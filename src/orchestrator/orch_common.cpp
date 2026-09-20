// =============================================================================
// src/orchestrator/orch_common.cpp -- shared orchestrator plumbing
// (unified-writer rank test, solve() HDF5 persistence finalizer, the
// exact-small thermal env probe).
// Part of the workflow orchestrator; see orchestrator_internal.h for the
// file map.
// =============================================================================

#include "orchestrator_internal.h"

namespace ed::workflows {
namespace orch_detail {

// ---------------------------------------------------------------------------
// Helper: which rank should own the unified ``ed_results.h5`` writer?
//
// "Universal save contract" follow-up (May 2026): distributed lanes
// (MPI / MPI+CUDA) were previously skipping the orchestrator's
// HDF5 finalizer entirely on the assumption that every rank holds
// only a slab of the result. That is true for slab-distributed
// eigenvectors / TPQ state vectors -- those still get written to
// per-rank ``rank_<r>.h5`` files by ``ed_distributed_main`` -- but
// the *aggregate* fields (eigenvalues, thermo curves, S(omega)) are
// already broadcast / reduced onto every rank, so rank 0 can write a
// single, unified ``ed_results.h5`` next to the per-rank slabs.
//
// Contract:
//   - serial (non-MPI) lane:  returns true on the single process.
//   - MPI lane (any backend): returns true on rank 0 only.
//   - MPI not initialised inside the orchestrator: returns true (we
//     are the only writer).
// ---------------------------------------------------------------------------
bool is_unified_writer(const Geometry& geom) {
#ifdef WITH_MPI
    if (geom.is_distributed()) {
        int inited = 0;
        MPI_Initialized(&inited);
        if (!inited) return true;
        int rank = 0;
        MPI_Comm_rank(geom.comm, &rank);
        return rank == 0;
    }
#else
    (void)geom;
#endif
    return true;
}

// ---------------------------------------------------------------------------
// solve() persistence finalizer (extracted helper, May 2026 follow-up).
//
// Centralises the post-kernel HDF5 emission so every dispatch lane --
// the ``lanczos_real`` real-Hermitian fast path, the standard complex
// Lanczos / BlockLanczos / KrylovSchur kernels, and the FullDiag
// fallback -- hits the same on-disk contract. Before this helper was
// pulled out the ``lanczos_real`` lane returned early (skipping every
// finalizer below the early-return), which left ``R.hdf5_path`` empty
// even when the caller had supplied ``opts.output_dir``.
//
// Contract:
//   * Serial lane (``!geom.is_distributed()``): writes
//     ``<output_dir>/ed_results.h5`` carrying ``/eigendata/eigenvalues``
//     and, when ``opts.compute_vectors == true`` and the kernel
//     populated ``R.eigenvectors->host``, the ``/eigendata/eigenvector_*``
//     datasets.
//   * MPI lane: writes the unified file only from rank 0 (see
//     ``is_unified_writer``) and only the aggregate eigenvalue array
//     (gathering the slab-distributed eigenvectors would require an
//     extra MPI_Gatherv pass; per-rank ``rank_<r>.h5`` files written by
//     ``ed_distributed_main`` remain the canonical location for
//     eigenvectors).
//
// No-ops when (a) the caller did not supply an ``output_dir``, (b) the
// path is the explicit "disabled" sentinel, or (c) ``R.hdf5_path`` is
// already populated by an upstream writer.
// ---------------------------------------------------------------------------
void apply_solve_save_finalizer(GroundStateResult& R,
                                const Geometry& geom,
                                const SolveOptions& opts) {
    if (opts.output_dir.empty()
            || HDF5IO::isDisabledOutputPath(opts.output_dir)) {
        return;
    }
    if (!R.hdf5_path.empty()) {
        return;  // upstream lane (e.g. FullDiag) already wrote.
    }

    // ----- Serial lane -----------------------------------------------
    if (!geom.is_distributed()) {
        // Eigenvectors path: kernel populated host-side eigenvectors
        // and caller asked for them.
        if (opts.compute_vectors
                && R.eigenvectors.has_value()
                && !R.eigenvectors->host.empty()) {
            try {
                HDF5IO::saveDiagonalizationResults(
                    opts.output_dir,
                    R.eigenvalues,
                    R.eigenvectors->host,
                    /*solver_name=*/"ed::workflows::solve");
                R.hdf5_path = opts.output_dir + "/ed_results.h5";
                R.eigenvectors->hdf5_path = R.hdf5_path;
            } catch (const std::exception& e) {
                R.backend.notes.emplace_back(
                    "eigenvector_save_failed", e.what());
            }
            return;
        }
        // Eigenvalues-only path: still want the unified ed_results.h5
        // so downstream tools (Python ``qed.solve``, post-processing
        // notebooks, etc.) have a single, canonical artefact to read.
        if (!R.eigenvalues.empty()) {
            try {
                std::error_code ec;
                std::filesystem::create_directories(opts.output_dir, ec);
                const std::string h5_path =
                    opts.output_dir + "/ed_results.h5";
                HDF5IO::createOrOpenFile(opts.output_dir);
                HDF5IO::saveEigenvalues(h5_path, R.eigenvalues);
                R.hdf5_path = h5_path;
            } catch (const std::exception& e) {
                R.backend.notes.emplace_back(
                    "eigenvalues_save_failed", e.what());
            }
        }
        return;
    }

    // ----- MPI lane: rank 0 only writes aggregate eigenvalues. -------
    if (is_unified_writer(geom) && !R.eigenvalues.empty()) {
        try {
            std::error_code ec;
            std::filesystem::create_directories(opts.output_dir, ec);
            const std::string h5_path =
                opts.output_dir + "/ed_results.h5";
            HDF5IO::createOrOpenFile(opts.output_dir);
            HDF5IO::saveEigenvalues(h5_path, R.eigenvalues);
            R.hdf5_path = h5_path;
            R.backend.notes.emplace_back(
                "mpi_unified_file",
                "Rank 0 wrote eigenvalues to ed_results.h5; "
                "eigenvectors remain slab-distributed across "
                "per-rank rank_<r>.h5 files.");
        } catch (const std::exception& e) {
            R.backend.notes.emplace_back(
                "mpi_unified_save_failed", e.what());
        }
    }
}

// ED_THERMAL_EXACT_SMALL=0 forces the real sampling kernel even at
// D <= SMALL_THERMAL_DIM. The fallback is a strict accuracy win for USERS, but
// it silently removes the sampling kernels from any accuracy test whose system
// fits under the cutoff -- test_thermal_dense_ref (N=6, dim=64) was already
// exercising this path instead of the mTPQ kernel it claimed to check. Tests
// that mean to gate a KERNEL set this to 0; nothing in production should.
// Read per call so a test can toggle it without restarting the process.
[[nodiscard]] bool exact_small_thermal_enabled() noexcept {
    return ed::env::flag("ED_THERMAL_EXACT_SMALL", true);
}

}  // namespace orch_detail
}  // namespace ed::workflows
