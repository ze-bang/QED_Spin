#pragma once

// =============================================================================
// ed/core/dispatch.h — the single canonical ED entry point.
//
// This header is the public C++ surface for "give me eigenvalues / vectors
// from this Hamiltonian directory". It replaces, and supersedes:
//
//     ed/core/ed_wrapper.h             (full-Hilbert / fixed-Sz / GPU)
//     ed/core/ed_wrapper_streaming.h   (streaming spatial symmetry)
//     ed/core/ed_dispatch_symmetry.h   (one-shot symmetry dispatcher)
//
// All three legacy headers remain available as thin transparent forwarders
// so that out-of-tree callers keep compiling unchanged, but the matvec-
// unification rollout (Phase 6) moves every in-tree caller onto this
// single facade.
//
// Public interface
// ----------------
//
// ed::exact_diagonalization(directory, ...)
// ed::exact_diagonalization(interaction_file, ...)
//
// Both overloads dispatch on the orthogonal axes recorded in `params`:
//
//   * use_symmetry  → route through the streaming-symmetry kernel
//                     (sector-by-sector matvec); GPU-aware per sector.
//   * use_fixed_sz  → project to a fixed Sz sector (chosen by
//                     params.n_up, defaulting to num_sites / 2 when
//                     n_up < 0 -- typically the ground-state-containing
//                     sector for nearest-neighbor Heisenberg-like
//                     Hamiltonians without a Zeeman field).
//   * use_gpu       → GPU matvec for the full-Hilbert and fixed-Sz
//                     paths; per-sector GPU dispatch inside the
//                     streaming-symmetry kernel.
//   * use_mpi       → distributed CPU/GPU matvec (delegated to the
//                     ed/distributed/* library).
//
// These axes are orthogonal and combine freely; the legacy combinatorial
// `exact_diagonalization_*_fixed_sz_gpu_streaming_...` entry points are
// fully reachable through this one call. The auto_pilot::solve front-end
// detects symmetry / Sz conservation and sets these axes automatically.
//
// All bit-flip matvec eventually funnels through the matvec-unification
// MatVecOperator interface (ed/matvec/matvec.h); this file is the only
// public C++ surface that users should need to include.
// =============================================================================

#include <ed/core/ed_wrapper.h>            // full-Hilbert + fixed-Sz + GPU kernels
#include <ed/core/ed_wrapper_streaming.h>  // streaming-symmetry kernel

#include <cstdint>
#include <string>

namespace ed {

/**
 * @brief Canonical ED entry point (directory form).
 *
 * Dispatches on the orthogonal axes in `params` -- use_symmetry,
 * use_fixed_sz, use_gpu, use_mpi -- to the appropriate kernel. The
 * choice is deterministic from `params` alone; auto_pilot::solve is
 * the recommended front-end if you want axis selection driven by
 * Hamiltonian analysis (Sz conservation, Zeeman field detection,
 * symmetry-info presence).
 *
 * @param directory               Hamiltonian directory (InterAll.dat /
 *                                Trans.dat / CounterTerm.dat /
 *                                ThreeBodyG.dat and, when
 *                                params.use_symmetry == true,
 *                                automorphism_results/ -- generated on
 *                                the fly if missing).
 * @param method                  Diagonalization method.
 * @param params                  Parameter bag (axes, sizes, options).
 * @param format                  Hamiltonian file format.
 * @param interaction_filename    Interaction-term filename inside `directory`.
 * @param single_site_filename    Single-site (Trans / Zeeman) filename.
 * @param counterterm_filename    Counter-term filename.
 * @param three_body_filename     Three-body term filename.
 * @return EDResults              Eigenvalues sorted ascending + optional
 *                                eigenvectors / timing / sector breakdown.
 */
inline EDResults exact_diagonalization(
    const std::string& directory,
    DiagonalizationMethod method = DiagonalizationMethod::LANCZOS,
    const EDParameters& params = EDParameters(),
    HamiltonianFileFormat format = HamiltonianFileFormat::STANDARD,
    const std::string& interaction_filename = "InterAll.dat",
    const std::string& single_site_filename = "Trans.dat",
    const std::string& counterterm_filename = "CounterTerm.dat",
    const std::string& three_body_filename = "ThreeBodyG.dat")
{
    // ----- Non-symmetry path: full-Hilbert / fixed-Sz / GPU / MPI -----
    // ed_wrapper.h's exact_diagonalization_from_directory already routes
    // on use_fixed_sz, use_gpu, and use_mpi internally.
    if (!params.use_symmetry) {
        return ::exact_diagonalization_from_directory(
            directory, method, params, format,
            interaction_filename, single_site_filename,
            counterterm_filename, three_body_filename);
    }

    // ----- Spatial-symmetry path: streaming kernel (per-sector matvec) -----
    // The streaming kernel honours use_gpu (per-sector GPU dispatch) and
    // use_fixed_sz (fixed-Sz orbit basis) inside its implementation.
    if (params.use_fixed_sz) {
        const std::int64_t n_up = (params.n_up >= 0)
            ? params.n_up
            : static_cast<std::int64_t>(params.num_sites / 2);
        return ::exact_diagonalization_streaming_symmetry_fixed_sz(
            directory, n_up, method, params,
            interaction_filename, single_site_filename,
            params.basis_cache_dir, params.precompute_basis_only);
    }
    return ::exact_diagonalization_streaming_symmetry(
        directory, method, params,
        interaction_filename, single_site_filename,
        params.basis_cache_dir, params.precompute_basis_only);
}

/**
 * @brief Canonical ED entry point (file-path form).
 *
 * Same contract as the directory overload but takes explicit file paths.
 * The directory used to locate `automorphism_results/` (when
 * params.use_symmetry == true) is derived as `dirname(interaction_file)`.
 */
inline EDResults exact_diagonalization(
    const std::string& interaction_file,
    const std::string& single_site_file = "",
    const std::string& counterterm_file = "",
    const std::string& three_body_file = "",
    DiagonalizationMethod method = DiagonalizationMethod::LANCZOS,
    const EDParameters& params = EDParameters(),
    HamiltonianFileFormat format = HamiltonianFileFormat::STANDARD)
{
    if (!params.use_symmetry) {
        return ::exact_diagonalization_from_files(
            interaction_file, single_site_file, counterterm_file,
            three_body_file, method, params, format);
    }

    // ----- Symmetry path: derive directory + basenames -----
    const auto split_path = [](const std::string& path,
                               std::string& dir, std::string& base) {
        const auto pos = path.find_last_of('/');
        if (pos == std::string::npos) {
            dir  = ".";
            base = path;
        } else {
            dir  = path.substr(0, pos);
            base = path.substr(pos + 1);
        }
    };

    std::string directory, interaction_basename;
    split_path(interaction_file, directory, interaction_basename);

    std::string single_site_basename = "Trans.dat";
    if (!single_site_file.empty()) {
        std::string ignored_dir;
        split_path(single_site_file, ignored_dir, single_site_basename);
    }

    if (params.use_fixed_sz) {
        const std::int64_t n_up = (params.n_up >= 0)
            ? params.n_up
            : static_cast<std::int64_t>(params.num_sites / 2);
        return ::exact_diagonalization_streaming_symmetry_fixed_sz(
            directory, n_up, method, params,
            interaction_basename, single_site_basename,
            params.basis_cache_dir, params.precompute_basis_only);
    }
    return ::exact_diagonalization_streaming_symmetry(
        directory, method, params,
        interaction_basename, single_site_basename,
        params.basis_cache_dir, params.precompute_basis_only);
}

}  // namespace ed
