// =============================================================================
// src/orchestrator.cpp
//
// Implementation of `ed::solve` / `ed::thermal` / `ed::spectral`
// (Phase 4.2 of the Minimalist ED Collapse, May 2026).
//
// The three entry points dispatch through `ed::select_backend` to choose
// one of the four concrete Backends (Cpu / Cuda / Mpi / MpiCuda) and
// invoke the matching templated kernel from Phase 2:
//
//     ed::solve     -> lanczos_kernel<Backend>            (single eig)
//                  or block_lanczos_kernel<Backend>      (multi eig, BLAS-3)
//                  or krylov_schur_kernel<Backend>       (many eigs / harder problems)
//                  or full_diag fallback                  (small dim)
//     ed::thermal   -> tpq_kernel<Backend>  (mTPQ)
//                  or the existing FTLM / LTLM / KpmDos kernels (CPU-only
//                     until they migrate to Backend; orchestrator routes
//                     CPU-friendly cases here).
//     ed::spectral  -> cf_spectral_kernel<Backend>
//
// The orchestrator carries the legacy CLI behaviour ONLY for the
// `output_dir` HDF5 trail; everything else is the new uniform Result
// shape from `include/ed/core/results.h`.
// =============================================================================

#include <ed/orchestrator.h>
#include <ed/core/solver_defaults.h>

#include <ed/core/hdf5_io.h>             // saveDiagonalizationResults (uniform eigenvector dump)
#include <ed/core/mem_guard.h>           // leaf working-set guard (clean error vs OOM)
#include <ed/krylov/block_lanczos_kernel.h>
#include <ed/krylov/block_krylov_schur_kernel.h>
#include <ed/krylov/krylov_schur_kernel.h>
#include <ed/krylov/lanczos_kernel.h>
#include <ed/krylov/ritz_convergence.h>
#include <ed/krylov/tridiag_eigensolver.h>  // solve_tridiag* (MPI-free)
#include <ed/krylov/subspace_policy.h>      // krylov_subspace_dim / krylov_vector_budget
// Execution planner / feasibility "dictator" removed (sensible defaults +
// env-override leaf hooks instead). The matvec/symmetry leaf policy hooks
// (csr_policy_hook, sym_matvec_policy_hook, basis_policy_hook) still provide
// the default + env-override behaviour, consumed lazily inside the operator
// backends; the orchestrator no longer overrides them.
#include <ed/core/operator.h>               // Operator introspection (term SoA, N)
#include <ed/symmetry/sector_operator.h>    // SectorOperator (group_size introspection)

#include <fstream>   // /proc/meminfo
#include <string>
#include <unistd.h>  // sysconf
#include <ed/matvec/backends/cpu_backend.h>
#include <ed/observables/cf_dynamical.h>
#include <ed/observables/cf_spectral_kernel.h>
#include <ed/observables/kpm_dynamical.h>
#include <ed/parallel/numa.h>            // pin_omp_threads_once
#include <ed/parallel/thread_budget.h>   // auto_threads_for_dim + ThreadBudgetScope
#include <ed/thermal/tpq_thermo.h>  // compute_tpq_thermo_from_trajectories aggregator
#include <ed/solvers/lanczos.h>  // FullDiag fallback (zheevd on the dense matrix)
#include <ed/thermal/ftlm_kernel.h>
#include <ed/thermal/oftlm_kernel.h>
#include <ed/thermal/kpm_dos_kernel.h>
#include <ed/thermal/ltlm_kernel.h>
#include <cstdio>
#include <ed/thermal/mtpq_kernel.h>
#include <ed/thermal/mtpq_f32.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>      // getenv (Wave 1.1 real-H fast-path opt-out)
#include <filesystem>
#include <iostream>
#include <random>
#include <stdexcept>
#include <type_traits>  // std::is_same_v (Wave 1.1)
#include <variant>

namespace ed::workflows {

namespace {

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
inline bool is_unified_writer(const Geometry& geom) {
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
inline void apply_solve_save_finalizer(GroundStateResult& R,
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

// Pick a sensible eigensolver when the caller leaves SolveMethod::Auto: full
// diagonalization for tiny spaces, Lanczos otherwise. (Replaces the planner's
// cost-model method choice.)
[[nodiscard]] SolveMethod default_method_for(const LinearOperator& H) {
    return (H.global_dim() <= 1024) ? SolveMethod::FullDiag : SolveMethod::Lanczos;
}

// ---------------------------------------------------------------------------
// Exact canonical thermodynamics from a complete eigenspectrum.
//
// Used as a small-sector fallback for mTPQ: when the Hilbert-space
// dimension is small (D <= SMALL_THERMAL_DIM), the stochastic mTPQ
// estimator has large per-sample variance (no algorithmic bug — the inherent
// limitation is that TPQ needs D >> num_samples for good typicality). For
// the sz_spatial symmetry mode on N=8, the (n_up, k) sub-sectors have
// D ≈ 1–9, which causes dE failures of ~0.12 even with 20 samples.
//
// Math:
//   Z(β) = Σ_j exp(-β E_j)
//   F(T)  = −T ln Z(β)     ← includes the ln(D) baseline automatically
//   E(T)  = Σ_j E_j w_j,   w_j = exp(−β E_j)/Z
//   Cv(T) = β² (Σ_j E_j² w_j − E(T)²)
//   S(T)  = β (E − F)
//
// The absolute free energy (F carries ln(D) at high T) is exactly what
// combine_sector_thermodynamics expects for the per-sector Boltzmann weight.
// ---------------------------------------------------------------------------
constexpr std::uint64_t SMALL_THERMAL_DIM = 512;

static ThermodynamicData compute_canonical_thermo_from_eigs(
    const std::vector<double>& eigs,
    const std::vector<double>& temperatures)
{
    ThermodynamicData td;
    if (eigs.empty() || temperatures.empty()) return td;
    const std::size_t nT = temperatures.size();
    td.temperatures = temperatures;
    td.energy.assign(nT, 0.0);
    td.specific_heat.assign(nT, 0.0);
    td.free_energy.assign(nT, 0.0);
    td.entropy.assign(nT, 0.0);

    const double E0 = *std::min_element(eigs.begin(), eigs.end());
    for (std::size_t t = 0; t < nT; ++t) {
        const double T = temperatures[t];
        if (!(T > 0.0)) continue;
        const double beta = 1.0 / T;
        double Z = 0.0, ZE = 0.0, ZE2 = 0.0;
        for (const double E : eigs) {
            const double w = std::exp(-beta * (E - E0));
            Z   += w;
            ZE  += w * E;
            ZE2 += w * E * E;
        }
        if (!(Z > 0.0)) continue;
        const double E_avg  = ZE  / Z;
        const double E2_avg = ZE2 / Z;
        const double lnZ    = std::log(Z) - beta * E0;
        td.energy[t]        = E_avg;
        td.specific_heat[t] = beta * beta * (E2_avg - E_avg * E_avg);
        td.free_energy[t]   = -lnZ / beta;
        td.entropy[t]       = beta * (E_avg - td.free_energy[t]);
    }
    return td;
}

template <typename Backend>
GroundStateResult solve_on(Backend& be,
                           const LinearOperator& H,
                           const SolveOptions& opts) {
    using Complex = std::complex<double>;
    const auto geom = H.geometry();
    auto matvec = H.template bind<Backend>();

    GroundStateResult R;

    // -----------------------------------------------------------------------
    // Sensible defaults (planner removed). The caller's explicit method /
    // max_iter win; otherwise a simple dim-based default. No memory-budget
    // pre-flight refusal, and no CSR / symmetry-matvec override -- the leaf
    // policy hooks (csr_policy_hook / sym_matvec_policy_hook / basis_policy_hook)
    // keep their default + env-override behaviour, consumed lazily at first
    // matvec.
    // -----------------------------------------------------------------------
    const SolveMethod method = (opts.method != SolveMethod::Auto)
        ? opts.method
        : default_method_for(H);
    const std::size_t max_iter =
        (opts.max_iter > 0) ? opts.max_iter : 2 * opts.num_eigs + 30;
    const std::uint64_t subspace_cap_vectors = 0;  // uncapped (no planner budget)

    // Leaf memory guard: throw cleanly before the dominant allocation rather
    // than OOM-crash. H.global_dim() is the actual working dimension (full /
    // fixed-Sz block / symmetry sector). Coarse: dense matrix for full diag;
    // stored Krylov basis (~max_iter vectors, x block_size) when eigenvectors
    // are kept; a handful of work vectors otherwise.
    {
        const std::uint64_t D = H.global_dim();
        constexpr std::uint64_t CX = 16;  // sizeof(complex<double>)
        std::uint64_t est;
        if (method == SolveMethod::FullDiag) {
            est = D * D * CX;
        } else if (opts.compute_vectors) {
            std::uint64_t vecs = std::max<std::uint64_t>(max_iter, 4);
            if (method == SolveMethod::BlockLanczos ||
                method == SolveMethod::BlockKrylovSchur)
                vecs *= std::max<std::size_t>(1, opts.block_size);
            est = D * vecs * CX;
        } else {
            est = D * 8ull * CX;  // eigenvalues-only: ring-buffer work vectors
        }
        ed::core::guard_working_set(est, "ed::solve");
    }

    const auto t0 = std::chrono::steady_clock::now();

    // Deterministic-ish seed for reproducibility within a single process.
    // The kernel expects the seed in the backend's memory space (host for
    // CPU/MPI, device for CUDA/MPI+CUDA). Build the seed on host first then
    // stage through `copy_from_host` into a backend-allocated vector so the
    // kernel's internal `be.copy(seed -> v_curr)` (which is a D2D for CUDA)
    // is given a properly-resident pointer.
    std::vector<Complex> seed_host(geom.local_dim);
    {
        std::mt19937_64 gen(0xCAFEBABEULL);
        std::normal_distribution<double> nd(0.0, 1.0);
        double sumsq = 0.0;
        for (auto& z : seed_host) {
            const double a = nd(gen), b = nd(gen);
            z = Complex(a, b);
            sumsq += a * a + b * b;
        }
        const double inv = (sumsq > 0.0) ? (1.0 / std::sqrt(sumsq)) : 1.0;
        for (auto& z : seed_host) z *= inv;
    }
    auto seed_backend = be.make_zero_vector(geom.local_dim);
    be.copy_from_host(seed_host.data(), seed_backend.get(), geom.local_dim);
    const Complex* seed = seed_backend.get();

    if (method == SolveMethod::Lanczos) {
        // -------------------------------------------------------------
        // Wave 1.1 of the SOTA Performance rollout (May 2026): on the
        // CPU backend, for the canonical eigenvalues-only ground-state
        // request on a real-Hermitian operator, dispatch to the
        // legacy `lanczos_real` lane. This was the engine the Apr 25
        // baseline measured (`bench_vs_xdiag_*.json` Python rows) and
        // remains 30-50% faster than the unified complex
        // `lanczos_kernel<CpuBackend>` thanks to fused BLAS-1, K=1
        // local-DGKS, zero-copy ring rotation, and a native-double
        // recurrence (`src/solvers/cpu/lanczos.cpp:1110-1258`).
        //
        // Eligibility (all must hold):
        //   * CpuBackend (no GPU / MPI lane affected),
        //   * single eigenvalue (the smallest --- num_eigs == 1),
        //   * eigenvalues only (caller did NOT request eigenvectors;
        //     CF spectral / per-state observables go through the
        //     complex kernel which keeps the basis),
        //   * H reports ``is_real_hermitian() == true``.
        //
        // Env opt-out: ``ED_FORCE_COMPLEX_LANCZOS=1`` returns the
        // pre-Wave-1.1 behaviour (unified complex kernel) for A/B
        // performance comparison and bisection.
        // -------------------------------------------------------------
        if constexpr (std::is_same_v<Backend, ed::matvec::CpuBackend>) {
            const bool force_complex = []() {
                const char* env = std::getenv("ED_FORCE_COMPLEX_LANCZOS");
                return env && env[0] == '1';
            }();
            if (!force_complex
                    && opts.num_eigs == 1
                    && !opts.compute_vectors
                    && H.is_real_hermitian()) {
                auto Hv_real = H.bind_real_cpu();
                std::vector<double> eigs;
                ::lanczos_real(
                    [Hv_real](const double* in, double* out, int n) {
                        Hv_real(in, out, static_cast<std::size_t>(n));
                    },
                    static_cast<std::uint64_t>(geom.local_dim),
                    static_cast<std::uint64_t>(max_iter),
                    /*exct=*/1u,
                    opts.tolerance,
                    eigs);
                if (!eigs.empty()) {
                    R.eigenvalues.assign(eigs.begin(),
                                         eigs.begin() + std::min<std::size_t>(
                                             opts.num_eigs, eigs.size()));
                }
                R.krylov.iters_done = 0;  // lanczos_real does not expose this
                R.krylov.converged  = true;
                const auto t1 = std::chrono::steady_clock::now();
                R.backend.wall_seconds =
                    std::chrono::duration<double>(t1 - t0).count();
                R.backend.notes.emplace_back(
                    "dispatch", "lanczos_real (Wave 1.1 real-H fast path)");
                // Phase D (May 2026): truthful lane reporting. The
                // lanczos_real fast path is guarded by the
                // CpuBackend ``constexpr`` branch above so the lane
                // label is the template's lane unconditionally. Using
                // ed::lane_label_for<Backend>() keeps the labels
                // consistent with the variant-driven helper used at
                // the bottom of solve() / thermal() / spectral().
                R.backend.lane = ed::lane_label_for<Backend>();
                R.backend.mpi_size = 1;
                // "Universal save contract" follow-up (May 2026): the
                // lanczos_real fast path used to ``return R`` here and
                // silently bypass every persistence finalizer below.
                // Run the shared finalizer so this lane lands on the
                // same ``ed_results.h5`` on-disk contract as every
                // other dispatch path.
                apply_solve_save_finalizer(R, geom, opts);
                return R;
            }
        }

        ed::krylov::LanczosKernelOptions kopts;
        kopts.max_iter      = max_iter;
        kopts.dim_cap       = static_cast<std::size_t>(geom.global_dim);
        kopts.keep_basis    = opts.compute_vectors;

        // Wave 2.1 + correction: LocalDGKS3 K=1 only ortho-projects
        // against the most recent two basis vectors. That is enough
        // for the EIGENVALUES-only path (tridiag eigvals don't need
        // mutually-orthogonal basis vectors), but if the orchestrator
        // is asked to RECONSTRUCT eigenvectors via
        //     psi_k = sum_i S(i, k) * V_i
        // (which is the path taken by ``ground_state_cf`` spectral and
        // any caller that sets ``compute_vectors = true``) the basis
        // MUST stay numerically orthogonal across all iterations.
        // FullCGS2 (against the kept basis) is the standard recipe.
        //
        // So: keep K=1 LocalDGKS3 (Wave 2.1) when basis is NOT kept,
        // and use FullCGS2 when it IS. ``ED_LANCZOS_REORTH_K`` still
        // overrides the local ring width when the user knows their
        // spectrum has near-degeneracies that K=1 can miss.
        if (kopts.keep_basis) {
            kopts.reorth = ed::krylov::ReorthPolicy::FullCGS2;
        } else {
            kopts.reorth          = ed::krylov::ReorthPolicy::LocalDGKS3;
            kopts.local_ring_size = 1;
            if (const char* k_env = std::getenv("ED_LANCZOS_REORTH_K")) {
                try {
                    const long k_val = std::stol(k_env);
                    if (k_val >= 1 && k_val <= 64) {
                        kopts.local_ring_size =
                            static_cast<std::size_t>(k_val);
                    }
                } catch (...) {
                    // malformed env: silently keep the default.
                }
            }
        }
        // Wire in Ritz-value early exit so the orchestrator matches the
        // legacy CPU `lanczos()` convergence behaviour (otherwise the
        // kernel always runs to `max_iter` -- 5-10x slower on small
        // problems and a noticeable hit even on large ones).
        kopts.convergence_check =
            ed::krylov::make_smallest_ritz_convergence(opts.num_eigs,
                                                       opts.tolerance,
                                                       /*min_iters=*/0);
        // Wave 2.6: check every-5 iterations to amortise the O(m^2)
        // LAPACK tridiag eigensolve. A few extra Lanczos iterations
        // (~ check_interval / 2) are cheaper than one extra dstevd
        // every iter past convergence. Matches the distributed lane
        // and the post-Wave-2.6 `lanczos()` default. Override via
        // env ``ED_LANCZOS_CHECK_EVERY``.
        kopts.convergence_check_interval = 5;
        if (const char* ce = std::getenv("ED_LANCZOS_CHECK_EVERY")) {
            try {
                const long ci = std::stol(ce);
                if (ci >= 1 && ci <= 1000) {
                    kopts.convergence_check_interval =
                        static_cast<std::size_t>(ci);
                }
            } catch (...) {
                // malformed env: keep the default.
            }
        }
        auto kres = ed::krylov::lanczos_kernel(be, matvec, geom.local_dim,
                                               seed, kopts);
        // Solve the small (m x m) real-symmetric tridiagonal for the
        // lowest `num_eigs` eigenvalues. When the caller didn't request
        // eigenvectors, use the eigenvalues-only Eigen path -- the
        // legacy `solve_tridiag_with_eigenvectors` did the full eigen
        // problem unconditionally, which is ~2-3x slower for the
        // common num_eigs=1 + compute_vectors=false workflow.
        std::vector<double> evals;
        std::vector<double> evec_coeffs;  // column-major m x m
        if (opts.compute_vectors) {
            std::vector<double> weights;
            ed::krylov::detail::solve_tridiag_with_eigenvectors(
                kres.alpha, kres.beta, kres.alpha.size(), evals, weights, evec_coeffs);
        } else {
            evals = ed::krylov::detail::solve_tridiag(
                kres.alpha, kres.beta, kres.alpha.size());
        }
        const std::size_t n_keep = std::min<std::size_t>(opts.num_eigs, evals.size());
        R.eigenvalues.assign(evals.begin(), evals.begin() + n_keep);

        // Reconstruct host-side eigenvectors from the kept Lanczos basis
        // when the caller requested them. evec_coeffs is the (m x m)
        // eigenvector matrix of the tridiag in column-major order; the
        // k-th eigenvector in the original Hilbert space is the linear
        // combination psi_k = sum_i evec_coeffs(i, k) * basis[i].
        if (opts.compute_vectors && !kres.basis.empty()) {
            const std::size_t m = kres.alpha.size();
            EigenvectorRef evref;
            evref.host.resize(n_keep,
                              std::vector<Complex>(geom.local_dim, Complex{0.0, 0.0}));
            std::vector<Complex> basis_host(geom.local_dim);
            for (std::size_t i = 0; i < m && i < kres.basis.size(); ++i) {
                be.copy_to_host(kres.basis[i].get(),
                                basis_host.data(), geom.local_dim);
                for (std::size_t k = 0; k < n_keep; ++k) {
                    const double c = evec_coeffs[i + k * m];
                    auto& out = evref.host[k];
                    for (std::size_t r = 0; r < geom.local_dim; ++r) {
                        out[r] += c * basis_host[r];
                    }
                }
            }
            R.eigenvectors = std::move(evref);
        }

        R.krylov.alpha = std::move(kres.alpha);
        R.krylov.beta  = std::move(kres.beta);
        R.krylov.iters_done = kres.iters_done;
    } else if (method == SolveMethod::BlockLanczos) {
        ed::krylov::BlockLanczosOptions kopts;
        kopts.num_eigs        = opts.num_eigs;
        kopts.block_size      = opts.block_size;
        kopts.max_iter        = max_iter;
        kopts.tolerance       = opts.tolerance;
        kopts.compute_vectors = opts.compute_vectors;
        kopts.output_dir      = opts.output_dir;
        // Reorth profile: full (stored basis) vs lean (local-only, eigenvalues).
        // Eigenvectors force full; else the planner's lean recommendation (the
        // full block basis would not fit the budget) OR the caller's opt forces
        // lean; ED_BLOCK_LANCZOS_LEAN=1 is the explicit override. The planner
        // path is what makes block-Lanczos memory-bounded (guarantee completion).
        if (opts.compute_vectors) {
            kopts.keep_basis = true;   // eigenvectors require the stored basis
        } else {
            // Planner removed: honour the caller's flag; ED_BLOCK_LANCZOS_LEAN=1
            // is the explicit lean override.
            kopts.keep_basis = opts.block_lanczos_keep_basis;
            if (const char* e = std::getenv("ED_BLOCK_LANCZOS_LEAN"))
                kopts.keep_basis = !(e[0] == '1' && e[1] == '\0');
        }
        auto kres = ed::krylov::block_lanczos_kernel(be, matvec,
            geom.local_dim, geom.global_dim, kopts);
        R.eigenvalues = std::move(kres.eigenvalues);
        if (opts.compute_vectors && !kres.eigenvectors.empty()) {
            EigenvectorRef evref;
            evref.host.reserve(kres.eigenvectors.size());
            std::vector<Complex> tmp(geom.local_dim);
            for (auto& v : kres.eigenvectors) {
                be.copy_to_host(v.get(), tmp.data(), geom.local_dim);
                evref.host.push_back(tmp);
            }
            R.eigenvectors = std::move(evref);
        }
        R.krylov.iters_done    = kres.blocks_built;
        R.krylov.converged     = kres.converged;
        R.krylov.ritz_residuals = kres.residuals;
        R.krylov.n_converged   = kres.n_converged;
        R.krylov.resid_history = kres.resid_history;
        if (!kres.residuals.empty())
            R.krylov.residual_norm = *std::max_element(kres.residuals.begin(),
                                                       kres.residuals.end());
    } else if (method == SolveMethod::BlockKrylovSchur) {
        ed::krylov::BlockKrylovSchurOptions kopts;
        kopts.num_eigs        = opts.num_eigs;
        kopts.block_size      = opts.block_size;
        kopts.max_iter        = max_iter;
        kopts.tolerance       = opts.tolerance;
        kopts.compute_vectors = opts.compute_vectors;
        kopts.output_dir      = opts.output_dir;
        kopts.global_n        = geom.global_dim;
        kopts.max_subspace_vectors = subspace_cap_vectors;
        auto kres = ed::krylov::block_krylov_schur_kernel(be, matvec,
            geom.local_dim, geom.global_dim, kopts);
        R.eigenvalues = std::move(kres.eigenvalues);
        if (opts.compute_vectors && !kres.eigenvectors.empty()) {
            EigenvectorRef evref;
            evref.host.reserve(kres.eigenvectors.size());
            std::vector<Complex> tmp(geom.local_dim);
            for (auto& v : kres.eigenvectors) {
                be.copy_to_host(v.get(), tmp.data(), geom.local_dim);
                evref.host.push_back(tmp);
            }
            R.eigenvectors = std::move(evref);
        }
        R.krylov.iters_done    = kres.restarts;
        R.krylov.converged     = kres.converged;
        R.krylov.ritz_residuals = kres.residuals;
        R.krylov.n_converged   = kres.n_converged;
        R.krylov.resid_history = kres.resid_history;
        if (!kres.residuals.empty())
            R.krylov.residual_norm = *std::max_element(kres.residuals.begin(),
                                                       kres.residuals.end());
    } else if (method == SolveMethod::KrylovSchur) {
        ed::krylov::KrylovSchurOptions kopts;
        kopts.num_eigs        = opts.num_eigs;
        kopts.max_iter        = max_iter;
        kopts.tolerance       = opts.tolerance;
        kopts.compute_vectors = opts.compute_vectors;
        kopts.global_n        = geom.global_dim;
        kopts.output_dir      = opts.output_dir;
        kopts.max_subspace_vectors = subspace_cap_vectors;
        auto kres = ed::krylov::krylov_schur_kernel(be, matvec,
            geom.local_dim, seed, kopts);
        R.eigenvalues = std::move(kres.eigenvalues);
        if (opts.compute_vectors && !kres.eigenvectors.empty()) {
            EigenvectorRef evref;
            evref.host.reserve(kres.eigenvectors.size());
            std::vector<Complex> tmp(geom.local_dim);
            for (auto& v : kres.eigenvectors) {
                be.copy_to_host(v.get(), tmp.data(), geom.local_dim);
                evref.host.push_back(tmp);
            }
            R.eigenvectors = std::move(evref);
        }
        R.krylov.iters_done = kres.iters_done;
        R.krylov.converged  = kres.converged;
    } else {
        // FullDiag lane: build the dense matrix-vector applied to every
        // basis vector and run LAPACK zheevd via the legacy
        // `full_diagonalization` helper. The orchestrator only takes this
        // path for small dimensions (<= 2^12 by default) so the O(N^3)
        // dense step is affordable.
        //
        // Distributed path (Wave A4 -- Full unified-interface collapse,
        // May 2026): gather the per-rank slab matvecs onto rank 0,
        // assemble the dense matrix there, run zheevd on rank 0, then
        // MPI_Bcast the eigenvalues. This trades the simplicity of a
        // ScaLAPACK redistribution for not introducing a new dependency,
        // and is correct precisely in the regime where FullDiag is the
        // orchestrator's chosen method (global_dim <= 2^12 -- ~12-13 MB
        // of dense complex<double> on rank 0). A ScaLAPACK path can
        // replace this when callers exercise FullDiag at larger
        // distributed dims.
#ifdef WITH_MPI
        if (geom.is_distributed()) {
            int mpi_rank = 0, mpi_size = 1;
            MPI_Comm_rank(geom.comm, &mpi_rank);
            MPI_Comm_size(geom.comm, &mpi_size);

            const auto Nlocal  = static_cast<int>(geom.local_dim);
            const auto Nglobal = static_cast<int>(geom.global_dim);

            std::vector<int> recv_counts(mpi_size, 0);
            std::vector<int> recv_displs(mpi_size, 0);
            MPI_Allgather(&Nlocal, 1, MPI_INT,
                          recv_counts.data(), 1, MPI_INT, geom.comm);
            for (int r = 1; r < mpi_size; ++r) {
                recv_displs[r] = recv_displs[r-1] + recv_counts[r-1];
            }

            // Dense matrix on rank 0; null elsewhere.
            std::vector<Complex> H_dense;
            if (mpi_rank == 0) {
                H_dense.assign(static_cast<std::size_t>(Nglobal)
                               * static_cast<std::size_t>(Nglobal),
                               Complex{0.0, 0.0});
            }

            // For each column k: build e_k as a slab-distributed
            // vector, apply H to get H * e_k (per-rank y_local),
            // gather y_local onto rank 0 into column k of H_dense.
            std::vector<Complex> ek_local(Nlocal, Complex{0.0, 0.0});
            std::vector<Complex> y_local(Nlocal, Complex{0.0, 0.0});
            for (int k = 0; k < Nglobal; ++k) {
                // Set e_k on the rank that owns global index k.
                std::fill(ek_local.begin(), ek_local.end(),
                          Complex{0.0, 0.0});
                const std::uint64_t lo = geom.local_offset;
                const std::uint64_t hi = lo + Nlocal;
                if (static_cast<std::uint64_t>(k) >= lo
                    && static_cast<std::uint64_t>(k) < hi) {
                    ek_local[static_cast<std::size_t>(
                        static_cast<std::uint64_t>(k) - lo)] =
                        Complex{1.0, 0.0};
                }

                matvec(ek_local.data(), y_local.data(),
                       static_cast<std::size_t>(Nlocal));

                // Gather column k onto rank 0.
                Complex* col_ptr = (mpi_rank == 0)
                    ? &H_dense[static_cast<std::size_t>(k)
                               * static_cast<std::size_t>(Nglobal)]
                    : nullptr;
                MPI_Gatherv(
                    y_local.data(), Nlocal, MPI_DOUBLE_COMPLEX,
                    col_ptr, recv_counts.data(), recv_displs.data(),
                    MPI_DOUBLE_COMPLEX, /*root=*/0, geom.comm);
            }

            std::vector<double> eigs(Nglobal, 0.0);
            if (mpi_rank == 0) {
                std::function<void(const Complex*, Complex*, int)> Hv =
                    [&](const Complex* in, Complex* out, int n) {
                        // Apply H_dense (column-major) once per call.
                        for (int i = 0; i < n; ++i) {
                            Complex acc{0.0, 0.0};
                            for (int j = 0; j < n; ++j) {
                                acc += H_dense[static_cast<std::size_t>(j)
                                                * static_cast<std::size_t>(n)
                                                + static_cast<std::size_t>(i)]
                                    * in[j];
                            }
                            out[i] = acc;
                        }
                    };
                full_diagonalization(Hv, static_cast<std::size_t>(Nglobal),
                                     opts.num_eigs, eigs,
                                     opts.output_dir,
                                     opts.compute_vectors);
                if (!opts.output_dir.empty()
                        && !HDF5IO::isDisabledOutputPath(opts.output_dir)) {
                    R.hdf5_path = opts.output_dir + "/ed_results.h5";
                }
            }
            MPI_Bcast(eigs.data(), Nglobal, MPI_DOUBLE, 0, geom.comm);
            const std::size_t n_keep = std::min<std::size_t>(
                opts.num_eigs, static_cast<std::size_t>(Nglobal));
            R.eigenvalues.assign(eigs.begin(), eigs.begin() + n_keep);
            R.krylov.iters_done = 0;
            R.krylov.converged  = true;
        } else
#endif
        {
            // "Universal save contract" follow-up (May 2026): the
            // FullDiag column-extraction loop in
            // ``::full_diagonalization`` (lanczos.cpp:1483-1488) calls
            // ``H(unit_vec.data(), col_j.data(), N)`` with host
            // ``std::vector<Complex>`` storage. If we hand it a matvec
            // bound to a non-CPU backend (e.g. the streaming-symmetry
            // GPU mirror, advertised via
            // ``Geometry::supports_device_matvec=true``), the lambda
            // dereferences the host pointers as device pointers and
            // ``cudaMemsetAsync`` returns "invalid argument".
            //
            // The dense build is O(N) matvecs and the LAPACK O(N^3)
            // call dominates, so there is no perf gain in keeping the
            // FullDiag column build on the GPU. Pin it to the CPU
            // binding (``LinearOperator::bind_cpu()`` is supported by
            // every Operator subclass and is the fallback path
            // ``LinearOperator::bind<CpuBackend>()`` selects). Krylov /
            // BlockLanczos / KrylovSchur lanes above keep the original
            // device-bound matvec since they operate entirely in the
            // backend's memory space.
            ed::LinearOperator::MatvecFn cpu_matvec = H.bind_cpu();
            std::function<void(const Complex*, Complex*, int)> Hv =
                [&](const Complex* in, Complex* out, int n) {
                    cpu_matvec(in, out, static_cast<std::size_t>(n));
                };
            std::vector<double> eigs;
            // Pass &H so the dense matrix is assembled DIRECTLY from the sparse
            // term structure in O(nnz) (full-space / fixed-Sz lanes) instead of N
            // full matvecs. Symmetry lanes (and any operator without direct
            // support) return false and fall back to the Hv column build, which
            // stays SEQUENTIAL because the CPU matvec is not reentrant.
            full_diagonalization(Hv, geom.local_dim, opts.num_eigs, eigs,
                                 opts.output_dir,
                                 opts.compute_vectors,
                                 /*op_for_dense=*/&H);
            if (!opts.output_dir.empty()
                    && !HDF5IO::isDisabledOutputPath(opts.output_dir)) {
                R.hdf5_path = opts.output_dir + "/ed_results.h5";
            }
            const std::size_t n_keep = std::min<std::size_t>(
                opts.num_eigs, eigs.size());
            R.eigenvalues.assign(eigs.begin(), eigs.begin() + n_keep);
            R.krylov.iters_done = 0;
            R.krylov.converged  = true;
        }
    }

    // ---------------------------------------------------------------------
    // Uniform eigenvector HDF5 dump (May 2026 contract).
    //
    // Before this block, only the FullDiag lane persisted eigenvectors when
    // `opts.output_dir` was set (via `full_diagonalization(...)`).
    // Lanczos / BlockLanczos / KrylovSchur silently dropped the
    // `output_dir` argument: the Krylov kernels accept it in their
    // `Options` structs but never read it (header-comment is an explicit
    // "TODO: not yet used"), so callers got `R.eigenvectors->host`
    // populated but `R.hdf5_path` empty and Python's
    // `EDResults.eigenvectors_path` silently empty as well.
    //
    // This finalizer hooks every method that lands eigenvectors in the
    // shared host buffer onto the same HDF5 path the FullDiag lane and
    // the streaming-symmetry CLI use. Guards:
    //   * `opts.compute_vectors` was actually requested,
    //   * the caller supplied a non-empty, non-/dev/null `output_dir`,
    //   * the kernel populated host-side eigenvectors,
    //   * no upstream path already wrote (and recorded) the file,
    //   * single-rank lane only (the MPI lane uses per-rank rank_*.h5
    //     files written by `ed_distributed_main` / the streaming-symmetry
    //     directory walker -- writing a single shared `ed_results.h5`
    //     from N ranks would clobber across processes).
    // ---------------------------------------------------------------------
    // ---------------------------------------------------------------------
    // Universal persistence finalizer (May 2026 follow-up). Pinned by
    // the long block-comment on ``apply_solve_save_finalizer`` above.
    //
    // Writes:
    //   * serial lane: ``<out>/ed_results.h5`` with
    //     ``/eigendata/eigenvalues`` (always) and
    //     ``/eigendata/eigenvector_*`` (when ``compute_vectors`` is set
    //     and the kernel populated host-side eigenvectors).
    //   * MPI lane: same file from rank 0 carrying only the aggregate
    //     eigenvalue array. Per-rank ``rank_<r>.h5`` files remain the
    //     canonical location for slab-distributed eigenvectors.
    //
    // No-op when ``R.hdf5_path`` was already filled by the FullDiag
    // upstream lane.
    // ---------------------------------------------------------------------
    apply_solve_save_finalizer(R, geom, opts);

    // Phase D (May 2026): truthful lane reporting. The legacy line
    //
    //     R.backend.lane = geom.is_device() ? "gpu" : "cpu";
    //
    // pulled the label from the operator's memory_space, which is
    // wrong for every SectorView (streaming-symmetry /
    // FixedSzStreamingSymmetry): those views report ``Host``
    // memory_space yet advertise ``supports_device_matvec=true``, so
    // ``select_backend`` actually picks ``CudaBackend`` and
    // ``bind_cuda()`` wires a lazy GPU mirror -- the label simply
    // misreported the lane. ``ed::lane_label_for<Backend>()`` reads
    // the template parameter directly so the label always matches
    // the lane ``std::visit`` dispatched to.
    R.backend.lane = ed::lane_label_for<Backend>();
    R.backend.mpi_size = 1;
    const auto t1 = std::chrono::steady_clock::now();
    R.backend.wall_seconds =
        std::chrono::duration<double>(t1 - t0).count();
    return R;
}

// ---------------------------------------------------------------------------
// GroundStateCF / KpmDynamical seed guard (Jul 2026).
//
// The in-memory spectral lanes seeded the continued-fraction kernel with the
// best-effort GS vector from ``solve()``, which caps its Lanczos and does NOT
// reorthogonalise the eigenVECTOR (the eigenVALUE converges to the extreme
// long before the vector does). A GS vector carrying ~1e-1..1e-3 residual
// makes phi = O|psi0> -- and its weight ||phi||^2 -- systematically wrong:
// the plain in-memory S(Q, omega) came out ~5% high vs the exact Lehmann /
// cross-irrep references, while the cross-irrep lane was exact BECAUSE it
// already refines its GS (ensure_gs_residual). This mirrors that refinement
// on the host vector: measure the residual; if it exceeds 1e-8, rerun a
// FullCGS2 kept-basis Lanczos on H seeded by the current vector, rebuild the
// Ritz pair, and update (seed_host, E0) in place. Cheap when the vector was
// already good (one matvec); decisive when it was not.
inline void refine_gs_seed_host(const LinearOperator&        H,
                                std::vector<std::complex<double>>& seed_host,
                                double&                      E0)
{
    using Complex = std::complex<double>;
    const std::size_t n = seed_host.size();
    if (n == 0) return;
    ed::matvec::CpuBackend be;
    auto apply_H = H.template bind<ed::matvec::CpuBackend>();

    auto residual = [&](const std::vector<Complex>& v, double e) {
        std::vector<Complex> hv(n, Complex(0.0, 0.0));
        apply_H(v.data(), hv.data(), n);
        double num = 0.0, den = 0.0;
        for (std::size_t i = 0; i < n; ++i) {
            num += std::norm(hv[i] - e * v[i]);
            den += std::norm(v[i]);
        }
        return (den > 0.0) ? std::sqrt(num / den)
                           : std::numeric_limits<double>::infinity();
    };

    if (residual(seed_host, E0) < 1e-8) return;

    ed::krylov::LanczosKernelOptions kopts;
    kopts.max_iter   = std::min<std::size_t>(n, 300);
    kopts.reorth     = ed::krylov::ReorthPolicy::FullCGS2;
    kopts.keep_basis = true;
    kopts.dim_cap    = n;
    auto kres = ed::krylov::lanczos_kernel(be, apply_H, n,
                                           seed_host.data(), kopts);
    const std::size_t m = kres.alpha.size();
    if (m == 0) return;   // leave the best-effort seed; CF still runs
    std::vector<double> diag = kres.alpha;
    std::vector<double> off(m > 1 ? m - 1 : 1, 0.0);
    for (std::size_t i = 0; i + 1 < m; ++i) off[i] = kres.beta[i + 1];
    std::vector<double> z(m * m, 0.0);
    const lapack_int info = LAPACKE_dstevd(
        LAPACK_COL_MAJOR, 'V', static_cast<lapack_int>(m),
        diag.data(), off.data(), z.data(), static_cast<lapack_int>(m));
    if (info != 0) return;
    std::vector<Complex> refined(n, Complex(0.0, 0.0));
    for (std::size_t j = 0; j < m; ++j) {
        const Complex* vj = kres.basis[j].get();
        const double   yj = z[j];               // column 0, row j
        if (std::abs(yj) < 1e-300) continue;
        for (std::size_t i = 0; i < n; ++i) refined[i] += yj * vj[i];
    }
    double nrm = 0.0;
    for (const auto& c : refined) nrm += std::norm(c);
    if (nrm <= 0.0) return;
    const double inv = 1.0 / std::sqrt(nrm);
    for (auto& c : refined) c *= inv;
    // Adopt the refined pair only if it is genuinely better.
    if (residual(refined, diag[0]) < residual(seed_host, E0)) {
        seed_host = std::move(refined);
        E0 = diag[0];
    }
}

}  // namespace

// ---------------------------------------------------------------------------
// Public entry points.
// ---------------------------------------------------------------------------

GroundStateResult solve(const LinearOperator& H, SolveOptions opts) {
    // Apply the same thread-budget hygiene the legacy `lanczos()` /
    // `block_lanczos()` / `krylov_schur()` entries do (Phase 6.1 of the
    // matvec-unification arc; see docs/history/PHASE_8_GPU_MPI_OPT.md).
    // Without this the orchestrator runs OpenBLAS + OpenMP at
    // `omp_get_max_threads()` for every BLAS-1 / SpMV call -- which at
    // N=14 dim=16k turns a ~1 ms/iter SpMV into a ~5 ms/iter SpMV due to
    // inter-core memory-bandwidth contention. ED_AUTO_THREADS=0 disables.
    const ed::parallel::ThreadBudgetScope budget(
        ed::parallel::auto_threads_for_dim(H.geometry().local_dim));
    ed::parallel::pin_omp_threads_once();

    auto variant = select_backend(H.geometry(), opts.backend);
    return std::visit(
        [&](auto& backend_uptr) -> GroundStateResult {
            using BPtr = std::decay_t<decltype(backend_uptr)>;
            using B = typename BPtr::element_type;
            return solve_on<B>(*backend_uptr, H, opts);
        },
        variant);
}

ThermalResult thermal(const LinearOperator& H, ThermalOptions opts) {
    // All lanes are wired: mTPQ dispatches through the unified
    // `tpq_kernel` via the Phase 2.4 facades; FTLM / LTLM / KpmDos
    // dispatch through their own `*_kernel<Backend>` templates (CPU
    // implementations today, GPU when the kernels migrate). The variant
    // visit at each lane keeps the dispatch backend-agnostic.
    using Complex = std::complex<double>;
    const ed::parallel::ThreadBudgetScope budget(
        ed::parallel::auto_threads_for_dim(H.geometry().local_dim));
    ed::parallel::pin_omp_threads_once();

    BackendVariant variant;
#ifdef WITH_CUDA
    // fp32 single-GPU mTPQ fits one H100 (2 x complex<float> = 68.7 GB at
    // 2^32). The default gpu_mem_fits() estimate uses complex<double> x fudge 8
    // (= 549 GB) and would REJECT the GPU, falling to the CPU lane -- whose
    // 2^32 spectral-bound Lanczos auto-tune then OOM-kills the host. Force the
    // GPU lane here; the fp32 driver manages its own (fitting) device memory.
    if (opts.mtpq_fp32
        && opts.method == ThermalOptions::Method::mTPQ
        && H.supports_cuda_f32()
        && ed::have_cuda()) {
        variant = BackendVariant{std::make_unique<ed::matvec::CudaBackend>()};
    } else
#endif
    {
        variant = select_backend(H.geometry(), opts.backend);
    }

    // COMPLETION GUARANTEE (thermal lane). The operator's basis is already built,
    // so the binding constraint is the kernel WORKING SET: FTLM / LTLM keep a
    // krylov_dim window of length-N vectors; TPQ / KPM a handful. Plan it and
    // refuse cleanly if it would not fit, before allocating those vectors. The
    // small-sector exact fallback (D <= SMALL_THERMAL_DIM) is tiny and always
    // passes. allow_infeasible (force) opts out.
    // Leaf memory guard (planner feasibility pre-flight removed): throw cleanly
    // before the Krylov-basis allocation rather than OOM-crash. H.global_dim()
    // is the per-call working dim (symmetry iterates sectors a level up, so this
    // is the sector dim there). LTLM stores GS + excitation bases (~2x krylov);
    // FTLM one sample's basis at a time; TPQ/KPM are O(1) state vectors.
    {
        const std::uint64_t D = H.global_dim();
        std::uint64_t vecs;
        std::uint64_t elem = 16ull;  // complex<double>
        bool fp32_lane = false;
#ifdef WITH_CUDA
        // fp32 single-GPU mTPQ lane: the lean driver keeps just two
        // complex<float> device vectors (psi + w), so the working set is
        // D * 2 * 8 bytes (68.7 GB at 2^32), a quarter of the double estimate
        // -- otherwise this guard's 8-vector complex<double> estimate (549 GB
        // at 2^32) would refuse the run before the driver ever allocates.
        fp32_lane = opts.mtpq_fp32
                 && opts.method == ThermalOptions::Method::mTPQ
                 && H.supports_cuda_f32();
#endif
        if (fp32_lane) {
            vecs = 2;
            elem = 8ull;  // complex<float>
        } else {
            switch (opts.method) {
                case ThermalOptions::Method::LTLM:
                    vecs = 2 * std::max<std::size_t>(opts.krylov_dim, 4); break;
                case ThermalOptions::Method::FTLM:
                    vecs = std::max<std::size_t>(opts.krylov_dim, 4) + 4; break;
                case ThermalOptions::Method::OFTLM:
                    // per-sample Krylov basis + the exact-eigenpair Lanczos basis
                    vecs = std::max<std::size_t>(opts.krylov_dim, 4)
                         + 2 * opts.num_exact + 34; break;
                default:  // mTPQ / KpmDos: a handful of state vectors
                    vecs = 8; break;
            }
        }
        ed::core::guard_working_set(D * vecs * elem, "ed::thermal");
    }

    // Surface unification follow-up (May 2026): when the caller does
    // not supply an explicit ``opts.betas`` grid, construct one from
    // the temperature-scan knobs (``temp_min``, ``temp_max``,
    // ``num_temp_bins``) and -- crucially -- mirror the resulting
    // temperature axis into ``R.thermo.temperatures`` so downstream
    // Python / CLI consumers can read the scan back without
    // recomputing it from ``opts.*``. Mirrors the legacy
    // ``finite_temperature_lanczos`` contract that every call site
    // relied on.
    if (opts.betas.empty() && opts.num_temp_bins > 0
        && opts.temp_min > 0.0 && opts.temp_max > opts.temp_min) {
        opts.betas.reserve(opts.num_temp_bins);
        const double t_lo = opts.temp_min;
        const double t_hi = opts.temp_max;
        const std::size_t n = opts.num_temp_bins;
        // Linear temperature axis (T = T_min + i*(T_max-T_min)/(n-1)),
        // descending in beta so the natural ascending-T print stays
        // ascending after the kernel.
        for (std::size_t i = 0; i < n; ++i) {
            const double T = (n == 1)
                ? t_lo
                : t_lo + (t_hi - t_lo) * static_cast<double>(i)
                          / static_cast<double>(n - 1);
            opts.betas.push_back(T > 0.0 ? 1.0 / T : 1.0 / 1e-300);
        }
    }

    ThermalResult R;
    if (!opts.betas.empty()) {
        R.thermo.temperatures.reserve(opts.betas.size());
        for (double b : opts.betas) {
            R.thermo.temperatures.push_back(b > 0.0 ? 1.0 / b : 0.0);
        }
    }
    const auto t0 = std::chrono::steady_clock::now();

    // -----------------------------------------------------------------------
    // Small-sector exact-thermal fallback for mTPQ.
    //
    // mTPQ is a stochastic method that needs D >> num_samples for
    // good typicality. For small sectors (D <= SMALL_THERMAL_DIM), the
    // per-sample variance is too high for the dE tolerance even with 20+
    // samples. The primary symptom is the sz_spatial mTPQ failure: within an
    // n_up block, translation k-sectors have D ≈ 1–9 for N=8, giving a
    // statistical error of ~0.12 with 20 samples (vs the 0.08 tolerance).
    //
    // For any D <= SMALL_THERMAL_DIM, diagonalise exactly and compute the
    // canonical partition function directly. The resulting ThermodynamicData
    // uses the same absolute free-energy convention (F includes ln(D) at
    // high T) as compute_tpq_thermo_from_trajectories, so it plugs in
    // correctly to combine_sector_thermodynamics for Sz/spatial recombination.
    // -----------------------------------------------------------------------
    const bool is_tpq_method = (opts.method == ThermalOptions::Method::mTPQ);
    if (is_tpq_method &&
        H.geometry().global_dim > 0 &&
        H.geometry().global_dim <= SMALL_THERMAL_DIM &&
        !R.thermo.temperatures.empty() &&
        // When the caller requested TPQ state snapshots (probe_betas), the exact
        // fallback cannot produce them -- run the real TPQ trajectory instead
        // (accepting the small-sector variance the user implicitly opted into).
        opts.probe_betas.empty()) {
        const std::uint64_t D = H.geometry().global_dim;
        std::vector<double> eigs;
        full_diagonalization(H, D, D, eigs, /*dir=*/"", /*compute_eigenvectors=*/false);
        if (!eigs.empty()) {
            R.thermo = compute_canonical_thermo_from_eigs(
                eigs, R.thermo.temperatures);
            R.ground_state_energy = eigs.front();
            // The exact fallback ran on the selected backend lane; label it like
            // the normal return path (this early return previously left lane
            // unset, failing the lane-metadata assertions).
            R.backend.lane = ed::lane_label_from_variant(variant);
            return R;
        }
    }

    if (opts.method == ThermalOptions::Method::mTPQ) {
        std::visit([&](auto& backend_uptr) {
            using BPtr = std::decay_t<decltype(backend_uptr)>;
            using B = typename BPtr::element_type;
            ed::thermal::MtpqOptions kopts;
            kopts.num_samples = opts.num_samples;
            kopts.random_seed = opts.random_seed;
            kopts.output_dir  = opts.output_dir;
            kopts.probe_betas = opts.probe_betas;
            auto matvec = H.template bind<B>();

            // -------------------------------------------------------------
            // SOTA mTPQ auto-tune (June 2026).
            //
            // The microcanonical iteration |psi_{k+1}> = (L*I - H)|psi_k>
            // advances the effective inverse temperature by
            //   Delta_beta ~ 2 / (L - <H>),     beta_k = 2 k / (L - E_k).
            // So L sets BOTH (a) the high-temperature resolution (small
            // Delta_beta needs large L) and -- together with the step
            // count -- (b) the coldest temperature reached.
            //
            // The previous heuristic used ``log2(global_dim)`` as a
            // stand-in for the spectral ceiling E_max and coupled L to
            // ``temp_min`` via ``L = 2*max_iter/beta_target + proxy``.
            // That is doubly wrong: the dim proxy has nothing to do with
            // the actual band edge, and lowering ``temp_min`` to reach
            // colder T *shrank* L, coarsening Delta_beta and biasing the
            // specific heat / entropy (and risking L < E_max, which makes
            // (L - H) indefinite and corrupts the trajectory).
            //
            // SOTA recipe:
            //   1. Measure the true spectral bounds (E_min, E_max) with a
            //      short Lanczos (reuse the KPM estimator).
            //   2. Pick L from ONE resolution knob: Delta_beta_target.
            //        L_res  = <H>_inf + 2 / Delta_beta_target
            //        L_stab = E_max + buffer   (strictly above the ceiling)
            //        L      = max(L_res, L_stab)
            //   3. Size the iteration count INDEPENDENTLY so the trajectory
            //      still brackets beta_max = 1/temp_min:
            //        steps ~ beta_max * (L - E_min) / 2.
            // Resolution and cold-reach are thus decoupled; matvecs are
            // cheap so over-provisioning steps is affordable.
            // -------------------------------------------------------------
            const double dbeta_target = 0.02;  // internal quality knob
            const double beta_max = (opts.temp_min > 0.0)
                ? 1.0 / opts.temp_min : 100.0;

            double e_min_est = 0.0, e_max_est = 0.0;
            bool have_bounds = false;
            if (std::isfinite(opts.e_min_override)
                && std::isfinite(opts.e_max_override)
                && opts.e_max_override > opts.e_min_override) {
                // Caller-supplied bounds (e.g. estimated once on the
                // largest sector and reused across the Sz loop).
                e_min_est = opts.e_min_override;
                e_max_est = opts.e_max_override;
                have_bounds = true;
            } else if constexpr (std::is_same_v<B, ed::matvec::CpuBackend>) {
                // CPU lane: run the shared Lanczos spectral-bound
                // estimator on a host-pointer wrapper of the matvec.
                ed::kpm_dos::MatVec legacy_H =
                    [&matvec](const std::complex<double>* in,
                              std::complex<double>* out, int n) {
                        matvec(in, out, static_cast<std::size_t>(n));
                    };
                const std::uint64_t bdim = H.geometry().local_dim;
                std::mt19937 gen(opts.random_seed
                                     ? static_cast<unsigned>(opts.random_seed)
                                     : 0x9E3779B9u);
                try {
                    const int kry = static_cast<int>(
                        std::min<std::uint64_t>(
                            60, std::max<std::uint64_t>(bdim, 1)));
                    // Extreme Ritz values converge first and are robust
                    // without reorthogonalization; skip it (the estimator
                    // does not retain the Krylov basis anyway) to avoid a
                    // spurious "reorth skipped" warning and wasted work.
                    ed::kpm_dos::estimate_spectral_bounds(
                        legacy_H, bdim, kry, /*full_reorth=*/false,
                        /*reorth_freq=*/0, /*tol=*/1e-10,
                        gen, e_min_est, e_max_est);
                    have_bounds = std::isfinite(e_min_est)
                                && std::isfinite(e_max_est)
                                && e_max_est > e_min_est;
                } catch (...) {
                    have_bounds = false;
                }
            }

            double L_auto;
            if (have_bounds) {
                const double W      = e_max_est - e_min_est;
                const double e_high = 0.5 * (e_min_est + e_max_est);
                const double L_res  = e_high + 2.0 / std::max(dbeta_target, 1e-6);
                const double L_stab = e_max_est + std::max(0.05 * W, 1e-9);
                L_auto = std::max(L_res, L_stab);
            } else {
                // Non-CPU backend or failed estimate: resolution-driven
                // floor plus a conservative dim-based pad (never below
                // the historical behaviour for high-T runs).
                const double bandwidth_proxy = std::log2(
                    static_cast<double>(std::max<std::uint64_t>(
                        H.geometry().global_dim, 2)));
                L_auto = 2.0 / std::max(dbeta_target, 1e-6) + bandwidth_proxy;
            }

            // Expert override (HPhi-style ``LargeValue``): a finite,
            // positive ``energy_shift`` pins L and skips the auto-tune.
            kopts.large_value = (opts.energy_shift > 0.0)
                ? opts.energy_shift : L_auto;

            // Iteration budget contract:
            //   * krylov_dim == 0  -> AUTO: size the step count so the
            //     trajectory brackets beta_max = 1/temp_min, using
            //     steps ~ beta_max*(L - E_min)/2 (capped for safety).
            //   * krylov_dim  > 0  -> RESPECT the caller's value exactly
            //     (expert override / explicit ``max_iterations=...``).
            // This keeps the default surface clean (no knob needed -- the
            // auto path guarantees the requested coldest T is reached)
            // while never silently overriding an explicit request.
            const double e_low = have_bounds ? e_min_est : 0.0;
            const std::size_t reach_iters = static_cast<std::size_t>(
                std::ceil(beta_max
                          * (kopts.large_value - e_low) / 2.0)) + 16;
            constexpr std::size_t MTPQ_HARD_CAP = 200000;
            kopts.max_iter = (opts.krylov_dim == 0)
                ? std::max<std::size_t>(std::min(reach_iters, MTPQ_HARD_CAP), 1)
                : opts.krylov_dim;

            ed::thermal::MtpqResult kres;
#ifdef WITH_CUDA
            // fp32 single-GPU lane: half-footprint state vectors let the full
            // 2^32 Hilbert space run mTPQ on one 80 GB H100. Reuses the L /
            // max_iter auto-tune computed above (kopts); the driver manages its
            // own device memory (bypasses the double CudaBackend vectors).
            if (std::getenv("ED_MTPQ_VERBOSE")) {
                std::fprintf(stderr,
                    "[mtpq-lane] mtpq_fp32=%d supports_cuda_f32=%d -> %s\n",
                    (int)opts.mtpq_fp32, (int)H.supports_cuda_f32(),
                    (opts.mtpq_fp32 && H.supports_cuda_f32()) ? "FP32-GPU"
                                                              : "double");
            }
            if (opts.mtpq_fp32 && H.supports_cuda_f32()) {
                kres = ed::thermal::mtpq_f32(H, kopts);
            } else
#endif
            {
                kres = ed::thermal::mtpq_kernel<B>(
                    *backend_uptr, matvec, H.geometry().local_dim,
                    H.geometry().global_dim, kopts);
            }
            R.ground_state_energy = kres.energies.empty()
                ? 0.0 : *std::min_element(kres.energies.begin(),
                                          kres.energies.end());
            // SOTA: aggregate per-sample (beta_k, E_k, var_k) trajectories
            // into ThermodynamicData on the requested temperature grid.
            // Closes the gap where mTPQ via qed.thermal raised
            // ``RuntimeError: solver returned no thermodynamic data``.
            if (!R.thermo.temperatures.empty()) {
                // Pass the (sub)space dimension so the aggregator can
                // reconstruct ABSOLUTE entropy / free energy (ln(D)
                // baseline). This is what makes per-sector F_s a valid
                // Boltzmann weight for U(1)/Sz + spatial recombination.
                ThermodynamicData td = ed::thermal::compute_tpq_thermo_from_trajectories(
                    kres.sample_inv_temps, kres.sample_energies,
                    kres.sample_variances, R.thermo.temperatures,
                    static_cast<double>(H.geometry().global_dim));
                if (!td.energy.empty()) {
                    // Mirror the LTLM/FTLM contract: the caller's T grid
                    // is authoritative -- overwrite R.thermo with the
                    // aggregator's output (which uses our T grid).
                    R.thermo = std::move(td);
                }
            }
            // Pillar 1 (May 2026): lift the per-sample TPQ trajectory
            // + state-vector snapshots into the outer ThermalResult so
            // the uniform finalizer (below) can persist them.
            R.tpq_sample_betas     = std::move(kres.sample_inv_temps);
            R.tpq_sample_energies  = std::move(kres.sample_energies);
            R.tpq_sample_variances = std::move(kres.sample_variances);
            for (std::size_t s = 0; s < kres.state_snapshots.size(); ++s) {
                for (std::size_t p = 0; p < kres.state_snapshots[s].size(); ++p) {
                    auto& psi = kres.state_snapshots[s][p];
                    if (psi.empty()) continue;
                    TpqStateSnapshot snap;
                    snap.sample_index   = s;
                    snap.requested_beta = (p < opts.probe_betas.size())
                                              ? opts.probe_betas[p]
                                              : 0.0;
                    snap.effective_beta = kres.state_snapshot_betas[s][p];
                    snap.psi            = std::move(psi);
                    R.tpq_state_snapshots.push_back(std::move(snap));
                }
            }
        }, variant);
    } else if (opts.method == ThermalOptions::Method::FTLM) {
        // Phase E of the "Close CPU/GPU Gaps" plan (May 2026): the
        // FTLM kernel facade now dispatches on Backend type internally
        // (see ftlm_kernel.h, mirroring LTLM at the block below).
        // Both CpuBackend and CudaBackend are supported; MpiBackend /
        // MpiCudaBackend are explicitly rejected by the kernel until
        // cross-rank Lanczos post-processing is wired.
        std::visit([&](auto& backend_uptr) {
            using BPtr = std::decay_t<decltype(backend_uptr)>;
            using B = typename BPtr::element_type;
            constexpr bool is_cpu =
                std::is_same_v<B, ed::matvec::CpuBackend>;
#ifdef WITH_CUDA
            constexpr bool is_cuda =
                std::is_same_v<B, ed::matvec::CudaBackend>;
#else
            constexpr bool is_cuda = false;
#endif
            if constexpr (!(is_cpu || is_cuda)) {
                throw std::runtime_error(
                    "ed::thermal: FTLM requires a CpuBackend or "
                    "CudaBackend; distributed backends are not yet "
                    "wired. Pin BackendConstraints to route through "
                    "the CPU/CUDA lanes.");
            } else {
                ed::thermal::FtlmOptions kopts;
                kopts.num_samples = opts.num_samples;
                kopts.krylov_dim  = opts.krylov_dim ? opts.krylov_dim : 100;
                kopts.betas       = opts.betas;
                kopts.random_seed = opts.random_seed;
                kopts.output_dir  = opts.output_dir;
                auto matvec = H.template bind<B>();
                auto kres = ed::thermal::ftlm_kernel<B>(
                    *backend_uptr, matvec, H.geometry().local_dim,
                    H.geometry().global_dim, kopts);
                R.thermo.energy = std::move(kres.energy);
                R.thermo.specific_heat = std::move(kres.heat_capacity);
                R.thermo.entropy = std::move(kres.entropy);
            }
            R.ground_state_energy = R.thermo.energy.empty()
                ? 0.0
                : *std::min_element(R.thermo.energy.begin(), R.thermo.energy.end());
        }, variant);
    } else if (opts.method == ThermalOptions::Method::OFTLM) {
        // OFTLM (Morita-Tohyama): FTLM + N_V exact low-lying states + random
        // vectors orthogonalized against them. CPU-only lane -- consumes the
        // host term-matvec directly (see oftlm_kernel.h).
        auto host_mv = H.bind_cpu();
        auto apply_H = [&host_mv](const Complex* in, Complex* out, int n) {
            host_mv(in, out, static_cast<std::size_t>(n));
        };
        ed::thermal::OftlmOptions kopts;
        kopts.num_samples = opts.num_samples;
        kopts.krylov_dim  = opts.krylov_dim ? opts.krylov_dim : 100;
        kopts.num_exact   = opts.num_exact;
        kopts.betas       = opts.betas;
        kopts.random_seed = opts.random_seed;
        kopts.output_dir  = opts.output_dir;
        auto kres = ed::thermal::oftlm_cpu(
            apply_H, H.geometry().global_dim, kopts);
        R.thermo.energy        = std::move(kres.energy);
        R.thermo.specific_heat = std::move(kres.heat_capacity);
        R.thermo.entropy       = std::move(kres.entropy);
        R.ground_state_energy = R.thermo.energy.empty()
            ? 0.0
            : *std::min_element(R.thermo.energy.begin(), R.thermo.energy.end());
    } else if (opts.method == ThermalOptions::Method::LTLM) {
        // Phase E2 of the "Backend x Symmetries x Workflows" plan
        // (May 2026): the LTLM kernel now dispatches on Backend type
        // internally (see ltlm_kernel.h). Both CpuBackend and
        // CudaBackend are supported; MpiBackend / MpiCudaBackend are
        // explicitly rejected by the kernel until cross-rank Lanczos
        // post-processing is wired.
        std::visit([&](auto& backend_uptr) {
            using BPtr = std::decay_t<decltype(backend_uptr)>;
            using B = typename BPtr::element_type;
            constexpr bool is_cpu =
                std::is_same_v<B, ed::matvec::CpuBackend>;
#ifdef WITH_CUDA
            constexpr bool is_cuda =
                std::is_same_v<B, ed::matvec::CudaBackend>;
#else
            constexpr bool is_cuda = false;
#endif
            if constexpr (!(is_cpu || is_cuda)) {
                throw std::runtime_error(
                    "ed::thermal: LTLM requires a CpuBackend or "
                    "CudaBackend; distributed backends are not yet "
                    "wired. Pin BackendConstraints to route through "
                    "the CPU/CUDA lanes.");
            } else {
                ed::thermal::LtlmOptions kopts;
                kopts.num_samples         = opts.num_samples;
                kopts.krylov_dim          = opts.krylov_dim ? opts.krylov_dim
                                            : ed::defaults::kLtlmKrylovDim;
                kopts.ground_state_krylov = opts.krylov_dim ? opts.krylov_dim
                                            : ed::defaults::kFtlmKrylovDim;
                kopts.betas               = opts.betas;
                kopts.random_seed         = opts.random_seed;
                kopts.output_dir          = opts.output_dir;
                auto matvec = H.template bind<B>();
                auto kres = ed::thermal::ltlm_kernel<B>(
                    *backend_uptr, matvec, H.geometry().local_dim,
                    H.geometry().global_dim, kopts);
                R.thermo.energy = std::move(kres.energy);
                R.thermo.specific_heat = std::move(kres.heat_capacity);
                R.thermo.entropy = std::move(kres.entropy);
                R.ground_state_energy = kres.ground_state_energy;
            }
        }, variant);
    } else if (opts.method == ThermalOptions::Method::KpmDos) {
        // Phase E1 of the "Backend x Symmetries x Workflows" plan
        // (May 2026): the KPM-DOS kernel now dispatches on Backend
        // type internally (see kpm_dos_kernel.h). Both CpuBackend and
        // CudaBackend are supported; MpiBackend / MpiCudaBackend are
        // explicitly rejected by the kernel until the cross-rank
        // Hutchinson reduction is wired.
        std::visit([&](auto& backend_uptr) {
            using BPtr = std::decay_t<decltype(backend_uptr)>;
            using B = typename BPtr::element_type;
            constexpr bool is_cpu =
                std::is_same_v<B, ed::matvec::CpuBackend>;
#ifdef WITH_CUDA
            constexpr bool is_cuda =
                std::is_same_v<B, ed::matvec::CudaBackend>;
#else
            constexpr bool is_cuda = false;
#endif
            if constexpr (!(is_cpu || is_cuda)) {
                throw std::runtime_error(
                    "ed::thermal: KpmDos requires a CpuBackend or "
                    "CudaBackend; distributed backends are not yet "
                    "wired. Pin BackendConstraints to route through "
                    "the CPU/CUDA lanes.");
            } else {
                ed::thermal::KpmDosOptions kopts;
                kopts.betas = opts.betas;
                kopts.random_seed = opts.random_seed;
                // Wave B3 (May 2026): pass-through caller-supplied
                // spectral bounds. NaN means "estimate via Lanczos".
                kopts.e_min_override = opts.e_min_override;
                kopts.e_max_override = opts.e_max_override;
                // Closing-the-gap follow-up (May 2026): forward the
                // user's KPM knobs. ``0`` keeps the kernel default
                // so legacy CLI call sites with no knob set are
                // unaffected.
                if (opts.kpm_num_moments > 0) {
                    kopts.num_moments = opts.kpm_num_moments;
                }
                if (opts.kpm_num_random_vectors > 0) {
                    kopts.num_random_vectors = opts.kpm_num_random_vectors;
                }
                auto matvec = H.template bind<B>();
                auto kres = ed::thermal::kpm_dos_kernel<B>(
                    *backend_uptr, matvec, H.geometry().local_dim,
                    H.geometry().global_dim, kopts);
                R.thermo.energy = std::move(kres.energy);
                R.thermo.specific_heat = std::move(kres.specific_heat);
                R.thermo.entropy = std::move(kres.entropy);
                R.thermo.free_energy = std::move(kres.free_energy);
                R.ground_state_energy = kres.e_min_estimate;
                // Surface the raw DOS the method exists to produce (was
                // computed and discarded; only its thermodynamics escaped).
                R.dos_energies = std::move(kres.dos_grid_energies);
                R.dos_values   = std::move(kres.dos_grid_values);
            }
        }, variant);
    } else {
        throw std::runtime_error(
            "ed::thermal: unknown ThermalOptions::Method enumerator.");
    }

    // Surface unification follow-up (May 2026): populate
    // ``R.thermo.free_energy = E - T * S`` post-hoc so downstream
    // consumers see the full thermodynamic quintet (T, E, Cv, S, F).
    // The FTLM/LTLM/KpmDos kernel facades return E/Cv/S only; the
    // legacy ``finite_temperature_lanczos`` populated F from the
    // partition function (F = -T ln Z), which is mathematically
    // equivalent to E - T S once normalised. We use the latter form
    // here since the kernel does not expose ln Z.
    if (R.thermo.free_energy.empty()
        && !R.thermo.energy.empty()
        && R.thermo.energy.size() == R.thermo.entropy.size()
        && R.thermo.energy.size() == R.thermo.temperatures.size()) {
        R.thermo.free_energy.reserve(R.thermo.energy.size());
        for (std::size_t i = 0; i < R.thermo.energy.size(); ++i) {
            R.thermo.free_energy.push_back(
                R.thermo.energy[i]
                - R.thermo.temperatures[i] * R.thermo.entropy[i]);
        }
    }

    // -----------------------------------------------------------------
    // Pillar 1 of the "Save and DSSF Upgrades" plan (May 2026): uniform
    // thermal persistence finalizer. Mirrors the contract that lives in
    // ``ed::workflows::solve`` (l. 458-510): when ``opts.output_dir``
    // is set and the run is single-rank (the shared-file save is not
    // safe under MPI; per-rank files are written elsewhere), persist
    // the result to ``<output_dir>/ed_results.h5`` and surface the
    // resulting path via ``R.hdf5_path``.
    //
    // Method-conditional payload (user-confirmed policy):
    //   - mTPQ: the full per-sample (beta, E, var, step)
    //     trajectory (one row per kernel step, appended via
    //     ``HDF5IO::appendTPQThermodynamics``), plus state vectors at
    //     the betas closest to ``opts.probe_betas`` written via
    //     ``HDF5IO::saveTPQState``.
    //   - FTLM / LTLM / KPM_DOS: aggregated thermodynamic curves
    //     (``T, E, Cv, S, F``) only -- no state vectors.
    // -----------------------------------------------------------------
    // MPI-aware single-file emission (May 2026 follow-up): rank 0 owns
    // the aggregate thermo / TPQ-trajectory data (the kernels recombine
    // per-sample columns onto rank 0 before the orchestrator wraps the
    // result), so we run the finalizer there too. Slab-distributed TPQ
    // state vectors are *not* re-gathered here -- the per-rank
    // ``rank_<r>.h5`` files written by ``ed_distributed_main`` remain
    // the canonical location for those. The unified file therefore
    // ships:
    //   * mTPQ: per-sample trajectory rows (always on rank 0).
    //     The probe-beta state snapshots are written only when they are
    //     populated -- which is the serial case; in the distributed
    //     lane ``R.tpq_state_snapshots`` is empty on rank 0 and the
    //     loop is a no-op.
    //   * FTLM / LTLM / KPM_DOS: aggregated thermodynamic curves.
    if (!opts.output_dir.empty()
            && !HDF5IO::isDisabledOutputPath(opts.output_dir)
            && is_unified_writer(H.geometry())) {
        try {
            std::error_code ec;
            std::filesystem::create_directories(opts.output_dir, ec);
            const std::string h5_path =
                opts.output_dir + "/ed_results.h5";
            // Creates the file (or opens it) and lays down the
            // standard group skeleton (``/eigendata``, ``/tpq``,
            // ``/spectral_data`` ...).
            HDF5IO::createOrOpenFile(opts.output_dir);

            const bool is_tpq =
                (opts.method == ThermalOptions::Method::mTPQ
);
            if (is_tpq) {
                // (a) Per-sample trajectory rows.
                const std::size_t S =
                    std::min({R.tpq_sample_betas.size(),
                              R.tpq_sample_energies.size(),
                              R.tpq_sample_variances.size()});
                for (std::size_t s = 0; s < S; ++s) {
                    HDF5IO::ensureTPQSampleGroup(h5_path, s);
                    const auto& bs = R.tpq_sample_betas[s];
                    const auto& es = R.tpq_sample_energies[s];
                    const auto& vs = R.tpq_sample_variances[s];
                    const std::size_t K =
                        std::min({bs.size(), es.size(), vs.size()});
                    for (std::size_t k = 0; k < K; ++k) {
                        HDF5IO::TPQThermodynamicPoint pt;
                        pt.beta     = bs[k];
                        pt.energy   = es[k];
                        pt.variance = vs[k];
                        pt.doublon  = 0.0;
                        pt.step     = static_cast<std::uint64_t>(k);
                        HDF5IO::appendTPQThermodynamics(h5_path, s, pt);
                    }
                }
                // (b) State-vector snapshots at probe-betas.
                for (const auto& snap : R.tpq_state_snapshots) {
                    if (snap.psi.empty()) continue;
                    HDF5IO::ensureTPQSampleGroup(h5_path, snap.sample_index);
                    HDF5IO::saveTPQState(h5_path,
                                         snap.sample_index,
                                         snap.effective_beta,
                                         snap.psi,
                                         /*overwrite=*/true);
                }
            } else {
                // FTLM / LTLM / KPM-DOS: aggregated thermodynamic
                // curves. The kernel facades do not surface per-T
                // standard errors (those live on ``FTLMResults`` for
                // the legacy CLI path); the shared-file saver below
                // requires parallel arrays of equal length, so we
                // ship zero-valued error vectors of matching size.
                const std::size_t N = R.thermo.temperatures.size();
                const std::vector<double> zeros(N, 0.0);
                const char* label =
                    (opts.method == ThermalOptions::Method::FTLM)   ? "FTLM"
                  : (opts.method == ThermalOptions::Method::LTLM)   ? "LTLM"
                  : (opts.method == ThermalOptions::Method::KpmDos) ? "KPM_DOS"
                  : "thermal";
                HDF5IO::saveFTLMThermodynamics(
                    h5_path,
                    R.thermo.temperatures,
                    R.thermo.energy, zeros,
                    R.thermo.specific_heat, zeros,
                    R.thermo.entropy, zeros,
                    R.thermo.free_energy, zeros,
                    static_cast<std::uint64_t>(opts.num_samples),
                    std::string(label));
            }
            R.hdf5_path = h5_path;
        } catch (const std::exception& e) {
            // Non-fatal: skip persistence on I/O failure but surface
            // the cause via a backend note. The caller still gets the
            // in-memory ``R`` back; the empty ``R.hdf5_path`` flags
            // that no on-disk file was produced.
            std::cerr << "ed::thermal: persistence finalizer failed: "
                      << e.what() << std::endl;
        }
    }

    // Phase D (May 2026): truthful lane reporting -- pull the lane
    // label from the actual ``BackendVariant`` ``select_backend``
    // returned, NOT the host operator's memory_space. SectorView (and
    // every other host-resident operator that lazily wires a GPU
    // mirror through ``bind_cuda()``) reports ``Host`` memory_space
    // but ``select_backend`` picks ``CudaBackend`` when
    // ``allow_gpu=true`` and ``supports_device_matvec=true``. Reading
    // the variant directly is the only way the label can tell the
    // truth across all symmetry / non-symmetry workflows.
    R.backend.lane = ed::lane_label_from_variant(variant);
    const auto t1 = std::chrono::steady_clock::now();
    R.backend.wall_seconds =
        std::chrono::duration<double>(t1 - t0).count();
    return R;
}

SpectralResult spectral(const LinearOperator&                      H,
                         const std::vector<const LinearOperator*>&  observables,
                         SpectralOptions                            opts) {
    if (observables.empty()) {
        throw std::invalid_argument(
            "ed::spectral: at least one observable is required.");
    }
    using Complex = std::complex<double>;

    const ed::parallel::ThreadBudgetScope budget(
        ed::parallel::auto_threads_for_dim(H.geometry().local_dim));
    ed::parallel::pin_omp_threads_once();

    auto variant = select_backend(H.geometry(), opts.backend);

    // COMPLETION GUARANTEE (spectral lane). Working set: GroundStateCF runs an
    // inner GS Lanczos (with vectors) + a continued-fraction krylov_dim window;
    // the dynamical lanes keep an FTLM / KPM window. Plan it + refuse cleanly if
    // it would not fit, before allocating. allow_infeasible (force) opts out.
    // Leaf memory guard (planner feasibility pre-flight removed): GS-CF stores
    // the GS eigenvector + a continued-fraction Krylov window; the dynamical
    // lanes keep an FTLM/KPM window (~2x krylov, per-sector dim for symmetry).
    {
        const std::uint64_t D = H.global_dim();
        const std::uint64_t vecs = 2 * std::max<std::size_t>(opts.krylov_dim, 4);
        ed::core::guard_working_set(D * vecs * 16ull, "ed::spectral");
    }

    SpectralResult R;
    const auto t0 = std::chrono::steady_clock::now();

    std::vector<double> omega(opts.num_omega);
    const double step = (opts.omega_max - opts.omega_min) /
        static_cast<double>(std::max<std::size_t>(1, opts.num_omega - 1));
    for (std::size_t i = 0; i < opts.num_omega; ++i) {
        omega[i] = opts.omega_min + i * step;
    }
    R.omega = omega;

    if (opts.method == SpectralOptions::Method::GroundStateCF) {
        // Pillar 3 of the "Save and DSSF Upgrades" plan (May 2026):
        // resolve the CF seed.
        //   - ``opts.initial_state`` non-empty -> renormalise and use
        //     it directly (TPQ-to-CF, user-staged warm states, ...).
        //   - else                              -> run an inner Lanczos
        //     ground-state solve with ``compute_vectors=true`` and use
        //     the resulting eigenvector. This closes the original
        //     "shortcut": the previous implementation seeded the CF
        //     kernel with a random vector, which gave a fundamentally
        //     wrong S(omega) (it computed the average response over a
        //     thermal state at T = infinity rather than the GS dynamic
        //     structure factor).
        double E0 = 0.0;
        std::vector<Complex> seed_host(H.geometry().local_dim);
        if (!opts.initial_state.empty()) {
            if (opts.initial_state.size() != H.geometry().local_dim) {
                throw std::invalid_argument(
                    "ed::spectral: opts.initial_state size ("
                    + std::to_string(opts.initial_state.size())
                    + ") does not match H.geometry().local_dim ("
                    + std::to_string(H.geometry().local_dim) + ").");
            }
            seed_host = opts.initial_state;
            double sumsq = 0.0;
            for (const auto& z : seed_host) sumsq += std::norm(z);
            const double inv = (sumsq > 0.0)
                ? (1.0 / std::sqrt(sumsq)) : 1.0;
            for (auto& z : seed_host) z *= inv;
            // Caller-supplied seed: no GS energy estimate; use the
            // user's ``energy_shift`` directly (legacy ``0.0`` -> the
            // CF kernel does its own tridiag-based auto-detect).
            E0 = 0.0;
        } else {
            SolveOptions sopts;
            sopts.num_eigs        = 1;
            sopts.compute_vectors = true;     // need the GS vector for the CF seed
            sopts.tolerance       = 1e-12;
            sopts.backend         = opts.backend;
            sopts.method          = SolveMethod::Lanczos;
            auto gs = solve(H, sopts);
            E0 = gs.eigenvalues.empty() ? 0.0 : gs.eigenvalues.front();
            if (!gs.eigenvectors.has_value() || gs.eigenvectors->host.empty()
                    || gs.eigenvectors->host[0].size()
                       != H.geometry().local_dim) {
                throw std::runtime_error(
                    "ed::spectral: GroundStateCF could not extract a "
                    "host-side ground-state vector from the inner "
                    "solve. Distributed lanes are not yet wired -- pin "
                    "BackendConstraints to a CPU/GPU single-rank lane.");
            }
            seed_host = gs.eigenvectors->host[0];
            double sumsq = 0.0;
            for (const auto& z : seed_host) sumsq += std::norm(z);
            const double inv = (sumsq > 0.0)
                ? (1.0 / std::sqrt(sumsq)) : 1.0;
            for (auto& z : seed_host) z *= inv;
            // Guard the GS vector before it seeds the CF weight ||O|psi0>||^2
            // (unrefined GS => ~5% S(omega) error vs the exact reference).
            refine_gs_seed_host(H, seed_host, E0);
        }
        const double shift = (std::abs(opts.energy_shift) > 1e-14)
            ? opts.energy_shift : E0;
        std::visit([&](auto& backend_uptr) {
            using BPtr = std::decay_t<decltype(backend_uptr)>;
            using B = typename BPtr::element_type;
            // Stage the host seed into backend-resident memory; the kernel
            // expects pointers in the backend's address space.
            auto seed_backend =
                backend_uptr->make_zero_vector(H.geometry().local_dim);
            backend_uptr->copy_from_host(seed_host.data(), seed_backend.get(),
                                         H.geometry().local_dim);
            ed::observables::CfSpectralOptions cfopts;
            cfopts.krylov_dim   = opts.krylov_dim;
            cfopts.broadening   = opts.broadening;
            cfopts.energy_shift = shift;
            cfopts.global_n     = H.geometry().global_dim;
            auto matvec_h = H.template bind<B>();
            auto matvec_o = observables[0]->template bind<B>();
            auto kres = ed::observables::cf_spectral_kernel(
                *backend_uptr, matvec_h, matvec_o,
                H.geometry().local_dim,
                seed_backend.get(), R.omega, cfopts);
            R.S_real = std::move(kres.spectral_function);
        }, variant);
        R.S_imag.assign(opts.num_omega, 0.0);
    } else if (opts.method == SpectralOptions::Method::KpmDynamical) {
        // Pillar 4 of the "Save and DSSF Upgrades" plan (May 2026):
        // KPM Chebyshev expansion of `delta(omega - H)` against a
        // single seed. Promotes
        // `ed::observables::kpm_dynamical_correlator` to a first-class
        // SpectralOptions::Method on equal footing with GroundStateCF
        // / FtlmDynamical.
        //
        // Seed resolution mirrors the GroundStateCF branch:
        //   - ``opts.initial_state`` (renormalised) when non-empty
        //     (TPQ-to-KPM warm seeding);
        //   - else: inner Lanczos GS solve with compute_vectors=true.
        if (observables.size() < 1) {
            throw std::invalid_argument(
                "ed::spectral: KpmDynamical requires at least one "
                "observable.");
        }
        std::vector<Complex> seed_host(H.geometry().local_dim);
        if (!opts.initial_state.empty()) {
            if (opts.initial_state.size() != H.geometry().local_dim) {
                throw std::invalid_argument(
                    "ed::spectral: opts.initial_state size ("
                    + std::to_string(opts.initial_state.size())
                    + ") does not match H.geometry().local_dim ("
                    + std::to_string(H.geometry().local_dim) + ").");
            }
            seed_host = opts.initial_state;
        } else {
            SolveOptions sopts;
            sopts.num_eigs        = 1;
            sopts.compute_vectors = true;
            sopts.tolerance       = 1e-12;
            sopts.backend         = opts.backend;
            sopts.method          = SolveMethod::Lanczos;
            auto gs = solve(H, sopts);
            if (!gs.eigenvectors.has_value() || gs.eigenvectors->host.empty()
                    || gs.eigenvectors->host[0].size()
                       != H.geometry().local_dim) {
                throw std::runtime_error(
                    "ed::spectral: KpmDynamical could not extract a "
                    "host-side ground-state vector from the inner "
                    "solve. Distributed lanes are not yet wired -- "
                    "pin BackendConstraints to a CPU/GPU single-rank "
                    "lane.");
            }
            seed_host = gs.eigenvectors->host[0];
        }
        // Renormalise to absorb any sloppiness in the user seed.
        {
            double sumsq = 0.0;
            for (const auto& z : seed_host) sumsq += std::norm(z);
            const double inv = (sumsq > 0.0)
                ? (1.0 / std::sqrt(sumsq)) : 1.0;
            for (auto& z : seed_host) z *= inv;
        }
        // Guard the GS seed for the KPM correlator (same rationale as
        // GroundStateCF). Only when NOT user-seeded: a caller-staged warm
        // state (TPQ-to-KPM) is deliberately not a GS eigenvector.
        if (opts.initial_state.empty()) {
            double e0_dummy = 0.0;
            refine_gs_seed_host(H, seed_host, e0_dummy);
        }

        const LinearOperator& O1 = *observables.front();
        const LinearOperator& O2 = (observables.size() >= 2)
            ? *observables[1] : O1;

        ed::observables::KpmDynamicalOptions kopts;
        kopts.num_moments         = opts.kpm_moments;
        kopts.kernel              = (opts.kpm_kernel
                                       == SpectralOptions::KpmKernel::Jackson)
            ? ed::observables::KpmKernel::Jackson
            : ed::observables::KpmKernel::Lorentz;
        kopts.lorentz_lambda      = opts.kpm_lorentz_lambda;
        kopts.spectral_bound_buffer = 0.05;
        kopts.spectral_bounds_krylov = static_cast<int>(
            std::max<std::size_t>(opts.krylov_dim, 32));

        // Phase G of the "Close CPU/GPU Gaps" plan (May 2026):
        // dispatch on Backend type. CpuBackend keeps the legacy host
        // body (delegates to ``compute_kpm_ltlm_from_states`` -- the
        // single source of truth for the CPU lane's intermediate
        // diagnostics + future HDF5 hooks). CudaBackend routes
        // through ``detail::kpm_dynamical_kernel_via_backend``, a
        // fully device-resident Chebyshev recursion (M matvecs + M
        // dot products on the GPU, host-side kernel-coefficient +
        // spectral-function evaluation).
        std::visit([&](auto& backend_uptr) {
            using BPtr = std::decay_t<decltype(backend_uptr)>;
            using B = typename BPtr::element_type;
            constexpr bool is_cpu =
                std::is_same_v<B, ed::matvec::CpuBackend>;
#ifdef WITH_CUDA
            constexpr bool is_cuda =
                std::is_same_v<B, ed::matvec::CudaBackend>;
#else
            constexpr bool is_cuda = false;
#endif
            if constexpr (!(is_cpu || is_cuda)) {
                throw std::runtime_error(
                    "ed::spectral: KpmDynamical requires a CpuBackend "
                    "or CudaBackend; distributed backends are not yet "
                    "wired. Pin BackendConstraints to route through "
                    "the CPU/CUDA lanes.");
            } else if constexpr (is_cpu) {
                auto kres = ed::observables::kpm_dynamical_correlator(
                    *backend_uptr,
                    static_cast<const ed::matvec::MatVecOperator&>(H),
                    static_cast<const ed::matvec::MatVecOperator&>(O1),
                    static_cast<const ed::matvec::MatVecOperator&>(O2),
                    seed_host.data(),
                    H.geometry().local_dim,
                    R.omega,
                    kopts);
                R.omega  = std::move(kres.omega);
                R.S_real = std::move(kres.spectral_real);
                R.S_imag = std::move(kres.spectral_imag);
            } else {
                // CudaBackend: device-resident Chebyshev recursion.
                auto matvec_h = H.template bind<B>();
                auto matvec_a = O1.template bind<B>();
                auto matvec_b = O2.template bind<B>();
                auto kres = ed::observables::detail::
                    kpm_dynamical_kernel_via_backend(
                        *backend_uptr, matvec_h, matvec_a, matvec_b,
                        seed_host.data(),
                        H.geometry().local_dim,
                        R.omega, kopts);
                R.omega  = std::move(kres.omega);
                R.S_real = std::move(kres.spectral_real);
                R.S_imag = std::move(kres.spectral_imag);
            }
        }, variant);
        if (R.S_imag.size() != R.S_real.size()) {
            R.S_imag.assign(R.S_real.size(), 0.0);
        }
    } else {
        // Phase F of the "Close CPU/GPU Gaps" plan (May 2026):
        // finite-temperature dynamical correlator via FTLM CF-Lanczos.
        // The CPU lane keeps the legacy ``::compute_dynamical_correlation``
        // body (preserves intermediate HDF5 sample dumps when
        // ``output_dir`` is set + matches existing per-sample
        // diagnostics). The CUDA lane routes through
        // ``detail::ftlm_dynamical_kernel_via_backend``, which mirrors
        // the LTLM dual-backend pattern: device-resident O2 / O1 /
        // Lanczos, host-side tridiag diagonalisation + Lorentzian sum.
        // Distributed backends (MpiBackend / MpiCudaBackend) still
        // require the cross-rank reduction story for FTLM sample
        // averaging; pin BackendConstraints to a single-rank lane until
        // that lands.
        if (observables.size() < 1) {
            throw std::invalid_argument(
                "ed::spectral: FtlmDynamical requires at least one "
                "observable.");
        }
        const LinearOperator& O1 = *observables.front();
        const LinearOperator& O2 = (observables.size() >= 2)
            ? *observables[1] : O1;

        std::visit([&](auto& backend_uptr) {
            using BPtr = std::decay_t<decltype(backend_uptr)>;
            using B = typename BPtr::element_type;
            constexpr bool is_cpu =
                std::is_same_v<B, ed::matvec::CpuBackend>;
#ifdef WITH_CUDA
            constexpr bool is_cuda =
                std::is_same_v<B, ed::matvec::CudaBackend>;
#else
            constexpr bool is_cuda = false;
#endif
            if constexpr (!(is_cpu || is_cuda)) {
                throw std::runtime_error(
                    "ed::spectral: FtlmDynamical requires a CpuBackend "
                    "or CudaBackend; distributed backends are not yet "
                    "wired. Pin BackendConstraints to route through "
                    "the CPU/CUDA lanes.");
            } else if constexpr (is_cpu) {
                std::function<void(const Complex*, Complex*, int)> H_apply =
                    [&H](const Complex* in, Complex* out, int n) {
                        H.apply(in, out, static_cast<std::size_t>(n));
                    };
                std::function<void(const Complex*, Complex*, int)> O1_apply =
                    [&O1](const Complex* in, Complex* out, int n) {
                        O1.apply(in, out, static_cast<std::size_t>(n));
                    };
                std::function<void(const Complex*, Complex*, int)> O2_apply =
                    [&O2](const Complex* in, Complex* out, int n) {
                        O2.apply(in, out, static_cast<std::size_t>(n));
                    };

                DynamicalResponseParameters params;
                params.krylov_dim               =
                    static_cast<std::uint64_t>(opts.krylov_dim);
                params.broadening               = opts.broadening;
                params.tolerance                = 1e-10;
                params.full_reorthogonalization = true;
                params.random_seed              = 0;

                const auto legacy = ::compute_dynamical_correlation(
                    H_apply, O1_apply, O2_apply,
                    static_cast<std::uint64_t>(H.geometry().local_dim),
                    params,
                    opts.omega_min, opts.omega_max,
                    static_cast<std::uint64_t>(opts.num_omega),
                    /*temperature=*/0.0,
                    opts.output_dir,
                    opts.energy_shift);

                R.S_real = legacy.spectral_function;
                R.S_imag = legacy.spectral_function_imag;
            } else {
                // CudaBackend lane: backend-templated body.
                ed::observables::FtlmDynamicalOptions kopts;
                kopts.krylov_dim   = opts.krylov_dim;
                kopts.num_samples  = (opts.kpm_moments > 0 ? 1u : 1u);
                kopts.broadening   = opts.broadening;
                kopts.temperature  = 0.0;
                kopts.energy_shift = opts.energy_shift;
                kopts.tolerance    = 1e-10;
                kopts.random_seed  = 0;
                kopts.global_n     = H.geometry().global_dim;
                auto matvec_h  = H.template bind<B>();
                auto matvec_o1 = O1.template bind<B>();
                auto matvec_o2 = O2.template bind<B>();
                auto kres = ed::observables::detail::
                    ftlm_dynamical_kernel_via_backend(
                        *backend_uptr,
                        matvec_h, matvec_o1, matvec_o2,
                        H.geometry().local_dim, R.omega, kopts);
                R.S_real = std::move(kres.spectral_real);
                R.S_imag = std::move(kres.spectral_imag);
            }
        }, variant);

        if (R.S_imag.size() != R.S_real.size()) {
            R.S_imag.assign(R.S_real.size(), 0.0);
        }
    }

    R.errors_real.assign(R.S_real.size(), 0.0);
    R.errors_imag.assign(R.S_imag.size(), 0.0);
    // Phase D (May 2026): truthful lane reporting (same rationale as
    // ``thermal()`` above) -- ``H.geometry().is_device()`` reads
    // ``false`` for SectorView yet ``select_backend`` may have picked
    // CudaBackend through ``supports_device_matvec=true``.
    R.backend.lane = ed::lane_label_from_variant(variant);
    const auto t1 = std::chrono::steady_clock::now();
    R.backend.wall_seconds =
        std::chrono::duration<double>(t1 - t0).count();

    // -----------------------------------------------------------------
    // "Universal save contract" follow-up (May 2026): uniform spectral
    // persistence finalizer. Mirrors the thermal finalizer above --
    // when the caller supplies a real ``output_dir`` and the current
    // process is the unified-file writer (every rank for the serial
    // lane, rank 0 only for MPI lanes), lay down the standard group
    // skeleton and persist (omega, S_real, S_imag, errors_real,
    // errors_imag) under ``/dynamical/<method>/...`` of
    // ``<output_dir>/ed_results.h5``.
    //
    // Method labels:
    //   GroundStateCF -> "ground_state_cf"
    //   FtlmDynamical -> "ftlm_dynamical"
    //   KpmDynamical  -> "kpm_dynamical"
    //
    // The legacy ``FtlmDynamical`` branch already passes
    // ``opts.output_dir`` to ``compute_dynamical_correlation`` which
    // writes ``/ftlm/samples/dynamical/...`` when
    // ``store_intermediate=true``. The uniform finalizer is
    // complementary: it ships the aggregated S(omega) at a stable,
    // method-tagged path regardless of which kernel produced it.
    //
    // MPI: ``R.omega`` / ``R.S_*`` are reduced onto every rank by the
    // kernels (CF / KPM use rank-local matvecs + ``MPI_Allreduce``;
    // FTLM averages locally and reduces at the orchestrator level),
    // so rank 0 holds the final aggregate. Per-rank ``rank_<r>.h5``
    // files (when produced by the legacy CLI) remain the canonical
    // location for any rank-local intermediates.
    // -----------------------------------------------------------------
    if (!opts.output_dir.empty()
            && !HDF5IO::isDisabledOutputPath(opts.output_dir)
            && is_unified_writer(H.geometry())) {
        try {
            std::error_code ec;
            std::filesystem::create_directories(opts.output_dir, ec);
            const std::string h5_path =
                opts.output_dir + "/ed_results.h5";
            HDF5IO::createOrOpenFile(opts.output_dir);

            const char* label =
                (opts.method == SpectralOptions::Method::GroundStateCF)
                    ? "ground_state_cf"
              : (opts.method == SpectralOptions::Method::KpmDynamical)
                    ? "kpm_dynamical"
              : (opts.method == SpectralOptions::Method::FtlmDynamical)
                    ? "ftlm_dynamical"
              : "spectral";

            HDF5IO::saveDynamicalResponseFull(
                h5_path,
                std::string(label),
                R.omega,
                R.S_real,
                R.S_imag,
                R.errors_real,
                R.errors_imag,
                /*total_samples=*/static_cast<std::uint64_t>(
                    std::max<std::size_t>(1, opts.num_samples)),
                /*temperature=*/0.0);

            R.hdf5_path = h5_path;
        } catch (const std::exception& e) {
            // Non-fatal: skip persistence on I/O failure but surface
            // the cause via stderr. The caller still gets the in-memory
            // ``R`` back; an empty ``R.hdf5_path`` flags that no
            // on-disk file was produced.
            std::cerr << "ed::spectral: persistence finalizer failed: "
                      << e.what() << "\n";
        }
    }
    return R;
}

}  // namespace ed::workflows
