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
//     ed::thermal   -> tpq_kernel<Backend>  (mTPQ, cTPQ)
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

#include <ed/krylov/block_lanczos_kernel.h>
#include <ed/krylov/krylov_schur_kernel.h>
#include <ed/krylov/lanczos_kernel.h>
#include <ed/krylov/ritz_convergence.h>
#include <ed/matvec/backends/cpu_backend.h>
#include <ed/observables/cf_spectral_kernel.h>
#include <ed/parallel/numa.h>            // pin_omp_threads_once
#include <ed/parallel/thread_budget.h>   // auto_threads_for_dim + ThreadBudgetScope
#include <ed/solvers/TPQ.h>      // compute_tpq_thermo_from_trajectories aggregator
#include <ed/solvers/lanczos.h>  // FullDiag fallback (zheevd on the dense matrix)
#include <ed/thermal/ctpq_kernel.h>
#include <ed/thermal/ftlm_kernel.h>
#include <ed/thermal/kpm_dos_kernel.h>
#include <ed/thermal/ltlm_kernel.h>
#include <ed/thermal/mtpq_kernel.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>      // getenv (Wave 1.1 real-H fast-path opt-out)
#include <iostream>
#include <random>
#include <stdexcept>
#include <type_traits>  // std::is_same_v (Wave 1.1)
#include <variant>

namespace ed::workflows {

namespace {

// ---------------------------------------------------------------------------
// Default heuristics for "method == Auto".
// ---------------------------------------------------------------------------
SolveMethod auto_solve_method(std::uint64_t global_dim,
                              std::size_t   num_eigs) {
    // Full diag below 2^12 is competitive.
    if (global_dim <= (1ULL << 12)) return SolveMethod::FullDiag;
    if (num_eigs == 1) return SolveMethod::Lanczos;
    if (num_eigs <= 8)  return SolveMethod::BlockLanczos;
    return SolveMethod::KrylovSchur;
}

std::size_t auto_max_iter(std::uint64_t global_dim,
                          std::size_t   num_eigs,
                          std::size_t   user_max_iter) {
    if (user_max_iter > 0) return user_max_iter;
    const std::size_t floor_iters = 2 * num_eigs + 30;
    const std::size_t cap_iters   = std::min<std::size_t>(
        floor_iters, static_cast<std::size_t>(std::min<std::uint64_t>(
            global_dim, 2000)));
    return std::max(floor_iters, cap_iters);
}

template <typename Backend>
GroundStateResult solve_on(Backend& be,
                           const LinearOperator& H,
                           const SolveOptions& opts) {
    using Complex = std::complex<double>;
    const auto geom = H.geometry();
    auto matvec = H.template bind<Backend>();

    GroundStateResult R;
    const std::size_t max_iter = auto_max_iter(
        geom.global_dim, opts.num_eigs, opts.max_iter);
    const SolveMethod method = (opts.method == SolveMethod::Auto)
        ? auto_solve_method(geom.global_dim, opts.num_eigs)
        : opts.method;

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
                const auto t1 = std::chrono::steady_clock::now();
                R.backend.wall_seconds =
                    std::chrono::duration<double>(t1 - t0).count();
                R.backend.notes.emplace_back(
                    "dispatch", "lanczos_real (Wave 1.1 real-H fast path)");
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
            ed::distributed::kernel::solve_tridiag_with_eigenvectors(
                kres.alpha, kres.beta, kres.alpha.size(), evals, weights, evec_coeffs);
        } else {
            evals = ed::distributed::kernel::solve_tridiag(
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
        R.krylov.iters_done = kres.blocks_built;
        R.krylov.converged  = kres.converged;
    } else if (method == SolveMethod::KrylovSchur) {
        ed::krylov::KrylovSchurOptions kopts;
        kopts.num_eigs        = opts.num_eigs;
        kopts.max_iter        = max_iter;
        kopts.tolerance       = opts.tolerance;
        kopts.compute_vectors = opts.compute_vectors;
        kopts.global_n        = geom.global_dim;
        kopts.output_dir      = opts.output_dir;
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
            std::function<void(const Complex*, Complex*, int)> Hv =
                [&](const Complex* in, Complex* out, int n) {
                    matvec(in, out, static_cast<std::size_t>(n));
                };
            std::vector<double> eigs;
            full_diagonalization(Hv, geom.local_dim, opts.num_eigs, eigs,
                                 opts.output_dir,
                                 opts.compute_vectors);
            const std::size_t n_keep = std::min<std::size_t>(
                opts.num_eigs, eigs.size());
            R.eigenvalues.assign(eigs.begin(), eigs.begin() + n_keep);
            R.krylov.iters_done = 0;
            R.krylov.converged  = true;
        }
    }

    R.backend.lane = (geom.is_distributed() ? "mpi" : "cpu");
    if (geom.is_device()) R.backend.lane = geom.is_distributed() ? "mpi_gpu" : "gpu";
    R.backend.mpi_size = 1;
    const auto t1 = std::chrono::steady_clock::now();
    R.backend.wall_seconds =
        std::chrono::duration<double>(t1 - t0).count();
    return R;
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
    // All five lanes are wired: mTPQ + cTPQ dispatch through the unified
    // `tpq_kernel` via the Phase 2.4 facades; FTLM / LTLM / KpmDos
    // dispatch through their own `*_kernel<Backend>` templates (CPU
    // implementations today, GPU when the kernels migrate). The variant
    // visit at each lane keeps the dispatch backend-agnostic.
    using Complex = std::complex<double>;
    const ed::parallel::ThreadBudgetScope budget(
        ed::parallel::auto_threads_for_dim(H.geometry().local_dim));
    ed::parallel::pin_omp_threads_once();

    auto variant = select_backend(H.geometry(), opts.backend);

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

    if (opts.method == ThermalOptions::Method::mTPQ) {
        std::visit([&](auto& backend_uptr) {
            using BPtr = std::decay_t<decltype(backend_uptr)>;
            using B = typename BPtr::element_type;
            ed::thermal::MtpqOptions kopts;
            kopts.num_samples = opts.num_samples;
            kopts.max_iter    = opts.krylov_dim ? opts.krylov_dim : 1000;
            kopts.random_seed = opts.random_seed;
            kopts.output_dir  = opts.output_dir;
            // SOTA large_value pick (May 2026): the (L*I - H) iteration
            // gives effective inverse temperature
            //   beta_k = 2 k / (L - E_k).
            // To cover the user-requested coldest temperature
            // ``temp_min`` (== 1/beta_max_target) with ``max_iter``
            // steps, we need
            //   L = 2 * max_iter / beta_max_target + |E|_typical.
            // Estimate |E|_typical from the system bandwidth proxy
            // ``log2(global_dim)`` (which equals the number of spins
            // for unsymmetrised spin-1/2 chains; for symmetrised /
            // FixedSz operators it's the symmetrised dim, which is a
            // conservative upper bound). Cap at the historical 1e5
            // default so we never go below it for high-T runs.
            const double beta_target = (opts.temp_min > 0.0)
                ? 1.0 / opts.temp_min : 100.0;
            const double bandwidth_proxy = std::log2(
                static_cast<double>(std::max<std::uint64_t>(
                    H.geometry().global_dim, 2)));
            const double L_auto = 2.0 * static_cast<double>(kopts.max_iter)
                                    / std::max(beta_target, 1e-6)
                                + bandwidth_proxy;
            kopts.large_value = std::max(L_auto, 1.0);
            auto matvec = H.template bind<B>();
            auto kres = ed::thermal::mtpq_kernel<B>(
                *backend_uptr, matvec, H.geometry().local_dim,
                H.geometry().global_dim, kopts);
            R.ground_state_energy = kres.energies.empty()
                ? 0.0 : *std::min_element(kres.energies.begin(),
                                          kres.energies.end());
            // SOTA: aggregate per-sample (beta_k, E_k, var_k) trajectories
            // into ThermodynamicData on the requested temperature grid.
            // Closes the gap where mTPQ via qed.thermal raised
            // ``RuntimeError: solver returned no thermodynamic data``.
            if (!R.thermo.temperatures.empty()) {
                ThermodynamicData td = compute_tpq_thermo_from_trajectories(
                    kres.sample_inv_temps, kres.sample_energies,
                    kres.sample_variances, R.thermo.temperatures);
                if (!td.energy.empty()) {
                    // Mirror the LTLM/FTLM contract: the caller's T grid
                    // is authoritative -- overwrite R.thermo with the
                    // aggregator's output (which uses our T grid).
                    R.thermo = std::move(td);
                }
            }
        }, variant);
    } else if (opts.method == ThermalOptions::Method::cTPQ) {
        std::visit([&](auto& backend_uptr) {
            using BPtr = std::decay_t<decltype(backend_uptr)>;
            using B = typename BPtr::element_type;
            ed::thermal::CtpqOptions kopts;
            kopts.num_samples  = opts.num_samples;
            // SOTA beta_max pick (May 2026): if the caller asked for
            // a coldest temperature ``temp_min`` smaller than the
            // default beta_max would cover, extend beta_max so the
            // trajectory actually brackets every target beta.
            const double user_beta_max = (opts.temp_min > 0.0)
                ? 1.0 / opts.temp_min : opts.beta_max;
            kopts.beta_max     = std::max(opts.beta_max, 1.1 * user_beta_max);
            kopts.delta_beta   = opts.delta_beta;
            kopts.taylor_order = opts.taylor_order;
            // Closing-the-symmetry-gap follow-up (May 2026): respect the
            // user's ``max_iterations`` knob as a hard CAP on the
            // Taylor step count. Without this cap, cTPQ silently runs
            // ``beta_max / delta_beta`` steps (default 20.0 / 0.01 =
            // 2000 steps) even when the user asked for ``max_iterations
            // = 200``, which inflates the per-sector matvec count by
            // 10x compared to mTPQ on the same call. The orchestrator
            // pipes ``max_iterations`` into ``opts.krylov_dim`` for
            // both mTPQ and cTPQ; mTPQ already honours it (l.550
            // above); this brings cTPQ to parity. ``0`` means "no
            // explicit cap" which preserves legacy behaviour.
            if (opts.krylov_dim > 0 && opts.delta_beta > 0.0) {
                const double cap_beta = static_cast<double>(opts.krylov_dim)
                                       * opts.delta_beta;
                if (cap_beta < kopts.beta_max) {
                    kopts.beta_max = cap_beta;
                }
            }
            kopts.random_seed  = opts.random_seed;
            kopts.output_dir   = opts.output_dir;
            auto matvec = H.template bind<B>();
            auto kres = ed::thermal::ctpq_kernel<B>(
                *backend_uptr, matvec, H.geometry().local_dim,
                H.geometry().global_dim, kopts);
            R.ground_state_energy = kres.energies.empty()
                ? 0.0 : *std::min_element(kres.energies.begin(),
                                          kres.energies.end());
            if (!R.thermo.temperatures.empty()) {
                ThermodynamicData td = compute_tpq_thermo_from_trajectories(
                    kres.sample_inv_temps, kres.sample_energies,
                    kres.sample_variances, R.thermo.temperatures);
                if (!td.energy.empty()) {
                    R.thermo = std::move(td);
                }
            }
        }, variant);
    } else if (opts.method == ThermalOptions::Method::FTLM) {
        // Wave B (Full unified-interface collapse, May 2026): the
        // FTLM/LTLM/KpmDos kernel facades are static_assert-gated to
        // `CpuBackend` (see ftlm_kernel.h). Filter to that variant
        // alternative explicitly so the std::visit doesn't try to
        // instantiate the template against CudaBackend / MpiBackend.
        std::visit([&](auto& backend_uptr) {
            using BPtr = std::decay_t<decltype(backend_uptr)>;
            using B = typename BPtr::element_type;
            if constexpr (!std::is_same_v<B, ed::matvec::CpuBackend>) {
                throw std::runtime_error(
                    "ed::thermal: FTLM lane requires a CpuBackend "
                    "today; the inner driver is host-side. Pin "
                    "BackendConstraints::allow_gpu = false / "
                    "allow_mpi = false to route through the CPU "
                    "lane explicitly.");
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
                kopts.krylov_dim          = opts.krylov_dim ? opts.krylov_dim : 200;
                kopts.ground_state_krylov = opts.krylov_dim ? opts.krylov_dim : 100;
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

    R.backend.lane = (H.geometry().is_distributed() ? "mpi" : "cpu");
    if (H.geometry().is_device()) {
        R.backend.lane = H.geometry().is_distributed() ? "mpi_gpu" : "gpu";
    }
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
        // Compute ground state via Lanczos on H; pass the GS as the seed
        // to cf_spectral_kernel.
        SolveOptions sopts;
        sopts.num_eigs       = 1;
        sopts.compute_vectors = false;
        sopts.tolerance       = 1e-12;
        sopts.backend         = opts.backend;
        sopts.method          = SolveMethod::Lanczos;  // skip FullDiag fallback
        auto gs = solve(H, sopts);
        const double E0 = gs.eigenvalues.empty() ? 0.0 : gs.eigenvalues.front();
        const double shift = (std::abs(opts.energy_shift) > 1e-14)
            ? opts.energy_shift : E0;

        // The seed must be the ground state. For the orchestrator's first
        // landing we keep this CPU-only and route the legacy
        // `compute_ground_state_dssf` for richer cases. The CF kernel here
        // produces S(omega) from a NORMALISED random seed (mirroring the
        // "compute_dynamical_correlation_state_cf" code path that takes an
        // arbitrary input state). Use it that way for now; a future
        // tightening will plumb the actual eigenvector through.
        std::vector<Complex> seed_host(H.geometry().local_dim);
        {
            std::mt19937_64 gen(0xC0FFEEULL);
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
    } else {
        // FtlmDynamical lane (Wave A4 -- Full unified-interface
        // collapse, May 2026): finite-temperature dynamical correlator
        // via the legacy FTLM CF-Lanczos routine. We delegate to the
        // existing `compute_dynamical_correlation` body since it is the
        // single source of truth for the FTLM dynamical pipeline and
        // already handles multi-sample averaging + per-temperature
        // weighting. The orchestrator's first landing keeps the
        // temperature axis at the legacy default (T = 0); per-T
        // scanning is plumbed through the CLI / Python entry points
        // until the SpectralOptions struct grows a `temperatures`
        // vector (Wave A5).
        if (observables.size() < 1) {
            throw std::invalid_argument(
                "ed::spectral: FtlmDynamical requires at least one "
                "observable.");
        }
        const LinearOperator& O1 = *observables.front();
        const LinearOperator& O2 = (observables.size() >= 2)
            ? *observables[1] : O1;

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
        if (R.S_imag.size() != R.S_real.size()) {
            R.S_imag.assign(R.S_real.size(), 0.0);
        }
    }

    R.errors_real.assign(opts.num_omega, 0.0);
    R.errors_imag.assign(opts.num_omega, 0.0);
    R.backend.lane = (H.geometry().is_distributed() ? "mpi" : "cpu");
    if (H.geometry().is_device()) {
        R.backend.lane = H.geometry().is_distributed() ? "mpi_gpu" : "gpu";
    }
    const auto t1 = std::chrono::steady_clock::now();
    R.backend.wall_seconds =
        std::chrono::duration<double>(t1 - t0).count();
    return R;
}

}  // namespace ed::workflows
