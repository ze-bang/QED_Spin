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
#include <ed/solvers/lanczos.h>  // FullDiag fallback (zheevd on the dense matrix)
#include <ed/thermal/ctpq_kernel.h>
#include <ed/thermal/ftlm_kernel.h>
#include <ed/thermal/kpm_dos_kernel.h>
#include <ed/thermal/ltlm_kernel.h>
#include <ed/thermal/mtpq_kernel.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <random>
#include <stdexcept>
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
        ed::krylov::LanczosKernelOptions kopts;
        kopts.max_iter      = max_iter;
        kopts.reorth        = ed::krylov::ReorthPolicy::LocalDGKS3;
        kopts.keep_basis    = opts.compute_vectors;
        kopts.dim_cap       = static_cast<std::size_t>(geom.global_dim);
        // Wire in Ritz-value early exit so the orchestrator matches the
        // legacy CPU `lanczos()` convergence behaviour (otherwise the
        // kernel always runs to `max_iter` -- 5-10x slower on small
        // problems and a noticeable hit even on large ones).
        kopts.convergence_check =
            ed::krylov::make_smallest_ritz_convergence(opts.num_eigs,
                                                       opts.tolerance,
                                                       /*min_iters=*/0);
        // Check every iteration (matches the legacy CPU `lanczos()` body);
        // every-5 introduced ~5 extra iters of overhead before triggering.
        kopts.convergence_check_interval = 1;
        auto kres = ed::krylov::lanczos_kernel(be, matvec, geom.local_dim,
                                               seed, kopts);
        // Solve the small (m x m) real-symmetric tridiagonal for the
        // lowest `num_eigs` eigenvalues. When the caller didn't request
        // eigenvectors, use the eigenvalues-only Eigen path -- the
        // legacy `solve_tridiag_with_eigenvectors` did the full eigen
        // problem unconditionally, which is ~2-3x slower for the
        // common num_eigs=1 + compute_vectors=false workflow.
        std::vector<double> evals;
        if (opts.compute_vectors) {
            std::vector<double> weights, evecs;
            ed::distributed::kernel::solve_tridiag_with_eigenvectors(
                kres.alpha, kres.beta, kres.alpha.size(), evals, weights, evecs);
        } else {
            evals = ed::distributed::kernel::solve_tridiag(
                kres.alpha, kres.beta, kres.alpha.size());
        }
        const std::size_t n_keep = std::min<std::size_t>(opts.num_eigs, evals.size());
        R.eigenvalues.assign(evals.begin(), evals.begin() + n_keep);
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

    ThermalResult R;
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
            auto matvec = H.template bind<B>();
            auto kres = ed::thermal::mtpq_kernel<B>(
                *backend_uptr, matvec, H.geometry().local_dim,
                H.geometry().global_dim, kopts);
            R.ground_state_energy = kres.energies.empty()
                ? 0.0 : *std::min_element(kres.energies.begin(), kres.energies.end());
        }, variant);
    } else if (opts.method == ThermalOptions::Method::cTPQ) {
        std::visit([&](auto& backend_uptr) {
            using BPtr = std::decay_t<decltype(backend_uptr)>;
            using B = typename BPtr::element_type;
            ed::thermal::CtpqOptions kopts;
            kopts.num_samples  = opts.num_samples;
            kopts.beta_max     = opts.beta_max;
            kopts.delta_beta   = opts.delta_beta;
            kopts.taylor_order = opts.taylor_order;
            kopts.random_seed  = opts.random_seed;
            kopts.output_dir   = opts.output_dir;
            auto matvec = H.template bind<B>();
            auto kres = ed::thermal::ctpq_kernel<B>(
                *backend_uptr, matvec, H.geometry().local_dim,
                H.geometry().global_dim, kopts);
            R.ground_state_energy = kres.energies.empty()
                ? 0.0 : *std::min_element(kres.energies.begin(), kres.energies.end());
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
        // Wave B: LTLM is CPU-only today (static_assert in ltlm_kernel.h).
        std::visit([&](auto& backend_uptr) {
            using BPtr = std::decay_t<decltype(backend_uptr)>;
            using B = typename BPtr::element_type;
            if constexpr (!std::is_same_v<B, ed::matvec::CpuBackend>) {
                throw std::runtime_error(
                    "ed::thermal: LTLM lane requires a CpuBackend "
                    "today; pin BackendConstraints to route via the "
                    "CPU lane.");
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
        // Wave B: KpmDos is CPU-only today (static_assert in kpm_dos_kernel.h).
        std::visit([&](auto& backend_uptr) {
            using BPtr = std::decay_t<decltype(backend_uptr)>;
            using B = typename BPtr::element_type;
            if constexpr (!std::is_same_v<B, ed::matvec::CpuBackend>) {
                throw std::runtime_error(
                    "ed::thermal: KpmDos lane requires a CpuBackend "
                    "today.");
            } else {
                ed::thermal::KpmDosOptions kopts;
                kopts.betas = opts.betas;
                kopts.random_seed = opts.random_seed;
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
