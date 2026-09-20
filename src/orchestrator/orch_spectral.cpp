// =============================================================================
// src/orchestrator/orch_spectral.cpp -- ed::workflows::spectral and its lanes
// (GroundStateCF / KpmDynamical / FtlmDynamical) plus the host ground-state
// seed refinement they share.
// Part of the workflow orchestrator; see orchestrator_internal.h for the
// file map.
// =============================================================================

#include "orchestrator_internal.h"

namespace ed::workflows {

using namespace orch_detail;

namespace {

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
    // Audit fix (2026-07-30): force the matrix-free apply for this bind.
    // The refine performs at most ~300 matvecs on a FRESH function-local
    // backend, so a CSR assembled here can never amortise -- and at
    // dim 2.7e6 (N=24 fixed-Sz, under the 2^22 lane cutoff) the build
    // alone cost ~47 s of the GPU spectral gate's wall. Explicit env
    // vars (ED_CSR_FORCE / ED_CSR_DIM_MAX) still take precedence over
    // this scoped override, per csr_policy_hook.h.
    ed::planner::ScopedCsrOverride no_csr(
        ed::planner::CsrOverride::MatrixFree);
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
    // Audit fix (2026-07-30): early-exit when the smallest Ritz value
    // stabilises instead of always burning the full 300 FullCGS2
    // iterations. Profiled at N=24 (dim 2.7e6): the flat-300 refine cost
    // 351 s (94% CPU reorthogonalisation) and was the dominant term in
    // the GPU spectral lane's 407 s wall (kill-hash gate budget: 30 s).
    // Ritz-stabilisation at 1e-13 checked every 5 steps reproduces the
    // same refined vector for any seed the inner solve hands over while
    // stopping as soon as the tower has converged.
    kopts.convergence_check =
        ed::krylov::make_smallest_ritz_convergence(1, 1e-13);
    kopts.convergence_check_interval = 5;
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

SpectralResult spectral(const LinearOperator&                      H,
                         const std::vector<const LinearOperator*>&  observables,
                         SpectralOptions                            opts) {
    require_hermitian_input(H, "ed::spectral");
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
        // Stage 12g (SU(2) rollout): label the CF source state's total
        // spin when the caller installed a labeler (the bindings do so
        // for SU(2)-invariant H). Works for the inner-solve GS AND a
        // caller-staged initial_state alike.
        if (opts.su2_labeler) {
            R.gs_two_S = opts.su2_labeler(seed_host.data(),
                                          seed_host.size(), &R.gs_s2);
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
            // Convergence bookkeeping (2026-09-11): the CF change between
            // half and full Krylov depth; > 5 % means "raise krylov_dim".
            R.krylov.iters_done    = kres.tridiag_size;
            R.krylov.residual_norm = kres.convergence_change;
            R.krylov.converged     = kres.convergence_change < 0.05;
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
        // Family-3 consolidation (audit 2026-07-31): ONE backend-generic
        // arm. The old split kept the legacy ``::compute_dynamical_
        // correlation`` on CPU "for HDF5 sample dumps" that were never
        // enabled (store_intermediate defaulted false and was never
        // set), silently used the legacy default of 40 samples on CPU
        // while the CUDA arm hardcoded 1 sample regardless of the
        // caller's request, and duplicated tolerance/seed constants per
        // arm (they had already drifted once). Both lanes now run
        // ``ftlm_dynamical_kernel_via_backend`` -- gated equivalent to
        // the legacy body at matching T (~5 decimals, Family-3 step 3)
        // -- and honour ``opts.num_samples``.
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
            } else {
                ed::observables::FtlmDynamicalOptions kopts;
                kopts.krylov_dim   = opts.krylov_dim;
                kopts.num_samples  = std::max<std::size_t>(
                    1, opts.num_samples);
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
