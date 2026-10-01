// =============================================================================
// src/orchestrator/orch_solve.cpp -- ed::workflows::solve and its backend
// lanes (Lanczos / KrylovSchur / FullDiag).
//
// This is the only orchestrator translation unit that instantiates a
// backend-templated eigensolver, so ``solve_on<CudaBackend>`` (and the CUDA
// half of the ``select_backend`` visit) is emitted here and nowhere else.
//
// Part of the workflow orchestrator; see orchestrator_internal.h for the
// file map.
// =============================================================================

#include "orchestrator_internal.h"

namespace ed::workflows {

namespace {

// Pick a sensible eigensolver when the caller leaves SolveMethod::Auto: full
// diagonalization for tiny spaces, Lanczos otherwise. On a device backend the
// dense lane would run on the host and leave the device idle, so every block
// above kDenseAlwaysDim runs the device Krylov lane instead.
[[nodiscard]] SolveMethod default_method_for(const LinearOperator& H,
                                             std::size_t num_eigs = 1,
                                             bool device = false) {
    if (!device && H.global_dim() <= 1024) return SolveMethod::FullDiag;
    // Single-vector Lanczos reports every degenerate level ONCE (measured: k = 6 on the 14-site ring, dim 3432, returns the
    // 5th level wrong), while Krylov-Schur resolves the multiplicities. A
    // window therefore defaults to Krylov-Schur; a single ground state keeps
    // the faster Lanczos lane.
    return (num_eigs > 1) ? SolveMethod::KrylovSchur : SolveMethod::Lanczos;
}

// Blocks this small are solved densely whatever Krylov method was requested:
// every Krylov lane measured wrong on dim 2..10 (E0 off by up to 96 % on a
// 2-state block, degenerate copies missing, k > dim), and dense LAPACK is
// exact in microseconds there. Windows covering half the block or more are
// treated the same way (a single-vector Krylov method cannot resolve them).
inline constexpr std::uint64_t kDenseAlwaysDim = 32;

template <typename Backend>
GroundStateResult solve_on(Backend& be,
                           const LinearOperator& H,
                           const SolveOptions& opts) {
    using Complex = std::complex<double>;
    const auto geom = H.geometry();
    // The assembled CSR is built directly (two-pass gather form, parallel)
    // at a cost of a few matvecs, so it pays off for every Krylov solve
    // inside the memory cutoff (ED_CSR_DIM_MAX, default 2^22).
    auto matvec = H.template bind<Backend>();

    GroundStateResult R;

    // -----------------------------------------------------------------------
    // Defaults. The caller's explicit method / max_iter win; otherwise a
    // simple dim-based default. No memory-budget
    // pre-flight refusal, and no CSR / symmetry-matvec override -- the leaf
    // policy hooks (sym_matvec_policy_hook, ED_CSR_* env)
    // keep their default + env-override behaviour, consumed lazily at first
    // matvec.
    // -----------------------------------------------------------------------
    SolveMethod method = (opts.method != SolveMethod::Auto)
        ? opts.method
        : default_method_for(H, opts.num_eigs, !std::is_same_v<Backend, ed::matvec::CpuBackend>);
    if (method != SolveMethod::FullDiag
            && (H.global_dim() <= kDenseAlwaysDim
                || 2 * static_cast<std::uint64_t>(opts.num_eigs) >= H.global_dim())) {
        R.backend.notes.emplace_back(
            "method", "requested Krylov method replaced by FullDiag: block dim "
                      + std::to_string(H.global_dim()) + " <= " + std::to_string(kDenseAlwaysDim)
                      + " or num_eigs >= dim/2 (Krylov lanes cannot resolve such windows)");
        method = SolveMethod::FullDiag;
    }
    // Default iteration budget when the caller left it at 0. The Krylov
    // lanes all have Ritz-value early exit, so the cap only has to be
    // generous: a tight cap (e.g. 2*num_eigs+30) returns UNCONVERGED
    // energies and eigenvectors for any dim above ~1e4 (the FullDiag lane
    // ignores it). For the restarted lanes ``max_iter`` is the PER-CYCLE
    // Krylov dimension (each cycle costs O(m^2 n) with full
    // reorthogonalisation and runs to m regardless of convergence), so it
    // must not scale with the dimension: a cap of min(dim, 1000) took 592 s
    // for three eigenvalues of a 4096-state chiral model against 0.5 s at
    // m = 200 (the Python facade defaults to max(200, 8k + 80) too).
    // Single-vector Lanczos stops on convergence, so its cap is
    // min(dim, 1000).
    const std::size_t max_iter =
        (opts.max_iter > 0) ? opts.max_iter
        : (method == SolveMethod::Lanczos)
            ? std::min<std::size_t>(std::max<std::uint64_t>(H.global_dim(), 1), 1000)
        : (method == SolveMethod::KrylovSchur)
            ? std::min<std::size_t>(std::max<std::uint64_t>(H.global_dim(), 1),
                                    std::max<std::size_t>(200, 8 * opts.num_eigs + 80))
            : 2 * opts.num_eigs + 30;
    // Two-pass Lanczos for eigenvectors: no kept basis, no
    // O(m^2 n) reorthogonalisation; the recurrence is rerun once and the
    // Ritz vectors accumulated on the fly.
    const bool eigvec_two_pass =
        opts.compute_vectors && method == SolveMethod::Lanczos;
    const std::uint64_t subspace_cap_vectors = 0;  // uncapped

    // Leaf memory guard: throw cleanly before the dominant allocation rather
    // than OOM-crash. H.global_dim() is the actual working dimension (full /
    // fixed-Sz block / symmetry sector). Coarse: dense matrix for full diag;
    // stored Krylov basis (~max_iter vectors) when eigenvectors
    // are kept; a handful of work vectors otherwise.
    {
        const std::uint64_t D = H.global_dim();
        constexpr std::uint64_t CX = 16;  // sizeof(complex<double>)
        std::uint64_t est;
        if (method == SolveMethod::FullDiag) {
            est = D * D * CX;
        } else if (opts.compute_vectors && !eigvec_two_pass) {
            std::uint64_t vecs = std::max<std::uint64_t>(max_iter, 4);
            est = D * vecs * CX;
        } else {
            est = D * 8ull * CX;  // eigenvalues-only: ring-buffer work vectors
        }
        ed::core::guard_working_set(est, "ed::solve");
    }

    const auto t0 = std::chrono::steady_clock::now();

    // Deterministic-ish seed for reproducibility within a single process.
    // The kernel expects the seed in the backend's memory space (host for
    // CPU, device for CUDA). Build the seed on host first then
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
    bool host_dense = false;   // the FullDiag lane runs on the host whatever the backend

    if (method == SolveMethod::Lanczos) {
        ed::krylov::LanczosKernelOptions kopts;
        kopts.max_iter      = max_iter;
        kopts.dim_cap       = static_cast<std::size_t>(geom.global_dim);
        // No kept basis: eigenvectors come from the two-pass
        // reconstruction below, so K=1 LocalDGKS3 suffices.
        // The kept-basis FullCGS2 lane is the certification fallback.
        kopts.keep_basis      = false;
        kopts.reorth          = ed::krylov::ReorthPolicy::LocalDGKS3;
        kopts.local_ring_size = 1;
        // Ritz-value early exit (otherwise the kernel always runs to
        // `max_iter` -- 5-10x slower on small problems and a noticeable
        // hit even on large ones).
        kopts.convergence_check =
            ed::krylov::make_smallest_ritz_convergence(opts.num_eigs,
                                                       opts.tolerance,
                                                       /*min_iters=*/0,
                                                       /*require_residual_bound=*/opts.compute_vectors);
        // Check every 5 iterations to amortise the O(m^2) LAPACK tridiag
        // eigensolve. A few extra Lanczos iterations (~ check_interval / 2)
        // are cheaper than one extra dstevd every iter past convergence.
        // Same interval as the `lanczos()` default.
        kopts.convergence_check_interval = 5;
        auto kres = ed::krylov::lanczos_kernel(be, matvec, geom.local_dim,
                                               seed, kopts);
        // Converged = the Ritz check fired before the cap, or the Krylov
        // space exhausted the full dimension.
        const std::size_t cap_hit_m = std::min<std::size_t>(
            max_iter, static_cast<std::size_t>(geom.global_dim));
        R.krylov.converged = (kres.alpha.size() < cap_hit_m) ||
                             (kres.alpha.size() == static_cast<std::size_t>(geom.global_dim));
        // Solve the small (m x m) real-symmetric tridiagonal for the
        // lowest `num_eigs` eigenvalues. When the caller didn't request
        // eigenvectors, use the eigenvalues-only path -- the full eigen
        // problem (`solve_tridiag_with_eigenvectors`) is ~2-3x slower for
        // the common num_eigs=1 + compute_vectors=false workflow.
        std::vector<double> evals;
        std::vector<double> evec_coeffs;  // column-major m x m
        // A requested WINDOW (num_eigs > 1) needs the tridiag eigenvectors
        // even on the eigenvalues-only path: the per-Ritz residual bound
        // |beta_m| * |z_{m,i}| is free once z exists, and without it
        // stalled interior Ritz values escape into a merged spectrum as
        // plausible-looking garbage (e.g. -2.686 where dense says -2.459;
        // not a reorth ghost -- K = 1..32 identical). num_eigs == 1 keeps the fast
        // eigenvalues-only path: the extreme pair is what Lanczos
        // converges first and the stall detector guards it adequately.
        const bool need_z = opts.compute_vectors || opts.num_eigs > 1;
        if (need_z) {
            std::vector<double> weights;
            ed::krylov::detail::solve_tridiag_with_eigenvectors(
                kres.alpha, kres.beta, kres.alpha.size(), evals, weights, evec_coeffs);
        } else {
            evals = ed::krylov::detail::solve_tridiag(
                kres.alpha, kres.beta, kres.alpha.size());
        }
        if (opts.num_eigs > 1 && !evec_coeffs.empty()) {
            // Drop Lanczos ghosts (Cullum-Willoughby) so a
            // window never reports a converged level twice; the tridiag
            // eigenvector columns are repacked to match.
            const std::size_t m = kres.alpha.size();
            const auto keep = ed::krylov::detail::cullum_willoughby_keep(
                kres.alpha, kres.beta, m, evals);
            if (keep.size() < evals.size()) {
                std::vector<double> ev2; ev2.reserve(keep.size());
                std::vector<double> z2(m * keep.size());
                for (std::size_t c = 0; c < keep.size(); ++c) {
                    ev2.push_back(evals[keep[c]]);
                    std::copy(evec_coeffs.begin() + static_cast<std::ptrdiff_t>(keep[c] * m),
                              evec_coeffs.begin() + static_cast<std::ptrdiff_t>((keep[c] + 1) * m),
                              z2.begin() + static_cast<std::ptrdiff_t>(c * m));
                }
                evals = std::move(ev2);
                evec_coeffs = std::move(z2);
            }
        }
        const std::size_t n_keep =
            std::min<std::size_t>(opts.num_eigs, evals.size());
        // The window is returned IN FULL (truncating to the certified
        // prefix would break the num_eigs COUNT contract in
        // environment-dependent ways) and every value carries its
        // residual bound |beta_m| * |z_{m,i}| in krylov.ritz_residuals --
        // consumers that MERGE windows (sector pools) filter on the bound;
        // a direct caller keeps num_eigs values plus the diagnostics.
        std::vector<double> ritz_bounds;
        // |beta_m| of pass 1: the last (unused) recurrence coefficient that
        // turns the tridiag eigenvector bottom component into the residual
        // bound ||H y - theta y|| = |beta_m| |z_{m,i}| (exact in exact
        // arithmetic; also used below to certify the two-pass vector).
        const double beta_last = [&] {
            const std::size_t m = kres.alpha.size();
            return (kres.beta.size() > m) ? std::abs(kres.beta[m])
                                          : (m < geom.local_dim && !kres.beta.empty()
                                                 ? std::abs(kres.beta.back())
                                                 : 0.0);
        }();
        if (opts.num_eigs > 1 && !evec_coeffs.empty()) {
            const std::size_t m = kres.alpha.size();
            ritz_bounds.reserve(n_keep);
            for (std::size_t i2 = 0; i2 < n_keep; ++i2)
                ritz_bounds.push_back(
                    beta_last * std::abs(evec_coeffs[(m - 1) + i2 * m]));
        }
        R.eigenvalues.assign(evals.begin(), evals.begin() + n_keep);

        // Reconstruct host-side eigenvectors when the caller requested
        // them. evec_coeffs is the (m x m) eigenvector matrix of the tridiag
        // in column-major order; psi_k = sum_i evec_coeffs(i, k) * V_i.
        if (opts.compute_vectors && !evec_coeffs.empty()) {
            // ---- Pass 2: rerun the identical recurrence and accumulate
            //      psi_k = sum_j y_{j,k} V_j as the basis vectors stream by.
            const std::size_t m = kres.alpha.size();
            std::vector<ed::matvec::Backend::UniqueVec> acc;
            for (std::size_t k = 0; k < n_keep; ++k)
                acc.push_back(be.make_zero_vector(geom.local_dim));
            std::size_t added = 0;
            ed::krylov::LanczosKernelOptions k2 = kopts;
            k2.convergence_check = nullptr;
            k2.max_iter          = m;
            k2.on_step_interval  = 1;
            k2.on_step = [&](std::size_t it, const std::vector<double>&,
                             const std::vector<double>&, const Complex*,
                             const Complex* v_prev, std::size_t n) {
                const std::size_t jidx = it - 1;          // v_prev = V_j
                if (jidx < m && jidx == added) {
                    for (std::size_t k = 0; k < n_keep; ++k)
                        be.axpy(Complex(evec_coeffs[jidx + k * m], 0.0),
                                v_prev, acc[k].get(), n);
                    ++added;
                }
            };
            (void)ed::krylov::lanczos_kernel(be, matvec, geom.local_dim, seed, k2);
            bool ok = (added == m);
            double resid = std::numeric_limits<double>::quiet_NaN();
            if (ok) {
                // Certify the ground-state vector: ||H psi - E psi|| / ||psi||.
                auto hpsi = be.make_zero_vector(geom.local_dim);
                matvec(acc[0].get(), hpsi.get(), geom.local_dim);
                const double npsi = be.nrm2(acc[0].get(), geom.local_dim);
                const double r = be.axpy_nrm2(Complex(-evals[0], 0.0), acc[0].get(),
                                              hpsi.get(), geom.local_dim);
                resid = (npsi > 0.0) ? r / npsi : std::numeric_limits<double>::infinity();
                R.krylov.residual_norm = resid;
                // Certification. The Ritz-value stop at `tolerance` gives a
                // vector whose residual scales like sqrt(tolerance) * |E|,
                // so an absolute gate (e.g. 1e-6) is never met at tol = 1e-10
                // and would send every N >= 20 run through the slow
                // kept-basis fallback. The right yardstick is the free
                // Lanczos bound |beta_m| |z_{m,0}| from pass 1: a faithful
                // reconstruction reproduces it to O(1); loss of orthogonality
                // (a ghost Ritz vector) violates it by orders of magnitude.
                const double scale  = std::max(1.0, std::abs(evals[0]));
                const double bound0 = beta_last * std::abs(evec_coeffs[m - 1]);
                const double gate   = std::max({10.0 * bound0,
                                                1e3 * opts.tolerance * scale,
                                                1e-12 * scale});
                ok = std::isfinite(resid) && (resid <= gate || !R.krylov.converged);
            }
            if (ok) {
                EigenvectorRef evref;
                evref.host.resize(n_keep, std::vector<Complex>(geom.local_dim, Complex{0.0, 0.0}));
                for (std::size_t k = 0; k < n_keep; ++k) {
                    const double nk = be.nrm2(acc[k].get(), geom.local_dim);
                    if (nk > 0.0) be.scale(Complex(1.0 / nk, 0.0), acc[k].get(), geom.local_dim);
                    be.copy_to_host(acc[k].get(), evref.host[k].data(), geom.local_dim);
                }
                R.eigenvectors = std::move(evref);
            } else {
                // Fallback (breakdown in pass 2 or an uncertified vector):
                // the kept-basis FullCGS2 lane.
                ed::krylov::LanczosKernelOptions k3 = kopts;
                k3.keep_basis = true;
                k3.reorth     = ed::krylov::ReorthPolicy::FullCGS2;
                k3.max_iter   = m;
                k3.convergence_check = nullptr;
                auto kres3 = ed::krylov::lanczos_kernel(be, matvec, geom.local_dim, seed, k3);
                std::vector<double> ev3, w3, z3;
                ed::krylov::detail::solve_tridiag_with_eigenvectors(
                    kres3.alpha, kres3.beta, kres3.alpha.size(), ev3, w3, z3);
                const std::size_t m3 = kres3.alpha.size();
                EigenvectorRef evref;
                evref.host.resize(n_keep, std::vector<Complex>(geom.local_dim, Complex{0.0, 0.0}));
                std::vector<Complex> basis_host(geom.local_dim);
                for (std::size_t i = 0; i < m3 && i < kres3.basis.size(); ++i) {
                    be.copy_to_host(kres3.basis[i].get(), basis_host.data(), geom.local_dim);
                    for (std::size_t k = 0; k < n_keep && k < ev3.size(); ++k) {
                        const double c = z3[i + k * m3];
                        auto& out = evref.host[k];
                        for (std::size_t r = 0; r < geom.local_dim; ++r) out[r] += c * basis_host[r];
                    }
                }
                R.eigenvectors = std::move(evref);
            }
        }

        R.krylov.alpha = std::move(kres.alpha);
        R.krylov.beta  = std::move(kres.beta);
        R.krylov.iters_done = kres.iters_done;
        if (!ritz_bounds.empty())
            R.krylov.ritz_residuals = std::move(ritz_bounds);
    } else if (method == SolveMethod::KrylovSchur) {
        ed::krylov::KrylovSchurOptions kopts;
        kopts.num_eigs        = opts.num_eigs;
        kopts.max_iter        = max_iter;
        kopts.tolerance       = opts.tolerance;
        kopts.compute_vectors = opts.compute_vectors;
        kopts.global_n        = geom.global_dim;
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
        // FullDiag lane: build the dense matrix and run LAPACK zheevd via
        // `full_diagonalization`. The default method picks this lane only
        // for small dimensions (<= 1024), so the O(N^3) dense step is
        // affordable.
        //
        // The FullDiag column-extraction loop in
        // ``::full_diagonalization`` (lanczos.cpp) calls
        // ``H(unit_vec.data(), col_j.data(), N)`` with host
        // ``std::vector<Complex>`` storage. If we hand it a matvec
        // bound to a non-CPU backend (e.g. the representative-sector
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
        // ``LinearOperator::bind<CpuBackend>()`` selects). The Lanczos /
        // KrylovSchur lanes above keep the device-bound matvec since they
        // operate entirely in the backend's memory space.
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
        std::vector<std::vector<Complex>> fd_vecs;
        full_diagonalization(Hv, geom.local_dim, opts.num_eigs, eigs,
                             opts.compute_vectors,
                             /*op_for_dense=*/&H,
                             opts.compute_vectors ? &fd_vecs : nullptr);
        if (opts.compute_vectors && !fd_vecs.empty()) {
            // The dense lane hands its vectors back in memory.
            EigenvectorRef evref;
            evref.host = std::move(fd_vecs);
            R.eigenvectors = std::move(evref);
        }
        const std::size_t n_keep = std::min<std::size_t>(
            opts.num_eigs, eigs.size());
        R.eigenvalues.assign(eigs.begin(), eigs.begin() + n_keep);
        R.krylov.iters_done = 0;
        R.krylov.converged  = true;
        host_dense = true;
    }

    // The lane label comes from the Backend template parameter, not the
    // operator's memory_space: a host-resident operator that advertises
    // ``supports_device_matvec=true`` runs on ``CudaBackend``, so only
    // ``ed::lane_label_for<Backend>()`` always matches the lane
    // ``std::visit`` dispatched to -- except FullDiag, which ran on the host.
    R.backend.lane  = host_dense ? "cpu" : ed::lane_label_for<Backend>();
    R.backend.dense = host_dense;
    const auto t1 = std::chrono::steady_clock::now();
    R.backend.wall_seconds =
        std::chrono::duration<double>(t1 - t0).count();
    return R;
}

}  // namespace

// ---------------------------------------------------------------------------
// Public entry point.
// ---------------------------------------------------------------------------

GroundStateResult solve(const LinearOperator& H, SolveOptions opts) {
    require_hermitian_input(H, "ed::solve");
    // Dimension-aware thread budget (the same one `lanczos()` applies).
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

}  // namespace ed::workflows
