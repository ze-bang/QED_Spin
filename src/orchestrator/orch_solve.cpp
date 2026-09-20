// =============================================================================
// src/orchestrator/orch_solve.cpp -- ed::workflows::solve and its backend
// lanes (Lanczos / BlockLanczos / BlockKrylovSchur / KrylovSchur / FullDiag).
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

using namespace orch_detail;

namespace {

// Pick a sensible eigensolver when the caller leaves SolveMethod::Auto: full
// diagonalization for tiny spaces, Lanczos otherwise. (Replaces the planner's
// cost-model method choice.)
[[nodiscard]] SolveMethod default_method_for(const LinearOperator& H,
                                             std::size_t num_eigs = 1) {
    if (H.global_dim() <= 1024) return SolveMethod::FullDiag;
    // Correctness (2026-09-11): single-vector Lanczos reports every degenerate
    // level ONCE (measured: k = 6 on the 14-site ring, dim 3432, returns the
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
    // Audit F4: the assembled CSR is now built directly (two-pass gather
    // form, parallel) at a cost of a few matvecs, so it pays off for every
    // Krylov solve inside the memory cutoff (ED_CSR_DIM_MAX, default 2^22);
    // the planner's dim-based decision stands.
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
    SolveMethod method = (opts.method != SolveMethod::Auto)
        ? opts.method
        : default_method_for(H, opts.num_eigs);
    if (method != SolveMethod::FullDiag
            && (H.global_dim() <= kDenseAlwaysDim
                || 2 * static_cast<std::uint64_t>(opts.num_eigs) >= H.global_dim())) {
        R.backend.notes.emplace_back(
            "method", "requested Krylov method replaced by FullDiag: block dim "
                      + std::to_string(H.global_dim()) + " <= " + std::to_string(kDenseAlwaysDim)
                      + " or num_eigs >= dim/2 (Krylov lanes cannot resolve such windows)");
        method = SolveMethod::FullDiag;
    }
    // Audit 2026-09 (correctness): the previous default of 2*num_eigs+30
    // (= 32 for the ground state) silently returned UNCONVERGED results
    // for every problem that needs more than 32 Krylov iterations (i.e.
    // any dim above ~1e4): energies wrong at 1e-7 and "eigenvectors" with
    // residuals of 1e-3..1e-2 at tolerance 1e-10. The Krylov lanes all
    // have Ritz-value early exit, so the cap only has to be generous; the
    // block methods count blocks and keep the old default.
    // Default iteration budget when the caller left it at 0. For the
    // restarted lanes ``max_iter`` is the PER-CYCLE Krylov dimension (each
    // cycle costs O(m^2 n) with full reorthogonalisation and runs to m
    // regardless of convergence), so it must not scale with the dimension:
    // the CLI's unset cap became min(dim, 1000) and a 4096-state chiral
    // model took 592 s for three eigenvalues (0.5 s at m = 200; the Python
    // facade already defaults to max(200, 8k + 80)). Single-vector Lanczos
    // stops on convergence, so its cap may stay at min(dim, 1000).
    const std::size_t max_iter =
        (opts.max_iter > 0) ? opts.max_iter
        : (method == SolveMethod::Lanczos)
            ? std::min<std::size_t>(std::max<std::uint64_t>(H.global_dim(), 1), 1000)
        : (method == SolveMethod::KrylovSchur)
            ? std::min<std::size_t>(std::max<std::uint64_t>(H.global_dim(), 1),
                                    std::max<std::size_t>(200, 8 * opts.num_eigs + 80))
            : 2 * opts.num_eigs + 30;
    // Two-pass Lanczos for eigenvectors (audit F3): no kept basis, no
    // O(m^2 n) reorthogonalisation; the recurrence is rerun once and the
    // Ritz vectors accumulated on the fly. ED_LANCZOS_EIGVEC_TWOPASS=0
    // restores the kept-basis FullCGS2 lane.
    const bool eigvec_two_pass = [&] {
        if (!(opts.compute_vectors && method == SolveMethod::Lanczos)) return false;
        return ed::env::flag("ED_LANCZOS_EIGVEC_TWOPASS", true);
    }();
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
        } else if (opts.compute_vectors && !eigvec_two_pass) {
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
    // Stage 12 (SU(2) rollout): caller-supplied seed transform (e.g. the
    // Lowdin total-spin projection). Applied on the host copy before
    // staging; the transform is responsible for leaving a usable
    // (normalisable) vector or throwing.
    if (opts.seed_transform) {
        opts.seed_transform(seed_host.data(), seed_host.size());
        double sumsq = 0.0;
        for (const auto& z : seed_host) sumsq += std::norm(z);
        if (!(sumsq > 0.0)) {
            throw std::runtime_error(
                "ed::solve: seed_transform produced a zero seed (the "
                "targeted symmetry sector has no weight in this block)");
        }
        const double inv = 1.0 / std::sqrt(sumsq);
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
                return ed::env::flag("ED_FORCE_COMPLEX_LANCZOS", false);
            }();
            // Audit F3/F5 (2026-09): the real-storage lane now also serves
            // eigenvalue WINDOWS (num_eigs > 1, with the Ritz residual
            // bounds the window contract requires) and EIGENVECTORS via a
            // real two-pass reconstruction, so the complex kernel is only
            // used when the operator is genuinely complex, a seed transform
            // is installed, or the two-pass lane is disabled by env.
            bool real_done = false;
            if (!force_complex
                    && !opts.seed_transform  // draws its own seed internally
                    && (!opts.compute_vectors || eigvec_two_pass)
                    && H.is_real_hermitian()) {
                auto Hv_real = H.bind_real_cpu();
                auto H_fn = [Hv_real](const double* in, double* out, int n) {
                    Hv_real(in, out, static_cast<std::size_t>(n));
                };
                const std::size_t n = geom.local_dim;
                // Real start vector: Re(seed); the seed is unit-norm complex
                // Gaussian so the real part is a perfectly good (renormalised
                // inside) random start. Deterministic => pass 2 replays pass 1.
                std::vector<double> v0(n);
                double v0_sq = 0.0;
                for (std::size_t i = 0; i < n; ++i) { v0[i] = seed_host[i].real(); v0_sq += v0[i] * v0[i]; }
                LanczosRealExtras ex;
                ex.v0        = (v0_sq > 0.0) ? v0.data() : nullptr;
                ex.want_ritz = opts.compute_vectors || opts.num_eigs > 1;
                ex.converge_vectors = opts.compute_vectors;   // windows without vectors keep the Ritz-value stop
                std::vector<double> eigs;
                std::uint64_t real_iters = 0;
                bool real_converged = false;
                ::lanczos_real(H_fn, static_cast<std::uint64_t>(n),
                               static_cast<std::uint64_t>(max_iter),
                               static_cast<std::uint64_t>(opts.num_eigs),
                               opts.tolerance, eigs, &real_iters, &real_converged, &ex);
                const std::size_t m = ex.alpha.size();
                const std::size_t n_keep = std::min<std::size_t>(opts.num_eigs, eigs.size());
                real_done = !eigs.empty() && (!ex.want_ritz || ex.ritz_vectors.size() >= m * n_keep);
                if (real_done && opts.compute_vectors && n_keep > 0) {
                    // ---- Pass 2 (real): psi_k = sum_j z_{j,k} V_j streamed. ----
                    std::vector<std::vector<double>> acc(n_keep, std::vector<double>(n, 0.0));
                    std::size_t added = 0;
                    LanczosRealExtras ex2;
                    ex2.v0 = ex.v0;
                    ex2.fixed_iterations = true;
                    ex2.on_basis_vector = [&](std::uint64_t j, const double* vj) {
                        if (j >= m || j != added) return;
                        for (std::size_t k = 0; k < n_keep; ++k) {
                            const double c = ex.ritz_vectors[j + k * m];
                            double* a = acc[k].data();
                            #pragma omp parallel for schedule(static) if(n > 8192)
                            for (long long i = 0; i < static_cast<long long>(n); ++i)
                                a[i] += c * vj[i];
                        }
                        ++added;
                    };
                    std::vector<double> eigs2;
                    ::lanczos_real(H_fn, static_cast<std::uint64_t>(n),
                                   static_cast<std::uint64_t>(m),
                                   static_cast<std::uint64_t>(opts.num_eigs),
                                   opts.tolerance, eigs2, nullptr, nullptr, &ex2);
                    double resid = std::numeric_limits<double>::quiet_NaN();
                    bool ok = (added == m);
                    if (ok) {
                        // Certify the ground-state vector against the pass-1
                        // Ritz bound (see the complex two-pass lane below).
                        std::vector<double> hpsi(n);
                        H_fn(acc[0].data(), hpsi.data(), static_cast<int>(n));
                        double num = 0.0, den = 0.0;
                        const double E0 = eigs[0];
                        #pragma omp parallel for reduction(+:num,den) schedule(static) if(n > 8192)
                        for (long long i = 0; i < static_cast<long long>(n); ++i) {
                            const double r = hpsi[i] - E0 * acc[0][i];
                            num += r * r; den += acc[0][i] * acc[0][i];
                        }
                        resid = (den > 0.0) ? std::sqrt(num / den)
                                            : std::numeric_limits<double>::infinity();
                        const double scale  = std::max(1.0, std::abs(E0));
                        const double bound0 = ex.ritz_bounds.empty() ? 0.0 : ex.ritz_bounds[0];
                        const double gate   = std::max({10.0 * bound0,
                                                        1e3 * opts.tolerance * scale,
                                                        1e-12 * scale});
                        ok = std::isfinite(resid) && (resid <= gate || !real_converged);
                    }
                    if (ok) {
                        EigenvectorRef evref;
                        evref.host.resize(n_keep, std::vector<Complex>(n, Complex{0.0, 0.0}));
                        for (std::size_t k = 0; k < n_keep; ++k) {
                            double nk = 0.0;
                            for (std::size_t i = 0; i < n; ++i) nk += acc[k][i] * acc[k][i];
                            const double inv = (nk > 0.0) ? 1.0 / std::sqrt(nk) : 1.0;
                            for (std::size_t i = 0; i < n; ++i)
                                evref.host[k][i] = Complex(acc[k][i] * inv, 0.0);
                        }
                        R.eigenvectors = std::move(evref);
                        R.krylov.residual_norm = resid;
                    } else {
                        std::cout << "Lanczos[real]: two-pass eigenvector not certified "
                                     "(residual = " << resid << "); using the complex "
                                     "kept-basis lane" << std::endl;
                        real_done = false;
                    }
                }
                if (real_done) {
                    R.eigenvalues.assign(eigs.begin(), eigs.begin() + n_keep);
                    if (opts.num_eigs > 1 && !ex.ritz_bounds.empty())
                        R.krylov.ritz_residuals.assign(ex.ritz_bounds.begin(),
                                                       ex.ritz_bounds.begin() + n_keep);
                    R.krylov.alpha = std::move(ex.alpha);
                    R.krylov.beta  = std::move(ex.beta);
                    R.krylov.iters_done = static_cast<std::size_t>(real_iters);
                    R.krylov.converged  = real_converged;
                }
            }
            if (real_done) {
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
        kopts.keep_basis    = opts.compute_vectors && !eigvec_two_pass;

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
            if (const char* k_env = ed::env::raw("ED_LANCZOS_REORTH_K")) {
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
                                                       /*min_iters=*/0,
                                                       /*require_residual_bound=*/opts.compute_vectors);
        // Wave 2.6: check every-5 iterations to amortise the O(m^2)
        // LAPACK tridiag eigensolve. A few extra Lanczos iterations
        // (~ check_interval / 2) are cheaper than one extra dstevd
        // every iter past convergence. Matches the distributed lane
        // and the post-Wave-2.6 `lanczos()` default. Override via
        // env ``ED_LANCZOS_CHECK_EVERY``.
        kopts.convergence_check_interval = 5;
        if (const char* ce = ed::env::raw("ED_LANCZOS_CHECK_EVERY")) {
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
        // Convergence bookkeeping (audit): the lane used to leave
        // `krylov.converged` false even when the Ritz check fired, and
        // said nothing when the cap was hit.
        const std::size_t cap_hit_m = std::min<std::size_t>(
            max_iter, static_cast<std::size_t>(geom.global_dim));
        R.krylov.converged = (kres.alpha.size() < cap_hit_m) ||
                             (kres.alpha.size() == static_cast<std::size_t>(geom.global_dim));
        // Solve the small (m x m) real-symmetric tridiagonal for the
        // lowest `num_eigs` eigenvalues. When the caller didn't request
        // eigenvectors, use the eigenvalues-only Eigen path -- the
        // legacy `solve_tridiag_with_eigenvectors` did the full eigen
        // problem unconditionally, which is ~2-3x slower for the
        // common num_eigs=1 + compute_vectors=false workflow.
        std::vector<double> evals;
        std::vector<double> evec_coeffs;  // column-major m x m
        // GAP 10 fix (2026-07-16): a requested WINDOW (num_eigs > 1) needs
        // the tridiag eigenvectors even on the eigenvalues-only path --
        // the per-Ritz residual bound |beta_m| * |z_{m,i}| is free once z
        // exists, and without it stalled interior Ritz values escaped
        // into the merged spectrum as plausible-looking garbage (measured:
        // -2.686 reported where dense says -2.459; NOT a reorth ghost --
        // K = 1..32 identical). num_eigs == 1 keeps the fast
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
            // Audit 2026-09: drop Lanczos ghosts (Cullum-Willoughby) so a
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
        // GAP-10 v2 (2026-07-17): the first fix TRUNCATED to the certified
        // prefix, which broke the num_eigs COUNT contract in
        // environment-dependent ways (an OpenBLAS runner trimmed a value
        // the local run kept; [unified-e2e] 83 asserted the count). The
        // window is now returned IN FULL and every value carries its
        // residual bound |beta_m| * |z_{m,i}| in krylov.ritz_residuals --
        // consumers that MERGE windows (the streaming sector pools, where
        // the original garbage did its damage) filter on the bound; a
        // direct caller keeps num_eigs values plus the diagnostics.
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

        // Reconstruct host-side eigenvectors from the kept Lanczos basis
        // when the caller requested them. evec_coeffs is the (m x m)
        // eigenvector matrix of the tridiag in column-major order; the
        // k-th eigenvector in the original Hilbert space is the linear
        // combination psi_k = sum_i evec_coeffs(i, k) * basis[i].
        if (opts.compute_vectors && eigvec_two_pass && !evec_coeffs.empty()) {
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
                             const Complex* v_prev, std::size_t n,
                             const std::vector<const Complex*>*) {
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
                // so an absolute gate (the first cut used 1e-6) is never met
                // at tol = 1e-10 and sent every N >= 20 run through the slow
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
        } else if (opts.compute_vectors && !kres.basis.empty()) {
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
        if (!ritz_bounds.empty())
            R.krylov.ritz_residuals = std::move(ritz_bounds);
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
            // Honour the caller's flag; ED_BLOCK_LANCZOS_LEAN=1 forces the lean
            // (no stored basis) mode. Any other value leaves the caller's choice alone
            // -- the variable used to FORCE keep_basis=true whenever it was set to
            // anything but "1", including "0" and "".
            kopts.keep_basis = opts.block_lanczos_keep_basis;
            if (ed::env::flag("ED_BLOCK_LANCZOS_LEAN", false)) kopts.keep_basis = false;
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
            std::vector<std::vector<Complex>> fd_vecs;
            full_diagonalization(Hv, geom.local_dim, opts.num_eigs, eigs,
                                 opts.output_dir,
                                 opts.compute_vectors,
                                 /*op_for_dense=*/&H,
                                 opts.compute_vectors ? &fd_vecs : nullptr);
            if (opts.compute_vectors && !fd_vecs.empty()) {
                // Audit 2026-09: the dense lane used to persist vectors to
                // HDF5 only; with output_dir empty the caller got nothing.
                EigenvectorRef evref;
                evref.host = std::move(fd_vecs);
                R.eigenvectors = std::move(evref);
            }
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

}  // namespace

// ---------------------------------------------------------------------------
// Public entry point.
// ---------------------------------------------------------------------------

GroundStateResult solve(const LinearOperator& H, SolveOptions opts) {
    require_hermitian_input(H, "ed::solve");
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

}  // namespace ed::workflows
