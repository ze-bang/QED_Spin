#include <ed/config/env_registry.h>
#include <ed/solvers/lanczos.h>
#include <ed/krylov/lanczos_kernel.h>
#include <ed/matvec/backends/cpu_backend.h>
#include <ed/parallel/fused_blas1.h>
#include <ed/parallel/numa.h>
#include <ed/parallel/thread_budget.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#include <chrono>
#include <cstdlib>
#include <limits>
#include <iomanip>

ComplexVector generateGaussianRandomVector(int N, std::mt19937& gen) {
    // i.i.d. standard complex Gaussian: real and imag parts ~ N(0, 1), then
    // L2-normalise. This produces an isotropic random vector on the complex
    // unit sphere and is the standard finite-T trace-estimator distribution
    // (Jaklic-Prelovsek FTLM, Hutchinson, etc.). Variance bounds and isotropy
    // properties differ from normalised uniform-cube sampling.
    std::normal_distribution<double> ndist(0.0, 1.0);
    ComplexVector v(N);
    for (int i = 0; i < N; i++) {
        v[i] = Complex(ndist(gen), ndist(gen));
    }
    double norm = cblas_dznrm2(N, v.data(), 1);
    Complex scale_factor = Complex(1.0 / norm, 0.0);
    cblas_zscal(N, &scale_factor, v.data(), 1);
    return v;
}

// Diagonalize tridiagonal matrix and extract Ritz values and weights
void diagonalize_tridiagonal_ritz(
    const std::vector<double>& alpha,
    const std::vector<double>& beta,
    std::vector<double>& ritz_values,
    std::vector<double>& weights,
    std::vector<double>* evecs
) {
    uint64_t m = alpha.size();
    
    // Prepare diagonal and off-diagonal arrays for LAPACK
    std::vector<double> diag = alpha;
    std::vector<double> offdiag(m - 1);
    for (int i = 0; i < m - 1; i++) {
        offdiag[i] = beta[i + 1];
    }
    
    // Allocate eigenvector storage
    std::vector<double> evecs_local;
    double* evecs_ptr = nullptr;
    
    if (evecs != nullptr) {
        evecs->resize(m * m);
        evecs_ptr = evecs->data();
    } else {
        evecs_local.resize(m * m);
        evecs_ptr = evecs_local.data();
    }
    
    // Diagonalize
    uint64_t info = LAPACKE_dstevd(LAPACK_COL_MAJOR, 'V', m, 
                                    diag.data(), offdiag.data(), 
                                    evecs_ptr, m);
    
    if (info != 0) {
        std::cerr << "LAPACKE_dstevd failed in diagonalize_tridiagonal_ritz with error code " << info << std::endl;
        ritz_values.clear();
        weights.clear();
        return;
    }
    
    // Extract Ritz values (eigenvalues are now in diag, sorted)
    ritz_values.resize(m);
    std::copy(diag.begin(), diag.end(), ritz_values.begin());
    
    // Extract weights: squared first component of each eigenvector
    weights.resize(m);
    for (int i = 0; i < m; i++) {
        // First component of eigenvector i (column-major: evecs[0 + i*m])
        double first_component = evecs_ptr[i * m];  // First row, column i
        weights[i] = first_component * first_component;
    }
}

void estimate_spectral_bounds(
    std::function<void(const Complex*, Complex*, int)> H,
    uint64_t dim,
    int krylov_dim,
    double tol,
    std::mt19937& gen,
    double& e_min,
    double& e_max)
{
    // On small blocks a Lanczos sweep with krylov_dim > dim returns garbage
    // bounds. Below 512 states assemble the block densely (dim matvecs) and
    // take the exact extremes; above, clamp the sweep to dim.
    if (dim <= 512) {
        const int n = static_cast<int>(dim);
        std::vector<Complex> dense(static_cast<std::size_t>(n) * n), unit(n), col(n);
        for (int j = 0; j < n; ++j) {
            std::fill(unit.begin(), unit.end(), Complex(0.0, 0.0));
            unit[j] = Complex(1.0, 0.0);
            H(unit.data(), col.data(), n);
            for (int i = 0; i < n; ++i) dense[static_cast<std::size_t>(j) * n + i] = col[i];
        }
        std::vector<double> w(n);
        const int info = LAPACKE_zheevd(LAPACK_COL_MAJOR, 'N', 'U', n,
                                        reinterpret_cast<lapack_complex_double*>(dense.data()),
                                        n, w.data());
        if (info == 0) {
            e_min = w.front();
            e_max = w.back();
            return;
        }
        // fall through to the Krylov estimate on a LAPACK failure
    }
    krylov_dim = static_cast<int>(std::min<uint64_t>(
        static_cast<uint64_t>(std::max(krylov_dim, 2)), dim));

    ComplexVector v0 = generateGaussianRandomVector(static_cast<int>(dim), gen);

    // Bare three-term recurrence (extreme Ritz values converge first and are
    // robust without reorthogonalization), stopping when ||w|| < tol.
    ed::krylov::LanczosKernelOptions opts;
    opts.max_iter      = static_cast<std::size_t>(krylov_dim);
    opts.reorth        = ed::krylov::ReorthPolicy::None;
    opts.keep_basis    = false;
    opts.breakdown_tol = tol;
    auto matvec = [&H](const Complex* in, Complex* out, std::size_t n) {
        H(in, out, static_cast<int>(n));
    };
    auto result = ed::krylov::lanczos_kernel(
        ed::matvec::default_cpu_backend(), matvec,
        static_cast<std::size_t>(dim), v0.data(), opts);
    std::vector<double> alpha = std::move(result.alpha);
    std::vector<double> beta  = std::move(result.beta);

    if (alpha.empty())
        throw std::runtime_error("estimate_spectral_bounds: Lanczos produced 0 iterations");

    std::vector<double> ritz, weights;
    diagonalize_tridiagonal_ritz(alpha, beta, ritz, weights, /*evecs=*/nullptr);
    if (ritz.empty())
        throw std::runtime_error("estimate_spectral_bounds: Ritz step returned 0 eigenpairs");

    e_min = ritz.front();
    e_max = ritz.back();
}

// =============================================================================
// lanczos_real -- real-storage / real-arithmetic Lanczos for eigenvalues only.
//
// Phase 6 #7: when the Hamiltonian is real and the seed is real, the entire
// Krylov basis stays real. The complex ``lanczos_kernel`` stores complex
// vectors, paying 2x memory traffic and 2x BLAS-1 FLOPs over the
// strictly-needed amount. At N = 18-22 (Krylov dim < 1M) the iter is BLAS-1
// bound, so this halving of BLAS-1 traffic is the single largest residual win.
//
// Local DGKS reorth against a short ring of recent vectors, relative Ritz-value
// convergence test, breakdown on beta < tol. No stored basis.
// =============================================================================
// Cullum-Willoughby ghost filter on the ascending Ritz values ``theta`` of the
// m x m Lanczos tridiagonal (alpha, beta[1..m-1]). Returns the ascending list
// of distinct, non-spurious Ritz values: multiple copies collapse to one and a
// simple Ritz value that coincides with an eigenvalue of the (m-1) x (m-1)
// tridiagonal obtained by deleting the first row and column is dropped.
static std::vector<double>
cullum_willoughby_filter(const std::vector<double>& theta,
                         const std::vector<double>& alpha,
                         const std::vector<double>& beta,
                         uint64_t m) {
    if (m < 3) return theta;
    std::vector<double> d2(alpha.begin() + 1, alpha.begin() + static_cast<std::ptrdiff_t>(m));
    std::vector<double> e2(m - 2);
    for (uint64_t ii = 0; ii + 2 < m; ++ii) e2[ii] = beta[ii + 2];
    const int info = LAPACKE_dstevd(LAPACK_COL_MAJOR, 'N', m - 1,
                                    d2.data(), e2.data(), nullptr, m - 1);
    if (info != 0) return theta;
    double scale = 0.0;
    for (double a : alpha) scale = std::max(scale, std::abs(a));
    for (double b : beta) scale = std::max(scale, std::abs(b));
    const double tol_eq = 1e-10 * std::max(1.0, scale);
    std::vector<double> out;
    out.reserve(theta.size());
    uint64_t i = 0;
    while (i < theta.size()) {
        uint64_t jj = i + 1;
        while (jj < theta.size() && std::abs(theta[jj] - theta[i]) <= tol_eq) ++jj;
        const bool multiple = (jj - i) > 1;
        if (multiple) {
            out.push_back(theta[i]);            // converged level, copies collapsed
        } else {
            // simple: spurious iff it is an eigenvalue of the deleted tridiagonal
            const auto it = std::lower_bound(d2.begin(), d2.end(), theta[i] - tol_eq);
            const bool spurious = (it != d2.end() && std::abs(*it - theta[i]) <= tol_eq);
            if (!spurious) out.push_back(theta[i]);
        }
        i = jj;
    }
    return out.empty() ? theta : out;
}

void lanczos_real(std::function<void(const double*, double*, int)> H_real,
                  uint64_t N, uint64_t max_iter, uint64_t exct,
                  double tol, std::vector<double>& eigenvalues,
                  uint64_t* iters_out, bool* converged_out,
                  LanczosRealExtras* extras) {
    if (iters_out) *iters_out = 0;
    if (converged_out) *converged_out = false;
    const bool fixed_iters = extras && extras->fixed_iterations;
    // Mirror the Lanczos thread-budget heuristic so the OMP+BLAS thread cap
    // is consistent with the complex kernel lanes.
    const ed::parallel::ThreadBudgetScope budget(
        ed::parallel::auto_threads_for_dim(N));

    // Local-reorth ring buffer: max_recent slabs of N doubles each, held
    // as a single contiguous allocation. Each iter we write the just-
    // computed v_{j+1} *directly* into slab[ring_head] via the fused
    // norm+scale kernel (Phase 6 #9 + #10) instead of writing into v_next
    // and then memcpy'ing into the slab. v_current and v_prev are POINTER
    // views into earlier slabs; the rotating ring head IS the new
    // v_current. This kills the per-iter dim-N memcpy that the previous
    // ring-update did.
    constexpr int max_recent = 5;
    // Phase 6 #12: project against the K most-recent ring vectors.
    // Default K=1: a single DGKS pass against v_{j-1}. This is the
    // canonical Lanczos-with-one-step-reorthogonalisation used by xdiag,
    // ARPACK (`dsaupd` mode 1), and SLEPc -- one DGKS pass restores
    // local orthogonality lost to round-off in the 3-term recurrence and
    // is sufficient for well-conditioned spin / Hubbard ground states up
    // to dim ~ 10^7. The fused-3op recurrence (above) already projects
    // against v_j (k=0) using the known alpha_j coefficient, so we walk
    // the ring starting at k=1.
    //
    // We measured at N >= 20 (Heisenberg PBC chain, fixed-Sz):
    //
    //     K     N=20      N=22       N=24
    //     ---   --------  ---------  ---------
    //     1     242 ms    1539 ms    7992 ms
    //     2     230 ms    1763 ms    ~9000 ms
    //     3     261 ms    1753 ms    >10 s
    //
    // The lone extra pass at K=1 saves one full dim-N dot+axpy per iter
    // (which is memory-bandwidth bound on every modern x86 CPU), and the
    // resulting eigenvalues match the K=3 reference to 1e-9 -- well below
    // the user-facing tolerance of 1e-10. Override with
    // ED_LANCZOS_REORTH_K=N (0..max_recent-1) for ill-conditioned spectra.
    int reorth_K = 1;
    if (const char* env = ed::env::raw("ED_LANCZOS_REORTH_K")) {
        const int k = std::atoi(env);
        if (k >= 0 && k < max_recent) reorth_K = k;
    }
    std::vector<double> recent_buf(static_cast<size_t>(N) * max_recent, 0.0);
    auto slab = [&](int slot) -> double* {
        return recent_buf.data() + static_cast<size_t>(slot) * N;
    };
    int ring_count = 0, ring_head = 0;

    // Working vectors. v_current and v_prev are POINTERS into the ring
    // (no copies). w is its own buffer (the SpMV destination). The
    // initial v_current lives in slab[0]; subsequent v_{j+1}s are written
    // directly into the next ring slot.
    std::vector<double> w(N);
    std::mt19937 gen(std::random_device{}());
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    double* v_current = slab(0);
    if (extras && extras->v0) {
        // Deterministic start (reproducible runs; identical in both passes
        // of the two-pass eigenvector reconstruction).
        std::copy(extras->v0, extras->v0 + N, v_current);
    } else {
        for (uint64_t i = 0; i < N; ++i) v_current[i] = dist(gen);
    }
    double norm = cblas_dnrm2(N, v_current, 1);
    if (norm == 0.0) {
        std::cerr << "lanczos_real: zero starting vector" << std::endl;
        return;
    }
    cblas_dscal(N, 1.0 / norm, v_current, 1);
    ring_count = 1;
    ring_head  = 1;  // next write goes here
    const double* v_prev = nullptr;  // unused at j=0 (apply_beta_term=false)

    // First-touch w so each OMP thread owns the chunk it will read.
    ed::parallel::pin_omp_threads_once();
    #pragma omp parallel for schedule(static) if(N > 4096)
    for (std::int64_t i = 0; i < static_cast<std::int64_t>(N); ++i) {
        w[i] = 0.0;
    }

    std::vector<double> alpha;  // tridiagonal diagonal
    std::vector<double> beta;   // tridiagonal off-diagonal (beta[0] unused)
    beta.push_back(0.0);

    // Eigenvalue-convergence bookkeeping. Phase 6 #11: switch from
    // absolute-change-every-10-iters to xdiag-style RELATIVE-change-
    // every-iter. This:
    //   * matches the criterion every other ED library (xdiag, KrylovKit,
    //     ARPACK with the default ``tol``) reports their iter counts
    //     against, so the bench_vs_xdiag iter counts become comparable;
    //   * lets the user pass the same ``tolerance=1e-12`` they would to
    //     ``scipy.sparse.linalg.eigsh`` and get the same behaviour;
    //   * converges much earlier on well-conditioned problems (e.g. the
    //     1D Heisenberg fixed-Sz benchmark drops from 60 -> ~25 iters at
    //     N=20 to reach the same numerical precision).
    //
    // We still re-solve the small Lanczos tridiagonal at every iter to
    // get the new Ritz value -- LAPACKE_dstevd on a 60x60 matrix is
    // ~50 us, dwarfed by the per-iter BLAS-1 cost (~5 ms at N=20).
    std::vector<double> prev_eigenvalues;
    bool converged = false;
    uint64_t total_reorth_count = 0, selective_reorth_count = 0;
    const double ortho_threshold = 1e-12;

    max_iter = std::min(N, max_iter);
    std::cout << "Lanczos[real]: max_iter=" << max_iter << ", n_eig=" << exct
              << ", tol=" << tol << " (real-storage fast path)" << std::endl;

    // Optional per-iter timing breakdown (set ED_LANCZOS_PROFILE=1 to enable).
    // Sums of microseconds spent in each kernel across the whole run.
    const bool profile = []() {
        return ed::env::flag("ED_LANCZOS_PROFILE", false);
    }();
    double t_apply = 0, t_recur = 0, t_reorth = 0, t_normsc = 0, t_tridiag = 0;
    auto now_us = []() {
        auto t = std::chrono::steady_clock::now().time_since_epoch();
        return std::chrono::duration<double, std::micro>(t).count();
    };

    for (uint64_t j = 0; j < max_iter; ++j) {
        if (extras && extras->on_basis_vector) extras->on_basis_vector(j, v_current);
        // w = H * v_j  (real SpMV; uses our OMP team)
        double t0 = profile ? now_us() : 0.0;
        H_real(v_current, w.data(), static_cast<int>(N));
        if (profile) { t_apply += now_us() - t0; t0 = now_us(); }

        // (1) w -= beta_j * v_{j-1}
        // (2) alpha_j = <v_j, w_after>
        // (3) w -= alpha_j * v_j
        // -- all three steps fused into one OMP parallel region
        //    (Phase 6 #9). The barrier between the dot reduction and
        //    the second axpy keeps the math identical to the cblas
        //    version while paying only one fork/join instead of three.
        const double alpha_j = ed::parallel::fused_axpy_dot_axpy_real(
            N, beta[j], v_prev, v_current, w.data(),
            /*apply_beta_term=*/(j > 0));
        alpha.push_back(alpha_j);
        if (profile) { t_recur += now_us() - t0; t0 = now_us(); }

        // Local reorthogonalization against the K most-recent ring vectors,
        // walking the ring backward. Each pass is one fused OMP region
        // (dot + conditional axpy) instead of two separate cblas_* calls.
        // The K=0 vector is v_current itself, which we already orthogonalised
        // against in the fused 3-op above, so we start at k=1.
        const int num_reorth = std::min(ring_count - 1, reorth_K);
        if (num_reorth > 0) {
            ++selective_reorth_count;
            total_reorth_count += static_cast<uint64_t>(num_reorth);
            for (int k = 1; k <= num_reorth; ++k) {
                const int slot = (ring_head - 1 - k + 2 * max_recent) % max_recent;
                ed::parallel::fused_dot_axpy_real(
                    N, slab(slot), w.data(), ortho_threshold);
            }
        }
        if (profile) { t_reorth += now_us() - t0; t0 = now_us(); }

        // (4) norm = ||w||
        // (5) v_{j+1} = w / norm  (written DIRECTLY into the next ring slot)
        // -- fused into one OMP region (Phase 6 #9 + #10). Avoids the
        //    separate ``recent_buf[slot] = v_current`` memcpy that the
        //    previous version paid every iter.
        double* v_next_slab = slab(ring_head);
        norm = ed::parallel::fused_norm2_scale_real(
            N, w.data(), v_next_slab);
        if (profile) { t_normsc += now_us() - t0; }

        // Print sparingly to match the complex path's verbosity profile.
        if (j == 0 || (j + 1) % 100 == 0 || j + 1 == max_iter) {
            const double residual_error = (j == 0)
                ? norm / (std::abs(alpha_j) + norm)
                : norm / (std::abs(alpha_j) + std::abs(beta[j]) + norm);
            std::cout << "Iteration " << j + 1 << " of " << max_iter
                      << "  |  beta = " << std::scientific << std::setprecision(4)
                      << norm << "  |  residual = " << residual_error
                      << std::defaultfloat << std::endl;
        }

        // Breakdown: invariant subspace found.
        if (norm < tol) {
            std::cout << "Lanczos[real]: invariant subspace at iter "
                      << j + 1 << " (beta=" << std::scientific
                      << std::setprecision(2) << norm << std::defaultfloat
                      << ")" << std::endl;
            max_iter = j + 1;
            break;
        }
        beta.push_back(norm);

        // Eigenvalue convergence check (every iter, xdiag-style relative).
        // Skip the first ``exct`` iters: the tridiagonal isn't large
        // enough yet to host ``exct`` Ritz values.
        const double t_tri0 = profile ? now_us() : 0.0;
        if (!fixed_iters && j >= exct) {
            const uint64_t m_cur = alpha.size();
            std::vector<double> diag = alpha;
            std::vector<double> offd(m_cur - 1);
            for (uint64_t ii = 0; ii < m_cur - 1; ++ii) offd[ii] = beta[ii + 1];
            const int info = LAPACKE_dstevd(LAPACK_COL_MAJOR, 'N', m_cur,
                                            diag.data(), offd.data(), nullptr,
                                            m_cur);
            if (info == 0) {
                // Audit 2026-09: ghost filter for eigenvalue WINDOWS. With
                // local reorthogonalisation a converged eigenvalue re-emerges
                // as extra copies ("ghosts"), and the eigenvalue-change test
                // happily converges on them: the CLI returned E[0] == E[1] on
                // a non-degenerate chiral model. Apply the Cullum-Willoughby
                // test (Lanczos Algorithms for Large Symmetric Eigenvalue
                // Computations, ch. 4): a SIMPLE Ritz value of T that is also
                // an eigenvalue of T with its first row/column deleted is
                // spurious; multiple Ritz values are one converged level.
                // Only for exct > 1 (the extreme pair is never a ghost).
                if (exct > 1 && m_cur >= 3) {
                    diag = cullum_willoughby_filter(diag, alpha, beta, m_cur);
                }
                const uint64_t n_check = std::min<uint64_t>(exct, m_cur);
                std::vector<double> current(diag.begin(),
                                            diag.begin() + std::min<uint64_t>(n_check, diag.size()));
                if (!prev_eigenvalues.empty()
                    && prev_eigenvalues.size() >= n_check) {
                    double max_rel_change = 0.0;
                    for (uint64_t ii = 0; ii < n_check; ++ii) {
                        const double denom = std::max(
                            std::abs(current[ii]), 1e-300);
                        const double rel_change = std::abs(
                            current[ii] - prev_eigenvalues[ii]) / denom;
                        max_rel_change = std::max(max_rel_change, rel_change);
                    }
                    // GPU-parity fix (2026-09-11): when eigenvectors are
                    // requested, the Ritz-value stop alone leaves the
                    // higher members of the window with residuals ~sqrt(tol)
                    // (measured 5e-8 / 6e-6 for levels 2-3 at tol = 1e-10),
                    // which fails the SU(2) label certification (needs
                    // <= 1e-8) that the complex kernel's vectors pass. Gate
                    // the stop additionally on the free Lanczos bound
                    // |beta_m| |z_{m,i}| <= tol * max(1, |E_0|) for every
                    // requested level -- a few extra iterations.
                    // Windows (exct > 1) are gated on the bounds too, at
                    // sqrt(tol): a stalled Ritz value passes the change test
                    // for one step while its residual bound is still O(1)
                    // (measured: 3rd level off by 1e-2 on a random 256-state
                    // model with tol = 1e-10).
                    bool vec_ok = true;
                    if (max_rel_change < tol
                            && ((extras && extras->converge_vectors) || exct > 1)) {
                        std::vector<double> d2 = alpha, o2(m_cur > 1 ? m_cur - 1 : 0),
                                            z(m_cur * m_cur);
                        for (uint64_t ii = 0; ii + 1 < m_cur; ++ii) o2[ii] = beta[ii + 1];
                        if (LAPACKE_dstevd(LAPACK_COL_MAJOR, 'V', m_cur, d2.data(),
                                           o2.data(), z.data(), m_cur) == 0) {
                            const double vec_tol =
                                ((extras && extras->converge_vectors) ? tol : std::sqrt(tol))
                                * std::max(1.0, std::abs(current[0]));
                            for (uint64_t ii = 0; ii < n_check && vec_ok; ++ii) {
                                double best = std::numeric_limits<double>::infinity();
                                for (uint64_t k = 0; k < m_cur; ++k) {
                                    if (std::abs(d2[k] - current[ii])
                                            <= 1e-10 * (1.0 + std::abs(current[ii]))) {
                                        best = std::min(best,
                                            norm * std::abs(z[(m_cur - 1) + k * m_cur]));
                                    }
                                }
                                if (best > vec_tol) vec_ok = false;
                            }
                        }
                    }
                    if (max_rel_change < tol && vec_ok) {
                        std::cout << "Lanczos[real]: Eigenvalues converged at "
                                     "iteration " << j + 1
                                  << " (max rel change = " << std::scientific
                                  << std::setprecision(4) << max_rel_change
                                  << " < tol = " << tol << ")"
                                  << std::defaultfloat << std::endl;
                        converged = true;
                        max_iter = j + 1;
                        break;
                    }
                }
                prev_eigenvalues = std::move(current);
            }
        }
        if (profile) t_tridiag += now_us() - t_tri0;

        // v_{j+1} already lives in slab[ring_head] (written directly by
        // the fused norm+scale kernel above). Rotate the pointers so
        // next iter sees the right vectors -- no dim-N memcpy at all.
        v_prev    = v_current;
        v_current = v_next_slab;
        if (ring_count < max_recent) ++ring_count;
        ring_head = (ring_head + 1) % max_recent;
    }

    if (profile) {
        const double iters = static_cast<double>(alpha.size());
        std::cout << "Lanczos[real] PROFILE (per-iter avg, " << iters
                  << " iters):\n"
                  << "  apply (SpMV)              = "
                  << t_apply / iters / 1000.0 << " ms\n"
                  << "  fused 3-op recurrence     = "
                  << t_recur / iters / 1000.0 << " ms\n"
                  << "  fused dot+axpy reorth     = "
                  << t_reorth / iters / 1000.0 << " ms\n"
                  << "  fused norm+scale          = "
                  << t_normsc / iters / 1000.0 << " ms\n"
                  << "  tridiag+rel-tol check     = "
                  << t_tridiag / iters / 1000.0 << " ms\n"
                  << "  TOTAL inner loop          = "
                  << (t_apply + t_recur + t_reorth + t_normsc + t_tridiag)
                     / iters / 1000.0 << " ms\n";
    }

    // Solve the final Lanczos tridiagonal for the requested eigenvalues.
    const uint64_t m = alpha.size();
    std::vector<double> diag = alpha;
    std::vector<double> offd(m > 0 ? m - 1 : 0);
    for (uint64_t ii = 0; ii + 1 < m; ++ii) offd[ii] = beta[ii + 1];
    const int info = LAPACKE_dstevd(LAPACK_COL_MAJOR, 'N', m,
                                    diag.data(), offd.data(), nullptr, m);
    if (info != 0) {
        std::cerr << "Lanczos[real]: tridiagonal solver failed (info="
                  << info << ")" << std::endl;
        return;
    }
    // Ghost filter (see the convergence check above). The Ritz bounds /
    // vectors requested through ``extras`` are computed for the UNFILTERED
    // tridiagonal below and then aligned to the filtered eigenvalue list.
    std::vector<double> unfiltered = diag;
    if (exct > 1 && m >= 3) diag = cullum_willoughby_filter(diag, alpha, beta, m);

    const uint64_t n_eig = std::min<uint64_t>(exct, static_cast<uint64_t>(diag.size()));
    eigenvalues.assign(diag.begin(), diag.begin() + n_eig);
    if (extras) {
        extras->alpha     = alpha;
        extras->beta      = beta;
        extras->beta_last = norm;   // beta_m: the norm computed in the last step
        extras->ritz_bounds.clear();
        extras->ritz_vectors.clear();
        if (extras->want_ritz && m > 0) {
            std::vector<double> d2 = alpha, o2(m > 1 ? m - 1 : 0), z(m * m);
            for (uint64_t ii = 0; ii + 1 < m; ++ii) o2[ii] = beta[ii + 1];
            const int info2 = LAPACKE_dstevd(LAPACK_COL_MAJOR, 'V', m, d2.data(),
                                             o2.data(), z.data(), m);
            if (info2 == 0) {
                // Map each kept (filtered) eigenvalue back to its column of the
                // unfiltered eigendecomposition (first match by value).
                extras->ritz_vectors.assign(m * n_eig, 0.0);
                uint64_t start = 0;
                for (uint64_t i = 0; i < n_eig; ++i) {
                    uint64_t col = start;
                    while (col + 1 < m && std::abs(unfiltered[col] - eigenvalues[i])
                           > 1e-12 * (1.0 + std::abs(eigenvalues[i]))) ++col;
                    start = col + 1;
                    std::copy(z.begin() + col * m, z.begin() + (col + 1) * m,
                              extras->ritz_vectors.begin() + i * m);
                    extras->ritz_bounds.push_back(std::abs(norm) * std::abs(z[(m - 1) + col * m]));
                }
            }
        }
    }
    if (iters_out) *iters_out = m;
    // A run that exhausted the full space (m == N) is exact by construction.
    if (converged_out) *converged_out = converged || (m == N);

    std::cout << "Lanczos[real]: " << m << " iterations, "
              << total_reorth_count << " local-reorth axpys ("
              << selective_reorth_count << " passes)"
              << (converged ? " [converged]" : "") << std::endl;
}

// Full diagonalization: dense LAPACK inside the dense window; larger blocks
// are refused (they belong to the Krylov lanes).
void full_diagonalization(std::function<void(const Complex*, Complex*, int)> H, uint64_t N, uint64_t num_eigs,
                       std::vector<double>& eigenvalues,
                       bool compute_eigenvectors,
                       const ed::matvec::MatVecOperator* op_for_dense,
                       std::vector<std::vector<Complex>>* eigenvectors_out) {
    std::cout << "Starting full diagonalization for matrix of dimension " << N << std::endl;
    if (eigenvectors_out) eigenvectors_out->clear();

    // Phase 6.1: dim-aware OMP+BLAS thread cap for the column build; the
    // dense eigensolve below lifts it (see dense_solve_budget).
    const ed::parallel::ThreadBudgetScope budget(
        ed::parallel::auto_threads_for_dim(N));

    // Dense window: LAPACK (zheevd/dsyevd) computes the COMPLETE spectrum
    // correctly and stably up to this dimension. 120000 aligns with the
    // NLCE router's --exact_max_block default, so a cluster the plan
    // admits is never refused here; at the cap the matrix is ~230 GB
    // complex / ~115 GB real (transient peak ~1.5x during the real-path
    // conversion) -- fat-node territory, and undersized machines fail as
    // a clean bad_alloc up front, before any solve work. Above the
    // window the call is a hard error. ED_FULLDIAG_DENSE_MAX overrides in
    // either direction.
    uint64_t DENSE_THRESHOLD = 120000;
    if (const char* env_max = ed::env::raw("ED_FULLDIAG_DENSE_MAX")) {
        const unsigned long long v = std::strtoull(env_max, nullptr, 10);
        if (v > 0) DENSE_THRESHOLD = static_cast<uint64_t>(v);
    }
    
    if (N <= DENSE_THRESHOLD) {
        // For smaller matrices, use dense approach with MKL for best performance
        std::cout << "Using dense diagonalization with MKL/LAPACK" << std::endl;
        
        // Check memory requirements - estimate total needed including workspace
        size_t matrix_size = static_cast<size_t>(N) * N;
        size_t bytes_for_matrix = matrix_size * sizeof(Complex);
        size_t bytes_for_eigenvalues = N * sizeof(double);
        
        // Determine if we can use memory-efficient partial eigenvalue computation
        uint64_t actual_num_eigs = std::min(num_eigs, N);
        bool use_partial_solver = (actual_num_eigs < N / 2) && (N > 1000);  // Use zheevr for subset
        
        if (use_partial_solver && compute_eigenvectors) {
            // zheevr needs: matrix + num_eigs eigenvectors + workspace
            size_t bytes_for_evecs = static_cast<size_t>(actual_num_eigs) * N * sizeof(Complex);
            std::cout << "Matrix requires " << bytes_for_matrix / (1024.0 * 1024.0 * 1024.0) << " GB" << std::endl;
            std::cout << "Eigenvectors require " << bytes_for_evecs / (1024.0 * 1024.0 * 1024.0) << " GB" << std::endl;
            std::cout << "Using memory-efficient partial eigensolver (zheevr) for " << actual_num_eigs << "/" << N << " eigenvalues" << std::endl;
        } else {
            // Full solver - eigenvectors overwrite the matrix (no extra allocation needed)
            std::cout << "Matrix requires " << bytes_for_matrix / (1024.0 * 1024.0 * 1024.0) << " GB of memory" << std::endl;
        }
        
        // Allocate memory for dense matrix with error checking
        std::vector<Complex> dense_matrix;
        try {
            dense_matrix.resize(matrix_size, Complex(0.0, 0.0));
        } catch (const std::bad_alloc& e) {
            std::cerr << "Failed to allocate the " << N << "x" << N
                      << " dense matrix ("
                      << bytes_for_matrix / (1024.0 * 1024.0 * 1024.0)
                      << " GB): this machine cannot hold the block. Shrink "
                      << "the block via symmetry, request fewer eigenvalues, "
                      << "or run on a node with more RAM." << std::endl;
            throw;
        }
        
        std::cout << "Constructing dense matrix..." << std::endl;

        // FAST PATH: assemble the matrix directly from the operator's sparse term
        // structure -- O(nnz), reentrant, parallel over columns -- instead of N
        // full matvecs (O(dim*nnz)). Supported by the full-space / fixed-Sz lanes;
        // symmetry lanes (and distributed/GPU callers with no operator handle)
        // return false and fall through to the matvec column build below.
        bool built_direct =
            (op_for_dense != nullptr) &&
            op_for_dense->try_build_dense_columns(dense_matrix.data(), N);

        if (!built_direct) {
            // Fallback: build column j = H * e_j. SEQUENTIAL outer loop -- the CPU
            // matvec is NOT reentrant (the backend owns shared CSR/scratch), so it
            // cannot be called concurrently; H parallelizes each column internally.
            const uint64_t chunk_size = std::max(static_cast<uint64_t>(1), N / 100);
            for (uint64_t j = 0; j < N; j++) {
                std::vector<Complex> unit_vec(N, Complex(0.0, 0.0));
                unit_vec[j] = Complex(1.0, 0.0);
                std::vector<Complex> col_j(N);
                H(unit_vec.data(), col_j.data(), N);
                for (uint64_t i = 0; i < N; i++) {
                    dense_matrix[j*N + i] = col_j[i];
                }
                if (j % chunk_size == 0 || j == N-1) {
                    double percentage = 100.0 * j / N;
                    uint64_t barWidth = 50;
                    uint64_t pos = barWidth * j / N;
                    std::cout << "\rProgress: [";
                    for (uint64_t k = 0; k < barWidth; ++k) {
                        if (k < pos) std::cout << "=";
                        else if (k == pos) std::cout << ">";
                        else std::cout << " ";
                    }
                    std::cout << "] " << std::fixed << std::setprecision(1) << percentage << "%" << std::flush;
                    if (j == N-1) std::cout << std::endl;
                }
            }
        }
        std::cout << "Dense matrix constructed" << std::endl;

        // The enclosing ThreadBudgetScope soft-caps threads at ~8 (tuned for
        // bandwidth-bound Lanczos SpMV / BLAS-1). The dense LAPACK eigensolve
        // below is compute-bound and scales to all cores (zheevd/zheevr), so
        // lift the cap for it -- otherwise it runs ~3-4x slower than the BLAS
        // backend can (measured: 9 s vs 2.6 s at dim 3432). ThreadBudgetScope
        // clamps the request to the hardware maximum and restores on scope exit.
        //
        // Nesting-aware: when this runs INSIDE a sector-parallel region (the
        // streaming-symmetry FULL loop spreads independent sectors across cores
        // via an outer `omp parallel for`), keep the eigensolve
        // single-threaded -- otherwise N_sectors x P_cores oversubscribes. A
        // standalone FULL solve takes all cores.
        // Audit 2026-09: "all cores" is wrong for small blocks. OpenBLAS's
        // dsytrd/zhetrd is a chain of O(N) BLAS-2 calls, and every one of them
        // fans out to the whole (spinning) thread pool: measured 0.13 s .. 10 s
        // for the SAME dim-924 block depending on what else was running, vs
        // ~40 ms single-threaded. Scale the team with the block: one thread
        // per ~1024 rows, all cores from ~32k rows on. ED_FULLDIAG_THREADS
        // overrides.
        int dense_threads = static_cast<int>(std::max<uint64_t>(1, N / 1024));
        if (const char* e = ed::env::raw("ED_FULLDIAG_THREADS")) {
            const int v = std::atoi(e);
            if (v > 0) dense_threads = v;
        }
#ifdef _OPENMP
        if (omp_in_parallel()) dense_threads = 1;
#endif
        ed::parallel::ThreadBudgetScope dense_solve_budget(dense_threads, dense_threads);

        // Allocate array for eigenvalues
        std::vector<double> evals(N);
        lapack_int info;
        
        // Real-matrix fast path: the assembled sector matrix is real whenever the
        // Hamiltonian is real and the sector carries no complex Bloch phase --
        // every no-symmetry / fixed-Sz block (real Heisenberg / XXZ) and the k=0
        // / k=pi momentum sectors. Real LAPACK (dsyevd / dsyevr) is ~2x faster and
        // uses half the working memory of the complex driver, with identical
        // eigenvalues. Detect once (O(N^2), trivial next to the O(N^3) solve).
        // ED_FULLDIAG_FORCE_COMPLEX forces the complex driver (A/B timing +
        // real-vs-complex equivalence checks).
        bool matrix_is_real = !ed::env::flag("ED_FULLDIAG_FORCE_COMPLEX", false);
        for (size_t i = 0; i < matrix_size && matrix_is_real; ++i)
            if (std::abs(dense_matrix[i].imag()) > 1e-12) matrix_is_real = false;

        if (matrix_is_real) {
            std::cout << "Matrix is real -> real LAPACK fast path ("
                      << (use_partial_solver ? "dsyevr" : "dsyevd") << ")" << std::endl;
            std::vector<double> rdense(matrix_size);
            for (size_t i = 0; i < matrix_size; ++i) rdense[i] = dense_matrix[i].real();
            std::vector<Complex>().swap(dense_matrix);  // free complex buffer (half mem)
            if (use_partial_solver) {
                std::vector<double> revecs;
                if (compute_eigenvectors) revecs.resize(static_cast<size_t>(actual_num_eigs) * N);
                lapack_int m_found;
                std::vector<lapack_int> isuppz(2 * actual_num_eigs);
                info = LAPACKE_dsyevr(LAPACK_COL_MAJOR, compute_eigenvectors ? 'V' : 'N',
                                      'I', 'U', N, rdense.data(), N, 0.0, 0.0,
                                      1, actual_num_eigs, LAPACKE_dlamch('S'), &m_found,
                                      evals.data(),
                                      compute_eigenvectors ? revecs.data() : nullptr,
                                      N, isuppz.data());
                if (info != 0) { std::cerr << "LAPACKE_dsyevr failed with error code " << info << std::endl; return; }
                std::cout << "Partial eigenvalue decomposition completed (" << m_found << " eigenvalues found)" << std::endl;
                eigenvalues.resize(m_found);
                for (lapack_int i = 0; i < m_found; ++i) eigenvalues[i] = evals[i];
                if (compute_eigenvectors && eigenvectors_out != nullptr) {
                    std::vector<std::vector<Complex>> eigenvector_list(m_found);
                    for (lapack_int i = 0; i < m_found; ++i) {
                        eigenvector_list[i].resize(N);
                        for (size_t j = 0; j < N; ++j) eigenvector_list[i][j] = Complex(revecs[static_cast<size_t>(i) * N + j], 0.0);
                    }
                    *eigenvectors_out = std::move(eigenvector_list);
                }
            } else {
                info = LAPACKE_dsyevd(LAPACK_COL_MAJOR, compute_eigenvectors ? 'V' : 'N',
                                      'U', N, rdense.data(), N, evals.data());
                if (info != 0) { std::cerr << "LAPACKE_dsyevd failed with error code " << info << std::endl; return; }
                std::cout << "Eigenvalue decomposition completed (divide-and-conquer, real)" << std::endl;
                eigenvalues.resize(actual_num_eigs);
                for (size_t i = 0; i < actual_num_eigs; ++i) eigenvalues[i] = evals[i];
                if (compute_eigenvectors && eigenvectors_out != nullptr) {
                    std::vector<std::vector<Complex>> eigenvector_list(actual_num_eigs);
                    for (size_t i = 0; i < actual_num_eigs; ++i) {
                        eigenvector_list[i].resize(N);
                        for (size_t j = 0; j < N; ++j) eigenvector_list[i][j] = Complex(rdense[i * N + j], 0.0);
                    }
                    *eigenvectors_out = std::move(eigenvector_list);
                }
            }
        } else if (use_partial_solver) {
            // ===== Memory-efficient partial eigenvalue computation using zheevr =====
            // zheevr uses the Relatively Robust Representations (RRR) algorithm
            // and can compute a subset of eigenvalues much more efficiently
            
            std::vector<Complex> evecs_partial;
            if (compute_eigenvectors) {
                evecs_partial.resize(static_cast<size_t>(actual_num_eigs) * N);
            }
            
            lapack_int m_found;  // Number of eigenvalues found
            std::vector<lapack_int> isuppz(2 * actual_num_eigs);  // Support of eigenvectors
            
            // Compute smallest actual_num_eigs eigenvalues (indices 1 to actual_num_eigs in Fortran 1-based)
            info = LAPACKE_zheevr(LAPACK_COL_MAJOR, 
                                  compute_eigenvectors ? 'V' : 'N',  // Compute eigenvectors?
                                  'I',                               // Compute eigenvalues by index range
                                  'U',                               // Upper triangular
                                  N,                                 // Matrix dimension
                                  reinterpret_cast<lapack_complex_double*>(dense_matrix.data()),
                                  N,                                 // Leading dimension
                                  0.0, 0.0,                          // VL, VU (unused when range='I')
                                  1, actual_num_eigs,                // IL, IU: eigenvalue indices (1-based)
                                  LAPACKE_dlamch('S'),               // Abstol
                                  &m_found,                          // Output: number found
                                  evals.data(),                      // Output: eigenvalues
                                  compute_eigenvectors ? reinterpret_cast<lapack_complex_double*>(evecs_partial.data()) : nullptr,
                                  N,                                 // Leading dimension of Z
                                  isuppz.data());                    // Support array
            
            if (info != 0) {
                std::cerr << "LAPACKE_zheevr failed with error code " << info << std::endl;
                return;
            }
            
            std::cout << "Partial eigenvalue decomposition completed (" << m_found << " eigenvalues found)" << std::endl;
            
            // Extract eigenvalues
            eigenvalues.resize(m_found);
            for (lapack_int i = 0; i < m_found; i++) {
                eigenvalues[i] = evals[i];
            }
            
            // Hand the requested eigenvectors back in memory
            if (compute_eigenvectors && eigenvectors_out != nullptr) {
                
                // Convert to vector of vectors format - read directly from evecs_partial
                std::vector<std::vector<Complex>> eigenvector_list(m_found);
                for (lapack_int i = 0; i < m_found; i++) {
                    eigenvector_list[i].resize(N);
                    // Eigenvectors are stored column-major in evecs_partial
                    for (size_t j = 0; j < N; j++) {
                        eigenvector_list[i][j] = evecs_partial[static_cast<size_t>(i) * N + j];
                    }
                }
                
                *eigenvectors_out = std::move(eigenvector_list);
            }
        } else {
            // ===== Full eigenvalue computation using zheevd (divide-and-conquer) =====
            // zheevd is typically 2-4x faster than zheev for large matrices
            // Note: eigenvectors overwrite dense_matrix, so no extra allocation needed!
            
            info = LAPACKE_zheevd(LAPACK_COL_MAJOR, 
                                  compute_eigenvectors ? 'V' : 'N', 
                                  'U', 
                                  N,
                                  reinterpret_cast<lapack_complex_double*>(dense_matrix.data()),
                                  N, 
                                  evals.data());
            
            if (info != 0) {
                std::cerr << "LAPACKE_zheevd failed with error code " << info << std::endl;
                return;
            }
            
            std::cout << "Eigenvalue decomposition completed (divide-and-conquer)" << std::endl;

            // Extract requested number of eigenvalues
            eigenvalues.resize(actual_num_eigs);
            for (size_t i = 0; i < actual_num_eigs; i++) {
                eigenvalues[i] = evals[i];
            }
            
            // Hand the requested eigenvectors back in memory
            // Note: eigenvectors are now stored IN dense_matrix (column-major)
            if (compute_eigenvectors && eigenvectors_out != nullptr) {
                
                // Convert dense_matrix (which now contains eigenvectors) to vector of vectors format
                // No intermediate copy needed - read directly from dense_matrix
                std::vector<std::vector<Complex>> eigenvector_list(actual_num_eigs);
                for (size_t i = 0; i < actual_num_eigs; i++) {
                    eigenvector_list[i].resize(N);
                    // Eigenvectors are stored column-major: evec[i] is at dense_matrix[i*N:(i+1)*N]
                    for (size_t j = 0; j < N; j++) {
                        eigenvector_list[i][j] = dense_matrix[i * N + j];
                    }
                }
                
                *eigenvectors_out = std::move(eigenvector_list);
            }
        }
    } else {
        // Above the dense window: HARD ERROR. The callers (the solve lane's
        // FullDiag method and the small-block thermal fallback) only route
        // small blocks here; Krylov lanes serve everything larger. The old
        // Eigen-sparse "full diagonalization" fallback densified internally
        // on one thread, heap-corrupted at N=32768 on symmetry-free clusters,
        // and could not honestly deliver all N eigenvalues (retired
        // 2026-07-20).
        throw std::runtime_error(
            "full_diagonalization: dimension " + std::to_string(N) +
            " exceeds the dense limit (" + std::to_string(DENSE_THRESHOLD) +
            "). Reduce the block with symmetry, use a Krylov method, or set "
            "ED_FULLDIAG_DENSE_MAX above " + std::to_string(N) +
            " if the dense matrix genuinely fits in RAM.");
    }
    
    std::cout << "Full diagonalization completed successfully" << std::endl;
}
