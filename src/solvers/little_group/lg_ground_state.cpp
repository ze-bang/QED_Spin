// =============================================================================
// src/solvers/little_group/lg_ground_state.cpp -- certified ground-state vector of one
// block, the streamed raw momentum sectors, and shared sector data.
// Part of the little-group engine; see lg_internal.h for the file map.
// =============================================================================

#include "lg_internal.h"

namespace ed::solvers {

using namespace lg_detail;

namespace lg_detail {   // solve_gs_vector backs solve_block_eigenpairs (lg_block_solve.cpp)

// TWO-PASS no-reorth ground-state Ritz vector.
//
// Pass 1: pure three-term recurrence (no reorth, no stored basis), tridiag
// only, with the same ghost-aware k=1 Paige-bound gate the star scan uses
// (a ghost is a COPY of the converged extreme, so the FIRST converged
// distinct Ritz value IS E0). Pass 2: replay the recurrence with the
// STORED alpha/beta -- no inner products, so the trajectory is identical
// arithmetic to pass 1 -- accumulating u = sum_j z_j V_j on the fly.
// Memory: four n-vectors, independent of the iteration count. The caller's
// residual guard stays the arbiter; on a miss we restart the whole
// two-pass seeded by the current u (Lanczos restarted on an approximate
// eigenvector converges rapidly), up to `restarts` times.
[[nodiscard]] std::pair<double, std::vector<Complex>>
solve_gs_vector_two_pass(const ed::matvec::MatVecOperator& hk,
                         std::size_t n)
{
    const std::size_t max_iter = std::min<std::size_t>(n, kLgGsTwoPassMaxIter);
    const int restarts = kLgGsRestarts;

    std::vector<Complex> v0(n);
    {
        std::mt19937_64 gen(0x51ED900DULL);
        std::normal_distribution<double> nd(0.0, 1.0);
        for (auto& v : v0) v = Complex(nd(gen), nd(gen));
    }
    auto nrm2 = [](const std::vector<Complex>& x) {
        double s = 0.0;
#ifdef _OPENMP
#   pragma omp parallel for reduction(+ : s) schedule(static)
#endif
        for (long long i = 0; i < static_cast<long long>(x.size()); ++i)
            s += std::norm(x[static_cast<std::size_t>(i)]);
        return std::sqrt(s);
    };
    auto scal = [](std::vector<Complex>& x, double a) {
#ifdef _OPENMP
#   pragma omp parallel for schedule(static)
#endif
        for (long long i = 0; i < static_cast<long long>(x.size()); ++i)
            x[static_cast<std::size_t>(i)] *= a;
    };

    double E0 = 0.0;
    std::vector<Complex> u;
    std::vector<Complex> seed = std::move(v0);
    {
        const double s0 = nrm2(seed);
        if (!(s0 > 0.0))
            throw std::runtime_error("little_group two-pass: zero seed");
        scal(seed, 1.0 / s0);
    }

    for (int attempt = 0; attempt <= restarts; ++attempt) {
        // ---------------- pass 1: tridiag only -------------------------
        std::vector<double> alpha, beta{0.0};
        std::vector<Complex> vp(n, Complex(0, 0)), vc = seed, w(n);
        bool done = false;
        std::vector<double> ritz_z;   // column 0 at exit
        std::size_t m = 0;
        while (!done && m < max_iter) {
            hk.apply(vc.data(), w.data(), n);
            double a = 0.0;
#ifdef _OPENMP
#   pragma omp parallel for reduction(+ : a) schedule(static)
#endif
            for (long long i = 0; i < static_cast<long long>(n); ++i)
                a += std::real(std::conj(vc[static_cast<std::size_t>(i)])
                               * w[static_cast<std::size_t>(i)]);
            alpha.push_back(a);
            const double bprev = beta.back();
#ifdef _OPENMP
#   pragma omp parallel for schedule(static)
#endif
            for (long long i = 0; i < static_cast<long long>(n); ++i) {
                const std::size_t ii = static_cast<std::size_t>(i);
                w[ii] -= a * vc[ii] + bprev * vp[ii];
            }
            const double b = nrm2(w);
            beta.push_back(b);
            ++m;
            if (!(b > 1e-300)) { done = true; break; }   // invariant subspace
            std::swap(vp, vc);
            std::swap(vc, w);
            scal(vc, 1.0 / b);
            if (m >= 3 && m % 10 == 0) {
                // Paige bound on the smallest Ritz value only.
                std::vector<double> d(alpha), e(m > 1 ? m - 1 : 1, 0.0);
                for (std::size_t i = 0; i + 1 < m; ++i) e[i] = beta[i + 1];
                std::vector<double> zz(m * m, 0.0);
                if (LAPACKE_dstevd(LAPACK_COL_MAJOR, 'V',
                                   static_cast<lapack_int>(m), d.data(),
                                   e.data(), zz.data(),
                                   static_cast<lapack_int>(m)) == 0) {
                    const double scale = std::max(
                        {std::abs(d[0]), std::abs(d[m - 1]), 1e-300});
                    if (beta[m] * std::abs(zz[m - 1]) < 1e-9 * scale)
                        done = true;
                }
            }
        }
        if (m == 0)
            throw std::runtime_error("little_group two-pass: empty tridiag");
        {
            std::vector<double> d(alpha), e(m > 1 ? m - 1 : 1, 0.0);
            for (std::size_t i = 0; i + 1 < m; ++i) e[i] = beta[i + 1];
            std::vector<double> zz(m * m, 0.0);
            if (LAPACKE_dstevd(LAPACK_COL_MAJOR, 'V',
                               static_cast<lapack_int>(m), d.data(), e.data(),
                               zz.data(), static_cast<lapack_int>(m)) != 0)
                throw std::runtime_error(
                    "little_group two-pass: tridiag eigensolve failed");
            E0 = d[0];
            ritz_z.assign(zz.begin(), zz.begin() + m);
        }
        // ---------------- pass 2: replay + accumulate ------------------
        u.assign(n, Complex(0, 0));
        std::fill(vp.begin(), vp.end(), Complex(0, 0));
        vc = seed;
        for (std::size_t j = 0; j < m; ++j) {
            const double zj = ritz_z[j];
#ifdef _OPENMP
#   pragma omp parallel for schedule(static)
#endif
            for (long long i = 0; i < static_cast<long long>(n); ++i)
                u[static_cast<std::size_t>(i)] +=
                    zj * vc[static_cast<std::size_t>(i)];
            if (j + 1 >= m) break;
            hk.apply(vc.data(), w.data(), n);
            const double a = alpha[j], bprev = beta[j], bnext = beta[j + 1];
#ifdef _OPENMP
#   pragma omp parallel for schedule(static)
#endif
            for (long long i = 0; i < static_cast<long long>(n); ++i) {
                const std::size_t ii = static_cast<std::size_t>(i);
                w[ii] -= a * vc[ii] + bprev * vp[ii];
            }
            std::swap(vp, vc);
            std::swap(vc, w);
            scal(vc, 1.0 / bnext);
        }
        const double un = nrm2(u);
        if (!(un > 0.0))
            throw std::runtime_error("little_group two-pass: zero Ritz vector");
        scal(u, 1.0 / un);
        // Residual check; restart seeded by u on a miss.
        hk.apply(u.data(), w.data(), n);
        double num = 0.0, ray = 0.0;
#ifdef _OPENMP
#   pragma omp parallel for reduction(+ : num, ray) schedule(static)
#endif
        for (long long i = 0; i < static_cast<long long>(n); ++i) {
            const std::size_t ii = static_cast<std::size_t>(i);
            ray += std::real(std::conj(u[ii]) * w[ii]);
        }
#ifdef _OPENMP
#   pragma omp parallel for reduction(+ : num) schedule(static)
#endif
        for (long long i = 0; i < static_cast<long long>(n); ++i) {
            const std::size_t ii = static_cast<std::size_t>(i);
            num += std::norm(w[ii] - ray * u[ii]);
        }
        E0 = ray;
        if (std::sqrt(num) <= kLgGsResidTol || attempt == restarts) break;
        seed = u;                       // restarted refinement
    }
    return {E0, std::move(u)};
}

// GS eigenpair of the PLAIN k0 sector with an in-memory eigenvector:
// dense for small blocks; above the dense crossover either the two-pass
// no-reorth lane (large n -- see kLgTwoPassMinDim) or FullCGS2 Lanczos
// + kept-basis Ritz vector (small n). Residual-
// guarded -- a failed vector THROWS (the caller's point_group='full'
// contract is loud, and there is no cheaper correct fallback for a
// vector consumer).
[[nodiscard]] std::pair<double, std::vector<Complex>>
solve_gs_vector(const ed::matvec::MatVecOperator& hk, int dense_max_dim)
{
    const std::size_t n = hk.dim();
    if (n == 0) throw std::runtime_error("little_group: empty GS sector");
    double E0 = 0.0;
    std::vector<Complex> u(n);
    if (n <= static_cast<std::size_t>(std::max(dense_max_dim, 2))) {
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> es(materialize(hk));
        E0 = es.eigenvalues()(0);
        for (std::size_t i = 0; i < n; ++i)
            u[i] = es.eigenvectors()(static_cast<Eigen::Index>(i), 0);
    } else if (n > kLgTwoPassMinDim) {
        auto pr = solve_gs_vector_two_pass(hk, n);
        E0 = pr.first;
        u  = std::move(pr.second);
    } else {
        ed::matvec::CpuBackend be;
        std::vector<Complex> v0(n);
        std::mt19937_64 gen(0x51ED900DULL);
        std::normal_distribution<double> nd(0.0, 1.0);
        for (auto& v : v0) v = Complex(nd(gen), nd(gen));
        ed::krylov::LanczosKernelOptions kopts;
        kopts.max_iter   = std::min<std::size_t>(n, kLgGsSmallMaxIter);
        kopts.reorth     = ed::krylov::ReorthPolicy::FullCGS2;
        kopts.keep_basis = true;
        kopts.dim_cap    = n;
        auto apply_H = [&hk](const Complex* in, Complex* out, std::size_t nn) {
            hk.apply(in, out, nn);
        };
        auto kres = ed::krylov::lanczos_kernel(be, apply_H, n, v0.data(),
                                               kopts);
        const std::size_t m = kres.alpha.size();
        if (m == 0) throw std::runtime_error("little_group: GS Lanczos "
                                             "produced an empty tridiag");
        std::vector<double> diag = kres.alpha;
        std::vector<double> off(m > 1 ? m - 1 : 1, 0.0);
        for (std::size_t i = 0; i + 1 < m; ++i) off[i] = kres.beta[i + 1];
        std::vector<double> z(m * m, 0.0);
        const lapack_int info = LAPACKE_dstevd(
            LAPACK_COL_MAJOR, 'V', static_cast<lapack_int>(m),
            diag.data(), off.data(), z.data(), static_cast<lapack_int>(m));
        if (info != 0)
            throw std::runtime_error("little_group: tridiag eigensolve "
                                     "failed (dstevd info != 0)");
        E0 = diag[0];
        std::fill(u.begin(), u.end(), Complex(0, 0));
        for (std::size_t j = 0; j < m; ++j) {
            const Complex* vj = kres.basis[j].get();
            const double   yj = z[j];              // column 0, row j
            if (std::abs(yj) < 1e-300) continue;
            for (std::size_t i = 0; i < n; ++i) u[i] += yj * vj[i];
        }
    }
    // Residual guard: the DSSF consumes this vector, so a stale pair is
    // silently-wrong physics -- verify before returning.
    std::vector<Complex> hu(n);
    hk.apply(u.data(), hu.data(), n);
    double num = 0.0, den = 1e-300;
    for (std::size_t i = 0; i < n; ++i) {
        num += std::norm(hu[i] - E0 * u[i]);
        den += std::norm(u[i]);
    }
    // Residual acceptance -- shared with the two-pass inner loop via
    // kLgGsResidTol.
    const double resid_tol = kLgGsResidTol;
    if (std::sqrt(num / den) > resid_tol) {
        char buf[64];
        std::snprintf(buf, sizeof buf, "%.3e", std::sqrt(num / den));
        throw std::runtime_error(
            std::string("little_group: GS eigenvector residual ") + buf
            + " exceeds tolerance " + std::to_string(resid_tol)
            + " -- declining the factorized DSSF.");
    }
    const double inv = 1.0 / std::sqrt(den);
    for (auto& c : u) c *= inv;
    return {E0, std::move(u)};
}

}  // namespace lg_detail

void little_group_k_sectors_stream(
    const ::Operator&                    op,
    const std::vector<std::vector<int>>& abelian_group,
    int                                  n_sites,
    int                                  n_up,
    int                                  sz_parity,
    const std::function<void(ed::symmetry::RepSectorData&)>& fn)
{
    // Build ONE raw momentum
    // sector at a time, hand it to ``fn``, then free it before building the
    // next. Holding every k-sector resident
    // costs ~15-20 GB/sector at N=36 half-filling -- 12 sectors
    // OOMs a 128 GB node. This keeps the resident set at one destination
    // sector for the factorized static/dynamical structure-factor loops.
    LittleGroupOptions o;
    o.n_up          = n_up;
    o.sz_parity     = sz_parity;
    o.spin_flip     = 0;      // destination sectors are RAW
    o.time_reversal = 0;
    EngineContext cx;
    bool tr_on = false;
    make_engine_context(op, abelian_group, {}, n_sites, o, cx, tr_on);
    for (int k = 0; k < cx.n_irr_raw; ++k) {
        auto rd = build_k_sector(cx, k, n_up);
        if (!rd.reps.empty()) fn(rd);
    }
}

std::shared_ptr<const ed::symmetry::RepSectorData>
share_rep_sector(ed::symmetry::RepSectorData rd)
{
    auto p = std::make_shared<ed::symmetry::RepSectorData>(std::move(rd));
    p->build_perm_lut();
    return p;
}

}  // namespace ed::solvers
