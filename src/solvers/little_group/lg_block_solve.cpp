// =============================================================================
// src/solvers/little_group/lg_block_solve.cpp -- the per-block eigensolve driver: the dense
// solve, the crossover, and the Backend-templated Krylov lanes (lowest-level scan, Krylov-Schur,
// certified GS vector, pruning estimate), instantiated for CpuBackend and CudaBackend.
// Part of the little-group engine; see lg_internal.h for the file map.
// =============================================================================

#include "lg_internal.h"

#include <algorithm>
#include <numeric>
#include <optional>
#include <random>
#include <type_traits>

namespace ed::solvers {

using namespace lg_detail;

namespace lg_detail {

// Per-block solve callbacks -----------------------------------------------

// Dense eigenvalues (ascending) of a materialized block through LAPACK
// divide-and-conquer (dsyevd / zheevd) -- the SAME threaded solver the
// full-diag lane uses (lanczos.cpp) -- not Eigen's SelfAdjointEigenSolver,
// whose tridiagonalisation is single-threaded (a 19,264-dim projected block
// spins for HOURS on one core while the OpenMP pool sits idle). Real blocks
// (real momenta under time reversal) take the ~2x cheaper dsyevd real path.
// The Eigen matrix is column-major ==
// LAPACK_COL_MAJOR, so the complex solve runs in place on its storage.
[[nodiscard]] std::vector<double>
dense_eigenvalues_inplace(Eigen::MatrixXcd& Hb) {
    const lapack_int n = static_cast<lapack_int>(Hb.rows());
    std::vector<double> w(static_cast<std::size_t>(n), 0.0);
    if (n == 0) return w;

    double max_imag = 0.0;             // is this block real (up to roundoff)?
    for (Eigen::Index j = 0; j < Hb.cols(); ++j)
        for (Eigen::Index i = j; i < Hb.rows(); ++i) {
            const double a = std::abs(Hb(i, j).imag());
            if (a > max_imag) max_imag = a;
        }

    lapack_int info;
    if (max_imag <= 1.0e-12) {
        Eigen::MatrixXd R = Hb.real();  // symmetric; LAPACK reads upper only
        info = LAPACKE_dsyevd(LAPACK_COL_MAJOR, 'N', 'U', n, R.data(), n,
                              w.data());
    } else {
        info = LAPACKE_zheevd(
            LAPACK_COL_MAJOR, 'N', 'U', n,
            reinterpret_cast<lapack_complex_double*>(Hb.data()), n, w.data());
    }
    if (info != 0)
        throw std::runtime_error(
            "little_group: dense block eigensolve failed (info = "
            + std::to_string(info) + ")");
    return w;
}

[[nodiscard]] std::vector<double>
dense_block_eigenvalues(const ed::LinearOperator& mv) {
    Eigen::MatrixXcd Hb = materialize(mv);
    return dense_eigenvalues_inplace(Hb);
}

// Full-spectrum: dense eigenvalues of the (projected or plain) block, through
// dense_eigenvalues_inplace (threaded LAPACK divide-and-conquer; real blocks
// take the ~2x cheaper real path).
[[nodiscard]] std::vector<double>
solve_block_full(const ed::LinearOperator& mv) {
    if (mv.dim() == 0) return {};
    return dense_block_eigenvalues(mv);
}

// Shared dense/Lanczos crossover for the lowest-k path. Kept in one place so
// the device lane of the sectors eigensolve (lg_sectors.cpp) makes exactly the same dense-vs-Lanczos
// decision as the CPU ``solve_block_lowest``.
[[nodiscard]] std::uint64_t lowest_dense_floor(std::size_t k, int dense_max_dim) {
    // An explicit crossover (EigsOptions::dense_max_dim >= 0) is the caller's: 0 sends
    // every block above dimension 2 to Krylov, a large value solves exactly.
    if (dense_max_dim >= 0) return static_cast<std::uint64_t>(dense_max_dim);
    // Automatic: sized by the eigenvalue-scan iteration cap max(40k, 400).
    const std::uint64_t max_iter_cap =
        std::max<std::uint64_t>(40u * static_cast<std::uint64_t>(k), 400u);
    // With the contiguous k-lowest Paige gate the Lanczos path is honest at
    // every dim (at worst a budget-capped block returns fewer values flagged
    // unconverged), so the floor only needs to cover the within-block
    // degeneracy regime: dense resolves true multiplicities that a
    // single-vector recurrence cannot. 4x the cap (1600 at k <= 10) covers
    // the validated degenerate cases (the 4x4 n_up=8 blocks ~800) and
    // leaves larger blocks to Lanczos, where threaded zheevd would cost
    // 8-15 s of latency-bound time PER BLOCK in the serial star walk
    // (an N=20 ring walk: 227 s dense vs 0.7 s). Pass a larger
    // dense_max_dim when a mid-band block needs exact multiplicities.
    return 4u * max_iter_cap;
}


// The `want` lowest levels of a block by a dense solve on the host: LAPACK
// divide-and-conquer for values, Eigen's eigensolver with vectors.
[[nodiscard]] BlockSolution solve_block_dense(const ed::LinearOperator& H, std::size_t want, bool vectors) {
    BlockSolution sol;
    const std::size_t nb = H.dim();
    if (nb == 0) return sol;
    const std::size_t k = std::min(std::max<std::size_t>(want, 1), nb);
    if (!vectors) {
        const std::vector<double> w = dense_block_eigenvalues(H);   // ascending
        sol.values.assign(w.begin(), w.begin() + static_cast<long>(std::min(k, w.size())));
        return sol;
    }
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> es(materialize(H));
    if (es.info() != Eigen::Success)
        throw std::runtime_error("little_group: dense block eigensolve failed");
    for (std::size_t j = 0; j < k; ++j) {
        sol.values.push_back(es.eigenvalues()(static_cast<Eigen::Index>(j)));
        std::vector<Complex> v(nb);
        for (std::size_t i = 0; i < nb; ++i)
            v[i] = es.eigenvectors()(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(j));
        sol.vectors.push_back(std::move(v));
    }
    return sol;
}

namespace {

// A unit-variance complex Gaussian start of dimension n from `seed`, drawn on the host and
// staged on the backend; the host draw is freed before the caller's kernel runs.
template <class B>
ed::matvec::Backend::UniqueVec staged_seed(B& be, std::size_t n, std::uint64_t seed) {
    auto v = be.make_zero_vector(n);
    std::vector<Complex> host(n);
    std::mt19937_64 gen(seed);
    std::normal_distribution<double> nd(0.0, 1.0);
    for (auto& c : host) c = Complex(nd(gen), nd(gen));
    be.copy_from_host(host.data(), v.get(), n);
    return v;
}

// A backend vector on the host.
template <class B>
std::vector<Complex> to_host(B& be, const Complex* v, std::size_t n) {
    std::vector<Complex> h(n);
    be.copy_to_host(v, h.data(), n);
    return h;
}

// H bound to the backend once, counting its applies.
struct CountedH {
    ed::LinearOperator::MatvecFn H;
    std::uint64_t applies = 0;
    void operator()(const Complex* in, Complex* out, std::size_t n) {
        ++applies;
        H(in, out, n);
    }
};

// The k lowest LEVELS of a Lanczos tridiagonal (Ritz values w[0..m) ascending; bound(j) the Paige
// bound |beta_m z_{m,j}| of value j), walked contiguously from the bottom. Values within 1e-9 scale
// of a level's first copy are that level: ghost copies of a converged extreme, or the copy of a
// degenerate level the recurrence picks up from roundoff once its first Krylov space is exhausted
// (the 7-state ring at k = 1). A level is converged when ANY copy carries a bound <= 1e-7 scale and
// is reported at its first converged copy. The walk stops at the first unconverged level, never
// backfilled from above (Lanczos converges the TOP extreme first). True when k levels were found;
// `out` (if given) receives the converged prefix.
template <class Bound>
bool lowest_levels(std::size_t m, const double* w, Bound&& bound, double scale, std::size_t k,
                   std::vector<double>* out) {
    std::size_t found = 0;
    for (std::size_t j = 0; j < m && found < k;) {
        const double first = w[j];
        std::size_t i = j;
        bool converged = false;
        double value = first;
        for (; i < m && std::abs(w[i] - first) <= 1e-9 * scale; ++i)
            if (!converged && bound(i) <= 1e-7 * scale) { converged = true; value = w[i]; }
        if (!converged) return false;
        if (out) out->push_back(value);
        ++found;
        j = i;
    }
    return found >= k;
}

}  // namespace

// Several lowest levels of one block above the dense crossover: thick-restart
// Krylov-Schur with locking. It reorthogonalises fully inside each cycle, so there
// are no ghost copies to dedup; it locks Ritz pairs strictly from the bottom (a level
// whose residual has not converged stops the locked prefix, it is never replaced by a
// higher one); and after locking, a fresh start deflated against the locked set finds
// the further copies of a degenerate level.
//
// Budgets. The per-cycle basis (m length-nb vectors) is capped by
// LanePolicy<B>::ks_budget_bytes() (the host: the RAM this job may still allocate,
// cgroup-aware); a cap too small to hold k + 8 vectors is a clean refusal, never a
// silent fall-back to the ghost-prone scan. The total iteration budget is
// max(200k, 2000), spent as restart cycles.
template <class B>
static BlockSolution krylov_schur_lane(B& be, const ed::LinearOperator& H, std::size_t k, bool vectors,
                                       std::uint64_t max_iter) {
    const std::size_t nb = H.dim();
    const std::uint64_t cap = ed::krylov::krylov_vector_budget(
        LanePolicy<B>::ks_budget_bytes(), nb, /*safety=*/0.5, /*reserve_vecs=*/8);
    if (cap > 0 && cap < k + 8) {
        throw std::runtime_error(
            "little_group: " + std::to_string(k) + " levels of a block of dimension "
            + std::to_string(nb) + " need at least " + std::to_string(k + 8)
            + " resident Krylov vectors (" + std::to_string((k + 8) * nb * 16 >> 20)
            + " MiB); only " + std::to_string(cap) + " fit in the memory this job may "
            "still allocate. Ask for fewer levels (k = 1 uses a basis-free scan) or more "
            "memory.");
    }
    // Default max(200k, 2000): each cycle restarts from ONE Ritz vector, so many
    // levels need many cycles (k = 10 left a block unconverged at the scan's 400).
    const std::uint64_t budget = max_iter > 0 ? max_iter
        : lg_lowest_max_iter(k, std::max<std::uint64_t>(200u * static_cast<std::uint64_t>(k), 2000u));
    // A cycle never exceeds the whole iteration budget (a starved budget must yield
    // an unconverged result, not one full-length cycle), nor the memory cap.
    const std::size_t per_cycle = std::min<std::size_t>(
        ed::krylov::krylov_subspace_dim(k, 2 * k + 60, nb, cap),
        static_cast<std::size_t>(std::max<std::uint64_t>(budget, k + 1)));
    const std::size_t restarts = static_cast<std::size_t>(std::max<std::uint64_t>(
        1u, budget / std::max<std::size_t>(per_cycle, 1)));
    // The kernels floor a cycle at 2k + 20 vectors unless the SUBSPACE cap says
    // otherwise, so the cycle length is imposed through that cap.
    const std::uint64_t cycle_cap = (cap > 0) ? std::min<std::uint64_t>(cap, per_cycle)
                                              : static_cast<std::uint64_t>(per_cycle);
    // absolute residual ||H x - theta x||
    const double tol = 1e-9;

    CountedH Hc{H.bind<B>()};
    auto v0 = staged_seed(be, nb, 0x51ED0B70ULL);   // same stream as the k = 1 scan
    ed::krylov::KrylovSchurOptions o;
    o.num_eigs             = k;
    o.max_iter             = per_cycle;
    o.max_restarts         = restarts;
    o.tolerance            = tol;
    o.max_subspace_vectors = cycle_cap;
    o.compute_vectors      = vectors;
    auto r = ed::krylov::krylov_schur_kernel(be, Hc, nb, v0.get(), o);
    v0.reset();
    std::vector<double> ev = std::move(r.eigenvalues);
    const bool conv  = r.converged;
    const bool whole = r.exhausted;   // every eigenvalue of the block was found (fewer than k when nb < k)
    std::vector<std::vector<Complex>> vv;    // Ritz vectors (block coordinates)
    if (vectors)
        for (auto& v : r.eigenvectors) {
            vv.push_back(to_host(be, v.get(), nb));
            v.reset();
        }
    // Ascending, vectors kept aligned with their values; then the k lowest.
    std::vector<std::size_t> order(ev.size());
    std::iota(order.begin(), order.end(), std::size_t{0});
    std::sort(order.begin(), order.end(),
              [&ev](std::size_t a, std::size_t b) { return ev[a] < ev[b]; });
    const bool have_vecs = vectors && vv.size() == ev.size();
    BlockSolution sol;
    for (std::size_t i = 0; i < order.size() && sol.values.size() < k; ++i) {
        sol.values.push_back(ev[order[i]]);
        if (have_vecs) sol.vectors.push_back(std::move(vv[order[i]]));
    }
    sol.converged = (conv || whole) && (sol.values.size() >= k || whole) && (!vectors || have_vecs);
    sol.applies   = Hc.applies;
    return sol;
}

// One level above the dense crossover: the kernel Lanczos with no stored basis, a real
// iteration budget and a k-lowest Ritz residual gate. A small budget (e.g. 2k+40) or no
// residual guard lets partially-converged and ghost Ritz values through as eigenvalues on
// near-degenerate blocks.
//
// WITHIN-BLOCK genuine degeneracy: a single-vector Lanczos returns exactly ONE Ritz value
// per eigenvalue no matter its true multiplicity (a random start has one component in a
// degenerate eigenspace), so an accidental degeneracy inside THIS (k, irrep, parity, Sz)
// block is undercounted. The dense crossover covers every block up to 4x the iteration cap
// (1600 at k <= 10) -- verified at 4x4 J2=1.0 (which DOES carry a within-block degeneracy at
// block ~800): normal operation matches the dense spectrum. The residual gap is a block that
// both exceeds the crossover AND carries an accidental degeneracy; two remedies: raise the
// dense crossover so the block goes dense (exact, memory permitting), or ask for k >= 2,
// which routes the block through Krylov-Schur, whose fresh starts after locking find every
// copy. This scan is the k = 1 lane only -- the one production uses at N = 36, where a
// Krylov basis of 1e8-dimensional vectors does not fit.
template <class B>
static BlockSolution lowest_scan_lane(B& be, const ed::LinearOperator& H, std::uint64_t max_iter) {
    constexpr std::size_t k = 1;
    const std::uint64_t nb = H.dim();
    ed::krylov::LanczosKernelOptions kopts;
    kopts.max_iter        = static_cast<std::size_t>(std::min<std::uint64_t>(
        nb, max_iter > 0 ? max_iter : lg_lowest_max_iter(k)));
    // Ring reorth is a MEMORY term at frontier dims: 8 ring vectors x 16 B
    // x nb is ~48 GB per 3.8e8-dim block (81 G RSS measured against a 96 G
    // budget on a 4x3 kagome block). This scan is
    // eigenvalues-only and the k-DISTINCT Paige-bound gate below is
    // ghost-aware by design, so above the two-pass dim floor we drop to
    // the pure three-term recurrence: ghosts cost duplicate converged
    // copies (deduped), not wrong eigenvalues. Small blocks keep the ring
    // -- it sharpens the excited window at negligible cost there.
    if (static_cast<std::size_t>(nb) > kLgTwoPassMinDim) {
        kopts.reorth          = ed::krylov::ReorthPolicy::None;
    } else {
        kopts.reorth          = ed::krylov::ReorthPolicy::LocalDGKS3;
        kopts.local_ring_size = 8;
    }
    kopts.keep_basis      = false;
    // k-LOWEST converged Ritz early exit: at 1e8 dims the window fills with
    // ghost COPIES of converged extremes, and a ghost is exactly as
    // stationary as an eigenvalue (a stationarity test burns the full
    // budget and returns 8x E0). The tridiagonal residual bound
    // |beta_m * z_{m,j}| is free, rigorous (Paige), and ghost-aware in
    // combination with dedup.
    //
    // CONTIGUITY: stopping once ANY k distinct Ritz values carry converged
    // bounds is wrong. Lanczos converges the TOP extreme first, so on large
    // blocks such a gate collects k converged top-of-spectrum values within
    // ~40 iterations and stops before the bottom has converged at all
    // (4x2 kagome BFG, 338019-dim blocks: block min +11.58 against a true
    // sector minimum of -6.57). The gate demands that the k LOWEST distinct
    // Ritz values, walked contiguously from the bottom, EACH carry a
    // converged bound -- the first unconverged distinct value vetoes the
    // exit.
    {
        const std::size_t kk = k;
        kopts.convergence_check =
            [kk](const std::vector<double>& alpha,
                 const std::vector<double>& beta) -> bool {
                const int m = static_cast<int>(alpha.size());
                if (static_cast<std::size_t>(m) < kk + 2) return false;
                Eigen::MatrixXd T = Eigen::MatrixXd::Zero(m, m);
                for (int i = 0; i < m; ++i)
                    T(i, i) = alpha[static_cast<std::size_t>(i)];
                for (int i = 1; i < m; ++i) {
                    const double b = beta[static_cast<std::size_t>(i)];
                    T(i, i - 1) = b;
                    T(i - 1, i) = b;
                }
                Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es;
                es.compute(T, Eigen::ComputeEigenvectors);
                if (es.info() != Eigen::Success) return false;
                const double beta_m =
                    (beta.size() > static_cast<std::size_t>(m))
                        ? std::abs(beta[static_cast<std::size_t>(m)]) : 0.0;
                const auto& w = es.eigenvalues();
                const double scale = std::max(
                    {std::abs(w(0)), std::abs(w(m - 1)), 1e-300});
                const auto& Z = es.eigenvectors();
                return lowest_levels(static_cast<std::size_t>(m), w.data(),
                                     [&](std::size_t j) { return beta_m * std::abs(Z(m - 1, static_cast<Eigen::Index>(j))); },
                                     scale, kk, nullptr);
            };
        kopts.convergence_check_interval = 10;
    }
    CountedH Hc{H.bind<B>()};
    // Fixed start-vector seed. Single-vector Lanczos returns ONE copy of a
    // genuinely degenerate pair (see above for the lanes that count copies).
    auto v0 = staged_seed(be, static_cast<std::size_t>(nb), 0x51ED0B70ULL);
    auto kres = ed::krylov::lanczos_kernel(be, Hc, static_cast<std::size_t>(nb), v0.get(), kopts);
    v0.reset();
    BlockSolution sol;
    sol.applies = Hc.applies;
    const std::size_t m = kres.alpha.size();
    if (m == 0) return sol;
    std::vector<double> diag = kres.alpha;
    std::vector<double> off(m > 1 ? m - 1 : 1, 0.0);
    for (std::size_t i = 0; i + 1 < m; ++i) off[i] = kres.beta[i + 1];
    std::vector<double> z(m * m, 0.0);
    const lapack_int info = LAPACKE_dstevd(
        LAPACK_COL_MAJOR, 'V', static_cast<lapack_int>(m),
        diag.data(), off.data(), z.data(), static_cast<lapack_int>(m));
    if (info != 0)
        throw std::runtime_error("little_group: lowest-k tridiag eigensolve "
                                 "failed (dstevd info != 0)");
    // Ghost handling: with a local reorth ring at dim ~1e8 the Ritz
    // window fills with ghost COPIES of converged extremes faster than
    // genuine upper levels converge. On this path a single-vector recurrence
    // cannot represent a true within-block degeneracy anyway (exact
    // arithmetic yields ONE copy per eigenvalue), so equal-to-tolerance
    // duplicates ARE one level (see lowest_levels). Additionally
    // keep only Ritz values whose tridiagonal residual bound
    // |beta_m * z_{m,j}| marks them converged -- both tests are free (the
    // tridiag is m <= a few hundred).
    const double beta_m = (kres.beta.size() > m) ? std::abs(kres.beta[m]) : 0.0;
    const double scale  = std::max(
        {std::abs(diag.front()), std::abs(diag[m - 1]), 1e-300});
    // CONTIGUITY (pairs with the gate above): walk the Ritz values
    // ASCENDING, merge ghost copies into levels, and take the k lowest levels
    // -- STOPPING at the first unconverged one. Skipping past it would
    // backfill with converged UPPER-spectrum values (Lanczos converges the
    // top extreme first) and report them as the "lowest" with
    // converged=true. An unconverged low value truncates the list and flags
    // the block unconverged; it is never silently replaced by a higher value.
    // A budget-capped block that could not deliver k converged levels must be
    // DISTINGUISHABLE from a converged one downstream.
    sol.converged = lowest_levels(m, diag.data(),
                                  [&](std::size_t j) { return beta_m * std::abs(z[(m - 1) + j * m]); },
                                  scale, k, &sol.values);
    return sol;
}

template <class B>
BlockSolution solve_block_lowest(B& be, const ed::LinearOperator& H, std::size_t want, std::uint64_t max_iter) {
    const std::uint64_t nb = H.dim();
    if (nb == 0) return {};
    const std::size_t k = static_cast<std::size_t>(std::max<std::uint64_t>(
        1u, std::min<std::uint64_t>(static_cast<std::uint64_t>(want), nb)));
    if (nb <= 2) return solve_block_dense(H, k, false);
    if (k > 1) return krylov_schur_lane(be, H, k, false, max_iter);
    return lowest_scan_lane(be, H, max_iter);
}

template <class B>
BlockSolution solve_block_eigenpairs(B& be, const ed::LinearOperator& H, std::size_t want,
                                     std::uint64_t max_iter) {
    const std::size_t nb = H.dim();
    if (nb == 0) return {};
    const std::size_t k = std::min<std::size_t>(std::max<std::size_t>(want, 1), nb);
    if (nb <= 2) return solve_block_dense(H, k, true);
    if (k > 1) return krylov_schur_lane(be, H, k, true, max_iter);
    BlockSolution sol;
    GsVector g = solve_gs_vector(be, H, LanePolicy<B>::gs_kept_basis_max_dim, max_iter);
    sol.applies = g.applies;
    if (!g.certified) {            // the residual guard failed: certify nothing
        sol.converged = false;
        return sol;
    }
    sol.values.push_back(g.energy);
    sol.vectors.push_back(std::move(g.vector));
    return sol;
}

// The lowest Ritz value after 40 Lanczos steps from a fixed random start: an upper bound on
// the block's lowest level.
template <class B>
BlockEstimate estimate_lowest(B& be, const ed::LinearOperator& H) {
    const std::size_t n = H.dim();
    BlockEstimate est;
    ed::krylov::LanczosKernelOptions kopts;
    kopts.max_iter   = std::min<std::size_t>(40, n);
    kopts.reorth     = ed::krylov::ReorthPolicy::None;
    kopts.keep_basis = false;
    CountedH Hc{H.bind<B>()};
    auto v0 = staged_seed(be, n, 0xE57A7EULL);
    const auto k = ed::krylov::lanczos_kernel(be, Hc, n, v0.get(), kopts);
    v0.reset();
    est.applies = Hc.applies;
    std::vector<double> d = k.alpha, e;
    for (std::size_t i = 1; i < k.alpha.size(); ++i) e.push_back(k.beta[i]);
    if (d.empty()) return est;
    e.resize(std::max<std::size_t>(d.size(), 1));
    if (LAPACKE_dstev(LAPACK_COL_MAJOR, 'N', static_cast<lapack_int>(d.size()), d.data(), e.data(),
                      nullptr, 1) != 0)
        return est;                                          // never prune on a failed estimate
    est.theta = *std::min_element(d.begin(), d.end());
    return est;
}

// TWO-PASS no-reorth ground-state Ritz vector, on the backend's BLAS-1.
//
// Pass 1: pure three-term recurrence (no reorth, no stored basis), tridiag
// only, with the same ghost-aware k=1 Paige-bound gate the star scan uses
// (a ghost is a COPY of the converged extreme, so the FIRST converged
// distinct Ritz value IS E0). Pass 2: replay the recurrence with the
// STORED alpha/beta -- no inner products, so the trajectory is identical
// arithmetic to pass 1 -- accumulating u = sum_j z_j V_j on the fly.
// Memory: five n-vectors, independent of the iteration count. The caller's
// residual guard stays the arbiter; on a miss we restart the whole
// two-pass seeded by the current u (Lanczos restarted on an approximate
// eigenvector converges rapidly), up to kLgGsRestarts times. A numerical
// failure (zero seed, empty tridiagonal, failed tridiagonal solve, zero
// Ritz vector) returns nullopt.
template <class B>
static std::optional<std::pair<double, std::vector<Complex>>>
gs_two_pass(B& be, CountedH& H, std::size_t n, std::uint64_t max_iter_override) {
    using UV = ed::matvec::Backend::UniqueVec;
    const std::size_t max_iter = std::min<std::size_t>(
        n, max_iter_override > 0 ? static_cast<std::size_t>(max_iter_override) : kLgGsTwoPassMaxIter);
    UV seed = staged_seed(be, n, 0x51ED900DULL);
    {
        const double s0 = be.nrm2(seed.get(), n);
        if (!(s0 > 0.0)) return std::nullopt;
        be.scale(Complex(1.0 / s0, 0.0), seed.get(), n);
    }
    UV vp = be.make_zero_vector(n), vc = be.make_zero_vector(n), w = be.make_zero_vector(n);
    UV u = be.make_zero_vector(n);
    double E0 = 0.0;
    for (int attempt = 0; attempt <= kLgGsRestarts; ++attempt) {
        // ---------------- pass 1: tridiag only -------------------------
        std::vector<double> alpha, beta{0.0};
        be.fill_zero(vp.get(), n);
        be.copy(seed.get(), vc.get(), n);
        bool done = false;
        std::size_t m = 0;
        while (!done && m < max_iter) {
            H(vc.get(), w.get(), n);
            const double a = std::real(be.dot(vc.get(), w.get(), n));
            alpha.push_back(a);
            const double bprev = beta.back();
            be.axpy(Complex(-a, 0.0), vc.get(), w.get(), n);
            be.axpy(Complex(-bprev, 0.0), vp.get(), w.get(), n);
            const double b = be.nrm2(w.get(), n);
            beta.push_back(b);
            ++m;
            if (!(b > 1e-300)) { done = true; break; }   // invariant subspace
            std::swap(vp, vc);
            std::swap(vc, w);
            be.scale(Complex(1.0 / b, 0.0), vc.get(), n);
            if (m >= 3 && m % 10 == 0) {
                // Paige bound on the smallest Ritz value only.
                std::vector<double> d(alpha), e(m > 1 ? m - 1 : 1, 0.0);
                for (std::size_t i = 0; i + 1 < m; ++i) e[i] = beta[i + 1];
                std::vector<double> zz(m * m, 0.0);
                if (LAPACKE_dstevd(LAPACK_COL_MAJOR, 'V', static_cast<lapack_int>(m), d.data(),
                                   e.data(), zz.data(), static_cast<lapack_int>(m)) == 0) {
                    const double scale = std::max({std::abs(d[0]), std::abs(d[m - 1]), 1e-300});
                    if (beta[m] * std::abs(zz[m - 1]) < 1e-9 * scale) done = true;
                }
            }
        }
        if (m == 0) return std::nullopt;
        std::vector<double> ritz_z;   // column 0
        {
            std::vector<double> d(alpha), e(m > 1 ? m - 1 : 1, 0.0);
            for (std::size_t i = 0; i + 1 < m; ++i) e[i] = beta[i + 1];
            std::vector<double> zz(m * m, 0.0);
            if (LAPACKE_dstevd(LAPACK_COL_MAJOR, 'V', static_cast<lapack_int>(m), d.data(), e.data(),
                               zz.data(), static_cast<lapack_int>(m)) != 0)
                return std::nullopt;
            E0 = d[0];
            ritz_z.assign(zz.begin(), zz.begin() + static_cast<long>(m));
        }
        // ---------------- pass 2: replay + accumulate ------------------
        be.fill_zero(u.get(), n);
        be.fill_zero(vp.get(), n);
        be.copy(seed.get(), vc.get(), n);
        for (std::size_t j = 0; j < m; ++j) {
            be.axpy(Complex(ritz_z[j], 0.0), vc.get(), u.get(), n);
            if (j + 1 >= m) break;
            H(vc.get(), w.get(), n);
            be.axpy(Complex(-alpha[j], 0.0), vc.get(), w.get(), n);
            be.axpy(Complex(-beta[j], 0.0), vp.get(), w.get(), n);
            std::swap(vp, vc);
            std::swap(vc, w);
            be.scale(Complex(1.0 / beta[j + 1], 0.0), vc.get(), n);
        }
        const double un = be.nrm2(u.get(), n);
        if (!(un > 0.0)) return std::nullopt;
        be.scale(Complex(1.0 / un, 0.0), u.get(), n);
        // Residual check; restart seeded by u on a miss.
        H(u.get(), w.get(), n);
        const double ray = std::real(be.dot(u.get(), w.get(), n));
        be.axpy(Complex(-ray, 0.0), u.get(), w.get(), n);
        const double resid = be.nrm2(w.get(), n);
        E0 = ray;
        if (resid <= kLgGsResidTol || attempt == kLgGsRestarts) break;
        be.copy(u.get(), seed.get(), n);   // restarted refinement
    }
    // Release the work vectors before the host copy: the lane exists to stay at a few n-vectors.
    seed.reset(); vp.reset(); vc.reset(); w.reset();
    return std::make_pair(E0, to_host(be, u.get(), n));
}

// The certified GS eigenpair of a block above the caller's dense crossover: FullCGS2
// Lanczos + kept-basis Ritz vector up to kept_basis_max_dim, the two-pass no-reorth lane
// above it. Residual-guarded: the vector is certified only when ||H u - E u|| <= kLgGsResidTol.
template <class B>
GsVector solve_gs_vector(B& be, const ed::LinearOperator& H, std::size_t kept_basis_max_dim,
                         std::uint64_t max_iter) {
    const std::size_t n = H.dim();
    if (n == 0) throw std::invalid_argument("little_group: empty GS sector");
    GsVector g;
    CountedH Hc{H.bind<B>()};
    double E0 = 0.0;
    std::vector<Complex> u;                  // allocated per branch: the two-pass returns its own
    if (n <= 2) {   // the caller sends blocks below its dense crossover to a dense solve
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> es(materialize(H));
        E0 = es.eigenvalues()(0);
        u.assign(n, Complex(0, 0));
        for (std::size_t i = 0; i < n; ++i)
            u[i] = es.eigenvectors()(static_cast<Eigen::Index>(i), 0);
    } else if (n > kept_basis_max_dim) {
        auto pr = gs_two_pass(be, Hc, n, max_iter);
        g.applies = Hc.applies;
        if (!pr) return g;
        E0 = pr->first;
        u  = std::move(pr->second);
    } else {
        ed::krylov::LanczosKernelOptions kopts;
        kopts.max_iter   = std::min<std::size_t>(
            n, max_iter > 0 ? static_cast<std::size_t>(max_iter) : kLgGsSmallMaxIter);
        kopts.reorth     = ed::krylov::ReorthPolicy::FullCGS2;
        kopts.keep_basis = true;
        auto v0 = staged_seed(be, n, 0x51ED900DULL);
        auto kres = ed::krylov::lanczos_kernel(be, Hc, n, v0.get(), kopts);
        v0.reset();
        const std::size_t m = kres.alpha.size();
        g.applies = Hc.applies;
        if (m == 0) return g;
        std::vector<double> diag = kres.alpha;
        std::vector<double> off(m > 1 ? m - 1 : 1, 0.0);
        for (std::size_t i = 0; i + 1 < m; ++i) off[i] = kres.beta[i + 1];
        std::vector<double> z(m * m, 0.0);
        const lapack_int info = LAPACKE_dstevd(
            LAPACK_COL_MAJOR, 'V', static_cast<lapack_int>(m),
            diag.data(), off.data(), z.data(), static_cast<lapack_int>(m));
        if (info != 0) return g;
        E0 = diag[0];
        u.assign(n, Complex(0, 0));
        std::vector<Complex> vj_host;
        for (std::size_t j = 0; j < m; ++j) {
            const double yj = z[j];              // column 0, row j
            if (std::abs(yj) < 1e-300) continue;
            const Complex* vj = kres.basis[j].get();
            if constexpr (!std::is_same_v<B, ed::matvec::CpuBackend>) {
                vj_host.resize(n);
                be.copy_to_host(vj, vj_host.data(), n);
                vj = vj_host.data();
            }
            for (std::size_t i = 0; i < n; ++i) u[i] += yj * vj[i];
        }
    }
    // Residual guard: the DSSF consumes this vector, so a stale pair is silently-wrong
    // physics -- verify before returning. H u is formed on the backend.
    std::vector<Complex> hu(n);
    if constexpr (std::is_same_v<B, ed::matvec::CpuBackend>) {
        Hc(u.data(), hu.data(), n);
    } else {
        auto du = be.make_zero_vector(n), dh = be.make_zero_vector(n);
        be.copy_from_host(u.data(), du.get(), n);
        Hc(du.get(), dh.get(), n);
        be.copy_to_host(dh.get(), hu.data(), n);
    }
    g.applies = Hc.applies;
    double num = 0.0, den = 1e-300;
    for (std::size_t i = 0; i < n; ++i) {
        num += std::norm(hu[i] - E0 * u[i]);
        den += std::norm(u[i]);
    }
    g.residual = std::sqrt(num / den);
    if (!(g.residual <= kLgGsResidTol)) return g;
    const double inv = 1.0 / std::sqrt(den);
    for (auto& c : u) c *= inv;
    g.energy    = E0;
    g.vector    = std::move(u);
    g.certified = true;
    return g;
}

#define ED_LG_LANES(B)                                                                            \
    template BlockSolution solve_block_lowest<B>(B&, const ed::LinearOperator&, std::size_t,      \
                                                 std::uint64_t);                                  \
    template BlockSolution solve_block_eigenpairs<B>(B&, const ed::LinearOperator&, std::size_t,  \
                                                     std::uint64_t);                              \
    template GsVector solve_gs_vector<B>(B&, const ed::LinearOperator&, std::size_t,              \
                                         std::uint64_t);                                          \
    template BlockEstimate estimate_lowest<B>(B&, const ed::LinearOperator&);
ED_LG_LANES(ed::matvec::CpuBackend)
#ifdef WITH_CUDA
ED_LG_LANES(ed::matvec::CudaBackend)
#endif
#undef ED_LG_LANES

}  // namespace lg_detail

}  // namespace ed::solvers
