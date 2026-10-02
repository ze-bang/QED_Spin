// =============================================================================
// src/engine/block_solve.cpp -- the per-block eigensolve driver: the dense
// solve, the crossover, and the Backend-templated Krylov lanes (lowest-level scan, Krylov-Schur,
// certified GS vector, pruning estimate), instantiated for CpuBackend and CudaBackend.
// Part of the little-group engine; see internal.h for the file map.
// =============================================================================

#include "internal.h"

#include <ed/core/footprint.h>

#include <algorithm>
#include <cmath>
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
// A block beyond what LAPACK can address (core/lapack.h) is refused before it is materialised.
static void require_lapack_dense(std::uint64_t n) {
    if (n > ed::core::lapack_max_dense_n())
        throw ed::Unsupported("a dense eigensolve of a block of " + std::to_string(n) + " states is beyond the "
                              "linked LAPACK (32-bit indices address at most "
                              + std::to_string(ed::core::lapack_max_dense_n()) + " x "
                              + std::to_string(ed::core::lapack_max_dense_n()) + "); split the block "
                              "with more symmetry, or use eigs / thermal(method='ftlm')");
}

[[nodiscard]] std::vector<double>
dense_eigenvalues_inplace(Eigen::MatrixXcd& Hb) {
    require_lapack_dense(static_cast<std::uint64_t>(Hb.rows()));
    const lapack_int n = static_cast<lapack_int>(Hb.rows());
    std::vector<double> w(static_cast<std::size_t>(n), 0.0);
    if (n == 0) return w;

    double max_imag = 0.0;             // is this block real (up to roundoff)?
    for (Eigen::Index j = 0; j < Hb.cols(); ++j)
        for (Eigen::Index i = j; i < Hb.rows(); ++i) {
            const double a = std::abs(Hb(i, j).imag());
            if (a > max_imag) max_imag = a;
        }

    double max_abs = 0.0;              // relative to the block's own scale (numerics.h)
    for (Eigen::Index j = 0; j < Hb.cols(); ++j)
        for (Eigen::Index i = j; i < Hb.rows(); ++i) max_abs = std::max(max_abs, std::abs(Hb(i, j)));
    lapack_int info;
    if (max_imag <= ed::numerics::kRealBlockRel * max_abs) {
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
    require_lapack_dense(mv.dim());
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
// the device lane of the sectors eigensolve (eigs.cpp) makes exactly the same dense-vs-Lanczos
// decision as the CPU ``solve_block_lowest``.
[[nodiscard]] std::uint64_t lowest_dense_floor(std::size_t k, int dense_max_dim, bool vectors) {
    // An explicit crossover (EigsOptions::dense_max_dim >= 0) is the caller's: 0 sends
    // every block above dimension 2 to Krylov, a large value solves exactly.
    // It is clamped to what LAPACK can address: a larger block takes the Krylov lanes.
    if (dense_max_dim >= 0)
        return std::min<std::uint64_t>(static_cast<std::uint64_t>(dense_max_dim), ed::core::lapack_max_dense_n());
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
    //
    // The floor grows with k (160 k), while Krylov-Schur needs only ~2k + 60 vectors: the automatic
    // floor never exceeds kAutoDenseCeiling, nor the largest block whose dense working set
    // (core/footprint.h) fits in half the RAM the job may still allocate.
    std::uint64_t floor_ = std::min<std::uint64_t>(4u * max_iter_cap, kAutoDenseCeiling);   // < lapack_max_dense_n
    const std::uint64_t avail = ed::core::mem_guard_off() ? 0 : ed::core::available_ram_bytes();
    if (avail > 0) {
        ed::core::Shape one;
        one.dim = 1;
        const double per = static_cast<double>(ed::core::footprint(
            vectors ? ed::core::Path::DenseVectors : ed::core::Path::DenseValues, one).host);
        floor_ = std::min<std::uint64_t>(floor_, static_cast<std::uint64_t>(std::sqrt(0.5 * static_cast<double>(avail) / per)));
    }
    return floor_;
}


// The `want` lowest levels of a block by a dense solve on the host: LAPACK
// divide-and-conquer for values, Eigen's eigensolver with vectors.
[[nodiscard]] BlockSolution solve_block_dense(const ed::LinearOperator& H, std::size_t want, bool vectors) {
    BlockSolution sol;
    const std::size_t nb = H.dim();
    if (nb == 0) return sol;
    const std::size_t k = std::min(std::max<std::size_t>(want, 1), nb);
    ed::core::Shape shape;
    shape.dim = nb;
    ed::core::guard_working_set(ed::core::footprint(vectors ? ed::core::Path::DenseVectors
                                                            : ed::core::Path::DenseValues, shape).host,
                                "dense block eigensolve");
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

// A unit-variance Gaussian start of dimension n from `seed`, drawn on the host and staged on
// the backend; the host draw is freed before the caller's kernel runs.
template <class B>
auto staged_seed(B& be, std::size_t n, std::uint64_t seed) {
    using Scalar = typename B::scalar_type;
    auto v = be.make_zero_vector(n);
    std::vector<Scalar> host(n);
    std::mt19937_64 gen(seed);
    std::normal_distribution<double> nd(0.0, 1.0);
    for (auto& c : host) c = ed::krylov::gaussian_entry<Scalar>(nd, gen);
    be.copy_from_host(host.data(), v.get(), n);
    return v;
}

// A backend vector on the host.
template <class B>
auto to_host(B& be, const typename B::scalar_type* v, std::size_t n) {
    std::vector<typename B::scalar_type> h(n);
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
        // scale-free: relative to the Ritz values' scale
        for (; i < m && std::abs(w[i] - first) <= 1e-9 * scale; ++i)
            if (!converged && bound(i) <= 1e-7 * scale) { converged = true; value = w[i]; }
        if (!converged) return false;
        if (out) out->push_back(value);
        ++found;
        j = i;
    }
    return found >= k;
}

// The leading invariant block of a Lanczos run (alpha[0..m), beta[i] the off-diagonal between steps
// i-1 and i): its size is the step before the first off-diagonal at roundoff level, 1e-12 of the
// largest |alpha| or |beta| seen. The kernel keeps iterating past an invariant subspace (FTLM wants
// the full tridiagonal), and from there the recurrence is built from roundoff: it re-finds the other
// copy of each degenerate level, coupled to the converged copy at that same roundoff, and the two
// mix at O(1), so no copy certifies (the 7-state magnon block of the odd Ising ring, the tiny tri9
// blocks). The leading block's Ritz values are exact eigenvalues. Without such an off-diagonal: m.
inline std::size_t leading_block(const std::vector<double>& alpha, const std::vector<double>& beta,
                                 std::size_t m) {
    double scale = 0.0;
    for (std::size_t i = 0; i < m; ++i) {
        scale = std::max(scale, std::abs(alpha[i]));
        if (i + 1 < beta.size()) scale = std::max(scale, std::abs(beta[i + 1]));
    }
    for (std::size_t i = 1; i < m; ++i)
        // scale-free: relative to the Ritz values' scale
        if (std::abs(beta[i]) <= 1e-12 * scale) return i;
    return m;
}

// The k = 1 gate and readout of a Lanczos run of m steps: lowest_levels on the lowest Ritz values,
// their Paige bounds and the Ritz values' scale, in O(m) through tridiag_ends. Four values cover
// the lowest level unless it fills them (a cluster of ghost copies); then the full solve decides.
bool lowest_level(const std::vector<double>& alpha, const std::vector<double>& beta, std::size_t m,
                  std::vector<double>* out) {
    const double beta_m = beta.size() > m ? std::abs(beta[m]) : 0.0;
    const ed::krylov::TridiagEnds t = ed::krylov::tridiag_ends(alpha, beta, m, 4);
    const std::size_t c = t.values.size();
    const double scale = std::max({std::abs(t.values.front()), std::abs(t.top), 1e-300});
    // scale-free: lowest_levels' level width, relative to the Ritz values' scale
    if (c < m && std::abs(t.values[c - 1] - t.values.front()) <= 1e-9 * scale) {
        const ed::krylov::TridiagEig f = ed::krylov::tridiag_eig(alpha, beta, m, /*vectors=*/true);
        return lowest_levels(m, f.values.data(), [&](std::size_t j) { return beta_m * std::abs(f.z(m - 1, j)); },
                             scale, 1, out);
    }
    return lowest_levels(c, t.values.data(), [&](std::size_t j) { return beta_m * std::abs(t.z(m - 1, j)); },
                         scale, 1, out);
}

}  // namespace

// The Krylov-Schur working set at cycle length m on lane B (core/footprint.h): the part that
// lives where the lane's vectors live.
template <class B>
static std::uint64_t ks_lane_bytes(std::uint64_t nb, std::size_t k, std::size_t m) {
    ed::core::Shape s;
    s.dim    = nb;
    s.k      = k;
    s.krylov = m;
    s.device = !ed::matvec::is_cpu_backend_v<B>;
    const ed::core::Footprint f = ed::core::footprint(ed::core::Path::KrylovSchur, s);
    return s.device ? f.device : f.host;
}

// The longest cycle whose working set fits in 90% of the memory lane B may still allocate
// (LanePolicy<B>::ks_budget_bytes(): the host's RAM, cgroup-aware, or the device's free
// memory); 0: no cap (ED_MEM_GUARD_OFF, or the memory cannot be queried).
template <class B>
static std::uint64_t ks_cycle_cap(std::uint64_t nb, std::size_t k) {
    const std::uint64_t avail = LanePolicy<B>::ks_budget_bytes();
    if (avail == 0 || nb == 0) return 0;
    const double fixed = static_cast<double>(ks_lane_bytes<B>(nb, k, 0));
    const double per   = static_cast<double>(ks_lane_bytes<B>(nb, k, 1)) - fixed;
    const double room  = 0.9 * static_cast<double>(avail) - fixed;
    return room < per ? 1 : static_cast<std::uint64_t>(room / per);
}

// Several lowest levels of one block above the dense crossover: thick-restart
// Krylov-Schur with locking. It reorthogonalises fully inside each cycle, so there
// are no ghost copies to dedup; it locks Ritz pairs strictly from the bottom (a level
// whose residual has not converged stops the locked prefix, it is never replaced by a
// higher one); and after locking, a fresh start deflated against the locked set finds
// the further copies of a degenerate level.
//
// Budgets. The cycle (m length-nb vectors) is capped
// by ks_cycle_cap; a cap below k + 8 vectors is a clean refusal, never a silent fall-back to the
// ghost-prone scan. The total iteration budget is max(200k, 2000), spent as restart cycles.
template <class B>
static BlockSolution krylov_schur_lane(B& be, const ed::LinearOperator& H, std::size_t k, bool vectors,
                                       std::uint64_t max_iter) {
    const std::size_t nb = H.dim();
    // Bound first: the device lane builds its mirror of H here, which the cap then sees.
    CountedH Hc{H.bind<B>()};
    const std::uint64_t cap = ks_cycle_cap<B>(nb, k);
    if (cap > 0 && cap < k + 8) {
        throw ed::ResourceLimit(
            "little_group: " + std::to_string(k) + " levels of a block of dimension "
            + std::to_string(nb) + " need a Krylov cycle of at least " + std::to_string(k + 8)
            + " vectors (" + std::to_string(ks_lane_bytes<B>(nb, k, k + 8) >> 20)
            + " MiB in all); only a cycle of " + std::to_string(cap) + " fits in the memory "
            + (ed::matvec::is_cpu_backend_v<B> ? "this job may still allocate" : "free on the device")
            + ". Ask for fewer levels (k = 1 uses a basis-free scan) or more memory.");
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
    // A thick-restart cycle adds m - p matvecs (m = min(per_cycle, 2p + 20), p = k + max(k/2, 8)
    // kept): the iteration budget buys that many cycles.
    const std::size_t p_keep = k + std::max<std::size_t>(k / 2, 8);
    const std::size_t fresh  = std::max<std::size_t>(std::min(per_cycle, 2 * p_keep + 20), p_keep + 1) - p_keep;
    const std::size_t restarts = static_cast<std::size_t>(std::max<std::uint64_t>(1u, budget / fresh));
    // The kernels floor a cycle at 2k + 20 vectors unless the SUBSPACE cap says
    // otherwise, so the cycle length is imposed through that cap.
    const std::uint64_t cycle_cap = (cap > 0) ? std::min<std::uint64_t>(cap, per_cycle)
                                              : static_cast<std::uint64_t>(per_cycle);
    // residual ||H x - theta x|| relative to the block's norm bound (numerics.h)
    const double scale = ed::numerics::scale_or_one(H.norm_bound());
    const double tol = ed::numerics::kLockRel * scale;

    auto v0 = staged_seed(be, nb, 0x51ED0B70ULL);   // same stream as the k = 1 scan
    ed::krylov::KrylovSchurOptions o;
    o.num_eigs             = k;
    o.max_iter             = per_cycle;
    o.max_restarts         = restarts;
    o.tolerance            = tol;
    o.breakdown_tol        = ed::numerics::kBreakdownRel * scale;
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
    // The pure three-term recurrence: this scan is eigenvalues-only and its gate is ghost-aware
    // (below), so a ghost costs a duplicate converged copy, merged into its level, not a wrong
    // value; and it holds no basis (at frontier dims one vector is ~6 GB). A local reorthogonalisation
    // ring bought nothing for the lowest level and cost 8 vectors (~48 GB on a 3.8e8-dim block).
    // Only a run that may span the whole block reorthogonalises fully: there the recurrence would
    // go on from roundoff once the space is exhausted. At this size full reorthogonalisation is
    // exact and cheap; blocks this small are dense at the default crossover.
    kopts.reorth     = static_cast<std::size_t>(nb) <= kopts.max_iter ? ed::krylov::ReorthPolicy::FullCGS2
                                                                       : ed::krylov::ReorthPolicy::None;
    kopts.keep_basis = kopts.reorth == ed::krylov::ReorthPolicy::FullCGS2;   // CGS2 projects on it
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
    // exit. It is O(m) (lowest_level), so it runs after every step.
    kopts.convergence_check = [](const std::vector<double>& alpha, const std::vector<double>& beta) -> bool {
        const std::size_t m_all = alpha.size();
        const std::size_t m = leading_block(alpha, beta, m_all);
        // An exhausted Krylov space (m < m_all) is exact; a live run needs a few steps.
        if (m == m_all && m_all < k + 2) return false;
        return lowest_level(alpha, beta, m, nullptr);
    };
    kopts.convergence_check_interval = 1;
    CountedH Hc{H.bind<B>()};
    // Fixed start-vector seed. Single-vector Lanczos returns ONE copy of a
    // genuinely degenerate pair (see above for the lanes that count copies).
    auto v0 = staged_seed(be, static_cast<std::size_t>(nb), 0x51ED0B70ULL);
    auto kres = ed::krylov::lanczos_kernel(be, Hc, static_cast<std::size_t>(nb), v0.get(), kopts);
    v0.reset();
    BlockSolution sol;
    sol.applies = Hc.applies;
    if (kres.alpha.empty()) return sol;
    // Ghost handling: the Ritz window fills with ghost COPIES of converged
    // extremes faster than genuine upper levels converge. On this path a
    // single-vector recurrence cannot represent a true within-block
    // degeneracy anyway (exact arithmetic yields ONE copy per eigenvalue),
    // so equal-to-tolerance duplicates ARE one level (see lowest_levels).
    // Additionally keep only Ritz values whose tridiagonal residual bound
    // |beta_m * z_{m,j}| marks them converged.
    //
    // CONTIGUITY (pairs with the gate above): walk the Ritz values
    // ASCENDING, merge ghost copies into levels, and take the k lowest levels
    // -- STOPPING at the first unconverged one. Skipping past it would
    // backfill with converged UPPER-spectrum values (Lanczos converges the
    // top extreme first) and report them as the "lowest" with
    // converged=true. An unconverged low value truncates the list and flags
    // the block unconverged; it is never silently replaced by a higher value.
    // A budget-capped block that could not deliver k converged levels must be
    // DISTINGUISHABLE from a converged one downstream. The readout is the
    // gate's own computation, so the two never disagree.
    const std::size_t m = leading_block(kres.alpha, kres.beta, kres.alpha.size());
    sol.converged = lowest_level(kres.alpha, kres.beta, m, &sol.values);
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
    if (k.alpha.empty()) return est;
    const std::size_t m = k.alpha.size();
    const ed::krylov::TridiagEig t = ed::krylov::tridiag_eig(k.alpha, k.beta, m, /*vectors=*/true);
    est.theta    = t.values.front();
    est.residual = k.beta.size() > m ? std::abs(k.beta[m]) * std::abs(t.z(m - 1, 0)) : 0.0;
    return est;
}

// How many Krylov vectors the GS lane may keep on lane B: none above kept_basis_max_dim; else as
// many as fit beside the lane's working vectors (core/footprint.h, GsKeptBasis) in half the memory
// lane B may still allocate, and never more than one attempt runs. Without a memory reading
// (ED_MEM_GUARD_OFF) the dimension rule alone decides.
template <class B>
static std::size_t gs_keep_cap(std::size_t n, std::size_t kept_basis_max_dim, std::size_t steps) {
    if (n > kept_basis_max_dim) return 0;
    const std::uint64_t avail = LanePolicy<B>::ks_budget_bytes();
    if (avail == 0) return steps;
    ed::core::Shape s;
    s.dim    = n;
    s.device = !ed::matvec::is_cpu_backend_v<B>;
    const auto lane = [&s](std::size_t kept) {
        s.krylov = kept;
        const ed::core::Footprint f = ed::core::footprint(ed::core::Path::GsKeptBasis, s);
        return static_cast<double>(s.device ? f.device : f.host);
    };
    const double fixed = lane(0), per = lane(1) - fixed;
    const double room  = 0.5 * static_cast<double>(avail) - fixed;
    if (!(per > 0.0) || room < per) return 0;
    return static_cast<std::size_t>(std::min(static_cast<double>(steps), std::floor(room / per)));
}

// The certified lowest eigenpair of a block above the caller's dense crossover, on the backend's
// BLAS-1: a three-term recurrence without reorthogonalisation, stopped by the Paige bound
// |beta_m z_{m,0}| of the lowest Ritz value (a ghost is a COPY of the converged extreme, so the
// first converged Ritz value IS E0). The Ritz vector u = sum_j z_j V_j comes from the basis kept on
// the way while it fits (keep_cap vectors, one parallel axpy_many), else from a replay of the
// recurrence with the stored alpha/beta -- no inner products, so the same arithmetic as the first
// pass. H u closes the attempt (its Rayleigh quotient and residual are the result); on a miss of
// resid_tol the next attempt starts from u (a Lanczos run restarted on an approximate eigenvector
// converges fast), up to kLgGsRestarts times. max_steps > 0 caps the recurrence steps of all
// attempts together (a test seam). A numerical failure (zero seed, zero Ritz vector) returns
// nullopt.
struct GsAttempt {
    double               energy   = 0.0;
    double               residual = std::numeric_limits<double>::infinity();
    std::vector<Complex> vector;
};

template <class B>
static std::optional<GsAttempt>
gs_lanczos(B& be, CountedH& H, std::size_t n, std::size_t kept_basis_max_dim, std::uint64_t max_steps,
           double resid_tol) {
    using UV = typename B::UniqueVec;
    const std::size_t per_attempt = std::min<std::size_t>(n, kLgGsMaxIter);
    std::size_t left = max_steps > 0 ? static_cast<std::size_t>(max_steps)
                                     : per_attempt * static_cast<std::size_t>(kLgGsRestarts + 1);
    const std::size_t keep_cap = gs_keep_cap<B>(n, kept_basis_max_dim, per_attempt);
    UV seed = staged_seed(be, n, 0x51ED900DULL);
    {
        const double s0 = be.nrm2(seed.get(), n);
        if (!(s0 > 0.0)) return std::nullopt;
        be.scale(Complex(1.0 / s0, 0.0), seed.get(), n);
    }
    UV vp = be.make_zero_vector(n), vc = be.make_zero_vector(n), w = be.make_zero_vector(n);
    UV u = be.make_zero_vector(n);
    std::vector<UV> kept;                     // V_1, V_2, ... (V_0 is the seed)
    GsAttempt out;
    for (int attempt = 0; attempt <= kLgGsRestarts && left > 0; ++attempt) {
        // ---------------- the recurrence, Paige-gated -------------------
        std::vector<double> alpha, beta{0.0};
        be.fill_zero(vp.get(), n);
        be.copy(seed.get(), vc.get(), n);
        bool keeping = keep_cap > 0;
        kept.clear();
        const std::size_t cap = std::min(per_attempt, left);
        std::size_t m = 0;
        double scale = 0.0;                    // largest |alpha| or |beta| so far
        while (m < cap) {
            H(vc.get(), w.get(), n);
            const double a = std::real(be.dot(vc.get(), w.get(), n));
            alpha.push_back(a);
            scale = std::max(scale, std::abs(a));
            const double bprev = beta.back();
            be.axpy(Complex(-a, 0.0), vc.get(), w.get(), n);
            be.axpy(Complex(-bprev, 0.0), vp.get(), w.get(), n);
            const double b = be.nrm2(w.get(), n);
            beta.push_back(b);
            ++m;
            // An invariant subspace (see leading_block): its tridiagonal is exact; going on would
            // build vectors from roundoff.
            // scale-free: relative to the Ritz values' scale
            if (!(b > 1e-12 * scale)) break;
            scale = std::max(scale, b);
            if (m >= 3) {
                // Paige bound on the smallest Ritz value only, O(m) (tridiag_ends): every step.
                const ed::krylov::TridiagEnds t = ed::krylov::tridiag_ends(alpha, beta, m, 1);
                const double tscale = std::max({std::abs(t.values[0]), std::abs(t.top), 1e-300});
                // scale-free: relative to the Ritz values' scale
                if (beta[m] * std::abs(t.z(m - 1, 0)) < 1e-9 * tscale) break;
            }
            if (m == cap) break;
            std::swap(vp, vc);
            std::swap(vc, w);
            be.scale(Complex(1.0 / b, 0.0), vc.get(), n);
            if (keeping && kept.size() >= keep_cap) { kept.clear(); keeping = false; }   // outgrown
            if (keeping) {
                kept.push_back(be.make_zero_vector(n));
                be.copy(vc.get(), kept.back().get(), n);
            }
        }
        left -= m;
        // The lowest Ritz vector of the tridiagonal.
        const std::vector<double> z = ed::krylov::tridiag_ends(alpha, beta, m, 1).vectors;
        // ---------------- u = sum_j z_j V_j ----------------------------
        be.fill_zero(u.get(), n);
        if (keeping) {
            std::vector<const Complex*> V{seed.get()};
            for (const UV& v : kept) V.push_back(v.get());
            const std::vector<Complex> c(z.begin(), z.end());
            be.axpy_many(c.data(), V.data(), m, u.get(), n);
            kept.clear();
        } else {
            be.fill_zero(vp.get(), n);
            be.copy(seed.get(), vc.get(), n);
            for (std::size_t j = 0; j < m; ++j) {
                be.axpy(Complex(z[j], 0.0), vc.get(), u.get(), n);
                if (j + 1 >= m) break;
                H(vc.get(), w.get(), n);
                be.axpy(Complex(-alpha[j], 0.0), vc.get(), w.get(), n);
                be.axpy(Complex(-beta[j], 0.0), vp.get(), w.get(), n);
                std::swap(vp, vc);
                std::swap(vc, w);
                be.scale(Complex(1.0 / beta[j + 1], 0.0), vc.get(), n);
            }
        }
        const double un = be.nrm2(u.get(), n);
        if (!(un > 0.0)) return std::nullopt;
        be.scale(Complex(1.0 / un, 0.0), u.get(), n);
        // ---------------- H u: the result, or the next start -------------
        H(u.get(), w.get(), n);
        out.energy = std::real(be.dot(u.get(), w.get(), n));
        be.axpy(Complex(-out.energy, 0.0), u.get(), w.get(), n);
        out.residual = be.nrm2(w.get(), n);
        if (out.residual <= resid_tol) break;
        be.copy(u.get(), seed.get(), n);   // restarted refinement
    }
    // Release the work vectors before the host copy: the replay exists to stay at a few n-vectors.
    seed.reset(); vp.reset(); vc.reset(); w.reset();
    out.vector = to_host(be, u.get(), n);
    return out;
}

// The certified GS eigenpair of a block above the caller's dense crossover (gs_lanczos; dense at
// n <= 2). Certified only when ||H u - E u|| <= gs_resid_tol(H).
template <class B>
GsVector solve_gs_vector(B& be, const ed::LinearOperator& H, std::size_t kept_basis_max_dim,
                         std::uint64_t max_iter) {
    const std::size_t n = H.dim();
    if (n == 0) throw std::invalid_argument("little_group: empty GS sector");
    GsVector g;
    const double tol = gs_resid_tol(H);
    if (n <= 2) {   // the caller sends blocks below its dense crossover to a dense solve
        const Eigen::MatrixXcd M = materialize(H);
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> es(M);
        const Eigen::VectorXcd v = es.eigenvectors().col(0);
        g.energy    = es.eigenvalues()(0);
        g.residual  = (M * v - g.energy * v).norm();
        g.vector.assign(v.data(), v.data() + n);
        g.certified = g.residual <= tol;
        return g;
    }
    CountedH Hc{H.bind<B>()};
    auto r = gs_lanczos(be, Hc, n, kept_basis_max_dim, max_iter, tol);
    g.applies = Hc.applies;
    if (!r) return g;
    g.residual = r->residual;
    if (!(g.residual <= tol)) return g;
    g.energy    = r->energy;
    g.vector    = std::move(r->vector);
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
