// =============================================================================
// src/solvers/little_group/lg_block_solve.cpp -- per-block eigensolves (dense / Lanczos / Krylov-Schur), crossover
// Part of the little-group engine; see lg_internal.h for the file map.
// =============================================================================

#include "lg_internal.h"

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

// Several lowest levels of one block above the dense crossover: thick-restart
// Krylov-Schur with locking. It reorthogonalises fully inside each cycle, so there
// are no ghost copies to dedup; it locks Ritz pairs strictly from the bottom (a level
// whose residual has not converged stops the locked prefix, it is never replaced by a
// higher one); and after locking, a fresh start deflated against the locked set finds
// the further copies of a degenerate level.
//
// Budgets. The per-cycle basis (m length-nb vectors) is capped by the RAM this job
// may still allocate (cgroup-aware); a cap too small to hold k + 8 vectors is a
// clean refusal, never a silent fall-back to the ghost-prone scan. The total
// iteration budget is max(200k, 2000), spent as restart cycles.
[[nodiscard]] std::vector<double>
solve_block_lowest_krylov_schur(const ed::LinearOperator& mv, std::size_t k,
                                bool* converged_out,
                                std::vector<std::vector<Complex>>* vecs_out) {
    const std::size_t nb = mv.dim();
    const std::uint64_t cap = ed::krylov::krylov_vector_budget(
        ed::core::available_ram_bytes(), nb, /*safety=*/0.5, /*reserve_vecs=*/8);
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
    const std::uint64_t budget = lg_lowest_max_iter(
        k, std::max<std::uint64_t>(200u * static_cast<std::uint64_t>(k), 2000u));
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

    auto apply_H = [&mv](const Complex* in, Complex* out, std::size_t nn) {
        mv.apply(in, out, nn);
    };
    ed::matvec::CpuBackend be;
    std::mt19937_64 gen(0x51ED0B70ULL);       // same stream as the k = 1 scan
    std::normal_distribution<double> nd(0.0, 1.0);
    std::vector<Complex> v0(nb);
    for (auto& v : v0) v = Complex(nd(gen), nd(gen));
    ed::krylov::KrylovSchurOptions o;
    o.num_eigs             = k;
    o.max_iter             = per_cycle;
    o.max_restarts         = restarts;
    o.tolerance            = tol;
    o.max_subspace_vectors = cycle_cap;
    o.compute_vectors      = vecs_out != nullptr;
    auto r = ed::krylov::krylov_schur_kernel(be, apply_H, nb, v0.data(), o);
    std::vector<double> ev = std::move(r.eigenvalues);
    const bool conv  = r.converged;
    const bool whole = r.exhausted;   // every eigenvalue of the block was found (fewer than k when nb < k)
    std::vector<std::vector<Complex>> vv;    // Ritz vectors (block coordinates)
    if (vecs_out)
        for (auto& v : r.eigenvectors) vv.emplace_back(v.get(), v.get() + nb);
    // Ascending, vectors kept aligned with their values; then the k lowest.
    std::vector<std::size_t> order(ev.size());
    std::iota(order.begin(), order.end(), std::size_t{0});
    std::sort(order.begin(), order.end(),
              [&ev](std::size_t a, std::size_t b) { return ev[a] < ev[b]; });
    const bool have_vecs = vecs_out != nullptr && vv.size() == ev.size();
    std::vector<double> ev_sorted;
    std::vector<std::vector<Complex>> vv_sorted;
    for (std::size_t i = 0; i < order.size() && ev_sorted.size() < k; ++i) {
        ev_sorted.push_back(ev[order[i]]);
        if (have_vecs) vv_sorted.push_back(std::move(vv[order[i]]));
    }
    if (vecs_out) *vecs_out = std::move(vv_sorted);
    if (converged_out)
        *converged_out = (conv || whole) && (ev_sorted.size() >= k || whole) && (!vecs_out || have_vecs);
    return ev_sorted;
}

// Lowest-k: dense values-only eigensolve (as ``solve_block_full``) below a
// crossover; above it Krylov-Schur for several levels, and for one level the
// kernel Lanczos with no stored basis, a real iteration budget and a
// k-lowest Ritz residual gate. A small budget (e.g. 2k+40) or no residual
// guard lets partially-converged and ghost Ritz values through as
// eigenvalues on near-degenerate blocks.
[[nodiscard]] std::vector<double>
solve_block_lowest(const ed::LinearOperator& mv, int want,
                   int dense_max_dim, bool* converged_out) {
    if (converged_out) *converged_out = true;
    const std::uint64_t nb = mv.dim();
    if (nb == 0) return {};
    const std::size_t k = static_cast<std::size_t>(std::max<std::uint64_t>(
        1u, std::min<std::uint64_t>(static_cast<std::uint64_t>(want), nb)));
    // Dense crossover: a performance + within-block-multiplicity decision
    // (the contiguous k-lowest Paige gate below makes Lanczos correct at
    // any dim: an unconverged low value truncates and flags, it is never
    // replaced by a higher one). Dense resolves true multiplicities and is
    // cheapest below ~4x the iteration cap. See lowest_dense_floor for the
    // sizing and the explicit dense_max_dim (raise it for exact multiplicities
    // on a suspect block; 0 in tests forces the Lanczos path at toy dims).
    const std::uint64_t dense_floor = lowest_dense_floor(k, dense_max_dim);
    if (nb <= dense_floor || nb <= 2) {
        const std::vector<double> w = dense_block_eigenvalues(mv);  // ascending
        const std::size_t m = std::min<std::size_t>(k, w.size());
        return std::vector<double>(w.begin(), w.begin() + m);
    }

    // Several levels: Krylov-Schur (see above). The basis-free scan below stays the
    // k = 1 lane -- the one production uses at N = 36, where a Krylov basis of
    // 1e8-dimensional vectors does not fit.
    if (k > 1)
        return solve_block_lowest_krylov_schur(mv, k, converged_out);

    // WITHIN-BLOCK genuine degeneracy: a single-vector Lanczos returns
    // exactly ONE Ritz value per eigenvalue no matter its true multiplicity
    // (a random start has one component in a degenerate eigenspace), so an
    // accidental degeneracy inside THIS (k, irrep, parity, Sz) block is
    // undercounted. The DENSE branch above resolves it exactly, and the dense
    // crossover covers every block up to 4x the iteration cap (1600 at
    // k <= 10) -- verified at 4x4 J2=1.0 (which DOES carry a within-block
    // degeneracy at block ~800): normal operation matches the dense
    // spectrum. The residual gap is a block that both exceeds the crossover AND
    // carries an accidental degeneracy; two remedies: raise the dense crossover so
    // the block goes dense (exact, memory permitting), or ask for k >= 2, which
    // routes the block through Krylov-Schur (above), whose fresh starts after
    // locking find every copy. This scan is the k = 1 lane only.
    ed::matvec::CpuBackend be;
    std::vector<Complex> v0(nb);
    // Fixed start-vector seed. Single-vector Lanczos returns ONE copy of a
    // genuinely degenerate pair (see above for the lanes that count copies).
    std::mt19937_64 gen(0x51ED0B70ULL);
    std::normal_distribution<double> nd(0.0, 1.0);
    for (auto& v : v0) v = Complex(nd(gen), nd(gen));

    ed::krylov::LanczosKernelOptions kopts;
    kopts.max_iter        = static_cast<std::size_t>(std::min<std::uint64_t>(
        nb, lg_lowest_max_iter(k)));
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
    kopts.dim_cap         = nb;
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
                std::size_t distinct = 0;
                double last = 0.0;
                for (int j = 0; j < m && distinct < kk; ++j) {
                    if (distinct > 0
                        && std::abs(w(j) - last) <= 1e-9 * scale)
                        continue;                 // ghost copy of `last`
                    const double bound =
                        beta_m * std::abs(es.eigenvectors()(m - 1, j));
                    if (bound > 1e-7 * scale) return false;  // lowest
                        // unconverged distinct value: keep iterating
                    last = w(j);
                    ++distinct;
                }
                return distinct >= kk;
            };
        kopts.convergence_check_interval = 10;
    }
    auto apply_H = [&mv](const Complex* in, Complex* out, std::size_t nn) {
        mv.apply(in, out, nn);
    };
    auto kres = ed::krylov::lanczos_kernel(be, apply_H,
                                           static_cast<std::size_t>(nb),
                                           v0.data(), kopts);
    const std::size_t m = kres.alpha.size();
    if (m == 0) return {};
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
    // duplicates ARE ghosts: keep the first of each cluster. Additionally
    // keep only Ritz values whose tridiagonal residual bound
    // |beta_m * z_{m,j}| marks them converged -- both tests are free (the
    // tridiag is m <= a few hundred). The dense branch (exact; genuine
    // duplicates possible) is untouched.
    const double beta_m = (kres.beta.size() > m) ? std::abs(kres.beta[m]) : 0.0;
    const double scale  = std::max(
        {std::abs(diag.front()), std::abs(diag[m - 1]), 1e-300});
    // CONTIGUITY (pairs with the gate above): walk the Ritz values
    // ASCENDING, dedup ghost copies, and take the k lowest distinct values
    // -- STOPPING at the first unconverged one. Skipping past it would
    // backfill with converged UPPER-spectrum values (Lanczos converges the
    // top extreme first) and report them as the "lowest" with
    // converged=true. An unconverged low value truncates the list and flags
    // the block unconverged; it is never silently replaced by a higher value.
    std::vector<double> keep;
    bool all_converged = true;
    for (std::size_t j = 0; j < m && keep.size() < k; ++j) {
        if (!keep.empty()
            && std::abs(diag[j] - keep.back()) <= 1e-9 * scale)
            continue;                             // ghost copy
        const double bound = beta_m * std::abs(z[(m - 1) + j * m]);
        if (bound > 1e-7 * scale) {               // lowest unconverged
            all_converged = false;                // distinct value: stop --
            break;                                // never backfill from above
        }
        keep.push_back(diag[j]);
    }
    // A budget-capped block that could not deliver k distinct converged
    // values must be DISTINGUISHABLE from a converged one downstream.
    if (converged_out) *converged_out = all_converged && keep.size() >= k;
    return keep;
}

// The lowest `want` eigenpairs of one block, in block coordinates. Dense below the
// lowest-k crossover (exact); one level through the certified ground-state solver
// (memory-light two-pass lane at frontier dimensions); several levels through
// Krylov-Schur with vectors. `converged` is false when the block could not
// certify the requested window; the certified prefix is still returned.
[[nodiscard]] std::pair<std::vector<double>, std::vector<std::vector<Complex>>>
solve_block_eigenpairs(const ed::LinearOperator& mv, int want,
                       int dense_max_dim, bool* converged) {
    *converged = true;
    const std::size_t nb = mv.dim();
    std::vector<double> ev;
    std::vector<std::vector<Complex>> vv;
    if (nb == 0) return {ev, vv};
    const std::size_t k = std::min<std::size_t>(std::max(want, 1), nb);
    if (nb <= lowest_dense_floor(k, dense_max_dim) || nb <= 2) {
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> es(materialize(mv));
        if (es.info() != Eigen::Success)
            throw std::runtime_error("little_group: dense block eigensolve failed");
        for (std::size_t j = 0; j < k; ++j) {
            ev.push_back(es.eigenvalues()(static_cast<Eigen::Index>(j)));
            std::vector<Complex> v(nb);
            for (std::size_t i = 0; i < nb; ++i)
                v[i] = es.eigenvectors()(static_cast<Eigen::Index>(i),
                                         static_cast<Eigen::Index>(j));
            vv.push_back(std::move(v));
        }
        return {ev, vv};
    }
    if (k == 1) {
        try {
            auto [e0, v] = solve_gs_vector(mv);
            ev.push_back(e0);
            vv.push_back(std::move(v));
        } catch (const std::runtime_error&) {
            *converged = false;            // residual guard failed: certify nothing
        }
        return {ev, vv};
    }
    ev = solve_block_lowest_krylov_schur(mv, k, converged, &vv);
    return {ev, vv};
}

}  // namespace lg_detail

}  // namespace ed::solvers
