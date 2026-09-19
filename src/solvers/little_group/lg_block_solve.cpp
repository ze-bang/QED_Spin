// =============================================================================
// src/solvers/little_group/lg_block_solve.cpp -- per-block eigensolves (dense / Lanczos), crossover, star filter
// Part of the little-group engine; see lg_internal.h for the file map.
// =============================================================================

#include "lg_internal.h"

namespace ed::solvers {

using namespace lg_detail;

namespace lg_detail {

// Per-block solve callbacks -----------------------------------------------

// Dense eigenvalues (ascending) of a materialized block through LAPACK
// divide-and-conquer (dsyevd / zheevd) -- the SAME threaded solver the
// abelian / full-diag lanes use (lanczos.cpp, orchestrator.cpp) --
// instead of Eigen's SelfAdjointEigenSolver, whose tridiagonalisation is
// single-threaded: a 19,264-dim projected block spun for HOURS on one
// core while the whole OpenMP pool sat idle (gdb-confirmed 2026-07-24),
// and at the lowest-path dense crossover the same wall made the serial
// star walk look hung (audit 2026-07-30). Real blocks (real momenta
// under time reversal) take the ~2x cheaper dsyevd real path, matching
// the abelian lane's arithmetic. The Eigen matrix is column-major ==
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
dense_block_eigenvalues(const ed::matvec::MatVecOperator& mv) {
    Eigen::MatrixXcd Hb = materialize(mv);
    return dense_eigenvalues_inplace(Hb);
}

// Full-spectrum: dense eigenvalues of the (projected or plain) block.
//
// Dense LAPACK divide-and-conquer (dsyevd/zheevd), threaded through the linked
// BLAS/LAPACK (AOCL here) -- the SAME solver the abelian / full-diag lane uses
// (lanczos.cpp, orchestrator.cpp). The previous Eigen SelfAdjointEigenSolver is
// single-threaded: on a 19,264-dim projected block it spun for hours on ONE
// core while the whole OpenMP pool sat idle, so the non-abelian little-group
// lane lost to the abelian lane it is supposed to beat (gdb-confirmed
// 2026-07-24). Real blocks (real momenta under time reversal) take the ~2x
// cheaper real path, matching the abelian lane's real arithmetic.
[[nodiscard]] std::vector<double>
solve_block_full(const ed::matvec::MatVecOperator& mv) {
    if (mv.dim() == 0) return {};
    return dense_block_eigenvalues(mv);
}

// Shared dense/Lanczos crossover for the lowest-k path. Kept in one place so
// the GPU deferred-batch lane below makes exactly the same dense-vs-Lanczos
// decision as the CPU ``solve_block_lowest``.
[[nodiscard]] std::uint64_t lowest_dense_floor(std::size_t k, int dense_max_dim) {
    // DEFAULT iteration cap on purpose (not lg_lowest_max_iter): raising
    // ED_SYM_LG_LOWEST_MAX_ITER for a frontier campaign must not widen
    // the dense band 4x with it. ED_SYM_LG_DENSE_FLOOR remains the
    // explicit dense-crossover override.
    const std::uint64_t max_iter_cap =
        std::max<std::uint64_t>(40u * static_cast<std::uint64_t>(k), 400u);
    // Audit 2026-07-30: the crossover was 32x the Lanczos cap (~1.3e4 at
    // k <= 10), sized to keep the OLD convergence gate -- which could
    // return converged top-of-spectrum values as the "lowest k" -- away
    // from any block it might corrupt. With the contiguous k-lowest
    // Paige gate the Lanczos path is honest at every dim (wrong is now
    // impossible; at worst a budget-capped block returns fewer values
    // flagged unconverged), so the floor only needs to cover the S1
    // within-block-degeneracy regime: dense resolves true multiplicities
    // that a single-vector recurrence cannot. 4x the cap (1600 at k <=
    // 10) still covers every historically-degenerate validated case
    // (the 4x4 n_up=8 blocks ~800) while releasing the 2e3-1.3e4 band
    // to Lanczos -- where the serial star walk was paying 8-15 s of
    // latency-bound threaded zheevd PER BLOCK (measured: the N=20 ring
    // walk cost 227 s against the abelian lane's 0.7 s). Raise
    // ED_SYM_LG_DENSE_FLOOR when a mid-band block needs exact
    // multiplicities (the documented S1 mitigation, unchanged).
    std::uint64_t dense_floor = std::max<std::uint64_t>(
        static_cast<std::uint64_t>(dense_max_dim), 4u * max_iter_cap);
    if (const long long df = ed::env::integer("ED_SYM_LG_DENSE_FLOOR", -1); df >= 0)
        dense_floor = static_cast<std::uint64_t>(df);
    return dense_floor;
}

// Batched-GPU-eigensolve gate: opt.use_gpu is the request; ED_SYM_LG_GPU=0 is
// the global little-group GPU veto (same env var that gates the rep-gather);
// a present device is required. Failures inside the lane degrade to the CPU
// path (the engine's graceful-degradation contract).
[[nodiscard]] bool lg_gpu_eigensolve_enabled(const LittleGroupOptions& opt) {
    if (!opt.use_gpu) return false;
    if (!ed::env::flag("ED_SYM_LG_GPU", true)) return false;   // =0 vetoes
    return ed::have_cuda();
}

// Several lowest levels of one block above the dense crossover: thick-restart
// Krylov-Schur with locking (single vector, block_size <= 1) or its block form
// (block_size = p resolves within-block multiplicities up to p). Both reorthogonalise
// fully inside each cycle, so there are no ghost copies to dedup, and both lock
// Ritz pairs strictly from the bottom: a level whose residual has not converged
// stops the locked prefix, it is never replaced by a higher one.
//
// Budgets. The per-cycle basis (m length-nb vectors) is capped by the RAM this job
// may still allocate (cgroup-aware); a cap too small to hold k + 8 vectors is a
// clean refusal, never a silent fall-back to the ghost-prone scan. The total
// iteration budget is max(200k, 2000) or ED_SYM_LG_LOWEST_MAX_ITER, spent
// as restart cycles -- so the environment lever keeps its meaning.
[[nodiscard]] std::vector<double>
solve_block_lowest_krylov_schur(const ed::matvec::MatVecOperator& mv, std::size_t k,
                                int block_size, bool* converged_out,
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
    // ED_SYM_LG_LOWEST_MAX_ITER still overrides absolutely.
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
    constexpr double tol = 1e-9;     // absolute residual ||H x - theta x||

    auto apply_H = [&mv](const Complex* in, Complex* out, std::size_t nn) {
        mv.apply(in, out, nn);
    };
    ed::matvec::CpuBackend be;
    std::vector<double> ev;
    std::vector<std::vector<Complex>> vv;    // Ritz vectors (block coordinates)
    bool conv = false;
    if (block_size <= 1) {
        std::uint64_t seed = 0x51ED0B70ULL;       // same stream as the k = 1 scan
        seed ^= static_cast<std::uint64_t>(ed::env::integer("ED_SYM_LG_SEED", 0))
                * 0x9E3779B97F4A7C15ULL;
        std::mt19937_64 gen(seed);
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
        ev   = std::move(r.eigenvalues);
        conv = r.converged;
        if (vecs_out)
            for (auto& v : r.eigenvectors) vv.emplace_back(v.get(), v.get() + nb);
    } else {
        ed::krylov::BlockKrylovSchurOptions o;
        o.num_eigs             = k;
        o.block_size           = static_cast<std::size_t>(block_size);
        o.max_iter             = per_cycle;
        o.max_restarts         = restarts;
        o.tolerance            = tol;
        o.max_subspace_vectors = cycle_cap;
        o.compute_vectors      = vecs_out != nullptr;
        auto r = ed::krylov::block_krylov_schur_kernel(be, apply_H, nb, nb, o);
        ev   = std::move(r.eigenvalues);
        conv = r.converged;
        if (vecs_out)
            for (auto& v : r.eigenvectors) vv.emplace_back(v.get(), v.get() + nb);
    }
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
        *converged_out = conv && ev_sorted.size() >= k && (!vecs_out || have_vecs);
    return ev_sorted;
}

// Lowest-k: dense on small blocks, Lanczos otherwise.
// Stage-9f verification fix (2026-07-12). The previous body delegated to the
// legacy ``::lanczos`` wrapper with an iteration budget of ``max_it = 2k+40``
// -- far too small to converge k eigenvalues on near-degenerate little-group
// blocks -- and no residual guard, so partially-converged and ghost Ritz
// values (K=1 local-ring reorth) were returned as eigenvalues (caught at 4x4
// J1-J2, J2=0.15, n_up=8, k=10: ghost -8.461485 beside the true -8.461508
// doublet, spurious -8.44734 between genuine levels).  Replaced with a
// direct kernel call: dense values-only eigensolve (mirroring
// ``solve_block_full``) below a crossover, and above it the kernel Lanczos
// with a LocalDGKS3 ring of 8, no stored basis, a real iteration budget,
// and a k-lowest Ritz stationarity gate.
[[nodiscard]] std::vector<double>
solve_block_lowest(const ed::matvec::MatVecOperator& mv, int want,
                   int dense_max_dim, bool* converged_out, int block_size) {
    if (converged_out) *converged_out = true;
    const std::uint64_t nb = mv.dim();
    if (nb == 0) return {};
    const std::size_t k = static_cast<std::size_t>(std::max<std::uint64_t>(
        1u, std::min<std::uint64_t>(static_cast<std::uint64_t>(want), nb)));
    // Dense crossover (Jul 2026 fix; resized 2026-07-30): originally 32x
    // the Lanczos cap because the OLD convergence gate could return a
    // spurious extreme (the 4x4 n_up=8 ~800-dim block returned an
    // interior level -7.75 instead of the true GS -8.57). That failure
    // class is closed by the contiguous k-lowest Paige gate below (an
    // unconverged low value now truncates and flags -- it can never be
    // replaced by a higher one), so the floor is back to a PERF+S1
    // decision: dense resolves true within-block multiplicities and is
    // cheapest below ~4x the iteration cap; above it the honest Lanczos
    // wins (the serial star walk was paying 8-15 s of latency-bound
    // threaded zheevd per mid-band block). See lowest_dense_floor for
    // the sizing rationale and the ED_SYM_LG_DENSE_FLOOR override
    // (raise it for exact multiplicities on a suspect block; set it to
    // 1 in tests to force the Lanczos path at toy dims).
    const std::uint64_t dense_floor = lowest_dense_floor(k, dense_max_dim);
    if (nb <= dense_floor || nb <= 2) {
        const std::vector<double> w = dense_block_eigenvalues(mv);  // ascending
        const std::size_t m = std::min<std::size_t>(k, w.size());
        return std::vector<double>(w.begin(), w.begin() + m);
    }

    // Several levels, or an explicit block size: Krylov-Schur (see above). The
    // basis-free scan below stays the k = 1 lane -- the one production uses at
    // N = 36, where a Krylov basis of 1e8-dimensional vectors does not fit.
    if (k > 1 || block_size > 1)
        return solve_block_lowest_krylov_schur(mv, k, block_size, converged_out);

    // S1 (WITHIN-BLOCK genuine degeneracy): a single-vector Lanczos returns
    // exactly ONE Ritz value per eigenvalue no matter its true multiplicity
    // (a random start has one component in a degenerate eigenspace), so an
    // accidental degeneracy inside THIS (k, irrep, parity, Sz) block is
    // undercounted. The DENSE branch above resolves it exactly, and the dense
    // crossover covers every block up to 4x the iteration cap (1600 at
    // k <= 10) -- verified at 4x4 J2=1.0 (which DOES carry a within-block
    // degeneracy at block ~800): normal operation matches the dense
    // spectrum. The residual gap is a block that both exceeds the crossover AND
    // carries an accidental degeneracy; two remedies: raise ED_SYM_LG_DENSE_FLOOR
    // so the block goes dense (exact, memory permitting), or ask for block_size >= 2,
    // which routes the block through block Krylov-Schur (above) and resolves
    // multiplicities up to the block size. This scan is the k = 1 lane only.
    ed::matvec::CpuBackend be;
    std::vector<Complex> v0(nb);
    // ED_SYM_LG_SEED offsets the start vector (default 0): the multi-seed
    // verification protocol for within-block degeneracy suspicion -- two
    // runs with different seeds must agree on every distinct level (a level
    // with accidentally tiny overlap against one seed shows up with the
    // other). NOTE: single-vector Lanczos still returns ONE copy of a
    // genuinely degenerate pair regardless of seed; multiplicity needs
    // block Lanczos (ledger #2).
    std::uint64_t seed = 0x51ED0B70ULL;
    seed ^= static_cast<std::uint64_t>(ed::env::integer("ED_SYM_LG_SEED", 0))
            * 0x9E3779B97F4A7C15ULL;
    std::mt19937_64 gen(seed);
    std::normal_distribution<double> nd(0.0, 1.0);
    for (auto& v : v0) v = Complex(nd(gen), nd(gen));

    ed::krylov::LanczosKernelOptions kopts;
    kopts.max_iter        = static_cast<std::size_t>(std::min<std::uint64_t>(
        nb, lg_lowest_max_iter(k)));
    // Ring reorth is a MEMORY term at frontier dims: 8 ring vectors x 16 B
    // x nb is ~48 GB per 3.8e8-dim block (measured 81 G RSS against a 96 G
    // budget on the 4x3 kagome campaign, 2026-07-19). This scan is
    // eigenvalues-only and the k-DISTINCT Paige-bound gate below is
    // ghost-aware by design, so above the two-pass dim floor we drop to
    // the pure three-term recurrence: ghosts cost duplicate converged
    // copies (deduped), not wrong eigenvalues. Small blocks keep the ring
    // -- it sharpens the excited window at negligible cost there.
    if (static_cast<std::size_t>(nb) > lg_two_pass_min_dim()) {
        kopts.reorth          = ed::krylov::ReorthPolicy::None;
    } else {
        kopts.reorth          = ed::krylov::ReorthPolicy::LocalDGKS3;
        kopts.local_ring_size = 8;
    }
    kopts.keep_basis      = false;
    kopts.dim_cap         = nb;
    // k-LOWEST converged Ritz early exit (Jul 2026; CONTIGUITY fix
    // 2026-07-30): at 1e8 dims the window fills with ghost COPIES of
    // converged extremes, and a ghost is exactly as stationary as an
    // eigenvalue -- the first production block burned its full budget and
    // returned 8x E0. The tridiagonal residual bound |beta_m * z_{m,j}| is
    // free, rigorous (Paige), and ghost-aware in combination with dedup.
    //
    // CONTIGUITY (the 2026-07-30 ghost-eigenvalue fix): the previous gate
    // stopped once ANY k distinct Ritz values carried converged bounds.
    // Lanczos converges the TOP extreme first, so on large blocks the gate
    // collected k converged top-of-spectrum values within ~40 iterations
    // and stopped before the bottom had converged at all; the keep loop
    // below then returned those top values AS the "lowest k", flagged
    // converged (measured: 4x2 kagome BFG, 338019-dim blocks, block min
    // reported +11.58 while the true sector minimum is -6.57 -- and the
    // fabricated value was near-identical across all momentum stars
    // because the Ising-dominated spectrum top barely feels k). The gate
    // must demand that the k LOWEST distinct Ritz values, walked
    // contiguously from the bottom, EACH carry a converged bound -- the
    // first unconverged distinct value vetoes the exit.
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
    // Ghost handling (Jul 2026; the first 126M-dim production block returned
    // EIGHT copies of E0): with a local reorth ring at dim ~1e8 the Ritz
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
    // CONTIGUITY fix (2026-07-30, pairs with the gate above): walk the
    // Ritz values ASCENDING, dedup ghost copies, and take the k lowest
    // distinct values -- STOPPING at the first unconverged one. The old
    // loop `continue`d past unconverged low values and backfilled with
    // converged UPPER-spectrum values, which is precisely how the tower
    // scan fabricated "lowest" eigenvalues near the spectrum TOP with
    // converged=true (Lanczos converges the top extreme first). An
    // unconverged low value now truncates the list and flags the block
    // unconverged; it is never silently replaced by a higher value.
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
    // 1b: a budget-capped block that could not deliver k distinct converged
    // values must be DISTINGUISHABLE from a converged one downstream.
    if (converged_out) *converged_out = all_converged && keep.size() >= k;
    return keep;
}

// Dimension of the (n_up | parity | full) subspace this walk must tile.
[[nodiscard]] std::uint64_t
subspace_dim_of(int n_sites, const LittleGroupOptions& opt) {
    if (opt.n_up >= 0) {
        long double c = 1.0L;
        const int kk = std::min(opt.n_up, n_sites - opt.n_up);
        for (int i = 0; i < kk; ++i)
            c = c * (n_sites - i) / (i + 1);
        return static_cast<std::uint64_t>(c + 0.5L);
    }
    if (opt.sz_parity >= 0) return std::uint64_t{1} << (n_sites - 1);
    return std::uint64_t{1} << n_sites;
}

// -----------------------------------------------------------------------------
// Star selection from the environment (ED_SYM_LG_ONLY_K0), the job-splitting form
// of LittleGroupOptions::only_k0.
//
// Precedence: an explicit opt.only_k0 WINS -- the environment is consulted only when
// the caller named no star. (It used to be the other way round: an exported variable
// silently replaced the argument, so a driver that passed only_k0 could be made to
// solve a different star by a leftover export in the job script.) When both are
// present the variable is ignored, with one line on stderr saying so.
// "plan" flips plan mode where the caller honours it; prefer opt.plan_only.
// -----------------------------------------------------------------------------
void parse_only_k0_env(std::set<int>& only_k0, bool& plan_only) {
    const std::string fs = ed::env::text("ED_SYM_LG_ONLY_K0");
    if (fs.empty()) return;
    if (!only_k0.empty()) {
        static bool noted = false;
        if (!noted) {
            noted = true;
            std::fprintf(stderr,
                "[little_group] ED_SYM_LG_ONLY_K0=%s ignored: the caller passed only_k0 "
                "explicitly, and an argument takes precedence over the environment\n",
                fs.c_str());
        }
        return;
    }
    if (fs == "plan") {
        plan_only = true;
        return;
    }
    std::size_t pos = 0;
    while (pos < fs.size()) {
        const std::size_t c = fs.find(',', pos);
        const std::string tok = fs.substr(pos, c == std::string::npos ? c : c - pos);
        if (!tok.empty()) only_k0.insert(std::stoi(tok));
        if (c == std::string::npos) break;
        pos = c + 1;
    }
}

}  // namespace lg_detail

}  // namespace ed::solvers
