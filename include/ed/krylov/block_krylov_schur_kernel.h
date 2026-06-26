#pragma once
// =============================================================================
// include/ed/krylov/block_krylov_schur_kernel.h
//
// Block Krylov-Schur (thick-restart block Lanczos) for Hermitian operators.
//
// This is the block generalization of `krylov_schur_kernel` and the restarted
// generalization of `block_lanczos_kernel`. For a Hermitian H, Krylov-Schur
// reduces to thick-restart Lanczos; the *block* variant carries `p` vectors per
// step, so it resolves eigenvalues of multiplicity up to `p` (and tightly
// clustered groups) in a single cycle -- exactly the regime where single-vector
// Lanczos/Krylov-Schur stall or miss copies (e.g. frustrated lattices).
//
// Algorithm (one cycle):
//   1. Block-Lanczos factorization of up to `m` blocks from the current
//      starting block V0, deflating every new block against the LOCKED set
//      (so converged eigenpairs are not re-found) -> block-tridiagonal T.
//   2. Dense eigensolve of T (LAPACKE_zheevd) -> Ritz values + vectors Y.
//   3. Residual ||B_last * Y_lastblock|| per Ritz pair; lock the converged
//      contiguous prefix from the bottom (reconstruct phi = sum_blk V_blk Y_blk,
//      orthonormalize against the locked set, append).
//   4. If enough locked, stop; else THICK-RESTART: re-seed V0 with the `p`
//      lowest non-locked Ritz vectors (filled with random columns if fewer than
//      `p` remain) and repeat.
//
// Templated on Backend + matvec functor exactly like the other kernels, so the
// reduction axis (Full / Sz / abelian / non-abelian symmetry) and the device
// axis (CPU / CUDA) are inherited through the same `MatVecOperator` seam --
// nothing symmetry-specific lives here. MPI backends are a documented deferral
// (same static_assert as block_lanczos_kernel).
// =============================================================================

#include <ed/krylov/block_lanczos_kernel.h>   // detail helpers + Complex + LAPACKE
#include <ed/krylov/subspace_policy.h>        // krylov_subspace_dim (shared sizing)
#include <ed/matvec/backend.h>

#include <algorithm>
#include <cstddef>
#include <limits>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <stdexcept>
#include <vector>

namespace ed::krylov {

struct BlockKrylovSchurOptions {
    std::size_t   num_eigs        = 1;
    std::size_t   block_size      = 4;
    std::size_t   max_iter        = 60;   ///< max blocks built per cycle
    std::size_t   max_restarts    = 50;
    double        tolerance       = 1e-10;
    bool          compute_vectors = false;
    std::string   output_dir;
    std::uint64_t global_n        = 0;
    std::size_t   reorth_period   = 1;    ///< full reorth every K blocks (1 = always)
    /// Memory cap on the per-cycle subspace, in resident length-N vectors
    /// (m_blocks*block_size). 0 = no cap. Set by the orchestrator from available
    /// RAM/VRAM so the footprint is predictable. See krylov_subspace_dim.
    std::uint64_t max_subspace_vectors = 0;
};

struct BlockKrylovSchurResult {
    std::vector<double>                         eigenvalues;
    std::vector<ed::matvec::Backend::UniqueVec> eigenvectors;  ///< if compute_vectors
    std::size_t                                 restarts  = 0;
    bool                                        converged = false;
    // ---- convergence diagnostics ----
    /// Per-eigenvalue residual ‖B_last · y_lastblock‖ ≈ ‖H xᵢ − θᵢ xᵢ‖ at lock
    /// time, aligned with `eigenvalues`.
    std::vector<double>                         residuals;
    /// Number of returned eigenvalues that met `tolerance` (locked == converged).
    std::size_t                                 n_converged = 0;
    /// Best (lowest) not-yet-locked residual after each restart cycle.
    std::vector<double>                         resid_history;
};

template <typename Backend, typename MatvecFn>
BlockKrylovSchurResult block_krylov_schur_kernel(Backend&                       be,
                                                 MatvecFn&&                     apply_H,
                                                 std::size_t                    local_n,
                                                 std::uint64_t                  /*global_n*/,
                                                 const BlockKrylovSchurOptions& opts)
{
    static_assert(
        std::is_base_of_v<ed::matvec::CpuBackend, std::decay_t<Backend>>
#ifdef WITH_CUDA
            || std::is_base_of_v<ed::matvec::CudaBackend, std::decay_t<Backend>>
#endif
        ,
        "block_krylov_schur_kernel currently supports CpuBackend / CudaBackend. "
        "Distributed backends use src/distributed/* directly (Phase 2.3).");

    using ed::krylov::detail::hermitianize_inplace;
    using ed::krylov::detail::build_projected_matrix;

    const std::size_t N = local_n;
    if (N == 0) throw std::invalid_argument("block_krylov_schur_kernel: local_n == 0");

    const std::size_t b = std::max<std::size_t>(1, std::min<std::size_t>(opts.block_size, N));
    const std::size_t k = std::max<std::size_t>(1, std::min<std::size_t>(opts.num_eigs, N));
    const double      tol       = (opts.tolerance <= 0.0) ? 1e-12 : opts.tolerance;
    constexpr double  breakdown = 1e-12;
    // Blocks per cycle. The per-cycle Krylov subspace is m_blocks*b vectors; it
    // must be large enough to converge k eigenvalues, or every cycle locks
    // nothing and the thick restart makes no progress (returns 0 eigenvalues on
    // a large sector -- the block analogue of the single-vector KS subspace-cap
    // bug). So we size it like single-vector KS: a 2k+20-vector floor, GROWN by
    // the user's `max_iter` budget (interpreted in Krylov-vector units, divided
    // by the block width), and only then capped by N/b. `max_iter` thus GROWS
    // the subspace rather than only shrinking it.
    const std::size_t max_blocks_dim = (N + b - 1) / b;
    // Memory-bounded subspace (shared with KS + the orchestrator + the planner):
    // floor 2k+20, grown by max_iter, CAPPED by the memory budget so m_blocks*b
    // vectors cannot OOM. Then round up to whole blocks and clamp to N/b.
    const std::size_t want_dim = ed::krylov::krylov_subspace_dim(
        k, opts.max_iter, static_cast<std::uint64_t>(N), opts.max_subspace_vectors);
    std::size_t m_blocks = std::max<std::size_t>(2, (want_dim + b - 1) / b);
    m_blocks = std::min(m_blocks, max_blocks_dim);

    const Complex one(1.0, 0.0), zero(0.0, 0.0), neg_one(-1.0, 0.0);

    // ---- locked (converged) eigenpairs -------------------------------------
    std::vector<ed::matvec::Backend::UniqueVec> locked_vecs;  // one N-vector each
    std::vector<double>                          locked_evals;
    std::vector<double>                          locked_resid; // residual at lock time

    // Deflate a single N-column (device) against the whole locked set (CGS2).
    auto deflate_against_locked = [&](Complex* col) {
        for (int pass = 0; pass < 2; ++pass)
            for (auto& lv : locked_vecs) {
                const Complex c = be.dot(lv.get(), col, N);
                be.axpy(-c, lv.get(), col, N);
            }
    };

    // ---- starting block V0 (random, orthonormalized) -----------------------
    auto V_seed = be.make_zero_vector(N * b);
    {
        std::vector<Complex> hb(N * b);
        std::mt19937 gen(12345u);
        std::uniform_real_distribution<double> dist(-1.0, 1.0);
        for (auto& z : hb) z = Complex(dist(gen), dist(gen));
        be.copy_from_host(hb.data(), V_seed.get(), N * b);
    }

    auto V_prev = be.make_zero_vector(N * b);
    auto V_curr = be.make_zero_vector(N * b);
    auto W      = be.make_zero_vector(N * b);
    auto AB     = be.make_zero_vector(b * b);

    BlockKrylovSchurResult R;
    bool converged = false;
    std::size_t stall_cycles = 0;   // consecutive no-lock cycles at the growth cap

    for (std::size_t restart = 0; restart < opts.max_restarts; ++restart) {
        R.restarts = restart + 1;
        const std::size_t need = (k > locked_evals.size()) ? (k - locked_evals.size()) : 0;
        if (need == 0) { converged = true; break; }

        // Deflate the starting block against locked, then orthonormalize it.
        for (std::size_t c = 0; c < b; ++c) deflate_against_locked(V_seed.get() + c * N);
        {
            std::vector<Complex> R0(b * b);
            be.qr_thin(V_seed.get(), N, b, R0.data());
        }
        be.copy(V_seed.get(), V_curr.get(), N * b);
        be.fill_zero(V_prev.get(), N * b);

        std::vector<ed::matvec::Backend::UniqueVec> basis;
        basis.reserve(m_blocks);
        std::vector<std::vector<Complex>> alpha_blocks, beta_blocks;
        std::vector<Complex> Aj(b * b), Bj(b * b), Bprev(b * b, zero);

        // ---- block-Lanczos factorization (deflated against locked) ----------
        for (std::size_t j = 0; j < m_blocks; ++j) {
            auto keep = be.make_zero_vector(N * b);
            be.copy(V_curr.get(), keep.get(), N * b);
            basis.emplace_back(std::move(keep));

            for (std::size_t c = 0; c < b; ++c)
                apply_H(V_curr.get() + c * N, W.get() + c * N, N);

            // A_j = V_curr^H W  (Hermitianized).
            be.gemm('C', 'N', b, b, N, one, V_curr.get(), N, W.get(), N, zero, AB.get(), b);
            be.copy_to_host(AB.get(), Aj.data(), b * b);
            be.all_reduce_sum_vec(Aj.data(), b * b);
            hermitianize_inplace(Aj.data(), b);
            be.copy_from_host(Aj.data(), AB.get(), b * b);

            be.gemm('N', 'N', N, b, b, neg_one, V_curr.get(), N, AB.get(), b, one, W.get(), N);
            if (j > 0) {
                be.copy_from_host(Bprev.data(), AB.get(), b * b);
                be.gemm('N', 'C', N, b, b, neg_one, V_prev.get(), N, AB.get(), b, one, W.get(), N);
            }

            // Full CGS2 against the stored basis AND the locked set (deflation).
            const bool full = (opts.reorth_period > 0) && (j % opts.reorth_period == 0);
            const int passes = full ? 2 : 1;
            for (int pass = 0; pass < passes; ++pass) {
                const std::size_t upto = full ? basis.size() : std::min<std::size_t>(basis.size(), 1);
                for (std::size_t blk = 0; blk < (full ? upto : basis.size()); ++blk) {
                    const Complex* Vp = basis[basis.size() - 1 - blk].get();
                    be.gemm('C', 'N', b, b, N, one, Vp, N, W.get(), N, zero, AB.get(), b);
                    be.gemm('N', 'N', N, b, b, neg_one, Vp, N, AB.get(), b, one, W.get(), N);
                    if (!full) break;
                }
                for (std::size_t c = 0; c < b; ++c) deflate_against_locked(W.get() + c * N);
            }

            alpha_blocks.push_back(Aj);
            if (j > 0) beta_blocks.push_back(Bprev);

            be.qr_thin(W.get(), N, b, Bj.data());
            double min_diag = std::numeric_limits<double>::max();
            for (std::size_t i = 0; i < b; ++i)
                min_diag = std::min(min_diag, std::abs(Bj[i + i * b]));
            if (min_diag < breakdown) break;

            std::swap(V_prev, V_curr);
            be.copy(W.get(), V_curr.get(), N * b);
            Bprev = Bj;
        }

        // ---- eigensolve of the block-tridiagonal T --------------------------
        const std::size_t mb    = alpha_blocks.size();
        if (mb == 0) break;
        const std::size_t total = mb * b;
        std::vector<Complex> T;
        build_projected_matrix(alpha_blocks, beta_blocks, b, T);
        std::vector<double> theta(total);
        if (LAPACKE_zheevd(LAPACK_COL_MAJOR, 'V', 'U',
                           static_cast<lapack_int>(total),
                           reinterpret_cast<lapack_complex_double*>(T.data()),
                           static_cast<lapack_int>(total), theta.data()) != 0)
            break;
        // T now holds eigenvectors Y (column-major, total x total), theta ascending.

        // Residual of Ritz pair i: || B_last * Y[(mb-1)*b : mb*b, i] ||.
        auto ritz_residual = [&](std::size_t i) {
            std::vector<Complex> r(b);
            cblas_zgemv(CblasColMajor, CblasNoTrans, b, b, &one,
                        Bj.data(), b,
                        &T[(mb - 1) * b + i * total], 1, &zero, r.data(), 1);
            return cblas_dznrm2(static_cast<int>(b), r.data(), 1);
        };

        // Reconstruct Ritz vector i (device) into `out` (N): sum_blk V_blk * Y_blk,i.
        auto reconstruct = [&](std::size_t i, Complex* out) {
            be.fill_zero(out, N);
            std::vector<Complex> yblk(b);
            for (std::size_t blk = 0; blk < mb; ++blk) {
                for (std::size_t r = 0; r < b; ++r)
                    yblk[r] = T[blk * b + r + i * total];
                be.copy_from_host(yblk.data(), AB.get(), b);   // reuse AB as b-scratch
                be.gemm('N', 'N', N, 1, b, one, basis[blk].get(), N, AB.get(), b, one, out, N);
            }
        };

        // ---- lock the converged contiguous prefix from the bottom -----------
        std::size_t newly = 0;
        for (std::size_t i = 0; i < total && newly < need; ++i) {
            const double rho = ritz_residual(i);
            if (rho > tol) break;                       // stop at first unconverged
            auto phi = be.make_zero_vector(N);
            reconstruct(i, phi.get());
            deflate_against_locked(phi.get());
            const double nrm = be.nrm2(phi.get(), N);
            if (nrm < 1e-13) continue;
            be.scale(Complex(1.0 / nrm, 0.0), phi.get(), N);
            locked_evals.push_back(theta[i]);
            locked_resid.push_back(rho);
            locked_vecs.emplace_back(std::move(phi));
            ++newly;
        }
        // Convergence curve: residual of the first not-yet-converged Ritz pair
        // (those below it were locked this cycle).
        if (newly < total) R.resid_history.push_back(ritz_residual(newly));

        if (locked_evals.size() >= k) { converged = true; break; }

        // Adaptive subspace growth: the simple re-seed thick restart can STALL
        // (the residual estimate plateaus above tol) on a tight-gap spectrum, so
        // a fixed small per-cycle subspace would never lock anything. When a
        // cycle locks nothing, grow the per-cycle block count so the next cycle
        // explores a larger Krylov space. The growth is CAPPED (not at full N/b)
        // because the per-cycle cost is O(m_blocks^2) with reorth_period=1; an
        // uncapped grow-to-full-dim is pathologically slow on small sectors. The
        // cap is generous enough to converge typical gaps; harder cases return
        // the converged prefix (>= the ground state) rather than hang. Use
        // BLOCK_LANCZOS for the efficient degeneracy solve.
        // ``2k+20`` is the base subspace floor (matching krylov_subspace_dim's
        // floor before the max_iter growth); 8x its block count is a generous
        // growth ceiling. (Was ``floor_dim`` before that local was folded into
        // krylov_subspace_dim.)
        const std::size_t floor_blocks = (2 * k + 20 + b - 1) / b;
        const std::size_t grow_cap =
            std::min(max_blocks_dim,
                     std::max<std::size_t>(8 * floor_blocks, 128));
        if (newly == 0 && m_blocks < grow_cap)
            m_blocks = std::min(m_blocks + m_blocks / 2 + 1, grow_cap);
        // Early-out: once at the cap, a few more no-lock cycles will not help the
        // weak thick restart -- stop and return the converged prefix (the lowest
        // eigenvalues, including the ground state) rather than burning the full
        // max_restarts on slow large-subspace cycles.
        if (newly == 0 && m_blocks >= grow_cap) { if (++stall_cycles >= 3) break; }
        else stall_cycles = 0;

        // ---- thick restart: re-seed V0 with the lowest non-locked Ritz block.
        be.fill_zero(V_seed.get(), N * b);
        std::size_t filled = 0;
        for (std::size_t i = newly; i < total && filled < b; ++i, ++filled)
            reconstruct(i, V_seed.get() + filled * N);
        if (filled < b) {                               // pad with random columns
            std::vector<Complex> hb((b - filled) * N);
            std::mt19937 gen(67890u + restart);
            std::uniform_real_distribution<double> dist(-1.0, 1.0);
            for (auto& z : hb) z = Complex(dist(gen), dist(gen));
            be.copy_from_host(hb.data(), V_seed.get() + filled * N, (b - filled) * N);
        }
    }

    // ---- assemble ascending result -----------------------------------------
    std::vector<std::size_t> ord(locked_evals.size());
    for (std::size_t i = 0; i < ord.size(); ++i) ord[i] = i;
    std::sort(ord.begin(), ord.end(),
              [&](std::size_t a, std::size_t c) { return locked_evals[a] < locked_evals[c]; });
    const std::size_t out_k = std::min<std::size_t>(k, locked_evals.size());
    R.eigenvalues.reserve(out_k);
    R.residuals.reserve(out_k);
    for (std::size_t i = 0; i < out_k; ++i) {
        R.eigenvalues.push_back(locked_evals[ord[i]]);
        R.residuals.push_back(locked_resid[ord[i]]);
    }
    if (opts.compute_vectors) {
        R.eigenvectors.reserve(out_k);
        for (std::size_t i = 0; i < out_k; ++i)
            R.eigenvectors.emplace_back(std::move(locked_vecs[ord[i]]));
    }
    // Every returned (locked) eigenvalue passed the residual test by construction.
    R.n_converged = out_k;
    R.converged = converged || (locked_evals.size() >= k);
    return R;
}

}  // namespace ed::krylov
