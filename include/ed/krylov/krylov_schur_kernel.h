#pragma once
// =============================================================================
// include/ed/krylov/krylov_schur_kernel.h
//
// Backend-templated thick-restart Krylov-Schur kernel. Builds on top of
// `ed::krylov::lanczos_kernel<Backend>`:
//
//   for restart cycle r = 0..R-1:
//     1. Re-orthogonalise the seed against the locked Ritz set.
//     2. Run `lanczos_kernel<Backend>` with `aux_ortho_ptrs = locked set`.
//        This builds an m-step Lanczos factorisation orthogonal to both
//        the current cycle's basis AND the locked Ritz vectors.
//     3. Solve the m x m projected tridiagonal eigenproblem on host
//        (LAPACKE_dstevd via `solve_tridiag_with_eigenvectors`).
//     4. For each Ritz pair, evaluate residual = |β_last * y[m-1, i]|.
//        If below `tolerance`, reconstruct the Ritz vector
//        (`V_local * y` --- a local linear combination of the basis)
//        and append to the locked set.
//     5. Re-seed with a non-locked Ritz vector and continue.
//
// Single body drives the four deployment targets (CPU / CUDA / MPI /
// MPI+CUDA). The only Backend-specific operations are the basis-vector
// reconstruction (one `axpy_many` per locked vector) and the seed
// generation (caller-provided via `seed_local`).
//
// Phase 2.2 of the Minimalist ED Collapse (May 2026): replaces the
// previous CPU-only `static_assert`-guarded forward to the legacy
// `::krylov_schur` body.
// =============================================================================

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include <ed/distributed/distributed_lanczos_kernel.h>  // solve_tridiag_with_eigenvectors
#include <ed/krylov/lanczos_kernel.h>
#include <ed/matvec/backend.h>

namespace ed::krylov {

using Complex = std::complex<double>;

struct KrylovSchurOptions {
    std::size_t num_eigs        = 1;
    std::size_t max_iter        = 100;
    double      tolerance       = 1e-10;
    bool        compute_vectors = false;
    /// Maximum restart cycles before we give up (matches the legacy
    /// `::krylov_schur` body's hard limit).
    std::size_t max_restarts    = 30;
    /// Optional output directory (kept for ABI compat; not consumed
    /// here -- the kernel returns eigenvalues / eigenvectors via the
    /// result struct).
    std::string output_dir;
    /// Global problem dimension. Distributed backends MUST set this so
    /// the per-cycle `lanczos_kernel<Backend>` knows the global cap.
    /// Default 0 means "use local_n" (single-rank / single-GPU runs).
    std::uint64_t global_n      = 0;
    /// Breakdown threshold passed to the per-cycle `lanczos_kernel`.
    /// 1e-13 matches the historical Krylov-Schur restart body.
    double      breakdown_tol   = 1e-13;
};

struct KrylovSchurResult {
    std::vector<double>                    eigenvalues;
    /// Locked Ritz vectors in backend memory, one per converged
    /// eigenvalue. Only populated when `opts.compute_vectors == true`.
    std::vector<ed::matvec::Backend::UniqueVec> eigenvectors;
    std::size_t                            iters_done = 0;
    std::size_t                            restarts   = 0;
    bool                                   converged  = false;
};

namespace detail {

inline std::size_t ks_subspace_size(std::size_t k, std::size_t max_iter,
                                    std::size_t global_dim) {
    // Match the legacy heuristic: m = min(2k+20, max_iter, global_dim).
    std::size_t m = 2 * k + 20;
    if (m > max_iter) m = max_iter;
    if (global_dim > 0 && m > global_dim) m = global_dim;
    if (m < k + 1) m = k + 1;
    return m;
}

}  // namespace detail

/// Run thick-restart Krylov-Schur on `matvec` starting from `seed_local`
/// (already in backend memory, dimension `local_n`). For distributed
/// backends `seed_local` is the rank-local slab of a globally
/// scattered seed; caller is responsible for that scatter (see
/// `ed::distributed::scatter_initial_vector` for the CPU+MPI path).
template <typename Backend, typename MatvecFn>
KrylovSchurResult krylov_schur_kernel(Backend&       be,
                                      MatvecFn&&     matvec,
                                      std::size_t    local_n,
                                      const Complex* seed_local,
                                      const KrylovSchurOptions& opts)
{
    using ed::matvec::Backend;
    using ed::distributed::kernel::solve_tridiag_with_eigenvectors;

    if (opts.max_iter == 0) {
        throw std::invalid_argument("krylov_schur_kernel: max_iter == 0");
    }

    const std::uint64_t global_dim =
        (opts.global_n > 0) ? opts.global_n
                            : static_cast<std::uint64_t>(local_n);
    const std::size_t   k_target = std::max<std::size_t>(1, opts.num_eigs);
    const std::size_t   m_max    = detail::ks_subspace_size(
        k_target, opts.max_iter, static_cast<std::size_t>(global_dim));

    // --- locked Ritz set (in backend memory) ---------------------------
    std::vector<ed::matvec::Backend::UniqueVec> locked_vecs;
    std::vector<double>                          locked_evals;
    locked_vecs.reserve(k_target);
    locked_evals.reserve(k_target);

    // --- seed vector (own copy in backend memory) ----------------------
    auto v_seed = be.make_zero_vector(local_n);
    if (local_n > 0) {
        be.copy(seed_local, v_seed.get(), local_n);
        const double n0 = be.nrm2(v_seed.get(), local_n);
        if (n0 > 0.0) {
            be.scale(Complex(1.0 / n0, 0.0), v_seed.get(), local_n);
        }
    }

    KrylovSchurResult R;
    bool              converged = false;

    for (std::size_t restart = 0; restart < opts.max_restarts; ++restart) {
        // --- step 1: project seed against the locked set (CGS2 by hand) -
        for (int pass = 0; pass < 2; ++pass) {
            for (auto& lv : locked_vecs) {
                const Complex c = be.dot(lv.get(), v_seed.get(), local_n);
                be.axpy(-c, lv.get(), v_seed.get(), local_n);
            }
        }
        const double seed_norm = be.nrm2(v_seed.get(), local_n);
        if (seed_norm < 1e-13) {
            // Seed collapsed onto the locked subspace -- nothing more to
            // extract via this cycle. Bail out cleanly.
            break;
        }
        be.scale(Complex(1.0 / seed_norm, 0.0), v_seed.get(), local_n);

        // --- step 2: per-cycle Lanczos -------------------------------
        std::vector<const Complex*> aux;
        aux.reserve(locked_vecs.size());
        for (auto& lv : locked_vecs) aux.push_back(lv.get());

        LanczosKernelOptions kopts;
        kopts.max_iter      = m_max;
        kopts.reorth        = ReorthPolicy::FullCGS2;
        kopts.keep_basis    = true;
        kopts.breakdown_tol = opts.breakdown_tol;
        kopts.dim_cap       = static_cast<std::size_t>(global_dim);
        kopts.aux_ortho_ptrs = std::move(aux);

        auto kres = lanczos_kernel(be, matvec, local_n, v_seed.get(), kopts);
        R.iters_done += kres.iters_done;
        R.restarts    = restart + 1;
        if (kres.alpha.empty()) break;

        // --- step 3: solve the projected tridiag -----------------------
        std::vector<double> evals, weights, evecs_cm;
        solve_tridiag_with_eigenvectors(kres.alpha, kres.beta,
                                        kres.alpha.size(),
                                        evals, weights, evecs_cm);
        const std::size_t m_eff    = kres.alpha.size();
        const double      beta_last = kres.beta.back();

        std::vector<std::size_t> idx(m_eff);
        std::iota(idx.begin(), idx.end(), std::size_t{0});
        std::sort(idx.begin(), idx.end(),
                  [&](std::size_t a, std::size_t b) {
                      return evals[a] < evals[b];
                  });

        const std::size_t need = (k_target > locked_evals.size())
                                     ? (k_target - locked_evals.size())
                                     : 0;
        std::size_t newly_locked = 0;
        for (std::size_t r = 0; r < std::min(need, m_eff); ++r) {
            const std::size_t i = idx[r];
            const double residual =
                std::abs(beta_last) *
                std::abs(evecs_cm[i * m_eff + (m_eff - 1)]);
            if (residual >= opts.tolerance) break;

            // Reconstruct phi = sum_j evec[i][j] * basis[j] (local).
            auto phi = be.make_zero_vector(local_n);
            {
                std::vector<Complex> coefs(m_eff);
                std::vector<const Complex*> basis_ptrs(m_eff);
                for (std::size_t j = 0; j < m_eff; ++j) {
                    coefs[j] = Complex(evecs_cm[i * m_eff + j], 0.0);
                    basis_ptrs[j] = kres.basis[j].get();
                }
                be.axpy_many(coefs.data(), basis_ptrs.data(), m_eff,
                             phi.get(), local_n);
            }
            // Project against the existing locked set (CGS2).
            for (int pass = 0; pass < 2; ++pass) {
                for (auto& lv : locked_vecs) {
                    const Complex c = be.dot(lv.get(), phi.get(), local_n);
                    be.axpy(-c, lv.get(), phi.get(), local_n);
                }
            }
            const double pn = be.nrm2(phi.get(), local_n);
            if (pn < 1e-14) break;
            be.scale(Complex(1.0 / pn, 0.0), phi.get(), local_n);
            locked_evals.push_back(evals[i]);
            locked_vecs.emplace_back(std::move(phi));
            ++newly_locked;
        }

        if (locked_evals.size() >= k_target) {
            converged = true;
            break;
        }

        // --- step 5: re-seed with a non-locked Ritz vector ----------
        const std::size_t seed_rank =
            std::min<std::size_t>(newly_locked, m_eff - 1);
        const std::size_t i_seed = idx[seed_rank];
        {
            be.fill_zero(v_seed.get(), local_n);
            std::vector<Complex> coefs(m_eff);
            std::vector<const Complex*> basis_ptrs(m_eff);
            for (std::size_t j = 0; j < m_eff; ++j) {
                coefs[j] = Complex(evecs_cm[i_seed * m_eff + j], 0.0);
                basis_ptrs[j] = kres.basis[j].get();
            }
            be.axpy_many(coefs.data(), basis_ptrs.data(), m_eff,
                         v_seed.get(), local_n);
        }
        if (kres.iters_done == 0) break;
    }

    // Sort the locked spectrum ascending.
    std::vector<std::size_t> order(locked_evals.size());
    std::iota(order.begin(), order.end(), std::size_t{0});
    std::sort(order.begin(), order.end(),
              [&](std::size_t a, std::size_t b) {
                  return locked_evals[a] < locked_evals[b];
              });
    R.eigenvalues.reserve(locked_evals.size());
    for (std::size_t i : order) R.eigenvalues.push_back(locked_evals[i]);

    if (opts.compute_vectors) {
        R.eigenvectors.reserve(locked_vecs.size());
        for (std::size_t i : order) {
            R.eigenvectors.emplace_back(std::move(locked_vecs[i]));
        }
    }
    R.converged = converged || (locked_evals.size() >= k_target);
    return R;
}

}  // namespace ed::krylov
