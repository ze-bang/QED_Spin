#pragma once
// =============================================================================
// include/ed/krylov/krylov_schur_kernel.h
//
// Backend-templated restarted Krylov-Schur with locking, built on
// `ed::krylov::lanczos_kernel<Backend>`:
//
//   for restart cycle r = 0..R-1:
//     1. Re-orthogonalise the seed against the locked Ritz set.
//     2. Run `lanczos_kernel<Backend>` with `aux_ortho_ptrs = locked set`.
//        This builds an m-step Lanczos factorisation orthogonal to both
//        the current cycle's basis AND the locked Ritz vectors.
//     3. Solve the m x m projected tridiagonal eigenproblem on host
//        (`solve_tridiag_with_eigenvectors`).
//     4. For each Ritz pair, evaluate residual = |β_last * y[m-1, i]|.
//        If below `tolerance`, reconstruct the Ritz vector
//        (`V_local * y` --- a local linear combination of the basis)
//        and append to the locked set.
//     5. Re-seed with a non-locked Ritz vector and continue -- or, once a cycle has
//        locked its whole Krylov space (an exact invariant subspace), with a fresh
//        random start; two such starts inside the locked span exhaust the space.
//   then the degeneracy probe: one cycle from a fresh random start (see the body).
//
// The same body serves every Backend (CPU / CUDA); only the basis-vector
// reconstruction (`axpy_many`) and the seed transfer touch backend memory.
// =============================================================================

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include <ed/krylov/lanczos_kernel.h>
#include <ed/krylov/subspace_policy.h>      // krylov_subspace_dim (shared sizing)
#include <ed/krylov/tridiag_eigensolver.h>  // solve_tridiag_with_eigenvectors
#include <ed/matvec/backend.h>

namespace ed::krylov {

using Complex = std::complex<double>;

struct KrylovSchurOptions {
    std::size_t num_eigs        = 1;
    std::size_t max_iter        = 100;
    double      tolerance       = 1e-10;
    bool        compute_vectors = false;
    /// Maximum restart cycles before we give up.
    std::size_t max_restarts    = 30;
    /// Global problem dimension, forwarded as the per-cycle
    /// `lanczos_kernel<Backend>` dimension cap.
    /// Default 0 means "use local_n" (CPU / single-GPU runs).
    std::uint64_t global_n      = 0;
    /// Breakdown threshold passed to the per-cycle `lanczos_kernel`.
    double      breakdown_tol   = 1e-13;
    /// Memory cap on the per-cycle Krylov subspace, in resident length-N
    /// vectors (the basis held during each restart cycle is the dominant cost).
    /// 0 = no cap. The orchestrator sets this from available RAM/VRAM so the
    /// footprint is PREDICTABLE: m_max <= max_subspace_vectors regardless of
    /// max_iter. See ed::krylov::krylov_subspace_dim.
    std::uint64_t max_subspace_vectors = 0;
    /// After convergence, look for a level the single-vector restarts skipped (a second copy
    /// of a degenerate eigenvalue) with one extra cycle from a fresh random start.
    bool        probe_degeneracy = true;
};

struct KrylovSchurResult {
    std::vector<double>                    eigenvalues;
    /// Locked Ritz vectors in backend memory, one per converged
    /// eigenvalue. Only populated when `opts.compute_vectors == true`.
    std::vector<ed::matvec::Backend::UniqueVec> eigenvectors;
    std::size_t                            iters_done = 0;
    std::size_t                            restarts   = 0;
    bool                                   converged  = false;
    /// The locked vectors span the whole space: every eigenvalue was found (fewer than
    /// num_eigs when the space is smaller).
    bool                                   exhausted  = false;
};

namespace detail {

// Convenience wrapper with no memory cap. The memory-bounded sizing lives
// in ed::krylov::krylov_subspace_dim (subspace_policy.h) and is what the kernel,
// orchestrator, and planner all use so they AGREE on the footprint.
inline std::size_t ks_subspace_size(std::size_t k, std::size_t max_iter,
                                    std::size_t global_dim) {
    return ed::krylov::krylov_subspace_dim(k, max_iter,
                                           static_cast<std::uint64_t>(global_dim),
                                           /*max_vectors=*/0);
}

}  // namespace detail

/// Run thick-restart Krylov-Schur on `matvec` starting from `seed_local`
/// (already in backend memory, dimension `local_n`).
template <typename Backend, typename MatvecFn>
KrylovSchurResult krylov_schur_kernel(Backend&       be,
                                      MatvecFn&&     matvec,
                                      std::size_t    local_n,
                                      const Complex* seed_local,
                                      const KrylovSchurOptions& opts)
{
    using ed::matvec::Backend;
    using ed::krylov::detail::solve_tridiag_with_eigenvectors;

    if (opts.max_iter == 0) {
        throw std::invalid_argument("krylov_schur_kernel: max_iter == 0");
    }

    const std::uint64_t global_dim =
        (opts.global_n > 0) ? opts.global_n
                            : static_cast<std::uint64_t>(local_n);
    const std::size_t   k_target = std::max<std::size_t>(1, opts.num_eigs);
    // Per-cycle subspace: grown by max_iter, but CAPPED by the memory budget
    // (max_subspace_vectors) so the basis footprint is predictable / cannot OOM.
    const std::size_t   m_max    = ed::krylov::krylov_subspace_dim(
        k_target, opts.max_iter, global_dim, opts.max_subspace_vectors);

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

    // CGS2 against the locked set.
    auto deflate = [&](Complex* v) {
        for (int pass = 0; pass < 2; ++pass) {
            for (auto& lv : locked_vecs) {
                const Complex c = be.dot(lv.get(), v, local_n);
                be.axpy(-c, lv.get(), v, local_n);
            }
        }
    };
    struct Cycle {
        LanczosKernelResult      kres;
        std::vector<double>      evals, evecs_cm;
        std::vector<std::size_t> idx;          // ascending Ritz values
        std::size_t              m = 0;
        double                   beta_last = 0.0;
    };
    // One Lanczos factorisation from v_seed, orthogonal to the locked set. False when the
    // seed lies in the locked span or nothing was built.
    auto run_cycle = [&](Cycle& c) -> bool {
        deflate(v_seed.get());
        const double seed_norm = be.nrm2(v_seed.get(), local_n);
        if (seed_norm < 1e-13) return false;
        be.scale(Complex(1.0 / seed_norm, 0.0), v_seed.get(), local_n);
        std::vector<const Complex*> aux;
        aux.reserve(locked_vecs.size());
        for (auto& lv : locked_vecs) aux.push_back(lv.get());
        LanczosKernelOptions kopts;
        kopts.max_iter       = m_max;
        kopts.reorth         = ReorthPolicy::FullCGS2;
        kopts.keep_basis     = true;
        kopts.breakdown_tol  = opts.breakdown_tol;
        kopts.dim_cap        = static_cast<std::size_t>(global_dim);
        kopts.aux_ortho_ptrs = std::move(aux);
        c.kres = lanczos_kernel(be, matvec, local_n, v_seed.get(), kopts);
        R.iters_done += c.kres.iters_done;
        ++R.restarts;
        if (c.kres.alpha.empty()) return false;
        std::vector<double> weights;
        solve_tridiag_with_eigenvectors(c.kres.alpha, c.kres.beta, c.kres.alpha.size(),
                                        c.evals, weights, c.evecs_cm);
        c.m         = c.kres.alpha.size();
        c.beta_last = c.kres.beta.back();
        c.idx.resize(c.m);
        std::iota(c.idx.begin(), c.idx.end(), std::size_t{0});
        std::sort(c.idx.begin(), c.idx.end(),
                  [&](std::size_t a, std::size_t b) { return c.evals[a] < c.evals[b]; });
        return true;
    };
    auto ritz_vector = [&](const Cycle& c, std::size_t i, Complex* out) {
        be.fill_zero(out, local_n);
        std::vector<Complex> coefs(c.m);
        std::vector<const Complex*> basis_ptrs(c.m);
        for (std::size_t j = 0; j < c.m; ++j) {
            coefs[j]      = Complex(c.evecs_cm[i * c.m + j], 0.0);
            basis_ptrs[j] = c.kres.basis[j].get();
        }
        be.axpy_many(coefs.data(), basis_ptrs.data(), c.m, out, local_n);
    };
    // A fresh Gaussian start (deflated in run_cycle): the way on past an exact invariant
    // subspace. A cycle that ends on a breakdown spans one vector per distinct eigenvalue
    // of its start; once those are locked every Ritz vector of it is locked too, so a
    // re-seed from it deflates to zero, while a fresh start reaches the further copies of
    // a degenerate level (the Ising ring: a few distinct levels, each thousands of times).
    std::mt19937_64 fresh_gen(0xF8E5A7C3ULL);
    std::normal_distribution<double> fresh_nd(0.0, 1.0);
    std::vector<Complex> fresh_host;
    auto fresh_seed = [&] {
        fresh_host.resize(local_n);
        for (auto& z : fresh_host) z = Complex(fresh_nd(fresh_gen), fresh_nd(fresh_gen));
        be.copy_from_host(fresh_host.data(), v_seed.get(), local_n);
    };
    // Set when two fresh starts in a row lie in the locked span: the locked vectors span the
    // whole space, so every eigenvalue has been found.
    bool exhausted = false;
    // Restart cycles until `target` pairs are locked or `budget` cycles are spent. Pairs lock
    // strictly from the bottom of each cycle; the next cycle starts from the lowest unlocked
    // Ritz vector, or from a fresh start when the cycle locked all it had.
    auto lock_until = [&](std::size_t target, std::size_t budget) {
        int null_starts = 0;
        for (std::size_t cycle = 0; cycle < budget && locked_evals.size() < target; ++cycle) {
            Cycle c;
            if (!run_cycle(c)) {
                if (++null_starts >= 2) { exhausted = true; return; }
                fresh_seed();
                continue;
            }
            null_starts = 0;
            const std::size_t need = target - locked_evals.size();
            std::size_t newly_locked = 0;
            for (std::size_t r = 0; r < std::min(need, c.m); ++r) {
                const std::size_t i = c.idx[r];
                const double residual =
                    std::abs(c.beta_last) * std::abs(c.evecs_cm[i * c.m + (c.m - 1)]);
                if (residual >= opts.tolerance) break;
                auto phi = be.make_zero_vector(local_n);
                ritz_vector(c, i, phi.get());
                deflate(phi.get());
                const double pn = be.nrm2(phi.get(), local_n);
                if (pn < 1e-14) break;
                be.scale(Complex(1.0 / pn, 0.0), phi.get(), local_n);
                locked_evals.push_back(c.evals[i]);
                locked_vecs.emplace_back(std::move(phi));
                ++newly_locked;
            }
            if (locked_evals.size() >= target) return;
            if (newly_locked >= c.m) fresh_seed();          // the cycle's whole Krylov space is locked
            else ritz_vector(c, c.idx[newly_locked], v_seed.get());
        }
    };

    lock_until(k_target, opts.max_restarts);
    bool converged = locked_evals.size() >= k_target;

    // Degeneracy probe. Every cycle restarts from one Ritz vector orthogonal to the locked
    // set, so a second copy of a locked eigenvalue carries only roundoff weight from then on
    // and a higher level can be locked in its place. A fresh random start deflated against
    // the locked set has a generic component on it; a Ritz value of that start below the
    // highest locked level is an upper bound on a level that was skipped. Lock it, keep the
    // k lowest, and look again. Runs whenever something was locked and the space is not
    // exhausted: an unconverged block can have skipped copies too.
    if (!locked_evals.empty() && !exhausted && opts.probe_degeneracy) {
        std::mt19937_64 gen(0xDE6E4E7AULL);
        std::normal_distribution<double> nd(0.0, 1.0);
        std::vector<Complex> host(local_n);
        for (std::size_t round = 0; round < k_target; ++round) {
            const double top = *std::max_element(locked_evals.begin(), locked_evals.end());
            const double gap = std::max(10.0 * opts.tolerance, 1e-8 * std::max(1.0, std::abs(top)));
            for (auto& z : host) z = Complex(nd(gen), nd(gen));
            be.copy_from_host(host.data(), v_seed.get(), local_n);
            Cycle c;
            if (!run_cycle(c) || c.evals[c.idx[0]] >= top - gap) break;
            ritz_vector(c, c.idx[0], v_seed.get());
            const std::size_t before = locked_evals.size();
            lock_until(before + 1, opts.max_restarts);
            if (locked_evals.size() == before) { converged = false; break; }   // skipped, not recovered
            while (locked_evals.size() > k_target) {
                const auto hi = std::max_element(locked_evals.begin(), locked_evals.end()) - locked_evals.begin();
                locked_evals.erase(locked_evals.begin() + hi);
                locked_vecs.erase(locked_vecs.begin() + hi);
            }
        }
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
    R.converged = converged;
    R.exhausted = exhausted;
    return R;
}

}  // namespace ed::krylov
