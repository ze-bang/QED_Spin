// =============================================================================
// include/ed/distributed/distributed_krylov_schur.h
//
// Phase 9: distributed Krylov-Schur for Hermitian operators (Layer 3).
//
// Implements explicitly-restarted Lanczos with Ritz-pair locking
// (a thick-restart variant of Krylov-Schur for symmetric problems).
//
// Each cycle:
//   1. Build a Lanczos basis of size up to ``max_iter`` rooted at the
//      current seed vector, with full reorthogonalisation against
//      *both* the in-cycle basis AND every previously locked Ritz
//      vector.
//   2. Solve the (m_eff x m_eff) tridiagonal eigenproblem on every rank.
//   3. Estimate Ritz-pair residuals as r_i = beta_m * |y_{m-1, i}| and
//      promote any that satisfy ``residual < tol`` into the locked set.
//   4. Pick the leading unlocked Ritz vector as the next cycle's seed.
//
// The result struct mirrors ``DistributedLanczosResult`` so callers and
// the standalone CLI binary can swap entry points without changing
// other code.
//
// Why call this "Krylov-Schur"?  For Hermitian/symmetric problems the
// Schur form is diagonal and the canonical Krylov-Schur restart
// (Stewart 2001) reduces to the Wu/Simon thick-restart Lanczos. The
// implementation here uses the simpler "explicit restart with
// locking" variant -- mathematically equivalent to thick-restart for
// the locked Ritz pairs, just without keeping the (k+l) auxiliary
// vectors between cycles. The simplification is justified by the
// distributed setting: holding O(m * local_n) basis vectors per rank
// dominates memory, so rebuilding the basis from scratch each cycle
// is cheaper than carrying it forward.
//
// Cost per cycle:
//   * m matvecs (the Lanczos build).
//   * Reorthogonalisation against locked vectors and the in-cycle
//     basis: O((|locked| + j) * local_n) per Lanczos step j.
//   * One (m x m) Eigen tridiagonal solve, replicated on every rank.
//
// Memory per rank:
//   * m * local_n (in-cycle Krylov basis).
//   * |locked| * local_n (locked Ritz vectors, freed at function exit).
//
// Convergence semantics:
//   * The residual norm estimate beta_m * |y_{m-1, i}| is the textbook
//     Lanczos residual bound (Saad 2011, eq. 6.11). For Hermitian
//     operators it is sharp.
//   * The function returns once ``options.exct`` Ritz pairs are locked,
//     or after ``options.max_iter * 6`` matvecs have been spent
//     (whichever comes first).
// =============================================================================

#pragma once

#ifdef WITH_MPI

#include <ed/distributed/distributed_lanczos.h>
#include <ed/distributed/distributed_operator.h>

namespace ed::distributed {

/// Distributed Krylov-Schur (thick-restart Lanczos with locking).
/// Collective on ``op.comm()`` -- every rank must call.
DistributedLanczosResult distributed_krylov_schur(
    const DistributedOperator& op,
    const DistributedLanczosOptions& options = {});

}  // namespace ed::distributed

#endif  // WITH_MPI
