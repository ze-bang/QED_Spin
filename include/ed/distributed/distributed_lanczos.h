// =============================================================================
// include/ed/distributed/distributed_lanczos.h
//
// Phase 3b #2: distributed-memory Lanczos solver.
//
// Each rank owns ONLY its slab of every Krylov basis vector. Inner products
// (alpha_j = <V_j | H V_j>) and norms (beta_{j+1} = ||w||_2) collapse to
// local zdotc/dnrm2 + a single MPI_Allreduce; axpy / zdscal are fully local.
// At convergence, the small (max_iter x max_iter) tridiagonal eigenproblem
// is solved redundantly on every rank, producing the same eigenvalue array
// everywhere -- no Bcast needed.
//
// Three-term recurrence with optional full re-orthogonalization:
//   * full_reorth = false  -- canonical lanczos_no_ortho (cheapest;
//                             O(N) per iteration)
//   * full_reorth = true   -- modified Gram-Schmidt against every prior
//                             basis vector kept rank-local (each MGS step
//                             is local zdotc + Allreduce + axpy; total
//                             O(j N) per iteration j)
//
// At Phase 3b first-cut scope, eigenvectors are NOT returned (only the
// Ritz values). Eigenvector reconstruction at distributed scale needs the
// rank-local Krylov basis to be replayed in a second pass; we leave that
// hook for Phase 3b #2.5 and document it honestly in PHASE_3_SUMMARY.md.
// =============================================================================

#pragma once

#ifdef WITH_MPI

#include <ed/distributed/distributed_operator.h>

#include <complex>
#include <cstdint>
#include <vector>

namespace ed::distributed {

struct DistributedLanczosOptions {
    /// Maximum Lanczos iterations (Krylov space dimension cap).
    std::uint64_t max_iter = 200;

    /// Number of eigenvalues to return (smallest-first). Caller still gets
    /// every Ritz value; this controls the output size.
    std::uint64_t exct = 1;

    /// Convergence tolerance for the smallest exct eigenvalues.
    double tol = 1e-12;

    /// If true, MGS-reorthogonalise w against every prior basis vector at
    /// every iteration (rank-local zdotc + MPI_Allreduce + axpy per pair).
    /// Default false to mirror lanczos_no_ortho.
    bool full_reorth = false;

    /// If true, rank 0 prints per-iteration diagnostics. Other ranks stay
    /// silent regardless.
    bool verbose = false;

    /// Seed for the rank-0-generated initial vector. Other ranks receive
    /// their slab via MPI_Scatterv, so different `seed` values produce
    /// different runs across all ranks consistently.
    unsigned long seed = 12345UL;
};

struct DistributedLanczosResult {
    /// Ritz values, sorted ascending. Length = min(exct, iterations).
    /// Replicated on every rank (bit-identical given the same seed).
    std::vector<double> eigenvalues;

    /// Number of iterations actually performed (<= max_iter; can be less
    /// if the recurrence breaks down via beta_j ~= 0, which signals an
    /// invariant Krylov subspace).
    int iterations = 0;
};

/// Distributed Lanczos for a DistributedOperator. Collective on
/// op.comm() -- every rank must call.
DistributedLanczosResult distributed_lanczos(
    const DistributedOperator& op,
    const DistributedLanczosOptions& options = {});

}  // namespace ed::distributed

#endif  // WITH_MPI
