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
// Phase 3b #6 wires up rank-local eigenvector reconstruction:
// `compute_eigenvectors = true` retains the rank-local Krylov basis V_local
// AND the (m x m) tridiagonal eigenvector matrix U so that
//   psi_k_local = V_local @ U[:, k]
// can be assembled per Ritz pair without re-running Lanczos. The
// `reconstruct_local_eigenvector` free function is the public hook used by
// distributed_ftlm() for observable expectation values and by callers that
// need ground-state Ritz vectors directly.
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

    /// If true, populate `tridiag_eigenvalues` and `tridiag_weights` in
    /// the result struct (one extra (m x m) eigensolve in addition to
    /// the per-iteration tridiagonal solves; m = max_iter).  Required
    /// by distributed_ftlm() for its trace estimator; default false to
    /// keep the eigenvalue-only path lean.
    bool compute_weights = false;

    /// If true, also retain the rank-local Krylov basis (m vectors of
    /// length local_n each, packed in `krylov_basis_local`) AND the
    /// (m x m) tridiagonal eigenvector matrix in `tridiag_eigenvectors`
    /// so that callers can reconstruct
    ///   psi_k_local = V_local @ U[:, k]
    /// via `reconstruct_local_eigenvector(...)`. Implies the basis MUST
    /// be kept, which costs `m * local_n * 16 B` per rank -- same as
    /// `full_reorth = true`. We force `full_reorth = true` internally
    /// when this is set, otherwise the retained basis is not numerically
    /// orthonormal and the reconstructed eigenvectors are unreliable.
    /// Default false.
    bool compute_eigenvectors = false;
};

struct DistributedLanczosResult {
    /// Ritz values, sorted ascending. Length = min(exct, iterations).
    /// Replicated on every rank (bit-identical given the same seed).
    std::vector<double> eigenvalues;

    /// Number of iterations actually performed (<= max_iter; can be less
    /// if the recurrence breaks down via beta_j ~= 0, which signals an
    /// invariant Krylov subspace).
    int iterations = 0;

    /// Full Ritz spectrum (same length as `tridiag_alpha`, replicated on
    /// every rank). Caller may use this with `tridiag_weights` for
    /// FTLM-style trace estimators when `compute_weights = true`. Empty
    /// otherwise.
    std::vector<double> tridiag_eigenvalues;

    /// Squared first component of each tridiagonal eigenvector
    /// (|<e_0 | psi_k>|^2 in tridiagonal-basis). Combined with the
    /// initial-vector overlap in the FTLM kernel:
    ///   Z(beta) ~ sum_k tridiag_weights[k] * exp(-beta * tridiag_eigenvalues[k])
    /// when v0 is L2-normalised. Empty unless compute_weights = true.
    std::vector<double> tridiag_weights;

    /// Tridiagonal eigenvectors in column-major flat layout: column k
    /// (m doubles) starts at `tridiag_eigenvectors[k * m]`, where
    /// `m = tridiag_eigenvalues.size()`. Same ordering as
    /// `tridiag_eigenvalues` (Eigen's natural ascending). Empty unless
    /// `compute_eigenvectors = true`.
    std::vector<double> tridiag_eigenvectors;

    /// Rank-local Krylov basis: `krylov_basis_local[j]` is the
    /// `local_n`-element slab of V_j on this rank. Length =
    /// `iterations`. Empty unless `compute_eigenvectors = true`.
    /// Memory: `iterations * local_n * 16 B` per rank.
    std::vector<std::vector<std::complex<double>>> krylov_basis_local;
};

/// Distributed Lanczos for a DistributedOperator. Collective on
/// op.comm() -- every rank must call.
DistributedLanczosResult distributed_lanczos(
    const DistributedOperator& op,
    const DistributedLanczosOptions& options = {});

/// Reconstruct the rank-local slab of the k-th Ritz eigenvector from a
/// `DistributedLanczosResult` produced with `compute_eigenvectors = true`.
///
///   psi_k_local = sum_j krylov_basis_local[j] * tridiag_eigenvectors[k * m + j]
///
/// where m = tridiag_eigenvalues.size(). The output `psi_k_local` is
/// resized to `local_n = krylov_basis_local[0].size()`. If `result` does
/// not carry eigenvector data (basis empty / U empty), throws
/// `std::invalid_argument`.
///
/// This is a purely local op (no MPI). Caller may follow up with
/// `MPI_Allgatherv` if a global eigenvector is needed for, e.g., a
/// dense reference comparison.
void reconstruct_local_eigenvector(
    const DistributedLanczosResult& result,
    std::size_t k,
    std::vector<std::complex<double>>& psi_k_local);

/// Result of distributed_lanczos_eigenvectors().
struct DistributedEigenpairsResult {
    /// Smallest `n_keep = min(exct, iterations)` Ritz values, ascending.
    std::vector<double> eigenvalues;

    /// Rank-local slabs of the corresponding Ritz vectors. Layout:
    /// `eigenvectors_local[k]` has length `local_n` and is the slab of
    /// the k-th eigenvector owned by this rank. Length = `n_keep`.
    std::vector<std::vector<std::complex<double>>> eigenvectors_local;

    /// Lanczos iteration count actually performed.
    int iterations = 0;
};

/// Convenience wrapper: run distributed_lanczos with
/// `compute_eigenvectors = true` and reconstruct the smallest
/// `min(options.exct, iterations)` Ritz vectors per rank. Collective.
DistributedEigenpairsResult distributed_lanczos_eigenvectors(
    const DistributedOperator& op,
    const DistributedLanczosOptions& options = {});

}  // namespace ed::distributed

// -----------------------------------------------------------------------------
// Phase 3b #7 stage 3: distributed Lanczos on the symmetry-projected operator.
//
// `distributed_lanczos_symmetry` runs the same kernel as `distributed_lanczos`
// (via the templated helper in `distributed_lanczos_kernel.h`) but on a
// `DistributedSymmetryOperator` --  i.e. the LPT-balanced orbit-row operator
// over the symmetry-projected basis. The slab geometry is rank-major with
// scrambled orbit ids; this entry point handles the initial-vector scatter
// + result reproduction so callers don't need to know about the orbit
// permutation.
//
// Eigenvalue / weights / eigenvector extraction options match the
// `DistributedLanczosOptions` semantics exactly. Eigenvector reconstruction
// produces the rank-local slab in *rank-major orbit ordering*; callers that
// want a global eigenvector in *natural orbit ordering* can use
// `partition.rank_orbits[r][k]` to permute back, exactly like the
// test harness in `test_distributed_symmetry_operator.cpp`.
// -----------------------------------------------------------------------------

#include <ed/distributed/distributed_symmetry_operator.h>

namespace ed::distributed {

DistributedLanczosResult distributed_lanczos_symmetry(
    const DistributedSymmetryOperator& op,
    const DistributedLanczosOptions& options = {});

}  // namespace ed::distributed

#endif  // WITH_MPI
