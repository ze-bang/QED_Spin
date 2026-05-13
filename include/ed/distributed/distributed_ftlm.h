// =============================================================================
// include/ed/distributed/distributed_ftlm.h
//
// Phase 3b #3: distributed-memory Finite-Temperature Lanczos Method (FTLM).
//
// Two-level parallelism:
//
//   * outer = MPI groups across samples (one group per random vector)
//   * inner = MPI ranks within a group, each owning a slab of v / Krylov
//             basis (handled transparently by DistributedOperator +
//             DistributedLanczos).
//
// The sample loop structure mirrors the existing serial FTLM in
// src/solvers/cpu/ftlm.cpp (Jaklic & Prelovsek, PRB 49, 5065 (1994)) so
// the partition-function and observable-normalisation conventions match.
//
// Phase 3b #5 wires up arbitrary observable expectation values via the
// canonical Jaklic-Prelovsek (J&P 1994) trace estimator:
//
//   <O>(beta) = ( sum_s sum_j q_j(s)^* f_j(s, beta) )
//             / ( sum_s sum_k w_k(s) exp(-beta E_k(s)) )
//
// where q_j(s) = <V_s[j] | O r_s>, f_j(s, beta) = sum_k U_s[j,k] U_s[0,k]
// exp(-beta E_k(s)), and r_s = V_s[0] is the Lanczos seed. Computational
// cost vs the Z-only path is O(1) extra DistributedOperator matvecs per
// sample (one for u = O r_s) plus m extra rank-local zdotc + Allreduce.
//
// Honest scope notes (carried into docs/history/PHASE_3_SUMMARY.md):
//
//   * For honest 40 the inner DistributedOperator hits the slab-decomposition
//     limits documented in the DistributedOperator header. Honest-40 FTLM
//     therefore needs the same symmetry-aware slabbing fix that
//     DistributedOperator does, before this module is run on the real-world
//     spectrum target.
//   * The observable Operator MUST share the same Hilbert space (n_bits) as
//     the Hamiltonian Operator, since we reuse the same balanced 1D row
//     decomposition. We do not currently check this -- caller is responsible.
// =============================================================================

#pragma once

#ifdef WITH_MPI

#include <ed/distributed/distributed_operator.h>

#include <cstdint>
#include <memory>
#include <vector>

namespace ed::distributed {

struct DistributedFtlmOptions {
    /// Number of random samples (R in J&P notation). Distributed evenly
    /// across `n_groups` MPI groups; each group runs the corresponding
    /// subset of samples.
    int n_samples = 32;

    /// Number of MPI groups across samples (i.e., outer parallelism).
    /// Must divide MPI_Comm_size(comm). Set to 1 for a "every rank does
    /// every sample" replicated-sample strategy (matches the J&P serial
    /// FTLM with one DistributedOperator across all ranks).
    int n_groups = 1;

    /// Lanczos parameters per sample.
    std::uint64_t lanczos_max_iter = 100;

    /// Inverse temperatures at which to evaluate Z(beta) and (in the
    /// future) <O>(beta). Empty -> {1.0}.
    std::vector<double> betas;

    /// Random seed offset; each sample s uses (seed_offset + s).
    unsigned long seed_offset = 0UL;

    /// If true, rank 0 prints per-sample / per-beta diagnostics.
    bool verbose = false;

    /// Optional observable. If non-null, distributed_ftlm() additionally
    /// computes <O>(beta) at every entry of `betas` and stores the result
    /// in `DistributedFtlmResult::O_expectation`. The observable Operator
    /// MUST act on the same Hilbert space as `op` (same n_bits / no
    /// symmetry-sector projection); callers are responsible. Internally
    /// adds one DistributedOperator constructor per group plus one
    /// matvec + m local zdotc + Allreduce per sample. Default null.
    std::shared_ptr<class ::Operator> observable_op = nullptr;
};

struct DistributedFtlmResult {
    /// Partition function estimate at each beta (same length as
    /// options.betas, or 1 if betas was empty).
    /// All ranks in the world communicator hold the same array on return.
    std::vector<double> Z;

    /// log Z(beta) at each beta, computed in shifted-exponent form so it
    /// stays finite at low T where Z itself underflows. Replicated on
    /// every rank. Same length as `Z`.
    ///
    /// Numerically: lnZ[b] = log(D/R) - beta * E_shift
    ///                       + log( sum_s sum_k w_k(s) exp(-beta(E_k(s) - E_shift)) ),
    /// where E_shift is the global minimum Lanczos Ritz value across all
    /// samples and ranks (Allreduce MPI_MIN). Use this instead of
    /// log(Z[b]) when beta * |E| is large (Z[b] may have underflowed to
    /// zero while lnZ[b] remains accurate).
    std::vector<double> lnZ;

    /// Global minimum Ritz eigenvalue across all samples / ranks. Used as
    /// the universal exponent shift so that exp(-beta(E_k - E_shift)) is
    /// O(1) for the lowest-energy sample. Useful for cross-sector
    /// recombination where every sector reports its own E_shift.
    double E_shift = 0.0;

    /// Observable expectation values at each beta. Empty unless
    /// options.observable_op was non-null. Replicated on every rank.
    std::vector<double> O_expectation;

    /// Number of samples actually reduced into Z (=options.n_samples,
    /// modulo dropped sample slots if n_groups doesn't divide evenly).
    int samples_used = 0;
};

/// Distributed FTLM. Collective on `world_comm` -- every rank must call.
/// Internally splits world_comm into options.n_groups subcommunicators
/// (MPI_Comm_split) and runs DistributedLanczos inside each.
DistributedFtlmResult distributed_ftlm(
    std::shared_ptr<class ::Operator> op,
    const DistributedFtlmOptions& options,
    MPI_Comm world_comm);

/// Symmetry-projected variant of `distributed_ftlm` (Phase D step 4).
/// Same trace-estimator algorithm, but every per-sample SpMV runs
/// inside ONE symmetry sector (`sector_idx`) of the underlying
/// `Operator` (and the optional `observable_op`). Internally builds
/// `DistributedSymmetryOperator` instances on the per-group
/// subcommunicator, then routes through `distributed_lanczos_symmetry`.
/// The returned `Z[b]` and `O_expectation[b]` are the contributions
/// from this sector alone (un-multiplied by any irrep-multiplicity
/// weight); the caller is responsible for aggregating across sectors
/// when reconstructing the full-space partition function.
///
/// Collective on `world_comm` -- every rank must call.
DistributedFtlmResult distributed_ftlm_symmetry(
    std::shared_ptr<class ::Operator> op,
    std::size_t sector_idx,
    const DistributedFtlmOptions& options,
    MPI_Comm world_comm);

}  // namespace ed::distributed

#endif  // WITH_MPI
