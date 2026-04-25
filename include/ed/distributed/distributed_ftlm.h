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
// Honest scope notes (carried into PHASE_3_SUMMARY.md):
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

}  // namespace ed::distributed

#endif  // WITH_MPI
