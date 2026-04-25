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
// Honest scope notes (carried into PHASE_3_SUMMARY.md):
//
//   * The first cut returns ONLY the partition-function trace estimate
//     Z(beta) = (1/n_samples) sum_s sum_k |v0_s . V_k_s|^2 exp(-beta*E_k_s).
//     Operator expectation values are wired in as a second pass: every
//     rank can replay the rank-local Krylov basis from its v0_s seed,
//     compute <V_k_s | O | V_l_s> with another DistributedOperator wrapping
//     O, then assemble the trace contribution. We document that hook here
//     and leave the actual O-side wiring for the `compute_o_estimator`
//     follow-up (Phase 3b #3.5).
//   * For honest 40 the inner DistributedOperator hits the slab-decomposition
//     limits documented in the DistributedOperator header. Honest-40 FTLM
//     therefore needs the same symmetry-aware slabbing fix that
//     DistributedOperator does, before this module is run on the real-world
//     spectrum target.
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
};

struct DistributedFtlmResult {
    /// Partition function estimate at each beta (same length as
    /// options.betas, or 1 if betas was empty).
    /// All ranks in the world communicator hold the same array on return.
    std::vector<double> Z;

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
