// =============================================================================
// include/ed/distributed/distributed_tpq.h
//
// Phase 3b #8: distributed-memory canonical Thermal Pure Quantum (cTPQ)
// state evolution.
//
// Convention (Sugiura & Shimizu 2012 / Hyuga et al. 2014):
//
//   |psi(beta)> = e^{-beta H / 2} |r>  /  || e^{-beta H / 2} |r> ||
//
// where |r> is a unit Gaussian random vector. Thermal expectation:
//
//   <O>_thermal(beta) ~ <psi(beta) | O | psi(beta)>  +  O(1/sqrt(D))
//
// We propagate in small `delta_beta` substeps using a Taylor expansion of
// the imaginary-time propagator:
//
//   e^{-(delta_beta/2) H} |psi> ~ sum_{n=0}^{taylor_order}
//                                   (-delta_beta / 2)^n H^n |psi> / n!
//
// followed by an L2 renormalisation. Each H^n |psi> term is computed as
// a sequence of n DistributedOperator matvecs (one MPI_Alltoallv per
// matvec), local zaxpy accumulation, and one MPI_Allreduce for the
// normalisation norm at the end of each step.
//
// Two-level parallelism mirrors distributed_ftlm:
//   * outer = MPI groups across samples (one group per random vector)
//   * inner = MPI ranks within a group, each owning a slab of |psi>
//
// At each measure_beta, the per-group rank-0 sums energy / variance into
// a world-level Allreduce, which is then divided by `n_samples` so all
// ranks return the same sample-averaged result.
//
// Honest scope notes (carried into docs/history/PHASE_3_SUMMARY.md):
//
//   * The Taylor expansion is stable up to ||delta_beta * H|| of order 1.
//     For honest 40 the full operator norm scales like O(N), so practical
//     delta_beta drops as 1/N. We do NOT auto-tune; the caller picks
//     `delta_beta` and `taylor_order` and is responsible for stability.
//   * Variance is computed as <psi|H^2|psi> - <psi|H|psi>^2, with an extra
//     matvec per measure_beta. Set `compute_variance = false` to skip.
//   * Like distributed_ftlm, the inner DistributedOperator hits the
//     1D slab-decomposition limit at honest 40; symmetry-aware slabbing
//     remains the next load-bearing item.
// =============================================================================

#pragma once

#ifdef WITH_MPI

#include <ed/distributed/distributed_operator.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace ed::distributed {

struct DistributedTpqOptions {
    /// Number of random samples (R in the J&P notation; here used as the
    /// number of independent TPQ trajectories whose observables we
    /// average). Distributed across `n_groups` groups round-robin.
    int n_samples = 8;

    /// Number of MPI groups across samples. Must divide world_comm size.
    int n_groups = 1;

    /// Inverse-temperature substep for the Taylor expansion of
    /// e^{-(delta_beta/2) H}. Caller chooses this; smaller = more stable
    /// but more matvecs.
    double delta_beta = 0.05;

    /// Truncation order of the Taylor series for e^{-(delta_beta/2) H}.
    /// 30 is a safe default for ||delta_beta H|| <~ 1.
    std::uint64_t taylor_order = 30;

    /// Inverse temperatures at which to record E(beta) (and optionally
    /// variance). Must be sorted ascending. Each consecutive pair is
    /// bridged by floor((b_{i+1} - b_i)/delta_beta) full substeps plus
    /// one short final substep so beta lands exactly on the target.
    /// Empty -> {1.0}.
    std::vector<double> betas;

    /// Random seed offset; sample s uses seed_offset + s.
    unsigned long seed_offset = 0UL;

    /// If true, also compute E2 = <psi|H^2|psi> at every measure_beta and
    /// expose `variance = E2 - E^2` in the result. Costs one extra
    /// DistributedOperator matvec per (sample, measure_beta).
    bool compute_variance = false;

    /// If true, rank 0 prints per-sample, per-beta diagnostics.
    bool verbose = false;
};

struct DistributedTpqResult {
    /// Z-weighted sample-averaged <H>(beta) at each entry of options.betas.
    /// Replicated on every rank in world_comm. Defined as
    ///   <H>(beta) = (sum_r w_r(beta) E_r(beta)) / (sum_r w_r(beta))
    /// where w_r(beta) = <r|e^{-beta H}|r> is the per-sample weight that
    /// the cTPQ propagator implicitly accumulates as the product of
    /// pre-renormalisation step squared-norms. This is the J&P-consistent
    /// canonical estimator and is what makes cross-sector recombination
    /// Z(beta) = sum_alpha Z_alpha(beta), <H>(beta) = sum_alpha Z_alpha
    /// <H>_alpha / sum_alpha Z_alpha exact in the limit of many samples.
    std::vector<double> energy;

    /// Z-weighted sample-averaged variance <H^2>(beta) - <H>(beta)^2.
    /// Empty unless options.compute_variance was true.
    std::vector<double> variance;

    /// Per-beta partition function estimate
    ///   Z(beta) = (D / R) * sum_r w_r(beta)
    /// where D is the (sector) dimension exposed by the operator and R is
    /// the number of samples actually reduced. Replicated on every rank.
    /// This is the field the Phase-H Python aggregator consumes via
    /// the HDF5 dataset `/Z`.
    std::vector<double> Z;

    /// log Z(beta) computed via logsumexp of the per-sample log-weights;
    /// numerically robust at large |beta| where Z itself may underflow.
    /// Same length as `energy`.
    std::vector<double> lnZ;

    /// Number of samples actually reduced.
    int samples_used = 0;
};

/// Distributed canonical TPQ. Collective on `world_comm` -- every rank
/// must call. Splits world_comm into options.n_groups subcommunicators
/// (MPI_Comm_split), distributes samples round-robin, propagates each
/// rank-local |psi(beta)> via Taylor-truncated e^{-(delta/2) H}, and
/// reduces sample-averaged observables across the world.
DistributedTpqResult distributed_tpq(
    std::shared_ptr<class ::Operator> op,
    const DistributedTpqOptions& options,
    MPI_Comm world_comm);

/// Symmetry-projected variant of `distributed_tpq` (Phase E).
/// Same canonical-TPQ algorithm, but every per-sample SpMV runs inside
/// ONE symmetry sector (`sector_idx`) of the underlying `Operator`.
/// Internally builds a `DistributedSymmetryOperator` on the per-group
/// subcommunicator. The returned `energy[b]` is the sample-averaged
/// `<H>(beta)` measured WITHIN this sector (not weighted by the
/// sector's contribution to the full-space partition function); the
/// caller is responsible for FTLM-style aggregation across sectors
/// when reconstructing full-space thermal observables.
///
/// Collective on `world_comm` -- every rank must call.
DistributedTpqResult distributed_tpq_symmetry(
    std::shared_ptr<class ::Operator> op,
    std::size_t sector_idx,
    const DistributedTpqOptions& options,
    MPI_Comm world_comm);

}  // namespace ed::distributed

#endif  // WITH_MPI
