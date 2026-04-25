// =============================================================================
// include/ed/distributed/orbit_partition.h    (Phase 3b #7, stage 1)
//
// "Orbit-respecting" row partition for distributed-memory ED with full
// lattice symmetry. The Phase 3b #1 `DistributedOperator` partitions the
// *unsymmetrised* basis by `balanced_slab(global_dim, rank, size)` -- one
// contiguous row range per rank. That works at the bounded-N test scale
// but blows up the `MPI_Alltoallv` halo at "honest 40" with full
// point-group + Sz: a single matvec touches every orbit element of the
// representative, so the row-slab boundaries cut orbits in half and the
// per-rank halo grows like the slab itself.
//
// The proper fix is to partition the *sector basis indices* (one entry per
// orbit) such that:
//
//   1. Every orbit lives entirely on one rank.
//   2. The per-rank weight (sum of orbit weights) is balanced within a
//      bounded factor of the mean.
//
// `balanced_orbit_slab(orbit_weights, n_ranks)` returns one such partition
// using a deterministic LPT (Longest Processing Time first) greedy:
//
//   * Sort orbits by weight, descending.
//   * Iterate orbits in that order, append each to the rank with the
//     currently smallest accumulated weight.
//   * For ties, prefer the lower-indexed rank.
//
// LPT achieves a 4/3 - 1/(3n_ranks) approximation of the optimal makespan
// (Graham 1969); for our orbit-size distributions (typically dominated by
// many same-sized orbits with O(1) heavy outliers from short-period
// stabiliser subgroups) it is very close to optimal in practice.
//
// IMPORTANT: this stage 1 helper does NOT yet wire orbit partitions into
// `DistributedOperator`. That is stage 2 (a follow-up landing). What this
// header gives us is the load-balancing primitive plus the data layout
// that the future `DistributedSymmetryOperator` will consume:
//
//   * `OrbitPartition::orbit_owner[i]`    -- which rank owns orbit i
//   * `OrbitPartition::rank_orbits[r]`    -- list of orbit indices on rank r
//                                             (in ascending order so a
//                                             SortedUint64Index stays valid)
//   * `OrbitPartition::rank_offsets[r]`   -- prefix-sum of rank_orbit_count
//                                             so callers can map (rank, k)
//                                             to a contiguous global
//                                             "sector basis index".
//
// Lockdown:
//   * `tests/unit/test_orbit_partition.cpp` exercises the helper on
//     pathological inputs (empty, single orbit, all equal, geometrically
//     skewed) and checks the LPT approximation bound.
// =============================================================================

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace ed::distributed {

/**
 * Result of an orbit-aware row partition. Every field is replicable on every
 * rank because the partition is a deterministic function of `(orbit_weights,
 * n_ranks)`. No MPI required to build (the test suite covers it without
 * launching MPI), but designed so the future `DistributedSymmetryOperator`
 * constructor can take this struct directly and skip its own decomposition.
 */
struct OrbitPartition {
    /// Number of ranks the partition was built for.
    int n_ranks = 0;

    /// `orbit_owner[i]` = rank that owns orbit i. Length = number of orbits.
    /// Values in `[0, n_ranks)`.
    std::vector<int> orbit_owner;

    /// `rank_orbits[r]` = sorted-ascending list of orbit indices owned by
    /// rank r. `concat(rank_orbits[0..n_ranks-1])` is a permutation of
    /// `[0, n_orbits)`. Stored as the partitioning side-product so the
    /// `DistributedSymmetryOperator` can build its rank-local sector basis
    /// without re-walking `orbit_owner`.
    std::vector<std::vector<std::size_t>> rank_orbits;

    /// `rank_offsets[r]` = sum of `rank_orbits[0..r-1].size()`, with
    /// `rank_offsets[n_ranks] = n_orbits`. Acts as the "global sector basis
    /// index" prefix-sum: orbit `rank_orbits[r][k]` lands at global index
    /// `rank_offsets[r] + k` in the rank-major ordering. This is the
    /// quantity that downstream Lanczos / FTLM / TPQ kernels will treat as
    /// the analogue of `local_offset_` from `DistributedOperator`.
    std::vector<std::size_t> rank_offsets;

    /// `rank_weights[r]` = sum of `orbit_weights[orbit_index]` for every
    /// orbit owned by rank r. Exposed for diagnostics / load-balance
    /// quality checks.
    std::vector<std::uint64_t> rank_weights;

    /// Per-orbit reverse lookup: `orbit_local_index[i] = k` means orbit i
    /// is the k-th orbit on its owner rank (i.e. `rank_orbits[owner][k] ==
    /// i`). Built alongside the partition so the future
    /// `DistributedSymmetryOperator::owner_local_index()` is O(1) instead
    /// of O(log local_size). Length = number of orbits.
    std::vector<std::size_t> orbit_local_index;

    /// Convenience accessors.
    std::size_t n_orbits() const noexcept { return orbit_owner.size(); }
    std::size_t local_size(int r) const noexcept {
        return (r >= 0 && r < n_ranks) ? rank_orbits[r].size() : 0;
    }
    std::uint64_t local_weight(int r) const noexcept {
        return (r >= 0 && r < n_ranks) ? rank_weights[r] : 0;
    }

    // -------------------------------------------------------------------------
    // DistributedOperator-shaped accessors (Phase 3b #7 stage 2 prep).
    //
    // These mirror the API surface of `ed::distributed::DistributedOperator`
    // (owner_rank / local_offset / local_size) so a future
    // `DistributedSymmetryOperator` can drop into call-sites that today
    // expect contiguous-row geometry but will be ported to orbit-aware
    // geometry. Every accessor is O(1) given the auxiliary tables above.
    // -------------------------------------------------------------------------

    /// Owner rank of orbit `i`. Returns -1 if `i` is out of range (caller
    /// must not pass an invalid orbit id; the negative return is a
    /// debug-friendly cliff rather than a hard contract).
    int owner_rank(std::size_t orbit_id) const noexcept {
        if (orbit_id >= orbit_owner.size()) return -1;
        return orbit_owner[orbit_id];
    }

    /// k-th position of orbit `i` on its owner rank. Equivalent to
    /// `std::lower_bound(rank_orbits[owner].begin(), ..., i)` via the
    /// precomputed `orbit_local_index` table.
    std::size_t owner_local_index(std::size_t orbit_id) const noexcept {
        return (orbit_id < orbit_local_index.size())
                   ? orbit_local_index[orbit_id]
                   : 0;
    }

    /// Global rank-major orbit index: orbit `i` lands at
    /// `rank_offsets[owner] + owner_local_index(i)` in the future
    /// `DistributedSymmetryOperator`'s rank-concatenated basis ordering.
    /// Useful for callers that need the analogue of
    /// `DistributedOperator::local_offset() + local_pos`.
    std::size_t global_rank_major_index(std::size_t orbit_id) const noexcept {
        const int r = owner_rank(orbit_id);
        if (r < 0) return 0;
        return rank_offsets[static_cast<std::size_t>(r)]
             + owner_local_index(orbit_id);
    }
};

/**
 * Build a load-balanced orbit partition using the Longest-Processing-Time
 * (LPT) greedy: at each step, the next-heaviest orbit is appended to the
 * rank with the currently smallest accumulated weight.
 *
 * @param orbit_weights  per-orbit weight (orbit size, or projected work
 *                       estimate). Length = number of orbits. Must be
 *                       non-negative; zero-weight orbits are still
 *                       assigned (to the lightest rank) so that
 *                       downstream code can address every orbit.
 * @param n_ranks        number of MPI ranks. Must be > 0.
 * @return               filled `OrbitPartition`. Throws
 *                       `std::invalid_argument` on n_ranks <= 0.
 *
 * Determinism: same inputs always produce the same output, byte-for-byte
 * (no RNG, stable tie-breaks on ascending rank index then ascending orbit
 * index). This is mandatory for collective MPI use -- every rank must
 * compute the same partition independently.
 *
 * Approximation quality:
 *   max(rank_weight) <= (4/3 - 1/(3 n_ranks)) * optimal_makespan
 * (Graham 1969). For our orbit-weight distributions (mostly equal
 * |G|-sized orbits with a few heavier ones from non-trivial stabilisers)
 * the LPT solution is typically within < 1% of optimal.
 */
OrbitPartition balanced_orbit_slab(
    const std::vector<std::uint64_t>& orbit_weights,
    int n_ranks);

/**
 * Compute the worst-case load-balance imbalance ratio
 *   max_r rank_weights[r] / mean(rank_weights)
 * Returns 1.0 for a perfectly balanced partition. Useful for regression
 * tests and diagnostic output. Returns 1.0 if total weight is 0.
 */
double load_imbalance(const OrbitPartition& part) noexcept;

}  // namespace ed::distributed
