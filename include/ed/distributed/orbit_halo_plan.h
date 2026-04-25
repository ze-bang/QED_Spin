// =============================================================================
// include/ed/distributed/orbit_halo_plan.h    (Phase 3b #7, stage 2 prep)
//
// Orbit-aware halo plan: the load-bearing primitive that the future
// `DistributedSymmetryOperator` will use to keep its `MPI_Alltoallv`
// halo size proportional to the number of orbit boundaries crossed
// by H, not to `|patterns| * local_n` (which is what the current
// `DistributedOperator` does on the unsymmetrised basis).
//
// Inputs:
//
//   * `OrbitPartition`  -- which rank owns each orbit (Phase 3b #7
//     stage 1 LPT greedy result; `n_ranks` and `orbit_owner[]` tell
//     us where every orbit lives).
//   * `needed_orbits`   -- the "I need amplitude(orbit_id) from its
//     owner rank during my next H * v" set. This is computed once
//     by walking the operator's term storage and projecting every
//     `(rep_i, rep_j)` matrix-element into a "rank rank(rep_i)
//     needs orbit rep_j" edge, then dedup'd. The future
//     `DistributedSymmetryOperator` constructor will call this
//     helper with the result of that walk.
//
// Outputs:
//
//   * `OrbitHaloPlan::recv_counts[r]`   -- # orbits this rank needs
//     from rank r (0 for `r == this_rank`).
//   * `OrbitHaloPlan::recv_displs[r]`   -- prefix-sum of recv_counts
//     so a single contiguous recv buffer can be carved up.
//   * `OrbitHaloPlan::recv_orbit_id[]`  -- the global orbit id at
//     each recv-buffer slot (rank-major).
//   * `OrbitHaloPlan::send_counts[r]`   -- # orbits this rank must
//     send to rank r (computed via a single `MPI_Alltoall` of the
//     recv_counts array under the contract that
//     "what rank A says it wants from rank B == what rank B owes
//     rank A").
//   * `OrbitHaloPlan::send_displs[r]`   -- prefix-sum of send_counts.
//   * `OrbitHaloPlan::send_orbit_id[]`  -- the global orbit id at
//     each send-buffer slot (rank-major).
//   * `OrbitHaloPlan::send_local_idx[]` -- where in the rank-local
//     orbit-amplitude vector each send slot reads from (i.e., the
//     `OrbitPartition::owner_local_index(orbit_id)` of the orbit at
//     `send_orbit_id[k]`). This is what `apply()`'s pack phase
//     consumes.
//
// Contract:
//
//   * Built collectively (every rank in `comm` must call). The MPI
//     traffic at construction is one `MPI_Alltoall` (size=int) and
//     one `MPI_Alltoallv` (size=uint64_t ids). Both bounded by
//     O(n_ranks) and O(total halo orbits) respectively -- much
//     cheaper than per-iteration halo cost.
//   * Once built, `exchange(local_amplitudes, halo_amplitudes)`
//     does ONE `MPI_Alltoallv` of complex<double> values per call.
//   * Size estimate: `plan_bytes()` returns the total memory held
//     by the recv/send arrays for diagnostics.
//
// Honest scope:
//
//   This header gives us the halo *machinery*. It does NOT yet wire
//   the inner SpMV (which still needs the orbit-projected matrix
//   elements from the serial symmetry-aware operator). Wiring that
//   in is the next stage. The unit test
//   `tests/unit/test_orbit_halo_plan.cpp` exercises the plan on a
//   synthetic 1-D ring of orbits where each orbit talks to its two
//   neighbours -- enough to lock down counts/displs/exchange
//   semantics across np ∈ {1, 2, 4}.
// =============================================================================

#pragma once

#ifdef WITH_MPI

#include <ed/distributed/orbit_partition.h>

#include <complex>
#include <cstddef>
#include <cstdint>
#include <vector>

#include <mpi.h>

namespace ed::distributed {

class OrbitHaloPlan {
public:
    using Complex = std::complex<double>;

    /**
     * Build a halo plan.
     *
     * @param part            Orbit partition (every rank computed it
     *                        deterministically from
     *                        `balanced_orbit_slab`). Must satisfy
     *                        `part.n_ranks == comm_size`.
     * @param needed_orbits   This rank's "I need this orbit's
     *                        amplitude during the next SpMV" set.
     *                        May contain locally-owned orbits
     *                        (silently filtered out -- callers
     *                        often build the set without first
     *                        checking ownership). Duplicates are
     *                        deduplicated.
     * @param comm            Communicator (same as the future
     *                        `DistributedSymmetryOperator`'s comm).
     *
     * Throws `std::invalid_argument` on `part.n_ranks != comm_size`
     * or `comm == MPI_COMM_NULL`.
     * Throws `std::runtime_error` on MPI failure.
     */
    OrbitHaloPlan(const OrbitPartition& part,
                  const std::vector<std::size_t>& needed_orbits,
                  MPI_Comm comm);

    /**
     * Exchange local-orbit amplitudes -> halo-orbit amplitudes.
     *
     *   local_amplitudes  : length `part.local_size(rank)`. Indexed
     *                       by `OrbitPartition::owner_local_index`
     *                       of the locally-owned orbit ids.
     *   halo_amplitudes   : length `recv_total()`. After return, slot
     *                       k holds the amplitude of the orbit
     *                       `recv_orbit_id_[k]`, which is now in
     *                       this rank's working set for the SpMV.
     *
     * Single `MPI_Alltoallv` of complex<double>. Collective.
     */
    void exchange(const Complex* local_amplitudes,
                  Complex* halo_amplitudes) const;

    // Diagnostics / capacity accessors.
    int           rank()       const noexcept { return rank_; }
    int           comm_size()  const noexcept { return size_; }
    MPI_Comm      comm()       const noexcept { return comm_; }
    std::size_t   recv_total() const noexcept { return recv_orbit_id_.size(); }
    std::size_t   send_total() const noexcept { return send_orbit_id_.size(); }

    /// Per-rank send / recv counts (length = comm_size).
    const std::vector<int>& send_counts() const noexcept { return send_counts_; }
    const std::vector<int>& recv_counts() const noexcept { return recv_counts_; }

    /// Globally-indexed orbit ids associated with each slot in the
    /// recv / send buffers. `recv_orbit_id_[k]` is the orbit id
    /// whose amplitude lands in slot k of the halo buffer.
    const std::vector<std::uint64_t>& recv_orbit_id() const noexcept {
        return recv_orbit_id_;
    }
    const std::vector<std::uint64_t>& send_orbit_id() const noexcept {
        return send_orbit_id_;
    }

    /// For each send slot, the local orbit index in the rank-local
    /// amplitude vector to read from. Useful if a caller wants to
    /// pre-pack outside `exchange()`.
    const std::vector<std::size_t>& send_local_idx() const noexcept {
        return send_local_idx_;
    }

    /// Approximate per-rank memory held by the plan (bytes; excludes
    /// the local/halo amplitude buffers which the caller owns).
    std::size_t plan_bytes() const noexcept;

private:
    MPI_Comm                  comm_;
    int                       rank_  = 0;
    int                       size_  = 0;

    std::vector<int>          recv_counts_;
    std::vector<int>          recv_displs_;
    std::vector<std::uint64_t> recv_orbit_id_;

    std::vector<int>          send_counts_;
    std::vector<int>          send_displs_;
    std::vector<std::uint64_t> send_orbit_id_;
    std::vector<std::size_t>  send_local_idx_;
};

}  // namespace ed::distributed

#endif  // WITH_MPI
