// =============================================================================
// include/ed/distributed/distributed_symmetry_operator.h    (Phase 3b #7, stage 2)
//
// `DistributedSymmetryOperator` -- the distributed-memory analogue of
// `Operator` projected onto a single irreducible representation (sector)
// of a site-permutation symmetry group, with:
//
//   * Orbit-respecting row partition: one row per orbit, every orbit
//     lives entirely on one rank, balanced by LPT greedy
//     (`OrbitPartition` from `balanced_orbit_slab`).
//   * Orbit-aware halo exchange: each rank pulls amplitudes for the
//     remote orbits its local rows actually touch (`OrbitHaloPlan`).
//   * Sparse SpMV in the symmetry-projected basis: at construction
//     each rank materialises only the rows it owns of the projected
//     matrix `H_q[i, j] = <i,q|H|j,q>` (in the orbit basis); at apply
//     time it does ONE `MPI_Alltoallv` halo + one local CSR-style
//     sweep.
//
// What this class does NOT do (honest scope):
//
//   * Lazy / on-the-fly enumeration. The orbit basis and the projected
//     matrix elements are built collectively at construction time by
//     walking the full Hilbert space (size `2^N`), so this primitive
//     is bounded by the same `N` ceiling as the serial
//     `generateSymmetrizedBasis*` path. The win over the serial path
//     is that:
//       (a) The full sparse projected matrix is partitioned across
//           ranks (each rank stores only its row slab), so memory at
//           apply time scales as `O(local_nnz)` instead of `O(global_nnz)`.
//       (b) The hot-path apply is `O(1)` MPI exchange + `O(local_nnz)`
//           inner loop instead of an HDF5 round-trip per matrix-vector
//           product.
//     Lazy enumeration (so construction itself becomes distributed
//     and the `2^N` ceiling lifts) is the next refactor; the apply
//     path here is invariant under that change.
//
// Sector convention:
//
//   The ctor takes `sector_index` indexing into
//   `op->symmetry_info.sectors[]`. Sector quantum numbers and phase
//   factors come straight from `SectorMetadata`; the projection
//   formula matches `Operator::createSymmetrizedVector`:
//
//       ~|i> = sum_g chi_q(g)* |g(rep_i)>
//       |i>  = ~|i> / sqrt(N_i),  N_i = ||~|i>||^2
//
//   Orbits whose `N_i` is below `kZeroNormTolerance` are filtered out
//   (they contribute nothing to this sector).  This is the same
//   "phantom orbit" filter that the serial HDF5 path applies after
//   building each orbit's symmetrised vector.
//
// Lockdown:
//
//   `tests/unit/test_distributed_symmetry_operator.cpp` cross-checks
//   `apply()` vs an independently-built dense reference of the
//   symmetry-projected matrix on N=4, N=6 Heisenberg chains with
//   translation symmetry, at np in {1, 2, 4}.
// =============================================================================

#pragma once

#ifdef WITH_MPI
#include <mpi.h>

#include <complex>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include <ed/distributed/orbit_halo_plan.h>
#include <ed/distributed/orbit_partition.h>

class Operator;

namespace ed::distributed {

class DistributedSymmetryOperator {
public:
    using Complex = std::complex<double>;

    /// Orbits whose unnormalised projection norm is below this threshold
    /// are dropped from the symmetry-projected basis (they encode the
    /// "phantom orbit" cancellations that arise when the orbit
    /// stabiliser's character on the chosen sector is non-trivial).
    static constexpr double kZeroNormTolerance = 1e-10;

    /// Matrix elements with magnitude below this threshold are dropped
    /// during construction (they are numerical noise from the dense
    /// projection reduction, which is bounded by O(|G|^2 * dim *
    /// machine_eps)).
    static constexpr double kSparsityTolerance = 1e-12;

    /**
     * Construct the distributed symmetry operator.
     *
     * @param op            Serial Operator carrying the term storage AND
     *                      a populated `symmetry_info` (max_clique,
     *                      power_representation, sectors). Must be the
     *                      same on every rank in `comm`. Kept alive by
     *                      `op_` and never mutated.
     * @param sector_index  Index into `op->symmetry_info.sectors[]`.
     * @param comm          MPI communicator. Must satisfy
     *                      `comm != MPI_COMM_NULL`.
     *
     * Collective: every rank in `comm` must call. The ctor:
     *   1. Walks the full Hilbert space (size `2^N`) to enumerate
     *      orbits + the unnormalised projection coefficients
     *      `~|i>_b = sum_{g: g(rep_i) = b} chi_q(g)*`.
     *   2. Filters out zero-norm orbits.
     *   3. Builds an `OrbitPartition` (LPT greedy on orbit sizes) so
     *      every orbit has a unique owner rank.
     *   4. For each orbit `j`, computes `H * ~|j>` densely on the full
     *      Hilbert space via `op->apply`, then projects onto every
     *      orbit `i` and keeps `(j, H_ij)` if `i` is locally owned
     *      and `|H_ij| > kSparsityTolerance`.
     *   5. Builds an `OrbitHaloPlan` for the set of remote `j` indices
     *      that any local row references.
     *
     * Throws on:
     *   - null `op`
     *   - missing or malformed `symmetry_info`
     *   - `sector_index` out of range
     *   - `n_bits >= 64`
     *   - `comm == MPI_COMM_NULL`
     */
    DistributedSymmetryOperator(std::shared_ptr<Operator> op,
                                std::size_t sector_index,
                                MPI_Comm comm);

    /**
     * Distributed symmetry-projected SpMV.
     *
     *   y_local = (H_q * x_global)[local rows of this rank]
     *
     * `x_local` and `y_local` are sized exactly `local_size()` and
     * indexed by the rank-local orbit index (i.e., the k-th local
     * orbit corresponds to global orbit
     * `partition().rank_orbits[rank()][k]`).
     *
     * Performs ONE `MPI_Alltoallv` halo exchange + one rank-local
     * CSR-style sparse-matvec. Collective on `comm()`.
     */
    void apply(const Complex* x_local, Complex* y_local) const;

    // -------------------------------------------------------------------------
    // Geometry / diagnostics
    // -------------------------------------------------------------------------
    std::uint64_t global_dim()   const noexcept;  // # nonzero-norm orbits
    std::uint64_t local_size()   const noexcept;  // # orbits on this rank
    std::uint64_t local_offset() const noexcept;  // rank-major prefix offset
    int           rank()         const noexcept { return rank_; }
    int           comm_size()    const noexcept { return size_; }
    MPI_Comm      comm()         const noexcept { return comm_; }
    std::size_t   sector_index() const noexcept { return sector_index_; }

    /// LPT-built orbit partition (deterministic, replicated on every rank).
    const OrbitPartition& partition() const noexcept { return partition_; }

    /// Orbit-aware halo plan (built once at construction). nullptr means
    /// no remote orbits are needed (np=1 or fully local SpMV).
    const OrbitHaloPlan* halo_plan() const noexcept { return halo_plan_.get(); }

    /// Canonical (lex-min) representative of orbit `i` in the full Hilbert
    /// space, indexed by orbit id (0..n_orbits()).  Replicated on every rank.
    const std::vector<std::uint64_t>& orbit_reps() const noexcept {
        return orbit_reps_;
    }

    /// Unnormalised projection norm `N_i = ||~|i>||^2` for orbit `i`.
    /// Always > kZeroNormTolerance for orbits in the kept basis.
    const std::vector<double>& orbit_norms_sq() const noexcept {
        return orbit_norms_sq_;
    }

    /// Number of full-Hilbert states in orbit `i` (equals |G| / |stabiliser|).
    const std::vector<std::uint64_t>& orbit_sizes() const noexcept {
        return orbit_sizes_;
    }

    /// Per-rank sparse-matrix nonzero count for the local row slab.
    /// Useful for load-balance diagnostics (the LPT greedy balances
    /// orbit sizes, not nonzeros, so a downstream rebalance step might
    /// want this signal).
    std::size_t local_nnz() const noexcept;

    /// Per-row CSR-style accessors for the locally-owned row slab.
    /// `csr_row_col_idx()[r][k]` is either an index into a length-
    /// `local_size()` amplitude vector (when `csr_row_is_local()[r][k]
    /// != 0`) or an index into the halo recv buffer of `halo_plan()`
    /// (when 0). `csr_row_coeff()[r][k]` is the matrix element
    /// `H_q[i_local(r), j]`. These are exposed so a GPU mirror
    /// (`DistributedSymmetryOperatorGPU`) can flatten the row slab
    /// into device-resident SoA arrays at construction time.
    const std::vector<std::vector<std::size_t>>&  csr_row_col_idx() const noexcept {
        return row_col_idx_;
    }
    const std::vector<std::vector<std::uint8_t>>& csr_row_is_local() const noexcept {
        return row_is_local_;
    }
    const std::vector<std::vector<Complex>>&      csr_row_coeff() const noexcept {
        return row_coeff_;
    }

private:
    std::shared_ptr<Operator> op_;
    MPI_Comm                  comm_   = MPI_COMM_NULL;
    int                       rank_   = 0;
    int                       size_   = 0;
    std::size_t               sector_index_ = 0;

    OrbitPartition                partition_;
    std::unique_ptr<OrbitHaloPlan> halo_plan_;

    /// `orbit_reps_[i]` = lex-min state in orbit i (full-Hilbert basis).
    std::vector<std::uint64_t> orbit_reps_;

    /// `orbit_norms_sq_[i]` = N_i = ||~|i>||^2 (unnormalised).
    std::vector<double> orbit_norms_sq_;

    /// `orbit_sizes_[i]` = |orbit i| in the full Hilbert space.
    std::vector<std::uint64_t> orbit_sizes_;

    /// Map global orbit_id -> halo index in the recv buffer of `halo_plan_`,
    /// for orbits NOT owned by this rank that some local row touches.
    /// `halo_index_[orbit_id] = k` means the orbit's amplitude lands in
    /// `recv_buf[k]` after `halo_plan_->exchange(...)`. Sentinel value
    /// `kHaloMissing` for unused orbits.
    static constexpr std::size_t kHaloMissing = static_cast<std::size_t>(-1);
    std::vector<std::size_t> halo_index_;

    /// One row per local orbit. `row_col_idx_` holds either a local
    /// index (when `row_is_local_[r][k] == true`, indexing into
    /// `x_local`) or a halo index (when false, indexing into the recv
    /// buffer of `halo_plan_`).  `row_coeff_[r][k]` is `H_ij`.
    std::vector<std::vector<std::size_t>> row_col_idx_;
    std::vector<std::vector<std::uint8_t>> row_is_local_;
    std::vector<std::vector<Complex>>     row_coeff_;

    // -------------------------------------------------------------------------
    // Phase 8: persistent halo recv buffer.
    //
    // Same rationale as DistributedOperator::send_buf_/recv_buf_ -- the size
    // is fixed by the OrbitHaloPlan after construction, so we allocate once
    // and reuse on every apply(). Marked mutable because apply() is const.
    // -------------------------------------------------------------------------
    mutable std::vector<Complex> halo_buf_;
};

}  // namespace ed::distributed

#endif  // WITH_MPI
