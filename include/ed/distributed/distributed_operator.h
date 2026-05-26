// =============================================================================
// include/ed/distributed/distributed_operator.h
//
// Phase 3b #1: distributed-memory matrix-free SpMV (the load-bearing piece
// for "honest 40").
//
//   y_local = (H * v_global)[local_offset, local_offset + local_size)
//
// where every rank holds ONLY its slab of `v` and `y`. No rank ever
// materialises the full state vector -- that is the entire point of Phase
// 3b versus the trivial "Allgatherv the world to every rank" pattern.
//
// Design (one-time at construction):
//
//   1.  Decompose [0, global_dim) into balanced 1D row slabs:
//         rank r owns rows [offset_r, offset_r + n_r).
//   2.  Walk the operator's canonical SoA term storage
//       (Operator::terms_.{diag_one_body, offdiag_one_body, diag_two_body,
//       mixed_two_body, offdiag_two_body, three_body}) and extract
//       the unique set of column-flip patterns
//         { p in [0, 2^n_bits) : exists term s.t. col = row XOR p }
//       For Heisenberg/Hubbard-style spin Hamiltonians this is bounded by
//       O(N^2) patterns, each of popcount <= 3.
//   3.  For each local row r and each pattern p, the column c = r XOR p is
//       either local (in our slab) or remote. For every remote c, append
//       to a per-rank "I want this global index" list, then sort + dedupe.
//   4.  MPI_Alltoall to communicate counts, MPI_Alltoallv to communicate
//       the actual global indices we want from each rank. The receiving
//       side translates those into LOCAL slab offsets for the future send
//       phase of every apply().
//   5.  Build a `SortedUint64Index` (Phase 3a #5) `recv_lookup_` that maps
//       a remote global column -> its position in the contiguous recv
//       buffer. O(log) lookups during the SpMV inner loop.
//
// Per-apply (hot path):
//
//   1.  Pack send buffer: send_buf[k] = v_local[send_local_idx_[k]].
//   2.  Single MPI_Alltoallv exchanging Complex values (NOT indices --
//       indices were resolved at construction time).
//   3.  Matrix-free apply in GATHER form:
//         for each local row r in [local_offset, local_offset + local_n):
//           for each H term that touches r:
//             c = r XOR (term flip pattern)
//             read v[c] from local slab if c in [offset, offset + n_local),
//                                else from recv_buf via recv_lookup_.
//             y_local[r - local_offset] += <r|H|c> * v[c]
//
// Why GATHER and not SCATTER:
//
//   The matrix-free apply in `Operator::apply_optimized` runs in SCATTER
//   form (iterate input basis b, compute output target c, atomic-add into
//   y[c]). For distributed-memory SpMV with row-owned y, this would require
//   a sparse all-to-all of (index, value) pairs -- 24 bytes per nonzero
//   instead of 16. GATHER form, by contrast, exchanges only Complex values
//   (16 bytes/entry) and the resulting y_local is rank-private (no
//   cross-rank atomics needed).
//
//   GATHER for spin Hamiltonians is mathematically equivalent because H is
//   Hermitian and the term coefficients are real (or, for complex H,
//   conjugated correctly below). The condition rewrites are derived in
//   distributed_operator.cpp -- search for "Gather rewrite".
//
// Honest scope notes (carried into docs/history/PHASE_3_SUMMARY.md):
//
//   * The bit-flip-pattern enumeration is bounded by O(local_n_ *
//     |patterns|) raw column references; for Heisenberg with O(N) bonds
//     the recv-buffer per rank is bounded by min(local_n_ * O(N^2),
//     global_dim - local_n_).
//   * Honest-40 (dim = 2^40, e.g., spin-1/2 chain on 40 sites without
//     symmetry projection) needs ~7e10 local rows per rank with 16 ranks.
//     The raw 1D row-slab decomposition produces a recv-buffer per rank
//     bounded by ~15 * local_n_ at worst -- which, at 16 GB/Complex per
//     2^30, is too large. The PROPER fix is symmetry-aware slabbing: keep
//     each fixed-Sz orbit on a single rank, then the bit-flip patterns
//     within an orbit are local. We document this honestly in
//     docs/architecture/SCALING.md §6 / docs/history/PHASE_3_SUMMARY.md and leave it as the next
//     load-bearing item.
//   * For the bounded tests in this PR (N <= 16, np up to 4) the recv
//     buffer is small enough that the slab decomposition works directly
//     and validates the architecture end-to-end.
// =============================================================================

#pragma once

#ifdef WITH_MPI
#include <mpi.h>

#include <complex>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include <ed/core/linear_operator.h>
#include <ed/core/sorted_uint64_index.h>
#include <ed/matvec/matvec.h>          // MatVecOperator interface (Phase 2)
#include <ed/matvec/memory_space.h>    // DistributedHost tag

// Forward declaration to avoid pulling all of construct_ham.h into this header.
class Operator;

namespace ed::distributed {

class DistributedOperator : public ed::LinearOperator {
public:
    using Complex = std::complex<double>;

    /**
     * Construct a distributed wrapper around a serial Operator.
     *
     * Collective: every rank in `comm` must call this with an Operator that
     * carries the SAME term lists (i.e., the same Hamiltonian definition).
     * The serial Operator is kept alive by `op_` and never mutated by this
     * class; we only read its term storage.
     *
     * Builds the communication plan during construction. Throws on:
     *   - n_bits >= 64 (would overflow uint64_t basis indices)
     *   - any term coefficient with |imag| > 1e-15 in operator-flagged real
     *     mode -- caller may pre-validate via op->isReal() if needed.
     */
    DistributedOperator(std::shared_ptr<Operator> op, MPI_Comm comm);

    /**
     * y_local = (H * v_global)[local_offset, local_offset + local_size).
     *
     * v_local and y_local must be sized exactly ``local_size()``.
     * Internally performs ONE MPI_Alltoallv of Complex values + one
     * matrix-free local SpMV. Thread-safe within a single ``apply()``
     * call (the inner loop is parallelised with OpenMP).
     *
     * Both the 2-arg "legacy" entry and the 3-arg ``MatVecOperator``
     * override now route through the same private ``apply_local_``
     * implementation -- the 2-arg form is the back-compat fast path
     * (trusts caller-sized buffers), the 3-arg form is the polymorphic
     * surface and adds a defensive ``check_size`` first. Audit ref:
     * STRUCTURAL_AUDIT.md S1 #10.
     */
    void apply(const Complex* v_local, Complex* y_local) const {
        apply_local_(v_local, y_local);
    }

    // -------------------------------------------------------------------------
    // Matvec-unification Phase 2: MatVecOperator override.
    //
    // dim() reports the LOCAL (per-rank) length so solvers that walk
    // through the polymorphic interface size their work buffers
    // correctly. The orthogonal `global_dim()` accessor (already in
    // the original API and now a MatVecOperator virtual) reports the
    // full Hilbert-space length so reductions / norms can normalise.
    // Memory space is DistributedHost -- solvers must use the MPI
    // Backend variant for axpy/dot/norm; using a plain CpuBackend would
    // give wrong norms (it does not allreduce across ranks).
    // -------------------------------------------------------------------------
    void apply(const ed::matvec::Complex* v_local,
               ed::matvec::Complex* y_local,
               std::size_t size) const override
    {
        check_size(size);
        // ed::matvec::Complex is exactly std::complex<double>; identity cast.
        apply_local_(reinterpret_cast<const Complex*>(v_local),
                     reinterpret_cast<Complex*>(y_local));
    }
    [[nodiscard]] std::size_t dim() const override {
        return static_cast<std::size_t>(local_n_);
    }
    [[nodiscard]] std::size_t global_dim() const override {
        return static_cast<std::size_t>(global_dim_);
    }
    [[nodiscard]] ed::matvec::MemorySpace memory_space() const override {
        return ed::matvec::MemorySpace::DistributedHost;
    }
    [[nodiscard]] bool is_hermitian() const override { return true; }
    [[nodiscard]] std::string description() const override {
        return "DistributedOperator(local_n=" + std::to_string(local_n_)
            + ", global_dim=" + std::to_string(global_dim_)
            + ", nproc=" + std::to_string(size_) + ")";
    }

    // -------------------------------------------------------------------
    // bind_<Backend> overrides (Wave A2 -- Full unified-interface
    // collapse, May 2026).
    //
    // `apply()` already performs the MPI_Alltoallv halo exchange and
    // the local SpMV; the matching `bind_mpi` returns a callable that
    // wraps it. Kernels using `MpiBackend::all_reduce_sum` for norm /
    // dot reductions then compose with this matvec correctly --- the
    // matvec emits per-rank y_local slices and the reductions sum
    // across ranks. `bind_cpu` / `bind_cuda` are explicitly
    // unsupported: a single-rank caller through `bind_cpu` would
    // (correctly, on np = 1) just receive the same wrapper, but on
    // np > 1 the `apply()` body needs the MpiBackend's collectives to
    // make the per-rank vector lengths add up; routing through
    // `bind_cpu` invites silent miscalibration. We allow `bind_cpu`
    // when the operator's communicator is single-rank, matching the
    // "degenerate to plain Operator" semantics callers expect.
    // -------------------------------------------------------------------
    [[nodiscard]] MatvecFn bind_mpi() const override {
        return [this](const ed::matvec::Complex* in,
                      ed::matvec::Complex* out, std::size_t n) {
            this->apply(in, out, n);
        };
    }
    [[nodiscard]] MatvecFn bind_cpu() const override {
        if (size_ > 1) {
            throw std::runtime_error(
                "DistributedOperator: bind_cpu() is not supported on a "
                "multi-rank communicator. Use bind<MpiBackend>() so "
                "axpy/dot reductions allreduce across ranks.");
        }
        return [this](const ed::matvec::Complex* in,
                      ed::matvec::Complex* out, std::size_t n) {
            this->apply(in, out, n);
        };
    }
    [[nodiscard]] MatvecFn bind_cuda() const override {
        throw std::runtime_error(
            "DistributedOperator: bind_cuda() is not supported -- this "
            "operator is host-resident. For the MPI+CUDA lane wrap a "
            "ed::distributed::DistributedGPUOperator around this "
            "instance and pair it with an MpiCudaBackend (see "
            "ed::WithMpiCudaBackend in select_backend.h).");
    }
    [[nodiscard]] MatvecFn bind_mpi_cuda() const override {
        throw std::runtime_error(
            "DistributedOperator: bind_mpi_cuda() is not supported -- "
            "this operator is host-resident. Use "
            "ed::distributed::DistributedGPUOperator for the MPI+CUDA "
            "lane.");
    }

    // -------------------------------------------------------------------
    // Wave 4 (May 2026, "Unify all 16 matvec cells" plan): real-arithmetic
    // overrides. The Hamiltonian is real-Hermitian when the underlying
    // serial ``Operator`` reports ``isReal() == true`` (the distributed
    // wrapper doesn't add imaginary couplings of its own). The Wave 4
    // ``bind_real_mpi`` here uses the default complex shim path; a
    // native double-precision halo (MPI_DOUBLE instead of
    // MPI_C_DOUBLE_COMPLEX) lands as a follow-up so the bandwidth halves
    // for real-Hermitian workloads.
    // -------------------------------------------------------------------
    [[nodiscard]] bool is_real_hermitian() const noexcept override;
    [[nodiscard]] RealMatvecFn bind_real_mpi() const override {
        return bind_real_cpu();
    }

    // -------------------------------------------------------------------------
    // Slab geometry (queryable on every rank).
    // -------------------------------------------------------------------------
    std::uint64_t local_offset() const noexcept { return local_offset_; }
    std::uint64_t local_size()   const noexcept { return local_n_; }
    int           rank()         const noexcept { return rank_; }
    int           comm_size()    const noexcept { return size_; }
    MPI_Comm      comm()         const noexcept { return comm_; }

    // Phase 3.1 of the Minimalist ED Collapse (May 2026): expose the
    // distributed slab geometry to `ed::select_backend` / orchestrators.
    [[nodiscard]] ed::Geometry geometry() const override {
        ed::Geometry g;
        g.local_dim    = static_cast<std::size_t>(local_n_);
        g.global_dim   = global_dim_;
        g.local_offset = local_offset_;
        g.memory_space = ed::matvec::MemorySpace::DistributedHost;
        g.comm         = comm_;
        return g;
    }

    /// Owner rank for a given GLOBAL row/column index, under this object's
    /// balanced 1D decomposition.
    int owner_rank(std::uint64_t global_idx) const noexcept;

    /// Convert a global index to a local index (only meaningful when
    /// owner_rank(global_idx) == this->rank()).
    std::uint64_t to_local(std::uint64_t global_idx) const noexcept {
        return global_idx - local_offset_;
    }

    /// Sum of approximate per-rank memory bytes held by the comm plan.
    /// Exposed for diagnostics / logging. Excludes v_local / y_local
    /// themselves, which the caller owns.
    std::size_t plan_bytes() const noexcept;

    /// Number of unique bit-flip patterns extracted from the operator
    /// (informational; relevant for capacity planning / honest scope notes).
    std::size_t num_flip_patterns() const noexcept { return flip_patterns_.size(); }

    // -------------------------------------------------------------------------
    // Comm plan view (Phase 3c stage 3). Exposed so that
    // `DistributedGPUOperator` can mirror the same MPI_Alltoallv-derived
    // plan onto the device and run a pack + NCCL pairwise SendRecv halo
    // without reconstructing it from the operator definition. All pointers
    // are valid for the lifetime of `*this`; do NOT cache them past a move
    // of the parent (this object is non-copyable but moves invalidate
    // pointers as usual).
    // -------------------------------------------------------------------------
    struct CommPlanView {
        const int*           send_counts   = nullptr;
        const int*           send_displs   = nullptr;
        const int*           recv_counts   = nullptr;
        const int*           recv_displs   = nullptr;
        const int*           send_local_idx = nullptr;  // length total_send
        const std::uint64_t* recv_keys     = nullptr;   // sorted; length total_recv
        const std::size_t*   recv_values   = nullptr;   // length total_recv
        int total_send = 0;
        int total_recv = 0;
        int comm_size  = 0;
    };
    CommPlanView comm_plan_view() const noexcept {
        CommPlanView v;
        v.send_counts    = send_counts_.data();
        v.send_displs    = send_displs_.data();
        v.recv_counts    = recv_counts_.data();
        v.recv_displs    = recv_displs_.data();
        v.send_local_idx = send_local_idx_.data();
        v.recv_keys      = recv_lookup_.empty()
                              ? nullptr : recv_lookup_.keys().data();
        v.recv_values    = recv_lookup_.empty()
                              ? nullptr : recv_lookup_.values().data();
        v.total_send = total_send_;
        v.total_recv = total_recv_;
        v.comm_size  = size_;
        return v;
    }

    /// Underlying serial operator. The GPU operator pulls its SoA term
    /// tables from this. Lifetime tied to `*this`.
    std::shared_ptr<Operator> serial_operator() const noexcept { return op_; }

    // -------------------------------------------------------------------------
    // Static helpers (collective-friendly: each rank can call
    // independently, given the same args, and get the same answers).
    // -------------------------------------------------------------------------

    /// Balanced 1D row slab for given (global_dim, rank, size).
    /// First (global_dim mod size) ranks get ceil(global_dim/size) rows,
    /// the rest get floor(global_dim/size) rows.
    static void balanced_slab(std::uint64_t global_dim, int rank, int size,
                              std::uint64_t& out_offset,
                              std::uint64_t& out_n) noexcept;

    /// Owner rank for a given global index under balanced_slab().
    static int balanced_owner_rank(std::uint64_t global_idx,
                                   std::uint64_t global_dim,
                                   int size) noexcept;

private:
    /**
     * Canonical SpMV implementation -- one body, two public surfaces.
     * Called from both ``apply(v, y)`` (legacy 2-arg, back-compat) and
     * ``apply(v, y, size)`` (MatVecOperator override). Trusts that the
     * caller-supplied buffers are sized ``local_n_``; the 3-arg form
     * runs ``check_size`` before delegating.
     */
    void apply_local_(const Complex* v_local, Complex* y_local) const;

    void extract_flip_patterns_();
    void build_comm_pattern_();

    std::shared_ptr<Operator> op_;
    MPI_Comm     comm_;
    int          rank_;
    int          size_;
    std::uint64_t global_dim_;
    std::uint64_t local_offset_;
    std::uint64_t local_n_;

    /// rank_offsets_[r] = global index of the first row owned by rank r.
    /// rank_offsets_[size_] = global_dim_. Used by owner_rank() for O(log)
    /// lookup via std::upper_bound (much faster than a per-call recompute
    /// of balanced_slab() for every call).
    std::vector<std::uint64_t> rank_offsets_;

    /// Unique column-flip patterns from H. ALWAYS contains 0 (diagonal).
    std::vector<std::uint64_t> flip_patterns_;

    /// MPI_Alltoallv counts (Complex elements, not bytes).
    std::vector<int> send_counts_;
    std::vector<int> send_displs_;
    std::vector<int> recv_counts_;
    std::vector<int> recv_displs_;
    int total_send_ = 0;
    int total_recv_ = 0;

    /// For each entry in the send buffer (length total_send_), the LOCAL
    /// index in v_local to copy from. Filled at construction time.
    std::vector<int> send_local_idx_;

    /// recv_lookup_[global_col] = offset into recv_buf (built once,
    /// queried during every apply()'s inner loop).
    ed::core::SortedUint64Index recv_lookup_;

    // -------------------------------------------------------------------------
    // Phase 8: reusable halo-exchange staging buffers.
    //
    // The pre-Phase-8 implementation allocated `send_buf` and `recv_buf` on
    // every call to apply(). For Lanczos / TPQ where apply() is called
    // hundreds of times that is two heap round-trips per matvec (and two
    // full-buffer first-touch faults if the allocator handed back fresh
    // pages). The buffers are sized exactly by the comm plan and never
    // change once build_comm_pattern_ has run, so we keep them as instance
    // members and only mutate their contents on each apply().
    //
    // mutable: apply() must remain const because callers (Lanczos kernels,
    // distributed_tpq, distributed_ftlm) pass `const DistributedOperator&`
    // and the SpMV is logically read-only on the operator. The buffers are
    // pure scratch -- conceptually they belong to the apply() call, not to
    // the matrix definition.
    // -------------------------------------------------------------------------
    mutable std::vector<Complex> send_buf_;
    mutable std::vector<Complex> recv_buf_;
};

}  // namespace ed::distributed

#endif  // WITH_MPI
