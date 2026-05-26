#pragma once
// =============================================================================
// include/ed/distributed/distributed_fixed_sz_operator.h
//
// Wave 2 of the "Unify all 16 matvec cells under apply_terms<BasisPolicy,
// Scalar, Backend>" plan (May 2026).
//
// Cell 2C of the (backend x Hilbert-mode) matrix: distributed-memory
// matrix-free SpMV on a Fixed-Sz sector. Mirror of
// ``ed::distributed::DistributedOperator`` (which handles cell 1C) but
// with bitstring-driven flip-pattern enumeration and a Fixed-Sz
// ``index_of`` lookup via the underlying ``FixedSzOperator``'s
// ``LinIndexTable``.
//
// Closes the historical Phase G workaround that routed Fixed-Sz +
// MPI through ``DistributedSymmetryOperator`` with a trivial 1-element
// symmetry group + popcount filter. The new path is a native cell,
// removes the orbit-replay overhead, and lets the MPI lane benefit
// from the same ``gather_row_basis`` kernel as the serial path.
// =============================================================================

#ifdef WITH_MPI
#include <mpi.h>

#include <complex>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include <ed/core/linear_operator.h>
#include <ed/core/sorted_uint64_index.h>
#include <ed/matvec/matvec.h>
#include <ed/matvec/memory_space.h>

class FixedSzOperator;

namespace ed::distributed {

class DistributedFixedSzOperator : public ed::LinearOperator {
public:
    using Complex = std::complex<double>;

    /**
     * Construct a distributed wrapper around a serial FixedSzOperator.
     *
     * Collective: every rank in ``comm`` must call this with a
     * FixedSzOperator that carries the SAME basis_states + term lists
     * (i.e., the same Hamiltonian definition at the same n_up). The
     * serial operator is kept alive by ``op_`` and never mutated.
     *
     * Builds the comm plan during construction. Each rank owns a
     * contiguous slab of FIXED-SZ ARRAY-INDICES ``[local_offset,
     * local_offset + local_n)`` over the C(N, n_up)-sized global
     * Fixed-Sz basis.
     */
    DistributedFixedSzOperator(std::shared_ptr<FixedSzOperator> op,
                               MPI_Comm comm);

    /**
     * y_local = (H * v_global)[local_offset, local_offset + local_n).
     *
     * Both v_local and y_local must be sized ``local_size()``. One
     * MPI_Alltoallv + one local SpMV via
     * ``ed::matvec::kernel::gather_row_basis<FixedSzBasisPolicy>``.
     */
    void apply(const Complex* v_local, Complex* y_local) const {
        apply_local_(v_local, y_local);
    }

    void apply(const ed::matvec::Complex* v_local,
               ed::matvec::Complex* y_local,
               std::size_t size) const override
    {
        check_size(size);
        apply_local_(reinterpret_cast<const Complex*>(v_local),
                     reinterpret_cast<Complex*>(y_local));
    }
    [[nodiscard]] std::size_t dim() const override { return static_cast<std::size_t>(local_n_); }
    [[nodiscard]] std::size_t global_dim() const override { return static_cast<std::size_t>(global_dim_); }
    [[nodiscard]] ed::matvec::MemorySpace memory_space() const override {
        return ed::matvec::MemorySpace::DistributedHost;
    }
    [[nodiscard]] bool is_hermitian() const override { return true; }
    [[nodiscard]] std::string description() const override {
        return "DistributedFixedSzOperator(local_n=" + std::to_string(local_n_)
            + ", global_dim=" + std::to_string(global_dim_)
            + ", nproc=" + std::to_string(size_) + ")";
    }

    [[nodiscard]] MatvecFn bind_mpi() const override {
        return [this](const ed::matvec::Complex* in,
                      ed::matvec::Complex* out, std::size_t n) {
            this->apply(in, out, n);
        };
    }
    [[nodiscard]] MatvecFn bind_cpu() const override {
        if (size_ > 1) {
            throw std::runtime_error(
                "DistributedFixedSzOperator: bind_cpu() is not supported on "
                "a multi-rank communicator. Use bind<MpiBackend>().");
        }
        return [this](const ed::matvec::Complex* in,
                      ed::matvec::Complex* out, std::size_t n) {
            this->apply(in, out, n);
        };
    }
    [[nodiscard]] MatvecFn bind_cuda() const override {
        throw std::runtime_error(
            "DistributedFixedSzOperator: bind_cuda() not supported. The "
            "MPI+CUDA lane for Fixed-Sz is Wave 5 of the unification plan.");
    }
                [[nodiscard]] MatvecFn bind_mpi_cuda() const override {
                    throw std::runtime_error(
                        "DistributedFixedSzOperator: bind_mpi_cuda() not supported "
                        "(Wave 5).");
                }

                // ----- Wave 6 (apply_batch fusion planning) ---------------------
                // The base ``LinearOperator::apply_batch`` default invokes
                // ``apply()`` per column, which is correct but pays one
                // ``MPI_Alltoallv`` per column. The Wave 6 follow-up fuses
                // ``batch`` halo exchanges into a single Alltoallv whose
                // element type is ``batch * MPI_C_DOUBLE_COMPLEX`` (or
                // a struct datatype, depending on alignment) and shares
                // the local SpMV scatter buffers across batch columns.
                // Wired into KPM-DOS Chebyshev moment evaluation and
                // FTLM block-Lanczos. Until then the default behaviour
                // is the correct (unfused) fallback.

                // ----- Wave 4: real-arithmetic overrides ----------------------
                // Real-Hermitian iff the underlying serial FixedSzOperator's
                // term lists are all real-coefficient. The bind_real_mpi
                // path uses the default complex shim; a native double halo
                // is a follow-up (Wave 4 stretch goal).
                [[nodiscard]] bool is_real_hermitian() const noexcept override;
                [[nodiscard]] RealMatvecFn bind_real_mpi() const override {
                    return bind_real_cpu();
                }

    // ----- Slab geometry --------------------------------------------------
    std::uint64_t local_offset() const noexcept { return local_offset_; }
    std::uint64_t local_size()   const noexcept { return local_n_; }
    int           rank()         const noexcept { return rank_; }
    int           comm_size()    const noexcept { return size_; }
    MPI_Comm      comm()         const noexcept { return comm_; }

    [[nodiscard]] ed::Geometry geometry() const override {
        ed::Geometry g;
        g.local_dim    = static_cast<std::size_t>(local_n_);
        g.global_dim   = global_dim_;
        g.local_offset = local_offset_;
        g.memory_space = ed::matvec::MemorySpace::DistributedHost;
        g.comm         = comm_;
        return g;
    }

    int owner_rank(std::uint64_t global_idx) const noexcept;
    std::uint64_t to_local(std::uint64_t global_idx) const noexcept {
        return global_idx - local_offset_;
    }

    std::shared_ptr<FixedSzOperator> serial_operator() const noexcept { return op_; }

    static void balanced_slab(std::uint64_t global_dim, int rank, int size,
                              std::uint64_t& out_offset,
                              std::uint64_t& out_n) noexcept;
    static int balanced_owner_rank(std::uint64_t global_idx,
                                   std::uint64_t global_dim,
                                   int size) noexcept;

private:
    void apply_local_(const Complex* v_local, Complex* y_local) const;
    void build_comm_pattern_();

    std::shared_ptr<FixedSzOperator> op_;
    MPI_Comm     comm_;
    int          rank_;
    int          size_;
    std::uint64_t global_dim_;
    std::uint64_t local_offset_;
    std::uint64_t local_n_;

    std::vector<std::uint64_t> rank_offsets_;

    std::vector<int> send_counts_;
    std::vector<int> send_displs_;
    std::vector<int> recv_counts_;
    std::vector<int> recv_displs_;
    int total_send_ = 0;
    int total_recv_ = 0;

    std::vector<int> send_local_idx_;

    ed::core::SortedUint64Index recv_lookup_;

    mutable std::vector<Complex> send_buf_;
    mutable std::vector<Complex> recv_buf_;
};

}  // namespace ed::distributed

#endif  // WITH_MPI
