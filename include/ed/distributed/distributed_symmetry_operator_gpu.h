// =============================================================================
// include/ed/distributed/distributed_symmetry_operator_gpu.h    (Phase C)
//
// Multi-GPU sibling of `DistributedSymmetryOperator`. Wraps an existing
// CPU `DistributedSymmetryOperator` (which carries the orbit basis, the
// per-orbit projection norms, the LPT orbit partition, the sparse
// `H_q[i, j]` row slabs, AND the OrbitHaloPlan), and uploads the row
// slabs + halo plan to device for an entirely on-device SpMV in the
// symmetry-projected basis.
//
// What this class provides:
//   * `apply(d_x_local, d_y_local, stream)` -- one device-side pack
//     kernel + NCCL pairwise SendRecv halo exchange + one CSR-style
//     SpMV kernel. The CSR layout mirrors the CPU class exactly:
//     each non-zero is tagged `is_local` (column index points into
//     `d_x_local`) or `is_halo` (column index points into the device
//     recv buffer that the NCCL halo just filled).
//
// Honest scope:
//   * The orbit basis enumeration, projection-coefficient evaluation,
//     and the actual `H_q[i, j]` projection still happen on CPU at
//     construction time (in the wrapped `DistributedSymmetryOperator`).
//     This class only accelerates the *apply* hot path; construction
//     time is unchanged. That mirrors the CPU/GPU `DistributedOperator`
//     split in stage 3 of phase 3c.
//   * No on-device "phase factor" table is needed because the projection
//     has already been folded into the per-row coefficient list at
//     construction. The GPU kernel is just a sparse CSR matvec on
//     orbit indices.
//
// Build: compiled iff `WITH_MPI=ON && WITH_CUDA=ON && NCCL_FOUND`.
// Consumers guard the include with `#ifdef ED_HAVE_NCCL`.
// =============================================================================

#pragma once

#ifdef WITH_MPI

#include <ed/distributed/distributed_symmetry_operator.h>
#include <ed/distributed/multi_gpu.h>

#include <complex>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace ed::distributed {

class DistributedSymmetryOperatorGPU {
public:
    using Complex = std::complex<double>;

    /**
     * Build a GPU-resident wrapper around a CPU `DistributedSymmetryOperator`.
     * Collective on `gpu_comm.mpi_comm()` (which must equal `op->comm()`;
     * we check). On construction:
     *   * Sets the CUDA device to `gpu_comm.device()`.
     *   * Flattens the per-row CSR coordinates from the CPU class into
     *     contiguous device arrays: `d_row_offsets_[n_rows+1]`,
     *     `d_col_idx_[nnz]`, `d_is_local_[nnz]` (uint8),
     *     `d_coeff_re_[nnz]`, `d_coeff_im_[nnz]`.
     *   * Uploads the halo-plan metadata: `d_send_local_idx_[total_send]`
     *     (uint32 device-friendly cast of the CPU `std::size_t`), and
     *     keeps host-side counts/displs for NCCL pairwise SendRecv.
     *   * Allocates persistent device send/recv buffers sized to
     *     `total_send` and `total_recv`.
     *
     * Throws `std::logic_error` if NCCL is not compiled in or if
     * `gpu_comm.mpi_comm() != op->comm()`.
     */
    DistributedSymmetryOperatorGPU(
        std::shared_ptr<DistributedSymmetryOperator> op,
        const multi_gpu::MultiGpuCommunicator& gpu_comm);

    ~DistributedSymmetryOperatorGPU();

    DistributedSymmetryOperatorGPU(const DistributedSymmetryOperatorGPU&) = delete;
    DistributedSymmetryOperatorGPU& operator=(const DistributedSymmetryOperatorGPU&) = delete;
    DistributedSymmetryOperatorGPU(DistributedSymmetryOperatorGPU&&) = delete;
    DistributedSymmetryOperatorGPU& operator=(DistributedSymmetryOperatorGPU&&) = delete;

    /**
     * y_local = H_q * x_global   restricted to local rows of this rank.
     * Both buffers are device pointers; layout is `cuDoubleComplex`-
     * compatible. Sequence on `stream`:
     *   1. Pack kernel:  d_send_buf[k] = d_x_local[d_send_local_idx_[k]]
     *   2. NCCL pairwise SendRecv halo (one ncclGroupStart/End).
     *   3. CSR SpMV kernel: one thread per row; each non-zero loads
     *      from d_x_local or d_recv_buf depending on the is_local tag.
     *
     * Asynchronous on the stream; caller is responsible for
     * synchronisation if the host needs to read d_y_local.
     *
     * Collective on `gpu_comm`.
     */
    void apply(const multi_gpu::MultiGpuCommunicator& gpu_comm,
               const Complex* d_x_local, Complex* d_y_local,
               cudaStream_t stream = nullptr) const;

    /// Non-owning access to the underlying CPU class.
    const DistributedSymmetryOperator& cpu() const noexcept { return *op_; }

    /// Approximate device-resident bytes held by the row slabs + halo
    /// plan (excludes the persistent send/recv buffers reported below).
    std::size_t device_plan_bytes()   const noexcept { return plan_bytes_; }
    std::size_t device_buffer_bytes() const noexcept { return buf_bytes_; }

private:
    std::shared_ptr<DistributedSymmetryOperator> op_;
    int device_index_ = -1;

    int n_rows_   = 0;
    int nnz_      = 0;
    int local_n_  = 0;          // rows on this rank == cpu()->local_size()
    int total_send_ = 0;
    int total_recv_ = 0;

    // CSR row slab on device (column-major-style SoA).
    int*           d_row_offsets_ = nullptr;  // [n_rows + 1]
    int*           d_col_idx_     = nullptr;  // [nnz]
    std::uint8_t*  d_is_local_    = nullptr;  // [nnz]
    double*        d_coeff_re_    = nullptr;  // [nnz]
    double*        d_coeff_im_    = nullptr;  // [nnz]

    // Halo plan on device.
    int*           d_send_local_idx_ = nullptr;  // [total_send_]
    void*          d_send_buf_       = nullptr;  // [total_send_ * 16 B]
    void*          d_recv_buf_       = nullptr;  // [total_recv_ * 16 B]

    // Host-side per-peer counts/displs (NCCL takes int per-peer scalars).
    std::vector<int> send_counts_;
    std::vector<int> send_displs_;
    std::vector<int> recv_counts_;
    std::vector<int> recv_displs_;

    std::size_t plan_bytes_ = 0;
    std::size_t buf_bytes_  = 0;

    void release_();
};

}  // namespace ed::distributed

#endif  // WITH_MPI
