// =============================================================================
// include/ed/distributed/distributed_gpu_operator.h    (Phase 3c stage 3)
//
// Fully GPU-resident sibling of `DistributedOperator`:
//
//   * Halo exchange runs through `ncclGroupStart() -> ncclSend / ncclRecv ->
//     ncclGroupEnd()` on device buffers. No host staging in the hot path.
//   * SpMV is a CUDA kernel that reads from the on-device input slab
//     `d_v_local` and the on-device receive buffer `d_recv_buf`, with
//     binary-search lookups via the same SortedUint64Index keys/values
//     mirrored to device.
//   * Term storage (diag_one_body, offdiag_one_body, diag_two_body,
//     mixed_two_body, offdiag_two_body) is uploaded once at construction
//     and then read inside the SpMV kernel.
//
// Lockdown: validated against `DistributedOperator::apply` in
// `tests/unit/test_distributed_gpu_operator.cpp` for random complex
// vectors at np = 1, 2, 4 on Heisenberg N = 4..8. Three-body terms are
// out of scope for stage 3 and rejected at construction time -- the
// distributed code paths we currently exercise (Heisenberg / Hubbard) do
// not use them.
//
// Compiled iff:
//     WITH_MPI
//   && WITH_CUDA
//   && NCCL_FOUND   (ED_HAVE_NCCL = 1)
//
// On builds missing any of those, the constructor throws
// `std::logic_error`; the header still compiles cleanly so callers can
// guard via `ed::distributed::multi_gpu::nccl_compiled_in()`.
//
// Honest scope notes:
//   * `DistributedGPUOperator` re-uses the same balanced 1D row-slab
//     decomposition as the parent `DistributedOperator`. Symmetry-aware
//     orbit slabbing (Phase 3b stage 2 proper) is independent and lives
//     in `ed::distributed::orbit_partition` / `orbit_halo_plan`. When
//     that lands the GPU operator will accept either decomposition via a
//     small adapter, but the apply() loop body is identical.
//   * Pinned-host scratch for D2H / H2D transfers is a future
//     optimisation (would matter once the SpMV is fast enough that PCIe
//     turnaround is the bottleneck again, which is not the case at the
//     scales we currently test).
// =============================================================================

#pragma once

#ifdef WITH_MPI

#include <ed/distributed/distributed_operator.h>
#include <ed/distributed/multi_gpu.h>

#include <complex>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace ed::distributed {

class DistributedGPUOperator {
public:
    using Complex = std::complex<double>;

    /**
     * Build a GPU-resident wrapper around an existing CPU
     * `DistributedOperator`. Collective on `gpu_comm.mpi_comm()` (which
     * must equal `op->comm()`; we check).
     *
     * On construction:
     *   * Sets the CUDA device to `gpu_comm.device()` (idempotent if the
     *     caller already did so).
     *   * Allocates and uploads the comm plan: `d_send_local_idx_`,
     *     `d_recv_keys_`, `d_recv_values_`, and host-resident copies of
     *     send/recv counts/displs (kept on host because NCCL
     *     send/recv-pair calls accept host-side ints).
     *   * Allocates persistent device send/recv buffers sized to
     *     `total_send` and `total_recv`.
     *   * Uploads the SoA term tables (diag_one_body, offdiag_one_body,
     *     diag_two_body, mixed_two_body, offdiag_two_body). 3-body terms
     *     trigger `std::invalid_argument`.
     *   * Throws `std::logic_error` if `nccl_compiled_in()` is false.
     */
    DistributedGPUOperator(std::shared_ptr<DistributedOperator> op,
                           const multi_gpu::MultiGpuCommunicator& gpu_comm);

    ~DistributedGPUOperator();

    DistributedGPUOperator(const DistributedGPUOperator&) = delete;
    DistributedGPUOperator& operator=(const DistributedGPUOperator&) = delete;
    // Move is non-trivial because of CUDA pointer ownership; defer it
    // until a callsite needs it.
    DistributedGPUOperator(DistributedGPUOperator&&) = delete;
    DistributedGPUOperator& operator=(DistributedGPUOperator&&) = delete;

    /**
     * y_local = (H * v_global)[local_offset, local_offset + local_size).
     * Both buffers are device pointers (`std::complex<double>*` pointing
     * to GPU memory; layout-compatible with `cuDoubleComplex`).
     *
     * Sequence on `stream`:
     *   1. Pack kernel: `d_send_buf[k] = d_v_local[d_send_local_idx_[k]]`.
     *   2. Halo: NCCL group with pairwise ncclSend / ncclRecv per peer
     *      with non-zero count (treated as 2*N float64).
     *   3. SpMV kernel: GATHER form, one thread per local row, walking
     *      the SoA term tables. Off-rank columns binary-search into
     *      d_recv_keys_ / d_recv_values_.
     *
     * `stream` defaults to the legacy default stream (== nullptr). This
     * function is asynchronous on the stream; the caller is responsible
     * for stream synchronisation if it needs to read y_local from the
     * host.
     *
     * Collective on `gpu_comm` (every rank in the comm must call apply
     * with matching v/y buffer sizes inside the same logical step).
     */
    void apply(const multi_gpu::MultiGpuCommunicator& gpu_comm,
               const Complex* d_v_local, Complex* d_y_local,
               cudaStream_t stream = nullptr) const;

    /// Non-owning access to the underlying CPU DistributedOperator (for
    /// queries like `local_size()` / `local_offset()`).
    const DistributedOperator& cpu() const noexcept { return *op_; }

    /// Approximate device-resident bytes held by the comm plan + term
    /// tables (excludes the persistent send/recv buffers which scale
    /// with total_send/total_recv and are reported separately).
    std::size_t device_plan_bytes() const noexcept { return plan_bytes_; }
    std::size_t device_buffer_bytes() const noexcept { return buf_bytes_; }

private:
    std::shared_ptr<DistributedOperator> op_;
    int      device_index_ = -1;

    // Comm plan: send/recv counts/displs stay on host (fed to ncclSend /
    // ncclRecv as scalars per peer). The indirection arrays live on
    // device.
    std::vector<int> send_counts_;
    std::vector<int> send_displs_;
    std::vector<int> recv_counts_;
    std::vector<int> recv_displs_;
    int total_send_ = 0;
    int total_recv_ = 0;

    int*            d_send_local_idx_ = nullptr;  // [total_send_]
    std::uint64_t*  d_recv_keys_      = nullptr;  // [total_recv_]
    std::size_t*    d_recv_values_    = nullptr;  // [total_recv_]
    void*           d_send_buf_       = nullptr;  // [total_send_ * 16 B]
    void*           d_recv_buf_       = nullptr;  // [total_recv_ * 16 B]

    // Term tables on device. Each "table" is parallel arrays of plain
    // POD scalars so the SpMV kernel is simple and warp-friendly.
    int n_diag_one_       = 0;
    int n_offdiag_one_    = 0;
    int n_diag_two_       = 0;
    int n_mixed_two_      = 0;
    int n_offdiag_two_    = 0;

    std::uint64_t*  d_d1_site_  = nullptr;
    double*         d_d1_re_    = nullptr;
    double*         d_d1_im_    = nullptr;

    std::uint64_t*  d_o1_site_  = nullptr;
    std::uint8_t*   d_o1_op_    = nullptr;
    double*         d_o1_re_    = nullptr;
    double*         d_o1_im_    = nullptr;

    std::uint64_t*  d_d2_s1_    = nullptr;
    std::uint64_t*  d_d2_s2_    = nullptr;
    double*         d_d2_re_    = nullptr;
    double*         d_d2_im_    = nullptr;

    std::uint64_t*  d_m2_sz_    = nullptr;
    std::uint64_t*  d_m2_flip_  = nullptr;
    std::uint8_t*   d_m2_op_    = nullptr;
    double*         d_m2_re_    = nullptr;
    double*         d_m2_im_    = nullptr;

    std::uint64_t*  d_o2_s1_    = nullptr;
    std::uint64_t*  d_o2_s2_    = nullptr;
    std::uint8_t*   d_o2_op1_   = nullptr;
    std::uint8_t*   d_o2_op2_   = nullptr;
    double*         d_o2_re_    = nullptr;
    double*         d_o2_im_    = nullptr;

    // Cached scalars used inside the SpMV kernel.
    double  spin_      = 0.5;
    double  spin_sq_   = 0.25;
    std::uint64_t local_offset_ = 0;
    std::uint64_t local_n_      = 0;

    std::size_t plan_bytes_ = 0;
    std::size_t buf_bytes_  = 0;

    void upload_comm_plan_();
    void upload_term_tables_();
    void release_();
};

}  // namespace ed::distributed

#endif  // WITH_MPI
