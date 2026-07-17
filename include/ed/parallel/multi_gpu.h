// =============================================================================
// include/ed/parallel/multi_gpu.h    (Phase 3c, runtime)
//
// Multi-GPU collectives over NCCL for distributed-memory ED. Replaces the
// detection-only `multi_gpu_stub.h` with an actual implementation surface
// that the future `distributed_lanczos_gpu` (Phase 3c #2) will consume.
//
// Design:
//
//   * `MultiGpuCommunicator` is a RAII wrapper around `ncclComm_t`. It is
//     constructed COLLECTIVELY across an `MPI_Comm`: rank 0 generates a
//     `ncclUniqueId`, broadcasts it via `MPI_Bcast`, every rank calls
//     `cudaSetDevice(local_device_index)` and `ncclCommInitRank` to join
//     the NCCL communicator. Destructor calls `ncclCommDestroy`.
//
//     Local device selection: by default we use a node-local sub-communicator
//     (`MPI_Comm_split_type(MPI_COMM_TYPE_SHARED)`) and bind rank `r` to
//     `cudaSetDevice(node_local_rank_r % visible_device_count)`. This is the
//     conventional 1 GPU per rank layout (4 ranks per node x 4 H100s on
//     rorqual `gpubase_*`). Override by passing an explicit device index to
//     the constructor for non-standard layouts (e.g. dev/test on a single
//     MIG slice with multiple ranks sharing the same device).
//
//   * Free-function collective wrappers cover the operations the distributed
//     Lanczos / FTLM kernels actually need:
//       - sum-reduction over `double` arrays (norm/dot reductions)
//       - sum-reduction over `std::complex<double>` arrays (zdotc;
//         implemented as a 2x-length `double` reduction since
//         (a+bi) + (c+di) = (a+c) + (b+d)i; equivalent because NCCL's
//         ncclSum is element-wise)
//       - broadcast over `double` (initial-vector scatter sub-step)
//
//     Each collective takes the `MultiGpuCommunicator&`, raw device
//     pointers, an element count, and an optional CUDA stream
//     (`cudaStream_t`, default 0 = legacy default stream).
//
//   * Pre-existing `nccl_compiled_in()` / `nccl_status_string()` (formerly
//     in `multi_gpu_stub.h`) remain inline and visible to every TU --
//     they are how callers detect whether the build supports multi-GPU at
//     all.
//
// Honest scope:
//   * If `ED_HAVE_NCCL` is NOT defined, this header still compiles but
//     every method on `MultiGpuCommunicator` and every free-function
//     collective throws `std::logic_error` at construction time. Callers
//     must guard with `nccl_compiled_in()` or rely on CMake `target_link`
//     ordering to keep these TUs out of non-NCCL builds.
//   * NCCL collectives that don't lower naturally to one of the typed
//     wrappers above (e.g. complex max-reduction, reduce-scatter on
//     non-power-of-two sizes) are deliberately omitted -- add them when
//     the calling code actually needs them.
//   * No `ed::distributed::DistributedGPUOperator` here yet -- the SpMV
//     halo exchange continues to ride MPI in Phase 3c stage 1 and the
//     NCCL collectives only handle the dot/norm reductions. GPU-Direct
//     RDMA halo is Phase 3c stage 2.
// =============================================================================

#pragma once

#include <complex>
#include <cstddef>
#include <cstdint>
#include <string>

// Forward-declare CUDA / NCCL handles to keep `<cuda_runtime.h>` and
// `<nccl.h>` out of the public header. The implementation TU pulls them
// in normally. `cudaStream_t` is a `struct CUstream_st*` and `ncclComm_t`
// is `struct ncclComm*`; we declare them as opaque pointer-to-struct so
// the typedefs match across host TUs without dragging in the runtime
// headers.
struct CUstream_st;
using cudaStream_t = CUstream_st*;
struct ncclComm;
using ncclComm_t = ncclComm*;

#ifdef WITH_MPI
#  include <mpi.h>
#endif

namespace ed::distributed::multi_gpu {

/// True iff this build was configured with NCCL discovered AND with
/// CUDA + MPI enabled. Safe to query from any TU (no NCCL headers
/// needed). Carries forward the `multi_gpu_stub.h` API so callers
/// migrating from the stub do not need to touch their detection guards.
inline constexpr bool nccl_compiled_in() noexcept {
#ifdef ED_HAVE_NCCL
    return true;
#else
    return false;
#endif
}

/// Human-readable status string for diagnostics / CLI banners.
std::string nccl_status_string();

/// Sentinel: pass to `MultiGpuCommunicator(MPI_Comm, int)` to request the
/// default node-local-rank-mod-num-visible-devices binding.
inline constexpr int kAutoDeviceIndex = -1;

#ifdef WITH_MPI

/**
 * RAII wrapper around `ncclComm_t`, built collectively from an MPI_Comm.
 *
 * Construction is collective: every rank in `mpi_comm` must call the
 * constructor. Internally we:
 *   1. `cudaSetDevice(device_index)` (or the auto-detected node-local
 *      device index).
 *   2. Generate a `ncclUniqueId` on rank 0, broadcast via `MPI_Bcast`.
 *   3. Call `ncclCommInitRank(&comm_, world_size, id, world_rank)`.
 *
 * Destruction calls `ncclCommDestroy(comm_)`. On failure the destructor
 * swallows the error after printing a diagnostic to stderr -- aborting
 * during stack unwind would mask the original exception that caused the
 * destructor to run.
 *
 * Thread safety: this object is *not* thread-safe. Treat it as a
 * per-rank singleton, just like an MPI_Comm.
 */
class MultiGpuCommunicator {
public:
    /**
     * Build a NCCL communicator over `mpi_comm`.
     *
     * @param mpi_comm     MPI communicator. Every rank in this comm joins
     *                     the NCCL group. Must be a valid, live comm.
     * @param device_index Local CUDA device index for this rank. Pass
     *                     `kAutoDeviceIndex` (default) to use
     *                     `node_local_rank % cudaGetDeviceCount()`.
     *                     Throws `std::runtime_error` if no CUDA device
     *                     is visible.
     *
     * Throws:
     *   * `std::logic_error` if `nccl_compiled_in()` is false.
     *   * `std::runtime_error` on cuda / nccl / MPI failure (with the
     *     vendor error string included in `what()`).
     */
    explicit MultiGpuCommunicator(MPI_Comm mpi_comm,
                                  int device_index = kAutoDeviceIndex);

    /// Destroys the underlying ncclComm_t (best-effort).
    ~MultiGpuCommunicator();

    MultiGpuCommunicator(const MultiGpuCommunicator&) = delete;
    MultiGpuCommunicator& operator=(const MultiGpuCommunicator&) = delete;
    MultiGpuCommunicator(MultiGpuCommunicator&&) noexcept;
    MultiGpuCommunicator& operator=(MultiGpuCommunicator&&) noexcept;

    /// NCCL handle (do NOT call `ncclCommDestroy` on this; we own it).
    ncclComm_t nccl() const noexcept { return comm_; }

    /// MPI communicator we were built from.
    MPI_Comm mpi_comm() const noexcept { return mpi_comm_; }

    /// World rank inside `mpi_comm`.
    int rank() const noexcept { return rank_; }

    /// World size of `mpi_comm`.
    int size() const noexcept { return size_; }

    /// CUDA device index this rank was bound to.
    int device() const noexcept { return device_index_; }

    /// `true` once the NCCL handle is valid (i.e., not moved-from / not
    /// destroyed). Cheap; safe to call after move.
    bool valid() const noexcept { return comm_ != nullptr; }

private:
    MPI_Comm    mpi_comm_      = MPI_COMM_NULL;
    int         rank_          = 0;
    int         size_          = 0;
    int         device_index_  = -1;
    ncclComm_t  comm_          = nullptr;  // owned
};

/**
 * sum-reduce `count` doubles in-place across all ranks of `comm`.
 *   d_buf must point to device memory accessible from `comm.device()`.
 *   On return every rank's d_buf holds the rank-summed result.
 * `stream` defaults to the legacy default stream; pass a real stream for
 * overlap with compute. Throws on NCCL failure.
 */
void all_reduce_sum_double(const MultiGpuCommunicator& comm,
                           double* d_buf, std::size_t count,
                           cudaStream_t stream = nullptr);

/**
 * sum-reduce `count` complex<double> in-place across all ranks of `comm`.
 *   The buffer must point to device memory; treated as 2*count doubles.
 *   Equivalent to `all_reduce_sum_double(d_buf, 2*count)` because complex
 *   sum is element-wise on (real, imag). The operand is alias-safe.
 */
void all_reduce_sum_complex_double(const MultiGpuCommunicator& comm,
                                   std::complex<double>* d_buf,
                                   std::size_t count,
                                   cudaStream_t stream = nullptr);

/**
 * Broadcast `count` doubles from `root` to every rank of `comm`. The
 * `d_buf` must point to device memory on every rank. On the root the
 * input is read; on non-root ranks the input is overwritten.
 */
void broadcast_double(const MultiGpuCommunicator& comm,
                      double* d_buf, std::size_t count, int root,
                      cudaStream_t stream = nullptr);

/// Block until the NCCL stream associated with `comm` is idle. Wraps
/// `cudaStreamSynchronize`; useful to mark a clear ordering point between
/// a NCCL collective and the next host action.
void synchronize_stream(cudaStream_t stream);

#endif  // WITH_MPI

}  // namespace ed::distributed::multi_gpu
