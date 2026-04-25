// =============================================================================
// src/distributed/multi_gpu.cu    (Phase 3c, runtime)
//
// NCCL collective wrappers for distributed-memory ED. See
// include/ed/distributed/multi_gpu.h for the design and the honest scope.
//
// Compiled iff:
//     WITH_MPI       (CMake)
//   && WITH_CUDA      (CMake; pulls in CUDA::cudart, etc.)
//   && NCCL_FOUND     (CMake; pulls in nccl_LIBRARIES, ED_HAVE_NCCL)
//
// On builds without all three, this TU is excluded from `ed_distributed_gpu`
// and the corresponding header methods throw `std::logic_error` so
// downstream code that #includes the header still compiles cleanly.
// =============================================================================

#ifdef ED_HAVE_NCCL

#include <ed/distributed/multi_gpu.h>

#include <cuda_runtime.h>
#include <nccl.h>

#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

namespace ed::distributed::multi_gpu {

namespace {

[[noreturn]] void throw_cuda(cudaError_t e, const char* what) {
    std::ostringstream os;
    os << "MultiGpuCommunicator: CUDA error in " << what << ": "
       << cudaGetErrorString(e) << " (code=" << static_cast<int>(e) << ")";
    throw std::runtime_error(os.str());
}

[[noreturn]] void throw_nccl(ncclResult_t e, const char* what) {
    std::ostringstream os;
    os << "MultiGpuCommunicator: NCCL error in " << what << ": "
       << ncclGetErrorString(e) << " (code=" << static_cast<int>(e) << ")";
    throw std::runtime_error(os.str());
}

inline void check_cuda(cudaError_t e, const char* what) {
    if (e != cudaSuccess) throw_cuda(e, what);
}

inline void check_nccl(ncclResult_t e, const char* what) {
    if (e != ncclSuccess) throw_nccl(e, what);
}

#ifdef WITH_MPI
// Pick the device index for this rank.
//   * If the caller passed an explicit `device_index >= 0`, use it.
//   * Otherwise, build a node-local sub-communicator via
//     MPI_Comm_split_type(MPI_COMM_TYPE_SHARED) so we can find this
//     rank's "node-local rank id" (0..N-1 where N = ranks on the same
//     physical node), then bind to (node_local_rank % visible_devices).
int resolve_device_index(MPI_Comm mpi_comm, int requested,
                         int /*world_rank*/) {
    int n_devices = 0;
    cudaError_t status = cudaGetDeviceCount(&n_devices);
    if (status != cudaSuccess || n_devices < 1) {
        throw std::runtime_error(
            "MultiGpuCommunicator: cudaGetDeviceCount returned "
            + std::to_string(n_devices) + " devices (status="
            + cudaGetErrorString(status) + "). NCCL requires at least one"
            " visible CUDA device per MPI rank.");
    }
    if (requested >= 0) {
        if (requested >= n_devices) {
            throw std::runtime_error(
                "MultiGpuCommunicator: requested device_index="
                + std::to_string(requested) + " >= visible device count "
                + std::to_string(n_devices));
        }
        return requested;
    }
    MPI_Comm node_comm = MPI_COMM_NULL;
    int err = MPI_Comm_split_type(mpi_comm, MPI_COMM_TYPE_SHARED, 0,
                                  MPI_INFO_NULL, &node_comm);
    if (err != MPI_SUCCESS || node_comm == MPI_COMM_NULL) {
        throw std::runtime_error(
            "MultiGpuCommunicator: MPI_Comm_split_type(SHARED) failed");
    }
    int node_rank = 0;
    MPI_Comm_rank(node_comm, &node_rank);
    MPI_Comm_free(&node_comm);
    return node_rank % n_devices;
}
#endif  // WITH_MPI

}  // namespace

std::string nccl_status_string() {
    int nccl_major = 0, nccl_minor = 0, nccl_patch = 0;
    int nccl_version = 0;
    ncclGetVersion(&nccl_version);
    nccl_major = nccl_version / 10000;
    nccl_minor = (nccl_version / 100) % 100;
    nccl_patch = nccl_version % 100;
    std::ostringstream os;
    os << "ed::distributed::multi_gpu: NCCL "
       << nccl_major << "." << nccl_minor << "." << nccl_patch
       << " linked";
    return os.str();
}

#ifdef WITH_MPI

MultiGpuCommunicator::MultiGpuCommunicator(MPI_Comm mpi_comm,
                                           int device_index)
    : mpi_comm_(mpi_comm) {
    if (mpi_comm == MPI_COMM_NULL) {
        throw std::invalid_argument(
            "MultiGpuCommunicator: mpi_comm == MPI_COMM_NULL");
    }
    MPI_Comm_rank(mpi_comm_, &rank_);
    MPI_Comm_size(mpi_comm_, &size_);

    device_index_ = resolve_device_index(mpi_comm_, device_index, rank_);
    check_cuda(cudaSetDevice(device_index_), "cudaSetDevice");

    // Generate + broadcast unique ID. NCCL_UNIQUE_ID_BYTES is 128 in
    // current NCCL releases; we read it from the type to stay
    // version-independent.
    ncclUniqueId id;
    if (rank_ == 0) {
        check_nccl(ncclGetUniqueId(&id), "ncclGetUniqueId");
    }
    int err = MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, mpi_comm_);
    if (err != MPI_SUCCESS) {
        throw std::runtime_error(
            "MultiGpuCommunicator: MPI_Bcast(unique_id) failed");
    }

    // Wrap the per-rank init in a NCCL group so the call sequence is
    // valid even when multiple comms are created back-to-back. Single
    // init is fine without the group, but the explicit grouping mirrors
    // NCCL examples and protects against future multi-comm extensions.
    check_nccl(ncclGroupStart(), "ncclGroupStart");
    ncclResult_t init = ncclCommInitRank(&comm_, size_, id, rank_);
    ncclResult_t end  = ncclGroupEnd();
    check_nccl(init, "ncclCommInitRank");
    check_nccl(end,  "ncclGroupEnd");

    if (!comm_) {
        throw std::runtime_error(
            "MultiGpuCommunicator: ncclCommInitRank returned null comm");
    }
}

MultiGpuCommunicator::~MultiGpuCommunicator() {
    if (comm_ != nullptr) {
        ncclResult_t e = ncclCommDestroy(comm_);
        if (e != ncclSuccess) {
            // Don't throw from a dtor; just warn.
            std::cerr << "MultiGpuCommunicator: ncclCommDestroy failed: "
                      << ncclGetErrorString(e) << " (rank " << rank_
                      << ", device " << device_index_ << ")" << std::endl;
        }
        comm_ = nullptr;
    }
}

MultiGpuCommunicator::MultiGpuCommunicator(MultiGpuCommunicator&& o) noexcept
    : mpi_comm_(o.mpi_comm_),
      rank_(o.rank_),
      size_(o.size_),
      device_index_(o.device_index_),
      comm_(o.comm_) {
    o.comm_     = nullptr;
    o.mpi_comm_ = MPI_COMM_NULL;
}

MultiGpuCommunicator& MultiGpuCommunicator::operator=(
    MultiGpuCommunicator&& o) noexcept {
    if (this != &o) {
        if (comm_ != nullptr) {
            (void)ncclCommDestroy(comm_);
        }
        mpi_comm_     = o.mpi_comm_;
        rank_         = o.rank_;
        size_         = o.size_;
        device_index_ = o.device_index_;
        comm_         = o.comm_;
        o.comm_       = nullptr;
        o.mpi_comm_   = MPI_COMM_NULL;
    }
    return *this;
}

void all_reduce_sum_double(const MultiGpuCommunicator& comm,
                           double* d_buf, std::size_t count,
                           cudaStream_t stream) {
    if (!comm.valid()) {
        throw std::logic_error(
            "all_reduce_sum_double: comm is not valid (moved-from?)");
    }
    if (d_buf == nullptr || count == 0) return;
    check_nccl(ncclAllReduce(d_buf, d_buf, count, ncclFloat64, ncclSum,
                             comm.nccl(), stream),
               "ncclAllReduce(double, sum)");
}

void all_reduce_sum_complex_double(const MultiGpuCommunicator& comm,
                                   std::complex<double>* d_buf,
                                   std::size_t count,
                                   cudaStream_t stream) {
    // (a+bi) + (c+di) = (a+c) + (b+d)i; element-wise sum of doubles.
    if (!comm.valid()) {
        throw std::logic_error(
            "all_reduce_sum_complex_double: comm is not valid");
    }
    if (d_buf == nullptr || count == 0) return;
    auto* as_double = reinterpret_cast<double*>(d_buf);
    check_nccl(ncclAllReduce(as_double, as_double, count * 2,
                             ncclFloat64, ncclSum, comm.nccl(), stream),
               "ncclAllReduce(complex<double>, sum)");
}

void broadcast_double(const MultiGpuCommunicator& comm,
                      double* d_buf, std::size_t count, int root,
                      cudaStream_t stream) {
    if (!comm.valid()) {
        throw std::logic_error("broadcast_double: comm is not valid");
    }
    if (root < 0 || root >= comm.size()) {
        throw std::invalid_argument(
            "broadcast_double: root=" + std::to_string(root)
            + " out of range [0, " + std::to_string(comm.size()) + ")");
    }
    if (d_buf == nullptr || count == 0) return;
    check_nccl(ncclBroadcast(d_buf, d_buf, count, ncclFloat64, root,
                             comm.nccl(), stream),
               "ncclBroadcast(double)");
}

void synchronize_stream(cudaStream_t stream) {
    cudaError_t e = cudaStreamSynchronize(stream);
    if (e != cudaSuccess) throw_cuda(e, "cudaStreamSynchronize");
}

#endif  // WITH_MPI

}  // namespace ed::distributed::multi_gpu

#else  // !ED_HAVE_NCCL
// ============================================================================
// Fallback TU when NCCL was not discovered. We provide nccl_status_string()
// because callers may include the header from non-NCCL builds (e.g. CPU-only
// CI). MultiGpuCommunicator constructors are NOT defined here -- linkers
// will reject any code that tries to actually instantiate one in a non-NCCL
// build, which is the desired behaviour (compile-time error rather than
// silent runtime failure).
// ============================================================================
#include <ed/distributed/multi_gpu.h>
#include <string>

namespace ed::distributed::multi_gpu {
std::string nccl_status_string() {
    return "ed::distributed::multi_gpu: NCCL not compiled in";
}
}  // namespace ed::distributed::multi_gpu

#endif  // ED_HAVE_NCCL
