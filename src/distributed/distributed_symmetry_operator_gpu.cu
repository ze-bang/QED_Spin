// =============================================================================
// src/distributed/distributed_symmetry_operator_gpu.cu    (Phase C)
//
// Multi-GPU SpMV for the symmetry-projected operator. NCCL pairwise
// SendRecv halo + CSR sparse-matvec on the row slab built by the wrapped
// CPU `DistributedSymmetryOperator`. See header for design + scope.
//
// Compiled iff `ED_HAVE_NCCL` is set (WITH_MPI && WITH_CUDA && NCCL).
// =============================================================================

#ifdef ED_HAVE_NCCL

#include <ed/distributed/distributed_symmetry_operator_gpu.h>
#include <ed/distributed/orbit_halo_plan.h>

#include <cuComplex.h>
#include <cuda_runtime.h>
#include <nccl.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace ed::distributed {

namespace {

[[noreturn]] void throw_cuda(cudaError_t e, const char* what) {
    std::ostringstream os;
    os << "DistributedSymmetryOperatorGPU: CUDA error in " << what << ": "
       << cudaGetErrorString(e) << " (code=" << static_cast<int>(e) << ")";
    throw std::runtime_error(os.str());
}

[[noreturn]] void throw_nccl(ncclResult_t e, const char* what) {
    std::ostringstream os;
    os << "DistributedSymmetryOperatorGPU: NCCL error in " << what << ": "
       << ncclGetErrorString(e) << " (code=" << static_cast<int>(e) << ")";
    throw std::runtime_error(os.str());
}

inline void check_cu(cudaError_t e, const char* what) {
    if (e != cudaSuccess) throw_cuda(e, what);
}

inline void check_nccl(ncclResult_t e, const char* what) {
    if (e != ncclSuccess) throw_nccl(e, what);
}

template <typename T>
T* device_alloc_copy(const T* host, std::size_t count, std::size_t& running) {
    if (count == 0) return nullptr;
    T* d = nullptr;
    const std::size_t b = count * sizeof(T);
    check_cu(cudaMalloc(reinterpret_cast<void**>(&d), b), "cudaMalloc");
    check_cu(cudaMemcpy(d, host, b, cudaMemcpyHostToDevice),
             "cudaMemcpy(H2D)");
    running += b;
    return d;
}

template <typename T>
void device_free(T*& p) {
    if (p) {
        cudaFree(p);
        p = nullptr;
    }
}

}  // namespace

// ---------------------------------------------------------------------------
// Pack kernel: send_buf[k] = x_local[send_local_idx[k]]
// ---------------------------------------------------------------------------
__global__ void sym_pack_send_kernel(const cuDoubleComplex* __restrict__ x_local,
                                     const int* __restrict__ send_local_idx,
                                     cuDoubleComplex* __restrict__ send_buf,
                                     int total_send) {
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= total_send) return;
    send_buf[k] = x_local[send_local_idx[k]];
}

// ---------------------------------------------------------------------------
// CSR SpMV kernel: one thread per local row.
//   y[r] = sum_{k in row r} coeff[k] * (is_local[k] ? x_local[col[k]]
//                                                   : halo[col[k]])
// ---------------------------------------------------------------------------
__global__ void sym_spmv_kernel(const cuDoubleComplex* __restrict__ x_local,
                                const cuDoubleComplex* __restrict__ halo,
                                cuDoubleComplex* __restrict__ y,
                                const int* __restrict__ row_offsets,
                                const int* __restrict__ col_idx,
                                const std::uint8_t* __restrict__ is_local,
                                const double* __restrict__ coeff_re,
                                const double* __restrict__ coeff_im,
                                int n_rows) {
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= n_rows) return;
    const int s = row_offsets[r];
    const int e = row_offsets[r + 1];
    cuDoubleComplex acc = make_cuDoubleComplex(0.0, 0.0);
    for (int k = s; k < e; ++k) {
        const int  c    = col_idx[k];
        const bool loc  = (is_local[k] != 0);
        const cuDoubleComplex x = loc ? x_local[c] : halo[c];
        const cuDoubleComplex co = make_cuDoubleComplex(coeff_re[k],
                                                        coeff_im[k]);
        acc = cuCadd(acc, cuCmul(co, x));
    }
    y[r] = acc;
}

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------
DistributedSymmetryOperatorGPU::DistributedSymmetryOperatorGPU(
    std::shared_ptr<DistributedSymmetryOperator> op,
    const multi_gpu::MultiGpuCommunicator& gpu_comm)
    : op_(std::move(op))
{
    if (!multi_gpu::nccl_compiled_in()) {
        throw std::logic_error(
            "DistributedSymmetryOperatorGPU: NCCL not compiled in "
            "(rebuild with WITH_MPI=ON WITH_CUDA=ON and NCCL_FOUND).");
    }
    if (!op_) {
        throw std::invalid_argument(
            "DistributedSymmetryOperatorGPU: null operator");
    }
    if (!gpu_comm.valid()) {
        throw std::invalid_argument(
            "DistributedSymmetryOperatorGPU: gpu_comm is not valid");
    }

    // Ensure same MPI communicator.
    int same = 0;
    MPI_Comm_compare(op_->comm(), gpu_comm.mpi_comm(), &same);
    if (same != MPI_IDENT && same != MPI_CONGRUENT) {
        throw std::invalid_argument(
            "DistributedSymmetryOperatorGPU: gpu_comm.mpi_comm() does not "
            "match op->comm() (must be MPI_IDENT or MPI_CONGRUENT).");
    }

    device_index_ = gpu_comm.device();
    if (device_index_ >= 0) {
        check_cu(cudaSetDevice(device_index_), "cudaSetDevice");
    }

    local_n_ = static_cast<int>(op_->local_size());
    n_rows_  = local_n_;

    // -------------------------------------------------------------------
    // Flatten the CPU CSR row slab. We reach into the CPU class via the
    // public apply() pattern: the per-row arrays row_col_idx_ etc are
    // private, so we reconstruct equivalent flat buffers by re-running
    // the same logic exposed through the partition / halo plan + a
    // friend-free probe. Here we instead expose those arrays via the
    // existing public interface --- the CPU class makes only the
    // *aggregate* `local_nnz()` public, so we drive the projection
    // through a lightweight host-side mirror: walk the CPU object's
    // halo_plan() and partition() to get the geometry, and pull the
    // projected CSR values via a one-shot matvec probe pattern is too
    // heavy. The cleanest route is direct member access; we add a
    // narrow accessor on the CPU side. For now, since the CPU class is
    // in the same library and we control both, we use a direct member
    // query exposed through the `cpu_csr_*` accessors below.
    //
    // (Implementation detail: the CPU class already exposes everything
    // we need EXCEPT the per-row arrays. Rather than perturb the public
    // header just for this, we bind via friendship in the .cu by
    // forward-declaring the internal accessors. The cleanest approach
    // is to add a thin public accessor `csr_rows()` to the CPU class
    // returning const refs to the three vectors -- which we do as part
    // of Phase C below.)
    // -------------------------------------------------------------------

    const auto& row_cols  = op_->csr_row_col_idx();
    const auto& row_isloc = op_->csr_row_is_local();
    const auto& row_coefs = op_->csr_row_coeff();

    if (static_cast<int>(row_cols.size()) != n_rows_) {
        throw std::runtime_error(
            "DistributedSymmetryOperatorGPU: CPU csr row count ("
            + std::to_string(row_cols.size()) + ") != local_size ("
            + std::to_string(n_rows_) + ")");
    }

    std::vector<int>          h_row_offsets(n_rows_ + 1, 0);
    for (int r = 0; r < n_rows_; ++r) {
        h_row_offsets[r + 1] = h_row_offsets[r]
                             + static_cast<int>(row_cols[r].size());
    }
    nnz_ = h_row_offsets[n_rows_];

    std::vector<int>          h_col_idx(nnz_);
    std::vector<std::uint8_t> h_is_local(nnz_);
    std::vector<double>       h_coeff_re(nnz_);
    std::vector<double>       h_coeff_im(nnz_);
    for (int r = 0; r < n_rows_; ++r) {
        const auto& cols  = row_cols[r];
        const auto& isloc = row_isloc[r];
        const auto& coefs = row_coefs[r];
        const int   off   = h_row_offsets[r];
        for (std::size_t k = 0; k < cols.size(); ++k) {
            // cols[k] is std::size_t; on the CPU it indexes either a
            // local-amplitude slot (range < local_n_) or a halo slot
            // (range < halo_recv_total). Both fit comfortably in int32
            // for any feasible problem size; assert defensively.
            if (cols[k] > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
                throw std::runtime_error(
                    "DistributedSymmetryOperatorGPU: CSR column index exceeds "
                    "INT_MAX (problem too large for int32 indexing)");
            }
            h_col_idx[off + k]  = static_cast<int>(cols[k]);
            h_is_local[off + k] = isloc[k];
            h_coeff_re[off + k] = coefs[k].real();
            h_coeff_im[off + k] = coefs[k].imag();
        }
    }

    d_row_offsets_ = device_alloc_copy(h_row_offsets.data(),
                                       h_row_offsets.size(), plan_bytes_);
    if (nnz_ > 0) {
        d_col_idx_  = device_alloc_copy(h_col_idx.data(),  h_col_idx.size(),  plan_bytes_);
        d_is_local_ = device_alloc_copy(h_is_local.data(), h_is_local.size(), plan_bytes_);
        d_coeff_re_ = device_alloc_copy(h_coeff_re.data(), h_coeff_re.size(), plan_bytes_);
        d_coeff_im_ = device_alloc_copy(h_coeff_im.data(), h_coeff_im.size(), plan_bytes_);
    }

    // -------------------------------------------------------------------
    // Halo plan: counts/displs stay on host (per-peer NCCL scalars);
    // send_local_idx goes to device.
    // -------------------------------------------------------------------
    const OrbitHaloPlan* plan = op_->halo_plan();
    const int S = gpu_comm.size();
    send_counts_.assign(S, 0);
    recv_counts_.assign(S, 0);
    send_displs_.assign(S, 0);
    recv_displs_.assign(S, 0);
    total_send_ = 0;
    total_recv_ = 0;

    if (plan != nullptr) {
        const auto& sc = plan->send_counts();
        const auto& rc = plan->recv_counts();
        if (static_cast<int>(sc.size()) != S
            || static_cast<int>(rc.size()) != S) {
            throw std::runtime_error(
                "DistributedSymmetryOperatorGPU: halo plan comm_size mismatch");
        }
        for (int p = 0; p < S; ++p) {
            send_counts_[p] = sc[p];
            recv_counts_[p] = rc[p];
        }
        for (int p = 1; p < S; ++p) {
            send_displs_[p] = send_displs_[p - 1] + send_counts_[p - 1];
            recv_displs_[p] = recv_displs_[p - 1] + recv_counts_[p - 1];
        }
        total_send_ = static_cast<int>(plan->send_total());
        total_recv_ = static_cast<int>(plan->recv_total());

        if (total_send_ > 0) {
            const auto& sli = plan->send_local_idx();
            std::vector<int> h_sli(total_send_);
            for (int k = 0; k < total_send_; ++k) {
                if (sli[k] > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
                    throw std::runtime_error(
                        "DistributedSymmetryOperatorGPU: send_local_idx exceeds INT_MAX");
                }
                h_sli[k] = static_cast<int>(sli[k]);
            }
            d_send_local_idx_ = device_alloc_copy(h_sli.data(), h_sli.size(),
                                                  plan_bytes_);
            check_cu(cudaMalloc(&d_send_buf_,
                                static_cast<std::size_t>(total_send_)
                                    * sizeof(cuDoubleComplex)),
                     "cudaMalloc(d_send_buf)");
            buf_bytes_ += static_cast<std::size_t>(total_send_)
                        * sizeof(cuDoubleComplex);
        }
        if (total_recv_ > 0) {
            check_cu(cudaMalloc(&d_recv_buf_,
                                static_cast<std::size_t>(total_recv_)
                                    * sizeof(cuDoubleComplex)),
                     "cudaMalloc(d_recv_buf)");
            buf_bytes_ += static_cast<std::size_t>(total_recv_)
                        * sizeof(cuDoubleComplex);
        }
    }
}

DistributedSymmetryOperatorGPU::~DistributedSymmetryOperatorGPU() {
    release_();
}

void DistributedSymmetryOperatorGPU::release_() {
    device_free(d_row_offsets_);
    device_free(d_col_idx_);
    device_free(d_is_local_);
    device_free(d_coeff_re_);
    device_free(d_coeff_im_);
    device_free(d_send_local_idx_);
    if (d_send_buf_) { cudaFree(d_send_buf_); d_send_buf_ = nullptr; }
    if (d_recv_buf_) { cudaFree(d_recv_buf_); d_recv_buf_ = nullptr; }
}

// ---------------------------------------------------------------------------
// Apply
// ---------------------------------------------------------------------------
void DistributedSymmetryOperatorGPU::apply(
    const multi_gpu::MultiGpuCommunicator& gpu_comm,
    const Complex* d_x_local, Complex* d_y_local,
    cudaStream_t stream) const {

    if (!gpu_comm.valid()) {
        throw std::logic_error(
            "DistributedSymmetryOperatorGPU::apply: gpu_comm is not valid");
    }
    auto* x_d    = reinterpret_cast<const cuDoubleComplex*>(d_x_local);
    auto* y_d    = reinterpret_cast<cuDoubleComplex*>(d_y_local);
    auto* send_d = static_cast<cuDoubleComplex*>(d_send_buf_);
    auto* recv_d = static_cast<cuDoubleComplex*>(d_recv_buf_);

    // (1) Pack send buffer.
    if (total_send_ > 0) {
        const int block = 256;
        const int grid  = (total_send_ + block - 1) / block;
        sym_pack_send_kernel<<<grid, block, 0, stream>>>(
            x_d, d_send_local_idx_, send_d, total_send_);
        check_cu(cudaGetLastError(), "sym_pack_send_kernel launch");
    }

    // (2) NCCL pairwise SendRecv halo (one group). Each complex<double>
    // is sent as 2 ncclFloat64 elements (no reduction in send/recv, so
    // the reinterpretation is exact -- same convention as
    // DistributedGPUOperator::apply).
    const int S = gpu_comm.size();
    ncclComm_t comm = gpu_comm.nccl();
    check_nccl(ncclGroupStart(), "ncclGroupStart");
    for (int peer = 0; peer < S; ++peer) {
        const int sc = send_counts_[peer];
        if (sc > 0) {
            cuDoubleComplex* base = send_d + send_displs_[peer];
            check_nccl(ncclSend(base, static_cast<std::size_t>(sc) * 2,
                                ncclFloat64, peer, comm, stream),
                       "ncclSend");
        }
        const int rc = recv_counts_[peer];
        if (rc > 0) {
            cuDoubleComplex* base = recv_d + recv_displs_[peer];
            check_nccl(ncclRecv(base, static_cast<std::size_t>(rc) * 2,
                                ncclFloat64, peer, comm, stream),
                       "ncclRecv");
        }
    }
    check_nccl(ncclGroupEnd(), "ncclGroupEnd");

    // (3) CSR SpMV kernel.
    if (n_rows_ > 0) {
        const int block = 128;
        const int grid  = (n_rows_ + block - 1) / block;
        sym_spmv_kernel<<<grid, block, 0, stream>>>(
            x_d, recv_d, y_d, d_row_offsets_, d_col_idx_, d_is_local_,
            d_coeff_re_, d_coeff_im_, n_rows_);
        check_cu(cudaGetLastError(), "sym_spmv_kernel launch");
    }
}

}  // namespace ed::distributed

#endif  // ED_HAVE_NCCL
