// =============================================================================
// src/distributed/distributed_gpu_operator.cu    (Phase 3c stage 3)
//
// Fully GPU-resident DistributedOperator: NCCL pairwise SendRecv halo +
// CUDA SpMV kernel reading the same comm plan that the CPU
// `DistributedOperator` builds. See
// include/ed/distributed/distributed_gpu_operator.h for design + scope.
//
// Compiled iff `ED_HAVE_NCCL` is set (i.e. WITH_MPI && WITH_CUDA &&
// NCCL_FOUND, all wired through cmake/EDLibraries.cmake).
// =============================================================================

#ifdef ED_HAVE_NCCL

#include <ed/distributed/distributed_gpu_operator.h>

// Pull in the full Operator definition for the SoA fields.
#include <ed/core/construct_ham.h>

#include <cuComplex.h>
#include <cuda_runtime.h>
#include <nccl.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace ed::distributed {

namespace {

[[noreturn]] void throw_cuda(cudaError_t e, const char* what) {
    std::ostringstream os;
    os << "DistributedGPUOperator: CUDA error in " << what << ": "
       << cudaGetErrorString(e) << " (code=" << static_cast<int>(e) << ")";
    throw std::runtime_error(os.str());
}

[[noreturn]] void throw_nccl(ncclResult_t e, const char* what) {
    std::ostringstream os;
    os << "DistributedGPUOperator: NCCL error in " << what << ": "
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
T* device_alloc_copy(const T* host, std::size_t count, std::size_t& running_bytes) {
    if (count == 0) return nullptr;
    T* d = nullptr;
    const std::size_t b = count * sizeof(T);
    check_cu(cudaMalloc(reinterpret_cast<void**>(&d), b), "cudaMalloc");
    check_cu(cudaMemcpy(d, host, b, cudaMemcpyHostToDevice),
             "cudaMemcpy(H2D)");
    running_bytes += b;
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
// Pack kernel: d_send_buf[k] = d_v_local[d_send_local_idx[k]]
// ---------------------------------------------------------------------------
__global__ void pack_send_buf_kernel(const cuDoubleComplex* __restrict__ v_local,
                                     const int* __restrict__ send_local_idx,
                                     cuDoubleComplex* __restrict__ send_buf,
                                     int total_send) {
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= total_send) return;
    send_buf[k] = v_local[send_local_idx[k]];
}

// ---------------------------------------------------------------------------
// Device-side binary search into recv_keys (sorted ascending). Returns
// the index in [0, total_recv) such that recv_keys[idx] == key, or -1 if
// the key is not present.
// ---------------------------------------------------------------------------
__device__ inline int recv_lookup_find(const std::uint64_t* __restrict__ keys,
                                       int total_recv,
                                       std::uint64_t key) {
    int lo = 0;
    int hi = total_recv;
    while (lo < hi) {
        int mid = lo + ((hi - lo) >> 1);
        std::uint64_t kv = keys[mid];
        if (kv < key)       lo = mid + 1;
        else if (kv > key)  hi = mid;
        else                return mid;
    }
    return -1;
}

// Helper: fetch v[c] for an arbitrary global column c that the SpMV
// kernel touches. Local hits are O(1); off-rank hits are O(log).
__device__ inline cuDoubleComplex
read_v_global(const cuDoubleComplex* __restrict__ v_local,
              const cuDoubleComplex* __restrict__ recv_buf,
              const std::uint64_t* __restrict__ recv_keys,
              const std::size_t* __restrict__ recv_values,
              std::uint64_t local_offset, std::uint64_t local_n,
              int total_recv,
              std::uint64_t c) {
    if (c >= local_offset && c < local_offset + local_n) {
        return v_local[c - local_offset];
    }
    int idx = recv_lookup_find(recv_keys, total_recv, c);
    if (idx < 0) {
        // build_comm_pattern_ guarantees coverage; treat a miss as zero
        // to be safe (kernels can't throw). The CPU-side assert in
        // distributed_operator.cpp catches the same condition.
        return make_cuDoubleComplex(0.0, 0.0);
    }
    std::size_t pos = recv_values[idx];
    return recv_buf[pos];
}

// ---------------------------------------------------------------------------
// SpMV kernel (GATHER form). One thread per local row.
// ---------------------------------------------------------------------------
__global__ void distributed_gpu_spmv_kernel(
    const cuDoubleComplex* __restrict__ v_local,
    cuDoubleComplex*       __restrict__ y_local,
    const cuDoubleComplex* __restrict__ recv_buf,
    const std::uint64_t*   __restrict__ recv_keys,
    const std::size_t*     __restrict__ recv_values,
    int                    total_recv,
    std::uint64_t          local_offset,
    std::uint64_t          local_n,
    double                 spin,
    double                 spin_sq,
    int                    n_d1,
    const std::uint64_t*   __restrict__ d1_site,
    const double*          __restrict__ d1_re,
    const double*          __restrict__ d1_im,
    int                    n_o1,
    const std::uint64_t*   __restrict__ o1_site,
    const std::uint8_t*    __restrict__ o1_op,
    const double*          __restrict__ o1_re,
    const double*          __restrict__ o1_im,
    int                    n_d2,
    const std::uint64_t*   __restrict__ d2_s1,
    const std::uint64_t*   __restrict__ d2_s2,
    const double*          __restrict__ d2_re,
    const double*          __restrict__ d2_im,
    int                    n_m2,
    const std::uint64_t*   __restrict__ m2_sz,
    const std::uint64_t*   __restrict__ m2_flip,
    const std::uint8_t*    __restrict__ m2_op,
    const double*          __restrict__ m2_re,
    const double*          __restrict__ m2_im,
    int                    n_o2,
    const std::uint64_t*   __restrict__ o2_s1,
    const std::uint64_t*   __restrict__ o2_s2,
    const std::uint8_t*    __restrict__ o2_op1,
    const std::uint8_t*    __restrict__ o2_op2,
    const double*          __restrict__ o2_re,
    const double*          __restrict__ o2_im) {

    const std::uint64_t r_local =
        static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (r_local >= local_n) return;

    const std::uint64_t r = local_offset + r_local;

    double acc_re = 0.0;
    double acc_im = 0.0;

    // diag_one_body (Sz)
    for (int t = 0; t < n_d1; ++t) {
        double sign = ((r >> d1_site[t]) & 1ULL) ? -1.0 : 1.0;
        cuDoubleComplex vc = v_local[r_local];
        double scale = spin * sign;
        double cre = d1_re[t] * scale;
        double cim = d1_im[t] * scale;
        acc_re += cre * vc.x - cim * vc.y;
        acc_im += cre * vc.y + cim * vc.x;
    }

    // offdiag_one_body (S+ / S-)
    for (int t = 0; t < n_o1; ++t) {
        std::uint64_t bit = (r >> o1_site[t]) & 1ULL;
        if (bit != static_cast<std::uint64_t>(o1_op[t])) {
            std::uint64_t c = r ^ (1ULL << o1_site[t]);
            cuDoubleComplex vc = read_v_global(
                v_local, recv_buf, recv_keys, recv_values,
                local_offset, local_n, total_recv, c);
            double cre = o1_re[t];
            double cim = o1_im[t];
            acc_re += cre * vc.x - cim * vc.y;
            acc_im += cre * vc.y + cim * vc.x;
        }
    }

    // diag_two_body (Sz_i Sz_j)
    for (int t = 0; t < n_d2; ++t) {
        double si = ((r >> d2_s1[t]) & 1ULL) ? -1.0 : 1.0;
        double sj = ((r >> d2_s2[t]) & 1ULL) ? -1.0 : 1.0;
        cuDoubleComplex vc = v_local[r_local];
        double scale = spin_sq * si * sj;
        double cre = d2_re[t] * scale;
        double cim = d2_im[t] * scale;
        acc_re += cre * vc.x - cim * vc.y;
        acc_im += cre * vc.y + cim * vc.x;
    }

    // mixed_two_body
    for (int t = 0; t < n_m2; ++t) {
        std::uint64_t flip_bit = (r >> m2_flip[t]) & 1ULL;
        if (flip_bit != static_cast<std::uint64_t>(m2_op[t])) {
            std::uint64_t b = r ^ (1ULL << m2_flip[t]);
            double sz_sign = ((b >> m2_sz[t]) & 1ULL) ? -1.0 : 1.0;
            cuDoubleComplex vc = read_v_global(
                v_local, recv_buf, recv_keys, recv_values,
                local_offset, local_n, total_recv, b);
            double scale = spin * sz_sign;
            double cre = m2_re[t] * scale;
            double cim = m2_im[t] * scale;
            acc_re += cre * vc.x - cim * vc.y;
            acc_im += cre * vc.y + cim * vc.x;
        }
    }

    // offdiag_two_body
    for (int t = 0; t < n_o2; ++t) {
        std::uint64_t bit_1 = (r >> o2_s1[t]) & 1ULL;
        std::uint64_t bit_2 = (r >> o2_s2[t]) & 1ULL;
        if (bit_1 != static_cast<std::uint64_t>(o2_op1[t]) &&
            bit_2 != static_cast<std::uint64_t>(o2_op2[t])) {
            std::uint64_t c = r ^ (1ULL << o2_s1[t]) ^ (1ULL << o2_s2[t]);
            cuDoubleComplex vc = read_v_global(
                v_local, recv_buf, recv_keys, recv_values,
                local_offset, local_n, total_recv, c);
            double cre = o2_re[t];
            double cim = o2_im[t];
            acc_re += cre * vc.x - cim * vc.y;
            acc_im += cre * vc.y + cim * vc.x;
        }
    }

    y_local[r_local] = make_cuDoubleComplex(acc_re, acc_im);
}

// ===========================================================================
// Member functions
// ===========================================================================

DistributedGPUOperator::DistributedGPUOperator(
    std::shared_ptr<DistributedOperator> op,
    const multi_gpu::MultiGpuCommunicator& gpu_comm)
    : op_(std::move(op)) {

    if (!multi_gpu::nccl_compiled_in()) {
        throw std::logic_error(
            "DistributedGPUOperator: NCCL not compiled in (build with "
            "WITH_CUDA=ON and NCCL_FOUND=ON)");
    }
    if (!op_) {
        throw std::invalid_argument("DistributedGPUOperator: null op");
    }
    if (gpu_comm.size() != op_->comm_size() ||
        gpu_comm.rank() != op_->rank()) {
        throw std::invalid_argument(
            "DistributedGPUOperator: gpu_comm rank/size mismatch with op");
    }

    auto serial = op_->serial_operator();
    if (!serial->three_body_data_.empty()) {
        throw std::invalid_argument(
            "DistributedGPUOperator: 3-body terms not supported in stage 3 "
            "(rejected at construction; use the host-staged "
            "distributed_lanczos_gpu path or extend the SpMV kernel)");
    }

    device_index_ = gpu_comm.device();
    check_cu(cudaSetDevice(device_index_), "cudaSetDevice");

    spin_         = static_cast<double>(serial->getSpin());
    spin_sq_      = spin_ * spin_;
    local_offset_ = op_->local_offset();
    local_n_      = op_->local_size();

    upload_comm_plan_();
    upload_term_tables_();
}

void DistributedGPUOperator::upload_comm_plan_() {
    auto plan = op_->comm_plan_view();
    const int S = plan.comm_size;
    send_counts_.assign(plan.send_counts, plan.send_counts + S);
    send_displs_.assign(plan.send_displs, plan.send_displs + S);
    recv_counts_.assign(plan.recv_counts, plan.recv_counts + S);
    recv_displs_.assign(plan.recv_displs, plan.recv_displs + S);
    total_send_ = plan.total_send;
    total_recv_ = plan.total_recv;

    plan_bytes_ = 0;
    if (total_send_ > 0) {
        d_send_local_idx_ = device_alloc_copy(
            plan.send_local_idx, static_cast<std::size_t>(total_send_),
            plan_bytes_);
    }
    if (total_recv_ > 0 && plan.recv_keys) {
        d_recv_keys_ = device_alloc_copy(
            plan.recv_keys, static_cast<std::size_t>(total_recv_),
            plan_bytes_);
        d_recv_values_ = device_alloc_copy(
            plan.recv_values, static_cast<std::size_t>(total_recv_),
            plan_bytes_);
    }

    buf_bytes_ = 0;
    if (total_send_ > 0) {
        const std::size_t b = static_cast<std::size_t>(total_send_) *
                              sizeof(cuDoubleComplex);
        check_cu(cudaMalloc(&d_send_buf_, b), "cudaMalloc(send_buf)");
        buf_bytes_ += b;
    }
    if (total_recv_ > 0) {
        const std::size_t b = static_cast<std::size_t>(total_recv_) *
                              sizeof(cuDoubleComplex);
        check_cu(cudaMalloc(&d_recv_buf_, b), "cudaMalloc(recv_buf)");
        buf_bytes_ += b;
    }
}

void DistributedGPUOperator::upload_term_tables_() {
    auto serial = op_->serial_operator();

    n_diag_one_ = static_cast<int>(serial->diag_one_body_.size());
    if (n_diag_one_ > 0) {
        std::vector<std::uint64_t> sites(n_diag_one_);
        std::vector<double> re(n_diag_one_), im(n_diag_one_);
        for (int i = 0; i < n_diag_one_; ++i) {
            sites[i] = serial->diag_one_body_[i].site_index;
            re[i]    = serial->diag_one_body_[i].coefficient.real();
            im[i]    = serial->diag_one_body_[i].coefficient.imag();
        }
        d_d1_site_ = device_alloc_copy(sites.data(), n_diag_one_, plan_bytes_);
        d_d1_re_   = device_alloc_copy(re.data(),    n_diag_one_, plan_bytes_);
        d_d1_im_   = device_alloc_copy(im.data(),    n_diag_one_, plan_bytes_);
    }

    n_offdiag_one_ = static_cast<int>(serial->offdiag_one_body_.size());
    if (n_offdiag_one_ > 0) {
        std::vector<std::uint64_t> sites(n_offdiag_one_);
        std::vector<std::uint8_t>  ops(n_offdiag_one_);
        std::vector<double> re(n_offdiag_one_), im(n_offdiag_one_);
        for (int i = 0; i < n_offdiag_one_; ++i) {
            sites[i] = serial->offdiag_one_body_[i].site_index;
            ops[i]   = serial->offdiag_one_body_[i].op_type;
            re[i]    = serial->offdiag_one_body_[i].coefficient.real();
            im[i]    = serial->offdiag_one_body_[i].coefficient.imag();
        }
        d_o1_site_ = device_alloc_copy(sites.data(), n_offdiag_one_, plan_bytes_);
        d_o1_op_   = device_alloc_copy(ops.data(),   n_offdiag_one_, plan_bytes_);
        d_o1_re_   = device_alloc_copy(re.data(),    n_offdiag_one_, plan_bytes_);
        d_o1_im_   = device_alloc_copy(im.data(),    n_offdiag_one_, plan_bytes_);
    }

    n_diag_two_ = static_cast<int>(serial->diag_two_body_.size());
    if (n_diag_two_ > 0) {
        std::vector<std::uint64_t> s1(n_diag_two_), s2(n_diag_two_);
        std::vector<double> re(n_diag_two_), im(n_diag_two_);
        for (int i = 0; i < n_diag_two_; ++i) {
            s1[i] = serial->diag_two_body_[i].site_index_1;
            s2[i] = serial->diag_two_body_[i].site_index_2;
            re[i] = serial->diag_two_body_[i].coefficient.real();
            im[i] = serial->diag_two_body_[i].coefficient.imag();
        }
        d_d2_s1_ = device_alloc_copy(s1.data(), n_diag_two_, plan_bytes_);
        d_d2_s2_ = device_alloc_copy(s2.data(), n_diag_two_, plan_bytes_);
        d_d2_re_ = device_alloc_copy(re.data(), n_diag_two_, plan_bytes_);
        d_d2_im_ = device_alloc_copy(im.data(), n_diag_two_, plan_bytes_);
    }

    n_mixed_two_ = static_cast<int>(serial->mixed_two_body_.size());
    if (n_mixed_two_ > 0) {
        std::vector<std::uint64_t> sz(n_mixed_two_), fl(n_mixed_two_);
        std::vector<std::uint8_t>  op(n_mixed_two_);
        std::vector<double> re(n_mixed_two_), im(n_mixed_two_);
        for (int i = 0; i < n_mixed_two_; ++i) {
            sz[i] = serial->mixed_two_body_[i].sz_site;
            fl[i] = serial->mixed_two_body_[i].flip_site;
            op[i] = serial->mixed_two_body_[i].flip_op_type;
            re[i] = serial->mixed_two_body_[i].coefficient.real();
            im[i] = serial->mixed_two_body_[i].coefficient.imag();
        }
        d_m2_sz_   = device_alloc_copy(sz.data(), n_mixed_two_, plan_bytes_);
        d_m2_flip_ = device_alloc_copy(fl.data(), n_mixed_two_, plan_bytes_);
        d_m2_op_   = device_alloc_copy(op.data(), n_mixed_two_, plan_bytes_);
        d_m2_re_   = device_alloc_copy(re.data(), n_mixed_two_, plan_bytes_);
        d_m2_im_   = device_alloc_copy(im.data(), n_mixed_two_, plan_bytes_);
    }

    n_offdiag_two_ = static_cast<int>(serial->offdiag_two_body_.size());
    if (n_offdiag_two_ > 0) {
        std::vector<std::uint64_t> s1(n_offdiag_two_), s2(n_offdiag_two_);
        std::vector<std::uint8_t>  op1(n_offdiag_two_), op2(n_offdiag_two_);
        std::vector<double> re(n_offdiag_two_), im(n_offdiag_two_);
        for (int i = 0; i < n_offdiag_two_; ++i) {
            s1[i]  = serial->offdiag_two_body_[i].site_index_1;
            s2[i]  = serial->offdiag_two_body_[i].site_index_2;
            op1[i] = serial->offdiag_two_body_[i].op_type_1;
            op2[i] = serial->offdiag_two_body_[i].op_type_2;
            re[i]  = serial->offdiag_two_body_[i].coefficient.real();
            im[i]  = serial->offdiag_two_body_[i].coefficient.imag();
        }
        d_o2_s1_  = device_alloc_copy(s1.data(),  n_offdiag_two_, plan_bytes_);
        d_o2_s2_  = device_alloc_copy(s2.data(),  n_offdiag_two_, plan_bytes_);
        d_o2_op1_ = device_alloc_copy(op1.data(), n_offdiag_two_, plan_bytes_);
        d_o2_op2_ = device_alloc_copy(op2.data(), n_offdiag_two_, plan_bytes_);
        d_o2_re_  = device_alloc_copy(re.data(),  n_offdiag_two_, plan_bytes_);
        d_o2_im_  = device_alloc_copy(im.data(),  n_offdiag_two_, plan_bytes_);
    }
}

void DistributedGPUOperator::release_() {
    device_free(d_send_local_idx_);
    device_free(d_recv_keys_);
    device_free(d_recv_values_);
    if (d_send_buf_) { cudaFree(d_send_buf_); d_send_buf_ = nullptr; }
    if (d_recv_buf_) { cudaFree(d_recv_buf_); d_recv_buf_ = nullptr; }
    device_free(d_d1_site_);
    device_free(d_d1_re_); device_free(d_d1_im_);
    device_free(d_o1_site_); device_free(d_o1_op_);
    device_free(d_o1_re_); device_free(d_o1_im_);
    device_free(d_d2_s1_); device_free(d_d2_s2_);
    device_free(d_d2_re_); device_free(d_d2_im_);
    device_free(d_m2_sz_); device_free(d_m2_flip_); device_free(d_m2_op_);
    device_free(d_m2_re_); device_free(d_m2_im_);
    device_free(d_o2_s1_); device_free(d_o2_s2_);
    device_free(d_o2_op1_); device_free(d_o2_op2_);
    device_free(d_o2_re_); device_free(d_o2_im_);
}

DistributedGPUOperator::~DistributedGPUOperator() {
    release_();
}

void DistributedGPUOperator::apply(
    const multi_gpu::MultiGpuCommunicator& gpu_comm,
    const Complex* d_v_local, Complex* d_y_local,
    cudaStream_t stream) const {

    if (!gpu_comm.valid()) {
        throw std::logic_error(
            "DistributedGPUOperator::apply: gpu_comm is not valid");
    }
    auto* v_d    = reinterpret_cast<const cuDoubleComplex*>(d_v_local);
    auto* y_d    = reinterpret_cast<cuDoubleComplex*>(d_y_local);
    auto* send_d = static_cast<cuDoubleComplex*>(d_send_buf_);
    auto* recv_d = static_cast<cuDoubleComplex*>(d_recv_buf_);

    // ---- (1) Pack the send buffer on device. -----------------------------
    if (total_send_ > 0) {
        const int block = 256;
        const int grid  = (total_send_ + block - 1) / block;
        pack_send_buf_kernel<<<grid, block, 0, stream>>>(
            v_d, d_send_local_idx_, send_d, total_send_);
        check_cu(cudaGetLastError(), "pack_send_buf_kernel launch");
    }

    // ---- (2) NCCL halo exchange. -----------------------------------------
    // Pairwise ncclSend / ncclRecv per peer with non-zero count, wrapped
    // in a single ncclGroupStart / ncclGroupEnd so NCCL can fuse and
    // schedule them deadlock-free.
    //
    // Each std::complex<double> is sent as 2 ncclFloat64 elements
    // because NCCL does not have a native complex type but
    // (a+bi) + (c+di) is element-wise on (real, imag), and
    // send / recv don't perform any reduction so the reinterpretation
    // is exact.
    const int S = gpu_comm.size();
    ncclComm_t comm = gpu_comm.nccl();
    check_nccl(ncclGroupStart(), "ncclGroupStart");
    for (int peer = 0; peer < S; ++peer) {
        const int sc = (peer < static_cast<int>(send_counts_.size()))
                           ? send_counts_[peer] : 0;
        if (sc > 0) {
            cuDoubleComplex* base = send_d + send_displs_[peer];
            check_nccl(ncclSend(base, static_cast<std::size_t>(sc) * 2,
                                ncclFloat64, peer, comm, stream),
                       "ncclSend");
        }
        const int rc = (peer < static_cast<int>(recv_counts_.size()))
                           ? recv_counts_[peer] : 0;
        if (rc > 0) {
            cuDoubleComplex* base = recv_d + recv_displs_[peer];
            check_nccl(ncclRecv(base, static_cast<std::size_t>(rc) * 2,
                                ncclFloat64, peer, comm, stream),
                       "ncclRecv");
        }
    }
    check_nccl(ncclGroupEnd(), "ncclGroupEnd");

    // ---- (3) SpMV kernel. ------------------------------------------------
    if (local_n_ > 0) {
        const int block = 128;
        const int grid  = static_cast<int>((local_n_ + block - 1) / block);
        distributed_gpu_spmv_kernel<<<grid, block, 0, stream>>>(
            v_d, y_d, recv_d, d_recv_keys_, d_recv_values_, total_recv_,
            local_offset_, local_n_, spin_, spin_sq_,
            n_diag_one_,    d_d1_site_, d_d1_re_, d_d1_im_,
            n_offdiag_one_, d_o1_site_, d_o1_op_, d_o1_re_, d_o1_im_,
            n_diag_two_,    d_d2_s1_, d_d2_s2_, d_d2_re_, d_d2_im_,
            n_mixed_two_,   d_m2_sz_, d_m2_flip_, d_m2_op_, d_m2_re_, d_m2_im_,
            n_offdiag_two_, d_o2_s1_, d_o2_s2_, d_o2_op1_, d_o2_op2_,
            d_o2_re_, d_o2_im_);
        check_cu(cudaGetLastError(), "distributed_gpu_spmv_kernel launch");
    }
}

}  // namespace ed::distributed

#endif  // ED_HAVE_NCCL
