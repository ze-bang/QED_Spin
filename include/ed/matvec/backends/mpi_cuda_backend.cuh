#pragma once
// =============================================================================
// include/ed/matvec/backends/mpi_cuda_backend.cuh
//
// MpiCudaBackend: GPU-resident state + NCCL collectives for the cross-rank
// reductions inside `dot`, `nrm2`, `all_reduce_sum`, and the batched
// `dot_many` used by CGS2.
//
// Phase 3 of the gap-fill rollout (May 2026 day 11+). Completes the
// four-corners deployment matrix that the Krylov-kernel unification
// promised:
//
//                 single-rank          multi-rank
//   CPU memory    CpuBackend           MpiBackend
//   GPU memory    CudaBackend          MpiCudaBackend   <- this header
//
// Architectural model: same as `MpiBackend` over `CpuBackend`. The
// derived class inherits all of `CudaBackend`'s local primitives
// (pool-backed `allocate`, cuBLAS BLAS-1, batched `cublasZgemv` for
// `dot_many` / `axpy_many`, fused `cublasZgeam` for `axpby`) and
// overrides only the reduction-bearing primitives to add an NCCL
// `ncclAllReduce` after the local cuBLAS pass. The single biggest
// win is `dot_many`: one `ncclAllReduce` over a 2*M-double buffer
// instead of M separate ones (matches the MpiBackend pattern).
//
// Pointer convention follows the Backend contract: all `Complex*`
// are rank-local device slabs of dimension `n`; the matvec callback
// handed to the Lanczos kernel performs any halo exchange itself.
//
// Build-time gate: this header only compiles when `ED_HAVE_NCCL` is
// defined (set PUBLIC by the `ed_distributed_gpu` CMake target when
// `WITH_MPI && WITH_CUDA && NCCL_FOUND`). Consumers that include the
// header must already be inside that target -- the kernel and the
// `distributed_lanczos_gpu.cu` consumer are.
//
// NCCL data path: NCCL collectives operate on device memory only,
// so each reduction stages the local scalar / vector through a
// persistent device scratch buffer (`reduce_buf_dev_`), then copies
// the rank-summed result back to host (for `dot` / `nrm2`) or back
// to the caller's host-visible buffer (for `dot_many` /
// `all_reduce_sum`). The two H<->D copies are tiny (M scalars at
// most) and dwarfed by the local cuBLAS work in any practical
// Lanczos run.
// =============================================================================

#ifdef ED_HAVE_NCCL

#include <cmath>
#include <complex>
#include <cstddef>
#include <stdexcept>
#include <string>

#include <cuda_runtime.h>
#include <mpi.h>
#include <nccl.h>

#include <ed/core/blas_lapack_wrapper.h>
#include <ed/parallel/multi_gpu.h>
#include <ed/matvec/backend.h>
#include <ed/matvec/backends/cuda_backend.cuh>
#include <ed/matvec/memory_space.h>

namespace ed::matvec {

/// GPU memory + MPI/NCCL collectives. Inherits CudaBackend's pool-backed
/// allocator + cuBLAS BLAS-1 / BLAS-2 + fused axpby; overrides only the
/// reductions to chain an `ncclAllReduce` after the local cuBLAS pass.
class MpiCudaBackend final : public CudaBackend {
public:
    /// Construct with a multi-GPU communicator (RAII over `ncclComm_t`
    /// built from an MPI_Comm; see `ed/parallel/multi_gpu.h`). The
    /// referenced comm must outlive the backend.
    explicit MpiCudaBackend(
        const ed::distributed::multi_gpu::MultiGpuCommunicator& comm)
        : comm_(comm) {}

    ~MpiCudaBackend() override {
        if (reduce_buf_dev_) cudaFree(reduce_buf_dev_);
    }

    MpiCudaBackend(const MpiCudaBackend&)            = delete;
    MpiCudaBackend& operator=(const MpiCudaBackend&) = delete;
    MpiCudaBackend(MpiCudaBackend&&)                 = delete;
    MpiCudaBackend& operator=(MpiCudaBackend&&)      = delete;

    [[nodiscard]] MemorySpace memory_space() const override {
        return MemorySpace::DistributedCudaDevice;
    }
    [[nodiscard]] std::string description() const override {
        return "MpiCudaBackend(NCCL+cuBLAS, np=" +
               std::to_string(comm_.size()) + ")";
    }

    [[nodiscard]] const ed::distributed::multi_gpu::MultiGpuCommunicator&
    communicator() const noexcept { return comm_; }

    // ------------------------------------------------------------------
    // Scalar reductions: ferry a single scalar through a device buffer.
    // ------------------------------------------------------------------
    [[nodiscard]] Complex all_reduce_sum(Complex v) const override {
        ensure_reduce_buf_(sizeof(Complex));
        cuda_backend_detail::check_cuda(
            cudaMemcpyAsync(reduce_buf_dev_, &v, sizeof(Complex),
                            cudaMemcpyHostToDevice, /*stream=*/0),
            "MpiCudaBackend: stage scalar for all_reduce_sum(Complex)");
        ed::distributed::multi_gpu::all_reduce_sum_complex_double(
            comm_, reinterpret_cast<Complex*>(reduce_buf_dev_), /*count=*/1);
        Complex out;
        cuda_backend_detail::check_cuda(
            cudaMemcpy(&out, reduce_buf_dev_, sizeof(Complex),
                       cudaMemcpyDeviceToHost),
            "MpiCudaBackend: read back all_reduce_sum(Complex)");
        return out;
    }
    [[nodiscard]] double all_reduce_sum(double v) const override {
        ensure_reduce_buf_(sizeof(double));
        cuda_backend_detail::check_cuda(
            cudaMemcpyAsync(reduce_buf_dev_, &v, sizeof(double),
                            cudaMemcpyHostToDevice, /*stream=*/0),
            "MpiCudaBackend: stage scalar for all_reduce_sum(double)");
        ed::distributed::multi_gpu::all_reduce_sum_double(
            comm_, reinterpret_cast<double*>(reduce_buf_dev_), /*count=*/1);
        double out;
        cuda_backend_detail::check_cuda(
            cudaMemcpy(&out, reduce_buf_dev_, sizeof(double),
                       cudaMemcpyDeviceToHost),
            "MpiCudaBackend: read back all_reduce_sum(double)");
        return out;
    }

    /// Vector all-reduce on host-side buffers. We stage to device and
    /// allreduce there (NCCL requires device pointers), then ferry back.
    /// Used by `qr_thin`'s CholeskyQR2 to reduce a b x b Gram block
    /// (b <= 8 in practice, so the H<->D copies are tiny).
    void all_reduce_sum_vec(Complex* buf, std::size_t count) const override {
        if (count == 0) return;
        const std::size_t bytes = count * sizeof(Complex);
        ensure_reduce_buf_(bytes);
        cuda_backend_detail::check_cuda(
            cudaMemcpyAsync(reduce_buf_dev_, buf, bytes,
                            cudaMemcpyHostToDevice, /*stream=*/0),
            "MpiCudaBackend: stage vec for all_reduce_sum_vec(Complex)");
        ed::distributed::multi_gpu::all_reduce_sum_complex_double(
            comm_, reinterpret_cast<Complex*>(reduce_buf_dev_), count);
        cuda_backend_detail::check_cuda(
            cudaMemcpy(buf, reduce_buf_dev_, bytes, cudaMemcpyDeviceToHost),
            "MpiCudaBackend: read back all_reduce_sum_vec(Complex)");
    }
    void all_reduce_sum_vec(double* buf, std::size_t count) const override {
        if (count == 0) return;
        const std::size_t bytes = count * sizeof(double);
        ensure_reduce_buf_(bytes);
        cuda_backend_detail::check_cuda(
            cudaMemcpyAsync(reduce_buf_dev_, buf, bytes,
                            cudaMemcpyHostToDevice, /*stream=*/0),
            "MpiCudaBackend: stage vec for all_reduce_sum_vec(double)");
        ed::distributed::multi_gpu::all_reduce_sum_double(
            comm_, reinterpret_cast<double*>(reduce_buf_dev_), count);
        cuda_backend_detail::check_cuda(
            cudaMemcpy(buf, reduce_buf_dev_, bytes, cudaMemcpyDeviceToHost),
            "MpiCudaBackend: read back all_reduce_sum_vec(double)");
    }

    // ------------------------------------------------------------------
    // BLAS-1 overrides: compute the local term via CudaBackend::*, then
    // reduce. We deliberately don't reuse the single-pair `dot` from
    // `dot_many` so we can batch the NCCL allreduce.
    // ------------------------------------------------------------------
    [[nodiscard]] Complex dot(const Complex* x, const Complex* y,
                              std::size_t n) const override {
        const Complex local = CudaBackend::dot(x, y, n);
        return all_reduce_sum(local);
    }

    [[nodiscard]] double nrm2(const Complex* x, std::size_t n) const override {
        // Local ||x||^2 = real(<x,x>); square-and-allreduce gives the
        // correct global ||x||. We avoid `CudaBackend::nrm2` here because
        // it returns sqrt(local), and we'd have to square it back to
        // sum across ranks -- one round-trip of precision wasted.
        Complex c       = CudaBackend::dot(x, x, n);
        double  local_sq = c.real();
        double  global_sq = all_reduce_sum(local_sq);
        return std::sqrt(global_sq);
    }

    // ------------------------------------------------------------------
    // Batched dot_many: local partial products land in coeffs_out
    // (host) via the inherited CudaBackend::dot_many; stage back to
    // device, do ONE ncclAllReduce, copy back. At M=100 this is
    // ~100x fewer NCCL collectives than calling `dot` in a loop.
    // ------------------------------------------------------------------
    void dot_many(const Complex* const* basis,
                  std::size_t           num_basis,
                  const Complex*        v,
                  std::size_t           n,
                  Complex*              coeffs_out) const override {
        if (num_basis == 0) return;

        // Local partial products in coeffs_out (host).
        CudaBackend::dot_many(basis, num_basis, v, n, coeffs_out);

        // Stage to device, allreduce, stage back. We piggy-back on the
        // CudaBackend's `coeffs_dev_` device scratch -- it's already
        // sized to fit num_basis complex doubles from the local call.
        // Allocate a separate buffer here only if for some reason the
        // size policy diverged (defensive).
        const std::size_t bytes = num_basis * sizeof(Complex);
        ensure_reduce_buf_(bytes);

        cuda_backend_detail::check_cuda(
            cudaMemcpyAsync(reduce_buf_dev_, coeffs_out, bytes,
                            cudaMemcpyHostToDevice, /*stream=*/0),
            "MpiCudaBackend: stage coeffs for dot_many allreduce");
        ed::distributed::multi_gpu::all_reduce_sum_complex_double(
            comm_, reinterpret_cast<Complex*>(reduce_buf_dev_),
            /*count=*/num_basis);
        cuda_backend_detail::check_cuda(
            cudaMemcpy(coeffs_out, reduce_buf_dev_, bytes,
                       cudaMemcpyDeviceToHost),
            "MpiCudaBackend: read back dot_many allreduce");
    }

    // axpy_many is purely local; CudaBackend's batched `cublasZgemv`
    // is the right code path on each rank. No override.

    // ------------------------------------------------------------------
    // Distributed tall-skinny QR via CholeskyQR2 (Phase 1 of the
    // Minimalist ED Collapse). Same algorithm as MpiBackend, but the
    // local gemm/trsm use cuBLAS (inherited from CudaBackend) and the
    // Gram-block allreduce goes through NCCL via
    // `all_reduce_sum_vec(Complex*, count)` above.
    // ------------------------------------------------------------------
    void qr_thin(Complex* A_dev, std::size_t m_local, std::size_t b,
                 Complex* R_host) const override {
        if (b == 0) return;

        std::vector<Complex> G_host(b * b);
        std::vector<Complex> R1(b * b);
        std::vector<Complex> R2(b * b);

        // Device scratch for the b x b Gram block.
        const std::size_t G_bytes = b * b * sizeof(Complex);
        Complex* G_dev = nullptr;
        cuda_backend_detail::check_cuda(
            cudaMalloc(reinterpret_cast<void**>(&G_dev), G_bytes),
            "MpiCudaBackend::qr_thin cudaMalloc(G_dev)");
        Complex* R_dev = nullptr;
        cuda_backend_detail::check_cuda(
            cudaMalloc(reinterpret_cast<void**>(&R_dev), G_bytes),
            "MpiCudaBackend::qr_thin cudaMalloc(R_dev)");

        auto cholesky_pass = [&](Complex* R_out) {
            // Local Gram (b x b on device)
            if (m_local > 0) {
                CudaBackend::gemm('C', 'N', b, b, m_local,
                    Complex{1.0, 0.0},
                    A_dev, m_local, A_dev, m_local,
                    Complex{0.0, 0.0}, G_dev, b);
                cuda_backend_detail::check_cuda(
                    cudaMemcpy(G_host.data(), G_dev, G_bytes,
                               cudaMemcpyDeviceToHost),
                    "MpiCudaBackend::qr_thin D2H Gram");
            } else {
                std::fill_n(G_host.begin(), b * b, Complex{0.0, 0.0});
            }
            // NCCL allreduce on the (small) host buffer, ferried via
            // `all_reduce_sum_vec` (stages H -> D -> NCCL -> D -> H).
            all_reduce_sum_vec(G_host.data(), b * b);
            // Host Cholesky: G = R^H R.
            int info = LAPACKE_zpotrf(LAPACK_COL_MAJOR, 'U',
                                      static_cast<int>(b),
                                      reinterpret_cast<lapack_complex_double*>(G_host.data()),
                                      static_cast<int>(b));
            if (info != 0) {
                cudaFree(G_dev); cudaFree(R_dev);
                throw std::runtime_error(
                    "MpiCudaBackend::qr_thin: CholeskyQR factorization "
                    "failed (info=" + std::to_string(info) + ")");
            }
            std::fill_n(R_out, b * b, Complex{0.0, 0.0});
            for (std::size_t j = 0; j < b; ++j) {
                for (std::size_t i = 0; i <= j; ++i) {
                    R_out[i + j * b] = G_host[i + j * b];
                }
            }
            // Stage R to device for the trsm.
            cuda_backend_detail::check_cuda(
                cudaMemcpy(R_dev, R_out, G_bytes, cudaMemcpyHostToDevice),
                "MpiCudaBackend::qr_thin H2D R");
            if (m_local > 0) {
                // A <- A * R^{-1}
                CudaBackend::trsm('R', 'U', 'N', 'N', m_local, b,
                    Complex{1.0, 0.0}, R_dev, b, A_dev, m_local);
            }
        };

        cholesky_pass(R1.data());
        cholesky_pass(R2.data());

        // R_host = R2 * R1 (b x b host-side)
        std::fill_n(R_host, b * b, Complex{0.0, 0.0});
        for (std::size_t j = 0; j < b; ++j) {
            for (std::size_t i = 0; i <= j; ++i) {
                Complex acc{0.0, 0.0};
                for (std::size_t l = i; l <= j; ++l) {
                    acc += R2[i + l * b] * R1[l + j * b];
                }
                R_host[i + j * b] = acc;
            }
        }

        cudaFree(G_dev);
        cudaFree(R_dev);
    }

private:
    const ed::distributed::multi_gpu::MultiGpuCommunicator& comm_;

    // Persistent device scratch for the H<->D ferry. Sized lazily.
    mutable void*       reduce_buf_dev_     = nullptr;
    mutable std::size_t reduce_buf_capacity_ = 0;

    void ensure_reduce_buf_(std::size_t bytes) const {
        if (bytes <= reduce_buf_capacity_) return;
        if (reduce_buf_dev_) cudaFree(reduce_buf_dev_);
        reduce_buf_dev_ = nullptr;
        cuda_backend_detail::check_cuda(
            cudaMalloc(&reduce_buf_dev_, bytes),
            "MpiCudaBackend: cudaMalloc(reduce scratch)");
        reduce_buf_capacity_ = bytes;
    }
};

}  // namespace ed::matvec

#endif  // ED_HAVE_NCCL
