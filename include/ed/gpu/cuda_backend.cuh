#pragma once
// =============================================================================
// include/ed/gpu/cuda_backend.cuh
//
// CudaBackend: CUDA-device realisation of the `Backend` interface
// (sibling of `CpuBackend`).
//
// One header-only RAII class. Each instance owns a cuBLAS handle whose
// pointer-mode is CUBLAS_POINTER_MODE_HOST -- meaning the dot/nrm2
// results are *returned to the caller as host scalars* and the
// alpha/beta arguments to axpy/scale are read from host memory. This
// matches the `Backend` interface contract (which returns
// `std::complex<double>` / `double` by value) and is the right choice
// for short-iteration Lanczos kernels where the host has to read alpha
// to advance the recurrence anyway.
//
// All `Complex*` arguments are interpreted as `cuDoubleComplex*` (the
// binary layouts are guaranteed compatible by the C++17 layout-equiv
// rules and matches CUDA's documented contract).
//
// The header is `.cuh` because it transitively pulls in `<cublas_v2.h>`
// and `<cuda_runtime.h>`; only `.cu` (or nvcc-compiled) sources should
// include it. CPU-only code paths see this header through forward
// declaration only.
//
// Performance features:
//   * Pool-backed `allocate` / `deallocate` via `cudaMallocAsync` on
//     the default stream. The default device memory pool is configured
//     with `cudaMemPoolAttrReleaseThreshold = UINT64_MAX` so freed
//     allocations stay in the pool ready for reuse (avoids the
//     ~50us per-cudaMalloc latency that bites Krylov runs at M=100).
//   * Batched primitives `dot_many` / `axpy_many` as one `cublasZgemv` per
//     run of basis columns that lie back to back in memory -- a contiguous
//     basis (Krylov-Schur's V, the Lanczos basis) is one run -- instead of
//     the M sequential `cublasZdotc` / `cublasZaxpy` calls of the Backend
//     default, with no copy of the basis and no state kept between calls
//     (P7.3: the staging cache and its pointer fingerprint are gone, so one
//     backend may serve any number of solves).
//   * Fused `axpby` via `cublasZgeam` -- one launch instead of a
//     `scale` + `axpy` pair.
// =============================================================================

#include <algorithm>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuComplex.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <ed/core/errors.h>
#include <ed/matvec/backend.h>
#include <ed/matvec/memory_space.h>

namespace ed::matvec {

// Internal helper: turn a cudaError_t / cublasStatus_t into a thrown
// std::runtime_error, including the call site in the message. (Defined
// inline in the header so we don't need a .cu source for CudaBackend.)
namespace cuda_backend_detail {

inline void check_cuda(cudaError_t err, const char* what) {
    if (err == cudaErrorMemoryAllocation) {
        cudaGetLastError();
        throw ed::ResourceLimit(std::string("CudaBackend: ") + what +
                                " failed: the device is out of memory");
    }
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("CudaBackend: ") + what +
                                 " failed: " + cudaGetErrorString(err));
    }
}

inline void check_cublas(cublasStatus_t err, const char* what) {
    if (err != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error(std::string("CudaBackend: ") + what +
                                 " failed with cuBLAS status " +
                                 std::to_string(static_cast<int>(err)) +
                                 " (last CUDA error: " +
                                 cudaGetErrorString(cudaPeekAtLastError()) + ")");
    }
}

inline cublasOperation_t to_cublas_op(char op) {
    switch (op) {
        case 'N': case 'n': return CUBLAS_OP_N;
        case 'T': case 't': return CUBLAS_OP_T;
        case 'C': case 'c': case 'H': case 'h': return CUBLAS_OP_C;
        default:
            throw std::invalid_argument(
                std::string("CudaBackend: invalid trans op '") + op + "'");
    }
}

}  // namespace cuda_backend_detail

// The device backend for vectors of Scalar. Only std::complex<double> is defined (below).
template <class Scalar>
class BasicCudaBackend;

template <>
class BasicCudaBackend<Complex> : public BasicBackend<Complex> {
public:
    /// Every cuBLAS BLAS-1 entry point takes a 32-bit count. A bare
    /// `static_cast<int>` would silently wrap at n >= 2^31 (a 32 GiB
    /// complex vector -- reachable on 80 GB parts) and compute garbage;
    /// guard the narrowing once, loudly.
    [[nodiscard]] static int as_blas_int(std::size_t n) {
        if (n > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
            throw std::length_error(
                "CudaBackend: vector length " + std::to_string(n) +
                " exceeds the 32-bit cuBLAS count limit (2^31 - 1); "
                "split the operation or use a 64-bit BLAS path.");
        }
        return static_cast<int>(n);   // narrow-ok: checked above
    }

    /// Construct a CudaBackend on the *current* CUDA device. Set the
    /// device with `cudaSetDevice(id)` BEFORE constructing if you want
    /// to pin to a specific GPU; the handle binds to whichever device
    /// is active at construction time.
    BasicCudaBackend() {
        using namespace cuda_backend_detail;
        check_cublas(cublasCreate(&handle_), "cublasCreate");
        // Host pointer mode -- dot/nrm2 results go to host memory,
        // which is what the abstract `Backend` interface promises.
        check_cublas(cublasSetPointerMode(handle_, CUBLAS_POINTER_MODE_HOST),
                     "cublasSetPointerMode(HOST)");

        // Configure the default device memory pool to retain freed
        // allocations indefinitely (until OS reclaim). Eliminates the
        // ~50us-per-cudaMalloc latency that turns Krylov runs at
        // M=100 into 5ms of allocator churn per iteration. If the
        // device / driver doesn't support memory pools (very old
        // hardware), `cudaDeviceGetDefaultMemPool` returns an error
        // and we fall back to the synchronous `cudaMalloc` path
        // (see `allocate_impl_`).
        // Every call here may fail without consequence; each failure is cleared
        // so it cannot surface later as a misattributed kernel error.
        int dev = -1, pools = 0;
        cudaMemPool_t pool = nullptr;
        std::uint64_t threshold = UINT64_MAX;
        pool_available_ =
            cudaGetDevice(&dev) == cudaSuccess
            && cudaDeviceGetAttribute(&pools, cudaDevAttrMemoryPoolsSupported, dev) == cudaSuccess
            && pools != 0
            && cudaDeviceGetDefaultMemPool(&pool, dev) == cudaSuccess
            && cudaMemPoolSetAttribute(pool, cudaMemPoolAttrReleaseThreshold, &threshold) == cudaSuccess;
        if (!pool_available_) cudaGetLastError();
    }

    ~BasicCudaBackend() override {
        // Use raw cuda* calls (noexcept) inside the destructor; errors
        // here would only surface as `cudaGetLastError()` from a later
        // call. The driver tears down resources at process exit anyway.
        if (coeffs_dev_)   cudaFree(coeffs_dev_);
        if (handle_)       cublasDestroy(handle_);
    }

    BasicCudaBackend(const BasicCudaBackend&)            = delete;
    BasicCudaBackend& operator=(const BasicCudaBackend&) = delete;

    // Move-only. Move transfers ownership of the cuBLAS handle and the
    // coefficient buffer to the destination; the source is reset to a
    // default-constructed state so its destructor is a no-op. Needed so
    // callers can return a CudaBackend by value (NRVO can't elide every
    // case once the backend is wrapped in a struct alongside non-trivial
    // state).
    BasicCudaBackend(BasicCudaBackend&& other) noexcept
        : handle_(other.handle_),
          pool_available_(other.pool_available_),
          coeffs_dev_(other.coeffs_dev_),
          coeffs_capacity_(other.coeffs_capacity_)
    {
        other.handle_          = nullptr;
        other.coeffs_dev_      = nullptr;
        other.coeffs_capacity_ = 0;
    }
    BasicCudaBackend& operator=(BasicCudaBackend&& other) noexcept {
        if (this != &other) {
            if (coeffs_dev_) cudaFree(coeffs_dev_);
            if (handle_)     cublasDestroy(handle_);
            handle_                = other.handle_;
            pool_available_        = other.pool_available_;
            coeffs_dev_            = other.coeffs_dev_;
            coeffs_capacity_       = other.coeffs_capacity_;
            other.handle_          = nullptr;
            other.coeffs_dev_      = nullptr;
            other.coeffs_capacity_ = 0;
        }
        return *this;
    }

    [[nodiscard]] MemorySpace memory_space() const override {
        return MemorySpace::CudaDevice;
    }
    [[nodiscard]] std::string description() const override {
        int dev = -1;
        cudaGetDevice(&dev);
        return "CudaBackend(cuBLAS, device=" + std::to_string(dev) + ")";
    }

    // ------------------------------------------------------------------
    // Memory management.
    //
    // Pool-backed via `cudaMallocAsync` on the default stream when the
    // device supports memory pools (CUDA >= 11.2, all current NVIDIA
    // datacenter parts and recent consumer parts). Freed allocations
    // stay in the pool because we set the release threshold to
    // UINT64_MAX in the ctor; the next `allocate` of comparable size
    // is then a cheap pool-pop rather than a fresh cudaMalloc. This
    // is the difference between ~5ms of allocator latency per Krylov
    // run at M=100 (one cudaMalloc per basis vector) and ~50us
    // (amortised pool hits).
    //
    // Falls back to synchronous `cudaMalloc` when the device doesn't
    // expose a default pool.
    // ------------------------------------------------------------------
    [[nodiscard]] Complex* allocate(std::size_t n) const override {
        if (n == 0) return nullptr;
        void* p = nullptr;
        if (pool_available_) {
            cuda_backend_detail::check_cuda(
                cudaMallocAsync(&p, n * sizeof(Complex), /*stream=*/0),
                "cudaMallocAsync");
        } else {
            cuda_backend_detail::check_cuda(
                cudaMalloc(&p, n * sizeof(Complex)), "cudaMalloc");
        }
        return static_cast<Complex*>(p);
    }
    void deallocate(Complex* p) const noexcept override {
        if (!p) return;
        // Noexcept path: ignore errors. The pool/non-pool branch must
        // match the path taken in `allocate`; we track this implicitly
        // by `pool_available_` being a const-ish field (set once in the
        // ctor and never flipped).
        const cudaError_t err = pool_available_ ? cudaFreeAsync(p, /*stream=*/0) : cudaFree(p);
        if (err != cudaSuccess) cudaGetLastError();   // nothing to do here; do not let it linger
    }
    void fill_zero(Complex* p, std::size_t n) const override {
        if (n == 0 || !p) return;
        cuda_backend_detail::check_cuda(
            cudaMemset(p, 0, n * sizeof(Complex)), "cudaMemset");
    }
    void copy(const Complex* src, Complex* dst, std::size_t n) const override {
        if (n == 0) return;
        cuda_backend_detail::check_cuda(
            cudaMemcpy(dst, src, n * sizeof(Complex),
                       cudaMemcpyDeviceToDevice),
            "cudaMemcpy(D2D)");
    }
    void copy_from_host(const Complex* host_src,
                        Complex* device_dst,
                        std::size_t n) const override {
        if (n == 0) return;
        cuda_backend_detail::check_cuda(
            cudaMemcpy(device_dst, host_src, n * sizeof(Complex),
                       cudaMemcpyHostToDevice),
            "cudaMemcpy(H2D)");
    }
    void copy_to_host(const Complex* device_src,
                      Complex* host_dst,
                      std::size_t n) const override {
        if (n == 0) return;
        cuda_backend_detail::check_cuda(
            cudaMemcpy(host_dst, device_src, n * sizeof(Complex),
                       cudaMemcpyDeviceToHost),
            "cudaMemcpy(D2H)");
    }

    // ------------------------------------------------------------------
    // Level-1 BLAS, complex-double. Direct cuBLAS pass-through.
    // ------------------------------------------------------------------
    void axpy(Complex alpha, const Complex* x, Complex* y,
              std::size_t n) const override {
        if (n == 0) return;
        const cuDoubleComplex a = make_cuDoubleComplex(alpha.real(), alpha.imag());
        cuda_backend_detail::check_cublas(
            cublasZaxpy(handle_, as_blas_int(n), &a,
                        reinterpret_cast<const cuDoubleComplex*>(x), 1,
                        reinterpret_cast<cuDoubleComplex*>(y), 1),
            "cublasZaxpy");
    }
    void scale(Complex alpha, Complex* x, std::size_t n) const override {
        if (n == 0) return;
        const cuDoubleComplex a = make_cuDoubleComplex(alpha.real(), alpha.imag());
        cuda_backend_detail::check_cublas(
            cublasZscal(handle_, as_blas_int(n), &a,
                        reinterpret_cast<cuDoubleComplex*>(x), 1),
            "cublasZscal");
    }
    [[nodiscard]] Complex dot(const Complex* x, const Complex* y,
                              std::size_t n) const override {
        if (n == 0) return Complex{0.0, 0.0};
        cuDoubleComplex result{0.0, 0.0};
        cuda_backend_detail::check_cublas(
            cublasZdotc(handle_, as_blas_int(n),
                        reinterpret_cast<const cuDoubleComplex*>(x), 1,
                        reinterpret_cast<const cuDoubleComplex*>(y), 1,
                        &result),
            "cublasZdotc");
        return Complex{cuCreal(result), cuCimag(result)};
    }
    [[nodiscard]] double nrm2(const Complex* x, std::size_t n) const override {
        if (n == 0) return 0.0;
        double result = 0.0;
        cuda_backend_detail::check_cublas(
            cublasDznrm2(handle_, as_blas_int(n),
                         reinterpret_cast<const cuDoubleComplex*>(x), 1,
                         &result),
            "cublasDznrm2");
        return result;
    }

    // Fused axpby via `cublasZgeam` (single GPU launch):
    //
    //   y[i] = alpha * x[i] + beta * y[i]
    //
    // ZGEAM computes `C = alpha*op(A) + beta*op(B)` for matrices A/B/C.
    // We model the vector as an (n x 1) column-major matrix; A=x,
    // B=y, C=y (in-place output is permitted: cuBLAS documents that
    // C and B can overlap). One launch + the per-launch ~3us
    // overhead, vs a two-launch `scale` + `axpy` pair with an
    // implicit sync between the two kernels.
    void axpby(Complex alpha, const Complex* x,
               Complex beta,  Complex* y, std::size_t n) const override {
        if (n == 0) return;
        const cuDoubleComplex a = make_cuDoubleComplex(alpha.real(), alpha.imag());
        const cuDoubleComplex b = make_cuDoubleComplex(beta.real(),  beta.imag());
        cuda_backend_detail::check_cublas(
            cublasZgeam(handle_,
                CUBLAS_OP_N, CUBLAS_OP_N,
                as_blas_int(n), /*cols=*/1,
                &a, reinterpret_cast<const cuDoubleComplex*>(x), as_blas_int(n),
                &b, reinterpret_cast<const cuDoubleComplex*>(y), as_blas_int(n),
                    reinterpret_cast<cuDoubleComplex*>(y),       as_blas_int(n)),
            "cublasZgeam(axpby)");
    }

    // ------------------------------------------------------------------
    // Batched primitives (CGS2 reorth fast path).
    //
    //   dot_many  : `coeffs = B^H * v`     <=> cublasZgemv(OP_C)
    //   axpy_many : `v     += B * alphas`  <=> cublasZgemv(OP_N)
    //
    // One `cublasZgemv` per run of columns that sit back to back in device
    // memory (basis[k + 1] == basis[k] + n): a contiguous basis -- Krylov-
    // Schur's V, the Lanczos basis -- is a single run, read in place (no
    // copy, no state kept between calls); a separately allocated vector is
    // a run of one. The coefficients go through one device buffer: dot_many
    // reads them back once (the caller needs them at once), axpy_many
    // uploads them once and never waits.
    // ------------------------------------------------------------------
    void dot_many(const Complex* const* basis,
                  std::size_t           num_basis,
                  const Complex*        v,
                  std::size_t           n,
                  Complex*              coeffs_out) const override {
        if (num_basis == 0) return;
        if (n == 0) {
            for (std::size_t k = 0; k < num_basis; ++k) {
                coeffs_out[k] = Complex{0.0, 0.0};
            }
            return;
        }
        ensure_coeffs_(num_basis);
        const cuDoubleComplex one  = make_cuDoubleComplex(1.0, 0.0);
        const cuDoubleComplex zero = make_cuDoubleComplex(0.0, 0.0);
        for_each_run_(basis, num_basis, n, [&](std::size_t k, std::size_t len) {
            cuda_backend_detail::check_cublas(
                cublasZgemv(handle_, CUBLAS_OP_C, as_blas_int(n), as_blas_int(len), &one,
                            reinterpret_cast<const cuDoubleComplex*>(basis[k]), as_blas_int(n),
                            reinterpret_cast<const cuDoubleComplex*>(v), 1, &zero,
                            reinterpret_cast<cuDoubleComplex*>(coeffs_dev_ + k), 1),
                "cublasZgemv(dot_many)");
        });
        cuda_backend_detail::check_cuda(
            cudaMemcpy(coeffs_out, coeffs_dev_, num_basis * sizeof(Complex), cudaMemcpyDeviceToHost),
            "cudaMemcpy(dot_many D2H)");
    }

    void axpy_many(const Complex*        alphas,
                   const Complex* const* basis,
                   std::size_t           num_basis,
                   Complex*              v,
                   std::size_t           n) const override {
        if (num_basis == 0 || n == 0) return;
        ensure_coeffs_(num_basis);
        cuda_backend_detail::check_cuda(
            cudaMemcpyAsync(coeffs_dev_, alphas, num_basis * sizeof(Complex), cudaMemcpyHostToDevice,
                            /*stream=*/0),
            "cudaMemcpyAsync(alphas H2D)");
        const cuDoubleComplex one = make_cuDoubleComplex(1.0, 0.0);
        for_each_run_(basis, num_basis, n, [&](std::size_t k, std::size_t len) {
            cuda_backend_detail::check_cublas(
                cublasZgemv(handle_, CUBLAS_OP_N, as_blas_int(n), as_blas_int(len), &one,
                            reinterpret_cast<const cuDoubleComplex*>(basis[k]), as_blas_int(n),
                            reinterpret_cast<const cuDoubleComplex*>(coeffs_dev_ + k), 1, &one,
                            reinterpret_cast<cuDoubleComplex*>(v), 1),
                "cublasZgemv(axpy_many)");
        });
    }

    // ------------------------------------------------------------------
    // Level-3 BLAS via cuBLAS. All matrices column-major;
    // pointers are device pointers.
    // ------------------------------------------------------------------
    void gemm(char opA, char opB,
              std::size_t m, std::size_t n, std::size_t k,
              Complex alpha,
              const Complex* A, std::size_t lda,
              const Complex* B, std::size_t ldb,
              Complex beta,
              Complex* C, std::size_t ldc) const override {
        if (m == 0 || n == 0) return;
        const cuDoubleComplex a = make_cuDoubleComplex(alpha.real(), alpha.imag());
        const cuDoubleComplex b = make_cuDoubleComplex(beta.real(),  beta.imag());
        cuda_backend_detail::check_cublas(
            cublasZgemm(handle_,
                cuda_backend_detail::to_cublas_op(opA),
                cuda_backend_detail::to_cublas_op(opB),
                as_blas_int(m), as_blas_int(n), as_blas_int(k),
                &a,
                reinterpret_cast<const cuDoubleComplex*>(A), as_blas_int(lda),
                reinterpret_cast<const cuDoubleComplex*>(B), as_blas_int(ldb),
                &b,
                reinterpret_cast<cuDoubleComplex*>(C), as_blas_int(ldc)),
            "cublasZgemm");
    }

private:
    cublasHandle_t handle_         = nullptr;
    bool           pool_available_ = false;

    // Device buffer for the M coefficients computed by `dot_many` /
    // consumed by `axpy_many`. cuBLAS in HOST pointer-mode wants the
    // x/y vectors of `cublasZgemv` in device memory; we copy to host
    // (`dot_many`) or from host (`axpy_many`) once per call.
    mutable Complex*     coeffs_dev_      = nullptr;
    mutable std::size_t  coeffs_capacity_ = 0;  // bytes

    void ensure_coeffs_(std::size_t m) const {
        const std::size_t needed = m * sizeof(Complex);
        if (needed > coeffs_capacity_) {
            if (coeffs_dev_) cudaFree(coeffs_dev_);
            coeffs_dev_ = nullptr;
            cuda_backend_detail::check_cuda(
                cudaMalloc(reinterpret_cast<void**>(&coeffs_dev_), needed),
                "cudaMalloc(coeffs)");
            coeffs_capacity_ = needed;
        }
    }

    // f(k, len) for each maximal run basis[k .. k + len) of columns that sit back to back
    // (basis[k + i] == basis[k] + i n), in order.
    template <class F>
    static void for_each_run_(const Complex* const* basis, std::size_t m, std::size_t n, F&& f) {
        for (std::size_t k = 0; k < m;) {
            std::size_t len = 1;
            while (k + len < m && basis[k + len] == basis[k] + len * n) ++len;
            f(k, len);
            k += len;
        }
    }
};

using CudaBackend = BasicCudaBackend<Complex>;

}  // namespace ed::matvec
