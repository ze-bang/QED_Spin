#pragma once
// =============================================================================
// include/ed/matvec/backends/cuda_backend.cuh
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
//   * Batched primitives `dot_many` / `axpy_many` as a single
//     `cublasZgemv` each, instead of the M sequential `cublasZdotc` /
//     `cublasZaxpy` calls of the Backend default. The kernel passes the
//     same growing basis pointer set on every CGS2 pass within a Lanczos
//     step, so the staging copy is cached with a pointer-fingerprint and
//     incrementally extended (only the new column is staged per
//     Lanczos iteration).
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
#include <cusolverDn.h>

#include <ed/matvec/backend.h>
#include <ed/matvec/memory_space.h>

namespace ed::matvec {

// Internal helper: turn a cudaError_t / cublasStatus_t into a thrown
// std::runtime_error, including the call site in the message. (Defined
// inline in the header so we don't need a .cu source for CudaBackend.)
namespace cuda_backend_detail {

inline void check_cuda(cudaError_t err, const char* what) {
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

inline void check_cusolver(cusolverStatus_t err, const char* what) {
    if (err != CUSOLVER_STATUS_SUCCESS) {
        throw std::runtime_error(std::string("CudaBackend: ") + what +
                                 " failed with cuSolver status " +
                                 std::to_string(static_cast<int>(err)));
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

class CudaBackend : public Backend {
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
        return static_cast<int>(n);
    }

    /// Construct a CudaBackend on the *current* CUDA device. Set the
    /// device with `cudaSetDevice(id)` BEFORE constructing if you want
    /// to pin to a specific GPU; the handle binds to whichever device
    /// is active at construction time.
    CudaBackend() {
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

    ~CudaBackend() override {
        // Use raw cuda* calls (noexcept) inside the destructor; errors
        // here would only surface as `cudaGetLastError()` from a later
        // call. The driver tears down resources at process exit anyway.
        if (qr_work_dev_)  cudaFree(qr_work_dev_);
        if (qr_tau_dev_)   cudaFree(qr_tau_dev_);
        if (qr_info_dev_)  cudaFree(qr_info_dev_);
        if (staging_buf_)  cudaFree(staging_buf_);
        if (coeffs_dev_)   cudaFree(coeffs_dev_);
        if (cusolver_)     cusolverDnDestroy(cusolver_);
        if (handle_)       cublasDestroy(handle_);
    }

    CudaBackend(const CudaBackend&)            = delete;
    CudaBackend& operator=(const CudaBackend&) = delete;

    // Move-only. Move transfers ownership of the cuBLAS handle AND the
    // staging buffer cache to the destination; the source is reset to
    // a default-constructed state so its destructor is a no-op. Needed
    // so callers can return a CudaBackend by value (NRVO can't elide
    // every case once the backend is wrapped in a struct alongside
    // non-trivial state).
    CudaBackend(CudaBackend&& other) noexcept
        : handle_(other.handle_),
          cusolver_(other.cusolver_),
          pool_available_(other.pool_available_),
          staging_buf_(other.staging_buf_),
          staging_capacity_(other.staging_capacity_),
          staging_n_(other.staging_n_),
          staging_fingerprint_(std::move(other.staging_fingerprint_)),
          coeffs_dev_(other.coeffs_dev_),
          coeffs_capacity_(other.coeffs_capacity_),
          qr_work_dev_(other.qr_work_dev_),
          qr_work_capacity_(other.qr_work_capacity_),
          qr_tau_dev_(other.qr_tau_dev_),
          qr_tau_capacity_(other.qr_tau_capacity_),
          qr_info_dev_(other.qr_info_dev_)
    {
        other.handle_            = nullptr;
        other.cusolver_          = nullptr;
        other.staging_buf_       = nullptr;
        other.staging_capacity_  = 0;
        other.staging_n_         = 0;
        other.coeffs_dev_        = nullptr;
        other.coeffs_capacity_   = 0;
        other.qr_work_dev_       = nullptr;
        other.qr_work_capacity_  = 0;
        other.qr_tau_dev_        = nullptr;
        other.qr_tau_capacity_   = 0;
        other.qr_info_dev_       = nullptr;
    }
    CudaBackend& operator=(CudaBackend&& other) noexcept {
        if (this != &other) {
            if (qr_work_dev_) cudaFree(qr_work_dev_);
            if (qr_tau_dev_)  cudaFree(qr_tau_dev_);
            if (qr_info_dev_) cudaFree(qr_info_dev_);
            if (staging_buf_) cudaFree(staging_buf_);
            if (coeffs_dev_)  cudaFree(coeffs_dev_);
            if (cusolver_)    cusolverDnDestroy(cusolver_);
            if (handle_)      cublasDestroy(handle_);
            handle_              = other.handle_;
            cusolver_            = other.cusolver_;
            pool_available_      = other.pool_available_;
            staging_buf_         = other.staging_buf_;
            staging_capacity_    = other.staging_capacity_;
            staging_n_           = other.staging_n_;
            staging_fingerprint_ = std::move(other.staging_fingerprint_);
            coeffs_dev_          = other.coeffs_dev_;
            coeffs_capacity_     = other.coeffs_capacity_;
            qr_work_dev_         = other.qr_work_dev_;
            qr_work_capacity_    = other.qr_work_capacity_;
            qr_tau_dev_          = other.qr_tau_dev_;
            qr_tau_capacity_     = other.qr_tau_capacity_;
            qr_info_dev_         = other.qr_info_dev_;
            other.handle_            = nullptr;
            other.cusolver_          = nullptr;
            other.staging_buf_       = nullptr;
            other.staging_capacity_  = 0;
            other.staging_n_         = 0;
            other.coeffs_dev_        = nullptr;
            other.coeffs_capacity_   = 0;
            other.qr_work_dev_       = nullptr;
            other.qr_work_capacity_  = 0;
            other.qr_tau_dev_        = nullptr;
            other.qr_tau_capacity_   = 0;
            other.qr_info_dev_       = nullptr;
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
    // Both overrides cast the M-vector batched primitive into one
    // `cublasZgemv` over a contiguous staging buffer that holds the
    // basis as an (n x M) column-major matrix. Staging is cached
    // across calls with a pointer fingerprint; the typical Lanczos
    // pattern (basis grows by one vector each iteration, and CGS2
    // passes 1 + 2 within an iteration use the same growing basis,
    // and `axpy_many` is called right after `dot_many` on the same
    // basis) gets cache hits for everything except the new column.
    //
    //   dot_many  : `coeffs = B^H * v`     <=> cublasZgemv(OP_C)
    //   axpy_many : `v     += B * alphas`  <=> cublasZgemv(OP_N)
    //
    // Per CGS2 pass at M=100, n=65536: 1 staging memcpy (the new
    // column) + 1 cublasZgemv launch + 1 D2H memcpy of M coeffs.
    // That replaces M individual cublasZdotc launches, each with
    // its own implicit host-sync. The win is most pronounced at
    // small/medium n, where per-vector launches are launch-bound
    // rather than bandwidth-bound.
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
        ensure_staging_(n, num_basis);
        ensure_coeffs_(num_basis);
        stage_basis_(basis, num_basis, n);

        // coeffs_dev_ = 1 * staging_buf_^H * v + 0 * coeffs_dev_
        const cuDoubleComplex one  = make_cuDoubleComplex(1.0, 0.0);
        const cuDoubleComplex zero = make_cuDoubleComplex(0.0, 0.0);
        cuda_backend_detail::check_cublas(
            cublasZgemv(handle_,
                CUBLAS_OP_C,
                as_blas_int(n), static_cast<int>(num_basis),
                &one,
                reinterpret_cast<const cuDoubleComplex*>(staging_buf_),
                as_blas_int(n),
                reinterpret_cast<const cuDoubleComplex*>(v), 1,
                &zero,
                reinterpret_cast<cuDoubleComplex*>(coeffs_dev_), 1),
            "cublasZgemv(dot_many)");

        // Synchronous D2H: subsequent host code reads coeffs_out
        // immediately to build the negated axpy_many call.
        cuda_backend_detail::check_cuda(
            cudaMemcpy(coeffs_out, coeffs_dev_,
                       num_basis * sizeof(Complex),
                       cudaMemcpyDeviceToHost),
            "cudaMemcpy(dot_many D2H)");
    }

    void axpy_many(const Complex*        alphas,
                   const Complex* const* basis,
                   std::size_t           num_basis,
                   Complex*              v,
                   std::size_t           n) const override {
        if (num_basis == 0 || n == 0) return;
        ensure_staging_(n, num_basis);
        ensure_coeffs_(num_basis);
        stage_basis_(basis, num_basis, n);

        // Stage alphas (host -> device).
        cuda_backend_detail::check_cuda(
            cudaMemcpyAsync(coeffs_dev_, alphas,
                            num_basis * sizeof(Complex),
                            cudaMemcpyHostToDevice, /*stream=*/0),
            "cudaMemcpyAsync(alphas H2D)");

        // v = 1 * staging_buf_ * alphas + 1 * v
        const cuDoubleComplex one = make_cuDoubleComplex(1.0, 0.0);
        cuda_backend_detail::check_cublas(
            cublasZgemv(handle_,
                CUBLAS_OP_N,
                as_blas_int(n), static_cast<int>(num_basis),
                &one,
                reinterpret_cast<const cuDoubleComplex*>(staging_buf_),
                as_blas_int(n),
                reinterpret_cast<const cuDoubleComplex*>(coeffs_dev_), 1,
                &one,
                reinterpret_cast<cuDoubleComplex*>(v), 1),
            "cublasZgemv(axpy_many)");
    }

    // ------------------------------------------------------------------
    // Level-3 BLAS via cuBLAS / cuSolver. All matrices column-major;
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
                static_cast<int>(m), as_blas_int(n), static_cast<int>(k),
                &a,
                reinterpret_cast<const cuDoubleComplex*>(A), static_cast<int>(lda),
                reinterpret_cast<const cuDoubleComplex*>(B), static_cast<int>(ldb),
                &b,
                reinterpret_cast<cuDoubleComplex*>(C), static_cast<int>(ldc)),
            "cublasZgemm");
    }

    /// In-place tall-skinny QR via cuSolver ZGEQRF + ZUNGQR. `A` is on
    /// device (m_local x b column-major); R_host is host scratch
    /// (b x b column-major).
    void qr_thin(Complex* A, std::size_t m_local, std::size_t b,
                 Complex* R_host) const override {
        if (b == 0 || m_local == 0) return;
        if (m_local < b) {
            throw std::runtime_error(
                "CudaBackend::qr_thin: m_local < b is not supported");
        }
        ensure_cusolver_();
        const int M = static_cast<int>(m_local);
        const int N = static_cast<int>(b);

        ensure_qr_tau_(b);
        if (!qr_info_dev_) {
            cuda_backend_detail::check_cuda(
                cudaMalloc(&qr_info_dev_, sizeof(int)), "cudaMalloc(qr_info)");
        }

        int lwork_geqrf = 0;
        cuda_backend_detail::check_cusolver(
            cusolverDnZgeqrf_bufferSize(cusolver_, M, N,
                reinterpret_cast<cuDoubleComplex*>(A), M, &lwork_geqrf),
            "cusolverDnZgeqrf_bufferSize");
        int lwork_ungqr = 0;
        cuda_backend_detail::check_cusolver(
            cusolverDnZungqr_bufferSize(cusolver_, M, N, N,
                reinterpret_cast<cuDoubleComplex*>(A), M,
                reinterpret_cast<cuDoubleComplex*>(qr_tau_dev_),
                &lwork_ungqr),
            "cusolverDnZungqr_bufferSize");
        const int lwork = std::max(lwork_geqrf, lwork_ungqr);
        ensure_qr_work_(static_cast<std::size_t>(lwork));

        cuda_backend_detail::check_cusolver(
            cusolverDnZgeqrf(cusolver_, M, N,
                reinterpret_cast<cuDoubleComplex*>(A), M,
                reinterpret_cast<cuDoubleComplex*>(qr_tau_dev_),
                reinterpret_cast<cuDoubleComplex*>(qr_work_dev_), lwork,
                static_cast<int*>(qr_info_dev_)),
            "cusolverDnZgeqrf");
        int info_host = 0;
        cuda_backend_detail::check_cuda(
            cudaMemcpy(&info_host, qr_info_dev_, sizeof(int),
                       cudaMemcpyDeviceToHost), "qr_thin info D2H");
        if (info_host != 0) {
            throw std::runtime_error(
                "CudaBackend::qr_thin: cusolverDnZgeqrf info=" +
                std::to_string(info_host));
        }

        // Copy device A (which currently holds Householder factors in
        // the lower part and the upper-triangular R in the upper part)
        // back to host as a b x b extract before ZUNGQR overwrites it.
        std::vector<Complex> top_block(m_local * b);
        const std::size_t copy_rows = std::min(m_local, b);
        for (std::size_t j = 0; j < b; ++j) {
            cuda_backend_detail::check_cuda(
                cudaMemcpy(top_block.data() + j * copy_rows,
                           A + j * m_local,
                           copy_rows * sizeof(Complex),
                           cudaMemcpyDeviceToHost),
                "qr_thin: D2H column for R extract");
        }
        for (std::size_t j = 0; j < b; ++j) {
            for (std::size_t i = 0; i <= j; ++i) {
                R_host[i + j * b] = top_block[i + j * copy_rows];
            }
            for (std::size_t i = j + 1; i < b; ++i) {
                R_host[i + j * b] = Complex{0.0, 0.0};
            }
        }

        cuda_backend_detail::check_cusolver(
            cusolverDnZungqr(cusolver_, M, N, N,
                reinterpret_cast<cuDoubleComplex*>(A), M,
                reinterpret_cast<cuDoubleComplex*>(qr_tau_dev_),
                reinterpret_cast<cuDoubleComplex*>(qr_work_dev_), lwork,
                static_cast<int*>(qr_info_dev_)),
            "cusolverDnZungqr");
        cuda_backend_detail::check_cuda(
            cudaMemcpy(&info_host, qr_info_dev_, sizeof(int),
                       cudaMemcpyDeviceToHost), "qr_thin info D2H (ungqr)");
        if (info_host != 0) {
            throw std::runtime_error(
                "CudaBackend::qr_thin: cusolverDnZungqr info=" +
                std::to_string(info_host));
        }
    }

private:
    cublasHandle_t             handle_         = nullptr;
    mutable cusolverDnHandle_t cusolver_       = nullptr;
    bool                       pool_available_ = false;

    // Persistent staging buffer for batched dot_many / axpy_many. Sized
    // lazily to fit (n x m) complex<double>. Owned by the backend so
    // the cost amortises across all Lanczos / FTLM / KS calls on this
    // instance.
    mutable Complex*     staging_buf_       = nullptr;
    mutable std::size_t  staging_capacity_  = 0;  // bytes
    mutable std::size_t  staging_n_         = 0;  // row count of staged content
    mutable std::vector<const Complex*> staging_fingerprint_;

    // Device buffer for the M coefficients computed by `dot_many` /
    // consumed by `axpy_many`. cuBLAS in HOST pointer-mode wants the
    // x/y vectors of `cublasZgemv` in device memory; we copy to host
    // (`dot_many`) or from host (`axpy_many`) once per call.
    mutable Complex*     coeffs_dev_      = nullptr;
    mutable std::size_t  coeffs_capacity_ = 0;  // bytes

    // qr_thin: persistent cuSolver scratch.
    mutable void*       qr_work_dev_      = nullptr;
    mutable std::size_t qr_work_capacity_ = 0;  // bytes
    mutable Complex*    qr_tau_dev_       = nullptr;
    mutable std::size_t qr_tau_capacity_  = 0;  // count of Complex
    mutable void*       qr_info_dev_      = nullptr;

    void ensure_cusolver_() const {
        if (cusolver_) return;
        cuda_backend_detail::check_cusolver(
            cusolverDnCreate(&cusolver_), "cusolverDnCreate");
    }

    void ensure_qr_work_(std::size_t lwork) const {
        const std::size_t bytes = lwork * sizeof(Complex);
        if (bytes <= qr_work_capacity_) return;
        if (qr_work_dev_) cudaFree(qr_work_dev_);
        qr_work_dev_ = nullptr;
        cuda_backend_detail::check_cuda(
            cudaMalloc(&qr_work_dev_, bytes), "cudaMalloc(qr_work)");
        qr_work_capacity_ = bytes;
    }

    void ensure_qr_tau_(std::size_t count) const {
        if (count <= qr_tau_capacity_) return;
        if (qr_tau_dev_) cudaFree(qr_tau_dev_);
        qr_tau_dev_ = nullptr;
        cuda_backend_detail::check_cuda(
            cudaMalloc(reinterpret_cast<void**>(&qr_tau_dev_),
                       count * sizeof(Complex)),
            "cudaMalloc(qr_tau)");
        qr_tau_capacity_ = count;
    }

    void ensure_staging_(std::size_t n, std::size_t m) const {
        const std::size_t needed = n * m * sizeof(Complex);
        if (needed > staging_capacity_) {
            if (staging_buf_) cudaFree(staging_buf_);
            staging_buf_ = nullptr;
            cuda_backend_detail::check_cuda(
                cudaMalloc(reinterpret_cast<void**>(&staging_buf_), needed),
                "cudaMalloc(staging)");
            staging_capacity_ = needed;
            // Re-malloc invalidates everything that was previously
            // staged; force a re-stage on the next call.
            staging_fingerprint_.clear();
            staging_n_ = 0;
        }
    }

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

    // Stage `num_basis` device pointers into `staging_buf_` as an
    // (n x num_basis) column-major matrix. Cache layout uses
    // `staging_fingerprint_` (a copy of the basis pointer array
    // from the last call); we recognise three cases:
    //
    //   (a) exact match (same pointers, same count) -> no work.
    //   (b) prefix match (cached array is a prefix of current) ->
    //       stage only the new tail columns. This is the typical
    //       Lanczos pattern.
    //   (c) anything else -> re-stage all columns.
    //
    // The fingerprint is also invalidated when `staging_n_` changes
    // (different problem size, or capacity grew). All D2D memcpys
    // are async on the default stream; the subsequent cublasZgemv
    // sees them via stream ordering.
    void stage_basis_(const Complex* const* basis,
                      std::size_t           num_basis,
                      std::size_t           n) const {
        const bool n_match = (staging_n_ == n);

        std::size_t start = 0;
        if (n_match) {
            if (staging_fingerprint_.size() == num_basis &&
                std::equal(staging_fingerprint_.begin(),
                           staging_fingerprint_.end(), basis)) {
                return;  // exact cache hit
            }
            if (staging_fingerprint_.size() < num_basis &&
                std::equal(staging_fingerprint_.begin(),
                           staging_fingerprint_.end(), basis)) {
                start = staging_fingerprint_.size();  // prefix hit
            }
        }

        for (std::size_t k = start; k < num_basis; ++k) {
            cuda_backend_detail::check_cuda(
                cudaMemcpyAsync(staging_buf_ + k * n, basis[k],
                                n * sizeof(Complex),
                                cudaMemcpyDeviceToDevice, /*stream=*/0),
                "cudaMemcpyAsync(stage)");
        }
        staging_fingerprint_.assign(basis, basis + num_basis);
        staging_n_ = n;
    }
};

}  // namespace ed::matvec
