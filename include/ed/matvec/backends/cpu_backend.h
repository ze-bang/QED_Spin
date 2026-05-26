#pragma once
// =============================================================================
// include/ed/matvec/backends/cpu_backend.h
//
// CpuBackend: host-memory, OpenMP-parallel realisation of the Backend
// interface. Drives every CPU solver in the codebase (Lanczos, FTLM,
// LTLM, TPQ, CG, KPM, time evolution) once the matvec-unification revamp
// is complete.
//
// Vector primitives delegate to BLAS via the existing
// `ed/core/blas_lapack_wrapper.h` shim (the same code path the legacy
// solvers already use), so wall-clock performance is identical or better
// (one less indirection per call).
//
// Allocations use aligned new (64-byte alignment for AVX-512) so the
// inner SpMV / level-1 BLAS loops can rely on aligned moves.
// =============================================================================

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <new>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef _OPENMP
#  include <omp.h>
#endif

#include <ed/core/blas_lapack_wrapper.h>
#include <ed/matvec/backend.h>
#include <ed/matvec/memory_space.h>

namespace ed::matvec {

// NOT `final`: `MpiBackend` derives from this class to reuse the host
// allocators / BLAS-1 locals and override only the reduction-bearing
// primitives. The original `final` keyword was a copy-paste from
// `CudaBackend`; removed (May 2026, day 8) when the
// `distributed_lanczos` migration onto `lanczos_kernel<MpiBackend>`
// actually instantiated `MpiBackend` for the first time and tripped
// the contradiction.
class CpuBackend : public Backend {
public:
    [[nodiscard]] MemorySpace memory_space() const override {
        return MemorySpace::Host;
    }
    [[nodiscard]] std::string description() const override {
        return "CpuBackend(OpenMP)";
    }

    // 64-byte aligned alloc keeps level-1 BLAS happy on AVX-512 nodes.
    [[nodiscard]] Complex* allocate(std::size_t n) const override {
        if (n == 0) return nullptr;
        void* p = nullptr;
#if defined(_ISOC11_SOURCE) || defined(__APPLE__) || defined(_WIN32)
        p = std::aligned_alloc(
            64, ((n * sizeof(Complex) + 63) / 64) * 64);
#else
        if (posix_memalign(&p, 64, n * sizeof(Complex)) != 0) p = nullptr;
#endif
        if (!p) throw std::bad_alloc{};
        return static_cast<Complex*>(p);
    }
    void deallocate(Complex* p) const noexcept override {
        std::free(p);
    }
    void fill_zero(Complex* p, std::size_t n) const override {
        if (n == 0 || !p) return;
        std::memset(p, 0, n * sizeof(Complex));
    }
    void copy(const Complex* src, Complex* dst, std::size_t n) const override {
        if (n == 0) return;
        std::memcpy(dst, src, n * sizeof(Complex));
    }
    void copy_from_host(const Complex* host_src, Complex* dst,
                        std::size_t n) const override {
        copy(host_src, dst, n);
    }
    void copy_to_host(const Complex* src, Complex* host_dst,
                      std::size_t n) const override {
        copy(src, host_dst, n);
    }

    // -----------------------------------------------------------------
    // Level-1 BLAS. We deliberately re-implement here with OpenMP
    // instead of cblas_zaxpy / cblas_zdotc so the Backend has zero
    // mandatory link-time dependencies (the BLAS shim continues to be
    // used by the legacy code path during the migration). After Phase
    // 4 we can swap these for BLAS calls if a profiler ever shows
    // them to be the bottleneck --- they currently bind-and-stream at
    // memory bandwidth, which BLAS would not improve.
    // -----------------------------------------------------------------
    void axpy(Complex alpha, const Complex* x, Complex* y,
              std::size_t n) const override {
        if (n == 0) return;
        #pragma omp parallel for schedule(static) if(n > 8192)
        for (long long i = 0; i < static_cast<long long>(n); ++i) {
            y[i] += alpha * x[i];
        }
    }
    void scale(Complex alpha, Complex* x, std::size_t n) const override {
        if (n == 0) return;
        #pragma omp parallel for schedule(static) if(n > 8192)
        for (long long i = 0; i < static_cast<long long>(n); ++i) {
            x[i] *= alpha;
        }
    }
    [[nodiscard]] Complex dot(const Complex* x, const Complex* y,
                              std::size_t n) const override {
        if (n == 0) return Complex{0.0, 0.0};
        double re = 0.0, im = 0.0;
        #pragma omp parallel for reduction(+:re,im) schedule(static) if(n > 8192)
        for (long long i = 0; i < static_cast<long long>(n); ++i) {
            const Complex xc = std::conj(x[i]);
            const Complex y_  = y[i];
            re += xc.real() * y_.real() - xc.imag() * y_.imag();
            im += xc.real() * y_.imag() + xc.imag() * y_.real();
        }
        return Complex{re, im};
    }
    [[nodiscard]] double nrm2(const Complex* x, std::size_t n) const override {
        if (n == 0) return 0.0;
        double sum = 0.0;
        #pragma omp parallel for reduction(+:sum) schedule(static) if(n > 8192)
        for (long long i = 0; i < static_cast<long long>(n); ++i) {
            const Complex v = x[i];
            sum += v.real() * v.real() + v.imag() * v.imag();
        }
        return std::sqrt(sum);
    }
    void axpby(Complex alpha, const Complex* x,
               Complex beta,  Complex* y, std::size_t n) const override {
        if (n == 0) return;
        #pragma omp parallel for schedule(static) if(n > 8192)
        for (long long i = 0; i < static_cast<long long>(n); ++i) {
            y[i] = alpha * x[i] + beta * y[i];
        }
    }

    // ----------------------------------------------------------------
    // Batched primitives (CGS2 reorth fast path).
    //
    // dot_many: each thread sweeps a chunk of `i in [0, n)` and
    // accumulates partial sums of <basis[k], v> for every k. Final
    // reduction across threads via a critical section (or partial
    // arrays followed by sum). One streaming pass over `v` feeds all
    // k inner products --- bandwidth-bound, but only one read of v.
    //
    // axpy_many: each thread sweeps a chunk of `i` and accumulates
    // sum_k alphas[k] * basis[k][i] into v[i]. One streaming pass
    // over v.
    // ----------------------------------------------------------------
    void dot_many(const Complex* const* basis,
                  std::size_t           num_basis,
                  const Complex*        v,
                  std::size_t           n,
                  Complex*              coeffs_out) const override {
        if (num_basis == 0) return;
        if (n == 0) {
            for (std::size_t k = 0; k < num_basis; ++k) coeffs_out[k] = {0, 0};
            return;
        }

#ifdef _OPENMP
        const int nthreads = omp_get_max_threads();
#else
        const int nthreads = 1;
#endif
        // Per-thread partial-sum scratch: nthreads * num_basis doubles
        // each for re / im. Phase 6.1 of the Krylov-unification gap-fill
        // (May 2026): held in `mutable` member storage so a tight Lanczos
        // loop doesn't re-alloc on every iteration. Per-instance and only
        // touched serially from the calling thread (the OMP parallel
        // region below operates on disjoint slices via `tid * num_basis`
        // offset, so the buffer is safe to share across OMP child
        // threads). The `CpuBackend` singleton is process-global; if a
        // higher-level caller dispatches two `dot_many` calls in
        // parallel on the same singleton, the calls must synchronize
        // (the kernel and every legacy caller does so by construction:
        // `dot_many` is always invoked from a serial section).
        const std::size_t need = static_cast<std::size_t>(nthreads) * num_basis;
        if (scratch_partial_re_.size() < need) scratch_partial_re_.resize(need);
        if (scratch_partial_im_.size() < need) scratch_partial_im_.resize(need);
        std::fill_n(scratch_partial_re_.begin(), need, 0.0);
        std::fill_n(scratch_partial_im_.begin(), need, 0.0);
        double* const partial_re = scratch_partial_re_.data();
        double* const partial_im = scratch_partial_im_.data();

        #pragma omp parallel if(n > 8192)
        {
#ifdef _OPENMP
            const int tid = omp_get_thread_num();
#else
            const int tid = 0;
#endif
            double* re = &partial_re[tid * num_basis];
            double* im = &partial_im[tid * num_basis];

            #pragma omp for schedule(static) nowait
            for (long long i = 0; i < static_cast<long long>(n); ++i) {
                const Complex vi = v[i];
                for (std::size_t k = 0; k < num_basis; ++k) {
                    // <basis[k], v> = sum_i conj(basis[k][i]) * v[i].
                    const Complex bk = basis[k][i];
                    // conj(bk) * vi = (br - i*bi)*(vr + i*vi)
                    //              = br*vr + bi*vi + i*(br*vi - bi*vr)
                    re[k] += bk.real() * vi.real() + bk.imag() * vi.imag();
                    im[k] += bk.real() * vi.imag() - bk.imag() * vi.real();
                }
            }
        }

        for (std::size_t k = 0; k < num_basis; ++k) {
            double r = 0.0, i = 0.0;
            for (int t = 0; t < nthreads; ++t) {
                r += partial_re[t * num_basis + k];
                i += partial_im[t * num_basis + k];
            }
            coeffs_out[k] = Complex(r, i);
        }
    }

    void axpy_many(const Complex*        alphas,
                   const Complex* const* basis,
                   std::size_t           num_basis,
                   Complex*              v,
                   std::size_t           n) const override {
        if (num_basis == 0 || n == 0) return;
        #pragma omp parallel for schedule(static) if(n > 8192)
        for (long long i = 0; i < static_cast<long long>(n); ++i) {
            Complex acc = v[i];
            for (std::size_t k = 0; k < num_basis; ++k) {
                acc += alphas[k] * basis[k][i];
            }
            v[i] = acc;
        }
    }

    // -----------------------------------------------------------------
    // Level-3 BLAS via cBLAS / LAPACKE (Phase 1 of the Minimalist ED
    // Collapse). All matrix arguments column-major.
    // -----------------------------------------------------------------
    void gemm(char opA, char opB,
              std::size_t m, std::size_t n, std::size_t k,
              Complex alpha,
              const Complex* A, std::size_t lda,
              const Complex* B, std::size_t ldb,
              Complex beta,
              Complex* C, std::size_t ldc) const override {
        if (m == 0 || n == 0) return;
        const CBLAS_TRANSPOSE tA = trans_(opA);
        const CBLAS_TRANSPOSE tB = trans_(opB);
        cblas_zgemm(CblasColMajor, tA, tB,
                    static_cast<int>(m), static_cast<int>(n), static_cast<int>(k),
                    &alpha, A, static_cast<int>(lda),
                            B, static_cast<int>(ldb),
                    &beta,  C, static_cast<int>(ldc));
    }

    void gemv(char opA,
              std::size_t m, std::size_t n,
              Complex alpha,
              const Complex* A, std::size_t lda,
              const Complex* x, std::size_t incx,
              Complex beta,
              Complex* y, std::size_t incy) const override {
        if (m == 0 || n == 0) return;
        cblas_zgemv(CblasColMajor, trans_(opA),
                    static_cast<int>(m), static_cast<int>(n),
                    &alpha, A, static_cast<int>(lda),
                            x, static_cast<int>(incx),
                    &beta,  y, static_cast<int>(incy));
    }

    void trsm(char side, char uplo, char transA, char diag,
              std::size_t m, std::size_t n,
              Complex alpha,
              const Complex* A, std::size_t lda,
              Complex* B, std::size_t ldb) const override {
        if (m == 0 || n == 0) return;
        const CBLAS_SIDE      sd = (side == 'L' || side == 'l') ? CblasLeft  : CblasRight;
        const CBLAS_UPLO      up = (uplo == 'U' || uplo == 'u') ? CblasUpper : CblasLower;
        const CBLAS_TRANSPOSE tr = trans_(transA);
        const CBLAS_DIAG      dg = (diag == 'U' || diag == 'u') ? CblasUnit  : CblasNonUnit;
        cblas_ztrsm(CblasColMajor, sd, up, tr, dg,
                    static_cast<int>(m), static_cast<int>(n),
                    &alpha, A, static_cast<int>(lda),
                            B, static_cast<int>(ldb));
    }

    /// In-place tall-skinny QR using LAPACK ZGEQRF + ZUNGQR. On exit
    /// the column-major A holds Q (m x b), and R_host (column-major,
    /// b x b) holds the upper triangle from ZGEQRF (extracted before
    /// ZUNGQR overwrites A with Q).
    void qr_thin(Complex* A, std::size_t m_local, std::size_t b,
                 Complex* R_host) const override {
        if (b == 0 || m_local == 0) return;
        if (m_local < b) {
            throw std::runtime_error("CpuBackend::qr_thin: m_local < b is not supported");
        }
        const int M  = static_cast<int>(m_local);
        const int N  = static_cast<int>(b);
        std::vector<Complex> tau(b);
        int info = LAPACKE_zgeqrf(LAPACK_COL_MAJOR, M, N,
            reinterpret_cast<lapack_complex_double*>(A), M,
            reinterpret_cast<lapack_complex_double*>(tau.data()));
        if (info != 0) {
            throw std::runtime_error("CpuBackend::qr_thin: LAPACKE_zgeqrf info=" +
                                     std::to_string(info));
        }
        for (std::size_t j = 0; j < b; ++j) {
            for (std::size_t i = 0; i <= j; ++i) {
                R_host[i + j * b] = A[i + j * m_local];
            }
            for (std::size_t i = j + 1; i < b; ++i) {
                R_host[i + j * b] = Complex{0.0, 0.0};
            }
        }
        info = LAPACKE_zungqr(LAPACK_COL_MAJOR, M, N, N,
            reinterpret_cast<lapack_complex_double*>(A), M,
            reinterpret_cast<lapack_complex_double*>(tau.data()));
        if (info != 0) {
            throw std::runtime_error("CpuBackend::qr_thin: LAPACKE_zungqr info=" +
                                     std::to_string(info));
        }
    }

private:
    static CBLAS_TRANSPOSE trans_(char op) {
        switch (op) {
            case 'N': case 'n': return CblasNoTrans;
            case 'T': case 't': return CblasTrans;
            case 'C': case 'c': case 'H': case 'h': return CblasConjTrans;
            default:
                throw std::invalid_argument(
                    std::string("CpuBackend: invalid trans op '") + op + "'");
        }
    }

private:
    // Phase 6.1 (Krylov-unification gap-fill, May 2026): persistent
    // per-thread accumulation scratch for `dot_many`. Sized lazily on
    // first call, grow-only across the lifetime of this backend. See
    // the comment block in `dot_many` for the concurrency contract.
    mutable std::vector<double> scratch_partial_re_;
    mutable std::vector<double> scratch_partial_im_;
};

// Singleton accessor. The CPU backend is stateless so a single static
// instance is fine; we hand out references rather than ownership so
// callers don't keep allocating it.
[[nodiscard]] inline CpuBackend& default_cpu_backend() {
    static CpuBackend instance;
    return instance;
}

} // namespace ed::matvec
