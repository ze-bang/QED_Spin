#pragma once
// =============================================================================
// include/ed/matvec/backends/cpu_backend.h
//
// CpuBackend: host-memory, OpenMP-parallel realisation of the Backend
// interface, used by the CPU solvers (Lanczos, FTLM, LTLM, TPQ, CG,
// time evolution).
//
// Level-1 vector primitives are OpenMP loops; the level-3 primitive (gemm)
// calls cBLAS through the `ed/core/blas_lapack_wrapper.h` shim.
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
        // Parallel first touch so Krylov vectors are distributed
        // across NUMA nodes with the same static chunking the BLAS-1 and
        // matvec kernels use (a serial memset places every page on the
        // calling thread's node).
        #pragma omp parallel for schedule(static) if(n > 65536)
        for (long long i = 0; i < static_cast<long long>(n); ++i) {
            p[i] = Complex{0.0, 0.0};
        }
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
    // Level-1 BLAS as OpenMP loops rather than cblas_zaxpy / cblas_zdotc.
    // These kernels stream at memory bandwidth, which BLAS would not
    // improve, and their static chunking matches fill_zero's first touch.
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
    // Fused Lanczos primitives: single streaming pass.
    // `axpy_dot_local` / `axpy_nrm2sq_local` are the local pieces.
    // ----------------------------------------------------------------
    [[nodiscard]] Complex axpy_dot_local(Complex alpha, const Complex* x, Complex* y,
                                         const Complex* z, std::size_t n) const {
        if (n == 0) return Complex{0.0, 0.0};
        double re = 0.0, im = 0.0;
        const double ar = alpha.real(), ai = alpha.imag();
        #pragma omp parallel for reduction(+:re,im) schedule(static) if(n > 8192)
        for (long long i = 0; i < static_cast<long long>(n); ++i) {
            const Complex xi = x[i];
            const double yr = y[i].real() + ar * xi.real() - ai * xi.imag();
            const double yi = y[i].imag() + ar * xi.imag() + ai * xi.real();
            y[i] = Complex(yr, yi);
            const Complex zi = z[i];
            re += zi.real() * yr + zi.imag() * yi;
            im += zi.real() * yi - zi.imag() * yr;
        }
        return Complex{re, im};
    }
    [[nodiscard]] double axpy_nrm2sq_local(Complex alpha, const Complex* x, Complex* y,
                                           std::size_t n) const {
        if (n == 0) return 0.0;
        double sq = 0.0;
        const double ar = alpha.real(), ai = alpha.imag();
        #pragma omp parallel for reduction(+:sq) schedule(static) if(n > 8192)
        for (long long i = 0; i < static_cast<long long>(n); ++i) {
            const Complex xi = x[i];
            const double yr = y[i].real() + ar * xi.real() - ai * xi.imag();
            const double yi = y[i].imag() + ar * xi.imag() + ai * xi.real();
            y[i] = Complex(yr, yi);
            sq += yr * yr + yi * yi;
        }
        return sq;
    }
    [[nodiscard]] Complex axpy_dot(Complex alpha, const Complex* x, Complex* y,
                                   const Complex* z, std::size_t n) const override {
        return axpy_dot_local(alpha, x, y, z, n);
    }
    [[nodiscard]] double axpy_nrm2(Complex alpha, const Complex* x, Complex* y,
                                   std::size_t n) const override {
        return std::sqrt(axpy_nrm2sq_local(alpha, x, y, n));
    }

    // ----------------------------------------------------------------
    // Batched primitives (CGS2 reorth fast path).
    //
    // dot_many: each thread sweeps a chunk of `i in [0, n)` and
    // accumulates partial sums of <basis[k], v> for every k. Final
    // reduction sums the per-thread partial arrays. One streaming pass over `v` feeds all
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
        // each for re / im, held in `mutable` member storage so a tight
        // Lanczos loop doesn't re-alloc on every iteration. The OMP
        // parallel region below writes disjoint slices (`tid * num_basis`
        // offset), so the buffer is safe to share across OMP child
        // threads. Two concurrent `dot_many` calls on the same instance
        // would race; `default_cpu_backend()` is thread_local for that
        // reason.
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
    // Level-3 BLAS via cBLAS. All matrix arguments column-major.
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
    // Persistent per-thread accumulation scratch for `dot_many`. Sized
    // lazily on first call, grow-only across the lifetime of this backend. See
    // the comment block in `dot_many` for the concurrency contract.
    mutable std::vector<double> scratch_partial_re_;
    mutable std::vector<double> scratch_partial_im_;
};

// Thread-local accessor. CpuBackend holds mutable scratch buffers in
// dot_many() that are not safe to share across concurrent callers. Using
// thread_local gives each thread its own instance so the outer sector-
// parallel loop can call lanczos_kernel from
// multiple OMP threads simultaneously without racing on the scratch storage.
[[nodiscard]] inline CpuBackend& default_cpu_backend() {
    static thread_local CpuBackend instance;
    return instance;
}

} // namespace ed::matvec
