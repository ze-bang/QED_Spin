#pragma once
// =============================================================================
// include/ed/matvec/cpu_backend.h
//
// CpuBackend: host-memory, OpenMP-parallel realisation of the Backend
// interface, used by the CPU solvers (Lanczos, FTLM, LTLM, TPQ, CG,
// time evolution).
//
// Level-1 vector primitives are OpenMP loops; the level-3 primitive (gemm)
// calls cBLAS through the `ed/core/lapack.h` shim.
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
#include <initializer_list>
#include <limits>
#include <new>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#ifdef _OPENMP
#  include <omp.h>
#endif

#include <ed/core/errors.h>
#include <ed/core/lapack.h>
#include <ed/matvec/backend.h>
#include <ed/matvec/memory_space.h>

namespace ed::matvec {

/// sum_{i < n} f(i) with a result that does not depend on thread timing: each thread reduces
/// its own contiguous chunk and the partials are added in thread order (an OpenMP reduction
/// combines them in arrival order, so repeated runs differed in the last bits). Serial below
/// 8193 terms. f may write element i (the fused Lanczos primitives update y in the same pass).
template <class T, class F>
[[nodiscard]] inline T ordered_sum(std::size_t n, F&& f) {
    T total{};
#ifdef _OPENMP
    if (n > 8192) {
        std::vector<T> part(static_cast<std::size_t>(omp_get_max_threads()), T{});
        #pragma omp parallel
        {
            const std::size_t t = static_cast<std::size_t>(omp_get_thread_num());
            const std::size_t nt = static_cast<std::size_t>(omp_get_num_threads());
            T s{};
            for (std::size_t i = n * t / nt, end = n * (t + 1) / nt; i < end; ++i) s += f(i);
            if (t < part.size()) part[t] = s;
        }
        for (const T& p : part) total += p;
        return total;
    }
#endif
    for (std::size_t i = 0; i < n; ++i) total += f(i);
    return total;
}

// The host backend for vectors of Scalar: std::complex<double>, and double for real blocks (P6.4).
// One body; the complex arithmetic is written out on (re, im) pairs so the compiler vectorises it
// and the real instantiation is the same loops without the imaginary parts.
template <class Scalar>
class BasicCpuBackend : public BasicBackend<Scalar> {
    static_assert(std::is_same_v<Scalar, double> || std::is_same_v<Scalar, Complex>,
                  "BasicCpuBackend: Scalar is double or std::complex<double>");
    static constexpr bool kComplex = std::is_same_v<Scalar, Complex>;

public:
    [[nodiscard]] MemorySpace memory_space() const override {
        return MemorySpace::Host;
    }
    [[nodiscard]] std::string description() const override {
        return kComplex ? "CpuBackend(OpenMP)" : "CpuBackend<double>(OpenMP)";
    }

    // 64-byte aligned alloc keeps level-1 BLAS happy on AVX-512 nodes.
    [[nodiscard]] Scalar* allocate(std::size_t n) const override {
        if (n == 0) return nullptr;
        void* p = nullptr;
#if defined(_ISOC11_SOURCE) || defined(__APPLE__) || defined(_WIN32)
        p = std::aligned_alloc(
            64, ((n * sizeof(Scalar) + 63) / 64) * 64);
#else
        if (posix_memalign(&p, 64, n * sizeof(Scalar)) != 0) p = nullptr;
#endif
        if (!p) throw std::bad_alloc{};
        return static_cast<Scalar*>(p);
    }
    void deallocate(Scalar* p) const noexcept override {
        std::free(p);
    }
    void fill_zero(Scalar* p, std::size_t n) const override {
        if (n == 0 || !p) return;
        // Parallel first touch so Krylov vectors are distributed
        // across NUMA nodes with the same static chunking the BLAS-1 and
        // matvec kernels use (a serial memset places every page on the
        // calling thread's node).
        #pragma omp parallel for schedule(static) if(n > 65536)
        for (long long i = 0; i < static_cast<long long>(n); ++i) {
            p[i] = Scalar{};
        }
    }
    void copy(const Scalar* src, Scalar* dst, std::size_t n) const override {
        if (n == 0) return;
        if (n <= 65536) {
            std::memcpy(dst, src, n * sizeof(Scalar));
            return;
        }
        // In parallel, with the static split the kernels use (a serial memcpy of a 3e7-state
        // vector takes ~50 ms, against ~5 ms on the team).
        #pragma omp parallel
        {
#ifdef _OPENMP
            const std::size_t t = static_cast<std::size_t>(omp_get_thread_num());
            const std::size_t nt = static_cast<std::size_t>(omp_get_num_threads());
#else
            const std::size_t t = 0, nt = 1;
#endif
            const std::size_t i0 = n * t / nt, i1 = n * (t + 1) / nt;
            if (i1 > i0) std::memcpy(dst + i0, src + i0, (i1 - i0) * sizeof(Scalar));
        }
    }
    void copy_from_host(const Scalar* host_src, Scalar* dst,
                        std::size_t n) const override {
        copy(host_src, dst, n);
    }
    void copy_to_host(const Scalar* src, Scalar* host_dst,
                      std::size_t n) const override {
        copy(src, host_dst, n);
    }

    // -----------------------------------------------------------------
    // Level-1 BLAS as OpenMP loops rather than cblas_zaxpy / cblas_zdotc.
    // These kernels stream at memory bandwidth, which BLAS would not
    // improve, and their static chunking matches fill_zero's first touch.
    // -----------------------------------------------------------------
    void axpy(Scalar alpha, const Scalar* x, Scalar* y,
              std::size_t n) const override {
        if (n == 0) return;
        #pragma omp parallel for schedule(static) if(n > 8192)
        for (long long i = 0; i < static_cast<long long>(n); ++i) {
            y[i] += alpha * x[i];
        }
    }
    void scale(Scalar alpha, Scalar* x, std::size_t n) const override {
        if (n == 0) return;
        #pragma omp parallel for schedule(static) if(n > 8192)
        for (long long i = 0; i < static_cast<long long>(n); ++i) {
            x[i] *= alpha;
        }
    }
    [[nodiscard]] Scalar dot(const Scalar* x, const Scalar* y,
                             std::size_t n) const override {
        if (n == 0) return Scalar{};
        return ordered_sum<Scalar>(n, [&](std::size_t i) -> Scalar {
            if constexpr (kComplex) {
                const Complex xc = std::conj(x[i]);
                const Complex y_  = y[i];
                return Complex(xc.real() * y_.real() - xc.imag() * y_.imag(),
                               xc.real() * y_.imag() + xc.imag() * y_.real());
            } else {
                return x[i] * y[i];
            }
        });
    }
    [[nodiscard]] double nrm2(const Scalar* x, std::size_t n) const override {
        if (n == 0) return 0.0;
        return std::sqrt(ordered_sum<double>(n, [&](std::size_t i) {
            if constexpr (kComplex) {
                const Complex v = x[i];
                return v.real() * v.real() + v.imag() * v.imag();
            } else {
                return x[i] * x[i];
            }
        }));
    }
    void axpby(Scalar alpha, const Scalar* x,
               Scalar beta,  Scalar* y, std::size_t n) const override {
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
    [[nodiscard]] Scalar axpy_dot_local(Scalar alpha, const Scalar* x, Scalar* y,
                                        const Scalar* z, std::size_t n) const {
        if (n == 0) return Scalar{};
        return ordered_sum<Scalar>(n, [&](std::size_t i) -> Scalar {
            if constexpr (kComplex) {
                const double ar = alpha.real(), ai = alpha.imag();
                const Complex xi = x[i];
                const double yr = y[i].real() + ar * xi.real() - ai * xi.imag();
                const double yi = y[i].imag() + ar * xi.imag() + ai * xi.real();
                y[i] = Complex(yr, yi);
                const Complex zi = z[i];
                return Complex(zi.real() * yr + zi.imag() * yi, zi.real() * yi - zi.imag() * yr);
            } else {
                y[i] += alpha * x[i];
                return z[i] * y[i];
            }
        });
    }
    [[nodiscard]] double axpy_nrm2sq_local(Scalar alpha, const Scalar* x, Scalar* y,
                                           std::size_t n) const {
        if (n == 0) return 0.0;
        return ordered_sum<double>(n, [&](std::size_t i) {
            if constexpr (kComplex) {
                const double ar = alpha.real(), ai = alpha.imag();
                const Complex xi = x[i];
                const double yr = y[i].real() + ar * xi.real() - ai * xi.imag();
                const double yi = y[i].imag() + ar * xi.imag() + ai * xi.real();
                y[i] = Complex(yr, yi);
                return yr * yr + yi * yi;
            } else {
                y[i] += alpha * x[i];
                return y[i] * y[i];
            }
        });
    }
    [[nodiscard]] Scalar axpy_dot(Scalar alpha, const Scalar* x, Scalar* y,
                                  const Scalar* z, std::size_t n) const override {
        return axpy_dot_local(alpha, x, y, z, n);
    }
    [[nodiscard]] double axpy_nrm2(Scalar alpha, const Scalar* x, Scalar* y,
                                   std::size_t n) const override {
        return std::sqrt(axpy_nrm2sq_local(alpha, x, y, n));
    }

    // ----------------------------------------------------------------
    // Batched primitives (CGS2 reorth fast path), in chunks of kManyChunk elements: within a
    // chunk the slice of v stays in cache while each basis vector streams past once, through a
    // vectorised loop. (Interleaving all the basis vectors element by element -- the old form --
    // ran ~40 memory streams at once and reached a third of the bandwidth at 3e7 states.)
    //
    // dot_many: each thread sweeps its static share of the chunks and accumulates partial sums
    // of <basis[k], v> for every k; the per-thread partials are added in thread order, so the
    // result does not depend on thread timing.
    //
    // axpy_many: v += sum_k alphas[k] basis[k], each element accumulated in k order.
    // ----------------------------------------------------------------
    static constexpr std::size_t kManyChunk = 2048;   // 32 KiB of complex v

    void dot_many(const Scalar* const* basis,
                  std::size_t          num_basis,
                  const Scalar*        v,
                  std::size_t          n,
                  Scalar*              coeffs_out) const override {
        if (num_basis == 0) return;
        if (n == 0) {
            for (std::size_t k = 0; k < num_basis; ++k) coeffs_out[k] = Scalar{};
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
        const long long chunks = static_cast<long long>((n + kManyChunk - 1) / kManyChunk);

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
            for (long long c = 0; c < chunks; ++c) {
                const std::size_t i0  = static_cast<std::size_t>(c) * kManyChunk;
                const std::size_t len = std::min(kManyChunk, n - i0);
                const double* x = reinterpret_cast<const double*>(v + i0);
                for (std::size_t k = 0; k < num_basis; ++k) {
                    const double* b = reinterpret_cast<const double*>(basis[k] + i0);
                    if constexpr (kComplex) {
                        // <b, v> = sum_i conj(b_i) v_i = sum_i (br vr + bi vi) + i (br vi - bi vr)
                        double sr = 0.0, si = 0.0;
                        #pragma omp simd reduction(+ : sr, si)
                        for (std::size_t i = 0; i < len; ++i) {
                            const double br = b[2 * i], bi = b[2 * i + 1], vr = x[2 * i], vi = x[2 * i + 1];
                            sr += br * vr + bi * vi;
                            si += br * vi - bi * vr;
                        }
                        re[k] += sr;
                        im[k] += si;
                    } else {
                        double s = 0.0;
                        #pragma omp simd reduction(+ : s)
                        for (std::size_t i = 0; i < len; ++i) s += b[i] * x[i];
                        re[k] += s;
                    }
                }
            }
        }

        for (std::size_t k = 0; k < num_basis; ++k) {
            double r = 0.0, i = 0.0;
            for (int t = 0; t < nthreads; ++t) {
                r += partial_re[t * num_basis + k];
                i += partial_im[t * num_basis + k];
            }
            if constexpr (kComplex) coeffs_out[k] = Complex(r, i);
            else                    coeffs_out[k] = r;
        }
    }

    void axpy_many(const Scalar*        alphas,
                   const Scalar* const* basis,
                   std::size_t          num_basis,
                   Scalar*              v,
                   std::size_t          n) const override {
        if (num_basis == 0 || n == 0) return;
        const long long chunks = static_cast<long long>((n + kManyChunk - 1) / kManyChunk);
        #pragma omp parallel for schedule(static) if(n > 8192)
        for (long long c = 0; c < chunks; ++c) {
            const std::size_t i0  = static_cast<std::size_t>(c) * kManyChunk;
            const std::size_t len = std::min(kManyChunk, n - i0);
            double* y = reinterpret_cast<double*>(v + i0);
            for (std::size_t k = 0; k < num_basis; ++k) {
                const double* b = reinterpret_cast<const double*>(basis[k] + i0);
                if constexpr (kComplex) {
                    const double ar = alphas[k].real(), ai = alphas[k].imag();
                    #pragma omp simd
                    for (std::size_t i = 0; i < len; ++i) {
                        const double br = b[2 * i], bi = b[2 * i + 1];
                        y[2 * i]     += ar * br - ai * bi;
                        y[2 * i + 1] += ar * bi + ai * br;
                    }
                } else {
                    const double a = alphas[k];
                    #pragma omp simd
                    for (std::size_t i = 0; i < len; ++i) y[i] += a * b[i];
                }
            }
        }
    }

    // -----------------------------------------------------------------
    // Level-3 BLAS via cBLAS. All matrix arguments column-major.
    // -----------------------------------------------------------------
    void gemm(char opA, char opB,
              std::size_t m, std::size_t n, std::size_t k,
              Scalar alpha,
              const Scalar* A, std::size_t lda,
              const Scalar* B, std::size_t ldb,
              Scalar beta,
              Scalar* C, std::size_t ldc) const override {
        if (m == 0 || n == 0) return;
        // cblas takes int: a dimension past it would wrap into a call BLAS skips or misreads.
        for (const std::size_t d : {m, n, k, lda, ldb, ldc})
            if (d > static_cast<std::size_t>(std::numeric_limits<int>::max()))
                throw ed::ResourceLimit("CpuBackend::gemm: a dimension of " + std::to_string(d)
                                        + " exceeds the 32-bit BLAS index range");
        const CBLAS_TRANSPOSE tA = trans_(opA);
        const CBLAS_TRANSPOSE tB = trans_(opB);
        if constexpr (kComplex) {
            cblas_zgemm(CblasColMajor, tA, tB,   // narrow-ok: every dimension checked above
                        static_cast<int>(m), static_cast<int>(n), static_cast<int>(k),
                        &alpha, A, static_cast<int>(lda),
                                B, static_cast<int>(ldb),
                        &beta,  C, static_cast<int>(ldc));
        } else {
            cblas_dgemm(CblasColMajor, tA, tB,   // narrow-ok: every dimension checked above
                        static_cast<int>(m), static_cast<int>(n), static_cast<int>(k),
                        alpha, A, static_cast<int>(lda),
                               B, static_cast<int>(ldb),
                        beta,  C, static_cast<int>(ldc));
        }
    }

private:
    static CBLAS_TRANSPOSE trans_(char op) {
        switch (op) {
            case 'N': case 'n': return CblasNoTrans;
            case 'T': case 't': return CblasTrans;
            case 'C': case 'c': case 'H': case 'h': return CblasConjTrans;   // dgemm: the transpose
            default:
                throw std::invalid_argument(
                    std::string("CpuBackend: invalid trans op '") + op + "'");
        }
    }

    // Persistent per-thread accumulation scratch for `dot_many`. Sized
    // lazily on first call, grow-only across the lifetime of this backend. See
    // the comment block in `dot_many` for the concurrency contract.
    mutable std::vector<double> scratch_partial_re_;
    mutable std::vector<double> scratch_partial_im_;
};

using CpuBackend = BasicCpuBackend<Complex>;

// Whether B is a host backend (of any Scalar): the lanes copy to and from host memory only
// for the others.
template <class B>
inline constexpr bool is_cpu_backend_v = false;
template <class Scalar>
inline constexpr bool is_cpu_backend_v<BasicCpuBackend<Scalar>> = true;

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
