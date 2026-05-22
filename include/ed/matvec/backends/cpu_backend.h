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
#include <complex>
#include <cstdlib>
#include <cstring>
#include <new>
#include <stdexcept>
#include <string>

#ifdef _OPENMP
#  include <omp.h>
#endif

#include <ed/matvec/backend.h>
#include <ed/matvec/memory_space.h>

namespace ed::matvec {

class CpuBackend final : public Backend {
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
};

// Singleton accessor. The CPU backend is stateless so a single static
// instance is fine; we hand out references rather than ownership so
// callers don't keep allocating it.
[[nodiscard]] inline CpuBackend& default_cpu_backend() {
    static CpuBackend instance;
    return instance;
}

} // namespace ed::matvec
