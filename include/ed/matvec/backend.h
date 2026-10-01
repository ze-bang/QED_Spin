#pragma once
// =============================================================================
// include/ed/matvec/backend.h
//
// Backend: the vector-primitives half of the matvec layer. While
// MatVecOperator says "how to apply H to a vector", Backend says "how to do
// every *other* linear-algebra primitive that the surrounding Krylov /
// thermal solver needs on that vector": axpy, dot, norm, scale, copy,
// memory allocation.
//
// Backends are paired 1:1 with MemorySpace:
//
//     MemorySpace                Backend implementation
//     ---------------------     -----------------------
//     Host                      CpuBackend
//     CudaDevice                CudaBackend
//
// Why a separate object instead of methods on MatVecOperator? Because
// vector primitives are independent of which Hamiltonian you're applying.
// One backend can drive many different MatVecOperators (e.g. the
// Hamiltonian and an observable, used together in FTLM-style spectral
// kernels).
//
// All operations are synchronous from the caller's point of view: when
// they return, the result is visible. Internally Backends may chain CUDA
// streams, but the API is sync, which is what the solvers assume.
//
// NOTE: the matvec dispatch strategy (matrix-free vs assembled-CSR, real
// vs complex specialisation) lives in a SEPARATE header
// ``ed/matvec/matvec_backend.h``. The two are orthogonal: Backend is the
// vector-primitives backend the solver talks to; MatVecBackendBase
// (CpuMatVecBackend) is the SpMV-kernel strategy the Operator talks to.
// =============================================================================

#include <complex>
#include <cstddef>
#include <memory>
#include <stdexcept>
#include <string>

#include <ed/matvec/memory_space.h>

namespace ed::matvec {

using Complex = std::complex<double>;

// ----------------------------------------------------------------------------
// Backend interface. Concrete backends live in ed/matvec/backends/*.h. The
// interface intentionally takes raw pointers to keep it host/device
// agnostic --- the *meaning* of those pointers (host RAM vs device memory)
// is determined by memory_space().
//
// All vector arguments are dimension `n`.
// ----------------------------------------------------------------------------
class Backend {
public:
    virtual ~Backend() = default;

    // Identity --- which memory space am I responsible for?
    [[nodiscard]] virtual MemorySpace memory_space() const = 0;
    [[nodiscard]] virtual std::string description() const = 0;

    // ------------------------------------------------------------------
    // Memory management. Returned pointers must be released with the
    // matching deallocate() on the same Backend. The unique_ptr helper
    // below handles that pairing for the common case.
    // ------------------------------------------------------------------
    [[nodiscard]] virtual Complex* allocate(std::size_t n) const = 0;
    virtual void deallocate(Complex* p) const noexcept = 0;
    virtual void fill_zero(Complex* p, std::size_t n) const = 0;
    virtual void copy(const Complex* src, Complex* dst, std::size_t n) const = 0;

    // Host <-> backend transfer. Mainly used by I/O code that materialises
    // initial vectors from disk or pushes results out; the hot path
    // touches these rarely.
    virtual void copy_from_host(const Complex* host_src,
                                Complex* backend_dst,
                                std::size_t n) const = 0;
    virtual void copy_to_host(const Complex* backend_src,
                              Complex* host_dst,
                              std::size_t n) const = 0;

    // ------------------------------------------------------------------
    // Level-1 BLAS primitives, complex-double. Naming mirrors BLAS.
    //   axpy:  y <- alpha * x + y
    //   scale: x <- alpha * x
    //   dot:   returns x^H * y   (conj on left)
    //   nrm2:  returns ||x||_2
    // ------------------------------------------------------------------
    virtual void   axpy(Complex alpha, const Complex* x, Complex* y, std::size_t n) const = 0;
    virtual void   scale(Complex alpha, Complex* x, std::size_t n) const = 0;
    [[nodiscard]] virtual Complex dot(const Complex* x, const Complex* y, std::size_t n) const = 0;
    [[nodiscard]] virtual double  nrm2(const Complex* x, std::size_t n) const = 0;

    // Convenience: y <- alpha*x + beta*y in one pass (saves one stream
    // through y for fused-update inner loops in Lanczos & TPQ).
    virtual void axpby(Complex alpha, const Complex* x,
                       Complex beta,  Complex* y, std::size_t n) const = 0;

    // ------------------------------------------------------------------
    // Fused Lanczos-recurrence primitives.
    // One streaming pass each instead of two:
    //   axpy_dot : y <- y + alpha*x ; returns z^H * y      (reduced)
    //   axpy_nrm2: y <- y + alpha*x ; returns ||y||_2      (reduced)
    // The defaults compose the primitives above so every backend stays
    // correct; CpuBackend overrides them with single-pass
    // kernels (`lanczos_kernel` issues three fused calls per
    // iteration instead of seven separate BLAS-1 calls).
    // ------------------------------------------------------------------
    [[nodiscard]] virtual Complex axpy_dot(Complex alpha, const Complex* x, Complex* y,
                                           const Complex* z, std::size_t n) const {
        axpy(alpha, x, y, n);
        return dot(z, y, n);
    }
    [[nodiscard]] virtual double axpy_nrm2(Complex alpha, const Complex* x, Complex* y,
                                           std::size_t n) const {
        axpy(alpha, x, y, n);
        return nrm2(y, n);
    }

    // ------------------------------------------------------------------
    // Batched BLAS-1 primitives used by classical Gram-Schmidt-2 (CGS2)
    // reorthogonalisation. The "many" forms amortise the cache-residency
    // of `v` (one streaming pass over `v` feeds k inner dots, instead of
    // k streaming passes for k single-pair calls).
    //
    // The default implementations loop over the single-pair calls,
    // trading reduction amortisation for simplicity. Concrete backends
    // override them for the batched fast path (CpuBackend uses a
    // single OpenMP region; CudaBackend routes through cuBLAS gemv).
    // ------------------------------------------------------------------

    /// Compute `coeffs_out[k] = <basis[k], v>` for k in [0, num_basis).
    /// `basis[k]` and `v` are dimension-`n` vectors. The default impl
    /// loops over `dot()`.
    virtual void dot_many(const Complex* const* basis,
                          std::size_t           num_basis,
                          const Complex*        v,
                          std::size_t           n,
                          Complex*              coeffs_out) const {
        for (std::size_t k = 0; k < num_basis; ++k) {
            coeffs_out[k] = dot(basis[k], v, n);
        }
    }

    /// Compute `v += sum_k alphas[k] * basis[k]`. No reductions
    /// involved (axpy is a local operation). The default impl loops
    /// over `axpy()`.
    virtual void axpy_many(const Complex*        alphas,
                           const Complex* const* basis,
                           std::size_t           num_basis,
                           Complex*              v,
                           std::size_t           n) const {
        for (std::size_t k = 0; k < num_basis; ++k) {
            axpy(alphas[k], basis[k], v, n);
        }
    }

    // ------------------------------------------------------------------
    // Level-3 BLAS primitives. All matrix arguments are column-major.
    //
    // Defaults throw; CpuBackend and CudaBackend override.
    // ------------------------------------------------------------------

    /// Standard ZGEMM: C = alpha * op(A) * op(B) + beta * C, where
    /// op = 'N' (none), 'T' (transpose), 'C' (conjugate-transpose).
    /// All matrices column-major. `m`, `n`, `k` follow BLAS convention:
    /// op(A) is m x k, op(B) is k x n, C is m x n.
    virtual void gemm(char /*opA*/, char /*opB*/,
                      std::size_t /*m*/, std::size_t /*n*/, std::size_t /*k*/,
                      Complex /*alpha*/,
                      const Complex* /*A*/, std::size_t /*lda*/,
                      const Complex* /*B*/, std::size_t /*ldb*/,
                      Complex /*beta*/,
                      Complex* /*C*/, std::size_t /*ldc*/) const {
        throw std::runtime_error("Backend::gemm not implemented for this backend");
    }

    // Convenience helper: allocate a zero-filled work vector. Many
    // Krylov inner loops need a couple of these per iteration; this is
    // the cleanest way to express it without forcing every backend to
    // ship a Vector wrapper.
    struct Deleter {
        const Backend* be;
        void operator()(Complex* p) const noexcept {
            if (be && p) be->deallocate(p);
        }
    };
    using UniqueVec = std::unique_ptr<Complex, Deleter>;

    [[nodiscard]] UniqueVec make_zero_vector(std::size_t n) const {
        Complex* p = allocate(n);
        fill_zero(p, n);
        return UniqueVec{p, Deleter{this}};
    }
};

} // namespace ed::matvec
