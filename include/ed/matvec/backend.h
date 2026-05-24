#pragma once
// =============================================================================
// include/ed/matvec/backend.h
//
// Backend: the second half of the matvec-unification pair. While
// MatVecOperator says "how to apply H to a vector", Backend says "how to do
// every *other* linear-algebra primitive that the surrounding Krylov /
// thermal solver needs on that vector": axpy, dot, norm, scale, copy,
// memory allocation, reductions.
//
// Backends are paired 1:1 with MemorySpace:
//
//     MemorySpace                Backend implementation
//     ---------------------     -----------------------
//     Host                      CpuBackend
//     CudaDevice                CudaBackend
//     DistributedHost           MpiBackend
//     DistributedCudaDevice     MpiCudaBackend     (NCCL + cuBLAS)
//
// Why a separate object instead of methods on MatVecOperator? Because
// vector primitives are independent of which Hamiltonian you're applying.
// One backend can drive many different MatVecOperators (e.g. the
// Hamiltonian and an observable, used together in FTLM-style spectral
// kernels). The split also lets us swap reduction strategies (NCCL vs
// host-staged MPI) without touching operator code.
//
// All operations are synchronous from the caller's point of view: when
// they return, the result is visible. Internally Backends may chain CUDA
// streams or pipeline MPI requests, but the API is sync. This matches
// what every existing solver in the codebase already assumes.
//
// Phase 1 of the matvec-unification revamp.
//
// NOTE: the matvec dispatch strategy (matrix-free vs assembled-CSR, real
// vs complex specialisation) lives in a SEPARATE header
// ``ed/matvec/matvec_backend.h``. The two are orthogonal: Backend is the
// vector-primitives backend the solver talks to; MatVecBackend is the
// SpMV-kernel strategy the Operator talks to.
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
// All vector arguments are dimension `n` (rank-local for distributed
// backends; the implementation handles global reductions internally).
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
    //   dot:   returns x^H * y   (conj on left, MPI-reduced if distributed)
    //   nrm2:  returns ||x||_2   (MPI-reduced if distributed)
    //   set_value: x[i] <- v for i < n  (used for unit vector seeding)
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
    // Reductions. For non-distributed backends these are no-ops returning
    // their argument; for MPI / NCCL backends they MPI_Allreduce /
    // ncclAllReduce across the communicator.
    // ------------------------------------------------------------------
    [[nodiscard]] virtual Complex all_reduce_sum(Complex v) const { return v; }
    [[nodiscard]] virtual double  all_reduce_sum(double  v) const { return v; }

    /// Vector all-reduce (sum). Used by block-Lanczos to reduce the
    /// rank-local (b x b) Gram blocks. Default is a no-op for
    /// non-distributed backends.
    virtual void all_reduce_sum_vec(Complex* /*buf*/, std::size_t /*count*/) const {}
    virtual void all_reduce_sum_vec(double*  /*buf*/, std::size_t /*count*/) const {}

    // ------------------------------------------------------------------
    // Batched BLAS-1 primitives used by classical Gram-Schmidt-2 (CGS2)
    // reorthogonalisation. The "many" forms exist to amortise both
    //   (1) the cache-residency of `v` (one streaming pass over `v`
    //       feeds k inner dots, instead of k streaming passes for k
    //       single-pair calls), and
    //   (2) the MPI / NCCL reduction count: one Allreduce over a
    //       k-element buffer instead of k Allreduces over scalars
    //       (huge difference for the FTLM tridiagonal builder at
    //       M=100, where it turns ~100 round-trips into 1).
    //
    // Returned coefficients in `dot_many` are already-reduced for
    // distributed backends, matching the single-pair `dot()` contract.
    //
    // Default implementations are provided so that pre-existing
    // concrete Backends compile unchanged; they trade reduction
    // amortisation for code simplicity. Concrete backends override
    // them when they want the batched fast path (CpuBackend uses a
    // single OpenMP region; MpiBackend collapses k Allreduces into
    // 1; future GpuBackend will route through cuBLAS gemv).
    // ------------------------------------------------------------------

    /// Compute `coeffs_out[k] = <basis[k], v>` for k in [0, num_basis).
    /// `basis[k]` and `v` are dimension-`n` rank-local vectors. The
    /// returned coefficients are MPI/NCCL-reduced for distributed
    /// backends. The default impl loops over `dot()`.
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
    // Level-3 BLAS primitives. Added Phase 1 of the Minimalist ED
    // Collapse (May 2026). All matrix arguments are column-major. For
    // distributed backends, `gemm`/`gemv`/`trsm` operate purely on the
    // rank-local slabs --- the caller chains `all_reduce_sum_vec` on
    // the result block when a cross-rank reduction is required (this
    // matches the BLAS-1 contract where `dot()` reduces internally
    // but `axpy()` doesn't, and avoids hardcoding reduction semantics
    // into the wrong primitive).
    //
    // Defaults throw; backends that need to expose BLAS-3 (currently
    // every concrete backend: CpuBackend, CudaBackend, MpiBackend,
    // MpiCudaBackend) override.
    // ------------------------------------------------------------------

    /// Standard ZGEMM: C = alpha * op(A) * op(B) + beta * C, where
    /// op = 'N' (none), 'T' (transpose), 'C' (conjugate-transpose).
    /// All matrices column-major. `m`, `n`, `k` follow BLAS convention:
    /// op(A) is m x k, op(B) is k x n, C is m x n. Strictly local; no
    /// cross-rank reduction.
    virtual void gemm(char /*opA*/, char /*opB*/,
                      std::size_t /*m*/, std::size_t /*n*/, std::size_t /*k*/,
                      Complex /*alpha*/,
                      const Complex* /*A*/, std::size_t /*lda*/,
                      const Complex* /*B*/, std::size_t /*ldb*/,
                      Complex /*beta*/,
                      Complex* /*C*/, std::size_t /*ldc*/) const {
        throw std::runtime_error("Backend::gemm not implemented for this backend");
    }

    /// Standard ZGEMV: y = alpha * op(A) * x + beta * y. op(A) is m x n.
    /// Strictly local for distributed backends.
    virtual void gemv(char /*opA*/,
                      std::size_t /*m*/, std::size_t /*n*/,
                      Complex /*alpha*/,
                      const Complex* /*A*/, std::size_t /*lda*/,
                      const Complex* /*x*/, std::size_t /*incx*/,
                      Complex /*beta*/,
                      Complex* /*y*/, std::size_t /*incy*/) const {
        throw std::runtime_error("Backend::gemv not implemented for this backend");
    }

    /// Triangular solve: solve op(A) X = alpha B  or  X op(A) = alpha B
    /// in-place, writing X to B. A is `ka x ka` upper- or lower-triangular,
    /// B is `m x n`. `side` = 'L' (left) or 'R' (right); `uplo` = 'U' or
    /// 'L'; `transA` = 'N'/'T'/'C'; `diag` = 'N' (non-unit) or 'U' (unit).
    /// Strictly local for distributed backends. Used by the
    /// CholeskyQR-based `qr_thin` to recover Q from B * R^{-1}.
    virtual void trsm(char /*side*/, char /*uplo*/, char /*transA*/, char /*diag*/,
                      std::size_t /*m*/, std::size_t /*n*/,
                      Complex /*alpha*/,
                      const Complex* /*A*/, std::size_t /*lda*/,
                      Complex* /*B*/, std::size_t /*ldb*/) const {
        throw std::runtime_error("Backend::trsm not implemented for this backend");
    }

    /// In-place tall-skinny QR. On entry, `A` is `m_local x b`
    /// column-major (row-slab partition for distributed backends).
    /// On exit, `A` holds Q (orthonormal columns, same partition),
    /// and `R_host` (size `b*b`, column-major) holds the upper-
    /// triangular R block --- identical on every rank for
    /// distributed backends. Internally chains gemm/trsm + Cholesky
    /// + `all_reduce_sum_vec` for distributed paths (CholeskyQR2);
    /// uses LAPACK / cuSolver `geqrf` + `ungqr` for single-rank paths.
    virtual void qr_thin(Complex* /*A*/, std::size_t /*m_local*/, std::size_t /*b*/,
                         Complex* /*R_host*/) const {
        throw std::runtime_error("Backend::qr_thin not implemented for this backend");
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
