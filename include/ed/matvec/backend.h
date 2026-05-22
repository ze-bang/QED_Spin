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
// =============================================================================

#include <complex>
#include <cstddef>
#include <memory>
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
