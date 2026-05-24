#pragma once
// =============================================================================
// include/ed/core/linear_operator.h
//
// LinearOperator: the unified operator concept consumed by the Phase-4
// orchestrators (`ed::solve`, `ed::thermal`, `ed::spectral`). Folds the
// existing `MatVecOperator` polymorphic apply with the geometry + binding
// metadata an orchestrator needs to pick a Backend at runtime.
//
// Design:
//   * `LinearOperator` derives from `ed::matvec::MatVecOperator`, so every
//     existing concrete operator (`MatVecOperator`, `DistributedOperator`,
//     `GPUOperator`, `DistributedGPUOperator`, ...) remains usable via the
//     legacy `MatVecOperator*` API; the new entry points just need them
//     to also expose `geometry()` and the matching `bind<Backend>` lane.
//   * `Geometry` captures every piece of metadata `ed::select_backend`
//     reads to choose a backend (rank-local dim, global dim, MPI comm,
//     memory space).
//   * `bind<Backend>` returns a `std::function` matching the matvec
//     signature kernels already consume (`(const Complex*, Complex*,
//     std::size_t) -> void`). The default implementation is a thin
//     wrapper over `apply()` --- concrete operators with a faster
//     backend-specialised path override the specific lane.
//
// Phase 3.1 of the Minimalist ED Collapse (May 2026).
// =============================================================================

#include <complex>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>

#include <ed/matvec/matvec.h>
#include <ed/matvec/memory_space.h>

#ifdef WITH_MPI
#include <mpi.h>
#endif

namespace ed {

using Complex = std::complex<double>;

// ---------------------------------------------------------------------------
// Geometry --- the geometry/runtime metadata a Phase-4 orchestrator
// needs to pick a Backend. Constructible from a `MatVecOperator*` for
// the non-distributed cases (single-rank or single-GPU operators).
// ---------------------------------------------------------------------------
struct Geometry {
    /// Rank-local dimension (= global_dim for single-rank cases).
    std::size_t           local_dim    = 0;
    /// Global Hilbert-space dimension across all ranks.
    std::uint64_t         global_dim   = 0;
    /// Offset of this rank's slab in the global ordering. 0 on single-rank.
    std::uint64_t         local_offset = 0;
    /// Where the apply() expects its buffers to live.
    ed::matvec::MemorySpace memory_space = ed::matvec::MemorySpace::Host;

#ifdef WITH_MPI
    /// MPI communicator the operator runs on. `MPI_COMM_NULL` means the
    /// operator is single-rank (no MPI).
    MPI_Comm              comm         = MPI_COMM_NULL;
#endif

    [[nodiscard]] bool is_distributed() const noexcept {
        return ed::matvec::is_distributed(memory_space);
    }
    [[nodiscard]] bool is_device() const noexcept {
        return ed::matvec::is_device(memory_space);
    }
};

// ---------------------------------------------------------------------------
// LinearOperator
// ---------------------------------------------------------------------------
class LinearOperator : public ed::matvec::MatVecOperator {
public:
    /// Geometry + memory-space metadata used by `ed::select_backend`.
    /// Default implementation derives geometry from the existing
    /// `MatVecOperator` getters (dim / global_dim / memory_space), so
    /// every existing operator becomes a single-rank `LinearOperator`
    /// for free. Override when the rank-local offset / MPI comm
    /// differ from the trivial single-rank values.
    [[nodiscard]] virtual Geometry geometry() const {
        Geometry g;
        g.local_dim    = this->dim();
        g.global_dim   = this->global_dim();
        g.local_offset = 0;
        g.memory_space = this->memory_space();
#ifdef WITH_MPI
        g.comm         = MPI_COMM_NULL;
#endif
        return g;
    }

    // -------------------------------------------------------------------
    // bind<Backend> --- return a callable matching the matvec signature
    // every kernel in the project consumes:
    //   void(const Complex*, Complex*, std::size_t)
    //
    // The default just calls apply(). Concrete operators with a faster
    // backend-specialised path override the appropriate overload below.
    // The template is non-virtual; specialisation happens via the
    // backend-tagged virtual hooks `bind_cpu`, `bind_cuda`, `bind_mpi`,
    // `bind_mpi_cuda`. Each defaults to the legacy `apply()` so an
    // operator that doesn't yet specialise still works.
    // -------------------------------------------------------------------

    using MatvecFn = std::function<void(const Complex*, Complex*, std::size_t)>;

    [[nodiscard]] virtual MatvecFn bind_cpu() const {
        return [this](const Complex* in, Complex* out, std::size_t n) {
            this->apply(in, out, n);
        };
    }
    [[nodiscard]] virtual MatvecFn bind_cuda() const { return bind_cpu(); }
    [[nodiscard]] virtual MatvecFn bind_mpi() const { return bind_cpu(); }
    [[nodiscard]] virtual MatvecFn bind_mpi_cuda() const { return bind_cpu(); }

    /// Tagged dispatch helper. The template indirection makes calls
    /// from the orchestrators read more naturally:
    ///     auto mv = op.bind<CpuBackend>();
    /// The implementation matches on the backend type's
    /// `MemorySpace` and forwards to the corresponding virtual hook.
    template <typename Backend>
    [[nodiscard]] MatvecFn bind() const;
};

}  // namespace ed

// Specialisations of `bind` live at the bottom so concrete Backend types
// (CpuBackend, CudaBackend, MpiBackend, MpiCudaBackend) declared in
// `ed/matvec/backends/*.h` don't pull this header into a dependency cycle.
#include <ed/matvec/backends/cpu_backend.h>
#ifdef WITH_CUDA
#include <ed/matvec/backends/cuda_backend.cuh>
#endif
#ifdef WITH_MPI
#include <ed/matvec/backends/mpi_backend.h>
#  ifdef ED_HAVE_NCCL
#  include <ed/matvec/backends/mpi_cuda_backend.cuh>
#  endif
#endif

namespace ed {

template <>
inline LinearOperator::MatvecFn LinearOperator::bind<ed::matvec::CpuBackend>() const {
    return bind_cpu();
}

#ifdef WITH_CUDA
template <>
inline LinearOperator::MatvecFn LinearOperator::bind<ed::matvec::CudaBackend>() const {
    return bind_cuda();
}
#endif

#ifdef WITH_MPI
template <>
inline LinearOperator::MatvecFn LinearOperator::bind<ed::matvec::MpiBackend>() const {
    return bind_mpi();
}
#  ifdef ED_HAVE_NCCL
template <>
inline LinearOperator::MatvecFn LinearOperator::bind<ed::matvec::MpiCudaBackend>() const {
    return bind_mpi_cuda();
}
#  endif
#endif

}  // namespace ed
