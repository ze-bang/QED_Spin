#pragma once
// =============================================================================
// include/ed/core/linear_operator.h
//
// LinearOperator: the operator concept consumed by the orchestrators
// (`ed::workflows::solve`, `ed::workflows::thermal`). Combines the
// `MatVecOperator` polymorphic apply with the geometry + binding
// metadata an orchestrator needs to pick a Backend at runtime.
//
// Design:
//   * `LinearOperator` derives from `ed::matvec::MatVecOperator`, so every
//     concrete operator is also usable via the `MatVecOperator*` API; the
//     orchestrators additionally need `geometry()` and the matching
//     `bind<Backend>` lane.
//   * `Geometry` captures every piece of metadata `ed::select_backend`
//     reads to choose a backend (local dim, global dim, memory space).
//   * `bind<Backend>` returns a `std::function` matching the matvec
//     signature kernels already consume (`(const Complex*, Complex*,
//     std::size_t) -> void`). The default implementation is a thin
//     wrapper over `apply()` --- concrete operators with a faster
//     backend-specialised path override the specific lane.
// =============================================================================

#include <complex>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include <ed/matvec/matvec.h>
#include <ed/matvec/memory_space.h>

namespace ed {

using Complex = std::complex<double>;

// ---------------------------------------------------------------------------
// Geometry --- the geometry/runtime metadata an orchestrator needs to
// pick a Backend.
// ---------------------------------------------------------------------------
struct Geometry {
    /// Dimension of the vectors apply() acts on (= global_dim unless the
    /// operator is partitioned).
    std::size_t           local_dim    = 0;
    /// Global Hilbert-space dimension.
    std::uint64_t         global_dim   = 0;
    /// Offset of the local slab in the global ordering (0 when unpartitioned).
    std::uint64_t         local_offset = 0;
    /// Where the apply() expects its buffers to live.
    ed::matvec::MemorySpace memory_space = ed::matvec::MemorySpace::Host;

    /// Decouples DEVICE CAPABILITY from STORAGE: when
    /// `true`, the host operator advertises that it can lazily
    /// promote to a GPU mirror via `bind_cuda()`. `select_backend`
    /// inspects this flag to decide whether to pick `CudaBackend`,
    /// even when `memory_space == Host`.
    ///
    /// The contract for an operator opting in:
    ///   1. `bind_cuda()` must NOT throw -- it must lazily build a
    ///      device mirror on first call and cache it.
    ///   2. The mirror must obey the same semantics as the host
    ///      `apply()` (bit-exact within FP atomic-ordering tol).
    ///
    /// Default `false` (host-only operator). The little-group ``RepSectorMatVec`` flips this to
    /// `true` on CUDA builds (its bind_cuda builds the device rep
    /// mirror).
    bool                  supports_device_matvec = false;

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
    /// Default implementation derives geometry from the
    /// `MatVecOperator` getters (dim / global_dim / memory_space).
    /// Override when the local offset is not zero.
    [[nodiscard]] virtual Geometry geometry() const {
        Geometry g;
        g.local_dim    = this->dim();
        g.global_dim   = this->global_dim();
        g.local_offset = 0;
        g.memory_space = this->memory_space();
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
    // backend-tagged virtual hooks `bind_cpu`, `bind_cuda`. Each defaults
    // to `apply()` so an operator without a specialised path still works.
    // -------------------------------------------------------------------

    using MatvecFn = std::function<void(const Complex*, Complex*, std::size_t)>;

    [[nodiscard]] virtual MatvecFn bind_cpu() const {
        return [this](const Complex* in, Complex* out, std::size_t n) {
            this->apply(in, out, n);
        };
    }
    [[nodiscard]] virtual MatvecFn bind_cuda() const { return bind_cpu(); }

    /// k vectors per call on the device: outs[i] = A ins[i] (device pointers, each of dim()).
    /// Operators whose device kernel can serve several vectors in one pass return it; the
    /// default (empty) means callers apply the vectors one at a time.
    using MultiMatvecFn = std::function<void(const Complex* const* ins, Complex* const* outs,
                                             std::size_t n, std::size_t k)>;
    [[nodiscard]] virtual MultiMatvecFn bind_cuda_multi() const { return {}; }

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
// (CpuBackend, CudaBackend) declared in `ed/matvec/backends/*.h` don't
// pull this header into a dependency cycle.
#include <ed/matvec/backends/cpu_backend.h>
#ifdef WITH_CUDA
#include <ed/matvec/backends/cuda_backend.cuh>
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

}  // namespace ed
