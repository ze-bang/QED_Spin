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
#include <vector>

#include <ed/matvec/matvec.h>
#include <ed/matvec/memory_space.h>

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

    /// Phase 2 of the "Unified CPU/GPU symmetry architecture" plan
    /// (May 2026). Decouples DEVICE CAPABILITY from STORAGE: when
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
    /// Default `false` keeps every existing host-only operator
    /// unchanged. ``SectorOperator`` / ``FixedSzOperator`` flip this to
    /// `true` on CUDA builds (their bind_cuda builds the device rep
    /// mirror).
    bool                  supports_device_matvec = false;

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
    /// for free. Override when the rank-local offset differs from the
    /// trivial single-rank value.
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
    // to the legacy `apply()` so an operator that doesn't yet specialise
    // still works.
    // -------------------------------------------------------------------

    using MatvecFn = std::function<void(const Complex*, Complex*, std::size_t)>;
    /// Real-valued matvec lambda --- only meaningful when
    /// ``is_real_hermitian() == true``. Default implementation throws
    /// to make sure the orchestrator never silently routes a complex
    /// operator through the real-only Lanczos lane.
    using RealMatvecFn =
        std::function<void(const double*, double*, std::size_t)>;

    [[nodiscard]] virtual MatvecFn bind_cpu() const {
        return [this](const Complex* in, Complex* out, std::size_t n) {
            this->apply(in, out, n);
        };
    }
    [[nodiscard]] virtual MatvecFn bind_cuda() const { return bind_cpu(); }

    // -------------------------------------------------------------------
    // Single-precision (fp32) CUDA device matvec binding --- the memory-
    // halving lane that lets the full 2^32 Hilbert space fit two vectors
    // on one 80 GB H100 for mTPQ. The in/out pointers are
    // ``cuFloatComplex*`` in DEVICE memory, erased to ``void*`` so this
    // header (consumed by CPU-only TUs too) need not include
    // ``<cuComplex.h>``. Only ``Operator`` (full-Hilbert) overrides it;
    // every other operator reports ``supports_cuda_f32() == false`` and
    // the default binding throws. Consumed by ``ed::thermal::mtpq_f32``.
    // -------------------------------------------------------------------
    using Fp32DeviceMatvecFn =
        std::function<void(const void*, void*, std::size_t)>;

    [[nodiscard]] virtual bool supports_cuda_f32() const noexcept {
        return false;
    }
    [[nodiscard]] virtual Fp32DeviceMatvecFn bind_cuda_f32() const {
        throw std::runtime_error(
            "LinearOperator::bind_cuda_f32: this operator has no fp32 "
            "device matvec (only the full-Hilbert Operator supports it)");
    }

    // -------------------------------------------------------------------
    // Wave 1.1 of the SOTA Performance rollout (May 2026): orchestrator
    // real-Hermitian fast-path dispatch.
    //
    // Every solver in the project consumes a complex matvec by default
    // because the matvec abstraction (`MatVecOperator::apply`) is
    // complex-valued. However a large fraction of production workloads
    // (Heisenberg, t-J, Hubbard with real hoppings, all real spin
    // chains) are real-Hermitian, and the legacy `lanczos_real` lane
    // (`src/solvers/cpu/lanczos.cpp:1110-1258`) is 30-50% faster than
    // the unified complex `lanczos_kernel<CpuBackend>` thanks to
    // fused BLAS-1 and a native-double recurrence.
    //
    // The two virtuals below let the orchestrator detect such cases
    // and dispatch. The defaults are conservative: ``is_real_hermitian``
    // returns false, ``bind_real_cpu`` throws. Concrete subclasses
    // (notably ``Operator``, ``SectorView``, ``DistributedOperator``,
    // ``GPUOperator``) override only when their internal storage
    // genuinely supports a `double*`-typed apply.
    // -------------------------------------------------------------------

    /// Whether the operator is both real-coefficient AND Hermitian (so
    /// the Lanczos algorithm is exact under real arithmetic). Default
    /// is ``false`` -- orchestrators take the standard complex path.
    [[nodiscard]] virtual bool is_real_hermitian() const noexcept {
        return false;
    }

    /// Real-only matvec binding. Only legal to call when
    /// ``is_real_hermitian() == true``. Defaults to a wrapper that
    /// shuttles real to complex through ``apply()``; subclasses that
    /// have a native real path (``Operator::apply_real``) override.
    [[nodiscard]] virtual RealMatvecFn bind_real_cpu() const {
        return [this](const double* in, double* out, std::size_t n) {
            std::vector<Complex> cin(n);
            std::vector<Complex> cout(n);
            for (std::size_t i = 0; i < n; ++i) {
                cin[i] = Complex(in[i], 0.0);
            }
            this->apply(cin.data(), cout.data(), n);
            for (std::size_t i = 0; i < n; ++i) {
                out[i] = cout[i].real();
            }
        };
    }

    /// CUDA real-only matvec binding. Default routes through bind_real_cpu;
    /// real GPU lanes land in Wave 5 of the unification plan.
    [[nodiscard]] virtual RealMatvecFn bind_real_cuda() const {
        return bind_real_cpu();
    }

    // -------------------------------------------------------------------
    // Wave C2 (May 2026): batched multi-column matvec.
    //
    // Block solvers (KPM Chebyshev moments over R random vectors,
    // FTLM block-Lanczos, GMRES variants) benefit from amortising
    // any per-matvec overhead (orbit walk, term-table prefetch, OMP
    // team spin-up) across a "batch" of input columns processed
    // simultaneously. Default implementations just loop the existing
    // single-column ``apply`` / ``apply_real`` so every operator
    // already supports the API; operators with a faster batched
    // path (notably ``StreamingSymmetryOperator::SectorView`` which
    // can share orbit traversal) override.
    //
    // Buffers are laid out as ``in_block[col * dim + i]`` and
    // ``out_block[col * dim + i]`` (column-major contiguous).
    // -------------------------------------------------------------------

    /// Complex batched matvec. Defaults to ``batch`` calls to
    /// ``apply()``. Override when there is a faster per-batch path.
    virtual void apply_batch(const Complex* in_block,
                             Complex* out_block,
                             std::size_t dim,
                             std::size_t batch) const {
        for (std::size_t b = 0; b < batch; ++b) {
            this->apply(in_block + b * dim, out_block + b * dim, dim);
        }
    }

    /// Real batched matvec. Only legal when ``is_real_hermitian()``.
    /// Default routes each column through ``bind_real_cpu()``'s
    /// single-column lane.
    virtual void apply_batch_real(const double* in_block,
                                  double* out_block,
                                  std::size_t dim,
                                  std::size_t batch) const {
        auto mv = this->bind_real_cpu();
        for (std::size_t b = 0; b < batch; ++b) {
            mv(in_block + b * dim, out_block + b * dim, dim);
        }
    }

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
