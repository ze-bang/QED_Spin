#pragma once
// =============================================================================
// include/ed/matvec/linear_operator.h
//
// LinearOperator: the one operator interface every solver consumes. A concrete
// operator supplies the host apply and its dimension; the solvers pair it with
// a Backend (axpy/dot/norm/scale/copy) through bind<Backend>().
//
//   * apply(in, out, n): out = A in on HOST buffers; out is overwritten.
//   * has_device_kernel(): the operator can bind a device apply (bind_cuda).
//     The default bind_cuda() throws ed::DeviceUnsupported instead of handing
//     the host apply device pointers.
//   * bind<CpuBackend>() -> bind_cpu(), bind<CudaBackend>() -> bind_cuda(),
//     bind<BasicCpuBackend<double>>() -> bind_cpu_real() for an operator that is_real().
//
// Internal kernels stay templated (term_kernels.h), so the virtual call costs
// one indirection per apply.
// =============================================================================

#include <complex>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>

#include <ed/core/errors.h>

namespace ed {

using Complex = std::complex<double>;

class LinearOperator {
public:
    virtual ~LinearOperator() = default;

    /// out = A in. Buffers are host memory, distinct, of dim() elements.
    virtual void apply(const Complex* in, Complex* out, std::size_t n) const = 0;
    [[nodiscard]] virtual std::size_t dim() const = 0;
    [[nodiscard]] virtual bool is_hermitian() const { return true; }
    /// An upper bound of ||A||_2 (the scale the solvers' tolerances are relative to,
    /// <ed/core/numerics.h>), or 0 when unknown.
    [[nodiscard]] virtual double norm_bound() const { return 0.0; }
    [[nodiscard]] virtual std::string description() const { return "LinearOperator"; }

    /// bind_cuda() returns a device apply.
    [[nodiscard]] virtual bool has_device_kernel() const { return false; }

    using MatvecFn = std::function<void(const Complex*, Complex*, std::size_t)>;

    [[nodiscard]] virtual MatvecFn bind_cpu() const {
        return [this](const Complex* in, Complex* out, std::size_t n) {
            this->apply(in, out, n);
        };
    }
    [[nodiscard]] virtual MatvecFn bind_cuda() const {
        throw ed::DeviceUnsupported(description() + " has no device kernel");
    }

    /// The operator is real in its basis (real matrix elements): bind_cpu_real() applies it to
    /// real host vectors, so a real symmetric eigenproblem runs in real arithmetic. May build the
    /// operator's representation (it answers from it).
    [[nodiscard]] virtual bool is_real() const { return false; }
    using RealMatvecFn = std::function<void(const double*, double*, std::size_t)>;
    [[nodiscard]] virtual RealMatvecFn bind_cpu_real() const {
        throw ed::Unsupported(description() + " is not real");
    }

    /// k vectors per call on the device: outs[i] = A ins[i] (device pointers, each of dim()).
    /// Operators whose device kernel can serve several vectors in one pass return it; the
    /// default (empty) means callers apply the vectors one at a time.
    using MultiMatvecFn = std::function<void(const Complex* const* ins, Complex* const* outs,
                                             std::size_t n, std::size_t k)>;
    [[nodiscard]] virtual MultiMatvecFn bind_cuda_multi() const { return {}; }

    /// The apply on vectors of a Backend: bind<CpuBackend>() is bind_cpu(), bind<CudaBackend>()
    /// bind_cuda(), bind<BasicCpuBackend<double>>() bind_cpu_real().
    template <typename Backend>
    using BoundFn = std::function<void(const typename Backend::scalar_type*, typename Backend::scalar_type*,
                                       std::size_t)>;
    template <typename Backend>
    [[nodiscard]] BoundFn<Backend> bind() const;
};

}  // namespace ed

// Specialisations of `bind` live at the bottom so concrete Backend types
// (CpuBackend, CudaBackend) declared in `ed/matvec/cpu_backend.h` / `ed/gpu/cuda_backend.cuh` don't
// pull this header into a dependency cycle.
#include <ed/matvec/cpu_backend.h>
#ifdef WITH_CUDA
#include <ed/gpu/cuda_backend.cuh>
#endif

namespace ed {

template <>
inline LinearOperator::MatvecFn LinearOperator::bind<ed::matvec::CpuBackend>() const {
    return bind_cpu();
}

template <>
inline LinearOperator::RealMatvecFn LinearOperator::bind<ed::matvec::BasicCpuBackend<double>>() const {
    return bind_cpu_real();
}

#ifdef WITH_CUDA
template <>
inline LinearOperator::MatvecFn LinearOperator::bind<ed::matvec::CudaBackend>() const {
    return bind_cuda();
}
#endif

}  // namespace ed
