#pragma once
// =============================================================================
// include/ed/core/linear_operator.h
//
// LinearOperator: the one operator interface every solver consumes. A concrete
// operator supplies the host apply and its dimension; the solvers pair it with
// a Backend (axpy/dot/norm/scale/copy) through bind<Backend>().
//
//   * apply(in, out, n): out = A in on HOST buffers; out is overwritten.
//   * has_device_kernel(): the operator can bind a device apply (bind_cuda).
//     The default bind_cuda() throws ed::DeviceUnsupported instead of handing
//     the host apply device pointers.
//   * bind<CpuBackend>() -> bind_cpu(), bind<CudaBackend>() -> bind_cuda().
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
    [[nodiscard]] virtual std::string description() const { return "LinearOperator"; }

    /// Optional fast dense assembly: fill the dim() x dim() COLUMN-MAJOR matrix
    /// `dense` (pre-zeroed) from the operator's sparse structure in O(nnz).
    /// Reentrant. false (the default) tells the caller to build the columns
    /// with apply().
    [[nodiscard]] virtual bool try_build_dense_columns(Complex* /*dense*/,
                                                       std::size_t /*n*/) const {
        return false;
    }

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

    /// k vectors per call on the device: outs[i] = A ins[i] (device pointers, each of dim()).
    /// Operators whose device kernel can serve several vectors in one pass return it; the
    /// default (empty) means callers apply the vectors one at a time.
    using MultiMatvecFn = std::function<void(const Complex* const* ins, Complex* const* outs,
                                             std::size_t n, std::size_t k)>;
    [[nodiscard]] virtual MultiMatvecFn bind_cuda_multi() const { return {}; }

    /// bind<CpuBackend>() is bind_cpu(), bind<CudaBackend>() is bind_cuda().
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
