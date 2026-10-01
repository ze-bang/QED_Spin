#pragma once
// =============================================================================
// tests/common/dense_operator.h -- test-only operators for the block lanes.
//
//   DenseOperator     an explicit matrix: the host apply is a dense mat-vec; on CUDA builds
//                     it has a device kernel (a device copy of the matrix and ZGEMM), so the
//                     lanes can be run on CudaBackend against the same block.
//   ThrowingOperator  wraps another operator and throws ed::ResourceLimit on its k-th
//                     apply, on either backend: the lanes must let it through.
// =============================================================================

#include <Eigen/Dense>

#include <ed/core/errors.h>
#include <ed/core/linear_operator.h>
#ifdef WITH_CUDA
#include <ed/matvec/backends/cuda_backend.cuh>
#endif

#include <complex>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>

namespace ed_tests {

class DenseOperator final : public ed::LinearOperator {
public:
    explicit DenseOperator(Eigen::MatrixXcd A) : A_(std::move(A)) {}

    void apply(const Complex* in, Complex* out, std::size_t n) const override {
        Eigen::Map<const Eigen::VectorXcd> x(in, static_cast<Eigen::Index>(n));
        Eigen::Map<Eigen::VectorXcd> y(out, static_cast<Eigen::Index>(n));
        y.noalias() = A_ * x;
    }
    [[nodiscard]] std::size_t dim() const override { return static_cast<std::size_t>(A_.rows()); }
    [[nodiscard]] std::string description() const override { return "DenseOperator"; }
    [[nodiscard]] const Eigen::MatrixXcd& matrix() const noexcept { return A_; }

#ifdef WITH_CUDA
    [[nodiscard]] bool has_device_kernel() const override { return true; }
    [[nodiscard]] MatvecFn bind_cuda() const override {
        struct State {
            ed::matvec::CudaBackend be;
            ed::matvec::Backend::UniqueVec A;
            explicit State(const Eigen::MatrixXcd& M)
                : A(be.make_zero_vector(static_cast<std::size_t>(M.size()))) {
                be.copy_from_host(M.data(), A.get(), static_cast<std::size_t>(M.size()));   // column-major
            }
        };
        auto st = std::make_shared<State>(A_);
        return [st](const Complex* in, Complex* out, std::size_t n) {
            st->be.gemm('N', 'N', n, 1, n, Complex(1, 0), st->A.get(), n, in, n, Complex(0, 0), out, n);
        };
    }
#endif

private:
    Eigen::MatrixXcd A_;
};

class ThrowingOperator final : public ed::LinearOperator {
public:
    ThrowingOperator(std::shared_ptr<const ed::LinearOperator> inner, std::uint64_t throw_at)
        : inner_(std::move(inner)), throw_at_(throw_at) {}

    void apply(const Complex* in, Complex* out, std::size_t n) const override {
        tick();
        inner_->apply(in, out, n);
    }
    [[nodiscard]] std::size_t dim() const override { return inner_->dim(); }
    [[nodiscard]] std::string description() const override { return "ThrowingOperator"; }
    [[nodiscard]] bool has_device_kernel() const override { return inner_->has_device_kernel(); }
    [[nodiscard]] MatvecFn bind_cuda() const override {
        auto f = inner_->bind_cuda();
        return [this, f](const Complex* in, Complex* out, std::size_t n) {
            tick();
            f(in, out, n);
        };
    }

private:
    void tick() const {
        if (++applies_ >= throw_at_)
            throw ed::ResourceLimit("ThrowingOperator: apply " + std::to_string(applies_));
    }
    std::shared_ptr<const ed::LinearOperator> inner_;
    std::uint64_t throw_at_;
    mutable std::uint64_t applies_ = 0;
};

}  // namespace ed_tests
