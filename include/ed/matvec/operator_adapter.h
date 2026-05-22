#pragma once
// =============================================================================
// include/ed/matvec/operator_adapter.h
//
// Phase 2 of the matvec-unification revamp: this header used to wrap the
// legacy `Operator` / `FixedSzOperator` classes in adapter shims so they
// could be consumed through the `MatVecOperator` interface. After Phase 2
// `Operator` itself derives from `MatVecOperator` and overrides `apply()`
// / `dim()` / `memory_space()` / `is_hermitian()` / `description()` ---
// so the adapters no longer add anything.
//
// We keep the `adapt(...)` free function as a thin convenience wrapper
// that returns a unique_ptr<MatVecOperator> --- useful when a caller
// wants to own the operator polymorphically but doesn't have the
// concrete type at hand. Concrete `Operator&` / `FixedSzOperator&`
// callers should just pass the reference straight to solvers; there is
// no longer any reason to box it.
// =============================================================================

#include <memory>

#include <ed/core/fixed_sz_operator.h>
#include <ed/core/operator.h>
#include <ed/matvec/matvec.h>

namespace ed::matvec {

// Non-owning adoption: wrap an externally-owned Operator as a
// MatVecOperator without copying. Mostly redundant now that Operator IS
// a MatVecOperator, but useful when a caller wants a unique_ptr to
// hand off to a function expecting one (e.g. a factory return value).
class OperatorRef final : public MatVecOperator {
public:
    explicit OperatorRef(const Operator& op) noexcept : op_(&op) {}

    void apply(const Complex* in, Complex* out, std::size_t size) const override {
        op_->apply(in, out, size);
    }
    [[nodiscard]] std::size_t dim() const override { return op_->dim(); }
    [[nodiscard]] MemorySpace memory_space() const override {
        return op_->memory_space();
    }
    [[nodiscard]] bool is_hermitian() const override {
        return op_->is_hermitian();
    }
    [[nodiscard]] std::string description() const override {
        return "OperatorRef(" + op_->description() + ")";
    }

private:
    const Operator* op_;
};

[[nodiscard]] inline std::unique_ptr<MatVecOperator>
adapt(const Operator& op) {
    return std::make_unique<OperatorRef>(op);
}
// FixedSzOperator IS-A Operator, so the same overload picks it up
// correctly --- the virtual `apply()` / `dim()` will dispatch through
// the vtable. No special overload needed any more.

} // namespace ed::matvec
