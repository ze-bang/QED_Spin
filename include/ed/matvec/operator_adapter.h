#pragma once
// =============================================================================
// include/ed/matvec/operator_adapter.h
//
// Adapters wrapping the legacy `Operator` / `FixedSzOperator` classes
// as `MatVecOperator` instances. They keep the term storage and
// matvec implementations on the legacy class for now --- the unified
// term kernel (term_kernels.h) takes over in Phase 3 of the revamp
// once we have validated that the new boundary plumbs correctly.
//
// These adapters are non-owning: they hold a reference to a long-lived
// Operator instance. The adapter is meant to be constructed cheaply at
// the dispatch boundary and handed straight to a solver. Construction is
// always on the host side and the matvec runs on the host.
//
// Phase 1 of the matvec-unification revamp.
// =============================================================================

#include <cstddef>
#include <string>

#include <ed/core/fixed_sz_operator.h>
#include <ed/core/operator.h>
#include <ed/matvec/matvec.h>

namespace ed::matvec {

// ----------------------------------------------------------------------------
// Adapter for the full Hilbert space Operator.
// ----------------------------------------------------------------------------
class OperatorAdapter final : public MatVecOperator {
public:
    explicit OperatorAdapter(const Operator& op) noexcept
        : op_(&op) {}

    void apply(const Complex* in, Complex* out, std::size_t size) const override {
        check_size(size);
        op_->apply(in, out, size);
    }
    [[nodiscard]] std::size_t dim() const override {
        return static_cast<std::size_t>(1ULL << op_->getNumBits());
    }
    [[nodiscard]] MemorySpace memory_space() const override {
        return MemorySpace::Host;
    }
    [[nodiscard]] bool is_hermitian() const override {
        return true;
    }
    [[nodiscard]] std::string description() const override {
        return "OperatorAdapter(full, n_bits="
            + std::to_string(op_->getNumBits()) + ")";
    }
    [[nodiscard]] const Operator& underlying() const noexcept { return *op_; }

private:
    const Operator* op_;
};

// ----------------------------------------------------------------------------
// Adapter for the fixed-Sz projected Operator.
// ----------------------------------------------------------------------------
class FixedSzOperatorAdapter final : public MatVecOperator {
public:
    explicit FixedSzOperatorAdapter(const FixedSzOperator& op) noexcept
        : op_(&op) {}

    void apply(const Complex* in, Complex* out, std::size_t size) const override {
        check_size(size);
        op_->apply(in, out, size);
    }
    [[nodiscard]] std::size_t dim() const override {
        return static_cast<std::size_t>(op_->getFixedSzDim());
    }
    [[nodiscard]] MemorySpace memory_space() const override {
        return MemorySpace::Host;
    }
    [[nodiscard]] bool is_hermitian() const override {
        return true;
    }
    [[nodiscard]] std::string description() const override {
        return "FixedSzOperatorAdapter(n_bits="
            + std::to_string(op_->getNumBits())
            + ", dim=" + std::to_string(op_->getFixedSzDim()) + ")";
    }
    [[nodiscard]] const FixedSzOperator& underlying() const noexcept { return *op_; }

private:
    const FixedSzOperator* op_;
};

// ----------------------------------------------------------------------------
// Factory: build the right adapter for an Operator / FixedSzOperator.
//
// Operator is intentionally non-polymorphic (the legacy design pre-dates
// this revamp), so we cannot use dynamic_cast to discriminate. We instead
// rely on function overloading: callers with a FixedSzOperator& pick up
// the more-specific overload; callers with an Operator& get the full-
// space adapter. This is the same trick auto_pilot::solve uses today and
// makes the call site type-safe at compile time. Phase 2 of the revamp
// makes Operator polymorphic and replaces this with a virtual factory.
// ----------------------------------------------------------------------------
[[nodiscard]] inline std::unique_ptr<MatVecOperator>
adapt(const Operator& op) {
    return std::make_unique<OperatorAdapter>(op);
}
[[nodiscard]] inline std::unique_ptr<MatVecOperator>
adapt(const FixedSzOperator& op) {
    return std::make_unique<FixedSzOperatorAdapter>(op);
}

} // namespace ed::matvec
