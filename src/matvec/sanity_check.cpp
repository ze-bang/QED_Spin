// =============================================================================
// src/matvec/sanity_check.cpp
//
// Phase 1 sanity TU: verify the new matvec layer compiles and links by
// including every public header and instantiating the adapter once. This
// file does NOT replace any unit test; it just guarantees the headers
// stay buildable as Phase 2/3 progress.
// =============================================================================

#include <ed/matvec/backend.h>
#include <ed/matvec/backends/cpu_backend.h>
#include <ed/matvec/basis_policy.h>
#include <ed/matvec/matvec.h>
#include <ed/matvec/memory_space.h>
#include <ed/matvec/operator_adapter.h>
#include <ed/matvec/term_kernels.h>

namespace ed::matvec::detail {

// Tiny non-virtual sanity wrapper so we never link this into the public
// surface. Compiler-checked only --- it forces every header to be parsed
// and every template instantiation we care about for Phase 1 to fire.
[[maybe_unused]] inline void compile_check(const Operator& op,
                                           const FixedSzOperator& fz_op) {
    // Adapter construction.
    auto a1 = adapt(op);
    auto a2 = adapt(fz_op);
    (void)a1->dim();
    (void)a2->dim();
    (void)a1->memory_space();
    (void)default_cpu_backend().memory_space();

    // Force one instantiation of the unified term kernel against each
    // basis policy and each scalar type. We never *call* this --- the
    // compiler just needs to see that every duck-typed field access
    // resolves cleanly against the Operator SoA term structs. This
    // catches divergence between the kernel's expected schema and the
    // Operator class as Phase 2/3 progress.
    const auto& diag1 = op.diag_one_body_;
    const auto& off1  = op.offdiag_one_body_;
    const auto& diag2 = op.diag_two_body_;
    const auto& mix2  = op.mixed_two_body_;
    const auto& off2  = op.offdiag_two_body_;
    const auto& tri   = op.three_body_data_;

    // Full basis, complex.
    basis::FullBasisPolicy full = basis::make_full_basis(op.getNumBits());
    std::complex<double> dummy_in{}, dummy_out{};
    kernel::apply_terms<basis::FullBasisPolicy, std::complex<double>>(
        full, op.getSpin(), diag1, off1, diag2, mix2, off2, tri,
        &dummy_in, &dummy_out);

    // Full basis, real (the fast path).
    double dr_in{}, dr_out{};
    kernel::apply_terms<basis::FullBasisPolicy, double>(
        full, op.getSpin(), diag1, off1, diag2, mix2, off2, tri,
        &dr_in, &dr_out);

    // Fixed-Sz basis, complex.
    auto fz = basis::make_fixed_sz_basis(fz_op.getBasisStates(), fz_op.lin_index_table());
    kernel::apply_terms<basis::FixedSzBasisPolicy, std::complex<double>>(
        fz, fz_op.getSpin(), diag1, off1, diag2, mix2, off2, tri,
        &dummy_in, &dummy_out);
}

} // namespace ed::matvec::detail
