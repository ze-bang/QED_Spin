#pragma once
// =============================================================================
// include/ed/matvec/basis_policy.h
//
// Basis policies: compile-time-known descriptions of how a
// (array-index, bitstring) pairing works in a given Hilbert subspace. They
// are the second template argument of the unified term kernel
// (term_kernels.h). This header defines FullBasisPolicy (full Hilbert
// space). The symmetry sectors use RepSymmetryBasisPolicy with its own
// kernels, see rep_symmetry_basis_policy.h.
//
// A basis policy is a small value-type that exposes:
//
//   uint64_t dim() const noexcept
//       Number of basis states (== size of the in/out arrays).
//
//   uint64_t state_of(uint64_t idx) const noexcept
//       Bitstring at array position `idx`. Must be valid for idx < dim().
//
//   int64_t  index_of(uint64_t state) const noexcept
//       Array position of `state`, or -1 if `state` is not in this basis.
//
//   static constexpr bool may_leave_basis
//       Compile-time hint: do off-diagonal terms ever produce a state
//       outside this basis? `false` for the full Hilbert space (every
//       length-N bitstring is in the basis); `true` for any restricted
//       basis. The term kernel uses this to skip the
//       `index_of() >= 0` check for the full basis at zero runtime cost.
//
//   static constexpr bool needs_orbit_walk / has_coeff_modifier
//       Both false here: one computational state per row, no per-emit
//       projection factor.
//
// Policies are passed by value (or by `const&`) to the term kernel; they
// hold POD or pointers-to-POD and are trivially copyable so the kernel
// can keep them in registers across the inner loops.
// =============================================================================

#include <cstdint>
#include <complex>
#include <utility>
#include <vector>

#include <ed/basis/bits.h>
#include <ed/basis/combinadic.h>   // BinomialTable, rank_state/unrank_to_state (tableless mode)

namespace ed::matvec::basis {

// ---------------------------------------------------------------------------
// Full Hilbert space basis: array index == bitstring.
// ---------------------------------------------------------------------------
struct FullBasisPolicy {
    uint64_t n_bits;

    [[nodiscard]] inline uint64_t dim() const noexcept { return 1ULL << n_bits; }
    [[nodiscard]] inline uint64_t state_of(uint64_t idx) const noexcept { return idx; }
    [[nodiscard]] inline int64_t index_of(uint64_t state) const noexcept {
        // The full basis contains every bitstring of length n_bits.
        // Out-of-range is technically impossible if the caller is well-
        // behaved (term application never produces > 2^n_bits), but the
        // check keeps the documented index_of contract (-1 for states
        // outside the basis).
        return state < (1ULL << n_bits) ? static_cast<int64_t>(state) : -1;
    }

    static constexpr bool may_leave_basis = false;

    // One computational state per row, no per-emit projection factor.
    static constexpr bool needs_orbit_walk = false;
    static constexpr bool has_coeff_modifier = false;
};

// ---------------------------------------------------------------------------
// Convenience factory. Saves callers from spelling out the field types
// each time they need a fresh view.
// ---------------------------------------------------------------------------
[[nodiscard]] inline FullBasisPolicy make_full_basis(uint64_t n_bits) noexcept { return FullBasisPolicy{n_bits}; }

} // namespace ed::matvec::basis
