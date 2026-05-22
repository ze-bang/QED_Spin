#pragma once
// =============================================================================
// include/ed/matvec/basis_policy.h
//
// Basis policies: compile-time-known descriptions of how a
// (array-index, bitstring) pairing works in a given Hilbert subspace. They
// are the second template argument of the unified term kernel
// (term_kernels.h) and the *only* thing that distinguishes:
//
//   * full Hilbert space matvec
//   * fixed total-Sz sector matvec
//   * (eventually) symmetry-projected sector matvec
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
//       length-N bitstring is in the basis); `true` for fixed-Sz (S+/S-
//       changes the popcount). The term kernel uses this to skip the
//       `index_of() >= 0` check for the full basis at zero runtime cost.
//
// Policies are passed by value (or by `const&`) to the term kernel; they
// hold POD or pointers-to-POD and are trivially copyable so the kernel
// can keep them in registers across the inner loops. The Hamiltonian
// class owns the backing arrays (basis_states_, lin_index_); the policy
// is constructed once per `apply()` call as a thin view onto them.
//
// Phase 1 of the matvec-unification revamp.
// =============================================================================

#include <cstdint>
#include <vector>

#include <ed/core/basis_utils.h>  // LinIndexTable

namespace ed::matvec::basis {

// ---------------------------------------------------------------------------
// Full Hilbert space basis: array index == bitstring.
// ---------------------------------------------------------------------------
struct FullBasisPolicy {
    uint64_t n_bits;

    [[nodiscard]] inline uint64_t dim() const noexcept {
        return 1ULL << n_bits;
    }
    [[nodiscard]] inline uint64_t state_of(uint64_t idx) const noexcept {
        return idx;
    }
    [[nodiscard]] inline int64_t index_of(uint64_t state) const noexcept {
        // The full basis contains every bitstring of length n_bits.
        // Out-of-range is technically impossible if the caller is well-
        // behaved (term application never produces > 2^n_bits), but we
        // keep the check so the kernel signature is identical to the
        // fixed-Sz one (lets the compiler dedup template instantiations).
        return state < (1ULL << n_bits) ? static_cast<int64_t>(state) : -1;
    }

    static constexpr bool may_leave_basis = false;
};

// ---------------------------------------------------------------------------
// Fixed total-Sz sector basis. The backing arrays live on the owning
// Hamiltonian (FixedSzOperator); the policy is a non-owning view.
// ---------------------------------------------------------------------------
struct FixedSzBasisPolicy {
    const std::vector<uint64_t>* basis_states;  // sorted, length C(N, n_up)
    const LinIndexTable*         lin_index;     // Lin (1990) lookup
    uint64_t                     dim_;          // cached basis_states->size()

    [[nodiscard]] inline uint64_t dim() const noexcept {
        return dim_;
    }
    [[nodiscard]] inline uint64_t state_of(uint64_t idx) const noexcept {
        return (*basis_states)[idx];
    }
    [[nodiscard]] inline int64_t index_of(uint64_t state) const noexcept {
        return lin_index->lookup(state);
    }

    static constexpr bool may_leave_basis = true;
};

// ---------------------------------------------------------------------------
// Convenience factories. Saves callers from spelling out the field types
// each time they need a fresh view.
// ---------------------------------------------------------------------------
[[nodiscard]] inline FullBasisPolicy make_full_basis(uint64_t n_bits) noexcept {
    return FullBasisPolicy{n_bits};
}

[[nodiscard]] inline FixedSzBasisPolicy make_fixed_sz_basis(
    const std::vector<uint64_t>& basis_states,
    const LinIndexTable&         lin_index) noexcept
{
    return FixedSzBasisPolicy{
        &basis_states, &lin_index, basis_states.size()
    };
}

} // namespace ed::matvec::basis
