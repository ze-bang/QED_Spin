#pragma once
// =============================================================================
// include/ed/matvec/basis_policy.h
//
// Basis policies: compile-time-known descriptions of how a
// (array-index, bitstring) pairing works in a given Hilbert subspace. They
// are the second template argument of the unified term kernel
// (term_kernels.h) and the *only* thing that distinguishes:
//
//   * full Hilbert space matvec       -- FullBasisPolicy
//   * fixed total-Sz sector matvec    -- FixedSzBasisPolicy
//
// (The symmetry sectors use RepSymmetryBasisPolicy with its own kernels,
// see rep_symmetry_basis_policy.h.)
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
//   static constexpr bool needs_orbit_walk / has_coeff_modifier
//       Both false here: one computational state per row, no per-emit
//       projection factor.
//
// Policies are passed by value (or by `const&`) to the term kernel; they
// hold POD or pointers-to-POD and are trivially copyable so the kernel
// can keep them in registers across the inner loops. The Hamiltonian
// class owns the backing arrays (basis_states_, lin_index_); the policy
// is constructed once per `apply()` call as a thin view onto them.
// =============================================================================

#include <cstdint>
#include <complex>
#include <utility>
#include <vector>

#include <ed/core/basis_utils.h>  // LinIndexTable
#include <ed/core/combinadic.h>   // BinomialTable, rank_state/unrank_to_state (tableless mode)

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

    static constexpr bool may_leave_basis  = false;

    // One computational state per row, no per-emit projection factor.
    static constexpr bool needs_orbit_walk  = false;
    static constexpr bool has_coeff_modifier = false;
};

// ---------------------------------------------------------------------------
// Fixed total-Sz sector basis. The backing arrays live on the owning
// Hamiltonian (FixedSzOperator); the policy is a non-owning view.
// ---------------------------------------------------------------------------
struct FixedSzBasisPolicy {
    const std::vector<uint64_t>* basis_states;  // sorted, length C(N, n_up)
    const LinIndexTable*         lin_index;     // Lin (1990) lookup
    uint64_t                     dim_;          // cached basis_states->size()

    // Tableless combinadic mode (Track A, Jun 2026). When ``binom != nullptr``
    // the basis is represented IMPLICITLY by (n_bits_, n_up_) + a small O(N^2)
    // BinomialTable instead of the materialized ``basis_states`` vector +
    // ``lin_index`` table -- the per-element lookup becomes O(N) combinadic
    // rank/unrank rather than an O(1) table read, but the C(N,n_up)-sized basis
    // vector (the ~72 GB wall at N=36) and the Lin table are never allocated.
    // The colex combinadic rank coincides with the ascending-sorted index, so
    // the two modes are index-compatible and produce identical matvecs.
    const ed::core::combinadic::BinomialTable* binom = nullptr;
    int                          n_bits_ = 0;
    int                          n_up_   = 0;

    [[nodiscard]] inline uint64_t dim() const noexcept {
        return dim_;
    }
    [[nodiscard]] inline uint64_t state_of(uint64_t idx) const noexcept {
        return binom
            ? ed::core::combinadic::unrank_to_state(idx, n_bits_, n_up_, *binom)
            : (*basis_states)[idx];
    }
    [[nodiscard]] inline int64_t index_of(uint64_t state) const noexcept {
        // Audit F1: the tableless mode now carries an implicit Lin table
        // (``LinIndexTable::build_implicit``), so the O(1) two-table read is
        // the primary path in BOTH modes; the O(N) combinadic rank is only
        // the fallback for a policy constructed without a table.
        if (lin_index) return lin_index->lookup(state);
        if (binom) {
            return (__builtin_popcountll(state) == n_up_)
                ? ed::core::combinadic::rank_state(state, n_bits_, n_up_, *binom)
                : int64_t{-1};
        }
        return int64_t{-1};
    }

    // Audit F1: sequential row enumeration. In tableless mode the basis is
    // the ascending list of popcount-n_up words, so a contiguous row chunk
    // can be walked with Gosper's hack from ONE unrank at the chunk start
    // instead of an O(N) unrank per row. ``for_each_row_state`` (gather
    // driver / CSR assembly / diagonal build) consults these.
    [[nodiscard]] inline bool sequential_states() const noexcept {
        return binom != nullptr;
    }
    [[nodiscard]] static inline uint64_t next_state(uint64_t v) noexcept {
        // Gosper's hack: next larger word with the same popcount (v != 0).
        const uint64_t t = v | (v - 1);
        return (t + 1) | (((~t & (t + 1)) - 1) >> (__builtin_ctzll(v) + 1));
    }

    static constexpr bool may_leave_basis  = true;

    // One computational state per row, no per-emit projection factor.
    static constexpr bool needs_orbit_walk  = false;
    static constexpr bool has_coeff_modifier = false;
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

// Tableless combinadic fixed-Sz basis view. ``binom`` must be sized for at
// least ``n_bits`` and outlive the returned policy. ``dim`` is C(n_bits,n_up).
[[nodiscard]] inline FixedSzBasisPolicy make_combinadic_fixed_sz_basis(
    int                                        n_bits,
    int                                        n_up,
    const ed::core::combinadic::BinomialTable& binom,
    uint64_t                                   dim,
    const LinIndexTable*                       lin = nullptr) noexcept
{
    FixedSzBasisPolicy p{};
    p.basis_states = nullptr;
    p.lin_index    = lin;   // implicit Lin table (audit F1); nullptr => rank fallback
    p.dim_         = dim;
    p.binom        = &binom;
    p.n_bits_      = n_bits;
    p.n_up_        = n_up;
    return p;
}

} // namespace ed::matvec::basis
