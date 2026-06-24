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
//   * symmetry-projected sector       -- SymmetryBasisPolicy   (Wave 1)
//   * Sz + symmetry sector            -- FixedSzSymmetryBasisPolicy (Wave 1)
//   * distributed slab of any above   -- DistributedBasisPolicy<...> (Wave 2)
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
// Optional ABI (Wave 0 of the close-the-symmetry-gap unification, May
// 2026). Trivial policies inherit no-op defaults so the kernel's
// ``if constexpr`` branches compile to nothing:
//
//   static constexpr bool needs_orbit_walk = false
//       When true, the outer apply_terms loop calls ``iter_orbit(i, cb)``
//       instead of using ``state_of(i)`` directly. Lets the symmetry
//       policies walk |G| computational states per orbit representative.
//
//   template <class Callback>
//   void iter_orbit(uint64_t src_idx, Callback&& cb) const
//       Yields ``(state s, Scalar pre_phase)`` for each computational
//       basis state in the orbit of ``src_idx``. Trivial policies yield
//       ``(state_of(src_idx), Scalar(1))`` exactly once.
//
//   static constexpr bool has_coeff_modifier = false
//       When true, the per-term emit multiplies by
//       ``coeff_modifier<Scalar>(s, s_prime, src_idx, dst_idx)`` before
//       pushing to the radix-sort buffer. Symmetry policies return
//       ``conj(beta_{s'}) * group_norm / norm_{dst}``.
//
//   template <class Scalar>
//   Scalar coeff_modifier(uint64_t s, uint64_t s_prime,
//                         uint64_t src_idx, uint64_t dst_idx) const
//       Optional per-emit phase / normalization factor. Default = 1.
//
//   static constexpr bool is_distributed = false
//       When true, indicates the basis is a slab across MPI ranks and
//       the kernel may need to consult ``is_local`` / ``local_offset``.
//
//   bool     is_local(uint64_t global_idx) const noexcept   -- default: true
//   uint64_t local_offset() const noexcept                  -- default: 0
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

    // Wave 0 ABI extension (May 2026): trivial-policy no-ops. ``apply_terms``
    // gates these behind ``if constexpr`` so a compliant compiler elides
    // them entirely for FullBasisPolicy / FixedSzBasisPolicy.
    static constexpr bool needs_orbit_walk  = false;
    static constexpr bool has_coeff_modifier = false;
    static constexpr bool is_distributed    = false;

    template <class Callback>
    inline void iter_orbit(uint64_t src_idx, Callback&& cb) const {
        // Trivial: orbit is the single state itself, phase 1.
        // Kept for ABI completeness so a future apply_terms variant that
        // doesn't constexpr-elide the orbit branch still compiles.
        using cb_phase_t =
            decltype(std::declval<Callback>()(uint64_t{}, std::complex<double>{}));
        (void)cb_phase_t{};
        cb(state_of(src_idx), std::complex<double>(1.0, 0.0));
    }
    template <class Scalar>
    [[nodiscard]] inline Scalar coeff_modifier(uint64_t /*s*/,
                                               uint64_t /*s_prime*/,
                                               uint64_t /*src_idx*/,
                                               uint64_t /*dst_idx*/) const noexcept {
        return Scalar(1);
    }
    [[nodiscard]] inline bool is_local(uint64_t /*g*/) const noexcept {
        return true;
    }
    [[nodiscard]] inline uint64_t local_offset() const noexcept { return 0; }
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
        if (binom) {
            return (__builtin_popcountll(state) == n_up_)
                ? ed::core::combinadic::rank_state(state, n_bits_, n_up_, *binom)
                : int64_t{-1};
        }
        return lin_index->lookup(state);
    }

    static constexpr bool may_leave_basis  = true;

    // Wave 0 ABI extension (May 2026): trivial-policy no-ops. See the
    // twin block in ``FullBasisPolicy`` for the contract.
    static constexpr bool needs_orbit_walk  = false;
    static constexpr bool has_coeff_modifier = false;
    static constexpr bool is_distributed    = false;

    template <class Callback>
    inline void iter_orbit(uint64_t src_idx, Callback&& cb) const {
        cb(state_of(src_idx), std::complex<double>(1.0, 0.0));
    }
    template <class Scalar>
    [[nodiscard]] inline Scalar coeff_modifier(uint64_t /*s*/,
                                               uint64_t /*s_prime*/,
                                               uint64_t /*src_idx*/,
                                               uint64_t /*dst_idx*/) const noexcept {
        return Scalar(1);
    }
    [[nodiscard]] inline bool is_local(uint64_t /*g*/) const noexcept {
        return true;
    }
    [[nodiscard]] inline uint64_t local_offset() const noexcept { return 0; }
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
    uint64_t                                   dim) noexcept
{
    FixedSzBasisPolicy p{};
    p.basis_states = nullptr;
    p.lin_index    = nullptr;
    p.dim_         = dim;
    p.binom        = &binom;
    p.n_bits_      = n_bits;
    p.n_up_        = n_up;
    return p;
}

} // namespace ed::matvec::basis
