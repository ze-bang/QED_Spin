#pragma once
// =============================================================================
// include/ed/symmetry/fixed_sz_membership.h
//
// Tableless, O(1)-memory stand-in for `FixedSzSubspace` used during streaming
// symmetry-sector construction. `compute_orbit_for_state`
// (projector_chain.h) consults the subspace ONLY as a membership guard
// (`subspace.index_of(img) < 0` -> skip out-of-subspace images). For a fixed-Sz
// sector every group image of a fixed-popcount state is itself fixed-popcount
// (site permutations preserve popcount), so membership reduces to a popcount
// test -- no materialized C(N,n_up) basis vector and no LinIndexTable needed.
//
// `index_of` returns 0 (any non-negative sentinel) for in-subspace states and
// -1 otherwise; callers in the build path only test the sign. It is trivially
// copyable, so the deferred CSR-rebuild providers can capture it by value
// instead of co-owning a shared basis vector + Lin table.
// =============================================================================

#include <cstdint>

#if defined(__GNUC__) || defined(__clang__)
#  define ED_POPCOUNT64(x) __builtin_popcountll(x)
#else
#  include <bitset>
#  define ED_POPCOUNT64(x) static_cast<int>(std::bitset<64>(x).count())
#endif

namespace ed::symmetry {

class FixedSzMembershipSubspace {
public:
    FixedSzMembershipSubspace(std::uint64_t n_bits, int n_up) noexcept
        : n_bits_(n_bits), n_up_(n_up) {}

    /// Membership guard: 0 if popcount(state) == n_up (and in range), else -1.
    /// Only the sign is consumed by `compute_orbit_for_state`.
    [[nodiscard]] std::int64_t index_of(std::uint64_t state) const noexcept {
        return (ED_POPCOUNT64(state) == n_up_) ? 0 : -1;
    }

    [[nodiscard]] std::uint64_t n_bits() const noexcept { return n_bits_; }
    [[nodiscard]] int           n_up()   const noexcept { return n_up_; }

private:
    std::uint64_t n_bits_;
    int           n_up_;
};

}  // namespace ed::symmetry

#undef ED_POPCOUNT64
