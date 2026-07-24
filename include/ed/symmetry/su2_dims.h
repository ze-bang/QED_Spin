#pragma once
// =============================================================================
// include/ed/symmetry/su2_dims.h
//
// Stage 12c of the SU(2) rollout (docs/architecture/SYMMETRY_V2_DESIGN.md):
// exact S-resolved dimension arithmetic from the highest-weight trick.
//
// Facts (N spin-1/2 sites):
//   * Number of spin-S multiplets:
//       M(N, S) = C(N, N/2 - S) - C(N, N/2 - S - 1)
//     (Catalan-triangle counting; each multiplet contributes 2S+1 states).
//   * Every S-multiplet appears EXACTLY ONCE as a highest-weight state in
//     the Sz = S sector. Since every site permutation (and hence every
//     momentum / point-group sector) commutes with the global raising
//     operator S+, the same statement holds inside ANY spatial symmetry
//     sector:
//       dim(sector, spin S) = dim(sector, Sz = S) - dim(sector, Sz = S+1).
//     The right-hand side is two calls to the existing exact Burnside /
//     Molien per-sector formula (make_operator.h sector_dims_burnside) at
//     adjacent n_up -- NO orbit walk, free at planning time. The
//     group-aware wrapper `sector_dims_s_resolved` lives next to
//     `sector_dims_burnside` in make_operator.h; this header holds the
//     group-free combinatorics so light-weight callers (tests, Python
//     pre-flight, the projector's allowed-S sets) need no operator
//     machinery.
//
// All spins are carried as two_S = 2S (doubled integers) so odd-N
// half-integer towers stay exact.
// =============================================================================

#include <cstdint>
#include <stdexcept>
#include <string>

namespace ed::symmetry {

/// C(n, k) with the usual out-of-range-is-zero convention (k < 0 or
/// k > n). Overflow-safe for the n <= 63 site counts this library
/// supports (C(63, 31) < 2^63).
[[nodiscard]] inline std::uint64_t
binomial_or_zero(int n, int k) noexcept {
    if (k < 0 || k > n) return 0;
    if (k > n - k) k = n - k;
    std::uint64_t r = 1;
    for (int i = 0; i < k; ++i) {
        r = r * static_cast<std::uint64_t>(n - i) /
            static_cast<std::uint64_t>(i + 1);
    }
    return r;
}

/// Is two_S an admissible total-spin label for n_sites spin-1/2 sites?
[[nodiscard]] inline bool
two_S_admissible(int n_sites, int two_S) noexcept {
    return two_S >= 0 && two_S <= n_sites && (two_S % 2) == (n_sites % 2);
}

/// Number of spin-S multiplets, M(N, S) = C(N, (N-2S)/2) - C(N, (N-2S)/2 - 1).
[[nodiscard]] inline std::uint64_t
multiplet_count(int n_sites, int two_S) {
    if (n_sites <= 0 || n_sites >= 64) {
        throw std::invalid_argument(
            "multiplet_count: n_sites must be in [1, 63]");
    }
    if (!two_S_admissible(n_sites, two_S)) return 0;
    const int k = (n_sites - two_S) / 2;
    return binomial_or_zero(n_sites, k) - binomial_or_zero(n_sites, k - 1);
}

/// n_up of the highest-weight (Sz = +S) sector for a spin-S tower:
/// Sz = n_up - N/2 = S  =>  n_up = (N + 2S)/2. Throws on inadmissible
/// two_S so callers can't silently target an empty tower.
[[nodiscard]] inline int
n_up_of_highest_weight(int n_sites, int two_S) {
    if (!two_S_admissible(n_sites, two_S)) {
        throw std::invalid_argument(
            "n_up_of_highest_weight: two_S = " + std::to_string(two_S) +
            " is not admissible for n_sites = " + std::to_string(n_sites) +
            " (needs 0 <= two_S <= n_sites with matching parity)");
    }
    return (n_sites + two_S) / 2;
}

}  // namespace ed::symmetry
