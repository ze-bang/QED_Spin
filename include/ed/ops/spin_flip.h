#pragma once
// =============================================================================
// include/ed/ops/spin_flip.h -- where the global spin flip X = prod_i sigma^x_i closes.
//
// X maps a state with n_up up spins to one with N - n_up, so it preserves a fixed-Sz block
// only at n_up = N/2, an Sz-parity half only for even N, and the full space always.
// Whether H commutes with X is ed::ops::flip_invariant (invariance.h).
// =============================================================================

namespace ed::symmetry {

/// THE flip-subspace closure rule, single-sourced for every consumer (do
/// not restate it locally). prod sigma^x preserves:
///   * a fixed-Sz block   iff n_up == N/2,
///   * an Sz-parity half  iff N is even,
///   * the full space     always.
/// ``n_up >= 0`` wins over ``sz_parity`` (mutually exclusive upstream).
[[nodiscard]] inline bool flip_subspace_admissible(int n_up, int sz_parity, int n_sites) noexcept {
    if (n_up >= 0) return 2 * n_up == n_sites;
    if (sz_parity >= 0) return n_sites % 2 == 0;
    return true;
}

}  // namespace ed::symmetry
