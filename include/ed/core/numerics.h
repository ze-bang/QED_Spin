#pragma once
// =============================================================================
// include/ed/core/numerics.h -- the engine's numerical tolerances, relative to the scale of H.
//
// A threshold that judges an energy (a residual, a Krylov breakdown, a degeneracy window) is
// a multiple of s_H, an upper bound of ||H||: the sum of |c_t| over H's canonical terms (each
// site factor I, Z, S+, S- has norm 1; LinearOperator::norm_bound()). So a Hamiltonian in
// other units (s * H) gives s times the same answer. A threshold on a quantity that is already
// scale-free (a unit vector's norm, a character, a phase) stays absolute at its site.
// =============================================================================

#include <limits>

namespace ed::numerics {

inline constexpr double kEps = std::numeric_limits<double>::epsilon();

/// A Lanczos / Krylov-Schur step breaks down (invariant subspace) when beta <= this * s_H.
inline constexpr double kBreakdownRel = 64.0 * kEps;
/// Krylov-Schur locks a Ritz pair when its residual bound is below this * s_H.
inline constexpr double kLockRel = 1e-10;
/// A ground-state vector is certified when ||H u - E u|| <= this * s_H.
inline constexpr double kGsResidRel = 1e-9;
/// A dense block is real when max |Im H_ij| <= this * max |H_ij|.
inline constexpr double kRealBlockRel = 32.0 * kEps;

/// s_H from LinearOperator::norm_bound(): the bound itself, or 1 when it is unknown (0).
[[nodiscard]] inline double scale_or_one(double bound) noexcept { return bound > 0.0 ? bound : 1.0; }

}  // namespace ed::numerics
