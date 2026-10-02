// =============================================================================
// include/ed/ops/invariance.h -- symmetry verdicts on the canonical term map (algebra.h).
//
// g is a symmetry of H iff g H g^-1 == H, compared term by term on the canonical map:
// the expansion of an operator into canonical terms is unique, so the verdict does not
// depend on how H was written (cancelling records, Cartesian or ladder form, same-site
// products). Every verdict uses one relative tolerance, kInvarianceRtol times the largest
// |coefficient| of H, so it does not depend on H's overall scale either.
// =============================================================================
#pragma once

#include <ed/ops/algebra.h>

#include <vector>

class Operator;   // ops/operator.h: the term records the kernels read

namespace ed::ops {

/// Two coefficients agree when they differ by at most this times the largest |coefficient|.
inline constexpr double kInvarianceRtol = 1e-10;

/// The operator a ::Operator's term records describe, records summed and same-site products
/// reduced exactly, in the order the kernels apply them: a two-body record is O1 O2 with O2
/// acting first (classify_route), a three-body record O1 acting first (its sites are
/// distinct, so the order does not matter). Throws std::invalid_argument for an op_type
/// outside {0 = S+, 1 = S-, 2 = Sz} or a site outside [0, n).
[[nodiscard]] MaskedOperator masked(const ::Operator& op);

/// g H g^-1 == H, given the image.
[[nodiscard]] inline bool invariant(const MaskedOperator& H, const MaskedOperator& image,
                                    double rtol = kInvarianceRtol) {
    return H.equals(image, rtol);
}

/// [H, U_perm] = 0 for the site permutation `perm` (either direction: H is invariant under
/// a permutation iff under its inverse). Throws std::invalid_argument unless perm is a
/// permutation of 0..n-1.
[[nodiscard]] bool commutes_with_permutation(const MaskedOperator& H, const std::vector<int>& perm,
                                             double rtol = kInvarianceRtol);

/// [H, prod sigma^x] = 0: the global spin flip.
[[nodiscard]] inline bool flip_invariant(const MaskedOperator& H, double rtol = kInvarianceRtol) {
    return invariant(H, H.image(MaskedOperator::Map::F), rtol);
}

/// K H K = H: H is real in the S^z basis (time reversal pairs k with -k).
[[nodiscard]] inline bool conjugation_invariant(const MaskedOperator& H, double rtol = kInvarianceRtol) {
    return invariant(H, H.image(MaskedOperator::Map::K), rtol);
}

/// H^dagger = H.
[[nodiscard]] inline bool hermitian(const MaskedOperator& H, double rtol = kInvarianceRtol) {
    return invariant(H, H.dagger(), rtol);
}

/// What H conserves of S^z: every term keeps the number of up spins (U1), changes it by
/// even amounts only (Parity: prod sigma^z is conserved), or neither. Terms below the
/// tolerance do not count (U(theta) H U(theta)^dagger differs from H by exactly them).
enum class SzContent { U1, Parity, None };
[[nodiscard]] SzContent sz_content(const MaskedOperator& H, double rtol = kInvarianceRtol);

/// [H, S^a_tot] = 0 for a = +, -, z: full spin-rotation invariance.
[[nodiscard]] bool su2_invariant(const MaskedOperator& H, double rtol = kInvarianceRtol);

}  // namespace ed::ops
