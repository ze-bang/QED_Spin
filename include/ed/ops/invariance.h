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

#include <cmath>
#include <ed/ops/algebra.h>

#include <complex>
#include <optional>
#include <string>
#include <vector>

class Operator;   // ops/operator.h: the term records the kernels read

namespace ed::ops {

/// Two coefficients agree when they differ by at most this times the largest |coefficient|.
// scale-free: relative tolerance (to the operator's largest coefficient)
inline constexpr double kInvarianceRtol = 1e-10;

/// The operator a ::Operator describes: its records summed, same-site products reduced exactly,
/// plus its extra canonical terms; in the order the kernels apply them: a two-body record is O1 O2 with O2
/// acting first (classify_route), a three-body record O1 acting first (its sites are
/// distinct, so the order does not matter). Throws std::invalid_argument for an op_type
/// outside {0 = S+, 1 = S-, 2 = Sz} or a site outside [0, n).
[[nodiscard]] MaskedOperator masked(const ::Operator& op);

/// The Operator of `m`'s canonical terms, in key order: a term on k <= 3 sites is one k-body
/// record (Z = 2 Sz folded into the coefficient; the identity is the record Sz_0 Sz_0 = 1/4),
/// a term on more sites an extra canonical term (Operator::add_extra_term). masked(to_operator(m))
/// == m.
[[nodiscard]] ::Operator to_operator(const MaskedOperator& m);

/// One canonical term as a product: coeff * prod_k ops[k](sites[k]), ops over '+', '-', 'z'
/// (S+, S-, S^z), sites ascending; the identity has no factors.
struct ProductTerm {
    std::complex<double> coeff;
    std::string          ops;
    std::vector<int>     sites;
};
/// m's canonical terms in key order, each as a product of S+, S-, S^z (Z = 2 S^z folded in).
[[nodiscard]] std::vector<ProductTerm> product_terms(const MaskedOperator& m);

/// Throws std::invalid_argument unless perm is a permutation of 0..n-1.
void require_permutation(const std::vector<int>& perm, int n);

/// Which changes of the spin counts a term may make: any, none (S^z conserved), or an even
/// number (S^z parity conserved).
enum class SzKeep { All, Zero, Even };
/// O without the terms whose S^z change `keep` excludes.
[[nodiscard]] MaskedOperator keep_sz_changes(const MaskedOperator& O, SzKeep keep);

/// O averaged over the group G of site permutations (every element listed, image() convention;
/// over a closed group a permutation and its inverse give the same average) and, with `flip`,
/// the global spin flip: (1 / (|G| (1 + flip))) sum_g sum_f U O U^dagger. The average commutes
/// with every element, so it is block diagonal in their sectors.
[[nodiscard]] MaskedOperator group_average(const MaskedOperator& O, const std::vector<std::vector<int>>& G, bool flip);

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

/// H^dagger = H. A non-finite coefficient (NaN, inf) is never Hermitian: the tolerance test
/// cannot compare it.
[[nodiscard]] inline bool hermitian(const MaskedOperator& H, double rtol = kInvarianceRtol) {
    return std::isfinite(H.l1_norm()) && invariant(H, H.dagger(), rtol);
}

/// What H conserves of S^z: every term keeps the number of up spins (U1), changes it by
/// even amounts only (Parity: prod sigma^z is conserved), or neither. Terms below the
/// tolerance do not count (U(theta) H U(theta)^dagger differs from H by exactly them).
enum class SzContent { U1, Parity, None };
[[nodiscard]] SzContent sz_content(const MaskedOperator& H, double rtol = kInvarianceRtol);

/// [H, S^a_tot] = 0 for a = +, -, z: full spin-rotation invariance.
[[nodiscard]] bool su2_invariant(const MaskedOperator& H, double rtol = kInvarianceRtol);

/// The h for which H - h S^z_tot is SU(2) invariant (0 when H is), else nullopt: an
/// SU(2)-symmetric H in a uniform field along z keeps S^2 and S^z as good quantum numbers.
[[nodiscard]] std::optional<double> su2_field(const MaskedOperator& H, double rtol = kInvarianceRtol);

/// The SU(2)-scalar part of O: its average over every global spin rotation,
/// int dR U_R O U_R^dagger, whose expectation in a spin multiplet is the multiplet average of O
/// and whose thermal trace with an SU(2)-symmetric H is O's. Computed exactly as the average over
/// the 60 rotations of the icosahedral group, which has no invariant harmonic below l = 6, so it
/// equals the full rotation average on every term of at most 5 sites (more raise Unsupported).
/// An SU(2)-invariant O is returned as it is.
[[nodiscard]] MaskedOperator su2_scalar_part(const MaskedOperator& O);

}  // namespace ed::ops
