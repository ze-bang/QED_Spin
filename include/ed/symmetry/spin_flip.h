#pragma once
// =============================================================================
// include/ed/symmetry/spin_flip.h
//
// Stage 5 of the SymmetryEngine v2 plan
// (docs/architecture/SYMMETRY_V2_DESIGN.md): global spin-flip Z2.
//
// The global flip X = prod_i sigma^x_i acts on the term algebra as
//
//     X Sz_i X = -Sz_i        X S+_i X = S-_i        X S-_i X = S+_i
//
// ``hamiltonian_is_spin_flip_symmetric`` checks [H, X] = 0 at the TERM
// level on the canonical SoA storage:
//
//   * diag_one_body   (h Sz)         : image -h Sz    -> any Zeeman kills it
//   * diag_two_body   (J Sz Sz)      : invariant      -> always fine
//   * offdiag_one_body(c S+/-)       : image  c S-/+  -> needs the op-flipped
//                                      partner with the SAME coefficient
//                                      (transverse field hx*Sx passes: X Sx X = Sx)
//   * mixed_two_body  (c Sz S+/-)    : image -c Sz S-/+ -> partner with the
//                                      flipped op and NEGATED coefficient
//   * offdiag_two_body(c S^a S^b)    : image  c S^a~ S^b~ (both ops flipped)
//                                      -> partner with both ops flipped, same
//                                      coefficient (site order either way)
//   * three_body                     : conservative -> any presence returns
//                                      false (extend when needed)
//
// The check is exact multiset matching with a small tolerance on the
// coefficients; term counts are O(bonds), so the O(n^2) partner search is
// negligible next to any orbit scan.
//
// Physics consequence exploited by the SectorTransporter (workflow level):
// X commutes with every site permutation (it acts on the internal spin
// index, permutations on sites), so it maps the (n_up, irrep k) sector to
// (N - n_up, SAME k) with an identical spectrum:
//
//     Z_{n_up, k}(beta) == Z_{N - n_up, k}(beta).
//
// The all-Sz thermal loop therefore solves only n_up <= N/2 and mirrors
// the thermodynamic entries -- no projection, no new basis machinery.
// (The n_up == N/2 in-sector projection -- flip as a CompiledGroup element
// halving the biggest sector -- is the Stage-5b follow-up.)
// =============================================================================

#include <cmath>
#include <complex>
#include <cstdlib>
#include <vector>

#include <ed/matvec/term_storage.h>

namespace ed::symmetry {

namespace detail {

inline bool coeff_eq(const std::complex<double>& a,
                     const std::complex<double>& b,
                     double eps = 1e-12) noexcept {
    return std::abs(a - b) <= eps * (1.0 + std::abs(a) + std::abs(b));
}

}  // namespace detail

/// Env gate for the Stage-5 Sz transporter (default ON; set
/// ED_SYM_SPIN_FLIP=0 to disable for bisection). Read per call so tests
/// can toggle it from Python without process restarts.
[[nodiscard]] inline bool spin_flip_transport_enabled() noexcept {
    const char* v = std::getenv("ED_SYM_SPIN_FLIP");
    return !(v != nullptr && v[0] == '0' && v[1] == '\0');
}

/// [H, X] == 0 at the term level (see header comment for the mapping).
[[nodiscard]] inline bool
hamiltonian_is_spin_flip_symmetric(const ed::matvec::TermStorage& t) noexcept {
    constexpr double kZero = 1e-14;

    // Zeeman: any nonzero Sz field breaks the flip.
    for (const auto& d : t.diag_one_body)
        if (std::abs(d.coefficient) > kZero) return false;

    // Conservative on three-body content.
    if (!t.three_body.empty()) return false;

    // offdiag_one_body: multiset must be invariant under op 0 <-> 1 with the
    // SAME coefficient.
    {
        const auto& v = t.offdiag_one_body;
        std::vector<char> used(v.size(), 0);
        for (std::size_t i = 0; i < v.size(); ++i) {
            bool matched = false;
            for (std::size_t j = 0; j < v.size(); ++j) {
                if (used[j]) continue;
                if (v[j].site_index == v[i].site_index &&
                    v[j].op_type == (v[i].op_type ^ 1u) &&
                    detail::coeff_eq(v[j].coefficient, v[i].coefficient)) {
                    used[j] = 1;
                    matched = true;
                    break;
                }
            }
            if (!matched) return false;
        }
    }

    // mixed_two_body: partner with flipped S+/- op and NEGATED coefficient.
    {
        const auto& v = t.mixed_two_body;
        std::vector<char> used(v.size(), 0);
        for (std::size_t i = 0; i < v.size(); ++i) {
            bool matched = false;
            for (std::size_t j = 0; j < v.size(); ++j) {
                if (used[j]) continue;
                if (v[j].sz_site == v[i].sz_site &&
                    v[j].flip_site == v[i].flip_site &&
                    v[j].flip_op_type == (v[i].flip_op_type ^ 1u) &&
                    detail::coeff_eq(v[j].coefficient, -v[i].coefficient)) {
                    used[j] = 1;
                    matched = true;
                    break;
                }
            }
            if (!matched) return false;
        }
    }

    // offdiag_two_body: partner with BOTH ops flipped, same coefficient
    // (accept the site pair in either stored order).
    {
        const auto& v = t.offdiag_two_body;
        std::vector<char> used(v.size(), 0);
        for (std::size_t i = 0; i < v.size(); ++i) {
            bool matched = false;
            for (std::size_t j = 0; j < v.size(); ++j) {
                if (used[j]) continue;
                const bool same_order =
                    v[j].site_index_1 == v[i].site_index_1 &&
                    v[j].site_index_2 == v[i].site_index_2 &&
                    v[j].op_type_1 == (v[i].op_type_1 ^ 1u) &&
                    v[j].op_type_2 == (v[i].op_type_2 ^ 1u);
                const bool swapped_order =
                    v[j].site_index_1 == v[i].site_index_2 &&
                    v[j].site_index_2 == v[i].site_index_1 &&
                    v[j].op_type_1 == (v[i].op_type_2 ^ 1u) &&
                    v[j].op_type_2 == (v[i].op_type_1 ^ 1u);
                if ((same_order || swapped_order) &&
                    detail::coeff_eq(v[j].coefficient, v[i].coefficient)) {
                    used[j] = 1;
                    matched = true;
                    break;
                }
            }
            if (!matched) return false;
        }
    }

    // diag_two_body invariant; nothing to check.
    return true;
}

}  // namespace ed::symmetry
