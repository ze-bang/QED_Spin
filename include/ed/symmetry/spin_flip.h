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

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstdlib>
#include <vector>

#include <ed/matvec/term_storage.h>

namespace ed::symmetry {

// -----------------------------------------------------------------------------
// Diagonal (S^z-basis-diagonal) symmetry axis of H, from the term-level
// Delta(n_up) content:
//   * U1     -- every term conserves popcount (S+S- pairs, diagonals):
//               the full U(1) tower of fixed-Sz sectors applies.
//   * Parity -- every term changes popcount by an EVEN amount (adds
//               S+S+ / S-S- pair terms, e.g. J_{+-+-}-type anisotropic
//               exchange): U(1) is broken but (-1)^{n_up} = prod sigma^z
//               is conserved -- the Hilbert space splits into two
//               2^{N-1} halves.
//   * None   -- some term changes popcount by an odd amount (lone S+-,
//               Sz S+- mixed terms, transverse fields): no diagonal
//               reduction.
// This is the three-valued refinement of the old binary conserves-Sz
// check; ``Parity`` is the Z2 remnant of the broken U(1).
// -----------------------------------------------------------------------------
enum class SzAxis : std::uint8_t { None = 0, Parity = 1, U1 = 2 };

[[nodiscard]] inline SzAxis
sz_axis_of(const ed::matvec::TermStorage& t) noexcept {
    // odd popcount changes -> None
    if (!t.offdiag_one_body.empty() || !t.mixed_two_body.empty()) {
        return SzAxis::None;
    }
    auto dpop = [](std::uint8_t op) -> int {
        // op codes: 0 = S+, 1 = S-, 2 = Sz (see term_storage.h);
        // only parity of the total change matters here.
        return (op == 2) ? 0 : 1;
    };
    bool u1 = true;
    for (const auto& tb : t.offdiag_two_body) {
        // Both ops are off-diagonal S+- here; S+S- conserves, S+S+ /
        // S-S- changes popcount by 2 (even either way).
        if (tb.op_type_1 == tb.op_type_2) u1 = false;
    }
    for (const auto& tb : t.three_body) {
        const int d = dpop(tb.op_type_1) + dpop(tb.op_type_2)
                    + dpop(tb.op_type_3);
        if (d % 2 != 0) return SzAxis::None;
        // Even but possibly nonzero: S+S+SzSz-free 3-body raising pairs
        // etc. break U(1) unless the +/- content pairs off exactly;
        // conservative: any off-diagonal 3-body content demotes U(1).
        if (d != 0) u1 = false;
    }
    return u1 ? SzAxis::U1 : SzAxis::Parity;
}


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

    // three_body: partner each term with its flip image -- all S+/- ops
    // swapped (op 0 <-> 1), coefficient scaled by (-1)^{# Sz factors}
    // (X Sz X = -Sz). Factors on DISTINCT sites commute, so compare
    // site-sorted factor lists. Same-site three-body products don't commute
    // under the sort, so stay conservative (reject) when any term repeats a
    // site -- correctness over coverage for that exotic case (C10).
    {
        const auto& v = t.three_body;
        using Fac = std::array<std::pair<std::uint64_t, int>, 3>;
        auto facs = [](const ed::matvec::ThreeBodyTerm& tb, bool flip) -> Fac {
            Fac f = {{ {tb.site_index_1, tb.op_type_1},
                       {tb.site_index_2, tb.op_type_2},
                       {tb.site_index_3, tb.op_type_3} }};
            if (flip) for (auto& x : f) if (x.second != 2) x.second ^= 1;
            std::sort(f.begin(), f.end());
            return f;
        };
        auto sz_sign = [](const ed::matvec::ThreeBodyTerm& tb) -> double {
            const int nz = (tb.op_type_1 == 2) + (tb.op_type_2 == 2)
                         + (tb.op_type_3 == 2);
            return (nz & 1) ? -1.0 : 1.0;
        };
        for (const auto& tb : v) {
            if (tb.site_index_1 == tb.site_index_2 ||
                tb.site_index_1 == tb.site_index_3 ||
                tb.site_index_2 == tb.site_index_3)
                return false;   // same-site 3-body: conservative
        }
        std::vector<char> used(v.size(), 0);
        for (std::size_t i = 0; i < v.size(); ++i) {
            const Fac fi = facs(v[i], /*flip=*/true);
            const std::complex<double> ci = v[i].coefficient * sz_sign(v[i]);
            bool matched = false;
            for (std::size_t j = 0; j < v.size(); ++j) {
                if (used[j]) continue;
                if (facs(v[j], /*flip=*/false) == fi &&
                    detail::coeff_eq(v[j].coefficient, ci)) {
                    used[j] = 1;
                    matched = true;
                    break;
                }
            }
            if (!matched) return false;
        }
    }

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
