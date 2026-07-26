#pragma once
// =============================================================================
// include/ed/symmetry/su2.h
//
// Stage 12b of the SU(2) rollout (docs/architecture/SYMMETRY_V2_DESIGN.md):
// term-level detection of full SU(2) spin-rotation invariance,
// [H, S^a_tot] = 0 for a = x, y, z.
//
// A spin-1/2 term Hamiltonian commutes with the total-spin algebra iff it
// is a sum of isotropic exchanges J_b * (S_i . S_j) over bonds b = {i, j}
// (plus constants). At the TermStorage level that means:
//
//   * diag_one_body    (h Sz)        : any Zeeman field breaks it
//   * offdiag_one_body (c S+/-)      : any transverse field breaks it
//   * mixed_two_body   (c Sz S+/-)   : DM / Gamma-type anisotropy breaks it
//   * offdiag_two_body (c S^a S^b)   : S+S+ / S-S- (op1 == op2) breaks it;
//                                      S+S- content must aggregate, per
//                                      unordered pair, to
//                                        c(+-) == c(-+) == c(zz) / 2
//                                      matching the isotropic emission
//                                      J * (0.5 S+S- + 0.5 S-S+ + SzSz)
//                                      (hamiltonian_builder.h heisenberg()).
//   * diag_two_body    (c Sz Sz)     : joins the per-pair ratio above;
//                                      site_1 == site_2 entries are pure
//                                      identity shifts (spin_sq = 1/4 per
//                                      state) and are ignored.
//   * three_body                     : conservative -> any presence breaks it
//                                      (scalar chirality S_i.(S_j x S_k) IS
//                                      SU(2)-invariant; extend when needed).
//
// Isotropy is PER BOND: a lattice with different Heisenberg J on different
// bonds still qualifies. The check mirrors sz_axis_of /
// hamiltonian_is_spin_flip_symmetric (same file family, same SoA input,
// same conservative bias) and satisfies the algebra containment
//     su2  =>  (sz_axis_of == U1)  &&  spin_flip  &&  time_reversal-real
// which the unit tests assert as an internal consistency tripwire.
// =============================================================================

#include <cmath>
#include <complex>
#include <cstdlib>
#include <map>
#include <utility>

#include <ed/matvec/term_storage.h>

namespace ed::symmetry {

/// Env gate for the SU(2) axis (default ON; set ED_SYM_SU2=0 to veto
/// detection-driven labeling/exploitation for bisection). Read per call so
/// tests can toggle from Python without process restarts.
[[nodiscard]] inline bool su2_enabled() noexcept {
    const char* v = std::getenv("ED_SYM_SU2");
    return !(v != nullptr && v[0] == '0' && v[1] == '\0');
}

/// [H, S_tot] == 0 at the term level (see header comment for the mapping).
[[nodiscard]] inline bool
hamiltonian_is_su2_symmetric(const ed::matvec::TermStorage& t,
                             double tol = 1e-12) noexcept {
    using Cx = std::complex<double>;
    const auto nonzero = [tol](const Cx& c) noexcept {
        return std::abs(c) > tol;
    };

    for (const auto& d : t.diag_one_body)
        if (nonzero(d.coefficient)) return false;
    for (const auto& d : t.offdiag_one_body)
        if (nonzero(d.coefficient)) return false;
    for (const auto& d : t.mixed_two_body)
        if (nonzero(d.coefficient)) return false;
    for (const auto& d : t.three_body)
        if (nonzero(d.coefficient)) return false;

    // Aggregate per unordered pair {i < j}:
    //   czz  : coefficient of Sz_i Sz_j
    //   cpm  : coefficient of S+_i S-_j   (covers (0,1,i,j) and (1,0,j,i))
    //   cmp  : coefficient of S-_i S+_j   (covers (1,0,i,j) and (0,1,j,i))
    struct PairCoeffs {
        Cx czz{0.0, 0.0};
        Cx cpm{0.0, 0.0};
        Cx cmp{0.0, 0.0};
    };
    std::map<std::pair<std::uint64_t, std::uint64_t>, PairCoeffs> pairs;
    const auto key = [](std::uint64_t a, std::uint64_t b) {
        return (a < b) ? std::make_pair(a, b) : std::make_pair(b, a);
    };

    for (const auto& d : t.diag_two_body) {
        if (d.site_index_1 == d.site_index_2) continue;  // identity shift
        pairs[key(d.site_index_1, d.site_index_2)].czz += d.coefficient;
    }
    for (const auto& o : t.offdiag_two_body) {
        if (!nonzero(o.coefficient) ) continue;
        if (o.site_index_1 == o.site_index_2) return false;  // ill-formed
        if (o.op_type_1 == o.op_type_2) return false;        // S+S+ / S-S-
        auto& pc = pairs[key(o.site_index_1, o.site_index_2)];
        const bool sorted = o.site_index_1 < o.site_index_2;
        // S+ on the smaller site index -> contributes to cpm.
        const bool plus_on_min = sorted ? (o.op_type_1 == 0)
                                        : (o.op_type_2 == 0);
        (plus_on_min ? pc.cpm : pc.cmp) += o.coefficient;
    }

    const auto close = [tol](const Cx& a, const Cx& b) noexcept {
        return std::abs(a - b) <= tol * (1.0 + std::abs(a) + std::abs(b));
    };
    for (const auto& [k, pc] : pairs) {
        (void)k;
        // J S_i.S_j  ==  czz SzSz + (czz/2)(S+S- + S-S+): both transverse
        // aggregates must equal czz/2. Bonds missing any component fail.
        const Cx half_zz = 0.5 * pc.czz;
        if (!close(pc.cpm, half_zz) || !close(pc.cmp, half_zz)) return false;
    }
    return true;
}

}  // namespace ed::symmetry
