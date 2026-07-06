#pragma once
// =============================================================================
// include/ed/symmetry/commute_check.h
//
// Term-level [H, U_g] = 0 validation for a site-permutation symmetry element.
//
// The abelian rep lane trusts the generators it is handed: an automorphism
// from ``find_symmetries`` commutes with H by construction (it is a symmetry
// of H's own coloured interaction graph), but an EXPLICIT generator set --
// e.g. a permutation list passed straight to ``qed.solve(symmetry=[...])``,
// or an NLCE bridge that maps a cluster onto a lattice embedding -- is
// unchecked. A wrong permutation (site-ordering mismatch, an off-by-one in a
// translation) then produces SILENTLY WRONG spectra with correct-looking
// per-sector sum rules. This checker makes that loud.
//
// Method: H commutes with the state permutation U_g iff relabelling every
// term's sites by ``perm`` leaves the term multiset (with coefficients)
// invariant -- an EXACT, O(#terms log #terms) test that needs no matvec and no
// 2^N vector. Distinct-site factors commute, so a term's factors are ordered
// canonically by permuted site; same-site factors keep their order (the
// conservative case -- it can only ever raise a false ALARM, never pass a
// genuinely non-commuting element, because a non-commuting relabelling changes
// the site multiset itself).
// =============================================================================

#include <ed/core/operator.h>           // Operator::TransformData
#include <ed/matvec/term_storage.h>     // ThreeBodyTerm

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <map>
#include <tuple>
#include <vector>

namespace ed::symmetry {

namespace detail {

// A canonical (op, permuted-site) factor.
using CommuteFactor = std::pair<int, std::uint64_t>;   // (op_type, perm[site])

// Order factors by permuted site (stable: equal sites keep input order, the
// only non-commuting case). Returns the sorted key list.
inline std::vector<CommuteFactor>
canon_factors(std::vector<CommuteFactor> f) {
    std::stable_sort(f.begin(), f.end(),
                     [](const CommuteFactor& a, const CommuteFactor& b) {
                         return a.second < b.second;
                     });
    return f;
}

inline std::uint64_t permuted_site(std::uint64_t s, const std::vector<int>& perm) {
    return (s < perm.size()) ? static_cast<std::uint64_t>(perm[s]) : s;
}

// Canonical term multiset under a site relabelling (identity perm => the
// operator's own canonical form). Keyed by the ordered factor list.
inline std::map<std::vector<CommuteFactor>, std::complex<double>>
canonical_terms(const std::vector<Operator::TransformData>&    t2,
                const std::vector<ed::matvec::ThreeBodyTerm>&  t3,
                const std::vector<int>&                        perm) {
    std::map<std::vector<CommuteFactor>, std::complex<double>> m;
    for (const auto& t : t2) {
        std::vector<CommuteFactor> f;
        f.emplace_back(static_cast<int>(t.op_type),
                       permuted_site(t.site_index, perm));
        if (t.is_two_body)
            f.emplace_back(static_cast<int>(t.op_type_2),
                           permuted_site(t.site_index_2, perm));
        m[canon_factors(std::move(f))] += t.coefficient;
    }
    for (const auto& t : t3) {
        std::vector<CommuteFactor> f = {
            {static_cast<int>(t.op_type_1), permuted_site(t.site_index_1, perm)},
            {static_cast<int>(t.op_type_2), permuted_site(t.site_index_2, perm)},
            {static_cast<int>(t.op_type_3), permuted_site(t.site_index_3, perm)},
        };
        m[canon_factors(std::move(f))] += t.coefficient;
    }
    for (auto it = m.begin(); it != m.end();)
        it = (std::abs(it->second) < 1e-13) ? m.erase(it) : std::next(it);
    return m;
}

}  // namespace detail

/// True iff relabelling H's term sites by ``perm`` leaves the term multiset
/// invariant, i.e. [H, U_perm] = 0. ``perm`` is the ``applyPermutation``
/// convention (output bit i reads input bit perm[i]); invariance under a
/// permutation and its inverse coincide, so the direction is immaterial.
[[nodiscard]] inline bool
hamiltonian_commutes_with_permutation(
    const std::vector<Operator::TransformData>&   t2,
    const std::vector<ed::matvec::ThreeBodyTerm>& t3,
    const std::vector<int>&                       perm) {
    std::vector<int> identity(perm.size());
    for (std::size_t i = 0; i < perm.size(); ++i) identity[i] = static_cast<int>(i);
    const auto base = detail::canonical_terms(t2, t3, identity);
    const auto perm_m = detail::canonical_terms(t2, t3, perm);
    if (base.size() != perm_m.size()) return false;
    for (const auto& [k, v] : base) {
        const auto it = perm_m.find(k);
        if (it == perm_m.end()) return false;
        if (std::abs(it->second - v) > 1e-10 * (1.0 + std::abs(v))) return false;
    }
    return true;
}

}  // namespace ed::symmetry
