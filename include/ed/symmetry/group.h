// =============================================================================
// include/ed/symmetry/group.h
//
// `ed::sym` -- a small, programmatic DSL for building site-permutation
// symmetry groups without going through the JSON detour
// (`automorphism_results/{max_clique, minimal_generators,
// sector_metadata}.json`) that the legacy `automorphism_finder.py`
// pipeline assumes (P2.11 / audit §3.10).
//
// Why?
// ----
// The existing engine consumes a fully-elaborated `SymmetryGroupInfo`
// (defined in `ed/core/construct_ham.h`): the full `max_clique` of group
// elements, the minimal `generators` + their orders, the corresponding
// `power_representation`, and the desired list of irrep `sectors`.
// Today users get there in two ways:
//
//   1. They write a `connectivity_graph.json`, run
//      `automorphism_finder.py`, then point the C++ engine at the
//      resulting `automorphism_results/` directory.
//   2. They open the `SymmetryGroupInfo` struct manually and fill its
//      seven public fields by hand, with no validation.
//
// Path 1 is heavyweight (Python, Nauty, JSON, file I/O) and path 2 is a
// landmine (you have to know the conventions for the BFS-based
// `power_representation` and the abelian sector enumeration).
//
// The DSL below covers the four cases that account for ~95% of
// collaborator usage:
//
//   * 1D translation group  (`translation_group_1d(N)`)
//   * 1D translation × reflection D_N
//     (`translation_group_with_reflection_1d(N)`)
//   * arbitrary point group from a list of permutation generators
//     (`group_from_generators(N, gens, [gen_orders])`)
//   * Z2 spatial inversion (`reflection_1d(N)`) -- the single-generator
//     building block that powers the others
//
// All builders return a `SymmetryGroupInfo` ready to be assigned to
// `Operator::symmetry_info` (or `FixedSzOperator::symmetry_info`); from
// that point on, the existing `generateSymmetrySectors*` machinery
// works unchanged.
//
// What this DSL does NOT do
// -------------------------
//   * Internal Z2 spin-flip. The on-the-fly engine only handles
//     site-permutation symmetries; spin-flip is a bit-XOR action, not
//     a permutation, so it lives outside this DSL. (The fixed-Sz
//     workflow already takes care of U(1) charge conservation.)
//   * 2D / 3D point groups beyond what the user can describe by listing
//     site permutations. For lattices with non-trivial site labels
//     (e.g. kagome, pyrochlore), keep using the Python automorphism
//     pipeline -- this DSL is the programmatic shortcut for the simple
//     cases.
//
// All functions throw `std::invalid_argument` on malformed input
// (wrong-length permutation, non-bijective permutation, generator
// order zero, ...). Validation is cheap (<1us per generator), so we
// always run it.
// =============================================================================

#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

// Bring in SymmetryGroupInfo (declared in ed/core/construct_ham.h, alongside
// the Operator definition that consumes it).
#include <ed/core/construct_ham.h>

namespace ed::sym {

/// A site permutation: `perm[i]` is the new label of site `i` after the
/// permutation acts. Composition convention: `(a o b)[i] = a[b[i]]`,
/// i.e. `b` is applied first.
using Permutation = std::vector<int>;

// ---------------------------------------------------------------------------
// Permutation algebra (header-inlined: fast, branch-light, no link cost).
// ---------------------------------------------------------------------------

/// Identity permutation on `n_sites` sites.
[[nodiscard]] inline Permutation identity(int n_sites) {
    if (n_sites <= 0) {
        throw std::invalid_argument(
            "ed::sym::identity: n_sites must be positive (got " +
            std::to_string(n_sites) + ")");
    }
    Permutation p(static_cast<std::size_t>(n_sites));
    for (int i = 0; i < n_sites; ++i) p[static_cast<std::size_t>(i)] = i;
    return p;
}

/// Throws if `g` is not a valid permutation of `{0, ..., n_sites-1}`.
inline void validate(const Permutation& g, int n_sites) {
    if (static_cast<int>(g.size()) != n_sites) {
        throw std::invalid_argument(
            "ed::sym::validate: permutation length " +
            std::to_string(g.size()) + " != n_sites " +
            std::to_string(n_sites));
    }
    std::vector<char> seen(static_cast<std::size_t>(n_sites), 0);
    for (int x : g) {
        if (x < 0 || x >= n_sites) {
            throw std::invalid_argument(
                "ed::sym::validate: permutation entry " + std::to_string(x) +
                " out of range [0, " + std::to_string(n_sites) + ")");
        }
        if (seen[static_cast<std::size_t>(x)]) {
            throw std::invalid_argument(
                "ed::sym::validate: permutation is not a bijection (entry " +
                std::to_string(x) + " appears twice)");
        }
        seen[static_cast<std::size_t>(x)] = 1;
    }
}

/// `(a o b)[i] = a[b[i]]`. `a` and `b` must have equal length.
[[nodiscard]] inline Permutation compose(const Permutation& a,
                                         const Permutation& b) {
    if (a.size() != b.size()) {
        throw std::invalid_argument(
            "ed::sym::compose: length mismatch (" +
            std::to_string(a.size()) + " vs " +
            std::to_string(b.size()) + ")");
    }
    Permutation out(a.size());
    for (std::size_t i = 0; i < a.size(); ++i) out[i] = a[b[i]];
    return out;
}

/// `g^k` for `k >= 0` (`g^0 = identity`).
[[nodiscard]] inline Permutation power(const Permutation& g, int k) {
    if (k < 0) {
        throw std::invalid_argument(
            "ed::sym::power: negative exponent " + std::to_string(k) +
            " not supported (use inverse() first)");
    }
    Permutation out = identity(static_cast<int>(g.size()));
    for (int i = 0; i < k; ++i) out = compose(g, out);
    return out;
}

/// Smallest `k > 0` with `g^k == identity`, capped at `g.size()!` (which
/// is way more than any realistic group element actually needs). Throws
/// if no such `k` is found within the cap (cannot happen for a valid
/// permutation, but keeps the loop bounded as a defensive measure).
[[nodiscard]] inline int order(const Permutation& g) {
    const Permutation id = identity(static_cast<int>(g.size()));
    Permutation cur = g;
    for (int k = 1; k <= static_cast<int>(g.size()) * 2 + 1; ++k) {
        if (cur == id) return k;
        cur = compose(g, cur);
    }
    throw std::runtime_error(
        "ed::sym::order: ran past 2*n_sites+1 iterations without "
        "returning to identity (input is probably not a permutation)");
}

// ---------------------------------------------------------------------------
// Builders for common 1D groups
// ---------------------------------------------------------------------------

/// Cyclic translation by `shift` sites on a 1D ring of `n_sites` sites:
/// `T[i] = (i - shift) mod n_sites`. Convention: `T` shifts state by
/// `+shift` so site `i` of `T(state)` is `state[(i - shift) mod N]`,
/// matching the Bloch-momentum sign convention `e^{+i k r}`.
[[nodiscard]] inline Permutation translation(int n_sites, int shift = 1) {
    if (n_sites <= 0) {
        throw std::invalid_argument(
            "ed::sym::translation: n_sites must be positive (got " +
            std::to_string(n_sites) + ")");
    }
    Permutation p(static_cast<std::size_t>(n_sites));
    for (int i = 0; i < n_sites; ++i) {
        const int j = ((i - shift) % n_sites + n_sites) % n_sites;
        p[static_cast<std::size_t>(i)] = j;
    }
    return p;
}

/// Spatial reflection (site reversal) on a 1D chain: `R[i] = n_sites - 1 - i`.
/// This is the only Z2 the on-the-fly engine knows how to handle (internal
/// Z2 spin-flip would be a bit-XOR, which is not a site permutation).
[[nodiscard]] inline Permutation reflection_1d(int n_sites) {
    if (n_sites <= 0) {
        throw std::invalid_argument(
            "ed::sym::reflection_1d: n_sites must be positive (got " +
            std::to_string(n_sites) + ")");
    }
    Permutation p(static_cast<std::size_t>(n_sites));
    for (int i = 0; i < n_sites; ++i) {
        p[static_cast<std::size_t>(i)] = n_sites - 1 - i;
    }
    return p;
}

/// Swap exactly two sites `(a, b)`; identity on every other site.
[[nodiscard]] inline Permutation site_swap(int n_sites, int a, int b) {
    if (a < 0 || b < 0 || a >= n_sites || b >= n_sites) {
        throw std::invalid_argument(
            "ed::sym::site_swap: indices out of range");
    }
    Permutation p = identity(n_sites);
    std::swap(p[static_cast<std::size_t>(a)], p[static_cast<std::size_t>(b)]);
    return p;
}

// ---------------------------------------------------------------------------
// Group / sector enumeration
// ---------------------------------------------------------------------------

/// Expand a list of generators into the full group (BFS by left-multiplication).
/// Result is sorted (by lexicographic permutation order) so every caller gets
/// a canonical, deterministic ordering -- important because
/// `power_representation[i]` is keyed by position in `max_clique`.
[[nodiscard]] std::vector<Permutation>
generate_group(const std::vector<Permutation>& generators);

/// Build a fully-populated `SymmetryGroupInfo` from `generators`.
///
/// The returned struct is ready to be assigned to
/// `Operator::symmetry_info` (or `FixedSzOperator::symmetry_info`):
///
///   - `generators`         = the input list (validated)
///   - `generator_orders`   = `order(g)` for each `g`
///   - `max_clique`         = `generate_group(generators)`
///   - `power_representation[i]` = BFS solution `(p_0,...,p_{k-1})` with
///                                  `g_0^{p_0} o ... o g_{k-1}^{p_{k-1}} = max_clique[i]`
///   - `sectors`            = the abelian-irrep enumeration described below
///
/// If `sector_quantum_numbers` is empty (the default) the abelian
/// product `Z_{o_0} x ... x Z_{o_{k-1}}` is enumerated automatically:
/// every tuple `(q_0, ..., q_{k-1})` with `0 <= q_i < o_i` becomes a
/// sector with phase factors `phi_a = prod_k exp(-2 pi i q_k p_{a,k} / o_k)`
/// where `(p_{a,k})` is the power representation of group element
/// `max_clique[a]`. The legacy `filterInvalidSectors()` step removes
/// any "phantom irreps" produced by non-trivial generator relations.
///
/// @throws std::invalid_argument on any malformed generator (wrong
///         length, non-bijective, etc.).
[[nodiscard]] SymmetryGroupInfo
group_from_generators(int n_sites,
                      std::vector<Permutation> generators,
                      std::vector<std::vector<int>> sector_quantum_numbers = {});

/// Convenience: the cyclic translation group `Z_N` on a 1D ring of `N`
/// sites, with all `N` momentum sectors enumerated. Equivalent to
/// `group_from_generators(N, {translation(N)})`.
[[nodiscard]] SymmetryGroupInfo translation_group_1d(int n_sites);

/// Convenience: the dihedral group `D_N = Z_N x_| Z_2` on a 1D ring,
/// generated by `translation(N)` (order N) and `reflection_1d(N)`
/// (order 2). All `2N` sectors of the full abelian projection are
/// enumerated and then `filterInvalidSectors()` prunes the phantom
/// ones produced by the relation `R T R = T^{-1}`.
[[nodiscard]] SymmetryGroupInfo
translation_group_with_reflection_1d(int n_sites);

} // namespace ed::sym
