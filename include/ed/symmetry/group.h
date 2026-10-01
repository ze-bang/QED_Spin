// =============================================================================
// include/ed/symmetry/group.h
//
// `ed::sym` -- a small, programmatic DSL for site-permutation groups:
// the permutation algebra (identity / validate /
// compose / power / order), the common builders (translation,
// reflection_1d, site_swap) and `generate_group`, which closes a list of
// generators into the full group. Exposed to Python as `qed._core.sym`.
//
// What this DSL does NOT do
// -------------------------
//   * Internal Z2 spin-flip. Spin-flip is a bit-XOR action, not a site
//     permutation, so it lives outside this DSL.
//   * Irreps / sectors: the symmetry engine derives characters from the
//     group itself (ed/symmetry/irreps.h).
//
// All functions throw `std::invalid_argument` on malformed input
// (wrong-length permutation, non-bijective permutation, ...). Validation
// is cheap (<1us per generator), so we always run it.
// =============================================================================

#pragma once

#include <cstdint>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

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

/// Smallest `k > 0` with `g^k == identity`: the least common multiple of the cycle lengths.
/// Throws when `g` is not a permutation of 0..n-1.
[[nodiscard]] inline int order(const Permutation& g) {
    const int n = static_cast<int>(g.size());
    std::vector<char> hit(static_cast<std::size_t>(n), 0), seen(static_cast<std::size_t>(n), 0);
    for (int x : g) {
        if (x < 0 || x >= n || hit[static_cast<std::size_t>(x)])
            throw std::invalid_argument("ed::sym::order: input is not a permutation");
        hit[static_cast<std::size_t>(x)] = 1;
    }
    long long k = 1;
    for (int s = 0; s < n; ++s) {
        if (seen[static_cast<std::size_t>(s)]) continue;
        long long len = 0;
        for (int t = s; !seen[static_cast<std::size_t>(t)]; t = g[static_cast<std::size_t>(t)]) {
            seen[static_cast<std::size_t>(t)] = 1;
            ++len;
        }
        k = k / std::gcd(k, len) * len;
        if (k > std::numeric_limits<int>::max())
            throw std::overflow_error("ed::sym::order: the order does not fit in an int");
    }
    return static_cast<int>(k);
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
// Group closure
// ---------------------------------------------------------------------------

/// Expand a list of generators into the full group (BFS by left-multiplication).
/// Result is sorted (by lexicographic permutation order) so every caller gets
/// a canonical, deterministic ordering.
[[nodiscard]] std::vector<Permutation>
generate_group(const std::vector<Permutation>& generators);

} // namespace ed::sym
