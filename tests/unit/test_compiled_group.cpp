// =============================================================================
// tests/unit/test_compiled_group.cpp
//
// Bit-identity contract of CompiledGroup:
//
//   CompiledGroup::apply(s, g) == applyPermutation(s, perm[g]) ^ flip[g]
//
// for every state and element, across the N range the byte-LUT covers
// (1..64), plus:
//
//   * content_hash is stable under recompilation and sensitive to any
//     element change.
// =============================================================================
#include "common/catch2_harness.h"

#include <ed/core/basis_utils.h>          // applyPermutation (scalar reference)
#include <ed/symmetry/compiled_group.h>

#include <algorithm>
#include <cstdint>
#include <random>
#include <vector>

using ed::symmetry::CompiledGroup;

namespace {

std::vector<int> random_perm(int n, std::mt19937_64& gen) {
    std::vector<int> p(n);
    for (int i = 0; i < n; ++i) p[i] = i;
    std::shuffle(p.begin(), p.end(), gen);
    return p;
}

std::uint64_t random_state(int n, std::mt19937_64& gen) {
    const std::uint64_t mask =
        (n >= 64) ? ~0ULL : ((1ULL << n) - 1ULL);
    return gen() & mask;
}

}  // namespace

TEST_CASE("CompiledGroup::apply matches scalar applyPermutation across N",
          "[compiled_group]") {
    std::mt19937_64 gen(20260702);
    for (int n : {1, 5, 8, 13, 16, 24, 31, 32, 37, 48, 63, 64}) {
        std::vector<std::vector<int>> perms;
        for (int g = 0; g < 6; ++g) perms.push_back(random_perm(n, gen));
        // Include the identity explicitly (is_identity contract).
        std::vector<int> ident(n);
        for (int i = 0; i < n; ++i) ident[i] = i;
        perms.push_back(ident);

        const CompiledGroup cg = CompiledGroup::from_permutations(perms, n);
        REQUIRE(cg.size() == perms.size());
        REQUIRE(cg.is_identity(perms.size() - 1));

        for (int trial = 0; trial < 200; ++trial) {
            const std::uint64_t s = random_state(n, gen);
            for (std::size_t g = 0; g < perms.size(); ++g) {
                REQUIRE(cg.apply(s, g) == applyPermutation(s, perms[g]));
            }
        }
    }
}

TEST_CASE("CompiledGroup flip elements XOR after the permutation",
          "[compiled_group]") {
    std::mt19937_64 gen(42);
    const int n = 20;
    std::vector<std::vector<int>> perms{random_perm(n, gen),
                                        random_perm(n, gen)};
    const std::uint64_t all_ones = (1ULL << n) - 1ULL;
    std::vector<std::uint64_t> flips{0ULL, all_ones};

    const CompiledGroup cg = CompiledGroup::from_elements(perms, flips, n);
    for (int trial = 0; trial < 200; ++trial) {
        const std::uint64_t s = random_state(n, gen);
        REQUIRE(cg.apply(s, 0) == applyPermutation(s, perms[0]));
        REQUIRE(cg.apply(s, 1) ==
                (applyPermutation(s, perms[1]) ^ all_ones));
    }
    // A pure flip on the identity permutation is NOT the identity element.
    std::vector<int> ident(n);
    for (int i = 0; i < n; ++i) ident[i] = i;
    const CompiledGroup cg2 = CompiledGroup::from_elements(
        {ident, ident}, {0ULL, all_ones}, n);
    REQUIRE(cg2.is_identity(0));
    REQUIRE(!cg2.is_identity(1));
}

TEST_CASE("CompiledGroup content_hash: stable and element-sensitive",
          "[compiled_group]") {
    std::mt19937_64 gen(7);
    const int n = 16;
    std::vector<std::vector<int>> perms{random_perm(n, gen),
                                        random_perm(n, gen)};
    const auto a = CompiledGroup::from_permutations(perms, n);
    const auto b = CompiledGroup::from_permutations(perms, n);
    REQUIRE(a.content_hash() == b.content_hash());

    // Any element change must change the hash.
    auto perms2 = perms;
    std::swap(perms2[1][0], perms2[1][1]);
    const auto c = CompiledGroup::from_permutations(perms2, n);
    REQUIRE(a.content_hash() != c.content_hash());

    // Same permutations, different flips: different hash.
    const auto d = CompiledGroup::from_elements(
        perms, {0ULL, (1ULL << n) - 1ULL}, n);
    REQUIRE(a.content_hash() != d.content_hash());
}
