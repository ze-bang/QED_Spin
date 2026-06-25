// =============================================================================
// test_symmetry_dsl (Catch2 v3, P2.11)
//
// Lock down the programmatic `ed::sym` DSL (`ed/symmetry/group.h`):
//
//   1. Permutation algebra (compose / power / order / identity / validate).
//   2. Builders for translation / reflection_1d / site_swap.
//   3. `generate_group` produces a deterministic, sorted list closed under
//      composition.
//   4. `group_from_generators` populates SymmetryGroupInfo correctly:
//        * generators + generator_orders matches the input
//        * max_clique = generate_group(generators)
//        * power_representation[i] reproduces max_clique[i]
//        * default sector enumeration produces |sectors| <= |G| (after
//          phantom-irrep filter)
//        * phase_factors[a] for sector q satisfies the abelian character
//          identity
//   5. `translation_group_1d(N)` matches the QuSpin/Bloch convention:
//      sector q encodes momentum k = 2 pi q / N and phase_factors[a] for
//      group element T^a is exp(-2 pi i q a / N).
//   6. `translation_group_with_reflection_1d(N)` correctly filters phantom
//      irreps via the `R T R = T^{-1}` relation.
// =============================================================================

#include "common/catch2_harness.h"

#include <catch2/catch_approx.hpp>

#include <ed/symmetry/group.h>

#include <cmath>
#include <complex>
#include <set>
#include <stdexcept>
#include <vector>

using ed::sym::Permutation;

namespace {

bool is_identity(const Permutation& p) {
    for (std::size_t i = 0; i < p.size(); ++i) {
        if (p[i] != static_cast<int>(i)) return false;
    }
    return true;
}

constexpr double kTwoPi = 6.283185307179586;

} // namespace

TEST_CASE("ed::sym permutation algebra basics", "[symmetry][p2-11][dsl]") {
    using namespace ed::sym;

    const auto id = identity(6);
    REQUIRE(is_identity(id));

    auto T = translation(6, 1);
    REQUIRE_NOTHROW(validate(T, 6));
    REQUIRE(order(T) == 6);

    auto T6 = power(T, 6);
    REQUIRE(is_identity(T6));

    auto R = reflection_1d(6);
    REQUIRE(order(R) == 2);
    REQUIRE(is_identity(power(R, 2)));

    // Dihedral relation: R T R == T^{-1} == T^{N-1}.
    auto lhs = compose(R, compose(T, R));
    auto rhs = power(T, 5);
    REQUIRE(lhs == rhs);
}

TEST_CASE("ed::sym validate rejects malformed permutations",
          "[symmetry][p2-11][dsl]") {
    using namespace ed::sym;
    REQUIRE_THROWS_AS(validate({0, 1, 1}, 3), std::invalid_argument);
    REQUIRE_THROWS_AS(validate({0, 1, 3}, 3), std::invalid_argument);
    REQUIRE_THROWS_AS(validate({0, 1},    3), std::invalid_argument);
    REQUIRE_THROWS_AS(identity(0),            std::invalid_argument);
    REQUIRE_THROWS_AS(power({0, 1, 2}, -1),   std::invalid_argument);
}

TEST_CASE("ed::sym generate_group is closed and deterministic",
          "[symmetry][p2-11][dsl]") {
    using namespace ed::sym;

    const int N = 5;
    auto group = generate_group({translation(N, 1)});
    REQUIRE(group.size() == static_cast<std::size_t>(N));

    // Closure: g o h is in the group for every (g, h).
    std::set<Permutation> set(group.begin(), group.end());
    for (const auto& g : group) {
        for (const auto& h : group) {
            REQUIRE(set.count(compose(g, h)) == 1);
        }
    }

    // Determinism: a second call returns the same vector verbatim.
    REQUIRE(generate_group({translation(N, 1)}) == group);
}

TEST_CASE("ed::sym group_from_generators populates SymmetryGroupInfo",
          "[symmetry][p2-11][dsl]") {
    using namespace ed::sym;

    const int N = 6;
    auto info = translation_group_1d(N);

    REQUIRE(info.num_generators   == 1);
    REQUIRE(info.generators.size() == 1);
    REQUIRE(info.generator_orders == std::vector<int>{N});
    REQUIRE(info.max_clique.size() == static_cast<std::size_t>(N));
    REQUIRE(info.power_representation.size() == info.max_clique.size());
    // Without a generator relation, |sectors| == |G|.
    REQUIRE(info.sectors.size() == static_cast<std::size_t>(N));

    // Each power_representation[i] reconstructs max_clique[i].
    for (std::size_t i = 0; i < info.max_clique.size(); ++i) {
        const int p0 = info.power_representation[i].at(0);
        Permutation rebuilt = power(info.generators[0], p0);
        REQUIRE(rebuilt == info.max_clique[i]);
    }

    // Bloch / QuSpin convention: phase factor for translation T^a in
    // momentum sector q is exp(-2 pi i q a / N).
    for (std::size_t s = 0; s < info.sectors.size(); ++s) {
        const int q = info.sectors[s].quantum_numbers.at(0);
        for (std::size_t a = 0; a < info.max_clique.size(); ++a) {
            const int pa = info.power_representation[a].at(0);
            const double angle = -kTwoPi * static_cast<double>(q * pa) /
                                 static_cast<double>(N);
            const std::complex<double> expected{std::cos(angle), std::sin(angle)};
            const std::complex<double> got = info.sectors[s].phase_factors[a];
            REQUIRE(std::abs(expected - got) < 1e-12);
        }
    }
}

TEST_CASE("ed::sym dihedral group is restricted to its abelian (Z_N) subgroup",
          "[symmetry][p2-11][dsl]") {
    using namespace ed::sym;

    const int N = 4;
    auto info = translation_group_with_reflection_1d(N);

    // D_N (|G|=2N) is NON-abelian, but the projection layer is abelian-only.
    // The non-abelian guard (group.cpp) restricts the input to a maximal abelian
    // subgroup -- the Z_N translations (|A|=N) -- which is a complete & correct
    // reduction (orbits AND characters agree), just coarser than the full 2N.
    // This replaces the previous (incomplete) "enumerate D_N then take the 8
    // one-dimensional Z_4 x Z_2 sectors" behavior, which silently dropped the
    // 2-dimensional irrep content.
    REQUIRE(info.max_clique.size() == static_cast<std::size_t>(N));   // |A| = N, not 2N

    // Z_N has exactly N one-dimensional irreps -> N momentum sectors {0..N-1}.
    REQUIRE(info.sectors.size() == static_cast<std::size_t>(N));
    std::set<int> got;
    for (const auto& s : info.sectors) got.insert(s.quantum_numbers.at(0));
    REQUIRE(got == std::set<int>{0, 1, 2, 3});
}

TEST_CASE("ed::sym filterInvalidSectors prunes phantom irreps",
          "[symmetry][p2-11][dsl]") {
    using namespace ed::sym;

    // Two commuting Z_2 site_swaps on a 4-site chain that happen to be
    // *equal*: g_0 == g_1 == swap(0,1). The naive product Z_2 x Z_2 has
    // 4 sectors but the generator relation `g_0 g_1 = e` (since
    // g_1^{-1} = g_0) means only the q_0 + q_1 == 0 (mod 2) sectors are
    // valid, leaving 2 sectors == |G|.
    auto g = site_swap(4, 0, 1);
    auto info = group_from_generators(4, {g, g});
    REQUIRE(info.max_clique.size() == 2);                // {e, swap(0,1)}
    REQUIRE(info.sectors.size()    == 2);                // phantom-pruned
    std::set<std::pair<int,int>> kept;
    for (const auto& s : info.sectors) {
        kept.insert({s.quantum_numbers[0], s.quantum_numbers[1]});
    }
    const std::set<std::pair<int,int>> expected = {{0,0}, {1,1}};
    REQUIRE(kept == expected);
}

TEST_CASE("ed::sym group_from_generators rejects bad input",
          "[symmetry][p2-11][dsl]") {
    using namespace ed::sym;
    REQUIRE_THROWS_AS(group_from_generators(4, {}), std::invalid_argument);
    REQUIRE_THROWS_AS(group_from_generators(0, {translation(4)}),
                      std::invalid_argument);
    REQUIRE_THROWS_AS(group_from_generators(4, {{0, 1, 1, 3}}),
                      std::invalid_argument);

    // Custom sectors with wrong cardinality should also bail out.
    REQUIRE_THROWS_AS(
        group_from_generators(4, {translation(4)}, {{0, 0}}),
        std::invalid_argument);
    REQUIRE_THROWS_AS(
        group_from_generators(4, {translation(4)}, {{4}}),
        std::invalid_argument);
}
