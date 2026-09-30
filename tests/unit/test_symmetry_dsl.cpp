// =============================================================================
// test_symmetry_dsl (Catch2 v3, P2.11)
//
// Lock down the programmatic `ed::sym` DSL (`ed/symmetry/group.h`):
//
//   1. Permutation algebra (compose / power / order / identity / validate).
//   2. Builders for translation / reflection_1d / site_swap.
//   3. `generate_group` produces a deterministic, sorted list closed under
//      composition.
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
