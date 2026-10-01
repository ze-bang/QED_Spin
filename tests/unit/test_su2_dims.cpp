// =============================================================================
// tests/unit/test_su2_dims.cpp
//
// Exact S-resolved dimensions from the
// highest-weight / Burnside-differencing trick
// (include/ed/basis/su2_dims.h).
//
// Oracles (exact integer identities, the strongest kind in this repo):
//   * multiplet counting sum rule: sum_S (2S+1) M(N,S) = 2^N and
//     sum_S M(N,S) = C(N, floor(N/2));
//   * n_up_of_highest_weight admissibility guards.
// =============================================================================
#include "common/catch2_harness.h"

#include <ed/basis/su2_dims.h>

#include <cstdint>
#include <vector>

using ed::symmetry::binomial_or_zero;
using ed::symmetry::multiplet_count;
using ed::symmetry::n_up_of_highest_weight;
using ed::symmetry::two_S_admissible;

TEST_CASE("multiplet counting sum rules", "[su2_dims]") {
    for (int N : {3, 4, 7, 8, 12}) {
        std::uint64_t states = 0, multiplets = 0;
        for (int ts = N % 2; ts <= N; ts += 2) {
            const auto m = multiplet_count(N, ts);
            states += static_cast<std::uint64_t>(ts + 1) * m;  // (2S+1) each
            multiplets += m;
        }
        REQUIRE(states == (1ULL << N));
        REQUIRE(multiplets == binomial_or_zero(N, N / 2));
    }
    // Spot values: N = 4 -> 2 singlets, 3 triplets, 1 quintet.
    REQUIRE(multiplet_count(4, 0) == 2);
    REQUIRE(multiplet_count(4, 2) == 3);
    REQUIRE(multiplet_count(4, 4) == 1);
    // Inadmissible two_S counts zero.
    REQUIRE(multiplet_count(4, 1) == 0);
    REQUIRE(multiplet_count(4, 6) == 0);
}

TEST_CASE("highest-weight n_up mapping and guards", "[su2_dims]") {
    REQUIRE(n_up_of_highest_weight(8, 0) == 4);   // Sz = 0
    REQUIRE(n_up_of_highest_weight(8, 8) == 8);   // fully polarized
    REQUIRE(n_up_of_highest_weight(5, 1) == 3);   // odd N, S = 1/2
    REQUIRE_FALSE(two_S_admissible(8, 1));        // parity mismatch
    REQUIRE_FALSE(two_S_admissible(8, 10));       // S > N/2
    REQUIRE_THROWS(n_up_of_highest_weight(8, 1));
}
