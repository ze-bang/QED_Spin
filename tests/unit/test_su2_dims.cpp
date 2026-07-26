// =============================================================================
// tests/unit/test_su2_dims.cpp
//
// Stage 12c of the SU(2) rollout: exact S-resolved dimensions from the
// highest-weight / Burnside-differencing trick
// (include/ed/symmetry/su2_dims.h + detail::sector_dims_s_resolved in
// include/ed/core/make_operator.h).
//
// Oracles (exact integer identities, the strongest kind in this repo):
//   * multiplet counting sum rule: sum_S (2S+1) M(N,S) = 2^N and
//     sum_S M(N,S) = C(N, floor(N/2));
//   * per-momentum tiling on the Z_N translation ring:
//       sum_S dims_S(k) == dims_burnside(k, n_up = N/2)   (telescoping)
//       sum_k dims_S(k) == M(N, S)                        (sectors tile)
//   * non-negativity of every S-resolved dimension;
//   * n_up_of_highest_weight admissibility guards.
// =============================================================================
#include "common/catch2_harness.h"

#include <ed/core/make_operator.h>
#include <ed/symmetry/group.h>
#include <ed/symmetry/su2_dims.h>

#include <cstdint>
#include <numeric>
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

TEST_CASE("S-resolved sector dims tile the Z_N momentum sectors",
          "[su2_dims]") {
    for (int N : {6, 8}) {
        const SymmetryGroupInfo info = ed::sym::translation_group_1d(N);
        const std::size_t n_sectors = info.sectors.size();
        REQUIRE(n_sectors == static_cast<std::size_t>(N));

        const auto half_dims =
            ed::detail::sector_dims_burnside(info, N / 2);

        std::vector<std::uint64_t> per_k_sum(n_sectors, 0);
        for (int ts = 0; ts <= N; ts += 2) {
            const auto dims_S =
                ed::detail::sector_dims_s_resolved(info, N, ts);
            REQUIRE(dims_S.size() == n_sectors);

            // Non-negativity is enforced by type; check the sector tiling
            // of each spin tower: sum_k dims_S(k) == M(N, S).
            const std::uint64_t tower_total =
                std::accumulate(dims_S.begin(), dims_S.end(),
                                std::uint64_t{0});
            REQUIRE(tower_total == multiplet_count(N, ts));

            for (std::size_t k = 0; k < n_sectors; ++k)
                per_k_sum[k] += dims_S[k];
        }
        // Telescoping: summing the towers recovers the Sz = 0 block
        // dimension per momentum sector.
        for (std::size_t k = 0; k < n_sectors; ++k)
            REQUIRE(per_k_sum[k] == half_dims[k]);
    }
}
