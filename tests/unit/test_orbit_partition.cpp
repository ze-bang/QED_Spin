// =============================================================================
// test_orbit_partition  (Phase 3b #7, stage 1)
//
// Catch2 v3 lockdown for ed::distributed::balanced_orbit_slab() and the
// LPT-greedy partition it produces. No MPI required -- the partition is a
// deterministic function of (orbit_weights, n_ranks) and can be exercised
// in serial.
//
// Sections:
//   1. Trivial: empty orbit set; n_ranks=1; single orbit; n_ranks > n_orbits.
//   2. Determinism: same inputs always produce byte-identical output.
//   3. Coverage invariants: every orbit is owned by exactly one rank;
//      rank_orbits[r] is sorted ascending; rank_offsets is a strict
//      prefix-sum.
//   4. Equal-weight orbits: every rank gets the same count up to
//      remainder distribution, and load_imbalance() ~ 1.
//   5. Heavy outlier: one giant orbit + many small ones. The heavy orbit
//      pins the makespan; LPT achieves the textbook 4/3 - 1/(3 n_ranks)
//      bound. We assert load_imbalance() <= 4/3 over a range of weight
//      distributions.
//   6. Random stress: 10k random orbits, 13 ranks; partition matches the
//      LPT bound and every orbit is assigned exactly once.
// =============================================================================

#include "common/catch2_harness.h"

#include <ed/distributed/orbit_partition.h>

#include <algorithm>
#include <cstdint>
#include <numeric>
#include <random>
#include <unordered_set>

using ed::distributed::balanced_orbit_slab;
using ed::distributed::load_imbalance;
using ed::distributed::OrbitPartition;

namespace {

// Reference: every orbit appears exactly once in the partition; the
// rank_orbits arrays sum to a permutation of [0, n_orbits); rank_offsets
// is a strict prefix-sum of the per-rank counts.
void check_basic_invariants(const OrbitPartition& part,
                            std::size_t expected_n_orbits) {
    REQUIRE(part.n_orbits() == expected_n_orbits);
    REQUIRE(static_cast<int>(part.rank_orbits.size()) == part.n_ranks);
    REQUIRE(static_cast<int>(part.rank_weights.size()) == part.n_ranks);
    REQUIRE(static_cast<int>(part.rank_offsets.size()) == part.n_ranks + 1);

    // Every orbit owner is in [0, n_ranks).
    for (int o : part.orbit_owner) {
        REQUIRE(o >= 0);
        REQUIRE(o < part.n_ranks);
    }

    // rank_orbits[r] is sorted ascending.
    for (int r = 0; r < part.n_ranks; ++r) {
        const auto& v = part.rank_orbits[r];
        for (std::size_t i = 1; i < v.size(); ++i) {
            REQUIRE(v[i - 1] < v[i]);
        }
    }

    // Concatenation is a permutation of [0, n_orbits).
    std::vector<int> seen_count(expected_n_orbits, 0);
    for (int r = 0; r < part.n_ranks; ++r) {
        for (std::size_t orbit_idx : part.rank_orbits[r]) {
            REQUIRE(orbit_idx < expected_n_orbits);
            seen_count[orbit_idx] += 1;
            REQUIRE(part.orbit_owner[orbit_idx] == r);
        }
    }
    for (std::size_t i = 0; i < expected_n_orbits; ++i) {
        REQUIRE(seen_count[i] == 1);
    }

    // rank_offsets prefix-sum.
    REQUIRE(part.rank_offsets[0] == 0);
    for (int r = 0; r < part.n_ranks; ++r) {
        REQUIRE(part.rank_offsets[r] + part.rank_orbits[r].size()
                == part.rank_offsets[r + 1]);
    }
    REQUIRE(part.rank_offsets[part.n_ranks] == expected_n_orbits);
}

}  // namespace

TEST_CASE("balanced_orbit_slab: trivial inputs",
          "[orbit_partition][trivial]") {
    SECTION("empty orbit set") {
        OrbitPartition p = balanced_orbit_slab({}, 4);
        REQUIRE(p.n_ranks == 4);
        REQUIRE(p.n_orbits() == 0);
        for (int r = 0; r < 4; ++r) {
            REQUIRE(p.rank_orbits[r].empty());
            REQUIRE(p.rank_weights[r] == 0);
            REQUIRE(p.rank_offsets[r] == 0);
        }
        REQUIRE(p.rank_offsets[4] == 0);
        REQUIRE(load_imbalance(p) == 1.0);
    }

    SECTION("single rank: every orbit lands on rank 0") {
        std::vector<std::uint64_t> w{3, 1, 4, 1, 5, 9, 2, 6};
        OrbitPartition p = balanced_orbit_slab(w, 1);
        check_basic_invariants(p, w.size());
        REQUIRE(p.rank_orbits[0].size() == w.size());
        REQUIRE(p.rank_weights[0]
                == std::accumulate(w.begin(), w.end(), std::uint64_t{0}));
        REQUIRE(load_imbalance(p) == 1.0);
    }

    SECTION("single orbit, multiple ranks") {
        OrbitPartition p = balanced_orbit_slab({7}, 4);
        check_basic_invariants(p, 1u);
        // The lone orbit must end up on exactly one rank.
        int hosting = -1;
        for (int r = 0; r < 4; ++r) {
            if (!p.rank_orbits[r].empty()) {
                REQUIRE(hosting == -1);  // only one rank
                hosting = r;
                REQUIRE(p.rank_orbits[r].size() == 1u);
                REQUIRE(p.rank_orbits[r][0] == 0u);
                REQUIRE(p.rank_weights[r] == 7u);
            } else {
                REQUIRE(p.rank_weights[r] == 0u);
            }
        }
        REQUIRE(hosting != -1);
    }

    SECTION("more ranks than orbits") {
        std::vector<std::uint64_t> w{1, 1, 1};
        OrbitPartition p = balanced_orbit_slab(w, 8);
        check_basic_invariants(p, w.size());
        // Five ranks must be empty; three hold one orbit each.
        int empty_ranks = 0;
        int loaded_ranks = 0;
        for (int r = 0; r < 8; ++r) {
            if (p.rank_orbits[r].empty()) {
                ++empty_ranks;
                REQUIRE(p.rank_weights[r] == 0u);
            } else {
                ++loaded_ranks;
                REQUIRE(p.rank_orbits[r].size() == 1u);
                REQUIRE(p.rank_weights[r] == 1u);
            }
        }
        REQUIRE(empty_ranks == 5);
        REQUIRE(loaded_ranks == 3);
    }

    SECTION("zero-weight orbits are still assigned") {
        std::vector<std::uint64_t> w{0, 0, 0, 0};
        OrbitPartition p = balanced_orbit_slab(w, 2);
        check_basic_invariants(p, w.size());
        REQUIRE(p.rank_weights[0] == 0u);
        REQUIRE(p.rank_weights[1] == 0u);
        REQUIRE(load_imbalance(p) == 1.0);
    }

    SECTION("n_ranks <= 0 throws") {
        REQUIRE_THROWS_AS(balanced_orbit_slab({1, 2, 3}, 0),
                          std::invalid_argument);
        REQUIRE_THROWS_AS(balanced_orbit_slab({1, 2, 3}, -1),
                          std::invalid_argument);
    }
}

TEST_CASE("balanced_orbit_slab: deterministic across calls",
          "[orbit_partition][determinism]") {
    std::vector<std::uint64_t> w(257);
    std::mt19937_64 gen(0xDEADBEEFULL);
    std::uniform_int_distribution<std::uint64_t> d(1, 100);
    for (auto& x : w) x = d(gen);

    OrbitPartition a = balanced_orbit_slab(w, 7);
    OrbitPartition b = balanced_orbit_slab(w, 7);

    REQUIRE(a.n_ranks == b.n_ranks);
    REQUIRE(a.orbit_owner == b.orbit_owner);
    REQUIRE(a.rank_offsets == b.rank_offsets);
    REQUIRE(a.rank_weights == b.rank_weights);
    for (int r = 0; r < a.n_ranks; ++r) {
        REQUIRE(a.rank_orbits[r] == b.rank_orbits[r]);
    }
}

TEST_CASE("balanced_orbit_slab: equal-weight orbits balance well",
          "[orbit_partition][equal_weight]") {
    SECTION("count divisible by n_ranks") {
        std::vector<std::uint64_t> w(100, 5);
        OrbitPartition p = balanced_orbit_slab(w, 10);
        check_basic_invariants(p, w.size());
        for (int r = 0; r < 10; ++r) {
            REQUIRE(p.rank_orbits[r].size() == 10u);
            REQUIRE(p.rank_weights[r] == 50u);
        }
        REQUIRE(load_imbalance(p) == 1.0);
    }

    SECTION("count NOT divisible: imbalance bounded by ceil/floor ratio") {
        std::vector<std::uint64_t> w(103, 5);
        OrbitPartition p = balanced_orbit_slab(w, 10);
        check_basic_invariants(p, w.size());
        // 10 ranks share 103 orbits -> three ranks get 11, seven get 10.
        int n11 = 0, n10 = 0;
        for (int r = 0; r < 10; ++r) {
            std::size_t k = p.rank_orbits[r].size();
            REQUIRE((k == 10u || k == 11u));
            if (k == 11u) ++n11;
            if (k == 10u) ++n10;
        }
        REQUIRE(n11 == 3);
        REQUIRE(n10 == 7);
        // 55 / 51.5 = 1.068... -- well within the 4/3 bound.
        REQUIRE(load_imbalance(p) < 4.0 / 3.0);
    }
}

TEST_CASE("balanced_orbit_slab: heavy outlier respects LPT bound",
          "[orbit_partition][heavy_outlier]") {
    SECTION("one giant + many small") {
        // One orbit of weight 1000 + 99 orbits of weight 1.
        std::vector<std::uint64_t> w(100, 1);
        w[0] = 1000;
        OrbitPartition p = balanced_orbit_slab(w, 8);
        check_basic_invariants(p, w.size());
        // The 1000-weight orbit pins one rank's weight at >= 1000; the
        // rest of the work (99 small) spreads across the remaining seven
        // ranks (~14 each, weight ~14 each). Mean total weight = 1099/8 ~
        // 137; the heavy rank's weight is ~ 1000. Imbalance is ~7.3
        // because no greedy can do better in this pathological case.
        // We DO require LPT to put the small orbits on the OTHER ranks
        // first (i.e. NOT add them to the heavy rank), which is exactly
        // what the heap-based LPT does.
        // Heaviest rank weight should be exactly 1000 (the outlier alone).
        std::uint64_t mx = 0;
        for (int r = 0; r < 8; ++r) {
            if (p.rank_weights[r] > mx) mx = p.rank_weights[r];
        }
        REQUIRE(mx == 1000u);
    }

    SECTION("LPT 4/3 bound on geometrically-spaced weights") {
        // Geometric-like weight distribution -- the case where LPT is at
        // its closest to the worst-case 4/3 - 1/(3 n_ranks) bound.
        // For n_ranks = 8 the bound is 4/3 - 1/24 = 1.291666...
        std::vector<std::uint64_t> w;
        for (int i = 0; i < 50; ++i) {
            w.push_back(std::uint64_t{1} << (i % 8));
        }
        const int n_ranks = 8;
        OrbitPartition p = balanced_orbit_slab(w, n_ranks);
        check_basic_invariants(p, w.size());
        const double bound = 4.0 / 3.0 - 1.0 / (3.0 * n_ranks);
        const double imb = load_imbalance(p);
        INFO("imbalance=" << imb << " bound=" << bound);
        REQUIRE(imb <= bound + 1e-9);
    }
}

TEST_CASE("balanced_orbit_slab: random stress",
          "[orbit_partition][stress]") {
    std::vector<std::uint64_t> w(10'000);
    std::mt19937_64 gen(0xC0FFEE);
    std::uniform_int_distribution<std::uint64_t> d(1, 4096);
    for (auto& x : w) x = d(gen);

    const int n_ranks = 13;
    OrbitPartition p = balanced_orbit_slab(w, n_ranks);
    check_basic_invariants(p, w.size());

    // No orbit assignment ambiguity.
    std::unordered_set<std::size_t> all;
    for (int r = 0; r < n_ranks; ++r) {
        for (auto idx : p.rank_orbits[r]) {
            REQUIRE(all.insert(idx).second);
        }
    }
    REQUIRE(all.size() == w.size());

    // Sum of per-rank weights must equal total.
    std::uint64_t total = std::accumulate(w.begin(), w.end(),
                                          std::uint64_t{0});
    std::uint64_t sum_rank_w = std::accumulate(p.rank_weights.begin(),
                                               p.rank_weights.end(),
                                               std::uint64_t{0});
    REQUIRE(sum_rank_w == total);

    // Imbalance should be very close to 1 with so many small-weight
    // orbits; we use a generous bound to guard against regressions.
    const double bound = 4.0 / 3.0 - 1.0 / (3.0 * n_ranks);
    REQUIRE(load_imbalance(p) <= bound + 1e-9);
}

// =============================================================================
// Phase 3b #7 stage 2 prep: DistributedOperator-shaped accessors.
//
// `OrbitPartition` exposes owner_rank() / owner_local_index() /
// global_rank_major_index() so the future `DistributedSymmetryOperator`
// can build its halo plan in the same shape as the existing
// `DistributedOperator` (which uses balanced_slab + rank_offsets +
// upper_bound). These tests lock down the round-trip
//
//   orbit_id -> (owner_rank, owner_local_index) -> rank_orbits[owner][k]
//             -> orbit_id
//
// for a variety of partitions and ensure global_rank_major_index() is a
// permutation of [0, n_orbits).
// =============================================================================
TEST_CASE("OrbitPartition: owner_rank / owner_local_index round-trip",
          "[orbit_partition][stage2_accessors]") {
    SECTION("equal weights, divisible") {
        std::vector<std::uint64_t> w(40, 3);
        OrbitPartition p = balanced_orbit_slab(w, 8);
        check_basic_invariants(p, w.size());
        for (std::size_t i = 0; i < w.size(); ++i) {
            const int r = p.owner_rank(i);
            const std::size_t k = p.owner_local_index(i);
            REQUIRE(r >= 0);
            REQUIRE(r < p.n_ranks);
            REQUIRE(k < p.rank_orbits[r].size());
            REQUIRE(p.rank_orbits[r][k] == i);
        }
    }

    SECTION("heavy outlier + small fries") {
        std::vector<std::uint64_t> w(50, 1);
        w[3] = 1000;
        w[17] = 500;
        w[42] = 200;
        OrbitPartition p = balanced_orbit_slab(w, 6);
        check_basic_invariants(p, w.size());
        for (std::size_t i = 0; i < w.size(); ++i) {
            const int r = p.owner_rank(i);
            const std::size_t k = p.owner_local_index(i);
            REQUIRE(p.rank_orbits[static_cast<std::size_t>(r)][k] == i);
        }
    }

    SECTION("global_rank_major_index is a permutation of [0, n_orbits)") {
        std::vector<std::uint64_t> w(257);
        std::mt19937_64 gen(0xFEEDFACEULL);
        std::uniform_int_distribution<std::uint64_t> d(1, 64);
        for (auto& x : w) x = d(gen);
        OrbitPartition p = balanced_orbit_slab(w, 11);
        check_basic_invariants(p, w.size());

        std::vector<int> hits(w.size(), 0);
        for (std::size_t i = 0; i < w.size(); ++i) {
            const std::size_t g = p.global_rank_major_index(i);
            REQUIRE(g < w.size());
            hits[g] += 1;
        }
        for (std::size_t g = 0; g < w.size(); ++g) {
            REQUIRE(hits[g] == 1);
        }
    }

    SECTION("owner_rank() out-of-range returns -1") {
        OrbitPartition p = balanced_orbit_slab({1, 2, 3}, 2);
        REQUIRE(p.owner_rank(0) >= 0);
        REQUIRE(p.owner_rank(2) >= 0);
        REQUIRE(p.owner_rank(99) == -1);
    }

    SECTION("rank-major reordering matches DistributedOperator's "
            "balanced_slab semantics for equal weights") {
        // When orbit_weights are all equal AND n_orbits == n_ranks * k,
        // the LPT greedy degenerates to a contiguous round-robin (orbit i
        // -> rank i / k, position i % k).  Note: the heap-based LPT may
        // not produce contiguous rank assignments for all-equal weights
        // (it assigns to the lightest rank with ties broken by ascending
        // rank index), but the count per rank is exactly k.
        const int n_ranks = 4;
        const std::size_t k = 7;
        std::vector<std::uint64_t> w(n_ranks * k, 5);
        OrbitPartition p = balanced_orbit_slab(w, n_ranks);
        for (int r = 0; r < n_ranks; ++r) {
            REQUIRE(p.local_size(r) == k);
            REQUIRE(p.rank_offsets[r] == r * k);
        }
        REQUIRE(p.rank_offsets[n_ranks] == n_ranks * k);
    }
}
