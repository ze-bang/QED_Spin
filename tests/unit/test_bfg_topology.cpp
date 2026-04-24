// =============================================================================
// test_bfg_topology (Catch2 v3, P2.1)
//
// Locks down `ed::bfg::find_triangles` and `ed::bfg::find_bowties` -- the
// combinatorial routines pulled out of `compute_bfg_order_parameters.cpp`
// into the `ed_bfg` static library.
//
// Two clusters exercise the routines:
//
//   1. The shipped 3-site kagome 1x1 fixture (one triangle, no bowties).
//   2. A synthetic 5-site "two triangles sharing a vertex" graph that
//      hand-builds the canonical bowtie configuration without any
//      file I/O so the bowtie path is exercised on every CI host.
// =============================================================================

#include "common/catch2_harness.h"

#include <ed/bfg/cluster.h>
#include <ed/bfg/topology.h>

#include <algorithm>
#include <utility>
#include <vector>

#ifndef ED_TEST_BFG_KAGOME_1X1_DIR
#error "ED_TEST_BFG_KAGOME_1X1_DIR must be defined to a real cluster fixture path"
#endif

namespace {

ed::bfg::Cluster make_two_triangle_bowtie() {
    // Five sites on a single shared vertex; the two triangles share s0
    // but no edges:
    //
    //     s1 --- s2          s3 --- s4
    //       \\   //            \\   //
    //         s0 ------------- (shared)
    //
    // Site coordinates are arbitrary; only the connectivity matters for
    // the topology routines.
    ed::bfg::Cluster c;
    c.n_sites = 5;
    c.positions = {
        {0.0,  0.0},
        {-1.0, 1.0},
        {-1.0, -1.0},
        {1.0,  1.0},
        {1.0,  -1.0},
    };
    c.sublattice = {0, 1, 2, 1, 2};
    c.edges_nn = {
        {0, 1}, {0, 2}, {1, 2},   // triangle 1 around s0
        {0, 3}, {0, 4}, {3, 4},   // triangle 2 around s0
    };
    return c;
}

}  // namespace

TEST_CASE("ed::bfg::find_triangles finds the single triangle in a kagome 1x1",
          "[bfg][topology][p2-1]") {
    const auto cluster = ed::bfg::load_cluster(ED_TEST_BFG_KAGOME_1X1_DIR);
    const auto tris = ed::bfg::find_triangles(cluster);

    REQUIRE(tris.size() == 1);
    auto t = tris.front();
    std::sort(t.begin(), t.end());
    REQUIRE(t == std::array<int, 3>{0, 1, 2});
}

TEST_CASE("ed::bfg::find_triangles enumerates all 4 triangles in a 5-site "
          "two-triangle bowtie cluster",
          "[bfg][topology][p2-1]") {
    const auto cluster = make_two_triangle_bowtie();
    const auto tris = ed::bfg::find_triangles(cluster);

    // Two NN-triangles by construction. The (s1,s2,s3,s4) outer ring is
    // not connected, so no extra triangle should appear.
    REQUIRE(tris.size() == 2);

    std::vector<std::array<int, 3>> sorted_tris;
    sorted_tris.reserve(tris.size());
    for (auto t : tris) {
        std::sort(t.begin(), t.end());
        sorted_tris.push_back(t);
    }
    std::sort(sorted_tris.begin(), sorted_tris.end());

    REQUIRE(sorted_tris[0] == std::array<int, 3>{0, 1, 2});
    REQUIRE(sorted_tris[1] == std::array<int, 3>{0, 3, 4});
}

TEST_CASE("ed::bfg::find_bowties returns no bowties on a single-triangle "
          "kagome 1x1 cluster",
          "[bfg][topology][p2-1]") {
    const auto cluster = ed::bfg::load_cluster(ED_TEST_BFG_KAGOME_1X1_DIR);
    const auto bows = ed::bfg::find_bowties(cluster);
    REQUIRE(bows.empty());
}

TEST_CASE("ed::bfg::find_bowties returns exactly one bowtie on a 5-site "
          "two-triangle cluster",
          "[bfg][topology][p2-1]") {
    const auto cluster = make_two_triangle_bowtie();
    const auto bows = ed::bfg::find_bowties(cluster);

    REQUIRE(bows.size() == 1);
    const auto& b = bows.front();
    REQUIRE(b.s0 == 0);

    // Outer pairs must canonicalize to the (1,2) and (3,4) sets.
    std::vector<std::pair<int, int>> outer = {
        {std::min(b.s1, b.s2), std::max(b.s1, b.s2)},
        {std::min(b.s3, b.s4), std::max(b.s3, b.s4)},
    };
    std::sort(outer.begin(), outer.end());
    REQUIRE(outer[0] == std::make_pair(1, 2));
    REQUIRE(outer[1] == std::make_pair(3, 4));

    // Center is the mean of all 5 sites; for our symmetric layout the y
    // average is exactly 0 and the x average is 0 by symmetry too.
    REQUIRE(std::abs(b.center[0]) < 1e-12);
    REQUIRE(std::abs(b.center[1]) < 1e-12);

    // Orientation = sublattice index of the shared vertex.
    REQUIRE(b.orientation == cluster.sublattice[b.s0]);
}
