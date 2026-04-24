// =============================================================================
// test_bfg_cluster (Catch2 v3, P2.1 lockdown)
//
// Smoke tests for the new `ed::bfg::Cluster` + `load_cluster` library that
// was extracted from `src/apps/compute_bfg_order_parameters.cpp` so the same
// loader can be shared with the GPU binary, future Python bindings, and
// the in-progress library split below the main() driver.
//
// What we lock down here:
//   * load_cluster reads a minimal kagome 1x1 fixture and reports the
//     correct site count + NN-bond count.
//   * The distance-fallback NN detector finds all 3 nearest-neighbour
//     bonds when no `_nn_list.dat` file is present.
//   * Default kagome lattice vectors are filled in when no
//     `_lattice_parameters.dat` file is present, and reciprocal vectors
//     are derived consistently.
//   * `bond_orientation` is populated for every emitted edge, in both
//     orderings.
//   * `minimum_image_displacement(i, j)` is the negative of (j, i) for
//     the loaded geometry (basic sanity).
//
// The fixture lives at tests/fixtures/bfg_kagome_1x1/ and contains only
// `positions.dat` -- exercising the bare-minimum loader path.
// =============================================================================

#include "common/catch2_harness.h"

#include <ed/bfg/cluster.h>

#include <cmath>
#include <stdexcept>

namespace {

constexpr double kTol = 1e-9;

#ifndef ED_TEST_BFG_KAGOME_1X1_DIR
#error "ED_TEST_BFG_KAGOME_1X1_DIR must be defined to a real cluster fixture path"
#endif

}  // namespace

TEST_CASE("ed::bfg::load_cluster reads a 3-site kagome 1x1 fixture",
          "[bfg][cluster][load]") {
    const auto cluster = ed::bfg::load_cluster(ED_TEST_BFG_KAGOME_1X1_DIR);

    REQUIRE(cluster.n_sites == 3);
    REQUIRE(cluster.positions.size() == 3);
    REQUIRE(cluster.sublattice == std::vector<int>{0, 1, 2});

    // Distance-fallback NN detector should pick up all 3 unique edges.
    REQUIRE(cluster.edges_nn.size() == 3);
    // Both halves of every edge should land in nn_list.
    REQUIRE(cluster.nn_list.size() == 3);
    for (int s = 0; s < 3; ++s) {
        REQUIRE(cluster.nn_list.at(s).size() == 2);
    }
}

TEST_CASE("ed::bfg::load_cluster fills default kagome lattice vectors when "
          "no _lattice_parameters.dat file is present",
          "[bfg][cluster][defaults]") {
    const auto cluster = ed::bfg::load_cluster(ED_TEST_BFG_KAGOME_1X1_DIR);

    REQUIRE(std::abs(cluster.a1[0] - 1.0) < kTol);
    REQUIRE(std::abs(cluster.a1[1])       < kTol);
    REQUIRE(std::abs(cluster.a2[0] - 0.5) < kTol);
    REQUIRE(std::abs(cluster.a2[1] - std::sqrt(3.0) / 2.0) < kTol);

    // Reciprocal vectors derived from a1, a2.
    const double det = cluster.a1[0] * cluster.a2[1]
                     - cluster.a1[1] * cluster.a2[0];
    REQUIRE(std::abs(det - std::sqrt(3.0) / 2.0) < kTol);
    REQUIRE(std::abs(cluster.b1[0]
                     - 2.0 * M_PI * cluster.a2[1] / det) < kTol);
    REQUIRE(std::abs(cluster.b2[1]
                     - 2.0 * M_PI * cluster.a1[0] / det) < kTol);
}

TEST_CASE("ed::bfg::Cluster::bond_orientation is populated for every NN edge "
          "in both orderings",
          "[bfg][cluster][geometry]") {
    const auto cluster = ed::bfg::load_cluster(ED_TEST_BFG_KAGOME_1X1_DIR);

    for (const auto& [i, j] : cluster.edges_nn) {
        REQUIRE(cluster.bond_orientation.count({i, j}) == 1);
        REQUIRE(cluster.bond_orientation.count({j, i}) == 1);
        REQUIRE(cluster.bond_orientation.at({i, j})
                == cluster.bond_orientation.at({j, i}));
        const int o = cluster.bond_orientation.at({i, j});
        REQUIRE((o == 0 || o == 1 || o == 2));
    }
}

TEST_CASE("ed::bfg::Cluster::minimum_image_displacement is antisymmetric",
          "[bfg][cluster][geometry]") {
    const auto cluster = ed::bfg::load_cluster(ED_TEST_BFG_KAGOME_1X1_DIR);

    for (int i = 0; i < cluster.n_sites; ++i) {
        for (int j = 0; j < cluster.n_sites; ++j) {
            if (i == j) continue;
            const auto dij = cluster.minimum_image_displacement(i, j);
            const auto dji = cluster.minimum_image_displacement(j, i);
            REQUIRE(std::abs(dij[0] + dji[0]) < kTol);
            REQUIRE(std::abs(dij[1] + dji[1]) < kTol);
        }
    }
}

TEST_CASE("ed::bfg::load_cluster throws if positions.dat is missing",
          "[bfg][cluster][validation]") {
    REQUIRE_THROWS_AS(ed::bfg::load_cluster("/path/that/does/not/exist"),
                      std::runtime_error);
}
