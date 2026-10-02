// =============================================================================
// test_input_library (Catch2 v3)
//
// Lockdown for the standalone `ed::input` C++ library (the lattices; the Hamiltonian
// builder is Python, python/tests/test_input.py). Coverage:
//
//   1. `lattice::chain` produces the expected NN bond structure (PBC + OBC).
//   2. `lattice::square` produces |E| = 2 * Lx * Ly under PBC.
//   3. `lattice::pyrochlore` produces 4 * Lx * Ly * Lz sites with each up
//      tetrahedron contributing 6 NN bonds.
// =============================================================================

#include "common/catch2_harness.h"

#include <ed/ops/construct_ham.h>
#include <ed/input/input.h>

#include <Eigen/Dense>
#include <complex>
#include <memory>
#include <vector>

using namespace ed_tests;
using ed::input::Op;
namespace lat = ed::input::lattice;

TEST_CASE("ed::input::lattice::chain produces canonical NN bonds",
          "[input][lattice]") {
    SECTION("OBC chain has L-1 bonds") {
        auto L = lat::chain(6, /*pbc=*/false);
        REQUIRE(L.num_sites == 6);
        REQUIRE(L.nn_bonds.size() == 5);
        REQUIRE(L.pbc == false);
        REQUIRE(L.nn_bonds.front().i == 0);
        REQUIRE(L.nn_bonds.front().j == 1);
        REQUIRE(L.nn_bonds.back().j == 5);
    }
    SECTION("PBC chain has L bonds, the wrap bond oriented L-1 -> 0") {
        auto L = lat::chain(6, /*pbc=*/true);
        REQUIRE(L.nn_bonds.size() == 6);
        REQUIRE(L.pbc == true);
        REQUIRE(L.nn_bonds.back().i == 5);
        REQUIRE(L.nn_bonds.back().j == 0);
        REQUIRE(L.nnn_bonds.size() == 6);
        REQUIRE(L.nnnn_bonds.size() == 3);   // (i, i+3): one pair per antipodal couple
    }
}

TEST_CASE("ed::input::lattice::pyrochlore PBC has both tetrahedra",
          "[input][lattice]") {
    auto L = lat::pyrochlore(2, 2, 2, /*pbc=*/true);
    REQUIRE(L.num_sites == 32);
    REQUIRE(L.nn_bonds.size() == 96);
    std::vector<int> coord(L.num_sites, 0);
    for (const auto& b : L.nn_bonds) {
        ++coord[b.i];
        ++coord[b.j];
        REQUIRE(L.sublattice[b.i] < L.sublattice[b.j]);
    }
    for (int c : coord) REQUIRE(c == 6);
}

TEST_CASE("ed::input::lattice::square PBC bond count is 2*Lx*Ly",
          "[input][lattice]") {
    auto L = lat::square(3, 4, /*pbc=*/true);
    REQUIRE(L.num_sites == 12);
    REQUIRE(L.nn_bonds.size() == 24);  // 2 directions x 3*4 sites
}

TEST_CASE("ed::input::lattice::pyrochlore site + bond accounting",
          "[input][lattice]") {
    auto L = lat::pyrochlore(1, 1, 1, /*pbc=*/false);
    REQUIRE(L.num_sites == 4);
    // 1 up tetrahedron only (no PBC neighbours when Lx=Ly=Lz=1, OBC).
    REQUIRE(L.nn_bonds.size() == 6);
    for (int u = 0; u < 4; ++u) {
        REQUIRE(L.sublattice[u] == u);
    }
}
