// =============================================================================
// test_input_library (Catch2 v3)
//
// Lockdown for the standalone `ed::input` C++ library that replaces the
// Python `edlib/helper_*.py` workflow. Coverage:
//
//   1. `lattice::chain` produces the expected NN bond structure (PBC + OBC).
//   2. `lattice::square` produces |E| = 2 * Lx * Ly under PBC.
//   3. `lattice::pyrochlore` produces 4 * Lx * Ly * Lz sites with each up
//      tetrahedron contributing 6 NN bonds.
//   4. `HamiltonianBuilder::heisenberg` materialises the same matrix-free
//      Operator as the existing programmatic `build_heisenberg_chain`
//      fixture (spectra match to 1e-12).
//   5. `HamiltonianBuilder::write_directory` -> `Operator::loadFromDirectory`
//      round-trips the same Heisenberg chain (loader path + builder path
//      give identical spectra).
//   6. `HamiltonianBuilder::xxz` collapses to the Heisenberg case when
//      Jxy == Jz.
// =============================================================================

#include "common/catch2_harness.h"

#include <ed/core/construct_ham.h>
#include <ed/input/input.h>

#include <Eigen/Dense>
#include <complex>
#include <filesystem>
#include <memory>
#include <vector>

using namespace ed_tests;
using ed::input::HamiltonianBuilder;
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
    SECTION("PBC chain has L bonds") {
        auto L = lat::chain(6, /*pbc=*/true);
        REQUIRE(L.nn_bonds.size() == 6);
        REQUIRE(L.pbc == true);
    }
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

TEST_CASE("HamiltonianBuilder::heisenberg matches programmatic Operator",
          "[input][builder][heisenberg]") {
    const uint64_t N = 4;
    const uint64_t dim = 1ULL << N;

    auto chain = lat::chain(N, /*pbc=*/false);
    HamiltonianBuilder builder(N);
    builder.heisenberg(chain.nn_pairs(), 1.0);

    auto op_built = builder.to_operator();
    auto built = reference_from_operator(*op_built, dim);

    auto op_ref = build_heisenberg_chain(N, 1.0);
    auto ref = reference_from_operator(*op_ref, dim);

    require_eigs_close(built.eigs, ref.eigs, dim, 1e-12,
                       "HamiltonianBuilder vs build_heisenberg_chain");
    double err = (built.H - ref.H).norm() /
                 std::max(ref.H.norm(), 1e-30);
    INFO("||H_built - H_ref|| / ||H_ref|| = " << err);
    REQUIRE(err < 1e-12);
}

TEST_CASE("HamiltonianBuilder::write_directory + loadFromDirectory roundtrip",
          "[input][builder][io]") {
    const uint64_t N = 4;
    const uint64_t dim = 1ULL << N;
    auto chain = lat::chain(N, /*pbc=*/false);

    HamiltonianBuilder builder(N);
    builder.heisenberg(chain.nn_pairs(), 1.0);

    std::string dir = make_scratch_dir("input_builder_io");
    ed::input::FileOptions opts;
    opts.write_lattice_metadata = true;
    builder.write_directory(dir, &chain, opts);

    REQUIRE(std::filesystem::exists(dir + "/Trans.dat"));
    REQUIRE(std::filesystem::exists(dir + "/InterAll.dat"));
    REQUIRE(std::filesystem::exists(dir + "/positions.dat"));

    auto op_loaded = std::make_unique<Operator>(N, 0.5f);
    op_loaded->loadFromFile(dir + "/Trans.dat");
    op_loaded->loadFromInterAllFile(dir + "/InterAll.dat");
    auto loaded = reference_from_operator(*op_loaded, dim);

    auto op_ref = build_heisenberg_chain(N, 1.0);
    auto ref = reference_from_operator(*op_ref, dim);

    require_eigs_close(loaded.eigs, ref.eigs, dim, 1e-10,
                       "loaded-from-builder-files vs programmatic Heisenberg");
    double err = (loaded.H - ref.H).norm() /
                 std::max(ref.H.norm(), 1e-30);
    REQUIRE(err < 1e-10);
}

TEST_CASE("HamiltonianBuilder::xxz collapses to heisenberg when Jxy == Jz",
          "[input][builder][xxz]") {
    const uint64_t N = 4;
    const uint64_t dim = 1ULL << N;
    auto chain = lat::chain(N, /*pbc=*/false);

    HamiltonianBuilder b1(N), b2(N);
    b1.heisenberg(chain.nn_pairs(), 0.7);
    b2.xxz(chain.nn_pairs(), 0.7, 0.7);

    auto h1 = reference_from_operator(*b1.to_operator(), dim);
    auto h2 = reference_from_operator(*b2.to_operator(), dim);
    require_eigs_close(h1.eigs, h2.eigs, dim, 1e-12,
                       "xxz(0.7,0.7) == heisenberg(0.7)");
    double err = (h1.H - h2.H).norm() / std::max(h1.H.norm(), 1e-30);
    REQUIRE(err < 1e-12);
}

TEST_CASE("HamiltonianBuilder::on_site_field adds diagonal Sz",
          "[input][builder][onsite]") {
    const uint64_t N = 3;
    const uint64_t dim = 1ULL << N;

    HamiltonianBuilder b(N);
    b.on_site_field(0.5);

    auto op = b.to_operator();
    auto r = reference_from_operator(*op, dim);
    // H = 0.5 * sum_i Sz_i; spectrum should be {-3/4, -1/4, -1/4, -1/4, 1/4, 1/4, 1/4, 3/4}
    std::vector<double> expected = {-0.75, -0.25, -0.25, -0.25,
                                    0.25, 0.25, 0.25, 0.75};
    require_eigs_close(r.eigs, expected, dim, 1e-12,
                       "on_site_field spectrum");
}
