// =============================================================================
// test_dssf_operator_spec (Catch2 v3, P1.10 / P2.5 / DSSF PR-A+F lockdown)
//
// Sanity tests for the new `ed::dssf::build_observable_pairs` /
// `ed::dssf::compute_transverse_bases` library that replaced ~500 LOC of
// duplicated operator-construction logic in `src/apps/ed_main.cpp` and
// `src/apps/TPQ_DSSF.cpp`.
//
// What we lock down here:
//   * `compute_transverse_bases`: Q × polarization basis math, including
//     the parallel fallback to {y, polarization} or {x, polarization}.
//   * `build_observable_pairs`:
//       - `sum`              -> 1 pair per (combo, Q), correct names
//       - `transverse`       -> 2 pairs per (combo, Q) (SF then NSF)
//       - `sublattice` filter / no-filter modes
//       - `single_obs_only`  -> obs_2 stays empty, names use single op
//       - argument validation: empty inputs / wrong sizes throw
//
// We deliberately avoid asserting the matrix elements of the constructed
// Operators (that's covered by test_operator_apply / test_observables);
// here we only assert the *shape* of the output and the bookkeeping that
// the legacy CLI has implicitly relied on.
// =============================================================================

#include "common/catch2_harness.h"

#include <ed/dssf/operator_spec.h>

#include <cmath>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {

constexpr double kBaseTol = 1e-12;

#ifndef ED_TEST_POSITIONS_4SITE
#error "ED_TEST_POSITIONS_4SITE must be defined to a real positions file path"
#endif

ed::dssf::OperatorSpec base_spec() {
    ed::dssf::OperatorSpec s;
    s.operator_type    = "sum";
    s.basis            = "ladder";
    s.spin_combinations = {{2, 2}};                 // SzSz
    s.momentum_points  = {{0.0, 0.0, 0.0}};
    s.polarization     = {1.0, 0.0, 0.0};
    s.unit_cell_size   = 4;
    s.num_sites        = 4;
    s.spin_length      = 0.5f;
    s.use_fixed_sz     = false;
    s.n_up             = 0;
    s.positions_file   = ED_TEST_POSITIONS_4SITE;
    return s;
}

}  // namespace

TEST_CASE("compute_transverse_bases: Q ⊥ polarization yields y-axis e2",
          "[dssf][transverse][geometry]") {
    const auto [e1, e2] = ed::dssf::compute_transverse_bases(
        /*Q=*/{0.0, 0.0, 1.0},
        /*pol=*/{1.0, 0.0, 0.0});

    REQUIRE(std::abs(e1[0] - 1.0) < kBaseTol);
    REQUIRE(std::abs(e1[1])       < kBaseTol);
    REQUIRE(std::abs(e1[2])       < kBaseTol);

    // Q × pol = (0, 0, 1) × (1, 0, 0) = (0, 1, 0)
    REQUIRE(std::abs(e2[0])       < kBaseTol);
    REQUIRE(std::abs(e2[1] - 1.0) < kBaseTol);
    REQUIRE(std::abs(e2[2])       < kBaseTol);
}

TEST_CASE("compute_transverse_bases: Q ∥ polarization triggers fallback basis",
          "[dssf][transverse][geometry]") {
    const auto [e1, e2] = ed::dssf::compute_transverse_bases(
        /*Q=*/{1.0, 0.0, 0.0},
        /*pol=*/{1.0, 0.0, 0.0});

    REQUIRE(std::abs(e1[0] - 1.0) < kBaseTol);
    // pol_x dominates -> fallback uses (0,1,0) × pol = (0, 0, -1) -> normalize
    const double e2_norm = std::sqrt(e2[0]*e2[0] + e2[1]*e2[1] + e2[2]*e2[2]);
    REQUIRE(std::abs(e2_norm - 1.0) < kBaseTol);
    // e2 must be orthogonal to e1 (the polarization vector)
    const double dot = e1[0]*e2[0] + e1[1]*e2[1] + e1[2]*e2[2];
    REQUIRE(std::abs(dot) < kBaseTol);
}

TEST_CASE("compute_transverse_bases: validates input shapes",
          "[dssf][transverse][validation]") {
    REQUIRE_THROWS_AS(ed::dssf::compute_transverse_bases({1.0}, {1.0, 0.0, 0.0}),
                      std::invalid_argument);
    REQUIRE_THROWS_AS(ed::dssf::compute_transverse_bases({1.0, 0.0, 0.0}, {1.0, 0.0}),
                      std::invalid_argument);
}

TEST_CASE("build_observable_pairs: sum operator -- 1 pair per (combo, Q)",
          "[dssf][build_pairs][sum]") {
    auto spec = base_spec();
    spec.spin_combinations = {{2, 2}, {0, 1}};
    spec.momentum_points  = {{0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}};

    auto out = ed::dssf::build_observable_pairs(spec);

    // 2 momenta * 2 combos = 4 pairs
    REQUIRE(out.obs_1.size() == 4);
    REQUIRE(out.obs_2.size() == 4);
    REQUIRE(out.names.size() == 4);

    // Names should be deterministic and contain the Q components.
    for (const auto& n : out.names) {
        REQUIRE(n.find("_q_Qx") != std::string::npos);
    }
}

TEST_CASE("build_observable_pairs: transverse operator -- 2 pairs per (combo, Q)",
          "[dssf][build_pairs][transverse]") {
    auto spec = base_spec();
    spec.operator_type   = "transverse";
    spec.momentum_points = {{0.0, 0.0, 1.0}};
    spec.spin_combinations = {{2, 2}};

    auto out = ed::dssf::build_observable_pairs(spec);

    REQUIRE(out.obs_1.size() == 2);
    REQUIRE(out.obs_2.size() == 2);
    REQUIRE(out.names.size() == 2);

    // Legacy ordering: NSF then SF. Lock that in.
    REQUIRE(out.names[0].find("_NSF") != std::string::npos);
    REQUIRE(out.names[1].find("_SF")  != std::string::npos);
}

TEST_CASE("build_observable_pairs: sublattice -- full triangle vs filter",
          "[dssf][build_pairs][sublattice]") {
    auto spec = base_spec();
    spec.operator_type  = "sublattice";
    spec.unit_cell_size = 2;
    spec.spin_combinations = {{2, 2}};
    spec.momentum_points  = {{0.0, 0.0, 0.0}};

    SECTION("no filter -> emits the upper-triangular {(0,0),(0,1),(1,1)}") {
        auto out = ed::dssf::build_observable_pairs(spec);
        REQUIRE(out.obs_1.size() == 3);
        REQUIRE(out.obs_2.size() == 3);
        REQUIRE(out.names.size() == 3);
        REQUIRE(out.names[0].find("_sub0_sub0") != std::string::npos);
        REQUIRE(out.names[1].find("_sub0_sub1") != std::string::npos);
        REQUIRE(out.names[2].find("_sub1_sub1") != std::string::npos);
    }

    SECTION("filter -> emits exactly that one pair") {
        spec.sublattice_filter = std::make_pair<std::uint64_t, std::uint64_t>(0, 1);
        auto out = ed::dssf::build_observable_pairs(spec);
        REQUIRE(out.obs_1.size() == 1);
        REQUIRE(out.obs_2.size() == 1);
        REQUIRE(out.names.size() == 1);
        REQUIRE(out.names[0].find("_sub0_sub1") != std::string::npos);
    }
}

TEST_CASE("build_observable_pairs: single_obs_only skips obs_2 and uses single name",
          "[dssf][build_pairs][single_obs_only]") {
    auto spec = base_spec();
    spec.single_obs_only = true;
    spec.spin_combinations = {{2, 2}};

    auto out = ed::dssf::build_observable_pairs(spec);
    REQUIRE(out.obs_1.size() == 1);
    REQUIRE(out.obs_2.empty());
    REQUIRE(out.names.size() == 1);

    // Name should start with single op label "Sz" (not "SzSz").
    REQUIRE(out.names[0].rfind("Sz", 0) == 0);
    REQUIRE(out.names[0].find("SzSz") == std::string::npos);
}

TEST_CASE("build_observable_pairs: rejects malformed input",
          "[dssf][build_pairs][validation]") {
    SECTION("empty spin combinations") {
        auto spec = base_spec();
        spec.spin_combinations.clear();
        REQUIRE_THROWS_AS(ed::dssf::build_observable_pairs(spec), std::invalid_argument);
    }
    SECTION("empty momentum points") {
        auto spec = base_spec();
        spec.momentum_points.clear();
        REQUIRE_THROWS_AS(ed::dssf::build_observable_pairs(spec), std::invalid_argument);
    }
    SECTION("polarization not a 3-vector") {
        auto spec = base_spec();
        spec.polarization = {1.0, 0.0};
        REQUIRE_THROWS_AS(ed::dssf::build_observable_pairs(spec), std::invalid_argument);
    }
    SECTION("num_sites = 0") {
        auto spec = base_spec();
        spec.num_sites = 0;
        REQUIRE_THROWS_AS(ed::dssf::build_observable_pairs(spec), std::invalid_argument);
    }
    SECTION("unknown operator_type") {
        auto spec = base_spec();
        spec.operator_type = "totally_made_up";
        REQUIRE_THROWS_AS(ed::dssf::build_observable_pairs(spec), std::invalid_argument);
    }
}
