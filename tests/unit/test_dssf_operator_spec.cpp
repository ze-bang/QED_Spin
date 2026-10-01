// =============================================================================
// test_dssf_operator_spec (Catch2 v3)
//
// Sanity tests for `ed::dssf::build_observables` /
// `ed::dssf::compute_transverse_bases`, the single source of truth for
// DSSF observable construction.
//
// What we lock down here:
//   * `compute_transverse_bases`: Q × polarization basis math, including
//     the parallel fallback to {y, polarization} or {x, polarization}.
//   * `build_observables`:
//       - `sum`          -> 1 observable per (Q, component), single-component names
//       - `transverse`   -> 2 per (Q, component) (NSF then SF)
//       - `sublattice`   -> one per sublattice, or the one selected
//       - `experimental` -> one per Q
//       - argument validation: empty inputs / wrong sizes throw
//
// We deliberately avoid asserting the matrix elements of the constructed
// Operators (that's covered by test_operator_apply);
// here we only assert the *shape* of the output and the bookkeeping that
// downstream observable naming depends on.
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
    s.components       = {2};                          // Sz
    s.momentum_points  = {{0.0, 0.0, 0.0}};
    s.polarization     = {1.0, 0.0, 0.0};
    s.unit_cell_size   = 4;
    s.num_sites        = 4;
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

TEST_CASE("build_observables: sum operator -- 1 observable per (Q, component)",
          "[dssf][build_observables][sum]") {
    auto spec = base_spec();
    spec.components      = {2, 0};
    spec.momentum_points = {{0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}};

    auto out = ed::dssf::build_observables(spec);

    // 2 momenta * 2 components = 4 observables, momentum-major.
    REQUIRE(out.operators.size() == 4);
    REQUIRE(out.names.size() == 4);
    REQUIRE(out.names[0].rfind("Sz_q_Qx", 0) == 0);
    REQUIRE(out.names[1].rfind("Sp_q_Qx", 0) == 0);   // component 0 is S+, not relabelled

    // Names should be deterministic and contain the Q components.
    for (const auto& n : out.names) {
        REQUIRE(n.find("_q_Qx") != std::string::npos);
    }
}

TEST_CASE("build_observables: transverse operator -- 2 observables per (Q, component)",
          "[dssf][build_observables][transverse]") {
    auto spec = base_spec();
    spec.operator_type   = "transverse";
    spec.momentum_points = {{0.0, 0.0, 1.0}};

    auto out = ed::dssf::build_observables(spec);

    REQUIRE(out.operators.size() == 2);
    REQUIRE(out.names.size() == 2);

    // Output ordering is NSF then SF; downstream names depend on it.
    REQUIRE(out.names[0].find("_NSF") != std::string::npos);
    REQUIRE(out.names[1].find("_SF")  != std::string::npos);
}

TEST_CASE("build_observables: sublattice -- every sublattice vs one",
          "[dssf][build_observables][sublattice]") {
    auto spec = base_spec();
    spec.operator_type   = "sublattice";
    spec.unit_cell_size  = 2;
    spec.momentum_points = {{0.0, 0.0, 0.0}};

    SECTION("no selection -> one observable per sublattice") {
        auto out = ed::dssf::build_observables(spec);
        REQUIRE(out.operators.size() == 2);
        REQUIRE(out.names.size() == 2);
        REQUIRE(out.names[0].find("_sub0") != std::string::npos);
        REQUIRE(out.names[1].find("_sub1") != std::string::npos);
    }

    SECTION("selection -> exactly that sublattice") {
        spec.sublattice = 1;
        auto out = ed::dssf::build_observables(spec);
        REQUIRE(out.operators.size() == 1);
        REQUIRE(out.names.size() == 1);
        REQUIRE(out.names[0].find("_sub1") != std::string::npos);
    }
}

TEST_CASE("build_observables: experimental -- one per Q, components ignored",
          "[dssf][build_observables][experimental]") {
    auto spec = base_spec();
    spec.operator_type   = "experimental";
    spec.components.clear();
    spec.momentum_points = {{0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}};

    auto out = ed::dssf::build_observables(spec);
    REQUIRE(out.operators.size() == 2);
    REQUIRE(out.names[0].rfind("Experimental_q_Qx", 0) == 0);
}

TEST_CASE("build_observables: rejects malformed input",
          "[dssf][build_observables][validation]") {
    SECTION("empty components") {
        auto spec = base_spec();
        spec.components.clear();
        REQUIRE_THROWS_AS(ed::dssf::build_observables(spec), std::invalid_argument);
    }
    SECTION("empty momentum points") {
        auto spec = base_spec();
        spec.momentum_points.clear();
        REQUIRE_THROWS_AS(ed::dssf::build_observables(spec), std::invalid_argument);
    }
    SECTION("polarization not a 3-vector") {
        auto spec = base_spec();
        spec.polarization = {1.0, 0.0};
        REQUIRE_THROWS_AS(ed::dssf::build_observables(spec), std::invalid_argument);
    }
    SECTION("num_sites = 0") {
        auto spec = base_spec();
        spec.num_sites = 0;
        REQUIRE_THROWS_AS(ed::dssf::build_observables(spec), std::invalid_argument);
    }
    SECTION("unknown operator_type") {
        auto spec = base_spec();
        spec.operator_type = "totally_made_up";
        REQUIRE_THROWS_AS(ed::dssf::build_observables(spec), std::invalid_argument);
    }
}
