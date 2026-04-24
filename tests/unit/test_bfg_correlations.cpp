// =============================================================================
// test_bfg_correlations (Catch2 v3, P2.1)
//
// Locks down the two-body spin correlations / bond expectations that were
// pulled out of `compute_bfg_order_parameters.cpp` into the `ed_bfg`
// library (`include/ed/bfg/correlations.h`).
//
// We test against analytic single-state expectations on a 2-site spin-1/2
// system because every value is hand-checkable:
//
//   convention: bit=0 -> spin UP (Sz=+1/2), bit=1 -> spin DOWN (Sz=-1/2)
//   states (LSB = site 0):
//     |00> = up   up    Sz tot = +1
//     |01> = down up    Sz tot =  0
//     |10> = up   down  Sz tot =  0
//     |11> = down down  Sz tot = -1
//
// 1. |psi> = |00> (both spins up)
//    <S^- S^+>     = diag(0, 0)   ;  off-diag = 0
//    <S^z S^z>     = diag(1/4)    ;  off-diag = +1/4
//    XY bond (0,1) = 0  (no flip-flop possible)
//
// 2. |psi> = (|01> + |10>)/sqrt(2)  (singlet/triplet symmetric mix in the
//    Sz=0 sector)
//    <S^z S^z>(0,1) = -1/4
//    <XY>     (0,1) = +1   (= S+S- + S-S+ matrix element on this state)
//    <S+S->   (0,1) = +1/2
//    <Heisenberg>(0,1) = -1/4 + (1/2)*1 = +1/4
// =============================================================================

#include "common/catch2_harness.h"

#include <ed/bfg/cluster.h>
#include <ed/bfg/correlations.h>

#include <catch2/catch_approx.hpp>

#include <complex>
#include <vector>

namespace {

ed::bfg::Cluster make_two_site_cluster() {
    ed::bfg::Cluster c;
    c.n_sites = 2;
    c.positions = {{0.0, 0.0}, {1.0, 0.0}};
    c.sublattice = {0, 1};
    c.edges_nn = {{0, 1}};
    return c;
}

}  // namespace

using Catch::Approx;
using ed::bfg::Complex;

TEST_CASE("ed::bfg::compute_smsp_correlations on |up,up> is all zero",
          "[bfg][correlations][p2-1]") {
    std::vector<Complex> psi(4, 0.0);
    psi[0b00] = 1.0;
    auto m = ed::bfg::compute_smsp_correlations(psi, 2);

    REQUIRE(m.size() == 2);
    REQUIRE(m[0].size() == 2);
    REQUIRE(m[0][0].real() == Approx(0.0).margin(1e-12));
    REQUIRE(m[1][1].real() == Approx(0.0).margin(1e-12));
    REQUIRE(std::abs(m[0][1]) < 1e-12);
    REQUIRE(std::abs(m[1][0]) < 1e-12);
}

TEST_CASE("ed::bfg::compute_szsz_correlations on |up,up> matches +1/4",
          "[bfg][correlations][p2-1]") {
    std::vector<Complex> psi(4, 0.0);
    psi[0b00] = 1.0;
    auto m = ed::bfg::compute_szsz_correlations(psi, 2);

    REQUIRE(m[0][0] == Approx(0.25));
    REQUIRE(m[1][1] == Approx(0.25));
    REQUIRE(m[0][1] == Approx(0.25));
    REQUIRE(m[1][0] == Approx(0.25));
}

TEST_CASE("ed::bfg::compute_xy_bond_expectations on (|01>+|10>)/sqrt(2) "
          "equals +1",
          "[bfg][correlations][p2-1]") {
    const double inv_sqrt2 = 1.0 / std::sqrt(2.0);
    std::vector<Complex> psi(4, 0.0);
    psi[0b01] = inv_sqrt2;   // site 0 down, site 1 up
    psi[0b10] = inv_sqrt2;   // site 0 up,   site 1 down

    const auto cluster = make_two_site_cluster();
    auto bonds = ed::bfg::compute_xy_bond_expectations(psi, cluster);

    REQUIRE(bonds.count({0, 1}) == 1);
    REQUIRE(bonds.at({0, 1}).real() == Approx(1.0));
    REQUIRE(bonds.at({0, 1}).imag() == Approx(0.0).margin(1e-12));
}

TEST_CASE("ed::bfg::compute_spsm_bond_expectations on (|01>+|10>)/sqrt(2) "
          "equals +1/2",
          "[bfg][correlations][p2-1]") {
    const double inv_sqrt2 = 1.0 / std::sqrt(2.0);
    std::vector<Complex> psi(4, 0.0);
    psi[0b01] = inv_sqrt2;
    psi[0b10] = inv_sqrt2;

    const auto cluster = make_two_site_cluster();
    auto bonds = ed::bfg::compute_spsm_bond_expectations(psi, cluster);

    REQUIRE(bonds.at({0, 1}).real() == Approx(0.5));
    REQUIRE(bonds.at({0, 1}).imag() == Approx(0.0).margin(1e-12));
}

TEST_CASE("ed::bfg::compute_szsz_bond_expectations + Heisenberg combiner on "
          "(|01>+|10>)/sqrt(2)",
          "[bfg][correlations][p2-1]") {
    const double inv_sqrt2 = 1.0 / std::sqrt(2.0);
    std::vector<Complex> psi(4, 0.0);
    psi[0b01] = inv_sqrt2;
    psi[0b10] = inv_sqrt2;

    const auto cluster = make_two_site_cluster();
    auto szsz = ed::bfg::compute_szsz_bond_expectations(psi, cluster);
    auto xy   = ed::bfg::compute_xy_bond_expectations(psi, cluster);
    auto heis = ed::bfg::compute_heisenberg_bond_expectations(szsz, xy);

    REQUIRE(szsz.at({0, 1}) == Approx(-0.25));
    REQUIRE(heis.at({0, 1}) == Approx(0.25));
}
