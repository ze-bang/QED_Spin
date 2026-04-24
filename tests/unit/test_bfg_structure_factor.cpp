// =============================================================================
// test_bfg_structure_factor (Catch2 v3, P2.1)
//
// Locks down the bond-bilinear structure factor / Fourier-applied dimer
// kernels that were pulled out of `compute_bfg_order_parameters.cpp`
// into `ed_bfg::structure_factor`.
//
// We work on a 2-site spin-1/2 chain with a single bond (0,1) at q = 0
// because every matrix element is hand-checkable:
//
//   convention: bit=0 -> spin UP, bit=1 -> spin DOWN
//
// 1. |psi> = |up,up> = |00>
//    XY dimer  D = S+S- + S-S+: D|00> = 0, so apply_dimer_fourier(0) = 0,
//                              SF = ||D|psi>||^2 = 0, <D> = 0.
//    Heisenberg D = S.S = SzSz + (1/2)(S+S- + S-S+):
//                              D|00> = (1/4)|00>, SF = 1/16, <D> = 1/4.
//
// 2. |psi> = (|01> + |10>)/sqrt(2)  (Sz=0 symmetric superposition)
//    D_XY |01> = |10>,   D_XY |10> = |01>  -> D_XY|psi> = |psi>
//      SF_XY = 1, <D_XY> = 1.
//    D_H |01> = -1/4|01> + 1/2|10>,  D_H |10> = -1/4|10> + 1/2|01>
//      D_H|psi> = (1/4)|psi>  -> SF_H = 1/16, <D_H> = 1/4.
//
// 3. compute_dimer_dimer_correlation((0,1),(0,1)) on |psi> = (|01>+|10>)/sqrt(2):
//      D_XY|01> = |10>, conj(psi[10]) = 1/sqrt(2)
//      D_XY|10> = |01>, conj(psi[01]) = 1/sqrt(2)
//      <psi|D_XY^2|psi> = (1/sqrt(2))^2 + (1/sqrt(2))^2 = 1.
//
// 4. set_memory_efficient_mode(0) must leave the global flag false, since
//    `0 * sizeof(Complex)` is well under any threshold.
// =============================================================================

#include "common/catch2_harness.h"

#include <ed/bfg/structure_factor.h>

#include <catch2/catch_approx.hpp>

#include <array>
#include <cmath>
#include <complex>
#include <utility>
#include <vector>

using Catch::Approx;
using ed::bfg::Complex;

namespace {

const std::vector<std::pair<int, int>> kBond01 = {{0, 1}};
const std::vector<std::array<double, 2>> kCenter01 = {{0.5, 0.0}};
const std::array<double, 2> kQzero{0.0, 0.0};

}  // namespace

TEST_CASE("ed::bfg::set_memory_efficient_mode is off for tiny systems",
          "[bfg][structure_factor][p2-1]") {
    ed::bfg::set_memory_efficient_mode(0);
    REQUIRE(ed::bfg::memory_efficient_mode_enabled() == false);

    ed::bfg::set_memory_efficient_mode(64);
    REQUIRE(ed::bfg::memory_efficient_mode_enabled() == false);
}

TEST_CASE("ed::bfg::compute_dimer_sf_direct vanishes on |up,up>",
          "[bfg][structure_factor][p2-1]") {
    std::vector<Complex> psi(4, 0.0);
    psi[0b00] = 1.0;

    auto r = ed::bfg::compute_dimer_sf_direct(psi, kBond01, kCenter01, kQzero);
    REQUIRE(r.overlap.real() == Approx(0.0).margin(1e-12));
    REQUIRE(std::abs(r.expect_q1) < 1e-12);
    REQUIRE(std::abs(r.expect_q2) < 1e-12);
}

TEST_CASE("ed::bfg::compute_heisenberg_sf_direct on |up,up> equals 1/16",
          "[bfg][structure_factor][p2-1]") {
    std::vector<Complex> psi(4, 0.0);
    psi[0b00] = 1.0;

    auto r = ed::bfg::compute_heisenberg_sf_direct(psi, kBond01, kCenter01, kQzero);
    REQUIRE(r.overlap.real() == Approx(1.0 / 16.0));
}

TEST_CASE("ed::bfg::apply_dimer_fourier maps (|01>+|10>)/sqrt(2) to itself",
          "[bfg][structure_factor][p2-1]") {
    std::vector<Complex> psi(4, 0.0);
    const double s = 1.0 / std::sqrt(2.0);
    psi[0b01] = s;
    psi[0b10] = s;

    auto out = ed::bfg::apply_dimer_fourier(psi, kBond01, kCenter01, kQzero);
    REQUIRE(out.size() == psi.size());
    REQUIRE(out[0b01].real() == Approx(s));
    REQUIRE(out[0b10].real() == Approx(s));
    REQUIRE(std::abs(out[0b00]) < 1e-12);
    REQUIRE(std::abs(out[0b11]) < 1e-12);
}

TEST_CASE("ed::bfg::apply_heisenberg_dimer_fourier on (|01>+|10>)/sqrt(2)"
          " returns (1/4)|psi> with <D> = 1/4",
          "[bfg][structure_factor][p2-1]") {
    std::vector<Complex> psi(4, 0.0);
    const double s = 1.0 / std::sqrt(2.0);
    psi[0b01] = s;
    psi[0b10] = s;

    auto [out, expect] = ed::bfg::apply_heisenberg_dimer_fourier(
        psi, kBond01, kCenter01, kQzero);

    REQUIRE(out.size() == psi.size());
    REQUIRE(out[0b01].real() == Approx(0.25 * s));
    REQUIRE(out[0b10].real() == Approx(0.25 * s));
    REQUIRE(std::abs(out[0b00]) < 1e-12);
    REQUIRE(std::abs(out[0b11]) < 1e-12);

    REQUIRE(expect.real() == Approx(0.25));
    REQUIRE(std::abs(expect.imag()) < 1e-12);
}

TEST_CASE("ed::bfg::compute_dimer_dimer_correlation((0,1),(0,1)) on the symm"
          " Sz=0 state equals 1.0",
          "[bfg][structure_factor][p2-1]") {
    std::vector<Complex> psi(4, 0.0);
    const double s = 1.0 / std::sqrt(2.0);
    psi[0b01] = s;
    psi[0b10] = s;

    Complex r = ed::bfg::compute_dimer_dimer_correlation(psi, 0, 1, 0, 1);
    REQUIRE(r.real() == Approx(1.0));
    REQUIRE(std::abs(r.imag()) < 1e-12);
}

TEST_CASE("ed::bfg::compute_heisenberg_dimer_dimer_correlation((0,1),(0,1))"
          " on (|01>+|10>)/sqrt(2) equals 1/16",
          "[bfg][structure_factor][p2-1]") {
    std::vector<Complex> psi(4, 0.0);
    const double s = 1.0 / std::sqrt(2.0);
    psi[0b01] = s;
    psi[0b10] = s;

    // (S.S)^2 with S=1/2 has eigenvalue (-3/4)^2 = 9/16 on the singlet and
    // (1/4)^2 = 1/16 on the triplet. (|01>+|10>)/sqrt(2) is the |triplet,
    // m=0> state, so <(S.S)^2> = 1/16.
    double r = ed::bfg::compute_heisenberg_dimer_dimer_correlation(psi, 0, 1, 0, 1);
    REQUIRE(r == Approx(1.0 / 16.0));
}
