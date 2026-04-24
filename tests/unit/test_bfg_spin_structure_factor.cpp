// =============================================================================
// test_bfg_spin_structure_factor (Catch2 v3, P2.1 6th slice)
//
// Locks down `ed::bfg::compute_spin_structure_factor`, which was promoted
// out of `compute_bfg_order_parameters.cpp` into the `ed_bfg` library.
//
// We work on a 2-site spin-1/2 cluster -- the smallest geometry with a
// non-trivial Fourier sum -- and feed in handcomputed two-body
// correlation tables. The kernel only consumes
// `cluster.k_points`, `cluster.n_sites`, and
// `cluster.minimum_image_displacement(i, j)`; we leave
// `n_cells_x = n_cells_y = 0` so the naive `positions[j] - positions[i]`
// fallback is exercised (no PBC wrapping needed for this test).
//
// Convention: bit=0 -> spin UP, bit=1 -> spin DOWN.
// =============================================================================

#include "common/catch2_harness.h"

#include <ed/bfg/cluster.h>
#include <ed/bfg/spin_structure_factor.h>

#include <catch2/catch_approx.hpp>

#include <array>
#include <cmath>
#include <complex>
#include <vector>

using Catch::Approx;
using ed::bfg::Complex;

namespace {

// Build a degenerate 2-site cluster with positions (0,0) and (1,0).
// k_points are provided explicitly per test.
ed::bfg::Cluster make_two_site_cluster(
    const std::vector<std::array<double, 2>>& k_points
) {
    ed::bfg::Cluster c;
    c.n_sites = 2;
    c.positions = {{0.0, 0.0}, {1.0, 0.0}};
    c.k_points = k_points;
    // Keep n_cells_x = n_cells_y = 0 so minimum_image_displacement falls
    // through to (positions[j] - positions[i]).
    return c;
}

}  // namespace

TEST_CASE("ed::bfg::compute_spin_structure_factor on |up,up> "
          "puts all weight at q=0 with m_translation = 1/2",
          "[bfg][spin_structure_factor][p2-1]") {
    // |up,up>: <S-_i S+_j> = 0 for all (i,j) because S+ annihilates |up>.
    // <Sz_i Sz_j> = 1/4 for all (i,j).
    const std::vector<std::vector<Complex>> smsp = {
        {Complex(0, 0), Complex(0, 0)},
        {Complex(0, 0), Complex(0, 0)},
    };
    const std::vector<std::vector<double>> szsz = {
        {0.25, 0.25},
        {0.25, 0.25},
    };

    const std::array<double, 2> q0{0.0, 0.0};
    const std::array<double, 2> qpi{M_PI, 0.0};
    auto cluster = make_two_site_cluster({q0, qpi});

    auto r = ed::bfg::compute_spin_structure_factor(smsp, szsz, cluster);

    // q=0: s_q_szsz = (1/2) * sum_{i,j} 0.25 * 1 = 0.5; s_q_smsp = 0.
    REQUIRE(r.s_q_szsz[0].real() == Approx(0.5).margin(1e-12));
    REQUIRE(r.s_q_smsp[0].real() == Approx(0.0).margin(1e-12));
    REQUIRE(r.s_q[0].real() == Approx(0.5).margin(1e-12));

    // q=pi: s_q_szsz = (1/2) * (0.25 - 0.25 - 0.25 + 0.25) = 0.
    REQUIRE(r.s_q_szsz[1].real() == Approx(0.0).margin(1e-12));
    REQUIRE(r.s_q[1].real() == Approx(0.0).margin(1e-12));

    // Max abs(S(q)) is at q=0 with value 0.5.
    REQUIRE(r.q_max_idx == 0);
    REQUIRE(r.q_max[0] == Approx(0.0).margin(1e-12));
    REQUIRE(r.s_q_max.real() == Approx(0.5).margin(1e-12));

    // m_translation = sqrt(max / N) = sqrt(0.5 / 2) = 0.5.
    REQUIRE(r.m_translation == Approx(0.5).margin(1e-12));
}

TEST_CASE("ed::bfg::compute_spin_structure_factor on Neel |up,down> "
          "puts all weight at q=pi with m_translation = sqrt(1/2)",
          "[bfg][spin_structure_factor][p2-1]") {
    // |up,down>: only smsp[1][1] = <down|S-_1 S+_1|down> = 1, others zero.
    // szsz: diagonal +1/4, off-diagonal -1/4 (Neel correlations).
    const std::vector<std::vector<Complex>> smsp = {
        {Complex(0, 0), Complex(0, 0)},
        {Complex(0, 0), Complex(1, 0)},
    };
    const std::vector<std::vector<double>> szsz = {
        { 0.25, -0.25},
        {-0.25,  0.25},
    };

    const std::array<double, 2> q0{0.0, 0.0};
    const std::array<double, 2> qpi{M_PI, 0.0};
    auto cluster = make_two_site_cluster({q0, qpi});

    auto r = ed::bfg::compute_spin_structure_factor(smsp, szsz, cluster);

    // q=0:
    //   s_q_smsp = (1/2) * (0 + 0 + 0 + 1*1) = 0.5
    //   s_q_szsz = (1/2) * (0.25 - 0.25 - 0.25 + 0.25) * 1 = 0.0
    //   S(q=0) = 0.0 + Re(0.5) = 0.5
    REQUIRE(r.s_q_smsp[0].real() == Approx(0.5).margin(1e-12));
    REQUIRE(r.s_q_szsz[0].real() == Approx(0.0).margin(1e-12));
    REQUIRE(r.s_q[0].real() == Approx(0.5).margin(1e-12));

    // q=pi (positions[1]-positions[0] = (1, 0)):
    //   phases (i,j): (0,0)->1, (0,1)->-1, (1,0)->-1, (1,1)->1
    //   s_q_smsp = (1/2) * (0 + 0 + 0 + 1*1) = 0.5
    //   s_q_szsz = (1/2) * (0.25*1 + (-0.25)*(-1) + (-0.25)*(-1) + 0.25*1)
    //            = (1/2) * 1.0 = 0.5
    //   S(q=pi) = 0.5 + Re(0.5) = 1.0
    REQUIRE(r.s_q_smsp[1].real() == Approx(0.5).margin(1e-12));
    REQUIRE(r.s_q_szsz[1].real() == Approx(0.5).margin(1e-12));
    REQUIRE(r.s_q[1].real() == Approx(1.0).margin(1e-12));

    REQUIRE(r.q_max_idx == 1);
    REQUIRE(r.q_max[0] == Approx(M_PI).margin(1e-12));
    REQUIRE(r.s_q_max.real() == Approx(1.0).margin(1e-12));

    // m_translation = sqrt(1.0 / 2) = 1/sqrt(2).
    REQUIRE(r.m_translation == Approx(std::sqrt(0.5)).margin(1e-12));
}

TEST_CASE("ed::bfg::compute_spin_structure_factor handles empty k_points",
          "[bfg][spin_structure_factor][p2-1]") {
    const std::vector<std::vector<Complex>> smsp = {
        {Complex(0, 0), Complex(0, 0)},
        {Complex(0, 0), Complex(0, 0)},
    };
    const std::vector<std::vector<double>> szsz = {
        {0.25, 0.25},
        {0.25, 0.25},
    };

    auto cluster = make_two_site_cluster({});

    auto r = ed::bfg::compute_spin_structure_factor(smsp, szsz, cluster);
    REQUIRE(r.s_q.empty());
    REQUIRE(r.s_q_smsp.empty());
    REQUIRE(r.s_q_szsz.empty());
    // No q-points -> max scan never updates; defaults stay at 0.
    REQUIRE(r.q_max_idx == 0);
    REQUIRE(r.s_q_max.real() == Approx(0.0).margin(1e-12));
    REQUIRE(r.m_translation == Approx(0.0).margin(1e-12));
}
