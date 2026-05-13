// =============================================================================
// tests/unit/test_dssf_tune.cpp
//
// Pure-numerical lockdown of the C++ DSSF auto-tuner heuristics in
// `include/ed/auto/dssf_tune.h`. Mirrors python/tests/test_auto_tune.py
// so that drift between the two implementations breaks the build.
// =============================================================================

#include <ed/auto/dssf_tune.h>
#include <ed/core/ed_config.h>

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <cmath>

using namespace ed::auto_pilot::dssf;

TEST_CASE("pick_omega_window is symmetric around zero",
          "[dssf][auto_tune]") {
    auto [omin, omax] = pick_omega_window(8.0);
    REQUIRE(omin == Catch::Approx(-omax));
    REQUIRE(omax == Catch::Approx(8.0 * 1.1));
}

TEST_CASE("pick_eta is tighter for aggressive than conservative",
          "[dssf][auto_tune]") {
    const double bw = 4.0;
    const int npts = 400;
    const double eta_cons = pick_eta(bw, npts, TuneLevel::Conservative);
    const double eta_bal  = pick_eta(bw, npts, TuneLevel::Balanced);
    const double eta_agg  = pick_eta(bw, npts, TuneLevel::Aggressive);
    REQUIRE(eta_agg < eta_bal);
    REQUIRE(eta_bal < eta_cons);
}

TEST_CASE("pick_eta scales linearly with the omega-grid spacing",
          "[dssf][auto_tune]") {
    const double bw = 4.0;
    const double eta_400 = pick_eta(bw, 400);
    const double eta_200 = pick_eta(bw, 200);
    // 400 points -> Δω = (omax-omin)/399; 200 points -> /199. Ratio:
    REQUIRE(eta_200 == Catch::Approx(eta_400 * (399.0 / 199.0)));
}

TEST_CASE("pick_krylov_dim is sublinear and clamped",
          "[dssf][auto_tune]") {
    const int small = pick_krylov_dim(100);
    const int big   = pick_krylov_dim(1'000'000);
    REQUIRE(small <= big);
    REQUIRE(big <= 200);  // balanced upper bound
}

TEST_CASE("pick_num_random_vectors decreases with sector dim",
          "[dssf][auto_tune]") {
    const int tiny = pick_num_random_vectors(100);
    const int huge = pick_num_random_vectors(100'000'000);
    REQUIRE(tiny >= huge);
    REQUIRE(huge >= 1);
    REQUIRE(tiny >= 4);   // balanced lower bound
}

TEST_CASE("pick_device respects an explicit user request",
          "[dssf][auto_tune]") {
    REQUIRE(pick_device(100, DSSFDevice::CPU) == DSSFDevice::CPU);
}

TEST_CASE("apply_auto_tune leaves caller-supplied EDConfig fields untouched",
          "[dssf][auto_tune]") {
    EDConfig cfg;
    cfg.system.num_sites = 4;
    cfg.dynamical.broadening = 0.07;          // user-set, should NOT be overwritten
    cfg.dynamical.omega_min  = -2.5;          // user-set
    cfg.dynamical.omega_max  =  2.5;          // user-set
    AutoTuneOverrides ov;
    ov.verbose = false;
    apply_auto_tune(cfg, /*sector_dim=*/16, /*op=*/nullptr, ov);
    REQUIRE(cfg.dynamical.broadening == Catch::Approx(0.07));
    REQUIRE(cfg.dynamical.omega_min  == Catch::Approx(-2.5));
    REQUIRE(cfg.dynamical.omega_max  == Catch::Approx( 2.5));
}

TEST_CASE("apply_auto_tune fills in defaults when caller leaves them at sentinel",
          "[dssf][auto_tune]") {
    EDConfig cfg;
    cfg.system.num_sites = 4;
    // Leave dynamical.* at struct defaults (broadening=0.1, num_omega_points=1000,
    // omega_min=-5.0, omega_max=5.0, krylov_dim=400, num_random_states=20).
    AutoTuneOverrides ov;
    ov.verbose = false;
    apply_auto_tune(cfg, /*sector_dim=*/16, /*op=*/nullptr, ov);
    // Auto-tune must have changed broadening + krylov + num_random away from
    // their sentinel defaults.
    REQUIRE(cfg.dynamical.broadening != Catch::Approx(0.1));
    REQUIRE(cfg.dynamical.krylov_dim != 400);
    REQUIRE(cfg.dynamical.num_random_states != 20);
}
