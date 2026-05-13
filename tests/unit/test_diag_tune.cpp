// =============================================================================
// tests/unit/test_diag_tune.cpp
//
// Pure-numerical lockdown of the ED-solver auto-tuner heuristics in
// `include/ed/auto/diag_tune.h`. Mirrors the Python tune_diag tests in
// python/tests/test_auto_tune.py so that drift between the two
// implementations breaks the build.
// =============================================================================

#include <ed/auto/diag_tune.h>
#include <ed/core/ed_parameters.h>

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

using ed::auto_pilot::dssf::TuneLevel;
namespace td = ed::auto_pilot::diag;

TEST_CASE("pick_tolerance is monotone across levels",
          "[diag][auto_tune]") {
    REQUIRE(td::pick_tolerance(TuneLevel::Conservative)
            > td::pick_tolerance(TuneLevel::Balanced));
    REQUIRE(td::pick_tolerance(TuneLevel::Balanced)
            > td::pick_tolerance(TuneLevel::Aggressive));
}

TEST_CASE("pick_max_iterations is capped at sector_dim - 1",
          "[diag][auto_tune]") {
    REQUIRE(td::pick_max_iterations(4, 50) == 49u);
    // Big sector → max(floor=200, 8*1 + 80=88) = 200.
    REQUIRE(td::pick_max_iterations(1, 1ULL << 16, TuneLevel::Balanced)
            == 200u);
}

TEST_CASE("pick_max_subspace level ordering",
          "[diag][auto_tune]") {
    const std::uint64_t k = 4, D = 1ULL << 18;
    const auto cons = td::pick_max_subspace(k, D, TuneLevel::Conservative);
    const auto bal  = td::pick_max_subspace(k, D, TuneLevel::Balanced);
    const auto aggr = td::pick_max_subspace(k, D, TuneLevel::Aggressive);
    REQUIRE(cons < bal);
    REQUIRE(bal  < aggr);
}

TEST_CASE("pick_arpack_ncv >= 2k+1 across all levels",
          "[diag][auto_tune]") {
    for (std::uint64_t k : {1u, 4u, 16u, 64u}) {
        for (TuneLevel L : {TuneLevel::Conservative,
                            TuneLevel::Balanced,
                            TuneLevel::Aggressive}) {
            REQUIRE(td::pick_arpack_ncv(k, L)
                    >= static_cast<std::int64_t>(2 * k + 1));
        }
    }
}

TEST_CASE("pick_ftlm_krylov_dim and pick_ltlm_krylov_dim grow with level",
          "[diag][auto_tune]") {
    REQUIRE(td::pick_ftlm_krylov_dim(TuneLevel::Conservative)
            < td::pick_ftlm_krylov_dim(TuneLevel::Balanced));
    REQUIRE(td::pick_ftlm_krylov_dim(TuneLevel::Balanced)
            < td::pick_ftlm_krylov_dim(TuneLevel::Aggressive));
    REQUIRE(td::pick_ltlm_krylov_dim(TuneLevel::Conservative)
            < td::pick_ltlm_krylov_dim(TuneLevel::Balanced));
    REQUIRE(td::pick_ltlm_krylov_dim(TuneLevel::Balanced)
            < td::pick_ltlm_krylov_dim(TuneLevel::Aggressive));
}

TEST_CASE("pick_tpq_delta_beta is capped by 0.5 / bandwidth",
          "[diag][auto_tune]") {
    // Big bandwidth → 0.5 / W beats the level baseline.
    REQUIRE(td::pick_tpq_delta_beta(1000.0, TuneLevel::Balanced)
            == Catch::Approx(0.5 / 1000.0));
    // Small bandwidth → level baseline wins.
    REQUIRE(td::pick_tpq_delta_beta(1.0, TuneLevel::Balanced)
            == Catch::Approx(1e-2));
}

TEST_CASE("pick_tpq_taylor_order grows when bandwidth*dbeta is large",
          "[diag][auto_tune]") {
    const auto p_small = td::pick_tpq_taylor_order(
        /*bw=*/1.0, /*dbeta=*/1e-2, TuneLevel::Balanced);
    const auto p_large = td::pick_tpq_taylor_order(
        /*bw=*/100.0, /*dbeta=*/0.5, TuneLevel::Balanced);
    REQUIRE(p_small == 100u);
    REQUIRE(p_large >= p_small);
}

TEST_CASE("pick_num_thermal_samples decreases with sector_dim",
          "[diag][auto_tune]") {
    const auto big   = td::pick_num_thermal_samples(1ULL << 20,
                                                    TuneLevel::Balanced);
    const auto small = td::pick_num_thermal_samples(1024,
                                                    TuneLevel::Balanced);
    REQUIRE(small >= big);
    REQUIRE(big   >= 1u);
}

TEST_CASE("apply_auto_tune leaves caller-supplied EDParameters untouched",
          "[diag][auto_tune]") {
    EDParameters p;
    p.tolerance = 1.5e-7;        // user-set
    p.ftlm_krylov_dim = 999;     // user-set
    p.tpq_taylor_order = 42;     // user-set
    td::AutoTuneOverrides ov;
    ov.bandwidth = 4.0;
    ov.verbose = false;
    td::apply_auto_tune(p, /*sector_dim=*/1ULL << 16,
                        /*num_eigenvalues=*/4, /*op=*/nullptr, ov);
    REQUIRE(p.tolerance == Catch::Approx(1.5e-7));
    REQUIRE(p.ftlm_krylov_dim == 999u);
    REQUIRE(p.tpq_taylor_order == 42u);
}

TEST_CASE("apply_auto_tune fills sentinel-defaulted EDParameters fields",
          "[diag][auto_tune]") {
    EDParameters p;  // all struct defaults
    td::AutoTuneOverrides ov;
    ov.bandwidth = 4.0;
    ov.verbose = false;
    td::apply_auto_tune(p, /*sector_dim=*/1ULL << 16,
                        /*num_eigenvalues=*/4, /*op=*/nullptr, ov);
    // Sentinel defaults got replaced with auto-tuned values.
    REQUIRE(p.tolerance == Catch::Approx(1e-10));   // balanced default
    REQUIRE(p.max_iterations < 10000u);             // shrunk from sentinel
    REQUIRE(p.max_subspace >= 80u);                 // floor for L=balanced
    REQUIRE(p.arpack_ncv >= 2 * 4 + 1);             // >= 2k+1
    REQUIRE(p.ftlm_krylov_dim == 100u);             // balanced
    REQUIRE(p.ltlm_krylov_dim == 200u);             // balanced
    REQUIRE(p.tpq_delta_beta == Catch::Approx(1e-2));
    REQUIRE(p.tpq_taylor_order == 100u);
}
