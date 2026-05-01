// =============================================================================
// test_auto_dssf  (Catch2 v3)
//
// Smoke-tests the modern-C++ DSSF auto-pilot façade
// `ed::auto_pilot::dssf::pick_method(...)` and
// `ed::auto_pilot::dssf::compute(...)`. We only exercise the method-
// selection rule and the EDConfig-null guard here -- end-to-end DSSF
// numerics live in test_dssf_engine, test_dssf_io, etc.
// =============================================================================

#include "common/catch2_harness.h"

#include <ed/auto/dssf.h>
#include <ed/core/ed_config.h>
#include <ed/input/hamiltonian_builder.h>
#include <ed/input/lattice.h>

#include <filesystem>
#include <string>

using ed::auto_pilot::dssf::AutoDSSFOptions;
using ed::auto_pilot::dssf::compute;
using ed::auto_pilot::dssf::pick_method;
using ed::dssf::DSSFMethod;
using ed::dssf::DSSFRequest;

TEST_CASE("auto_pilot::dssf::pick_method follows the (T, omega) truth table",
          "[auto_pilot][dssf]") {
    REQUIRE(pick_method(/*T=*/false, /*w=*/false)
            == DSSFMethod::SINGLE_EXPECTATION);
    REQUIRE(pick_method(/*T=*/false, /*w=*/true)
            == DSSFMethod::GROUND_STATE_DSSF);
    REQUIRE(pick_method(/*T=*/true,  /*w=*/false)
            == DSSFMethod::STATIC_THERMAL);
    REQUIRE(pick_method(/*T=*/true,  /*w=*/true)
            == DSSFMethod::DYNAMICAL_THERMAL);
}

TEST_CASE("auto_pilot::dssf::compute throws when EDConfig is required but null",
          "[auto_pilot][dssf]") {
    DSSFRequest req;          // .config = nullptr by default
    req.output_dir = "/tmp/qed_dssf_autopilot_does_not_exist";

    AutoDSSFOptions opts;
    opts.has_temperature = true;
    opts.has_frequency   = true;
    opts.verbose         = false;

    REQUIRE_THROWS_AS(compute(req, opts), std::invalid_argument);
}

// ---------------------------------------------------------------------------
// End-to-end smoke: build a 4-site Heisenberg chain deck on disk via
// HamiltonianBuilder, populate an EDConfig + DSSFRequest, and round-trip
// through `ed::auto_pilot::dssf::compute(...)`. This exercises the full
// auto-pilot path -> ed::dssf::run -> compute_*_workflow on real data
// and is the C++ analog of the Python `qed.dssf.compute(...)` smoke test.
// ---------------------------------------------------------------------------
TEST_CASE("auto_pilot::dssf::compute runs static_thermal end-to-end on a "
          "4-site chain",
          "[auto_pilot][dssf][e2e]") {
    namespace fs = std::filesystem;
    using ed::input::HamiltonianBuilder;
    namespace lat = ed::input::lattice;

    const uint64_t N = 4;
    const std::string dir = ed_tests::make_scratch_dir("auto_dssf_e2e_static");
    const std::string out_dir = dir + "/output";
    fs::remove_all(out_dir);

    auto chain = lat::chain(N, /*pbc=*/true);
    HamiltonianBuilder builder(N);
    builder.heisenberg(chain.nn_pairs(), 1.0);
    ed::input::FileOptions fopts;
    fopts.write_lattice_metadata = true;
    builder.write_directory(dir, &chain, fopts);

    EDConfig cfg;
    cfg.system.num_sites       = N;
    cfg.system.spin_length     = 0.5f;
    cfg.system.hamiltonian_dir = dir;
    cfg.workflow.output_dir    = out_dir;
    // Keep the static-response workload tiny so the test stays fast.
    cfg.static_resp.num_random_states = 2;
    cfg.static_resp.krylov_dim        = 16;
    cfg.static_resp.temp_min          = 0.5;
    cfg.static_resp.temp_max          = 0.5;
    cfg.static_resp.num_temp_points   = 1;
    cfg.static_resp.spin_combinations = "2,2";       // SzSz only
    cfg.static_resp.momentum_points   = "0,0,0";     // Q = 0

    DSSFRequest req;
    req.config     = &cfg;
    req.output_dir = out_dir;
    req.operators.num_sites      = N;
    req.operators.spin_length    = 0.5f;
    req.operators.positions_file = dir + "/positions.dat";
    req.operators.spin_combinations = {{2, 2}};
    req.operators.momentum_points   = {{0.0, 0.0, 0.0}};

    AutoDSSFOptions opts;
    opts.has_temperature = true;
    opts.has_frequency   = false;
    opts.verbose         = false;

    auto result = compute(req, opts);
    REQUIRE(result.method == DSSFMethod::STATIC_THERMAL);
    REQUIRE(result.output_dir == out_dir);
    // The static_thermal workflow writes ed_results.h5 in output_dir.
    REQUIRE(fs::exists(out_dir + "/ed_results.h5"));
}
