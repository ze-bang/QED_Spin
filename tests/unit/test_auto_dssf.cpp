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
