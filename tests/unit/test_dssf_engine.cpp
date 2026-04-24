// =============================================================================
// test_dssf_engine (Catch2 v3, P2.2 / DSSF PR-C lockdown)
//
// Sanity tests for `ed::dssf::run(...)` / `DSSFMethod` -- the canonical
// dispatch entry point introduced in P2.2 (audit §3.10).
//
// What we lock down here:
//   * `to_string` / `method_from_string` round-trip for every enumerator.
//   * `method_from_string` is case-insensitive.
//   * `method_from_string` rejects unknown tokens with std::invalid_argument.
//   * `run(...)` rejects a null `request.config` with std::invalid_argument
//     (transitional P2.2 behaviour; will be lifted in P2.3 once the
//     workflow bodies move onto the engine seam directly).
//   * Numeric values of `DSSFMethod` are stable and contiguous (since they
//     are persisted in HDF5 metadata going forward).
//
// We do NOT exercise the underlying compute_*_workflow bodies here -- those
// are covered by the existing integration smokes (run_ed_smoke.sh,
// j3_h0_scan/) and by the dedicated workflow-level Catch2 fixtures that
// will land in P2.3 alongside the unified /dssf/ HDF5 schema.
// =============================================================================

#include "common/catch2_harness.h"

#include <ed/dssf/dssf_engine.h>

#include <stdexcept>
#include <string>

TEST_CASE("DSSFMethod string round-trip is the identity",
          "[dssf][engine][p2-2]") {
    using ed::dssf::DSSFMethod;
    using ed::dssf::method_from_string;
    using ed::dssf::to_string;

    for (auto m : {DSSFMethod::DYNAMICAL_THERMAL,
                   DSSFMethod::STATIC_THERMAL,
                   DSSFMethod::GROUND_STATE_DSSF,
                   DSSFMethod::SINGLE_EXPECTATION}) {
        const auto s = to_string(m);
        REQUIRE_FALSE(s.empty());
        const auto m2 = method_from_string(s);
        REQUIRE(static_cast<unsigned>(m2) == static_cast<unsigned>(m));
    }
}

TEST_CASE("method_from_string accepts mixed case",
          "[dssf][engine][p2-2]") {
    using ed::dssf::DSSFMethod;
    using ed::dssf::method_from_string;

    REQUIRE(method_from_string("Dynamical_Thermal") ==
            DSSFMethod::DYNAMICAL_THERMAL);
    REQUIRE(method_from_string("STATIC_THERMAL") ==
            DSSFMethod::STATIC_THERMAL);
    REQUIRE(method_from_string("Ground_State_DSSF") ==
            DSSFMethod::GROUND_STATE_DSSF);
    REQUIRE(method_from_string("single_EXPECTATION") ==
            DSSFMethod::SINGLE_EXPECTATION);
}

TEST_CASE("method_from_string rejects unknown tokens",
          "[dssf][engine][p2-2]") {
    using ed::dssf::method_from_string;
    REQUIRE_THROWS_AS(method_from_string(""),                std::invalid_argument);
    REQUIRE_THROWS_AS(method_from_string("dynamical"),       std::invalid_argument);
    REQUIRE_THROWS_AS(method_from_string("ftlm"),            std::invalid_argument);
    REQUIRE_THROWS_AS(method_from_string("dynamical thermal"), std::invalid_argument);
}

TEST_CASE("run() rejects null EDConfig in transitional P2.2 mode",
          "[dssf][engine][p2-2]") {
    using namespace ed::dssf;

    DSSFRequest req;
    req.method  = DSSFMethod::DYNAMICAL_THERMAL;
    req.config  = nullptr;
    REQUIRE_THROWS_AS(run(req), std::invalid_argument);
}

TEST_CASE("DSSFMethod numeric values are stable",
          "[dssf][engine][p2-2]") {
    using ed::dssf::DSSFMethod;
    REQUIRE(static_cast<unsigned>(DSSFMethod::DYNAMICAL_THERMAL)  == 0u);
    REQUIRE(static_cast<unsigned>(DSSFMethod::STATIC_THERMAL)     == 1u);
    REQUIRE(static_cast<unsigned>(DSSFMethod::GROUND_STATE_DSSF)  == 2u);
    REQUIRE(static_cast<unsigned>(DSSFMethod::SINGLE_EXPECTATION) == 3u);
}
