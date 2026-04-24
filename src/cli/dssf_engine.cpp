// =============================================================================
// src/dssf/dssf_engine.cpp
//
// Implementation of `ed::dssf::run(...)` -- the canonical DSSF/SSSF
// dispatcher introduced in P2.2 (DSSF PR-C, audit §3.10).
//
// This translation unit deliberately stays small and *only* contains the
// dispatch table + free functions on `DSSFMethod`. The actual workflow
// bodies live (for now) in `src/cli/workflows.cpp`. P2.3 / P2.4 will move
// them onto this seam one at a time so `ctest` stays green between commits.
// =============================================================================

#include <ed/dssf/dssf_engine.h>

#include <ed/cli/workflows.h>

#include <algorithm>
#include <stdexcept>
#include <string>

namespace ed::dssf {

namespace {

constexpr const char* kDynamicalThermal = "dynamical_thermal";
constexpr const char* kStaticThermal    = "static_thermal";
constexpr const char* kGroundStateDSSF  = "ground_state_dssf";
constexpr const char* kSingleExpect     = "single_expectation";

} // namespace

std::string to_string(DSSFMethod method) {
    switch (method) {
        case DSSFMethod::DYNAMICAL_THERMAL:  return kDynamicalThermal;
        case DSSFMethod::STATIC_THERMAL:     return kStaticThermal;
        case DSSFMethod::GROUND_STATE_DSSF:  return kGroundStateDSSF;
        case DSSFMethod::SINGLE_EXPECTATION: return kSingleExpect;
    }
    // Future-proof: unrecognised numeric value (shouldn't happen for a
    // strongly-typed enum, but be defensive against persisted enum values
    // from a future schema).
    throw std::invalid_argument(
        "ed::dssf::to_string: unrecognised DSSFMethod value " +
        std::to_string(static_cast<std::uint32_t>(method)));
}

DSSFMethod method_from_string(const std::string& token) {
    // Normalise to lowercase for forgiving CLI parsing.
    std::string normalized;
    normalized.reserve(token.size());
    std::transform(token.begin(), token.end(), std::back_inserter(normalized),
                   [](unsigned char c) { return std::tolower(c); });

    if (normalized == kDynamicalThermal) return DSSFMethod::DYNAMICAL_THERMAL;
    if (normalized == kStaticThermal)    return DSSFMethod::STATIC_THERMAL;
    if (normalized == kGroundStateDSSF)  return DSSFMethod::GROUND_STATE_DSSF;
    if (normalized == kSingleExpect)     return DSSFMethod::SINGLE_EXPECTATION;

    throw std::invalid_argument(
        "ed::dssf::method_from_string: unrecognised method token '" +
        token + "'. Valid tokens: dynamical_thermal, static_thermal, "
        "ground_state_dssf, single_expectation.");
}

DSSFResult run(const DSSFRequest& request) {
    DSSFResult result;
    result.method     = request.method;
    result.output_dir = request.output_dir;

    if (request.config == nullptr) {
        throw std::invalid_argument(
            "ed::dssf::run: request.config is null. The transitional "
            "P2.2 implementation requires the legacy EDConfig blob; this "
            "requirement will be lifted in P2.3.");
    }

    // Provisional task count: one per (operator pair × momentum point ×
    // temperature). The actual workflow may collapse some of these into
    // a single Lanczos run when multi-temperature optimisation kicks in;
    // we surface the *attempted* count for now (good enough for the
    // smoke-check use case described in DSSFResult's docstring).
    const auto num_pairs = request.operators.spin_combinations.size() *
                           request.operators.momentum_points.size();
    const auto num_temps =
        (request.method == DSSFMethod::GROUND_STATE_DSSF)
            ? 1u
            : (request.config->dynamical.num_temp_bins > 0
                   ? request.config->dynamical.num_temp_bins
                   : 1u);
    result.num_tasks_attempted =
        static_cast<std::uint64_t>(num_pairs) *
        static_cast<std::uint64_t>(num_temps);

    switch (request.method) {
        case DSSFMethod::DYNAMICAL_THERMAL:
            compute_dynamical_response_workflow(*request.config);
            return result;

        case DSSFMethod::STATIC_THERMAL:
        case DSSFMethod::SINGLE_EXPECTATION:
            // Legacy TPQ_DSSF treated single_expectation as a special
            // mode of the static workflow; we preserve that wiring here
            // for one release. P2.3 will give SINGLE_EXPECTATION its own
            // dedicated workflow body.
            compute_static_response_workflow(*request.config);
            return result;

        case DSSFMethod::GROUND_STATE_DSSF:
            compute_ground_state_dssf_workflow(*request.config);
            return result;
    }

    throw std::invalid_argument(
        "ed::dssf::run: unrecognised DSSFMethod value " +
        std::to_string(static_cast<std::uint32_t>(request.method)));
}

} // namespace ed::dssf
