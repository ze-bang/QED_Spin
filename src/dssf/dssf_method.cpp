// =============================================================================
// src/dssf/dssf_method.cpp
//
// Pure helpers on `ed::dssf::DSSFMethod` -- `to_string` and
// `method_from_string`. These live in `ed_dssf` (NOT `ed_cli`) so any
// code that already links `ed_dssf` (including `dssf_io.cpp` itself,
// future Python bindings, and downstream users via `find_package(ED)`)
// can stamp the on-disk `@method` attribute and parse CLI tokens
// without dragging in the workflow machinery.
//
// The `run(...)` dispatcher continues to live in `src/cli/dssf_engine.cpp`
// because it calls into `compute_*_workflow` functions that are part of
// `ed_cli` -- moving it down here would create a circular dependency.
//
// Split out of `src/cli/dssf_engine.cpp` while wiring up the unified
// `/dssf/` HDF5 schema (P2.3 / DSSF PR-D, audit §3.10).
// =============================================================================

#include <ed/dssf/dssf_engine.h>

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <iterator>
#include <stdexcept>
#include <string>

namespace ed::dssf {

namespace {

constexpr const char* kDynamicalThermal  = "dynamical_thermal";
constexpr const char* kStaticThermal     = "static_thermal";
constexpr const char* kGroundStateDSSF   = "ground_state_dssf";
constexpr const char* kSingleExpect      = "single_expectation";
constexpr const char* kKPMThermodynamics = "kpm_thermodynamics";

} // namespace

std::string to_string(DSSFMethod method) {
    switch (method) {
        case DSSFMethod::DYNAMICAL_THERMAL:   return kDynamicalThermal;
        case DSSFMethod::STATIC_THERMAL:      return kStaticThermal;
        case DSSFMethod::GROUND_STATE_DSSF:   return kGroundStateDSSF;
        case DSSFMethod::SINGLE_EXPECTATION:  return kSingleExpect;
        case DSSFMethod::KPM_THERMODYNAMICS:  return kKPMThermodynamics;
    }
    throw std::invalid_argument(
        "ed::dssf::to_string: unrecognised DSSFMethod value " +
        std::to_string(static_cast<std::uint32_t>(method)));
}

DSSFMethod method_from_string(const std::string& token) {
    std::string normalized;
    normalized.reserve(token.size());
    std::transform(token.begin(), token.end(),
                   std::back_inserter(normalized),
                   [](unsigned char c) { return std::tolower(c); });

    if (normalized == kDynamicalThermal)  return DSSFMethod::DYNAMICAL_THERMAL;
    if (normalized == kStaticThermal)     return DSSFMethod::STATIC_THERMAL;
    if (normalized == kGroundStateDSSF)   return DSSFMethod::GROUND_STATE_DSSF;
    if (normalized == kSingleExpect)      return DSSFMethod::SINGLE_EXPECTATION;
    if (normalized == kKPMThermodynamics) return DSSFMethod::KPM_THERMODYNAMICS;

    throw std::invalid_argument(
        "ed::dssf::method_from_string: unrecognised method token '" +
        token + "'. Valid tokens: dynamical_thermal, static_thermal, "
        "ground_state_dssf, single_expectation, kpm_thermodynamics.");
}

} // namespace ed::dssf
