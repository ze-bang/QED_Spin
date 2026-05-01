#pragma once
// =============================================================================
// include/ed/auto/dssf.h
//
// `ed::auto_pilot::dssf::compute(...)` — companion to
// `ed::auto_pilot::solve(...)`. Maps a (T-given?, ω-given?) tuple to the
// right `DSSFMethod` enum value, then forwards to `ed::dssf::run(...)`.
//
// This is intentionally a thin layer: the heavy lifting (OperatorSpec
// construction, EDConfig population, observable assembly) still lives
// in the canonical CLI workflow path. The auto-pilot just removes the
// "which method should I pick?" decision from the user's path.
//
// Selection rule (matches the Python `qed.dssf.compute(...)` and the
// `ed::dssf::DSSFMethod` enum at include/ed/dssf/dssf_engine.h):
//
//     T given,  ω given      -> DYNAMICAL_THERMAL
//     T given,  ω absent     -> STATIC_THERMAL
//     T absent, ω given      -> GROUND_STATE_DSSF
//     T absent, ω absent     -> SINGLE_EXPECTATION
// =============================================================================

#include <ed/dssf/dssf_engine.h>

#include <iostream>
#include <stdexcept>
#include <string>

namespace ed::auto_pilot::dssf {

/// Explicit "is this axis present?" flags so callers don't have to go
/// through std::optional gymnastics for what is fundamentally a boolean
/// question. Either flag may be true; both being false is allowed and
/// selects SINGLE_EXPECTATION.
struct AutoDSSFOptions {
    bool   has_temperature = false;
    bool   has_frequency   = false;
    bool   verbose         = true;
};

/// Pure method picker. No I/O, no side effects. Useful in tests and as
/// a stepping stone for callers that still want to populate the
/// `DSSFRequest` fields by hand but don't want to encode the rule
/// themselves.
inline ed::dssf::DSSFMethod pick_method(bool has_temperature,
                                        bool has_frequency) noexcept {
    using ed::dssf::DSSFMethod;
    if (has_temperature && has_frequency)  return DSSFMethod::DYNAMICAL_THERMAL;
    if (has_temperature)                   return DSSFMethod::STATIC_THERMAL;
    if (has_frequency)                     return DSSFMethod::GROUND_STATE_DSSF;
    return DSSFMethod::SINGLE_EXPECTATION;
}

/// Auto-pilot DSSF runner. Mirrors `ed::auto_pilot::solve(...)`'s API
/// shape. The caller is still responsible for building a `DSSFRequest`
/// (operators, output_dir, optional EDConfig*); the auto-pilot only
/// fills in `request.method` from `options` and validates that the
/// supplied EDConfig is non-null when the chosen kernel needs one.
///
/// `request.method` IS OVERWRITTEN to keep the API a single source of
/// truth for the auto-rule. Callers who want to set the method
/// themselves should call `ed::dssf::run(request)` directly.
inline ed::dssf::DSSFResult compute(ed::dssf::DSSFRequest request,
                                    const AutoDSSFOptions& options = {}) {
    using ed::dssf::DSSFMethod;
    request.method = pick_method(options.has_temperature,
                                 options.has_frequency);
    if (options.verbose) {
        std::cerr << "[ed::auto_pilot::dssf::compute] method="
                  << ed::dssf::to_string(request.method)
                  << "  (T given: " << (options.has_temperature ? "true" : "false")
                  << ", omega given: " << (options.has_frequency ? "true" : "false")
                  << ")\n";
    }
    // Defensive: every kernel except SINGLE_EXPECTATION currently reads
    // its workflow knobs out of EDConfig (frequency window, broadening,
    // krylov dim, ...). Surface a clear error rather than letting the
    // dispatcher throw a less informative one downstream.
    if (request.config == nullptr
        && request.method != DSSFMethod::SINGLE_EXPECTATION) {
        throw std::invalid_argument(
            "ed::auto_pilot::dssf::compute: request.config must be non-null "
            "for method " + ed::dssf::to_string(request.method)
            + " (the workflow needs the per-method knobs from EDConfig).");
    }
    return ed::dssf::run(request);
}

} // namespace ed::auto_pilot::dssf
