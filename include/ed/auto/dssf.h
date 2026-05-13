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

#include <ed/auto/dssf_tune.h>
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

    // Phase 9.2 (May 2026): auto-tune knobs forwarded to
    // `apply_auto_tune` when `auto_tune == true`. Anything left at the
    // sentinel value (NaN / 0 / DSSFDevice::Auto) is picked by the
    // tuner; any explicit value overrides it. See dssf_tune.h.
    bool                auto_tune       = true;
    AutoTuneOverrides   tune_overrides  = {};

    // Sector dimension hint used for tuning when no Operator is
    // attached to the request. When 0 (default), we fall back to
    // `request.config->system.num_sites`-derived 2^N.
    std::uint64_t       sector_dim_hint = 0;
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

    // Phase 9.2: auto-tune missing EDConfig knobs (eta / omega / krylov /
    // num_random / device) based on sector dim and operator coefficients.
    // Only fields still at their struct-default sentinel are overwritten,
    // so caller-supplied values pass through untouched.
    //
    // DSSFRequest::config is `const EDConfig*`, so we copy into a local
    // mutable EDConfig, mutate that, then re-point request.config at it
    // for the dispatcher call.
    EDConfig tuned_config;
    if (options.auto_tune
        && request.config != nullptr
        && request.method != DSSFMethod::SINGLE_EXPECTATION) {
        tuned_config = *request.config;
        std::uint64_t sector_dim = options.sector_dim_hint;
        if (sector_dim == 0) {
            const auto N = static_cast<std::uint64_t>(
                tuned_config.system.num_sites);
            sector_dim = (N > 0 && N < 64) ? (1ULL << N) : 1ULL;
        }
        AutoTuneOverrides tov = options.tune_overrides;
        tov.verbose = options.verbose;
        apply_auto_tune(tuned_config, sector_dim, /*op=*/nullptr, tov);
        request.config = &tuned_config;
    }

    return ed::dssf::run(request);
}

} // namespace ed::auto_pilot::dssf
