// =============================================================================
// src/cli/dssf_engine.cpp
//
// Implementation of `ed::dssf::run(...)` -- the canonical DSSF/SSSF
// dispatcher introduced in P2.2 (DSSF PR-C, audit §3.10).
//
// This translation unit deliberately stays small and *only* contains the
// `run(...)` dispatcher (it has to live in `ed_cli` so it can call into
// the `compute_*_workflow` bodies that live in `src/cli/workflows.cpp`).
// The pure helpers `to_string` / `method_from_string` were split out
// into `src/dssf/dssf_method.cpp` (P2.3) so they can be linked from
// `ed_dssf` consumers (e.g. `dssf_io.cpp`) without dragging in `ed_cli`.
// =============================================================================

#include <ed/dssf/dssf_engine.h>

#include <ed/cli/workflows.h>

#include <stdexcept>
#include <string>

namespace ed::dssf {

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
