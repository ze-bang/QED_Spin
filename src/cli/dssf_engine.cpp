// =============================================================================
// src/cli/dssf_engine.cpp
//
// Implementation of `ed::dssf::run(...)` -- the canonical DSSF/SSSF
// dispatcher introduced in P2.2 (DSSF PR-C, audit §3.10).
//
// This translation unit deliberately stays small and *only* contains the
// `run(...)` dispatcher (it has to live in `ed_cli` so it can call into
// the `compute_*_workflow` bodies that live in `src/cli/workflows.cpp`)
// and `run_cli(...)`, the `ED dssf` subcommand body shared by the `ED`
// executable and the `_core.dssf_run` Python binding (WP9.8).
// The pure helpers `to_string` / `method_from_string` were split out
// into `src/dssf/dssf_method.cpp` (P2.3) so they can be linked from
// `ed_dssf` consumers (e.g. `dssf_io.cpp`) without dragging in `ed_cli`.
// =============================================================================

#include <ed/dssf/dssf_engine.h>

#include <ed/cli/workflows.h>
#include <ed/core/system_utils.h>      // create_directory_mpi_safe

#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

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
            // SINGLE_EXPECTATION is the single-operator diagnostic flavour
            // of the static workflow (no Hermitian conjugate, no O₁†O₂
            // product); the static kernel branches on
            // `OperatorSpec::single_obs_only` so both flavours share the
            // same FTLM thermal averaging machinery.
            compute_static_response_workflow(*request.config);
            return result;

        case DSSFMethod::GROUND_STATE_DSSF:
            compute_ground_state_dssf_workflow(*request.config);
            return result;

        case DSSFMethod::KPM_THERMODYNAMICS:
            // Operator-free thermodynamics from the Chebyshev-expanded
            // density of states. The "structure" terminology is loose
            // here: KPM_THERMODYNAMICS does not consume an OperatorSpec
            // beyond ignoring it, but routing it through the DSSF
            // dispatcher gives consumers a single uniform entry point
            // for "give me thermodynamic observables for this H".
            compute_kpm_thermodynamics_workflow(*request.config);
            return result;
    }

    throw std::invalid_argument(
        "ed::dssf::run: unrecognised DSSFMethod value " +
        std::to_string(static_cast<std::uint32_t>(request.method)));
}

int run_cli(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Error: `ED dssf` requires a method argument.\n"
                  << "Usage: ED dssf <dynamical_thermal|static_thermal|"
                     "ground_state_dssf|single_expectation|"
                     "kpm_thermodynamics> "
                     "<directory> [options]\n";
        return 1;
    }

    DSSFMethod method;
    try {
        method = method_from_string(argv[2]);
    } catch (const std::invalid_argument& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    // Strip the "dssf <method>" prefix so EDConfig parses the rest of
    // argv as a normal ED invocation.
    std::vector<char*> cfg_argv;
    cfg_argv.reserve(argc - 1);
    cfg_argv.push_back(argv[0]);
    for (int i = 3; i < argc; ++i) cfg_argv.push_back(argv[i]);
    EDConfig sub_config = EDConfig::fromCommandLine(
        static_cast<int>(cfg_argv.size()), cfg_argv.data());

    if (!sub_config.validate()) {
        std::cerr << "\nConfiguration validation failed. Use --help.\n";
        return 1;
    }

    create_directory_mpi_safe(sub_config.workflow.output_dir);

    DSSFRequest request;
    request.method     = method;
    request.output_dir = sub_config.workflow.output_dir;
    request.config     = &sub_config;
    // operators left default-constructed: P2.2 transitional cut still
    // routes operator construction through the workflow body (which
    // reads sub_config.dynamical / .static_resp). P2.3 will populate
    // request.operators here from the same EDConfig fields.

    try {
        const auto result = run(request);
        std::cout << "\n[ED dssf] method=" << to_string(result.method)
                  << " tasks=" << result.num_tasks_attempted
                  << " output=" << result.output_dir << "\n";
    } catch (const std::exception& e) {
        std::cerr << "\nError: " << e.what() << "\n";
        return 1;
    }
    return 0;
}

} // namespace ed::dssf
