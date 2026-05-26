#pragma once
// =============================================================================
// include/ed/core/ed_legacy_types.h
//
// Slim residue of the ~2.1k-line `ed_wrapper.h` that the surface-
// unification collapse (May 2026) hard-deleted. Carries the small
// set of legacy POD types that the orchestrator's adapters and the
// CLI's standard / streaming workflows still consume:
//
//   * `EDResults`     - eigenvalue + thermo + FTLM result envelope
//                       (read by the Python `_ed_result_from_*`
//                       adapters and by the CLI's HDF5 emit step).
//
// The `EDParameters` parameter bag lives in
// `<ed/core/ed_parameters.h>` (separate file already split out by
// the modernization audit).
//
// Everything else that used to live in `ed_wrapper.h` -- the
// `exact_diagonalization_*` free-function family, the
// `HamiltonianFileFormat` enum (consumed only by those functions),
// the `hamiltonian_conserves_sz` probe, the `ed_internal::`
// file-loading helpers, etc. -- was deleted in lockstep with the
// pybind11 forwarder deletion. External consumers migrate to
// `ed::make_operator(OperatorSpec)` + `ed::workflows::{solve,
// thermal, spectral}` per `docs/MIGRATION.md`.
// =============================================================================

#include <ed/core/results.h>      // ThermodynamicData + FTLMResults

#include <string>
#include <vector>

/**
 * @brief Eigenvalue + thermodynamics + FTLM result envelope.
 *
 * Returned by the Python adapter (`qed.workflow._ed_result_from_*`)
 * and used by the CLI's HDF5 emit step (`run_standard_workflow` and
 * `run_streaming_symmetry_workflow` in `src/cli/workflows.cpp`).
 *
 * For ground-state lanes only `eigenvalues` and
 * `eigenvectors_*` are populated. For thermal lanes
 * (FTLM / LTLM / TPQ / KPM-DOS) the `thermo_data` block carries the
 * temperature scan; `ftlm_results` carries error-bar statistics.
 */
struct EDResults {
    std::vector<double> eigenvalues;
    bool eigenvectors_computed = false;
    std::string eigenvectors_path;
    ThermodynamicData thermo_data;
    FTLMResults       ftlm_results;
};
