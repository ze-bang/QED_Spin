// =============================================================================
// python/qed/_bindings/workflow_bindings.h
//
// Forward declaration of the `bind_workflows(py::module_&)` entry point
// that populates `qed._core` with the Minimalist ED Collapse surface
// (`workflows_solve` / `workflows_thermal` / `workflows_spectral`
// + SolveOptions / ThermalOptions / SpectralOptions
// + GroundStateResult / ThermalResult / SpectralResult
// + KrylovDiagnostics / BackendMetadata / EigenvectorRef
// + BackendConstraints + the SolveMethod / ThermalMethod / SpectralMethod
//   enums).
//
// The implementation lives in `workflow_bindings.cpp` and is invoked from
// inside `PYBIND11_MODULE(_core, ...)` in `qed_bindings.cpp`.
//
// ED Cleanup Sweep Phase 1 (May 2026).
// =============================================================================

#pragma once

#include <pybind11/pybind11.h>

void bind_workflows(pybind11::module_& m);
