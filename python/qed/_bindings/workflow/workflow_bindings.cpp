// =============================================================================
// python/qed/_bindings/workflow/workflow_bindings.cpp
//
// pybind11 surface for the Minimalist ED Collapse entry points:
//
//   * ed::workflows::solve    -> qed._core.workflows_solve(op, opts)
//   * ed::workflows::thermal  -> qed._core.workflows_thermal(op, opts)
//   * ed::workflows::spectral -> qed._core.workflows_spectral(op, obs, opts)
//
// Plus the option / result / enum / diagnostics types.
//
// The legacy dispatcher surface that used to live in
// `dispatcher_bindings.cpp` (the `exact_diagonalization_*` family) was
// hard-removed in the surface-unification collapse (May 2026); all
// Python entry points (`qed.solve` / `qed.thermal` / `qed.spectral`)
// now route through the bindings below.
//
// WP11 (Sep 2026) split the 5k-line monolith into per-area translation
// units in this directory; `bind_workflows` stays the single entry point
// declared by `python/qed/_bindings/workflow_bindings.h` and called from
// PYBIND11_MODULE(_core, ...) in `qed_bindings.cpp`. The registrars run in
// the same order the areas appeared in the old file: pybind11 renders a
// signature with the raw C++ type name for any type not yet registered, so
// registration order is part of the Python surface.
// =============================================================================

#include "../workflow_bindings.h"

#include "workflow_bindings_internal.h"

void bind_workflows(py::module_& m) {
    bind_workflows_types(m);
    bind_workflows_symmetry(m);
    bind_workflows_solve(m);
    bind_workflows_thermal(m);
    bind_workflows_spectral(m);
    bind_workflows_solve_streaming(m);
    bind_workflows_thermal_streaming(m);
    bind_workflows_spectral_streaming(m);
    bind_workflows_spectral_cross_irrep(m);
    bind_workflows_spectral_multiq(m);
    bind_workflows_spectral_ftlm(m);
    bind_workflows_thermal_all_sz(m);
}
