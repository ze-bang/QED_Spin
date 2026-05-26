// =============================================================================
// python/qed/_bindings/dispatcher_bindings.h
//
// Expose the residual "type-level" bindings to Python: the legacy
// ``EDParameters`` / ``EDResults`` / ``ThermodynamicData`` envelope
// types, the ``DiagonalizationMethod`` enum, the in-process
// symmetry attribute setters/getters on
// ``Operator`` / ``FixedSzOperator``, and the build-introspection
// probes (``has_cuda_build`` / ``has_mpi_build``).
//
// Surface unification (May 2026): the five
// ``exact_diagonalization_*`` Python forwarders that lived here in
// previous releases were deleted in lockstep with the C++
// ``ed::exact_diagonalization_*`` family removal. The canonical
// entry points are now ``_core.workflows_solve`` /
// ``workflows_thermal`` / ``workflows_spectral`` (in
// ``workflow_bindings.cpp``) and the public Python surface
// ``qed.solve`` / ``qed.thermal`` / ``qed.spectral``.
// =============================================================================

#pragma once

#include <pybind11/pybind11.h>

void bind_dispatcher(pybind11::module_& m);
