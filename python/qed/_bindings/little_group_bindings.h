// =============================================================================
// python/qed/_bindings/little_group_bindings.h
//
// Entry point that registers the _core.little_group_* verbs. Implemented in
// little_group_bindings.cpp; called from PYBIND11_MODULE(_core, ...) in
// qed_bindings.cpp once the Operator types exist.
// =============================================================================

#pragma once

#include <pybind11/pybind11.h>

void bind_little_group(pybind11::module_& m);
