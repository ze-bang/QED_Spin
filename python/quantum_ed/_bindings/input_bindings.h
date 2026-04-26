// =============================================================================
// python/quantum_ed/_bindings/input_bindings.h
//
// Forward declaration of the `bind_input(py::module_&)` entry point that
// populates `quantum_ed._core.input` with the `ed::input` C++ library
// surface (Lattice + lattice::* generators + HamiltonianBuilder +
// FileOptions + low-level file writers).
//
// The implementation lives in `input_bindings.cpp` and is invoked from
// inside `PYBIND11_MODULE(_core, ...)` in `quantum_ed_bindings.cpp`.
// =============================================================================

#pragma once

#include <pybind11/pybind11.h>

void bind_input(pybind11::module_& m);
