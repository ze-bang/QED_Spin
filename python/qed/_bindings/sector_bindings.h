// =============================================================================
// python/qed/_bindings/sector_bindings.h
//
// Forward declaration of the `bind_sectors(py::module_&)` entry point that
// exposes `ed::symmetry::SectorOperator` to Python as a first-class,
// applicable object.
//
// The workflow bindings already build symmetry sector sets internally, but
// they only ever hand back finished results (eigenvalues, thermodynamic
// curves, spectra). Callers that need to drive their own outer iteration --
// a blocked/filtered eigensolver over a band far wider than `num_eigenvalues`
// is meant for -- need the per-sector matvec itself. This surface provides
// exactly that and nothing more.
//
// The implementation lives in `sector_bindings.cpp` and is invoked from
// inside `PYBIND11_MODULE(_core, ...)` in `qed_bindings.cpp`.
// =============================================================================

#pragma once

#include <pybind11/pybind11.h>

void bind_sectors(pybind11::module_& m);
