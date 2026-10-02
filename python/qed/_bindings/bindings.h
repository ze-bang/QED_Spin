// =============================================================================
// python/qed/_bindings/bindings.h -- the submodule entry points PYBIND11_MODULE(_core, ...) in
// core.cpp calls: bind_input fills qed._core.input (input.cpp: Lattice, the lattice
// generators, HamiltonianBuilder), bind_sectors qed._core.sectors (sectors.cpp: the engine).
// =============================================================================
#pragma once

#include <pybind11/pybind11.h>

void bind_input(pybind11::module_& m);
void bind_sectors(pybind11::module_& m);
