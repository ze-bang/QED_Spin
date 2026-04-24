# exact_diagonalization

A modern C++17 / CUDA / Python toolkit for **exact diagonalization (ED)** of
spin-1/2 Hamiltonians, with first-class support for:

- Full diagonalization, Lanczos, block-Lanczos (CPU **and** GPU);
- Finite-temperature methods: **FTLM** and **LTLM**;
- **Dynamical / static structure factors (DSSF / SSSF)** with momentum and
  sublattice resolution;
- Symmetry sectors (translation, point-group, fixed-Sz);
- HDF5 I/O for eigenvectors, thermodynamic observables, and DSSF traces;
- A first-class **Python interface** (`quantum_ed`) built via `pybind11` +
  `scikit-build-core`, plus a compatibility shim for the legacy `edlib`
  package.

This site combines the **C++ API reference** (auto-extracted from the headers
under `include/ed/` via Doxygen + Breathe) with **prose chapters** that walk
through how the pieces fit together end-to-end.

```{toctree}
:maxdepth: 2
:caption: Getting started

guides/install
guides/quickstart
guides/python_quickstart
```

```{toctree}
:maxdepth: 2
:caption: Reference

api/cpp
api/python
```

```{toctree}
:maxdepth: 1
:caption: Project

../CHANGELOG
../CONTRIBUTING
```

## Why a "modern" rewrite?

The codebase is being modernized so collaborators can reuse it without
needing institutional knowledge:

- **CMake first**, with `find_package(ED CONFIG)` for downstream consumers,
  install rules, presets, and a shared static-library layout
  (`ed_core` / `ed_io` / `ed_dssf` / `ed_solvers_cpu` / `ed_solvers_gpu`).
- **Catch2 v3** unit tests gating every commit (37+ test cases).
- **CI** lanes for GCC Release, Clang Debug, clang-tidy, Python wheel +
  pytest, and CUDA build-only.
- **Python bindings** that expose the high-value entry points
  (`Operator`, `FixedSzOperator`, `full_diagonalization`, `lanczos`,
  `finite_temperature_lanczos`, …) under the new `quantum_ed` namespace.

See [`MODERNIZATION_AUDIT.md`](../MODERNIZATION_AUDIT.md) at the repo root
for the full rolling roadmap.

## Indices and tables

- {ref}`genindex`
- {ref}`search`
