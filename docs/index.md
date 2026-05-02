# exact_diagonalization

A modern C++17 / CUDA / Python toolkit for **exact diagonalization (ED)** of
spin-1/2 Hamiltonians, with first-class support for:

- Full diagonalization, Lanczos, block-Lanczos (CPU **and** GPU);
- Finite-temperature methods: **FTLM** and **LTLM**;
- **Dynamical / static structure factors (DSSF / SSSF)** with momentum and
  sublattice resolution;
- Symmetry sectors (translation, point-group, fixed-Sz);
- HDF5 I/O for eigenvectors, thermodynamic observables, and DSSF traces;
- A first-class **Python interface** (`qed`) built via `pybind11` +
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
guides/workflow
guides/usage
guides/python_api_coverage
```

```{toctree}
:maxdepth: 2
:caption: Reference

api/cpp
api/python
```

```{toctree}
:maxdepth: 2
:caption: Architecture

architecture/CODEMAP
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
  `finite_temperature_lanczos`, …) under the new `qed` namespace.

## Project documents

The following live at the repository root and are rendered on GitHub:

- [`CHANGELOG.md`](https://github.com/ze-bang/exact_diagonalization_cpp/blob/main/CHANGELOG.md)
  — versioned release notes.
- [`CONTRIBUTING.md`](https://github.com/ze-bang/exact_diagonalization_cpp/blob/main/CONTRIBUTING.md)
  — how to set up a dev environment and submit changes.
- [`docs/architecture/CODEMAP.md`](https://github.com/ze-bang/exact_diagonalization_cpp/blob/main/docs/architecture/CODEMAP.md)
  — static libraries, file-level tree, `ED` flowcharts, MPI vs GPU layers, redundancies.
- [`docs/architecture/IMPLEMENTATION_REPORT.md`](https://github.com/ze-bang/exact_diagonalization_cpp/blob/main/docs/architecture/IMPLEMENTATION_REPORT.md)
  — exhaustive subsystem reference.
- [`docs/architecture/SCALING.md`](https://github.com/ze-bang/exact_diagonalization_cpp/blob/main/docs/architecture/SCALING.md)
  — scaling envelope, memory tables, environment-variable controls.
- [`docs/architecture/IMPLEMENTATION_NOTES.md`](https://github.com/ze-bang/exact_diagonalization_cpp/blob/main/docs/architecture/IMPLEMENTATION_NOTES.md)
  — deferred work and HPC-gated milestones.
- [`docs/benchmarks/BENCHMARKS.md`](https://github.com/ze-bang/exact_diagonalization_cpp/blob/main/docs/benchmarks/BENCHMARKS.md)
  — head-to-head benchmark write-up vs QuSpin / SciPy.
- [`docs/guides/python_api_coverage.md`](https://github.com/ze-bang/exact_diagonalization_cpp/blob/main/docs/guides/python_api_coverage.md)
  — what `import qed` covers vs the full `ED` binary.
- [`examples/`](https://github.com/ze-bang/exact_diagonalization_cpp/tree/main/examples)
  — runnable C++ / Python / CLI examples (one per use case).
- [`docs/history/`](https://github.com/ze-bang/exact_diagonalization_cpp/tree/main/docs/history)
  — historical phase summaries (`MODERNIZATION_AUDIT`, `PHASE_3A`, `PHASE_3`,
  [`PHASE_7_SOLVER_AXES`](https://github.com/ze-bang/exact_diagonalization_cpp/blob/main/docs/history/PHASE_7_SOLVER_AXES.md),
  [`PHASE_7_1_SYMMETRY_AXIS`](https://github.com/ze-bang/exact_diagonalization_cpp/blob/main/docs/history/PHASE_7_1_SYMMETRY_AXIS.md),
  [`PHASE_8_GPU_MPI_OPT`](https://github.com/ze-bang/exact_diagonalization_cpp/blob/main/docs/history/PHASE_8_GPU_MPI_OPT.md)).

## Indices and tables

- {ref}`genindex`
- {ref}`search`
