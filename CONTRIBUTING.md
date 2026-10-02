# Contributing to QED_Spin

## Build and test

```bash
scripts/build.sh --variant cpu --tests          # -> build/cpu, with the C++ unit tests
ctest --test-dir build/cpu --output-on-failure
export PYTHONPATH=$PWD/python QED_CORE_DIR=$PWD/build/cpu/python/qed
python -m pytest python/tests                   # API tests; add `-m grid` for the coverage grid
```

`scripts/build.sh --variant cuda` builds the CUDA variant into `build/cuda`. On a cluster,
run builds and tests inside jobs; `scripts/gate/` submits the whole gate as SLURM arrays:
the build, the C++ unit tests, pytest, the grid, the golden suite and the examples. A
change is ready when every stage of the gate reports 0.

`CMakePresets.json` has one preset, `ci-linux`, which the GitHub CI uses. `-DBLAS_PROFILE=` picks
the BLAS / LAPACK provider (`AUTO`, `FLEXIBLAS`, `OPENBLAS`, `MKL`; `cmake/EDBlas.cmake`).

## Style

- C++17, CUDA C++17. `clang-format` enforces formatting (`.clang-format`; the pre-commit
  hook runs it). `#pragma once` in headers.
- Comments describe the code as it is: what it does and why. History goes in commit
  messages and `CHANGELOG.md`.
- Environment variables are read only through `ed::env` and must be rows of
  `include/ed/core/config.h` (`scripts/check_env_registry.sh` checks both ways).

## Tests

- C++ code lands with a Catch2 test under `tests/unit/`; Python code with a pytest test
  under `python/tests/`.
- A new task, symmetry or backend path gets grid cells (`python/tests/grid`) checked
  against the dense reference; sampled GPU paths must reproduce the CPU path at the same
  seeds.
- Changes that move numbers on purpose re-bless the golden suite
  (`tests/golden/golden.py bless`) with a reason.

## Where things live

- **A task or a symmetry**: the sector drivers in `src/engine/` (`eigs.cpp`, `thermal.cpp`, `dynamics.cpp`, `expect.cpp`),
  the star walk and block operators in `walk.h`, and the Python verbs in
  `python/qed/_verbs/`. See [`docs/architecture.md`](docs/architecture.md).
- **A kernel**: `include/ed/krylov/`, `include/ed/thermal/`, `include/ed/dynamics/`,
  written once against the backend interface (`include/ed/matvec/backend.h`).
- **A device kernel**: `include/ed/gpu/term_kernels.cuh` and
  `src/gpu/rep_matvec.cu`.
- **An example**: `examples/`, one script per family of verbs; the gate runs them all.

## Commits

One logical change per commit, imperative subject line, a body that says why. Every
commit on `main` passes the gate.

## Reporting bugs

Include the build command, compiler and CUDA versions, and a short Python script that
reproduces the problem.
