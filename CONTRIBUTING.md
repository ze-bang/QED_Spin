# Contributing to QED

QED is a research-grade exact-diagonalization toolkit. Contributions
from collaborators are welcome. The goal is "research-grade with sane
engineering": fast iteration, scientifically correct output, easy to
debug, easy for new lab members to pick up.

## TL;DR

```bash
git clone https://github.com/ze-bang/QED.git
cd QED
pre-commit install                  # one-time, after `pip install pre-commit`
cmake --preset default              # Release + OpenBLAS, no CUDA/MPI
cmake --build --preset default -j   # build
ctest --preset default              # all tests must pass
```

If `ctest` is not 100 % green on `main`, that is a bug — file an issue.

## Build presets

See `CMakePresets.json` for the full list. The most common ones:

| Preset             | Use when                                                 |
|--------------------|----------------------------------------------------------|
| `default`          | First-time setup, no CUDA, no MPI                        |
| `debug`            | Stepping through with `gdb`                              |
| `debug-asan`       | Hunting memory bugs / undefined behavior (slow)          |
| `release-mpi`      | Distributed Lanczos / FTLM / TPQ runs                    |
| `release-cuda`     | GPU Lanczos / FTLM / TPQ                                 |
| `release-cuda-mpi` | Full HPC build (NCCL + CUDA + MPI)                       |
| `ci-linux`         | What CI uses; pinned to system `gcc`/`g++`               |

## Local developer overrides

The two paths the upstream `CMakeLists.txt` used to hardcode (LAPACKE
root, BLAS shim dir) are cache variables:

```cmake
-DED_LAPACKE_ROOT=/path/to/your/lapacke
-DED_BLAS_SHIM_DIR=/path/to/blas_shim
```

If you want them set automatically every time you configure, copy
`local.cmake.example` to `local.cmake` (gitignored) and pass it via
`cmake --preset default -C local.cmake`.

## Style

- C++17 (CUDA: C++17). No C++20 modules.
- `clang-format` enforces formatting. Run `clang-format -i
  path/to/file.cpp` before committing, or rely on the `pre-commit` hook.
- `clang-tidy` runs in CI as warnings-only.
- 4-space indent, 100-col, pointer/reference attached to type. See
  `.clang-format`.
- `#pragma once` for include guards (no `#ifndef X_H` boilerplate).
- Use `std::filesystem`, not `system("mkdir -p ...")`.
- Use `nlohmann::json`, not the bespoke parser in `construct_ham.h`.

## Tests

- All new C++ code should land with at least one Catch2 test under
  `tests/unit/`. Integration tests live under `tests/integration/`.
- For numerics-changing PRs, add a regression test that pins a
  known-good value (energy, spectral peak, etc.) on a small system
  you can compute analytically.
- Cross-checks: a CPU/GPU equivalence test on a 4–6 site lattice
  catches 90 % of GPU bugs and runs in seconds. Tag it
  `[gpu][cpu-equivalent]` so CI can opt in/out.
- New Python code should land with at least one `pytest` test under
  `python/tests/`.

## Commits and PRs

- One logical change per commit. Big refactors land as a series of
  small commits where every intermediate state still builds and
  `ctest` is green.
- Commit messages: imperative subject ("add", "fix", "refactor"),
  72-col first line, then a body that explains *why*.
- PRs: small, focused, one author. Self-review before requesting
  review.
- CI must be green before merge. No exceptions for `main`.

## Where things live

When extending the codebase, the relevant entry points are:

- **A new solver / kernel** — under `include/ed/krylov/` (Krylov
  family) or `include/ed/thermal/` (finite-T family), plus
  implementation under `src/solvers/`. Register the new
  `DiagonalizationMethod` enum in `include/ed/core/ed_types.h` and
  wire it into `src/orchestrator.cpp`.
- **A new symmetry axis** (spin-flip Z2, time reversal, SU(2)
  total-S, etc.) — see
  [`docs/architecture/SYMMETRY.md`](docs/architecture/SYMMETRY.md) §6
  for the design pattern: extend `ProjectorChain` with a new
  `Projector` (in `include/ed/symmetry/projector.h`) or add a new
  `Subspace` specialisation (in `include/ed/symmetry/subspace.h`).
  The operator hierarchy stays untouched.
- **A new basis policy** — see
  [`docs/architecture/ADD_NEW_BASIS_POLICY.md`](docs/architecture/ADD_NEW_BASIS_POLICY.md).
- **A new GPU lane** — see
  [`docs/architecture/ADD_NEW_GPU_CELL.md`](docs/architecture/ADD_NEW_GPU_CELL.md).
- **A new MPI lane** — see
  [`docs/architecture/ADD_NEW_MPI_CELL.md`](docs/architecture/ADD_NEW_MPI_CELL.md).
- **A new example** — the per-cell example tree was retired; the
  canonical usage documentation is now the tour
  (`examples/tour/0N_<topic>.py`, one verb per file, heavily
  commented, runs standalone in seconds). Extend the existing script
  for the matching verb rather than adding a new file; if a genuinely
  new verb/workflow needs its own script, add it to the tour, index it
  in `examples/README.md` and the top-level `README.md`, and make sure
  it passes in the `linux-tour` CI lane (which runs every tour script
  on each push). Exhaustive per-configuration coverage belongs in the
  test suites, not in examples.

## Reporting bugs

Open a GitHub issue with:

- The exact `cmake` command you used (or the preset name).
- The compiler version (`g++ --version`, `nvcc --version`).
- The full `ctest` output (`ctest --output-on-failure`).
- A minimal config that reproduces (`configs/<your_repro>.cfg`) or a
  short Python script.

## Architecture documents

- [`docs/architecture/ARCHITECTURE.md`](docs/architecture/ARCHITECTURE.md)
  — post-collapse architectural picture (read first).
- [`docs/architecture/SYMMETRY.md`](docs/architecture/SYMMETRY.md)
  — symmetry math + `Subspace × ProjectorChain` decomposition.
- [`docs/architecture/CODEMAP.md`](docs/architecture/CODEMAP.md)
  — directory-by-directory tour.
- [`docs/architecture/SCALING.md`](docs/architecture/SCALING.md)
  — memory + N envelope, env-var knobs.
- [`docs/history/`](docs/history/) — historical phase summaries
  (frozen time capsules).
- [`CHANGELOG.md`](CHANGELOG.md) — versioned release notes.
- [`README.md`](README.md) — user-facing introduction.
