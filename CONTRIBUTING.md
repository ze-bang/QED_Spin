# Contributing to `exact_diagonalization_cpp`

This is a lab-internal exact-diagonalization toolkit. Contributions from collaborators are welcome. The goal is "research-grade with sane engineering": fast iteration, scientifically correct output, easy to debug, easy for new lab members to pick up.

## TL;DR

```bash
git clone <repo>
cd exact_diagonalization_cpp
pre-commit install                  # one-time, after `pip install pre-commit`
cmake --preset default              # Release + OpenBLAS, no CUDA/MPI
cmake --build --preset default -j   # build
ctest --preset default              # 146/146 must pass
```

If `ctest` is not 146/146 green on `main`, that is a bug. File an issue.

## Build presets

See `CMakePresets.json` for the full list. The most common ones:

| Preset             | Use when                                                 |
|--------------------|----------------------------------------------------------|
| `default`          | First-time setup, no CUDA, no MPI                        |
| `debug`            | Stepping through with `gdb`                              |
| `debug-asan`       | Hunting memory bugs / undefined behavior (slow!)         |
| `release-mpi`      | Distributed TPQ samples, ScaLAPACK runs                  |
| `release-cuda`     | GPU Lanczos / FTLM / TPQ                                 |
| `release-cuda-mpi` | Full HPC build                                           |
| `ci-linux`         | What CI uses; pinned to system `gcc`/`g++`               |

## Local developer overrides

The two paths the upstream `CMakeLists.txt` used to hardcode (LAPACKE root, BLAS shim dir) are now cache variables:

```cmake
-DED_LAPACKE_ROOT=/path/to/your/lapacke
-DED_BLAS_SHIM_DIR=/path/to/blas_shim
```

If you want them set automatically every time you configure, copy `local.cmake.example` to `local.cmake` (gitignored) and pass it via `cmake --preset default -C local.cmake`.

## Style

- C++17 (CUDA: C++17). No C++20 modules.
- `clang-format` enforces formatting. Run `clang-format -i path/to/file.cpp` before committing, or just rely on the `pre-commit` hook.
- `clang-tidy` runs in CI as warnings-only for now (will become an error gate in Phase 2).
- 4-space indent, 100-col, pointer/reference attached to type. See `.clang-format`.
- `#pragma once` for include guards (no `#ifndef X_H` boilerplate).
- Use `std::filesystem`, not `system("mkdir -p ...")`.
- Use `nlohmann::json`, not the bespoke parser in `construct_ham.h`.

## Tests

- All new C++ code should land with at least one Catch2 test under `tests/unit/`. Until the Catch2 migration completes (Phase 1), the existing bespoke test harness in `tests/common/test_harness.h` is acceptable.
- For numerics-changing PRs, add a regression test that pins a known-good value (energy, spectral peak, etc.) on a small system you can compute analytically.
- Cross-checks: a CPU/GPU equivalence test on a 4–6 site lattice catches 90% of GPU bugs and runs in seconds. Tag it `[gpu][cpu-equivalent]` so CI can opt in/out.

## Commits and PRs

- One logical change per commit. Big refactors land as a series of small commits where every intermediate state still builds and `ctest` is green.
- Commit messages: imperative subject ("add", "fix", "refactor"), 72-col first line, then a body that explains *why*.
- Reference the audit phase if applicable: `(P0.7)`, `(P1.3)`, `(P2.2)`, etc. See `docs/history/MODERNIZATION_AUDIT.md` for the legacy numbering scheme.
- PRs: small, focused, one author. Self-review before requesting review.
- CI must be green before merge. No exceptions for `main`.

## When adding a new method

1. Add the source under `src/solvers/cpu/` or `src/solvers/gpu/`. Header under `include/ed/solvers/`.
2. If it changes the `DiagonalizationMethod` enum, update `include/ed/core/ed_types.h` (single source of truth — there used to be a duplicate in `src/core/ed_config.cpp`; do not re-introduce it).
3. Wire the method into the dispatch in `src/core/ed_config.cpp`.
4. Add a unit test that compares it against a brute-force ground state on a small system.
5. Document the new method in `README.md` under the methods table.

## Reporting bugs

Open a GitHub issue with:

- The exact `cmake` command you used (or the preset name).
- The compiler version (`g++ --version`, `nvcc --version`).
- The full ctest output (`ctest --output-on-failure`).
- A minimal config that reproduces (`configs/<your_repro>.cfg`).

## Architecture documents

- `docs/architecture/IMPLEMENTATION_REPORT.md` — exhaustive subsystem reference. Read this before making structural changes.
- `docs/architecture/SCALING.md` — performance envelope and tunable knobs.
- `docs/architecture/IMPLEMENTATION_NOTES.md` — deferred / HPC-gated work; pick from here if you have cluster time.
- `docs/history/MODERNIZATION_AUDIT.md`, `docs/history/PHASE_3*_SUMMARY.md` — historical roadmap and phase summaries (frozen).
- `README.md` — user-facing introduction to the toolkit.
