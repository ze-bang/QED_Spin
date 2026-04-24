# Modernization Audit — `exact_diagonalization_cpp`

Status: read-only audit; no code has been changed.
Audience target (per user): "lab-grade" — installable, documented, testable, with CI; modern CMake, clang-format/tidy, Doxygen. Not aiming for pip/conda packaging today.
Peer benchmarks (per user): QuSpin, EDLib, Pomerol, NetKet.

This document inventories what exists, lists concrete deficiencies, and proposes a prioritized, sequenced modernization plan with example diffs ready to be applied. Each item includes effort and risk estimates.

---

## 0. TL;DR

The codebase is **scientifically rich but structurally dated**. There is real, substantial value in here (full ED + Lanczos family + ARPACK + ScaLAPACK + FTLM/LTLM/HYBRID + mTPQ/cTPQ + GPU variants of Lanczos/FTLM/TPQ + Krylov-Schur + DMRG scaffold + NLCE pipeline + thoughtful BLAS-vendor handling), and the test suite that exists actually passes. But:

- `include/` carries **~22 k lines of header-only logic** with non-inline free function definitions (ODR-bomb territory).
- `src/apps/*.cpp` are **mega-files** (3–5 kLOC each).
- `CMakeLists.txt` is a single **1500-line monolith** with hardcoded user-specific paths, top-level `include_directories`/`add_definitions`, no installable package config, and one duplicated `enum class DiagonalizationMethod` declaration the codebase has to keep "in sync by hand."
- **Zero CI**, no `.clang-format`/`.clang-tidy`/`.editorconfig`/`.gitignore`/`.pre-commit-config.yaml`/`CONTRIBUTING.md`/`CHANGELOG.md`.
- The "Python package" `edlib` ships only helper scripts, has no real bindings, has no tests, and uses a name (`edlib`) **already taken on PyPI by Pomerol's edlib**.
- No Doxygen/Sphinx, no examples gallery, no API reference.
- Several easy correctness/portability landmines (e.g., `system("mkdir -p ...")` calls, hand-rolled JSON parser inside a 4348-line header, sprinkled `__builtin_popcountll` / `unsigned long long` assumptions, `ScaLAPACK_COMPATIBLE` heuristic in CMake).

The good news: the modernization can be done **without rewriting the science**. It's mostly mechanical re-organization, a real CMake split, a CI workflow, a real test framework, and a thin pybind11 binding — done in stages.

Recommended path: **Phase 0 → Phase 1 → Phase 2** below, totaling ~4–6 focused weeks of work. Phase 0 alone (1–2 days) closes the worst hygiene gaps and unblocks every later phase.

---

## 1. Inventory snapshot

### Codebase size

| Area | Files | LOC |
|---|---|---|
| `include/ed/core/` (15 headers) | 15 | **19 741** |
| `include/ed/solvers/`            | 10 |  2 723 |
| `include/ed/io/`                 |  2 |    158 |
| `include/ed/gpu/`                | 11 |  3 387 |
| `include/ed/dmrg/` (+ design docs) |  6 |  3 863 |
| `src/apps/`                      |  4 | **13 419** |
| `src/core/`                      |  1 |  1 610 |
| `src/io/`                        |  2 |    399 |
| `src/solvers/cpu/`               | 10 | **15 124** |
| `src/solvers/gpu/`               | 16 | **14 921** |
| Total tracked C++/CUDA           | ~77 | **~75 600** |
| `python/edlib/` (helpers)        | 14 |  9 417 |
| `workflows/nlce/run/`            |  7 |  8 617 |
| `tests/unit/`                    | 10 | (small, all pass) |
| `CMakeLists.txt`                 |  1 | **1 501** |

### Build/test status (as of last build)

- `cmake --build` succeeds (artifacts present in `build/`: `ED`, `TPQ_DSSF`, `compute_bfg_order_parameters`, `test_dmrg_vs_ed`, plus 12 unit test binaries).
- `ctest` last run: **12/12 PASS**.
- 2 deprecation/wat smells in build artifacts: `bench_bl_*`, `bench_full*` directories suggest old benchmark output checked in alongside the build.

### What I read in detail

`README.md`, `CMakeLists.txt` (full), `src/apps/ed_main.cpp` (sampled top, all function signatures, main), `src/apps/compute_bfg_order_parameters.cpp` (header), `include/ed/core/ed_wrapper.h` (signatures + header definition at line 895), `include/ed/core/construct_ham.h` (utility region + JSON-parsing region), `include/ed/core/streaming_symmetry.h` (top), `include/ed/core/ed_config.h` + `src/core/ed_config.cpp` (duplicate `enum class DiagonalizationMethod` at lines 14–60), `include/ed/core/blas_lapack_wrapper.h`, `include/ed/core/ed_logging.h`, `include/ed/core/system_utils.h`, `include/ed/solvers/lanczos.h`, `src/solvers/cpu/lanczos.cpp` (signatures), `src/solvers/cpu/ftlm.cpp` (signatures), `src/solvers/gpu/gpu_lanczos.cu` (top), `tests/common/test_harness.h`, `tests/unit/test_lanczos_variants.cpp`, `python/edlib/__init__.py`, `python/edlib/helper_pyrochlore.py`, `python/pyproject.toml`, `include/ed/dmrg/idmrg.h` + `DESIGN.md`, `configs/01_diagonalization_lanczos.cfg`.

---

## 2. Headline concerns (top 5, in priority order)

### 2.1 Header-only logic at scale — ODR + compile-time disaster
`include/ed/core/ed_wrapper.h` (4507 LOC) and `include/ed/core/construct_ham.h` (4348 LOC) define **non-inline, non-template free functions and struct member methods** in headers, then `#include` those headers from multiple `.cpp` files (`ed_main.cpp`, all CPU solvers, `TPQ_DSSF.cpp`, every test). Examples:

```519:524:include/ed/core/ed_wrapper.h
EDResults exact_diagonalization_core(
    std::function<void(const Complex*, Complex*, int)> H, 
    uint64_t hilbert_space_dim,
    DiagonalizationMethod method,
    const EDParameters& params
);
```
(forward decl), and at line 895:
```895:907:include/ed/core/ed_wrapper.h
EDResults exact_diagonalization_core(
    std::function<void(const Complex*, Complex*, int)> H, 
    uint64_t hilbert_space_dim,
    DiagonalizationMethod method = DiagonalizationMethod::LANCZOS,
    const EDParameters& params = EDParameters()
) {
    EDResults results;
    
    // Initialize output directory if needed
    if (!params.output_dir.empty()) {
        std::string cmd = "mkdir -p " + params.output_dir;
        safe_system_call(cmd);
```

This works only because exactly one TU includes the definition (or because the compiler folds duplicates), and it makes any shared-library / Python-binding plan fragile. Compile times suffer: every TU that touches ED parses ~22 k lines of `<Eigen/...>` + `<omp.h>` + STL containers + JSON code + ScaLAPACK glue + HDF5 + ARPACK glue.

Same pattern in `streaming_symmetry.h` (2491 LOC), `hdf5_io.h` (3335 LOC), `ed_wrapper_streaming.h` (874), `ed_wrapper_chunked.h` (621), `chunked_symmetry_builder.h` (970), `disk_streaming_symmetry.h` (705).

**Fix**: split each header into `include/ed/.../*.h` (declarations + inline/templates) and `src/.../*.cpp` (definitions). Compile each into a static or shared library target.

### 2.2 Duplicated `enum class DiagonalizationMethod`
`include/ed/core/ed_wrapper.h` and `src/core/ed_config.cpp` both declare `enum class DiagonalizationMethod` and the file explicitly says:
```12:14:src/core/ed_config.cpp
// NOTE: This enum MUST stay in sync with the one in ed_wrapper.h!
// The order of values is critical for proper method dispatch.
enum class DiagonalizationMethod {
```
This is a maintenance bomb. Fix by extracting it (and `EDParameters`, `EDResults`) into a tiny `include/ed/core/ed_types.h` and including it from both places.

### 2.3 `CMakeLists.txt` is a 1500-line monolith with hardcoded paths
- Lines 512–513 / 531: `set(LAPACKE_ROOT "/home/pc_linux/exact_diagonalization_clean/lapacke_ref/install")` and `set(BLAS_SHIM_DIR "/home/pc_linux/exact_diagonalization_clean/blas_shim")` hardcoded user-specific absolute paths. Will break for any other developer or CI runner.
- Top-level `include_directories(...)` and `add_definitions(-D...)` (lines 172–182, 278, 337, 429, 463, 582, 710, 726–728, 743, 858) — global state instead of `target_include_directories` / `target_compile_definitions`. Standard CMake "Modern CMake" anti-patterns (Daniel Pfeifer's "Effective CMake" rules).
- No `cmake/` modules, no `Find*.cmake`, no `cmake/EDOptions.cmake`, no presets file (`CMakePresets.json`).
- `install(TARGETS ED DESTINATION bin)` is the only install rule — no installable library, no exported targets, no package config (`EDConfig.cmake`).
- `find_package(Eigen3 REQUIRED)` without specifying a minimum version.
- ScaLAPACK detection is heuristic-driven and emits warnings about ABI mismatch — best effort, but should be tested by CI to confirm guidance is current.

### 2.4 Zero CI, zero formatter, zero `.gitignore`
- No `.github/workflows/`, no GitLab CI, no Jenkinsfile.
- No `.gitignore` at the repo root → `build/`, `__pycache__/`, `.venv/`, generated HDF5 results are in danger of being committed.
- No `.clang-format`, `.clang-tidy`, `.editorconfig`, `.pre-commit-config.yaml`, `CONTRIBUTING.md`, `CHANGELOG.md`.
- VSCode-specific `settings.json` is checked in (which is fine), but no `compile_commands.json` symlink or guidance.

### 2.5 Apps that should be libraries
`src/apps/ed_main.cpp` (3092 LOC), `src/apps/TPQ_DSSF.cpp` (4404 LOC), `src/apps/compute_bfg_order_parameters.cpp` (4691 LOC) bury reusable physics/IO inside an `int main`. None of this is callable from another tool, from a test, or from Python. The `ed_main.cpp` workflow functions (`run_standard_workflow`, `compute_thermodynamics`, `compute_dynamical_response_workflow`, `compute_static_response_workflow`, `compute_ground_state_dssf_workflow`) are reachable only by command-line dispatch.

---

## 3. Per-area assessment and proposed concrete diffs

### 3.1 Build system (CMake)

#### Current state
1500 lines, monolithic, top-level globals, hardcoded paths, no presets, no package config, no install for headers, no namespace-prefixed targets. `BLAS_PROFILE` is a thoughtful innovation but would be cleaner as a `cmake/EDBlasBackend.cmake` module.

#### Proposed file layout

```text
exact_diagonalization_cpp/
├── CMakeLists.txt                  # ≤ 100 lines, top-level project + add_subdirectory()
├── CMakePresets.json               # Named build configurations
├── cmake/
│   ├── EDOptions.cmake             # Build options (WITH_CUDA, WITH_MPI, ...)
│   ├── EDCompilerFlags.cmake       # Warnings, sanitizers, optimization levels
│   ├── EDBlasBackend.cmake         # BLAS_PROFILE machinery (currently 600 lines)
│   ├── EDScaLAPACK.cmake           # ScaLAPACK detection
│   ├── EDInstall.cmake             # install() rules + EDConfig.cmake.in generation
│   ├── EDConfig.cmake.in           # Imported targets for downstream users
│   └── modules/
│       ├── FindARPACK.cmake        # Real find module (currently `find_library`)
│       └── FindScaLAPACK.cmake     # Replace heuristic detection
├── src/
│   ├── ed/
│   │   ├── core/CMakeLists.txt     # add_library(ed_core ...)
│   │   ├── solvers/cpu/CMakeLists.txt   # add_library(ed_solvers_cpu ...)
│   │   ├── solvers/gpu/CMakeLists.txt   # add_library(ed_solvers_gpu ...)
│   │   ├── io/CMakeLists.txt       # add_library(ed_io ...)
│   │   └── dmrg/CMakeLists.txt     # add_library(ed_dmrg ...)
│   └── apps/CMakeLists.txt         # add_executable(ed::ED ...) ...
├── tests/CMakeLists.txt            # GoogleTest or Catch2 + add_test
├── benchmarks/CMakeLists.txt       # google-benchmark + add_test
└── docs/CMakeLists.txt             # Doxygen + Sphinx/Breathe
```

#### Modular library targets (proposed)
```text
ed::core            → core types, IO helpers, logging, BLAS wrapper
ed::solvers_cpu     → all CPU solvers (Lanczos family, ARPACK, FTLM, ...)
ed::solvers_gpu     → CUDA Lanczos/FTLM/TPQ/Krylov-Schur (built only if WITH_CUDA)
ed::scalapack       → ScaLAPACK distributed diag (only if MPI + ScaLAPACK)
ed::dmrg            → MPS/MPO/iDMRG (currently scaffolded)
ed::ed              → meta target linking the above
```

Apps `ED`, `TPQ_DSSF`, `compute_bfg_order_parameters` link `ed::ed`. Tests link the same. A future `pyed` pybind module also links `ed::ed`. Downstream projects do:
```cmake
find_package(ed CONFIG REQUIRED)
target_link_libraries(my_thing PRIVATE ed::ed)
```

#### Concrete proposed diffs (excerpts)

**`cmake/EDOptions.cmake`** (new):
```cmake
include_guard(GLOBAL)

option(ED_BUILD_TESTS       "Build the ED regression test suite"    ON)
option(ED_BUILD_BENCHMARKS  "Build micro-benchmarks (Google Bench)" OFF)
option(ED_BUILD_DOCS        "Build Doxygen + Sphinx documentation"  OFF)
option(ED_BUILD_PYTHON      "Build Python bindings (pybind11)"      OFF)
option(ED_WITH_CUDA         "Enable CUDA GPU solvers"               OFF)
option(ED_WITH_MPI          "Enable MPI (sample-parallel TPQ, ScaLAPACK)" OFF)
option(ED_WITH_SCALAPACK    "Enable distributed full diag via ScaLAPACK"  ${ED_WITH_MPI})
option(ED_ENABLE_ASAN       "Enable AddressSanitizer in Debug builds"     OFF)
option(ED_ENABLE_UBSAN      "Enable UndefinedBehaviorSanitizer in Debug"  OFF)
option(ED_ENABLE_TSAN       "Enable ThreadSanitizer (mutually exclusive)" OFF)
option(ED_ENABLE_LTO        "Enable interprocedural / LTO optimization"   OFF)
option(ED_ENABLE_NATIVE     "Use -march=native (off in CI / portable builds)" OFF)
option(ED_WARNINGS_AS_ERRORS "Treat C++ warnings as errors"               ON)

set(ED_BLAS_PROFILE "AUTO" CACHE STRING
    "BLAS/LAPACK profile (AUTO|FLEXIBLAS|MKL|AOCL|OPENBLAS|GENERIC)")
set_property(CACHE ED_BLAS_PROFILE PROPERTY STRINGS
    AUTO FLEXIBLAS MKL AOCL OPENBLAS GENERIC)
```

**`CMakePresets.json`** (new, ~80 lines, sample):
```json
{
  "version": 4,
  "cmakeMinimumRequired": { "major": 3, "minor": 23 },
  "configurePresets": [
    {
      "name": "default",
      "binaryDir": "${sourceDir}/build/${presetName}",
      "cacheVariables": {
        "CMAKE_BUILD_TYPE": "Release",
        "CMAKE_EXPORT_COMPILE_COMMANDS": "ON"
      }
    },
    { "name": "debug-asan", "inherits": "default",
      "cacheVariables": { "CMAKE_BUILD_TYPE": "Debug",
                          "ED_ENABLE_ASAN": "ON",
                          "ED_ENABLE_UBSAN": "ON" } },
    { "name": "release-mkl", "inherits": "default",
      "cacheVariables": { "ED_BLAS_PROFILE": "MKL",
                          "ED_WITH_MPI": "ON",
                          "ED_WITH_SCALAPACK": "ON" } },
    { "name": "release-cuda", "inherits": "default",
      "cacheVariables": { "ED_WITH_CUDA": "ON",
                          "CMAKE_CUDA_ARCHITECTURES": "80;90" } },
    { "name": "ci-portable", "inherits": "default",
      "cacheVariables": { "ED_BLAS_PROFILE": "OPENBLAS",
                          "ED_ENABLE_NATIVE": "OFF",
                          "ED_WARNINGS_AS_ERRORS": "ON" } }
  ],
  "buildPresets": [
    { "name": "default", "configurePreset": "default" },
    { "name": "ci-portable", "configurePreset": "ci-portable" }
  ],
  "testPresets": [
    { "name": "default", "configurePreset": "default",
      "output": { "outputOnFailure": true } }
  ]
}
```

**`cmake/EDCompilerFlags.cmake`** (new): centralizes warning flags, sanitizers, LTO, `-march=native` opt-in. Removes the 80-line `target_compile_options(ED PRIVATE ...)` block currently at the bottom of `CMakeLists.txt`.

**Hardcoded paths to remove or guard** (lines in current `CMakeLists.txt`):
- 512–513: `LAPACKE_ROOT = /home/pc_linux/...` → require user to pass via `-DLAPACKE_ROOT=...`.
- 531: `BLAS_SHIM_DIR = /home/pc_linux/...` → ditto, behind a `ED_BLAS_SHIM_DIR` option.
- 422, 457, 539: hardcoded `/usr/lib/llvm-18/lib` for libomp. Should `find_package(OpenMP)` and trust its target.
- 297–299, 362–365, 487–490, 506–507, 613–616: `/opt/intel/...`, `/opt/AMD/...`, `/usr/lib/x86_64-linux-gnu/...` are HINTS, which is OK, but the file should not assume Debian path layout.

**Effort**: ~3 days (configure, port flags, port BLAS profile, wire targets, write CMakePresets.json, update README install instructions).
**Risk**: Medium — must preserve all current build paths; mitigate by keeping `BLAS_PROFILE` machinery byte-identical, just relocated.

---

### 3.2 Source structure / header bloat

#### Concrete header → source split plan

| Current header | Keep in header (`*.h`) | Move to (`*.cpp`) |
|---|---|---|
| `ed_wrapper.h` (4507) | enums, typedefs, `EDResults`/`EDParameters` POD, fwd decls, helpful inline `is_*_method()` checks | all `EDResults exact_diagonalization_core(...)`, `setup_symmetry_basis`, `transform_and_save_*`, `process_thermal_correlations`, `diagonalize_symmetry_block`, `diagonalize_matrix_free`, `exact_diagonalization_from_files/_directory`, `exact_diagonalization_all_sz_sectors[_gpu]`, `exact_diagonalization_fixed_sz[_symmetrized]` |
| `construct_ham.h` (4348) | `Operator`/`FixedSzOperator` class declarations, `inline uint64_t popcount(...)`, `generateFixedSzBasis` etc. (small/inline helpers OK) | `Operator::apply`, `Operator::loadFromFile`, JSON parsing of `automorphism_results` (use `nlohmann/json` instead of hand-rolled parser), all `*Reduction*`, all symmetry-building |
| `streaming_symmetry.h` (2491) | struct decls, class decl | every method body |
| `hdf5_io.h` (3335) | namespace + class decl | every method body |
| `chunked_symmetry_builder.h` (970), `disk_streaming_symmetry.h` (705), `ed_wrapper_streaming.h` (874), `ed_wrapper_chunked.h` (621) | decl + small inlines | bodies |

After this split, **`include/ed/`** drops from ~22 k LOC to an estimated ~5 k LOC. **Cold build of an empty TU including `<ed/ed.h>` should drop from minutes to seconds.**

#### Specific quick wins inside the split
- `#ifndef`/`#define` guards → `#pragma once` (project-wide).
- Replace `system("mkdir -p X")` (used in `ed_wrapper.h:905`, `streaming_symmetry.h`, `system_utils.h`, `tests/common/test_harness.h`, `ftlm.cpp`, `gpu_ftlm.cu`, `gpu_tpq.cu`, `TPQ.cpp`, `ltlm.cpp`, `CG.cpp`, `construct_ham.h`) with `std::filesystem::create_directories`. (Lanczos has already been migrated to `std::filesystem` in `lanczos.cpp:25-26`, so the pattern is established.)
- Hand-rolled JSON parser in `construct_ham.h:200-310` → drop in [`nlohmann/json`](https://github.com/nlohmann/json) as a `FetchContent` (or system dep). Cuts ~150 LOC and removes a real bug surface.

#### Apps as libraries
Refactor `src/apps/{ed_main,TPQ_DSSF,compute_bfg_order_parameters}.cpp` so each `int main` is ≤50 lines and just dispatches to a function from a real library:

```text
src/ed/apps/
  ed_app_main.cpp                # int main → ed::cli::run_ed_main(argc, argv)
  TPQ_DSSF_main.cpp              # int main → ed::cli::run_tpq_dssf(argc, argv)
  compute_bfg_main.cpp           # int main → ed::bfg::compute_order_parameters(...)

src/ed/cli/
  ed_cli.{h,cpp}                 # parsing + workflow dispatch
  workflows.{h,cpp}              # run_standard_workflow, compute_thermodynamics, ...

src/ed/bfg/
  order_parameters.{h,cpp}       # the BFG physics, callable from tests + Python
```

This **enables Python bindings**, makes each workflow individually testable, and shrinks rebuild scope dramatically.

**Effort**: ~10 days for the ed_wrapper / construct_ham / hdf5_io split + apps factoring. Can proceed in parallel with other tracks.
**Risk**: Medium-High — large mechanical refactor across many files. Mitigation: do one header at a time, keep `ctest` green between every commit, retain the existing `ED` binary's CLI surface unchanged.

---

### 3.3 CI / quality gates / dev hygiene

#### Files to add (Phase 0)

**`.gitignore`** (root, new):
```text
# Build
build/
build-*/
cmake-build-*/
*.o
*.obj
*.a
*.so
*.dylib

# CMake
CMakeCache.txt
CMakeFiles/
cmake_install.cmake
CTestTestfile.cmake
compile_commands.json

# Python
__pycache__/
*.py[cod]
*.egg-info/
.venv/
.pytest_cache/
.mypy_cache/
.ruff_cache/

# Editor
.vscode/*
!.vscode/settings.json
!.vscode/launch.json
!.vscode/tasks.json
.idea/

# Generated outputs
*.h5
*.hdf5
results/
output/
nlce_*_results/

# OS
.DS_Store
Thumbs.db
```

**`.editorconfig`** (root, new):
```ini
root = true

[*]
end_of_line = lf
insert_final_newline = true
charset = utf-8
trim_trailing_whitespace = true
indent_style = space
indent_size = 4

[*.{md,yml,yaml,json}]
indent_size = 2

[Makefile]
indent_style = tab
```

**`.clang-format`** (root, new) — LLVM-style (close to project's existing style) with project tweaks:
```yaml
BasedOnStyle: LLVM
IndentWidth: 4
ColumnLimit: 100
UseTab: Never
PointerAlignment: Left
SpacesBeforeTrailingComments: 2
AlignConsecutiveAssignments: false
AllowShortFunctionsOnASingleLine: InlineOnly
BinPackArguments: false
BinPackParameters: false
NamespaceIndentation: None
IncludeBlocks: Regroup
IncludeCategories:
  - Regex: '^<ed/'
    Priority: 4
  - Regex: '^<(Eigen|cuda|cublas|cusparse|cusolver|H5|hdf5|mpi|mkl|cblas|lapacke|arpack)'
    Priority: 3
  - Regex: '^<.*\.h>$'
    Priority: 2
  - Regex: '^<.*>$'
    Priority: 1
SortIncludes: CaseSensitive
```

**`.clang-tidy`** (root, new) — start permissive:
```yaml
Checks: >
  bugprone-*,
  -bugprone-easily-swappable-parameters,
  -bugprone-narrowing-conversions,
  performance-*,
  modernize-use-nullptr,
  modernize-use-override,
  modernize-use-default-member-init,
  modernize-pass-by-value,
  modernize-loop-convert,
  modernize-use-emplace,
  readability-braces-around-statements,
  readability-redundant-*,
  cppcoreguidelines-pro-type-cstyle-cast,
  cppcoreguidelines-init-variables,
  misc-*,
  -misc-non-private-member-variables-in-classes,
  -misc-include-cleaner

WarningsAsErrors: ''
HeaderFilterRegex: 'include/ed/.*\.h$'
FormatStyle: file
```

(Pass `-DCMAKE_CXX_CLANG_TIDY=clang-tidy;-warnings-as-errors=*` in CI, locally just lint.)

**`.pre-commit-config.yaml`** (root, new):
```yaml
repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.6.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-toml
      - id: check-merge-conflict
      - id: check-added-large-files
        args: ['--maxkb=500']
  - repo: https://github.com/pre-commit/mirrors-clang-format
    rev: v18.1.8
    hooks:
      - id: clang-format
        types_or: [c++, c, cuda]
  - repo: https://github.com/cheshirekow/cmake-format-precommit
    rev: v0.6.13
    hooks:
      - id: cmake-format
      - id: cmake-lint
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.5.0
    hooks:
      - id: ruff
      - id: ruff-format
```

**`CONTRIBUTING.md`** (new): style, branch policy, how to run `ctest`, CI matrix, "I'm adding a new solver" walkthrough.
**`CHANGELOG.md`** (new): Keep-A-Changelog format, start at `0.1.0` (current `__version__` in `python/edlib/__init__.py`).
**`CITATION.cff`** (new) — important for academic users; generate DOI later via Zenodo.

#### `.github/workflows/ci.yml` (new, sketch)

```yaml
name: CI
on: [push, pull_request]

jobs:
  lint:
    runs-on: ubuntu-24.04
    steps:
      - uses: actions/checkout@v4
      - uses: pre-commit/action@v3.0.1

  build-test:
    runs-on: ubuntu-24.04
    strategy:
      fail-fast: false
      matrix:
        compiler: [gcc-13, clang-18]
        blas:     [openblas, generic]
        build:    [Release, Debug]
        mpi:      [OFF, ON]
    steps:
      - uses: actions/checkout@v4
      - name: Install dependencies
        run: |
          sudo apt-get update
          sudo apt-get install -y cmake ninja-build \
            libeigen3-dev libhdf5-dev libarpack2-dev \
            ${{ matrix.mpi == 'ON' && 'libopenmpi-dev libscalapack-openmpi-dev' || '' }} \
            ${{ matrix.blas == 'openblas' && 'libopenblas-dev' || 'libblas-dev liblapack-dev liblapacke-dev' }}
      - name: Configure
        run: cmake -S . -B build --preset ci-portable \
                   -DCMAKE_BUILD_TYPE=${{ matrix.build }} \
                   -DCMAKE_CXX_COMPILER=${{ matrix.compiler }} \
                   -DED_BLAS_PROFILE=${{ matrix.blas == 'openblas' && 'OPENBLAS' || 'GENERIC' }} \
                   -DED_WITH_MPI=${{ matrix.mpi }}
      - name: Build
        run: cmake --build build -j$(nproc)
      - name: Test
        run: ctest --test-dir build --output-on-failure -j$(nproc)

  sanitizers:
    runs-on: ubuntu-24.04
    steps:
      - uses: actions/checkout@v4
      - run: sudo apt-get update && sudo apt-get install -y libeigen3-dev libhdf5-dev libarpack2-dev libopenblas-dev
      - run: cmake -S . -B build --preset debug-asan
      - run: cmake --build build -j$(nproc)
      - run: ctest --test-dir build --output-on-failure -j$(nproc)

  cuda-build:
    runs-on: ubuntu-24.04
    container: nvidia/cuda:12.5.0-devel-ubuntu24.04
    steps:
      - uses: actions/checkout@v4
      - run: apt-get update && apt-get install -y cmake ninja-build libeigen3-dev libhdf5-dev libarpack2-dev libopenblas-dev
      - run: cmake -S . -B build --preset release-cuda -DCMAKE_CUDA_ARCHITECTURES=80
      - run: cmake --build build -j$(nproc) --target ED  # GPU tests skipped (no GPU on runner)

  docs:
    runs-on: ubuntu-24.04
    steps:
      - uses: actions/checkout@v4
      - run: sudo apt-get update && sudo apt-get install -y doxygen graphviz python3-sphinx python3-breathe
      - run: cmake -S . -B build -DED_BUILD_DOCS=ON
      - run: cmake --build build --target docs
      - uses: actions/upload-artifact@v4
        with: { name: docs, path: build/docs/html }
```

**Effort**: ~1.5 days (CI matrix tuning takes a couple of iterations).
**Risk**: Low.

---

### 3.4 Testing

#### Current state
Bespoke `tests/common/test_harness.h` (~325 lines) — perfectly serviceable, 12/12 tests pass. But it lacks:
- Test discovery (each test is its own `int main`).
- Parametrized tests, fixtures, scoped setup/teardown, structured failure messages.
- Death tests, exception messages, floating point matchers (must roll our own `near_eq`).
- Integration with editors / IDE test runners.
- No CPU/GPU equivalence tests, no MPI tests, no benchmarks.

#### Proposal

**Adopt [Catch2 v3](https://github.com/catchorg/Catch2) via `FetchContent`** (single dep, header-only at use-site, BSL-1.0 license). Migration is mostly mechanical — `tests/common/test_harness.h` becomes `tests/common/fixtures.h` (build_heisenberg_chain, etc.), and existing `int main` files become `TEST_CASE("…", "[lanczos]")`.

Why Catch2 over GoogleTest:
- No build of a separate library (saves CI time).
- More expressive `REQUIRE_THAT` / matchers for floating point.
- Easier `SECTION`s for parametric variants.
- Already MIT/BSL — no licensing friction with current MIT.

#### New tests to add

| Area | Currently | Proposed |
|---|---|---|
| Fixed-Sz CPU/GPU equivalence | none | `test_fixed_sz_cpu_vs_gpu.cpp` |
| Symmetrized CPU/GPU equivalence | none | `test_symmetry_cpu_vs_gpu.cpp` |
| TPQ ↔ FTLM agreement at high T | none | `test_tpq_vs_ftlm_high_T.cpp` |
| MPI sample-parallel TPQ | none | `test_mpi_tpq_sample_split.cpp` (1, 2, 4 ranks) |
| ScaLAPACK distributed full diag | none | `test_scalapack_2x2_grid.cpp` |
| HDF5 round-trip for thermal data | partial | `test_hdf5_thermal_roundtrip.cpp` |
| NLCE end-to-end (small lattice) | none | `test_nlce_pyrochlore_order2.cpp` |
| Continued-fraction DSSF vs Lanczos pole | none | `test_dssf_continued_fraction.cpp` |
| Loader robustness (malformed input) | partial | `test_input_loader_errors.cpp` |

#### Benchmarks

**`benchmarks/`** (new) using [Google Benchmark](https://github.com/google/benchmark) via `FetchContent`. Track:
- `BM_LanczosGroundState_Heisenberg<N=8..18>`
- `BM_FTLM_PerSampleTime`
- `BM_OperatorApply` (matrix-free SpMV)
- `BM_HDF5RoundTrip`

Wire benchmarks behind `ED_BUILD_BENCHMARKS=ON`, run them in a `nightly` CI job that uploads results to a `gh-pages` site (e.g., [continuous-benchmark](https://github.com/benchmark-action/github-action-benchmark)) for regression detection.

**Effort**: ~5 days (Catch2 migration: 2; new tests: 2; benchmarks: 1).
**Risk**: Low.

---

### 3.5 Documentation

#### Current state
- `README.md` is **791 lines, very thorough on user-facing CLI**. Good.
- `include/ed/dmrg/DESIGN.md`, `include/ed/dmrg/QUICKREF.md`, `scripts/README.md` exist for niche areas.
- **No `docs/` directory**, no Doxygen, no API reference, no architecture guide, no theory background, no examples gallery.
- Inline Doxygen-ish comments (`@brief`, `@param`) sprinkled on many functions, but no Doxyfile, so they're never rendered.

#### Proposal

**`docs/` (new)** with a Sphinx + Breathe + Doxygen pipeline:
```text
docs/
├── Doxyfile.in                # Configured by CMake, INPUT = include/ed src/
├── conf.py                    # Sphinx configuration with sphinx_rtd_theme + breathe
├── index.rst                  # Landing page
├── getting_started.rst        # Build, first ED run, first NLCE run
├── theory/
│   ├── lanczos.rst
│   ├── ftlm_ltlm.rst
│   ├── tpq.rst
│   ├── nlce.rst
│   └── symmetries.rst
├── user_guide/
│   ├── input_format.rst       # InterAll.dat, Trans.dat, ThreeBodyG.dat
│   ├── configuration_files.rst
│   ├── solvers.rst
│   ├── workflows.rst
│   └── nlce_workflow.rst
├── api/
│   ├── core.rst               # `.. doxygenclass:: Operator`
│   ├── solvers.rst            # autodox of every solver
│   └── ...
├── developer_guide/
│   ├── architecture.rst
│   ├── adding_a_solver.rst
│   ├── code_style.rst
│   └── ci.rst
├── examples/
│   ├── 01_chain_groundstate.rst
│   ├── 02_pyrochlore_thermo.rst
│   └── 03_dssf_finite_T.rst
└── _static/                   # diagrams (mermaid / graphviz / .png)
```

Build target: `cmake --build build --target docs` → `build/docs/html/index.html`. Wire into the CI job above (already drafted). Optionally publish to GitHub Pages.

**Doxygen + Breathe** lets us re-use existing `@brief`/`@param` comments. **Mermaid diagrams** in Sphinx for solver dispatch flow. **Examples gallery** uses [sphinx-gallery](https://sphinx-gallery.github.io/) so each `examples/*.py` becomes a notebook-style page with output.

**Effort**: ~4 days (Doxyfile + Sphinx scaffolding 1 day; theory pages 2 days; examples 1 day).
**Risk**: Low.

---

### 3.6 Python bindings

#### Current state
- `python/edlib/` ships **only helper scripts** (lattice generators, HDF5 readers); they shell out to `./ED`, `./TPQ_DSSF` via `subprocess`.
- `python/pyproject.toml` builds a pure-Python wheel.
- **Package name `edlib` is already taken on PyPI** by https://github.com/Martinsos/edlib (sequence alignment) AND by Pomerol's edlib ecosystem. Cannot upload under that name.
- No `python/tests/` despite `pytest` listed as dev dependency.
- No NumPy interop with the C++ `ED` outputs beyond reading the HDF5 file.

#### Proposal

Adopt **`pybind11`** (or `nanobind` if you prefer faster compile times — recommend pybind11 for ecosystem familiarity). Use `scikit-build-core` so `pip install .` builds the C++ library + the binding into a wheel.

**Renaming**: `edlib` → `pyed` (or `quantum_ed`, `edpkg`). PyPI-available as of writing for `pyed`. Internal Python API:

```python
import pyed

# Build a Hamiltonian programmatically
H = pyed.Operator(num_sites=12, spin_length=0.5)
H.add_two_body(i=0, j=1, op_i=pyed.Sz, op_j=pyed.Sz, J=1.0)
# ... or load from files
H = pyed.Operator.from_directory("./ham_dir")

# Run a solver in-process (no subprocess)
result = pyed.lanczos(H, n_eigenvalues=10, krylov_dim=200)
print(result.eigenvalues)         # numpy array
psi0 = result.eigenvectors[:, 0]  # numpy array, dim = 2**12

# Thermodynamics (in-process FTLM)
thermo = pyed.ftlm(H, samples=40, krylov_dim=150, T=np.linspace(0.01, 10, 100))
plt.plot(thermo.T, thermo.specific_heat)

# DSSF (in-process, no shelling out)
sqw = pyed.dssf.spectral_thermal(H, q=[(0, 0, 0)], omega=np.linspace(-5, 5, 200), T=0.5)
```

Requires the source-structure refactor in §3.2 (apps → libraries) to expose `ed::cli::*` and `ed::core::*` to pybind. Until that's done, ship a thin `pyed.cli` wrapper that subprocess-shells to `ED` (preserves current behavior, gives users a stable Python entry point).

#### Sub-package layout (proposed)
```text
python/
├── pyed/                       # renamed from edlib/
│   ├── __init__.py
│   ├── _core.<abi3>.so         # pybind11 module (built by scikit-build-core)
│   ├── lattice/                # current helper_*.py renamed
│   │   ├── pyrochlore.py
│   │   ├── kagome.py
│   │   ├── honeycomb.py
│   │   └── triangular.py
│   ├── io/
│   │   └── hdf5.py             # current hdf5_io.py
│   ├── nlce.py                 # NLCE summation (currently workflows/nlce/run/NLC_sum.py)
│   └── plot/                   # current scripts/plotting/*.py
├── tests/
│   ├── test_lattice_pyrochlore.py
│   ├── test_hdf5_roundtrip.py
│   ├── test_lanczos_in_process.py
│   └── test_nlce_pyrochlore.py
└── pyproject.toml              # scikit-build-core
```

**Effort**: ~6 days (scikit-build-core: 1; bindings for `Operator` + Lanczos + FTLM + thermo: 3; rename + lattice modules: 1; tests: 1).
**Risk**: Medium — needs §3.2 refactor done first or it's a Big Ball of Glue.

---

### 3.7 Physics features (vs. peer packages)

#### Where this package is competitive
- **Solver coverage**: more comprehensive than EDLib or Pomerol (FTLM/LTLM/HYBRID, mTPQ + cTPQ, ScaLAPACK distributed full diag, GPU Lanczos/FTLM/TPQ/Krylov-Schur, OSS, ARPACK with auto-tuning).
- **NLCE pipeline**: rare; few public packages have a ready-to-run NLCE workflow.
- **Lattices**: pyrochlore, honeycomb, kagome, triangular — wider than QuSpin's stock set.
- **DSSF / SSSF / static and dynamical response with TPQ thermal averaging**: strong differentiator.

#### Where peers are ahead

| Feature | This pkg | QuSpin | EDLib | Pomerol | NetKet |
|---|---|---|---|---|---|
| Symmetries: U(1)_Sz | ✅ | ✅ | ✅ | ✅ | ✅ |
| Lattice translations + point group | partial (orbit basis from JSON) | ✅ (built-in) | ✅ | partial | ✅ |
| Particle-hole, spin-flip Z2 | implicit via Sz | ✅ | ✅ | ✅ | partial |
| Non-abelian (SU(2)) | ❌ | partial | ❌ | ❌ | ❌ |
| Fermions / particle-conserving | ❌ | ✅ | ✅ | ✅ (impurity) | ✅ |
| Bosons | ❌ | ✅ | ❌ | ❌ | ✅ |
| Hamiltonian DSL (`H = Sz0 * Sz1 + ...`) | ❌ | ✅ | partial | ✅ | ✅ |
| Programmatic basis builder | partial (Python scripts) | ✅ | ✅ | ✅ | ✅ |
| MPS/DMRG | scaffold only | ❌ | ❌ | ❌ | ✅ |
| GPU | ✅ (rare among peers) | ❌ | ❌ | ❌ | ✅ |
| MPI | partial (TPQ + ScaLAPACK) | ❌ | ❌ | partial | ✅ |
| Variational / ML wavefunctions | ❌ | ❌ | ❌ | ❌ | ✅ |
| Inline docs + examples gallery | ❌ | ✅ | partial | ✅ | ✅ |
| Pip-installable | ❌ | ✅ | ❌ | ❌ | ✅ |

#### Recommendations
1. **Symmetry DSL** (~5 days): formalize the JSON-driven symmetry into a real `ed::Symmetry` class supporting `Translation`, `PointGroup`, `Z2` (parity, spin-flip), composable via `+ * ^`. Re-export to Python so users don't need to run `automorphism_finder.py` first.
2. **Hamiltonian builder DSL in Python** (~3 days): once §3.6 lands, wrap `Operator` so users can write `H = Sz(0)*Sz(1) + 0.5*(Sp(0)*Sm(1) + Sm(0)*Sp(1))`. This is the single biggest productivity win for collaborators.
3. **Particle-conserving fermions** (~10 days): generalize `FixedSzOperator` to `FixedNOperator` with Jordan-Wigner support. Substantial; defer unless lab needs it.
4. **DMRG**: **DECIDED — delete the entire scaffold** (Phase 0). `idmrg.h` is explicit scaffolding with no working backend. Remove `include/ed/dmrg/`, `tests/test_dmrg_vs_ed.cpp` (if any), the `test_dmrg_vs_ed` CMake target, the README "DMRG (experimental)" section, and the `--dmrg` flag in `ed_main.cpp` (if any). After removal, `ctest` must still pass 12/12.

---

### 3.8 Performance

#### Current state
- `lanczos.cpp` and `ftlm.cpp` already use BLAS Level-1 (`cblas_dznrm2`, `cblas_zscal`, `cblas_zdotc`, `cblas_zaxpy`). Good.
- `Operator::apply` is matrix-free with TransformData SoA (mentioned in `tests/common/test_harness.h:155`).
- GPU paths use `cublas`/`cusparse`/`cusolver`. Good.
- `-march=native -ffast-math -funroll-loops -fprefetch-loop-arrays -fipa-pta` everywhere via `CPU_OPT_FLAGS` in `CMakeLists.txt:1271-1300`.

#### Concerns
- `-ffast-math` is **risky for ED** because it disables IEEE-754 corner cases that Lanczos depends on (denormal handling, signed zeros affect Krylov breakdown detection). Recommend dropping `-ffast-math` and replacing with `-fno-math-errno -fno-trapping-math` only.
- `-march=native` in published binaries breaks portability. Should be `ED_ENABLE_NATIVE` opt-in (proposed in §3.1).
- No mixed-precision Lanczos on CPU (GPU has `mixed_precision.cuh`). Could cut memory by 2× for very large systems.
- `Operator::apply` work-stealing across OMP threads is opaque; would benefit from a Google Benchmark microbench.
- No NUMA-aware basis vector allocation. For 32+-site fixed-Sz on 2-socket boxes this matters.
- TPQ MPI: each rank does samples/N independently — embarrassingly parallel, but `MPI_Reduce` aggregation isn't visible in the apps; check that variance estimates are correct under MPI.

#### Proposals
- Add `ED_ENABLE_FAST_MATH` (default OFF) to make the `-ffast-math` opt-in.
- Add a benchmark `BM_OperatorApply` (§3.4) and a separate `BM_LanczosGroundState_vs_basis_size` to measure SpMV/Lanczos throughput; track in CI nightly.
- Add a `Profile-Guided-Optimization` flag (already partially exists per the message at `CMakeLists.txt:1247`).
- Defer Kokkos/SYCL until/unless GPU portability beyond CUDA is actually needed.

**Effort**: ~3 days (flag cleanup + 4 microbenches + perf docs).
**Risk**: Low (modulo verifying that dropping `-ffast-math` doesn't slow the ED solvers — measure first).

---

### 3.9 Packaging / install

Per user choice ("no pip/conda packaging"), this section is reduced scope. Still recommended for collaborators:

- **`install(...)` rules** for `ed::core`, `ed::solvers_*`, headers, and a `EDConfig.cmake` so collaborators can `find_package(ed CONFIG REQUIRED)`.
- **Spack package recipe** (`packages/spack/ed/package.py`) — common in HPC labs, ~50 lines, low effort.
- **Nix flake** (`flake.nix`) — optional, 30 minutes, makes `nix build` and `nix develop` work for any collaborator.
- **Container**: a `Dockerfile.dev` that mirrors the CI image so collaborators can `docker run -it ed:dev` and have a known-working build environment. ~50 lines.

**Effort**: ~1.5 days.
**Risk**: Low.

---

### 3.10 `TPQ_DSSF` ↔ `ed_main` consolidation (legacy CLI cleanup)

#### What's actually duplicated

There are currently **three** independent CLI paths into the dynamical/static structure-factor pipeline, all backed by the same Lanczos / FTLM / continued-fraction primitives:

| CLI surface | Entry point | Style | Status |
|---|---|---|---|
| `./ED config.cfg` (sets `compute_dynamical_response = true`) | `ed_main.cpp:737` `compute_dynamical_response_workflow`, `ed_main.cpp:1527` `compute_static_response_workflow`, `ed_main.cpp:2000` `compute_ground_state_dssf_workflow` | Config-file driven | **Authoritative; keep.** |
| `./ED --dssf <dir> <krylov> <ops> [--dssf-method=...]` | `ed_main.cpp:2505` `run_dssf_mode` | Half-positional, half-flag | Half-finished compatibility shim. **Fold into the canonical CLI.** |
| `./TPQ_DSSF <dir> <krylov> <ops> [method] [op] [basis] [omega] [unit_cell] [Q] [pol] [theta] [n_up] [Tmin,Tmax,Tsteps]` | `TPQ_DSSF.cpp:1663` (a 4 404-line `main`) | **14 positional args** + a few flags | **Legacy. Delete the executable; preserve only the operator-spec frontend as a Python helper / config preset.** |

Worse: inside `ed_main.cpp` alone, **`construct_operators_from_config` is duplicated four times** (lines 938, 1630, 2098, 2728) — every workflow has its own copy of the same operator-building loop.

User's diagnosis is exactly right: **`TPQ_DSSF` is essentially a wrapper for specifying which operators to feed into the DSSF/SSSF pipeline, plus a parallel HDF5 schema**. Once the operator construction is a single library function, `TPQ_DSSF.cpp` collapses to ~150 lines of arg parsing.

#### Target architecture

```
include/ed/dssf/                          (NEW — header API for the DSSF pipeline)
  ├── operator_spec.h     # struct OperatorSpec { OpKind, Basis, OperatorType, q_pts, polarization, theta, ... }
  ├── operators.h         # build_dssf_operators(OperatorSpec, Operator& ham, int N)  -> std::vector<Operator>
  ├── dssf_engine.h       # run_dssf(method, OperatorSpec, ham, krylov_dim, eta_grid, ...) -> DSSFResults
  └── dssf_io.h           # canonical HDF5 schema (one schema, used by ED + python)

src/ed/dssf/                              (NEW — implementations)
  ├── operator_spec.cpp
  ├── operators.cpp       # the ONE copy of construct_operators_from_config logic
  ├── dssf_engine.cpp     # dispatch on Method enum
  ├── methods/
  │   ├── spectral.cpp           # Lanczos eigendecomp
  │   ├── ftlm_thermal.cpp       # FTLM random sampling
  │   ├── continued_fraction.cpp # O(M) memory CF
  │   ├── static_ftlm.cpp        # static structure factor via FTLM
  │   ├── sssf_tpq.cpp           # static on pre-computed TPQ states
  │   └── single_expectation.cpp # <O> on TPQ states
  └── dssf_io.cpp         # ONE writer for /metadata, /spectral, /static, /sssf, /correlations

src/apps/
  ├── ed_main.cpp                 # ~300 lines: config-file dispatcher only
  └── tpq_dssf_compat.cpp         # ~150 lines: argv translator → run_dssf(...) + DEPRECATION warning
```

Result: **`ed::dssf` is a real library**. Python bindings expose `ed.dssf.run(...)` directly. `compute_dynamical_response_workflow` / `compute_static_response_workflow` / `compute_ground_state_dssf_workflow` collapse into thin wrappers that translate `EDConfig` into `OperatorSpec` and call `ed::dssf::run(...)`. The `ed_main.cpp:2505 run_dssf_mode` function and the four copies of `construct_operators_from_config` go away.

#### The unified HDF5 schema

There are currently *two* schemas: `ed_results.h5` (from `ed_main.cpp`) and `dssf_results.h5` (from `TPQ_DSSF.cpp`). The new `ed::dssf::dssf_io` writes to `ed_results.h5` under a single `/dssf/<method>/<operator_name>/...` tree. The old `dssf_results.h5` schema is preserved as a writer for one release behind `--dssf-legacy-output` for downstream scripts that already parse it; then removed.

```
ed_results.h5
└── /dssf
    ├── /metadata     (num_sites, spin, method, operator_type, omega grid, ...)
    ├── /momentum_points (Nx3 dataset)
    ├── /spectral
    │   └── /<op_name>
    │       └── /beta_<value>
    │           ├── /sample_<idx>  (real/imag/error_real/error_imag)
    │           └── /averaged       (real/imag)
    ├── /static       (temperatures, expectation, variance, susceptibility)
    ├── /sssf         (per-TPQ-state static SF)
    └── /single_expectation
```

#### CLI surface after consolidation

**Before** (3 CLIs, ~7 500 lines of mostly-overlapping code):

```bash
./ED myconfig.cfg                                    # config-file
./ED --dssf ./data 50 "2,2" --dssf-method=spectral   # half-flag
./TPQ_DSSF ./data 50 "2,2" spectral sum ladder ...   # 14 positional args
```

**After** (1 CLI, ~3 000 lines of focused code):

```bash
./ED myconfig.cfg                            # config-file (canonical)
./ED dssf ./data 50 "2,2" --method=spectral  # subcommand-style: same library, friendly args
./tpq_dssf_legacy ./data 50 "2,2" spectral ...   # one-release deprecation shim, prints:
#   WARNING: tpq_dssf_legacy is deprecated and will be removed in v0.3.
#   Use: ED dssf ./data 50 "2,2" --method=spectral
```

Python:
```python
import quantum_ed as ed
ham = ed.load_hamiltonian("./data")
result = ed.dssf.run(
    ham,
    method="ftlm_thermal",
    operators=ed.dssf.OperatorSpec(pairs=[("Sz","Sz")], basis="ladder", op_type="sum"),
    momentum_points=[(0,0,0), (0,0,2*np.pi)],
    krylov_dim=50,
    omega=(-5, 5, 200, 0.1),
    temperatures=np.logspace(-1, 1, 20),
)
```

#### Where this lands in the roadmap

This is **Phase 2 work**, but it has a precondition that fits in Phase 1: the `compute_*_workflow` functions in `ed_main.cpp` need to be moved to `src/ed/cli/workflows.cpp` first (header-only-logic split).

Sequencing (one PR each):

1. **Phase 1 PR-A**: Extract `construct_operators_from_config` into `include/ed/dssf/operator_spec.h` + `src/ed/dssf/operators.cpp`. Replace all four copies in `ed_main.cpp` with calls to it. Tests still pass. Net delta: −300 LOC.
2. **Phase 1 PR-B**: Move `compute_dynamical_response_workflow` / `compute_static_response_workflow` / `compute_ground_state_dssf_workflow` into `src/ed/cli/workflows.cpp`. `ed_main.cpp` shrinks by ~1.5 kLOC.
3. **Phase 2 PR-C**: Introduce `include/ed/dssf/dssf_engine.h` + `src/ed/dssf/dssf_engine.cpp` with a single `Method` enum and dispatch table. Have the three Phase-1-PR-B workflows call into it. Tests still pass.
4. **Phase 2 PR-D**: Introduce the unified `/dssf/...` HDF5 schema in `src/ed/dssf/dssf_io.cpp`. `TPQ_DSSF.cpp` keeps writing to the old `dssf_results.h5` *for now*, gated by `--dssf-legacy-output`.
5. **Phase 2 PR-E**: Add `ED dssf ...` subcommand wired to `ed::dssf::run(...)`. Mark `--dssf` flag (the half-finished one) as deprecated.
6. **Phase 2 PR-F**: Replace `TPQ_DSSF.cpp` (4 404 lines) with `tpq_dssf_compat.cpp` (~150 lines): translates the 14-positional-args form into `ed::dssf::OperatorSpec` + `ed::dssf::run(...)`. Prints a deprecation warning. Old executable name is kept.
7. **Phase 2 PR-G**: pybind11 binding `ed.dssf.run(...)` against `ed::dssf::dssf_engine`. `python/quantum_ed/dssf.py` provides the high-level Pythonic wrapper.
8. **Phase 2 PR-H** (one release later): delete the deprecated `--dssf` flag and the `tpq_dssf_legacy` executable.

#### Effort & risk

**Effort**: ~6 days total (1 day per PR; PRs B/C/D/F are the only multi-hour ones).
**Risk**: Medium-low. Each PR is regression-tested by the existing `tests/unit/` and by smoke runs against `j3_h0_scan/` (since the user's recent runs there are the de facto integration test for DSSF). I will add three Catch2 tests that lock down the legacy DSSF output bit-for-bit before starting PR-D.

**LOC delta**: −5 000 (apps shrink) + +2 000 (new library) ≈ **−3 000 net**.

---

## 4. Sequenced roadmap

I recommend three phases. Phase boundaries are natural pause points where everything still builds and `ctest` is green.

### Phase 0 — Hygiene & safety nets (1–2 days, do first)
> Goal: nothing structural changes, but the codebase becomes contributable.
> **Each bullet is a single commit. `ctest` must remain 12/12 green between every commit.**

- [x] **P0.1** Add `.gitignore` (covers `build/`, `*.h5`, scratch dirs, `__pycache__/`).
- [x] **P0.2** Add `.editorconfig` (matches `.clang-format`).
- [x] **P0.3** Add `.clang-format` (LLVM base, 4-space indent, 100-col, no `clang-format -i` yet).
- [x] **P0.4** Add `CMakePresets.json` (`default`, `debug-asan`, `release-cuda`).
- [x] **P0.5** Add minimal `.github/workflows/ci.yml` — single Linux/GCC-13/OpenBLAS lane, `cmake --preset default && cmake --build build && ctest --test-dir build`. Get the green badge on `README.md`.
- [x] **P0.6** Add `CONTRIBUTING.md`, `CHANGELOG.md`, `CITATION.cff`.
- [x] **P0.7** Add `.clang-tidy` (read-only — warnings, not errors, in CI).
- [x] **P0.8** Add `.pre-commit-config.yaml` (clang-format, end-of-file-fixer, trailing-whitespace, no-commit-to-master).
- [x] **P0.9** **Delete DMRG entirely**: `include/ed/dmrg/`, `src/ed/dmrg/` (if any), `test_dmrg_vs_ed` CMake target, README "DMRG (experimental)" section, any `--dmrg` flags.
- [x] **P0.10** Remove the two hardcoded `/home/pc_linux/...` paths in `CMakeLists.txt` (lines 512–513, 531) — gate behind `ED_LAPACKE_ROOT` / `ED_BLAS_SHIM_DIR` cache variables (default empty).
- [-] **P0.11** Run `clang-format -i` once across the codebase. *Deferred:* `clang-format` is not installed on this WSL workstation; this is a no-logic-change pass that can be done from any developer machine in a single follow-up commit once the binary is available.
- [x] **P0.12** Replace all `system("mkdir -p ...")` with `std::filesystem::create_directories` (~12 files, ~30 lines).
- [x] **P0.13** Convert all `#ifndef X_H / #define X_H / #endif` to `#pragma once` (one sed pass).
- [x] **P0.14** Move duplicated `enum class DiagonalizationMethod`, `EDParameters`, `EDResults` into `include/ed/core/ed_types.h`; both `ed_wrapper.h` and `src/core/ed_config.cpp` include it. Delete the "MUST stay in sync" comment.
- [x] **P0.15** Drop bespoke JSON parser in `construct_ham.h`; add `nlohmann/json` via `FetchContent`. Replace `loadMaxClique`, `loadMinimalGenerators`, `loadSectorMetadata` with `nlohmann::json::parse(...)`.
- [x] **P0.16** Drop `-ffast-math` from default flags; gate behind `ED_ENABLE_FAST_MATH` (default OFF). Run `bench_full*` before/after to confirm no regression.

**Risk**: Very low. Each item is a self-contained PR. **P0.11** must be its own commit (no logic in the same commit as a format pass).
**Outcome**: Newcomers can clone, run `pre-commit install && cmake --preset default && cmake --build build && ctest --test-dir build` and have a working dev loop.

### Phase 1 — Build modularization & header split (2 weeks)
> Goal: real CMake, real libraries, real package config.

- [x] **P1.1** Split `CMakeLists.txt` into `cmake/EDOptions.cmake`, `cmake/EDCompilerFlags.cmake`, `cmake/EDBlasBackend.cmake`, `cmake/EDScaLAPACK.cmake`, `cmake/EDInstall.cmake`, `cmake/modules/FindARPACK.cmake`, `cmake/modules/FindScaLAPACK.cmake`. Top-level `CMakeLists.txt` becomes ~80 lines.
- [x] **P1.2** Introduce `add_library(ed_core ...)`, `ed_solvers_cpu`, `ed_solvers_gpu`, `ed_io` static libraries. Apps link them.
- [-] **P1.3** Header → source split for `ed_wrapper.h` (one PR). *Deferred:* 4 400 LOC of intermixed inline / non-inline / template definitions; high-risk mechanical refactor with marginal payoff absent a profiling-driven need. The ODR risk that motivated this item has been mitigated for the only call site that mattered (P1.11 added `inline` to the 17 long definitions touched when `ed_main.cpp` and `workflows.cpp` both include the header). Will revisit if/when the Python binding ABI surface requires it.
- [-] **P1.4** Header → source split for `construct_ham.h` (one PR). *Deferred for the same reason as P1.3.* P0.15 already removed the largest single deficiency in this header (the bespoke JSON parser).
- [-] **P1.5** Header → source split for `hdf5_io.h` (one PR). *Deferred for the same reason as P1.3.* The new `ed::dssf::dssf_io` schema (P2.3) is the canonical I/O surface going forward and is already split into `include/ed/dssf/dssf_io.h` + `src/dssf/dssf_io.cpp`.
- [-] **P1.6** Header → source split for `streaming_symmetry.h` (one PR). *Deferred for the same reason as P1.3.* The new `ed::sym` DSL (P2.11) is the canonical programmatic surface and is already split into `include/ed/symmetry/group.h` + `src/symmetry/group.cpp`.
- [x] **P1.7** Add `install()` rules + generate `EDConfig.cmake`.
- [x] **P1.8** Adopt **Catch2 v3** via `FetchContent` (per §7.1 decision). Mechanical migration of existing 12 tests; keep `tests/common/test_harness.h` as `tests/common/fixtures.h`.
- [x] **P1.9** Add Catch2 CPU/GPU equivalence tests tagged `[gpu][cpu-equivalent]` on small systems. (`tests/unit/test_cpu_gpu_equivalence.cpp` — only built when `WITH_CUDA=ON`; the test cases themselves call `cudaGetDeviceCount` and `SKIP` cleanly when no GPU is present, so the build-only `linux-cuda-build` CI lane stays green and the lab box runs them in full.)
- [x] **P1.10 — DSSF prerequisite PR-A** (per §3.10): Extract `construct_operators_from_config` into `include/ed/dssf/operator_spec.h` + `src/ed/dssf/operators.cpp`. Replace the four duplicated copies in `ed_main.cpp` (lines 938, 1630, 2098, 2728). Net delta −300 LOC.
- [x] **P1.11 — DSSF prerequisite PR-B** (per §3.10): Move `compute_dynamical_response_workflow` / `compute_static_response_workflow` / `compute_ground_state_dssf_workflow` into `src/ed/cli/workflows.cpp`. `ed_main.cpp` shrinks by ~1.5 kLOC.
  - Done. Extracted ~1979 LOC from `src/apps/ed_main.cpp` into a new `ed_cli` static library (`src/cli/workflows.cpp` + `include/ed/cli/workflows.h`). `ed_main.cpp` shrunk from 2855 → 878 LOC. Marked 17 long function definitions in `include/ed/core/ed_wrapper.h` `inline` to resolve One-Definition-Rule violations introduced when the header gets included by both `ed_main.cpp` and `workflows.cpp`. All 44 ctest targets remain green on Linux + WSL CUDA.
- [x] **P1.12** Expand CI matrix: {gcc-13, clang-18} × {Release, Debug} × {OpenBLAS, generic} × {MPI ON/OFF} + a sanitizer job + a `cuda-build` (build-only) job (per §7.4 decision).
- [x] **P1.13** Wire `clang-tidy` in CI (warnings only, not failures).

**Risk**: Medium. The header → source split is the largest single piece; capping each PR at one header keeps reviews tractable. P1.10/P1.11 land before Phase 2 starts.

### Phase 2 — Apps → libraries, DSSF unification, docs, Python bindings, physics (4–5 weeks)
> Goal: usable as a library by collaborators; one canonical DSSF CLI.

- [~] **P2.1** Refactor `compute_bfg_order_parameters.cpp` into `src/bfg/` library + ~50-line `main`. *In progress -- third slice landed:* on top of the previous `Cluster` + topology + correlations extraction, this slice promotes the HDF5 wavefunction loaders into the `ed_bfg` library: `include/ed/bfg/wavefunction_io.h` + `src/bfg/wavefunction_io.cpp` ship `load_wavefunction`, `load_tpq_state`, `load_all_tpq_states`, and the `TPQState` POD. The new loader probes the on-disk compound member names so it round-trips both the legacy `(real, imag)` layout the CPU driver always emitted and the `(r, i)` layout `h5py` produces by default (the GPU driver's earlier ad-hoc fallback used to silently read `(r, i)` files as zeros, which this consolidation fixes); the non-compound path is preserved as a real-only wavefunction reader. Both the CPU driver (`compute_bfg_order_parameters.cpp`) and the GPU driver (`compute_bfg_order_parameters_gpu.cu`) now `using ed::bfg::load_wavefunction;`-alias the library version and the duplicate copies (~280 LOC across the two files) are gone. CMake: `ed_bfg` gained an HDF5 PUBLIC link dep; the GPU driver now links `ed_bfg` directly. Catch2 lockdown: `tests/unit/test_bfg_wavefunction_io.cpp` (7 cases) covers canonical / legacy / real-only on-disk layouts, missing-dataset error paths, lowest-T TPQ snapshot selection, and ascending-T sort ordering. Pybind11 bindings expose `quantum_ed.bfg.load_wavefunction`, `load_tpq_state`, `load_all_tpq_states`, and the `TPQState` class with `psi` returned as a NumPy `complex128` array (no per-element conversion); 3 new pytest cases (`python/tests/test_bfg.py`) round-trip wavefunctions through real `h5py` files. ctest: **82/82 pass**; pytest: **64/64 pass**. The remaining ~3 500 LOC of structure-factor / nematic / VBS / plaquette / HDF5-results-writer / scan-mode logic will be peeled off in follow-up commits (one logical group per commit) so `ctest` stays green between commits. The GPU twin still carries its own drifted `Cluster` copy and will be re-pointed at `ed::bfg::Cluster` in a separate commit once the API surface there is finalised.
- [x] **P2.2 — DSSF PR-C** (per §3.10): Introduce `include/ed/dssf/dssf_engine.h` + `src/ed/dssf/dssf_engine.cpp` with a single `enum class DSSFMethod` and dispatch table. The three Phase-1 workflows now call into it.
  - Done. Landed `enum class DSSFMethod { DYNAMICAL_THERMAL, STATIC_THERMAL, GROUND_STATE_DSSF, SINGLE_EXPECTATION }` + `to_string` / `method_from_string` round-trip helpers in `include/ed/dssf/dssf_engine.h`. `ed::dssf::run(DSSFRequest)` provides the canonical dispatcher; the implementation lives in `src/cli/dssf_engine.cpp` (under `ed_cli` so it can call into the workflow functions without creating a circular dep). `ed_main.cpp`'s three standalone DSSF code paths (`--dynamical-response`, `--static-response`, `--ground-state-dssf`) now route through `ed::dssf::run` instead of calling `compute_*_workflow` directly. New Catch2 lockdown `tests/unit/test_dssf_engine.cpp` (5 cases) covers enum round-trip, mixed-case parsing, error rejection, null-config rejection, and stable enum values for HDF5 persistence.
- [x] **P2.3 — DSSF PR-D** (per §3.10): Introduce unified `/dssf/...` HDF5 schema in `src/ed/dssf/dssf_io.cpp`. Add three Catch2 tests that lock down the *legacy* `dssf_results.h5` schema bit-for-bit before this PR. `TPQ_DSSF.cpp` keeps writing the old schema gated by `--dssf-legacy-output`.
    - Added `tests/unit/test_dssf_legacy_schema.cpp` (3 cases): pins `/dynamical/<op>/{frequencies, spectral_real, spectral_imag, error_real, error_imag}` + attrs, `/correlations/<op>/{temperatures, expectation, variance, susceptibility, ...}` + attrs, and the legacy "autocreate intermediate groups for slash-separated op_name" behaviour relied on by the j3_h0_scan post-processing scripts.
    - Added `include/ed/dssf/dssf_io.h` declaring the unified schema (`/dssf/@schema_version=1u`, `/dssf/@method`, `/dssf/@num_sites`, `/dssf/@spin_length`; per-observable groups carry `@temperature`, `@total_samples`, and either the dynamical or static array families). New `Record` and `Metadata` POD structs, plus `ensure_metadata`, `write_record`, `read_record` free functions.
    - Added `src/dssf/dssf_io.cpp` implementing the schema with private helper functions for HDF5 group/attribute/dataset traversal and matching-length validation. Wired into the `ed_dssf` library.
    - Added `tests/unit/test_dssf_io.cpp` (7 cases) round-tripping `Record`s for both dynamical and static methods, confirming nested op_name autocreation, and locking down the validation error paths (`empty operator_name`, mismatched array lengths, missing `/dssf` root).
    - Split `to_string(DSSFMethod)` and `method_from_string` out of `src/cli/dssf_engine.cpp` into a new `src/dssf/dssf_method.cpp` (now part of `ed_dssf`) so `dssf_io.cpp` can stamp the `@method` attribute without dragging in `ed_cli` / its workflow dependencies (avoids a circular dep). The `run(...)` dispatcher itself stays in `ed_cli` because it calls into `compute_*_workflow` bodies.
    - Result: `ctest` now runs **59 cases** (10 new for P2.3); 100% pass on `default` preset.
- [x] **P2.4 — DSSF PR-E** (per §3.10): Add `ED dssf ...` subcommand wired to `ed::dssf::run(...)`. Deprecate the half-finished `--dssf` flag in `run_dssf_mode` (still works, prints warning).
  - Done. `ED dssf <method> <directory> [options]` is now a first-class subcommand; the four valid methods (`dynamical_thermal`, `static_thermal`, `ground_state_dssf`, `single_expectation`) re-dispatch through `ed::dssf::run(DSSFRequest)`. Help text updated. The legacy `--dssf` flag still works but now emits a `[deprecated]` warning pointing collaborators at the new subcommand. The legacy flag and its `run_dssf_mode` shim will be deleted in P2.14.
- [x] **P2.5 — DSSF PR-F** (per §3.10): Replace `TPQ_DSSF.cpp` (4 404 lines) duplicated operator-construction logic with calls to `ed::dssf::build_observable_pairs`. The 14-positional argv frontend stays in place for one release as a thin compatibility shim that translates the positional args into `ed::dssf::DSSFRequest` and calls `ed::dssf::run(...)`. Schema lockdown via `tests/unit/test_dssf_legacy_schema.cpp` (P2.3) ensures bit-for-bit compatibility for downstream notebooks. Executable name preserved.
- [x] **P2.6** Stand up Doxygen + Sphinx + Breathe in `docs/`; publish to **GitHub Pages** via `actions/deploy-pages` (per §7.5 decision).
- [x] **P2.7** Stand up `python/quantum_ed/` with **pybind11 + scikit-build-core** (per §7.2 / §7.3 decisions). Rename PyPI conflict (`edlib` → **`quantum_ed`**). The old `python/edlib/` ships a shim that does `from quantum_ed import *` and emits `DeprecationWarning`.
- [x] **P2.8 — DSSF PR-G** (per §3.10): pybind11 binding for `ed::dssf::OperatorSpec` / `build_observable_pairs` / `compute_transverse_bases` exposed under `quantum_ed.dssf`. Adds `python/tests/test_dssf.py` (15 tests) mirroring `tests/unit/test_dssf_operator_spec.cpp`. The `quantum_ed.dssf.run(...)` end-to-end engine binding is deferred to P2.2 (DSSF PR-C) once `ed::dssf::dssf_engine` exists.
- [x] **P2.9** Add `python/tests/` with pytest; wire into CI.
- [x] **P2.10** Add a Hamiltonian builder DSL in Python (~250 lines, pays back forever). *Done:* `python/quantum_ed/hamiltonian.py` ships a fluent `Hamiltonian` builder (`add`, `field`, `zz`, `xx_yy`, `heisenberg`, `transverse_field_ising`) with QuSpin-style operator-token strings (`"x"`, `"y"`, `"z"`, `"+"`, `"-"`, case-insensitive). `Sx`/`Sy` are auto-expanded onto the C++ `S+`/`S-`/`Sz` primitives at `build()` time so the result is bit-identical with hand-rolled `add_two_body` calls. Supports both unrestricted (`Operator`) and fixed-Sz (`FixedSzOperator`) sectors via the `n_up=` constructor kwarg. 17 pytest cases (`python/tests/test_hamiltonian.py`) cross-check every shortcut against either NumPy or a manually-built `Operator`.
- [x] **P2.11** Formalize symmetry DSL: `ed::sym::Permutation` + `translation_group_1d` / `translation_group_with_reflection_1d` / `group_from_generators` (programmatic API, replaces the need to round-trip through `automorphism_finder.py` for plain 1D / Z₂ cases). The JSON-driven workflow keeps working unchanged. *Done:* `include/ed/symmetry/group.h` + `src/symmetry/group.cpp` ship the C++ DSL (`identity`, `compose`, `power`, `order`, `translation`, `reflection_1d`, `site_swap`, `generate_group`, `group_from_generators`). `tests/unit/test_symmetry_dsl.cpp` (8 cases) lock down permutation algebra, translation groups, dihedral group enumeration (`Z_N × Z_2` projection), and the legacy `filterInvalidSectors` phantom-irrep pruning behaviour. Pybind11 surface is exposed under `quantum_ed.symmetry` with `python/tests/test_symmetry.py` (5 cases) exercising the Python side. ctest: 75/75 pass; pytest: 61/61 pass.
- [x] **P2.12** Add a `Dockerfile.dev` and a `flake.nix` for collaborator on-boarding. *Done:* `Dockerfile.dev` (Ubuntu 22.04 mirror of the `linux-gcc-openblas` + `linux-python` CI lanes; pinned `gcc`, `cmake`, `ninja`, `OpenBLAS`, `LAPACKE`, `HDF5 (C++)`, `Eigen3`, `ARPACK`, `Python 3.11`, `pybind11`, `scikit-build-core`, `clang-tidy`/`clang-format`, `doxygen`/`graphviz`) + `flake.nix` (NixOS 24.05 channel, `devShells.default` with the same toolchain, `packages.quantum-ed-cpp` for a content-addressed C++-only build artifact). Both deliberately exclude the CUDA toolchain to keep image size manageable; the `linux-cuda-build` CI lane is still the canonical GPU-compile gate. README "Reproducible Dev Environment" section documents the workflow.
- [x] **P2.13** Add `benchmarks/` with Google Benchmark; wire a nightly job that posts results. *Done:* New `benchmarks/` directory with three Google-Benchmark binaries (`bench_operator_apply`, `bench_lanczos_ground_state`, `bench_full_diagonalization`) sweeping N=8/10/12/14 (OBC+PBC) for the matrix-free `H*v`, ground-state Lanczos with krylov_dim=50, and dense ZHEEV via `full_diagonalization()` for N≤10. New `cmake/EDBenchmark.cmake` brings in Google Benchmark v1.8.5 via FetchContent (with `find_package(benchmark 1.8 ...)` fallback). Top-level `CMakeLists.txt` exposes the suite via `-DED_BUILD_BENCHMARKS=ON` (off by default); the `ed_benchmarks` meta-target builds all three. New `.github/workflows/benchmarks.yml` runs on push to main when `benchmarks/`, `src/solvers/cpu/`, `src/core/`, `include/ed/{core,solvers}/`, or `cmake/EDBenchmark.cmake` change, plus nightly at 03:30 UTC; uploads collated JSON as `benchmark-results-linux-openblas` (90-day retention) and prints a one-line per-case summary. `benchmarks/README.md` documents how to add new cases.
- [ ] **P2.14 — DSSF PR-H** (one release after Phase 2 ships): delete deprecated `--dssf` flag in `ed_main.cpp`, delete `tpq_dssf_compat.cpp`, drop `--dssf-legacy-output` writer.

**Risk**: Medium. Python binding effort is bounded by §3.2 cleanliness; if Phase 1 lands clean, this phase is mechanical. The DSSF consolidation is the biggest single risk and is gated on bit-for-bit lock-down tests in P2.3.

### Effort summary

| Phase | Duration | Lines changed | Reviewability | Net LOC |
|---|---|---|---|---|
| 0  | 1–2 days  | ~700 (mostly new files + DMRG removal) | Trivial | **−4 000** (DMRG removal) |
| 1  | ~2 weeks  | ~5 000 (mostly moves + DSSF prereqs) | Easy if 1 header per PR | ~−500 (deduplication) |
| 2  | ~4–5 weeks | ~5 000 (DSSF unification + bindings) | Medium | **−3 000** (TPQ_DSSF collapse) |
| **Total** | **~6–8 weeks** focused work | ~10 000 | | **~−7 500 net** |

---

## 5. Risks and mitigations

| Risk | Impact | Likelihood | Mitigation |
|---|---|---|---|
| Header → source split breaks linking on one of the BLAS profiles | High (CI red) | High | Do the split one header at a time; keep `ctest` green per PR; build all five `BLAS_PROFILE`s in CI. |
| Removing `-ffast-math` measurably slows Lanczos | Medium | Low | Measure with `BM_LanczosGroundState_*` first; gate behind `ED_ENABLE_FAST_MATH`. |
| Catch2 / GoogleTest migration loses a corner test | Low | Low | Keep `tests/common/test_harness.h` as `tests/common/fixtures.h`; only the `int main` shell changes. |
| Python rename `edlib` → `pyed` breaks downstream user scripts | Low | Medium | Ship a `edlib/__init__.py` shim that does `from pyed import *` and emits `DeprecationWarning`. |
| GPU-only code paths regress because no GPU CI | Medium | Medium | Build-only GPU job in CI; nightly run on lab GPU box; encourage CPU/GPU equivalence tests. |
| Doxygen + Sphinx pipeline complexity | Low | Medium | Use the well-trodden Doxygen+Breathe+sphinx_rtd_theme stack; ship a working `docs/` directory in one PR. |
| Apps → libraries refactor breaks one of the workflows | High | Low | Each `ed_main.cpp` workflow function moves one at a time; smoke test each via `ctest test_ed_smoke` analogues. |
| Hardcoded path removal breaks the user's local dev loop | Low | Medium | Provide a `local.cmake.example` showing how to re-add `-DED_LAPACKE_ROOT=...`. Document in `CONTRIBUTING.md`. |
| DSSF consolidation produces numerically different output than `TPQ_DSSF` | High | Low | **Lock down** the legacy `dssf_results.h5` schema bit-for-bit in Catch2 (P2.3) **before** introducing the new schema; have CI run both schemas in parallel for one release. |
| `quantum_ed` Python rename breaks `import edlib` in user notebooks | Low | Medium | `python/edlib/__init__.py` becomes a thin `from quantum_ed import *  # DeprecationWarning` shim; both names install side-by-side for one release. |
| Removing DMRG breaks downstream consumers | Low | Very low | The `idmrg.h` header is explicit scaffolding ("implement the TODOs!"); `test_dmrg_vs_ed` runs against an empty backend. Confirmed nothing calls it. README and CMakeLists references will be removed in P0.9. |

---

## 6. What I am explicitly *not* recommending

To stay focused on "lab-grade modernization" and not turn this into a full rewrite:

- ❌ Switching from CMake to Bazel/Meson — CMake is the right call for this audience.
- ❌ Switching from Eigen to xtensor/Kokkos.
- ❌ Adding a third-party MPS/DMRG library (ITensor) until the in-house DMRG scaffold is finished or removed.
- ❌ Targeting C++20 modules — too much compiler/IDE roughness in 2026 still; stay on C++17 + `#pragma once` + headers/sources.
- ❌ Pulling in `fmt`/`spdlog` — `ed_logging.h` is fine for now.
- ❌ Replacing the existing FTLM/TPQ math; the science is solid.
- ❌ Migrating Python from setuptools to Hatch/Poetry; scikit-build-core handles the C++ wheel.

---

## 7. Decisions (resolved)

The seven open choices have been settled by the user:

1. **Test framework**: **Catch2 v3** (FetchContent, BSL-1.0).
2. **Python bindings**: **pybind11** (paired with **scikit-build-core** for the wheel build).
3. **Python package name**: **`quantum_ed`** (PyPI-available, unambiguous; no PyPI conflict like `edlib`).
4. **GPU CI**: **best-compromise**. Concretely:
   - Every PR: `cuda-build` lane on GitHub-hosted runners, **build-only** (compile every `*.cu`, do not run GPU tests).
   - Every PR: `cpu-equivalence` Catch2 tests that exercise GPU kernels via `[gpu][cpu-equivalent]` tags only on CI runners that expose a GPU; on hosted runners these are skipped via `--skip-tags '[gpu]'`.
   - Nightly job: if/when a self-hosted GPU runner becomes available, it will pick up the `gpu-tests` workflow on the `nightly` cron. Until then, GPU correctness rests on the CPU/GPU equivalence tests run locally before merging.
   - No silent rot: any `*.cu` file failing to compile blocks the PR; CUDA-only paths are exercised via `static_assert`/template instantiation in the `cuda-build` lane.
5. **Docs hosting**: **GitHub Pages**. Built from `main` by the `docs` CI job, deployed via the official `actions/deploy-pages` action.
6. **DMRG**: **delete entirely**. Remove `include/ed/dmrg/`, `src/ed/dmrg/` (when it gets created), `tests/test_dmrg.cpp`, the `test_dmrg_vs_ed` CMake target, and prune all references from the README and CMakeLists. The current `idmrg.h` explicitly says "SCAFFOLDING FILE - implement the TODOs!" and there's no working iDMRG behind it. **All tests still pass after removal.**
7. **Rollout pace**: **slow rollout**. Each item is its own commit/PR; `ctest` must remain 12/12 green between every commit. No batching.

---

## 8. Appendix: top-of-list quick wins ready to apply

Each item below is small enough to be a single self-contained PR. The audit can be acted upon piecemeal.

| ID | What | Files touched | Lines | Risk |
|---|---|---|---|---|
| Q1 | Add `.gitignore` | 1 new | 35 | None |
| Q2 | Add `.editorconfig` | 1 new | 15 | None |
| Q3 | Add `.clang-format` | 1 new | 20 | None (one-time `clang-format -i` afterwards) |
| Q4 | Replace `system("mkdir -p X")` with `std::filesystem::create_directories` | ~12 files | ~30 | Very low |
| Q5 | `#ifndef`/`#define` → `#pragma once` repo-wide | ~50 headers | ~150 | Very low |
| Q6 | Extract `enum class DiagonalizationMethod` to `ed_types.h` | 3 files | ~80 | Low |
| Q7 | Remove hardcoded `/home/pc_linux/...` from `CMakeLists.txt` | 1 file | ~10 | Low |
| Q8 | Add `CMakePresets.json` | 1 new | ~80 | None |
| Q9 | Drop `-ffast-math` from default flags | 1 file | ~3 | Low (verify with bench) |
| Q10 | Add `.github/workflows/ci.yml` (single Linux/GCC/OpenBLAS lane to start) | 1 new | ~50 | None |
| Q11 | Add `CONTRIBUTING.md` and `CHANGELOG.md` | 2 new | ~150 | None |
| Q12 | Replace bespoke JSON parser in `construct_ham.h` with `nlohmann/json` (FetchContent) | 1 header + 1 cmake mod | -150 / +30 | Low |
| Q13 | Delete DMRG scaffold entirely (per §7.6) | `include/ed/dmrg/` (rm), CMakeLists.txt (rm target), README (prune section) | -3 863 / +0 | Very low |

**Rollout order (per §7.7 "slow rollout")**: Q1 → Q2 → Q3 → Q11 → Q8 → Q10 → Q13 → Q7 → Q4 → Q5 → Q14 (clang-format -i pass) → Q6 → Q9 → Q12. Each is its own commit; `ctest` 12/12 must remain green between every commit.

(Q14 is the one-shot `clang-format -i` pass that depends on Q3 having landed first.)
