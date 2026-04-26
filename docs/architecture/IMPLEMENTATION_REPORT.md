---
title: Exact Diagonalization Toolkit — Implementation Report
audience: contributors, reviewers, advanced users
last_updated: 2026-04-24
orphan: true
---

# Exact Diagonalization Toolkit — Implementation Report

This document is an exhaustive, pedagogical walk-through of the working
functionality of the C++/CUDA/Python codebase rooted at
`exact_diagonalization_cpp/`. It traces every layer that is reachable from
the user-facing CLIs (`ED`, `compute_bfg_order_parameters[_gpu]`, the Python
extension `quantum_ed._core`, and the NLCE workflow package) down to the
algorithmic kernels and on-disk schemas they ultimately invoke.

The goal is twofold: (i) help a new contributor reason about the codebase
end-to-end without having to reverse-engineer the headers, and (ii) give an
experienced contributor a stable, citeable index of what each subsystem is
actually doing today.

Throughout, paths are relative to `exact_diagonalization_cpp/` unless noted.

---

## Table of Contents

1. [Project framing](#1-project-framing)
2. [Repository layout & build system](#2-repository-layout--build-system)
3. [Core types — `Operator`, `FixedSzOperator`, `EDConfig`](#3-core-types)
4. [`ed_wrapper.h` — the dispatch hub](#4-ed_wrapperh--the-dispatch-hub)
5. [CPU solvers](#5-cpu-solvers)
6. [GPU solvers](#6-gpu-solvers)
7. [Symmetry / projection pipeline](#7-symmetry--projection-pipeline)
8. [DSSF / SSSF engine and workflows](#8-dssf--sssf-engine-and-workflows)
9. [BFG: bond-bilinear, structure factors, order parameters](#9-bfg-bond-bilinear-structure-factors-order-parameters)
10. [I/O layer (HDF5 schema & basis buffers)](#10-io-layer-hdf5-schema--basis-buffers)
11. [CLI: `ED` driver](#11-cli-ed-driver)
12. [Python bindings (`quantum_ed`) and `edlib`](#12-python-bindings)
13. [NLCE workflow package](#13-nlce-workflow-package)
14. [Tests, benchmarks, and operational notes](#14-tests-benchmarks-and-operational-notes)
15. [Subsystem cross-reference](#15-subsystem-cross-reference)

---

## 1. Project framing

The toolkit's stated purpose (from `README.md`) is to solve quantum lattice
spin Hamiltonians by **exact diagonalization (ED)** — ground states, full
spectra, finite-temperature thermodynamics, and dynamical / static response
functions — with optional **GPU acceleration**, **MPI** parallelism,
**symmetry-projected** matrix-free representations, and an end-to-end
**Numerical Linked Cluster Expansion (NLCE)** orchestrator.

Concretely, the toolkit exposes four public surfaces:

- **`ED`** — the canonical CLI binary (`src/apps/ed_main.cpp`). Single
  entry-point for diagonalization, thermodynamics, and the
  `ED dssf <method>` subcommand for response functions.
- **`compute_bfg_order_parameters` / `_gpu`** — research drivers that
  post-process saved wavefunctions or TPQ states to compute bond-bilinear
  observables, structure factors, and VBS / nematic / plaquette order
  parameters.
- **`quantum_ed`** — a `pybind11` Python package that wraps the CPU
  solver/observable layer for use from notebooks and other tools.
- **`workflows.nlce`** — a Python package that orchestrates ED across
  cluster expansions (geometry × pipeline matrix), driving the `ED`
  binary as a subprocess.

The codebase is sized at ~71,000 lines (≈42 k lines of C++/CUDA `.cpp`/`.cu`
plus ≈29 k lines of `.h`/`.cuh` headers), with a tested kernel surface of
**103 Catch2 tests** registered through CTest.

---

## 2. Repository layout & build system

### 2.1 Layout

```
include/ed/                # Public headers
├── core/                  # Operator, FixedSzOperator, EDConfig, ed_wrapper, HDF5IO
├── solvers/               # Lanczos, ARPACK, FTLM/LTLM, TPQ, dynamics, ScaLAPACK, CG
├── gpu/                   # GPU operator, kernels, Lanczos/Davidson/FTLM/TPQ/dynamics
├── symmetry/              # `ed::sym` permutation DSL
├── dssf/                  # DSSF engine seam + OperatorSpec + DSSF I/O
├── bfg/                   # Cluster, correlations, structure factor, order params, ring obs
├── io/                    # In-RAM Lanczos basis buffer, HDF5 basis-vector storage
└── cli/                   # Workflow declarations consumed by `ED` and `dssf_engine`

src/                       # Implementations (mirrors `include/ed/...`)
├── apps/                  # ed_main.cpp + BFG drivers (CPU + CUDA)
├── core/ed_config.cpp     # EDConfig parser/validator/saver
├── cli/                   # workflows.cpp + dssf_engine.cpp (the engine seam)
├── dssf/                  # operator_spec.cpp + dssf_method.cpp + dssf_io.cpp
├── solvers/cpu/           # All CPU solver TUs
├── solvers/gpu/           # All CUDA TUs
├── symmetry/group.cpp     # `ed::sym` impls
├── bfg/                   # BFG library
└── io/                    # Basis storage + buffer

python/quantum_ed/         # pybind11 module (CPU only)
python/edlib/              # Lattice/automorphism helpers (Python)
workflows/nlce/            # NLCE orchestrator (calls `ED` binary)
tests/unit/                # Catch2 v3 unit tests, registered via ctest
benchmarks/                # Google Benchmark micro-benchmarks
docs/                      # Doxygen + Sphinx scaffolding
configs/                   # Worked example .cfg INI files
```

### 2.2 Build options

`CMakeLists.txt` exposes:

- `WITH_CUDA` — toggles every `ed_solvers_gpu` target and the `ED` /
  BFG GPU links. Defaults match the host's CUDA availability, then the
  user's `local.cmake` (see `local.cmake.example`).
- `WITH_MPI` — toggles MPI in `ed_solvers_cpu`, `ed_cli`, the BFG drivers,
  and the optional ScaLAPACK path. The wheel build (`pyproject.toml`)
  forces it `OFF`.
- `WITH_SCALAPACK` — gated by `cmake/EDMpiScalapack.cmake`; only enabled
  when `WITH_MPI=ON` and a compatible BLAS profile (`MKL`, `AOCL`, etc.)
  is detected. ScaLAPACK is consumed by `scalapack_diag.cpp`.
- `BLAS_PROFILE` (`AUTO`, `MKL`, `AOCL`, `OPENBLAS`, `REFBLAS`) —
  selects the BLAS/LAPACK shim included by `blas_lapack_wrapper.h`.
- `ED_BUILD_PYTHON` — adds `python/quantum_ed` as a CMake subdirectory
  and produces `_core*.so` next to `__init__.py` (so `import quantum_ed`
  works in-tree).
- HDF5 (C++ API) and ARPACK are **required**; they are not gated by an
  option. They are looked up in `cmake/EDDependencies.cmake`.

### 2.3 Static libraries (in dependency order)

Defined in `cmake/EDLibraries.cmake`:

| Target              | Sources (representative)                                   | Notes                                       |
|---------------------|------------------------------------------------------------|---------------------------------------------|
| `ed_io`             | `lanczos_basis_buffer`, `basis_vector_storage`             | Low-level I/O primitives                    |
| `ed_core`           | `ed_config.cpp`                                            | Pulls in `include/ed/core/*` headers        |
| `ed_solvers_cpu`    | All `src/solvers/cpu/*.cpp`                                | Lanczos, ARPACK, FTLM/LTLM, TPQ, dynamics, ScaLAPACK |
| `ed_solvers_gpu`    | All `src/solvers/gpu/*.cu` (+ conversion `.cpp`)           | Only built when `WITH_CUDA=ON`              |
| `ed_dssf`           | `operator_spec.cpp`, `dssf_method.cpp`, `dssf_io.cpp`      | Pure observable construction + I/O          |
| `ed_symmetry`       | `group.cpp`                                                | `ed::sym` permutation DSL                   |
| `ed_bfg`            | All `src/bfg/*.cpp`                                        | Research-grade order parameters             |
| `ed_cli`            | `workflows.cpp`, `dssf_engine.cpp`                         | The high-level workflow seam                |

### 2.4 Executables

| Target                                     | Source                                              |
|--------------------------------------------|-----------------------------------------------------|
| `ED`                                       | `src/apps/ed_main.cpp`                              |
| `compute_bfg_order_parameters`             | `src/apps/compute_bfg_order_parameters.cpp`         |
| `compute_bfg_order_parameters_gpu`         | `src/apps/compute_bfg_order_parameters_gpu.cu`      |

### 2.5 Presets

`CMakePresets.json` ships configurations for `default`, `release-mpi`,
`release-cuda`, `release-cuda-mpi`, `debug`, and `debug-asan`. The
`build-clang-debug/` directory illustrates an alternative compiler tree.

---

## 3. Core types

### 3.1 `Operator` (`include/ed/core/construct_ham.h`)

`Operator` is the **matrix-free** spin Hamiltonian. It models a Hilbert
space of dimension \(2^N\) where each basis state is a length-`N` bit
pattern (one bit per site, spin-½ encoding). It never stores a dense
matrix in normal use; `apply(in, out, dim)` is a sparse matrix-vector
product driven by per-term bit logic.

#### File formats

`Operator` consumes three text files in the toolkit's canonical layout:

- **`Trans.dat`** (single-site / one-body) via `loadFromFile`.
  Header: skip 2 lines, read a `... numLines` line, skip 3 more, then
  one term per line: `Op indx E F` where `Op ∈ {0, 1, 2}` selects
  `S+ / S- / Sz`, `indx` is the site, and `E + iF` is the (complex)
  coefficient.
- **`InterAll.dat`** (two-body) via `loadFromInterAllFile`. Same header
  pattern, then `Op_i indx_i Op_j indx_j E F` per line.
- **`ThreeBodyG.dat`** (three-body) via `loadThreeBodyTerm`. Per-line
  format: `op_type_1 site_1 op_type_2 op_type_3 op_type_4 site_2 real imag`.
  By convention `op_type_3` is repurposed as a **third-site index** in the
  legacy file format.

#### Internal storage

Two parallel representations coexist:

- A unified list `transform_data_` of `TransformData` (used by the legacy
  `addTransform` path and the dense / sparse builders).
- A **structure-of-arrays** decomposition built lazily by
  `separateTransformsByType()` and consumed by the hot `apply` path:
  `diag_one_body_`, `offdiag_one_body_`, `diag_two_body_`,
  `mixed_two_body_`, `offdiag_two_body_`. This decomposition removes
  branches from the inner loop.
- `three_body_data_` for three-body `ThreeBodyTransformData`.
- Optional assembled CSR (`sparseMatrix_` / `sparseMatrixReal_`) cached
  for the few code paths that opt in via `ED_USE_SPARSE` / `ED_SPARSE_DIM_MAX`.

#### `apply(const Complex* in, Complex* out, size_t dim)`

The hot path. Three branches:

1. If a CSR has already been built or env vars demand sparse, use an
   assembled CSR SpMV (real-only when both `H` and `x` are real).
2. Else if `H` is real and `dim` is large, dispatch to `apply_real`.
3. Else default to `apply_optimized`: an OpenMP-parallel,
   cache-blocked loop over basis indices, with per-thread `(index, value)`
   buffers, a radix sort to merge per-target writes, and an atomic add
   into the complex `out` (real + imag halves are atomically added as
   doubles).

Bit-level matrix elements use the same rules as the symbolic builders:
flips for `S±`, sign from the bit for `Sz`, products of `spin_l_` factors.

### 3.2 `FixedSzOperator`

A subclass that **restricts the Hilbert space** to states with a fixed
total `S^z` (i.e. a fixed up-spin count `n_up`). The dimension drops to
\(\binom{N}{n_{\mathrm{up}}}\).

Key features:

- **Basis enumeration.** `generateFixedSzBasis` walks all `N`-bit
  integers with exactly `n_up` set, in lexicographic order, using
  Gosper's hack (`nextFixedSzState`).
- **Index mapping.** A constant-time **`LinIndexTable`** (H. Q. Lin
  1990) maps a state to its index after `build()`, with a
  `binarySearchState` fallback.
- **Term projection.** Same loaders as `Operator` push terms; in
  `apply`, off-diagonal targets that fall outside the sector get
  `lookupState(...) == -1` and contribute nothing.
- **Symmetric arithmetic only.** Terms that change `n_up` (lone `S±` in
  the Hamiltonian) are dropped; for *observables* outside the
  Hamiltonian they may still be applied if the resulting state lies in
  the same sector.

### 3.3 `EDConfig` (`include/ed/core/ed_config.h`, `src/core/ed_config.cpp`)

`EDConfig` is the canonical **run configuration**. It is built either
from CLI arguments (`fromCommandLine`) or an INI file (`fromFile`),
validated (`validate`), saved (`save`), and merged (`merge`).

Sub-structs:

| Struct                   | Field                  | Concern                                 |
|--------------------------|------------------------|-----------------------------------------|
| `DiagonalizationConfig`  | `diag`                 | num_eigenvalues, max_iter, tolerance, eigenvectors, shift, block_size, max_subspace, target window |
| `ThermalConfig`          | `thermal`              | TPQ / FTLM / LTLM / Hybrid knobs, T grid, samples |
| `ObservableConfig`       | `observable`           | TPQ flags, ω/τ grids, operator file lists |
| `DynamicalResponseConfig`| `dynamical`            | DSSF: ω grid, samples, krylov, broadening, GPU flag, operator-construction strings |
| `StaticResponseConfig`   | `static_resp`          | SSSF: T grid, samples, krylov, GPU flag, operator-construction strings |
| `SystemConfig`           | `system`               | num_sites, spin_length, sublattice, file names, fixed-Sz, n_up |
| `ArpackConfig`           | `arpack`               | ARPACK_ADVANCED tuning |
| `WorkflowConfig`         | `workflow`             | which workflows to run, sector filters, output dir, basis cache dir, thresholds |

Top-level: `EDConfig::method` of type `DiagonalizationMethod` (see
`include/ed/core/ed_types.h`).

`fromCommandLine` is a long `if/else` chain in `ed_config.cpp`. Notable
behaviours:

- `--symm` / `--symmetrized` / `--streaming-symmetry` all set the same
  `workflow.run_symm_auto = true` (the latter two are deprecated aliases).
- `--use-gpu` sets **both** `dynamical.use_gpu` and `static_resp.use_gpu`;
  `--dyn-use-gpu` and `--static-use-gpu` set only one.
- `--eigenvalues=FULL` is sentinelled as `-1`; the dispatcher later
  translates it to `2^num_sites`.
- After the parse, `autoDetectNumSites()` reads `positions.dat` and infers
  `num_sites` from `max_site_id + 1`.
- A late "auto-skip-ED" rule sets `skip_ed = true` if **only** response
  workflows are requested without a diagonalization workflow, so the
  user is not forced to repeat ED for every DSSF run.

`validate()` checks dimension bounds, positivity, and per-workflow
parameter consistency. `merge()` performs a pragmatic per-field override
where the CLI takes precedence over file values; `save()` round-trips a
small reproducible subset (mostly `[Diagonalization]`, `[System]`, and
core `[Workflow]` flags).

### 3.4 `EDConfigAdapter`

`include/ed/core/ed_config_adapter.h` provides
`ed_adapter::toEDParameters(EDConfig)` and
`ed_adapter::fromEDParameters(...)`. The forward direction copies all
diagonalization/thermal/observable/system/ARPACK fields plus
`workflow.output_dir` and `selected_sectors`; it intentionally **does
not** carry the response or workflow flags into `EDParameters`, because
those drive the CLI/workflow layer rather than the core ED kernel.

### 3.5 Supporting headers

- `ed_types.h` — `enum class DiagonalizationMethod` (the canonical list
  of every supported solver); `using` alias for global scope.
- `ed_logging.h` — `ed_log` namespace: `header / subheader / info /
  progress / success / warning / error / debug / Timer`. Verbosity is
  global and switchable.
- `thermal_types.h` — `ThermodynamicData` and `FTLMResults` data PODs.
- `blas_lapack_wrapper.h` — picks MKL / AOCL / generic CBLAS/LAPACKE at
  configure time and exposes a uniform set of declarations.
- `system_utils.h` — file-hash-based symmetry-cache validation,
  MPI-safe `create_directory_*`, and `generate_automorphisms` (which
  shells out to `python/edlib/automorphism_finder.py`).

---

## 4. `ed_wrapper.h` — the dispatch hub

`include/ed/core/ed_wrapper.h` is a 4 466-line header that ties the
operator layer to the solver layer.

### 4.1 Result/parameter PODs

- `EDResults` — `eigenvalues`, optional `eigenvectors` (full-space
  complex columns), per-method metadata.
- `EDParameters` — every field a CPU/GPU solver might need
  (matches the 1:1 schema used by `ed_config_adapter`).

### 4.2 Method classification

`ed_internal` provides metaprogramming-style helpers:

- `is_gpu_method`, `is_deprecated_fixed_sz_method`,
  `normalize_method`, `normalize_method_and_fixed_sz`,
  `supports_fixed_sz` — handle the legacy `*_FIXED_SZ` enums by
  rewriting them to the base method plus `use_fixed_sz=true`, and
  enumerate which GPU methods support fixed-Sz (currently
  `LANCZOS_GPU`, `BLOCK_LANCZOS_GPU`, `FTLM_GPU`, `DAVIDSON_GPU`,
  `LOBPCG_GPU`, `KRYLOV_SCHUR_GPU`, `BLOCK_KRYLOV_SCHUR_GPU`,
  `mTPQ_GPU`, `cTPQ_GPU`, `FULL_GPU`).
- `is_tpq_method`, `is_ftlm_method`, `requires_ground_state_sector`,
  `requires_sector_combination` — used by the symmetry-projected paths
  to decide whether to combine sectors before persisting results.

### 4.3 The `exact_diagonalization_core` switch

`exact_diagonalization_core(H, dim, method, params)` is the central
**`switch (method)`** for non-GPU paths. The mapping from enum value to
underlying solver call is:

| `DiagonalizationMethod`           | Calls                                                       |
|-----------------------------------|-------------------------------------------------------------|
| `FULL`                            | `full_diagonalization`                                      |
| `SCALAPACK`, `SCALAPACK_MIXED`    | `scalapack_diagonalization` (or fallback `full_diagonalization`) |
| `LANCZOS`                         | `lanczos` (or `arpack_ground_state` if `ED_USE_ARPACK_DEFAULT=1`) |
| `LANCZOS_SELECTIVE`               | `lanczos_selective_reorth`                                  |
| `LANCZOS_NO_ORTHO`                | `lanczos_no_ortho`                                          |
| `SHIFT_INVERT`                    | `shift_invert_lanczos`                                      |
| `SHIFT_INVERT_ROBUST`             | `shift_invert_lanczos` (with stderr notice)                 |
| `DAVIDSON`                        | `davidson_method`                                           |
| `LOBPCG`                          | `lobpcg_diagonalization`                                    |
| `KRYLOV_SCHUR`                    | `krylov_schur`                                              |
| `BLOCK_KRYLOV_SCHUR`              | `block_krylov_schur`                                        |
| `IMPLICIT_RESTART_LANCZOS`        | `implicitly_restarted_lanczos`                              |
| `THICK_RESTART_LANCZOS`           | `thick_restart_lanczos`                                     |
| `OSS`                             | `optimal_spectrum_solver`                                   |
| `mTPQ`                            | `microcanonical_tpq`                                        |
| `cTPQ`                            | `canonical_tpq`                                             |
| `BLOCK_LANCZOS`                   | `block_lanczos`                                             |
| `CHEBYSHEV_FILTERED`              | `chebyshev_filtered_lanczos`                                |
| `ARPACK_SM` / `ARPACK_LM`         | `arpack_ground_state` / `arpack_largest`                    |
| `ARPACK_SHIFT_INVERT`             | `arpack_shift_invert`                                       |
| `ARPACK_ADVANCED`                 | `arpack_eigs_advanced`                                      |
| `FTLM`                            | `finite_temperature_lanczos` + `save_ftlm_results`          |
| `LTLM`                            | `low_temperature_lanczos` + `save_ltlm_results`             |
| `HYBRID`                          | `hybrid_thermal_method` + `save_hybrid_thermal_results`     |
| `mTPQ_MPI`, GPU enums             | Throw / no-op — must use the file-based or symmetry-projected entry points |

### 4.4 File-based and projected entry points

- `exact_diagonalization_from_files`, `exact_diagonalization_from_directory`
  — load `Trans.dat` / `InterAll.dat` (and optionally three-body and
  counter-term files), apply normalization, and dispatch into the core
  switch or into the GPU/CSR branch.
- `exact_diagonalization_fixed_sz` — same with `FixedSzOperator`.
- `exact_diagonalization_all_sz_sectors` / `_gpu` — block by total `S^z`
  (`--full-sz-split`).
- `exact_diagonalization_from_directory_symmetrized`,
  `exact_diagonalization_fixed_sz_symmetrized` — older
  pre-streaming symmetrized variants kept for completeness.

### 4.5 Streaming/Chunked wrappers

- `ed_wrapper_streaming.h` — implements the canonical symmetry path used
  by `--symm`. Calls `generate_automorphisms` (Python helper if needed),
  builds the streaming sectors, optionally consults a basis cache, and
  for each sector dispatches to GPU (`is_gpu_method` ⇒
  `dispatchGPUSymmetrizedSector` ⇒ `GPUEDWrapper::createGPUSymmetrizedOperator`)
  or CPU (`hamiltonian.applySymmetrized` ⇒ `exact_diagonalization_core`).
- `ed_wrapper_chunked.h` — orchestrates the disk-based chunked builder
  for very large clusters.

---

## 5. CPU solvers

All CPU solvers obey one shared contract: the Hamiltonian arrives as
`std::function<void(const Complex* in, Complex* out, int n)>` (matrix-free
SpMV). Vectors are contiguous `std::vector<std::complex<double>>`; matrices
are LAPACK column-major. OpenMP appears in the hottest loops of
`lanczos.cpp` (Ritz reconstruction) and FTLM.

### 5.1 Lanczos family — `include/ed/solvers/lanczos.h`

Public entry points and their roles:

- `lanczos` — default driver. Uses a **ring buffer** of recent basis
  vectors with **local reorthogonalization** against the three most
  recent (threshold `√ε`); basis is held in either an in-RAM buffer or
  on-disk `.dat` files; the tridiagonal is solved by
  `solve_tridiagonal_matrix`. Note: the comment mentions
  Parlett–Simon and there is an `omega` array allocated, but the
  active reorth is the local-ring strategy.
- `lanczos_no_ortho` — pure three-term recurrence; periodic tridiagonal
  check for convergence.
- `lanczos_selective_reorth` — threshold-based reorth combined with
  periodic full reorth, with a 5-vector ring of recent indices.
- `block_lanczos` — block variant for degeneracies.
- `chebyshev_filtered_lanczos` — Chebyshev pre-filter to bias the
  Krylov build into a target spectral window.
- `shift_invert_lanczos` — runs Lanczos on \((H-\sigma I)^{-1}\) by
  solving \((H-\sigma I)\,w = v_j\) every step with PCG.
- `full_diagonalization` — for moderate `N`, builds the dense matrix
  via `N` SpMVs, then `zheev`/`zheevr` (subset).
- `krylov_schur`, `block_krylov_schur`, `implicitly_restarted_lanczos`,
  `thick_restart_lanczos` — restarted Krylov methods.
- `estimate_eigenvalue_count` — stochastic Chebyshev expansion (Jackson
  damped) for eigenvalue counts in an interval.
- `optimal_spectrum_solver` — heuristic dispatcher across the above.

Helpers exposed in the same header:

- `generateRandomVector` / `generateGaussianRandomVector` /
  `generateOrthogonalVector` — start vectors. `ED_LANCZOS_COMPLEX_SEED`
  controls real vs complex seeding (real keeps a real Krylov when `H`
  is real, which is faster).
- `refine_eigenvector_with_cg` — CG refinement of one Ritz vector.
- `refine_degenerate_eigenvectors` — small projected `zheev` cleanup.
- `read_basis_vector` / `write_basis_vector` — basis I/O (delegates to
  the in-RAM buffer or `temp_dir/basis_<i>.dat`).
- `diagonalize_tridiagonal_ritz` — `dstevd` on the tridiagonal,
  returning Ritz values, weights `|e_1^{(i)}|²`, and optional `m × m`
  eigenvectors.
- `solve_tridiagonal_matrix` — chooses `dstemr` (subset) vs `dstevd`,
  reconstructs full Ritz vectors with Kahan summation under OpenMP, and
  writes HDF5.

Performance knobs:

- `BasisBufferScope` registers an in-memory storage for the duration of
  one Lanczos run; see §10.2 for the contract.
- `ED_LANCZOS_VERBOSE` toggles per-iteration logging.
- `ED_LANCZOS_DISK=1` forces the legacy `.dat` per-vector file storage
  even when a buffer is registered.

### 5.2 FTLM — `include/ed/solvers/ftlm.h`

The Finite-Temperature Lanczos Method approximates `Z`, `<E>`, `C_v`,
`S` from a small number of stochastic short Lanczos runs. The code
extends this to two-operator dynamical and static correlations, T=0
DSSF via continued fraction, and multi-temperature reuse where one
Krylov build serves all `T`.

Key entry points (selected):

- `finite_temperature_lanczos` — main driver; produces FTLM thermo.
- `compute_ftlm_thermodynamics`, `average_ftlm_samples`,
  `combine_ftlm_sector_results` — sample-level math and merging.
- `compute_dynamical_correlation` / `compute_dynamical_correlation_state`
  — two-operator `S(ω)` over random or fixed states.
- `compute_dynamical_correlation_state_cf` — continued-fraction
  spectral function for `O₁ = O₂` (memory-efficient).
- `continued_fraction_spectral_function` — close-form `S(ω)` from
  `(α, β)` chains.
- `compute_lanczos_spectral_data` — stash temperature-independent
  Lanczos data for reuse.
- `compute_spectral_function_from_lanczos_data` — assemble
  Boltzmann-weighted `S(ω, T)` from prior data.
- `compute_dynamical_correlation_state_multi_temperature` and
  `compute_dynamical_correlation_multi_sample_multi_temperature` — the
  optimized "one Lanczos, many T" paths used by the workflows.
- `compute_thermal_expectation_value`, `compute_static_response`,
  `save_static_response_results` — SSSF.
- `compute_ground_state_dssf`, `compute_ground_state_cross_correlation`,
  `load_ground_state_from_file` — T=0 DSSF.

Several functions in `ftlm.cpp` were recently optimized to replace
nested `O(m²)` C++ loops with two `cblas_dgemv` calls (see the BLAS-2
paths in `compute_dynamical_correlation`, `compute_dynamical_correlation_state`,
and `compute_ground_state_cross_correlation`).

### 5.3 LTLM — `include/ed/solvers/ltlm.h`

Low-Temperature Lanczos Method: takes a Lanczos chain rooted at the
ground state. Provides `find_ground_state_lanczos`,
`build_excitation_spectrum`, `compute_ltlm_thermodynamics`,
`low_temperature_lanczos`. Used both standalone and as the low-T half
of the hybrid method.

### 5.4 Hybrid LTLM/FTLM — `include/ed/solvers/hybrid_thermal.h`

Single T-grid driver that uses LTLM below a crossover \(T_c\) and FTLM
above. `estimate_optimal_crossover` does a small Lanczos to estimate
the gap and bounds \(T_c\) into `[0.01, 10]`. `auto_crossover`
controls whether the heuristic kicks in.

### 5.5 ARPACK — `include/ed/solvers/arpack.h`

Wraps ARPACK-NG (`znaupd_`/`zneupd_`) with shift-invert support via an
inner PCG/MINRES-CR/CGNR for `(H - σI)⁻¹`. Public:

- `arpack_eigs(..., which="SM")` — generic.
- `arpack_ground_state` — forces `which="SR"` (smallest real, the right
  choice for Hermitian ground states).
- `arpack_largest` — `which="LR"`.
- `arpack_shift_invert`, `arpack_shift_invert_prec`.
- `arpack_eigs_advanced` — adaptive `ncv`, two-phase tolerance,
  optional auto shift-invert fallback (parameterised by
  `ArpackAdvancedOptions`).

Eigenvectors are sorted by real part and **phase-canonicalised** so the
largest-magnitude component is positive real.

### 5.6 CG / Davidson / LOBPCG — `include/ed/solvers/CG.h`

- `davidson_method` — random multi-vector start, `Eigen::SelfAdjointEigenSolver`
  on the projected `V†HV`, residual-driven corrections, restart at
  `max_subspace`.
- `lobpcg_method`, `lobpcg_diagonalization` — block Rayleigh + residual,
  with an optional toy diagonal preconditioner (placeholder).
- (`bicg_eigenvalues` exists in `CG.cpp` but is not part of the public
  header surface.)

### 5.7 Dynamics — `include/ed/solvers/dynamics.h`

Real-time and imaginary-time evolution and time-domain correlations:

- `time_evolve_taylor`, `imaginary_time_evolve_taylor` — adaptive Taylor.
- `time_evolve_krylov` — short Lanczos with selective reorth on \(\beta_j\) shrinkage.
- `time_evolve_chebyshev` — rescaled `H ∈ [-1, 1]`, Chebyshev series.
- `time_evolve_rk4`, `time_evolve_adaptive`.
- `create_time_evolution_operator` — factory for fixed-step `U(Δt)`.
- `compute_time_correlation`, `compute_multiple_time_correlations`,
  `compute_time_correlations_with_U_t`, `compute_time_correlations_incremental`.
- `compute_spectral_function` — broadened FT.
- `compute_operator_dynamics` — high-level driver.

### 5.8 Observables — `include/ed/solvers/observables.h`

`calculate_thermodynamics_from_spectrum(eigenvalues, T_grid)` produces
`ThermodynamicData` (energy, heat capacity, entropy, free energy) using
log-sum-exp on a log-spaced T grid. Special-cases `T → 0` using the
ground-state degeneracy.

### 5.9 ScaLAPACK — `include/ed/solvers/scalapack_diag.h`

When `WITH_MPI=ON`, fills a distributed Hermitian matrix from `H` and
calls `pcheevd`/`pzheevd`. Provides
`scalapack_diagonalization`, `scalapack_diagonalization_outofcore`
(future slicing entry), grid lifecycle helpers
`initialize_scalapack_grid` / `finalize_scalapack_grid`,
`get_optimal_block_size`, `estimate_distributed_memory`,
`is_scalapack_available`, `is_elpa_available`, and
mixed-precision helpers including `refine_eigenpair`.

Without MPI, `scalapack_diagonalization` returns an empty result and
prints an error.

### 5.10 TPQ — `include/ed/solvers/TPQ.h`

Implements **microcanonical** TPQ (random initial state + dynamics
along a "quenching" schedule) and **canonical** TPQ (imaginary-time
Taylor expansion of `e^{-Δβ H}`). Includes site-spin operator factories
(`createSzOperators`, `createSxOperators`, `createSyOperators`),
fluctuation writers, file I/O, MPI parallelism over independent samples
(`microcanonical_tpq` and friends), and converters from raw TPQ runs to
the unified thermo format (`convert_tpq_to_unified_thermo*`).

---

## 6. GPU solvers

The GPU layer mirrors the CPU layer through a stable façade
`GPUEDWrapper` (`include/ed/gpu/gpu_ed_wrapper.h`,
`src/solvers/gpu/gpu_ed_wrapper.cu`) and a set of CUDA classes for the
operator and each solver family. All GPU code is gated by
`WITH_CUDA=ON`; without it, the wrapper degrades to
no-ops/`nullptr`/`false` returns.

### 6.1 GPU operators

#### `GPUOperator` (`gpu_operator.cuh`, `gpu_operator.cu`)

The flagship matrix-free GPU Hamiltonian. Stores interaction terms in
two layouts:

- **Legacy unified array** of `GPUTransformData` with `op_type`, sites,
  `is_two_body` flag, and a `cuDoubleComplex` coefficient.
- **Separated SoA (v2)** built by `separateTransformsByType()` and
  uploaded by `copySeparatedTransformsToDevice()`:
  `GPUDiagonalOneBody`, `GPUOffDiagonalOneBody`, `GPUDiagonalTwoBody`,
  `GPUMixedTwoBody`, `GPUOffDiagonalTwoBody`. This decomposition removes
  branches inside the SpMV kernels.
- Optional **assembled CSR** built on-device by `buildCsrOnDevice(N)`,
  which enumerates all non-zero couplings, sorts COO, merges duplicates,
  builds CSR, and registers `cusparseSpMat` + workspace.

`matVecGPU` selects one of four kernel pathways via heuristics:

1. `CUSPARSE_CSR` (`cusparseSpMV`, `CUSPARSE_SPMV_CSR_ALG2`) when
   `N ≥ ED_GPU_CUSPARSE_MIN_DIM` (default 32 768) and CSR built OK.
2. `WARP_REDUCTION` for high-`T`/high-`N` regimes (one warp per output row).
3. `BRANCH_FREE_SCATTER` for moderate `T`.
4. `SHARED_MEMORY` for small `T`.

`ED_GPU_TIMING=1` enables per-`matVecGPU` event timing.

#### `GPUFixedSzOperator` (`gpu_fixed_sz_operator.cu`)

Restricts to fixed-`S^z` blocks. Generates the basis on-device with
`generateFixedSzBasisKernel` (Gosper's hack), then dispatches a
state-parallel or transform-parallel matvec, with **binary search**
`lookupState` for off-diagonal targets. The operator overrides
`matVec` / `matVecGPU` and disables async matvec (shared basis +
atomics).

#### `GPUSymmetrizedOperator` (`gpu_symmetrized_operator.cu`)

Implements the symmetry-projected matvec on GPU. Stores **CSR-style
orbits** (`d_orbit_elements_`, `d_orbit_coefficients_`,
`d_orbit_offsets_`, `d_orbit_norms_`) and an **open-addressing hash
table** (`d_hash_table_` of `GPUHashEntry`) that maps each computational
state in any orbit back to its symmetrized basis index along with a
**precomputed projection factor** `conj(α_s) · |G|⁻¹ / norm_j`. The
matvec kernel is launched as a 2D grid over `(orbit_element, transform)`.

### 6.2 Eigensolvers

Declared in `include/ed/gpu/gpu_lanczos.cuh`:

| Class                      | Behaviour                                                         |
|----------------------------|-------------------------------------------------------------------|
| `GPULanczos`               | Lanczos with partial reorth, GPU matvec + cuBLAS, host LAPACK on tridiagonal |
| `GPUBlockLanczos`          | Block Lanczos with QR via cuSOLVER                                |
| `GPUKrylovSchur`           | Restarted Arnoldi via cuBLAS + cuSOLVER                           |
| `GPUBlockKrylovSchur`      | Block-restarted Arnoldi                                            |
| `gpuFullDiagonalization`   | Builds dense `N × N` on GPU via `N` matvecs, then `cusolverDnZheevd` |
| `GPUIterativeSolver`       | Davidson and LOBPCG (latter solves the small generalized eigenproblem on the host via Eigen, see `lobpcg_eigen_solve.cpp` to dodge nvcc + Eigen ABI issues) |

Each is wrapped by a `runGPU<Method>` entry on `GPUEDWrapper`, which
optionally writes results through `HDF5IO::saveDiagonalizationResults`
under tags such as `"GPU_LANCZOS"`, `"GPU_BLOCK_LANCZOS"`, etc.

### 6.3 Thermal & dynamical GPU paths

- **`GPUFTLMSolver`** (`gpu_ftlm.cuh`, `gpu_ftlm.cu`) — full FTLM with
  cuBLAS, cuRAND, optional cuSOLVER tridiagonal. The
  `computeDynamicalCorrelationMultiTemp` path matches the CPU
  multi-T trick.
- **`GPUTPQSolver`** (`gpu_tpq.cuh`, `gpu_tpq.cu`) — mTPQ and cTPQ;
  uses dual CUDA streams for compute/transfer overlap.
- **`GPUDynamicsSolver`** (`gpu_dynamics.cuh`, `gpu_dynamics.cu`) —
  Krylov / Taylor time evolution and `<O₂(t) O₁(0)>` style correlations.
  Note: the `O₁`/`O₂` callbacks are `std::function` on CPU for
  flexibility; the Hamiltonian step uses `GPUOperator`.

### 6.4 The `GPUEDWrapper` façade

`GPUEDWrapper` is a static class. Public API summary
(see `include/ed/gpu/gpu_ed_wrapper.h` for full signatures):

- **Lifecycle / introspection.** `isGPUAvailable`, `printGPUInfo`,
  `estimateGPUMemory`, `shouldUseGPU`.
- **Operator construction.** `createGPUOperatorDirect` (interaction
  tuples), `createGPUOperatorFromFiles` (Trans/InterAll),
  `createGPUFixedSzOperatorDirect`, `createGPUSymmetrizedOperator`
  (orbit + files), `destroyGPUOperator`. **Note:** the
  `createGPUOperatorFromCPU` method is currently a stub that returns
  `false`; production code uses
  `convertOperatorToGPU` from `gpu_operator_conversion.cpp` directly.
- **Eigensolvers.** `runGPULanczos[FixedSz]`,
  `runGPUBlockLanczos[FixedSz]`, `runGPUDavidson[FixedSz]`,
  `runGPUKrylovSchur[FixedSz]`, `runGPUBlockKrylovSchur[FixedSz]`,
  `runGPULOBPCG[FixedSz]`, `runGPUFullDiag`.
- **TPQ.** `runGPUMicrocanonicalTPQ[FixedSz]`,
  `runGPUCanonicalTPQ[FixedSz]`.
- **FTLM.** `runGPUFTLM[FixedSz]`.
- **Dynamical.** `runGPUDynamicalResponse`,
  `runGPUDynamicalResponseThermal`, `runGPUDynamicalCorrelation`,
  `runGPUDynamicalCorrelationState`,
  `runGPUDynamicalCorrelationStateCF`,
  `runGPUDynamicalCorrelationMultiTemp` (the workhorse for
  `compute_dynamical_response_workflow`'s GPU path).
- **Static.** `runGPUThermalExpectation`, `runGPUStaticCorrelation`.

### 6.5 Kernels (`gpu_kernels.cu`)

- `atomicAddDouble` — native on `sm ≥ 6`, CAS loop fallback.
- `matVecKernelOptimized` — state-parallel SpMV with shared cache and
  `__ldg` loads, atomics on `y[new_state]`.
- `matVecFixedSzTransformParallel`, `matVecFixedSzKernelOptimized` and
  branch-free variants per separated-type.
- `matVecWarpReductionFused` — gather-style (no output atomics).
- `matVecSymmetrized` — 2D orbit × transform grid.

### 6.6 Infrastructure

- `cuda_raii.cuh` — RAII wrappers (cuBLAS, cuSPARSE, cuRAND, streams,
  events, `GpuMemory<T>`).
- `mixed_precision.cuh` — TF32 / `cublasSetMathMode`,
  `ScopedMixedPrecision` (opt-in; not on the default matvec path).
- `kernel_config.h` — `BLOCK_SIZE = 256`, `MAX_SITES = 32`,
  `NNZ_PER_STATE_ESTIMATE`, `CUDA_CHECK` macro.
- `bit_operations.cuh` — `GPUBitOps` (popcount, `apply_sp/sm/sz/sx/sy`,
  `flip_bit`, permutation helpers).

### 6.7 BFG GPU app

`src/apps/compute_bfg_order_parameters_gpu.cu` is **not** wired through
`GPUEDWrapper`. It is a standalone post-processing tool that loads
saved wavefunctions and runs custom kernels for `S+S-`, XY bonds,
bond-bond correlations, and bowtie ring expectations using
`atomicAdd` reductions over Hilbert basis indices.

---

## 7. Symmetry / projection pipeline

### 7.1 Mathematical setup

The toolkit targets **abelian site-permutation symmetries**. A spin
configuration is a length-`N` bit string; a group element `g ∈ G` is a
site permutation realised by `applyPermutation`. For each `s` the orbit
is `O(s) = { g · s : g ∈ G }`, and the canonical orbit representative
is `rep(s) = min_g g · s` (lexicographic minimum as an unsigned
integer). A state is a representative iff `s == rep(s)`.

A symmetry sector is labelled by **integer quantum numbers**
`(q_0, …, q_{K-1})` for a chosen set of commuting generators
`g_0, …, g_{K-1}` of orders `o_0, …, o_{K-1}`. For each `g`, the **power
representation** decomposes `g = g_0^{p_0} ⋯ g_{K-1}^{p_{K-1}}`, and the
1D irrep character is

\[
\chi(g) = \prod_k e^{2\pi i q_k p_k / o_k}.
\]

The projected (unnormalized) state from a representative `|s_0⟩` is

\[
P|s_0⟩ = \sum_{g \in G} \chi(g)^* |g · s_0⟩,
\]

which is materialised by `computeOrbitData` as a sparse map (key =
image state, value = accumulated `conj(χ(g))`). Its squared norm,
divided by `|G|`, gives `‖P|s_0⟩‖²`.

For fixed-`S^z`, each site permutation preserves the up-spin count, so
the same projector applies on top of a fixed-`S^z` enumeration.

The **`filterInvalidSectors`** step prunes `(q_0, …, q_{K-1})` tuples
that do not correspond to a valid character (because of relations
between generators) — i.e. "phantom sectors" implied by the naive
product `∏ o_k`.

### 7.2 `ed::sym` DSL — `include/ed/symmetry/group.h`

A small permutation toolkit:

- `using Permutation = std::vector<int>` (composition `(a∘b)[i]=a[b[i]]`).
- `identity`, `validate`, `compose`, `power`, `order`.
- Constructors: `translation`, `reflection_1d`, `site_swap`.
- `generate_group` — BFS closure to a sorted element list.
- `group_from_generators(n_sites, generators, sector_qn={})` — builds a
  full `SymmetryGroupInfo` (clique, power representation, sectors,
  phantom filtering) so users do not need the JSON pipeline for
  programmatic groups.
- Convenience: `translation_group_1d(n_sites)`,
  `translation_group_with_reflection_1d(n_sites)`.

JSON-based loading lives elsewhere: `SymmetryGroupInfo::loadFromDirectory`
(`include/ed/core/construct_ham.h`) parses
`automorphism_results/{automorphisms,max_clique,minimal_generators,sector_metadata}.json`.

### 7.3 Streaming sector generation — `include/ed/core/streaming_symmetry.h`

Two-pass, matrix-free enumeration:

1. **Pass 1.** For each basis state, compute `rep(s)` over `max_clique`.
   Keep only states with `s == rep(s)`. Full-Hilbert version uses
   OpenMP `static` parallelism; thread-local rep lists are merged and
   sorted for determinism.
2. **Pass 2.** For each irrep (sector) and each orbit rep,
   `computeOrbitData` produces a sparse `SymBasisState` (`orbit_elements`,
   `orbit_coefficients`, `norm`).

A per-sector **lookup table** `state_to_sector_basis_[si][s] = j` lets
`H` applied to a bit string project back into the symmetrized index in
O(1) plus O(log|orbit|) for `findCoeff`.

`applySymmetrized` zeroes `out`, loops over `j` and orbit elements,
applies all Hamiltonian terms (including three-body), and projects back
through the lookup. The fixed-`S^z` variant
`FixedSzStreamingSymmetryOperator` uses Gosper-iteration over the
fixed-`n_up` basis with sector-aware permutation lookups.

The same module manages an HDF5 **basis cache** under `basis_cache_dir`
(file pattern `orbit_basis_N{n_sites}_fullspace.h5`). Reads happen
freely; writes only happen in `--precompute-basis-only` mode (avoiding
clobbering shared caches).

### 7.4 Chunked and disk-based variants

- `include/ed/core/chunked_symmetry_builder.h` — two-pass orbit
  enumeration that walks Hilbert in chunks, persists per-sector data
  under `…/sector_cache_chunked/`, and supports a fixed-`S^z` variant
  and a fully disk-based variant `DiskBasedChunkedSymmetryBuilder` that
  k-way merges sorted rep chunks before sector assignment.
- `include/ed/core/disk_streaming_symmetry.h` — `DiskStreamingSymmetryOperator`
  serialises one sector at a time as binary files (sector header, QN,
  phase factors, then `SymBasisState` records). The matvec keeps a
  reverse lookup map for the loaded sector only. The wrapper
  `exact_diagonalization_disk_streaming` is **CPU-only**.

### 7.5 Symmetry-sector I/O — `hdf5_symmetry_io.h`

`HDF5SymmetryIO` writes a **separate** `symmetry_data.h5` (truncate on
recreate) with `/metadata/{num_sectors, sector_dimensions}`,
`/basis/vector_<i>` (compound `{index, real, imag}`), and
`/blocks/block_<i>` (compound `{row, col, real, imag}`) for the older
block-matrix workflows. The streaming/CSR caches and the per-run
`ed_results.h5` use the regular `HDF5IO` path and group conventions
described in §10.

### 7.6 GPU dispatch in the symmetry path

`include/ed/core/ed_wrapper_streaming.h` extracts orbit CSR data via
`extractOrbitData` and calls
`GPUEDWrapper::createGPUSymmetrizedOperator` for `LANCZOS_GPU*`,
`BLOCK_LANCZOS_GPU*`, `DAVIDSON_GPU`, `KRYLOV_SCHUR*`,
`BLOCK_KRYLOV_SCHUR*`, and `FULL_GPU`. The CPU fallback wraps
`hamiltonian.applySymmetrized` in `exact_diagonalization_core`.

### 7.7 The Python automorphism pipeline

`python/edlib/automorphism_finder.py` reads `Trans.dat` and
`InterAll.dat`, builds a Weisfeiler–Lehman colored graph, calls
`pynauty.autgrp` for graph automorphisms, expands generators, projects
back to site permutations, and finally
`filter_hamiltonian_automorphisms` keeps the ones that map every
interaction term to an existing term. Outputs land under
`{data_dir}/automorphism_results/`:

- `automorphisms.json`, `vertex_mapping.json`, `max_clique.json`,
  `minimal_generators.json`, `sector_metadata.json`, optionally
  `.translation_only` marker.

`AutomorphismCliqueAnalyzer.find_maximum_clique` (NetworkX) selects a
maximum mutually commuting set; `--translation_only` skips the clique
search and uses translations from `positions.dat` +
`*_lattice_parameters.dat`.

The lattice-specific `helper_*.py` modules (`helper_cluster*.py`,
`helper_honeycomb*.py`, `helper_kagome_bfg*.py`, `helper_pyrochlore*.py`,
etc.) generate `positions.dat`, `Trans.dat`, `InterAll.dat`, and
`*_lattice_parameters.dat` — the inputs the C++ side ingests.

---

## 8. DSSF / SSSF engine and workflows

### 8.1 The engine seam

`include/ed/dssf/dssf_engine.h` defines the canonical DSSF/SSSF API:

- `enum class DSSFMethod { DYNAMICAL_THERMAL = 0, STATIC_THERMAL = 1,
  GROUND_STATE_DSSF = 2, SINGLE_EXPECTATION = 3 }` with persistent
  numeric values.
- `to_string` / `method_from_string` (defined in
  `src/dssf/dssf_method.cpp` so I/O and Python bindings can link them
  without pulling in workflows).
- `DSSFRequest { OperatorSpec operators; DSSFMethod method;
  std::string output_dir; const EDConfig* config; }`. The transitional
  `config` pointer is required today; it carries the same knobs the
  legacy CLI uses.
- `DSSFResult { method; num_tasks_attempted; output_dir; }`. Spectra
  are not returned in memory — they are written to HDF5 by the
  workflows.

`ed::dssf::run(request)` (`src/cli/dssf_engine.cpp`) validates the
config pointer, computes a heuristic `num_tasks_attempted`, and
dispatches:

- `DYNAMICAL_THERMAL` → `compute_dynamical_response_workflow(*config)`
- `STATIC_THERMAL` and `SINGLE_EXPECTATION` →
  `compute_static_response_workflow(*config)`
- `GROUND_STATE_DSSF` → `compute_ground_state_dssf_workflow(*config)`

### 8.2 `OperatorSpec` and observable construction

`include/ed/dssf/operator_spec.h` defines the operator schema shared by
the CLI and Python bindings:

| Field                     | Meaning                                                                         |
|---------------------------|---------------------------------------------------------------------------------|
| `operator_type`           | `"sum"`, `"transverse"`, `"sublattice"`, `"experimental"`, `"transverse_experimental"` |
| `basis`                   | `"ladder"` (S+, S-, Sz) or `"xyz"` (Sx, Sy, Sz)                                 |
| `spin_combinations`       | Pairs of spin indices (0/1/2)                                                   |
| `momentum_points`         | List of 3-vectors `Q` (cross-product with spin combinations)                    |
| `polarization`            | Unit 3-vector for the SF/NSF axes                                               |
| `theta`                   | Tilt angle for "experimental" types                                             |
| `unit_cell_size`          | Sublattice count for sublattice operators                                       |
| `num_sites`, `spin_length`| System size and spin                                                            |
| `use_fixed_sz`, `n_up`    | Fixed-`S^z` or full Hilbert                                                     |
| `positions_file`          | Passed to every operator constructor                                            |
| `single_obs_only`         | Build only `obs_1`; skip ladder swap; flat names like `"Sz"` instead of `"SzSz"`|
| `sublattice_filter`       | Optional single sublattice pair                                                 |

`build_observable_pairs(spec) → ObservablePairs`
(`src/dssf/operator_spec.cpp`):

- Validates non-empty spin/momentum lists, 3-vector polarization,
  `num_sites > 0`.
- For each `Q` and spin pair, constructs `Operator` wrappers
  (`SumOperator` / `FixedSzSumOperator` / transverse / sublattice /
  experimental / transverse-experimental) with consistent names.
- For ladder pair products (excluding `single_obs_only`), swaps spin
  index `0 ↔ 1` so `<S⁻†S⁺>` matches the legacy HDF5 convention.
- `compute_transverse_bases(Q, pol)` returns `(e1, e2)` with
  `e1 = polarization`, `e2 = normalize(Q × pol)` (with degenerate-Q
  fallback pinned by tests).

`construct_operators_from_config` (`src/cli/workflows.cpp`) is the
adapter from the scalar CLI fields into a populated `OperatorSpec`.

### 8.3 `compute_dynamical_response_workflow`

Workflow steps:

1. **Banner** prints a one-line GPU status:
   - `"GPU: requested but disabled (Fixed-Sz GPU path not implemented; falling back to CPU)"`,
   - `"GPU: enabled (multi-temperature path; single-T and 1-sample tasks fall back to CPU)"`,
   - or `"GPU: requested but unavailable (build has no CUDA support; using CPU)"`.
2. Loads the Hamiltonian from `Trans.dat`, `InterAll.dat`, optionally a
   three-body file. Computes the Hilbert dimension as `2^N` or the
   binomial for fixed-`S^z`.
3. Resolves the **ground-state energy shift**: rank 0 tries
   `ed_results.h5/eigendata/eigenvalues[0]`, then `loadTPQThermodynamics`
   minimum, then runs Lanczos. The result is broadcast.
4. Builds a log-spaced T grid.
5. Constructs operators (config path) or loads `--dyn-operator[/2]`
   files (legacy path).
6. **MPI tasking.**
   - For `num_temp_bins > 1`: one task per operator,
     `is_multi_temp = true`, all ranks run
     `process_operator_all_temps` in lockstep (the underlying CPU
     multi-T routine uses MPI collectives internally, so dynamic
     scheduling is intentionally *not* used here).
   - For `num_temp_bins == 1`: tasks are `(T, op)` pairs, dispatched
     master/worker with `TASK_TAG`/`DONE_TAG`/`STOP_TAG`.
7. **GPU multi-T path** (in `process_operator_all_temps`): when
   `WITH_CUDA` and `use_gpu` and *not* fixed-Sz, calls
   `convertOperatorToGPU` on the Hamiltonian and the two observables,
   then `runGPUDynamicalCorrelationMultiTemp`. Spectral errors are zero
   on the GPU path (kernel does not yet propagate them).
8. **CPU paths.** `num_samples == 1` →
   `compute_dynamical_correlation_state_multi_temperature`; else
   `compute_dynamical_correlation_multi_sample_multi_temperature`. The
   single-T path uses `compute_dynamical_correlation`.
9. Persists with `HDF5IO::saveDynamicalResponseFull` under
   `/dynamical/<op_name>/...` (see §10).

### 8.4 `compute_static_response_workflow`

Same shape as the dynamical workflow but the tasks are operator pairs
only (no T axis at the task level). The GPU branch calls
`runGPUStaticCorrelation` and maps the real part to `expectation`;
susceptibility is currently CPU-only (TODO in source). MPI uses
master/worker scheduling. Persistence goes through
`HDF5IO::saveStaticResponse` at `/correlations/<op_name>/...`. The
legacy file-based path supports a `single_operator_mode` that calls
`compute_thermal_expectation_value`.

### 8.5 `compute_ground_state_dssf_workflow`

CPU-only. Workflow steps:

1. Banner emits a single note if `--use-gpu` was requested
   (no GPU path for T=0 DSSF).
2. Hamiltonian and dimension as above.
3. Constructs `GroundStateDSSFParameters` (ω grid, broadening,
   `krylov_dim`, etc.) from `config.dynamical`.
4. Builds operators via `construct_operators_from_config`.
5. **Ground state**: tries to load eigenvalue + eigenvector from
   `ed_results.h5`; otherwise calls `find_ground_state_lanczos`. Rank 0
   may save the eigenpair back.
6. **MPI** uses static round-robin (`for i in range(rank, n, size)`).
7. Per task, calls `compute_ground_state_cross_correlation`.
8. Saves with `HDF5IO::saveDynamicalResponseFull` under the
   `ground_state_dssf/<op_name>` prefix; `total_samples = 1`,
   `temperature = 0`. A trailing `MPI_Barrier` sync precedes return.

### 8.6 The new `/dssf` schema

`include/ed/dssf/dssf_io.h` introduces a versioned, library-friendly
schema:

- `kSchemaVersion = 1`.
- `Metadata` (`method`, `num_sites`, `spin_length`, optional
  `created_at`).
- `Record` (operator name, T, total_samples, dynamical/static arrays).
- `ensure_metadata` / `write_record` / `read_record` (with strict
  schema-version check).

Production workflows still write through `HDF5IO`'s **legacy** layout
(`/dynamical/<op>/...`, `/correlations/<op>/...`); `dssf_io.h` is the
designated future home for a single `/dssf/...` tree. Readers that need
it today should use `HDF5IO`.

---

## 9. BFG: bond-bilinear, structure factors, order parameters

The BFG ("Bond Field & Geometry") library is a research add-on for
post-processing wavefunctions and TPQ snapshots into bond bilinear
expectations, plaquette / triangle / bowtie observables, dimer
structure factors, spin structure factors, and VBS / nematic / plaquette
order parameters.

### 9.1 `cluster.h` / `cluster.cpp`

`Cluster` carries sites, 2D positions, sublattice assignments,
nearest-neighbour lists and edges, reciprocal/direct lattice vectors,
`k_points`, cell grid metadata, and PBC helpers
(`minimum_image_displacement`, `bond_center_pbc`). `load_cluster(dir)`
populates it from disk and prints diagnostics; missing
`positions.dat` throws.

### 9.2 `correlations.h`

Full Hilbert-space expectations over a complex state vector, OpenMP
where built:

- `compute_smsp_correlations` — `<S⁻_i S⁺_j>` matrix.
- `compute_szsz_correlations` — `<S^z_i S^z_j>` matrix.
- `compute_xy_bond_expectations` — symmetrized XY per NN edge.
- `compute_spsm_bond_expectations` — asymmetric `<S⁺_i S⁻_j>` per edge.
- `compute_szsz_bond_expectations`.
- `compute_heisenberg_bond_expectations` — combines SzSz + XY into
  `<S_i · S_j>`.

Dimer-dimer correlations live in `structure_factor.h`.

### 9.3 `order_parameters.h` / `order_parameters.cpp`

- `compute_nematic_order` — complex-bond nematic
  (`ψ_nem`, `m_nem`, anisotropy).
- `compute_nematic_order_real` — real-bond variant (typical for SzSz).
- `compute_vbs_order` — VBS / dimer SF pipeline using ψ, XY and
  Heisenberg bond expectations; populates a `VBSResult` with
  `S_d_xy_oriented`, `S_d_heis_oriented`, peak `q_max`, `m_vbs_*`,
  `D_mean_*`.
- `compute_plaquette_order` — bowtie / triangle plaquette bundle
  (`PlaquetteResult`).
- `compute_sq_2d_grid` — 2D `q`-grid spin SF from precomputed
  correlation tables.
- `compute_all_order_parameters` — one-shot scalar summary
  `OrderParameterResults` for scan modes.

### 9.4 `structure_factor.h`

- `DimerSFResult` — overlap and expectations for
  `S_D(q) = overlap − |<D(q)>|²`.
- Global `set_memory_efficient_mode(n_states)` /
  `memory_efficient_mode_enabled()` control whether per-thread
  accumulators or atomic accumulation is used (heuristic ≈ 4 GB).
- `compute_dimer_sf_direct`, `compute_heisenberg_sf_direct` — bond
  bilinear `S_D` at a single `q`.
- `apply_dimer_fourier`, `apply_heisenberg_dimer_fourier` — apply
  `D(q)|ψ⟩`.
- `compute_dimer_dimer_correlation`,
  `compute_heisenberg_dimer_dimer_correlation`.

### 9.5 `spin_structure_factor.h`

`StructureFactorResult` (per-`k` `S(q)`, SMSP / SzSz channels,
`q_max`, `m_translation`) and `compute_spin_structure_factor`.

### 9.6 `ring_observables.h`

- `apply_bowtie_fourier` — Fourier bowtie ring operator on `|ψ⟩`.
- `compute_bowtie_resonance` — single-bowtie
  `<S+S−S+S− + h.c.>`.
- `compute_triangle_chiral` — symmetrized 3-spin ring
  `<S+S−S+ + S−S+S−>`.

### 9.7 `topology.h`

`Bowtie` POD plus `find_triangles` / `find_bowties` to enumerate NN
triangles and bowtie plaquettes from the cluster.

### 9.8 `wavefunction_io.h`

`TPQState`, `load_wavefunction` (probes multiple HDF5 paths and complex
layouts for backward compatibility), `load_all_tpq_states`,
`load_tpq_state` (lowest-T snapshot).

### 9.9 `results_io.h`

PODs `NematicResult`, `VBSResult`, `PlaquetteResult`,
`Sq2DGridResult`, `OrderParameterResults` (with `is_valid()` check via
NaN `jpm`), plus `save_results`, `save_temperature_scan_results`,
`save_scan_results` writers.

### 9.10 `cli.h` / `cli.cpp` and `compute_bfg_order_parameters.cpp`

`SingleFileOptions`, `ScanOptions`, `print_usage`,
`process_all_temperatures`, `scan_jpm_directories`, `run_single_file`,
`run_scan`. The driver `compute_bfg_order_parameters.cpp` initialises
MPI, parses argv, dispatches to `run_single_file` or `run_scan`, and
handles errors and timing.

### 9.11 GPU BFG driver

`compute_bfg_order_parameters_gpu.cu` is a standalone CUDA app with
custom kernels for `S+S-`, XY bonds, bowtie resonance, and bond-bond
correlation. Each kernel is a 1D grid over Hilbert basis indices with
`atomicAdd` reduction. The host orchestrates O(N²) `S+S-` launches,
per-edge XY launches, per-bowtie launches, and CPU loops over sites
and `k_points` for the higher-level S(q) and order parameters.

---

## 10. I/O layer (HDF5 schema & basis buffers)

### 10.1 `HDF5IO` and `ed_results.h5`

The canonical results file is `<output_dir>/ed_results.h5`. `HDF5IO`
(declared in `include/ed/core/hdf5_io.h`) writes a tree shaped roughly
as:

| Path pattern | Contents |
|--------------|----------|
| `/eigendata/eigenvalues` | 1D double eigenvalue list (replaced on save) |
| `/eigendata/eigenvector_<k>` | Complex eigenvector |
| `/thermodynamics/temperatures` | Shared T grid (created once) |
| `/thermodynamics/<observable>` | Curve vs T |
| `/correlations/<name>_real`, `_imag` | 2D or 1D real/imag split correlations |
| `/correlations/<operator_name>/...` | Static response: `temperatures`, `expectation`, `expectation_error`, `variance`, `variance_error`, `susceptibility`, `susceptibility_error` |
| `/dynamical/frequencies` | Shared ω grid |
| `/dynamical/<operator_name>` | Legacy 1D spectrum + attrs |
| `/dynamical/<operator_name>/...` | Full dynamical response: `frequencies`, `spectral_real`, `spectral_imag`, `error_real`, `error_imag`; group attrs `total_samples`, `temperature` |
| `/dynamical/time_correlations/<op>_<label>/...` | Time-domain correlations |
| `/ftlm/samples/sample_<i>/eigenvalues` | Per-sample eigenvalues |
| `/ftlm/samples/{thermodynamic,dynamical,dynamical_correlation,static}/sample_<i>/...` | Per-sample observables |
| `/ftlm/averaged/...` | Averaged FTLM curves + errors; hybrid adds `method_indicator` and `crossover_*` attrs |
| `/tpq/samples/sample_<i>/thermodynamics` | 5-column extensible table (`beta`, `energy`, `variance`, `doublon`, `step`) |
| `/tpq/samples/sample_<i>/norm` | 4-column table (`beta`, `norm`, `first_norm`, `step`) |
| `/tpq/samples/sample_<i>/states/beta_<value>` | TPQ state vectors |
| `/tpq/averaged/...` | Averaged TPQ curves + errors |

Note that "ground-state DSSF" is not a separate top-level group: it is
written under `/dynamical/...` with a `ground_state_dssf/` prefix in
the operator name.

`ed::dssf::write_record` (see §8.6) writes a future, versioned `/dssf/...`
tree; current production paths still use the legacy keys above.

#### `createOrOpenFile` and concurrency

`createOrOpenFile(directory, filename)` opens with `H5F_ACC_RDWR` if
the file exists, else `H5F_ACC_TRUNC`, then `ensureStandardGroups`
populates missing top-level groups. **Multi-process concurrent writers
to the same file are not solved by POSIX locking**; the documented MPI
pattern is `getPerRankFilePath` / `createPerRankFile` (per-rank
files) and `mergePerRankTPQFiles` on rank 0 (which copies/merges TPQ
sample trees into the unified file).

#### Compression / chunking

Optional deflate + shuffle through `ED_HDF5_COMPRESSION_LEVEL`,
`ED_HDF5_CHUNK_TARGET_BYTES`, `ED_HDF5_SHUFFLE` (see
`makeAdaptiveDsetProps`).

#### Selected `HDF5IO` API

(See §10 for the full list — every `save*` and `load*` is named by its
target dataset family.)

- `saveEigenvalues`, `loadEigenvalues`, `saveEigenvector`,
  `loadEigenvector`, `saveDiagonalizationResults`.
- `saveThermodynamics`, `loadThermodynamicObservable`.
- `saveCorrelationMatrix`, `saveCorrelationData`.
- `saveDynamicalResponse`, `saveDynamicalResponseFull`,
  `saveStaticResponse`.
- `saveFTLMSample`, `saveFTLMThermodynamicSample`,
  `saveFTLMDynamicalSample`, `saveFTLMStaticSample`,
  `saveFTLMThermodynamics`, `saveHybridThermalResults`.
- `appendTPQThermodynamics`, `appendTPQNorm`,
  `truncateAndRewriteTPQThermodynamics`,
  `truncateAndRewriteTPQNorm`, `saveTPQThermodynamics`, `saveTPQNorm`,
  `loadTPQThermodynamics`, `loadTPQNorm`, `listTPQSamples`,
  `saveTPQAveragedThermodynamics`, `saveTPQState`, `loadTPQState`,
  `listTPQStates`, `loadTPQStateByName`.
- `getPerRankFilePath`, `createPerRankFile`, `mergePerRankTPQFiles`,
  `copyTPQSamples`.

`HDF5SymmetryIO` writes a separate `symmetry_data.h5` (see §7.5).

### 10.2 In-memory Lanczos basis buffer (`lanczos_basis_buffer.{h,cpp}`)

`BasisBufferScope` registers an in-RAM storage keyed by `temp_dir` for
the duration of one Lanczos run. While registered, `read_basis_vector`
/ `write_basis_vector` (in `lanczos.cpp`) serve vectors from the buffer
instead of disk. Behaviour:

- `force_disk_storage()` returns true when `ED_LANCZOS_DISK=1` is set,
  in which case registration is **skipped** and the legacy
  `temp_dir/basis_<i>.dat` files are used.
- If no buffer is registered and the env var is not forcing disk, I/O
  still falls through to `.dat` files (so no crash if a caller
  forgets to scope).
- The reservation cap is 2²⁶ pointer slots to avoid accidental
  `vector::reserve` blow-ups.

There is **no automatic spill-to-disk threshold**; the choice is
explicit and per-run.

### 10.3 HDF5-backed basis storage (`basis_vector_storage.{h,cpp}`)

Independent of the Lanczos buffer: a tiny HDF5 file with a root
attribute `dimension` and datasets `/basis_<i>` shaped `(N, 2)` real/imag
doubles. `read_basis_vector_h5` / `write_basis_vector_h5` are
self-contained convenience wrappers.

### 10.4 Scratch directory layout

Lanczos drivers create scratch subdirectories under the run's
`output_dir` (or `./` if empty): `lanczos_basis_vectors`,
`block_lanczos_basis`, `chebyshev_lanczos_basis`,
`krylov_schur_temp`, `irl_basis_vectors`, `trl_basis_vectors`. Each is
keyed both as the `temp_dir` for the buffer and as the on-disk
fallback.

---

## 11. CLI: `ED` driver

`src/apps/ed_main.cpp` is intentionally thin (~590 lines). It is
responsible for help printing, the `--method-info=<name>` switch, the
`ED dssf <method>` subcommand, and the legacy positional CLI that
`EDConfig::fromCommandLine` understands.

### 11.1 The legacy positional CLI

The classical invocation:

```
ED <directory> [options]
ED --config=<file> [options]
```

After parsing into `EDConfig`, validating, and saving the config back
to `output_dir/ed_config.txt`, `ED` runs whichever workflows are flagged:

```cpp
if (config.workflow.precompute_basis_only) run_streaming_symmetry_workflow(config);
if (config.workflow.run_symm_auto && !skip_ed) sym_results = run_streaming_symmetry_workflow(config);
if (config.workflow.run_standard && !skip_ed) standard_results = run_standard_workflow(config);
if (config.workflow.run_disk_streaming && !skip_ed) ... run_disk_streaming_workflow(config);
if (config.workflow.run_chunked_symmetry && !skip_ed) ... run_chunked_symmetry_workflow(config);
if (config.workflow.compute_dynamical_response) dispatch_dssf(DYNAMICAL_THERMAL);
if (config.workflow.compute_static_response)    dispatch_dssf(STATIC_THERMAL);
if (config.workflow.compute_ground_state_dssf)  dispatch_dssf(GROUND_STATE_DSSF);
```

The four `run_*_workflow` functions (`include/ed/cli/workflows.h`) are
thin shims over the Hamiltonian-loading / sector-generation /
solver-dispatch chain in `ed_wrapper*.h`. Each prints a short
eigenvalue summary, runs `compute_thermodynamics` if `--thermo`, and
returns an `EDResults`.

### 11.2 The `ED dssf` subcommand

`ED dssf <dynamical_thermal|static_thermal|ground_state_dssf|single_expectation> <directory> [options]`

The handler:

1. Parses the method via `ed::dssf::method_from_string`.
2. Strips `dssf <method>` from `argv` and re-runs `EDConfig::fromCommandLine`.
3. Validates, creates the output directory, builds a `DSSFRequest` with
   the parsed method and a pointer to the new config (operators left
   default-constructed in this transitional design — see §8.1).
4. Calls `ed::dssf::run(request)` and prints a one-line summary.

### 11.3 Behaviour for flag combinations

After the recent CLI cleanup audit:

- `--symm` is the canonical flag; `--symmetrized` and
  `--streaming-symmetry` remain as deprecated aliases (still parsed).
- `--use-gpu` is honoured for the multi-T dynamical workflow and the
  static workflow when the config-operator path is taken and
  `--fixed-sz` is not set; otherwise the workflow banner emits a
  one-line "fall back to CPU" note. `--ground-state-dssf` has no GPU
  path and emits a one-line "using CPU" note when GPU was requested.
- `--disk-streaming` and `--chunked-symm` warn and downgrade GPU
  diagonalization methods to CPU Lanczos (their matrix-free CPU
  representation is incompatible with the GPU operator classes).

---

## 12. Python bindings

### 12.1 The `quantum_ed._core` extension

`python/quantum_ed/CMakeLists.txt` builds a `pybind11` module from
`_bindings/quantum_ed_bindings.cpp` that links against
`ed_solvers_cpu`, `ed_dssf`, `ed_symmetry`, and `ed_bfg` (the **CPU
stack**; GPU is intentionally out of scope for the Python module). The
artifact is placed next to `python/quantum_ed/__init__.py` so
`import quantum_ed` works in-tree and from a wheel.

Top-level Python surface (`quantum_ed/__init__.py.__all__`):

- `Operator`, `FixedSzOperator` — matrix-free Hamiltonian classes.
- `OP_SPLUS`, `OP_SMINUS`, `OP_SZ` — operator type tags.
- `full_diagonalization`, `lanczos`,
  `compute_thermodynamics_from_spectrum`,
  `finite_temperature_lanczos`, `low_temperature_lanczos`,
  `hybrid_thermal_method`.
- `FTLMParameters`, `LTLMParameters`, `HybridThermalParameters` —
  tunable structs for the thermal solvers.
- Submodules `dssf`, `symmetry`, `bfg`.
- `hamiltonian` — fluent builder DSL (see below).
- `helpers` — lazy aliases to `edlib` lattice helpers.
- `__version__`.

The bindings release the GIL around long-running C++ calls.

### 12.2 The fluent Hamiltonian DSL — `hamiltonian.py`

`Hamiltonian` exposes builder methods (`add`, `field`, `zz`, `xx_yy`,
`heisenberg`, `transverse_field_ising`, …) that emit symbolic terms
in the same `S+ / S- / Sz` convention as the C++ loaders, then
`build()` returns either an `Operator` or a `FixedSzOperator`.
`OP_TOKENS` is the token-to-term-list mapping.

### 12.3 DSSF, symmetry, BFG submodules

- `quantum_ed.dssf` re-exports `OperatorSpec`, `ObservablePairs`,
  `build_observable_pairs`, `compute_transverse_bases`. This guarantees
  the **same operator construction** as the `ED dssf` CLI, so
  observable names and ordering match the on-disk HDF5 layout.
- `quantum_ed.symmetry` re-exports the `ed::sym` permutation DSL
  (`identity`, `compose`, `power`, `order`, `translation`,
  `reflection_1d`, `site_swap`, `generate_group`, `group_from_generators`,
  `translation_group_1d`). The full-group helpers return `dict`s that
  mirror `SymmetryGroupInfo` / the JSON schema.
- `quantum_ed.bfg` re-exports the BFG cluster, topology, correlation,
  structure-factor, and TPQ I/O helpers.

### 12.4 `edlib`

The legacy data-prep package. Headline modules:

- `automorphism_finder.py` — see §7.7.
- `hdf5_io.py` — Python HDF5 helpers parallel to the C++ `HDF5IO`.
- Lattice helpers: `helper_cluster.py`, `helper_cluster_triangular*.py`,
  `helper_honeycomb*.py`, `helper_kagome_bfg*.py`,
  `helper_pyrochlore.py`, `helper_pyrochlore_super.py`. These generate
  `positions.dat`, `Trans.dat`, `InterAll.dat`, and lattice parameter
  files for the C++ pipeline.

`quantum_ed.helpers` lazy-imports these so they appear under the modern
namespace.

---

## 13. NLCE workflow package

`workflows/nlce/` implements a unified Numerical Linked Cluster
Expansion driver. The matrix is **(geometry × pipeline)**:

| Pipeline (`pipelines/`) | Strategy |
|-------------------------|----------|
| `full_ed`               | Dense `FULL` (or `SCALAPACK_MIXED` for large clusters); summation via `NLC_sum.py` (pyrochlore) or `NLC_sum_triangular.py` (triangular). |
| `ftlm`                  | Hybrid: `FULL` for small clusters (`num_sites ≤ hybrid_threshold`, default 10) else `FTLM` (or `FTLM_GPU`); adaptive Krylov; summation via `NLC_sum_ftlm.py`. |
| `lanczos_boost`         | Small clusters `FULL`; large `LANCZOS` with `--compute_eigenvectors`; summation via `NLC_sum_LB.py`. |

| Geometry (`geometries/`)     | Expansion unit | Generator script                       |
|------------------------------|----------------|----------------------------------------|
| `pyrochlore`                 | tetrahedra     | `generate_pyrochlore_clusters.py`      |
| `triangular_site`            | sites          | `generate_triangular_clusters.py`      |
| `triangular_triangle`        | triangles      | `generate_triangle_nlce_clusters.py`   |

Each `Geometry` provides `generate_clusters`, `prepare_hamiltonian`,
and (for triangular_site) `precompute_basis` for the
`--streaming-symmetry` option. Each `Pipeline` contributes
`make_ed_options`, `build_ed_command`, and `summation_command`.

The CLI (`__main__.py` → `cli.main()`) uses two-phase argparse: pick
`--geometry` and `--pipeline`, then merge their option sets. `--list`
enumerates the available geometries and pipelines.

A typical run does:

1. Cluster generation → `clusters_order_*`.
2. Hamiltonian preparation → `hamiltonians_order_*`.
3. Optional basis precomputation → cached orbit HDF5.
4. ED — `Pipeline.make_ed_options` → `build_ed_command` →
   `run_ed_subprocess`, with optional parallel execution.
5. NLCE summation — `Pipeline.summation_command` → Python driver.

`workflows/nlce/run/` and `analysis/` host the supporting summation
scripts and post-processing.

---

## 14. Tests, benchmarks, and operational notes

### 14.1 Tests (`tests/unit/`, registered through CTest)

Twenty-three test executables compile against the libraries and exercise:

| Test | Subsystem |
|------|-----------|
| `test_operator_apply` | `Operator::apply` correctness and OpenMP determinism |
| `test_fixed_sz_operator` | `FixedSzOperator` basis enumeration and projection |
| `test_full_diagonalization` | Dense LAPACK reference paths |
| `test_lanczos_variants` | All Lanczos drivers vs reference |
| `test_lanczos_basis_buffer` | RAM-buffer registration and disk fallback |
| `test_thermal_methods` | FTLM / LTLM / Hybrid against exact spectrum |
| `test_observables` | `calculate_thermodynamics_from_spectrum` |
| `test_dssf_engine` | `ed::dssf::run` dispatch |
| `test_dssf_io` | `dssf_io.h` schema round-trips |
| `test_dssf_legacy_schema` | `HDF5IO` legacy DSSF readout |
| `test_dssf_operator_spec` | `OperatorSpec` / `build_observable_pairs` |
| `test_hdf5_io` | Generic `HDF5IO` save/load round-trips |
| `test_trans_interall_loading` | File-format parsers |
| `test_symmetry`, `test_symmetry_dsl` | Sectoring, sector identities, the `ed::sym` DSL |
| `test_cpu_gpu_equivalence` | 4-site and 8-site Heisenberg CPU vs GPU `H·v` and ground-state |
| `test_bfg_*` (cluster, correlations, ring observables, structure factor, spin structure factor, topology, wavefunction I/O) | BFG library |
| `run_ed_smoke.sh` | End-to-end smoke test for the `ED` binary on a 4-site Heisenberg ring |

`tests/common/` provides `catch2_harness.h` and `test_harness.h`
shared infrastructure.

`ctest -j` runs all 103 cases in ~3 s on a typical workstation.

### 14.2 Benchmarks

`benchmarks/` contains Google Benchmark micro-benchmarks; the JSON
result snapshots `bench_cpp_results.json` and `bench_vs_quspin_results.json`
illustrate the typical comparison layout (vs QuSpin and across BLAS
backends).

### 14.3 Useful environment variables

| Variable | Effect |
|----------|--------|
| `ED_LANCZOS_DISK=1` | Force disk-backed basis storage (skip RAM buffer) |
| `ED_LANCZOS_VERBOSE=1` | Per-iteration Lanczos logging |
| `ED_LANCZOS_COMPLEX_SEED=0/1` | Seed Lanczos with a real (faster real-`H` path) or complex random vector |
| `ED_USE_SPARSE`, `ED_SPARSE_DIM_MAX` | Force/limit assembled CSR `apply` paths |
| `ED_USE_ARPACK_DEFAULT=1` | Make `--method=LANCZOS` route to ARPACK ground state for small `k` |
| `ED_GPU_TIMING=1` | Per-call GPU matvec event timing |
| `ED_GPU_DISABLE_CUSPARSE=1` | Disable cuSPARSE CSR fast path |
| `ED_GPU_CUSPARSE_MIN_DIM` | Threshold above which cuSPARSE is preferred (default 32 768) |
| `ED_HDF5_COMPRESSION_LEVEL`, `ED_HDF5_CHUNK_TARGET_BYTES`, `ED_HDF5_SHUFFLE` | Tune HDF5 compression / chunking |
| `DEBUG_BLAS_BACKEND=1` | Print the selected BLAS profile at startup |

### 14.4 MPI patterns

- **TPQ** parallelism is per-sample (independent random states across
  ranks).
- **Dynamical multi-T** uses internal MPI collectives; ranks must
  process operators in lockstep (no master/worker scheduling for that
  workflow).
- **Dynamical single-T** and **static** use master/worker dynamic
  scheduling with `TASK_TAG`/`DONE_TAG`/`STOP_TAG`.
- **Ground-state DSSF** uses static round-robin across ranks.
- **HDF5** writes from multiple ranks to a single file are not
  concurrency-safe; the documented workaround is per-rank files via
  `getPerRankFilePath` / `createPerRankFile` and
  `mergePerRankTPQFiles` on rank 0.

### 14.5 GPU fallback policy

| Combination | Behaviour |
|-------------|-----------|
| `--use-gpu` + `--fixed-sz` (DSSF/SSSF) | Banner notes the fallback; CPU path is taken |
| `--use-gpu` + `--ground-state-dssf` | Banner notes "no GPU path"; CPU path is taken |
| `--use-gpu` + dynamical, single-T, single-sample | CPU path is taken (no message; documented in workflow comment) |
| `--disk-streaming` / `--chunked-symm` + GPU method | Warning, downgrade to CPU Lanczos |
| GPU build absent + `--use-gpu` | Banner notes "build has no CUDA support; using CPU" |

---

## 15. Subsystem cross-reference

The following cheat sheet maps "what the user wants" to the call chain
that delivers it:

### Compute the ground state of a small spin Hamiltonian

```
ED <dir> --method=LANCZOS [--fixed-sz --n-up=N/2]
  → run_standard_workflow
    → exact_diagonalization_from_directory       (full Hilbert)
    → exact_diagonalization_fixed_sz             (sector)
      → ed_internal::create_hamiltonian_apply_function
      → exact_diagonalization_core (switch on method)
        → lanczos / arpack_ground_state / ...
        → HDF5IO::saveDiagonalizationResults
```

### Compute thermodynamics from a finite spectrum

```
ED ... --thermo
  → compute_thermodynamics
    → calculate_thermodynamics_from_spectrum
    → HDF5IO::saveThermodynamics
```

### FTLM thermodynamics on a single Hamiltonian

```
ED <dir> --method=FTLM --temp_min=... --temp_max=...
  → exact_diagonalization_core (FTLM branch)
    → finite_temperature_lanczos
    → save_ftlm_results → HDF5IO::saveFTLMThermodynamics
```

### Thermal `S(q, ω)` (DSSF) with GPU multi-T

```
ED dssf dynamical_thermal <dir> --use-gpu --dyn-temp-bins=20 ...
  → ed::dssf::run(DYNAMICAL_THERMAL)
    → compute_dynamical_response_workflow
      → construct_operators_from_config (build_observable_pairs)
      → process_operator_all_temps (per operator, all T at once)
        GPU branch: convertOperatorToGPU + GPUEDWrapper::runGPUDynamicalCorrelationMultiTemp
        CPU branch: compute_dynamical_correlation_*_multi_temperature
      → HDF5IO::saveDynamicalResponseFull
```

### `T = 0` ground-state DSSF

```
ED dssf ground_state_dssf <dir> --fixed-sz --n-up=N/2
  → ed::dssf::run(GROUND_STATE_DSSF)
    → compute_ground_state_dssf_workflow
      → construct_operators_from_config
      → find_ground_state_lanczos (or load from HDF5)
      → compute_ground_state_cross_correlation (per task, MPI round-robin)
      → HDF5IO::saveDynamicalResponseFull (under ground_state_dssf/<op>)
```

### Static structure factor (SSSF)

```
ED dssf static_thermal <dir> [--use-gpu]
  → ed::dssf::run(STATIC_THERMAL)
    → compute_static_response_workflow
      → process_task per operator
        GPU branch: GPUEDWrapper::runGPUStaticCorrelation
        CPU branch: compute_static_response (FTLM-style)
      → HDF5IO::saveStaticResponse
```

### Symmetry-projected ED

```
ED <dir> --symm [--method=LANCZOS_GPU]
  → run_streaming_symmetry_workflow
    → exact_diagonalization_streaming_symmetry
      → generate_automorphisms (Python helper, cached)
      → generateSymmetrySectorsStreaming (or load HDF5 cache)
      → for each sector:
          GPU: dispatchGPUSymmetrizedSector → GPUEDWrapper::runGPULanczos / etc.
          CPU: hamiltonian.applySymmetrized → exact_diagonalization_core
        → HDF5IO::saveEigenvalues (+ expanded eigenvectors)
```

### BFG order parameters from saved wavefunctions

```
compute_bfg_order_parameters[_gpu] <dir> [scan options]
  → ed::bfg::cli::run_single_file / run_scan
    → load_cluster, load_wavefunction / load_all_tpq_states
    → compute_smsp / szsz / xy_bond / dimer-SF / VBS / nematic / plaquette / SF
    → ed::bfg::save_results / save_temperature_scan_results / save_scan_results
```

### NLCE driver

```
python -m workflows.nlce --geometry=... --pipeline=...
  → workflows.nlce.cli.main
    → Geometry.generate_clusters
    → Geometry.prepare_hamiltonian (per cluster)
    → Geometry.precompute_basis (optional)
    → Pipeline.make_ed_options + build_ed_command + run_ed_subprocess (calls ED binary)
    → Pipeline.summation_command → NLC_sum*.py
```

---

*This report reflects the codebase as of the audit performed on
2026-04-24. It does not modify any source file; it is a descriptive
artifact intended to live alongside `MODERNIZATION_AUDIT.md`.*
