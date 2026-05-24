# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Full Unified-Interface Collapse — Remaining waves C2, E1-E4, F-partial (May 2026)

Continues the collapse landed in commit 6983531 ("Waves A, B, E-pilot, G").
This drop shrinks the legacy-symbol cone again: the two pilot CLI workflows,
all in-tree Python dispatch sites, and the C++ `dispatch.h` header are now
gone or routed through the unified `make_operator + workflows::*` shape.

#### Wave C2 — CLI workflow pilot (2 of 7 workflows migrated)

* **`run_standard_workflow`** in `src/cli/workflows.cpp` no longer calls the
  legacy `ed::exact_diagonalization(directory, method, params, ...)` entry
  from `<ed/core/dispatch.h>`. It now builds an `OperatorSpec`, calls
  `ed::make_operator(spec)`, and dispatches through `ed::workflows::solve`.
  An explicit `HDF5IO::saveEigenvalues(...)` keeps CLI parity with the
  orchestrator (which does not auto-save). The `params.full_sz_split &&
  method == FULL` branch (`ALL_SZ_SECTORS` in the legacy world) is
  reproduced as an explicit Sz loop that merges + sorts eigenvalues across
  sectors.
* **`run_streaming_symmetry_workflow`** in the same file uses
  `spec.streaming_symmetry = true` (with optional `spec.fixed_sz`) and
  iterates symmetry sectors explicitly via the downcast
  `StreamingSymmetryOperator::sector(k)` / `FixedSzStreamingSymmetryOperator
  ::sector(k)` -> `SectorView`. Each `SectorView` is solved independently
  through `ed::workflows::solve`, with per-sector HDF5 emission.
* **`include/ed/core/ed_config_adapter.h`** gains
  `ed_adapter::toSolveOptions(EDParameters, DiagonalizationMethod)`, which
  maps the legacy `EDParameters` field set + method enum onto
  `ed::workflows::SolveOptions` (including the CLI-parity knobs:
  `use_fixed_sz`, `use_symmetry`, `n_up`, `basis_cache_dir`,
  `precompute_basis_only`).
* **Build system** — `ed_distributed` is now a `PUBLIC` dependency of
  `ed_cli` in `cmake/EDLibraries.cmake` (under `WITH_MPI`), so any binary
  linking `ed_cli` automatically resolves the `DistributedOperator` /
  `DistributedSymmetryOperator` symbols pulled in transitively by the
  inline `ed::make_operator` factory.

#### Wave E1-E4 — Python `qed.*` wrappers point at the unified C++ surface

* **E1 — `python/qed/workflow.py`**: ground-state dispatch in `qed.diag`
  now routes through new helper `_diag_via_workflows_solve(...)`, which
  builds an in-memory `Operator`, maps `EDParameters` -> `SolveOptions`
  via `_ed_params_to_solve_options`, calls `_core.workflows_solve`, and
  reshapes the `GroundStateResult` into the legacy `EDResults` shape.
  Thermal methods (FTLM / TPQ / KPM-DOS) fall through to the existing
  `exact_diagonalization_core` path, which now emits a `DeprecationWarning`
  -- a new `_suppress_legacy_dispatch_warning` context manager silences
  those internal fallback calls so user-facing pytest runs are clean.
* **E2 — `python/qed/thermal.py`**: the two `exact_diagonalization_from_
  directory` call-sites repoint at `_core.workflows_thermal` when the
  orchestrator covers the lane (no spatial symmetry, or any TPQ method --
  which disables symmetry anyway). Builds an `Operator` from directory
  files, maps `EDParameters` -> `ThermalOptions`, and reshapes
  `ThermalResult` (now exposing `thermo` + `per_sector`) back into
  `EDResults`. Spatial-symmetry cases continue through the legacy lane.
* **E2 binding follow-up — `python/qed/_bindings/workflow_bindings.cpp`**:
  `ThermalResult` now exposes `thermo` (`ThermodynamicData`) and
  `per_sector` (`std::vector<ThermalSectorEntry>`), and a new
  `py::class_<ThermalSectorEntry>` binding surfaces `sz_index`,
  `ground_state_energy`, and `thermo`.
* **E3 (skip)** — `python/qed/dssf.py`: `qed.dssf.compute` still shells
  out to `./ED dssf`. The in-process `_core.workflows_spectral` path
  needs a non-trivial cross-correlator pair assembler in Python (or a new
  C++ `workflows_dssf_compute` orchestrator), so it is documented in the
  docstring as a tracked follow-up rather than landed now.
* **E4 — `python/qed/_bindings/dispatcher_bindings.cpp`**: the 5
  ED-related `m.def(...)` registrations (`exact_diagonalization_core`
  -- both overloads, `exact_diagonalization_streaming_symmetry`,
  `_streaming_symmetry_fixed_sz`, `exact_diagonalization_from_directory`)
  are now thin **deprecation-warning forwarders**. Each emits a Python
  `DeprecationWarning` pointing at the new `qed.workflows` /
  `qed.solve` / `qed.thermal` surface, then routes through the C++
  `ed_wrapper.h` / `ed_wrapper_streaming.h` entries directly (the
  `exact_diagonalization_from_directory` binding inlines the
  symmetry-detect + method-canonicalize logic that previously lived in
  `dispatch.h`).
* **E acceptance** — `python/tests/test_workflow.py` gains
  `test_cpu_path_lands_in_workflows_solve`, which monkeypatches
  `exact_diagonalization_core` and `workflows_solve` and asserts that
  `qed.diag(method=lanczos, ...)` lands in the unified entry. The full
  Python workflow suite passes under `pytest -W error::DeprecationWarning`
  (modulo pre-existing path / `edlib`-import failures unrelated to this
  drop).

#### Wave F-partial — first hard removals

* **`include/ed/core/dispatch.h` deleted** (~312 LOC). Its last in-tree
  callers were `run_standard_workflow`, `run_streaming_symmetry_workflow`,
  the Python `exact_diagonalization_from_directory` binding, and the
  `test_dispatch_streaming_thermo` unit test. The first two migrated to
  `make_operator + workflows::solve` in Wave C2; the Python binding now
  inlines the symmetry-detect + method-canonicalize routing in Wave E4;
  the unit test was deleted (the streaming-symmetry + FTLM end-to-end
  behaviour it covered is now exercised by `test_auto_thermal`).
* **`tests/unit/test_dispatch_streaming_thermo.cpp` deleted** and its
  `add_test(test_dispatch_streaming_thermo ...)` block removed from
  `CMakeLists.txt`.
* **Docs** — `docs/MIGRATION.md` and `docs/architecture/STRUCTURAL_AUDIT.md`
  updated: `dispatch.h` removed, the remaining legacy headers listed
  with their gating story.

#### Net effect

| Surface | Before this drop | After |
|---|---|---|
| `ed::exact_diagonalization` in-tree production callers (`.cpp`) | 2 in `workflows.cpp` + 1 in `dispatcher_bindings.cpp` + 2 in `test_dispatch_streaming_thermo.cpp` | 0 (only doc-comment references remain) |
| Python `qed.diag` ground-state lane | `_core.exact_diagonalization_core` (legacy) | `_core.workflows_solve` (unified) |
| Python `qed.thermal` (no-spatial-symmetry / TPQ lane) | `_core.exact_diagonalization_from_directory` | `_core.workflows_thermal` |
| LOC removed | -- | ~362 (`dispatch.h` + deleted test) |
| C++ tests green sequentially | 279 / 279 | 277 / 277 (the 2 removed were the `test_dispatch_streaming_thermo` Catch2 cases) |

#### Deferred (tracked in `plans/remaining_unified-interface_waves_*.plan.md`)

* Wave C3: migrate the 5 remaining heavy CLI workflows
  (`compute_dynamical_response_workflow`, `compute_static_response_workflow`,
  `compute_ground_state_dssf_workflow`, `compute_kpm_thermodynamics_workflow`,
  and the four DSSF helpers). Blocked on either extending
  `SpectralResult` / `ThermalResult` with the missing CLI-only HDF5
  fields (variance, susceptibility, ground-state vector, KPM moments) or
  keeping the HDF5 plumbing in the CLI and only swapping the inner
  `GPUEDWrapper::*` calls.
* Wave D: distributed CLI + 19 distributed tests (blocked on Wave B
  kernel-delegation inversion for the FTLM / TPQ distributed thermal
  lanes).
* Wave E5: `qed.dssf.compute` in-process path.
* Wave F2-F5: hard-rm of `ed_wrapper.h`, `ed_wrapper_streaming.h`, the 8
  distributed-solver shells, `gpu_ed_wrapper.h`, `gpu_lanczos.cuh`,
  `ed/solvers/{TPQ,ftlm,ltlm,kpm_dos}.h`. Each is gated on the
  corresponding C3 / D / B wave clearing the in-tree callers.

### Full Unified-Interface Collapse — Waves A, B, E, G (May 2026)

The "fragmented surface" the user kept hitting (`ed::exact_diagonalization_*`,
`GPUEDWrapper::*`, `distributed_lanczos*`, `qed.exact_diagonalization_*` plus
the new `ed::workflows::*` orchestrator surface co-existing) is now collapsed
into a single canonical entry-point shape **for every consumer that has been
migrated**:

    OperatorSpec  ->  ed::make_operator(spec)  ->  ed::workflows::{solve,thermal,spectral}

The same three-line shape works from C++ and from Python (`qed.workflows.*`),
with the backend (CPU / GPU / MPI / MPI+GPU) auto-selected internally from
the operator's `geometry()`.

#### Wave A — structural enablers (additive, zero behaviour change)

* **A1. `include/ed/core/make_operator.h`** — extended to honour every spec
  axis: `fixed_sz`, `streaming_symmetry`, `distributed`, and their pairwise
  combinations. Returns `std::unique_ptr<LinearOperator>`. New
  `InMemoryOperator` / `FilePaths` / `DirectoryPath` `std::variant` source
  alternative makes the factory work for both programmatic operators and
  file-based loaders. Companion `sector_view_from(spec, sector_idx)` for the
  streaming-symmetry per-sector lane.
* **A2.** Real `bind_<Backend>` overrides on `GPUOperator`,
  `DistributedOperator`, `DistributedSymmetryOperator`, and
  `StreamingSymmetryOperator::SectorView`. The default no-op forwarder is
  gone for these subclasses; wrong-backend calls now throw a documented
  `std::runtime_error` instead of silently miscompiling.
* **A3. `make_mpi_cuda_backend()` factory** in
  `include/ed/core/select_backend.h`. RAII `MpiCudaBackendContext` bundles
  the NCCL `MultiGpuCommunicator` and the `MpiCudaBackend` that references
  it; the backend reference into the comm is now lifetime-safe by
  construction. Used by the eventual distributed CLI to wrap externally-
  constructed MPI+GPU backends in the orchestrator's `BackendVariant`.
* **A4. Two dead lanes wired** in `src/orchestrator.cpp`:
  * `solve()` FullDiag on distributed -> gather-to-root + LAPACK `zheevd`
    on rank 0 + `MPI_Bcast` of eigenvalues (correct in the regime where
    FullDiag is picked: `global_dim <= 2^12`).
  * `spectral()` `FtlmDynamical` -> delegate to
    `::compute_dynamical_correlation` (the legacy FTLM-dynamical body in
    `ed/solvers/ftlm.h`).
* **A5.** Extended `SolveOptions` / `ThermalOptions` / `SpectralOptions`
  with the full CLI-parity knob set (`use_fixed_sz`, `use_symmetry`,
  `n_up`, `basis_cache_dir`, `precompute_basis_only`, `temp_min/max`,
  `num_temp_bins`, `broadening`, `num_samples`, `temperatures`,
  `observable_type`) so the upcoming CLI migration (Wave C) is a pure
  refactor with no feature loss.
* **A acceptance**: build green, **279 / 279 tests pass sequentially**;
  the parallel-ctest run has 2 intermittent races on the FullDiag
  HDF5-output tests (`test #49`, `#52`) — pre-existing, unrelated to
  this wave (both pass solo and under `-j 1`).

#### Wave B — thermal kernel-shim honest dispatch

The full delegation inversion (move algorithm bodies from CPU `.cpp` into
templated `*_kernel.h` headers, so FTLM/LTLM/KPM-DOS actually run on
CUDA / MPI backends) is deferred. What landed here makes the current
state explicit and crash-safe:

* `ftlm_kernel.h`, `ltlm_kernel.h`, `kpm_dos_kernel.h` now carry a
  `static_assert(std::is_same_v<Backend, CpuBackend>, ...)` declaring that
  CPU is the only supported backend today. The mTPQ / cTPQ kernels are
  already fully backend-templated and remain so.
* `src/orchestrator.cpp` thermal lanes use `if constexpr` filters around
  the `std::visit` so the FTLM / LTLM / KPM-DOS branches compile cleanly
  under `WITH_CUDA` / `WITH_MPI`, and raise a documented
  `std::runtime_error` if a caller asks for an unsupported backend. The
  mTPQ / cTPQ paths run on every backend variant.

#### Wave E — Python unified surface

* New module **`qed.workflows`** (`python/qed/workflows.py`) exposing
  `workflows.solve(op, opts)`, `workflows.thermal(op, opts)`,
  `workflows.spectral(op, observables, opts)` as thin wrappers over the
  C++ orchestrator bindings. Re-exports the option / result / method-tag
  classes (`SolveOptions`, `ThermalOptions`, `SpectralOptions`,
  `GroundStateResult`, etc.) so callers never have to reach into
  `qed._core`.
* `qed.solve` / `qed.spectral` still go through the auto-pilot
  (`qed.diag` / `qed.dssf.compute`) for smart-defaults users; the
  `qed.workflows.*` module is the canonical entry point for callers
  who want explicit control over the orchestrator surface.

Verified end-to-end on N=6 Heisenberg PBC: `qed.workflows.solve` runs,
returns the correct eigenvalues, and reports `backend.lane = 'cpu'` with
`wall_seconds` populated.

#### Wave G — examples rewritten + a new canonical end-to-end demo

* New **`examples/00_unified_interface.cpp`** — single self-contained
  walkthrough of the unified API: ground-state Lanczos, fixed-Sz sector
  solve, mTPQ thermal, and ground-state CF spectral, all in the same
  `OperatorSpec -> make_operator -> workflows::*` shape. Builds + runs
  end-to-end, prints E0 / E1, mTPQ E_min, and S(omega) samples.
* **Examples 01-04 + 09-10 rewritten** onto the unified API:
  * `01_cpp_ground_state.cpp`: in-memory Heisenberg chain ->
    `ed::workflows::solve(*op, Lanczos)`.
  * `02_cpp_full_spectrum.cpp`: J1-J2 chain ->
    `ed::workflows::solve(*op, FullDiag)`.
  * `03_cpp_ftlm_thermal.cpp`: Heisenberg PBC chain ->
    `ed::workflows::thermal(*op, FTLM)`.
  * `04_cpp_gpu_lanczos.cpp`: `GPUOperator` ->
    `ed::workflows::solve(op, Lanczos)`; backend auto-picked to
    `CudaBackend` because the operator's `geometry().memory_space ==
    Device`. Verified runs with `backend lane = gpu`.
  * `09_python_quickstart.py`: same Heisenberg chain via
    `qed.workflows.solve`.
  * `10_python_dssf.py`: full DSSF pipeline via `qed.dssf` +
    `qed.workflows.solve` + `qed.workflows.spectral`.

#### Real bug fix shaken loose by the GPU example rewrite

* `src/orchestrator.cpp` — Lanczos / Krylov-Schur / spectral seed-vector
  staging was using `seed.data()` (host pointer) as the initial vector
  passed to `lanczos_kernel<Backend>` / `krylov_schur_kernel<Backend>` /
  `cf_spectral_kernel<Backend>`. The kernels then call
  `be.copy(v0_local, v_curr.get(), local_n)` which for `CudaBackend` is
  a device-to-device `cudaMemcpy` — passing a host pointer there
  produces `cudaErrorInvalidValue` at runtime. Fixed by allocating a
  backend-resident vector via `be.make_zero_vector` and staging the
  host seed through `be.copy_from_host` before invoking the kernel.
  Caught by `ex04_cpp_gpu_lanczos`; this was a latent bug nothing
  else exercised because the existing GPU tests use `GPULanczos` /
  `GPUEDWrapper` directly, never the orchestrator's CudaBackend
  Lanczos lane.

#### Status of the remaining waves

The plan's Waves C (CLI migration of the seven `src/cli/workflows.cpp`
functions), D (distributed CLI + examples + tests), F (hard-removal of
~7 K LOC of legacy headers), and E4 (collapse `dispatcher_bindings.cpp`)
are not landed in this commit. They are non-trivial — each of the seven
CLI workflows is hundreds of lines of HDF5 plumbing + sector iteration
+ method canonicalization, and the hard-removal step is gated on those
migrations finishing. The Wave A enablers above are deliberately
sufficient for those migrations to proceed: the factory understands
every operator axis, the bind overrides are real, the
`make_mpi_cuda_backend()` factory is in place, and the option structs
carry every CLI knob. The unified surface is the canonical path for
**new** code; the legacy `ed::exact_diagonalization_*` / `GPUEDWrapper`
entry points remain only as in-tree CLI implementation detail until
those waves land.

### Day-17 follow-up — header-graph dead-code subtraction (May 2026)

Re-scanned the include graph after the main cleanup sweep and found two
genuinely unreachable headers (zero `#include` references anywhere in
`src/`, `tests/`, `benchmarks/`, `examples/`, `python/`):

* **`include/ed/core/hdf5_symmetry_io.h` (496 LOC) — deleted**. Defined
  the `HDF5SymmetryIO` class for sparse symmetry-basis / block-Hamiltonian
  HDF5 I/O. The symmetry path migrated to the streaming kernel in
  `ed/core/ed_wrapper_streaming.h` (matrix-free per sector), so the
  on-disk basis/block format this header described is no longer
  materialised. The companion `.cpp` was already removed in an earlier
  sweep; this finishes the removal.
* **`include/ed/distributed/multi_gpu_stub.h` (16 LOC) — deleted**.
  Backwards-compat shim that just forwarded to `ed/distributed/multi_gpu.h`
  after Phase 3c promoted the stub API to a real implementation.
  Historical doc mentions in `docs/history/PHASE_3_SUMMARY.md` are
  intentionally preserved.

`include/ed/core/make_operator.h` (168 LOC) was also flagged with zero
includers but kept: it's the documented future-API surface referenced
from CHANGELOG / MIGRATION.md / ARCHITECTURE.md and is the planned
first-consumer target for the Phase 4 CLI migration.

Net subtraction this day: **512 LOC** across 2 files. Tree size table in
`STRUCTURAL_AUDIT.md` VII.4 refreshed: total source files 324 → 322,
LOC (incl. tests, excl. docs) ~81 900 → ~81 400. Build green; 271/271
tests pass in isolation (2 intermittent ctest failures remain in
parallel runs — known parallel-HDF5 contention races, both pass solo).

### Orchestrator Lanczos lane — perf tuning + small dead-code subtraction (May 2026, day 17)

Follow-up to the cleanup sweep that closes part of the ~1.45× constant
overhead gap measured in `bench_minimalist_collapse`:

* `src/orchestrator.cpp` — `ed::workflows::solve` Lanczos lane:
  * `convergence_check_interval` dropped from `5` → `1` so the orchestrator
    matches the legacy CPU `lanczos()`'s every-iteration Ritz-value
    convergence check (previously ran up to 4 extra iterations after
    convergence).
  * Tridiag solve now picks the cheaper Eigen path: `solve_tridiag(...)`
    (eigenvalues-only) when `opts.compute_vectors == false`,
    `solve_tridiag_with_eigenvectors(...)` only when the eigenvectors are
    actually requested. The legacy `_with_eigenvectors` path runs the full
    SelfAdjointEigenSolver unconditionally — ~2-3× slower for the common
    "num_eigs=1, vectors=false" workflow.
* Effect on `bench_minimalist_collapse` (N = 14, single rank, CPU):
  workflows Lanczos lane goes from **5.32 ms → 4.74 ms**, narrowing the
  gap to the legacy lane from 1.45× to 1.24×.
* `include/ed/core/ed_wrapper.h` — removed the empty pass-through shim
  `ed_internal::normalize_method_and_fixed_sz()` (zero call sites; the
  `_FIXED_SZ` enum variants it was guarding were retired earlier in the
  minimalist refactor).

### ED cleanup sweep — phased deletion of legacy dispatcher / auto-pilot surface (May 2026, days 15-16)

The Minimalist ED Collapse landed the new `ed::workflows::solve / thermal /
spectral` surface on top of five `<Backend>`-templated kernels. The legacy
surface was kept buildable to avoid breaking the tree. This sweep retires
the parts that are now genuinely unreachable, lands the FTLM/LTLM/KPM-DOS
orchestrator wiring that Phase 6 needed, and documents the remaining
phases whose deletions are blocked by the CLI / GPU-wrapper migrations
still in flight.

#### Phase 0 — Dead-code subtraction (~1.2 K LOC out, zero migration)

* Deleted `include/ed/gpu/gpu_dynamics.cuh` + `src/solvers/gpu/gpu_dynamics.cu`
  (`GPUDynamicsSolver` had zero callers).
* Deleted `include/ed/core/ed_dispatch_symmetry.h` (55 LOC, 0 callers).
* Removed dead methods from `GPUEDWrapper`:
  `createGPUOperatorFromCPU`, `gpuMatVec`, `shouldUseGPU`,
  `estimateGPUMemory`.
* Removed dead methods from GPU solvers: `GPULanczos::runWithStartVector`,
  `GPUBlockLanczos::runWithStartBlock`, `setReorthStrategy`,
  `GPUKrylovSchur::setMaxOuterIterations`.
* Removed `#if 0` archive blocks in `src/solvers/cpu/lanczos.cpp` and
  `src/solvers/cpu/ftlm.cpp` (git is the archive).
* Stripped the stale ARPACK benchmark from
  `benchmarks/bench_lanczos_ground_state.cpp` (header `arpack.h` had
  already been removed in a prior session).

#### Phase 1 — `ed::workflows::*` Python bindings

* New `python/qed/_bindings/workflow_bindings.{cpp,h}` bind
  `ed::workflows::solve / thermal / spectral` (plus `SolveOptions /
  ThermalOptions / SpectralOptions / SolveMethod` enum, `GroundStateResult
  / ThermalResult / SpectralResult`, `KrylovDiagnostics`, `BackendMetadata`)
  into `qed._core.workflows_*`.
* Wired into `python/qed/CMakeLists.txt`.
* Smoke-tested end-to-end on a 10-site Heisenberg chain.

#### Phase 2 — C++ unit-test migration off `ed::auto_pilot::*`

* `tests/unit/test_auto_solve.cpp` (7 cases),
  `tests/unit/test_auto_thermal.cpp` (4 cases),
  `tests/unit/test_auto_dssf.cpp` (3 cases),
  `tests/unit/test_ed_solver_matrix_e2e.cpp` (7 cases) migrated from
  `ed::auto_pilot::*` to `ed::workflows::*`.
* `src/orchestrator.cpp` now wires the `FullDiag` lane to the legacy
  `full_diagonalization` function for small `dim` (≤ 2¹²); fixed the
  `LinearOperator::bind<Backend>()` call signature to invoke the returned
  `std::function` directly.

#### Phase 3 — Python test + API migration

* `python/tests/test_dispatcher.py` Lanczos / BlockLanczos / KrylovSchur
  / FullDiag method-loop cases (plus FixedSz + FullDiag fixtures) now
  go through `qed._core.workflows_solve`.
* `python/tests/test_workflow.py` retains its `qed.diag` surface but
  the GPU-routing test monkeypatches both legacy and workflows names so
  the directory dispatcher test stays green.
* `python/qed/__init__.py`: legacy `qed.exact_diagonalization_*` and
  `qed._from_directory*` names are now `_deprecated_alias` wrappers that
  emit `DeprecationWarning` and forward to `qed.solve(...)`.

#### Phase 4 — Helpers landed; CLI migration deferred

* `include/ed/core/select_backend.h`: added `WithMpiCudaBackend(...)`
  helper for the distributed `--gpu` lane.
* `include/ed/core/make_operator.h`: extended `make_operator(OperatorSpec)`
  to cover the full Hamiltonian file matrix (`InterAll.dat`, `Trans.dat`,
  `CounterTerm.dat`, `ThreeBodyG.dat`) and to dispatch `fixed_sz`
  projection via `FixedSzOperator`.
* CLI body refactor of `src/cli/workflows.cpp` (4 callsites) and
  `src/cli/ed_distributed_main.cpp` (~18 callsites) DEFERRED to follow-up
  PR; depends on Phase 6 distributed FTLM/TPQ kernel landings and a
  directory-driven streaming-symmetry path on `workflows::*`.

#### Phase 5 — Hard removal of auto-pilot heuristic headers (~2.5 K LOC out)

* Deleted `include/ed/auto/{solve,thermal,dssf,diag_tune,dssf_tune}.h`.
* Deleted `tests/unit/test_diag_tune.cpp` and `tests/unit/test_dssf_tune.cpp`.
* `include/ed/core/{dispatch,ed_wrapper,ed_wrapper_streaming}.h` and the
  full `python/qed/_bindings/dispatcher_bindings.cpp` collapse DEFERRED:
  depends on Phase 4 (CLI migration) and Phase 6 (distributed FTLM/TPQ
  kernel landings).

#### Phase 6 — FTLM/LTLM/KPM-DOS orchestrator wiring

* `src/orchestrator.cpp`: `ed::workflows::thermal` now dispatches
  `ThermalOptions::Method::{FTLM,LTLM,KpmDos}` through the existing
  `ed::thermal::{ftlm,ltlm,kpm_dos}_kernel<Backend>` templates instead
  of throwing `runtime_error`.
* `tests/unit/test_auto_thermal.cpp` "FTLM/LTLM/KpmDos lanes throw" test
  flipped to verify the lanes now run end-to-end on a 4-site Heisenberg
  chain with a small beta grid.
* Distributed solver shell removal (8 headers + 8 sources + 19 tests
  retarget + 4 example binaries + CLI) DEFERRED: ~3 K LOC of mechanical
  retargeting depending on Phase 4 (CLI) completion.

#### Phase 7 — GPU shell collapse (DEFERRED)

* Blocked by Phase 4 (`src/cli/workflows.cpp` still uses
  `GPUEDWrapper::runGPUDynamicalCorrelationMultiTemp`) and Phase 5
  (`include/ed/core/ed_wrapper.h` still includes `GPUEDWrapper`).
* Requires landing `tpq_kernel<CudaBackend>` and
  `block_lanczos_kernel<CudaBackend>` facades (~1 K LOC of new code)
  before deletion of `gpu_lanczos.cuh`, `gpu_ed_wrapper.{h,cu}`,
  `gpu_solvers.h`.

#### Phase 8 — CPU solver shell collapse (DEFERRED)

* Blocked: all 12 `include/ed/solvers/*.h` headers are still referenced
  by the kernel-shim layer (`ctpq_kernel.h` delegates to legacy
  `TPQ.cpp::run_tpq`; `ftlm_kernel.h` delegates to
  `finite_temperature_lanczos`; etc), by the CLI (`src/cli/workflows.cpp`),
  by examples (01_cpp_ground_state.cpp, 02_cpp_full_spectrum.cpp,
  03_cpp_ftlm_thermal.cpp), and by 14 unit/bench files. Cleanly
  collapsing them needs either (a) native kernel-body implementations, or
  (b) inverting the delegation so legacy entry points dispatch through
  `workflows::*`. Both are >1 K LOC and depend on Phase 4 / 7.

#### Phase 9 — Docs + benchmark

* This CHANGELOG entry.
* `docs/architecture/STRUCTURAL_AUDIT.md` Part VII appended with the
  phase-by-phase LOC deltas.
* `docs/MIGRATION.md` updated: `include/ed/auto/{solve,thermal,dssf,
  diag_tune,dssf_tune}.h` wording changed from "scheduled for removal" to
  "removed".
* `docs/architecture/ARCHITECTURE.md` legacy pre-collapse picture trimmed
  to a single paragraph + link.

##### Benchmark — `bench_minimalist_collapse` (N = 6..14 Heisenberg, single rank, CPU)

`benchmarks/bench_minimalist_collapse.cpp` compares the new
`ed::workflows::solve` lane against the legacy
`exact_diagonalization_core` lane against direct LAPACK `zheevd` full
diagonalization. All on the same 1-D periodic Heisenberg chain.

| N  | dim    | `ed::workflows::solve` (Lanczos) | Legacy `EDCore` (Lanczos) | LAPACK full diag |
|---:|-------:|---------------------------------:|---------------------------:|-----------------:|
| 6  | 64     | 0.147 ms                         | 0.099 ms                   | 1.43 ms          |
| 8  | 256    | 0.270 ms                         | 0.188 ms                   | 23.6 ms          |
| 10 | 1 024  | 0.589 ms                         | 0.427 ms                   | 332 ms           |
| 12 | 4 096  | 1.79 ms                          | 1.43 ms                    | 8 857 ms         |
| 14 | 16 384 | 4.74 ms                          | 3.81 ms                    | (not run)        |

Two observations from the sweep:

1. **Lanczos vs LAPACK crossover stays where it was**: the workflows
   lane beats LAPACK by ~9× at N = 8 and >100× by N = 12 — same
   asymptotic behaviour as the legacy lane.
2. **The new lane carries a ~1.2–1.5× constant overhead**: visible at
   N = 6 (0.15 ms vs 0.10 ms ≈ 1.5×) and narrowing toward N = 14
   (4.7 ms vs 3.8 ms ≈ 1.24×) where matvec time dominates. After the
   orchestrator-side perf pass (May 2026 cleanup-sweep follow-up: drop
   `convergence_check_interval` from 5→1, and use the eigenvalues-only
   tridiag solve when `compute_vectors == false`), the residual gap
   tracks entirely to (a) `select_backend` + `unique_ptr<CpuBackend>`
   allocation per call, and (b) `LinearOperator::bind<CpuBackend>`
   constructing a `std::function` envelope around the matvec. Both are
   amortizable across a longer-running session; neither is a correctness
   issue.

#### Summary of LOC subtracted in this sweep

| Phase | LOC out | Status |
|------:|--------:|--------|
| 0 | ~1 200 | landed |
| 1 | +400 (new bindings) | landed |
| 2 | 0 (test rewrite) | landed |
| 3 | 0 (test rewrite) | landed |
| 4 | 0 (helpers landed; CLI deferred) | partial |
| 5 | ~2 500 | partial |
| 6 | 0 (orchestrator wiring; deletions deferred) | partial |
| 7 | 0 | deferred |
| 8 | 0 | deferred |
| 9 | docs only | landed |
| **Net this sweep** | **~3 700 LOC out** | — |

The remaining ~14 K LOC of deletions (distributed solver shells, GPU
solver shells, CPU solver shells, `ed_wrapper.h` family,
`dispatcher_bindings.cpp` collapse) will land in a follow-up PR once the
CLI migration (Phase 4) and GPU-kernel facades (Phase 7) are in.

---

### Minimalist ED Collapse — unified `ed::workflows::*` entry points (May 2026, days 13-14)

The minimalist-architecture rollout collapses the entire C++ public API
down to three top-level entry points: `ed::workflows::solve`,
`ed::workflows::thermal`, and `ed::workflows::spectral`. Every other
public symbol that used to take an `Operator&` (or worse, parsed files
itself) is now either a forwarder to these three or marked for removal.

#### Phase 1 — Backend BLAS-3

* `include/ed/matvec/backends/backend.h` — added `gemm`, `gemv`, `trsm`,
  `qr_thin` to the `Backend` abstract interface. These dense BLAS-3 ops
  let block kernels stay backend-agnostic.
* `include/ed/matvec/backends/cpu_backend.h` — implemented via LAPACKE.
* `include/ed/matvec/backends/cuda_backend.cuh` — implemented via cuBLAS
  / cuSOLVER (`cublasZgemm`, `cusolverDnZgeqrf` + `Zungqr`).
* `include/ed/matvec/backends/mpi_backend.h` — implemented as local
  dispatch (BLAS-3 ops are intrinsically per-rank in the partitioning
  used here; communication happens via `all_reduce_sum_vec`).
* `tests/unit/test_backend_blas3.cpp` — round-trip correctness on
  `CpuBackend` and (under `WITH_CUDA`) `CudaBackend`.

#### Phase 2 — Kernel unification

* **2.1 — `LocalDGKS3` reorth policy + Lanczos resume**:
  `include/ed/krylov/lanczos_kernel.h` gained a `ReorthPolicy::LocalDGKS3`
  three-pass local DGKS variant (matches the legacy CPU Lanczos
  Phase-7 reorth) and a `LanczosResumeState` struct that bridges
  `lanczos_checkpoint.h` to the kernel. `src/solvers/cpu/lanczos.cpp::lanczos()`
  body is now a 30-line orchestrator over the kernel.
* **2.2 — Krylov-Schur kernel templated**:
  `include/ed/krylov/krylov_schur_kernel.h` lifted the `CpuBackend`
  static_assert. `src/solvers/cpu/lanczos.cpp::krylov_schur()` body is a
  thin orchestrator; the legacy hand-rolled restart loop is preserved
  under `#if 0` for archival.
* **2.3 — Block-Lanczos kernel templated**:
  `include/ed/krylov/block_lanczos_kernel.h` rewritten on top of
  `Backend::gemm` + `Backend::qr_thin`. Delivers CPU + CUDA out of the
  box; MPI / MPI+CUDA `TSQR` deferred (`static_assert` enforces).
  `src/solvers/cpu/lanczos.cpp::block_lanczos()` is now an orchestrator.
* **2.4 — TPQ kernel**:
  `include/ed/thermal/tpq_kernel.h` is a new unified
  `tpq_kernel<Backend>` that covers microcanonical TPQ (mTPQ) and
  canonical Taylor TPQ (cTPQ) via per-step callbacks.
  `include/ed/thermal/mtpq_kernel.h` and `ctpq_kernel.h` are now thin
  facades.
* **2.5 — Continued-fraction spectral kernel**:
  `include/ed/observables/cf_spectral_kernel.h` wraps
  `lanczos_kernel<Backend>(keep_basis=false)` + the existing host-side
  `continued_fraction_spectral_function`. The CF Lanczos path in
  `src/solvers/cpu/ftlm.cpp::compute_dynamical_correlation_state_cf` is
  now an orchestrator over this kernel.

#### Phase 3 — Concepts

* **3.1 — `LinearOperator` concept**:
  `include/ed/core/linear_operator.h` defines the `Geometry`
  (local_dim, global_dim, local_offset, memory_space, MPI_Comm) +
  `bind<Backend>()` polymorphic interface that operators implement.
* **3.2 — Operator models**: `Operator`, `DistributedOperator`,
  `DistributedSymmetryOperator`, `GPUOperator`, and
  `StreamingSymmetryOperator::SectorView` now all inherit from
  `LinearOperator` (transparently through `MatVecOperator`).
* **3.3 — Unified result types**: `include/ed/core/results.h` defines
  `GroundStateResult`, `ThermalResult`, `SpectralResult`,
  `BackendMetadata`, `KrylovDiagnostics`. Legacy `EDResults`,
  `DistributedLanczosResult`, etc. remain available; new entry points
  populate the unified shape.

#### Phase 4 — Orchestrators

* **4.1 — `select_backend`**: `include/ed/core/select_backend.h`
  implements the runtime decision logic (CUDA available? MPI ranks > 1?
  GPU memory fits?) and returns a `std::variant<CpuBackend, CudaBackend,
  MpiBackend, MpiCudaBackend*>` for `std::visit` dispatch. Respects the
  operator's declared `memory_space()` to avoid host/device mismatches.
* **4.2 — Workflow orchestrators**:
  `include/ed/orchestrator.h` + `src/orchestrator.cpp` deliver
  `ed::workflows::solve(op, opts) -> GroundStateResult`,
  `ed::workflows::thermal(op, opts) -> ThermalResult`,
  `ed::workflows::spectral(op, observables, opts) -> SpectralResult`.
  Each orchestrator uses `select_backend` + `std::visit` to dispatch
  through the new kernel family. Default heuristics for
  `SolveMethod::Auto` (Lanczos / Krylov-Schur / Block-Lanczos /
  FullDiag) live in one place.
* **4.3 — `make_operator(OperatorSpec)` factory**:
  `include/ed/core/make_operator.h` exposes a unified entry point that
  consumes either `FilePaths`, `DirectoryPath`, or an `InMemoryOperator`
  (`std::variant`-based) and returns a `std::unique_ptr<Operator>`.
  This collapses the old
  `exact_diagonalization_from_files / _from_directory / _streaming_symmetry*`
  entry surface to one.

#### Phase 5 — Legacy entry-point soft-break

The plan called for hard deletion of the legacy surface. In practice the
in-tree call graph (CLI binary, Python binding, ~30 unit/integration
tests) still depends on `ed::auto_pilot::*`, `ed_dispatch::*`, and the
`exact_diagonalization_*` family, so a literal `rm -rf` would leave the
tree non-buildable. The collapse landed as a **soft hard-break** in this
release:

* Top-of-file deprecation notices added to `ed/core/dispatch.h`,
  `ed/core/ed_wrapper.h`, `ed/auto/solve.h`, `ed/auto/thermal.h`,
  `ed/auto/dssf.h` documenting the `ed::workflows::*` migration path.
* CHANGELOG and `docs/MIGRATION.md` (new) explain how to port C++ and
  Python callers; in-tree migration to be completed by follow-up PRs.

#### Phase 6 — Python bindings

* `python/qed/__init__.py` and the pybind11 binding gained
  `qed.solve / qed.thermal / qed.spectral` as the canonical entry
  points. `qed.diag`, `qed.dssf.compute`, `qed.finite_temperature_lanczos`,
  and `qed.lanczos` are preserved as deprecation aliases routed through
  the new entry points so existing notebooks continue to run.

#### Phase 7 — Tests + docs

* `tests/unit/test_backend_blas3.cpp` — `gemm`, `qr_thin` round-trip.
* `tests/unit/test_orchestrator.cpp` — small Heisenberg-chain smoke
  through `ed::workflows::solve / thermal / spectral`.
* `tests/unit/test_kernel_facades.cpp` — extended with
  `krylov_schur_kernel<CpuBackend>` and `tpq_kernel<CpuBackend>`
  facade tests.
* `docs/architecture/STRUCTURAL_AUDIT.md` Part VI ("Minimalist
  collapse") summarises the new architecture.

### Krylov-unification gap-fill — fill every remaining Krylov-kernel deployment gap (May 2026, days 11-12)

The structural audit shipped in Part IV of `docs/architecture/STRUCTURAL_AUDIT.md`
identified the remaining Krylov-kernel unification gaps. This rollout closes
them across all four (CPU / GPU / CPU+MPI / GPU+MPI) deployment lanes:

#### Phase 1 — `CudaBackend` perf foundations

* `include/ed/matvec/backends/cuda_backend.cuh` — overrode `dot_many` and
  `axpy_many` with a single `cublasZgemv` call over a contiguous device
  staging buffer that is incrementally grown with fingerprinted cache
  hits. Replaces M sequential BLAS-1 round-trips with one BLAS-2 call;
  ~600-1000 us latency saved per CGS2 pass at M=100. Added pool-backed
  allocator via `cudaMemPool_t` + `cudaMallocFromPoolAsync` (release
  threshold `UINT64_MAX` so allocations persist across runs). Replaced
  `scale + axpy` with a fused `cublasZgeam`-based `axpby` override.
* `tests/unit/test_cuda_backend.cpp` — new cases for `dot_many` /
  `axpy_many` correctness vs sequential reference; pool churn smoke
  test; fused `axpby` agreement check.

#### Phase 2 — Single-GPU consumer migrations

* `src/solvers/gpu/gpu_ftlm.cu` — `buildLanczosTridiagonalFromVector` and
  `buildLanczosTridiagonalWithBasis` are now thin shims over a new
  `run_ftlm_lanczos_kernel_facade` (in `gpu_lanczos_kernel_facade.cu`)
  that wraps `lanczos_kernel<CudaBackend>` directly. `cudaFree` calls
  for pool-backed allocations switched to `cudaFreeAsync(_, 0)`.
* `src/solvers/gpu/gpu_krylov_schur.cu` — non-migration decision
  documented in-file: the contiguous-basis layout is fundamental to the
  `cublasZgemm`-based thick-restart and there is no perf gain (KS
  already uses batched GEMV). Tracked as a separate "Backend BLAS-3
  view" workstream.

#### Phase 3 — GPU+MPI lane (Phase C completion)

* `include/ed/matvec/backends/mpi_cuda_backend.cuh` — new
  `MpiCudaBackend final : public CudaBackend`. Inherits every local
  primitive; overrides `dot`, `nrm2`, `all_reduce_sum(Complex|double)`,
  and `dot_many` to chain an `ncclAllReduce` after the local cuBLAS
  pass. Single batched ncclAllReduce per CGS2 pass replaces M sequential
  ones. NCCL-gated via `ED_HAVE_NCCL`.
* `src/distributed/distributed_lanczos_gpu.cu` — entirely replaced the
  ~220-LOC hand-rolled three-term recurrence (which had NO
  reorthogonalisation — audit D2 silent-wrong-answer bug) with a call
  to `lanczos_kernel<MpiCudaBackend>` configured for FullCGS2 reorth
  and a `make_smallest_ritz_convergence` predicate. Closes audit items
  **D2** and **D3** simultaneously.
* `src/distributed/distributed_krylov_schur_gpu.cu` — `ks_gpu_impl`'s
  per-cycle inner Arnoldi loop replaced with one
  `lanczos_kernel<MpiCudaBackend>` call. `aux_ortho_ptrs` carries the
  locked Ritz set so the kernel's CGS2 keeps the cycle basis
  orthogonal to it. Output basis is copied back into the existing
  `d_basis` contiguous slab to preserve the downstream restart logic.
* `src/distributed/distributed_ftlm_gpu.cu` — per-sample Lanczos body
  (`lanczos_loop_gpu`, ~150 LOC) gated behind
  `ED_FTLM_GPU_LEGACY_LANCZOS`; the production per-sample build now
  calls `lanczos_kernel<MpiCudaBackend>` and copies the resulting basis
  into the pre-existing `d_basis` slab so the observable-projection
  code path remains unchanged.
* `tests/unit/test_distributed_lanczos_gpu.cpp` — new
  `|| V^H V - I ||_inf < 1e-10` orthogonality lockdown calling
  `lanczos_kernel<MpiCudaBackend>` directly.
* `tests/unit/test_mpi_cuda_backend.cpp` — new NCCL-gated lockdown
  suite for the `MpiCudaBackend` overrides: `dot`, `nrm2`, `dot_many`
  parity vs `MPI_Allreduce` over CPU dot; `axpy_many` purely-local
  correctness; `all_reduce_sum(double|Complex)` parity.

#### Phase 4 — Top-level CPU `lanczos()`

* `include/ed/krylov/lanczos_kernel.h` — new
  `LanczosKernelOptions::on_step` checkpoint hook + matching
  `on_step_interval` (Phase 4.1, infrastructure ready). Designed for
  the `lanczos.cpp` checkpoint / restart + disk basis I/O path.
* `src/solvers/cpu/lanczos.cpp` — body migration is **deferred** with
  an in-file design note explaining the two interlocking blockers
  (LocalDGKS3 reorth policy not in the kernel + kernel resume-from-state
  not yet exposed). The on_step hook lands the foundational kernel API
  change; the body migration itself is tracked as a follow-up rev.

#### Phase 5 — FTLM / LTLM / Hybrid direct-kernel call

* `include/ed/krylov/lanczos_tridiag.h` — new
  `ed::krylov::lanczos_tridiag<MatvecFn>(...)` helper that delegates to
  `lanczos_kernel<CpuBackend>` without the `std::function` adapter or
  the `UniqueVec -> ComplexVector` translation copy.
* `src/solvers/cpu/ftlm.cpp` — `build_lanczos_tridiagonal` now calls
  the kernel directly through `lanczos_tridiag` on the full-reorth
  fast path.
* `include/ed/solvers/lanczos.h` —
  `build_lanczos_tridiagonal_with_basis` marked `[[deprecated]]`
  (Phase 5.3). The attribute is gated by `ED_BUILDING_INTERNAL` so the
  library's own remaining legacy callsites do not produce warnings;
  every EXTERNAL caller (Python bindings, downstream apps) gets the
  diagnostic pointing at the kernel-direct replacement.

#### Phase 6 — Hygiene + polish

* `include/ed/matvec/backends/cpu_backend.h` — `CpuBackend::dot_many`
  per-thread accumulator scratch is now persistent `mutable` member
  storage instead of `std::vector<double>` constructed every call.
  Saves ~50 KB of allocator churn per Lanczos step (Phase 6.1).
* `include/ed/matvec/backends/mpi_backend.h` — `MpiBackend` constructor
  now `MPI_Comm_dup`s the caller-supplied communicator so subsequent
  app-level `MPI_Comm_split` / `MPI_Comm_free` cannot stomp the
  library's stored handle (audit D6 / Phase 6.2). Destructor checks
  `MPI_Finalized` before freeing to be safe at process shutdown.
* `src/distributed/distributed_lanczos_gpu.cu` — the runtime no-reorth
  warning from the legacy path is gone (replaced by the kernel-driven
  body in Phase 3.2).

#### Audit items closed

* **D2** (`distributed_lanczos_gpu` no reorth) — fixed via Phase 3.2.
* **D3** (`distributed_lanczos_gpu` incorrect convergence check) —
  fixed via Phase 3.2.
* **D6** (`MpiBackend` parent-comm hygiene) — fixed via Phase 6.2.
* **S1 #32** (FTLM `std::function` matvec adapter) — fixed via Phase 5.

#### Net LOC delta

Roughly **-700 LOC** of duplicated hand-rolled Lanczos bodies deleted
(`distributed_lanczos_gpu.cu` ~220 LOC, `distributed_krylov_schur_gpu.cu`
~80 LOC inner loop, `distributed_ftlm_gpu.cu` ~150 LOC behind legacy
flag, `gpu_ftlm.cu` ~120 LOC); **+450 LOC** new infrastructure
(`MpiCudaBackend`, `lanczos_tridiag` helper, on_step hook, tests). The
four-cell deployment matrix is now fully unified on
`ed::krylov::lanczos_kernel` except for `lanczos.cpp::lanczos` (Phase
4.2-4.3 deferred).

### Minimalist-architecture rollout — distributed Krylov-Schur per-cycle Lanczos consolidated onto `lanczos_kernel<MpiBackend>` via new `aux_ortho_ptrs` (May 2026, day 9)

The thick-restart distributed Krylov-Schur implementation in
`src/distributed/distributed_krylov_schur.cpp` carried its own inline
Lanczos build (recurrence + per-step CGS2 reorth against the union of
the locked Ritz set and the current cycle's basis). The only thing that
prevented it from using the unified `ed::krylov::lanczos_kernel` was
that the kernel's CGS2 only knew about its own growing basis — not an
extra fixed set the caller wanted to project out. Day 9 closes that gap
with a single additive kernel option and migrates KS onto it.

* `include/ed/krylov/lanczos_kernel.h` — new
  `LanczosKernelOptions::aux_ortho_ptrs` (a `std::vector<const
  Complex*>` of fixed backend-memory pointers). When non-empty AND
  `reorth != ReorthPolicy::None`, every CGS2 pass projects `w` against
  the union `aux_ortho_ptrs ∪ basis_ptrs` in a single batched
  `dot_many` / `axpy_many` call. The pointers must remain valid for
  the duration of the kernel run; the kernel does not take ownership
  and does not append to the aux set. Per-step Allreduce count is
  unchanged (still 2 per step for FullCGS2 — one per pass over the
  combined set), so callers get aux orthogonality "for free" at the
  network-traffic level.
* `src/distributed/distributed_krylov_schur.cpp` — the per-restart-cycle
  inline Lanczos body (three-term recurrence + per-vector
  `reorth_against` loop + breakdown check + swap-rotate, ~50 LOC) is
  replaced with one `ed::krylov::lanczos_kernel<MpiBackend>(...)` call
  that takes the locked Ritz set as `aux_ortho_ptrs`. The kernel hands
  back `alpha`, `beta`, and the backend-owned `basis`; we marshal the
  basis into the `std::vector<std::vector<Complex>>` shape the locking
  step + Ritz reconstruction below already speak. `breakdown_tol =
  1e-13` matches the historical KS bar (distinct from the kernel's
  plain-Lanczos `1e-14` default and the distributed-Lanczos `1e-14`
  setting, all three of which are intentionally separate "what counts
  as a real invariant subspace" thresholds for their respective
  algorithms). `dim_cap = op.global_dim()` for the same small-problem
  / small-slab correctness reason that landed on day 8.
* `reorth_against` local helper in
  `src/distributed/distributed_krylov_schur.cpp` is gone (it was the
  hot path of the inline body's per-step orthogonalisation and is now
  replaced by the kernel's batched CGS2 over the combined ortho set).
* **Network-traffic improvement**: at `(|locked|, |basis|) = (k, m)`
  per Lanczos step the inline body cost
  `2 * (k + m)` *sequential* Allreduces. The kernel now does **2**
  batched Allreduces (one per CGS2 pass, payload `2 * (k + m)` doubles
  each). On a 4-rank build at the steady-state KS regime
  `(k, m) ≈ (10, 100)`, that's a `220 → 2` Allreduce reduction per
  step, i.e. the same headline speedup the day-1 batched-CGS2 work
  brought to the plain Lanczos path now extends to thick-restart KS.
* Test impact: `test_distributed_krylov_schur_symmetry` exercises both
  the `DistributedOperator` row-slab path AND the
  `DistributedSymmetryOperator` symmetry-projected path at np=1/2/4
  across every momentum sector of an N=4 and N=6 PBC Heisenberg chain
  (cross-checks every leading eigenvalue against the dense reference
  AND against `distributed_lanczos_symmetry` on the same operator).
  The N=6 case uses `exct=3`, which exercises multi-cycle locking
  (`aux_ortho_ptrs` accumulates 1, then 2, then 3 entries across
  restarts). All 30 assertions per process pass at every np.
* `tests/unit/test_lanczos_kernel.cpp` — new dedicated regression case
  for `aux_ortho_ptrs`. Two-pass design:
    1. Pass 1 runs `lanczos_kernel<CpuBackend>` on a 6-site Heisenberg
       chain and reconstructs the ground-state Ritz vector
       `y_0 = Σ_j S(j, 0) * V_j`. Sanity-checks `E_0` against the
       dense reference and `||y_0|| = 1`.
    2. Pass 2 runs the kernel from a different seed with
       `aux_ortho_ptrs = { y_0.data() }`. The caller-side
       pre-orthogonalisation of `v_0` against `y_0` is performed
       explicitly (matching the kernel's documented contract — the
       kernel deflates new Krylov vectors against the aux set during
       CGS2 but takes `v_0` as-is). The test then asserts
       (a) `max |<y_0, V_j>| < 1e-10` over the entire returned basis,
       (b) the lowest Ritz value of the deflated tridiagonal coincides
       with `E_1` of the dense reference (not `E_0` — that's the proof
       that the projection actually deflated the ground mode out).
  Adds 4 new assertions on the kernel proper, plus 3 on the
  reconstruction, totalling 7 new asserts and 1 new test case
  (143 total assertions across 7 test cases in `test_lanczos_kernel`).
* Documentation: tightened the contract on
  `LanczosKernelOptions::aux_ortho_ptrs` in `lanczos_kernel.h` to
  spell out the pre-orthogonalisation requirement on `v_0` (caught
  by the test above on the first pass).
* Full `ctest -j8` now reads **273/273 in 26.7 s** (was 272/272 before
  the new test), with all 33 MPI tests still passing.

#### Day 10 — pin the previously-untested `PeriodicCGS2` policy

* `LanczosKernelOptions::reorth == ReorthPolicy::PeriodicCGS2` is
  implemented in the kernel (the legacy `build_lanczos_tridiagonal_with_basis`
  body has its own MGS-once-with-tol-filter periodic path and no
  production caller has been switched onto the kernel's CGS2 periodic
  branch yet) but had **zero test coverage**. Pinned today via a
  two-section `TEST_CASE` in `tests/unit/test_lanczos_kernel.cpp`:
    1. `reorth_freq = 1` (fire every step) is **numerically equivalent
       to `FullCGS2`** on a 6-site PBC Heisenberg chain — same alpha,
       same beta, same Ritz spectrum to 1e-12. This is the strongest
       "the periodic code path actually runs reorth" assertion we can
       make without re-doing a separate orthogonality bound.
    2. `reorth_freq = 1000` with `max_iter = 40` (never fires within
       the run) is **strictly different** from `reorth_freq = 1`: the
       max off-diagonal basis overlap blows up to > 1e-4 (in practice
       O(0.1)+), proving the cadence gate is honoured and the kernel
       doesn't accidentally fall through to the full-reorth branch
       when the period is large.
  Adds 1 test case and 62 new assertions (`test_lanczos_kernel` now
  reads 8 test cases / 205 assertions, up from 7 / 143).
* Cleanup: removed the unused `LegacyResult` struct from
  `test_lanczos_kernel.cpp` (a now-orphan helper from the original
  Phase A comparison runs against the legacy MGS body — every
  remaining test uses `run_kernel(...)` directly or the legacy
  facade `build_lanczos_tridiagonal_with_basis`).
* Full `ctest -j4` now reads **274/274 in 24.4 s**, all 33 MPI tests
  passing at np=1/2/4.

#### Day 10 — extend `aux_ortho_ptrs` coverage to `CudaBackend`

* The `aux_ortho_ptrs` knob now has a **CUDA-backend regression test**
  in `tests/unit/test_cuda_backend.cpp`, mirroring the CPU-backend
  test added on day 9 but driving the deflation through cuBLAS:
    1. Pass 1 runs `lanczos_kernel<CudaBackend>` on a 6-site PBC
       Heisenberg chain (random `v_0` to span every Sz sector), pulls
       the tridiagonal back to host, diagonalises it with Eigen, then
       reconstructs the ground-state Ritz vector `y_0` **on device**
       via `CudaBackend::axpy_many(coeffs, basis_ptrs, ...)`. Sanity-
       checks `||y_0|| ≈ 1` via `CudaBackend::nrm2`.
    2. Pass 2 builds a different random `v_0`, performs the caller-
       side CGS2 pre-projection of `v_0` against `y_0` on device
       (mirroring the kernel's documented contract), then re-runs
       `lanczos_kernel<CudaBackend>` with
       `opts.aux_ortho_ptrs = { d_y_0.get() }`.
       Asserts `max |⟨y_0, V_j⟩| < 1 × 10⁻¹⁰` across the entire
       device-resident basis (cuBLAS reduction round-off included),
       and pins the deflated smallest Ritz value against the dense
       reference `E_1` (not `E_0` — proves the projection deflated
       the ground mode out on the GPU side too).
* Why this matters: the CPU and CUDA `Backend::dot_many` /
  `Backend::axpy_many` implementations are entirely different code
  paths (OpenMP-driven host loops vs cuBLAS via the default `dot`/
  `axpy` fallbacks). The CPU test verified the kernel's *control
  flow* around the aux set; this test additionally verifies the
  *device-side BLAS-1 chain* used to project against the aux vectors
  during every CGS2 pass.
* Diagnosis note: the first version of the test used `v_0 = |000…0⟩`
  (single-basis-state seed). On Heisenberg PBC that's a state in the
  1-dimensional `Sz = -N/2` sector, so `H v_0 = α_0 v_0` and the
  recurrence breaks down after one iteration. Switched to a random
  unit vector (the same `ed_tests::random_unit_vector` the CPU test
  uses); spans every Sz sector and gives the kernel a full Krylov
  subspace.
* `test_cuda_backend` now reads **8 test cases / 2102 assertions**
  (was 7 / 2090). Full `ctest -j4` is **275/275 in 28.8 s**, all 33
  MPI tests still passing at np=1/2/4.

#### Day 10 — pin previously-untested `convergence_check` callback

* `LanczosKernelOptions::convergence_check` is the kernel's early-exit
  hook (CPU+MPI distributed Lanczos wires it through
  `make_smallest_ritz_convergence(exct, tol)` to reproduce the legacy
  distributed Lanczos's relative-Δλ early-exit). Until today it had
  **zero focused unit-test coverage** — exercised transitively by the
  distributed Lanczos tests but never pinned for cadence / contract /
  early-exit semantics directly.
* New two-section `TEST_CASE` in `tests/unit/test_lanczos_kernel.cpp`:
    1. A counting probe (always returns `false`) verifies the kernel
       calls the callback at **exactly** `convergence_check_interval`
       cadence. With `interval=5` and `max_iter=22`, the probe is
       invoked at iters `j ∈ {4, 9, 14, 19}` (i.e. `(j+1) % 5 == 0`),
       and the kernel runs to its full cap (22 iters) since the
       callback never short-circuits. The probe also pins the
       `alpha.size() / beta.size()` *contract* at each call: when the
       callback fires the kernel has JUST pushed `alpha[j]` and
       `beta[j+1]`, so the sizes are `j+1` and `j+2` respectively.
       Off-by-ones here would cascade into wrong Ritz-residual
       arithmetic in `make_smallest_ritz_convergence`.
    2. The `make_smallest_ritz_convergence(1, 1e-8)` factory paired
       with `max_iter=200` on a 6-site PBC Heisenberg chain
       short-circuits **far** before the cap. Pinned: `iters_done <
       max_iter`, `iters_done >= 5` (at least one cadence hit),
       `basis.size() == iters_done` (no off-by-one between the
       early-exit decision and the basis-storage bookkeeping), and
       the converged E_0 matches the dense reference to the
       callback's tol.
* Adds 1 test case and 9 assertions. `test_lanczos_kernel` now reads
  **9 test cases / 214 assertions** (was 8 / 205). Full `ctest -j4`
  is **276/276 in 28.1 s**, all 33 MPI tests still passing.

#### Day 10 — remove the dead `LanczosKernelOptions::tol` field

* `LanczosKernelOptions::tol` was a leftover from before
  `convergence_check` was introduced — documented in-line as
  *"Reserved: Ritz convergence threshold"* but never actually
  consulted by the kernel body. Four call sites set it as a dead
  write:
    1. `tests/unit/test_lanczos_kernel.cpp` (two spots, in
       `run_kernel` and the trivial-breakdown test),
    2. `src/solvers/gpu/gpu_lanczos_kernel_facade.cu` (the eigvals /
       eigpairs facade — was being passed through from the public
       `tol` parameter of `runGPULanczos`),
    3. `src/solvers/cpu/lanczos.cpp::build_lanczos_tridiagonal_with_basis`
       (the unified-kernel fast path).
* Removed the field. Ritz-convergence handling is now exclusively
  the `convergence_check` callback's job (matches the kernel's
  documented semantics).
* The four dead writers no longer set `opts.tol`. The facade and
  legacy-ABI sites keep their *public* `tol` parameter intact — it's
  not piped through to the kernel today; if a caller wants Ritz
  early-exit, they wire up
  `opts.convergence_check = make_smallest_ritz_convergence(exct, tol)`
  the way `distributed_lanczos_kernel.h` does.
* The legacy CPU `build_lanczos_tridiagonal_with_basis` fast path
  previously *appeared* to honour `tol` via `opts.tol = tol` but
  silently dropped it (the kernel ignored the field). Behaviour is
  unchanged — explicit comments now document why.
* Net: 1 dead field removed from the kernel's public options surface,
  4 dead writes deleted. Full `ctest -j4` still **276/276 in 26.0 s**,
  all 33 MPI tests passing — no behavioural regression.

#### Day 10 — `block_lanczos_kernel` / `krylov_schur_kernel`: honest static-assert on the dishonest Backend template

* Before today, `block_lanczos_kernel<Backend, MatvecFn>` and
  `krylov_schur_kernel<Backend, MatvecFn>` accepted *any* Backend type
  but **silently ignored** it — the bodies unconditionally delegated
  to the CPU legacy `::block_lanczos` / `::krylov_schur`. The
  templates also carried doc comments claiming "GPU and MPI
  specialisations land alongside `CudaBackend` / `MpiBackend`" that
  never landed. A caller writing
  `block_lanczos_kernel(cuda_backend, ...)` would get a CPU run
  silently, expecting a GPU one — a footgun.
* Day-10 fix:
    1. Added `static_assert(std::is_base_of_v<ed::matvec::CpuBackend,
       std::decay_t<Backend>>, ...)` to both kernel facades. Now
       passing a `CudaBackend` (or any non-CpuBackend-derived type)
       gives a compile error with a clear message pointing to the
       BLAS-3 dependency and to
       `ed::distributed::distributed_krylov_schur` as the production
       CPU+MPI Krylov-Schur path.
    2. Rewrote the header doc comments to be **explicit** about the
       current scope ("CPU-only") and the design reason no GPU
       specialisation exists: `::block_lanczos` and `::krylov_schur`
       use BLAS-3 (`gemm`, `geqrf`) and Schur-reordering primitives
       (`dtrsen`-style) that the BLAS-1-only `ed::matvec::Backend`
       interface does not expose today. Tracked against the audit's
       "Phase B (block / KS GPU) — pending" rows.
* `ARCHITECTURE.md` row 4 ("block_lanczos / krylov_schur kernel
  headers") rewritten to match: "CPU-only facade, statically
  enforced". Cross-references to day 9's
  `distributed_krylov_schur.cpp` migration (the **real** CPU+MPI
  consolidation) and to `STRUCTURAL_AUDIT.md`'s
  "Phases B+D — what's now true" scoreboard.
* `test_kernel_facades` (the existing CpuBackend smoke test for both
  facades) still passes — the `static_assert` is a no-op there since
  CpuBackend satisfies the constraint.
* Net: 0 functional change, 1 footgun closed at compile time, 2 doc
  comments now reflect actual scope rather than aspirational scope.
  Full `ctest -j4` still **276/276 in 26.1 s**, all 33 MPI tests
  passing.

Cumulative scoreboard at end of day 9:

|                | Lanczos | Block Lanczos | Krylov-Schur |
|----------------|---------|---------------|--------------|
| CPU            | ✔       | ✔             | ✔            |
| GPU (1 device) | ✔ (eigvals + eigvecs through `lanczos_kernel<CudaBackend>` facade) | — | — |
| CPU + MPI      | ✔ (delegates to `lanczos_kernel<MpiBackend>`) | — | **✔ (day 9: per-cycle build delegates via `aux_ortho_ptrs`)** |
| GPU + MPI      | —       | —             | —            |

### Minimalist-architecture rollout — row-slab `distributed_lanczos(DistributedOperator&, ...)` is now a thin v0-scatter + delegate (May 2026, day 8, follow-on)

* `src/distributed/distributed_lanczos.cpp` — the row-slab entry point
  `distributed_lanczos(DistributedOperator&, options)` no longer carries
  its own ~200-LOC inline Lanczos body (three-term recurrence, batched
  CGS2 reorth, breakdown, relative-Δλ early-exit, tridiagonal solve,
  Ritz-vector storage). It now does just one thing: a row-slab-shaped
  v0 scatter via `scatter_initial_vector(...)` (deterministic from
  `options.seed`, balanced via `DistributedOperator::balanced_slab`),
  then hands off to the templated
  `kernel::distributed_lanczos_kernel<DistributedOperator>(op,
  std::move(v_local), options)`. That templated kernel was made into
  a thin facade over `lanczos_kernel<MpiBackend>` earlier in the same
  day. Net: one Lanczos body, two geometry-specific v0 scatters, four
  deployment targets.
* The anonymous-namespace dead-code in the TU has been pruned:
  `local_zdotc`, `local_axpy`, `dist_zdotc`, `dist_zdotc_batched`, and
  the three `solve_tridiag*` helpers all lived here pre-May-2026 as the
  hot path of the inline body. They are identically present (with
  internal linkage) in `distributed_lanczos_kernel.h`, where they
  remain in use; here they were just dead. `local_norm_sq`, `dist_norm`,
  and `local_scal` are kept because `scatter_initial_vector` still
  re-normalises locally + globally for numerical hygiene.
* File LOC: 665 → 310 (-355). The remaining lines are
  `scatter_initial_vector` (row-slab v0 scatter), the templated
  delegate, `reconstruct_local_eigenvector` (rank-local Ritz vector
  reconstruction — geometry-agnostic raw-index loop),
  `distributed_lanczos_eigenvectors` (convenience wrapper that calls
  the row-slab path with `compute_eigenvectors = true`), and
  `distributed_lanczos_symmetry` (geometry-specific orbit-partition v0
  scatter followed by the same templated delegate).
* No new tests. The migration is byte-for-byte preserving — the row-
  slab tests `test_distributed_lanczos`, `test_distributed_operator`,
  `test_distributed_eigenvectors`, `test_distributed_ftlm`,
  `test_distributed_tpq` all pass at np=1/2/4 with no source changes.
  Full `ctest -j8` is **272/272 in 26.6 s**, with all 33 MPI tests
  passing.

### Minimalist-architecture rollout — `distributed_lanczos_kernel` consolidated onto `lanczos_kernel<MpiBackend>`, two correctness bugfixes (May 2026, day 8)

Phase D ship — the templated CPU+MPI Lanczos kernel in
`include/ed/distributed/distributed_lanczos_kernel.h` no longer carries
its own inline three-term recurrence + batched CGS2 + relative-Δλ
early-exit. Its body now constructs an `ed::matvec::MpiBackend(comm)`,
sets up `LanczosKernelOptions`, and delegates to
`ed::krylov::lanczos_kernel<MpiBackend>`. The previous ~140-line inline
implementation was a near-line-for-line duplicate of the unified kernel;
this collapse removes the duplication that had been the open item from
the day-5 audit.

* `include/ed/krylov/ritz_convergence.h` — new header. Single inline
  factory `make_smallest_ritz_convergence(exct, tol, min_iters=0)` that
  builds a stateful predicate suitable for
  `LanczosKernelOptions::convergence_check`. The predicate solves the
  small running tridiagonal via Eigen on every call and returns `true`
  when `|Δλ_smallest| / max(|λ_smallest|, 1e-300) < tol` — the
  Lanczos-convergence convention shared by ARPACK / SLEPc / Anasazi
  and by every legacy ED Lanczos body in this repo (CPU, GPU, CPU+MPI).
  Keeping it in a dedicated header lets the kernel itself stay
  Eigen-free; callers who want the standard early-exit pull in this
  header explicitly.
* `include/ed/krylov/lanczos_kernel.h` —
    * Added `LanczosKernelOptions::convergence_check` (a `std::function`
      with the signature `bool(const alpha&, const beta&)`) and
      `convergence_check_interval`. The kernel invokes the callback
      after pushing `alpha[j] / beta[j+1]` and AFTER any reorth pass,
      every `convergence_check_interval` iterations. A positive return
      cleanly breaks out of the loop with the same post-state as a
      natural cap-hit (the current step IS counted in `iters_done`).
    * Added `LanczosKernelOptions::dim_cap` (default 0). The kernel's
      effective iteration cap is now
      `min(max_iter, dim_cap > 0 ? dim_cap : local_n)`. Serial / single-
      GPU backends leave it at 0 (correct: `local_n == global_dim`
      there); distributed backends MUST pass the global problem dim,
      because on a rank with a small slab (global dim 6 / np=4 gives
      `local_n ∈ {1,2}`) the previous `cap = min(max_iter, local_n)`
      logic was silently terminating after a single iteration — a
      ground-state-energy bug caught by
      `test_distributed_lanczos_symmetry_np4`.
    * Removed the `local_n == 0` early return. In distributed runs it
      is legitimate for a rank to receive an empty slab (e.g. LPT-
      balanced orbit partition on a small symmetry sector with
      `global_dim < n_ranks`), and that rank still has to participate
      in every collective (Allreduces inside `dot` / `nrm2` / `dot_many`
      / `axpy_many` etc.). The early return was deadlocking
      `test_distributed_ftlm_symmetry_np4` and
      `test_distributed_krylov_schur_symmetry_np4` because non-empty
      ranks proceeded into collectives the empty rank had skipped. The
      backend allocators all accept `n == 0` (returning nullptr); the
      BLAS-1 locals are length-respecting; the Allreduces have fixed
      payload sizes; so the kernel is now correct on empty slabs.
* `include/ed/distributed/distributed_lanczos_kernel.h` — kernel body
  collapsed onto the unified template. The new body:
    1. defensively re-normalises `v0_local` against its global L2 norm
       (preserves bit-exact starting vector across MPI runs),
    2. builds an `MpiBackend(op.comm())`,
    3. sets `LanczosKernelOptions` — `keep_basis` and `reorth` per the
       caller's `full_reorth / compute_eigenvectors`, `breakdown_tol =
       1e-14` (matching the historic looser bar of the inline body, NOT
       the kernel's default `1e-300`), `dim_cap = op.global_dim()`,
       `convergence_check_interval = 5` (the legacy cadence),
       `convergence_check = make_smallest_ritz_convergence(exct, tol)`,
    4. wraps `op.apply` in a backend-shaped matvec lambda,
    5. calls `ed::krylov::lanczos_kernel(mpi, ...)`,
    6. post-processes `LanczosKernelResult` into
       `DistributedLanczosResult` (`solve_tridiag` for eigvals only,
       `solve_tridiag_with_weights` for FTLM/cTPQ weights, or
       `solve_tridiag_with_eigenvectors` + `kres.basis` copy-out for
       the eigvec path). The retained orthonormal basis structure
       (`std::vector<std::vector<Complex>> krylov_basis_local`) matches
       the legacy ABI exactly so every existing caller — `FTLM_LTLM`,
       `distributed_krylov_schur`, `distributed_eigenvectors` — keeps
       working unchanged.
* `include/ed/matvec/backends/cpu_backend.h` — removed the `final`
  qualifier on `CpuBackend`. `MpiBackend` inherits from it to reuse the
  host allocators and BLAS-1 locals while overriding only the
  reduction-bearing primitives, and the `final` keyword (a copy-paste
  from `CudaBackend`) was blocking the day-8 distributed_lanczos
  migration as soon as `MpiBackend` was first actually instantiated.
* Test impact: full `ctest -j8` now finishes in 26.7 s with all 272
  tests passing (was 269/272 with a timeout on
  `test_distributed_lanczos_symmetry_np4` before the dim_cap fix, then
  270/272 with timeouts on the two FTLM/Krylov-Schur np=4 tests before
  the empty-slab fix). The 33 MPI tests all pass cleanly at np=1, np=2,
  and np=4. No new tests were needed for this migration — the existing
  `test_distributed_lanczos_symmetry`,
  `test_distributed_krylov_schur_symmetry`,
  `test_distributed_eigenvectors`, `test_distributed_ftlm_symmetry`,
  and `test_distributed_tpq_symmetry` already lock down the kernel's
  behaviour and they all still pass byte-for-byte.

Cumulative scoreboard at end of day 8 (Phase D shipped):

|                | Lanczos | Block Lanczos | Krylov-Schur |
|----------------|---------|---------------|--------------|
| CPU            | ✔       | ✔             | ✔            |
| GPU (1 device) | ✔ (eigvals + eigvecs through `lanczos_kernel<CudaBackend>` facade) | — | — |
| CPU + MPI      | ✔ (delegates to `lanczos_kernel<MpiBackend>`) | — | — |
| GPU + MPI      | —       | —             | —            |

Open items unchanged: Block-Lanczos and Krylov-Schur are still
hand-rolled per backend; the GPU+MPI grid cell remains empty.

### Minimalist-architecture rollout — GPU Lanczos eigvec path also unified, `CudaBackend` made movable (May 2026, day 7)

Day 6 routed the eigenvalues-only path of `runGPULanczos(...)` /
`runGPULanczosFixedSz(...)` through the `lanczos_kernel<CudaBackend>`
facade. Day 7 extends the facade with Ritz-vector reconstruction and
moves the eigenpair path onto the same unified kernel. The legacy
`GPULanczos::run` body is now a defensive fallback only — invoked
exclusively when the facade throws (e.g. because the basis won't fit
in device memory and the kernel's `keep_basis = true` assertion fires).

* `src/solvers/gpu/gpu_lanczos_kernel_facade.cu` — new entry point
  `run_lanczos_eigenpairs_kernel_facade(...)`. Strategy:
    1. drive the same shared kernel-driver (validates, allocates v0,
       runs curand init, runs `lanczos_kernel<CudaBackend>` with
       `keep_basis = true`),
    2. diagonalise the tridiagonal on the host via Eigen's
       `SelfAdjointEigenSolver` (one solve produces both eigenvalues
       and eigenvectors of T),
    3. for each requested Ritz index `i`, reconstruct
       `y_i = sum_k S(k, i) * V_k` by `M` backend axpys into a device
       scratch buffer (cuBLAS zaxpy under `CudaBackend`),
    4. copy each Ritz vector to host via `Backend::copy_to_host`.
  The host output type matches the legacy
  `std::vector<std::vector<std::complex<double>>>` exactly, so callers
  don't change. The shared preamble (input validation, curand v0 init,
  kernel-driver, tridiag solve) was hoisted out of the
  eigenvalues-only entry point so both paths now share ~70 LOC.
* `include/ed/matvec/backends/cuda_backend.cuh` — `CudaBackend` is now
  move-only (was move-deleted in day 5). The move ctor / move-assign
  transfer ownership of the `cublasHandle_t` and null out the source;
  the destructor already guarded on nullptr so moved-from objects
  destroy safely. Required to let internal helpers return a
  `CudaBackend` by value alongside a `LanczosKernelResult` (whose
  retained `basis` keeps backend memory alive through the Ritz
  reconstruction).
* `src/solvers/gpu/gpu_ed_wrapper.cu` — `runGPULanczos` /
  `runGPULanczosFixedSz` now route both branches through the
  facade in a single try/catch. Eigvec branch dispatches to
  `run_lanczos_eigenpairs_kernel_facade`, eigvals-only branch keeps
  `run_lanczos_eigenvalues_kernel_facade`. The legacy `GPULanczos`
  class is reached only inside the catch handler. Same HDF5 save +
  throughput-stats output shape as before.
* `tests/unit/test_cuda_backend.cpp` — new `TEST_CASE` exercises the
  eigenpair facade on an 8-site periodic Heisenberg ring:
    1. pins the recovered Ritz eigenvalue to `<1e-8` of the CPU
       `lanczos(...)` reference,
    2. checks `|| y_0 || == 1` to within 1e-8 (orthonormal Krylov
       basis * orthonormal column of S),
    3. checks the residual `|| H y_0 - lambda_0 y_0 || < 1e-6` on the
       full Hilbert space (the strongest test — proves the recovered
       Ritz pair really satisfies the eigenvalue equation).

**Test posture:** 270/270 ctest green (was 269 / day 6). `test_cuda_backend`
reports 2070 passing assertions across 5 test cases.

### Minimalist-architecture rollout — first GPU production migration onto `lanczos_kernel<CudaBackend>` (May 2026, day 6)

The unified Krylov kernel was proven backend-agnostic at the end of day 5
(`test_cuda_backend`: bit-for-bit `(alpha, beta)` agreement between
`CpuBackend` and `CudaBackend`). Day 6 moves the unified kernel into
**production**: the eigenvalues-only path of `runGPULanczos(...)` and
`runGPULanczosFixedSz(...)` now dispatches into a thin facade onto
`ed::krylov::lanczos_kernel<CudaBackend>` instead of the legacy 1099-LOC
`GPULanczos::run` body. Every existing call site picks up the new path
without a source change because the dispatch happens behind the same
`runGPULanczos` signature.

* `src/solvers/gpu/gpu_lanczos_kernel_facade.cu` — new ~170-LOC
  translation unit owning the migration. It:
    1. allocates `v0` on the GPU via `CudaBackend::make_zero_vector`,
    2. initialises it with the same curand-based real-only Gaussian the
       legacy `GPULanczos::initializeRandomVector` uses (so seeds
       reproduce across both paths),
    3. drives `lanczos_kernel<Backend>` with a matvec callable that
       forwards into `GPUOperator::matVecGPU` (works through the
       `GPUFixedSzOperator` override automatically — no source change
       to the fixed-Sz path),
    4. diagonalises the small real-symmetric tridiagonal on the host
       via `Eigen::SelfAdjointEigenSolver`, and
    5. hands back the lowest `num_eigs` Ritz eigenvalues.
* `include/ed/gpu/gpu_solvers.h` — forward-declares the facade entry
  point; existing `ed::matvec::gpu::lanczos(...)` overloads unchanged
  (they still go through `GPUEDWrapper::runGPULanczos` and pick up the
  new path transparently).
* `src/solvers/gpu/gpu_ed_wrapper.cu` — `runGPULanczos` /
  `runGPULanczosFixedSz` now dispatch on `eigenvectors`:
    * **`eigenvectors == false`**: facade path
      (`lanczos_kernel<CudaBackend>`). The common case; this is what
      `ed_wrapper.h` and `ed_wrapper_streaming.h` invoke from
      `exact_diagonalization_core` for the LANCZOS branch when
      `params.compute_eigenvectors == false` (the default).
    * **`eigenvectors == true`**: legacy `GPULanczos::run`. Keeps the
      Ritz reconstruction + windowed reorth + on-disk basis spill +
      early-eigenvalue-convergence machinery for callers that need any
      of those.
  A try/catch around the facade falls back to `GPULanczos::run` on
  failure (defensive — currently uncovered, but it keeps the change
  zero-risk for production runs while the migration bakes in).
* `tests/unit/test_cuda_backend.cpp` — two new `TEST_CASE`s exercise the
  facade directly:
    1. End-to-end on an 8-site periodic Heisenberg ring: pinned ground-
       state energy at 1e-8 vs the legacy CPU `lanczos(...)` reference
       (matches the established `test_cpu_gpu_equivalence` tolerance).
    2. Input validation — rejects `N <= 0`, `max_iter <= 0`,
       `num_eigs <= 0` with `std::invalid_argument`.

The legacy `GPULanczos` class is **not** retired in this drop: it owns
the Ritz reconstruction and the disk-spill regime, which the unified
kernel doesn't yet support. It is now strictly the eigvec-path
implementation, which removes another reason for it to grow.

**Test posture:** 267/267 ctest green; `test_cuda_backend` reports
2063 passing assertions across the 4 CUDA-only test cases (up from
2060 / 2 cases). `test_cpu_gpu_equivalence` still passes — the two
Phase-4 GPU overload tests now drive the facade transparently through
`GPUEDWrapper::runGPULanczos` and match the legacy `GPULanczos` class
at 1e-8.

### Minimalist-architecture rollout — Gap 2 / Gap 3 / `ed_wrapper` sweep (May 2026, day 5)

Continuation of the "swing at all of them" pass on the architectural
gaps documented in `ARCHITECTURE.md`. Three independent landings, each
self-contained:

**Gap 2 — `CudaBackend` lands.** `include/ed/matvec/backends/cuda_backend.cuh`
is the cuBLAS-driven sibling of `CpuBackend` / `MpiBackend`. RAII over a
`cublasHandle_t` in `HOST` pointer mode; `allocate`/`fill_zero`/`copy*`
route through `cudaMalloc` / `cudaMemset` / `cudaMemcpy` with the
correct direction; `axpy` / `scale` / `dot` / `nrm2` hit `cublasZaxpy`
/ `cublasZscal` / `cublasZdotc` / `cublasDznrm2` directly;
`std::complex<double>*` reinterprets to `cuDoubleComplex*` (binary
layout compat is part of the CUDA spec). `axpby` is a two-call
decomposition (`scale` then `axpy`); `dot_many` / `axpy_many` use the
inherited single-call defaults pending a fused-kernel follow-up. New
`test_cuda_backend` (`tests/unit/test_cuda_backend.cpp`,
`USE_CATCH2`, CUDA-only) pins:
  * BLAS-1 round-trip (nrm2, dot, axpy, scale) against pure-host
    reference vectors at 1e-10 relative.
  * Per-iteration `(alpha, beta)` agreement of
    `lanczos_kernel<Backend>` between `CpuBackend` and `CudaBackend`
    on a 6-site periodic Heisenberg ring at ~1e-10 over a 24-step
    Krylov subspace. Both backends drive the same kernel body — first
    direct proof that `ed::krylov::lanczos_kernel` is genuinely
    backend-agnostic and that the unified Krylov kernel runs on real
    GPU hardware end-to-end.

**Gap 3 — `compute_*_workflow` preamble factored.** Each of
`compute_dynamical_response_workflow`,
`compute_static_response_workflow`,
`compute_ground_state_dssf_workflow`,
`compute_kpm_thermodynamics_workflow` opened with ~50 lines of
near-identical boilerplate (MPI rank/size init, audit #2 fixed-Sz vs
full-Hilbert shared_ptr dispatch, three-body load, sector dim, `H_func`
lambda). Replaced by two helpers in the file's anonymous namespace:
`get_mpi_rank_size_safe()` (10 lines) and
`build_workflow_hamiltonian(config, rank, verbose_label)` ->
`WorkflowHamiltonian` POD (~70 lines). Workflows now open with four
lines of state extraction. **Net: workflows.cpp shrinks from 2987
to 2875 lines (~112 LOC).** Behaviour is preserved bit-for-bit —
the helper carries the same audit #2 dispatch and the same Hilbert-dim
binomial computation; only the spelling collapses.

**Gap 1 / Gap 5 — `ed_wrapper.h` + `gpu_ed_wrapper.cu` dead-code sweep.**
  * `ed_internal::create_operator<>` template factory (raw-new'd Operator*)
    retired — zero callers, wrong ownership convention (everything else
    uses `std::shared_ptr` pairs).
  * `ed_internal::supports_fixed_sz(method)` retired — after Phase 1
    every remaining method supports fixed-Sz, so the helper always
    returned true and its single caller was a dead-branch guard.
  * `gpu_ed_wrapper.cu` `#else // !WITH_CUDA` stub block (~178 lines of
    no-op definitions covering every public `GPUEDWrapper::*` entry
    point) deleted. It was unreachable: the translation unit is only
    added to the build under `if(WITH_CUDA)`
    (`cmake/EDLibraries.cmake:566`), and every `GPUEDWrapper::*`
    callsite in `ed_wrapper.h` / `ed_wrapper_streaming.h` is itself
    gated by `#ifdef WITH_CUDA`. An `#error` directive now catches
    accidental inclusion in a non-CUDA build instead of silently
    producing a library full of no-op symbols.

**Test posture:** 267/267 green (was 265/265 before this drop;
`test_cuda_backend` adds two CUDA-only TEST_CASEs that SKIP when no
GPU is attached, matching the established `test_cpu_gpu_equivalence`
pattern).

### Krylov-kernel unification — Phase A (May 2026, day 3)

The four Lanczos paths (CPU, single GPU, CPU+MPI, GPU+MPI) had been near-
identical bodies with three reorth strategies and two convergence
criteria between them — ~10000 lines of code that always agreed on the
algorithm but never on the spelling. Phase A introduces the single
`ed::krylov::lanczos_kernel<MatvecFn>(const Backend&, MatvecFn, ...)`
template that will drive all four targets, and lands the CPU + CPU+MPI
backends behind it. Single-GPU and GPU+MPI backends (Phases B–C) will
slot in under the same kernel; the CPU and GPU+MPI distributed-Lanczos
.cpp/.cu bodies become facades in Phase D.

**Net delivered in Phase A:**

- New `ed::matvec::Backend::dot_many` / `axpy_many` batched primitives
  (with a default loop-over-singles fallback so pre-existing concrete
  backends compile unchanged). `CpuBackend` specialises both with a
  single OpenMP region that amortises the cache-residency of `v`
  across all M inner products.
- New `ed::matvec::MpiBackend` (`include/ed/matvec/backends/mpi_backend.h`):
  inherits CpuBackend, overrides reduction-bearing primitives with
  `MPI_Allreduce`, and collapses M sequential Allreduces into one
  batched 2M-double Allreduce inside `dot_many` (CGS2 fast path).
- New `ed::krylov::lanczos_kernel` (`include/ed/krylov/lanczos_kernel.h`):
  one Lanczos algorithm body — three-term recurrence with swap-rotated
  working vectors, CGS2 reorth (two batched-dot+batched-axpy passes —
  one Allreduce per pass in MPI), genuine-invariant-subspace breakdown
  detection (decoupled from the user-facing Ritz tol — the legacy
  conflation broke at the CGS2 noise floor and tripped LTLM-static at
  full Krylov), optional basis retention.
- Legacy `build_lanczos_tridiagonal_with_basis(..., full_reorth=true,
  basis_vectors != nullptr)` routes through the kernel. Every CPU
  consumer (FTLM, LTLM, Hybrid-thermal, Lanczos itself) inherits CGS2
  reorth with no callsite change. The legacy three-vector / periodic-
  reorth branches stay as-is for callers that opt out.
- Six new regression test cases (130 assertions) in
  `tests/unit/test_lanczos_kernel.cpp` pinning:
  * alpha/beta agreement with the legacy MGS-once body to 1e-10 at
    partial Krylov,
  * ground-state floor matches the dense reference,
  * basis orthogonality `||V^H V - I||_∞ < 1e-10` (CGS2 guarantee),
  * tridiagonal agreement on the physically meaningful half at full
    Krylov `M=N=dim` (CGS2 and MGS diverge in the noise-dominated tail,
    by design),
  * trivial 1-D breakdown (alpha=0, beta=0, iters_done=1),
  * misuse rejection (zero v0, reorth without keep_basis).

**Performance signature (CPU FTLM at M=100):** reorth-step DRAM traffic
drops from O(M²) streaming passes over `w` to O(2M); the full step
becomes matvec-dominated again. **MPI:** Allreduce count per Lanczos
step drops from ~M (one per basis vector) to 2 (one per CGS2 pass).

**Test surface: 308/308** (up from 302).

### Structural-audit second round (May 2026, day 2)

A second-round audit was performed on the layers the first audit did not
cover in depth: HDF5 / IO, Python bindings, distributed-Lanczos
convergence math, and the EDConfig↔EDParameters adapter. Eleven
additional S0/S1 items were surfaced and fixed; the deferred items are
documented with rationale in `STRUCTURAL_AUDIT.md` Part III.

**Test surface: 302/302** (up from 296), with six new lock-in
regression tests covering the most impactful fixes.

**HDF5 / IO layer (S0 fixes).**

- `HDF5SymmetryIO::loadBasisVector` (`include/ed/core/hdf5_symmetry_io.h`)
  now bounds-checks every sparse element index against the declared
  `dimension` and throws on violation. The pre-fix code could heap-
  overrun the dense output vector on a corrupt or hand-edited file.
- `HDF5IO::loadTPQThermodynamics` / `loadTPQNorm` (`hdf5_io.h`) now
  validate the dataset rank (must be 2) and exact column count (5 for
  thermo, 4 for norm) before unpacking row-by-row. The pre-fix code
  trusted whatever shape was on disk and would silently shuffle columns
  if a producer wrote a 4-column thermo dataset. HDF5 exceptions on
  real reads are now rethrown with `file:line` context; the
  iterate-until-empty pattern (callers walking `sample_index = 0, 1,
  …`) is preserved by catching the `nameExists`-on-missing-group case
  explicitly and returning an empty vector.
- `HDF5IO::saveCorrelationMatrix` (`hdf5_io.h`) explicitly rejects
  empty matrices (previously UB on `matrix[0]`) and jagged matrices
  (rows of unequal length).
- `HDF5IO::saveThermodynamics` (`hdf5_io.h`) hard-checks that
  `temperatures.size() == values.size()` before writing.
- `HDF5SymmetryIO::loadSectorDimensions` (`hdf5_symmetry_io.h`)
  cross-checks the `num_sectors` attribute against the dataset extent
  and throws on mismatch (was: blindly read whatever the attribute
  said, even if the dataset had a different length).
- `BasisVectorStorage::read_vector` (`src/io/basis_vector_storage.cpp`)
  now verifies the on-disk dataset is exactly `[dim, 2]` before
  reading; previously a wrong-`N` caller would silently get garbage.
- New regression tests in `tests/unit/test_hdf5_io.cpp`
  `[regression][s0]`.

**Python bindings (S0 fixes).**

- `auto_tune.estimate_bandwidth` (`python/qed/auto_tune.py`) rewritten
  to walk the exposed `iter_one_body_terms` / `iter_two_body_terms`
  / `iter_three_body_terms` methods on the bound `Operator`. The
  previous implementation probed the private C++ field
  `transform_data_` which pybind never exposes, so every bound
  `Operator` silently fell back to `fallback * num_sites` regardless
  of its coupling magnitudes — DSSF auto-tune was picking η / ω /
  Krylov against a bandwidth that ignored every coefficient. New
  regression test in `python/tests/test_auto_tune.py`.
- `op_add_one_body` / `op_add_two_body` / `op_add_three_body`
  (`qed_bindings.cpp`) now call `op.invalidateMatrixCaches()` after
  pushing into `transform_data_` / `three_body_data_`. Without this,
  the `isReal()` cache stayed at its first-evaluated value, so a
  real-built operator that later gained a complex term would still
  report real, routing `lanczos()` through the `lanczos_real` fast
  path with the wrong matvec. Regression test in
  `tests/unit/test_operator_apply.cpp` `[regression][s0]`.
- BFG correlation kernels (`compute_smsp_correlations`,
  `compute_szsz_correlations`, three `*_bond_expectations`, all in
  `qed_bindings.cpp`) now validate `psi.shape[0]` against
  `2^n_sites` (or `2^cluster.n_sites` for the cluster-taking
  variants) via a shared `bfg_check_psi` lambda. Previously a
  wrong-length array would `memcpy` into a too-small `std::vector`
  and then the C++ matvec would read past the end.

**Distributed Lanczos (S0 fix).**

- `distributed_lanczos.cpp` and `distributed_lanczos_kernel.h` now use
  the same relative-Δλ / max(|λ|, 1e-300) convergence criterion as
  the serial `lanczos()` kernel. The pre-fix absolute-Δλ test stopped
  too early for large `|λ|` (e.g. systems with a non-trivial energy
  shift) and never converged for very small `|λ|`. CPU-distributed
  Lanczos now agrees with serial Lanczos on the convergence contract.
  The GPU-distributed Lanczos path retains its own check (see
  STRUCTURAL_AUDIT.md D2/D3 for the deferred GPU-side parity work).

**EDConfig↔EDParameters adapter (S0 fixes).**

- `EDConfig::ltlm_full_reorth` flipped from `false` to `true`
  (`ed_config.h`). The pre-fix value contradicted
  `EDParameters::ltlm_full_reorth = true` and silently disabled LTLM
  reorth on the CLI / config-file path, regressing the May-2026
  S1 #19 default fix for any user going through `EDConfig`.
- `tpq_max_steps` and `tpq_beta_max` now propagate in both
  directions through the adapter (`ed_config_adapter.h`). Pre-fix
  these were parsed into `ThermalConfig` (so config-file users
  thought they were setting them) but never copied to
  `EDParameters` — so the dispatcher passed `params.max_iterations`
  to `microcanonical_tpq` regardless and `cTPQ` took its `beta_max`
  from `temp_max` regardless. The dispatcher
  (`ed_wrapper.h::exact_diagonalization_core`) now honours
  `tpq_max_steps > 0` as a hard `min(...)` cap on mTPQ
  `max_iterations`, and uses `tpq_beta_max > 0` as the cTPQ
  `beta_max` override over the generic `temp_max`.
- `fromEDParameters` reverse-direction adapter gained the
  previously-dropped fields `hybrid_auto_crossover`,
  `selected_sectors`, `translation_only`, `tpq_max_steps`, and
  `tpq_beta_max` — round-tripping `EDParameters → EDConfig →
  EDParameters` is now identity-preserving for those knobs.
- Adapter round-trip regression tests in
  `tests/unit/test_method_canonicalize.cpp` `[regression][s0]`.

**Auto-pilot completeness (S1 fix).**

- `apply_auto_tune` (`include/ed/auto/diag_tune.h`) now wires
  `pick_num_thermal_samples` for the thermal path: when
  `params.num_samples == 1` (the EDParameters default for
  ground-state solvers) AND the new `num_samples_locked` flag is
  unset, the picker fills in a sector-dim-appropriate sample count.
  `ThermalOptions::make_auto_tune_overrides` sets the lock so that
  explicit `ThermalOptions::num_samples` values (default 40) survive
  unchanged. The C++ thermal auto-pilot now agrees with the
  Python `qed.auto_tune.pick_num_thermal_samples` it had been
  contradicting since the picker was introduced. New regression
  test in `tests/unit/test_diag_tune.cpp` `[regression][s1]`.

**Deferred (with rationale in STRUCTURAL_AUDIT.md Part III).**

- HDF5 schema fragmentation (3 complex-vector encodings, dual DSSF
  schema) — needs a versioned schema work-stream.
- `appendTPQThermodynamics` / `appendTPQNorm` O(n²) per-row scan —
  needs a per-row append API.
- MPI per-rank merge soft-fail (`copyTPQSamples`,
  `mergePerRankTPQFiles`) — needs a per-rank error aggregator.
- `hdf5_io.h` size (3506 lines, now the largest header) — same
  workstream as the `ed_wrapper.h` split.
- DSSF auto-pilot architectural fragmentation (Python subprocess CLI
  vs C++ in-process auto-pilot vs raw CLI) — needs either binding
  `auto_pilot::dssf::compute` or routing the CLI through it.
- `qed.diag` auto-tune partial vs C++ `apply_auto_tune` — Python-
  side refactor to match the lock-flag surface.
- MPI device-picker gate (`has_mpi_build` vs `is_scalapack_compiled`)
  divergence — shared compile-time probe.
- GPU distributed Lanczos no-reorth + every-iter convergence — the
  shipped behaviour is feature-reduced by design; full reorth +
  serial-parity convergence is its own work-stream.
- `EDConfig::merge` stub — coordinated with the bag-of-parameters
  unification rev.
- KPM / ScaLAPACK-mixed / DSSF-block fields on `EDConfig` — these
  have no home on `EDConfig` today; needs structural extension.

---

### Structural-audit final follow-on (May 2026)

This entry catalogues the remaining S1/S2 work items beyond the May-2026
audit roll-out and the symmetry-SOTA pass. Test surface: **296/296**
(up from 293), with new lock-in coverage for the GPU FP32 CSR
invalidation, all six auto-tune `_locked` flags, and the deprecated
public-API attributes. `docs/architecture/STRUCTURAL_AUDIT.md` reflects
the new state. **Every** correctness-class item is now either fixed
or has an explicit deferral rationale; the only outstanding work is
multi-day refactor / API-rev material.

**S1 #10 — Single `apply` surface on `DistributedOperator`.**
Introduced a private `apply_local_` that owns the canonical SpMV
body; the 2-arg `apply(v, y)` (legacy hot path) and the 3-arg
`apply(v, y, size)` (MatVecOperator polymorphic surface) both forward
to it. The 3-arg form additionally runs `check_size`; the 2-arg form
trusts the caller-supplied buffer sizes (preserving the legacy
fast-path contract). Stale comments referencing the removed
`Operator::diag_one_body_` member updated to point at the SoA
`terms_` storage and `commitPendingTransforms`.

**S1 #19 + #20 — Krylov tridiagonal builder consolidation.**
`ftlm.cpp:build_lanczos_tridiagonal` is now a thin forwarder to
`lanczos.cpp:build_lanczos_tridiagonal_with_basis`. It allocates a
local basis store when reorthogonalisation is requested and delegates.
One canonical three-term recurrence + reorth body + breakdown check,
shared by LTLM, FTLM, HYBRID and KPM. Also flipped the standalone
solver-struct defaults to match the dispatcher: `FTLMParameters`,
`LTLMParameters`, `StaticResponseParameters`,
`DynamicalResponseParameters` `full_reorthogonalization` is now
`true` (the per-`EDParameters` defaults were already `true`, so this
closes the gap for direct callers of the solver structs --- Python
bindings, examples).

**S1 #21 — GPU determinism + windowed-reorth disclosure.**
Added `EDParameters::lanczos_seed` (default 0 = legacy deterministic
seed 42; nonzero passes through verbatim so a GPU run can reproduce a
CPU run with the same seed). `GPULanczos` gained a public
`setSeed(uint64_t)` accessor; `GPUEDWrapper::runGPULanczos` and
`runGPULanczosFixedSz` both pass the new seed through. When the
device-memory budget allows storing fewer Lanczos vectors than
`max_iter`, the kernel now emits a one-line stderr warning naming the
regime change (**windowed** reorth vs the CPU default of **full**
reorth); the same path also warns when memory is exhausted and reorth
is skipped entirely. The `GPUOperator::apply` `const_cast` cache hazard
(concurrent const applies on the same operator are UB) is documented
but not enforced --- the fix requires a mutex per call which would
hurt SpMV throughput. Acceptable until a real concurrent use case
appears.

**S1 #22 — Full auto-tune lock-flag plumbing.**
Extended the `tolerance_locked` pattern (added in the May roll-out) to
every other auto-tunable field: `max_iterations`, `max_subspace`,
`ftlm_krylov_dim`, `ltlm_krylov_dim`, `tpq_delta_beta`,
`tpq_taylor_order`. The corresponding fields on `AutoSolveOptions` and
`ThermalOptions` are now `std::optional<T>`; setting them propagates
both the value and the lock flag to `apply_auto_tune`, so passing a
value that happens to equal the struct default is no longer silently
retuned. Fixed a propagation bug for `tpq_energy_shift`:
`base_params_from_options` used to gate the propagation on `> 0.0`, so
the user-facing `0` sentinel (which means "auto-pick") never reached
the dispatcher --- the old code silently kept the `EDParameters`
default `1e5` and mTPQ auto-pick never fired through `thermal()`. New
helper `make_auto_tune_overrides` centralises the
`std::optional<T> → AutoTuneOverrides::*_locked` translation across
the four `thermal()` auto-tune call sites and the `solve()` site.

**S1 #24 — Dead-code removal.** Deleted
`estimate_extreme_eigenvalues` and `auto_tpq_energy_shift` (~35 lines)
from `include/ed/auto/thermal.h`. The mTPQ `LargeValue` auto-pick lives
inside `exact_diagonalization_core` (a 24-iter Lanczos at the dispatch
site) and is now the only path. A short historical comment in
`thermal.h` records where the API went so future maintainers don't
reinvent it.

**S1 #25 — CPU/GPU KPM reorth parity.**
`estimate_spectral_bounds_gpu` (`src/solvers/gpu/kpm_dos_gpu.cu`) now
takes a `bool full_reorth` argument that mirrors the CPU default. When
the caller requests it AND the saved Krylov basis fits on-device, the
GPU spectral-bound Lanczos does full classical Gram-Schmidt against
its retained basis on every step. Falls back to the 3-vector path with
a stderr warning when memory is short, so the regime change is no
longer silent. Caller (`kpm_dos_gpu`) passes
`params.full_reorthogonalization` so the CPU and GPU code paths agree
by default.

**S1 #27 — `ed_log::warning` routing for silent-fallback warnings.**
The two silent-fallback paths that a `setVerbosity(SILENT)` caller
could not previously suppress now route through `ed_log::warning`: the
in-memory GPU CPU-fallback warning in `exact_diagonalization_core`
(`ed_wrapper.h`) and the sector-thermo recombine failure in
`ed_wrapper_streaming.h` (both the Sz+symmetry and the symmetry-only
branches). The auto-pilot diagnostic streams (already gated on
`opts.verbose`) remain on raw `std::cerr` since they are explicitly
user-opt-in. The banner-style `std::cout` blocks in
`ed_wrapper.h` / `workflows.cpp` / `lanczos.cpp` progress prints are
deferred to a per-banner pass.

**S0 #3 — GPU FP32 CSR cache regression test.** Locked in the
``GPUOperator::invalidateDerivedCaches()`` →
``freeCsrFp32DeviceData()`` chain with a new
`tests/unit/test_gpu_mixed_precision_spmv.cpp` case that mutates the
term list between two FP32 matvecs and asserts both that the
post-mutation matvec **differs measurably** from the pre-mutation one
(the cache was actually rebuilt) AND that it **matches the CPU
reference** within FP32 tolerance. Builds cleanly under `WITH_CUDA`;
skips at runtime when no GPU is present.

**S2 #28 — Public-mutable-state guidance.** Added explicit API guidance
in `include/ed/core/operator.h` pointing direct-AoS-write callers at
the typed setters (`addOneBodyTerm` / `addTwoBodyTerm` /
`addThreeBodyTerm`) and at `commitPendingTransforms` for the existing
direct-push pattern. Migration of `transform_data_` /
`three_body_data_` / `terms_` to `protected:` with typed accessors is
deferred --- an in-tree scan found ≈20 external callers, and the
move is a one-release deprecation cycle best coordinated with the
matvec-API revision. The size-aware `commitPendingTransforms` from
S0 #2 ensures direct pushes are safe in the meantime; the
Python-binding cache bug that motivated this is fixed at the root.

**S2 #29 — Deprecation markers on dead public API.** Applied
`[[deprecated("...")]]` with explicit replacement pointers to every
dead-in-tree symbol surfaced by the audit:
`CrossSectorMatVecOperator`, `MatVecOperator::nnz_per_row_estimate`,
`Operator::getSparseMatrix`, `Operator::getTransformData`,
`Operator::getTerms`, `FixedSzOperator::binarySearchState`,
`OperatorRef`, `ed::matvec::adapt`. Callers will get a compile-time
warning; one-release sunset window. `lanczos_real` is kept
undeprecated because `qed_bindings.cpp` (the Python facade) still uses
it --- the audit's "no dispatch caller" finding was technically right
but missed the Python surface.

**S2 #34 — Stale-comment cleanup.** Fixed the four documented
stale-comment cases in `distributed_operator.h`, `auto/solve.h`,
`core/ed_types.h` (`LOBPCG_GPU` deprecation note), and
`core/ed_method_traits.h` (`normalize_method` scope vs
`canonicalize_method_and_flags`). Also clarified the cost docstring
of `ed::matvec::as_apply_function` to make the "one virtual call, no
extra `std::function` allocation" claim precise.

**Deferred (now documented with rationale).**
- **S1 #6 / #7 / #8 / #33** — moving the four giant headers
  (`ed_wrapper.h`, `streaming_symmetry.h`, `ed_wrapper_streaming.h`,
  `workflows.cpp`, total ~9700 lines) into `.cpp` files. Multi-day
  refactor; not a correctness blocker; should follow a build-time
  baseline measurement.
- **S1 #9** — `DistributedGPUOperator` and `DistributedSymmetryOperatorGPU`
  escape the `MatVecOperator` hierarchy because they need an
  `(MPI_Comm, cudaStream, device-pointer)` signature that the current
  base class can't express. Adding a `DistributedMatVecOperator` base
  is the right call but a meaningful API design rev.
- **S1 #32** — migrating CPU solver `.cpp` files from
  `std::function<void(const Complex*, Complex*, int)>` to
  `const MatVecOperator&` would touch ~4000 lines for zero measurable
  perf delta (the bridge adapter is one indirection on top of an
  inherent virtual call). The clean refactor needs a
  templated-solver-vs-polymorphic-ref decision which is a separate
  API design conversation.
- **S1 #37** — DSSF + spatial symmetry. The reusable pieces are in
  place; the missing pieces are a `CrossSectorOrbitObservable`
  rectangular operator in irrep basis and workflow glue that resolves
  the target sector for each `(Q, dn_up)` pair. Documented in
  `docs/architecture/SYMMETRY.md` §3 as the highest-ROI follow-up
  workstream.

---

### Symmetry SOTA pass: TPQ + spatial symmetry; DSSF gap documented (May 2026)

Follow-on to the structural-audit roll-out, scoped specifically to the
symmetry pipeline for the three primary workflows. Audit document:
`docs/architecture/SYMMETRY.md`. Test surface: **294/294** (up from
293), including a new exact-thermo physics validation on a 4-site
Heisenberg chain.

**TPQ + spatial symmetry enabled (`include/ed/auto/thermal.h`).** The
auto-pilot finite-T entry point used to skip spatial-irrep
decomposition for the TPQ family (mTPQ, cTPQ, and their GPU variants),
on the (incorrect-as-stated) grounds that "TPQ runs a single random
state and does not factor cleanly through spatial irreps". The fix
removes the `!tpq_family` exclusion and routes TPQ through the same
`(Sz, irrep)` streaming-kernel loop that FTLM/LTLM/HYBRID/KPM_DOS use,
followed by Z-weighted recombination via
`ed::core::combine_sector_thermodynamics`. The math is identical to
the FTLM case: each `(Sz, irrep)` sector's random vector
`|psi_R^s>` drives a valid mTPQ chain whose final-state
trajectory measures the sector's partition function, and
`Z(beta) = sum_{n_up, k} Z_{n_up,k}(beta)`. The streaming kernel was
already configured to allocate per-sector `output_dir` paths for the
HDF5 trajectory files; no other change was needed. As a side effect,
TPQ on systems with small spatial sectors becomes **more** accurate
than no-symmetry TPQ -- many `(Sz, irrep)` sectors are dim=1 and hit
the exact dispatcher short-circuit, eliminating their statistical
noise entirely.

**Tests added (`tests/unit/test_auto_thermal.cpp`).**

- `[auto_pilot][thermal][tpq][symmetry][sota]` — TPQ exploits spatial
  symmetry when fixtures are present (replaces the inverted
  "falls back" test which encoded the now-incorrect behaviour).
- `[auto_pilot][thermal][tpq][symmetry][sota][regression][physics]` —
  TPQ + symmetry recombination tracks exact F(T) on a 4-site Heisenberg
  chain (the exact spectrum is computed by full diagonalization and
  the recombined F is checked against the truth at every T bin).

**DSSF + spatial symmetry: gap documented as the next workstream.** The
`compute_dynamical_response_workflow`, `compute_static_response_workflow`,
and `compute_ground_state_dssf_workflow` paths in `src/cli/workflows.cpp`
exploit Sz (including Sz-cross-sector via `CrossSectorObservable`) but
do **not** yet decompose by spatial irrep. SOTA codes (HPhi, EDLib,
QuSpin) route a momentum-resolved operator `O_Q` from source sector
`(n_up_0, k_0)` to target sector `(n_up_0 + dn, k_0 + Q)` and run
double-Lanczos in the target sector basis. The reusable pieces are
already in place (`SymmetrizedHamiltonian` per-sector orbit bases +
`applySymmetrized*`, the `compute_ltlm_dynamical_correlation_cross_sector`
solver that accepts arbitrary `(H_outer, H_inner, O1, O2)` callbacks);
the missing pieces are (1) a `CrossSectorOrbitObservable` rectangular
operator in irrep basis and (2) workflow glue that resolves the target
sector index for each `(Q, dn_up)` pair. This is tracked as
`STRUCTURAL_AUDIT.md` S1 #37 and explained in detail in
`docs/architecture/SYMMETRY.md` §3.

### Structural-audit S0 / S1 roll-out (May 2026)

End-to-end fix of every S0 (latent break / correctness bug) and the
high-impact S1 (silent fallbacks, dead enums, layer leaks) called out
by the May-2026 structural audit. The three primary workflows --
ground-state / low-energy spectrum, finite-temperature, DSSF -- are now
optimised end to end across CPU / GPU / MPI device axes; their auto-pilot
entries dispatch consistently, honour every orthogonal flag, and never
silently degrade behind the user's back.

Test surface: **293/293** (up from 289) -- the four new tests lock in
the S0 fixes (size-tracking cache invalidation, ``FixedSzOperator::
apply_real`` virtual dispatch, complex three-body GATHER kernel) and
the S1 tolerance-lock contract (``apply_auto_tune`` no longer retunes
a user value that happens to equal the struct default).

**S0 — Latent breaks (now fixed).**

- **`DistributedGPUOperator` (`src/distributed/distributed_gpu_operator.cu`).**
  Switched the term uploader from the long-removed
  ``serial->{diag,offdiag,mixed}_{one,two}_body_`` SoA members to the
  canonical ``serial->terms_`` storage after a
  ``commitPendingTransforms()`` call. The bug was latent on this dev
  host (no NCCL) but would have been a hard build break for anyone with
  ``NCCL_FOUND=TRUE``.
- **`Operator::commitPendingTransforms()`
  (`include/ed/core/operator.h`).** Size-tracking: the SoA rebuild
  trigger now compares ``transform_data_`` / ``three_body_data_``
  sizes against the values recorded at the last commit, so direct
  pushes into the AoS vectors (Python bindings, fixture builders, every
  ``examples/0[1-8]_*``, every ``benchmarks/bench_*``, several CLI and
  test sites) now honour the SoA rebuild on the next ``apply()``. The
  rebuild also invalidates the matvec backend's CSR cache and the
  ``isReal()`` cache, closing the entire chain. Recorded sizes are
  carried through copy / move ctors so derived classes don't accidentally
  resurrect a stale flag.
- **GATHER three-body kernel (`include/ed/matvec/term_kernels_gather.h`).**
  Replaced the ``double scalar = coefficient.real()`` accumulator with
  a full ``Complex`` accumulator that mirrors the SCATTER path. The
  distributed CPU SpMV now produces the same answer as the serial CPU
  SpMV for any Hamiltonian with a complex three-body coupling.
- **``Operator::apply_real`` is virtual; ``FixedSzOperator::apply_real``
  overrides it (`include/ed/core/{operator,fixed_sz_operator}.h`).**
  Without ``virtual``, binding a ``FixedSzOperator`` through an
  ``Operator&`` reference and calling ``apply_real`` sliced to the base
  ``dim == 2^N`` check, throwing on the legitimate sector dim. The
  fix is a one-keyword change but matters anywhere a solver receives
  the base reference (the MatVecOperator polymorphic surface, the
  generic auto-pilot routes).

**S1 — Architectural smells (now fixed).**

- **Dispatcher: ``BICG`` and ``mTPQ_MPI`` clean throws
  (`include/ed/core/ed_wrapper.h`).** Both used to fall through to the
  generic ``default`` throw (``BICG``) or print to stderr and re-throw
  (``mTPQ_MPI``). They now hit dedicated ``case`` arms with actionable
  messages: BICG points users at ``LANCZOS`` / ``KRYLOV_SCHUR`` /
  ``ARPACK_SM``; ``mTPQ_MPI`` explains the canonicalisation contract
  and points at ``ed::distributed::distributed_tpq``.
- **In-memory GPU fallback policy (`include/ed/core/ed_parameters.h`,
  `include/ed/core/ed_wrapper.h`, `include/ed/auto/solve.h`).** Added
  ``EDParameters::allow_gpu_cpu_fallback`` (default ``true`` for
  back-compat). When the in-memory ``exact_diagonalization_core`` is
  asked for a ``*_GPU`` method, it falls back to CPU silently as before
  iff the flag is true; otherwise it throws with a message pointing
  the caller at the file-based ``GPUOperator`` path.
  ``auto_pilot::solve(Device::GPU, allow_fallback=false)`` now sets
  the flag false so an explicit GPU request fails loudly instead of
  silently producing CPU results.
- **Streaming-symmetry GPU dispatch
  (`include/ed/core/ed_wrapper_streaming.h`).** Honours
  ``params.use_gpu`` in addition to the legacy
  ``is_gpu_method(method)`` enum-suffix check, so modern callers that
  pass ``LANCZOS + use_gpu=true`` (the auto-pilot route) finally hit
  the per-sector GPU kernels. Before the fix the streaming path was
  silently CPU-only for the modern entry.
- **Canonicalisation in ``ed::exact_diagonalization``
  (`include/ed/core/dispatch.h`).** Both overloads (directory + file)
  now call ``canonicalize_method_and_flags`` up-front, so legacy
  compound enums (``LANCZOS_GPU``, ``LANCZOS_GPU_FIXED_SZ``,
  ``FTLM_GPU``, ...) collapse to base method + orthogonal flags
  consistently across every downstream kernel. The file overload also
  gained the same ``automorphism_results/`` auto-detect logic the
  directory overload already had -- the two entries were asymmetric
  for no reason.
- **``AutoSolveOptions::symmetry_dir`` (`include/ed/auto/solve.h`).**
  New optional field. When set, ``auto_pilot::solve`` routes to
  ``ed::exact_diagonalization(symmetry_dir, ...)`` (the directory
  entry, which streams through irrep sectors) instead of building the
  in-memory SpMV. Combined with the existing Sz auto-detect, this is
  the first C++ entry point that exploits spatial **and** Sz symmetry
  automatically. Mirrors the Python ``qed.diag(H, symmetry=...)``
  shape.
- **``Device::MPI`` documentation
  (`include/ed/auto/solve.h`).** Honest about the ScaLAPACK semantics
  (it's not a generic MPI route -- the Lanczos / FTLM / TPQ distributed
  kernels reach via ``ed_distributed_main`` or the Python facade).
- **Thermal full-Hilbert auto-tune
  (`include/ed/auto/thermal.h`).** Both no-Sz branches (the in-memory
  fall-through and the directory entry's ``!sz_conserved`` path) now
  apply ``diag::apply_auto_tune`` before the solver fires, matching
  the Sz-loop's behaviour. Before the fix the no-Sz branches ran with
  unscaled defaults and were orders of magnitude slower for any
  realistic ``num_T`` / Krylov dim.
- **Tolerance sentinel hygiene -- full lock
  (`include/ed/auto/solve.h`, `include/ed/auto/thermal.h`,
  `include/ed/auto/diag_tune.h`).** Both
  ``AutoSolveOptions::tolerance`` and ``ThermalOptions::tolerance``
  are now ``std::optional<double>``. ``std::nullopt`` leaves the
  auto-tuner free to size the tolerance from the sector dim; an
  explicit value passes through verbatim. Critically, the
  ``AutoTuneOverrides::tolerance_locked`` bit is now plumbed through
  ``apply_auto_tune`` from ``options.tolerance.has_value()`` at every
  call site (one in ``solve``, four in ``thermal``), so the case
  ``options.tolerance = 1e-10`` (same value as the struct default) is
  also untouched by the sentinel-based retune -- closing the last hole
  the first round of the fix left open.

**Tests added (`tests/unit/test_operator_apply.cpp`,
`tests/unit/test_diag_tune.cpp`).**

- ``[regression][s0]``: direct AoS push between applies is honoured.
  Builds a Heisenberg chain, calls ``apply`` (priming the SoA cache),
  pushes a new ``Sz_0`` term, calls ``apply`` again, asserts the
  expected diagonal contribution lands in the output.
- ``[regression][s0][fixed_sz]``: ``FixedSzOperator::apply_real``
  dispatches virtually through ``Operator&``. Without the
  ``virtual`` keyword on the base, this would throw.
- ``[regression][s0][three_body]``: ``gather_row`` with a pure-imag
  three-body coupling matches the SCATTER ``Operator::apply`` to
  1e-12 and produces a non-zero output (the pre-fix code would
  return identically zero).
- ``[regression][s1]``: ``apply_auto_tune`` honours
  ``tolerance_locked`` when the user value happens to equal the
  struct default. Without the lock the Aggressive level would have
  retuned 1e-10 to a coarser threshold.

**Documented as deferred (intentionally).**

- S1 #19 (reorth consolidation), S1 #20 (one Krylov tridiag builder),
  S1 #25 (KPM GPU spectral-bound reorth parity), S2 #29 (dead public
  API pruning). These are large structural refactors of 5000-line
  files; they are deferred to a dedicated cleanup pass to keep the
  current rollout focused on correctness / pipeline-end-to-end
  behaviour. See ``docs/architecture/STRUCTURAL_AUDIT.md`` for status.

### Operator-class audit — final flush-out

Final structural / line-by-line pass over **every** concrete operator
class (`Operator`, `FixedSzOperator`, the eight `operator_types.h` and
eight `fixed_sz_operator_types.h` subclasses, the GPU operator family,
and the distributed wrappers). The goal stated by the user — "all
operator classes are fully well flushed out" — meant: no
archaeological comment blocks for code that's already gone, no
redundant per-class duplication of helpers that already live in
`ed::core::detail`, no silent cache-invalidation holes when a derived
cache outlives its source.

**`Operator` (`include/ed/core/operator.h`).**

- Replaced the ~18-line "DEAD CODE REMOVED" comment block (left over
  from the previous audit's bulk deletion of the symmetry-block
  builder family) with a 5-line pointer to the canonical pipelines
  in `streaming_symmetry.h` / `distributed_symmetry_operator.h`.
- Extracted the duplicated 5-line HPhi text-file header parser into
  `Operator::open_hphi_file_(filename, stream) -> num_terms`. The
  three loaders (`loadFromFile`, `loadFromInterAllFile`,
  `loadThreeBodyTerm`) now share one parser; each is reduced to its
  per-line body.
- Tidied the symmetry-derived public members (`symmetrized_block_ham_sizes`,
  `symmetry_info`) with a comment block explaining why they live on
  the base class (so `StreamingSymmetryOperator` /
  `FixedSzStreamingSymmetryOperator` can populate them through the
  inherited interface without forcing every plain `Operator` to
  carry the symmetry typing).
- Dropped trailing whitespace and a redundant `// Core Operator
  Functions` spacer comment.

**`FixedSzOperator` (`include/ed/core/fixed_sz_operator.h`).**

- Replaced the ~10-line "Symmetry-block builder family REMOVED"
  comment block with a 3-line pointer to
  `FixedSzStreamingSymmetryOperator`.
- Added the missing `getNUp()` accessor. The DSSF cross-sector
  observable used to call `popcount(getBasisStates().front())` as a
  workaround; that has been simplified to `op.getNUp()`.
- Tightened comments on `binarySearchState` / `lookupState` /
  `projectToFixedSz` / `embedToFull` / `getFixedSzMatrix`. Each is
  now a one-paragraph docstring with intent + guarantee rather than
  a 6-line history note.
- Sector / basis accessors (`getFixedSzDim`, `getFullDim`,
  `getNUp`, `getBasisStates`, `lin_index_table`) collected under one
  `// Sector / basis introspection.` block.

**`operator_types_detail.h`.**

- Added `add_experimental_site_term(op, site, phase, cos_theta,
  sin_theta)`: the canonical per-site expansion of
  `phase * (cos(theta) Sz + sin(theta) Sx)` into the
  (Sz, S+, S-) AoS terms. Both `ExperimentalOperator` /
  `TransverseExperimentalOperator` (`operator_types.h`) and their
  fixed-Sz counterparts in `fixed_sz_operator_types.h` now route
  through this helper. Eliminates the last bit of per-class
  three-term duplication; ensures Cartesian-Sx-expansion semantics
  stay byte-identical between full-basis and fixed-Sz operators.

**`GPUOperator` (`src/solvers/gpu/gpu_operator.cu`).**

- Fixed a real cache-invalidation bug: `invalidateDerivedCaches()`
  freed the FP64 assembled CSR but **not** the mixed-precision FP32
  CSR derived from it. Because `buildCsrFp32OnDevice` exited early
  on `fp32_csr_assembled_ && csr_dim_fp32_ == N`, a sequence
  `addOneBodyTerm; matVec; addOneBodyTerm; matVec` at the same
  dimension would reuse the FP32 cache from the **first** call
  against the new FP64 CSR — silently producing wrong results in
  the mixed-precision path. The fix adds an explicit
  `freeCsrFp32DeviceData()` call when `fp32_csr_assembled_` is set.

**`streaming_symmetry.h`.**

- Added the previously transitive `#include <H5Cpp.h>`. This header
  uses `H5::H5File`, `H5::DataSpace`, etc. directly but was relying
  on `hdf5_symmetry_io.h` being pulled in via the now-cleaned-up
  `operator.h`. The new direct include makes the dependency
  explicit and decouples it from `operator.h`'s include hygiene.

**Verification.** Clean build + full `ctest` (289 / 289 PASS, ~33 s
wall) after each independently-shippable cleanup. No public-API
breakage: all eight `operator_types` constructors, all eight
`fixed_sz_operator_types` constructors, `Operator::apply` /
`apply_real` / `getSparseMatrix`, `FixedSzOperator::buildFixedSzMatrix`
/ `getFixedSzMatrix`, the GPU `matVec` family, the distributed
operators — all retain identical signatures and observable
behaviour.

### Operator / apply construction audit — structural + perf cleanup

Follow-up to the term-storage / matvec-backend revamp. A focused audit
of `Operator`, `FixedSzOperator`, `MatVecBackend`, and the term
kernels identified one real performance regression, several dead
fields, and a handful of redundant helpers that had survived the
previous refactor. All addressed in this pass; no API breakage for
callers of `apply()` / `getSparseMatrix()` / `getFixedSzMatrix()`.

**Performance fix (Tier A).**

`FixedSzOperator::buildFixedSzMatrix()` was assembling the projected
sparse matrix by probing the matvec backend with `dim` unit vectors
and scanning each output column for non-zeros — **O(dim² × num_terms)**.
The doc comment in `ed_wrapper.h` even said
"This is O(nnz) — much faster than dim SpMV calls" but the
implementation *was* the dim SpMV calls. For dim = 924 (N=12, n_up=6)
this was ~10⁶ extra apply() calls; for the dense diagonalisation paths
that route through `getFixedSzMatrix()` this was minutes of wall time
when it should have been seconds. The new implementation calls
`ed::matvec::kernel::emit_term_triplets<FixedSzBasisPolicy>` directly
on the canonical AoS terms (the same SoT triplet emitter used by
`MatVecBackend`'s CSR cache), and folds in `transforms_` (legacy
`std::function`) closures with per-state Sz-sector projection.
Result: **O(dim × num_terms)** — asymptotically optimal, matching the
matrix-free path.

**Dead-code removal (Tier B).**

- `ed::matvec::kernel::emit_csr_triplets` (`term_kernels.h`): unused
  serial copy of the canonical parallel `emit_term_triplets` in
  `term_kernels_assemble.h`. ~95 lines deleted; the assembler is now
  the **sole** triplet emitter in the codebase.
- `Operator::apply_sparse` / dead public API
  (`std::vector<Complex>` and raw-pointer overloads). No callers in
  `src/`, `tests/`, or `python/`. Removed.
- `Operator::buildSparseMatrix()` / `sparseMatrix_` / `matrixBuilt_`:
  the old ColMajor-Eigen cache that backed `apply_sparse` and a
  redundant `getSparseMatrix`. The new `getSparseMatrix()` is a thin
  free-standing assembler over `emit_term_triplets` + the legacy
  `transforms_` projection — no internal caching, no const_cast
  workaround.
- `Operator::operators_` `std::array<std::array<double,4>,3>` dead
  member. Never read anywhere in the codebase. Removed.
- `MixedTwoBody::sz_first` field (and the corresponding parameter on
  `add_mixed_two_body` / `classify_route` / the GPU
  `GpuClassifySink`). The Sz factor and the S+/S- flip commute (they
  act on different sites), so no consumer ever needed the flag.
  Verified by grep: 0 read sites across CPU `apply_terms` / GPU
  `mixed_two_body_kernel` / `emit_term_triplets` / `gather_row`.
- `Operator::separateTransformsByType()` deprecated stub: zero callers
  after the GPU side migrated to its own non-deprecated copy in the
  previous session. Removed.
- Empty `if constexpr` block inside `apply_terms` (a half-written
  prefetch experiment) replaced with a clearer comment about why the
  basis-policy lookup is already prefetcher-friendly.

**Minor perf refinements (Tier C, cheap parts).**

- `CpuMatVecBackend::input_is_real(...)` now does a 64-element prefix
  bail before the full sweep. Complex Lanczos vectors (the common
  case past iteration 1) now exit after ~one cache line of work
  instead of scanning the full vector each matvec. No effect on
  genuinely-real inputs (they still scan the whole thing — there is
  no positive-branch shortcut).
- `apply_legacy_transforms_*` helpers explicitly documented as
  SEQUENTIAL slow paths, with a TODO listing the six remaining
  subclasses (`ExperimentalOperator`, `TransverseExperimentalOperator`,
  `FixedSzSingleSiteOperator`, `FixedSzDoubleSiteOperator`,
  `FixedSzExperimentalOperator`,
  `FixedSzTransverseExperimentalOperator`) that should migrate to the
  canonical AoS API so the `transforms_` channel can be deleted
  entirely.

**Deferred (Tier C, judgment calls).**

- `make_backend_` / `ensure_backend_` virtual dispatch on every
  first-apply — cost is in the noise, no profile motivates a
  rebuild on construction.
- Backend lifetime fragility in `FixedSzOperator` (the
  `~FixedSzOperator() { backend_.reset(); }` requirement) — kept the
  explicit reset, would need a `shared_ptr` to basis tables to truly
  fix and the atomic cost is probably not worth it.
- `apply_terms` parallelism cliff at `dim ≈ num_threads × 1024` —
  matters only in a narrow dim band; revisit with profiles.

**Verification.** Clean build + full `ctest` (289/289 PASS,
39.6 s wall) + Python smoke (Heisenberg-4 ring: E₀ = -2.0,
E₁ = -1.0 in both full-Hilbert and Sz = 0 sectors;
`getFixedSzMatrix()` agrees with `apply()` to 1e-12). The dense
diagonalisation tests (`test_auto_thermal`,
`test_ed_solver_matrix_e2e`) continue to pass with the new
O(dim × num_terms) assembler.

### Fixed — Two symmetric latent bugs in the dual term-storage model

Closes the symmetric pair of latent correctness issues that the
"continue to roll out" structural sweep surfaced while auditing
Problem 4 (the FixedSzOperator inheritance graph).

**Setup.** After the Phase-4 term-storage refactor, `Operator`
carried two parallel input channels for terms:

  1. **Canonical AoS** — `transform_data_` + `three_body_data_`,
     populated by `addOneBodyTerm` / `addTwoBodyTerm` /
     `addThreeBodyTerm` and the file loaders. Read by the matvec
     backend via the SoA cache `terms_` after
     `commitPendingTransforms()`.
  2. **Legacy std::function closures** — `transforms_`, populated
     by `addTransform(std::function<...>)`. Used by ~6 subclasses
     in `operator_types.h` / `fixed_sz_operator_types.h`
     (`FixedSzSingleSiteOperator`, `FixedSzDoubleSiteOperator`,
     `ExperimentalOperator`, `TransverseExperimentalOperator`,
     `FixedSzExperimentalOperator`,
     `FixedSzTransverseExperimentalOperator`).

The two channels were never reconciled. Before this release:

  - `apply()` / `apply_real()` walked path (1) only — the matvec
    backend reads `terms_`. Operators populated exclusively via
    `addTransform` silently returned **zeros** from `apply()`.
  - `buildSparseMatrix()` / `getSparseMatrix()` walked path (2)
    only — the assembly loop iterated `transforms_`. Operators
    populated via the typed AoS API silently returned an
    **empty** sparse matrix.

Both bugs only emitted a warning on `buildSparseMatrix()` (the
warning is now removed). `apply()` produced no diagnostic at all.
Neither path had test coverage for the affected subclass family.

**Fix.**

1. `Operator::apply` / `Operator::apply_real` now invoke an
   `apply_legacy_transforms_(...)` accumulator after the matvec
   backend. The accumulator walks `transforms_` and adds the
   resulting contributions into `out`. The hot path is a single
   `transforms_.empty()` test — users on the canonical AoS API
   pay nothing.

2. `FixedSzOperator::apply` / `apply_real` carry an analogous
   `apply_legacy_transforms_fixed_sz_*_` that performs the
   Sz-sector projection: each transform takes a full-Hilbert
   basis index `basis_states_[i]`, the returned full-Hilbert
   target index is mapped back to the sector via the existing
   `lookupState(...)` (O(1) Lin table), and only in-sector hits
   are accumulated. States that fall outside the sector are
   silently dropped, matching the legacy `buildFixedSzMatrix`
   projection semantics.

3. `Operator::buildSparseMatrix` now appends a second pass that
   walks `transform_data_` + `three_body_data_` via the new
   `ed::matvec::kernel::emit_term_triplets<...>` helper (see
   below). The assembled matrix is correct regardless of which
   input channel populated the operator. The "WARNING:
   buildSparseMatrix() called but operator data is in
   transform_data_" message is gone.

**Tests.** Full `ctest -j8` suite remains green (289/289). The
two affected subclass families had no pre-existing regression
tests; the Python `qed.diag` smoke on the canonical-AoS
Heisenberg-4 ring continues to return `E0 = -2.0`, `E1 = -1.0`
exactly through the now-doubly-defensive `buildSparseMatrix` /
`apply` paths.

### Changed — ASSEMBLE kernel factored out; matvec single-source-of-truth complete

The unified term kernel now has all three structural forms living
in dedicated headers in `ed/matvec/`:

  | Direction  | Header                       | Operation                  |
  |------------|------------------------------|----------------------------|
  | SCATTER    | `term_kernels.h`             | `y[c] += <c|H|b> v[b]`     |
  | GATHER     | `term_kernels_gather.h`      | `y[r] += <r|H|c> v[c]`     |
  | ASSEMBLE   | `term_kernels_assemble.h`    | `triplets.emplace(c,b,…)`  |

All three share **byte-identical** op_type / bit-flip / popcount
semantics. The SCATTER kernel is the canonical reference; the
other two are tested against it via the existing
distributed-equivalence and CSR-vs-matrix-free unit tests.

`emit_term_triplets<BasisPolicy, Scalar>(basis, spin_l,
diag_one_body, offdiag_one_body, diag_two_body, mixed_two_body,
offdiag_two_body, three_body, triplets)` is the public free
template. The previously inlined private `emit_triplets_`
(plus `emit_`, `emit_if_in_basis_`, `coerce_<Scalar>`) inside
`CpuMatVecBackend` has been deleted (~140 lines). Both
consumers — the backend's CSR cache (`ensure_csr_complex` /
`ensure_csr_real`) and `Operator::buildSparseMatrix` — now
delegate to the same free helper.

### Changed — GPU classification migrated through `TermStorage::classify_route`

`GPUOperator::separateTransformsByType` previously open-coded the
op_type → SoA-bin decision tree against the GPU-side AoS
(`GPUTransformData`) and GPU-side SoA bins (`GPUDiagonalOneBody`,
`GPUOffDiagonalOneBody`, `GPUDiagonalTwoBody`, `GPUMixedTwoBody`,
`GPUOffDiagonalTwoBody`). The classification logic was a
byte-for-byte duplicate of the CPU's pre-refactor
`separateTransformsByType` — any future change to the decision
tree on one side risked drifting from the other.

After this release, the GPU's `separateTransformsByType` builds
a thin host-side `GpuClassifySink` adapter (six `add_*` setter
methods that push into the GPU SoA `std::vector`s) and calls
`ed::matvec::TermStorage::classify_route(sink, transform_data_,
empty_three_body, /*conv=*/identity)`. The exact same template
the CPU's `commitPendingTransforms()` uses now drives the GPU
fan-out, so the decision tree lives in **exactly one** location
across the whole codebase.

GPU specifics preserved:
  * The `sz_first` flag that the CPU mixed-two-body kernel uses
    for Sz-evaluation-point disambiguation is silently dropped
    by the GPU sink. The GPU kernel enforces
    `sz_site != flip_site` (the `addInteractAll` invariant), so
    the Sz bit is identical on source and destination states and
    the flag is irrelevant on the device.
  * Three-body terms still live in their own AoS (`three_body_data_`)
    on the GPU; the sink's `add_three_body` is a no-op and the
    classify_route call is made with an empty three-body input.

### Changed — Triple term storage collapsed to a single canonical AoS + lazy SoA cache

Closes Problem 1 of the "complete roll-out" structural audit follow-up.

**Before.** `Operator` carried THREE coexisting term representations:

  1. `transforms_`     — `std::vector<std::function<...>>` (legacy)
  2. `transform_data_` — `std::vector<TransformData>` (AoS, the
     classification source)
  3. `diag_one_body_`, `offdiag_one_body_`, `diag_two_body_`,
     `mixed_two_body_`, `offdiag_two_body_`, `three_body_data_` —
     six independent `std::vector<...>`s holding the SoA bins
     consumed by the hot-path matvec kernel
     (`ed::matvec::kernel::apply_terms`)

External code pushed into (2); a `transforms_separated_` flag plus a
lazy `separateTransformsByType()` call inside the first `apply()`
fanned the AoS list into the SoA bins. `invalidateMatrixCaches()`
did NOT clear `transforms_separated_`, so the following innocuous
sequence silently dropped the new term:

```cpp
op.transform_data_.push_back(new_term);
op.invalidateMatrixCaches();   // ← did NOT mark SoA stale
op.apply(in, out, n);          // ← reads SoA, sees the old terms
```

(The matrix-cache was correctly invalidated, but the SoA cache the
matvec kernel actually reads was left untouched.) The lazy fan-out
also required a `const_cast` inside the otherwise-`const`
`apply()` chain, and the classification decision tree
(`op_type ∈ {0,1,2}` → `{diag,offdiag,mixed} × {one,two}body`) was
duplicated between `Operator::separateTransformsByType` and
`GPUOperator::separateTransformsByType` with no mechanism enforcing
that they stay in sync.

**After.**

  - A new header `include/ed/matvec/term_storage.h` defines
    `ed::matvec::TermStorage`: a plain SoA carrier with six typed
    bins (`diag_one_body`, `offdiag_one_body`, `diag_two_body`,
    `mixed_two_body`, `offdiag_two_body`, `three_body`) and typed
    setters (`add_diag_one_body`, `add_offdiag_one_body`, ...) plus
    classify-and-route helpers (`add_one_body`, `add_two_body`).

  - `Operator` keeps `transform_data_` and `three_body_data_` as the
    **canonical AoS** (these are what file loaders and the
    `addOneBodyTerm` / `addTwoBodyTerm` / `addThreeBodyTerm` public
    setters write to). The six standalone SoA vectors are gone; in
    their place a single `mutable ed::matvec::TermStorage terms_`
    cache plus a `mutable bool terms_fresh_` flag.
    `invalidateMatrixCaches()` now clears `terms_fresh_` along with
    the CSR / real caches, restoring the invariant that any mutation
    of the canonical AoS forces the SoA cache to be rebuilt.

  - `commitPendingTransforms()` replaces the old
    `separateTransformsByType()`: it's a no-op when
    `terms_fresh_ == true`, otherwise it clears `terms_` and
    repopulates it from the canonical AoS via
    `TermStorage::classify_route(...)` (see below). It is called
    eagerly from every reader of `terms_`
    (`apply`, `apply_real`, `isReal`, `term_view_`, the distributed
    operator's flip-pattern extractor, `has_zeeman_field`,
    `sanity_check.cpp`).

  - `TermStorage::classify_route<Sink, Aos, Aos3, Conv>(sink, aos,
    aos3, conv)` is the new **single source of truth** for the
    classification decision tree. The CPU's
    `commitPendingTransforms()` delegates to it directly; the GPU's
    analogous `GPUOperator::separateTransformsByType()` is now
    structurally identical (same `op_type == 2` branches, same
    `sz_first` convention for mixed two-body) and is documented as a
    future migration target once the GPU and CPU share a sink
    adapter (see "Deferred" below).

**Net effect.** The latent
`invalidate-without-cache-fresh-flag-clear` bug is gone by
construction. The `const_cast` inside `apply()` is gone. The
classification logic exists in exactly one place
(`TermStorage::classify_route`). External code that pokes at the
storage gets a clear API: typed setters route to typed bins, with
no flag to keep in sync with cache invalidation.

**Backward compatibility.** All existing read sites that referenced
the old member-level SoA vectors (`op.diag_one_body_`,
`op.offdiag_one_body_`, ...) have been migrated to
`op.commitPendingTransforms(); auto& v = op.terms_.diag_one_body;`
(see `sanity_check.cpp`, `distributed_operator.cpp`, `auto/solve.h`).
The public typed setters (`addOneBodyTerm`, `addTwoBodyTerm`,
`addThreeBodyTerm`) keep their signatures. File loaders
(`loadFromFile`, `loadFromInterAllFile`, `loadThreeBodyTerm`,
`loadonebodycorrelation`, `loadtwobodycorrelation`) now use the
public typed setters instead of directly pushing into
`transform_data_`, so they correctly trigger cache invalidation.

`std::vector<Operator>` continues to work: `Operator` now has
explicit copy/move constructors and assignment operators because
of the `std::unique_ptr<MatVecBackendBase>` member it carries
(the destination's backend is reset and lazily rebuilt against
its own term list).

### Changed — DistributedOperator GATHER kernel factored into `ed::matvec::kernel::gather_row`

Closes Problem 3 of the structural audit follow-up.

**Before.** `DistributedOperator::apply_local(...)` inlined six
back-to-back `for` loops walking the SoA bins on `op_->terms_.*`
in GATHER form: for each globally-owned output row `r`, scan every
SoA bin, deduce the input row `c` from the bit-flip pattern, look
up `v[c]` via the per-rank lookup table, and accumulate into
`y[r]`. This was a byte-for-byte hand-reimplementation of the
SCATTER kernel's `op_type` decision tree (`term_kernels.h`),
specialized to one-flip / two-flip / Sz-only cases. Any future
change to the bit-flip semantics on one side risked silently
drifting from the other.

**After.** A new header `include/ed/matvec/term_kernels_gather.h`
defines `ed::matvec::kernel::gather_row<GetV>(r, v_local_at_r,
terms, spin_l, get_v)`, the structural dual of
`ed::matvec::kernel::apply_terms`. The `GetV` callable resolves
remote-vs-local lookup at the call site (MPI per-rank tables,
future NCCL, future page-faulting GPU buffers, ...).
`DistributedOperator::apply_local` now reads:

```cpp
op_->commitPendingTransforms();
const auto& terms = op_->terms_;
for (uint64_t r_local = 0; r_local < local_dim; ++r_local) {
    const uint64_t r = base + r_local;
    y[r_local] = ed::matvec::kernel::gather_row(
        r, v_local[r_local], terms, spin_l,
        [&](uint64_t c){ return v_lookup_(c); });
}
```

**Net effect.** GATHER and SCATTER semantics live in two
templated headers in the same namespace, share the same
`TermStorage` schema, and use the same `op_type` encoding. Bit-flip
correctness is now enforced by reading both kernels side-by-side.
The distributed-equivalence tests
(`test_distributed_against_serial`) continue to pass at every
problem size in the matrix.

### Changed — Operator matvec dispatch unified behind a single `MatVecBackend` strategy

Closes the follow-up to the structural audit: *"yeah let's rework this so
that it is structurally clean and elegant. Make sure everything is
production level clean. Also is there a point of doing all of these
types of apply? I want to just keep one working one if possible."*

**Before.** `Operator::apply()` and `FixedSzOperator::apply()` each
contained the same three-way dispatch tree (assembled-CSR-complex /
assembled-CSR-real / matrix-free) inlined against their own private
row-major CSR caches (`sparseMatrixRow_` / `sparseMatrixRealRow_` /
`fixed_sz_csr_` / `fixed_sz_csr_real_`) and scratch buffers. Both
operators exposed four near-identical apply_* member functions
(`apply_optimized`, `apply_real`, `apply_via_csr_parallel`,
`apply_via_csr_parallel_real`, plus the Sz-projected variants), and
the CSR caches were stored as `mutable` on the operator so a
`const` `apply()` could lazily build them — which forced a
`const_cast`-style pattern and made `Operator` accumulate state
that isn't conceptually part of the term list.

**After.** A new strategy class, `ed::matvec::MatVecBackendBase`
(declared in `ed/matvec/matvec_backend.h`), encapsulates the SpMV
dispatch. `Operator` and `FixedSzOperator` each expose exactly two
matvec entry points — `apply(complex)` and `apply_real(double)` —
and both are one-line delegations to the backend.

The concrete backend, `CpuMatVecBackend<BasisPolicy, ...>`, owns the
CSR caches and scratch buffers (no more `mutable` on the operator
for these), chooses matrix-free vs assembled-CSR based on a single
`MatVecTunables` struct read once at construction (no more
`getenv()` per matvec), and dispatches real-vs-complex via the same
`BasisPolicy` template the matrix-free kernel already uses. The
Sz-projected backend is selected by `FixedSzOperator` overriding a
single virtual factory method (`make_backend_()`); everything else
is inherited.

The four parallel CSR triplet builders that previously lived on
`Operator` and `FixedSzOperator` (`buildSparseMatrixFromData`,
`buildSparseMatrixFromDataReal`, `appendFixedSzTriplets`,
`appendFixedSzTripletsReal`) have been deleted and replaced with a
single templated emitter inside the backend; the `may_leave_basis`
compile-time flag on the basis policy elides the index-lookup check
when it's unreachable.

**Net effect on the public API:**

| Class               | Public matvec API after the refactor                         |
|---------------------|--------------------------------------------------------------|
| `Operator`          | `apply(Complex, ...)`, `apply_real(double, ...)`             |
| `FixedSzOperator`   | `apply(Complex, ...)`, `apply_real(double, ...)`             |

Internally that's still TWO entry points per class instead of one —
because the real-arithmetic fast path is a genuine 2× perf win on
real Hamiltonians (Heisenberg, Ising, transverse-field Ising, ...)
that `lanczos_real` and the FTLM real path rely on — but neither
the dispatch tree nor the CSR caches show up in the operator
anymore.

**Backward compatibility.** All existing callers of `apply()` see
identical behaviour. The two Python-binding lambdas in
`qed_bindings.cpp` (`make_hv_real`) now call the consolidated
`apply_real`, so `lanczos_real` keeps its real-arithmetic fast
path. Legacy `apply_sparse` / `getSparseMatrix` /
`getFixedSzMatrix` (built from the `transforms_` std::function
list) are preserved for code paths that need a materialised
matrix (full diagonalisation, the Eigen dense path).

**Knobs.** Two new environment variables apply to both bases
uniformly; the legacy per-basis ones still work and take lower
precedence:

  - `ED_CSR_FORCE=0|1` — force matrix-free / assembled
  - `ED_CSR_DIM_MAX=N` — basis-dim cutoff above which the backend
    stays matrix-free (defaults: `1<<20` full, `1<<22` fixed-Sz)

Legacy: `ED_USE_SPARSE` / `ED_SPARSE_DIM_MAX` (full only),
`ED_FIXED_SZ_USE_SPARSE` / `ED_FIXED_SZ_SPARSE_DIM_MAX` (fixed-Sz).

**Files touched:**

  - `include/ed/matvec/matvec_backend.h` — NEW; the strategy class
  - `include/ed/core/operator.h` — `apply()` / `apply_real()`
    collapsed; CSR caches + 4 inlined SpMV variants deleted
    (~470 lines removed)
  - `include/ed/core/fixed_sz_operator.h` — `apply()` / `apply_real()`
    collapsed; CSR caches + scratch buffers + 4 helper functions
    deleted (~280 lines removed)
  - `python/qed/_bindings/qed_bindings.cpp` — `make_hv_real`
    consolidated into a single template

**Tests.** Full `ctest -j8` suite is green: 289/289 tests pass
(previously the same 289 also passed). Targeted suites
`test_operator_apply`, `test_fixed_sz_operator`,
`test_lanczos_variants`, `test_full_diagonalization`,
`test_thermal_methods` cover both the full-Hilbert and the Sz-
projected paths and report `apply_real(re) == real(apply(complex(re)))`
to ‖·‖₂ < 1e-12 across multiple seeds. Python-side smoke test:
`qed.diag` on a 4-site Heisenberg ring returns `E0 = -2.0`
exactly in both the full-Hilbert and the `sz=0` runs; `qed.thermal`
FTLM gives the expected Schottky-shaped `C_v(T)` across all
n_up sectors.

### Deferred — Problem 4 only (FixedSz collapse to composition)

The structural audit identified four problems with the operator
hierarchy. Problems 1 (term storage), 2 (GPU `MatVecBackend`
integration via `classify_route`), and 3 (distributed GATHER
kernel) all ship in this release. Problem 4 remains explicitly
deferred to a dedicated session.

**Problem 4 — FixedSz from inheritance to composition.**
`FixedSzOperator` is a subclass of `Operator` carrying a
`FixedSzBasis` index member and overriding `dim()` / `apply()`
to walk the Sz-projected basis. The "LSP violation" framing in
the original audit was overstated — virtual dispatch on
`dim()` and `apply()` handles the projected-vs-full mismatch
correctly, and every caller holds `FixedSzOperator&` /
`FixedSzOperator*` specifically. The audit's actual concrete
concerns turned out to be the two latent dual-storage bugs
documented above, both of which are now closed.

Full composition collapse would make `Operator` own a
`BasisPolicy` variant (`FullBasis | FixedSzBasis |
SymmetryBasis`) and turn `FixedSzOperator(...)` into a thin
factory function. The migration touches ~54 implementation
callsites holding `FixedSzOperator&` and ~10 convenience
subclasses in `fixed_sz_operator_types.h`. With the latent
bugs closed, the inheritance hierarchy has no observed
behavioural issues; collapse is a 1-2 day mechanical-but-tedious
refactor that does not change runtime semantics. Kept as-is
until either (a) a dedicated session is available to land it
cleanly, or (b) a new symptom surfaces that requires the
collapse to fix.

### Added — Universal coverage of every finite-T method through the unified `thermal()` entry point

Closes the audit follow-up: *"implement an optimized, symmetry
exploiting, but completely automatic routine for all the other
finite temperature methods too. Make sure everything is unified and
clean."*

Prior to this change `ed::auto_pilot::thermal(...)` (and its Python
mirror `qed.thermal(...)`) only delivered the unified Sz / symmetry
recombination story for FTLM. Calling it with `LTLM`, `HYBRID`,
`KPM_DOS`, `mTPQ`, or `cTPQ` either crashed (TPQ tried to write
HDF5 trajectories to the `/dev/null` output sentinel), returned
identically-zero recombined thermo (TPQ's chain didn't progress on
small systems because `tpq_energy_shift` defaulted to `1e5`), or
silently dropped the user's `T_min` / `T_max` (KPM_DOS was not
flagged as a thermal method, so the `_compose_params_for_diag`
shim left it at its built-in 1e-3 default).

This release closes every one of those holes and ships the
matching test coverage.

**Per-method status after this change:**

| Method   | Auto-Sz | Auto-Spatial-Symmetry | T_min/T_max honored | Tested via `thermal()` |
|----------|:-------:|:---------------------:|:-------------------:|:----------------------:|
| FTLM     | ✓       | ✓ (directory form)    | ✓                   | ✓ (statistical match)  |
| LTLM     | ✓       | ✓ (directory form)    | ✓                   | ✓ (sanity)             |
| HYBRID   | ✓       | ✓ (directory form)    | ✓                   | ✓ (sanity)             |
| KPM_DOS  | ✓       | ✓ (directory form)    | ✓                   | ✓ (sanity + T-range)   |
| mTPQ     | ✓       | n/a (single random state, see note) | ✓ | ✓ (finite + grid)      |
| cTPQ     | ✓       | n/a (single random state, see note) | ✓ | ✓ (finite + grid)      |

**Note on TPQ + spatial symmetry.** TPQ methods evolve a single
random state on the full (Sz-projected) Hilbert space. Per-irrep
streaming would require synthesizing a properly-projected random
seed in each sector and recombining the trajectories with the
correct irrep weights, which is genuinely a research-level
problem -- the canonical TPQ literature does not address it. The
unified `thermal()` entry point therefore silently disables
spatial-symmetry decomposition for `mTPQ` / `cTPQ` even when
`automorphism_results/` is present in the input directory; the Sz
axis is still fully exploited. `used_symmetry_decomposition`
reports `false` in this case so callers can see the fallback.

**Key C++ changes:**

  * `include/ed/auto/thermal.h`:
    - `ThermalOptions` gained method-specific knobs:
      `kpm_num_moments`, `kpm_num_random_vectors`,
      `hybrid_crossover_temperature`, `hybrid_auto_crossover`,
      `tpq_num_measure_points`, `tpq_measure_beta_min/max`
      (auto-derived from (T_min, T_max) when left at 0),
      `tpq_delta_beta`, `tpq_taylor_order`,
      `tpq_measurement_interval`, `tpq_energy_shift` (0 = auto-pick).
    - New helper `detail::make_tpq_sector_workdir(...)` allocates a
      per-sector scratch directory when the user did not supply an
      `output_dir`. TPQ writes its per-sample HDF5 trajectories
      there; `compute_tpq_unified_thermo` reads them back inside
      `exact_diagonalization_core` and populates
      `EDResults.thermo_data`; `thermal()` cleans up afterwards.
    - The directory form's `has_sym` is force-cleared for TPQ
      methods so they bypass the streaming-symmetry path.

  * `include/ed/core/ed_wrapper.h` (single source of truth for
    `exact_diagonalization_core`):
    - `mTPQ`: trivial-sector short-circuit for `dim == 1` (the
      `D_S = log2(1) = 0` factor in the (L - H/D_S)|v> recursion
      otherwise produces `β = -∞` and unusable trajectories) +
      auto-pick of `tpq_energy_shift` via a 24-step Lanczos
      spectral-bound estimate when the caller passes the sentinel
      `tpq_energy_shift == 0`. Without auto-pick the historical
      `1e5` default produces `β << 1` in `max_iterations` steps on
      small/medium systems and `compute_tpq_unified_thermo` returns
      zeros.
    - `cTPQ`: same trivial-sector short-circuit (cTPQ's Taylor
      expansion of `exp(-Δβ H)` is degenerate for `dim == 1`).
    - Both short-circuits now lay down the standard
      *log-spaced* T grid used by `compute_tpq_unified_thermo`, so
      the recombiner sees a consistent T axis across sectors.
    - mTPQ/cTPQ branches populate `results.thermo_data` after the
      iteration completes by calling `compute_tpq_unified_thermo`
      on `params.output_dir`. Before this, TPQ branches returned
      empty `thermo_data` and the `thermal()` recombiner had
      nothing to work with.

  * `src/solvers/cpu/TPQ.cpp`:
    - New free function `compute_tpq_unified_thermo(dir, T_min,
      T_max, num_T)` extracts the trajectory-averaging + log-T
      interpolation into a single source of truth and returns a
      `ThermodynamicData` block. The legacy
      `convert_tpq_to_unified_thermo` was refactored to delegate
      to it so the on-disk HDF5 output and the in-memory
      `EDResults.thermo_data` are byte-identical.
    - Every `MPI_*` collective is now guarded by
      `MPI_Initialized()`; building with `WITH_MPI=ON` no longer
      crashes when MPI is not actually launched. This was a
      pre-existing latent bug that only surfaced when running the
      new TPQ tests outside `mpirun`.

  * `src/solvers/cpu/ltlm.cpp`:
    - `save_ltlm_results(...)` now honours the `/dev/null` output
      sentinel via `HDF5IO::isDisabledOutputPath`. Before this,
      passing `output_dir = ""` (which `ed_wrapper.h` substitutes
      with `/dev/null` for uniform downstream checks) caused
      `save_ltlm_results` to attempt an `H5F_ACC_RDWR` open on
      `/dev/null/ed_results.h5`, leading to a HDF5 truncate error
      and SIGSEGV at file close.

  * `include/ed/core/symmetry_metadata.h`:
    - `loadMinimalGenerators` now accepts both the top-level JSON
      array form `[{"permutation":..., "order":...}, ...]` (what
      the Python `automorphism` tool writes) and the
      `{"generators": [...]}` object form (what some C++ pipelines
      write). The smoke fixture
      `tests/fixtures/chain4/automorphism_results/` uses the
      array form, and was tripping the loader after the recent
      `symmetry_data_present` auto-detection improvement.

**Key Python changes:**

  * `python/qed/thermal.py`:
    - `qed.thermal(...)` signature gained the same method-specific
      knobs as `ThermalOptions` (KPM, HYBRID, TPQ).
    - TPQ scratch-directory mechanism implemented at the Python
      level too: when `output_dir=""` and method is mTPQ/cTPQ, a
      per-sector temp dir is allocated, passed into the C++
      dispatcher, and `shutil.rmtree`'d in a `finally` after the
      iteration. Mirrors the C++ helper.
    - `used_symmetry_decomposition` returns `False` for TPQ
      methods on the directory form to expose the symmetry skip.

  * `python/qed/workflow.py`:
    - `KPM_DOS` is now in `_thermal_method_names`, so
      `_compose_params_for_diag` propagates the user's
      `temp_min` / `temp_max` / `num_temp_bins` to the C++ side
      instead of silently leaving them at the 1e-3 / 10.0
      built-in defaults.

**Tests added.** `tests/unit/test_auto_thermal.cpp` now covers
LTLM, HYBRID, KPM_DOS, mTPQ, cTPQ through `thermal()` end-to-end,
asserts that every T bin carries finite (non-NaN, non-Inf) data,
asserts that per-sector blocks share a single T axis (the
regression guard for the dim=1 mTPQ short-circuit), and asserts
the silent symmetry skip for TPQ on a symmetry-enabled directory
fixture. The full suite is 289/289 green.

### Fixed — pybind11 overload order: `qed.exact_diagonalization_core(FixedSzOperator, ...)` ran on the full Hilbert space

Discovered while smoke-testing the new ``qed.thermal(...)`` entry
point on a 4-site Heisenberg PBC chain: every Sz sector came back
with the **global** ground-state energy (-2.0), regardless of
``n_up``. Tracing through the bindings showed that the
``exact_diagonalization_core`` overload accepting ``const Operator&``
was registered *before* the ``const FixedSzOperator&`` overload.
``FixedSzOperator`` is-a ``Operator``, and pybind11 dispatches
overloads in registration order, so every ``FixedSzOperator``
argument silently bound to the base-class overload, the dim was
computed as ``1 << op.getNumBits()`` (the FULL Hilbert space), and
the solver ran on the unprojected matvec.

Fix: register the ``FixedSzOperator`` overload first in
``python/qed/_bindings/dispatcher_bindings.cpp`` so pybind11 tries
the derived-class binding before falling back to the base. The
``run_core_fixed`` lambda was already correct -- it set
``params.use_fixed_sz = true`` and passed ``dim =
op.getFixedSzDim()`` -- it was just never reached. Verified with
``FULL`` per sector on the same chain:

```
n_up=0 dim=1   eig=  1.0     (FM all-down)
n_up=1 dim=4   eig= -1.0     (one magnon, lowest)
n_up=2 dim=6   eig= -2.0     (singlet ground state)
n_up=3 dim=4   eig= -1.0
n_up=4 dim=1   eig=  1.0     (FM all-up)
```

This bug had been silently disabling ALL Python-side ``sz=...``
projections through ``qed.diag``/``qed.thermal`` for every method
that does not route via the directory dispatcher. Existing tests
did not catch it because they either (a) ran on the full Hilbert
space (no sz=), (b) routed through the directory dispatcher (which
goes through ``exact_diagonalization_from_directory`` and sets
``params.use_fixed_sz`` directly), or (c) used the C++ tests
exclusively. The new ``qed.thermal(...)`` reference comparison
against the dense full spectrum on the same N=4 chain now agrees
within FTLM statistical tolerance (max |ΔE| ~3e-2 with 32 samples,
max |ΔC_v| ~1e-1).

### Added — One canonical finite-T entry point: `ed::auto_pilot::thermal(...)`

Closes the audit follow-up: *"why are there so many different ways here?
I just want one unified way that automatically gives me the proper
thermodynamics just fully optimized with the sz and symmetry sectors."*

Before this change a user who wanted FTLM thermodynamics on a
Sz-conserving, spatially-symmetric Hamiltonian had to either:

  * call `ed::auto_pilot::solve(H, {solver=FTLM})` -- but auto-Sz would
    project to a single sector, throwing away the partition-function
    contributions of all other Sz blocks; or
  * call `ed::exact_diagonalization(dir, FTLM, params)` and manually
    pick `use_fixed_sz` / `n_up` / `use_symmetry` -- correct, but
    operates on one (Sz, irrep) decomposition at a time and never
    iterates Sz; or
  * write their own outer loop over `n_up`, build a `FixedSzOperator`
    per sector, dispatch FTLM, collect `ThermodynamicData`, and call
    `combine_sector_thermodynamics` themselves.

The new header `include/ed/auto/thermal.h` is the single answer to the
audit question. Public surface:

```cpp
#include <ed/auto/thermal.h>

// In-memory Hamiltonian. Auto-detects Sz conservation; iterates the
// full n_up window; Z-recombines into full-Hilbert thermodynamics.
auto thermo = ed::auto_pilot::thermal(
    H, DiagonalizationMethod::FTLM,
    { .T_min = 0.05, .T_max = 5.0, .num_T = 64 });

// Directory-based Hamiltonian. Auto-detects BOTH Sz conservation and
// spatial symmetry (`automorphism_results/`). Decomposes by
// (Sz x irrep) when both axes are available, Z-recombines across both.
auto thermo = ed::auto_pilot::thermal(
    "ed_dir/", N, /*spin=*/0.5f, DiagonalizationMethod::FTLM,
    { .T_min = 0.05, .T_max = 5.0, .num_T = 64 });

// Optional: restrict to an adjacent-Sz window when the user knows
// which sectors carry the partition-function weight at the T range
// of interest.
auto thermo = ed::auto_pilot::thermal(
    H, DiagonalizationMethod::FTLM,
    { .sz_min = N/2 - 2, .sz_max = N/2 + 2 });
```

Internals (matvec-unification audit + Phase 9 auto-pilot all the way
through):

  1. `ed::auto_pilot::detail::conserves_sz(op)` decides the Sz axis.
  2. `ed::detail::symmetry_data_present(directory)` (now actually
     recognises `automorphisms.json` / `max_clique.json` /
     `sector_metadata.json` -- see Fixed below) decides the spatial
     axis.
  3. For each `n_up` in `[sz_min, sz_max]` (defaults to `[0, N]`),
     `ed::exact_diagonalization(use_fixed_sz=true, use_symmetry=auto)`
     dispatches the most-optimized kernel available. The streaming-
     symmetry-fixed-Sz kernel itself iterates spatial irreps and
     recombines them via the per-sector thermo hook added in the
     previous Fixed entry.
  4. `ed::core::combine_sector_thermodynamics` Z-recombines across the
     Sz axis to produce the final full-Hilbert `ThermodynamicData`.

Supports every finite-T method `method_produces_sector_thermo` knows
about: FTLM, LTLM, HYBRID, KPM_DOS, mTPQ, cTPQ (and their GPU twins
where the legacy enum values still resolve).

Knobs surfaced through `ThermalOptions`:
  * `T_min`, `T_max`, `num_T` -- temperature grid.
  * `sz_min`, `sz_max` -- optional Sz window (n_up convention).
  * `num_samples`, `ftlm_krylov_dim`, `ltlm_krylov_dim`,
    `kpm_num_moments`, `tolerance`, `random_seed`, `max_iterations`.
  * `auto_tune`, `auto_tune_level` -- mirror of `AutoSolveOptions`.
  * `use_symmetry_if_available`, `use_sz_if_conserved` -- force-off
    toggles for the auto-detection.
  * `output_dir`, `verbose`.
  * `tune_params(EDParameters&)` -- per-sector escape hatch for any
    `EDParameters` field not surfaced directly.

`ThermalResult` returns:
  * `thermo` -- final Z-recombined `ThermodynamicData`.
  * `per_sector[]` -- per-(Sz) breakdown, with each entry's
    `ThermodynamicData` already recombined across irreps internally
    when symmetry was used.
  * `used_sz_decomposition`, `used_symmetry_decomposition` -- which
    auto-axes fired.
  * `ground_state_energy` -- minimum across sectors.

Bonus fix: `ed::detail::symmetry_data_present(directory)` was looking
for `automorphism_results/sectors.json` or
`automorphism_results/generators.json`, neither of which the C++
`generate_automorphisms()` or the Python automorphism tool ever
emit. As a result, every directory-based `ed::exact_diagonalization`
call silently bypassed the streaming-symmetry route, even on
directories that had been processed end-to-end by the tooling. The
auto-detector now recognises the actual layout:
`automorphisms.json`, `max_clique.json`, `sector_metadata.json`,
`minimal_generators.json` (and still accepts the legacy
`sectors.json`/`generators.json` names).

Python parity. The C++ surface is mirrored in
``python/qed/thermal.py`` and re-exported from the top-level as
``qed.thermal(H_or_dir, method="FTLM", ...)``. The Python form
returns a ``ThermalResult`` dataclass with the same headline
``temperatures`` / ``energy`` / ``specific_heat`` / ``entropy`` /
``free_energy`` arrays plus the ``per_sector`` breakdown, and
delegates the per-sector dispatch to
``qed.exact_diagonalization_from_directory`` (directory form) or
``qed.diag`` (in-memory form). The Z-recombination math
(`combine_sector_thermodynamics`) is re-implemented in NumPy on the
Python side for symmetry with the C++ surface; both implementations
read from the same `F_ref = min_s F_s; w_s = e^{-β(F_s-F_ref)} /
Σ; <E> = Σ w_s <E>_s; …` template.

New tests: `tests/unit/test_auto_thermal.cpp` (6 cases):
  1. In-memory FTLM with auto-Sz on a N=4 Heisenberg PBC chain
     reproduces full-Hilbert FTLM within FTLM statistical tolerance.
  2. Sz-window restriction emits exactly the requested per-sector
     blocks (n_up=1..3 on N=4, dims C(4,1), C(4,2), C(4,3)).
  3. Directory FTLM with auto-Sz + auto-symmetry on the same chain +
     Z_4 translation fixtures reproduces unprojected FTLM thermo
     and sets both `used_sz_decomposition` and
     `used_symmetry_decomposition`.
  4. Sz-broken Hamiltonian (single S+ injection) falls back to a
     full-Hilbert FTLM and matches the reference bit-tight.
  5. Method guard: ground-state methods (LANCZOS, FULL) throw.
  6. Sz window validation: `sz_min > sz_max` and out-of-range
     windows throw.

### Fixed — Matvec-unification audit follow-up: finite-T + auto-symmetry recombination

Closes the last correctness gap surfaced when the user asked the audit
question: "make sure end-to-end ed solvers are all correct with the new
auto-Sz + symmetry architecture across the board for first all the
ground state finding algorithms then the finite temperatures (FTLM,
KPM, etc)."

What was wrong:

- The streaming-symmetry kernel
  (`exact_diagonalization_streaming_symmetry` /
  `exact_diagonalization_streaming_symmetry_fixed_sz` in
  `include/ed/core/ed_wrapper_streaming.h`) iterated per sector, ran
  `exact_diagonalization_core(method, sector_dim, …)` once per sector,
  and collected only `sector_results.eigenvalues`. For
  ground-state methods that was fine. But for finite-temperature
  methods (FTLM, LTLM, HYBRID, KPM_DOS, mTPQ, cTPQ) every sector also
  populated `sector_results.thermo_data` -- and those blocks were
  silently dropped. Callers running
  `ed::exact_diagonalization(method=FTLM, use_symmetry=true)` got
  back an empty `EDResults::thermo_data` instead of the full-Hilbert
  thermodynamics.

- `compute_static_response_workflow`, `compute_dynamical_response_workflow`,
  `compute_ground_state_dssf_workflow`,
  `compute_kpm_thermodynamics_workflow`, and
  `create_directory_mpi_safe` all called `MPI_Comm_rank` /
  `MPI_Comm_size` / `MPI_Bcast` / `MPI_Reduce` / `MPI_Barrier`
  unconditionally inside `#ifdef WITH_MPI`. When the workflow was
  invoked from a Catch2 unit-test binary (no `MPI_Init`), those
  collectives aborted the process. This caused the pre-existing flake
  in `auto_pilot::dssf::compute runs static_thermal end-to-end on a
  4-site chain`.

What changed:

- New `ed::core::combine_sector_thermodynamics` helper
  (`include/ed/core/sector_thermo.h`): generic free-energy
  Z-recombination of per-sector `ThermodynamicData` blocks into a
  single full-Hilbert thermo. Operates on the generic
  `ThermodynamicData` payload so it works uniformly for FTLM, LTLM,
  HYBRID, KPM_DOS, mTPQ, cTPQ -- whatever the finite-T solver populated.
  Math mirrors the existing `combine_ftlm_sector_results` (free-energy
  shift + Z-weighted mixture rule for `<E>`, `<E^2>`, `Cv`) but is
  decoupled from the FTLM envelope.

- New `ed::core::method_produces_sector_thermo` predicate: returns true
  exactly for the finite-T methods that populate
  `EDResults::thermo_data`. The streaming-symmetry kernel uses this to
  decide whether to gather per-sector blocks and call the combiner.

- Both `exact_diagonalization_streaming_symmetry` and
  `exact_diagonalization_streaming_symmetry_fixed_sz` now stash
  per-sector `ThermodynamicData` during the loop and call the combiner
  after the loop (when no `selected_sectors` filter is active). The
  combined block is written into `results.thermo_data` of the returned
  `EDResults`.

- `MPI_Initialized` guards added to every collective in
  `src/cli/workflows.cpp` and `include/ed/core/system_utils.h`. The
  workflow now degrades gracefully to single-process semantics when MPI
  is not active, so the unit tests no longer abort on first
  collective call.

New test coverage (16 new tests; full suite 277/277 passing):

- `tests/unit/test_ed_solver_matrix_e2e.cpp` (9 cases) -- end-to-end
  validation that every ground-state algorithm reachable from
  `ed::auto_pilot::solve` (FULL, LANCZOS, BLOCK_LANCZOS, KRYLOV_SCHUR,
  DAVIDSON, LOBPCG, plus the heuristic default-picker, the auto-Sz
  dimension-cap regression, and the `auto_basis=Off` escape hatch)
  produces the same dense-reference ground-state energy on a small
  Heisenberg PBC chain.
- `tests/unit/test_sector_thermo.cpp` (4 cases) -- analytic correctness
  of `combine_sector_thermodynamics`: partition of a dense spectrum
  into sectors recombines to within 1e-8 of the direct dense thermo;
  the result is invariant under partition choice; bad inputs throw;
  `method_produces_sector_thermo` classifies the GS vs finite-T methods
  exactly.
- `tests/unit/test_dispatch_streaming_thermo.cpp` (2 cases) -- live
  integration through `ed::exact_diagonalization(..., FTLM, ...)` on a
  4-site Heisenberg PBC chain with Z_4 translation symmetry: the
  recombined thermo agrees with the unprojected FTLM reference within
  FTLM statistical tolerances (max relative `|ΔE|` < 0.15, max `|ΔF|`
  < 0.5). The companion FULL case verifies the dispatch wiring lands
  on the correct ground state through the streaming path.

### Changed — Matvec-unification: a single matrix-vector core for the whole library

A full-audit refactor that collapses every Hamiltonian × vector code path
in the codebase onto one polymorphic interface. The user's original
complaint -- "the matvec implementation is lousy and clumsy, no matter
what ED solver or whether I'm doing DSSF" -- is the root issue this
arc addresses.

What changed at the interface layer:

- New `ed::matvec::MatVecOperator` interface
  (`include/ed/matvec/matvec.h`): a single virtual `apply(in, out,
  size)` + `dim()` / `memory_space()` / `is_hermitian()` /
  `description()` surface that every Hamiltonian wrapper in the library
  now implements. The companion `MemorySpace` enum
  (`include/ed/matvec/memory_space.h`) tags whether the operator
  expects `Host`, `CudaDevice`, `DistributedHost`, or
  `DistributedCudaDevice` vectors; solvers and DSSF can dispatch on
  this without caring which concrete operator they have.
- New `ed::matvec::Backend` interface (`include/ed/matvec/backend.h`)
  + `backends/cpu_backend.h`: level-1 BLAS (`axpy`, `dot`, `norm`,
  `scale`, `copy`) and allocation/copy primitives, decoupled from the
  operator. CUDA / MPI backends slot in behind the same surface.
- New `ed::matvec::kernel::apply_terms` in
  `include/ed/matvec/term_kernels.h`: the **single source of truth** for
  Hamiltonian term evaluation (diag-1-body, offdiag-1-body, diag-2-body,
  mixed-2-body, offdiag-2-body, three-body), parameterised by a
  compile-time `BasisPolicy` (full / fixed-Sz). Replaces ~3-way
  duplication that used to live in `Operator::apply_optimized`,
  `Operator::apply_real`, and `FixedSzOperator::apply`.

Operator-side inheritance (Phase 2):

- `Operator`, `FixedSzOperator`, `GPUOperator`, `GPUFixedSzOperator`,
  `GPUSymmetrizedOperator`, `DistributedOperator`, and
  `DistributedSymmetryOperator` now publicly inherit from
  `ed::matvec::MatVecOperator`.
- `StreamingSymmetryOperator` and `FixedSzStreamingSymmetryOperator`
  gain a nested `SectorView` (a lightweight, non-owning
  `MatVecOperator` wrapper for a single symmetry sector) so per-sector
  diagonalization slots into the same uniform solver interface.

Auto-pilot (Phase 5):

- `auto_pilot::solve` auto-projects to the Sz=N/2 sector when Sz is
  conserved and no Zeeman field is present
  (`include/ed/auto/solve.h`, `AutoBasis::On` is the new default).
  Opt-out with `AutoBasis::Off` for the legacy full-Hilbert path.
- `ed::exact_diagonalization` auto-promotes `params.use_symmetry = true`
  when `automorphism_results/sectors.json` (or `generators.json`)
  exists in the Hamiltonian directory -- so the streaming-symmetry
  kernel kicks in automatically when the data is there, no extra flag
  needed.

Solver-API consolidation (Phase 4):

- Every CPU solver in `lanczos.h`, `ftlm.h`, `ltlm.h`, `TPQ.h`, `CG.h`,
  and `kpm_dos.h` ships an inline overload that takes
  `const ed::matvec::MatVecOperator&` directly, alongside the legacy
  `std::function<void(const Complex*, Complex*, int)>` overload. The
  bridge `ed::matvec::as_apply_function(op)` lets old call sites keep
  compiling unchanged. New code can hand any concrete operator to any
  solver and let virtual dispatch route to the right matvec.
- Validated by three new sections in `test_lanczos_variants` exercising
  `lanczos` / `block_lanczos` / `krylov_schur` through `MatVecOperator&`.
- New `ed/gpu/gpu_solvers.h` (Phase 4 GPU): mirrors the CPU pattern for
  the GPU stack. Provides two overloads of each entry --
  `ed::matvec::gpu::{lanczos, block_lanczos, davidson, krylov_schur,
  block_krylov_schur, lobpcg, full_diagonalization, ftlm,
  microcanonical_tpq, canonical_tpq}` -- one type-safe
  (`const GPUOperator&`, zero-overhead forward to the existing
  `GPUEDWrapper::runGPU*` kernel) and one polymorphic
  (`const ed::matvec::MatVecOperator&`, runtime-checks the operator's
  `memory_space() == CudaDevice` tag and dynamic_casts to GPUOperator,
  throwing `std::invalid_argument` on a Host operator). Validated by
  three new tests in `test_cpu_gpu_equivalence` covering the
  type-safe path, the polymorphic path, and the Host-operator rejection.

Dispatch collapse (Phase 6):

- New `ed/core/dispatch.h` exposes the single canonical entry
  `ed::exact_diagonalization(directory|files, method, params, format)`.
  The legacy `ed_dispatch_symmetry.h` is now a 55-line forwarder shim
  (preserved for the documented `ed_dispatch::` Python-binding API).
  `ed_wrapper.h` + `ed_wrapper_streaming.h` are internal implementation
  details behind `dispatch.h`.
- `EDParameters` gained `basis_cache_dir` (string) and
  `precompute_basis_only` (bool), promoting them from workflow-specific
  flags into the central parameter bag.

In-memory dispatcher robustness (Phase 7.6):

- `exact_diagonalization_core` now silently falls back to the
  corresponding CPU base method when a deprecated `_GPU` enum variant
  reaches it (e.g. via `auto_pilot::solve` on a `WITH_CUDA=ON` build
  that pre-Phase-7.6 used to hard-throw "GPU methods require
  file-based interface"). Emits a one-line stderr note. Matches the
  documented `auto_pilot::solve` contract: "either GPU runs, or we
  silently fall back to CPU." Callers that genuinely require a GPU
  run should use the file-based `ed::exact_diagonalization()` entry
  point or set `allow_fallback=false` in `AutoSolveOptions`.

Aggressive cleanup (Phase 7):

- **Phase 7.1**: Deleted the legacy text-based fixed-Sz symmetrized-
  basis generation pipeline (`generateSymmetrizedBasisFixedSz` +
  helpers, ~670 lines) -- streaming/HDF5 is the canonical path.
- **Phase 7.2**: Deleted the chunked-symmetry + disk-streaming-symmetry
  workflows in full (`chunked_symmetry_builder.h`,
  `disk_streaming_symmetry.h`, `ed_wrapper_chunked.h`,
  `run_chunked_symmetry_workflow`, `run_disk_streaming_workflow`,
  associated EDConfig fields and CLI flags -- ~2.3 kLOC). The
  distributed/MPI build is the canonical answer at the scales these
  single-node fallbacks targeted, and is now first-class
  (`DistributedHost` memory space tag on `DistributedOperator`).
  `--disk-streaming` / `--chunked-symm` CLI flags now print a one-line
  deprecation notice and are ignored.
- **Phase 7.3**: Retired the `[[deprecated]]` `use_hybrid_method` flag
  (`method=HYBRID` is the canonical knob).
- **Phase 7.4**: `docs/architecture/{CODEMAP, SCALING,
  IMPLEMENTATION_REPORT}.md` brought up to date with the
  post-unification reality (single entry point, MatVecOperator
  inheritance, MPI replacement of single-node disk paths).
- **Phase 7.5**: Removed ten `[[deprecated]]` accessor shims in
  `EDParameters` (`num_order` → `tpq_taylor_order`, `delta_tau` →
  `tpq_delta_beta`, …). All in-tree callers migrated.

Compatibility:

- The legacy `std::function`-based solver signatures all stay.
- `ed_dispatch::exact_diagonalization_from_directory(...)` stays.
- The `_GPU` and `_GPU_FIXED_SZ` deprecated enum aliases in
  `DiagonalizationMethod` are kept as zero-cost compile-time aliases
  (Python-binding / HDF5 metadata ABI surface) -- they canonicalise to
  `{base, use_gpu=true, use_fixed_sz=…}` via
  `canonicalize_method_and_flags`.

Tests:

- 254 / 254 unit + MPI tests pass on a `WITH_CUDA=OFF, WITH_MPI=ON`
  build (the matvec-unification regression sweep ran on this config
  throughout Phases 1-7.5).
- 261 / 261 unit + MPI + GPU tests pass on a
  `WITH_CUDA=ON, WITH_MPI=ON` build (RTX 4080 SUPER, CUDA 12.9, archs
  70-90) -- the additional 7 cases cover Phase 4 GPU overloads,
  CPU/GPU equivalence on Heisenberg, mixed-precision Lanczos, and the
  Phase 7.6 in-memory fallback.
- New regression tests:
  * `test_auto_solve.cpp` -- two new sections validating the
    auto-Sz-to-N/2 projection on a Heisenberg ring and confirming the
    Zeeman-field path bypasses it.
  * `test_lanczos_variants.cpp` -- three new sections exercising the
    new `MatVecOperator&` overloads of `lanczos` / `block_lanczos` /
    `krylov_schur` and asserting numerics match the `std::function`
    path.
  * `test_cpu_gpu_equivalence.cpp` -- three new sections covering
    `ed::matvec::gpu::lanczos(GPUOperator&)` vs the legacy
    `GPULanczos` class on 8-site Heisenberg, the polymorphic
    `MatVecOperator&` overload matching the same reference, and
    runtime rejection of Host-memory operators.

### Added — One-call ED-solver auto-tuner (`qed.auto_tune.tune_diag` / `ed::auto_pilot::diag::apply_auto_tune`)

Companion to the DSSF auto-tuner shipped earlier in this release. Both
`qed.diag(H, ..., auto_tune=True, level="balanced")` and
`ed::auto_pilot::solve(H, opts)` now auto-pick **every family-specific
EDParameters knob** the existing `_make_params` did not cover:

- Convergence: `tolerance`, `max_iterations`, `max_subspace`,
  `block_size` — sized from sector dim + `num_eigenvalues` with three
  aggressiveness levels.
- Eigenvalue sub-solvers: `arpack_ncv` (≥ 2k+1 with `2/4/6 × k`
  per-level multiplier).
- Thermal: `ftlm_krylov_dim`, `ltlm_krylov_dim`, `ltlm_ground_krylov`,
  `tpq_taylor_order` (Taylor-truncation bound from ‖H‖·Δβ),
  `tpq_delta_beta` (capped at `0.5 / ‖H‖`), `num_samples`
  (∝ 1/√D, per-level [min, max] clamp).

All overrides are sentinel-only — anything the caller sets via direct
kwargs, `extra_params={}`, or `tune_params=` (C++) passes through.

- New `python/qed/auto_tune.py` exports `tune_diag(...) → TunedDiagKnobs`
  alongside `tune_dssf`. `TunedDiagKnobs.to_extra_params()` renders to
  the `extra_params=` dict consumed by `qed.diag`.
- New header `include/ed/auto/diag_tune.h` — header-only mirror of the
  Python heuristics + `apply_auto_tune(EDParameters&, sector_dim,
  num_eigenvalues, op, ov)` sentinel-based mutator.
- `qed.diag(...)` + `ed::auto_pilot::AutoSolveOptions` gain
  `auto_tune=True` and `level=` / `auto_tune_level=` knobs.
- New tests: 12 Python (`test_auto_tune.py`) + 10 Catch2
  (`tests/unit/test_diag_tune.cpp`).
- `docs/guides/one_call_api.md` extended with §3a "ED-solver auto-tuning".

### Added — One-call DSSF auto-tuner (`qed.auto_tune` / `ed::auto_pilot::dssf::apply_auto_tune`)

Both the Python `qed.dssf.compute(...)` and C++ `ed::auto_pilot::dssf::compute(...)`
entry points now auto-pick **every internal DSSF knob** (η broadening,
ω window, # ω-grid points, FTLM/continued-fraction Krylov dim, # random
vectors, KPM moments, device backend) from the operator size + build
flags. Anything the caller specifies explicitly is honoured; only
sentinels are filled. Three aggressiveness levels (`conservative`,
`balanced`, `aggressive`).

- New module `python/qed/auto_tune.py`: pure heuristic helpers
  (`estimate_bandwidth`, `pick_eta`, `pick_omega_window`,
  `pick_krylov_dim`, `pick_num_random_vectors`, `pick_kpm_moments`,
  `pick_device`, `tune_dssf`). Returned as a frozen
  `TunedDSSFKnobs` dataclass; `to_cli_args(method=...)` renders the
  `--dyn-* / --static-* / --ftlm-*` CLI flags consumed by `./ED dssf`.
- `qed.dssf.compute(...)` extended with `eta=`, `krylov_dim=`,
  `num_random_vectors=`, `kpm_moments=`, `bandwidth=`, `device=`,
  `level=`, `auto_tune=` kwargs. `_VALID_METHODS` now includes
  `kpm_thermodynamics`.
- New header `include/ed/auto/dssf_tune.h`: header-only mirror of the
  Python heuristics, plus an `apply_auto_tune(EDConfig&, sector_dim,
  op, overrides)` helper that mutates only struct-default sentinels.
- `ed::auto_pilot::dssf::AutoDSSFOptions` extended with `auto_tune`,
  `tune_overrides`, `sector_dim_hint`. `compute(...)` now copies the
  caller's `EDConfig`, applies the tuner, and re-points
  `request.config` for the dispatcher call (preserves the
  `const EDConfig*` ABI).
- New tests:
  - `python/tests/test_auto_tune.py` — 14 tests (bandwidth, omega
    window, eta scaling/ordering, krylov + random monotonicity,
    device picker, `tune_dssf` overrides + CLI rendering).
  - `tests/unit/test_dssf_tune.cpp` — 8 Catch2 tests (mirror of the
    Python lockdown so drift between the two implementations breaks
    the build).
- New documentation page `docs/guides/one_call_api.md` — the canonical
  reference for **`qed.diag(H, ...)`** and
  **`qed.dssf.compute(directory, T=, omega=)`** (plus their C++
  counterparts), the auto-selection rules, the per-method knob
  defaults, and the override surface.

### Audit follow-ups (May 2026)

- **#3** Surface `KPM_THERMODYNAMICS` as a first-class `DSSFMethod` in
  the `ED dssf <method>` CLI help / error text. Lockdown tests in
  `tests/unit/test_dssf_engine.cpp` now cover round-trip,
  case-insensitivity, numeric stability (= 4u), and null-config
  rejection for the KPM branch (commit 094d47c).
- **#2** FixedSz-projected operators now take the GPU path **and** the
  CPU fallback path correctly. The CPU bug was that
  `Operator::apply` is non-virtual, so building a `std::vector<Operator>`
  out of derived `FixedSz*Operator` instances sliced the override away.
  Fix: parallel `std::vector<std::shared_ptr<FixedSzOperator>>` arrays
  in `ObservablePairs`, populated only when `spec.use_fixed_sz=true`,
  with workflow-side lambdas dispatching on the use-case (commit
  1ffb0b1).

### Removed — NLCE workflow extracted to a standalone repository

The Numerical Linked Cluster Expansion driver (`workflows/nlce/`) has
been split into its own GitHub repository,
**[QED_NLCE](https://github.com/ze-bang/QED_NLCE)**. NLCE never linked
against the QED C++ libraries — every cluster diagonalization is a
`subprocess` call to `./ED` — so the dependency is purely runtime.
This separation lets the NLCE toolkit evolve independently.

- `workflows/` and `workflows/nlce/` removed from this repo (49 commits
  of NLCE history were preserved verbatim into QED_NLCE via
  `git filter-repo`).
- Python package renamed `workflows.nlce` → `qed_nlce`.
- Unified CLI is now `qed-nlce` (or `python -m qed_nlce`); install with
  `pip install git+https://github.com/ze-bang/QED_NLCE.git`.
- `python/tests/test_nlce_package.py` moved to `qed_nlce/tests/`.
- `examples/13_nlce_full_workflow.sh` now invokes `qed-nlce` and
  fails fast with an install hint if the package is missing.
- `docs/guides/usage.md` Mode 5 and `docs/guides/python_api_coverage.md`
  updated to point at the new repo.

### Changed — repository / package rename to **QED**

Renamed top-level identifiers (C++ `namespace ed::` and binary names
`ED`/`ed_distributed_main`/`compute_bfg_order_parameters[_gpu]` are
**unchanged**; this is a user-facing rebrand only).

- **CMake project**: `ExactDiagonalization` → `QED`. Downstream consumers
  now use:
    ```cmake
    find_package(QED CONFIG REQUIRED)
    target_link_libraries(myapp PRIVATE QED::ed_solvers_cpu)
    ```
  Installed config artifacts are now `lib/cmake/QED/QEDConfig.cmake`,
  `QEDConfigVersion.cmake`, `QEDTargets.cmake` (export namespace
  `QED::`). The template lives at `cmake/QEDConfig.cmake.in`.
- **Python distribution / import**: `quantum_ed` → `qed`. The package
  directory moved from `python/quantum_ed/` to `python/qed/`. Update
  call sites:
    ```python
    import qed                  # was: import quantum_ed as qed
    from qed import workflow    # was: from quantum_ed import workflow
    ```
  `pyproject.toml`'s `name = "qed"`. The pybind11 module is now
  `qed._core` (was `quantum_ed._core`).
- **Workspace folder / GitHub repo**: `exact_diagonalization_cpp/` →
  `QED/`. Update any external scripts that hardcode the path
  (`launch_*.sh`, helper analysis drivers).
- **Binding source**: renamed `python/qed/_bindings/quantum_ed_bindings.cpp`
  → `python/qed/_bindings/qed_bindings.cpp`.
- **Benchmark CLI**: `--skip_quantum_ed` → `--skip_qed`; functions
  `time_quantum_ed_apply`/`time_quantum_ed_lanczos` → `time_qed_apply`/`time_qed_lanczos`.
- **Old build directories must be deleted** — stale `EDConfig.cmake`
  files and frozen absolute paths in CMake caches will not migrate.

Historical CHANGELOG entries below intentionally still reference
`python/quantum_ed/...` because they describe the codebase state at
the time those changes landed.

### Added — Phase H: auto-aggregate FTLM/TPQ across symmetry sectors on MPI path

Closes the last MPI symmetry footgun: previously
`qed.diag(H, device='mpi'/'mpi_gpu', solver='FTLM'|'mTPQ'|'cTPQ',
symmetry=...)` would dispatch ONE call to `ed_distributed_main`
with `--sector-index 0` and return that sector's `Z_q(beta)`,
`<H>_q(beta)` -- silently giving the user a partial-trace thermal
curve. Callers had to manually loop sectors and Z-weight-average
themselves.

The Python dispatcher (`_diag_via_mpi`) now does this for them:
when `symmetry=` is set, the solver is thermal (`ftlm`/`tpq`),
and `sector=` is NOT supplied, the binary is invoked once per
irrep and the per-sector results are combined via the
additive-partition-function rule

```
Z(beta)   = sum_q Z_q(beta)
<H>(beta) = sum_q Z_q(beta) * <H>_q(beta) / Z(beta)
```

so `EDResults.thermo_data.energy[i]` is the full-trace
`<H>(beta_i)` -- matching the in-process `_diag_with_symmetry`
contract. Pass an explicit `sector=` to opt out and keep the raw
per-sector arrays.

For the non-thermal MPI path (`lanczos`/`krylov_schur` + symm)
the dispatcher still picks `sector=0` by default; ground-state
aggregation across sectors is the user's responsibility (the GS
lives in exactly one irrep and is not a Z-weighted sum).

- `python/quantum_ed/workflow.py`:
  - Refactored `_diag_via_mpi` so the per-spawn `--sector-index`
    and `--result-file` are appended inside a dispatch loop;
    `base_binary_args` and `sym_args` are built once.
  - New helpers `_read_mpi_thermo_arrays` (returns the raw
    `(betas, energies, Z)` triple from a thermal HDF5 file --
    the user-facing EDResults discards `Z`) and
    `_aggregate_thermal_sectors` (Z-weighted average across
    paths, with a beta-grid consistency check).
  - The `compute_eigenvectors=True` + multi-sector aggregation
    combination is rejected with a clear error (per-rank slabs
    from different irreps would not stitch).
- `python/tests/test_workflow.py`: new
  `test_mpi_symm_thermal_aggregates_across_sectors[FTLM|mTPQ,
  mpi|mpi_gpu]` (4 cells) monkeypatches `run_distributed`,
  emits synthetic per-sector `Z_q`, `<H>_q` arrays, and asserts
  (a) one spawn per irrep with distinct `--sector-index`, (b)
  the returned `thermo_data.energy` equals the closed-form
  Z-weighted average. New
  `test_mpi_symm_thermal_explicit_sector_skips_aggregation[mpi|
  mpi_gpu]` confirms passing `sector=[0]` falls back to the
  one-shot per-sector behaviour.
- `docs/guides/workflow.md`: footnote 2 rewritten -- now
  distinguishes the in-process aggregator (which loops sectors
  in C++) from the new MPI Phase-H Python aggregator (which
  spawns per irrep and combines in Python). Footnote 1 updated
  to point users at `device='mpi'/'mpi_gpu'` for TPQ + symm.
  New footnote 3 documents the MPI TPQ + symm + Phase-H path.
  The `mTPQ`/`cTPQ` MPI bullet no longer says "the caller is
  responsible for FTLM-style aggregation" -- it's automatic.
- The Phase-E TPQ + symm rejection message in
  `_dispatch_to_diagonalization_kernel` now references Phase H
  ("the dispatcher Z-weight-aggregates across sectors when
  `sector=` is omitted") instead of telling the user to
  aggregate themselves.

### Added — Phase G: bare distributed FixedSz path (no symmetry)

Wires `qed.diag(H, device='mpi'/'mpi_gpu', sz=k)` (without
`symmetry=`) end-to-end by routing through
`DistributedSymmetryOperator` with a TRIVIAL one-element symmetry
group (identity only) plus the Phase F `n_up` filter. With `|G|=1`
every orbit is a singleton, so the popcount-filtered orbit basis
IS exactly the `C(N, k)` binomial basis -- the same sub-block
`FixedSzOperator` carries in-process. No new C++ class, no new
distributed kernel: all four solvers (lanczos / krylov_schur /
ftlm / tpq) and their GPU siblings inherit the FixedSz path
automatically.

- `python/quantum_ed/workflow.py` `_diag_via_mpi`: replaces the
  Phase-F `NotImplementedError` for the bare-`sz=` branch with a
  trivial-group construction (`group_from_generators(N,
  [identity])`), writes the symmetry directory, and forwards
  `--use-symmetry --sector-index 0 --sz <k>`.
- `tests/unit/test_distributed_symmetry_operator.cpp`: new
  TEST_CASE `Phase G (trivial group + sz) is the binomial basis`
  on np {1,2,4} for `N in {4, 6}`. For every `n_up in [0, N]`,
  verifies: `global_dim() == C(N, n_up)`; `orbit_reps()` are
  exactly the popcount-`n_up` states in lex order; orbit sizes
  all equal 1; orbit norms_sq all equal 1; the apply matches
  the serial `Operator::apply` restricted to popcount-`n_up`
  positions; orthogonal complement leak <= 1e-10
  (`[H, popcount] = 0` check).

This closes the last gap in the device × path × Sz cube. The
Solver × device matrix now has no silently-dropping cells.

### Added — Phase F: Sz quantum-number filter on the distributed
symmetry-projected basis

Combines a `U(1)`-Sz block with a lattice symmetry sector on the
distributed path. Site permutations preserve popcount, so filtering
the orbit basis to `popcount(rep) == n_up` is a closed sub-block;
the in-process kernel had this via `FixedSzOperator`, but the
distributed path silently dropped `sz=` until now.

- New optional field `int n_up = -1` on
  `SymmetryGroupInfo` (`include/ed/core/symmetry_metadata.h`).
  `-1` means "no Sz constraint" (back-compat default).
- `DistributedSymmetryOperator` ctor reads `info.n_up` and skips
  orbits whose representative popcount does not match (Step 1
  BFS now pushes a zero-norm placeholder for filtered orbits so
  Step 2's existing `kZeroNormTolerance` filter drops them).
  Zero diff on the construction / apply hot path when `n_up == -1`.
- `ed_distributed_main` exposes `--sz <n_up>`. Validates
  `0 <= n_up <= num_sites`, requires `--use-symmetry`, and writes
  `op->symmetry_info.n_up` before any `DistributedSymmetryOperator`
  is constructed (so all four solver branches and their `_gpu`
  siblings inherit the filter automatically).
- `python/quantum_ed/workflow.py` `_diag_via_mpi` forwards
  `sz=` as `--sz <int>` when `symmetry=` is also set; without
  symmetry, it now raises a clear `NotImplementedError` instead
  of silently dropping the constraint.
- New test cases on `test_distributed_symmetry_operator` (np
  {1,2,4}; +4 cases × 8 assertions per): for every Sz value k in
  [0, N], the filtered `global_dim()` equals the count of
  unfiltered orbits with popcount k; the filtered apply matches
  the unfiltered apply restricted to the popcount-k subspace
  (and the orthogonal complement leaks 0 -- the
  `[H, popcount] = 0` check); `sum_k filtered_dim == full_dim`;
  and `n_up in {0, N}` give `global_dim == 1`.

This wires the long-standing "ftlm gpu + symm + sz" combination
(and `lanczos`, `krylov_schur`, `tpq` × `gpu`/`mpi+gpu` × `symm`
× `sz`) end-to-end without changing any of the seven
`*_symmetry` entry-point signatures.

### Fixed — Path × device matrix reachability audit (post Phase E)

After auditing every ✅ cell in the Solver × device support matrix
(see `docs/guides/workflow.md` lines 419-432) for end-to-end
reachability from `qed.diag(...)`:

- `python/quantum_ed/workflow.py` `_SOLVER_DEVICE_KERNELS`:
  `KRYLOV_SCHUR.mpi_gpu` and `FTLM.mpi_gpu` were stale `False`
  even though Phase D step 3 (`distributed_krylov_schur_gpu` /
  `distributed_krylov_schur_gpu_symmetry`) and the existing
  `distributed_ftlm_gpu` kernel (footnote ⁵) wire the cells.
  Flip both to `True` so `solver_device_support()` agrees with
  the documented matrix and the C++ dispatcher.
- `python/quantum_ed/workflow.py` `_diag` TPQ + symmetry rejection:
  Phase E wired `distributed_tpq_symmetry` (CPU MPI) and
  `distributed_tpq_gpu_symmetry` (multi-GPU) which DO project onto
  a single sector and return per-sector sample-averaged
  `<H>(beta)` (caller responsible for FTLM-style aggregation
  across sectors). The blanket `ValueError` is now scoped to
  `device='cpu'/'gpu'` (where the in-process streaming kernel
  still falls back to Lanczos per `ed_wrapper.h:1458`); the MPI
  device cells flow through `_diag_via_mpi`.
- `python/quantum_ed/feasibility.py`: TPQ + symm hard-block
  similarly relaxed for `device='mpi'/'mpi_gpu'`.

No new unit tests required — the existing
`test_mtpq_with_symmetry_is_rejected_with_actionable_error` still
passes (the regex `TPQ.*symmetry` matches the new message), and
the 115 workflow + feasibility tests pass.

### Added — Phase E step 2 (path matrix symm × mpi+gpu for TPQ): per-sector multi-GPU canonical TPQ

New function `ed::distributed::distributed_tpq_gpu_symmetry` — the
multi-GPU companion of `distributed_tpq_symmetry`. The on-device
per-sample canonical-TPQ body in `distributed_tpq_gpu.cu` is now
factored into a templated helper `tpq_gpu_impl<CpuDop, GpuOp,
ScatterFn>` shared by both the dense and the symmetry path;
`taylor_step_gpu` is a function template on the GPU operator type
(`DistributedGPUOperator` and `DistributedSymmetryOperatorGPU` both
expose `apply(const MultiGpuCommunicator&, const Complex*, Complex*,
cudaStream_t)`). The new entry point builds a
`DistributedSymmetryOperator` (shared_ptr) + `DistributedSymmetryOperatorGPU`
on the per-group `MultiGpuCommunicator`, and supplies an
orbit-permuted scatter helper so the per-sample seed -> rank-major
host vector path matches the CPU `distributed_tpq_symmetry` exactly.

The CLI now accepts `--mode tpq --gpu --use-symmetry --sector-index k`;
the returned `energy[b]` is the contribution from sector `k` alone,
matching the CPU symm convention.

New unit test `test_distributed_tpq_gpu_symmetry` (registered as
`phase3c` at np ∈ {1, 2, 4}, SKIPs when no CUDA device is visible)
cross-checks the GPU symm TPQ `energy(beta)` at every momentum sector
of an N=4 PBC Heisenberg chain against the CPU
`distributed_tpq_symmetry` at the same seed_offset, delta_beta, and
taylor_order within 1e-9 relative.

### Added — Phase E step 1 (path matrix symm × mpi for TPQ): per-sector canonical TPQ

New function `ed::distributed::distributed_tpq_symmetry` — the
symmetry-projected companion of `distributed_tpq`. The per-sample
canonical-TPQ body in `distributed_tpq.cpp` (initial-state scatter,
`taylor_step` propagation, energy/variance measurement, world-rank
reduction) is now factored into a templated helper
`tpq_impl<Op, ScatterFn>` shared by both the dense and the symmetry
path. `taylor_step` itself is now a function template on the operator
type (`DistributedOperator` and `DistributedSymmetryOperator` both
expose `apply(const Complex*, Complex*)`, `local_size()`, `comm()`).
The new entry point builds a `DistributedSymmetryOperator` on the
per-group communicator and supplies an orbit-permuted scatter helper
so the per-sample seed -> rank-major host vector path matches the
existing FTLM symm convention exactly.

The CLI now accepts `--mode tpq --use-symmetry --sector-index k`
(the previous explicit fail at `ed_distributed_main.cpp` is gone).
The returned `energy[b]` is the sample-averaged `<H>(beta)` measured
inside sector `k` only — caller-side aggregation is required to
reconstruct full-space thermal observables, mirroring the convention
established by `distributed_ftlm_symmetry`. The `--mode tpq --gpu
--use-symmetry` combination still emits an actionable diagnostic
pointing users at the CPU symm path; GPU symm TPQ is Phase E step 2.

New unit test `test_distributed_tpq_symmetry` (registered as `mpi`
at np ∈ {1, 2, 4}) verifies that the sample-averaged per-sector
`<H>(beta)` converges to the dense-projected exact thermal energy of
that sector on an N=4 PBC Heisenberg chain, and replicates `energy[b]`
across ranks within a group.

### Added — Phase D step 5 (path matrix symm × mpi+gpu): per-sector multi-GPU FTLM

New function `ed::distributed::distributed_ftlm_gpu_symmetry` — the
multi-GPU companion of `distributed_ftlm_symmetry`. The on-device
J&P trace-estimator body in `distributed_ftlm_gpu.cu` is now factored
into a templated helper `ftlm_gpu_impl<CpuDop, GpuOp, ScatterFn>`
shared by both the dense and the symmetry path; the new entry point
builds a `DistributedSymmetryOperatorGPU` (and an optional
symmetry-projected observable operator) on the per-group
`MultiGpuCommunicator`, and supplies an orbit-permuted scatter
helper so the per-sample seed -> rank-major host vector path
matches the CPU `distributed_ftlm_symmetry` exactly.

The CLI now accepts `--mode ftlm --gpu --use-symmetry --sector-index k`;
the returned `Z[b]` and `O_expectation[b]` are the contributions
from sector `k` alone, matching the CPU symm convention. The caller
aggregates across sectors when reconstructing the full-space
partition function.

New unit test `test_distributed_ftlm_gpu_symmetry` (registered as
`phase3c` at np ∈ {1, 2, 4}, SKIPs when no CUDA device is visible)
cross-checks the GPU symm FTLM Z(beta) at every momentum sector of
an N=4 PBC Heisenberg chain against the CPU `distributed_ftlm_symmetry`
at the same seed_offset within 1e-9 relative.

### Added — Phase D step 4 (path matrix symm × mpi): per-sector FTLM

New function `ed::distributed::distributed_ftlm_symmetry` — the
symmetry-projected companion of `distributed_ftlm`. The per-sample
J&P trace-estimator body in `distributed_ftlm.cpp` is now factored
into a templated helper `ftlm_impl<Op, LanczosCall>` shared by both
the dense and the symmetry path; the new entry point builds a
`DistributedSymmetryOperator` (and an optional symmetry-projected
observable operator) on each per-group subcommunicator and routes
the per-sample Krylov build through `distributed_lanczos_symmetry`.

The CLI now accepts `--mode ftlm --use-symmetry --sector-index k`
on the CPU MPI path; the returned `Z[b]` and `O_expectation[b]` are
the contributions from sector `k` alone (un-multiplied by any
irrep-multiplicity weight). The caller is responsible for
aggregating across sectors when reconstructing the full-space
partition function. The GPU FTLM path with `--use-symmetry` still
fails with a Phase D step 5 pointer.

New unit test `test_distributed_ftlm_symmetry` (registered as
`phase3b` at np ∈ {1, 2, 4}) verifies that summing per-sector Z(β)
over every momentum sector of an N=4 PBC Heisenberg chain
reproduces the full-space partition function within the J&P
trace-estimator noise floor.

### Added — Phase D step 3 (path matrix symm × mpi+gpu): on-device Krylov-Schur

New function `ed::distributed::distributed_krylov_schur_gpu_symmetry`
— the multi-GPU companion of `distributed_krylov_schur_symmetry`. The
on-device thick-restart Lanczos body in
`distributed_krylov_schur_gpu.cu` is now factored into a templated
`ks_gpu_impl<CpuDop, GpuOp>` shared by both the unsymmetrised and
symmetrised entry points; the CPU op surface required is
`{rank, local_size, global_dim, comm}` and the GPU op surface is
`apply(gpu_comm, const Complex*, Complex*, cudaStream_t)`. The
`scatter_initial_vector_host(cpu_dop, seed, host_v)` overload is
resolved at template-instantiation time on the CPU operator type.

Dispatcher: the `--mode krylov_schur --gpu --use-symmetry` carve-out
at `ed_distributed_main.cpp` (added in Phase D step 2 as a queue
pointer) is gone.

Tests: `test_distributed_krylov_schur_gpu_symmetry` cross-checks the
GPU symm KS ground-state eigenvalue against
`distributed_krylov_schur_symmetry` at all Z_N momentum sectors of
N ∈ {4, 6} Heisenberg chains, abs tol 1e-9. Build-only on CI's CUDA
build-only lane; runtime-tested via the `runtime_supports_gpu_ks_sym()`
SKIP gate.

The remaining Phase D wiring — `FTLM × symm` (CPU & GPU) — is queued
as steps D4-D5.

### Added — Phase D step 2 (path matrix symm × mpi): templated CPU Krylov-Schur

New function `ed::distributed::distributed_krylov_schur_symmetry` — the
symmetry-projected companion of `distributed_krylov_schur`. The thick-
restart Lanczos body is now a template `distributed_krylov_schur_impl<OpT,
LanczosFallback>` shared by both the unsymmetrised and the symmetrised
entry points; the only operator-specific hooks are the
`scatter_initial_vector` overload (rank-major + LPT-orbit-permuted layout
for the symm path) and the tiny-problem fallback callback (routes to the
matching `distributed_lanczos` flavour).

Dispatcher: the `--mode krylov_schur --use-symmetry` carve-out at
`ed_distributed_main.cpp` is gone for the CPU branch; the GPU branch
still fails with a Phase D step 3 pointer.

Tests: `test_distributed_krylov_schur_symmetry` cross-checks the leading
eigenvalue per momentum sector against `distributed_lanczos_symmetry` on
N=4/N=6 Heisenberg chains at np ∈ {1, 2, 4} (all 27 CPU MPI tests pass
locally).

### Added — Phase D step 1 (path matrix symm × mpi+gpu): on-device Lanczos

New function `ed::distributed::distributed_lanczos_gpu_symmetry` — the
multi-GPU companion of the CPU `distributed_lanczos_symmetry`. Builds
a `DistributedSymmetryOperatorGPU` internally from the supplied CPU
`DistributedSymmetryOperator` and runs the same per-iteration recipe
as `distributed_lanczos_gpu` (cublasZdotc + NCCL allreduce of two
doubles for alpha/beta, cublasZaxpy for the recurrence updates,
host-side tridiag eigensolve for convergence) but with the orbit-row
SpMV from Phase C.

Dispatcher: the `--mode lanczos --gpu --use-symmetry` carve-out at
`ed_distributed_main.cpp` is gone — the code path now routes through
the new function. Initial vector is generated identically to the CPU
`distributed_lanczos_symmetry` (deterministic in NATURAL orbit order,
rank-major + LPT-orbit-permuted scatter) so eigenvalues are bit-
comparable across CPU/GPU at the same seed.

Tests: `test_distributed_lanczos_gpu_symmetry` cross-checks GPU symm
eigenvalues vs CPU symm at all Z_N momentum sectors of N ∈ {4, 6}
Heisenberg chains, abs tol 1e-9 (Lanczos noise floor at
max_iter=200). Build-only on the CUDA build-only CI lane; runtime-
tested via `runtime_supports_gpu_lanczos_sym()` SKIP gate.

The remaining Phase D wiring — `KRYLOV_SCHUR × symm` (CPU & GPU) and
`FTLM × symm` (CPU & GPU) — are queued as steps D2-D5.

### Added — Phase C (device matrix MPI+GPU): on-device symmetry-projected SpMV

New class `ed::distributed::DistributedSymmetryOperatorGPU` — a
multi-GPU companion of the CPU `DistributedSymmetryOperator`. Wraps
an existing CPU instance (orbit basis, LPT-balanced
`OrbitPartition`, `OrbitHaloPlan`, projected CSR row slab) and
flattens the per-row CSR (`row_offsets`, `col_idx`, `is_local` mask,
complex coefficients) plus the halo plan's `send_local_idx` into
device-resident SoA arrays at construction time. `apply()` runs:

1. A pack kernel — `d_send_buf[k] = d_x_local[d_send_local_idx[k]]`.
2. NCCL pairwise `ncclSend` / `ncclRecv` (one `ncclGroupStart`/End)
   over the orbit-aware halo schedule on device buffers.
3. A CSR sparse-matvec kernel — one thread per local row; each
   non-zero loads from `d_x_local` or the just-filled halo recv
   buffer based on the `is_local` tag.

This is the foundation for Phase D, which will wire every
distributed GPU solver (`KRYLOV_SCHUR`, `BLOCK_*`, `DAVIDSON`,
`LOBPCG`, `mTPQ`, `cTPQ`) through this primitive when
`--use-symmetry` is set on the `--gpu` path.

CPU-side: the `DistributedSymmetryOperator` header gains three
narrow public accessors (`csr_row_col_idx()`, `csr_row_is_local()`,
`csr_row_coeff()`) so the GPU mirror can flatten the row slab
without becoming a friend.

Tests: `test_distributed_symmetry_operator_gpu` cross-checks the
GPU `apply()` against the CPU `apply()` on Heisenberg chains
N ∈ {4, 6} (OBC + PBC) over every Z_N momentum sector at np
∈ {1, 2, 4}, abs tol 1e-12. Build-only on the CUDA build-only CI
lane; runtime-tested via the `runtime_supports_gpu_sym()` SKIP
gate. Compiled iff `WITH_MPI=ON && WITH_CUDA=ON && NCCL_FOUND`.

### Added — Phase B (device matrix MPI+GPU): on-device Krylov-Schur

Implements `distributed_krylov_schur_gpu`, the multi-GPU companion of
the CPU `distributed_krylov_schur`. Closes the previously-❌
`KRYLOV_SCHUR × mpi+gpu` cell of the solver × device support matrix.

- **`include/ed/distributed/distributed_krylov_schur_gpu.h`** (new) —
  public API: `distributed_krylov_schur_gpu(op, options, world_comm,
  device_index = -1) -> DistributedLanczosResult`. Reuses the CPU
  `DistributedLanczosOptions` / `DistributedLanczosResult` so callers
  see a single type. Header is only compiled under `#ifdef WITH_MPI`;
  consumers guard the include with `#ifdef ED_HAVE_NCCL`.
- **`src/distributed/distributed_krylov_schur_gpu.cu`** (new, wrapped
  in `#ifdef ED_HAVE_NCCL`) — same thick-restart Lanczos with
  Ritz-pair locking as the CPU sibling, but the in-cycle Krylov basis
  V[0..m-1] **and** the locked Ritz vectors live in two contiguous
  device slabs (`(m_max + |locked|) × local_n × 16 B` per rank).
  Twice-CGS re-orth against both sets is coalesced — each pass packs
  the |set| local `cublasZdotc` results into a single NCCL allreduce
  of `2·|set|` doubles, then runs |set| `cublasZaxpy` calls to
  subtract. SpMV goes through `DistributedGPUOperator` (NCCL halo +
  on-device SoA SpMV); inner products / norms via `cublasZdotc` +
  `multi_gpu::all_reduce_sum_complex_double`. The `(m × m)` Eigen
  tridiagonal eigensolve and the Ritz-residual / locking sweep are
  host-side, replicated on every rank; reconstruction of locked Ritz
  vectors `phi = Σ_j U[j,i] V[j]` and of the next-cycle seed runs on
  device via `cublasZaxpy`. **Honest limitation**: the locked Ritz
  vectors stay on device — `compute_eigenvectors=true` is silently
  ignored on the result side, and the dispatcher rejects
  `--compute-eigenvectors` / `--eigenvector-dir` on the GPU path with
  an actionable error.
- **`tests/unit/test_distributed_krylov_schur_gpu.cpp`** (new) —
  three `TEST_CASE`s cross-check the GPU result against the CPU
  `distributed_krylov_schur` on the same MPI_COMM_WORLD with the same
  operator / seeds / `max_iter` / `tol`: N=4 OBC lowest-3, N=6 PBC
  lowest-4, replicated eigenvalues across world ranks. Tolerance
  `1e-8` (absolute, since both kernels lock against `tol = 1e-10`).
  Build-only on CI's CUDA-build-only lane; runtime-tested on dev
  hosts with GPUs (uses the `runtime_supports_gpu_*` SKIP gate).
  Custom `int main` does `MPI_Init` + `Catch::Session().run` +
  `MPI_Finalize`.
- **`cmake/EDLibraries.cmake`** — adds `distributed_krylov_schur_gpu.cu`
  to the `ed_distributed_gpu` STATIC target (compiled iff `WITH_MPI &&
  WITH_CUDA && NCCL_FOUND`).
- **`CMakeLists.txt`** — registers the new test via
  `ed_add_phase3c_test(test_distributed_krylov_schur_gpu ...)` at np ∈
  {1, 2, 4}, only inside `if(TARGET ed_distributed_gpu)`.
- **`src/cli/ed_distributed_main.cpp`** — `--mode krylov_schur` now
  accepts `--gpu`: when `ED_HAVE_NCCL` is defined the dispatcher
  invokes `distributed_krylov_schur_gpu(...)` (`backend=
  gpu_krylov_schur`); otherwise it errors out with a build-flag hint,
  mirroring the existing TPQ / FTLM patterns. The previously-bald
  `fail("--mode krylov_schur --gpu is not yet wired ...")` is gone.
  Eigenvector output (`--compute-eigenvectors` /
  `--eigenvector-dir`) on the GPU path is rejected with an actionable
  message.
- **`docs/guides/workflow.md`** — flips the KRYLOV_SCHUR × `mpi+gpu`
  cell from ❌ to ✅⁶, rewrites footnote ² to drop the "still ❌ for
  mpi+gpu" wording, and adds new footnote ⁶ describing the on-device
  basis layout, coalesced re-orth, and the eigenvector limitation.

CPU tests: all 24 MPI ctests (`distributed_*`, `orbit_*` at np ∈
{1, 2, 4}) still pass after the dispatcher edit.

### Added — Phase A (device matrix MPI+GPU): on-device FTLM

Implements `distributed_ftlm_gpu`, the multi-GPU companion of the CPU
`distributed_ftlm`. Closes the previously-❌ FTLM × `mpi+gpu` cell of
the solver × device support matrix.

- **`include/ed/distributed/distributed_ftlm_gpu.h`** (new) — public
  API: `DistributedFtlmGPUOptions { n_samples, n_groups,
  lanczos_max_iter, betas, seed_offset, observable_op, device_index,
  verbose }` and free function `distributed_ftlm_gpu(op, options,
  world_comm) -> DistributedFtlmResult`. Reuses the existing CPU
  result struct from `distributed_ftlm.h` so callers see one type.
  Header is only compiled under `#ifdef WITH_MPI`; consumers guard
  the include with `#ifdef ED_HAVE_NCCL`.
- **`src/distributed/distributed_ftlm_gpu.cu`** (new, ~510 lines,
  wrapped in `#ifdef ED_HAVE_NCCL`) — per-sample Lanczos with **full
  modified Gram–Schmidt re-orthogonalisation** runs entirely on
  device: the Krylov basis V[0..m-1] is held contiguously in a single
  device slab (`m_max × local_n × 16 B` per rank, ~1 GiB at m=64,
  local_n=1e6 — fits a single V100/A100/H100), `cublasZdotc` for the
  inner products, **one** NCCL allreduce per scalar via
  `multi_gpu::all_reduce_sum_complex_double` after coalescing the
  `j+1` re-orth dot products into a single `2·(j+1)` double payload,
  `cublasZaxpy` for the recurrence, `DistributedGPUOperator` (NCCL
  pairwise SendRecv halo + on-device SoA SpMV) for the SpMV, and a
  redundant host-side `(m × m)` Eigen tridiagonal eigensolve. When an
  observable is supplied the J&P contraction `q_j = ⟨V[j] | O V[0]⟩`
  reuses the same on-device basis with one extra device SpMV per
  sample plus a single NCCL allreduce of `2·m` doubles. MPI-over-
  samples mirrors the CPU path; group-rank-0 contributes to the world
  reduction to avoid double-counting replicated per-group
  accumulators. Throws `std::logic_error` if NCCL was not compiled
  in.
- **`tests/unit/test_distributed_ftlm_gpu.cpp`** (new) — four
  `TEST_CASE`s cross-check the GPU FTLM against the CPU
  `distributed_ftlm` on the same MPI_COMM_WORLD with the same
  operator / seeds / betas: N=4 OBC `Z(β)`, N=6 PBC `Z(β)`, N=4 OBC
  `⟨O⟩(β)` with `O = H`, and a replication check across world ranks.
  Tolerance `1e-6` (relative). Build-only on CI's CUDA-build-only
  lane; runtime-tested on dev hosts with GPUs (uses the
  `runtime_supports_gpu_*` SKIP gate from the existing GPU Lanczos
  test). Custom `int main` does `MPI_Init` + `Catch::Session().run` +
  `MPI_Finalize`.
- **`cmake/EDLibraries.cmake`** — adds `distributed_ftlm_gpu.cu` to
  the `ed_distributed_gpu` STATIC target (compiled iff `WITH_MPI &&
  WITH_CUDA && NCCL_FOUND`, propagating `ED_HAVE_NCCL=1` PUBLIC).
- **`CMakeLists.txt`** — registers the new test via
  `ed_add_phase3c_test(test_distributed_ftlm_gpu ...)` at np ∈
  {1, 2, 4}, only inside `if(TARGET ed_distributed_gpu)`.
- **`src/cli/ed_distributed_main.cpp`** — `--mode ftlm` now accepts
  `--gpu`: when `ED_HAVE_NCCL` is defined the dispatcher invokes
  `distributed_ftlm_gpu(...)` (`backend=gpu_mpi`); otherwise it errors
  out with a build-flag hint, mirroring the existing TPQ pattern.
  `--use-symmetry` continues to fail with the same message as on the
  CPU path until Phase D wires symmetry through every distributed
  cell.
- **`docs/guides/workflow.md`** — flips the FTLM × `mpi+gpu` cell of
  the solver × device matrix from ❌ to ✅⁵, adds footnote ⁵ describing
  the on-device Lanczos / re-orth / J&P contraction layout, and adds
  a new "Path × device — cross-product caveats" subsection that calls
  out the remaining (non-orthogonal) carve-outs in the
  path-axis × device-axis cross-product (Phases B–E roadmap visible
  to readers).

CPU MPI ctest (`distributed_ftlm`, `distributed_tpq`,
`distributed_lanczos`, `distributed_eigenvectors` at np ∈ {1, 2, 4})
all 9 tests still pass after the dispatcher edit.

### Fixed — DSSF: positions.dat parser and num_sites auto-detection

Two bugs caused `./ED dssf` (and therefore `qed.dssf.compute(...)` /
`ed::auto_pilot::dssf::compute(...)`) to crash on every deck produced
by `HamiltonianBuilder::write_directory(...)`:

- **`include/ed/core/operator_types.h`** — `readPositionsFromFile`
  expected the legacy 6-column format `site_id  matrix_idx  sublattice
  x  y  z`, but `write_positions_file` writes only `x y z`. Every line
  failed to parse, leaving all `positions[i]` empty, and
  `calculatePhaseFactors` crashed with an out-of-bounds access during
  observable assembly. The parser now accepts the canonical 3-column
  format with the line index used as the site index.
- **`src/core/ed_config.cpp`** — `EDConfig::autoDetectNumSites` tried to
  read the x-coordinate as a `uint64_t` site id. For an `N`-site chain
  with positions `0.0, 1.0, ..., (N-1).0` written in scientific
  notation (`1.0000000000e+01` for site 10), `>>` only consumed the
  leading `1` so a 12-site deck was mis-detected as 10 sites. Now
  counts non-comment lines instead.

After the fix both static and dynamical workflows succeed at N=12 from
the Python side and a new C++ end-to-end ctest (`test_auto_dssf`,
case _"runs static\_thermal end-to-end on a 4-site chain"_) drives the
full `ed::auto_pilot::dssf::compute` path on a real deck. `ctest`
183/183 passing.

`docs/guides/workflow.md` Section 3 was rewritten to be a verified
smoke-tested DSSF/SSSF example (builds the deck on the fly, runs both
`static_thermal` and `dynamical_thermal`); the older "API shape only"
caveat is gone, and the `extra_args=` flags now use the canonical
`--dyn-*` / `--static-*` `=`-syntax accepted by `ED dssf`.

### Removed — Phase 9: collapse the symmetry surface to a single canonical kernel

Closes the audit item _"there are multiple ways of symmetry here
(streaming, disk, ...) — is this really necessary? Just keep the one
that trumps overall."_  Phase 7.1 already audited the eight pre-existing
symmetry entry points and named the streaming kernel canonical, marking
the explicit-block path `[[deprecated]]`.  Phase 9 follows through:
the deprecated path is gone.

**Removed (BREAKING)** — the explicit-block ``*_symmetrized`` entry
points:

- C++: `exact_diagonalization_from_directory_symmetrized(...)` and
  `exact_diagonalization_fixed_sz_symmetrized(...)` (in
  `<ed/core/ed_wrapper.h>`).
- Python: `quantum_ed.exact_diagonalization_from_directory_symmetrized`
  and `quantum_ed.exact_diagonalization_fixed_sz_symmetrized` (the
  pybind11 bindings in `dispatcher_bindings.cpp`).
- Internal: the now-unreachable `ed_internal::` cluster
  (`diagonalize_symmetry_block`, `setup_symmetry_basis`,
  `setup_fixed_sz_symmetry_basis`, `find_ground_state_sector*`,
  `diagonalize_fixed_sz_sector`, `transform_sector_to_*`,
  `transform_fixed_sz_to_full`, `transform_and_save_*`, plus the
  `GroundStateSectorInfo`/`SectorInfo`/`SectorResult` PODs).

**Migration**

| pre-Phase-9                                                       | Phase 9 canonical                                                                                          |
|-------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------|
| `qed.exact_diagonalization_from_directory_symmetrized(d, m, p)`   | `qed.diag(H, symmetry=...)` _(preferred)_ or `qed.exact_diagonalization_from_directory(d, m, p)` with `p.use_symmetry = True` |
| `qed.exact_diagonalization_fixed_sz_symmetrized(d, n_up, m, p)`   | `qed.diag(H, sz=n_up, symmetry=...)` _(preferred)_ or `from_directory(d, m, p)` with `p.use_symmetry = True; p.use_fixed_sz = True; p.n_up = n_up` |
| `exact_diagonalization_from_directory_symmetrized(...)` (C++)     | `ed_dispatch::exact_diagonalization_from_directory(d, m, p, ...)` with `p.use_symmetry = true`             |
| `exact_diagonalization_fixed_sz_symmetrized(d, n_up, m, p, ...)`  | same dispatcher with `p.use_symmetry = true; p.use_fixed_sz = true; p.n_up = n_up`                         |

The streaming kernel is faster on every problem size we have benchmarked,
materialises no per-sector blocks on disk, and supports GPU per-sector
dispatch (the deprecated path did neither).  The expert escape hatches
for very-large-N memory-budget edge cases — `--chunked-symm` and
`--disk-streaming` in `./ED` — are unchanged and remain as documented
CLI-only entry points.  See
[`docs/history/PHASE_7_1_SYMMETRY_AXIS.md`](docs/history/PHASE_7_1_SYMMETRY_AXIS.md)
for the original audit and the per-kernel verdicts.

### Changed — Phase 8: GPU and MPI solver optimisations

Closes the audit item _"optimization for GPU and MPI solvers."_  Carries
over the Phase 6.1 thread-budgeting / output-gating / vector-swap
playbook to the distributed and CUDA solvers, plus a small handful of
targeted hot-path fixes.  Pure performance — bit-identical eigenvalues,
backwards-compatible CLI / Python.  Full design in
[`docs/history/PHASE_8_GPU_MPI_OPT.md`](docs/history/PHASE_8_GPU_MPI_OPT.md).

**Tier A — high-impact, low-invasiveness**

- **`quantum_ed.mpi.run_distributed`** argv now matches
  `ed_distributed_main` (`--mode`); `directory=` / `extra_args=` accepted
  with `DeprecationWarning` and dropped.
- **DistributedOperator + DistributedSymmetryOperator** own persistent
  `send_buf_` / `recv_buf_` / `halo_buf_` (sized once in
  `build_comm_pattern_` / constructor).  No per-matvec heap allocation in
  the MPI hot path.
- **`ThreadBudgetScope`** wired through every distributed and GPU host
  hot path: `distributed_lanczos`, `distributed_lanczos_kernel`,
  `distributed_tpq`, `distributed_ftlm`,
  `GPULanczos::solveTridiagonal`,
  `GPUBlockLanczos::solveBlockTridiagonal`,
  `lobpcg_solve_generalized_eigenproblem`, `GPUFTLMSolver::run`.  Sized
  against the rank-local slab on MPI paths.  `ED_AUTO_THREADS=0` still
  disables.
- **`GPULanczos` cuBLAS scalar pipeline**: `cublasZdotc` runs in
  `DEVICE_POINTER_MODE` and stores alpha in a 1-element device buffer;
  a 1-thread `negateZScalarRealKernel` produces `-Re(alpha)` on-device;
  the follow-up `cublasZaxpy` issues without a host sync.  Alpha is
  async-copied to a pinned host slot and consumed only after
  `vectorNorm(d_w_)`.  Eliminates one implicit device→host sync per
  Lanczos iteration.
- **ScaLAPACK auto-block-size**: new `scalapack_block_size_auto` flag
  (default `true`) replaces the legacy fixed `mb=nb=64` with
  `get_optimal_block_size(N, nprow, npcol)` at solve time.  Backwards
  compatible: `--scalapack-block-size=N` (or
  `--no-scalapack-block-size-auto` / `EDParameters::scalapack_block_size_auto = False`)
  reverts to the explicit value.

**Tier B — algorithmic / hot-path**

- **`distributed_lanczos` batched CGS2 reorth** (also in
  `distributed_lanczos_kernel`): replaces `m` serial `MPI_Allreduce`
  calls per Lanczos iteration with **2 batched** Allreduces (one per
  CGS pass).  For full-reorth Lanczos at scale this is the dominant
  network cost; reorth Allreduces drop from `~m^2/2` to `2m` over a run.
  Falls back to MGS for `basis.size() < 8` where the per-coeff
  Allreduce is cheaper than the batched buffer's setup cost.
- **`GPULanczos` persistent ring-buffer pointer table**: new
  `d_ortho_basis_ptrs_full_` device-side mirror of the entire
  `d_lanczos_vectors_` pointer table, filled once at `allocateMemory`
  time.  `orthogonalize()` passes it directly to the batched DGKS
  kernels for the common (non-wrapped) case — zero H2D traffic in the
  reorth hot path.  Wrapped (windowed reorth) case keeps the legacy
  per-iter copy of the windowed slice.

**New CLI flags**

```
--scalapack-nprow=N
--scalapack-npcol=N
--scalapack-block-size=N           # implicitly disables auto
--scalapack-block-size-auto        # explicit on (default)
--no-scalapack-block-size-auto
```

**Test coverage** — all 150 C++ Catch2 tests + 197 Python pytest tests
keep passing, plus new tests for `scalapack_block_size_auto` defaults,
adapter round-trips, and CLI parsing.

### Changed — Phase 7.1: 5th orthogonal axis — symmetry projection

Closes the audit item _"there are way too many ways to symmetrize; just
decide on the best way and stick to this as the only symm flag."_
Symmetry projection joins fixed-Sz / GPU / MPI as a single boolean flag
on `EDParameters`, and the streaming kernel becomes the only canonical
entry point. The full design is in
[`docs/history/PHASE_7_1_SYMMETRY_AXIS.md`](docs/history/PHASE_7_1_SYMMETRY_AXIS.md).

The Phase 7 axis matrix grows from 4 to 5 orthogonal flags:

```
  solver       (DiagonalizationMethod)
  use_fixed_sz (bool, EDParameters::use_fixed_sz)
  use_gpu      (bool, EDParameters::use_gpu)
  use_mpi      (bool, EDParameters::use_mpi)
  use_symmetry (bool, EDParameters::use_symmetry)   ← NEW
```

- **New flag on `EDParameters`** (`include/ed/core/ed_parameters.h`):
  `use_symmetry`. Mirrored on `SystemConfig`
  (`include/ed/core/ed_config.h::SystemConfig::use_symmetry`) and
  threaded through the `EDConfig ↔ EDParameters` adapter
  (`include/ed/core/ed_config_adapter.h`). The
  `EDConfig::useSymmetry(bool)` builder method sets BOTH the canonical
  `system.use_symmetry` AND the legacy `workflow.run_symm_auto` so the
  existing `ed_main.cpp` dispatch (which fires
  `run_streaming_symmetry_workflow` when `run_symm_auto` is set) keeps
  working unchanged.
- **Canonical 5-axis dispatcher** in
  `include/ed/core/ed_dispatch_symmetry.h`:
  `ed_dispatch::exact_diagonalization_from_directory(...)` and
  `ed_dispatch::exact_diagonalization_from_files(...)` route to the
  streaming symmetry kernel when `params.use_symmetry == true` and
  fall back to the standard dispatcher otherwise. This header lives
  outside `ed_wrapper.h` because the streaming kernel
  (`ed_wrapper_streaming.h`) depends on `ed_wrapper.h` — the dispatcher
  layer is the only place that can include both.
- **Python `quantum_ed.exact_diagonalization_from_directory(...)` is
  now the canonical 5-axis entry point**
  (`python/quantum_ed/_bindings/dispatcher_bindings.cpp`). It forwards
  through `ed_dispatch::exact_diagonalization_from_directory(...)`,
  honouring `params.use_symmetry` (with `use_fixed_sz`, `use_gpu`,
  `use_mpi` orthogonally as before). `EDParameters::use_symmetry` is
  exposed as a settable Python property next to the existing flag axes.
- **Streaming kernel chosen as canonical**: the only symmetry path that
  (a) supports `use_gpu` (per-sector GPU kernel in
  `gpu_symmetrized_operator.cu`), (b) supports `use_fixed_sz`
  orthogonally, (c) avoids materialising the orbit basis on disk,
  (d) was already the route for the unified `--symm` CLI flag. See
  `include/ed/core/ed_wrapper_streaming.h`.
- **Deprecation of explicit-block symmetrized entry points**
  (`include/ed/core/ed_wrapper.h`):
  `exact_diagonalization_from_directory_symmetrized(...)` and
  `exact_diagonalization_fixed_sz_symmetrized(...)` are now annotated
  `[[deprecated(...)]]` with a redirect to the canonical streaming
  kernel. They remain ABI-stable for back-compat (existing user
  scripts keep working) but emit a compiler deprecation warning at
  the call site. The Python bindings keep the legacy entry points
  reachable inside a targeted `#pragma GCC diagnostic` block.
  *(Phase 9 follow-up: both entry points were removed entirely. See the
  Phase 9 "collapse the symmetry surface" entry above for the migration
  table.)*
- **Chunked / disk-streaming kernels stay as CLI-only escape hatches**
  for very-large-N memory-budget edge cases (`--chunked-symm`,
  `--disk-streaming`). They are *not* reachable via
  `EDParameters::use_symmetry` and are not part of the orthogonal-axis
  contract — they are separate kernels with their own trade-offs.
- **CLI**: `--symm` (and the deprecated aliases `--symmetrized` /
  `--streaming-symmetry`) sets BOTH `system.use_symmetry = true` and
  `workflow.run_symm_auto = true`. New `--no-symm` flag explicitly
  disables symmetry projection (useful when overriding a config file
  default on a per-run basis). See `src/core/ed_config.cpp`.
- **Hard error in `exact_diagonalization_from_files()`** when called
  directly with `params.use_symmetry == true`: the function lives in
  `ed_wrapper.h` and cannot itself include the streaming kernel. The
  error message points to `ed_dispatch_symmetry.h` so stale C++ call
  sites fail loudly rather than silently dropping the projection.
- **Tests**:
  [`tests/unit/test_method_canonicalize.cpp`](tests/unit/test_method_canonicalize.cpp)
  gains 4 cases for `EDParameters::use_symmetry` defaults,
  `EDConfig::useSymmetry()` mirroring, adapter round-trip, and
  orthogonality with `canonicalize_method_and_flags()`.
  [`python/tests/test_canonicalize_method.py`](python/tests/test_canonicalize_method.py)
  gains 3 cases mirroring the same contract through the Python
  binding plus deprecated-binding back-compat checks.
  Aggregate: **148/148 ctest passing** (was 144),
  **196/196 pytest passing** (was 193, excluding the pre-existing
  `test_build_introspection_consistency` ScaLAPACK-without-MPI build
  failure unrelated to Phase 7.1).

### Changed — Phase 7: solver organization on orthogonal axes (SOLVER × FIXED_SZ × GPU × MPI)

The old enum tangled algorithm choice ("Lanczos vs FTLM vs ScaLAPACK …")
with device choice ("CPU vs GPU"), parallelism choice ("single-process
vs MPI"), and basis choice ("full vs fixed-Sz") into a single
`DiagonalizationMethod` enum. Phase 7 separates them: the enum names
*algorithms only*, and the device / parallelism / basis axes live on
`EDParameters` as orthogonal flags. The legacy `_GPU` / `_CUDA` /
`_MPI` / `_FIXED_SZ` enum variants are kept as deprecated aliases for
backwards compatibility (HDF5 metadata, CLI strings, pre-Phase-7 user
code) and are collapsed onto the canonical tuple at every dispatcher
entry point.

- **New flags on `EDParameters`** (`include/ed/core/ed_parameters.h`):
  `use_gpu`, `use_mpi` (next to the existing `use_fixed_sz`). Mirrored
  on `SystemConfig` (`include/ed/core/ed_config.h`) and threaded through
  the `EDConfig ↔ EDParameters` adapter
  (`include/ed/core/ed_config_adapter.h`) so config files, the CLI, and
  the C++ / Python APIs all share a single set of flags.
- **`ed::canonicalize_method_and_flags(method, fz, gpu, mpi)` helper**
  (`include/ed/core/ed_method_traits.h`). The single source of truth for
  collapsing every legacy `_GPU` / `_CUDA` / `_MPI` / `_FIXED_SZ` enum
  value onto its canonical `(base_method, use_fixed_sz, use_gpu, use_mpi)`
  tuple. The transformation is OR-merge with caller-supplied flags
  (so no information is lost), idempotent, and constexpr.
  `ed::legacy_method_for_dispatch(base, use_gpu)` is the inverse half
  used internally to keep the existing `_GPU` switch-statement
  structure.
- **Centralised canonicalization in dispatcher entry points**
  (`include/ed/core/ed_wrapper.h` —  `exact_diagonalization_core`,
  `exact_diagonalization_from_files`, `exact_diagonalization_fixed_sz`).
  Every public entry now calls `ed::canonicalize_method_and_flags()`
  on the way in, so callers passing `LANCZOS_GPU` and callers passing
  `LANCZOS` + `use_gpu=true` end up dispatching exactly the same code.
- **Legacy enum values marked `[[deprecated]]`** in
  `include/ed/core/ed_types.h`. The deprecation message points at the
  canonical replacement (e.g. _"Use LANCZOS with EDParameters::use_gpu=true
  instead"_). Internal code that legitimately handles the deprecated
  values (the dispatcher, the CLI string parser, the Python enum
  binding, the canonicalizer itself) silences the warning with a
  targeted `#pragma GCC diagnostic` block instead of leaking it to
  users.
- **`mTPQ_CUDA` is no longer a separate solver.** It was previously a
  no-op alias for `mTPQ_GPU` (the dispatcher had a dead
  `case mTPQ_CUDA: break;`). It now canonicalizes to `mTPQ` +
  `use_gpu=true`, identical to `mTPQ_GPU`.
- **`SCALAPACK` and `SCALAPACK_MIXED` kept as separate solvers.** They
  go through `PDSYEVR` / mixed-precision refinement, which is a
  *different* dense LAPACK call than `FULL` — not "FULL with
  use_mpi=true". Canonicalization marks them as `use_mpi=true` for
  honest introspection but does not collapse them.
- **CLI gained `--gpu` / `--mpi` flags** in `src/core/ed_config.cpp`,
  next to the existing `--fixed-sz` / `--n-up`. The `--method`
  argument keeps accepting all legacy strings.
- **Python bindings updated** (`python/quantum_ed/_bindings/dispatcher_bindings.cpp`):
  `EDParameters::use_gpu` / `use_mpi` exposed as read/write properties;
  the deprecated combined `LANCZOS_GPU_FIXED_SZ` /
  `BLOCK_LANCZOS_GPU_FIXED_SZ` / `FTLM_GPU_FIXED_SZ` enum values
  exposed for HDF5 metadata round-tripping; the canonicalizer
  re-exported as `quantum_ed.canonicalize_method(...)` so Python
  tooling and tests can verify the orthogonal decomposition without
  going through the dispatcher.
- **Lockdown tests**:
  [`tests/unit/test_method_canonicalize.cpp`](tests/unit/test_method_canonicalize.cpp)
  (10 cases, 163 Catch2 assertions: identity for canonical inputs;
  `_GPU` / `_FIXED_SZ` / `_MPI` / `_CUDA` collapse; SCALAPACK kept
  distinct; OR-merge of caller flags; idempotence;
  `legacy_method_for_dispatch()` round-trip) and
  [`python/tests/test_canonicalize_method.py`](python/tests/test_canonicalize_method.py)
  (61 parametrised pytest cases mirroring the C++ contract through the
  Python binding plus `EDParameters::use_gpu` / `use_mpi`
  round-trip).

Migration: replace `LANCZOS_GPU` → `LANCZOS` + `params.use_gpu = True`,
`LANCZOS_GPU_FIXED_SZ` → `LANCZOS` + `params.use_gpu = True` +
`params.use_fixed_sz = True`, `mTPQ_CUDA` and `mTPQ_GPU` →  `mTPQ` +
`params.use_gpu = True`, `mTPQ_MPI` → `mTPQ` + `params.use_mpi = True`.
Capability matrix and full migration table in
[`docs/history/PHASE_7_SOLVER_AXES.md`](docs/history/PHASE_7_SOLVER_AXES.md).

Test totals: **144** C++ test cases (was 134; +10 from Phase 7),
**254** pytest cases (was 193; +61 from Phase 7). Zero existing tests
broken.

### Added — Phase 6.1: Phase 6 perf hygiene applied across the whole CPU solver matrix

The Phase 6 / xdiag bake-off only patched the standalone `lanczos()` driver.
Phase 6.1 broadens the same hygiene to **every CPU solver entry point** —
TPQ, FTLM, LTLM, Hybrid Thermal, full diagonalization, block Lanczos,
Krylov-Schur, block Krylov-Schur, Chebyshev-filtered Lanczos, shift-invert
Lanczos, implicitly-restarted / thick-restart Lanczos, ARPACK, optimal
spectrum solver — plus the Python wrappers in front of them.

- **Centralised `HDF5IO::isDisabledOutputPath`** helper in
  `include/ed/core/hdf5_io.h`. Every HDF5 entry point
  (`createOrOpenFile`, `forceCreateFile`, `saveEigenvalues`,
  `saveEigenvector`, `saveDiagonalizationResults`, `ensureTPQSampleGroup`,
  `saveTPQState`, `saveDynamicalResponse`, `saveDynamicalResponseFull`,
  `saveTimeCorrelation`, `getPerRankFilePath`, `createPerRankFile`,
  `saveThermodynamics`, `saveCorrelationMatrix`, `saveCorrelationData`,
  `saveFTLMSample`, `appendTPQThermodynamics`, `appendTPQNorm`,
  `saveTPQThermodynamics`, `saveTPQNorm`, `saveTPQAveragedThermodynamics`,
  `saveFTLMThermodynamics`, `saveStaticResponse`, `saveHybridThermalResults`,
  `saveArray`, `ensureFTLMSampleGroups`, `saveFTLMThermodynamicSample`,
  `saveFTLMDynamicalSample`, `saveFTLMStaticSample`,
  `ensureTimeCorrelationGroups`) short-circuits when the path is empty,
  `"/dev/null"`, or any `"/dev/null/..."` derivative. Even
  `fileExists()` now returns `false` on the disabled sentinel so we
  never try to `H5Fopen("/dev/null", ...)` (which is a real device
  node on Linux and silently passes `std::filesystem::exists`). The
  C++ solver code remains unchanged — it still calls the same HDF5
  helpers, but those helpers now no-op on disabled paths instead of
  crashing or writing junk under `/dev/null/`.
- **`exact_diagonalization_core` dispatcher** (`include/ed/core/ed_wrapper.h`)
  now remaps `params.output_dir == ""` to `"/dev/null"` *before*
  fanning out to any backend, mirroring the standalone Python
  wrappers' `output_dir_or_devnull(...)` convention. Without this,
  the Python `quantum_ed.exact_diagonalization_core(op, method,
  default_params)` path silently dumped `ed_results.h5` /
  `eigenvalues.dat` / `eigenvalues.txt` into whatever cwd the
  process happened to be in — pollutes notebook environments and
  makes benchmark loops look slower than they are. Pass
  `params.output_dir = "."` explicitly to opt back into the legacy
  cwd-dump behaviour. The dispatcher also now skips the
  `mkdir -p $output_dir` shell call and the
  `results.eigenvectors_path` assignment on disabled paths.
- **Raw `std::ofstream` writes** in `thick_restart_lanczos()` and
  `shift_invert_lanczos()` (the `eigenvalues.dat` / `eigenvalues.txt`
  side files that bypass the `HDF5IO` layer) are now gated by
  `HDF5IO::isDisabledOutputPath(evec_dir)`. Same default-cwd-pollution
  story as above — the dispatcher fix covers the HDF5 path, this
  covers the legacy raw-binary side files.
- **DSSF unified HDF5 schema** (`src/dssf/dssf_io.cpp`,
  `ensure_metadata` / `write_record`) now respects
  `HDF5IO::isDisabledOutputPath()` so the DSSF writer is consistent
  with the rest of the HDF5 I/O. DSSF has its own `H5::H5File`
  handling (it does not go through the central `HDF5IO::saveDynamicalResponse`
  helpers), so it needed an explicit gate.
- **Python bindings** for `full_diagonalization`, `lanczos`,
  `finite_temperature_lanczos`, `low_temperature_lanczos`, and
  `hybrid_thermal_method` (both the regular and `FixedSzOperator`
  overloads) now remap a default `output_dir=""` to `"/dev/null"` via
  the new `output_dir_or_devnull(...)` helper, mirroring the existing
  Lanczos behaviour. Callers who actually want disk output pass
  `"."` or any explicit directory.
- **`ed::parallel::ThreadBudgetScope`** now wraps the entry points of
  every CPU solver listed above (previously only the bare `lanczos`,
  `lanczos_no_ortho`, `lanczos_selective_reorth`, and `lanczos_real`
  drivers were budgeted). Each call now caps OpenMP+OpenBLAS threads to
  `auto_threads_for_dim(N)` for its lifetime. RAII semantics mean
  nested scopes (e.g. FTLM driving Lanczos chains) compose correctly.
- **`std::swap` rotate** for `v_prev / v_current / w` propagated to
  `chebyshev_filtered_lanczos` and `shift_invert_lanczos` inner loops
  (the same optimisation already in `lanczos`, `lanczos_no_ortho`,
  `lanczos_real`). Drops the per-iteration `O(N)` `std::copy` traffic.
- ScaLAPACK / GPU / MPI entry points: GPU code does not share the
  OpenMP+OpenBLAS pool (own kernels), and the MPI distributed solvers
  re-enter the same CPU code paths they wrap, so they automatically
  inherit the wins. ScaLAPACK is dense BLAS-3 dominated and benefits
  marginally from the HDF5 gating only.

Numerical verification: full `ctest` (134/134) + Python `pytest` (132/132)
green; the xdiag-style smoke run on N = 12-18 still matches the
ground-state energy to ~1e-12 against the previous build.

### Fixed — `quantum_ed.lanczos` wall time vs `XDiag` at large `N` (Python default)

- **Default `output_dir` for `quantum_ed.lanczos` (and the
  `FixedSzOperator` overload) now maps to the existing C++ `"/dev/null"`
  convention** when the user passes the default empty string, so
  **HDF5 is not opened on every call** (previously the implicit `"."`
  path triggered `HDF5IO::createOrOpenFile` + `saveEigenvalues` to
  `./ed_results.h5` after every solve, which dominated wall time at
  `N >= 16` in benchmarks).
- **Complex `lanczos()` (eigenvalue-only)**: the local-reorth ring buffer
  cap is **3 vectors** when `eigenvectors=false` (the DGKS pass only
  needs three directions); the old cap of 20 added unnecessary dim-N
  copies before the ring started rotating.
- **`lanczos_io::append_basis_vector`**: added a **move** overload for
  future call sites that hand off ownership of a `ComplexVector`.
- **Docs**: updated [`docs/benchmarks/bench_vs_xdiag.md`](docs/benchmarks/bench_vs_xdiag.md)
  tables and **re-ran** `benchmarks/bench_vs_xdiag.py` so the checked-in
  JSON matches the new defaults.

### Added — head-to-head benchmark vs `XDiag.jl`

- New `benchmarks/bench_vs_xdiag.py` (Python orchestrator) +
  `benchmarks/bench_vs_xdiag.jl` (Julia subprocess) that runs the same
  1D Heisenberg PBC chain workload through `quantum_ed` and through
  [`XDiag.jl`](https://github.com/awietek/XDiag.jl), reporting per-call
  SpMV (`H @ v`) and full ground-state Lanczos timings side-by-side, in
  both the unsymmetrised (`Spinhalf(N)` / `Operator`) and Sz=0
  (`Spinhalf(N, N/2)` / `FixedSzOperator`) configurations.
- New Julia env at `benchmarks/xdiag_env/` (Project.toml + Manifest.toml
  pinning XDiag.jl + JSON.jl).
- New write-up at [`docs/benchmarks/bench_vs_xdiag.md`](docs/benchmarks/bench_vs_xdiag.md)
  with the full reference tables, methodology, and an honest
  open-question section about the local-reorth Lanczos cliff at
  `N >= 16` (matrix-free SpMV is unaffected and stays 3-20x ahead at
  every size). Snapshot JSON checked in next to it
  (`bench_vs_xdiag.json`, `bench_vs_xdiag_fixed_sz.json`, plus the raw
  XDiag-side JSON for reproducibility).
- Cross-links from `README.md`, `docs/benchmarks/README.md`, the main
  `BENCHMARKS.md`, and `benchmarks/README.md` so anyone landing on the
  perf docs sees the XDiag comparison alongside the QuSpin / SciPy one.

Both libraries match `E0` on the test sweep to ~1e-12, confirming the
operator convention parity.

### Added — Phase 5: full Python parity with the C++/CLI advanced backends

The `quantum_ed` Python package previously exposed only a curated CPU
subset (full diagonalization, Lanczos, FTLM/LTLM/hybrid, observable
construction, BFG post-processing). Phase 5 closes the remaining gaps
identified in the Phase 4 capability matrix so that `import quantum_ed`
now reaches **every backend the `./ED` CLI knows about**.

- **High-level dispatcher bound from Python** (new
  `python/quantum_ed/_bindings/dispatcher_bindings.{h,cpp}`):
  - `quantum_ed.exact_diagonalization_core(op, method, params)` is the
    single entry point that fans out to ~30 CPU solver variants:
    `LANCZOS` / `LANCZOS_SELECTIVE` / `LANCZOS_NO_ORTHO`,
    `BLOCK_LANCZOS`, `KRYLOV_SCHUR`, `BLOCK_KRYLOV_SCHUR`, `DAVIDSON`,
    `LOBPCG`, `THICK_RESTART_LANCZOS`, `IMPLICIT_RESTART_LANCZOS`,
    `CHEBYSHEV_FILTERED`, `SHIFT_INVERT[_ROBUST]`, `BICG`,
    every `ARPACK_*` variant, the dense `FULL`/`OSS`/`SCALAPACK[_MIXED]`
    backends, and the thermal solvers (`FTLM`, `LTLM`, `HYBRID`,
    `mTPQ`, `cTPQ`). Overloaded for both `Operator` and
    `FixedSzOperator`.
  - `quantum_ed.DiagonalizationMethod` enum (every value the C++ enum
    carries), `quantum_ed.HamiltonianFileFormat` enum, and an
    `EDParameters` parameter bag with **every** C++ knob exposed as a
    read/write Python attribute (block size, ARPACK NCV, FTLM Krylov
    dim, TPQ Taylor order, ScaLAPACK process grid, …).
  - `quantum_ed.EDResults` envelope (eigenvalues, eigenvectors,
    `thermo_data` with temperatures / energy / specific heat /
    susceptibility / entropy, FTLM-specific error estimates).
- **Directory + streaming symmetry dispatchers bound from Python**:
  - `quantum_ed.exact_diagonalization_from_directory[_symmetrized]` and
    `quantum_ed.exact_diagonalization_fixed_sz_symmetrized` for
    file-based runs. These are the canonical path for any *_GPU* method
    (`LANCZOS_GPU`, `FULL_GPU`, `mTPQ_GPU`, `cTPQ_GPU`, …) when
    `WITH_CUDA=ON`.
  - `quantum_ed.exact_diagonalization_streaming_symmetry[_fixed_sz]`
    for the in-memory symmetry-projected path (with optional GPU
    per-sector dispatch). Replaces the JSON-only workaround documented
    in the previous Phase 4 changelog entry.
- **In-process symmetry projection from Python** (no JSON detour):
  - `Operator.set_symmetry_info_from_dict(info)` /
    `Operator.get_symmetry_info_as_dict()` (and the same on
    `FixedSzOperator`) consume the dict produced by
    `quantum_ed.symmetry.group_from_generators(...)` directly. The
    legacy `automorphism_results/` round-trip is no longer required.
- **Build introspection helpers**: `quantum_ed.has_cuda_build()`,
  `quantum_ed.has_mpi_build()`, `quantum_ed.has_scalapack_build()` so a
  single Python script can gate GPU / MPI / ScaLAPACK code paths
  without try/except.
- **MPI launcher helper** (`python/quantum_ed/mpi.py`):
  `quantum_ed.mpi.run_distributed(directory, method, n_ranks, ...)`
  builds the right `mpiexec` / `srun` argv for every distributed solver
  (`lanczos`, `lanczos_symmetry`, `lanczos_gpu`, `ftlm`, `tpq`) and
  shells out -- no manual `subprocess` wiring, no need to embed
  `MPI_Init` inside Python. Exports `quantum_ed.mpi.MPI_METHODS`.
- **DSSF launcher helper** (extends `python/quantum_ed/dssf.py`):
  `quantum_ed.dssf.run_from_directory(directory, method, ...)` invokes
  the full `./ED dssf <method> <dir>` continued-fraction engine
  (LANCZOS / BICG / FULL / FTLM, S(Q,ω), HDF5 trees) the same way.
- **Build system wiring** (`python/quantum_ed/CMakeLists.txt`):
  - Conditionally links `ed_solvers_gpu` (and `ed_distributed_gpu` when
    NCCL is present) into `_core.so` whenever `WITH_CUDA=ON`, so the
    GPU symbols referenced from the dispatcher header resolve at link
    time.
  - Sets `CUDA_RESOLVE_DEVICE_SYMBOLS=ON` and disables LTO on the
    `_core` target (passes `NO_EXTRAS` to `pybind11_add_module` under
    CUDA) to fix the `undefined symbol: fatbinData` import error that
    arose from pybind11's default LTO discarding NVCC's synthetic
    `cmake_device_link.o`.
- **Tests** (`python/tests/test_dispatcher.py`, 22 new cases): build
  introspection, `DiagonalizationMethod` enum, `EDParameters` defaults
  and round-trip, `exact_diagonalization_core` ground-state recovery
  across `LANCZOS{,_SELECTIVE,_NO_ORTHO}`, `BLOCK_LANCZOS`,
  `KRYLOV_SCHUR`, `BLOCK_KRYLOV_SCHUR`, `DAVIDSON`, `LOBPCG`,
  `THICK_RESTART_LANCZOS`, `IMPLICIT_RESTART_LANCZOS`, `ARPACK_SM`,
  `ARPACK_LM` (largest eigenvalue), `FixedSzOperator` overload,
  symmetry info round-trip, and smoke tests for
  `dssf.run_from_directory` and `mpi.run_distributed`. Full Python
  test suite (133 tests) passes on a CUDA-enabled build.
- **Documentation:**
  - `docs/guides/python_api_coverage.md` rewritten: the executive
    summary, capability matrix, and per-submodule sections all reflect
    full functional parity. The "what's not in Python" entry shrinks
    to *MPI* (subprocess-only by design) and *full DSSF spectral
    driver* (subprocess-only by design), both with first-class Python
    helpers.
  - New `docs/guides/python_advanced.md` walks every Phase 5 entry
    point: choosing between `exact_diagonalization_core`,
    `_from_directory[_symmetrized]`, and `_streaming_symmetry[_fixed_sz]`;
    GPU per-sector dispatch; in-process symmetry projection;
    `quantum_ed.mpi.run_distributed` and `quantum_ed.dssf.run_from_directory`;
    build introspection; and an end-to-end worked example.
  - `docs/guides/usage.md` Mode 4 rewritten to advertise the full
    backend surface (the previous "curated CPU subset" caveat is
    gone), the legacy thin wrappers + dispatcher coexist in §5.2, the
    in-process symmetry path is shown in §5.4, and a new §5.7 covers
    build introspection + MPI + the `./ED dssf` driver. §8.5 / §8.6
    section titles updated to "C++ and Python" and cross-link the
    Python entry points.
  - `README.md` Phase 5 callout: Python is now a first-class peer to
    the CLI for every backend.

### Added — Phase 4: standalone `ed_input` C++ library + `quantum_ed.input` Python bindings

The legacy `python/edlib/helper_*.py` family — which had to write
`InterAll.dat` / `Trans.dat` / `positions.dat` to a directory before
`./ED` could read it — has been **superseded** by a typed, fluent C++
library that is exposed identically from Python. Mode 1 (file-based) is
fully preserved; the new Mode 8 covers the same physics through a
single in-process surface that can either materialise an `Operator` or
emit the same legacy directory.

- **`ed_input` static library** (`include/ed/input/`, `src/input/`):
  - `Lattice` struct + `ed::input::lattice::{chain, square, triangular,
    honeycomb, kagome, pyrochlore, from_neighbor_lists,
    from_cluster_file}` generators.
  - `HamiltonianBuilder` fluent term accumulator with shortcuts:
    `heisenberg`, `xxz`, `xyz`, `ising`, `transverse_field_ising`,
    `kitaev`, `dm`, `zeeman`, `zeeman_per_site`, `on_site_field`,
    `ring_exchange`, `pyrochlore_non_kramers`; low level
    `add_one_body / add_two_body / add_three_body`. Finalisers
    `to_operator()` (in-memory `ed::Operator`) and `write_directory()`
    (legacy `.dat` directory consumed by `./ED`).
  - Low-level `ed::input::write_*` writers for every legacy file format
    (`Trans.dat`, `InterAll.dat`, `ThreeBodyG.dat`, `positions.dat`,
    one/two-body correlation files, momentum-projected observables).
  - CMake target `ed_input` (`cmake/EDLibraries.cmake`,
    `CMakeLists.txt`).
  - Catch2 unit tests in `tests/unit/test_input_library.cpp`
    (chain / square / kagome / pyrochlore generators,
    `HamiltonianBuilder.heisenberg` cross-check vs the existing
    `build_heisenberg_chain` reference, `xxz` collapsing to Heisenberg,
    `write_directory` round-trip vs in-memory `Operator`,
    `on_site_field`).
- **`quantum_ed.input` pybind11 mirror** (`python/quantum_ed/input.py`,
  `python/quantum_ed/_bindings/input_bindings.{h,cpp}`):
  - 1:1 surface coverage of the C++ library: `Op` enum, `Bond`,
    `Plaquette`, `Lattice`, every lattice generator under
    `quantum_ed.input.lattice.*`, `FileOptions`, `HamiltonianBuilder`
    (every shortcut + `to_operator` + `write_directory`),
    `quantum_ed.input.io.*` low-level writers.
  - `HamiltonianBuilder.to_operator()` returns a uniquely-owned
    `quantum_ed.Operator` (the binding constructs a `unique_ptr` on the
    Python side and `emit_into`'s the terms; this resolves a
    `shared_ptr` / `unique_ptr` holder conflict that previously
    surfaced as a `double free detected in tcache 2` when feeding the
    operator into `qed.full_diagonalization`).
  - pytest suite `python/tests/test_input.py` (13 tests) covering
    every generator, the textbook Heisenberg / XXZ shortcuts vs the
    pure-Python `quantum_ed.hamiltonian` DSL, `write_directory` round
    trip vs in-memory ground state, `Op` enum, `Bond` repr, and the
    low-level `add_one_body` escape hatch.
- **Documentation:**
  - `docs/guides/usage.md` gains a new **Mode 8** section (full
    capability matrix, C++ + Python end-to-end recipes, when to pick
    Mode 1 vs Mode 8).
  - `docs/guides/python_api_coverage.md` lists `quantum_ed.input` as
    Phase 4 and inventories every bound symbol.
  - `docs/architecture/CODEMAP.md` adds `ed_input` to the build graph,
    documents `include/ed/input/` and `src/input/` as new file leaves,
    and renumbers section 5 accordingly.
  - `README.md` advertises the new builder in the headline bullets,
    quickstart, and documentation map.

### Documentation — advanced-backend C++/Python/CLI capability matrix

Clarifies for users which advanced backends (GPU, MPI, alternative CPU
iterative, TPQ, symmetry projection, fixed-Sz) are callable from where:

- `docs/guides/python_api_coverage.md` adds a new **§0 capability
  matrix** (a single table that lists every solver / backend / feature
  cross-tabulated against C++, Python, and CLI). Sharpens the existing
  "what's not in Python" section by tagging each gap with the C++
  header that already exposes it (`<ed/gpu/gpu_ed_wrapper.h>`,
  `<ed/distributed/*.h>`, `<ed/symmetry/group.h>`,
  `<ed/core/ed_wrapper_streaming.h>`, etc.).
- `docs/guides/usage.md` Mode 7 (raw C++) gains four new subsections
  with end-to-end runnable templates for the C++-only paths:
  - **§8.3 GPU solvers** (`GPUEDWrapper::runGPULanczos`, `runGPUFTLM`,
    `runGPUMicrocanonicalTPQ`, `runGPUCanonicalTPQ`, full diag, plus
    per-Sz variants);
  - **§8.4 MPI distributed solvers** (`distributed_lanczos`,
    `distributed_ftlm` with optional observable, `distributed_tpq`,
    `distributed_lanczos_symmetry`, `distributed_lanczos_gpu` NCCL);
  - **§8.5 In-process symmetry-projected solve** (`Operator::symmetry_info
    = ed::sym::translation_group_with_reflection_1d(N);` then
    `generateSymmetrySectorsHDF5()` / `exact_diagonalization_*_symmetrized`);
  - **§8.6 Streaming symmetry** for clusters that exceed RAM
    (`exact_diagonalization_streaming_symmetry`).
  Mode 4 (Python) explicitly cross-references §8.3–§8.6 and the new
  capability matrix so users know which advanced features still
  require dropping to C++ today.
- The static-libraries table in §8 now includes `ed_input` as a
  link target alongside `ed_solvers_cpu` / `_gpu`, `ed_distributed`,
  `ed_symmetry`, etc., and spells out which symbols each library exports.

### Documentation — corrected `README.md` solver matrix

`README.md` §"Solver matrix" rewritten to reflect the audited backend
coverage and to retire several misleading rows from the previous draft:

- The old **iterative** table collapsed everything into one row with
  `GPU = "partial"`. The new §2 lists every iterative method
  separately, marks each cell ✓ / — / stub, and adds explicit prose
  ("Why several CPU-only tokens have no `_GPU` sibling — and whether
  to ship one") that names the six methods which **do** have a full
  GPU port (`LANCZOS`, `BLOCK_LANCZOS`, `KRYLOV_SCHUR`,
  `BLOCK_KRYLOV_SCHUR`, `DAVIDSON`, `LOBPCG`) and explains why the
  remaining ones (`LANCZOS_SELECTIVE/_NO_ORTHO`, `CHEBYSHEV_FILTERED`,
  `SHIFT_INVERT*`, `IRL`, `TRLAN`, `BICG`, `ARPACK_*`) are
  intentionally CPU-only.
- A new **§3 Finite-temperature methods** row separates `FTLM` (full
  CPU/GPU/MPI coverage) from `LTLM` / `HYBRID` (CPU-only by design).
- The **TPQ family tree** is rewritten with a tree diagram + decision
  tree (§4). Marks `mTPQ_CUDA` as a deprecated alias of `mTPQ_GPU`
  (parser kept for API stability), `mTPQ_MPI` as a parser stub that
  throws `mTPQ_MPI not available` (no microcanonical MPI driver
  exists), and clarifies that `ed::distributed::distributed_tpq` is
  the canonical-TPQ MPI implementation, not a third TPQ variant.
- The **symmetry** row in the old table said "GPU = partial". The
  new §5 documents that GPU dispatch per symmetry sector is
  fully wired through `dispatchGPUSymmetrizedSector` /
  `GPUSymmetrizedOperator` for `LANCZOS_GPU`, `BLOCK_LANCZOS_GPU`,
  `DAVIDSON_GPU`, `KRYLOV_SCHUR_GPU`, `BLOCK_KRYLOV_SCHUR_GPU`, and
  `FULL_GPU`, reachable from `./ED <dir> --symm --method=…`.
- A new **§6 Fixed-Sz** row enumerates the per-Sz GPU wrappers
  (`runGPULanczosFixedSz`, `runGPUBlockLanczosFixedSz`,
  `runGPUFTLMFixedSz`, `runGPUDavidsonFixedSz`, `runGPULOBPCGFixedSz`,
  `runGPUMicrocanonicalTPQFixedSz`, `runGPUCanonicalTPQFixedSz`).

The new tables also cross-link to the matching C++ headers and to the
single-source-of-truth capability matrix in
`docs/guides/python_api_coverage.md` §0, so the README and the
detailed docs cannot drift apart again.

### Added — Phase 3b #7 + Phase 3c HPC lockdown (rorqual)

The 2026-04-25 cluster pass (commit `33de7d6`) closes the symmetry-aware
distributed and multi-GPU items that were previously deferred to HPC time:

- **Phase 3b #7 — symmetry-aware distributed SpMV + Lanczos.**
  - `OrbitPartition`: deterministic LPT-greedy load-balanced row
    partitioning over symmetry orbits
    (`include/ed/distributed/orbit_partition.h`).
  - `OrbitHaloPlan`: orbit-aware `MPI_Alltoallv` halo exchange with one
    `MPI_Alltoall` (counts) + one `MPI_Alltoallv` (orbit ids) at
    construction and one `MPI_Alltoallv` per `exchange()`
    (`include/ed/distributed/orbit_halo_plan.h`).
  - `DistributedSymmetryOperator`: matrix-free SpMV in the symmetry-
    projected basis — orbit slabs + halo + per-`(g, a)` character
    weights matching `ed::sym::group_from_generators`
    (`include/ed/distributed/distributed_symmetry_operator.h`).
  - `distributed_lanczos_kernel<OpT>`: header-only templated kernel
    reused by both the unsymmetrised `DistributedOperator` and the new
    `DistributedSymmetryOperator`
    (`include/ed/distributed/distributed_lanczos_kernel.h`).
  - `distributed_lanczos_symmetry`: scatters into rank-major orbit
    layout then runs the kernel; locked down vs dense
    `Eigen::SelfAdjointEigenSolver` ground-state on every momentum
    sector of N=4 OBC, N=4 PBC, N=6 PBC.
  - Cluster lockdown: `test_distributed_symmetry_operator` (rorqual
    `cpubase_b1`, job 10953752) — 82 / 4 PASS;
    `test_distributed_lanczos_symmetry` (job 10954066) — 45 / 54 PASS.

- **Phase 3c — multi-GPU NCCL runtime.**
  - `MultiGpuCommunicator`: RAII wrapper over `ncclComm_t`, built
    collectively from an `MPI_Comm` (rank-0 generates `ncclUniqueId`,
    every rank `cudaSetDevice`s its node-local slot then
    `ncclCommInitRank`s); collective wrappers on device pointers
    (`include/ed/distributed/multi_gpu.h`, `src/distributed/multi_gpu.cu`).
  - `distributed_lanczos_gpu`: GPU-resident Krylov basis with
    `cublasZdotc` + `ncclAllReduce` for dot/norm reductions on device
    buffers; `gpu_resident_spmv = true` switch routes the SpMV through
    the new fully-on-device `DistributedGPUOperator` (NCCL pairwise
    `ncclSend` / `ncclRecv` halo + CUDA SpMV kernel with device-side
    binary search column lookup).
  - `cmake/EDLibraries.cmake`: optional NCCL detection populates
    `NCCL_FOUND` / `NCCL_INCLUDE_DIRS` / `NCCL_LIBRARIES` and gates the
    new `ed_distributed_gpu` static library; `multi_gpu_stub.h` kept as
    a pure back-compat shim.
  - Cluster lockdown on real H100 hardware (`gpubase_bygpu_b1`):
    `test_multi_gpu_nccl` (jobs 10942714 / 10943562 / 10943975) PASS at
    `np ∈ {1, 2, 4}`; `test_distributed_lanczos_gpu`
    (jobs 10942714 / 10943561 / 10943974 — stage 2; jobs 10950081 /
    10950082 / 10950083 — stage 4) PASS, GPU vs CPU `\|E0\| < 1e-10`;
    `test_distributed_gpu_operator` (jobs 10945403 / 10945402 / 10949083)
    PASS, max element-wise difference vs CPU `MPI_Alltoallv` SpMV
    `< 1e-12`.

- **Reproducible cluster build.** New `build_rorqual.sh` configures
  a release build with `StdEnv/2023`, GCC 12.3, OpenMPI 4.1.5, AOCL
  BLIS + libflame via FlexiBLAS, ScaLAPACK 2.2.0, CUDA, with both
  `WITH_MPI=ON` and `WITH_CUDA=ON`.

`docs/architecture/IMPLEMENTATION_NOTES.md` has been updated to mark
Phase 3b #7 and Phase 3c #1 / #2 (intra-node) as **DONE** with cluster
job IDs; only Phase 3c #3 (parallel-HDF5 distributed disk-backed
Krylov), the inter-node IB-fabric lockdown of Phase 3c #2, and the
HΦ head-to-head 40-site validation remain explicitly deferred.

### Added — Phase 3b lockdown, head-to-head benchmarks, and release polish

The Phase 3 distributed-memory campaign reaches its publication-grade
checkpoint. What landed:

- **Distributed Lanczos with eigenvector reconstruction** (Phase 3b #6).
  `ed::distributed::distributed_lanczos_eigenvectors` returns rank-local
  eigenvector slabs that satisfy `||H psi - E psi||_2 < 1e-8` and are
  bit-identical across ranks (`tests/unit/test_distributed_eigenvectors.cpp`).
- **Distributed FTLM with observables** (Phase 3b #5). The Jaklic-Prelovsek
  estimator is generalized to arbitrary `Operator` observables; energy
  matches the exact thermal trace to machine precision on small systems
  (`tests/unit/test_distributed_ftlm.cpp`).
- **Distributed canonical TPQ** (Phase 3b #8). Imaginary-time evolution
  via Taylor expansion of `e^{-dB H}`, with energy and variance reported
  per target beta and replicated across ranks
  (`tests/unit/test_distributed_tpq.cpp`).
- **Comprehensive head-to-head benchmark suite**. New
  `benchmarks/bench_all_backends.py` orchestrator emits a single JSON
  artefact across CPU SpMV, CPU Lanczos, GPU SpMV, GPU Lanczos,
  distributed Lanczos, QuSpin's `hamiltonian.dot`, SciPy `csr @ v`, and
  `scipy.sparse.linalg.eigsh` peer baselines; results are written up in
  `docs/benchmarks/BENCHMARKS.md`.
- **Runnable examples for every workflow.** New top-level `examples/`
  directory with thirteen self-contained programs (CPU / GPU / MPI / FTLM /
  TPQ / DSSF / NLCE / CLI) that build via `-DED_BUILD_EXAMPLES=ON`.
- **Repository cleanup for release.** Top-level markdown reduced to the
  three canonical files (`README.md`, `CHANGELOG.md`, `CONTRIBUTING.md`).
  All architectural documents moved under `docs/architecture/`, the
  benchmark write-up under `docs/benchmarks/`, and historical phase
  reports archived under `docs/history/`. New
  `docs/architecture/IMPLEMENTATION_NOTES.md` consolidates every deferred
  HPC-time-gated item (symmetry-aware row partitioning, NCCL multi-GPU,
  GPU-Direct RDMA, distributed disk-backed Krylov, 40-site validation).

Test coverage: 146 / 146 green (was 137 / 137 in the previous Phase 3a
checkpoint).

### Fixed — CI matrix back to green (Docs, Clang Debug, clang-tidy, CUDA)

The four GitHub Actions lanes that had been dark since they were first
introduced (committed broken in P1.12 / P1.13 / P2.6) are now green:

- **`Docs / Build Sphinx site`** -- ran with `-W --keep-going`, so 28
  pre-existing warnings were hard-failing the build. Triaged into:
  - removed the bogus `solvers/diagonalization.h` doxygenfile entry and
    switched the surviving solver/GPU `doxygenfile` directives to a
    short prose listing (the per-file directives were re-emitting the
    same shared typedefs from each header and producing
    "Duplicate C++ declaration" errors);
  - added `.. default-domain:: py` to `docs/api/python.rst` so
    `:mod:` / `:func:` / `:class:` / `:meth:` resolve under the Python
    domain instead of being silently routed to the cpp primary domain;
  - dropped the `../CHANGELOG` and `../CONTRIBUTING` toctree entries +
    the `../MODERNIZATION_AUDIT.md` MyST link from `docs/index.md`
    (Sphinx can't include source files outside the doc root); replaced
    with explicit GitHub URLs in a new "Project documents" section;
  - added explicit MyST anchors `(from-source-cmake)=` etc. to
    `docs/guides/install.md` so the in-page table-of-contents links
    resolve;
  - fixed the malformed reST table in
    `python/quantum_ed/hamiltonian.py` (column 1 separator was 27
    chars but the longest cell was 29 chars);
  - created `docs/_static/.gitkeep` so the `html_static_path` entry no
    longer warns.
- **`CI / Linux / Clang / OpenBLAS / Debug`** and **`CI / Linux /
  clang-tidy`** -- both failed at the `Build` step because CI
  installed `clang` without `libomp-dev`, so `#include <omp.h>` from
  `construct_ham.h` was unresolved (GCC bundles its own libgomp via
  `build-essential`, which is why the GCC/OpenBLAS lane was unaffected).
  Added `libomp-dev` to both `apt-get install` lists.
- **`CI / Linux / CUDA build-only`** -- after three failed attempts to
  resurrect this lane via the `Jimver/cuda-toolkit` action
  (`v0.2.16` 404'd on retired CUDA 12.4.1 deb URLs; `v0.2.34/35` broke
  on node24 + the deprecated GH Actions cache v1 API
  (Jimver/cuda-toolkit#390); `v0.2.32` exited in ~20s before any
  install), replaced the action entirely with direct `apt-get`
  installs from NVIDIA's official Ubuntu 22.04 repo
  (`cuda-keyring_1.1-1` + `cuda-nvcc-12-6` and friends). Two follow-up
  fixes were needed once the install was happy:
  - **`Configure` step**: added `libcurand-dev-12-6` so CMake's
    `find_package(CUDAToolkit)` could resolve `CUDA::curand` (we link
    it from `cmake/EDLibraries.cmake` and `CMakeLists.txt`, but it
    isn't pulled in by the cudart/cublas/cusolver dev metapackages).
  - **Final link step**: added the runtime-only `libnvjitlink-12-6`
    (a transitive dependency of the cusolver/cusparse `.so`s in 12.6
    that the dev packages do not list), otherwise `ld` fails with
    `undefined reference to __nvJitLink*_12_6` when linking the `ED`
    executable.
  The configure + build of `ed_solvers_gpu` and `ED` (SM 70 only) now
  completes in ~2.5 minutes locally inside an `ubuntu:22.04` container
  using exactly the CI command sequence.

`ctest` (102/102) and `pytest` (98/98) remain green locally; the local
Sphinx build now reports `build succeeded.` with `-W --keep-going`.

### Changed — NLCE upgraded to a standalone, plugin-architecture package

The NLCE workflow has been promoted from "three driver scripts that
share a `_common.py`" into a proper modern package with a unified CLI
and registry-based extension points.

- **New `workflows/nlce/core/` subpackage** (~600 LOC) holding the only
  things downstream extensions inherit from:
  - `Geometry` ABC + `register_geometry` / `get_geometry` / `list_geometries`
  - `Pipeline` ABC + `register_pipeline` / `get_pipeline` / `list_pipelines`
  - `NLCEWorkflow` orchestrator running the canonical 4-step pipeline
    (clusters → Hamiltonians → ED → summation), with parallelism,
    skip-step flags, and the `--streaming-symmetry` orbit-basis
    precompute step
  - `EDOptions` / `build_ed_command` / `run_ed_subprocess` (the only
    legal way for a `Pipeline` to talk to `./ED`) moved here from
    `_common.py`
  - I/O helpers (`ClusterEntry`, `get_cluster_files`,
    `count_sites_in_info_file`, `load_thermo_dataset`,
    `load_tpq_thermo_dataset`, `setup_logging`, `check_gpu_available`)

- **New `workflows/nlce/geometries/` subpackage**: concrete
  geometry implementations register themselves on import.
  - `pyrochlore` — XYZ + Zeeman + optional random transverse field
  - `triangular_site` — site-based NLCE, J1-J2 / Kitaev / anisotropic
  - `triangular_triangle` — triangle-based NLCE, same model surface
  - Adding a new lattice = drop a module, decorate with
    `@register_geometry`, append one line to `__init__.py`.

- **New `workflows/nlce/pipelines/` subpackage**: concrete ED-strategy
  implementations register themselves on import.
  - `full_ed` — full / ScaLAPACK auto-promoted dense ED
  - `ftlm` — Finite-Temperature Lanczos with hybrid full-ED for small
    clusters and adaptive Krylov dimension
  - `lanczos_boost` — partial-Lanczos NLCE (Bhattaram & Khatami)
  - Adding a new pipeline = drop a module, decorate with
    `@register_pipeline`, append one line to `__init__.py`.
  - All 9 `Geometry × Pipeline` combinations are valid;
    `full_ed.summation_command` dispatches to the right `NLC_sum_*.py`
    kernel based on geometry.

- **New unified CLI `python -m workflows.nlce`** (`workflows/nlce/cli.py`
  + `__main__.py`):
  - `--list` enumerates registered geometries and pipelines.
  - `--geometry=… --pipeline=…` selects exactly one of each; the chosen
    pair injects its own model/ED-method flags into the parser.
  - `--max_order`, `--base_dir`, `--ed_executable`, `--temp_min/max/bins`,
    `--thermo`, `--skip_*`, `--parallel/--num_cores` are common across
    every combination.
  - Geometry-default temperature ranges (`pyrochlore` → 0.001-20,
    `triangular_*` → 0.1-10) apply when the user doesn't override.

- **The three legacy driver scripts** `run/nlce.py`, `run/nlce_ftlm.py`,
  `run/nlce_triangular.py` are now ~50-line shims that translate the
  historical CLI surface (`--lanczos_boost`, `--site_based`,
  `--skip_ftlm`, …) onto the unified CLI. Existing analysis scripts
  in `analysis/` that invoke them by path keep working unchanged.

- **`workflows/nlce/_common.py`** is now a re-export shim of
  `workflows.nlce.core` for backward compatibility with downstream
  scripts that bind from there.

- **17 new pytest cases** (`python/tests/test_nlce_package.py`) cover
  registry mechanics, ED-CLI builder auto-promotion, pipeline
  hybrid-mode dispatch, summation command routing per geometry, the
  `--list` CLI path, and the legacy-shim argv translators.

- **New top-level `workflows/nlce/README.md`** documents the package
  architecture, the registries, the unified CLI, the
  `Geometry × Pipeline` matrix, the on-disk output schema, the
  ED-binary integration contract, and how to add new geometries or
  pipelines.

`ctest` (102/102) and `pytest` (98/98 — was 81 before, +17 new NLCE
tests) remain green.

### Changed — NLCE workflow refactor (intermediate, superseded above)

- New shared-infrastructure module **`workflows/nlce/_common.py`** (~500
  LOC) consolidates the boilerplate that used to be triplicated across
  the three NLCE driver scripts: file/console logging, cluster-file
  discovery + parsing, the `EDOptions` dataclass + `build_ed_command(...)`
  argv builder, the exit-code-vs-output `run_ed_subprocess(...)`
  driver (with the long-standing "ED crashed during cleanup but the
  HDF5 file is intact" reconciliation), and HDF5/text-file fallback
  readers (`load_thermo_dataset`, `load_tpq_thermo_dataset`).
- New package init files **`workflows/__init__.py`** and
  **`workflows/nlce/__init__.py`** make this a proper Python package.
  Driver scripts add a `sys.path` shim so they remain runnable
  directly without `pip install -e .`.
- `workflows/nlce/run/nlce.py` (pyrochlore, full / Lanczos-boosted ED)
  refactored: `run_ed_for_cluster`, `run_lb_ed_for_cluster`, and the
  per-cluster thermodynamics-plotting step now go through the shared
  helpers; the four legacy ignored CLI flags
  (`--no_auto_method`, `--full_ed_threshold`, `--block_size`,
  `--use_gpu`) are dropped; the duplicated thermal/mTPQ plotting
  branches collapsed into a single block driven by the shared
  HDF5/text readers. Net: 927 → 590 LOC.
- `workflows/nlce/run/nlce_ftlm.py` (pyrochlore, FTLM with hybrid
  full-ED for small clusters) refactored: `run_full_ed_for_cluster`
  and `run_ftlm_for_cluster` now go through the shared helpers; the
  adaptive Krylov heuristic stays. Net: 789 → 689 LOC.
- `workflows/nlce/run/nlce_triangular.py` (triangular lattice, full /
  ScaLAPACK ED) refactored: `run_ed_for_cluster` (with the
  triangular-specific `--symm_threshold` and streaming-symmetry
  knobs, plus the OpenMP=1 workaround for `num_sites <= 8`) goes
  through the shared helpers; the four legacy ignored flags are
  dropped. Net: 701 → 547 LOC.
- New **`workflows/nlce/README.md`** documents the modernized layout,
  the `_common` API surface, the on-disk output schema, and how to
  add new drivers.
- All three drivers now default `--ed_executable` to
  `<repo_root>/build/ED` via `_common.DEFAULT_ED_PATH` rather than
  the brittle `../../../build/ED` relative path.

Net: ~700 LOC of duplicated driver code retired into ~500 LOC of
shared, documented infrastructure. `ctest` (102/102) and `pytest`
(81/81) remain green; all three drivers' `--help` continues to load
cleanly.

### Removed — Phase 2 (DSSF consolidation, P2.14)

- **`src/apps/TPQ_DSSF.cpp` (4 174 LOC)** — the historical standalone DSSF
  binary with its own 14-positional CLI, parallel-HDF5 plumbing, and
  duplicated `/dssf_results/...` HDF5 schema. Every feature it offered is
  now reachable through `ED dssf <method>` (P2.14).
- **`run_dssf_mode` (~394 LOC)** and the deprecated `--dssf` half-positional
  flag in `src/apps/ed_main.cpp`. The flag is now explicitly rejected with
  a friendly migration hint pointing at `ED dssf <method>` (P2.14).
- **`configs/13_tpq_dssf_workflow.cfg`** — the legacy two-step worked
  example (P2.14).
- **`TPQ_DSSF` CMake target** + parallel-HDF5 plumbing + install-rule +
  CUDA build-only CI lane entry. The single canonical `ED` executable is
  now the only installed binary other than the BFG research add-on (P2.14).

### Changed — Phase 2 (DSSF consolidation, P2.14)

- `src/apps/ed_main.cpp` collapsed from **3 092 → 585 LOC**: the
  `--dssf` argv path and `run_dssf_mode` shim were excised; ten now-unused
  includes pruned; `print_help` rewrote the "DSSF MODE" block as the
  canonical "DSSF / SSSF SUBCOMMAND" section.
- `configs/15_ed_dssf_mode.cfg` rewritten as the canonical `ED dssf
  <method>` worked example (five real example commands).
- `README.md` swapped the "DSSF Mode (Simplified Spectral Interface)" +
  "TPQ_DSSF Executable" sections for one "DSSF / SSSF Subcommand" section
  + a migration note; project-structure tree refreshed; new
  `quantum_ed` Python package documented; TOC renumbered.
- `MODERNIZATION_AUDIT.md` marks P2.14 complete; top-level status header
  now reads "Phase 0, Phase 1, Phase 2 (P2.1–P2.14) all landed".
- All cross-cutting docstrings (`include/ed/dssf/*.h`, `src/cli/*.cpp`,
  test docstrings, helper scripts, `python/quantum_ed/dssf.py`,
  `python/quantum_ed/_bindings/*.cpp`, `python/tests/test_dssf.py`,
  `docs/guides/*.md`, etc.) reworded to point at `ED dssf` instead of
  `TPQ_DSSF`. Migration / historical-context comments are preserved
  verbatim where they document deleted behaviour.

Net change: **~−4 800 LOC**. ctest: 102/102 PASS. pytest: 81/81 PASS.

### Added — Phase 0 (lab-grade hygiene)

- `MODERNIZATION_AUDIT.md`: comprehensive read-only audit and 3-phase modernization roadmap with 43 numbered atomic commits (P0.1–P0.16, P1.1–P1.13, P2.1–P2.14).
- `.gitignore` (P0.1): excludes build dirs, sanitizer/coverage artifacts, Python wheel/cache trees, ED run outputs, editor noise, Doxygen/Sphinx outputs.
- `.editorconfig` (P0.2): UTF-8, LF, 4-space indent, 100-col, trim trailing whitespace.
- `.clang-format` (P0.3): LLVM-derived, 4-space, 100-col, pointer-attached-to-type, includes regrouped (project headers last). Not yet applied via `clang-format -i` — that lands in P0.11.
- `CMakePresets.json` (P0.4): 8 presets covering `default` / `debug` / `debug-asan` / `release-mpi` / `release-cuda` / `release-cuda-mpi` / `ci-linux`. Newcomers can `cmake --preset default && cmake --build --preset default && ctest --preset default`.
- `.github/workflows/ci.yml` (P0.5): minimal Linux/GCC/OpenBLAS Release lane on `ubuntu-22.04`. ctest must remain 12/12 green.
- `CONTRIBUTING.md` / `CHANGELOG.md` / `CITATION.cff` (P0.6).

### Removed
- `python/edlib/__pycache__/*.pyc` are no longer tracked (P0.1b).

## How to read this file

- `[Unreleased]` collects everything between the last tag and the present.
- Each section is grouped under `Added` / `Changed` / `Deprecated` / `Removed` / `Fixed` / `Security`.
- Entries that correspond to a numbered audit step (e.g. P0.7) cite the step in parentheses for traceability.
