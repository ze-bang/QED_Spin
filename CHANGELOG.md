# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
