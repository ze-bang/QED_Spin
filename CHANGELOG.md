# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed — NLCE workflow modernization

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
