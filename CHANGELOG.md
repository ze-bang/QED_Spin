# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
