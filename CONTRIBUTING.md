# Contributing to QED_Spin

## Environment

Job scripts (the gate, the golden harness, the benches) source `scripts/env.sh` from the
repository root. It does the following, in order:

1. Sources the personal site file `QED_SITE_ENV` (default `~/.config/qed/site.env`), if it
   exists.
2. Loads `scripts/clusters/${QED_CLUSTER:-alliance}.env`.
3. Activates `QED_VENV`, if set.
4. Sets `OMP_NUM_THREADS` to `SLURM_CPUS_PER_TASK` (default 8).
5. Points `PYTHONPATH` at this checkout's `python/` and `QED_CORE_DIR` at
   `build/${QED_VARIANT:-cuda}/python/qed`. `QED_PYTHONPATH` (with an optional
   `QED_CORE_DIR`) selects another tree instead, for example a frozen snapshot.

Personal paths belong in the site file, never in the repository:

```bash
# ~/.config/qed/site.env
QED_VENV=$HOME/venvs/qed          # virtualenv with pytest and pynauty
QED_PYBIND11_DIR=/path/to/pybind11/share/cmake/pybind11   # only if CMake cannot find pybind11
QED_CLUSTER=alliance              # scripts/clusters/<name>.env
```

`scripts/clusters/alliance.env` loads the Alliance toolchain (StdEnv/2023, GCC 12.3, AOCL
through FlexiBLAS, CUDA, Python 3.11). `scripts/clusters/local.env` takes whatever is on
`PATH`.

## Build

`scripts/build.sh` is the one build entry point. Each variant owns its build directory. The
extension goes to `<build>/python/qed/`, never into the source tree.

```bash
scripts/build.sh --variant cpu --tests                  # -> build/cpu: engine, unit tests, qed._core
scripts/build.sh --cluster alliance --variant cuda      # -> build/cuda
export PYTHONPATH=$PWD/python QED_CORE_DIR=$PWD/build/cpu/python/qed
```

Build options:

- `--cluster NAME` (default `local`) and `--variant cpu|cuda` (default `cpu`).
- `--tests` builds the unit tests; `--no-python` skips `qed._core`.
- `--target T` (repeatable), `--clean`, `--jobs N` (default `SLURM_CPUS_PER_TASK`, else 4).
- `--name S` builds into `build/<variant>-S`. `--arch A` sets `-march` (default `native`).
- Arguments after `--` go to CMake.

If a build is killed midway, the next build finds the leftover `.build-running` marker,
deletes what the killed build wrote, and rebuilds it.

`CMakePresets.json` has one preset, `ci-linux`, which GitHub CI uses. `-DBLAS_PROFILE=`
chooses the BLAS / LAPACK provider: `AUTO`, `FLEXIBLAS`, `OPENBLAS` or `MKL`
(`cmake/EDBlas.cmake`).

To build the documentation site, configure with `-DED_BUILD_DOCS=ON` and run `--target sphinx`.
This needs Doxygen and the packages of `pyproject.toml`'s `docs` extra (sphinx, furo, breathe,
myst-parser). The output goes to `<build>/docs/sphinx/html`.

## Tests

```bash
ctest --test-dir build/cpu --output-on-failure
python -m pytest tests/python                         # API tests
python -m pytest tests/python/grid -m grid -k cpu     # the coverage grid, CPU cells
```

`pyproject.toml` deselects `slow`, `grid` and `regress` by default. The markers are:

| marker | meaning |
|---|---|
| `grid` | task × symmetry × backend cells against a dense reference |
| `regress` | audit repros (the gate runs `-m 'regress and not perf and not info'`) |
| `perf` | a regress case for a confirmed performance issue (not gated) |
| `info` | a regress case that is inconclusive or not reproduced (not gated) |
| `gpu` | needs a CUDA device; skipped without one |
| `slow` | long-running performance or scale guard |

`tests/python/conftest.py` pins the build under test. It uses `QED_CORE_DIR`, else
`ED_BUILD_DIR` (a CMake build tree), else the installed package. When a build is named, it
refuses to run if `import qed` resolves to any other package or extension.

### Adding a test

- **C++.**
  - Put Catch2 v3 `TEST_CASE`s in `tests/unit/test_<topic>.cpp` and register the file in
    `CMakeLists.txt` with `ed_add_test(test_<topic> tests/unit/test_<topic>.cpp)`. A test that
    needs engine-private headers adds `target_include_directories(test_<topic> PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src)`. Tests that need CUDA to compile go inside the
    `if(WITH_CUDA)` block.
  - A test case that needs a device calls `SKIP(...)` without one.
  - The tests link `tests/common/catch2_main.cpp`. A failure exits 1 and a run where every
    case was skipped exits 77, which ctest reads as a skip.
- **Python.**
  - Put tests in `tests/python/test_<topic>.py` and start with
    `qed = pytest.importorskip("qed")`.
  - Build references with `tests/python/support/oracle.py`. It turns term lists over
    `+-zudxyI` into numpy matrices without the library's matvec; `terms_of(op)` reads a
    library operator through its canonical terms.
  - The shared models are in `support/models.py` (`MODELS`).
  - Device tests carry the `gpu` marker or skip when `qed._core.cuda_device_count() == 0`.
- **A new task, symmetry content or device path** gets grid cells:
  - Add the model to `tests/python/support/models.py` (`MODELS`). Add the task to `TASKS`,
    or the content to `CONTENT_MODELS`, in `tests/python/grid/test_grid.py`.
  - A cell fails the run only when `baseline_cpu.json` / `baseline_gpu.json` record it as
    `pass`. Promote a cell there once the gate has measured it passing.
  - `QED_GRID_REPORT=<path>` writes the measured table.
  - Sampled GPU cells reproduce the CPU path at the same seeds.
- **A result with no closed-form reference**, such as a fixed-seed thermal curve or which call
  raises, gets a golden case (see below).
- **A confirmed bug** gets a regress repro before or with its fix (see below).
- **Representative choice.** The gate runs the test suites again with `ED_SYM_SUBLATTICE=1`
  (the `*_slc` stages). A test must hold under both representative rules, and
  `tests/python/test_sublattice.py` compares the two directly.

## The gate

```bash
scripts/gate/submit.sh <account>                        # everything, cuda variant
scripts/gate/submit.sh <account> ctest 'grid_cpu_*'     # stages by name or glob
scripts/gate/submit.sh <account> --variant cpu          # the CPU-only build and its table
scripts/gate/submit.sh <account> --build-only           # the build job alone
scripts/gate/submit.sh <account> --dry-run              # print the sbatch commands
```

`submit.sh` submits a build job (`build.sbatch`). It then submits job arrays (`task.sbatch`)
that depend on the build and run the stages `tasks.sh` lists: the CPU stages in two arrays (10-
and 20-minute limits), the GPU stages in one. Each task appends `<stage> <exit code>` to
`logs/gate/<build id>/rc` and keeps its log beside it.

```bash
awk '{c[$1]=$2} END {for (k in c) print k, c[k]}' logs/gate/<build id>/rc   # last run per stage
```

A change is ready when every stage reports 0. Submit the gate from a snapshot worktree synced
with `rsync --checksum`, not from the tree you are editing: the tasks import `python/` while
they run. A GPU task on a node without a usable device stops at once with a `NODE FAILURE`
line, naming the node, and rc 3; resubmit that stage on another node.

| where | stages |
|---|---|
| build job | the lints `env_registry`, `no_print`, `tolerance_literals`, `int_narrowing`; `build_tests` (`build/<variant>-tests`, unit tests, no Python); `build_core` (`build/<variant>`, target `_core`) |
| CPU array | `pytest`; `golden_cpu`; `examples` (every `examples/[0-9]*.py`); `grid_cpu_levels`, `grid_cpu_thermal`, `grid_cpu_dyn0_{zz,pm,3b}`, `grid_cpu_dynT_{zz,pm,3b}_{a,b}`; `regress_cpu_0` .. `regress_cpu_10`; `fuzz_cpu_s1` .. `fuzz_cpu_s4`; with `ED_SYM_SUBLATTICE=1`: `pytest_slc`, `grid_cpu_levels_slc`, `grid_cpu_thermal_slc`, `grid_cpu_dyn0_{zz,pm,3b}_slc` |
| GPU array (cuda variant, an H100 MIG slice) | `ctest`; `pytest_gpu` (`test_device.py`, `test_sublattice.py`); `golden_gpu`; `grid_gpu_levels`, `grid_gpu_exact_ftlm`, `grid_gpu_mtpq`, `grid_gpu_dyn0_{zz,pm,3b}`, `grid_gpu_dynT_{zz,pm,3b}`; `regress_gpu_0` .. `regress_gpu_2`; `fuzz_gpu_s1`, `fuzz_gpu_s2`; `ctest_slc`, `grid_gpu_levels_slc` |
| cpu variant | `ctest` and `ctest_slc` move to the CPU array; there is no GPU array |

The `regress_*`, `grid_*` and `fuzz_*` CPU tasks get 20 minutes, the other CPU tasks 10, and
GPU tasks 20. `GOLDEN_REF` names the golden reference directory (default `api-2026-09`).

Sanitizers run once per batch of commits, not on every gate:

```bash
sbatch --account=<acct> --export=ALL,KIND=asan scripts/gate/sanitize.sbatch     # ASan + UBSan: ctest, test_api.py
sbatch --account=<acct> --export=ALL,KIND=cusan --gpus-per-node=<gpu> scripts/gate/sanitize.sbatch
```

`cusan` runs compute-sanitizer memcheck and synccheck over the device unit tests of the gate's
cuda build and over `tests/python/regress/repros/X04-gpu-paths-under-sanitizer.py`. A
sanitizer report in a log fails its check whatever the exit code.

### Lints

The build job runs four checks, each pure grep or perl:

- `scripts/check_env_registry.sh`: every `ED_*` / `QED_*` variable the sources read is a row
  of `include/ed/core/config.h`. Every row is used, no row is declared twice, and no C++ reads
  one with a raw `getenv` (use `ed::env`).
- `scripts/check_no_print.sh`: no C++ in `include/`, `src/` or the bindings writes to
  stdout/stderr (use `ED_LOG`, `include/ed/core/log.h`). No module of `python/qed` prints
  (use `qed._log`).
- `scripts/check_tolerance_literals.sh`: no tolerance literal from 1e-8 to 1e-15 in C++
  outside `include/ed/core/numerics.h`. A threshold on an energy is relative to s_H. A literal
  on a scale-free quantity carries `// scale-free: <why>` on its line or the line above.
- `scripts/check_int_narrowing.sh`: no cast of a size (`dim`, `n`, `rows()`, `reps.size()`, ...)
  to `int`. Use `ed::core::checked_narrow<T>(x, what)`, or mark a bounded cast with
  `// narrow-ok: <why>`.

## Golden records

`tests/python/golden/golden.py` records what every verb returns on small systems, and later
commits are compared with it. The references are in
`tests/python/golden/refs/<tag>/{cpu,gpu}.json.gz`. Run the harness through
`scripts/golden/run.sbatch` with these variables:

- `DEVICE=cpu|gpu`.
- `MODE=compare|record|bless|bless_new|retire`.
- `REF=<file>`.
- `ONLY`, `REASON`, `TOL`, `GOT`, `ONCE` for the modes that take them.

The GPU harness sets `ED_SYM_LG_GPU=1`, so that toy blocks reach the device lane. The tiers
are `exact` (1e-10), `transport` (1e-7) and `stochastic` (fixed-seed sampling, 1e-10).

Rules:

- **A change lands only when both compare jobs exit 0.** A refactor that claims to move no
  number shows it bitwise: record the base build with `MODE=record ONCE=1` into a scratch
  file, then compare the new build against it with `TOL=0`.
- **Re-blessing a reference is its own commit**, right after the change that moves the
  records, and that commit carries the compare output that justified it.
  - `MODE=bless` takes exact case names (`ONLY="..."`) and a `REASON`, which the file's meta
    keeps. It runs each case twice and refuses a case whose two runs differ, or that raises
    (unless `golden.py bless` is given `--allow-raise`).
  - Re-bless only the records the change moves, and say in the body why each one moves and
    how it compares with the exact or dense answer.
- **New cases** are added to `cases.py` and entered with `MODE=bless_new REASON="..."`.
- **Removing a feature on purpose.** Delete its cases from `cases.py`, then run
  `MODE=retire ONLY="<names>" REASON="..."`.
  - `retire` refuses while `cases.py` still produces a named case.
  - The dropped records move to `meta.retired` with the commit and the reason.
- **Determinism.** `record` runs every case twice. A case whose runs disagree goes under
  `quarantine`: `compare` reports it, but it never gates.
- **Dense check.** Every record that holds the complete spectrum of a model of at most
  `cases.DENSE_MAX_N` (10) sites must equal the dense spectrum of the model's term list.
  `record` refuses otherwise, and `compare` checks it too.

## Regression ratchet

`tests/python/regress/repros/<ID>.py` holds one script per audit finding. Each script prints
exactly one `REPRO: <VERDICT> ...` line, where the verdict is `CONFIRMED`, `NOT_REPRODUCED` or
`INCONCLUSIVE`. `tests/python/regress/manifest.json` has one entry per script:

```json
{"id": "C01-pyapi-01", "rep": "C01-pyapi-01", "sev": "high", "axis": "correctness", "status": "fixed", "dev": "cpu", "seconds": 30}
```

| status | the case |
|---|---|
| `open` | a confirmed bug that is still there: `xfail(strict=True)`. Once a fix stops the script printing `CONFIRMED`, it XPASSes and fails the gate until the entry says `fixed` |
| `fixed` | must not print `CONFIRMED` again |
| `perf` | a confirmed performance issue (marker `perf`, not gated) |
| `info` | inconclusive, not reproduced, or a note (marker `info`, not gated) |

A script that crashes, exits non-zero, times out (after max(3 × `seconds`, 120) s) or does
not print exactly one `REPRO` line fails in every status.

- `dev` is `cpu`, `gpu` or `both`. CPU cases run with the GPUs hidden.
- `seconds` is the time cap that `QED_REGRESS_SHARD=k/n` uses to pack the shards: 11 CPU and
  3 GPU in the gate.

A new repro starts with these lines, then a docstring stating the claim:

```python
# AUDIT-ID: <ID>
# DEVICE: cpu
# SECONDS: 30
```

Run one script with `python tests/python/regress/run_one.py tests/python/regress/repros/<ID>.py`.
When a fix lands, flip its entry to `fixed` and name the fixing commit and the evidence. In
the history this reads `regress: <ID> fixed by <sha> (<evidence>; job <id>)`.

## Fuzzer

`tests/python/fuzz/fuzz.py` draws random small models (N ≤ 12) with verified symmetry content,
a symmetry request and a task, and compares the result with a dense numpy/scipy reference.
Each case runs in a worker process. The gate runs CPU seeds 1-4 and GPU seeds 1-2, 150 cases
each, with `--strict`.

- `tests/python/fuzz/known.json` lists the accepted failures. Each entry names a ledger id
  that is `open` in `manifest.json`, plus a predicate on the case record or explicit
  `[seed, index, device]` cases.
- `--strict` exits 1 on any unexplained non-pass record (a `harness_error` included), and 2
  when an entry names a ledger id that is no longer open.
- When the bug is fixed, flip the manifest entry and delete the known entry. Entries only ever
  go away; a new finding is fixed or entered in the ledger, never accepted directly.
- Run other seeds without `--strict` from time to time and triage what they find.
  `--replay SEED-INDEX` reruns one case in-process.

## Benchmarks

`bench/submit.sh <account> [case ...]` submits each case of `bench/cases.py` as its own job,
on the cuda build of the checkout, with `REPEATS` fresh runs (default 3). `bench/run.py`
appends one row per run to `bench/results/<commit>.jsonl`. A row holds the wall time, peak
RSS, the job id and the engine's per-block record.

`python bench/run.py --compare <baseline.jsonl> [--results <file>]` compares the median of the
repeats with the baseline. It exits 1 when a case is more than 10% slower or uses more than
15% more memory. Performance claims in commit messages cite these rows and their job ids.

## Style

- C++17 and CUDA C++17. `.clang-format` (clang-format 17) sets the format: 4-space indent, 120
  columns. Headers use `#pragma once` and are reached as `<ed/...>`.
- Python: ruff 0.6.9 (`pyproject.toml` `[tool.ruff]`): line length 120, quotes as written, the
  pyflakes, pycodestyle and bugbear checks.
- `.pre-commit-config.yaml` runs clang-format, ruff, ruff-format and the generic checks; install
  it once (`pip install pre-commit && pre-commit install`). CI runs `pre-commit run --all-files`
  and fails on any change it would make.
- `.clang-tidy`: bug-finding checks only, every warning an error. CI runs it over `src/`; silence
  a justified warning with `NOLINT(check)` and a reason on the line.
- Comments describe the code as it is: what it does and why. History goes in commit messages
  and `CHANGELOG.md`.
- Every environment variable is a row of `include/ed/core/config.h` and is read through
  `ed::env`; `qed.debug_env()` prints the table.
- The library writes nothing to the console.
- Errors are the types in `include/ed/core/errors.h` (`InvalidRequest`, `EmptySelection`,
  `Unsupported`, `DeviceUnavailable`, `DeviceUnsupported`, `ResourceLimit`,
  `ConvergenceError`). The bindings translate them into `qed.errors`.
- A device lane never falls back to the host under `device="gpu"`: it raises, naming the block.

## Where things live

- **A verb.**
  - Python surface: `python/qed/_verbs/<verb>.py`. Binding: `python/qed/_bindings/sectors.cpp`.
  - C++ entry point: `include/ed/sectors/{sectors,thermal,dynamics,expect}.h`.
  - Drivers: `src/engine/` (`eigs.cpp`, `thermal.cpp`, `dynamics.cpp`, `expect.cpp`). The star
    walk is in `walk.h`, the per-block solves in `block_solve.cpp`, the spin towers in
    `tower.cpp`.
- **A symmetry.**
  - `Symmetry.resolve` (`python/qed/_verbs/symmetry.py`), discovery and the group split
    (`python/qed/discovery.py`, `_groups.py`).
  - Sectors and representatives: `include/ed/basis/` (`compiled_group.h`, `orbit_table.h`,
    `rep_sector.h`, `irreps.h`, `sublattice_code.h`), `src/engine/stars.cpp` and
    `group_sector.cpp`.
- **An operator.** `include/ed/ops/`: `operator.h`, `algebra.h` (the exact algebra), `term.h`,
  `invariance.h` (canonical terms and symmetry verdicts), `program.h` (matrix elements between
  sectors), `row_walk.h`.
- **A kernel.** `include/ed/krylov/`, `include/ed/thermal/` and `include/ed/dynamics/`, written
  once against the backend interface (`include/ed/matvec/backend.h`).
- **Device code.** `include/ed/gpu/` and `src/gpu/`:
  - `rep_matvec.cu`: the sector gather and the device CSR;
  - `little_group.cu`: batched dense solves;
  - `rep_matrix_elements.cu`.

  Where a block runs is decided in one place: `ed::place` (`include/ed/core/select_backend.h`),
  reading the 'auto' table in `include/ed/core/device.h`.
- **An example.** `examples/`, one script per family of verbs; the gate runs them all.

## Commits

- One logical change per commit, landed with a green gate. A change that moves golden
  records on purpose may show exactly those records failing in its own gate; the re-bless
  commit that follows it turns the gate green.
- **Subject.** `<area>: <what changed>`, the area naming the part touched (`golden`, `grid`,
  `regress`, `bench`, `tests`, `parallel`, `Device CSR`, `FTLM dynamics`, ...), or a plain
  sentence when the change cuts across areas. A plan step or audit id ends the subject in
  parentheses, for example `(P6.2 step 6)` or `(P7.2, P2-gpu-04)`.
- **Body.** Why the change is made and what it moves, with the measured numbers, the SLURM job
  ids that measured them, and the gate that passed (`Gate <id> green`).
- **Bookkeeping commits** stand on their own: `golden: re-bless <records> (<why>)`,
  `golden: retire ...`, `regress: <ID> fixed by <sha> (...)`, `grid: ... baselined ...`.
- **No trailers.** Messages carry no trailers and no attribution lines, and neither do
  comments or documentation.
- **Changelog.** A change users can see adds a bullet to the `Unreleased` section of
  `CHANGELOG.md`: Python-visible changes first, then the C++ API list for the installed
  headers.

The version lives only in `pyproject.toml` (`version = "X.Y.Z"`). CMake (the project version,
compiled into `qed._core.__version__`, which `qed.__version__` is) and `docs/conf.py` read it,
and `tests/python/test_api.py` checks `qed.__version__` and `qed._core.__version__` against it.

## Reporting bugs

Include:

- the build command, the compiler and CUDA versions;
- the output of `qed.debug_env()`;
- a short Python script that reproduces the problem.
