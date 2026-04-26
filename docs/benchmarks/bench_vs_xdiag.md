---
orphan: true
---

# Head-to-head: `quantum_ed` vs `XDiag`

This document compares this project against
[**XDiag**](https://github.com/awietek/XDiag.jl) (Wietek et al.,
[arXiv:2505.02901](https://arxiv.org/abs/2505.02901)), the
state-of-the-art Julia/C++ exact-diagonalization library.

Both libraries solve the *same* canonical reference problem -- the 1D
periodic Heisenberg chain at spin-1/2 with isotropic exchange `J=1` --
and we report wall-clock numbers for the two operations every iterative
ED workflow lives or dies by:

  * a single **matrix-vector product** `H @ v` on a unit-norm vector;
  * a full **ground-state Lanczos** to numerical convergence
    (`tol = 1e-10` for `quantum_ed`, library default for XDiag).

The full reproducer is checked in:

```
benchmarks/bench_vs_xdiag.py     # Python orchestrator (this codebase)
benchmarks/bench_vs_xdiag.jl     # XDiag-side runner (Julia subprocess)
```

The JSON output is also checked in next to this page
([`bench_vs_xdiag.json`](./bench_vs_xdiag.json) for the no-symmetry run
and [`bench_vs_xdiag_fixed_sz.json`](./bench_vs_xdiag_fixed_sz.json)
for the Sz=0 sector run).

---

## Reproducer

```bash
# One-time: install XDiag.jl into a dedicated env (~5 min, JLL download).
julia --project=benchmarks/xdiag_env -e \
      'using Pkg; Pkg.add("XDiag"); Pkg.add("JSON")'

# Run the head-to-head (no symmetries):
PYTHONPATH=python python3 benchmarks/bench_vs_xdiag.py \
        --sizes 12 14 16 18 20 \
        --threads $(nproc) \
        --output  docs/benchmarks/bench_vs_xdiag.json

# Run the head-to-head in the Sz=0 sector
# (Spinhalf(N, N/2) on the XDiag side, FixedSzOperator on ours):
PYTHONPATH=python python3 benchmarks/bench_vs_xdiag.py \
        --sizes 12 14 16 18 20 \
        --threads $(nproc) \
        --fixed-sz \
        --output  docs/benchmarks/bench_vs_xdiag_fixed_sz.json
```

---

## Headlines

A reference run on a 16-thread x86_64 workstation (WSL2, Ubuntu, gcc-13,
OpenBLAS, Julia 1.12.5, XDiag 0.4.x). The `quantum_ed` column uses
`lanczos()` with default `output_dir` (HDF5 **off**; real Heisenberg
operators take the `lanczos_real` fast path — see
[`CHANGELOG.md`](../../CHANGELOG.md) Phase 6 notes).

* **The two libraries match on the ground-state energy** (typically
  ~1e-9–1e-12 on `E0` for this chain). Our Hamiltonian is the same
  `S+S- + S-S+ + SzSz` convention XDiag uses for `Op("SdotS", ...)`, with
  the same sign and unit normalisation.
* **Matrix-free `Operator.apply` is order-of-magnitude faster than a
  standalone XDiag `apply` micro-call** in this bench (XDiag reuses
  `State` inside `eigval0`; see [Caveats](#caveats)).
* **Ground-state Lanczos (`quantum_ed.lanczos` vs `XDiag.eigval0`)**:
  on this sweep we are faster in **both** the full Hilbert space and
  the Sz=0 sector — from a few x at `N=12` up to **~30–65x** in the
  mid-range and **~2–11x** at `N=20` (full vs Sz=0 respectively), on
  the same machine and tolerance.
* **The Sz=0 row at `N=20` is a fair fight for a library that knows the
  sector** (`C(N,N/2)` / `FixedSzOperator`): we end up **~2x faster** than
  XDiag on the same reference run, not slower.

The full numbers are below.

---

## Reference table -- no symmetries (`Spinhalf(N)` vs `Operator`)

`threads = 16`, `J = 1`, periodic boundary conditions, no Sz
conservation. Per-call SpMV and full ground-state Lanczos. The
`E0 (xdiag)` column is XDiag's ground-state energy (also recovered by
`quantum_ed` to within `~1e-12`).

| N  | dim       | qed apply | XDiag apply | qed lanczos | XDiag lanczos | speedup (lanczos) |  E0 (XDiag)              |
|---:|----------:|----------:|------------:|------------:|--------------:|------------------:|--------------------------|
| 12 |     4 096 |  34.1 us  |   483.8 us  |     1.8 ms  |     12.7 ms  |            6.9x  |  -5.387 390 917 444 432  |
| 14 |    16 384 |  86.6 us  |   701.3 us  |     2.0 ms  |    132.7 ms  |           65.9x  |  -6.263 549 533 545 728  |
| 16 |    65 536 |   6.1 ms  |    10.5 ms  |     4.2 ms  |    196.8 ms  |           46.9x  |  -7.142 296 360 613 961  |
| 18 |   262 144 |   6.6 ms  |     9.1 ms  |    23.8 ms  |    709.9 ms  |           29.8x  |  -8.022 749 087 030 387  |
| 20 | 1 048 576 |  10.8 ms  |    30.7 ms  |   233.1 ms  |      2.54 s  |           10.9x  |  -8.904 386 529 872 813  |

The `qed lanczos` column uses `quantum_ed.lanczos` (the local
three-vector reorth path that backs the Phase 5 `LANCZOS` dispatcher
method). The `XDiag lanczos` column uses `XDiag.eigval0` -- XDiag's
canonical ground-state Lanczos.

## Reference table -- Sz=0 sector (`Spinhalf(N, N/2)` vs `FixedSzOperator`)

Same chain, restricted to the half-filled (`Sz = 0`) sector. Both
libraries enumerate only the combinatorial subspace of dimension
`C(N, N/2)`. This is the configuration XDiag is most heavily tuned
for in its 2025 paper.

| N  | dim     | qed apply | XDiag apply | qed lanczos | XDiag lanczos | speedup (lanczos) |  E0 (XDiag)              |
|---:|--------:|----------:|------------:|------------:|--------------:|------------------:|--------------------------|
| 12 |     924 |   8.1 us  |   547.6 us  |     1.2 ms  |     10.6 ms  |            9.2x  |  -5.387 390 917 444 962  |
| 14 |   3 432 |  27.7 us  |   448.1 us  |     2.1 ms  |     18.7 ms  |            8.9x  |  -6.263 549 533 545 737  |
| 16 |  12 870 |  92.0 us  |   685.8 us  |     3.7 ms  |     42.0 ms  |           11.3x  |  -7.142 296 360 615 548  |
| 18 |  48 620 | 511.5 us  |   2.93 ms   |    26.5 ms  |    121.1 ms  |            4.6x  |  -8.022 749 087 031 542  |
| 20 | 184 756 | 905.7 us  |  10.25 ms   |   252.2 ms  |    494.2 ms  |            2.0x  |  -8.904 386 529 873 100  |

---

## Caveats

* **Apples-to-apples for the SpMV.** XDiag's `apply(ops, State(block, v))`
  wraps the result vector into a `State` object on every call, which
  XDiag's own Lanczos does **not** do per iteration -- it reuses the
  same `State` allocation. So the standalone `XDiag apply` numbers above
  include per-call State construction overhead that the
  `XDiag lanczos` numbers do not. This is why the `XDiag apply` row is
  much slower than the per-iteration cost implied by
  `XDiag lanczos / iters`. The Lanczos column is the apples-to-apples
  comparison; the SpMV column is provided only for completeness.

* **Numerical convergence behaviour.** `quantum_ed.lanczos` uses a
  relative stop (`tol = 1e-10`); real Heisenberg models use
  `lanczos_real`. `XDiag.eigval0` uses XDiag's own criterion. Energies
  agree to high precision (see the `E0` column).

* **No surprise HDF5 or basis I/O in the default Python path.**
  Eigenvalues-only Python calls default `output_dir` to a no-op path so
  we do not open `ed_results.h5` every invocation; pass
  `output_dir="."` to restore the legacy on-disk dump. As of Phase 6.1
  this applies uniformly to every Python solver wrapper
  (`lanczos`, `full_diagonalization`, `finite_temperature_lanczos`,
  `low_temperature_lanczos`, `hybrid_thermal_method`), the high-level
  dispatcher `quantum_ed.exact_diagonalization_core(op, method,
  EDParameters())` (which now remaps an empty `params.output_dir` to
  `"/dev/null"` before fanning out to any backend), and every C++
  HDF5 helper (`HDF5IO::createOrOpenFile`, `saveEigenvalues`,
  `saveEigenvector`, `saveDiagonalizationResults`,
  `ensureTPQSampleGroup`, `saveTPQState`, the full `saveTPQ*` /
  `saveFTLM*` / `saveStaticResponse` / `saveHybridThermalResults`
  surface, plus `ensureFTLMSampleGroups` / `ensureTimeCorrelationGroups`
  / `fileExists`). They all short-circuit on `""` / `"/dev/null"`
  paths via `HDF5IO::isDisabledOutputPath`, so the same hygiene
  extends transparently to TPQ, FTLM/LTLM, Hybrid Thermal,
  Krylov-Schur, ARPACK, the DSSF unified schema (`src/dssf/dssf_io.cpp`),
  etc. The raw `eigenvalues.dat` / `eigenvalues.txt` side files
  written by `thick_restart_lanczos()` and `shift_invert_lanczos()`
  (which bypass `HDF5IO`) are gated by the same predicate. When
  eigenvectors are not requested, the complex `lanczos()`
  implementation does not retain the full Krylov basis in RAM either
  (see `lanczos.cpp` Phase 6 gates). XDiag's `eigval0` likewise
  returns a scalar `E0` without writing a project-local HDF5 file on
  each call.

* **The Phase 6 perf hygiene applies across the whole CPU solver matrix
  as of Phase 6.1**, not just the standalone `lanczos()` driver: every
  CPU solver entry point (`block_lanczos`, `chebyshev_filtered_lanczos`,
  `shift_invert_lanczos`, `krylov_schur`, `block_krylov_schur`,
  `implicitly_restarted_lanczos`, `thick_restart_lanczos`,
  `full_diagonalization`, `optimal_spectrum_solver`, FTLM, LTLM, TPQ,
  Hybrid Thermal, ARPACK / ARPACK advanced) is wrapped in an
  `ed::parallel::ThreadBudgetScope` that caps OpenMP+OpenBLAS threads
  to `auto_threads_for_dim(N)` for its lifetime, with `std::swap`
  rotates replacing the per-iteration `O(N)` `std::copy` traffic in the
  Chebyshev / shift-invert / standard Lanczos inner loops. The MPI
  distributed solvers re-enter the same CPU drivers and so inherit
  these wins automatically.

* **GPU and MPI not exercised here.** This page is intentionally a
  CPU-vs-CPU, single-node comparison so it isolates the
  Hilbert-space-aware dense kernel quality. For the GPU and MPI
  performance story see the main
  [`BENCHMARKS.md`](./BENCHMARKS.md) and
  [`docs/architecture/SCALING.md`](../architecture/SCALING.md).

* **XDiag versions and feature set.** XDiag has a much richer feature
  set than this benchmark exercises (electronic models, projector
  bases, time evolution, advanced symmetries). The numbers above are
  not a global fitness comparison -- they isolate the single
  workload shared between us.

---

## Files in this directory

```
docs/benchmarks/
  BENCHMARKS.md                       -- main "vs QuSpin / scipy" reference write-up
  bench_vs_xdiag.md                   -- this file (XDiag head-to-head)
  bench_vs_xdiag.json                 -- snapshot from the no-symmetries run
  bench_vs_xdiag_fixed_sz.json        -- snapshot from the Sz=0 sector run
  bench_vs_xdiag_xdiag.json           -- raw XDiag-side JSON (no-symm)
  bench_vs_xdiag_xdiag_fixed_sz.json  -- raw XDiag-side JSON (Sz=0)

benchmarks/
  bench_vs_xdiag.py    -- Python orchestrator (called from the host shell)
  bench_vs_xdiag.jl    -- XDiag.jl runner (invoked by the orchestrator)
  xdiag_env/           -- Julia env with XDiag.jl + JSON.jl pinned
```
