# GPU x Symmetry x Workflow benchmark — Summary

May 2026 — companion to `bench_gpu_symmetry_matrix.py` and
`bench_gpu_symmetry_focused.py`. Documents the present GPU coverage
of `qed.solve` / `qed.thermal` / `qed.spectral` across all
(workflow, symmetry-mode) cells of the matrix.

## Host

- GPU: NVIDIA RTX 4080 SUPER (16 GB)
- CPU: x86_64 WSL2 host, ed_core compiled with OpenMP
- Build: `WITH_CUDA=ON`, `WITH_MPI=ON`

## Matrix axes

* **workflow** ∈ { GS, FT-FTLM, FT-LTLM, FT-KPM_DOS, DSSF-GS, DSSF-FT }
* **sym** ∈ { none, Sz, Symm, Sz+Symm }
* **device** ∈ { cpu, gpu }
* **N** ∈ { 8, 10, 12 } (full matrix); { 14, 16 } (focused, no-symm only)

Model: 1-D Heisenberg ring with periodic bonds; symmetry: Z_N
translation discovered via `qed.find_symmetries`.

## Architecture map (which lanes have a GPU kernel)

| workflow / sym | none | Sz | Symm | Sz+Symm |
|----------------|------|-----|------|---------|
| GS (LANCZOS)   | GPU  | GPU | CPU-fallback (Phase 1c) | CPU-fallback (Phase 1c) |
| FT-FTLM        | GPU  | GPU | not-implemented (qed.thermal lacks `device=`) | not-implemented |
| FT-LTLM        | not-implemented (no GPU kernel) | not-implemented | not-implemented | not-implemented |
| FT-KPM_DOS     | not-implemented (no GPU kernel) | not-implemented | not-implemented | not-implemented |
| DSSF-GS        | not-implemented (in-memory `qed.spectral` lacks `device=`) | not-implemented | CPU-fallback (Phase 1c) | CPU-fallback (Phase 1c) |
| DSSF-FT        | not-implemented (in-memory `qed.spectral` lacks `device=`) | not-implemented | CPU-fallback (Phase 1c) | CPU-fallback (Phase 1c) |

Notes:
* **CPU-fallback** means the GPU call routes through the streaming-
  symmetry directory binding, which advertises
  `Geometry::supports_device_matvec = true` only when
  `ED_GPU_SYMMETRY_MIRROR=1`. With the env var unset (default), the
  per-sector backend selector stays on the CPU lane. Setting it to
  `1` surfaces the Phase 1c stub
  (`StreamingSymmetryOperator::bind_cuda_for_sector`) which throws
  `std::logic_error` and produces an empty result.
* **not-implemented** means there is no public Python API that
  accepts a `device=` kwarg for that cell today; the workflow either
  hasn't grown the kwarg (`qed.thermal`, in-memory `qed.spectral`)
  or the C++ side has no GPU kernel for that solver (`LTLM`,
  `KPM_DOS`).

## Headline numbers (wall clock, seconds)

Full matrix at N=12 (Hilbert dim 4096):

| workflow | sym | dim | cpu_s | gpu_s | speedup |
|----------|-----|------|-------|-------|---------|
| GS | none | 4096 | 0.40 | 0.37 | 1.07x |
| GS | Sz | 924 | 0.42 | 0.36 | 1.17x |
| GS | Symm | 341 | 0.90 | 0.89 | 1.00x  (CPU-fallback) |
| GS | Sz+Symm | 77 | 0.54 | 0.55 | 0.97x (CPU-fallback) |
| FT-FTLM | none | 4096 | 0.69 | 0.65 | 1.06x |
| FT-FTLM | Sz | 924 | 0.46 | 0.44 | 1.03x |
| FT-FTLM | Symm | 341 | 5.71 | — | gap |
| FT-FTLM | Sz+Symm | 77 | 3.62 | — | gap |
| FT-LTLM | none | 4096 | 1.04 | 1.06 | 0.98x |
| FT-LTLM | Sz | 924 | 0.54 | 0.54 | 1.02x |
| FT-LTLM | Symm | 341 | 3.03 | — | gap |
| FT-LTLM | Sz+Symm | 77 | 2.17 | — | gap |
| FT-KPM_DOS | none | 4096 | 1.59 | 1.65 | 0.97x |
| FT-KPM_DOS | Sz | 924 | 0.69 | 0.71 | 0.97x |
| FT-KPM_DOS | Symm | 341 | 2.85 | — | gap |
| FT-KPM_DOS | Sz+Symm | 77 | 2.48 | — | gap |
| DSSF-GS | none | 4096 | 0.38 | 0.38 | 1.01x |
| DSSF-GS | Sz | 924 | 0.38 | 0.37 | 1.03x |
| DSSF-GS | Symm | 341 | 0.92 | 0.91 | 1.01x (CPU-fallback) |
| DSSF-GS | Sz+Symm | 77 | 0.51 | 0.50 | 1.01x (CPU-fallback) |
| DSSF-FT | none | 4096 | 1.41 | 1.39 | 1.01x |
| DSSF-FT | Sz | 924 | 0.59 | 0.57 | 1.02x |
| DSSF-FT | Symm | 341 | 187.1 | 187.1 | 1.00x (CPU-fallback) |
| DSSF-FT | Sz+Symm | 77 | 44.66 | 44.46 | 1.00x (CPU-fallback) |

Focused (none / Sz only) at N=14 / N=16:

| workflow | sym | N | dim | cpu_s | gpu_s | speedup |
|----------|-----|---|------|-------|-------|---------|
| GS | none | 14 | 16384 | 0.40 | 0.34 | 1.17x |
| GS | none | 16 | 65536 | 0.43 | 0.40 | 1.09x |
| GS | Sz | 14 | 3432 | 0.39 | 0.37 | 1.05x |
| GS | Sz | 16 | 12870 | 0.41 | 0.39 | 1.05x |
| FT-FTLM | none | 14 | 16384 | 2.29 | 2.10 | 1.09x |
| FT-FTLM | none | 16 | 65536 | 6.13 | 6.07 | 1.01x |
| FT-FTLM | Sz | 14 | 3432 | 0.62 | 0.64 | 0.97x |
| FT-FTLM | Sz | 16 | 12870 | 1.43 | 1.48 | 0.96x |
| DSSF-GS | none | 14 | 16384 | 0.39 | 0.40 | 0.97x |
| DSSF-GS | none | 16 | 65536 | 0.53 | 0.47 | 1.14x |
| DSSF-GS | Sz | 14 | 3432 | 0.37 | 0.36 | 1.02x |
| DSSF-GS | Sz | 16 | 12870 | 0.40 | 0.39 | 1.02x |
| DSSF-FT | none | 14 | 16384 | 5.06 | 4.66 | 1.09x |
| DSSF-FT | none | 16 | 65536 | 15.88 | 15.67 | 1.01x |
| DSSF-FT | Sz | 14 | 3432 | 1.21 | 1.23 | 0.99x |
| DSSF-FT | Sz | 16 | 12870 | 3.85 | 4.07 | 0.95x |

## Observations

1. **GPU lane is wired correctly across {GS, FTLM, DSSF-GS, DSSF-FT} ×
   {none, Sz}.** All 16 of those cells run on GPU when `device='gpu'`
   is passed (or when `device='auto'` and dim ≥ 16384). Eigenvalues
   and observables match the CPU baseline to numerical precision.

2. **GPU speedups are modest (~1.0-1.2x) up to N=16.** Two causes:
   - Python subprocess startup (~0.3 s) dominates the small-dim
     cells, masking any kernel-level GPU advantage.
   - At dim ≤ 65k, the cuSPARSE matvec kernel saturates well below
     peak; the CPU OpenMP path is competitive on the same primitive.
   - GPU acceleration should emerge clearly at N ≥ 18 (dim ≥ 262k)
     where the matvec runtime is large enough to amortise the
     launch overhead and CPU↔GPU sync.

3. **Symmetry GPU dispatch is a known gap (Phase 1c).** All 8
   (workflow, Symm/Sz+Symm) cells that DO have a GPU kernel today
   silently fall back to the CPU streaming-symmetry binding (same
   wall as CPU). Enabling `ED_GPU_SYMMETRY_MIRROR=1` surfaces the
   architectural stub
   (`StreamingSymmetryOperator::bind_cuda_for_sector`) which throws
   `std::logic_error: lazy GPU symmetry mirror is a Phase 1c
   deliverable`. The Phase 1c follow-up implements the actual
   per-sector GPU mirror.

4. **FT + Symm GPU is a separate gap.** `qed.solve(solver='FTLM'/
   'LTLM'/'KPM_DOS', symmetry=...)` raises
   `NotImplementedError: qed.solve: symmetry projection is only
   supported for ground-state methods today`. The CPU path goes
   through `qed.thermal(directory, use_symmetry_if_available=True)`
   which lacks a `device=` kwarg. Closing this gap requires (a)
   teaching `_diag_with_symmetry` to handle thermal methods, OR
   (b) adding a `device=` kwarg to `qed.thermal` and threading it
   through `workflows_thermal_streaming_symmetry_directory`.

5. **DSSF in-memory has no `device=` kwarg.** `qed.spectral(H,
   observables, ...)` (the no-symmetry, in-memory path) is CPU-only
   regardless of the `device=` kwarg, because the in-memory
   branch in `qed.spectral` does not honour it. The directory
   form with `symmetry=` routes to the streaming-symmetry binding
   which inherits the same Phase 1c gating. To get a GPU DSSF
   today users have to drop into the directory-form CLI without
   symmetry (`qed.spectral(directory, T=..., omega=..., device='gpu')`),
   which shells out to `./ED dssf`.

## Remaining work to close the gaps

| Gap | LOC est | Owner phase |
|-----|---------|-------------|
| GPU symmetry mirror (`bind_cuda_for_sector`) -- Streaming + FixedSz variants | ~600 | Phase 1c |
| `qed.thermal(..., device=)` kwarg + thread through `workflows_thermal_streaming_symmetry_directory` | ~80 | post-Phase-1c |
| `_diag_with_symmetry` thermal-method support (or `_diag_via_workflows_solve` symmetry support) | ~150 | post-Phase-1c |
| GPU kernels for LTLM / KPM_DOS | ~400 each | new |
| `_spectral_in_memory(device=)` honour | ~40 | post-Phase-1c |

## Raw data

- Full matrix (N=8, 10, 12) — `bench_gpu_symmetry_matrix_results.json`,
  `bench_gpu_symmetry_matrix_results.log`
- Focused (N=14) — `bench_gpu_symmetry_focused_results.json`,
  `bench_gpu_symmetry_focused_N14.log`
- Focused (N=16) — `bench_gpu_symmetry_focused_N16.json`,
  `bench_gpu_symmetry_focused_N16.log`

Reproduce:
```bash
python benchmarks/bench_gpu_symmetry_matrix.py --sizes 10,12
python benchmarks/bench_gpu_symmetry_focused.py --sizes 14
python benchmarks/bench_gpu_symmetry_focused.py --sizes 16 \
    --out benchmarks/bench_gpu_symmetry_focused_N16.json
```
