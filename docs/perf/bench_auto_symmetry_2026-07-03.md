# `symmetry="auto"` benchmark — 2026-07-03

`benchmarks/bench_auto_symmetry.py` on a Heisenberg ring,
`symmetry="off"` vs `symmetry="auto"` (U(1) Sz × spatial Z_N × spin
flip × time reversal, all auto-composed). WSL2, OpenBLAS, RTX-class
GPU. JSON: `bench_auto_symmetry_cpu_{small,large}.json`,
`bench_auto_symmetry_gpu.json`.

## Results

| verb | N | device | t_off [s] | t_auto [s] | speedup | blocks | max block dim |
|---|---|---|---|---|---|---|---|
| solve (GS, sz=N/2) | 12 | cpu | 0.06 | 0.59 | 0.1× | 1 | 44 |
| thermal (mTPQ) | 12 | cpu | 10.97 | 0.09 | **126×** | 146 | 66 |
| spectral (DSSF S^z_Q) | 12 | cpu | 0.07 | 0.59 | 0.1× | — | — |
| solve (GS, sz=N/2) | 16 | cpu | 0.01 | 0.05 | 0.3× | 1 | 415 |
| thermal (mTPQ) | 16 | cpu | 55.6 | 35.7 | 1.6× | 258 | 715 |
| spectral (DSSF S^z_Q) | 16 | cpu | 0.13 | 0.82 | 0.2× | — | — |
| solve (GS, sz=N/2) | 20 | cpu | 0.13 | 1.27 | 0.1× | 1 | 4 654 |
| spectral (DSSF S^z_Q) | 20 | cpu | 0.80 | 0.95 | 0.8× | — | — |
| solve (GS, sz=N/2) | 24 | cpu | — | 5.7 | — | 1 | 56 450 |
| spectral (DSSF S^z_Q) | 24 | cpu | — | 5.9 | — | — | — |
| solve (GS, sz=N/2) | 28 | cpu | — | 69 | — | 1 | 716 728 |
| spectral (DSSF S^z_Q) | 28 | cpu | — | 108 | — | — | — |
| solve (GS, sz=N/2) | 20 | gpu | 0.04 | 1.86 | — | 1 | 4 654 |
| solve (GS, sz=N/2) | 24 | gpu | — | 4.0 | — | 1 | 56 450 |

## Reading the numbers honestly

1. **The auto guarantee is *scaling*, not universal speedup.** The
   composition caps the largest block at `dim / (|G| · 2)`:
   at N=28 half filling that is 40 M → 717 k (56× smaller), which is
   the difference between a run that fits and one that does not. The
   N=28 rows have no `t_off` because the unsymmetrised lane is
   impractical there — that *is* the result.
2. **At small/medium N a single unsymmetrised shot is often faster.**
   The plain fixed-Sz Lanczos GS at N=20 takes 0.13 s; `auto` pays
   ~0.5 s of automorphism search plus sector construction. Don't use
   `auto` for one small GS; do use it when N is large, when you want
   per-irrep attribution / degeneracy structure, or when you sweep.
3. **Sweeps amortise the construction.** The OrbitTable disk cache
   (`basis_cache/sym_v2/`) makes the sector construction a one-time
   cost across parameter sweeps on the same lattice — the steady-state
   `t_auto` of a 200-point field sweep drops toward the pure solve
   time (see `SYMMETRY_V2_DESIGN.md` §3).
4. **Thermal is the flagship win** (126× at N=12): mTPQ/FTLM cost
   scales with matvec dim per sample, so hundreds of small blocks beat
   one big one — and the spin-flip mirror + TR pairing skip ~half to
   ¾ of the block solves outright. At N≥20 the flat pool becomes
   dominated by per-sector fixed costs (HDF5 writes, thread spin-up);
   that regime is being addressed separately (batch the I/O, not the
   physics).
5. **GPU rows behave the same** — the auto machinery is
   backend-independent (device flip masks, Stage 8b); GPU wins grow
   with block dim, so at these sizes CPU/GPU are comparable.
