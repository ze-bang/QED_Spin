# Capability matrix vs dense diagonalization — 2026-07-20 (rev 5)

Regenerated with `benchmarks/bench_capability_matrix.py --n 12 --devices cpu,gpu`
on the post-consolidation stack (little-group engine sole non-abelian engine,
qn/U-series routing). N=12 Heisenberg ring, **ground truth = full dense
`eigh`** (exact E₀, exact partition sum, exact Lehmann sum, η/π convention).
Raw numbers: `capability_matrix_N12.json`. All cells PASS at their expected
tolerance (exact lanes ≤ 1e-11, stochastic mTPQ/FTLM on the un-composed
full-space lanes at the few-% sampling level, composed lanes machine-exact via
the ≤512 exact fallback).

## What changed in rev 5 (read before comparing to rev 4)

1. **The non-abelian little-group engine is genuinely GPU-capable
   again.** Consolidation Family 6 removed the monolithic SAB engine
   and silently took its batched cuSOLVER block eigensolve down with
   it, turning `LittleGroupOptions::use_gpu` into a no-op; a later
   U-series revision then removed the routing veto on the (by-then
   false) claim that the project lane was still GPU end-to-end. Fixed
   at the root: the kernel is recovered and re-homed SAB-free as
   `little_group_gpu.{h,cu}` (`lg_blocks_batched_eigenvalues_gpu` —
   one packed upload, an 8-stream `cusolverDnZheevd` pool, one
   download), wired into BOTH the full-spectrum path and a new
   deferred-batch lane on the lowest-k path (dense-crossover blocks
   batch on the GPU; Lanczos-sized blocks solve inline, engaging the
   rep-gather at ≥2²⁰ reps). `point_group='auto'` and `point_group=
   'full'` now both run their non-abelian blocks ON the device under
   `device='gpu'`, and the strict per-row GPU-lane assertion holds for
   every GS/thermal/full_spectrum row below — including the GS/na and
   thermal/na non-abelian cells and the U(1)-broken model's
   nonabelian-full+parity cell, all new GPU rows this rev. GS-DSSF is a
   GPU lane too: under `device='gpu'` every receiving sector's
   continued-fraction matvec runs the forced device rep-gather
   (dimension floor dropped, reduced CSR demoted to fallback) and the
   GS-subspace scan batches its eigensolves on the device, with a
   truthful `gpu_engaged` report — the DSSF/na row below is asserted on
   it.
2. **cTPQ rows are gone for good**: the user-facing cTPQ method was removed
   in the final consolidation (commit 9843ebb); `qed.thermal` supports
   FTLM / LTLM / OFTLM / mTPQ / KPM-DOS.

## Supplementary cells (N=10 ring, dense-verified, 2026-07-20)

Beyond the standard sweep, the following previously-unswept cells were
probed against dense `eigh` (`benchmarks/audit_capability_gaps.py`):

| cell | result |
|---|---|
| GS little-group, `ED_SYM_LG_GPU=1` | PASS, dE0 = 0.0 (reduced CSR served the blocks — the GPU rep-gather is reachable only when the CSR declines, i.e. the ≥2^20-rep regime; not exercisable at N=10) |
| thermal little-group, `ED_SYM_LG_GPU=1` | PASS, max\|dE(T)\| = 5.3e-15 |
| `full_spectrum` symmetry, `device='gpu'` | PASS, max\|dE\| = 8.4e-15 over all 1024 values |
| `full_spectrum` `point_group='full'`, `device='gpu'` | PASS, 8.4e-15 over all 1024 values |
| DSSF `spin_flip='require'` (cpu + gpu) | PASS, max\|dS\| = 1.4e-13 — Stage 8d flip composition confirmed CONSUMED |
| DSSF `time_reversal='require'` | loud `NotImplementedError` BY DESIGN — the spectral verb does not exploit TR (no k↔−k fold); solve/thermal/full_spectrum do |
| DSSF `point_group='full'` (little-group) | PASS, max\|dS\| = 2.5e-14 |
| DSSF finite-T (T=1.0, FTLM, cpu + gpu) | PASS, max rel dev 6.9e-2, integral ratio 1.013 (stochastic regime) |
| thermal `method='cTPQ'` | removed by design — loud ValueError listing supported methods |

## Ground state (E0 vs dense)

| composition | device | lane | E0 | \|dE0\| | blocks | max dim | t [s] |
|---|---|---|---|---|---|---|---|
| none | cpu | cpu | -5.3873909174 | 3.2e-11 | 1 | 4096 | 0.180 |
| U(1) | cpu | cpu | -5.3873909174 | 4.4e-15 | 1 | 4096 | 0.059 |
| spatial | cpu | cpu | -5.3873909174 | 1.3e-11 | 1 | 352 | 0.047 |
| U(1)+spatial | cpu | cpu | -5.3873909174 | 1.5e-11 | 1 | 80 | 0.027 |
| U(1)+spatial+flip | cpu | cpu | -5.3873909174 | 5.3e-13 | 1 | 44 | 0.023 |
| U(1)+spatial+TR | cpu | cpu | -5.3873909174 | 1.2e-11 | 1 | 80 | 0.019 |
| U(1)+spatial+star | cpu | cpu | -5.3873909174 | 1.8e-14 | 1 | 4096 | 0.019 |
| U(1)+translation+star | cpu | cpu | -5.3873909174 | 5.3e-12 | 1 | 80 | 0.023 |
| U(1)+spatial+flip+TR+star | cpu | cpu | -5.3873909174 | 2.2e-14 | 1 | 4096 | 0.025 |
| none | gpu | gpu | -5.3873909174 | 1.8e-15 | 1 | 4096 | 0.063 |
| U(1) | gpu | gpu | -5.3873909174 | 5.3e-15 | 1 | 4096 | 0.040 |
| spatial | gpu | gpu | -5.3873909174 | 8.9e-15 | 1 | 352 | 0.279 |
| U(1)+spatial | gpu | gpu | -5.3873909174 | 6.2e-15 | 1 | 80 | 0.145 |
| U(1)+spatial+flip | gpu | gpu | -5.3873909174 | 2.0e-14 | 1 | 44 | 0.273 |
| U(1)+spatial+TR | gpu | gpu | -5.3873909174 | 6.2e-15 | 1 | 80 | 0.085 |
| U(1)+spatial+star | gpu | gpu | -5.3873909174 | 2.7e-15 | 1 | 4096 | 0.066 |
| U(1)+translation+star | gpu | gpu | -5.3873909174 | 6.2e-15 | 1 | 80 | 0.142 |
| U(1)+spatial+flip+TR+star | gpu | gpu | -5.3873909174 | 1.8e-15 | 1 | 4096 | 0.038 |

## Thermal mTPQ (E(T), C(T) vs exact partition sum)

| composition | device | E(0.5) | C(0.5) | max rel dev E(T) | blocks | max dim | t [s] |
|---|---|---|---|---|---|---|---|
| none | cpu | -4.269779 | 3.901213 | 1.4e-02 | 1 | 4096 | 0.826 |
| U(1) | cpu | -4.223071 | 3.835763 | 1.8e-02 | 13 | 924 | 1.203 |
| U(1)+spatial | cpu | -4.297161 | 4.201155 | 1.0e-15 | 134 | 80 | 0.413 |
| U(1)+spatial+flip | cpu | -4.297161 | 4.201155 | 1.2e-15 | 146 | 66 | 0.332 |
| U(1)+spatial+TR | cpu | -4.297161 | 4.201155 | 1.2e-15 | 134 | 80 | 0.662 |
| U(1)+spatial+star | cpu | -4.297161 | 4.201155 | 6.0e-16 | 95 | 78 | 0.428 |
| U(1)+translation+star | cpu | -4.297161 | 4.201155 | 1.0e-15 | 134 | 80 | 0.381 |
| U(1)+spatial+flip+TR+star | cpu | -4.297161 | 4.201155 | 2.7e-15 | 61 | 66 | 0.314 |
| none | gpu | -4.268786 | 3.913682 | 1.4e-02 | 1 | 4096 | 4.402 |
| U(1) | gpu | -4.222302 | 3.843498 | 1.8e-02 | 13 | 924 | 15.656 |
| U(1)+spatial | gpu | -4.297161 | 4.201155 | 1.0e-15 | 134 | 80 | 0.207 |
| U(1)+spatial+flip | gpu | -4.297161 | 4.201155 | 1.2e-15 | 146 | 66 | 0.128 |
| U(1)+spatial+TR | gpu | -4.297161 | 4.201155 | 1.2e-15 | 134 | 80 | 0.125 |
| U(1)+spatial+star | gpu | -4.297161 | 4.201155 | 6.0e-16 | 95 | 78 | 0.175 |
| U(1)+translation+star | gpu | -4.297161 | 4.201155 | 1.0e-15 | 134 | 80 | 0.206 |
| U(1)+spatial+flip+TR+star | gpu | -4.297161 | 4.201155 | 2.7e-15 | 61 | 66 | 0.112 |

## Other finite-T methods (fully composed vs none)

| method | composition | device | E(0.5) | C(0.5) | max rel dev E(T) | t [s] |
|---|---|---|---|---|---|---|
| FTLM | none | cpu | -4.306562 | 4.170578 | 1.8e-03 | 4.634 |
| FTLM | U(1)+spatial+flip+TR+star | cpu | -4.297161 | 4.201155 | 2.7e-15 | 0.295 |
| LTLM | none | cpu | -4.306562 | 4.170578 | 1.8e-03 | 13.273 |
| LTLM | U(1)+spatial+flip+TR+star | cpu | -4.297161 | 4.201155 | 2.7e-15 | 0.070 |
| FTLM | none | gpu | -4.301863 | 4.283484 | 1.9e-03 | 1.531 |
| FTLM | U(1)+spatial+flip+TR+star | gpu | -4.297161 | 4.201155 | 2.7e-15 | 0.061 |
| LTLM | none | gpu | -4.301863 | 4.283484 | 1.9e-03 | 2.913 |
| LTLM | U(1)+spatial+flip+TR+star | gpu | -4.297161 | 4.201155 | 2.7e-15 | 0.075 |

## DSSF S^z_{Q=pi}(omega) (vs dense Lehmann sum)

| composition | device | peak S | peak omega | max \|dS\| | t [s] |
|---|---|---|---|---|---|
| none | cpu | 2.588236 | 0.370 | 1.9e-13 | 0.765 |
| U(1)+spatial | cpu | 2.588236 | 0.370 | 6.7e-13 | 0.012 |
| none | gpu | 2.588236 | 0.370 | 5.2e-13 | 0.965 |
| U(1)+spatial | gpu | 2.588236 | 0.370 | 6.7e-13 | 0.015 |

## Extended cells: non-abelian full group + U(1)-broken model

| verb | composition | device | E0 | max dev | blocks | max dim | t [s] |
|---|---|---|---|---|---|---|---|
| GS/na | nonabelian-full+U(1) | cpu | -5.38739092 | 2.2e-14 | -- | -- | 0.023 |
| thermal/na | nonabelian-full(exact) | cpu | -5.18022550 | 1.4e-14 | -- | -- | 0.069 |
| GS/na | nonabelian-full+U(1) | gpu | -5.38739092 | 1.8e-15 | -- | -- | 0.457 |
| thermal/na | nonabelian-full(exact) | gpu | -5.18022550 | 1.4e-14 | -- | -- | 0.070 |
| DSSF/na | nonabelian-full | cpu | -- | 2.7e-13 | -- | -- | 0.089 |
| DSSF/na | nonabelian-full | gpu | -- | 2.8e-13 | -- | -- | 0.557 |
| GS[U(1)-broken] | none | cpu | -5.72132542 | 5.3e-15 | 1 | 4096 | 1.685 |
| GS[U(1)-broken] | spatial | cpu | -5.72132542 | 8.2e-14 | 12 | 352 | 0.254 |
| GS[U(1)-broken] | spatial+flipfull | cpu | -5.72132542 | 7.7e-14 | 24 | 180 | 0.115 |
| GS[U(1)-broken] | parity+spatial | cpu | -5.72132542 | 7.6e-14 | 24 | 180 | 0.123 |
| GS[U(1)-broken] | parity+spatial+flip | cpu | -5.72132542 | 7.2e-14 | 48 | 94 | 0.096 |
| GS[U(1)-broken] | parity+flip+TR+star | cpu | -5.72132542 | 1.3e-14 | 1 | 4096 | 0.087 |
| GS[U(1)-broken] | nonabelian-full+parity | cpu | -5.72132542 | 1.3e-14 | -- | -- | 0.031 |
| thermal[U(1)-broken] | parity(auto)+flip | cpu | -5.60894600 | 4.9e-15 | 18 | 176 | 0.063 |
| fulldense[U(1)-broken] | parity(auto)+flip | cpu | -5.72132542 | 8.3e-14 | -- | -- | 0.025 |
| GS[U(1)-broken] | none | gpu | -5.72132542 | 5.3e-15 | 1 | 4096 | 1.824 |
| GS[U(1)-broken] | spatial | gpu | -5.72132542 | 8.2e-14 | 12 | 352 | 0.280 |
| GS[U(1)-broken] | spatial+flipfull | gpu | -5.72132542 | 7.7e-14 | 24 | 180 | 0.244 |
| GS[U(1)-broken] | parity+spatial | gpu | -5.72132542 | 7.6e-14 | 24 | 180 | 0.191 |
| GS[U(1)-broken] | parity+spatial+flip | gpu | -5.72132542 | 7.2e-14 | 48 | 94 | 0.209 |
| GS[U(1)-broken] | parity+flip+TR+star | gpu | -5.72132542 | 7.1e-15 | 1 | 4096 | 0.279 |
| GS[U(1)-broken] | nonabelian-full+parity | gpu | -5.72132542 | 7.1e-15 | -- | -- | 0.101 |
| thermal[U(1)-broken] | parity(auto)+flip | gpu | -5.60894600 | 4.9e-15 | 18 | 176 | 0.073 |
| fulldense[U(1)-broken] | parity(auto)+flip | gpu | -5.72132542 | 8.3e-14 | -- | -- | 0.013 |
