# Capability matrix vs dense diagonalization — 2026-07-30 (rev 6)

Regenerated with `benchmarks/bench_capability_matrix.py --n 12 --devices cpu,gpu`
after the 2026-07-30 full-stack audit, on the audited working tree (main @
ccc75b1 + audit fixes). N=12 Heisenberg ring, ground truth = full dense
`eigh`. **All cells PASS** at their expected tolerance (exact lanes <= 1e-11,
stochastic mTPQ/FTLM on the un-composed full-space lanes at the few-%
sampling level, composed lanes machine-exact via the <=512 exact fallback).
Raw numbers: `capability_matrix_N12.json` (session scratchpad copy
`capability_matrix_N12_final.json`).

## What changed in rev 6 (read before comparing to rev 5)

1. **Audit fixes are in the measured tree.** Highlights relevant to the
   matrix surface: `sector=` now selects every extended slot of a named
   momentum (the 4-slot Sz-parity x flip layout silently dropped the
   odd-parity half before — 120/256 dense values missing from per-k
   solves); `flip=` on the abelian lane is a loud refusal instead of a
   silent both-halves merge; SU(2) `total_spin=` targeting no longer
   converges a ghost 0 when the tower minimum is positive (ghost-shift
   operator `P((H-mu)v) + mu v`); `qed.spectral(T=...)` on the plain
   in-memory lane raises instead of silently returning the T=0 / T=inf
   spectrum; SU(2) tower mTPQ reaches the full requested beta range.
2. **NEW: SU(2) total-spin cells** (N=8 ring vs an independent dense
   Kronecker reference, `su2_matrix_cells.py`): solve `total_spin=0`
   targeting (cpu + gpu), `total_spin='auto'` GS labeling, full-spectrum
   values, `thermal(total_spin='auto')` towers E(T)/C(T), and a single
   S=1 tower against a highest-weight-differencing reference — **all 7
   PASS at machine precision** (<= 1e-12).
3. **GPU spectral-lane perf regression fixed** (found by the kill-hash
   wall gates, which had been red since ~Jul-14): the `refine_gs_seed_host`
   guard ran a flat 300-iteration FullCGS2 CPU Lanczos (351 s at N=24)
   — now Ritz-convergence early-exit (45 iters, 11 s); and the Stage-12g
   S^2 labeler's one-shot apply assembled a ~13 GB CSR (48 s) — now
   scoped matrix-free. `qed.spectral` GPU at N=24 (dim 2.7e6): 411 s ->
   under the 30 s gate budget. All kill-hash gates green again.
4. Timing-methodology note: never run this benchmark concurrently with
   the pytest suite — a contended run inflated cells up to 100x and
   briefly looked like a regression.

## Ground state (E0 vs dense)

| composition | device | E0 | |dE0| | blocks | max dim | t [s] |
|---|---|---|---|---|---|---|---|---|
| none | cpu | -5.3873909174 | 7.9e-11 | 1 | 4096 | 0.512 |
| U(1) | cpu | -5.3873909174 | 4.4e-15 | 1 | 4096 | 0.061 |
| spatial | cpu | -5.3873909174 | 5.0e-11 | 1 | 352 | 0.032 |
| U(1)+spatial | cpu | -5.3873909174 | 2.2e-11 | 1 | 80 | 0.024 |
| U(1)+spatial+flip | cpu | -5.3873909174 | 9.5e-12 | 1 | 44 | 0.019 |
| U(1)+spatial+TR | cpu | -5.3873909174 | 3.8e-12 | 1 | 80 | 0.007 |
| U(1)+spatial+star | cpu | -5.3873909174 | 2.5e-14 | 1 | 4096 | 0.005 |
| U(1)+translation+star | cpu | -5.3873909174 | 5.3e-12 | 1 | 80 | 0.014 |
| U(1)+spatial+flip+TR+star | cpu | -5.3873909174 | 2.8e-14 | 1 | 4096 | 0.006 |
| none | gpu | -5.3873909174 | 1.8e-15 | 1 | 4096 | 0.076 |
| U(1) | gpu | -5.3873909174 | 5.3e-15 | 1 | 4096 | 0.044 |
| spatial | gpu | -5.3873909174 | 8.9e-15 | 1 | 352 | 0.375 |
| U(1)+spatial | gpu | -5.3873909174 | 6.2e-15 | 1 | 80 | 0.167 |
| U(1)+spatial+flip | gpu | -5.3873909174 | 2.0e-14 | 1 | 44 | 0.314 |
| U(1)+spatial+TR | gpu | -5.3873909174 | 6.2e-15 | 1 | 80 | 0.096 |
| U(1)+spatial+star | gpu | -5.3873909174 | 1.8e-15 | 1 | 4096 | 0.068 |
| U(1)+translation+star | gpu | -5.3873909174 | 6.2e-15 | 1 | 80 | 0.160 |
| U(1)+spatial+flip+TR+star | gpu | -5.3873909174 | 3.6e-15 | 1 | 4096 | 0.043 |

## Thermal mTPQ (E(T), C(T) vs exact partition sum)

| composition | device | E(0.5) | C(0.5) | E(2.0) | C(2.0) | max rel dev E(T) | blocks | max dim | t [s] |
|---|---|---|---|---|---|---|---|---|---|
| none | cpu | -4.269779 | 3.901213 | -1.254951 | 0.653766 | 1.4e-02 | 1 | 4096 | 0.729 |
| U(1) | cpu | -4.223071 | 3.835763 | -1.251468 | 0.659068 | 1.8e-02 | 13 | 924 | 0.586 |
| U(1)+spatial | cpu | -4.297161 | 4.201155 | -1.246986 | 0.661004 | 1.0e-15 | 134 | 80 | 0.076 |
| U(1)+spatial+flip | cpu | -4.297161 | 4.201155 | -1.246986 | 0.661004 | 1.2e-15 | 146 | 66 | 0.045 |
| U(1)+spatial+TR | cpu | -4.297161 | 4.201155 | -1.246986 | 0.661004 | 1.2e-15 | 134 | 80 | 0.120 |
| U(1)+spatial+star | cpu | -4.297161 | 4.201155 | -1.246986 | 0.661004 | 1.5e-15 | 95 | 78 | 0.193 |
| U(1)+translation+star | cpu | -4.297161 | 4.201155 | -1.246986 | 0.661004 | 1.0e-15 | 134 | 80 | 0.136 |
| U(1)+spatial+flip+TR+star | cpu | -4.297161 | 4.201155 | -1.246986 | 0.661004 | 2.2e-15 | 61 | 66 | 0.090 |
| none | gpu | -4.268786 | 3.913682 | -1.254880 | 0.654563 | 1.4e-02 | 1 | 4096 | 5.385 |
| U(1) | gpu | -4.222302 | 3.843498 | -1.251389 | 0.659515 | 1.8e-02 | 13 | 924 | 20.808 |
| U(1)+spatial | gpu | -4.297161 | 4.201155 | -1.246986 | 0.661004 | 1.0e-15 | 134 | 80 | 0.213 |
| U(1)+spatial+flip | gpu | -4.297161 | 4.201155 | -1.246986 | 0.661004 | 1.2e-15 | 146 | 66 | 0.131 |
| U(1)+spatial+TR | gpu | -4.297161 | 4.201155 | -1.246986 | 0.661004 | 1.2e-15 | 134 | 80 | 0.134 |
| U(1)+spatial+star | gpu | -4.297161 | 4.201155 | -1.246986 | 0.661004 | 1.5e-15 | 95 | 78 | 0.192 |
| U(1)+translation+star | gpu | -4.297161 | 4.201155 | -1.246986 | 0.661004 | 1.0e-15 | 134 | 80 | 0.188 |
| U(1)+spatial+flip+TR+star | gpu | -4.297161 | 4.201155 | -1.246986 | 0.661004 | 2.2e-15 | 61 | 66 | 0.116 |

## Other finite-T methods (fully composed vs none)

| method | composition | device | E(0.5) | C(0.5) | max rel dev E(T) | t [s] |
|---|---|---|---|---|---|---|
| FTLM | none | cpu | -4.306562 | 4.170578 | 1.8e-03 | 2.827 |
| FTLM | U(1)+spatial+flip+TR+star | cpu | -4.297161 | 4.201155 | 2.2e-15 | 0.081 |
| LTLM | none | cpu | -4.306562 | 4.170578 | 1.8e-03 | 12.194 |
| LTLM | U(1)+spatial+flip+TR+star | cpu | -4.297161 | 4.201155 | 2.2e-15 | 0.067 |
| FTLM | none | gpu | -4.301863 | 4.283484 | 1.9e-03 | 2.066 |
| FTLM | U(1)+spatial+flip+TR+star | gpu | -4.297161 | 4.201155 | 2.2e-15 | 0.074 |
| LTLM | none | gpu | -4.301863 | 4.283484 | 1.9e-03 | 4.077 |
| LTLM | U(1)+spatial+flip+TR+star | gpu | -4.297161 | 4.201155 | 2.2e-15 | 0.075 |

## DSSF S^z_{Q=pi}(omega) (vs dense Lehmann sum)

| composition | device | peak S | peak omega | max |dS| | t [s] |
|---|---|---|---|---|---|---|---|
| none | cpu | 2.588236 | 0.370 | 1.9e-13 | 0.794 |
| U(1)+spatial | cpu | 2.588236 | 0.370 | 6.7e-13 | 0.011 |
| none | gpu | 2.588236 | 0.370 | 5.2e-13 | 1.074 |
| U(1)+spatial | gpu | 2.588236 | 0.370 | 6.7e-13 | 0.012 |

## Extended cells: non-abelian full group + U(1)-broken model

| verb | composition | device | E0 | max dev | blocks | max dim | t [s] |
|---|---|---|---|---|---|---|---|
| GS/na | nonabelian-full+U(1) | cpu | -5.38739092 | 2.8e-14 | -- | -- | 0.024 |
| thermal/na | nonabelian-full(exact) | cpu | -5.18022550 | 1.2e-14 | -- | -- | 0.078 |
| GS/na | nonabelian-full+U(1) | gpu | -5.38739092 | 3.6e-15 | -- | -- | 0.551 |
| thermal/na | nonabelian-full(exact) | gpu | -5.18022550 | 1.2e-14 | -- | -- | 0.080 |
| DSSF/na | nonabelian-full | cpu | -- | 4.1e-13 | -- | -- | 0.095 |
| DSSF/na | nonabelian-full | gpu | -- | 4.3e-13 | -- | -- | 0.598 |
| GS[U(1)-broken] | none | cpu | -5.72132542 | 5.3e-15 | 1 | 4096 | 1.835 |
| GS[U(1)-broken] | spatial | cpu | -5.72132542 | 8.2e-14 | 12 | 352 | 0.708 |
| GS[U(1)-broken] | spatial+flipfull | cpu | -5.72132542 | 7.7e-14 | 24 | 180 | 0.127 |
| GS[U(1)-broken] | parity+spatial | cpu | -5.72132542 | 7.6e-14 | 24 | 180 | 0.107 |
| GS[U(1)-broken] | parity+spatial+flip | cpu | -5.72132542 | 7.2e-14 | 48 | 94 | 0.079 |
| GS[U(1)-broken] | parity+flip+TR+star | cpu | -5.72132542 | 3.4e-14 | 1 | 4096 | 0.092 |
| GS[U(1)-broken] | nonabelian-full+parity | cpu | -5.72132542 | 3.4e-14 | -- | -- | 0.040 |
| thermal[U(1)-broken] | parity(auto)+flip | cpu | -5.60894600 | 5.3e-15 | 18 | 176 | 0.059 |
| fulldense[U(1)-broken] | parity(auto)+flip | cpu | -5.72132542 | 8.3e-14 | -- | -- | 0.030 |
| GS[U(1)-broken] | none | gpu | -5.72132542 | 5.3e-15 | 1 | 4096 | 1.799 |
| GS[U(1)-broken] | spatial | gpu | -5.72132542 | 8.2e-14 | 12 | 352 | 0.467 |
| GS[U(1)-broken] | spatial+flipfull | gpu | -5.72132542 | 7.7e-14 | 24 | 180 | 0.222 |
| GS[U(1)-broken] | parity+spatial | gpu | -5.72132542 | 7.6e-14 | 24 | 180 | 0.216 |
| GS[U(1)-broken] | parity+spatial+flip | gpu | -5.72132542 | 7.2e-14 | 48 | 94 | 0.207 |
| GS[U(1)-broken] | parity+flip+TR+star | gpu | -5.72132542 | 3.6e-15 | 1 | 4096 | 0.309 |
| GS[U(1)-broken] | nonabelian-full+parity | gpu | -5.72132542 | 3.6e-15 | -- | -- | 0.118 |
| thermal[U(1)-broken] | parity(auto)+flip | gpu | -5.60894600 | 5.3e-15 | 18 | 176 | 0.084 |
| fulldense[U(1)-broken] | parity(auto)+flip | gpu | -5.72132542 | 8.3e-14 | -- | -- | 0.019 |
