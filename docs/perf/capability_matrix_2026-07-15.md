# Capability matrix vs dense diagonalization — 2026-07-15 (rev 4)

> Superseded by
> [`capability_matrix_2026-07-20.md`](capability_matrix_2026-07-20.md) (rev 5,
> post-consolidation stack + supplementary little-group-GPU / composed-DSSF /
> finite-T-DSSF cells).

Regenerated with `benchmarks/bench_capability_matrix.py --n 12 --devices cpu,gpu`.
N=12 Heisenberg ring (all four symmetry axes present). **Ground truth = full
dense `eigh` of the 4096-dim Hamiltonian**: exact E₀, exact partition sum for
E(T)/C(T), exact Lehmann sum for S^z_{Q=π}(ω) (η/π convention verified, ratio
≡ 1). Raw numbers: `capability_matrix_N12.json`.

Row hygiene: every composition is **pinned** (`spin_flip=` / `time_reversal=`
`"require"`/`"off"`, `point_group=` `"auto"`/`"off"`) so no silent fallback can
fake a label, and **every streaming-symmetry GPU row asserts the truthful
backend lane** (`result.backend.lane == "gpu"`) — a GPU row that ran on the CPU
fails the benchmark.

## What changed in rev 4 (read this before comparing to rev 3)

Two corrections land here; both change published numbers, and neither is a
measurement fluctuation.

1. **The LTLM rows in rev 3 were a BUG, not a method regime.** Rev 3 reported
   LTLM at `9.5e−01` / `4.6e−01` and explained it as "a method-regime property,
   reported as-is", citing agreement between the CPU and GPU lanes as evidence
   it was not a composition artifact. Both lanes agreed because both
   reimplemented the *same* defect: each summed the ground-state local DOS
   (`Σ|⟨0|ψₙ⟩|²e^(−βEₙ)`) instead of the thermal trace, so LTLM sat at E₀ at
   every temperature. **Twin agreement is not ground truth when the twins share
   lineage.** Fixed in 654ea06: for any function of H the symmetric LTLM
   estimator reduces exactly to the FTLM trace, so LTLM thermodynamics now
   dispatches through `ftlm_kernel`. Its rows below are FTLM's, to the digit.

2. **Composed finite-T rows are now machine-exact.** The `D ≤ 512` exact-thermal
   fallback used to be gated on mTPQ alone; it now covers every sampling method
   whose deliverable is thermodynamics (mTPQ / FTLM / LTLM / OFTLM). Composed
   lanes drive block dims to ≤ 80 here, so FTLM/LTLM take the exact route:
   `8.6e−03` → `1.1e−15`, and *faster* (0.60 s → 0.024 s). KpmDos is excluded on
   purpose — its Chebyshev density of states is a deliverable the exact path
   does not produce.

Also removed: rev 3's "Complete spectrum (dense)" section, whose rows were
labelled *"SAB (full D₁₂, d≥2 irreps)"*. The monolithic symmetry-adapted engine
had no production route after Stage 9c and was deleted outright in the
consolidation sweep; the non-abelian route is the factorized little-group engine
(`little_group_*`), covered by the `*/na` rows below. The section had no
generator left in the benchmark and is not reproduced.

## Ground state — E₀(dense) = −5.3873909174

| composition | device | E0 | |dE0| | blocks | max dim | t [s] |
|---|---|---|---|---|---|---|
| none | cpu | -5.3873909174 | 3.2e-11 | 1 | 4096 | 0.497 |
| U(1) | cpu | -5.3873909174 | 4.4e-15 | 1 | 4096 | 0.234 |
| spatial | cpu | -5.3873909174 | 3.3e-11 | 1 | 352 | 0.047 |
| U(1)+spatial | cpu | -5.3873909174 | 8.4e-12 | 1 | 80 | 0.018 |
| U(1)+spatial+flip | cpu | -5.3873909174 | 1.5e-11 | 1 | 44 | 0.014 |
| U(1)+spatial+TR | cpu | -5.3873909174 | 2.2e-11 | 1 | 80 | 0.007 |
| U(1)+spatial+star | cpu | -5.3873909174 | 1.8e-14 | 1 | 4096 | 0.005 |
| U(1)+translation+star | cpu | -5.3873909174 | 1.1e-11 | 1 | 80 | 0.014 |
| U(1)+spatial+flip+TR+star | cpu | -5.3873909174 | 2.2e-14 | 1 | 4096 | 0.005 |
| none | gpu | -5.3873909174 | 1.8e-15 | 1 | 4096 | 0.083 |
| U(1) | gpu | -5.3873909174 | 5.3e-15 | 1 | 4096 | 0.035 |
| spatial | gpu | -5.3873909174 | 8.9e-15 | 1 | 352 | 0.286 |
| U(1)+spatial | gpu | -5.3873909174 | 6.2e-15 | 1 | 80 | 0.171 |
| U(1)+spatial+flip | gpu | -5.3873909174 | 2.0e-14 | 1 | 44 | 0.335 |
| U(1)+spatial+TR | gpu | -5.3873909174 | 6.2e-15 | 1 | 80 | 0.107 |
| U(1)+spatial+star | gpu | -5.3873909174 | 1.8e-14 | 1 | 4096 | 0.006 |
| U(1)+translation+star | gpu | -5.3873909174 | 6.2e-15 | 1 | 80 | 0.173 |
| U(1)+spatial+flip+TR+star | gpu | -5.3873909174 | 2.2e-14 | 1 | 4096 | 0.006 |

Block-dim ladder: 4096 → 352 (spatial ÷11.6) → 80 (×U(1) ÷51) → 44 (×flip ÷93).

Reading the `star` rows: they report `blocks=1, max dim=4096` because
`point_group="auto"` now **projects** through the little-group engine (Stage 9c)
for eigenvalue-only solves, and that lane returns pooled eigenvalues without
per-block metadata — 4096 is "no block info", not a real block. Rev 3's `80` for
these rows came from the abelian lane that `auto` used to take. The physics is
unchanged (dE₀ ≤ 2.2e−14) and the solve is the fastest on the board (0.005 s).

GPU rows are 6–19× **slower** than CPU at these block dims (0.335 s vs 0.014 s
for the 44-dim blocks) — kernel-launch overhead dominating tiny sectors. This is
exactly what `BackendConstraints::gpu_dim_floor = 2^14` exists to prevent: the
GPU is not auto-promoted here, and these rows only run on the device because the
benchmark asks explicitly.

## Thermal mTPQ — exact E(0.5) = −4.297161, C(0.5) = 4.201155

| composition | device | E(0.5) | C(0.5) | E(2.0) | C(2.0) | max rel dev E(T) | blocks | max dim | t [s] |
|---|---|---|---|---|---|---|---|---|---|
| none | cpu | -4.269779 | 3.901213 | -1.254951 | 0.653766 | 1.4e-02 | 1 | 4096 | 0.687 |
| U(1) | cpu | -4.223071 | 3.835763 | -1.251468 | 0.659068 | 1.8e-02 | 13 | 924 | 0.747 |
| U(1)+spatial | cpu | -4.297161 | 4.201155 | -1.246986 | 0.661004 | 1.0e-15 | 134 | 80 | 0.075 |
| U(1)+spatial+flip | cpu | -4.297161 | 4.201155 | -1.246986 | 0.661004 | 1.2e-15 | 146 | 66 | 0.097 |
| U(1)+spatial+TR | cpu | -4.297161 | 4.201155 | -1.246986 | 0.661004 | 1.2e-15 | 134 | 80 | 0.115 |
| U(1)+spatial+star | cpu | -4.297161 | 4.201155 | -1.246986 | 0.661004 | 1.2e-15 | 134 | 80 | 0.122 |
| U(1)+translation+star | cpu | -4.297161 | 4.201155 | -1.246986 | 0.661004 | 1.0e-15 | 134 | 80 | 0.157 |
| U(1)+spatial+flip+TR+star | cpu | -4.297161 | 4.201155 | -1.246986 | 0.661004 | 1.1e-15 | 146 | 66 | 0.092 |
| none | gpu | -4.268786 | 3.913682 | -1.254880 | 0.654563 | 1.4e-02 | 1 | 4096 | 5.375 |
| U(1) | gpu | -4.222302 | 3.843498 | -1.251389 | 0.659515 | 1.8e-02 | 13 | 924 | 19.386 |
| U(1)+spatial | gpu | -4.297161 | 4.201155 | -1.246986 | 0.661004 | 1.0e-15 | 134 | 80 | 0.169 |
| U(1)+spatial+flip | gpu | -4.297161 | 4.201155 | -1.246986 | 0.661004 | 1.2e-15 | 146 | 66 | 0.156 |
| U(1)+spatial+TR | gpu | -4.297161 | 4.201155 | -1.246986 | 0.661004 | 1.2e-15 | 134 | 80 | 0.142 |
| U(1)+spatial+star | gpu | -4.297161 | 4.201155 | -1.246986 | 0.661004 | 1.2e-15 | 134 | 80 | 0.159 |
| U(1)+translation+star | gpu | -4.297161 | 4.201155 | -1.246986 | 0.661004 | 1.0e-15 | 134 | 80 | 0.225 |
| U(1)+spatial+flip+TR+star | gpu | -4.297161 | 4.201155 | -1.246986 | 0.661004 | 1.1e-15 | 146 | 66 | 0.111 |

Composed lanes are machine-exact (1e−15: every block ≤ 512 hits the exact
fallback) and ~9× faster than the unsymmetrised stochastic run, whose 1–2 % rows
are honest mTPQ sampling error. Flip splits the pool 134 → 146 blocks (max dim
80 → 66); TR and star drop solves without changing any number.

## Other finite-T methods (fully composed vs none)

| method | composition | device | E(0.5) | C(0.5) | max rel dev E(T) | t [s] |
|---|---|---|---|---|---|---|
| FTLM | none | cpu | -4.306562 | 4.170578 | 1.8e-03 | 2.760 |
| FTLM | U(1)+spatial+flip+TR+star | cpu | -4.297161 | 4.201155 | 1.1e-15 | 0.024 |
| LTLM | none | cpu | -4.306562 | 4.170578 | 1.8e-03 | 13.503 |
| LTLM | U(1)+spatial+flip+TR+star | cpu | -4.297161 | 4.201155 | 1.1e-15 | 0.031 |
| FTLM | none | gpu | -4.301863 | 4.283484 | 1.9e-03 | 1.877 |
| FTLM | U(1)+spatial+flip+TR+star | gpu | -4.297161 | 4.201155 | 1.1e-15 | 0.065 |
| LTLM | none | gpu | -4.301863 | 4.283484 | 1.9e-03 | 3.676 |
| LTLM | U(1)+spatial+flip+TR+star | gpu | -4.297161 | 4.201155 | 1.1e-15 | 0.059 |

LTLM ≡ FTLM to the digit, on both devices and both compositions — that identity
IS the contract (see rev-4 note 1), and it is pinned in
`test_thermal_dense_ref` ("LTLM thermodynamics IS the FTLM trace", 1e−12).

**Open perf item:** LTLM|none costs 13.5 s against FTLM's 2.76 s for a
bit-identical answer. `kLtlmKrylovDim` defaults to 200 where `kFtlmKrylovDim` is
100; with full reorthogonalisation the pass is ~O(m²), so ≈4× — matching the
measured 4.9×, while the trace estimate has already converged by m=100 (hence
identical digits). The 200 default is a leftover from when LTLM was a distinct
estimator; on the thermal verb it now buys nothing. Not changed here — it is a
knob with a physics history, and `compute_connected_qh_response_ltlm` (the
genuinely LTLM-only dM/dT estimator, which does not commute with H) may still
want the larger space.

## DSSF S^z_{Q=π}(ω) — dense Lehmann peak 2.588236 at ω = 0.370

| composition | device | peak S | peak omega | max |dS| | t [s] |
|---|---|---|---|---|---|
| none | cpu | 2.588236 | 0.370 | 1.9e-13 | 1.518 |
| U(1)+spatial | cpu | 2.588236 | 0.370 | 6.7e-13 | 0.011 |
| none | gpu | 2.588236 | 0.370 | 5.2e-13 | 1.322 |
| U(1)+spatial | gpu | 2.588236 | 0.370 | 6.7e-13 | 0.012 |

The sector route is ~2 orders tighter than the full-Hilbert continued fraction
and ~130× faster. Flip/TR/star are not yet consumed by the spectral lanes;
`"require"` still validates the Hamiltonian.

## Extended cells: non-abelian full group + the U(1)-broken model

The U(1)-broken model (Heisenberg ring + J±±-type S⁺S⁺/S⁻S⁻ terms, N=12, dense
E₀ = −5.72132542) exercises the mechanisms a U(1)-conserving Hamiltonian cannot:
**Sz parity** (the (−1)^{n_up} Z₂ remnant), **full-space ∏σˣ sectors**, and
their composition.

| verb | composition | device | E0 | max dev | blocks | max dim | t [s] |
|---|---|---|---|---|---|---|---|
| GS/na | nonabelian-full+U(1) | cpu | -5.38739092 | 2.2e-14 | -- | -- | 0.006 |
| thermal/na | nonabelian-full(exact) | cpu | -5.18022550 | 1.6e-13 | -- | -- | 0.016 |
| DSSF/na | nonabelian-full | cpu | -- | 2.6e-13 | -- | -- | 0.049 |
| GS[U(1)-broken] | none | cpu | -5.72132542 | 5.3e-15 | 1 | 4096 | 3.541 |
| GS[U(1)-broken] | spatial | cpu | -5.72132542 | 8.2e-14 | 12 | 352 | 0.909 |
| GS[U(1)-broken] | spatial+flipfull | cpu | -5.72132542 | 7.7e-14 | 24 | 180 | 0.211 |
| GS[U(1)-broken] | parity+spatial | cpu | -5.72132542 | 7.6e-14 | 24 | 180 | 0.307 |
| GS[U(1)-broken] | parity+spatial+flip | cpu | -5.72132542 | 7.2e-14 | 48 | 94 | 0.086 |
| GS[U(1)-broken] | parity+flip+TR+star | cpu | -5.72132542 | 1.3e-14 | 1 | 4096 | 0.084 |
| GS[U(1)-broken] | nonabelian-full+parity | cpu | -5.72132542 | 1.3e-14 | -- | -- | 0.033 |
| thermal[U(1)-broken] | parity(auto)+flip | cpu | -5.60894600 | 1.3e-14 | -- | -- | 0.364 |
| fulldense[U(1)-broken] | parity(auto)+flip | cpu | -5.72132542 | 8.3e-14 | -- | -- | 0.043 |
| GS[U(1)-broken] | none | gpu | -5.72132542 | 5.3e-15 | 1 | 4096 | 3.569 |
| GS[U(1)-broken] | spatial | gpu | -5.72132542 | 8.2e-14 | 12 | 352 | 0.974 |
| GS[U(1)-broken] | spatial+flipfull | gpu | -5.72132542 | 7.7e-14 | 24 | 180 | 0.699 |
| GS[U(1)-broken] | parity+spatial | gpu | -5.72132542 | 7.6e-14 | 24 | 180 | 0.403 |
| GS[U(1)-broken] | parity+spatial+flip | gpu | -5.72132542 | 7.2e-14 | 48 | 94 | 0.229 |
| GS[U(1)-broken] | parity+flip+TR+star | gpu | -5.72132542 | 1.3e-14 | 1 | 4096 | 0.075 |
| thermal[U(1)-broken] | parity(auto)+flip | gpu | -5.60894600 | 1.3e-14 | -- | -- | 0.548 |
| fulldense[U(1)-broken] | parity(auto)+flip | gpu | -5.72132542 | 8.3e-14 | -- | -- | 0.033 |

`*/na` rows are `point_group="full"` on the factorized little-group engine
(D₁₂, d≥2 irreps), composed with the diagonal axis automatically: GS at 0.006 s
(2.2e−14), exact thermodynamics 1.6e−13, GS-DSSF vs the dense Lehmann sum
2.6e−13. Ladder on the broken model: each Z₂ multiplies — spatial ÷12, ×∏σˣ ÷2,
×parity ÷2 (48 blocks, max dim 94 vs 4096). The `parity+flip+TR+star` row shows
`blocks=1, max dim=4096` for the same projection-lane reporting reason as the
star rows above.

## Notes

* Every cell is measured against dense diagonalization, never against another
  QED lane. That rule is not pedantry: rev 3's LTLM error came from trusting a
  CPU-vs-GPU comparison between two implementations that shared a bug.
* Timings are one WSL2 box (RTX-class GPU, shared with the desktop session) and
  are indicative, not a benchmark suite. The accuracy columns are the contract.
* Regenerate with:
  `python benchmarks/bench_capability_matrix.py --n 12 --devices cpu,gpu --json docs/perf/capability_matrix_N12.json`
