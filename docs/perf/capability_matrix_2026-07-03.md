# Capability matrix vs dense diagonalization — 2026-07-03 (rev 2)

`benchmarks/bench_capability_matrix.py --n 12 --devices cpu,gpu`.
N=12 Heisenberg ring (all four symmetry axes present). **Ground truth
= full dense `eigh` of the 4096-dim Hamiltonian**: exact E₀, exact
partition sum for E(T)/C(T), exact Lehmann sum for S^z_{Q=π}(ω)
(η/π convention verified, ratio ≡ 1). Raw numbers:
`capability_matrix_N12.json`.

Row hygiene: every composition is **pinned** (`spin_flip=` /
`time_reversal=` `"require"`/`"off"`, `point_group=` `"auto"`/`"off"`)
so no silent fallback can fake a label, and **every streaming-symmetry
GPU row asserts the truthful backend lane** (`result.backend.lane ==
"gpu"`) — a GPU row that ran on the CPU fails the benchmark. (The
legacy plain-lane result type carries no lane metadata; those two rows
are verified by the orchestrator's own GPU dispatch tests.)

New in rev 2: **star reduction** rows (`point_group="auto"` — the
non-abelian point-group residue folds the momentum sectors into
isospectral stars; solve one per star, copy the spectrum) and
**`symmetry="translation"`** rows (pure-translation projector via
`lattice=`, whole point group retained as stars).

## Ground state — E₀(dense) = −5.3873909174

| composition | device | E0 | |dE0| | blocks | max dim | t [s] |
|---|---|---|---|---|---|---|
| none | cpu | -5.3873909174 | 8.2e-11 | 1 | 4096 | 0.459 |
| U(1) | cpu | -5.3873909174 | 4.4e-15 | 1 | 4096 | 0.383 |
| spatial | cpu | -5.3873909174 | 5.7e-11 | 1 | 352 | 0.127 |
| U(1)+spatial | cpu | -5.3873909174 | 7.5e-11 | 1 | 80 | 0.037 |
| U(1)+spatial+flip | cpu | -5.3873909174 | 9.2e-12 | 1 | 44 | 0.017 |
| U(1)+spatial+TR | cpu | -5.3873909174 | 3.3e-11 | 1 | 80 | 0.011 |
| U(1)+spatial+star | cpu | -5.3873909174 | 1.6e-11 | 1 | 80 | 0.011 |
| U(1)+translation+star | cpu | -5.3873909173 | 1.1e-10 | 1 | 80 | 0.017 |
| U(1)+spatial+flip+TR+star | cpu | -5.3873909174 | 3.6e-12 | 1 | 44 | 0.010 |
| none | gpu | -5.3873909174 | 1.8e-15 | 1 | 4096 | 0.017 |
| U(1) | gpu | -5.3873909174 | 5.3e-15 | 1 | 4096 | 0.017 |
| spatial | gpu | -5.3873909174 | 1.3e-14 | 1 | 352 | 0.316 |
| U(1)+spatial | gpu | -5.3873909174 | 6.2e-15 | 1 | 80 | 0.182 |
| U(1)+spatial+flip | gpu | -5.3873909174 | 2.0e-14 | 1 | 44 | 0.325 |
| U(1)+spatial+TR | gpu | -5.3873909174 | 6.2e-15 | 1 | 80 | 0.111 |
| U(1)+spatial+star | gpu | -5.3873909174 | 6.2e-15 | 1 | 80 | 0.104 |
| U(1)+translation+star | gpu | -5.3873909174 | 6.2e-15 | 1 | 80 | 0.177 |
| U(1)+spatial+flip+TR+star | gpu | -5.3873909174 | 2.0e-14 | 1 | 44 | 0.189 |

Block-dim ladder: 4096 → 352 (spatial ÷11.6) → 80 (×U(1) ÷51) →
44 (×flip ÷93). Star/translation rows keep the 80-dim blocks (star
reduction cuts the *number of solves* — 7 of 12 sectors on the D₁₂
ring — not the block size).

## Thermal mTPQ — exact E(0.5) = −4.297161, C(0.5) = 4.201155

| composition | device | E(0.5) | C(0.5) | E(2.0) | C(2.0) | max rel dev E(T) | blocks | max dim | t [s] |
|---|---|---|---|---|---|---|---|---|---|
| none | cpu | -4.269779 | 3.901213 | -1.254951 | 0.653766 | 1.4e-02 | 1 | 4096 | 4.335 |
| U(1) | cpu | -4.223071 | 3.835763 | -1.251468 | 0.659068 | 1.8e-02 | 13 | 924 | 11.245 |
| U(1)+spatial | cpu | -4.297161 | 4.201155 | -1.246986 | 0.661004 | 1.0e-15 | 134 | 80 | 0.136 |
| U(1)+spatial+flip | cpu | -4.297161 | 4.201155 | -1.246986 | 0.661004 | 1.3e-15 | 146 | 66 | 0.106 |
| U(1)+spatial+TR | cpu | -4.297161 | 4.201155 | -1.246986 | 0.661004 | 1.2e-15 | 134 | 80 | 0.095 |
| U(1)+spatial+star | cpu | -4.297161 | 4.201155 | -1.246986 | 0.661004 | 1.2e-15 | 134 | 80 | 0.100 |
| U(1)+translation+star | cpu | -4.297161 | 4.201155 | -1.246986 | 0.661004 | 1.0e-15 | 134 | 80 | 0.125 |
| U(1)+spatial+flip+TR+star | cpu | -4.297161 | 4.201155 | -1.246986 | 0.661004 | 1.0e-15 | 146 | 66 | 0.088 |
| none | gpu | -4.268786 | 3.913682 | -1.254880 | 0.654563 | 1.4e-02 | 1 | 4096 | 8.912 |
| U(1) | gpu | -4.222302 | 3.843498 | -1.251389 | 0.659515 | 1.8e-02 | 13 | 924 | 28.207 |
| U(1)+spatial | gpu | -4.297161 | 4.201155 | -1.246986 | 0.661004 | 1.0e-15 | 134 | 80 | 0.226 |
| U(1)+spatial+flip | gpu | -4.297161 | 4.201155 | -1.246986 | 0.661004 | 1.3e-15 | 146 | 66 | 0.209 |
| U(1)+spatial+TR | gpu | -4.297161 | 4.201155 | -1.246986 | 0.661004 | 1.2e-15 | 134 | 80 | 0.134 |
| U(1)+spatial+star | gpu | -4.297161 | 4.201155 | -1.246986 | 0.661004 | 1.2e-15 | 134 | 80 | 0.144 |
| U(1)+translation+star | gpu | -4.297161 | 4.201155 | -1.246986 | 0.661004 | 1.0e-15 | 134 | 80 | 0.210 |
| U(1)+spatial+flip+TR+star | gpu | -4.297161 | 4.201155 | -1.246986 | 0.661004 | 1.0e-15 | 146 | 66 | 0.136 |
exact: E(0.5) = -4.297161, C(0.5) = 4.201155

Composed lanes (incl. star and translation+star) are machine-exact
(1e−15: every block ≤ 512 hits the exact fallback) and ~45× faster
than the unsymmetrised stochastic run whose 1–2 % rows are honest
mTPQ sampling error. Flip splits the pool 134 → 146 blocks (max dim
80 → 66); TR and star drop solves without changing any number.

## Other finite-T methods (fully composed vs none)

| method | composition | device | E(0.5) | C(0.5) | max rel dev E(T) | t [s] |
|---|---|---|---|---|---|---|
| FTLM | none | cpu | −4.306562 | 4.170578 | 1.8e−03 | 2.87 |
| FTLM | U(1)+spatial+flip+TR+star | cpu | −4.341584 | 4.178737 | 8.6e−03 | 0.60 |
| LTLM | none | cpu | −5.387391 | −0.000000 | 9.5e−01 | 0.62 |
| LTLM | U(1)+spatial+flip+TR+star | cpu | −4.625620 | 1.770994 | 4.6e−01 | 0.05 |
| FTLM | none | gpu | −4.301863 | 4.283484 | 1.9e−03 | 2.05 |
| FTLM | U(1)+spatial+flip+TR+star | gpu | −4.290482 | 4.153979 | 4.3e−03 | 23.5 |
| LTLM | none | gpu | −5.387391 | −0.000000 | 9.5e−01 | 0.49 |
| LTLM | U(1)+spatial+flip+TR+star | gpu | −4.625620 | 1.770994 | 4.6e−01 | 1.89 |

FTLM is stochastic in both lanes; the composed run's sampling error is
comparable to the unsymmetrised run's (the benchmark asserts it can't
be qualitatively worse) and 4.8× faster on CPU. **LTLM's accuracy at
N=12 with default knobs is poor in BOTH lanes** (the unsymmetrised run
collapses to the ground state at every T) — a method-regime property
reported as-is, not a composition artifact (the composed lane is
actually closer). FTLM-composed on GPU pays per-sector kernel-launch
overhead across 146 tiny sectors × samples — use the CPU lane at
small block dims. **cTPQ has been removed from the package** (July
2026): mTPQ + the exact small-block fallback covers its use cases;
the MPI-distributed thermal lane keeps its internal canonical
propagator.

## DSSF S^z_{Q=π}(ω) — dense Lehmann peak 2.588236 at ω = 0.370

| composition | device | peak S | peak omega | max |dS| | t [s] |
|---|---|---|---|---|---|
| none | cpu | 2.588236 | 0.370 | 1.4e-07 | 0.018 |
| U(1)+spatial | cpu | 2.588236 | 0.370 | 6.9e-13 | 0.701 |
| none | gpu | 2.588236 | 0.370 | 1.4e-07 | 0.187 |
| U(1)+spatial | gpu | 2.588236 | 0.370 | 6.9e-13 | 0.442 |

Sector route ~5 orders tighter than the full-Hilbert continued
fraction. Flip/TR/star are not yet consumed by the spectral lanes
(Stage 8d); `"require"` still validates the Hamiltonian.

## GPU parity and optimization status

* Every composition runs the SAME plan on both devices — flip masks
  live in the device basis policy (Stage 8b), star/TR orbit plans are
  backend-independent, and the per-sector solves ride the lazy CUDA
  mirrors. GPU rows reproduce dense at ≤ 2e−14.
* At N=12 block dims the GPU rows are kernel-launch-latency-bound
  (plus an external process holding this machine's GPU); GPU wins
  appear at block dims ≳ 10⁵ (see the GPU SpMV benchmarks in
  `docs/perf/bench_symmetry_gpu_matvec_2026-05-28.md`).
* Known GPU follow-up: the Stage-4 shared rank table has no device
  twin yet (per-sector device tables are budget-gated instead) —
  relevant only at N ≥ 30 aggregate scales.

## Notes

* Timings at N=12 measure fixed costs; the large-N scaling story is
  `bench_auto_symmetry_2026-07-03.md` (max block 40 M → 717 k at
  N=28).
* `symmetry="auto"` selects the flip+TR+star composition
  automatically when the Hamiltonian carries the symmetries;
  `symmetry="translation"` + `lattice=` pins the textbook
  space-group split (T projector + point-group stars).
