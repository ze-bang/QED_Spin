# Capability matrix vs dense diagonalization — 2026-07-03

`benchmarks/bench_capability_matrix.py --n 12 --devices cpu,gpu`.
N=12 Heisenberg ring (carries all four symmetry axes: U(1) Sz, Z₁₂
translation, spin flip, time reversal). **Ground truth = full dense
`eigh` of the 4096-dim Hamiltonian** (4.8 s): exact E₀, exact partition
sum for E(T)/C(T), exact Lehmann sum for S^z_{Q=π}(ω) with the same
η/π Lorentzian (convention verified: ratio ≡ 1).

Compositions are **pinned** with `spin_flip=`/`time_reversal=`
`"require"`/`"off"` — a row cannot silently fall back and fake its
label. Raw numbers: `capability_matrix_N12.json`.

## Ground state — E₀(dense) = −5.3873909174

| composition | device | E0 | \|dE0\| | max block dim | t [s] |
|---|---|---|---|---|---|
| none | cpu | −5.3873909174 | 3.0e−11 | 4096 | 0.473 |
| U(1) | cpu | −5.3873909174 | 4.4e−15 | 4096 | 0.117 |
| spatial | cpu | −5.3873909174 | 3.7e−12 | 352 | 0.092 |
| U(1)+spatial | cpu | −5.3873909174 | 1.6e−11 | 80 | 0.013 |
| U(1)+spatial+flip | cpu | −5.3873909174 | 5.8e−12 | 44 | 0.014 |
| U(1)+spatial+TR | cpu | −5.3873909174 | 2.2e−11 | 80 | 0.010 |
| U(1)+spatial+flip+TR | cpu | −5.3873909174 | 1.0e−11 | 44 | 0.009 |
| none | gpu | −5.3873909174 | 1.8e−15 | 4096 | 0.013 |
| U(1) | gpu | −5.3873909174 | 5.3e−15 | 4096 | 0.013 |
| spatial | gpu | −5.3873909174 | 7.1e−15 | 352 | 0.146 |
| U(1)+spatial | gpu | −5.3873909174 | 6.2e−15 | 80 | 0.166 |
| U(1)+spatial+flip | gpu | −5.3873909174 | 2.0e−14 | 44 | 0.301 |
| U(1)+spatial+TR | gpu | −5.3873909174 | 6.2e−15 | 80 | 0.101 |
| U(1)+spatial+flip+TR | gpu | −5.3873909174 | 2.0e−14 | 44 | 0.178 |

Every composition reproduces E₀ to ≤3e−11. Block-dim ladder:
4096 → 352 (spatial /11.6) → 80 (×U(1) /51) → 44 (×flip /93).

## Thermal mTPQ — exact E(0.5) = −4.297161, C(0.5) = 4.201155

| composition | device | E(0.5) | C(0.5) | E(2.0) | C(2.0) | max rel dev E(T) | blocks | max dim | t [s] |
|---|---|---|---|---|---|---|---|---|---|
| none | cpu | −4.269779 | 3.901213 | −1.254951 | 0.653766 | 1.4e−02 | 1 | 4096 | 4.24 |
| U(1) | cpu | −4.223071 | 3.835763 | −1.251468 | 0.659068 | 1.8e−02 | 13 | 924 | 10.99 |
| U(1)+spatial | cpu | **−4.297161** | **4.201155** | −1.246986 | 0.661004 | **1.0e−15** | 134 | 80 | 0.090 |
| U(1)+spatial+flip | cpu | −4.297161 | 4.201155 | −1.246986 | 0.661004 | 1.3e−15 | 146 | 66 | 0.127 |
| U(1)+spatial+TR | cpu | −4.297161 | 4.201155 | −1.246986 | 0.661004 | 1.2e−15 | 134 | 80 | 0.101 |
| U(1)+spatial+flip+TR | cpu | −4.297161 | 4.201155 | −1.246986 | 0.661004 | 1.0e−15 | 146 | 66 | 0.086 |
| none | gpu | −4.268786 | 3.913682 | −1.254880 | 0.654563 | 1.4e−02 | 1 | 4096 | 8.47 |
| U(1) | gpu | −4.222302 | 3.843498 | −1.251389 | 0.659515 | 1.8e−02 | 13 | 924 | 28.29 |
| U(1)+spatial | gpu | −4.297161 | 4.201155 | −1.246986 | 0.661004 | 1.0e−15 | 134 | 80 | 0.207 |
| U(1)+spatial+flip | gpu | −4.297161 | 4.201155 | −1.246986 | 0.661004 | 1.3e−15 | 146 | 66 | 0.142 |
| U(1)+spatial+TR | gpu | −4.297161 | 4.201155 | −1.246986 | 0.661004 | 1.2e−15 | 134 | 80 | 0.150 |
| U(1)+spatial+flip+TR | gpu | −4.297161 | 4.201155 | −1.246986 | 0.661004 | 1.0e−15 | 146 | 66 | 0.114 |

Two regimes, both by design:

* **Symmetry-composed lanes are machine-exact** (1e−15 vs the exact
  partition sum): every block is ≤ 512-dim, so the flat pool's exact
  small-sector fallback runs — the symmetry decomposition converts a
  stochastic method into an exact one at this size, **47× faster**
  (0.09 s vs 4.2 s) than the unsymmetrised run it also beats on
  accuracy by 13 orders of magnitude.
* The `none` / `U(1)` lanes are honest mTPQ: 1–2 % sampling error on
  4096/924-dim blocks (that error shrinks with `num_samples` and N).
  These rows are why the E(0.5)/C(0.5) columns differ from exact.
* Flip adds the (k,±) split (134 → 146 blocks, max dim 80 → 66); TR
  copies conjugate sectors (same numbers, fewer solves).

## DSSF S^z_{Q=π}(ω) — dense Lehmann peak 2.588236 at ω = 0.370

| composition | device | peak S | peak ω | max \|dS(ω)\| | t [s] |
|---|---|---|---|---|---|
| none | cpu | 2.588236 | 0.370 | 1.4e−07 | 0.018 |
| U(1)+spatial | cpu | 2.588236 | 0.370 | 6.9e−13 | 0.596 |
| none | gpu | 2.588236 | 0.370 | 1.4e−07 | 0.181 |
| U(1)+spatial | gpu | 2.588236 | 0.370 | 6.9e−13 | 0.415 |

The sector route reproduces the dense Lehmann spectrum ~5 orders
tighter than the full-Hilbert continued fraction (an 80-dim block's CF
is essentially exact). Spin-flip / time-reversal are not yet consumed
by the spectral lanes (Stage 8d); `"require"` still validates the
Hamiltonian.

## Notes

* Timings at N=12 measure *fixed costs*, not scaling — the sector
  construction is ~10 ms here and cached across runs; see
  `bench_auto_symmetry_2026-07-03.md` for the large-N scaling story
  (max block 40 M → 717 k at N=28).
* GPU rows at this size are launch-latency-bound (and the machine has
  an external process holding the GPU); GPU wins appear at block dims
  ≳ 10⁵.
* `symmetry="auto"` selects the U(1)+spatial+flip+TR row of each
  table automatically when the Hamiltonian carries the symmetries.
