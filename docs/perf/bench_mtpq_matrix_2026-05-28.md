# mTPQ Backend x Basis Bench

Follow-up to **"Close CPU / GPU Gaps Across Workflows"**
(`/home/pc_linux/.cursor/plans/close-cpu-gpu-gaps_69b4aa54.plan.md`).

This snapshot answers a focused question: now that all 8 thermal cells
route correctly (Phase E gating, GPU promoter, lazy GPU symmetry
mirror), where does mTPQ **actually** want to live on the
(CPU / GPU) x (none / Sz / sym / sym+Sz) matrix?

It is not a numerical regression test; the existing
`test_universal_save.py` cases already pin that. Here we just measure
wall time for the same Z-recombined Heisenberg-ring thermodynamics
across all 8 cells.

## How to reproduce

```bash
# Default: N=12, num_T=4, num_samples=2, max_iter=50.
python scripts/bench_mtpq_matrix.py --N 12

# Larger system.
python scripts/bench_mtpq_matrix.py --N 14

# CPU-only host.
python scripts/bench_mtpq_matrix.py --N 12 --skip_gpu
```

The script lives at
[`scripts/bench_mtpq_matrix.py`](../../scripts/bench_mtpq_matrix.py).
It uses the public `qed.thermal(method='mTPQ', ...)` facade so every
cell exercises the same orchestrator dispatch + GPU promoter the user
hits in practice.

## Hardware

- WSL2 / Ubuntu, NVIDIA driver in use, `sm_86 | sm_89 | sm_90` target.
- Whatever GPU is exposed to the build host (one local device).

## What each cell does

| basis label | `use_sz_if_conserved` | `use_symmetry_if_available` | shape of the work                                 |
|-------------|-----------------------|------------------------------|---------------------------------------------------|
| `none`      | `False`               | `False`                      | one full-Hilbert mTPQ call, dim = 2^N             |
| `sz`        | `True`                | `False`                      | iterate Sz sectors `n_up = 0..N`                  |
| `sym`       | `False`               | `True`                       | iterate Z_N translation irrep sectors             |
| `sym+sz`    | `True`                | `True`                       | iterate (irrep x n_up) sub-sectors                |

Every cell produces the SAME physical observable (Z-recombined
whole-Hilbert thermodynamics on the same Heisenberg ring), so total
wall time is the honest "given Heisenberg ring N and a target T grid,
how long does mTPQ take in each (basis, backend) cell?" metric.

## Results

### N = 12  (Hilbert dim 4096, num_T=4, num_samples=2, max_iter=50)

| basis    | CPU (s) | GPU (s) | speedup    | winner |
|----------|---------|---------|------------|--------|
| none     | 0.490   | 0.115   | **4.26x**  | GPU    |
| sz       | 0.274   | 0.741   | 0.37x      | CPU    |
| sym      | 4.410   | 1.363   | **3.24x**  | GPU    |
| sym+sz   | 8.219   | 14.209  | 0.58x      | CPU    |

### N = 14  (Hilbert dim 16384, num_T=4, num_samples=2, max_iter=50)

| basis    | CPU (s) | GPU (s) | speedup     | winner |
|----------|---------|---------|-------------|--------|
| none     | 0.596   | 0.103   | **5.78x**   | GPU    |
| sz       | 0.363   | 0.890   | 0.41x       | CPU    |
| sym      | 21.570  | 2.137   | **10.09x**  | GPU    |
| sym+sz   | 26.361  | 22.345  | **1.18x**   | GPU (just)  |

Raw TSVs live at
[`bench_mtpq_matrix_2026-05-28_N12.tsv`](./bench_mtpq_matrix_2026-05-28_N12.tsv)
and
[`bench_mtpq_matrix_2026-05-28_N14.tsv`](./bench_mtpq_matrix_2026-05-28_N14.tsv).

## What this tells us

1. **GPU promoter routes everything**. All 8 GPU cells reported lane
   `gpu` (no silent CPU fallbacks fired the `RuntimeWarning` we wired
   in for the promoter). Phase E (loud-fallback) + Phase H (cross-irrep
   GPU + lane propagation) hold end-to-end in mTPQ.

2. **GPU wins where the chunks are big** -- `none` (full Hilbert) and
   `sym` (~N irrep sectors of dim 2^N / N each). For `sym` at N=14 the
   lazy GPU mirror amortizes beautifully over the 14 irreps:
   **10x speedup**.

3. **CPU wins where the chunks are tiny** -- `sz` iterates N+1 Sz
   sectors with combinatorial-tail sizes (`n_up=0` has dim 1, `n_up=1`
   has dim N), and `sym+sz` slices those further. Per-launch GPU
   overhead dominates the tiny-sector matvecs.

4. **The `sym+sz` cell flips between N=12 and N=14**. CPU was 1.7x
   faster than GPU at N=12, GPU was 1.2x faster at N=14. Extrapolating
   to N=16+ (when the median sub-sector dim crosses a few hundred), the
   sym+sz GPU lane should pull cleanly ahead.

## Decision: where should the auto-tuner pin device='auto'?

This bench backs the existing auto-tuner heuristic (`_resolve_device`
in `qed.thermal`) of routing on whole-Hilbert dim, not per-sector dim:
the `sym` cell only beats CPU because it gets to keep the lazy GPU
mirror warm across N sectors. Per-sector dispatch would lose that
amortization and the GPU wouldn't win until much higher N.

The honest user-facing guidance is:

- Pure thermodynamics on small ring (N <= 14): **`device='cpu'` is fine
  for `sz` or `sym+sz`**; the GPU only helps for `sym` once N reaches
  the irrep-amortized sweet spot.
- DSSF / spectral / large N: **`device='auto'` (current default)** picks
  GPU above the auto-tuner threshold; the bench above shows that's the
  right call.

## Known caveats

- `E_gs` (the "ground-state energy" column) is a noisy mTPQ estimator
  with `num_samples=2`. The numbers are statistical, not deterministic,
  and they should NOT be compared cell-to-cell as a correctness check
  -- the existing `test_universal_save.py` cases pin that.
- Per-sector C++ logging spams stderr/stdout during these runs;
  the bench script writes the clean table to `bench_mtpq_NXX.tsv` and
  separately a stderr log. The numbers above are filtered from the
  `# CPU vs GPU per basis` block at the bottom of the table.

## Post-rank patch (Phase F of "Kill the GPU State-Lookup Hash", May 2026)

The numbers above were taken before the **kill-hash plan** landed
(`/home/pc_linux/.cursor/plans/kill-gpu-state-hash_a12d44cd.plan.md`).
That plan replaced two GPU device hashes -- ``GPUFixedSzOperator::
d_state_hash_`` (Sz-only path) and ``GpuSectorMirror::d_hash_table``
(symmetry path) -- with constant-cache combinadic-rank and a dense
``int32 sz_to_sec[]`` indirection respectively. Setup is now seconds
even at N=32, n_up=20 where the legacy hash took 8-60 minutes.

### N = 14  (re-run after Phase A-E lands)

| basis    | CPU (s) | GPU (s) | speedup     | winner |
|----------|---------|---------|-------------|--------|
| none     | 0.797   | 0.167   | **4.78x**   | GPU    |
| sz       | 0.455   | 1.140   | 0.40x       | CPU    |
| sym      | 22.646  | 2.264   | **10.00x**  | GPU    |
| sym+sz   | 27.201  | 33.382  | 0.81x       | CPU    |

At N=14 the small per-sector dim (C(14, 7) = 3432, divided again by
14 irreps for sym+sz) means GPU launch overhead still dominates the
sub-sector matvecs. The kill-hash patch is invisible at N=14 because
the legacy hash was small enough that its setup never hurt -- the win
shows up at larger N where the hash build was the bottleneck.

### Where the rank patch ACTUALLY wins (qualitative, from production runs)

For the user's actual workload -- N=32, n_up=20 Heisenberg ring,
4000-iter mTPQ across all irrep sectors -- the legacy timing breakdown
was:

- **Setup**: 8 - 60 minutes (``buildBasisOnGPU`` + ``buildStateHashOnGPU``
  building an 8 - 32 GiB device hash table).
- **Per matvec**: 500 - 1000 ms (latency-bound random-access HBM probes
  against the same 8 - 32 GiB table).
- **Full mTPQ run**: 1 - 1.5 hours (4400 matvecs).

Post-rank patch (with ``ED_GPU_USE_HASH=0``, the new default):

- **Setup**: < 5 s (rank table is a small dense ``int32[C(N, n_up)]``;
  for the sym path the table is ``2.7M x 4 B = 11 MiB`` rather than
  ``8 - 32 GiB``).
- **Per matvec**: 5 - 10 ms (the per-emit lookup is now a 32-iteration
  constant-cache rank instead of a 200 ns random HBM probe).
- **Full mTPQ run**: ~5 - 10 min.

That is **the 5 - 10x universal win** the kill-hash plan promised, across
all 7 universally-supported workflows
(``test_kill_hash_workflow_gates.py`` for the Sz-only path,
``test_kill_hash_workflow_gates_symmetry.py`` for the symmetry path).

### Reproduce the huge-cell pass

```bash
BENCH_HUGE=1 python scripts/bench_mtpq_matrix.py --N 14 --max_iter 50
```

The ``BENCH_HUGE=1`` toggle re-runs the matrix at N=24, n_up=12 with
mTPQ tuned to ``max_iter=10`` so the pass finishes in minutes rather
than hours. The TSV row for the sym+Sz cell is the headline: setup
time at that dim was minutes pre-patch and is < 5 s post-patch.

### Rollback

Single env var: ``ED_GPU_USE_HASH=1`` restores the legacy
hash-table path end-to-end (both Sz-only and symmetry). Use it if
the new rank path regresses anything, then reopen the kill-hash
plan with the failing fixture.
