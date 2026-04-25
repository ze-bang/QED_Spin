# Scaling envelope of `exact_diagonalization_cpp`

> **One-sentence answer.** This codebase is genuinely peer-grade with QuSpin
> / EDLib / Pomerol for **N ≤ 32 sites**, gets you to **N = 36 with care**
> on a fat single node, and is **not** in the SOTA 40–48 site regime today
> — that requires distributed-memory state vectors (Phase 3 work).
>
> Read this file *before* you spin up a 36-site DSSF run, and definitely
> before you tell anyone we can do 40+.

This document tells you, for a given system size N:

1. how much memory one Lanczos / FTLM / TPQ vector costs,
2. how big the symmetry-reduced sector basis is,
3. whether the current architecture can actually run that size,
4. which env-var knobs change the answer, and
5. what's missing to push the ceiling higher (Phase 3 roadmap).

Numbers are for **spin-1/2** with `Complex = std::complex<double>` (16 B per
state amplitude). For S>1/2 multiply dim by `(2S+1)^N / 2^N` and re-do the
arithmetic.

---

## 1. The dimensions you're actually fighting

Without symmetry, the Hilbert space is `2^N`. With Sz conservation and
half-filling, `C(N, N/2)`. With *additional* lattice point group ×
translation symmetry the typical reduction is a factor of `~G ≈ N` (more
on a square lattice with C₄ᵥ + translations: `8N`; less on a kagome
unit-cell-centered chain).

The conservative half-filling × `÷ N` column is what you should use to size
a node.

| N  | full Hilbert `2^N`     | half-filling `C(N,N/2)` | sector ≈ `C(N,N/2)/N` | one Cdouble vector | 100 Lanczos vectors |
|---:|-----------------------:|------------------------:|----------------------:|-------------------:|--------------------:|
| 16 | 6.6 × 10⁴              | 1.3 × 10⁴               | 8.0 × 10²             | 13 KB              | 1.3 MB              |
| 20 | 1.0 × 10⁶              | 1.8 × 10⁵               | 9.2 × 10³             | 147 KB             | 14.7 MB             |
| 24 | 1.7 × 10⁷              | 2.7 × 10⁶               | 1.1 × 10⁵             | **1.8 MB**         | 180 MB              |
| 28 | 2.7 × 10⁸              | 4.0 × 10⁷               | 1.4 × 10⁶             | 23 MB              | 2.3 GB              |
| 32 | 4.3 × 10⁹              | 6.0 × 10⁸               | 1.9 × 10⁷             | **300 MB**         | 30 GB               |
| 36 | 6.9 × 10¹⁰             | 9.1 × 10⁹               | 2.5 × 10⁸             | **4.0 GB**         | **400 GB**          |
| 40 | 1.1 × 10¹²             | 1.4 × 10¹¹              | 3.5 × 10⁹             | **56 GB**          | **5.6 TB**          |
| 44 | 1.8 × 10¹³             | 2.1 × 10¹²              | 4.8 × 10¹⁰            | 765 GB             | 76 TB               |
| 48 | 2.8 × 10¹⁴             | 3.2 × 10¹³              | 6.7 × 10¹¹            | **10.7 TB**        | **1.1 PB**          |

**One vector** is the absolute minimum; SpMV needs at least two (`v_in`,
`v_out`) and Lanczos with 3-term recurrence needs three. Selective re-orth
needs `m` of them; full re-orth needs all `m` simultaneously addressable
(can be on disk, see §3).

---

## 2. Where this codebase lives, today

### What works *out of the box* (single workstation, ≤ 64 GB RAM)

* **N ≤ 28**: every solver (Lanczos, FTLM, LTLM, mTPQ, cTPQ, Krylov–Schur,
  Davidson, LOBPCG, full diag, ScaLAPACK), CPU and GPU, with full
  re-orthogonalization, in seconds to minutes. This is QuSpin / EDLib
  territory.
* **N = 32 with Sz conservation**: also fits comfortably. Sector dim
  ~2 × 10⁷, one vector 300 MB, 100 vectors 30 GB. GPU Lanczos is faster
  than CPU here because the matrix-free SpMV saturates HBM bandwidth.

### What works *with care* (fat node, ≥ 256 GB RAM, NVMe scratch)

* **N = 36 with full point-group symmetry**: sector dim ~2.5 × 10⁸, one
  vector 4 GB. You need:
  * `ED_LANCZOS_DISK=1` so the Krylov basis spills to a directory under the
    lanczos working dir (the basis doesn't have to fit in RAM, only two
    active vectors and one accumulator do — ~12 GB working set).
  * Disk-streaming sector construction
    (`run_disk_streaming_workflow` / `run_chunked_symmetry_workflow` in
    `src/cli/workflows.cpp`) so the symmetry-projected basis is not built
    in RAM all at once.
  * Selective re-orth (the default after Batch 1) — full re-orth at m=200
    against 250 M-state vectors costs 8 TFLOPs per sweep just for the
    re-orth pass.
  * For dense thermodynamics (FTLM/TPQ): `ED_FTLM_PARALLEL=1` is **not**
    safe to enable here unless you've verified your `Operator` is
    thread-safe (the default `Operator::apply` may share scratch buffers).
    Run samples serially or distribute them over MPI ranks via
    `mpirun -n R ./ED ftlm.cfg`.
  * Multi-day wallclock. There is **no Krylov-state checkpoint/restart**
    today — a node failure means starting Lanczos from zero. See §6.

### What does NOT work (period)

* **N = 36 *without* symmetry**: dim 6.9 × 10¹⁰, one vector 1.1 TB. Beyond
  any single-node DRAM. Use symmetry or don't run it.
* **N = 40, with or without symmetry**: even one full-precision vector
  (56 GB at half-filling-with-symmetry, 1.1 TB without) does not fit on
  any node we test on. **The state vector is `std::vector<Complex>` on a
  single rank**; there is no MPI decomposition of it. This is the wall.
* **N = 44–48**: needs HΦ-on-Fugaku-class infrastructure: distributed
  multi-GPU Lanczos with NCCL all-reduce on dot products, halo-exchange
  SpMV, perfect-hash symmetry indexing because no `unordered_map<uint64_t>`
  can store the basis. This codebase is two engineering quarters away from
  that, not two weeks. See §6.

---

## 3. Knobs that change the answer

All of these are read at process start (most cached on first call) — set
them in your run script, not mid-run.

### Memory / disk strategy

| Env var | Default | What it does |
|---|---|---|
| `ED_LANCZOS_DISK` | `0` (in-memory) | If `1`/`true`/`yes`, the Krylov basis is spilled to a per-call working directory instead of registering in `lanczos_io::register_basis_buffer`. **Required for N≥36.** Costs disk I/O bandwidth per re-orth sweep; pair with NVMe. |
| `ED_USE_SPARSE` | unset (auto) | `0` forces matrix-free SpMV always; `1` forces CSR assembly. The auto threshold is `dim ≤ 2²⁰` (~1 M states), see `Operator::apply`. **Never set `=1` for N≥28** — assembling a 10⁸-row CSR will OOM you. |
| `ED_SPARSE_DIM_MAX` | `1<<20` | Custom dim cutoff for the sparse-vs-matrix-free dispatch. Raise carefully. |

### Numerics

| Env var | Default | What it does |
|---|---|---|
| `ED_LANCZOS_COMPLEX_SEED` | `0` (real seed) | If `1`, Lanczos starts from a fully complex random vector. Default is real-only so the operator can take the real-CSR / `apply_real` fast path for the entire Krylov space when H is real. Flip to `1` only when you specifically need to exercise complex spectra. |
| `ED_CTPQ_PROPAGATOR` | `taylor` | `krylov` switches canonical-TPQ imaginary-time evolution to the Lanczos-projected `expm(−Δτ T_m)` propagator added in Batch 2. Stable for larger Δτ; falls back to Taylor on Krylov breakdown. |
| `ED_CTPQ_KRYLOV_M` | `30` | Krylov subspace size for the cTPQ propagator above. |
| `ED_GPU_ALLOW_DROPPED_THREEBODY` | `0` | The GPU operator now hard-fails if `InterAll` contains 3-body terms (Batch 1, P0-8). Set to `1` only if you understand you're dropping those terms on GPU paths. |

### Parallelism

| Env var | Default | What it does |
|---|---|---|
| `ED_FTLM_PARALLEL` | `0` (serial) | Enables OpenMP over FTLM samples (Batch 2, P1-4). **Opt-in** because the default `Operator` is not guaranteed thread-safe — concurrent `apply()` calls can corrupt shared scratch. Safe with the per-sector-CSR Operator and with the chunked-symmetry Operator. Validate against the serial run before trusting averaged thermodynamics. |
| `ED_GPU_TIMING` | `0` | If `1`, GPU fixed-Sz matvec calls insert `cudaDeviceSynchronize()` and record per-call timings. Off by default for performance (Batch 2, P1-6). |

### Diagnostics

| Env var | Default | What it does |
|---|---|---|
| `ED_LANCZOS_VERBOSE` | `0` | Per-iteration progress prints inside Lanczos. Useful when debugging convergence; floods stdout in production (FTLM/TPQ call Lanczos hundreds of times). |
| `ED_LOG_LEVEL` | `INFO` | Standard `ed_log` level (`DEBUG` / `INFO` / `WARN` / `ERROR`). Batch 3 routed all surviving `[DEBUG]` prints in `ed_wrapper.h` through this. |

---

## 4. Worked example: a "publication-grade" 36-site Heisenberg ground state

You want the ground state energy and the lowest 5 eigenvalues of the
spin-1/2 Heisenberg model on a 6×6 square lattice (36 sites), to ~1e-9
precision.

* **Sector**: pick `Sz = 0`, momentum `(π,π)`, point-group `A₁`.
  Sector dim is roughly `C(36,18) / (36 × 8) ≈ 3 × 10⁷` — **fits on a
  workstation**.
* **Solver**: `LANCZOS` (default selective re-orth) or
  `IRLM` (implicitly-restarted, with locking) for nev=5.
* **Memory**: 3 × 10⁷ × 16 B = 480 MB per vector. m=200 in-memory basis is
  96 GB — fits on a fat node (`ED_LANCZOS_DISK=0`). On a 64 GB workstation,
  set `ED_LANCZOS_DISK=1` and let the basis spill.
* **Symmetry construction**: use `run_chunked_symmetry_workflow` if memory
  is tight, or `run_disk_streaming_workflow` if you want zero in-RAM
  intermediate basis at all.
* **Wallclock**: hours, not days. GPU path (`LANCZOS_GPU` with
  `BLOCK_LANCZOS_GPU` for nev=5) is 3–10× faster.
* **Honesty**: this is well-trodden territory. Multiple groups have
  published 36-site Heisenberg with this approach since the 2000s.

You want the *full* dynamic structure factor on the 36-site cluster at
finite temperature, all temperatures, all q in the BZ.

* That's *N_β × N_q* Lanczos runs in continued-fraction DSSF, plus FTLM
  averaging on top. At 250 M-state sector and a few thousand (β, q) points,
  you're looking at a multi-week single-node job.
* **Recommendation**: parallelize *over (β, q) points across MPI ranks*
  using the existing `MPI_Reduce` plumbing in `ftlm.cpp` (lines 3560–3577)
  — this is embarrassingly parallel and already works. Each rank does its
  own slab of the (β, q) grid.
* **Phase 3a** would let one Lanczos run distribute over multiple ranks
  (much more useful for the *single*-Lanczos use case below).

You want a 40-site Heisenberg ground state.

* **You can't.** Stop, read §6, decide whether to:
  * fall back to **DMRG / MPS** (different library — the `idmrg.h` scaffold
    in this repo was deleted in P0 because it was never finished), or
  * fall back to **Quantum Monte Carlo** (different library; QMC has no
    sign problem for unfrustrated bipartite Heisenberg), or
  * **wait for Phase 3** of this codebase, or
  * **use HΦ / dynamite / KQS** which already do distributed-memory ED.

---

## 5. Worked example: thermodynamics via FTLM / TPQ

FTLM / TPQ are *cheaper than DSSF* because each sample is a single short
Lanczos (m=50–100), and you average over R=10–100 i.i.d. random vectors.

* **N=32 FTLM, m=80, R=50**: 50 short Lanczos runs of dim 1.9 × 10⁷, each
  needing ~80 vectors × 300 MB = 24 GB working set. On a 64 GB node, run
  serially (`ED_FTLM_PARALLEL=0`) so only one sample is live at a time.
  Wallclock: hours.
* **N=32 FTLM with thread-safe Operator**: set `ED_FTLM_PARALLEL=1`. With 8
  OMP threads you'll get ~6× speedup if memory bandwidth permits.
  *Validate* against the serial run on a small problem first.
* **N=36 FTLM**: each sample wants 80 × 4 GB = 320 GB working set with full
  re-orth. Selective re-orth + `ED_LANCZOS_DISK=1` brings this down to
  ~12 GB resident + 320 GB on disk per sample. Run 1 sample per node with
  MPI distributing the R samples across nodes. Wallclock: 1–2 days for
  R=50, 30 nodes.
* **TPQ at N=36**: cheaper per sample than FTLM (no per-(β) stratification),
  but **does not benefit from `ED_LANCZOS_DISK`** the same way (TPQ is a
  long evolution, not a Krylov basis). Imaginary-time evolution is
  `2 × N_τ` SpMV per sample (Taylor) or `m × N_τ` SpMV (Krylov, set
  `ED_CTPQ_PROPAGATOR=krylov`). For canonical TPQ at very low T, prefer
  Krylov — it's stable for Δτ ~ 0.1 instead of the ~ 0.01 Taylor needs.

---

## 6. The Phase 3 roadmap (lifts the ceiling 36 → 40 → 48)

> Listed in dependency order. Each phase produces a usable library; the
> jumps in supported N are gated by the *biggest* missing piece.

### Phase 3a — "honest 36, fast 32" (~2–4 weeks)

Single-node improvements. No new MPI. Goal: make N=36 routine and N=32
publication-grade-fast on GPU.

1. **Krylov-state checkpoint/restart.** Persist `(α[0..k], β[0..k], v_{k-1},
   v_k, RNG state, iteration counter)` to HDF5 every M iterations and on
   SIGTERM. Resume from `--restart` flag. This is the #1 ask from anyone
   running multi-day jobs.
2. **Out-of-core blocked-tile reorthogonalization.** Today, when
   `ED_LANCZOS_DISK=1`, the re-orth pass random-accesses the on-disk basis
   one vector at a time. Refactor to walk the basis in tiles of
   `B = 8–32` vectors, holding `B × N` complex doubles in pinned RAM and
   calling a single `zgemv` per tile. Cuts re-orth I/O by `B×`.
3. **Mixed-precision SpMV (FP32 matvec + FP64 dot/normalize)** on GPU.
   Deferred from Batch 2 (P1-9). 1.7–2× speedup on
   memory-bandwidth-bound matvec at the cost of one extra Krylov iteration
   to compensate for the lower-precision intermediates. Easiest big win we
   are leaving on the table.
4. **NUMA-aware first-touch allocator + thread-pinning hooks.** On a
   4-socket node the difference between "OS scheduler decides" and
   "first-touch on the rank that owns the slice" is 30–50% on SpMV.
5. **Symmetry-projected basis: switch `unordered_map<uint64_t, size_t>`
   to a perfect hash** (e.g., compressed-sparse representative table +
   Robin-Hood probing, or a true minimal perfect hash via the BBHash /
   PTHash library). Today this map dominates RAM at N=36 and *fails* at
   N=40. The fix is local to `streaming_symmetry.h`.

After Phase 3a: N=36 with full symmetry is **routine** on a single fat
node, and N=32 GPU runs are 2× faster.

### Phase 3b — "honest 40" (~1–2 months)

Distributed memory. This is the regime change.

1. **`DistributedOperator` abstraction** parallel to `Operator`. Each MPI
   rank owns a contiguous slab of the basis; `apply(v_in_local,
   v_out_local)` does local SpMV + `MPI_Alltoallv` for the off-diagonal
   columns. Symmetry-aware: the slab boundaries should respect orbit
   structure so the off-rank traffic is bounded.
2. **Distributed Lanczos** built on top. Dot products via `MPI_Allreduce`,
   `axpy` is purely local. Re-orth is the painful one — needs either
   replicated short Krylov basis (`m ≤ 100`) or distributed re-orth via
   ring-exchange.
3. **Distributed FTLM / TPQ.** Once distributed Lanczos exists, FTLM
   becomes "distributed Lanczos × MPI-over-samples" and TPQ becomes
   "distributed-vector imaginary-time evolution" — both straightforward.
4. **Cluster-aware launcher.** A `scripts/launch_distributed.sh` that does
   the right SLURM / Open MPI / MVAPICH2 incantations for common HPC
   environments, with rank-binding and HBM-aware placement.

After Phase 3b: N=40 with symmetry on ~64 ranks is **possible**, and
N=36 thermodynamics finishes in hours instead of days.

### Phase 3c — "toward 48" (~2–3 months, requires HPC time)

Multi-GPU and the last 10× of scale. Validation requires actual cluster
hardware.

1. **NCCL-based multi-GPU Lanczos.** Replace `MPI_Allreduce` with
   `ncclAllReduce` for dot products on the same node; use NVLink/NVSwitch
   when available; fall back to MPI-over-Infiniband across nodes. Build
   on cuQuantum's `custatevec`-style API for the SpMV.
2. **GPU-Direct RDMA for halo exchange.** The off-diagonal columns in
   distributed SpMV are the bottleneck once compute is GPU-resident. RDMA
   directly between GPU memory across the IB fabric makes this work at
   100s-of-GPUs scale.
3. **Distributed disk-backed Krylov basis** with parallel I/O to a
   Lustre/GPFS scratch. m=200 Lanczos at N=44 means
   200 × 750 GB = 150 TB of basis spread across ranks; parallel HDF5 plus
   MPI-IO is the right tool here.
4. **Validation against HΦ on a published 40-site benchmark** before
   anyone claims 44/48-site numbers.

After Phase 3c: this codebase is in the same league as HΦ for distributed
ED. Until then, claim only what we can deliver.

---

## 7. What I am explicitly *not* claiming

* That the current code can do 40+ sites. It cannot.
* That MPI in the current code is "distributed Lanczos". It is not — MPI
  here is *only* sample-level parallelism in FTLM (and Jpm-sweep
  parallelism in BFG). The state vector is local to one rank.
* That `BasisBufferScope` / `ED_LANCZOS_DISK=1` lets you hold a vector
  bigger than node RAM. It only spills the *Krylov basis*; the *active
  vectors* still live in RAM.
* That GPU automatically helps at large N. It only helps when the vector
  fits in HBM (typically ≤ 80 GB on H100 — i.e., N ≤ 36 with symmetry).
  Past that point the host↔device transfer dominates and CPU becomes
  competitive again until distributed-multi-GPU lands.
* That mixed precision is wired up. The `mixed_precision.cuh` header was
  deleted in Batch 2 (P1-9) precisely because it was a misleading stub.
  Real mixed precision is Phase 3a item #3.
* That checkpoint/restart of Lanczos exists. *Intermediate observables* are
  checkpointed via HDF5; the *Krylov state* is not. Phase 3a item #1.

---

## 8. Provenance

This document was written after Batches 1, 2, and 3 of the modernization
audit landed (see `MODERNIZATION_AUDIT.md` §9 for the executive summary).
The numbers in §1 are exact (from `python -c "from math import comb;
print(comb(N, N//2))"`), the per-vector sizes assume
`sizeof(std::complex<double>) == 16`, and the per-N status in §2 is
calibrated against the actual code paths in `src/solvers/cpu/lanczos.cpp`,
`src/solvers/cpu/ftlm.cpp`, `src/solvers/cpu/TPQ.cpp`, the GPU twins, and
`include/ed/core/streaming_symmetry.h` as of this commit.

If you change the vector type (e.g., to `std::complex<float>`) or the
symmetry reduction (e.g., add SU(2) total-spin projection), redo the
arithmetic — don't trust the table blindly.
