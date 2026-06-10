---
orphan: true
---

# Scaling envelope of QED

> **One-sentence answer.** This codebase is genuinely peer-grade with QuSpin
> / EDLib / Pomerol for **N ≤ 32 sites**, gets you to **N = 36 with care**
> on a fat single node, and is **not** in the SOTA 40–48 site regime today
> — that requires distributed-memory state vectors (Phase 3 work).
>
> Read this file *before* you spin up a 36-site DSSF run, and definitely
> before you tell anyone we can do 40+.

> **Update (2026-06-09): backend-aware symmetry matvec + when symmetry
> actually helps.** The per-sector matvec path now follows the memory
> regime AND the backend (see [`SYMMETRY.md`](SYMMETRY.md)):
>
> * **CPU** — the precomputed orbit-CSR walk when the CSR fits (eager
>   regime), the CSR-free on-the-fly representative SpMV only when it does
>   not (lazy regime, e.g. ~24 GiB/sector at N=32). The rep kernel is
>   ~100x slower *per matvec* than the CSR walk when the CSR fits, so it is
>   reserved for the scale where the CSR is infeasible.
> * **GPU** — always the on-the-fly representative SpMV when usable: it
>   skips the orbit-CSR build + multi-GiB device upload and the per-element
>   recompute is hidden by the resident threads (measured N=24: rep 4.8 s
>   vs CSR-mirror 18.2 s end-to-end). Per-sector memory is just
>   `reps[] + inv_norms[] + perms + characters` (~600 MB at N=32).
>
> **Symmetry is a MEMORY tool and a FULL-SPECTRUM tool, not an
> iterative-ground-state speed tool.** The symmetry matvec carries an
> inherent ~|G| per-element cost (mapping each connected state back to its
> representative), and the |G| irrep blocks sum to the full Sz dimension,
> so scanning every irrep for a ground state does ~|G|x the matvec work of
> a single plain-Sz Lanczos (measured CPU Heisenberg ring: Sz+symm
> 1.2/3.2/31.7 s for N=14/16/18 vs ~0.5 s flat for Sz-only). Use it for:
>
> * **Memory-bound ground states** — at N=32 a plain-Sz vector is ~9.6 GB
>   (C(32,16)x16 B) and Lanczos needs several, so it will not fit a 16 GB
>   GPU. A symmetry block is |G|x smaller and fits. For a ground state,
>   target the block holding it with `sector=[q0,q1,...]` (e.g. k=0) rather
>   than scanning all irreps; in the lazy regime only that block is built.
>   If plain Sz *does* fit, it is faster (no |G| factor, no orbit
>   enumeration setup).
> * **Full spectrum** — `qed.full_spectrum(...)` /
>   `qed.solve(..., full_spectrum=True)`, or in NLCE the default
>   `full_ed --method FULL_SYMMETRIZED`. Dense per block is O((D/|G|)^3)
>   summed = O(D^3/|G|^2); the dense-vs-symmetry crossover (1D Heisenberg
>   ring, `benchmarks/bench_symmetry_full_spectrum.cpp`) is near N≈11; by
>   N=12 the symmetry path is ~6.6x faster at ~9.5x less peak RSS, widening
>   with N.
>
> Knobs: `ED_SYM_LAZY_SECTORS=1/0` forces the lazy/eager regime;
> `ED_SYM_REP=0` keeps the CPU on the CSR walk even in the lazy regime
> (bisection); `ED_GPU_SYMMETRY_REP=0` restores the GPU orbit-CSR mirror.

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
  * **Distributed/MPI build** (`ed_distributed_main`) for symmetry-projected
    bases that don't fit in single-node RAM. The chunked/disk-streaming
    single-node fallbacks were retired in matvec-unification Phase 7.2;
    the MPI path is faster and works on any cluster.
  * Selective re-orth (the default after Batch 1) — full re-orth at m=200
    against 250 M-state vectors costs 8 TFLOPs per sweep just for the
    re-orth pass.
  * For dense thermodynamics (FTLM/TPQ): `ED_FTLM_PARALLEL=1` is **not**
    safe to enable here unless you've verified your `Operator` is
    thread-safe (the default `Operator::apply` may share scratch buffers).
    Run samples serially or distribute them over MPI ranks via
    `mpirun -n R ./ED ftlm.cfg`.
  * Multi-day wallclock. **Krylov-state checkpoint/restart is now
    available** for the default `lanczos()` solver (Phase 3a #1, see §3
    "Memory / disk strategy" knobs and §6).

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
| `ED_SYM_LAZY_SECTORS` | unset (auto) | Controls whether the streaming-symmetry build materializes **all** per-sector orbit CSRs + reverse indices eagerly (Pass 2) or lazily one sector at a time (LRU-1, built at solve/bind time). `1` forces lazy, `0` forces eager. **Auto-lazy when the estimated all-sector footprint exceeds `ED_SYM_LAZY_SECTORS_BYTES_MAX`** — at N=32 half-filling the eager footprint is ~190 GB, so lazy auto-engages. The lazy path keeps only the deduped orbit-rep list (~600 MB at N=32) + the fixed-Sz basis resident; this is the **cleanest N=32 symmetry setup** and is the default. **On the GPU lane the orbit-CSR streaming this knob governs is superseded by the on-the-fly representative SpMV (`ED_GPU_SYMMETRY_REP`, default on)** — that path never materializes the per-sector orbit CSR at all, so the lazy↔eager distinction is moot for the GPU 32-site Sz+Symm matvec. |
| `ED_SYM_LAZY_SECTORS_BYTES_MAX` | `4 GiB` | Eager→lazy crossover for the symmetry-sector build above. Lower it (e.g. `2147483648` for 2 GiB) on a tight node to force lazy earlier; raise it only if you have the RAM for every sector's orbit CSR at once and want the fastest repeated matvec. |
| `ED_SYM_DENSE_LOOKUP_BYTES_MAX` | `512 MB` | Total budget (across all sectors) for the optional dense `int32_t` state→basis side-table that makes the SpMV reverse lookup O(1). Above this the operator stays on the compact `SortedUint64Index` (O(log N) binary search). Raise on a fat node for faster matvec; lower to save RAM. |
| `ED_GPU_SYMMETRY_REP` | `1` (on) | **On-the-fly representative GPU SpMV** for fixed-Sz + symmetry sectors (`SectorOperator::bind_cuda`). Instead of uploading the per-sector **orbit CSR** (≈`dim_Sz` images + character coefficients, ~14 GiB at N=32) and an O(`dim_Sz`) projection table (~10 GiB), it keeps only the representatives + `1/norm` + the `|G|` group permutations + the per-sector characters resident, and **regenerates the group action + projection arithmetically on the device** (`min_g g(s')` + combinadic rank + a `C(N,n_up)` rep→index table). Working set drops from ~30+ GiB to ~6 GiB at N=32 and per-SpMV traffic becomes just the in/out vectors — the genuine ÷\|G\| win. **This same gate also drives the host sector loop**: `StreamingSymmetryHandle::sector(k)` returns a CSR-free *lazy* `SectorOperator` (`make_rep_sector_operator_lazy`) that defers the per-sector host orbit CSR (~24 GiB at N=32: orbit images/coefficients + reverse-lookup index) — the GPU path never builds it; a CPU `apply` lazily materializes it only as a fallback. Set `=0` to opt back into the legacy orbit-CSR mirror **and** the eager host materialization (diagnostics / bisection). Only fixed-Sz sectors take this path; sym-only full-Hilbert sectors always use the orbit-CSR mirror. |
| `ED_SPARSE_DIM_MAX` | `1<<20` | Custom dim cutoff for the sparse-vs-matrix-free dispatch. Raise carefully. |
| `ED_LANCZOS_CHECKPOINT_DIR` | unset (off) | Directory for atomic Krylov-state checkpoints (Phase 3a #1, see `include/ed/io/lanczos_checkpoint.h`). When set, the default `lanczos()` writes `lanczos_checkpoint.h5` every `ED_LANCZOS_CHECKPOINT_INTERVAL` iterations and on the final iteration. Atomic write-then-rename, so a SIGKILL mid-write leaves the previous checkpoint intact. |
| `ED_LANCZOS_CHECKPOINT_INTERVAL` | `100` | Iterations between checkpoint writes. Lower for faster crash recovery, higher to amortize HDF5 I/O on long runs (each write is ~22 N complex doubles). |
| `ED_LANCZOS_RESUME` | `0` | If `1` and `ED_LANCZOS_CHECKPOINT_DIR` contains a checkpoint, `lanczos()` skips its random-vector init and resumes from `(α[0..k], β[0..k], v_{k-1}, v_k, ring buffer)`. **Eigenvalue-only mode** (`eigenvectors=false`) — eigenvector reconstruction needs the early basis vectors which a resumed run lacks; resuming with `eigenvectors=true` throws. |
| `ED_LANCZOS_REORTH_TILE` | `16` | Tile size `B` (in basis vectors) for the blocked-CGS reorthogonalization in `lanczos` and `lanczos_selective_reorth` (Phase 3a #2, see `include/ed/io/lanczos_reorth.h`). Each tile collapses `B` BLAS-1 `zdotc` + `zaxpy` pairs into two BLAS-2 `zgemv` calls, cutting per-iter file-open overhead by `B×` in disk mode. Clamped to `[1, 256]`; raise on machines with large L2/L3 (working set is `B × N` complex doubles), drop to `1` for the legacy per-vector behaviour. |
| `ED_GPU_MIXED_PRECISION_SPMV` | unset (off) | If `1`/`true`/`yes`, `GPUOperator::applyCusparse` runs the cuSPARSE SpMV in FP32 (`CUDA_C_32F`) instead of FP64 (Phase 3a #3, see `include/ed/gpu/gpu_mixed_precision.h`). Halves the value-array bandwidth on memory-bound matvec; outer Lanczos / FTLM dot / normalize / axpy stay in FP64 so orthogonality is preserved. Only takes effect on the CSR pathway (`N ≥ ED_GPU_CUSPARSE_MIN_DIM`); matrix-free pathways and symmetrized / fixed-Sz operators silently stay FP64. |

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
| `ED_FTLM_PARALLEL` | `0` (serial) | Enables OpenMP over FTLM samples (Batch 2, P1-4). **Opt-in** because the default `Operator` is not guaranteed thread-safe — concurrent `apply()` calls can corrupt shared scratch. Safe with the per-sector-CSR Operator. Validate against the serial run before trusting averaged thermodynamics. (The chunked-symmetry Operator was retired in matvec-unification Phase 7.2 and is no longer available as a thread-safe alternative.) |
| `ED_GPU_TIMING` | `0` | If `1`, GPU fixed-Sz matvec calls insert `cudaDeviceSynchronize()` and record per-call timings. Off by default for performance (Batch 2, P1-6). |
| `ED_NUMA_FIRST_TOUCH` | unset (off) | If `1`/`true`/`yes`, basis-sized work vectors (Lanczos `v_curr` / `v_prev` / `v_next` / `w`, the blocked-reorth tile) are parallel-zero-touched after allocation so each OpenMP thread owns the chunk of pages it will later read in `cblas_zaxpy` / `zdotc` / `zgemv` (Phase 3a #4, see `include/ed/parallel/numa.h`). On a multi-socket box this is the difference between every SpMV pulling its operand vector across the inter-socket link vs. straight from local DRAM (typically 2-4× SpMV bandwidth). No-op below a 256 KB threshold; never changes numerical results. Pair with `ED_NUMA_PIN_THREADS=1` so the thread-to-page assignment is stable across iterations. |
| `ED_NUMA_PIN_THREADS` | unset (off) | If `1`/`true`/`yes`, OpenMP worker threads are pinned compactly via `pthread_setaffinity_np` (thread `t` → CPU `t mod ncpus`) on first call into `lanczos` / `lanczos_selective_reorth` (Phase 3a #4). Idempotent within a process. Pairs with `ED_NUMA_FIRST_TOUCH=1` so each thread keeps owning the same page range across iterations; without pinning the kernel is free to migrate threads between cores and socket-local DRAM access is no longer guaranteed. Honour `OMP_PROC_BIND` / `OMP_PLACES` for non-compact layouts. |

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
* **Symmetry construction**: if the symmetry-projected basis does not fit
  in single-node RAM, switch to the distributed/MPI build
  (`ed_distributed_main`, exercised by the `phase3b` / `mpi` test labels).
  The Phase 7.2 cleanup retired the chunked / disk-streaming single-node
  fallbacks (`run_chunked_symmetry_workflow`,
  `run_disk_streaming_workflow`); the MPI path is faster, works on
  arbitrary cluster sizes, and is matvec-unification first-class
  (`DistributedHost` memory space tag on `DistributedOperator`).
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
* **N=32 mTPQ on a single large GPU, Sz + symmetry**: this is the
  motivating case for the **on-the-fly representative SpMV**
  (`ED_GPU_SYMMETRY_REP`, default on). Working in one k-sector reduces the
  fixed-Sz dim (601 M at n_up=16) to ≈`dim_Sz/|G|` ≈ 75 M. The legacy
  orbit-CSR mirror uploaded (or, in lazy mode, **streamed across PCIe every
  SpMV**) ~14 GiB of orbit indices + coefficients plus a ~10 GiB projection
  table, so the ÷8 FLOP win was entirely eaten by host↔device traffic. The
  rep path keeps only `reps` + `1/norm` + the `|G|` permutations + per-sector
  characters + a `C(N,n_up)` rep→index table (~6 GiB total at N=32, fits a
  40–80 GB A100/H100) and regenerates the group action on the device, so the
  per-SpMV traffic is just the two 75 M-element vectors — the genuine ÷|G|.
  Correctness is pinned bit-for-bit vs the CPU `applySymmetrized` reference
  by `test_rep_symmetry_gpu` (Z_N rings, incl. a |G|=8 complex-character
  fixed-Sz fixture).
  The matching **host** win lands in the production sector loop: with the rep
  path on, `StreamingSymmetryHandle::sector(k)` hands out a CSR-free *lazy*
  `SectorOperator` (`make_rep_sector_operator_lazy`) that knows its dimension
  up-front (Pass 1.5) and builds the CSR-free `RepSectorData` on demand for
  `bind_cuda` — it **never materializes the ~24 GiB/sector host orbit CSR**
  (14 GiB orbit images/coefficients + ~10 GiB `SortedUint64Index` reverse
  lookup). The host orbit CSR is materialized only if a CPU `apply` is invoked
  (the CPU-only fallback), so the GPU 32-site Sz+Symm mTPQ run no longer needs
  a 64 GB+ host just to stage a sector. Pinned end-to-end by
  `test_rep_lazy_sector_loop` (GPU rep matvec == CPU `apply`, with the
  host-CSR-never-materialized invariant asserted on the GPU path).

---

## 6. The Phase 3 roadmap (lifts the ceiling 36 → 40 → 48)

> Listed in dependency order. Each phase produces a usable library; the
> jumps in supported N are gated by the *biggest* missing piece.

### Phase 3a — "honest 36, fast 32" (~2–4 weeks)

Single-node improvements. No new MPI. Goal: make N=36 routine and N=32
publication-grade-fast on GPU.

1. **Krylov-state checkpoint/restart. — DONE (Phase 3a #1).** Persists
   `(α[0..k], β[0..k], v_{k-1}, v_k, ring buffer, RNG state, iteration
   counter, convergence cache)` to a single HDF5 file via atomic
   write-then-rename. Activated by `ED_LANCZOS_CHECKPOINT_DIR=...`,
   resumed via `ED_LANCZOS_RESUME=1`. Eigenvalue-only mode (the basis
   vectors needed for Ritz-vector reconstruction would have to be
   replicated separately, deferred to item #2). See
   `include/ed/io/lanczos_checkpoint.h` and the four-test lockdown in
   `tests/unit/test_lanczos_checkpoint.cpp`.
2. **Out-of-core blocked-tile reorthogonalization. — DONE (Phase 3a #2).**
   The periodic-full and selective re-orth passes in
   `lanczos_selective_reorth` (and the default `lanczos`) now walk the
   Krylov basis in tiles of `B` vectors. Per tile we issue two BLAS-2
   `zgemv` calls (`overlaps = V^H w`; `w := w − V * overlaps`) instead of
   `B` BLAS-1 `zdotc` + `zaxpy` pairs. The selective branch runs CGS2
   (two passes) which is backward-stable to the same bound as MGS
   (Giraud-Langou-Rozložník 2005). Tile size is configured by
   `ED_LANCZOS_REORTH_TILE` (default 16, clamped to `[1, 256]`); the
   in-memory basis buffer is zero-copy through `get_basis_vector_ptr`,
   the legacy on-disk store is read sequentially per tile so the OS page
   cache stays warm. New helpers in `include/ed/io/lanczos_reorth.h` +
   `src/io/lanczos_reorth.cpp`; covered by seven lockdown tests in
   `tests/unit/test_lanczos_reorth.cpp` (CGS-vs-MGS correctness on
   orthonormal V, threshold filter, skip predicate, in-memory and
   on-disk tile loading, end-to-end tile-size invariance for
   `lanczos_selective_reorth` across `B ∈ {1, 4, 16}`, and knob
   clamping).
3. **Mixed-precision SpMV (FP32 matvec + FP64 dot/normalize) on GPU. —
   DONE (Phase 3a #3).** When `ED_GPU_MIXED_PRECISION_SPMV=1` is set and
   the cuSPARSE CSR pathway is selected (`N ≥ ED_GPU_CUSPARSE_MIN_DIM`,
   default 32768), `GPUOperator::applyCusparse` lazily builds an FP32
   copy of the CSR value array (sharing the FP64 row/col index arrays),
   casts the FP64 input vector to FP32 with a tiny element-wise kernel,
   runs `cusparseSpMV` with `CUDA_C_32F`, and casts the FP32 output back
   to FP64. The Lanczos / FTLM outer dot/norm/axpy stay in FP64 (cuBLAS
   `cublasZdotc` / `cublasZdscal` / `cublasZaxpy` on FP64 vectors), so
   global orthogonality is preserved at FP64 precision. Halves the
   value-array bandwidth on a memory-bound SpMV; ground-state Lanczos
   eigenvalues converge to within 1e-5 of the FP64 result on the lockdown
   tests at the cost of ≤2 extra Krylov iterations. New files:
   `include/ed/gpu/gpu_mixed_precision.h`,
   `src/solvers/gpu/gpu_mixed_precision.cu`, plus the FP32 CSR cache
   members on `GPUOperator` (`d_csr_values_fp32_`, `csr_descr_fp32_`,
   workspace vectors); covered by two GPU lockdown tests in
   `tests/unit/test_gpu_mixed_precision_spmv.cpp` (H*v rel L2 < 5e-6 on
   N=10 Heisenberg PBC, ground-state Lanczos eigenvalue within 1e-5 of
   the dense reference at N=8). **Scope of this landing:** only the
   cuSPARSE CSR pathway; matrix-free WARP_REDUCTION /
   BRANCH_FREE_SCATTER / SHARED_MEMORY pathways stay FP64 (kernel
   templating for FP32 matrix-free is a separate, larger job and is
   deferred). Symmetrized / fixed-Sz operators do not currently build a
   CSR and so silently stay FP64.
4. **NUMA-aware first-touch allocator + thread-pinning hooks. — DONE
   (Phase 3a #4).** New module
   `include/ed/parallel/numa.h` + `src/parallel/numa.cpp` adds two
   default-off env knobs: `ED_NUMA_FIRST_TOUCH=1` makes
   `ed::parallel::first_touch_complex` parallel-zero the buffer it's
   given (page-aligned static schedule, so the same OMP thread later
   reads the chunk it just wrote), and `ED_NUMA_PIN_THREADS=1` calls
   `pthread_setaffinity_np` once per process to pin OMP workers
   compactly across logical cores. The Lanczos entry points
   (`lanczos`, `lanczos_selective_reorth`, `lanczos_no_ortho`) and the
   blocked-reorth tile loader (`load_basis_tile`) call into the helpers
   right after their basis-sized allocations. No libnuma dependency
   (works on any Linux + glibc + OpenMP); explicit `numa_alloc_onnode`
   / `mbind` placement is a separate follow-up. Knobs never change
   numerical results -- they only change DRAM page placement and OMP
   thread affinity. Below the 256 KB threshold the touch is a no-op
   (small scratch vectors fit in L2 anyway). Covered by seven lockdown
   tests in `tests/unit/test_numa.cpp` (env-knob defaults + truthy
   string parsing, off-state no-op, sub-threshold no-op, on-state
   counter contract, raw-byte first-touch, idempotent pinning,
   end-to-end Lanczos ground-state energy invariance with knobs on
   vs off to within 1e-12).
5. **Symmetry-projected basis: compact lookup index. — DONE
   (Phase 3a #5).** New header `include/ed/core/sorted_uint64_index.h`
   provides `ed::core::SortedUint64Index`, a sorted-vector + binary-search
   replacement for `std::unordered_map<uint64_t, size_t>`. Same build
   idiom (`m[state] = basis_idx` then `m.finalize()` once per sector);
   lookup returns `kNotFound` sentinel instead of an end-iterator.
   Wired into both `StreamingSymmetryOperator::state_to_sector_basis_`
   and `FixedSzStreamingSymmetryOperator::state_to_sector_basis_` plus
   the corresponding HDF5 reload paths and the inner SpMV kernels
   (`applyHamiltonianTermsFullSpace`, `applyHamiltonianTerms`). Cuts the
   per-entry footprint from ~32-40 B to a flat **16 B** (verified by the
   build-log line `Lookup index footprint: ... B/entry` -- 16.00 in
   practice). At N=36 with full point-group + Sz the dominant sector
   has ~3 × 10⁷ representatives, so this saves ~0.5-0.7 GB and pushes
   that regime from "barely fits" to "comfortably fits". Lookup
   performance is competitive with `unordered_map::find` because the
   binary search is on a contiguous, prefetchable array (~23 cmps for
   10⁷ keys, all on hot cache lines). Covered by `test_sorted_uint64_index`
   (8 sections, 111k assertions: empty / build / duplicate-key /
   sort-invariant / pre-finalize-throws / clear / size-bytes / 1 M-entry
   stress vs `unordered_map` oracle) plus the existing N=4 and N=6
   end-to-end symmetry-projected spectra in `test_symmetry` (still match
   dense reference to 1e-9). A *true* minimal perfect hash via PTHash /
   BBHash is deferred to Phase 3b: it would shave another ~6 B/entry
   but adds an external dep, and the dominant remaining cost at N=36 is
   the orbit-element CSR (`SymBasisState::orbit_elements`), not the
   lookup index.

After Phase 3a: N=36 with full symmetry is **routine** on a single fat
node, and N=32 GPU runs are 2× faster.

### Phase 3b — "honest 40" (~1–2 months)

Distributed memory. This is the regime change.

1. **`DistributedOperator` abstraction** parallel to `Operator`. — **DONE
   (Phase 3b #1).** Header at
   `include/ed/distributed/distributed_operator.h`. Each MPI rank owns a
   contiguous slab of the basis (1D row decomposition via
   `balanced_slab`); `apply(v_in_local, v_out_local)` extracts the
   bit-flip patterns of the Hamiltonian once at construction time, builds
   an `MPI_Alltoallv` plan + a `SortedUint64Index` lookup for the receive
   buffer (Phase 3a #5 reused), then runs a single Alltoallv per matvec
   followed by a gather-form OpenMP-parallel SpMV. Lockdown tests vs the
   serial `Operator` are bit-identical on N=4 OBC, N=6 PBC, N=8 OBC
   across `np ∈ {1,2,4}` (`tests/unit/test_distributed_operator.cpp`).
   *Honest scope:* the current 1D row partitioning is **not**
   symmetry-aware — for "honest 40" with full lattice symmetry we still
   need the orbit-respecting slab boundaries, otherwise Alltoallv counts
   may approach the slab size on long-range Hamiltonians.
2. **Distributed Lanczos** built on top. — **DONE (Phase 3b #2).** Header
   at `include/ed/distributed/distributed_lanczos.h`. Rank-local Krylov
   basis; dot products via `MPI_Allreduce`, `axpy` / `scal` purely local;
   tridiagonal eigensolve replicated on every rank (Eigen). Optional
   `full_reorth` for high-precision runs. The result struct now also
   exposes `tridiag_eigenvalues` + `tridiag_weights` (squared first
   component of each tridiagonal eigenvector) so FTLM / DOS estimators
   can use the proper J&P weights without re-running Lanczos. Lockdown
   tests check ground state vs dense reference on N=4 OBC, N=6 PBC, plus
   `full_reorth = true ↔ false` agreement and bit-replicated eigenvalues
   across all ranks (`tests/unit/test_distributed_lanczos.cpp`).
3. **Distributed FTLM (Z(β) AND ⟨O⟩(β)).** — **DONE (Phase 3b #3 + #5).**
   Header at `include/ed/distributed/distributed_ftlm.h`. Implements
   MPI-over-samples (`n_groups` outer parallelism via `MPI_Comm_split`)
   on top of `DistributedLanczos` with the proper J&P trace estimator
   `Z(β) ≈ (D/R) Σ_s Σ_k |⟨v0_s|ψ_k_s⟩|² exp(-β E_k_s)`. **Phase 3b #5
   add-on:** if `options.observable_op` is non-null, the same call
   computes the canonical J&P observable expectation
   `⟨O⟩(β) = N_O / N_Z` with
   `N_O = (D/R) Σ_s Re(Σ_j q_j(s)* f_j(s,β))`, where
   `q_j = ⟨V_s[j] | O r_s⟩` and
   `f_j(s,β) = Σ_k U_s[j,k] U_s[0,k] e^{-β E_k}`. Cost: one extra
   `DistributedOperator(O)` apply + `m` distributed inner products per
   sample. Lockdown tests recover the exact partition function on N=4
   OBC within statistical noise, recover the exact thermal energy
   `⟨H⟩(β)` for `O = H` at β ∈ {0.1, 0.5, 1.0}, and verify both Z(β) and
   ⟨O⟩(β) are bit-replicated across `np ∈ {1,2,4}`
   (`tests/unit/test_distributed_ftlm.cpp`).
4. **Distributed eigenvector reconstruction.** — **DONE (Phase 3b #6).**
   Setting `DistributedLanczosOptions::compute_eigenvectors = true`
   forces `full_reorth = true` and retains both the rank-local Krylov
   basis (`m × local_n × 16 B` per rank) and the small `m × m`
   tridiagonal eigenvector matrix `U` in the result. The new helper
   `reconstruct_local_eigenvector(result, k, ψ_k_local)` does the
   contraction `ψ_k_local = V_local @ U[:,k]` purely locally — no MPI
   traffic. The convenience wrapper
   `distributed_lanczos_eigenvectors(op, options)` returns the lowest
   `n_keep` eigenpairs as `(eigenvalues, eigenvectors_local)`. Lockdown
   tests assemble the full ψ via `MPI_Allgatherv`, apply the *serial*
   Operator, and check `‖H ψ - E ψ‖₂ < 1e-8` on N=4 OBC + N=6 PBC,
   plus a global-phase replication check across `np ∈ {1,2,4}`
   (`tests/unit/test_distributed_eigenvectors.cpp`). *Honest scope:* at
   "honest 40" with `m = 200` and `local_n ≈ 5e9` the basis costs
   ~16 TB per rank — way outside any sane budget. The recommended
   workflow at scale remains "distributed energy + symmetry-projected
   serial eigenvector"; the new API is honest about this in its
   docstring.
5. **Distributed canonical TPQ.** — **DONE (Phase 3b #8).** New header
   `include/ed/distributed/distributed_tpq.h`. `distributed_tpq(op,
   options, comm)` initialises a rank-local random complex state,
   evolves it via Taylor-truncated `e^{-(δβ/2) H}` substeps (one
   `DistributedOperator::apply` per Taylor order; local zaxpy + dist-norm
   renormalisation between substeps), and measures `E(β) = ⟨ψ|H|ψ⟩` (and
   optionally variance via `⟨H²⟩ = ‖Hψ‖²`) at every entry of
   `options.betas`. Same outer/inner MPI parallelism as
   `distributed_ftlm` (`n_groups` × ranks-per-group). Lockdown tests
   recover the exact thermal energy on N=4 OBC at β ∈ {0.5, 2.0} within
   statistical noise (R = 8 on D = 16), verify replication across ranks,
   and check `E(β=0) ≈ Tr(H)/D = 0` (`tests/unit/test_distributed_tpq.cpp`).
   *Honest scope:* the Taylor truncation is unconditionally stable only
   when `(δβ/2) ‖H‖ ≪ taylor_order`. Defaults (`δβ = 0.05`, order 30)
   cover `‖H‖ ≲ 30` per local site. For larger spectral radii, drop
   `δβ` or raise the order.
6. **Cluster-aware launcher.** — **DONE (Phase 3b #4).** Standalone
   driver `ed_distributed_main` (built when `WITH_MPI=ON`) at
   `src/cli/ed_distributed_main.cpp` exercises both
   `distributed_lanczos` and `distributed_ftlm` from the command line.
   Wrappers in `scripts/distributed/`:
   `run_dist.sh` for single-node `mpiexec`, `slurm_dist.sbatch` for
   slurm, `README.md` for the CLI surface. Smoke-test on N=12 PBC
   reproduces the exact ground state E0 = -5.387… on `np=4`.
7. **Symmetry-aware row partitioning.** — **OPEN (Phase 3b #7).** The
   current `balanced_slab` splits the unsymmetrised basis evenly. With
   full point-group + Sz the natural partition is by orbit, which is
   non-trivial to balance when orbit sizes vary by an order of
   magnitude. Required for "honest 40" before the Alltoallv halo size
   blows up.

After Phase 3b: N=40 with symmetry on ~64 ranks is **possible** in
principle, and N=36 thermodynamics finishes in hours instead of days.
The actual cluster lockdown awaits HPC time + Phase 3b #7.

### Phase 3c — "toward 48" (~2–3 months, requires HPC time)

Multi-GPU and the last 10× of scale. Validation requires actual cluster
hardware.

1. **NCCL-based multi-GPU Lanczos.** — **IMPLEMENTED (Phase 3c stages
   1-3); validation pending.** The build system discovers NCCL when
   both `WITH_CUDA=ON` and `WITH_MPI=ON` (sets `NCCL_FOUND`,
   `NCCL_INCLUDE_DIRS`, `NCCL_LIBRARIES`, propagates `ED_HAVE_NCCL` to
   `ed_distributed`). The runtime surface lives in
   `include/ed/distributed/multi_gpu.h`
   (`ed::distributed::multi_gpu::MultiGpuCommunicator` -- RAII
   communicator + NCCL collectives) and
   `include/ed/distributed/distributed_lanczos_gpu.h`
   (`distributed_lanczos_gpu` -- distributed Lanczos with
   GPU-resident vectors and NCCL halo exchange via
   `DistributedGPUOperator`). The older detection-only
   `multi_gpu_stub.h` remains as a back-compat shim that transitively
   includes `multi_gpu.h`. End-to-end validation on real
   multi-node multi-GPU hardware is still TODO; in-process tests
   currently exercise the single-process multi-GPU path under MPS.
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

* That the current code can do 40+ sites *as a routine production
  workflow*. The Phase 3b distributed solvers in `ed::distributed`
  (`DistributedOperator` / `DistributedLanczos` (energies + Ritz
  vectors) / `DistributedFTLM` (Z + ⟨O⟩) / `DistributedTPQ`, Phase
  3b #1–#6 + #8) are correctness-locked vs the serial path on
  Heisenberg N≤8 across `np ∈ {1,2,4}`, but they ride a 1D row
  decomposition that is **not** symmetry-aware and they have not yet
  been exercised at N=40 on a real cluster. "Honest 40" remains the
  Phase 3b #7 objective (orbit-respecting partition).
* That the MPI in the original `ED` driver / `ed_main.cpp` is
  "distributed Lanczos". It is not — there MPI is *only* sample-level
  parallelism in FTLM (and Jpm-sweep parallelism in BFG). For
  rank-distributed state vectors, use `ed::distributed::*` /
  `ed_distributed_main`.
* That Phase 3c multi-GPU is implemented. It is not. The build system
  detects NCCL when present and `ed::distributed::multi_gpu::
  nccl_compiled_in()` reports back, but no NCCL-backed kernels exist
  yet (validation needs HPC time we have not booked).
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
* That eigenvector reconstruction survives a Lanczos restart. The Krylov
  state is now checkpointed (Phase 3a #1) but the per-iteration basis
  vectors v_0..v_{k-1} required for Ritz-vector recomposition are not;
  resuming with `eigenvectors=true` therefore throws. The eigenvalue-only
  case (the common multi-day workhorse) does work end-to-end.

---

## 8. Provenance

The numbers in §1 are exact (from `python -c "from math import comb;
print(comb(N, N//2))"`), the per-vector sizes assume
`sizeof(std::complex<double>) == 16`, and the per-N status in §2 is
calibrated against the actual code paths in
`src/solvers/cpu/lanczos.cpp`, `src/solvers/cpu/ftlm.cpp`,
`src/solvers/cpu/TPQ.cpp`, the GPU twins, and
`include/ed/core/streaming_symmetry.h`.

If you change the vector type (e.g. to `std::complex<float>`) or
extend the symmetry projection (e.g. add SU(2) total-S via a new
`FixedS2Subspace`; see
[`docs/architecture/SYMMETRY.md`](SYMMETRY.md) §6), redo the
arithmetic — don't trust the table blindly.
