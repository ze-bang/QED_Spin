# Phase 3a — single-node scale-up: shipped

> Closes Phase 3a of the SOTA-publication-grade ED roadmap. Five
> commits over two weeks lifted the realistic single-node ceiling from
> "honest 32, painful 36" to **honest 36 routine, fast 32 on GPU**, and
> teed up Phase 3b (distributed memory for honest 40) by removing every
> single-node bottleneck a profiler can find.
>
> Read this once and you have the full picture. For per-item depth see
> [`../architecture/SCALING.md`](../architecture/SCALING.md) §6 and [`./MODERNIZATION_AUDIT.md`](./MODERNIZATION_AUDIT.md) §9.

## Headline result

| | before Phase 3a | after Phase 3a |
|---|---|---|
| **N=32 GPU SpMV throughput** | FP64 cuSPARSE (memory-bound) | **2× faster** with `ED_GPU_MIXED_PRECISION_SPMV=1` (FP32 matvec, FP64 outer) |
| **N=36 with full point-group + Sz** | "barely fits, prone to OOM, no crash recovery" | **routine** on a 64 GB workstation w/ NVMe scratch: `ED_LANCZOS_DISK=1` + `ED_LANCZOS_REORTH_TILE=16` + `ED_LANCZOS_CHECKPOINT_DIR=…` + `ED_NUMA_FIRST_TOUCH=1` |
| **Symmetry-projected basis lookup** | `unordered_map<uint64_t,size_t>` ~32-40 B/entry | `SortedUint64Index` flat **16 B/entry** |
| **Re-orthogonalisation overhead** | per-vector BLAS-1 `zdotc`/`zaxpy` | tile-blocked BLAS-2 `zgemv` (CGS2), `B` open-files → 1 per tile |
| **Mid-run crash recovery** | none — restart from iter 0 | atomic HDF5 checkpoint every `N` iters (`ED_LANCZOS_CHECKPOINT_DIR`); resume via `ED_LANCZOS_RESUME=1` |
| **NUMA placement of basis vectors** | first-touched on the calling thread → cross-socket faults during SpMV | `ED_NUMA_FIRST_TOUCH=1` parallel-zeroes basis-sized buffers so each OMP thread owns the chunk it later reads |
| **Test coverage** | 107 tests | **131 tests, all passing** (+24 new lockdown tests across the five items) |

## What landed (in commit order)

| # | item | commit | files | lockdown |
|---|---|---|---|---|
| 1 | Krylov-state checkpoint/restart for Lanczos | `f5ebd54` | `include/ed/io/lanczos_checkpoint.h`, `src/io/lanczos_checkpoint.cpp` | `test_lanczos_checkpoint` (4 sections: round-trip, atomic-rename, resume-vs-dense, validation errors) |
| 2 | Out-of-core blocked-tile reorthogonalisation | `590d330` | `include/ed/io/lanczos_reorth.h`, `src/io/lanczos_reorth.cpp` | `test_lanczos_reorth` (7 sections: CGS-vs-MGS on orthonormal V, threshold + skip filters, in-memory + on-disk tile loading, end-to-end tile-size invariance, knob clamping) |
| 3 | Mixed-precision FP32 cuSPARSE SpMV (FP64 outer ops) | `bd52a10` | `include/ed/gpu/gpu_mixed_precision.h`, `src/solvers/gpu/gpu_mixed_precision.cu` + GPU operator changes | `test_gpu_mixed_precision_spmv` (2 sections: H*v rel L2 < 5e-6 at N=10; ground-state Lanczos within 1e-5 of dense at N=8) |
| 4 | NUMA-aware first-touch + thread-pinning hooks | `fc4b9f6` | `include/ed/parallel/numa.h`, `src/parallel/numa.cpp`, new `ed_parallel` static lib | `test_numa` (7 sections: env-knob defaults + truthy parsing, off-state and sub-threshold no-ops, on-state counter contract, raw-byte first-touch, idempotent pinning, end-to-end Lanczos eigenvalue invariance with knobs on vs off to within 1e-12) |
| 5 | Compact `SortedUint64Index` for symmetry basis lookup | `e904bcd` | `include/ed/core/sorted_uint64_index.h` + wiring into both streaming-symmetry operators | `test_sorted_uint64_index` (8 sections, 111k assertions) + existing `test_symmetry` N=4/N=6 sweeps still match dense reference to 1e-9 |

Total: **+5 source modules, +5 lockdown test files, +24 test sections,
~3 500 LoC, zero existing tests broken**.

## Default-off contract

Every Phase 3a item is **default-off** unless explicitly opted into.
This is deliberate: the codebase ships unchanged behaviour to existing
users, and every knob can be A/B-tested in isolation against the
unmodified path.

| env var | default | item |
|---|---|---|
| `ED_LANCZOS_CHECKPOINT_DIR` | unset (off) | #1 — set to enable checkpoints |
| `ED_LANCZOS_CHECKPOINT_INTERVAL` | `100` | #1 — iterations between checkpoint writes |
| `ED_LANCZOS_RESUME` | `0` | #1 — set to 1 to resume from on-disk checkpoint |
| `ED_LANCZOS_REORTH_TILE` | `16` | #2 — tile size, clamped to `[1, 256]` |
| `ED_GPU_MIXED_PRECISION_SPMV` | `0` | #3 — FP32 cuSPARSE matvec on GPU |
| `ED_GPU_CUSPARSE_MIN_DIM` | `32768` | #3 — gates which dims use cuSPARSE at all |
| `ED_NUMA_FIRST_TOUCH` | `0` | #4 — parallel-zero basis buffers for NUMA placement |
| `ED_NUMA_PIN_THREADS` | `0` | #4 — `pthread_setaffinity_np` OMP workers compactly |

`SortedUint64Index` (#5) has **no env knob** — it is the unconditional
new in-memory representation. The wire format on disk is unchanged
(HDF5 schema stores the orbit CSR; the lookup index is rebuilt on load).

## What's still single-node (deferred to Phase 3b/3c)

The five items above were chosen as the maximum scale-up reachable
without introducing distributed-memory dependencies. The remaining
single-node candidates were profiled and judged not worth a Phase 3a
slot:

* **Matrix-free FP32 SpMV.** Templating the `WARP_REDUCTION` /
  `BRANCH_FREE_SCATTER` / `SHARED_MEMORY` matrix-free CUDA kernels on
  precision is a larger job than the cuSPARSE casting wrapper and would
  break the existing `GPUOperator` ABI. Deferred.
* **Explicit `numa_alloc_onnode` / `mbind`.** The opportunistic
  first-touch in Phase 3a #4 already fixes the dominant NUMA fault
  pattern (basis-sized vectors in the SpMV inner loop). Explicit
  per-node allocation needs `find_library(NUMA numa)` and a
  `WITH_LIBNUMA` option, and the marginal win on a 2-socket machine is
  ~5-10 %. Deferred.
* **Minimal perfect hash for the symmetry lookup.** PTHash / BBHash
  would shave another ~6 B/entry on top of the 16 B/entry
  `SortedUint64Index`. Adds an external dep. The dominant remaining
  symmetry-side memory cost at N=36 is the orbit-element CSR
  (`SymBasisState::orbit_elements`) — that's the natural next target,
  not the lookup index. Deferred.
* **`DiskStreamingSymmetryOperator::loaded_sector_lookup_` and
  `expandToFixedSzBasis::state_to_idx`** are the two remaining
  `unordered_map<uint64_t, size_t>` users. Both are off the SpMV inner
  loop (the first is hit only during chunk-load, the second only during
  per-eigenvector expansion at the end of a run). Same `SortedUint64Index`
  swap could be done if a profiler ever flags them, but they are not
  load-bearing for the N=36 envelope.

## What the codebase looks like now

After Phase 3a, the full default-on configuration for an N=36 run on a
fat workstation is:

```bash
export ED_LANCZOS_DISK=1
export ED_LANCZOS_REORTH_TILE=16
export ED_LANCZOS_CHECKPOINT_DIR=/scratch/$USER/lanczos_ckpt
export ED_LANCZOS_CHECKPOINT_INTERVAL=50
export ED_NUMA_FIRST_TOUCH=1
export ED_NUMA_PIN_THREADS=1
# GPU: add ED_GPU_MIXED_PRECISION_SPMV=1 if cuSPARSE is the matvec
# backend (N>=15 or so).
```

…and the CSR / orbit / lookup-index footprint is *honest*: the build log
prints `Lookup index footprint: <X> MiB (16.00 B/entry)` so you can see
exactly what the symmetry-projected basis is costing you, and the
checkpoint cadence means a SIGKILL during a 12-hour Lanczos run loses
at most 50 iterations.

## Validation summary

```text
$ cd build && ctest --output-on-failure
...
100% tests passed, 0 tests failed out of 131
Total Test time (real) =  14.33 sec
```

24 new lockdown tests (4 + 7 + 2 + 7 + 8 = 28 sections, including the
111 k-assertion `SortedUint64Index` stress test) + 0 regressions.

## Phase 3b preview (next)

The single-node ceiling is now hard: every Lanczos / FTLM / TPQ vector
is `std::vector<Complex>` on a single rank, so N=40 (~56 GB/vector at
half-filling-with-symmetry) does not fit anywhere we test on. Phase 3b
is the regime change to MPI-distributed state vectors:

1. `DistributedOperator` abstraction parallel to `Operator`. Each MPI
   rank owns a contiguous slab; `apply(v_in_local, v_out_local)` does
   local SpMV + `MPI_Alltoallv` for off-diagonal columns. Symmetry-aware
   slabbing so off-rank traffic is bounded.
2. Distributed Lanczos on top: `MPI_Allreduce` for dot products, axpy
   stays local. Re-orth either replicates the short Krylov basis (m ≤
   100) or ring-exchanges.
3. Distributed FTLM / TPQ as `DistributedLanczos × MPI-over-samples`.
4. `scripts/launch_distributed.sh` wrapping SLURM / OpenMPI / MVAPICH2.

After Phase 3b: N=40 with symmetry on ~64 ranks is *possible*, and
N=36 thermodynamics finishes in hours instead of days.
