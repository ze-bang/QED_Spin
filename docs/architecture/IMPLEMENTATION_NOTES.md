---
title: Implementation notes — deferred work and HPC-gated milestones
audience: future maintainers, HPC reviewers
status: design notes (no implementation)
last_updated: 2026-04-25
---

# Implementation notes

This file is the **single canonical landing pad** for work that is
*designed and scoped* but **not implemented** in the released codebase.
Each item below has a clear motivation, a concrete API sketch, and the
hardware / external-library prerequisites that gate it. The items live
here (rather than in `MODERNIZATION_AUDIT.md` or as scattered TODOs)
so a future contributor with HPC access can pick them up cold.

Everything here was descoped in the v1.0 release because it required
either:

1. **Multi-node / multi-GPU HPC time** that we have not booked, or
2. **Non-trivial design work on a 2-week-plus horizon** that is best
   done with a real production workload to validate against (not
   single-node toy tests).

If you start one of these, please also update the corresponding section
in `docs/architecture/SCALING.md` §6 from "OPEN" to "IN PROGRESS"
and link the PR.

---

## Table of contents

1. [Phase 3b #7 — symmetry-aware row partitioning](#phase-3b-7--symmetry-aware-row-partitioning)
2. [Phase 3c #1 — NCCL multi-GPU Lanczos](#phase-3c-1--nccl-multi-gpu-lanczos)
3. [Phase 3c #2 — GPU-Direct RDMA halo exchange](#phase-3c-2--gpu-direct-rdma-halo-exchange)
4. [Phase 3c #3 — distributed disk-backed Krylov basis](#phase-3c-3--distributed-disk-backed-krylov-basis)
5. [Phase 3c #4 — published 40-site validation against HΦ](#phase-3c-4--published-40-site-validation-against-h)
6. [Smaller deferred items](#smaller-deferred-items)

---

## Phase 3b #7 — symmetry-aware row partitioning

**Status:** design only. No code.
**Gates the milestone:** "honest 40 routine on a real cluster".
**Hardware needed for validation:** 16+ MPI ranks on a real
high-bandwidth cluster (Infiniband or equivalent), to actually exercise
`MPI_Alltoallv` halo cost as a function of partition geometry.

### Why we need it

The current `ed::distributed::DistributedOperator::balanced_slab`
splits the **unsymmetrised** basis evenly across ranks: rank `r` owns
indices `[r * N/P, (r+1) * N/P)` of a `2^N`-dimensional Hilbert space.
At small N this is fine and is what `test_distributed_operator` locks
down on N ∈ {4,6,8} across np ∈ {1,2,4}.

At "honest 40" with full point-group + Sz, the natural basis is the
*projected* one (orbits of the symmetry group), and orbit sizes can
vary by an order of magnitude. A naïve 1D row split of the projected
basis will then either:

- **Imbalance compute** (some ranks own 10× more orbits than others), or
- **Inflate halo exchange** (each Hamiltonian term that maps within an
  orbit but crosses a partition boundary requires an extra Alltoallv
  byte, and the worst-case cross-rank coupling for nearest-neighbour
  Heisenberg can approach `O(local_n)` rather than `O(local_n / N)`
  if the boundaries cut through high-multiplicity orbits).

### Sketch of the design

The right partitioning is a **graph-partitioning** problem on the
basis-overlap graph `G = (V, E)` where `V` are projected basis indices
and `E[i, j]` is non-zero iff some Hamiltonian term couples basis `i`
and basis `j`. Optimal partitions minimise edge-cut for fixed
vertex-balance. Standard tools:

- **METIS / ParMETIS** — partitions a static graph; offline.
- **Zoltan** (Trilinos) — partitions with balance / cut trade-off
  knobs; runtime.

For Heisenberg-class Hamiltonians, the basis-overlap graph is
extremely sparse (each row has `2 N` non-zeros for nearest-neighbour
terms), so even simple **Kahn-style topological orbit packing** with
a 5–10% imbalance budget should beat the naïve 1D split substantially.
Best to instrument the existing `DistributedOperator` first
(see "Smaller deferred items #1") to confirm the halo size *is* in
fact the bottleneck before hauling in a graph-partitioning library.

### Proposed API

```cpp
namespace ed::distributed {

// Pluggable partitioner. The default is the existing balanced_slab.
struct Partitioner {
    virtual ~Partitioner() = default;

    // Given the projected basis (or its orbit representatives) and
    // the comm size, return the (global_dim,)-vector mapping each
    // basis index to an owning rank.
    virtual std::vector<int> assign_owners(
        std::span<const std::uint64_t> orbit_reps,
        const Operator& H,
        int comm_size) = 0;
};

class BalancedSlabPartitioner : public Partitioner { /* current code */ };

class MetisPartitioner : public Partitioner {
public:
    explicit MetisPartitioner(double imbalance_budget = 0.05);
    std::vector<int> assign_owners(
        std::span<const std::uint64_t>,
        const Operator&, int) override;
};

// Existing constructor gets a defaulted partitioner argument.
DistributedOperator(std::shared_ptr<Operator> op,
                    MPI_Comm comm,
                    std::unique_ptr<Partitioner> p =
                        std::make_unique<BalancedSlabPartitioner>());

}  // namespace ed::distributed
```

### Test plan

1. **Unit test** (CPU, no cluster needed): on a small Heisenberg N=8
   PBC chain with full translation symmetry, compare the naïve 1D
   partition's halo size vs the METIS partition's. Assert
   `metis_halo_bytes < 0.6 * balanced_slab_halo_bytes` — the exact
   ratio depends on N and connectivity but should be under 60% for any
   N ≥ 6.
2. **Cluster lockdown** (when HPC time available): run N=20 PBC
   Heisenberg with full point-group on np=16,32,64 ranks; assert
   `Allreduce_time + Alltoallv_time` scales sub-linearly with `np` for
   the METIS partition and linearly for the naïve 1D partition.
3. **Bit-identical reproducibility**: the METIS partition's
   `distributed_lanczos` ground-state energy must match the
   `BalancedSlabPartitioner` energy to `1e-12` (since both compute the
   same matrix in different bases internally; the tridiagonal form is
   invariant under the partition).

### Estimated effort

~1 person-week for the graph-build, ~1 person-week for METIS
integration + tests, ~1 person-week for the cluster validation pass.
Total: ~3 weeks of focused work + actual HPC time.

---

## Phase 3c #1 — NCCL multi-GPU Lanczos

**Status:** detection-only stub at
`include/ed/distributed/multi_gpu_stub.h`. CMake discovers NCCL when
present (`NCCL_FOUND`, `NCCL_INCLUDE_DIRS`, `NCCL_LIBRARIES`,
`ED_HAVE_NCCL` propagated), but no kernels exist.
**Gates the milestone:** "fast 36 → 40 on a single 8-GPU node".
**Hardware needed:** ≥2 NVIDIA GPUs with NVLink or NVSwitch (ideally a
DGX-class node), NCCL ≥2.18 installed, CUDA ≥12.

### Why we need it

`ed::distributed::distributed_lanczos` today does its dot products and
norms via `MPI_Allreduce`. On a single multi-GPU node, that means:

1. Stage local result from device → host (PCIe DMA),
2. Cross-rank MPI all-reduce on host buffers,
3. Stage result host → device (PCIe DMA),
4. Use the result in the next BLAS-1 call on the device.

Steps 1 and 3 dominate at multi-GB rank-local state vectors. NCCL
(`ncclAllReduce` on device pointers, GPU-to-GPU via NVLink, never
touches host memory) collapses this to a single async device call.

### Sketch of the design

`include/ed/distributed/multi_gpu.h` (replacing the current stub)
exposes a thin wrapper:

```cpp
namespace ed::distributed::multi_gpu {

class NcclContext {
public:
    NcclContext(MPI_Comm comm, int device_id);
    ~NcclContext();

    // Sum-reduce `count` doubles in `dev_buf` across all ranks in
    // `comm`. dev_buf MUST be device-resident.
    void allreduce_sum(double* dev_buf, std::size_t count,
                       cudaStream_t stream = 0) const;

    // Same for complex<double>.
    void allreduce_sum(std::complex<double>* dev_buf,
                       std::size_t count,
                       cudaStream_t stream = 0) const;
};

}  // namespace ed::distributed::multi_gpu
```

`distributed_lanczos.cpp` then has a compile-time `if constexpr`
branch: when both `ED_HAVE_NCCL` is defined and the user passes
`DistributedLanczosOptions::use_nccl = true`, the inner loop's
`MPI_Allreduce` of `dot_local` gets replaced by
`NcclContext::allreduce_sum`. Everything else (Krylov basis, axpy,
scal, tridiagonal eigensolve) is unchanged.

### Test plan

1. **Single-rank, single-GPU smoke** (no cluster needed): with
   `ED_HAVE_NCCL=ON` and 1 MPI rank + 1 GPU, NCCL allreduce-sum of a
   degenerate 1-rank communicator must be a no-op; the result must be
   bit-identical to the MPI path on N=4 OBC.
2. **2-GPU lockdown**: on a 2-GPU node, `distributed_lanczos` ground
   state on N=12 PBC must match the 2-rank MPI path to `1e-12`.
3. **Performance**: on the 2-GPU node, measure
   `dot_product_time / matvec_time` for the MPI vs NCCL paths at
   N=20, dim ≈ 1M. NCCL should be ≥3× faster.

### Estimated effort

~1 person-week for `NcclContext` + the `distributed_lanczos` branch,
~1 person-week for `distributed_ftlm` and `distributed_tpq` to follow
suit, ~1 week of cluster validation + benchmarking.

---

## Phase 3c #2 — GPU-Direct RDMA halo exchange

**Status:** not started.
**Gates the milestone:** "honest 44–48 across a real cluster".
**Hardware needed:** RDMA-capable Infiniband fabric, `nv_peer_mem` or
GPUDirect RDMA enabled in the kernel, NCCL ≥2.18 with RDMA support.

### Why we need it

Once the Lanczos inner products are NCCL-resident (Phase 3c #1), the
remaining bottleneck on a multi-node multi-GPU run is the
`MPI_Alltoallv` of the halo state-vector entries inside each
`DistributedOperator::apply` call. Today this is staged through host
memory as well (device → host on the sender, MPI Alltoallv host →
host across the IB fabric, host → device on the receiver). With
GPU-Direct RDMA, the network adapter DMAs straight from the source
GPU's HBM to the target GPU's HBM via PCIe peer-to-peer + RDMA verbs,
bypassing host memory entirely.

### Design pointer

`DistributedOperator::apply` currently calls
`MPI_Alltoallv(MPI_C_DOUBLE_COMPLEX, ...)`. With GPU-Direct,
the right primitive is `ncclSendRecv` in a grouped call (NCCL ≥2.7
supports general send/recv with arbitrary tag), which on RDMA
fabrics will route via Infiniband verbs and stay device-resident.

The migration is two-step:
1. Stage the halo plan onto the device (the per-pair `sendcounts`,
   `recvcounts`, and `displs`).
2. Replace the Alltoallv with `ncclGroupStart` / `ncclSend` /
   `ncclRecv` / `ncclGroupEnd`, all on device pointers.

### Test plan

Functional lockdown vs the host-staged MPI path on a 2-node 4-GPU
cluster; performance lockdown of `apply()` wall-time at N=24 across
single-node-MPI vs single-node-NCCL vs 2-node-NCCL.

### Estimated effort

~2 person-weeks of focused work + actual HPC time on a real RDMA
cluster (development without that fabric is essentially impossible
because functional bugs in the device-resident halo plan only surface
under real RDMA verbs).

---

## Phase 3c #3 — distributed disk-backed Krylov basis

**Status:** not started.
**Gates the milestone:** "honest 44 with eigenvectors / dynamics".
**Hardware needed:** parallel filesystem (Lustre, GPFS, or equivalent)
mounted on every compute node, with measured aggregate write bandwidth
≥10 GB/s for the basis-flush path to keep up with compute.

### Why we need it

`m = 200` Lanczos at N = 44 means
`m × dim_local × P × 16 B = 200 × 750 GB ≈ 150 TB` of basis. No node
fits this in memory; the basis has to live on a parallel filesystem
and be tile-loaded for re-orthogonalisation.

Single-node disk-backed Krylov *is* shipped (`ED_LANCZOS_DISK=1` in
Phase 3a #2 — see `src/io/lanczos_reorth.cpp`); the missing piece is
the *distributed* version where each rank streams its own slab to and
from the parallel filesystem. The right tool here is parallel HDF5 +
MPI-IO with one shared `H5F_ACC_TRUNC` file per Krylov vector and one
chunked dataset per rank.

### Design pointer

`include/ed/io/distributed_lanczos_basis.h` should hide the parallel
HDF5 boilerplate behind the same in-memory interface that
`distributed_lanczos.cpp` already uses. The on-disk schema:

```
/krylov_basis
  /vector_0000  H5T_NATIVE_DOUBLE_COMPLEX  shape = (global_dim,)
  /vector_0001  ...
```

each `vector_*` dataset is chunked along axis 0 with `local_n`-sized
chunks, so each rank reads/writes only its own chunk.

### Test plan

Functional lockdown vs the in-memory path on a Lustre scratch with 2
ranks at N=10 PBC. Performance lockdown of write-bandwidth utilisation
on the actual cluster (target: ≥80% of measured aggregate write
bandwidth).

### Estimated effort

~2 person-weeks of focused work; not blocked on hardware because the
single-node parallel-HDF5 path can be developed and unit-tested on
any laptop with HDF5 ≥1.12.

---

## Phase 3c #4 — published 40-site validation against HΦ

**Status:** not started.
**Gates the milestone:** any external claim that this codebase is in
the same league as HΦ.

Before anyone publishes a 40-site result computed with this codebase,
we need a reproducible head-to-head against
[**HΦ**](https://github.com/issp-center-dev/HPhi) on a published
40-site benchmark. HΦ is the de-facto reference distributed-ED
implementation; until we cross-check ground-state energies (and at
least one finite-T `<H>(β)`) against HΦ on a 40-site spin-1/2 model,
we should claim only what the existing N≤8 lockdown supports.

The test harness should live under `benchmarks/vs_hphi/` and produce
a JSON artefact analogous to today's `bench_vs_quspin_results.json`.

---

## Smaller deferred items

These do not block any milestone but are worth picking up opportunistically.

1. **Halo-size instrumentation in `DistributedOperator`.** Add
   `DistributedOperator::halo_bytes_per_apply()` that returns the total
   number of bytes shuffled in one `MPI_Alltoallv` for the current
   partition. Useful for empirically validating Phase 3b #7 before
   landing METIS.

2. **`distributed_ftlm` over multiple observables in a single sweep.**
   Today, computing `<O>(β)` for two operators `O1` and `O2` requires
   two full FTLM sweeps. Caching the Krylov basis between calls would
   share the cost. Touching point: `DistributedFtlmOptions` would gain
   a `std::vector<std::shared_ptr<Operator>> observable_ops` instead of
   the current single `observable_op`.

3. **TPQ continuation-fraction DSSF.** `distributed_tpq` produces
   `|ψ(β)>` rank-locally; computing `<ψ(β)| (ω - H + iη)^-1 O |ψ(β)>`
   via the Mori continued-fraction expansion is the natural "honest 40"
   replacement for FTLM-DSSF. Needs a distributed Lanczos that takes a
   user-supplied initial vector — a 5-line change to
   `distributed_lanczos.cpp`.

4. **Eigenvector reconstruction memory budget.** Setting
   `compute_eigenvectors = true` on `DistributedLanczosOptions` retains
   `m × local_n × 16 B` per rank. At "honest 40" with `m = 200` and
   `local_n ≈ 5e9` that is ~16 TB per rank — way outside any sane
   budget. The right fix is a streaming reconstruction that re-runs
   Lanczos with the seed vector, replays the basis, and writes one
   eigenvector to disk at a time. Needs the disk-backed basis from
   Phase 3c #3.

5. **Replace the deprecated `createSymmetrizedVectorFixedSz` call site
   in `construct_ham.h`.** It is the only `[[deprecated]]` warning in
   the build; not load-bearing.

6. **A clang-format / clang-tidy pass on `src/distributed/` and
   `tests/unit/test_distributed_*.cpp`.** Items written under the
   Phase 3b time pressure inherited the Phase 2 style but were not
   re-formatted. ~30 min of `pre-commit run --all-files`.
