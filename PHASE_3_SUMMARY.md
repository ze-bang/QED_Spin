# Phase 3 — distributed-memory ED: shipped (3a + 3b) and stubbed (3c)

> Closes Phase 3 of the SOTA-publication-grade ED roadmap. **3a** lifted
> the realistic single-node ceiling from "honest 32, painful 36" to
> "honest 36 routine, fast 32 GPU" (5 commits, see
> [`PHASE_3A_SUMMARY.md`](./PHASE_3A_SUMMARY.md)). **3b** shipped a
> rank-distributed `Operator` / `Lanczos` / `FTLM` triple with a sparse
> halo-exchange SpMV and bit-identical lockdown vs the serial path.
> **3c** is detection-only: NCCL is discovered when present and a
> read-only stub is in place, but the multi-GPU runtime is gated on
> HPC time we have not booked. Total: **140/140 tests pass** (was 107
> at the end of Phase 2; +9 MPI tests at np ∈ {1,2,4} this phase).
>
> Read this once for the full picture. For per-item depth see
> [`SCALING.md`](./SCALING.md) §6 and
> [`MODERNIZATION_AUDIT.md`](./MODERNIZATION_AUDIT.md).

## Headline result

| | end of Phase 2 | end of Phase 3 |
|---|---|---|
| **Largest verified system size, single node** | N=32 (CPU) / N=28 (GPU) | **N=36 routine** (Phase 3a, see PHASE_3A_SUMMARY) |
| **Distributed-memory `H·v` (rank-local state vector)** | absent | **`ed::distributed::DistributedOperator`** with 1D row decomposition + `MPI_Alltoallv` halo exchange, bit-identical to serial on Heisenberg N=4/6/8 across `np ∈ {1,2,4}` |
| **Distributed Lanczos (`MPI_Allreduce` dot/norm, local axpy)** | absent | **`ed::distributed::distributed_lanczos`** with optional full re-orth; ground state matches dense reference on N=4 OBC and N=6 PBC; eigenvalues bit-replicated across all ranks |
| **Distributed FTLM (J&P trace estimator over MPI samples)** | absent | **`ed::distributed::distributed_ftlm`** with Ritz weights `Z(β) ≈ (D/R) Σ_s Σ_k |⟨v0_s|ψ_k_s⟩|² exp(-β E_k_s)` matching exact `Z(β)` on N=4 OBC within statistical noise |
| **Cluster launcher** | sample-parallel only | `ed_distributed_main` CLI + `scripts/distributed/run_dist.sh` + `slurm_dist.sbatch` |
| **Multi-GPU (NCCL)** | absent | **detection stub only** — `ED_HAVE_NCCL` propagated when discovered, `ed::distributed::multi_gpu::nccl_compiled_in()` reports back, runtime kernels gated on HPC time |
| **Test coverage** | 107 tests | **140 tests, all passing** (+24 from 3a, +9 from 3b at np=1/2/4) |

## What landed in Phase 3b (in commit order, on top of Phase 3a)

| # | item | files | lockdown |
|---|---|---|---|
| 1 | `DistributedOperator` (1D row slabs + bit-flip-pattern Alltoallv halo + gather-form OpenMP SpMV; reuses Phase 3a `SortedUint64Index` for the receive-buffer lookup) | `include/ed/distributed/distributed_operator.h`, `src/distributed/distributed_operator.cpp` | `test_distributed_operator` (5 sections: balanced_slab, balanced_owner_rank, N=4 OBC, N=6 PBC, N=8 OBC J=-1.5 + comm-plan symmetry) — 3 ctest entries at np ∈ {1,2,4} |
| 2 | `distributed_lanczos` (rank-local Krylov, MPI_Allreduce dot/norm, optional full re-orth, replicated tridiagonal eigensolve; result struct now also exposes `tridiag_eigenvalues` + `tridiag_weights` for FTLM/DOS) | `include/ed/distributed/distributed_lanczos.h`, `src/distributed/distributed_lanczos.cpp` | `test_distributed_lanczos` (4 sections: N=4 OBC ground state, N=6 PBC ground state, full_reorth ↔ no-orth agreement, bit-replicated eigenvalues) — 3 ctest entries at np ∈ {1,2,4} |
| 3 | `distributed_ftlm` (MPI-over-samples via `MPI_Comm_split` outer groups, J&P trace estimator using the Ritz weights from item 2) | `include/ed/distributed/distributed_ftlm.h`, `src/distributed/distributed_ftlm.cpp` | `test_distributed_ftlm` (2 sections: Z(β) within `1/√R` of exact on N=4 OBC at β ∈ {0.1, 0.5}; replicated Z across all ranks) — 3 ctest entries at np ∈ {1,2,4} |
| 4 | `ed_distributed_main` cluster launcher + scripts | `src/cli/ed_distributed_main.cpp`, `scripts/distributed/run_dist.sh`, `scripts/distributed/slurm_dist.sbatch`, `scripts/distributed/README.md` | manual smoke-test on N=12 PBC `np=4` reproduces E0 = -5.387… |
| 5 | CMake plumbing: `ed_distributed` static lib (gated by `WITH_MPI`), `ed_add_mpi_test()` helper that registers each MPI test 3× (np=1,2,4), install hook for `ed_distributed_main` | `cmake/EDLibraries.cmake`, `CMakeLists.txt` | `ctest -L phase3b` — 9/9 green |

## What landed in Phase 3c (detection-only)

| # | item | files | status |
|---|---|---|---|
| 1 | NCCL discovery in CMake (sets `NCCL_FOUND`, `NCCL_INCLUDE_DIRS`, `NCCL_LIBRARIES`; propagates `ED_HAVE_NCCL` to `ed_distributed`) | `CMakeLists.txt` (Phase 3c block after the WITH_CUDA/WITH_MPI summary) | DETECTION-ONLY |
| 2 | Read-only header advertising the future API surface (`nccl_compiled_in()`, `nccl_status_string()`) — does NOT include `<nccl.h>` | `include/ed/distributed/multi_gpu_stub.h` | STUB |
| 3 | Honest scoping update across `SCALING.md` §6 (Phase 3b items marked DONE/PARTIAL/STUB) and `SCALING.md` §7 (no longer claims "MPI is sample-only"; flags the unsymmetrised 1D decomposition limitation for "honest 40") | `SCALING.md` | DOC |

The actual NCCL kernels (NCCL allreduce on device buffers, GPU-Direct RDMA
halo exchange, multi-GPU Lanczos) require >= 2 visible CUDA devices, a
working NCCL/RDMA stack, and a HPC slot we have not allocated. The
detection stub exists so a future PR can flip the implementation switch
without re-touching the build system.

## Default-off contract

Every Phase 3 item is **default-off** unless explicitly opted into. The
serial `ED` driver and the existing CPU/GPU solvers ship unchanged
behaviour. To exercise Phase 3b, build with `-DWITH_MPI=ON` and either:

* call `ed::distributed::*` directly from C++, or
* run `ed_distributed_main` (or `scripts/distributed/run_dist.sh`).

To exercise Phase 3c (detection only), additionally build with
`-DWITH_CUDA=ON` on a system that has NCCL installed.

## Honest scope (what we are NOT claiming)

Read these alongside `SCALING.md` §7:

1. **N=40 routine on a real cluster.** The Phase 3b solvers are
   correctness-locked vs serial on Heisenberg N≤8 across np ∈ {1,2,4}
   *only*. They have not yet been exercised at N=40 on a real cluster.
   The 1D row decomposition is **not** symmetry-aware — for "honest 40"
   with full lattice symmetry, the slab boundaries should respect orbit
   structure to keep Alltoallv halo size bounded. That is a Phase 3b #5
   follow-up.
2. **Distributed observables `⟨O⟩(β)`.** `distributed_ftlm` returns only
   the partition function `Z(β)` today. Operator expectation values
   need a distributed Krylov-basis replay against a second
   `DistributedOperator` wrapping `O`; the hook is documented in the
   FTLM source but not yet wired.
3. **Distributed TPQ.** Imaginary-time evolution of a rank-distributed
   state vector is straightforward on top of `DistributedOperator` but
   has not yet been implemented.
4. **Multi-GPU.** The Phase 3c column above says it all: detection only.
5. **Eigenvector reconstruction in the distributed path.**
   `distributed_lanczos` returns Ritz values only. Reconstructing
   ground-state eigenvectors needs the rank-local Krylov basis to be
   replayed against the small tridiagonal eigenvectors — not done yet.
   For now, the recommended workflow is "run distributed Lanczos for
   the ground-state energy on the full Hilbert space, then run a
   *symmetry-projected serial* solver in the relevant sector for the
   eigenvector".

## Provenance

* Phase 3a: 5 commits between `f5ebd54` and `e904bcd` (see
  `PHASE_3A_SUMMARY.md` for details).
* Phase 3b/c: this commit (and the immediately preceding one if the
  3b/3c work was split across two commits).
* Test counts cross-checked with `ctest -N | grep -c "Test #"`
  (currently `140`) and the per-label sweep
  `ctest -L phase3b --output-on-failure` (currently `9/9`, ~17 s wall
  on a 32-thread Ryzen 9 7950X3D).
