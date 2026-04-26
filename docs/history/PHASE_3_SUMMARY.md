---
orphan: true
---

# Phase 3 — distributed-memory ED: 3a + 3b + 3c (stages 1-3) shipped

> Closes Phase 3 of the SOTA-publication-grade ED roadmap. **3a** lifted
> the realistic single-node ceiling from "honest 32, painful 36" to
> "honest 36 routine, fast 32 GPU" (5 commits, see
> [`PHASE_3A_SUMMARY.md`](./PHASE_3A_SUMMARY.md) — this directory). **3b** shipped a
> rank-distributed solver suite: `DistributedOperator`,
> `distributed_lanczos` (energies AND eigenvectors), `distributed_ftlm`
> (Z(β) AND `<O>(β)`), and `distributed_tpq` (canonical TPQ via
> Taylor-truncated `e^{-(δβ/2) H}`), all bit-identically locked down
> against the serial path on Heisenberg N ≤ 6 across `np ∈ {1, 2, 4}`.
> **3c** ships a real multi-GPU runtime: `MultiGpuCommunicator` (RAII
> NCCL over MPI), `distributed_lanczos_gpu` (GPU-resident Krylov +
> `cublasZdotc` + `ncclAllReduce` on device buffers), and
> `DistributedGPUOperator` (fully GPU-resident SpMV with NCCL pairwise
> SendRecv halo) — all locked down vs the CPU reference on real H100
> hardware at `np ∈ {1, 2, 4}` on rorqual `gpubase_bygpu_b1`.
>
> Read this once for the full picture. For per-item depth see
> [`../architecture/SCALING.md`](../architecture/SCALING.md) §6 and
> [`./MODERNIZATION_AUDIT.md`](./MODERNIZATION_AUDIT.md) (this directory).

## Headline result

| | end of Phase 2 | end of Phase 3 |
|---|---|---|
| **Largest verified system size, single node** | N=32 (CPU) / N=28 (GPU) | **N=36 routine** (Phase 3a, see PHASE_3A_SUMMARY) |
| **Distributed-memory `H·v` (rank-local state vector)** | absent | **`ed::distributed::DistributedOperator`** with 1D row decomposition + `MPI_Alltoallv` halo exchange, bit-identical to serial on Heisenberg N=4/6/8 across `np ∈ {1,2,4}` |
| **Distributed Lanczos (`MPI_Allreduce` dot/norm, local axpy)** | absent | **`ed::distributed::distributed_lanczos`** with optional full re-orth; ground state matches dense reference on N=4 OBC and N=6 PBC; eigenvalues bit-replicated across all ranks |
| **Distributed FTLM (J&P trace estimator over MPI samples)** | absent | **`ed::distributed::distributed_ftlm`** with Ritz weights `Z(β) ≈ (D/R) Σ_s Σ_k |⟨v0_s|ψ_k_s⟩|² exp(-β E_k_s)` matching exact `Z(β)` on N=4 OBC within statistical noise; **`<O>(β)` via canonical J&P** by passing `observable_op` (cross-checks against exact thermal `<H>`) |
| **Distributed eigenvectors** | absent | **`ed::distributed::distributed_lanczos_eigenvectors`** reconstructs rank-local Ritz vectors via `ψ_k_local = V_local @ U[:,k]`; `‖H ψ - E ψ‖₂ < 1e-8` lockdown after `MPI_Allgatherv` assembly on N=4 OBC + N=6 PBC |
| **Distributed canonical TPQ** | absent | **`ed::distributed::distributed_tpq`** propagates `\|ψ(β)>` via Taylor-truncated `e^{-(δβ/2) H}` on rank-local slabs; `<H>(β)` and `<H²>(β) - <H>(β)²` cross-check vs exact thermal energy on N=4 OBC |
| **Cluster launcher** | sample-parallel only | `ed_distributed_main` CLI + `scripts/distributed/run_dist.sh` + `slurm_dist.sbatch` |
| **Multi-GPU (NCCL)** | absent | **runtime shipped** — `MultiGpuCommunicator` (RAII NCCL over MPI), `distributed_lanczos_gpu` (GPU-resident Krylov + `cublasZdotc` + `ncclAllReduce`), `DistributedGPUOperator` (fully GPU-resident SpMV + `ncclSendRecv` pairwise halo); locked down on real H100×{1,2,4} on rorqual |
| **Symmetry-aware distributed `H · v` on the projected basis** | absent | **`ed::distributed::DistributedSymmetryOperator`** — orbit-respecting LPT-balanced row partition + orbit-aware `MPI_Alltoallv` halo + rank-local sparse SpMV; correctness-locked vs an independent serial dense projected reference on every momentum sector of Heisenberg `(N=4 OBC, N=4 PBC, N=6 PBC)` at `np ∈ {1,2,4}` (82 assertions / 4 cases PASS). Closes the load-bearing scaling bottleneck for "honest 40" with full point-group + Sz. |
| **Symmetry-aware distributed Lanczos** | absent | **`ed::distributed::distributed_lanczos_symmetry`** — distributed Lanczos rebased onto `DistributedSymmetryOperator` via a templated header-only `distributed_lanczos_kernel<OpT>` shared with the unsymmetrised path. Ground-state energy locked to a dense `Eigen::SelfAdjointEigenSolver` reference on every momentum sector of Heisenberg `(N=4 OBC, N=4 PBC, N=6 PBC)` at `np ∈ {1,2,4}` within `1e-8`. |
| **Test coverage** | 107 tests | **170 ctest entries** (`ctest -N`), all passing (+24 from 3a, +15 from 3b at np=1/2/4, +3 from `DistributedSymmetryOperator` and +3 from `distributed_lanczos_symmetry` at np=1/2/4, plus the orbit_partition / orbit_halo_plan / multi_gpu_nccl / distributed_lanczos_gpu / distributed_gpu_operator entries from 3c at np=1/2/4) |

## What landed in Phase 3b (in commit order, on top of Phase 3a)

| # | item | files | lockdown |
|---|---|---|---|
| 1 | `DistributedOperator` (1D row slabs + bit-flip-pattern Alltoallv halo + gather-form OpenMP SpMV; reuses Phase 3a `SortedUint64Index` for the receive-buffer lookup) | `include/ed/distributed/distributed_operator.h`, `src/distributed/distributed_operator.cpp` | `test_distributed_operator` (5 sections: balanced_slab, balanced_owner_rank, N=4 OBC, N=6 PBC, N=8 OBC J=-1.5 + comm-plan symmetry) — 3 ctest entries at np ∈ {1,2,4} |
| 2 | `distributed_lanczos` (rank-local Krylov, MPI_Allreduce dot/norm, optional full re-orth, replicated tridiagonal eigensolve; result struct now also exposes `tridiag_eigenvalues` + `tridiag_weights` for FTLM/DOS) | `include/ed/distributed/distributed_lanczos.h`, `src/distributed/distributed_lanczos.cpp` | `test_distributed_lanczos` (4 sections: N=4 OBC ground state, N=6 PBC ground state, full_reorth ↔ no-orth agreement, bit-replicated eigenvalues) — 3 ctest entries at np ∈ {1,2,4} |
| 3 | `distributed_ftlm` (MPI-over-samples via `MPI_Comm_split` outer groups, J&P trace estimator using the Ritz weights from item 2; **#5 add-on**: optional `observable_op` triggers the canonical J&P formula `<O>(β) = N_O / N_Z` with `N_O = Σ_s Re(Σ_j q_j(s)* f_j(s,β))`, `q_j = <V_s[j] | O r_s>`, `f_j(β) = Σ_k U[j,k] U[0,k] e^{-β E_k}`) | `include/ed/distributed/distributed_ftlm.h`, `src/distributed/distributed_ftlm.cpp` | `test_distributed_ftlm` (4 sections: Z(β) within `1/√R` of exact on N=4 OBC at β ∈ {0.1, 0.5}; replicated Z; **`<H>(β)` within stat noise of exact thermal energy at β ∈ {0.1, 0.5, 1.0}**; replicated `<O>` across ranks) — 3 ctest entries at np ∈ {1,2,4} |
| 4 | `ed_distributed_main` cluster launcher + scripts | `src/cli/ed_distributed_main.cpp`, `scripts/distributed/run_dist.sh`, `scripts/distributed/slurm_dist.sbatch`, `scripts/distributed/README.md` | manual smoke-test on N=12 PBC `np=4` reproduces E0 = -5.387… |
| 5 | CMake plumbing: `ed_distributed` static lib (gated by `WITH_MPI`), `ed_add_mpi_test()` helper that registers each MPI test 3× (np=1,2,4), install hook for `ed_distributed_main` | `cmake/EDLibraries.cmake`, `CMakeLists.txt` | `ctest -L phase3b` — 15/15 green |
| 6 | **#6 distributed eigenvector reconstruction**: `DistributedLanczosOptions::compute_eigenvectors` retains the rank-local Krylov basis `V_local` AND the (m × m) tridiagonal eigenvector matrix `U`; `reconstruct_local_eigenvector(result, k, ψ_k_local)` does `ψ_k_local = V_local @ U[:,k]`; convenience `distributed_lanczos_eigenvectors(...)` returns `n_keep` smallest eigenpairs with rank-local Ritz vectors | `include/ed/distributed/distributed_lanczos.h`, `src/distributed/distributed_lanczos.cpp` | `test_distributed_eigenvectors` (3 sections: `‖H ψ - E ψ‖₂ < 1e-8` after Allgatherv on N=4 OBC; same on N=6 PBC; assembled global ψ replicated to within `|<ψ_rank0|ψ_local>| - 1| < 1e-10` across ranks) — 3 ctest entries at np ∈ {1,2,4} |
| 8 | **#8 distributed canonical TPQ**: `distributed_tpq(op, options, world_comm)` propagates a rank-local `|ψ(β)>` via Taylor-truncated `e^{-(δβ/2) H}` (DistributedOperator matvecs + local zaxpy + dist-norm renormalisation), measures `E(β) = <ψ\|H\|ψ>` and (optionally) variance at every entry of `options.betas`. Same outer/inner MPI parallelism as `distributed_ftlm` (`n_groups` × ranks-per-group). | `include/ed/distributed/distributed_tpq.h`, `src/distributed/distributed_tpq.cpp` | `test_distributed_tpq` (3 sections: `<H>(β)` matches exact thermal energy at β ∈ {0.5, 2.0} on N=4 OBC within `0.6` (R=8 stat noise on D=16); E(β) replicated across ranks; `E(β=0)` ≈ `Tr(H)/D = 0` within R=16 stat noise) — 3 ctest entries at np ∈ {1,2,4} |

## What landed in Phase 3c (stage 1: NCCL collectives)

| # | item | files | status |
|---|---|---|---|
| 1 | NCCL discovery in CMake (`-DWITH_NCCL=ON` by default; sets `NCCL_FOUND`, `NCCL_INCLUDE_DIRS`, `NCCL_LIBRARIES`). Now run **before** `EDLibraries.cmake` so a downstream library can be added conditionally. | `CMakeLists.txt` (NCCL block above `include(EDLibraries)`) | DONE |
| 2 | `ed_distributed_gpu` static library — only built when `WITH_MPI && WITH_CUDA && NCCL_FOUND`. Carries the `multi_gpu.cu` TU + `ED_HAVE_NCCL=1` PUBLIC compile-def. CPU-only builds remain pure CXX (no CUDA toolchain dependency). | `cmake/EDLibraries.cmake` | DONE |
| 3 | `MultiGpuCommunicator` — RAII wrapper around `ncclComm_t` built collectively from an `MPI_Comm` (rank 0 generates `ncclUniqueId`, broadcasts via `MPI_Bcast`, every rank `cudaSetDevice`s its node-local-rank-mod-#GPUs slot then `ncclCommInitRank`s). Move-only; throws on cuda/nccl/MPI failure. | `include/ed/distributed/multi_gpu.h`, `src/distributed/multi_gpu.cu` | DONE |
| 4 | NCCL collective wrappers: `all_reduce_sum_double` (norm/dot reductions), `all_reduce_sum_complex_double` (treats `complex<double>` as 2N doubles since `ncclSum` is element-wise), `broadcast_double`, `synchronize_stream`. Each takes a `MultiGpuCommunicator&`, raw device pointer, count, and optional `cudaStream_t`. | same as #3 | DONE |
| 5 | `multi_gpu_stub.h` — back-compat shim that now just `#include <ed/distributed/multi_gpu.h>`. Any out-of-tree caller stays valid. | `include/ed/distributed/multi_gpu_stub.h` | SHIM |
| 6 | `ed_distributed_main` prints `nccl_status_string()` in its banner so cluster smoke logs record whether the GPU collectives path is alive. The launcher itself still uses CPU MPI Allreduce — Phase 3c stage 2 (below) wires NCCL into `distributed_lanczos`. | `src/cli/ed_distributed_main.cpp` | DONE |
| 7 | `test_multi_gpu_nccl` — Catch2 + MPI lockdown for the wrappers (allreduce double/complex matches `MPI_Allreduce` element-wise; broadcast matches `MPI_Bcast`; move semantics keep ownership exactly once). Registered at `np ∈ {1,2,4}`; SKIPs gracefully when no CUDA device is visible. | `tests/unit/test_multi_gpu_nccl.cpp` + `CMakeLists.txt` test block | DONE |
| 8 | `OrbitPartition` + `balanced_orbit_slab(orbit_weights, n_ranks)` LPT greedy primitive for Phase 3b #7 stage 1 (orbit-respecting row partitioning). Deterministic (every rank computes the same partition), worst-case `(4/3 - 1/(3 n_ranks))` × optimal makespan. | `include/ed/distributed/orbit_partition.h`, `src/distributed/orbit_partition.cpp` | DONE |
| 9 | `test_orbit_partition` — Catch2 lockdown for `balanced_orbit_slab` (trivial inputs, determinism, basic invariants, equal-weight/heavy-outlier/random-stress LPT bound). | `tests/unit/test_orbit_partition.cpp` + `CMakeLists.txt` test block | DONE |

## What landed in Phase 3c (stage 2: GPU-resident Lanczos with NCCL allreduce)

| # | item | files | status |
|---|---|---|---|
| 10 | `distributed_lanczos_gpu` — GPU-resident Krylov basis (`v_prev`, `v_curr`, `w` as `cuDoubleComplex*` device buffers); per-iteration alpha/beta computed via `cublasZdotc` + `ncclAllReduce` on device buffers (replacing `MPI_Allreduce` of the CPU path); axpy/scal via `cublasZaxpy`/`cublasZdscal`; SpMV is host-staged through the existing CPU `DistributedOperator` (each iteration: device→host, `op.apply`, host→device). Pointer rotation via `std::swap` to avoid extra copies. Replicated tridiagonal solve on every rank. | `include/ed/distributed/distributed_lanczos_gpu.h`, `src/distributed/distributed_lanczos_gpu.cu`, `cmake/EDLibraries.cmake` (linked into `ed_distributed_gpu` next to `multi_gpu.cu`; pulls `CUDA::cublas`) | DONE |
| 11 | `test_distributed_lanczos_gpu` — Catch2 + MPI lockdown that runs the GPU path AND the CPU `distributed_lanczos` on the same `DistributedOperator` with the same seed, and asserts `\|E0_gpu - E0_cpu\| < 1e-10` (also vs dense reference < 1e-8). Three sections: N=4 OBC ground state, N=6 PBC ground state, replicated eigenvalues across all ranks. SKIPs gracefully when no NCCL build / no visible CUDA device / `world_size > visible_devices` (multi-rank-on-same-device left for stage 3). Registered at `np ∈ {1, 2, 4}` via the new `ed_add_phase3c_test` helper. | `tests/unit/test_distributed_lanczos_gpu.cpp` + `CMakeLists.txt` | DONE |
| 12 | rorqual smoke jobs: `sbatch/dist_gpu_bygpu.sbatch` (1×/2×/4× H100 in `gpubase_bygpu_b1+` for backfill-friendly multi-rank NCCL runs) supplements `dist_gpu_full.sbatch` (whole 4×H100 NVLink node, low-priority). Stage 1 lockdown was already green on `dist_gpu_mig.sbatch` (1g.10gb MIG slice). | `/scratch/zhouzb79/ed_phase3/sbatch/dist_gpu_bygpu.sbatch` | DONE |

### Cluster validation (stage 2)

* **Single MIG slice (`nvidia_h100_80gb_hbm3_1g.10gb`, np=1)** — `test_distributed_lanczos_gpu` job 10942714: all 13 assertions / 3 test cases pass. Confirms that `MultiGpuCommunicator` initialises on a MIG slice, `cublasZdotc` + `ncclAllReduce` on device buffers reproduces the CPU `distributed_lanczos` ground-state energy bit-for-bit (within 1e-10), and the host-staged SpMV path is wired correctly.
* **Multi-GPU 2×H100 (`gpubase_bygpu_b1`, np=2)** — `test_distributed_lanczos_gpu` job **10943561** (13 / 3 PASS) + `test_multi_gpu_nccl` job **10943562** (594 / 6 PASS) on `rg21801` / `rg31606` after dropping `--gpu-bind=closest` (NCCL p2p needs full peer-device visibility per rank — see `sbatch/dist_gpu_bygpu.sbatch`). First end-to-end multi-rank lockdown of the GPU Lanczos with cross-device NCCL allreduce actually firing on real H100→H100 traffic.
* **Multi-GPU 4×H100 (`gpubase_bygpu_b1`, np=4 on `rg31702`)** — `test_distributed_lanczos_gpu` job **10943974** (13 / 3 PASS) + `test_multi_gpu_nccl` job **10943975** (594 / 6 PASS). Full single-node 4-rank lockdown.

### What landed in Phase 3b (stage 2 prep): orbit-partition accessors

| # | item | files | status |
|---|---|---|---|
| 7b | `OrbitPartition::owner_rank(orbit_id)` / `owner_local_index(orbit_id)` / `global_rank_major_index(orbit_id)` — DistributedOperator-shaped accessors that let the future `DistributedSymmetryOperator` build its halo plan in the same idiom (`owner_rank` + `local_offset` + rank-major prefix-sum) as the existing CPU `DistributedOperator`. Backed by a precomputed `orbit_local_index[]` table for O(1) reverse lookups. | `include/ed/distributed/orbit_partition.h`, `src/distributed/orbit_partition.cpp` | DONE |
| 7c | `test_orbit_partition` — new "stage2_accessors" section: round-trip `orbit_id → (owner_rank, owner_local_index) → rank_orbits[owner][k] → orbit_id` on equal-weight, heavy-outlier, and stress (n_orbits=257, n_ranks=11) cases; `global_rank_major_index` is a permutation of `[0, n_orbits)`; out-of-range `owner_rank` returns -1 instead of UB. **75,208 assertions** in 6 cases pass on the rorqual login node. | `tests/unit/test_orbit_partition.cpp` | DONE |
| 7d | `OrbitHaloPlan` — orbit-aware `MPI_Alltoallv` halo plan that the future `DistributedSymmetryOperator` will plug into the SpMV hot path. Constructor takes an `OrbitPartition` + a "this rank needs amplitudes for these orbit ids" set; runs **one** `MPI_Alltoall` (counts) and **one** `MPI_Alltoallv` (orbit ids) at construction; `exchange()` does **one** `MPI_Alltoallv` of `complex<double>` per call. Filters locally-owned orbits, deduplicates the request set, sorts per-rank slabs ascending so the plan is reproducible across calls. | `include/ed/distributed/orbit_halo_plan.h`, `src/distributed/orbit_halo_plan.cpp` | DONE |
| 7e | `test_orbit_halo_plan` — Catch2 + MPI lockdown on a synthetic 1-D ring of 32 orbits where every orbit talks to its two neighbours; `exchange()` populates the halo with the analytic `exp(2πi k / n)` reference for every recv slot at np ∈ {1, 2, 4} (np=1 → empty halo as expected). Plus heavy-outlier weight + dedup-of-the-universe sanity sections. **145 assertions** total across the three np values pass on the rorqual login node. | `tests/unit/test_orbit_halo_plan.cpp` + `CMakeLists.txt` | DONE |

## What landed in Phase 3c (stage 3: GPU-resident SpMV with NCCL pairwise SendRecv halo)

| # | item | files | status |
|---|---|---|---|
| 13 | `DistributedGPUOperator` — fully GPU-resident sibling of `DistributedOperator`. Constructor mirrors the CPU operator's communication plan (`send_local_idx`, `recv_keys`/`recv_values` from the existing `SortedUint64Index`) plus the SoA term tables (diag/offdiag one-/two-body, mixed two-body) onto the device, allocates persistent device send/recv buffers sized to `total_send`/`total_recv`, and exposes one entry point: `apply(MultiGpuCommunicator&, d_v, d_y, stream)`. The hot path runs three GPU phases on the caller's stream: (1) `pack_send_buf_kernel` builds the contiguous send buffer indirection-style on device; (2) `ncclGroupStart()` + per-peer `ncclSend` / `ncclRecv` (each `complex<double>` sent as 2× `ncclFloat64`) + `ncclGroupEnd()` exchanges the halo without ever touching the host; (3) `distributed_gpu_spmv_kernel` runs one thread per local row, walking the SoA term tables and resolving off-rank columns via a device-side binary search into `recv_keys`/`recv_values`. 3-body terms are explicitly rejected at construction (the existing distributed paths we exercise are 1-body + 2-body only). Required exposing `DistributedOperator::comm_plan_view()` + `serial_operator()` so the GPU operator can mirror the plan without re-deriving the communication structure. | `include/ed/distributed/distributed_gpu_operator.h`, `src/distributed/distributed_gpu_operator.cu`, `include/ed/distributed/distributed_operator.h` (added `CommPlanView` + getters), `cmake/EDLibraries.cmake` | DONE |
| 14 | `test_distributed_gpu_operator` — Catch2 + MPI lockdown that builds a deterministic random `complex<double>` global vector, scatters it the same way the GPU Lanczos does, and asserts `max\|y_gpu_local - y_cpu_local\| < 1e-12` element-wise after one `DistributedGPUOperator::apply` (full pack + NCCL halo + GPU SpMV) vs the existing CPU `DistributedOperator::apply`. Three sections: N=4 OBC, N=6 PBC, idempotent re-apply (same input → bit-identical output). Registered at `np ∈ {1,2,4}` via the same `ed_add_phase3c_test` helper; SKIPs cleanly on builds without NCCL or visible CUDA devices, and on `world_size > visible_devices` (1 GPU per rank). | `tests/unit/test_distributed_gpu_operator.cpp`, `CMakeLists.txt` | DONE |

### Cluster validation (stage 3)

* **Single H100 (`gpubase_bygpu_b1`, np=1 on `rg32001`)** — `test_distributed_gpu_operator` job **10945403**: 6 assertions / 3 cases PASS. Single-rank degenerate path (no halo traffic; SpMV kernel only) reproduces the CPU operator's output bit-identically.
* **Multi-GPU 2×H100 (`gpubase_bygpu_b1`, np=2 on `rg21801`)** — `test_distributed_gpu_operator` job **10945402**: 6 assertions / 3 cases PASS. NCCL pairwise SendRecv halo + GPU SpMV agrees with the CPU `MPI_Alltoallv`-driven path within 1e-12 element-wise on N=4 OBC and N=6 PBC random complex inputs. First end-to-end lockdown of a fully GPU-resident distributed `H · v`.
* **Multi-GPU 4×H100 (`gpubase_bygpu_b1`, np=4 on `rg32501`)** — `test_distributed_gpu_operator` job **10949083**: 6 assertions / 3 cases PASS. Full single-node 4-rank lockdown of the GPU SpMV with NCCL pairwise SendRecv halo. Closes the np ∈ {1, 2, 4} stage-3 sweep.

## What landed in Phase 3c (stage 4: GPU-resident SpMV inside `distributed_lanczos_gpu`)

| # | item | files | status |
|---|---|---|---|
| 15 | `DistributedLanczosGPUOptions::gpu_resident_spmv` (default `false`). When `true`, `distributed_lanczos_gpu` builds a `DistributedGPUOperator` once at startup (via an aliased `shared_ptr<DistributedOperator>` so the caller retains ownership) and replaces the per-iteration `D2H → CPU op.apply → H2D` sandwich with a single `gop.apply(gpu_comm, d_v_curr, d_w, /*stream=*/nullptr)`. The cuBLAS / NCCL allreduce calls already run on the legacy default stream, so device-side ordering is preserved without an explicit sync. The `false` path is preserved verbatim as the regression baseline / pre-`DistributedGPUOperator` fallback. | `include/ed/distributed/distributed_lanczos_gpu.h`, `src/distributed/distributed_lanczos_gpu.cu` | DONE |
| 16 | `test_distributed_lanczos_gpu` — new "stage4" test case: runs the same Lanczos twice on `(N=4 OBC, seed=12345, max_iter=60)` and `(N=6 PBC, seed=7, max_iter=80)`, once with `gpu_resident_spmv = false` (stage 2 host-staged path) and once with `gpu_resident_spmv = true` (stage 4 fully on-device path), and asserts `\|E0_stage2 - E0_stage4\| < 1e-10`. Catches stream-ordering / NCCL-vs-cuBLAS interleaving regressions that the standalone `test_distributed_gpu_operator` (single-shot apply) cannot. | `tests/unit/test_distributed_lanczos_gpu.cpp` | DONE |

### Cluster validation (stage 4)

* **Single H100 (`gpubase_bygpu_b1`, np=1 on `rg21802`)** — `test_distributed_lanczos_gpu` job **10950081**: **19 assertions / 4 cases** PASS. Stage-2 (host-staged SpMV) and stage-4 (`gpu_resident_spmv = true`) ground-state energies agree within 1e-10 on N=4 OBC and N=6 PBC.
* **Multi-GPU 2×H100 (`gpubase_bygpu_b1`, np=2 on `rg21708`)** — `test_distributed_lanczos_gpu` job **10950082**: 19 assertions / 4 cases PASS.
* **Multi-GPU 4×H100 (`gpubase_bygpu_b1`, np=4 on `rg32501`)** — `test_distributed_lanczos_gpu` job **10950083**: 19 assertions / 4 cases PASS. Closes the np ∈ {1, 2, 4} stage-4 sweep — fully GPU-resident distributed Lanczos (cuBLAS Krylov + `ncclAllReduce` dot/norm + `DistributedGPUOperator` SpMV with `ncclSendRecv` halo) is correctness-locked vs both the CPU `distributed_lanczos` and the stage-2 host-staged GPU path.

### Still open in Phase 3c (follow-ups)

* GPU-resident `distributed_ftlm` / `distributed_tpq` — the TPQ Taylor loop and FTLM Krylov inner trace are mechanically identical to `distributed_lanczos`; with the GPU Lanczos now rebased onto `DistributedGPUOperator`, all three can share the same `MultiGpuCommunicator` + `DistributedGPUOperator` foundation. The TU-level surgery is the same `gpu_resident_spmv = true` swap; the lockdown plan is to cross-check against the existing CPU distributed `<O>(β)` / `Z(β)` / `E(β)` paths.
* 3-body term support in `distributed_gpu_spmv_kernel`. The CPU path's 3-body apply is `apply_real`-style (real-only by design); a GPU port needs the same `walking`-state convention reproduced verbatim. Currently rejected at `DistributedGPUOperator` construction with a clear `std::invalid_argument`.
* Pinned-host scratch for the (rare) cases where users still want to read `y_local` back to host after every apply. Trivial to add (`cudaHostAllocMapped`); not on the critical path because the Lanczos loop above keeps everything on device anyway.

## What landed in Phase 3b (stage 2 proper): `DistributedSymmetryOperator`

| # | item | files | status |
|---|---|---|---|
| 7f | `DistributedSymmetryOperator` — the load-bearing distributed-memory operator on the symmetry-projected basis. The constructor (collective on `comm`) does five things: (1) BFS-enumerates every translation orbit on the unsymmetrised basis using `applyPermutation` + the symmetry-info generators, deterministically picking the lex-min element as the orbit representative; (2) projects each orbit into the requested momentum sector via the per-group-element character convention `chi_q(g_a) = sector.phase_factors[a]` (matches `ed::sym::group_from_generators` in `src/symmetry/group.cpp` L221-230 and the QuSpin/Bloch convention test in `test_symmetry_dsl.cpp` L127-139), filtering out zero-norm "phantom" orbits in this sector; (3) LPT-balances the surviving orbits across ranks via `balanced_orbit_slab(orbit_sizes, n_ranks)`; (4) builds the rank-local sparse rows of `H_q` by, for each global column `j`, inflating `~|j>` on the full Hilbert space, applying the serial `Operator::apply`, and projecting `H ~|j>` onto every orbit `i` to get `H_ij = (1/sqrt(N_i N_j)) Σ_b conj(phi[b]) (H ~|j>)[b]` — only entries with `i` locally owned and `|H_ij| > 1e-12` are kept; (5) builds an `OrbitHaloPlan` from the union of remote `j` ids the local rows touch, and rewrites every per-row column index to either `partition.owner_local_index(j)` (local) or `halo_index[j]` (remote). The hot `apply()` path is one `OrbitHaloPlan::exchange()` + one rank-local CSR-style sparse SpMV. The slab geometry is **rank-major** (orbit `i` lands at global index `partition.rank_offsets[owner] + partition.owner_local_index(i)`), not contiguous in orbit-id space — LPT scrambles ids across ranks, so callers feeding "natural orbit ordering" inputs must permute through `partition.rank_orbits[r][k]`. | `include/ed/distributed/distributed_symmetry_operator.h`, `src/distributed/distributed_symmetry_operator.cpp`, `cmake/EDLibraries.cmake` (added to `ed_distributed`) | DONE |
| 7g | `test_distributed_symmetry_operator` — Catch2 + MPI lockdown that builds an INDEPENDENT serial dense reference of the projected matrix `H_q` (mirroring the same per-group-element character convention so a bug in either side fails the test), constructs `DistributedSymmetryOperator(op, sector_idx, MPI_COMM_WORLD)`, applies it to a deterministic random complex input vector permuted into the operator's rank-major slab layout, gathers the result with `MPI_Allgatherv`, unscrambles it back to natural orbit ordering via `partition.rank_orbits`, and asserts `max\|y_dist - y_ref\| < 1e-10` element-wise. Coverage: every momentum sector of `(N=4 OBC, N=4 PBC, N=6 PBC)` Heisenberg chains across `np ∈ {1, 2, 4}`, plus halo-plan/load-balance diagnostics. Registered via `ed_add_mpi_test` and linked against `ed_symmetry`. **82 assertions / 4 cases** PASS at `np ∈ {1, 2, 4}` on the rorqual login node AND on a real compute node (cluster job **10953752** on `rc32014`, `cpubase_b1`). | `tests/unit/test_distributed_symmetry_operator.cpp`, `CMakeLists.txt`, `/scratch/zhouzb79/ed_phase3/sbatch/dist_symop_test.sbatch` | DONE |
| 7h | **Templated Lanczos kernel** (`distributed_lanczos_kernel.h`) — header-only `distributed_lanczos_kernel<OpT>(op, v0_local, options)` that runs the canonical three-term Lanczos recurrence on rank-local slabs with `MPI_Allreduce`-collapsed `dot`/`norm`, optional MGS full-reorth, and replicated tridiagonal eigensolve. Templated on the duck-typed "DistributedOperator-shaped" interface (`apply` / `rank` / `comm_size` / `comm` / `global_dim` / `local_size`) so both `DistributedOperator` (1D row-slab) and `DistributedSymmetryOperator` (LPT-balanced orbit slab) can call it. Eigenvalue / `compute_weights` / `compute_eigenvectors` extraction is identical to the existing `distributed_lanczos.cpp` kernel; the geometry-specific bits (initial-vector scatter, slab indexing) stay outside the templated kernel. | `include/ed/distributed/distributed_lanczos_kernel.h` | DONE |
| 7i | **`distributed_lanczos_symmetry`** — distributed Lanczos rebased onto `DistributedSymmetryOperator`. Generates a deterministic L2-normalised global random vector in NATURAL orbit ordering, permutes it into the rank-major LPT slab via `partition.rank_orbits`, scatters with `MPI_Scatterv`, then dispatches to the templated kernel. Returns the same `DistributedLanczosResult` as the unsymmetrised Lanczos so callers can reuse downstream code. | `include/ed/distributed/distributed_lanczos.h` (new declaration), `src/distributed/distributed_lanczos.cpp` (new entry point) | DONE |
| 7j | `test_distributed_lanczos_symmetry` — Catch2 + MPI lockdown that cross-checks `distributed_lanczos_symmetry` against an `Eigen::SelfAdjointEigenSolver` ground-state energy of the projected matrix `H_q` on Heisenberg `(N=4 OBC sector 0, N=4 PBC sector 0, N=6 PBC sector 0, N=6 PBC every momentum sector)`. Asserts `\|E0_dist - E0_dense\| < 1e-8` and bit-replicated eigenvalues across all ranks. **45 / 54 assertions in 4 cases** PASS at `np ∈ {1, 2, 4}` on the rorqual login node (45 at np=1 because the broadcast-replication assertion only fires at np>1) AND on a real compute node (cluster job **10954066** on `rc32329`, `cpubase_b1`). | `tests/unit/test_distributed_lanczos_symmetry.cpp`, `CMakeLists.txt`, `/scratch/zhouzb79/ed_phase3/sbatch/dist_symop_test.sbatch` | DONE |

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
   structure to keep Alltoallv halo size bounded. That is the Phase 3b
   #7 follow-up (still open).
2. **Symmetry-aware row partitioning.** The Phase 3b #7 follow-up
   (`OrbitPartition` + `OrbitHaloPlan` + `DistributedSymmetryOperator`
   + `distributed_lanczos_symmetry`) now gives us LPT-balanced orbit
   partitioning + orbit-aware `MPI_Alltoallv` halo + rank-local sparse
   SpMV on the projected basis, AND a distributed Lanczos solver loop
   on top of it. Correctness-locked vs serial dense reference on
   Heisenberg N=4 OBC, N=4 PBC, N=6 PBC across all momentum sectors at
   np ∈ {1,2,4}. The `apply()` hot path is now O(local_nnz) per
   iteration (not O(local_n × max_pattern_count)), so the previously
   load-bearing halo-blowup at "honest 40" with full point-group + Sz
   is closed. What is **not** yet shipped: `distributed_ftlm` /
   `distributed_tpq` rebased onto `DistributedSymmetryOperator`. The
   templated kernel (`distributed_lanczos_kernel<OpT>`) is in place;
   rebasing FTLM/TPQ is the natural follow-up landing.
3. **Multi-GPU (Phase 3c) at scale.** The NCCL allreduce path
   (`distributed_lanczos_gpu`) AND the fully-on-device SpMV with
   pairwise-`ncclSendRecv` halo (`DistributedGPUOperator`) are now
   correctness-locked on real H100×{1,2,4} hardware on rorqual
   `gpubase_bygpu_b1` against the CPU reference. They have NOT yet
   been swept on a many-node `gpubase_bynode_b1+` allocation, and
   they have not been profiled against an asynchronous double-buffered
   halo — only correctness is claimed.
4. **Stability of TPQ Taylor truncation.** `distributed_tpq` uses a
   degree-`taylor_order` Taylor truncation of `e^{-(δβ/2) H}` per
   substep. This is unconditionally stable only when
   `(δβ/2) ‖H‖ ≪ taylor_order`. Defaults (`δβ = 0.05`, order 30) cover
   `‖H‖ ≲ 30` per local site, which is fine for spin-1/2 Heisenberg
   models. For larger spectral radii, drop `δβ` or raise the order.
   Lockdown tested at N=4 OBC only; the formula is operator-agnostic
   but cluster-scale stability has not been swept.
5. **Distributed eigenvector reconstruction memory.** Setting
   `compute_eigenvectors = true` on `DistributedLanczosOptions` retains
   the entire rank-local Krylov basis (`m * local_n * 16 B` per rank,
   same as `full_reorth = true`). For "honest 40" with `m = 200` and
   `local_n ≈ 5e9`, that is ~16 TB per rank — well outside any sane
   single-node budget. The recommended workflow at scale remains
   "run distributed Lanczos for the ground-state *energy* on the full
   Hilbert space, then run a *symmetry-projected serial* solver in the
   relevant sector for the eigenvector". The new
   `distributed_lanczos_eigenvectors` API is honest about this cost in
   its docstring; the test suite exercises it on `local_n ≤ 64` only.

## Provenance

* Phase 3a: 5 commits between `f5ebd54` and `e904bcd` (see
  `PHASE_3A_SUMMARY.md` for details).
* Phase 3b/c: this commit (and the immediately preceding one if the
  3b/3c work was split across two commits).
* Test counts cross-checked with `ctest -N | grep -c "Test #"`
  (currently `167`) and the per-label sweep
  `ctest -L phase3b --output-on-failure` (currently `15/15`, ~120 s
  wall on a 32-thread Ryzen 9 7950X3D — TPQ at np=4 dominates).
* `DistributedSymmetryOperator` lockdown verified via `mpiexec -np
  ${np} ./test_distributed_symmetry_operator` for `np ∈ {1, 2, 4}`:
  82 / 82 assertions across 4 cases pass at every `np`. Cluster
  validation via `sbatch sbatch/dist_symop_test.sbatch` (single CPU
  node, runs `np ∈ {1, 2, 4}` back-to-back).
