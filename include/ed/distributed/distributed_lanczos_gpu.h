// =============================================================================
// include/ed/distributed/distributed_lanczos_gpu.h    (Phase 3c stage 2)
//
// GPU-resident sibling of `distributed_lanczos`. Krylov basis (v_prev,
// v_curr, w) lives in device memory; the dot/norm reductions go through
// `ncclAllReduce` on device buffers via the Phase 3c stage 1 wrappers
// (`ed::distributed::multi_gpu`).
//
// Honest scope (stage 2 only):
//
//   * SpMV is still host-side: each iteration we copy v_curr device→host,
//     apply the existing CPU `DistributedOperator` (which handles its
//     own MPI_Alltoallv halo), and copy the result back to device. This
//     means PCIe traffic dominates for small local_n and the wall-clock
//     win over the pure-CPU `distributed_lanczos` is modest. The point
//     of stage 2 is to validate the NCCL collectives in a real Lanczos
//     loop end-to-end (correctness lockdown vs the CPU path), not to
//     beat the CPU path on speed.
//
//   * Stage 3 (still open) replaces the host-staged SpMV with a fully
//     GPU-resident `DistributedGPUOperator` whose halo uses
//     `ncclSendRecv` between device buffers (GPU-Direct RDMA on
//     supported fabrics). At that point the GPU Lanczos beats the CPU
//     path on every metric.
//
//   * Re-orthogonalisation (full MGS) and eigenvector reconstruction are
//     deliberately omitted from stage 2 -- they require keeping the
//     full Krylov basis on device, which is the same memory footprint
//     issue we address via blocked re-orth on CPU and which would only
//     compound the host-staged SpMV cost. Add them when the GPU SpMV
//     lands in stage 3.
//
// API mirrors `distributed_lanczos` in spirit but is a separate type so
// that the CPU path keeps its full feature surface (full_reorth,
// compute_eigenvectors, FTLM weights) without conditional GPU
// branches.
// =============================================================================

#pragma once

#ifdef WITH_MPI

#include <ed/distributed/distributed_operator.h>

#include <cstdint>
#include <vector>

namespace ed::distributed {

struct DistributedLanczosGPUOptions {
    /// Maximum Lanczos iterations.
    std::uint64_t max_iter = 200;

    /// Number of smallest eigenvalues to return.
    std::uint64_t exct = 1;

    /// Convergence tolerance for the smallest exct eigenvalues.
    double tol = 1e-12;

    /// Seed for the rank-0 initial vector (scattered to every rank).
    unsigned long seed = 12345UL;

    /// Local CUDA device index for this rank. -1 = auto via
    /// `MPI_Comm_split_type(SHARED) -> node_local_rank % cudaGetDeviceCount()`.
    int device_index = -1;

    /// If true, rank 0 prints per-iteration diagnostics.
    bool verbose = false;

    /// Phase 3c stage 4: SpMV path selector.
    ///   * `false` (default, stage 2 path) -- per-iter `D2H -> CPU
    ///     DistributedOperator::apply -> H2D`. Validates the NCCL
    ///     allreduce path end-to-end without depending on the GPU SpMV
    ///     kernel; useful as a fallback / regression baseline.
    ///   * `true`  (stage 3+ path) -- builds a `DistributedGPUOperator`
    ///     once at startup and calls `gop.apply(gpu_comm, d_v, d_w)`
    ///     per iteration. Halo exchange runs through `ncclSendRecv` on
    ///     device buffers; no PCIe round-trips in the inner loop.
    bool gpu_resident_spmv = false;
};

struct DistributedLanczosGPUResult {
    /// Smallest `min(exct, iterations)` Ritz values. Replicated on every
    /// rank (bit-identical given the same seed).
    std::vector<double> eigenvalues;

    /// Number of Lanczos iterations actually performed (<= max_iter).
    int iterations = 0;

    /// Tridiagonal coefficients (replicated on every rank). `alphas`
    /// length = iterations; `betas` length = iterations + 1 (with
    /// `betas[0] = 0`). Useful for downstream FTLM/DOS work that
    /// wants to re-solve the tridiagonal with different weights.
    std::vector<double> alphas;
    std::vector<double> betas;
};

/**
 * GPU-resident distributed Lanczos. Collective on `op.comm()`.
 *
 * Behaviour:
 *   * Allocates `3 * local_n * 16 B` of device memory per rank (v_prev,
 *     v_curr, w). Plus a handful of 16 B device scalars for the
 *     reduction buffers.
 *   * Loops:
 *       1. SpMV: device→host, CPU DistributedOperator::apply, host→device.
 *       2. alpha_j = Re <v_curr | w> via cublasZdotc + ncclAllReduce.
 *       3. w -= alpha_j v_curr - beta_prev v_prev via cublasZaxpy.
 *       4. beta_{j+1} = sqrt(<w|w>) via cublasZdotc + ncclAllReduce + sqrt.
 *       5. Renormalise w; rotate (v_prev, v_curr) := (v_curr, w).
 *       6. Solve the (j+1)x(j+1) tridiagonal redundantly on every rank.
 *   * Throws `std::logic_error` if the build does not have NCCL.
 *   * Throws `std::runtime_error` on any cuda/cublas/nccl/MPI failure
 *     (with the vendor error string included in `what()`).
 *   * Throws `std::runtime_error` if the local rank cannot see at least
 *     one CUDA device.
 */
DistributedLanczosGPUResult distributed_lanczos_gpu(
    const DistributedOperator& op,
    const DistributedLanczosGPUOptions& options = {});

/**
 * GPU-resident distributed Lanczos on the **symmetry-projected** basis.
 * Collective on `op.comm()`.
 *
 * Builds a `DistributedSymmetryOperatorGPU` internally from the supplied
 * CPU `DistributedSymmetryOperator` and runs the same per-iteration
 * recipe as `distributed_lanczos_gpu` but with the orbit-row SpMV (one
 * NCCL pairwise SendRecv halo + one CSR sparse-matvec kernel per
 * iteration).
 *
 * The initial vector is generated deterministically from `seed` in
 * NATURAL orbit ordering, then permuted into rank-major-with-LPT-orbit
 * scrambling on rank 0 and scattered via `MPI_Scatterv` -- exactly the
 * same convention as the CPU `distributed_lanczos_symmetry` path, so
 * the resulting Ritz values are bit-comparable across CPU/GPU at the
 * same seed.
 *
 * `gpu_resident_spmv` is forced to `true` on this path (the CPU
 * fallback is meaningless because the symmetry SpMV needs the orbit
 * permutation that lives only inside `DistributedSymmetryOperator`).
 *
 * Throws `std::logic_error` if NCCL is not compiled in.
 */
DistributedLanczosGPUResult distributed_lanczos_gpu_symmetry(
    const class DistributedSymmetryOperator& op,
    const DistributedLanczosGPUOptions& options = {});

}  // namespace ed::distributed

#endif  // WITH_MPI
