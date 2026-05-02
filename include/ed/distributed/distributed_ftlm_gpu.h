// =============================================================================
// include/ed/distributed/distributed_ftlm_gpu.h    (Phase A: device matrix
// MPI+GPU FTLM)
//
// Multi-GPU sibling of `distributed_ftlm`. Same Jaklic-Prelovsek estimator,
// same MPI-over-samples grouping, same convention for `Z(beta)` and
// observable expectation values -- but the per-sample Lanczos loop runs
// entirely on device:
//
//   * Krylov basis V_local[0..m-1] is held in a single contiguous device
//     buffer (m * local_n * 16 B per rank).
//   * SpMV uses `DistributedGPUOperator::apply` -- NCCL pairwise
//     SendRecv halo + on-device SoA SpMV. No PCIe round-trips inside
//     the Lanczos loop.
//   * Dot products and norms use `cublasZdotc` + a single NCCL allreduce
//     per scalar (via the Phase 3c stage 1 wrappers in `multi_gpu.h`).
//   * Full modified Gram-Schmidt re-orthogonalisation against the locked
//     basis is done with a length-(j+1) cublasZdotc reduction (one NCCL
//     allreduce of 2*(j+1) doubles per Lanczos step) followed by m
//     cublasZaxpy calls. Required for clean FTLM weights when m
//     approaches the sector dimension.
//   * The (j+1)x(j+1) tridiagonal eigenproblem is solved redundantly on
//     every rank (Eigen::SelfAdjointEigenSolver). Same code path as the
//     CPU FTLM.
//   * Observable expectation values: optional. When `observable_op` is
//     non-null we build a per-group `DistributedGPUOperator` for O,
//     compute u = O * V[0] on device, then form
//         q_j = <V[j] | u>            (m parallel cublasZdotc + 1 NCCL
//                                      allreduce of 2*m doubles)
//         f_j(beta) = sum_k U[j,k] U[0,k] e^{-beta E_k}      (host)
//         contribution(beta) = sum_j Re(q_j) * f_j           (host)
//     and aggregate <O>(beta) = N_O / N_Z (the (D/R) prefactor cancels).
//
// MPI grouping (`n_groups`, `MPI_Comm_split`) and the
// "only-group-rank-0-contributes-to-world-reduce" trick are unchanged
// from `distributed_ftlm`.
//
// Result struct is reused: `DistributedFtlmResult` (same `Z`,
// `O_expectation`, `samples_used` fields). Cross-check against
// `distributed_ftlm` on the same problem and same seeds is bit-stable
// to round-off (NCCL allreduce ordering is implementation-defined; the
// scalar reductions are commutative).
//
// Compiled iff:
//     WITH_MPI
//   && WITH_CUDA
//   && NCCL_FOUND   (ED_HAVE_NCCL = 1)
//
// Honest scope:
//   * Like the CPU path, requires that the observable Operator share
//     the same Hilbert space (n_bits) as the Hamiltonian. Caller
//     responsible.
//   * Memory footprint is m * local_n * 16 B per rank for the basis
//     plus 4 working vectors (16 B each) plus the GPU operator term
//     tables. For m=64, local_n=10^6 that is ~1 GB -- fits a single
//     V100/A100/H100 with room to spare. For very large m / large
//     local_n the caller should split into more groups (smaller
//     local_n per rank) or reduce `lanczos_max_iter`.
//   * 3-body terms are out of scope for `DistributedGPUOperator`; the
//     constructor will throw with a clear message.
//   * Symmetry-projected operators are NOT yet wired (waits on Phase C
//     -- DistributedSymmetryOperatorGPU). Caller should pass a non-
//     symmetrised Hamiltonian.
// =============================================================================

#pragma once

#ifdef WITH_MPI

#include <ed/distributed/distributed_ftlm.h>   // re-use Result struct

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include <mpi.h>

namespace ed::distributed {

class DistributedSymmetryOperator;

struct DistributedFtlmGPUOptions {
    /// Number of random samples (R). Distributed evenly across `n_groups`
    /// MPI groups; each group runs the corresponding subset of samples.
    int n_samples = 32;

    /// Number of MPI groups across samples (outer parallelism). Must
    /// divide world_size.
    int n_groups = 1;

    /// Lanczos parameters per sample.
    std::uint64_t lanczos_max_iter = 100;

    /// Inverse temperatures at which to evaluate Z(beta) and (if an
    /// observable is set) <O>(beta). Empty -> {1.0}.
    std::vector<double> betas;

    /// Random seed offset; sample s uses (seed_offset + s).
    unsigned long seed_offset = 0UL;

    /// Optional observable. If non-null, also computes <O>(beta) at every
    /// entry of `betas`. Adds one DistributedGPUOperator constructor per
    /// group plus one device SpMV + m local cublasZdotc + 1 NCCL
    /// allreduce of 2*m doubles per sample. Default null.
    std::shared_ptr<class ::Operator> observable_op = nullptr;

    /// Local CUDA device index for this rank. -1 = auto via the node-
    /// local rank heuristic in MultiGpuCommunicator (same convention as
    /// distributed_lanczos_gpu / distributed_tpq_gpu).
    int device_index = -1;

    /// If true, rank 0 prints per-sample diagnostics.
    bool verbose = false;
};

/// Multi-GPU FTLM. Collective on `world_comm` -- every rank must call.
/// Internally splits world_comm into `options.n_groups` subcommunicators
/// and runs an on-device Lanczos inside each.
///
/// Result struct is reused from `distributed_ftlm`:
///   * `Z[b]`              -- replicated partition function at betas[b]
///   * `O_expectation[b]`  -- <O>(betas[b]) (empty if `observable_op` null)
///   * `samples_used`      -- = `options.n_samples`
///
/// Throws `std::logic_error` if NCCL is not compiled in.
DistributedFtlmResult distributed_ftlm_gpu(
    std::shared_ptr<class ::Operator> op,
    const DistributedFtlmGPUOptions& options,
    MPI_Comm world_comm);

/// Symmetry-projected variant of `distributed_ftlm_gpu` (Phase D step 5).
/// Same on-device J&P trace estimator, but every per-sample SpMV runs
/// inside ONE symmetry sector (`sector_idx`) of the underlying
/// `Operator` (and the optional `observable_op`). Internally builds
/// `DistributedSymmetryOperatorGPU` instances on the per-group
/// MultiGpuCommunicator. The returned `Z[b]` and `O_expectation[b]`
/// are the contributions from this sector alone -- the caller
/// aggregates across sectors when reconstructing the full-space
/// partition function (matching the CPU `distributed_ftlm_symmetry`
/// convention).
///
/// Collective on `world_comm`. Throws `std::logic_error` if NCCL is
/// not compiled in.
DistributedFtlmResult distributed_ftlm_gpu_symmetry(
    std::shared_ptr<class ::Operator> op,
    std::size_t sector_idx,
    const DistributedFtlmGPUOptions& options,
    MPI_Comm world_comm);

}  // namespace ed::distributed

#endif  // WITH_MPI
