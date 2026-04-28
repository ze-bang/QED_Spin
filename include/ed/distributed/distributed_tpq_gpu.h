// =============================================================================
// include/ed/distributed/distributed_tpq_gpu.h    (Phase 9 / Layer 2)
//
// Multi-GPU sibling of `distributed_tpq` (canonical Sugiura-Shimizu TPQ).
//
// Convention (identical to distributed_tpq.h):
//
//   |psi(beta)> = e^{-beta H / 2} |r>  /  || e^{-beta H / 2} |r> ||
//
// We propagate in `delta_beta` substeps using a Taylor expansion of
// e^{-(delta_beta/2) H}. Per substep:
//
//   * SpMV runs on every rank's GPU via `DistributedGPUOperator::apply`
//     (NCCL pairwise SendRecv halo + on-device SoA SpMV kernel).
//   * `dist_norm` and `dist_zdotc` for normalisation / observables run
//     through `cublasZdotc` + `multi_gpu::all_reduce_sum_complex_double`
//     (single NCCL allreduce per scalar reduction, no PCIe round-trip).
//   * The Taylor accumulator `result += coef * term` is a single
//     `cublasZaxpy` call per order.
//
// What this gives us:
//   * **multi-GPU canonical TPQ** end-to-end: device-resident |psi>,
//     GPU SpMV with NCCL halo, NCCL allreduce reductions. The CPU
//     never sees the wave function except for setup (initial vector
//     scatter on the host, then one H2D upload per sample).
//   * **MPI-over-samples** matches `distributed_tpq`: world_comm is
//     split into `n_groups` subcommunicators via MPI_Comm_split, each
//     sample lives on one group, world-level reductions sample-average
//     the energy / variance.
//   * **Results bit-compatible** with `distributed_tpq`: same betas
//     interpretation (strictly ascending; substeps cap at delta_beta;
//     final substep snaps onto the target), same compute_variance
//     semantics (E2 = ||H psi||^2 minus E^2), same `samples_used`
//     accounting.
//
// Honest scope:
//   * Like the CPU path, the Taylor expansion is stable up to
//     `||delta_beta * H|| ~ 1`. Caller picks delta_beta + taylor_order;
//     we don't auto-tune.
//   * We do **not** wire DistributedSymmetryOperator yet. Symmetry-
//     projected canonical TPQ is documented in distributed_tpq's
//     header as conceptually unsound (projection destroys the Z
//     normalisation), so the GPU path inherits the same restriction.
//   * 3-body terms are out of scope for `DistributedGPUOperator` and
//     therefore for this kernel; the constructor will throw with a
//     clear message.
//
// Compiled iff:
//     WITH_MPI
//   && WITH_CUDA
//   && NCCL_FOUND   (ED_HAVE_NCCL = 1)
//
// On builds missing NCCL, the function declaration stays valid but
// linking fails noisily; callers should guard via
// `multi_gpu::nccl_compiled_in()` (same convention as
// `distributed_lanczos_gpu`).
// =============================================================================

#pragma once

#ifdef WITH_MPI

#include <ed/distributed/distributed_tpq.h>     // DistributedTpqOptions/Result

#include <cstdint>
#include <memory>
#include <vector>

#include <mpi.h>

namespace ed::distributed {

struct DistributedTpqGPUOptions {
    /// Number of random samples (the R in the Sugiura-Shimizu sum). Each
    /// sample is round-robin assigned to one of `n_groups` MPI groups.
    int n_samples = 8;

    /// Number of MPI groups across samples. Must divide world_size.
    int n_groups = 1;

    /// Imaginary-time substep for the Taylor truncation of
    /// e^{-(delta_beta/2) H}. Smaller = more stable but more SpMVs.
    double delta_beta = 0.05;

    /// Truncation order for the Taylor series. 30 is safe for
    /// ||delta_beta H|| <~ 1.
    std::uint64_t taylor_order = 30;

    /// Inverse temperatures at which to record E(beta). Strictly
    /// ascending, non-negative. Empty -> {1.0}.
    std::vector<double> betas;

    /// Per-sample seed offset; sample s uses `seed_offset + s`.
    unsigned long seed_offset = 0UL;

    /// If true, also record E2 = ||H psi||^2 and surface
    /// `variance = E2 - E^2`. Costs one extra GPU SpMV per
    /// (sample, beta).
    bool compute_variance = false;

    /// Local CUDA device index for this rank. -1 = auto via the
    /// node-local rank heuristic in MultiGpuCommunicator.
    int device_index = -1;

    /// If true, rank 0 prints per-sample, per-beta diagnostics.
    bool verbose = false;
};

/// Multi-GPU canonical TPQ. Collective on `world_comm`; every rank must
/// call. Splits world_comm into `options.n_groups` group communicators,
/// distributes samples round-robin, propagates each rank-local |psi>
/// on-device via a Taylor-truncated e^{-(delta/2) H}, reduces
/// sample-averaged observables across the world.
///
/// Results are bit-compatible with `distributed_tpq` (same struct).
DistributedTpqResult distributed_tpq_gpu(
    std::shared_ptr<class ::Operator> op,
    const DistributedTpqGPUOptions& options,
    MPI_Comm world_comm);

}  // namespace ed::distributed

#endif  // WITH_MPI
