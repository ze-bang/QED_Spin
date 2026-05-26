// =============================================================================
// include/ed/distributed/distributed_krylov_schur_gpu.h
//
// Phase B (device matrix MPI+GPU): multi-GPU Krylov-Schur (thick-restart
// Lanczos with locking).
//
// Multi-GPU sibling of `distributed_krylov_schur`. Same algorithmic
// scaffolding -- explicitly-restarted Lanczos with Ritz-pair locking,
// full re-orthogonalisation against both the locked Ritz vectors and
// the in-cycle Krylov basis -- but with the locked vectors and the
// in-cycle basis held entirely in device memory. SpMV runs on
// `DistributedGPUOperator` (NCCL halo + on-device SoA SpMV);
// inner products / norms go through `cublasZdotc` +
// `multi_gpu::all_reduce_sum_complex_double` (one NCCL allreduce per
// scalar, with the in-cycle full-reorth dot products coalesced into a
// single 2*(j+1) double payload to amortise launch latency); the
// (m x m) tridiagonal eigensolve is done redundantly on every rank
// with Eigen.
//
// Result type and option semantics intentionally mirror the CPU
// `distributed_krylov_schur` so callers / dispatchers (the standalone
// CLI, `qed.solve`) can swap entry points without touching the
// surrounding code. The only piece this kernel honestly does NOT yet
// surface is `compute_eigenvectors`: the locked Ritz vectors live on
// device and are not staged back to host on exit. The result struct
// returns the locked spectrum and iteration count; setting
// `compute_eigenvectors = true` simply leaves
// `krylov_basis_local` / `tridiag_eigenvectors` empty.
//
// Memory per rank, per cycle:
//   * (m_max + |locked|) * local_n * 16 B           (basis + locked)
//   * O(m_max) host doubles for alpha / beta / Eigen tridiag solve.
//
// Build: compiled iff WITH_MPI=ON && WITH_CUDA=ON && NCCL_FOUND.
// Consumers guard the include with `#ifdef ED_HAVE_NCCL`.
// =============================================================================

#pragma once

#ifdef WITH_MPI

#include <ed/distributed/distributed_lanczos.h>
#include <ed/distributed/distributed_operator.h>

#include <memory>

#include <mpi.h>

namespace ed::distributed {

class DistributedSymmetryOperator;

/// Distributed Krylov-Schur (thick-restart Lanczos with Ritz-pair
/// locking) on the multi-GPU `DistributedGPUOperator`. Collective on
/// `world_comm` -- every rank must call.
///
/// Throws `std::logic_error` if NCCL was not compiled in (i.e. the
/// build did not satisfy `WITH_CUDA=ON && NCCL_FOUND`).
DistributedLanczosResult distributed_krylov_schur_gpu(
    std::shared_ptr<class ::Operator> op,
    const DistributedLanczosOptions& options,
    MPI_Comm world_comm,
    int device_index = -1);

/// Symmetry-projected variant of ``distributed_krylov_schur_gpu``
/// (Phase D step 3). Same algorithm as the unsymmetrised GPU KS, but
/// every SpMV is the on-device orbit-row matvec exposed by
/// ``DistributedSymmetryOperatorGPU`` (NCCL pairwise SendRecv halo +
/// CSR symm SpMV) and the initial-vector scatter matches the rank-
/// major + LPT-orbit-permuted layout that
/// ``distributed_krylov_schur_symmetry`` and
/// ``distributed_lanczos_symmetry`` use, so the three routines accept
/// the same seed and produce comparable spectra. Reuses the caller-
/// owned ``DistributedSymmetryOperator`` (held by non-owning
/// ``shared_ptr`` internally).
///
/// Throws ``std::logic_error`` if NCCL was not compiled in.
DistributedLanczosResult distributed_krylov_schur_gpu_symmetry(
    const DistributedSymmetryOperator& op,
    const DistributedLanczosOptions& options,
    int device_index = -1);

}  // namespace ed::distributed

#endif  // WITH_MPI
