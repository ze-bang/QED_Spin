// =============================================================================
// include/ed/gpu/kpm_dos_gpu.cuh
//
// GPU port of `ed::kpm_dos::compute_kpm_dos`.
//
// Same mathematical kernel as the CPU implementation in
// `src/solvers/cpu/kpm_dos.cpp`, but every D-dimensional vector lives on the
// GPU and every dot/axpy/scal goes through cuBLAS.  Mat-vec uses the supplied
// `GPUOperator` (typically a `GPUFixedSzOperator` for fixed-Sz sectors).
//
// Resource model
// --------------
//   * Three device complex vectors of length D   (3 * 16 B * D bytes).
//   * One device real scratch buffer of length D (used for cuBLAS norm only;
//     can be reused as a transient cuDoubleComplex when needed).
//   * No host buffers of length D — only the small (M-element) Chebyshev
//     moment array, the (N_quad-element) quadrature cache, and the (R-element)
//     per-sample diagnostic.
//
//   For D = C(32, 16) ≈ 6 × 10⁸ this is roughly 28 GiB of GPU memory, which
//   fits comfortably on a single H100 (80 GiB) alongside the FixedSz basis
//   table and the GPU operator's transform-data SoA.
//
// Doubling trick
// --------------
//   The KPM Chebyshev recurrence is v_{k+1} = 2 H_sc v_k − v_{k−1}, with
//   moments μ_k = ⟨r| T_k(H_sc) |r⟩.  Using the standard trick
//
//       μ_{2k}     = 2 ⟨v_k | v_k⟩       − μ_0
//       μ_{2k+1}   = 2 ⟨v_k | v_{k+1}⟩   − μ_1
//
//   we get *two* moments per mat-vec — half the matvecs of the naïve recursion
//   and we no longer need to retain |r⟩ during the loop, dropping a 4th vector.
//
// Spectral bound estimator
// ------------------------
//   Storing 100–150 Lanczos basis vectors at length 6 × 10⁸ would cost ≳ 1 TB
//   of device memory, so we instead run a 3-vector Lanczos *without*
//   reorthogonalization to get extreme Ritz values.  Loss of orthogonality
//   creates ghost copies of the converged extremes but does not move them, so
//   the [E_min, E_max] estimate stays accurate enough for the KPM rescaling
//   (which is then padded by the configured spectral_bound_buffer).
// =============================================================================

#pragma once

#ifdef WITH_CUDA

#include <cstdint>
#include <functional>
#include <vector>

#include <cuComplex.h>

#include <ed/solvers/kpm_dos.h>  // KPMDOSParameters, KPMDOSResult

class GPUOperator;

namespace ed::kpm_dos {

/// GPU implementation of `compute_kpm_dos`.
///
/// API mirrors the CPU version exactly so callers can swap CPU↔GPU based on
/// availability of the device matvec.  The returned `KPMDOSResult` is laid out
/// identically (Z, ⟨E⟩, Cv, S, F over `betas`, plus moments_raw /
/// moments_weighted, kpm_a/kpm_b, the spectral bound estimates, and an
/// optional reconstructed DOS on `dos_energies`).
///
/// @param gpu_op       Hermitian Hamiltonian on the device.  Must implement
///                     `matVecGPU(const cuDoubleComplex*, cuDoubleComplex*, int)`.
/// @param dim          Hilbert-space dimension D in the basis used by `gpu_op`.
/// @param betas        Inverse temperatures for thermodynamic post-processing.
/// @param dos_energies Optional energy grid for DOS reconstruction (skip if empty).
/// @param params       Control parameters; reused from the CPU version.
KPMDOSResult compute_kpm_dos_gpu(
    GPUOperator* gpu_op,
    std::uint64_t dim,
    const std::vector<double>& betas,
    const std::vector<double>& dos_energies,
    const KPMDOSParameters& params = {});

/// Phase E1 of the "Backend x Symmetries x Workflows" plan (May 2026):
/// matvec-callable variant of the GPU KPM driver. The callable must
/// take device-resident pointers (``const cuDoubleComplex*`` source,
/// ``cuDoubleComplex*`` destination) and implement ``y := H x`` in
/// device memory. This is the entry point used by
/// ``kpm_dos_kernel<CudaBackend>`` -- it lets us reuse the existing
/// GPU Chebyshev/Hutchinson loop with any device matvec (including
/// ``StreamingSymmetryOperator::bind_cuda_for_sector`` from Phase A),
/// not just ``GPUOperator``-backed ones.
using DeviceMatVec = std::function<void(const cuDoubleComplex*,
                                         cuDoubleComplex*, int)>;

KPMDOSResult compute_kpm_dos_gpu_with_matvec(
    DeviceMatVec matvec,
    std::uint64_t dim,
    const std::vector<double>& betas,
    const std::vector<double>& dos_energies,
    const KPMDOSParameters& params = {});

}  // namespace ed::kpm_dos

#endif  // WITH_CUDA
