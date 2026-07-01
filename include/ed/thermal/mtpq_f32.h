#pragma once
// =============================================================================
// include/ed/thermal/mtpq_f32.h
//
// Single-precision (fp32) single-GPU microcanonical TPQ over the FULL Hilbert
// space. This is the memory-halving lane that lets the full 2^32 (32-site)
// Hilbert space run mTPQ on ONE 80 GB H100: complex<double> needs 2 x 68.7 GB
// = 137 GB (OOM) even for the lean 2-vector loop, whereas complex<float> needs
// 2 x 34.4 GB = 68.7 GB, which fits.
//
// Why a dedicated driver instead of mtpq_kernel<CudaBackend>:
//   * The generic tpq_kernel/mtpq_kernel are hardcoded to complex<double> and
//     allocate 3-4 work vectors (psi + scratchA + scratchB + the on_step
//     scratch). In fp32 that is 4 x 34.4 GB = 137 GB -> still OOM.
//   * This driver runs a LEAN 2-vector loop (psi + w only) using the analytic
//     step-norm  ||L*psi - H*psi||^2 = L^2 - 2 L <H> + <H^2>  (psi normalised)
//     so each step is one matvec + one reduction, no extra vectors.
//
// Precision policy: the state vectors and the matvec run in fp32 (the memory
// win); the per-step reductions (<psi|psi>, Re<psi|H psi>, <H psi|H psi>) are
// accumulated in DOUBLE by a custom kernel, so the energies/temperatures stay
// double-accurate and only the residual fp32 matvec error (~1e-3 relative)
// remains -- quantified against the double 16-site reference.
//
// Requires ``H.supports_cuda_f32() == true`` (only the full-Hilbert Operator
// on a WITH_CUDA build). The result shape matches ``MtpqResult`` so callers
// feed it into the same ``compute_tpq_thermo_from_trajectories`` aggregator as
// the double mTPQ lane.
// =============================================================================

#include <ed/thermal/mtpq_kernel.h>  // MtpqOptions, MtpqResult

namespace ed { class LinearOperator; }

namespace ed::thermal {

/// Run fp32 full-Hilbert microcanonical TPQ on the GPU. ``opts.large_value``
/// (the microcanonical shift L) and ``opts.max_iter`` are taken as-is (the
/// orchestrator's auto-tune fills them exactly as for the double lane).
/// Throws ``std::runtime_error`` on allocation failure (device OOM) or when
/// ``H`` does not support the fp32 device matvec.
MtpqResult mtpq_f32(const ed::LinearOperator& H, const MtpqOptions& opts);

}  // namespace ed::thermal
