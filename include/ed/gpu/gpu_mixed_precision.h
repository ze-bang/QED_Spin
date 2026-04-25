#pragma once

// =============================================================================
// Mixed-precision GPU SpMV                                       (Phase 3a #3)
// =============================================================================
//
// On a memory-bandwidth-bound SpMV (the regime above N=2^15 where the
// `cusparseSpMV` CSR path takes over from the matrix-free kernels),
// switching the value array from FP64 complex (16 B) to FP32 complex (8 B)
// halves the value-traffic per element and yields a ~1.7-2x wallclock
// speedup at the cost of one or two extra Krylov iterations (for the dot /
// normalize / axpy steps to compensate for the lower-precision matvec
// intermediates).
//
// The implementation lives behind a single env knob:
//
//   ED_GPU_MIXED_PRECISION_SPMV
//       If set to "1" (and the cuSPARSE CSR pathway is selected for the
//       current problem), GPUOperator::applyCusparse routes through
//       applyCusparseMixed: it casts the FP64 input vector down to FP32,
//       runs cusparseSpMV with CUDA_C_32F against an FP32 copy of the CSR
//       value array (built lazily and cached), then casts the FP32 output
//       back to FP64. The host-side dot / normalize / axpy in
//       GPULanczos / GPUFTLM / etc. remain in FP64, so global
//       orthogonality is preserved at FP64 precision.
//
// Scope of the present landing:
//   * Only the cuSPARSE CSR pathway is mixed-precision-aware. The
//     matrix-free pathways (WARP_REDUCTION / BRANCH_FREE_SCATTER /
//     SHARED_MEMORY) stay FP64.
//   * Matrix-free mixed precision is a separate (much larger) kernel-
//     templating job and is deferred to a follow-up.
//   * When the env knob is set but the selected pathway is not
//     CUSPARSE_CSR, applyCusparse is not invoked and the run silently
//     stays in FP64 -- the knob requests but does not force.
//   * Symmetrized / fixed-Sz operators do not currently build a CSR;
//     they fall back to FP64.
//
// Cache invariants:
//   * The FP32 CSR is a *projection* of the FP64 CSR: row offsets and
//     column indices are aliased (same int32 arrays); only the value
//     array is duplicated as cuFloatComplex.
//   * On any operator mutation that calls invalidateDerivedCaches(), both
//     the FP64 and FP32 CSR caches are torn down via freeCsrDeviceData()
//     + freeCsrFp32DeviceData().
// =============================================================================

namespace ed {
namespace gpu {

// True iff the user has set ED_GPU_MIXED_PRECISION_SPMV to "1". Cached
// per-process (the knob is queried once on first call) -- mirrors the
// pattern used for ED_GPU_TIMING / ED_GPU_DISABLE_CUSPARSE.
bool gpu_mixed_precision_spmv_enabled();

}  // namespace gpu
}  // namespace ed
