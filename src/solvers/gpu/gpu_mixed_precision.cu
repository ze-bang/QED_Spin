// =============================================================================
// src/solvers/gpu/gpu_mixed_precision.cu
//
// Single-source-of-truth for the ED_GPU_MIXED_PRECISION_SPMV env knob
// (Phase 3a #3). The actual cuSPARSE-FP32 plumbing lives in
// gpu_operator.cu (buildCsrFp32OnDevice / applyCusparseMixed); the only
// reason this is a separate translation unit is so the cached env-flag
// initializer is *one* `static` across the GPU library, not duplicated in
// every caller.
// =============================================================================

#include "ed/gpu/gpu_mixed_precision.h"

#include <cstdlib>

namespace ed {
namespace gpu {

bool gpu_mixed_precision_spmv_enabled() {
    // Re-read each call: the FP32 path is gated per-operator via a lazy
    // build of the FP32 CSR cache (see GPUOperator::buildCsrFp32OnDevice),
    // not via a one-shot static init, so tests can flip the knob between
    // operators in the same process without re-loading the library.
    const char* v = std::getenv("ED_GPU_MIXED_PRECISION_SPMV");
    if (!v || !*v) return false;
    return v[0] == '1' || v[0] == 't' || v[0] == 'T' ||
           v[0] == 'y' || v[0] == 'Y';
}

}  // namespace gpu
}  // namespace ed
