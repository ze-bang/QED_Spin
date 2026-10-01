// =============================================================================
// src/orchestrator/orch_common.cpp -- shared orchestrator plumbing
// (the exact-small thermal env probe).
// Part of the workflow orchestrator; see orchestrator_internal.h for the
// file map.
// =============================================================================

#include "orchestrator_internal.h"

namespace ed::workflows {
namespace orch_detail {

// ED_THERMAL_EXACT_SMALL=0 forces the real sampling kernel even at
// D <= SMALL_THERMAL_DIM. The fallback is a strict accuracy win for USERS, but
// it silently removes the sampling kernels from any accuracy test whose system
// fits under the cutoff (e.g. test_thermal_dense_ref, N=6, dim=64). Tests
// that mean to gate a KERNEL set this to 0; nothing in production should.
// Read per call so a test can toggle it without restarting the process.
[[nodiscard]] bool exact_small_thermal_enabled() noexcept {
    return ed::env::flag("ED_THERMAL_EXACT_SMALL", true);
}

}  // namespace orch_detail
}  // namespace ed::workflows
