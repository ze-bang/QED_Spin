// =============================================================================
// include/ed/distributed/multi_gpu_stub.h    (Phase 3c)
//
// Backwards-compatibility shim: this header used to be a detection-only
// stub exposing `nccl_compiled_in()` / `nccl_status_string()` while the
// runtime kernels were still TBD. Phase 3c stage 1 promoted the API to
// a real implementation in `<ed/distributed/multi_gpu.h>` (with NCCL
// collectives + RAII communicator), so any existing user of the stub
// gets the new surface for free by transitively including the new
// header here.
//
// Prefer `<ed/distributed/multi_gpu.h>` in new code.
// =============================================================================
#pragma once

#include <ed/distributed/multi_gpu.h>
