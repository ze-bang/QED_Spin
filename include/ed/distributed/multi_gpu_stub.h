// =============================================================================
// include/ed/distributed/multi_gpu_stub.h    (Phase 3c)
//
// Compile-time NCCL availability flag and a minimal namespace that future
// multi-GPU code can hook into. THIS HEADER IS A DETECTION STUB. It does
// NOT pull in <nccl.h> -- that include lives in the (yet-to-be-written)
// implementation file under `src/distributed/multi_gpu.cu`, so that
// downstream targets that don't link NCCL never see the header.
//
// Status macros set by the build system:
//   ED_HAVE_NCCL  -- defined to 1 by CMake when NCCL was discovered AND
//                    both WITH_CUDA and WITH_MPI are ON. Otherwise undefined.
//
// Honest-scope notes (see SCALING.md "Phase 3c"):
//   * Validating multi-GPU correctness needs >= 2 visible CUDA devices,
//     a working NCCL/RDMA stack, and an HPC slot we have not booked.
//     Until that work happens, this header exists ONLY so that:
//        - the build system records whether NCCL is installed;
//        - any future ed::distributed::multi_gpu code can declare its
//          API in a stable place;
//        - users can opt into a NCCL-enabled build today and we can flip
//          the runtime switch in a follow-up PR without touching CMake.
//   * The current single-GPU GPULanczos in `ed::gpu` is unaffected.
//   * The Phase 3b CPU-MPI path (DistributedOperator / DistributedLanczos /
//     DistributedFTLM) remains the recommended distributed-memory route
//     for "honest 40" until Phase 3c lands.
// =============================================================================
#pragma once

#include <cstdint>
#include <string>

namespace ed::distributed::multi_gpu {

/// True iff this build was configured with NCCL discovered AND with
/// CUDA + MPI enabled. Safe to query from any TU (no NCCL headers needed).
inline constexpr bool nccl_compiled_in() noexcept {
#ifdef ED_HAVE_NCCL
    return true;
#else
    return false;
#endif
}

/// Human-readable status string for diagnostics / CLI banners.
inline std::string nccl_status_string() {
    if (nccl_compiled_in()) {
        return "ed::distributed::multi_gpu: NCCL detected at build time "
               "(Phase 3c stub; runtime kernels not yet implemented)";
    }
    return "ed::distributed::multi_gpu: NCCL NOT detected at build time "
           "(Phase 3c multi-GPU path disabled)";
}

}  // namespace ed::distributed::multi_gpu
