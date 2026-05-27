// =============================================================================
// src/api/build_introspection.cpp
//
// `ed::has_cuda_build()`, `ed::has_mpi_build()`, `ed::has_nccl_build()`.
//
// Mirrors Python's `qed.has_cuda_build()` / `qed.has_mpi_build()` /
// `qed.has_nccl_build()` so C++ examples can gate device-specific
// pathways at runtime without `#ifdef`-cluttering the example body.
//
// The three predicates are wired against the same preprocessor flags
// the rest of the build uses:
//
//   * WITH_CUDA      -- compiled with CUDA runtime + cuBLAS / cuSPARSE.
//   * WITH_MPI       -- compiled against an MPI implementation.
//   * ED_HAVE_NCCL   -- compiled with NCCL collectives (multi-GPU lane).
// =============================================================================

#include <ed/api.h>

namespace ed {

bool has_cuda_build() noexcept {
#ifdef WITH_CUDA
    return true;
#else
    return false;
#endif
}

bool has_mpi_build() noexcept {
#ifdef WITH_MPI
    return true;
#else
    return false;
#endif
}

bool has_nccl_build() noexcept {
#ifdef ED_HAVE_NCCL
    return true;
#else
    return false;
#endif
}

}  // namespace ed
