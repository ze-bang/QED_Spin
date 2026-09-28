// =============================================================================
// src/api/build_introspection.cpp
//
// `ed::has_cuda_build()`, `ed::has_mpi_build()`, `ed::has_nccl_build()`.
//
// Mirrors Python's `qed.has_cuda_build()` / `qed.has_mpi_build()` /
// `qed.has_nccl_build()` so C++ examples can gate device-specific
// pathways at runtime without `#ifdef`-cluttering the example body.
//
//   * has_cuda_build -- true when compiled with WITH_CUDA (CUDA runtime +
//                       cuBLAS).
//   * has_mpi_build, has_nccl_build -- always false; the library has no
//                       MPI or NCCL support.
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
    return false;
}

bool has_nccl_build() noexcept {
    return false;
}

}  // namespace ed
