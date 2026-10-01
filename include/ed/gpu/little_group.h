#pragma once
// =============================================================================
// include/ed/gpu/little_group.h
//
// Batched GPU eigensolver for the little-group engine's dense blocks. Its
// consumer is the ed::sectors dense batch (DenseBatch, walk.h).
//
// Design (minimal host<->device traffic): the host builds every dense block
// (irregular orbit/isotypic work), ONE cudaMemcpy uploads all packed blocks,
// an 8-stream cusolverDnZheevd pool overlaps the per-block eigensolves
// (the tiny-block batching mechanism), and ONE cudaMemcpy downloads the
// concatenated eigenvalue array. Two transfers total, any block count.
// =============================================================================

#include <complex>
#include <cstddef>
#include <vector>

namespace ed::solvers {

/// Column-major dense blocks packed back to back.
struct LgBlocksPacked {
    std::vector<std::size_t> offset;           ///< start of block b in `data`
    std::vector<int>         block_dim;        ///< n_b (block b is n_b x n_b)
    std::vector<int>         block_irrep_dim;  ///< d_sigma (bookkeeping only)
    std::vector<std::complex<double>> data;    ///< packed column-major blocks
};

/// Batched Hermitian eigensolve (eigenvalues only) of the packed blocks on the
/// GPU. Returns the eigenvalues concatenated per block, ascending within each
/// block. Throws on any CUDA/cuSOLVER failure (callers degrade to the CPU
/// path). Defined in src/gpu/little_group.cu -- only linked into
/// WITH_CUDA builds; call sites must be #ifdef WITH_CUDA guarded.
[[nodiscard]] std::vector<double>
lg_blocks_batched_eigenvalues_gpu(const LgBlocksPacked& P);

}  // namespace ed::solvers
