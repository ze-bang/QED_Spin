#pragma once
// =============================================================================
// include/ed/gpu/little_group.h
//
// Batched GPU eigensolver for the engine's dense blocks. Its consumer is the ed::sectors dense
// batch (DenseBatch, walk.h).
//
// Design (minimal host<->device traffic): the host builds every dense block (irregular orbit
// work), ONE cudaMemcpy uploads all packed blocks, an 8-stream pool of cuSOLVER's 64-bit syevd
// (cusolverDnXsyevd) overlaps the per-block eigensolves, and ONE cudaMemcpy downloads the
// concatenated eigenvalue array. A real block (a real symmetric matrix: a real momentum under
// time reversal, a real H in a real sector) travels and is solved in real arithmetic -- half the
// bytes, ~2x faster (H100, n = 12870: 1.9 s against 3.9 s, job 62670841).
// =============================================================================

#include <complex>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace ed::solvers {

/// Column-major dense blocks packed back to back: a real block as n^2 doubles, a complex one as
/// n^2 complex values (2 n^2 doubles, interleaved).
struct LgBlocksPacked {
    std::vector<std::size_t>  offset;      ///< start of block b in `data`, in doubles
    std::vector<std::int64_t> block_dim;   ///< n_b (block b is n_b x n_b)
    std::vector<char>         real;        ///< block b is stored (and solved) real
    std::vector<double>       data;

    [[nodiscard]] std::size_t bytes() const noexcept { return data.size() * sizeof(double); }
    /// Append an n x n column-major block: real (n^2 doubles) or complex (n^2 complex values).
    /// Every block starts on 256 bytes: a complex block after a real one of odd n^2 would sit 8
    /// bytes off the 16 its loads need (cuSOLVER faulted with a misaligned address, gate 62672849).
    void add_real(const double* a, std::int64_t n);
    void add_complex(const std::complex<double>* a, std::int64_t n);

private:
    void align() { data.resize((data.size() + 31) / 32 * 32, 0.0); }
};

inline void LgBlocksPacked::add_real(const double* a, std::int64_t n) {
    align();
    offset.push_back(data.size());
    block_dim.push_back(n);
    real.push_back(1);
    data.insert(data.end(), a, a + n * n);
}

inline void LgBlocksPacked::add_complex(const std::complex<double>* a, std::int64_t n) {
    align();
    offset.push_back(data.size());
    block_dim.push_back(n);
    real.push_back(0);
    const double* d = reinterpret_cast<const double*>(a);   // std::complex<double> is two doubles
    data.insert(data.end(), d, d + 2 * n * n);
}

/// Batched symmetric / Hermitian eigensolve (eigenvalues only) of the packed blocks on the GPU.
/// Returns the eigenvalues concatenated per block, ascending within each block. Throws on any
/// CUDA/cuSOLVER failure (callers degrade to the CPU path). Defined in src/gpu/little_group.cu --
/// only linked into WITH_CUDA builds; call sites must be #ifdef WITH_CUDA guarded.
[[nodiscard]] std::vector<double>
lg_blocks_batched_eigenvalues_gpu(const LgBlocksPacked& P);

/// The device bytes lg_blocks_batched_eigenvalues_gpu needs beyond the matrices for a single
/// block of dimension n (its workspace and eigenvalues). Throws like the solver.
[[nodiscard]] std::size_t lg_block_workspace_bytes_gpu(std::int64_t n, bool real);

}  // namespace ed::solvers
