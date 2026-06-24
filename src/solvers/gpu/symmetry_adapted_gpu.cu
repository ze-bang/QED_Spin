// =============================================================================
// src/solvers/gpu/symmetry_adapted_gpu.cu
//
// GPU consumers for the symmetry-adapted (Sz / abelian / non-abelian spatial /
// combined) reduction. The reduction yields MANY small dense blocks H_Γ; the
// FLOP-dominant, GPU-natural operation is a BATCHED Hermitian eigensolve of all
// blocks at once. Design (minimal host↔device traffic):
//
//   host: build every block H_Γ (cheap, irregular orbit/SAB work)  -- ONE pass
//   H->D: upload ALL packed blocks in a single cudaMemcpy
//   GPU : multi-stream cusolverDnZheevd  (concurrent per-block eigensolves)
//   D->H: download the (small) concatenated eigenvalue array  -- single copy
//   host: trivial GS / canonical-thermo reduction on the eigenvalues
//
// Two transfers total, regardless of block count. The eigensolve (Σ O(n_Γ^3))
// runs entirely on the device, overlapped across a stream pool.
// =============================================================================

#include <ed/symmetry/symmetry_adapted.h>

#include <cuComplex.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>

#include <algorithm>
#include <stdexcept>
#include <string>
#include <vector>

namespace ed::symmetry {
namespace {

#define ED_CUDA_CHECK(call)                                                    \
    do {                                                                       \
        cudaError_t _e = (call);                                               \
        if (_e != cudaSuccess)                                                 \
            throw std::runtime_error(std::string("symmetry_adapted_gpu CUDA: ")\
                                     + cudaGetErrorString(_e));                \
    } while (0)

#define ED_CUSOLVER_CHECK(call)                                                \
    do {                                                                       \
        cusolverStatus_t _s = (call);                                          \
        if (_s != CUSOLVER_STATUS_SUCCESS)                                     \
            throw std::runtime_error("symmetry_adapted_gpu cuSOLVER error "    \
                                     + std::to_string(static_cast<int>(_s)));  \
    } while (0)

// Batched Hermitian eigensolve (eigenvalues only) of the packed column-major
// blocks. Returns the eigenvalues concatenated per block (block b occupies
// [eig_offset[b], eig_offset[b]+block_dim[b])).
std::vector<double> batched_block_eigenvalues_gpu(const SymBlocksPacked& P) {
    const std::size_t nblk = P.block_dim.size();
    if (nblk == 0) return {};

    std::vector<std::size_t> eig_off(nblk);
    std::size_t total_eigs = 0;
    int maxdim = 0;
    for (std::size_t b = 0; b < nblk; ++b) {
        eig_off[b] = total_eigs;
        total_eigs += static_cast<std::size_t>(P.block_dim[b]);
        maxdim = std::max(maxdim, P.block_dim[b]);
    }

    // --- single upload of all blocks ---------------------------------------
    cuDoubleComplex* d_data = nullptr;
    double*          d_eigs = nullptr;
    ED_CUDA_CHECK(cudaMalloc(&d_data, P.data.size() * sizeof(cuDoubleComplex)));
    ED_CUDA_CHECK(cudaMalloc(&d_eigs, total_eigs * sizeof(double)));
    ED_CUDA_CHECK(cudaMemcpy(
        d_data, reinterpret_cast<const cuDoubleComplex*>(P.data.data()),
        P.data.size() * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));

    // --- stream/handle pool for concurrent per-block eigensolves -----------
    const int K = std::max(1, std::min<int>(static_cast<int>(nblk), 8));
    std::vector<cudaStream_t>       streams(K);
    std::vector<cusolverDnHandle_t> handles(K);
    std::vector<cuDoubleComplex*>   d_work(K, nullptr);
    std::vector<int*>               d_info(K, nullptr);

    // Workspace sized for the largest block (eigenvalues-only); reused by every
    // block on that handle's stream.
    int lwork_max = 1;
    {
        cusolverDnHandle_t h0;
        ED_CUSOLVER_CHECK(cusolverDnCreate(&h0));
        int lw = 0;
        ED_CUSOLVER_CHECK(cusolverDnZheevd_bufferSize(
            h0, CUSOLVER_EIG_MODE_NOVECTOR, CUBLAS_FILL_MODE_UPPER,
            maxdim, d_data, maxdim, d_eigs, &lw));
        lwork_max = std::max(1, lw);
        cusolverDnDestroy(h0);
    }
    for (int k = 0; k < K; ++k) {
        ED_CUDA_CHECK(cudaStreamCreate(&streams[k]));
        ED_CUSOLVER_CHECK(cusolverDnCreate(&handles[k]));
        ED_CUSOLVER_CHECK(cusolverDnSetStream(handles[k], streams[k]));
        ED_CUDA_CHECK(cudaMalloc(&d_work[k], static_cast<std::size_t>(lwork_max) * sizeof(cuDoubleComplex)));
        ED_CUDA_CHECK(cudaMalloc(&d_info[k], sizeof(int)));
    }

    // Distribute blocks round-robin across the handle/stream pool.
    for (std::size_t b = 0; b < nblk; ++b) {
        const int k = static_cast<int>(b % static_cast<std::size_t>(K));
        const int n = P.block_dim[b];
        cuDoubleComplex* A = d_data + P.offset[b];
        ED_CUSOLVER_CHECK(cusolverDnZheevd(
            handles[k], CUSOLVER_EIG_MODE_NOVECTOR, CUBLAS_FILL_MODE_UPPER,
            n, A, n, d_eigs + eig_off[b], d_work[k], lwork_max, d_info[k]));
    }
    for (int k = 0; k < K; ++k) ED_CUDA_CHECK(cudaStreamSynchronize(streams[k]));

    // --- single download of all eigenvalues --------------------------------
    std::vector<double> eigs(total_eigs);
    ED_CUDA_CHECK(cudaMemcpy(eigs.data(), d_eigs, total_eigs * sizeof(double),
                             cudaMemcpyDeviceToHost));

    for (int k = 0; k < K; ++k) {
        cudaFree(d_work[k]);
        cudaFree(d_info[k]);
        cusolverDnDestroy(handles[k]);
        cudaStreamDestroy(streams[k]);
    }
    cudaFree(d_data);
    cudaFree(d_eigs);
    return eigs;
}

// Expand the per-block eigenvalues into the full spectrum (each ×d_Γ).
std::vector<double> full_spectrum_from_blocks(const SymBlocksPacked& P,
                                              const std::vector<double>& block_eigs) {
    std::vector<double> full;
    std::size_t off = 0;
    for (std::size_t b = 0; b < P.block_dim.size(); ++b) {
        const int n = P.block_dim[b];
        const int d = P.block_irrep_dim[b];
        for (int e = 0; e < n; ++e)
            for (int r = 0; r < d; ++r)            // physical d_Γ degeneracy
                full.push_back(block_eigs[off + static_cast<std::size_t>(e)]);
        off += static_cast<std::size_t>(n);
    }
    return full;
}

}  // namespace

SymAdaptedSpectrum symmetry_adapted_spectrum_gpu(
    const ConnectFn&                     connect,
    const GroupIrreps&                   gi,
    const std::vector<std::vector<int>>& max_clique,
    int                                  n_sites,
    int                                  n_up)
{
    const SymBlocksPacked P =
        build_symmetry_blocks_packed(connect, gi, max_clique, n_sites, n_up);
    const std::vector<double> block_eigs = batched_block_eigenvalues_gpu(P);

    SymAdaptedSpectrum out;
    out.block_irrep_dim = P.block_irrep_dim;
    out.block_size      = P.block_dim;
    out.eigenvalues     = full_spectrum_from_blocks(P, block_eigs);
    std::sort(out.eigenvalues.begin(), out.eigenvalues.end());
    return out;
}

ThermodynamicData symmetry_adapted_thermodynamics_gpu(
    const ConnectFn&                     connect,
    const GroupIrreps&                   gi,
    const std::vector<std::vector<int>>& max_clique,
    int                                  n_sites,
    const std::vector<double>&           temperatures,
    int                                  n_up)
{
    const SymBlocksPacked P =
        build_symmetry_blocks_packed(connect, gi, max_clique, n_sites, n_up);
    const std::vector<double> block_eigs = batched_block_eigenvalues_gpu(P);
    const std::vector<double> full = full_spectrum_from_blocks(P, block_eigs);
    return canonical_thermo_from_eigs(full, temperatures);  // identical reduction to CPU
}

}  // namespace ed::symmetry
