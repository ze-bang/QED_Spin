// =============================================================================
// src/solvers/gpu/little_group_gpu.cu
//
// Batched GPU eigensolver for the little-group engine's dense blocks. See the
// header for history (recovered from the Family-6-removed SAB GPU twin) and
// the two-transfer design.
// =============================================================================

#include <ed/solvers/little_group_gpu.h>

#include <cuComplex.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>

#include <algorithm>
#include <stdexcept>
#include <string>
#include <vector>

namespace ed::solvers {
namespace {

#define ED_CUDA_CHECK(call)                                                    \
    do {                                                                       \
        cudaError_t _e = (call);                                               \
        if (_e != cudaSuccess)                                                 \
            throw std::runtime_error(std::string("little_group_gpu CUDA: ")    \
                                     + cudaGetErrorString(_e));                \
    } while (0)

#define ED_CUSOLVER_CHECK(call)                                                \
    do {                                                                       \
        cusolverStatus_t _s = (call);                                          \
        if (_s != CUSOLVER_STATUS_SUCCESS)                                     \
            throw std::runtime_error("little_group_gpu cuSOLVER error "        \
                                     + std::to_string(static_cast<int>(_s)));  \
    } while (0)

}  // namespace

std::vector<double>
lg_blocks_batched_eigenvalues_gpu(const LgBlocksPacked& P) {
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

    // Device resources, all null-initialised so the cleanup below is safe on
    // any partial-init throw path (no leak of memory / handles / streams).
    const int K = std::max(1, std::min<int>(static_cast<int>(nblk), 8));
    cuDoubleComplex* d_data = nullptr;
    double*          d_eigs = nullptr;
    int*             d_info = nullptr;              // per-block convergence code
    std::vector<cudaStream_t>       streams(K, nullptr);
    std::vector<cusolverDnHandle_t> handles(K, nullptr);
    std::vector<cuDoubleComplex*>   d_work(K, nullptr);
    int n_created = 0;                              // # fully-built pool slots

    auto cleanup = [&]() noexcept {
        for (int k = 0; k < n_created; ++k) {
            if (d_work[k])  cudaFree(d_work[k]);
            if (handles[k]) cusolverDnDestroy(handles[k]);
            if (streams[k]) cudaStreamDestroy(streams[k]);
        }
        if (d_info) cudaFree(d_info);
        if (d_eigs) cudaFree(d_eigs);
        if (d_data) cudaFree(d_data);
    };

    std::vector<double> eigs(total_eigs);
    try {
        // --- single upload of all blocks -----------------------------------
        ED_CUDA_CHECK(cudaMalloc(&d_data, P.data.size() * sizeof(cuDoubleComplex)));
        ED_CUDA_CHECK(cudaMalloc(&d_eigs, total_eigs * sizeof(double)));
        ED_CUDA_CHECK(cudaMalloc(&d_info, nblk * sizeof(int)));
        ED_CUDA_CHECK(cudaMemcpy(
            d_data, reinterpret_cast<const cuDoubleComplex*>(P.data.data()),
            P.data.size() * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));

        // Workspace sized for the largest block (lwork is non-decreasing in n,
        // so this upper bound is reused safely by every block on that handle).
        int lwork_max = 1;
        {
            cusolverDnHandle_t h0 = nullptr;
            ED_CUSOLVER_CHECK(cusolverDnCreate(&h0));
            int lw = 0;
            const cusolverStatus_t qs = cusolverDnZheevd_bufferSize(
                h0, CUSOLVER_EIG_MODE_NOVECTOR, CUBLAS_FILL_MODE_UPPER,
                maxdim, d_data, maxdim, d_eigs, &lw);
            cusolverDnDestroy(h0);
            ED_CUSOLVER_CHECK(qs);
            lwork_max = std::max(1, lw);
        }

        // --- stream/handle pool for concurrent per-block eigensolves -------
        for (int k = 0; k < K; ++k) {
            ED_CUDA_CHECK(cudaStreamCreate(&streams[k]));
            ED_CUSOLVER_CHECK(cusolverDnCreate(&handles[k]));
            ED_CUSOLVER_CHECK(cusolverDnSetStream(handles[k], streams[k]));
            ED_CUDA_CHECK(cudaMalloc(
                &d_work[k],
                static_cast<std::size_t>(lwork_max) * sizeof(cuDoubleComplex)));
            n_created = k + 1;
        }

        // Distribute blocks round-robin across the handle/stream pool. Blocks
        // on the same stream serialise (so they may share d_work[k]); blocks
        // on different streams overlap. d_data / d_eigs / d_info slices are
        // disjoint.
        for (std::size_t b = 0; b < nblk; ++b) {
            const int k = static_cast<int>(b % static_cast<std::size_t>(K));
            const int n = P.block_dim[b];
            cuDoubleComplex* A = d_data + P.offset[b];
            ED_CUSOLVER_CHECK(cusolverDnZheevd(
                handles[k], CUSOLVER_EIG_MODE_NOVECTOR, CUBLAS_FILL_MODE_UPPER,
                n, A, n, d_eigs + eig_off[b], d_work[k], lwork_max,
                d_info + static_cast<std::ptrdiff_t>(b)));
        }
        for (int k = 0; k < K; ++k)
            ED_CUDA_CHECK(cudaStreamSynchronize(streams[k]));

        // Convergence / argument check for every block.
        std::vector<int> info(nblk);
        ED_CUDA_CHECK(cudaMemcpy(info.data(), d_info, nblk * sizeof(int),
                                 cudaMemcpyDeviceToHost));
        for (std::size_t b = 0; b < nblk; ++b)
            if (info[b] != 0)
                throw std::runtime_error(
                    "little_group_gpu: cusolverDnZheevd did not converge for "
                    "block " + std::to_string(b) +
                    " (info=" + std::to_string(info[b]) + ")");

        // --- single download of all eigenvalues ----------------------------
        ED_CUDA_CHECK(cudaMemcpy(eigs.data(), d_eigs,
                                 total_eigs * sizeof(double),
                                 cudaMemcpyDeviceToHost));
    } catch (...) {
        cleanup();
        throw;
    }
    cleanup();
    return eigs;
}

}  // namespace ed::solvers
