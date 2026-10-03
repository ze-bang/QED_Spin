// =============================================================================
// src/gpu/little_group.cu
//
// Batched GPU eigensolver for the engine's dense blocks. See the header for the two-transfer
// design. cuSOLVER's 64-bit generic syevd takes the data type per call, so one pool solves real
// and complex blocks alike, and no block is limited by 32-bit indices.
// =============================================================================
#include <ed/gpu/little_group.h>

#include <cuda_runtime.h>
#include <cusolverDn.h>

#include <algorithm>
#include <stdexcept>
#include <string>
#include <vector>

namespace ed::solvers {

namespace {

#define ED_CUDA_CHECK(call)                                                                                            \
    do {                                                                                                               \
        cudaError_t _e = (call);                                                                                       \
        if (_e != cudaSuccess)                                                                                         \
            throw std::runtime_error(std::string("little_group_gpu CUDA: ") + cudaGetErrorString(_e));                 \
    } while (0)

#define ED_CUSOLVER_CHECK(call)                                                                                        \
    do {                                                                                                               \
        cusolverStatus_t _s = (call);                                                                                  \
        if (_s != CUSOLVER_STATUS_SUCCESS)                                                                             \
            throw std::runtime_error("little_group_gpu cuSOLVER error " + std::to_string(static_cast<int>(_s)));       \
    } while (0)

// syevd's device and host workspace for one n x n block of the given type (A and W may be null:
// the query reads only the sizes).
void workspace(cusolverDnHandle_t h, cusolverDnParams_t p, std::int64_t n, bool real, std::size_t& dev,
               std::size_t& host) {
    const cudaDataType t = real ? CUDA_R_64F : CUDA_C_64F;
    ED_CUSOLVER_CHECK(cusolverDnXsyevd_bufferSize(h, p, CUSOLVER_EIG_MODE_NOVECTOR, CUBLAS_FILL_MODE_UPPER, n, t,
                                                  nullptr, std::max<std::int64_t>(1, n), CUDA_R_64F, nullptr, t, &dev,
                                                  &host));
}

}  // namespace

std::size_t lg_block_workspace_bytes_gpu(std::int64_t n, bool real) {
    cusolverDnHandle_t h = nullptr;
    cusolverDnParams_t p = nullptr;
    std::size_t dev = 0, host = 0;
    try {
        ED_CUSOLVER_CHECK(cusolverDnCreate(&h));
        ED_CUSOLVER_CHECK(cusolverDnCreateParams(&p));
        workspace(h, p, n, real, dev, host);
    } catch (...) {
        if (p) cusolverDnDestroyParams(p);
        if (h) cusolverDnDestroy(h);
        throw;
    }
    cusolverDnDestroyParams(p);
    cusolverDnDestroy(h);
    return dev + static_cast<std::size_t>(n) * sizeof(double) + sizeof(int);
}

std::vector<double> lg_blocks_batched_eigenvalues_gpu(const LgBlocksPacked& P) {
    const std::size_t nblk = P.block_dim.size();
    if (nblk == 0) return {};

    std::vector<std::size_t> eig_off(nblk);
    std::size_t total_eigs = 0;
    std::int64_t max_real = 0, max_complex = 0;
    for (std::size_t b = 0; b < nblk; ++b) {
        eig_off[b] = total_eigs;
        total_eigs += static_cast<std::size_t>(P.block_dim[b]);
        if (P.real[b])
            max_real = std::max(max_real, P.block_dim[b]);
        else
            max_complex = std::max(max_complex, P.block_dim[b]);
    }

    // Device resources, all null-initialised so the cleanup below is safe on any partial-init
    // throw path (no leak of memory / handles / streams). The pool: up to 8 streams, fewer when
    // their workspaces (each sized for the largest block) would not fit beside the matrices.
    int K = std::max(1, std::min<int>(static_cast<int>(nblk), 8));
    {
        cusolverDnHandle_t h = nullptr;
        cusolverDnParams_t p = nullptr;
        std::size_t ws = 1, d = 0, hb = 0;
        try {
            ED_CUSOLVER_CHECK(cusolverDnCreate(&h));
            ED_CUSOLVER_CHECK(cusolverDnCreateParams(&p));
            if (max_real > 0) {
                workspace(h, p, max_real, true, d, hb);
                ws = std::max(ws, d);
            }
            if (max_complex > 0) {
                workspace(h, p, max_complex, false, d, hb);
                ws = std::max(ws, d);
            }
        } catch (...) {
            if (p) cusolverDnDestroyParams(p);
            if (h) cusolverDnDestroy(h);
            throw;
        }
        cusolverDnDestroyParams(p);
        cusolverDnDestroy(h);
        std::size_t free_b = 0, total_b = 0;
        if (cudaMemGetInfo(&free_b, &total_b) == cudaSuccess) {
            const std::size_t room =
                free_b > P.bytes() ? (free_b - P.bytes()) / 5 * 4 : 0;   // 80% of what the matrices leave
            K = std::max(1, std::min<int>(K, static_cast<int>(room / ws)));
        } else {
            cudaGetLastError();
        }
    }
    double* d_data = nullptr;
    double* d_eigs = nullptr;
    int* d_info = nullptr;                           // per-block convergence code
    std::vector<cudaStream_t> streams(K, nullptr);
    std::vector<cusolverDnHandle_t> handles(K, nullptr);
    std::vector<cusolverDnParams_t> params(K, nullptr);
    std::vector<void*> d_work(K, nullptr);
    std::vector<std::size_t> d_bytes(K, 0);
    std::vector<std::vector<char>> h_work(K);
    auto cleanup = [&]() noexcept {
        for (int k = 0; k < K; ++k) {
            if (d_work[k]) cudaFree(d_work[k]);
            if (params[k]) cusolverDnDestroyParams(params[k]);
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
        ED_CUDA_CHECK(cudaMalloc(&d_data, std::max<std::size_t>(1, P.bytes())));
        ED_CUDA_CHECK(cudaMalloc(&d_eigs, total_eigs * sizeof(double)));
        ED_CUDA_CHECK(cudaMalloc(&d_info, nblk * sizeof(int)));
        ED_CUDA_CHECK(cudaMemcpy(d_data, P.data.data(), P.bytes(), cudaMemcpyHostToDevice));

        // --- stream/handle pool, each slot's workspace sized for the largest block of either
        // type (the workspace is non-decreasing in n, so the bound serves every block on it) ---
        for (int k = 0; k < K; ++k) {
            ED_CUDA_CHECK(cudaStreamCreate(&streams[k]));
            ED_CUSOLVER_CHECK(cusolverDnCreate(&handles[k]));
            ED_CUSOLVER_CHECK(cusolverDnSetStream(handles[k], streams[k]));
            ED_CUSOLVER_CHECK(cusolverDnCreateParams(&params[k]));
            std::size_t dev = 1, host = 1, d, h;
            if (max_real > 0) {
                workspace(handles[k], params[k], max_real, true, d, h);
                dev = std::max(dev, d);
                host = std::max(host, h);
            }
            if (max_complex > 0) {
                workspace(handles[k], params[k], max_complex, false, d, h);
                dev = std::max(dev, d);
                host = std::max(host, h);
            }
            ED_CUDA_CHECK(cudaMalloc(&d_work[k], dev));
            d_bytes[k] = dev;
            h_work[k].resize(host);
        }

        // Blocks to streams largest first, each to the least-loaded stream (load ~ n^3, a complex
        // block twice a real one: H100 n = 12870 3.9 s against 1.9 s), so a large block runs beside
        // the small ones instead of after them. Blocks on one stream serialise (they share its
        // workspace); blocks on different streams overlap. The data / eigs / info slices are disjoint.
        std::vector<std::size_t> order(nblk);
        for (std::size_t b = 0; b < nblk; ++b) order[b] = b;
        std::stable_sort(order.begin(), order.end(),
                         [&P](std::size_t a, std::size_t c) { return P.block_dim[a] > P.block_dim[c]; });
        std::vector<double> load(static_cast<std::size_t>(K), 0.0);
        for (const std::size_t b : order) {
            const int k = static_cast<int>(std::min_element(load.begin(), load.end()) - load.begin());
            const double n3 = static_cast<double>(P.block_dim[b]) * static_cast<double>(P.block_dim[b])
                              * static_cast<double>(P.block_dim[b]);
            load[static_cast<std::size_t>(k)] += P.real[b] ? n3 : 2.0 * n3;
            const std::int64_t n = P.block_dim[b];
            const cudaDataType t = P.real[b] ? CUDA_R_64F : CUDA_C_64F;
            ED_CUSOLVER_CHECK(cusolverDnXsyevd(
                handles[k], params[k], CUSOLVER_EIG_MODE_NOVECTOR, CUBLAS_FILL_MODE_UPPER, n, t, d_data + P.offset[b],
                std::max<std::int64_t>(1, n), CUDA_R_64F, d_eigs + eig_off[b], t, d_work[k], d_bytes[k],
                h_work[k].data(), h_work[k].size(), d_info + static_cast<std::ptrdiff_t>(b)));
        }
        for (int k = 0; k < K; ++k) ED_CUDA_CHECK(cudaStreamSynchronize(streams[k]));

        // Convergence / argument check for every block.
        std::vector<int> info(nblk);
        ED_CUDA_CHECK(cudaMemcpy(info.data(), d_info, nblk * sizeof(int), cudaMemcpyDeviceToHost));
        for (std::size_t b = 0; b < nblk; ++b)
            if (info[b] != 0)
                throw std::runtime_error("little_group_gpu: syevd did not converge for block " + std::to_string(b)
                                         + " (n = " + std::to_string(P.block_dim[b])
                                         + ", info = " + std::to_string(info[b]) + ")");

        // --- single download of all eigenvalues ----------------------------
        ED_CUDA_CHECK(cudaMemcpy(eigs.data(), d_eigs, total_eigs * sizeof(double), cudaMemcpyDeviceToHost));
    } catch (...) {
        cleanup();
        throw;
    }
    cleanup();
    return eigs;
}

}  // namespace ed::solvers
