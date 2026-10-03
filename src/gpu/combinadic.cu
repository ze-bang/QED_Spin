// =============================================================================
// src/gpu/combinadic.cu
//
// Definition of the shared constant-memory Pascal triangle declared in
// ``include/ed/gpu/combinadic.cuh`` and its host uploader.
//
// d_pascal_shared[n][k] = C(n, k) for 0 <= n, k <= 64 (~33.8 KiB, uint64 --
// C(64, 32) = 1.83e18 fits). It is defined in exactly one TU; every other
// CUDA TU reads it through the ``extern __device__ __constant__`` declaration
// (device linking, CUDA_SEPARABLE_COMPILATION ON), so the device-link object
// carries a single 33 KiB table instead of one per TU.
// =============================================================================

#ifdef WITH_CUDA

#include <ed/gpu/combinadic.cuh>

#include <cuda_runtime.h>

#include <mutex>
#include <stdexcept>
#include <string>

namespace ed::gpu::combinadic {

__device__ __constant__ unsigned long long d_pascal_shared[65][65];

// Once per process, thread-safe; a failed upload throws and the next call tries again.
void upload_pascal_shared() {
    static std::once_flag once;
    std::call_once(once, [] {
        unsigned long long h_pascal[65][65] = {};
        for (int n = 0; n <= 64; ++n) {
            h_pascal[n][0] = 1ULL;
            for (int k = 1; k <= n; ++k) {
                unsigned long long left = h_pascal[n - 1][k - 1];
                unsigned long long right = (k < n) ? h_pascal[n - 1][k] : 0ULL;
                h_pascal[n][k] = left + right;
            }
        }
        const cudaError_t err = cudaMemcpyToSymbol(d_pascal_shared, h_pascal, sizeof(h_pascal));
        if (err != cudaSuccess) {
            cudaGetLastError();
            throw std::runtime_error(std::string("combinadic: uploading the Pascal table failed: ")
                                     + cudaGetErrorString(err));
        }
    });
}

}  // namespace ed::gpu::combinadic

#endif  // WITH_CUDA
