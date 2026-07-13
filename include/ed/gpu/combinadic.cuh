// =============================================================================
// ed/gpu/combinadic.cuh -- combinadic rank/unrank for fixed-Sz GPU lookups.
//
// Phase E.1 of the "Kill the GPU State-Lookup Hash" plan (May 2026).
//
// The symmetry path (``streaming_symmetry_gpu_mirror.cu``) needs the
// same constant-cache combinadic rank that the Sz-only matvec kernels
// (``gpu_kernels.cu``, Phase A.1) use. To stay within the 64 KiB
// per-TU constant-memory budget (a single 65x65 Pascal table is
// already ~33 KiB), we share ONE ``d_pascal_shared`` ``__constant__``
// table across all TUs that need combinadic lookups:
//
//   * Defined in ``gpu_kernels.cu`` (the same Pascal upload that the
//     Sz-only path already populates via ``ensure_pascal_uploaded``).
//   * Declared ``extern __device__ __constant__`` in this header so
//     downstream TUs (the symmetry GPU mirror, future helpers) read
//     from the same constant memory.
//
// Device-linking (``CUDA_SEPARABLE_COMPILATION ON``) resolves the
// extern reference across libraries -- all our CUDA TUs end up in
// one device-link object per binary, so the single 33 KiB table is
// shared.
//
// Inline implementations of rank/unrank live here so the symmetry
// kernel can inline them at the call site without the cross-TU
// device function call overhead.
// =============================================================================
#pragma once

#ifdef WITH_CUDA

#include <cuda_runtime.h>
#include <cstdint>

namespace ed::gpu::combinadic {

// Shared Pascal table. Definition lives in ``gpu_kernels.cu`` next to
// ``ensure_pascal_uploaded`` (single source of truth for the upload).
// Downstream TUs reference this symbol via device linking.
extern __device__ __constant__
unsigned long long d_pascal_shared[65][65];

// Host-callable uploader. Idempotent. Implemented in gpu_kernels.cu
// alongside ``ensure_pascal_uploaded`` (which uploads the same data
// into the legacy ``d_pascal`` symbol used by the older kernels).
void upload_pascal_shared();

__device__ __forceinline__
unsigned long long binomial(int n, int k) {
    if (k < 0 || k > n || n < 0 || n > 64) return 0ULL;
    return d_pascal_shared[n][k];
}

// Combinadic RANK in colex order: inverse of ``unrank``.
//
// For ``state`` with popcount exactly ``k``, returns the index in
// ``[0, C(n_bits, k))`` such that ``unrank(rank(state)) == state``.
// Scans bits low-to-high; the (seen)-th set bit at position ``bit``
// contributes ``C(bit, seen)`` to the rank.
//
// 64-bit rank: C(n_bits, k) exceeds INT32_MAX from N=34 half filling
// (C(36,18) ~ 9.1e9), so the rank must be accumulated and returned in
// 64 bits even though per-sector INDEX values stay int32.
//
// Returns 0 (which IS a valid rank) when ``popcount(state) != k``;
// callers must guard with their own popcount check if the input
// state might have escaped the fixed-Sz sector.
__device__ __forceinline__
std::int64_t rank_state(std::uint64_t state, int n_bits, int k) {
    std::int64_t rank = 0;
    int seen = 0;
    #pragma unroll
    for (int bit = 0; bit < 64; ++bit) {
        if (bit >= n_bits) break;
        if (seen >= k) break;
        if ((state >> bit) & 1ULL) {
            ++seen;
            rank += static_cast<std::int64_t>(binomial(bit, seen));
        }
    }
    return rank;
}

// Combinadic UNRANK: same colex convention as ``rank_state``.
__device__ __forceinline__
std::uint64_t unrank_to_state(std::uint64_t rank, int n_bits, int k) {
    std::uint64_t state = 0ULL;
    for (int i = k - 1; i >= 0; --i) {
        int p = i;
        while (p + 1 < n_bits && binomial(p + 1, i + 1) <= rank) ++p;
        state |= (1ULL << p);
        rank -= binomial(p, i + 1);
    }
    return state;
}

namespace detail {
// Backward-compat alias: ``DeviceSymmetryBasisPolicy::basis_view()``
// (and earlier drafts of this plan) used ``detail::upload_pascal()``.
// Forward to the canonical name now that the symbol is shared.
inline void upload_pascal() { upload_pascal_shared(); }
}  // namespace detail

}  // namespace ed::gpu::combinadic

#endif  // WITH_CUDA
