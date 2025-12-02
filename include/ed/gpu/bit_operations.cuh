#ifndef GPU_BIT_OPERATIONS_CUH
#define GPU_BIT_OPERATIONS_CUH

#ifdef WITH_CUDA
#include <cuda_runtime.h>
#include <cuComplex.h>
#include "kernel_config.h"

// Device functions for bit manipulation
namespace GPUBitOps {

/**
 * GPU-optimized population count (count number of 1 bits)
 */
__device__ __forceinline__ int popcount_gpu(uint64_t x) {
#ifdef __CUDA_ARCH__
    return __popcll(x);
#else
    return __builtin_popcountll(x);
#endif
}

/**
 * Check if bit is set at position
 */
__device__ __forceinline__ bool is_bit_set(uint64_t state, int pos) {
    return (state >> pos) & 1ULL;
}

/**
 * Set bit at position
 */
__device__ __forceinline__ uint64_t set_bit(uint64_t state, int pos) {
    return state | (1ULL << pos);
}

/**
 * Clear bit at position
 */
__device__ __forceinline__ uint64_t clear_bit(uint64_t state, int pos) {
    return state & ~(1ULL << pos);
}

/**
 * Flip bit at position
 */
__device__ __forceinline__ uint64_t flip_bit(uint64_t state, int pos) {
    return state ^ (1ULL << pos);
}

/**
 * Apply spin raising operator S+ at site i
 * Flips bit from 0 to 1 (down to up)
 * Returns {new_state, phase_factor}
 */
__device__ __forceinline__ void apply_sp(uint64_t state, int site, uint64_t& new_state, 
                                         cuDoubleComplex& amplitude) {
    if (is_bit_set(state, site)) {
        // Spin already up, operator gives zero
        amplitude = make_cuDoubleComplex(0.0, 0.0);
        new_state = 0;
    } else {
        new_state = set_bit(state, site);
        amplitude = make_cuDoubleComplex(1.0, 0.0);
    }
}

/**
 * Apply spin lowering operator S- at site i
 * Flips bit from 1 to 0 (up to down)
 */
__device__ __forceinline__ void apply_sm(uint64_t state, int site, uint64_t& new_state,
                                         cuDoubleComplex& amplitude) {
    if (!is_bit_set(state, site)) {
        // Spin already down, operator gives zero
        amplitude = make_cuDoubleComplex(0.0, 0.0);
        new_state = 0;
    } else {
        new_state = clear_bit(state, site);
        amplitude = make_cuDoubleComplex(1.0, 0.0);
    }
}

/**
 * Apply Sz operator at site i
 * Returns +1/2 for up spin, -1/2 for down spin
 */
__device__ __forceinline__ void apply_sz(uint64_t state, int site, uint64_t& new_state,
                                         cuDoubleComplex& amplitude) {
    new_state = state;
    double val = is_bit_set(state, site) ? 0.5 : -0.5;
    amplitude = make_cuDoubleComplex(val, 0.0);
}

/**
 * Apply Sx operator at site i
 * Sx = (S+ + S-) / 2
 */
__device__ __forceinline__ void apply_sx(uint64_t state, int site, uint64_t& new_state,
                                         cuDoubleComplex& amplitude) {
    new_state = flip_bit(state, site);
    amplitude = make_cuDoubleComplex(0.5, 0.0);
}

/**
 * Apply Sy operator at site i
 * Sy = (S+ - S-) / 2i
 */
__device__ __forceinline__ void apply_sy(uint64_t state, int site, uint64_t& new_state,
                                         cuDoubleComplex& amplitude) {
    new_state = flip_bit(state, site);
    double sign = is_bit_set(state, site) ? -1.0 : 1.0;
    amplitude = make_cuDoubleComplex(0.0, sign * 0.5);
}

/**
 * Apply permutation to a basis state
 */
__device__ __forceinline__ uint64_t apply_permutation(uint64_t state, const int* perm, int n_sites) {
    uint64_t result = 0;
    for (int i = 0; i < n_sites; ++i) {
        if (is_bit_set(state, i)) {
            result = set_bit(result, perm[i]);
        }
    }
    return result;
}

/**
 * Check if state is in fixed Sz sector
 */
__device__ __forceinline__ bool is_valid_sz_state(uint64_t state, int n_up) {
    return popcount_gpu(state) == n_up;
}

/**
 * Generate next state in lexicographic order with fixed population count (Gosper's hack)
 */
__device__ __forceinline__ uint64_t next_combination(uint64_t state) {
    uint64_t c = state & -state;  // rightmost bit
    uint64_t r = state + c;        // add 1 to rightmost bit
    return (((r ^ state) >> 2) / c) | r;
}

/**
 * Complex number operations for GPU
 */
__device__ __forceinline__ cuDoubleComplex complex_add(cuDoubleComplex a, cuDoubleComplex b) {
    return make_cuDoubleComplex(cuCreal(a) + cuCreal(b), cuCimag(a) + cuCimag(b));
}

__device__ __forceinline__ cuDoubleComplex complex_mul(cuDoubleComplex a, cuDoubleComplex b) {
    return cuCmul(a, b);
}

__device__ __forceinline__ cuDoubleComplex complex_scale(cuDoubleComplex a, double scale) {
    return make_cuDoubleComplex(cuCreal(a) * scale, cuCimag(a) * scale);
}

__device__ __forceinline__ double complex_abs_squared(cuDoubleComplex a) {
    return cuCreal(a) * cuCreal(a) + cuCimag(a) * cuCimag(a);
}

} // namespace GPUBitOps

#endif // WITH_CUDA

#endif // GPU_BIT_OPERATIONS_CUH
