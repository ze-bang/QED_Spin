#ifdef WITH_CUDA

#include <ed/gpu/gpu_operator.cuh>
#include <ed/gpu/bit_operations.cuh>

using namespace GPUBitOps;

// ============================================================================
// CUDA Kernels Implementation
// 
// OPERATOR ENCODING (used throughout all GPU kernels):
//   0 = S+ (raising operator, flips spin up)
//   1 = S- (lowering operator, flips spin down)
//   2 = Sz (diagonal, measures spin)
// ============================================================================

// Helper function for atomic add on double precision
// Uses native atomicAdd for compute capability >= 6.0, falls back to CAS for older GPUs
__device__ __forceinline__ double atomicAddDouble(double* address, double val) {
#if __CUDA_ARCH__ >= 600
    // For compute capability 6.0+, use native atomicAdd for double
    return atomicAdd(address, val);
#else
    // For older GPUs, use compare-and-swap implementation
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                       __double_as_longlong(val + __longlong_as_double(assumed)));
    } while (assumed != old);
    
    return __longlong_as_double(old);
#endif
}

namespace GPUKernels {

/**
 * OPTIMIZED: Matrix-vector product using Structure-of-Arrays
 * 
 * Key optimizations:
 * 1. Read-only cache for random access to input vector (via __ldg)
 * 2. Shared memory for transform data (reduces global memory traffic)
 * 3. Direct evaluation without function pointers
 * 4. Atomic writes to handle scatter pattern correctly
 * 
 * FIXED: Correctly implements y[new_state] += factor * x[state] 
 * Transform encodes: state -> new_state, so H[new_state, state] = factor
 * 
 * Uses grid-stride loop to handle arrays larger than max grid size.
 */
__global__ void matVecKernelOptimized(cudaTextureObject_t tex_x_unused, cuDoubleComplex* y,
                                      int N, int n_sites, float spin_l,
                                      const GPUTransformData* transforms, int num_transforms,
                                      const cuDoubleComplex* x) {
    // Use shared memory for transforms if small enough
    extern __shared__ GPUTransformData s_transforms[];
    
    // ALL threads in block participate in loading transforms into shared memory
    // This must happen BEFORE any early returns to avoid __syncthreads deadlock
    int num_loads = (num_transforms + blockDim.x - 1) / blockDim.x;
    for (int i = 0; i < num_loads; ++i) {
        int tidx = i * blockDim.x + threadIdx.x;
        if (tidx < num_transforms) {
            s_transforms[tidx] = transforms[tidx];
        }
    }
    __syncthreads();
    
    // Grid-stride loop to handle arrays larger than max grid size
    int grid_stride = blockDim.x * gridDim.x;
    
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < N; idx += grid_stride) {
        uint64_t state = static_cast<uint64_t>(idx);
        
        // Read input value once (coalesced read)
        cuDoubleComplex x_val = __ldg(&x[idx]);
        
        // Process all transforms for this basis state
        const GPUTransformData* t_data = (num_transforms <= 4096) ? s_transforms : transforms;
        
        #pragma unroll 4
        for (int t = 0; t < num_transforms; ++t) {
            const GPUTransformData& tdata = t_data[t];
            
            if (tdata.is_two_body) {
                // Two-body operator
                uint64_t bit1 = (state >> tdata.site_index) & 1;
                
                uint64_t new_state = state;
                cuDoubleComplex factor = tdata.coefficient;
                bool valid = true;
                
                // Apply first operator
                if (tdata.op_type == 2) {
                    // Sz operator
                    double sign = spin_l * ((bit1 == 0) ? 1.0 : -1.0);
                    factor = complex_scale(factor, sign);
                } else {
                    // S+ or S- operator
                    if (bit1 != tdata.op_type) {
                        new_state ^= (1ULL << tdata.site_index);
                    } else {
                        valid = false;
                    }
                }
                
                if (valid) {
                    // Apply second operator (update bit2 if first op flipped site 2)
                    uint64_t bit2_new = (new_state >> tdata.site_index_2) & 1;
                    
                    if (tdata.op_type_2 == 2) {
                        // Sz operator
                        double sign = spin_l * ((bit2_new == 0) ? 1.0 : -1.0);
                        factor = complex_scale(factor, sign);
                    } else {
                        // S+ or S- operator
                        if (bit2_new != tdata.op_type_2) {
                            new_state ^= (1ULL << tdata.site_index_2);
                        } else {
                            valid = false;
                        }
                    }
                }
                
                // CORRECT: Write to y[new_state], not y[state]
                if (valid && new_state < N) {
                    cuDoubleComplex contrib = cuCmul(factor, x_val);
                    atomicAddDouble(&y[new_state].x, cuCreal(contrib));
                    atomicAddDouble(&y[new_state].y, cuCimag(contrib));
                }
            } else {
                // One-body operator
                uint64_t bit = (state >> tdata.site_index) & 1;
                uint64_t new_state = state;
                cuDoubleComplex factor = tdata.coefficient;
                bool valid = true;
                
                if (tdata.op_type == 2) {
                    // Sz operator: diagonal
                    double sign = spin_l * ((bit == 0) ? 1.0 : -1.0);
                    factor = complex_scale(factor, sign);
                } else {
                    // S+ or S- operator: off-diagonal
                    if (bit != tdata.op_type) {
                        new_state ^= (1ULL << tdata.site_index);
                    } else {
                        valid = false;
                    }
                }
                
                // CORRECT: Write to y[new_state], not y[state]
                if (valid && new_state < N) {
                    cuDoubleComplex contrib = cuCmul(factor, x_val);
                    atomicAddDouble(&y[new_state].x, cuCreal(contrib));
                    atomicAddDouble(&y[new_state].y, cuCimag(contrib));
                }
            }
        }
    }  // end grid-stride loop
}

/**
 * Generate basis states for fixed Sz sector using parallel Gosper's hack
 */
__global__ void generateFixedSzBasisKernel(uint64_t* basis_states, int n_bits, int n_up,
                                          uint64_t start_state, int num_states) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_states) return;
    
    // Each thread generates one basis state
    uint64_t state = start_state;
    
    // Skip ahead to this thread's state
    for (int i = 0; i < idx; ++i) {
        state = next_combination(state);
        if (state >= (1ULL << n_bits)) return;
    }
    
    basis_states[idx] = state;
}

/**
 * OPTIMIZED: Device function to lookup state using binary search
 * Assumes basis_states array is sorted (which it naturally is for fixed Sz)
 * Returns index in basis, or -1 if not found
 * 
 * Much better than hash table with linear probing:
 * - No warp divergence (all threads follow same path)
 * - O(log N) complexity with minimal divergence
 * - Cache-friendly access pattern
 */
__device__ int lookupState(uint64_t state, const void* basis_states_ptr, int num_states) {
    const uint64_t* basis_states = static_cast<const uint64_t*>(basis_states_ptr);
    
    int left = 0;
    int right = num_states - 1;
    
    // Binary search (warp-coherent)
    while (left <= right) {
        int mid = (left + right) / 2;
        uint64_t mid_state = basis_states[mid];
        
        if (mid_state == state) {
            return mid;
        } else if (mid_state < state) {
            left = mid + 1;
        } else {
            right = mid - 1;
        }
    }
    
    return -1;  // Not found
}

/**
 * ULTRA-OPTIMIZED: Fixed-Sz matrix-vector product using Structure-of-Arrays
 * 
 * Key optimizations:
 * 1. Binary search for state lookup (warp-coherent)
 * 2. Shared memory for transform data
 * 3. Direct evaluation without function pointers
 * 4. __ldg for cached reads
 */
/**
 * GPU-NATIVE: Transform-parallel Fixed-Sz matrix-vector product
 * 
 * 2D parallelism: N × T threads for maximum GPU utilization
 * Uses binary search to map new_state → new_idx in basis_states[]
 * Atomic accumulation for output (hardware-optimized on modern GPUs)
 */
__global__ void matVecFixedSzTransformParallel(const cuDoubleComplex* x, cuDoubleComplex* y,
                                               const uint64_t* basis_states,
                                               const GPUTransformData* transforms,
                                               int num_transforms, int N, int n_sites, float spin_l) {
    // 2D thread mapping
    int state_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int transform_idx = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (state_idx >= N || transform_idx >= num_transforms) return;
    
    uint64_t state = basis_states[state_idx];
    const GPUTransformData& tdata = transforms[transform_idx];
    
    cuDoubleComplex factor = tdata.coefficient;
    uint64_t new_state = state;
    bool valid = true;
    
    if (tdata.is_two_body) {
        // Two-body operator
        uint64_t bit1 = (state >> tdata.site_index) & 1;
        
        // Apply first operator
        if (tdata.op_type == 2) {
            double sign = spin_l * ((bit1 == 0) ? 1.0 : -1.0);
            factor = complex_scale(factor, sign);
        } else {
            if (bit1 != tdata.op_type) {
                new_state ^= (1ULL << tdata.site_index);
            } else {
                valid = false;
            }
        }
        
        if (valid) {
            uint64_t bit2_new = (new_state >> tdata.site_index_2) & 1;
            
            if (tdata.op_type_2 == 2) {
                double sign = spin_l * ((bit2_new == 0) ? 1.0 : -1.0);
                factor = complex_scale(factor, sign);
            } else {
                if (bit2_new != tdata.op_type_2) {
                    new_state ^= (1ULL << tdata.site_index_2);
                } else {
                    valid = false;
                }
            }
        }
    } else {
        // One-body operator
        uint64_t bit = (state >> tdata.site_index) & 1;
        
        if (tdata.op_type == 2) {
            double sign = spin_l * ((bit == 0) ? 1.0 : -1.0);
            factor = complex_scale(factor, sign);
        } else {
            if (bit != tdata.op_type) {
                new_state ^= (1ULL << tdata.site_index);
            } else {
                valid = false;
            }
        }
    }
    
    // Binary search and atomic accumulation
    // CORRECT: Read from input state, write to output (new_state)
    // Transform encodes: state -> new_state, so H[new_state, state] = factor
    if (valid) {
        int new_idx = lookupState(new_state, basis_states, N);
        if (new_idx >= 0) {
            cuDoubleComplex x_val = __ldg(&x[state_idx]);
            cuDoubleComplex contrib = cuCmul(factor, x_val);
            
            // Atomic add for complex numbers - write to y[new_idx], not y[state_idx]
            atomicAddDouble(&y[new_idx].x, cuCreal(contrib));
            atomicAddDouble(&y[new_idx].y, cuCimag(contrib));
        }
    }
}

__global__ void matVecFixedSzKernelOptimized(const cuDoubleComplex* x, cuDoubleComplex* y,
                                             const uint64_t* basis_states,
                                             int N, int n_sites, float spin_l,
                                             const GPUTransformData* transforms, int num_transforms) {
    // Use shared memory for transforms if small enough
    extern __shared__ GPUTransformData s_transforms[];
    
    // ALL threads in block participate in loading transforms into shared memory
    // This must happen BEFORE any early returns to avoid __syncthreads deadlock
    int num_loads = (num_transforms + blockDim.x - 1) / blockDim.x;
    for (int i = 0; i < num_loads; ++i) {
        int tidx = i * blockDim.x + threadIdx.x;
        if (tidx < num_transforms) {
            s_transforms[tidx] = transforms[tidx];
        }
    }
    __syncthreads();
    
    // Grid-stride loop to handle arrays larger than max grid size
    int grid_stride = blockDim.x * gridDim.x;
    
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < N; idx += grid_stride) {
        // Get basis state for this index
        uint64_t state = basis_states[idx];
    
    // Read input value once (coalesced read)
    cuDoubleComplex x_val = __ldg(&x[idx]);
    
    // Process all transforms for this basis state
    const GPUTransformData* t_data = (num_transforms <= 4096) ? s_transforms : transforms;
    
    #pragma unroll 4
    for (int t = 0; t < num_transforms; ++t) {
        const GPUTransformData& tdata = t_data[t];
        
        if (tdata.is_two_body) {
            // Two-body operator
            uint64_t bit1 = (state >> tdata.site_index) & 1;
            
            uint64_t new_state = state;
            cuDoubleComplex factor = tdata.coefficient;
            bool valid = true;
            
            // Apply first operator
            if (tdata.op_type == 2) {
                double sign = spin_l * ((bit1 == 0) ? 1.0 : -1.0);
                factor = complex_scale(factor, sign);
            } else {
                if (bit1 != tdata.op_type) {
                    new_state ^= (1ULL << tdata.site_index);
                } else {
                    valid = false;
                }
            }
            
            if (valid) {
                uint64_t bit2_new = (new_state >> tdata.site_index_2) & 1;
                
                if (tdata.op_type_2 == 2) {
                    double sign = spin_l * ((bit2_new == 0) ? 1.0 : -1.0);
                    factor = complex_scale(factor, sign);
                } else {
                    if (bit2_new != tdata.op_type_2) {
                        new_state ^= (1ULL << tdata.site_index_2);
                    } else {
                        valid = false;
                    }
                }
            }
            
            // CORRECT: Write to y[new_idx], read from x[idx]
            if (valid) {
                int new_idx = lookupState(new_state, basis_states, N);
                if (new_idx >= 0) {
                    cuDoubleComplex contrib = cuCmul(factor, x_val);
                    atomicAddDouble(&y[new_idx].x, cuCreal(contrib));
                    atomicAddDouble(&y[new_idx].y, cuCimag(contrib));
                }
            }
        } else {
            // One-body operator
            uint64_t bit = (state >> tdata.site_index) & 1;
            uint64_t new_state = state;
            cuDoubleComplex factor = tdata.coefficient;
            bool valid = true;
            
            if (tdata.op_type == 2) {
                double sign = spin_l * ((bit == 0) ? 1.0 : -1.0);
                factor = complex_scale(factor, sign);
            } else {
                if (bit != tdata.op_type) {
                    new_state ^= (1ULL << tdata.site_index);
                } else {
                    valid = false;
                }
            }
            
            // CORRECT: Write to y[new_idx], read from x[idx]
            if (valid) {
                int new_idx = lookupState(new_state, basis_states, N);
                if (new_idx >= 0) {
                    cuDoubleComplex contrib = cuCmul(factor, x_val);
                    atomicAddDouble(&y[new_idx].x, cuCreal(contrib));
                    atomicAddDouble(&y[new_idx].y, cuCimag(contrib));
                }
            }
        }
    }
    }  // end grid-stride loop
}

// ============================================================================
// BRANCH-FREE KERNELS (v2 optimization)
// Each kernel handles one operator type - eliminates warp divergence
// All threads in a warp execute the same instructions (no if/else on op_type)
// ============================================================================

/**
 * One-body diagonal kernel (Sz only)
 * All threads do the same operation: multiply by spin eigenvalue
 * Writes to y[state] = y[input] (diagonal, no state lookup needed)
 */
__global__ void matVecDiagonalOneBody(const cuDoubleComplex* x, cuDoubleComplex* y,
                                      const GPUDiagonalOneBody* transforms,
                                      int num_transforms, int N, float spin_l) {
    int state_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int transform_idx = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (state_idx >= N || transform_idx >= num_transforms) return;
    
    uint64_t state = static_cast<uint64_t>(state_idx);
    const GPUDiagonalOneBody& t = transforms[transform_idx];
    
    // Sz eigenvalue: +spin_l for |0⟩, -spin_l for |1⟩
    uint64_t bit = (state >> t.site_index) & 1;
    double sign = spin_l * ((bit == 0) ? 1.0 : -1.0);
    
    cuDoubleComplex contrib = complex_scale(t.coefficient, sign);
    contrib = cuCmul(contrib, __ldg(&x[state_idx]));
    
    // Diagonal: output index = input index
    atomicAddDouble(&y[state_idx].x, cuCreal(contrib));
    atomicAddDouble(&y[state_idx].y, cuCimag(contrib));
}

/**
 * One-body off-diagonal kernel (S+ or S-)
 * TRULY BRANCH-FREE: Uses predicated execution via zero-masking
 * All threads execute identical instructions - divergent threads contribute zero
 */
__global__ void matVecOffDiagonalOneBody(const cuDoubleComplex* x, cuDoubleComplex* y,
                                         const GPUOffDiagonalOneBody* transforms,
                                         int num_transforms, int N) {
    int state_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int transform_idx = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (state_idx >= N || transform_idx >= num_transforms) return;
    
    uint64_t state = static_cast<uint64_t>(state_idx);
    const GPUOffDiagonalOneBody& t = transforms[transform_idx];
    
    uint64_t bit = (state >> t.site_index) & 1;
    
    // BRANCH-FREE: Compute validity mask (1.0 if valid, 0.0 if not)
    // S+ acts on |1⟩ (bit=1, op_type=0), S- acts on |0⟩ (bit=0, op_type=1)
    double valid_mask = (bit != t.op_type) ? 1.0 : 0.0;
    
    // Always compute new_state (cheap bit flip)
    uint64_t new_state = state ^ (1ULL << t.site_index);
    
    // Bounds check folded into mask
    valid_mask *= (new_state < N) ? 1.0 : 0.0;
    
    // All threads read and compute - invalid ones just contribute zero
    cuDoubleComplex x_val = __ldg(&x[state_idx]);
    cuDoubleComplex contrib = cuCmul(t.coefficient, x_val);
    double contrib_real = cuCreal(contrib) * valid_mask;
    double contrib_imag = cuCimag(contrib) * valid_mask;
    
    // Clamp new_state to valid range to avoid out-of-bounds atomic
    // (contribution is zero anyway for invalid states)
    new_state = min(new_state, static_cast<uint64_t>(N - 1));
    
    // All threads do atomic - invalid ones add zero (no-op but uniform execution)
    atomicAddDouble(&y[new_state].x, contrib_real);
    atomicAddDouble(&y[new_state].y, contrib_imag);
}

/**
 * Two-body diagonal kernel (Sz_i Sz_j)
 * All threads compute product of two spin eigenvalues
 */
__global__ void matVecDiagonalTwoBody(const cuDoubleComplex* x, cuDoubleComplex* y,
                                      const GPUDiagonalTwoBody* transforms,
                                      int num_transforms, int N, float spin_l) {
    int state_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int transform_idx = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (state_idx >= N || transform_idx >= num_transforms) return;
    
    uint64_t state = static_cast<uint64_t>(state_idx);
    const GPUDiagonalTwoBody& t = transforms[transform_idx];
    
    uint64_t bit1 = (state >> t.site_index_1) & 1;
    uint64_t bit2 = (state >> t.site_index_2) & 1;
    
    double sign1 = (bit1 == 0) ? 1.0 : -1.0;
    double sign2 = (bit2 == 0) ? 1.0 : -1.0;
    double factor = spin_l * spin_l * sign1 * sign2;
    
    cuDoubleComplex contrib = complex_scale(t.coefficient, factor);
    contrib = cuCmul(contrib, __ldg(&x[state_idx]));
    
    // Diagonal: output index = input index
    atomicAddDouble(&y[state_idx].x, cuCreal(contrib));
    atomicAddDouble(&y[state_idx].y, cuCimag(contrib));
}

/**
 * Two-body mixed kernel (Sz * S+/S-)
 * TRULY BRANCH-FREE: Uses predicated execution via zero-masking
 */
__global__ void matVecMixedTwoBody(const cuDoubleComplex* x, cuDoubleComplex* y,
                                   const GPUMixedTwoBody* transforms,
                                   int num_transforms, int N, float spin_l) {
    int state_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int transform_idx = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (state_idx >= N || transform_idx >= num_transforms) return;
    
    uint64_t state = static_cast<uint64_t>(state_idx);
    const GPUMixedTwoBody& t = transforms[transform_idx];
    
    uint64_t flip_bit = (state >> t.flip_site) & 1;
    uint64_t sz_bit = (state >> t.sz_site) & 1;
    
    // BRANCH-FREE: Compute all values, mask invalid contributions to zero
    double valid_mask = (flip_bit != t.flip_op_type) ? 1.0 : 0.0;
    double sz_sign = spin_l * ((sz_bit == 0) ? 1.0 : -1.0);
    
    uint64_t new_state = state ^ (1ULL << t.flip_site);
    valid_mask *= (new_state < N) ? 1.0 : 0.0;
    
    // All threads compute - invalid ones produce zero
    cuDoubleComplex x_val = __ldg(&x[state_idx]);
    cuDoubleComplex contrib = complex_scale(t.coefficient, sz_sign * valid_mask);
    contrib = cuCmul(contrib, x_val);
    
    new_state = min(new_state, static_cast<uint64_t>(N - 1));
    
    atomicAddDouble(&y[new_state].x, cuCreal(contrib));
    atomicAddDouble(&y[new_state].y, cuCimag(contrib));
}

/**
 * Two-body off-diagonal kernel (S+/S- * S+/S-)
 * TRULY BRANCH-FREE: Uses predicated execution via zero-masking
 */
__global__ void matVecOffDiagonalTwoBody(const cuDoubleComplex* x, cuDoubleComplex* y,
                                         const GPUOffDiagonalTwoBody* transforms,
                                         int num_transforms, int N) {
    int state_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int transform_idx = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (state_idx >= N || transform_idx >= num_transforms) return;
    
    uint64_t state = static_cast<uint64_t>(state_idx);
    const GPUOffDiagonalTwoBody& t = transforms[transform_idx];
    
    uint64_t bit1 = (state >> t.site_index_1) & 1;
    uint64_t bit2 = (state >> t.site_index_2) & 1;
    
    // BRANCH-FREE: Both conditions combined into single mask
    double valid_mask = ((bit1 != t.op_type_1) && (bit2 != t.op_type_2)) ? 1.0 : 0.0;
    
    uint64_t new_state = state ^ (1ULL << t.site_index_1) ^ (1ULL << t.site_index_2);
    valid_mask *= (new_state < N) ? 1.0 : 0.0;
    
    cuDoubleComplex x_val = __ldg(&x[state_idx]);
    double contrib_real = cuCreal(t.coefficient) * cuCreal(x_val) - cuCimag(t.coefficient) * cuCimag(x_val);
    double contrib_imag = cuCreal(t.coefficient) * cuCimag(x_val) + cuCimag(t.coefficient) * cuCreal(x_val);
    contrib_real *= valid_mask;
    contrib_imag *= valid_mask;
    
    new_state = min(new_state, static_cast<uint64_t>(N - 1));
    
    atomicAddDouble(&y[new_state].x, contrib_real);
    atomicAddDouble(&y[new_state].y, contrib_imag);
}

// ============================================================================
// Fixed-Sz Branch-Free Kernels
// ============================================================================

/**
 * Fixed-Sz one-body diagonal kernel
 * Diagonal terms stay in same sector - output index = input index
 */
__global__ void matVecFixedSzDiagonalOneBody(const cuDoubleComplex* x, cuDoubleComplex* y,
                                             const uint64_t* basis_states,
                                             const GPUDiagonalOneBody* transforms,
                                             int num_transforms, int N, float spin_l) {
    int state_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int transform_idx = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (state_idx >= N || transform_idx >= num_transforms) return;
    
    uint64_t state = basis_states[state_idx];
    const GPUDiagonalOneBody& t = transforms[transform_idx];
    
    uint64_t bit = (state >> t.site_index) & 1;
    double sign = spin_l * ((bit == 0) ? 1.0 : -1.0);
    
    cuDoubleComplex contrib = complex_scale(t.coefficient, sign);
    contrib = cuCmul(contrib, __ldg(&x[state_idx]));
    
    // Diagonal in fixed-Sz: index unchanged
    atomicAddDouble(&y[state_idx].x, cuCreal(contrib));
    atomicAddDouble(&y[state_idx].y, cuCimag(contrib));
}

/**
 * Fixed-Sz two-body diagonal kernel
 */
__global__ void matVecFixedSzDiagonalTwoBody(const cuDoubleComplex* x, cuDoubleComplex* y,
                                             const uint64_t* basis_states,
                                             const GPUDiagonalTwoBody* transforms,
                                             int num_transforms, int N, float spin_l) {
    int state_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int transform_idx = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (state_idx >= N || transform_idx >= num_transforms) return;
    
    uint64_t state = basis_states[state_idx];
    const GPUDiagonalTwoBody& t = transforms[transform_idx];
    
    uint64_t bit1 = (state >> t.site_index_1) & 1;
    uint64_t bit2 = (state >> t.site_index_2) & 1;
    
    double sign1 = (bit1 == 0) ? 1.0 : -1.0;
    double sign2 = (bit2 == 0) ? 1.0 : -1.0;
    double factor = spin_l * spin_l * sign1 * sign2;
    
    cuDoubleComplex contrib = complex_scale(t.coefficient, factor);
    contrib = cuCmul(contrib, __ldg(&x[state_idx]));
    
    atomicAddDouble(&y[state_idx].x, cuCreal(contrib));
    atomicAddDouble(&y[state_idx].y, cuCimag(contrib));
}

/**
 * Fixed-Sz two-body off-diagonal kernel (S+S- or S-S+ conserves Sz)
 * TRULY BRANCH-FREE: Uses predicated execution via zero-masking
 * Binary search for output index (unavoidable O(log N) divergence)
 */
__global__ void matVecFixedSzOffDiagonalTwoBody(const cuDoubleComplex* x, cuDoubleComplex* y,
                                                const uint64_t* basis_states,
                                                const GPUOffDiagonalTwoBody* transforms,
                                                int num_transforms, int N) {
    int state_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int transform_idx = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (state_idx >= N || transform_idx >= num_transforms) return;
    
    uint64_t state = basis_states[state_idx];
    const GPUOffDiagonalTwoBody& t = transforms[transform_idx];
    
    uint64_t bit1 = (state >> t.site_index_1) & 1;
    uint64_t bit2 = (state >> t.site_index_2) & 1;
    
    // BRANCH-FREE: Compute validity mask
    double valid_mask = ((bit1 != t.op_type_1) && (bit2 != t.op_type_2)) ? 1.0 : 0.0;
    
    uint64_t new_state = state ^ (1ULL << t.site_index_1) ^ (1ULL << t.site_index_2);
    
    // Binary search (unavoidable, but warp-coherent since all threads follow same path)
    int new_idx = lookupState(new_state, basis_states, N);
    valid_mask *= (new_idx >= 0) ? 1.0 : 0.0;
    
    cuDoubleComplex x_val = __ldg(&x[state_idx]);
    double contrib_real = cuCreal(t.coefficient) * cuCreal(x_val) - cuCimag(t.coefficient) * cuCimag(x_val);
    double contrib_imag = cuCreal(t.coefficient) * cuCimag(x_val) + cuCimag(t.coefficient) * cuCreal(x_val);
    contrib_real *= valid_mask;
    contrib_imag *= valid_mask;
    
    // Clamp to valid index range (contribution is zero anyway)
    new_idx = max(new_idx, 0);
    
    atomicAddDouble(&y[new_idx].x, contrib_real);
    atomicAddDouble(&y[new_idx].y, contrib_imag);
}

// ============================================================================
// WARP-REDUCTION (GATHER) KERNELS - Strategy 3
// 
// Each warp computes ONE complete output element by gathering from all inputs.
// This ELIMINATES atomic contention by design:
// - Multiple warps can READ same x[i] (reads don't conflict)
// - Each warp WRITES to unique y[j] (no write conflicts within kernel)
// - Warp shuffle reduction: O(log 32) = 5 steps, zero atomics within warp
//
// Trade-off: Scattered reads instead of scattered writes
//           (Reads are cheaper than atomic RMW operations!)
// ============================================================================

/**
 * WARP-REDUCTION: Fused kernel for all transform types
 * 
 * Each warp computes the COMPLETE output for one state.
 * Zero atomics within the kernel - single direct write per output.
 * 
 * Grid: ((N + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK) blocks
 * Block: WARPS_PER_BLOCK * 32 threads
 */
__global__ void matVecWarpReductionFused(
    const cuDoubleComplex* __restrict__ x,
    cuDoubleComplex* __restrict__ y,
    // Diagonal transforms
    const GPUDiagonalOneBody* __restrict__ diag1, int num_diag1,
    const GPUDiagonalTwoBody* __restrict__ diag2, int num_diag2,
    // Off-diagonal transforms  
    const GPUOffDiagonalOneBody* __restrict__ offdiag1, int num_offdiag1,
    const GPUMixedTwoBody* __restrict__ mixed2, int num_mixed2,
    const GPUOffDiagonalTwoBody* __restrict__ offdiag2, int num_offdiag2,
    int N, float spin_l
) {
    // Warp and lane identification
    const int lane_id = threadIdx.x & 31;  // threadIdx.x % 32
    const int warp_id_in_block = threadIdx.x >> 5;  // threadIdx.x / 32
    const int warps_per_block = blockDim.x >> 5;
    const int global_warp_id = blockIdx.x * warps_per_block + warp_id_in_block;
    
    // Each warp handles one output index
    const int out_idx = global_warp_id;
    if (out_idx >= N) return;
    
    const uint64_t out_state = static_cast<uint64_t>(out_idx);
    
    // Read x[out_idx] once for diagonal terms (input = output)
    cuDoubleComplex x_self = __ldg(&x[out_idx]);
    
    // Accumulator for this output (each lane has partial sum)
    double sum_real = 0.0;
    double sum_imag = 0.0;
    
    // ===== DIAGONAL ONE-BODY (Sz) =====
    // Input = output, just multiply by eigenvalue
    for (int t = lane_id; t < num_diag1; t += 32) {
        const GPUDiagonalOneBody& tr = diag1[t];
        uint64_t bit = (out_state >> tr.site_index) & 1;
        double eigenvalue = spin_l * ((bit == 0) ? 1.0 : -1.0);
        
        // contrib = eigenvalue * coefficient * x_self
        double c_real = cuCreal(tr.coefficient);
        double c_imag = cuCimag(tr.coefficient);
        double x_real = cuCreal(x_self);
        double x_imag = cuCimag(x_self);
        
        sum_real += eigenvalue * (c_real * x_real - c_imag * x_imag);
        sum_imag += eigenvalue * (c_real * x_imag + c_imag * x_real);
    }
    
    // ===== DIAGONAL TWO-BODY (Sz Sz) =====
    for (int t = lane_id; t < num_diag2; t += 32) {
        const GPUDiagonalTwoBody& tr = diag2[t];
        uint64_t bit1 = (out_state >> tr.site_index_1) & 1;
        uint64_t bit2 = (out_state >> tr.site_index_2) & 1;
        double sign1 = (bit1 == 0) ? 1.0 : -1.0;
        double sign2 = (bit2 == 0) ? 1.0 : -1.0;
        double eigenvalue = spin_l * spin_l * sign1 * sign2;
        
        double c_real = cuCreal(tr.coefficient);
        double c_imag = cuCimag(tr.coefficient);
        double x_real = cuCreal(x_self);
        double x_imag = cuCimag(x_self);
        
        sum_real += eigenvalue * (c_real * x_real - c_imag * x_imag);
        sum_imag += eigenvalue * (c_real * x_imag + c_imag * x_real);
    }
    
    // ===== OFF-DIAGONAL ONE-BODY (S+, S-) =====
    // GATHER: For output j, find input i = j XOR mask that maps to j
    for (int t = lane_id; t < num_offdiag1; t += 32) {
        const GPUOffDiagonalOneBody& tr = offdiag1[t];
        
        // Compute input state that would produce this output
        uint64_t flip_mask = 1ULL << tr.site_index;
        uint64_t in_state = out_state ^ flip_mask;
        
        // Selection rule (inverted for gather direction):
        // S+ (op_type=0) flips 1→0, so input must have bit=1
        // S- (op_type=1) flips 0→1, so input must have bit=0
        // The INPUT bit must be (1 - op_type)
        uint64_t in_bit = (in_state >> tr.site_index) & 1;
        
        // Valid if: input bit matches requirement AND in bounds
        bool valid = (in_bit == (1u - tr.op_type)) && (in_state < static_cast<uint64_t>(N));
        
        if (valid) {
            cuDoubleComplex x_in = __ldg(&x[in_state]);
            double c_real = cuCreal(tr.coefficient);
            double c_imag = cuCimag(tr.coefficient);
            double x_real = cuCreal(x_in);
            double x_imag = cuCimag(x_in);
            
            sum_real += c_real * x_real - c_imag * x_imag;
            sum_imag += c_real * x_imag + c_imag * x_real;
        }
    }
    
    // ===== MIXED TWO-BODY (Sz * S+/S-) =====
    for (int t = lane_id; t < num_mixed2; t += 32) {
        const GPUMixedTwoBody& tr = mixed2[t];
        
        // Only the flip_site changes between input and output
        uint64_t flip_mask = 1ULL << tr.flip_site;
        uint64_t in_state = out_state ^ flip_mask;
        
        // Selection rule for the flip operator
        uint64_t in_flip_bit = (in_state >> tr.flip_site) & 1;
        bool valid = (in_flip_bit == (1u - tr.flip_op_type)) && (in_state < static_cast<uint64_t>(N));
        
        if (valid) {
            // Sz eigenvalue at sz_site (evaluated on OUTPUT state, after the flip)
            uint64_t out_sz_bit = (out_state >> tr.sz_site) & 1;
            double sz_eigenvalue = spin_l * ((out_sz_bit == 0) ? 1.0 : -1.0);
            
            cuDoubleComplex x_in = __ldg(&x[in_state]);
            double c_real = cuCreal(tr.coefficient);
            double c_imag = cuCimag(tr.coefficient);
            double x_real = cuCreal(x_in);
            double x_imag = cuCimag(x_in);
            
            sum_real += sz_eigenvalue * (c_real * x_real - c_imag * x_imag);
            sum_imag += sz_eigenvalue * (c_real * x_imag + c_imag * x_real);
        }
    }
    
    // ===== OFF-DIAGONAL TWO-BODY (S+ S-, S- S+) =====
    for (int t = lane_id; t < num_offdiag2; t += 32) {
        const GPUOffDiagonalTwoBody& tr = offdiag2[t];
        
        // Both sites flip
        uint64_t flip_mask = (1ULL << tr.site_index_1) | (1ULL << tr.site_index_2);
        uint64_t in_state = out_state ^ flip_mask;
        
        // Selection rules for both operators
        uint64_t in_bit1 = (in_state >> tr.site_index_1) & 1;
        uint64_t in_bit2 = (in_state >> tr.site_index_2) & 1;
        
        bool valid = (in_bit1 == (1u - tr.op_type_1)) && 
                     (in_bit2 == (1u - tr.op_type_2)) &&
                     (in_state < static_cast<uint64_t>(N));
        
        if (valid) {
            cuDoubleComplex x_in = __ldg(&x[in_state]);
            double c_real = cuCreal(tr.coefficient);
            double c_imag = cuCimag(tr.coefficient);
            double x_real = cuCreal(x_in);
            double x_imag = cuCimag(x_in);
            
            sum_real += c_real * x_real - c_imag * x_imag;
            sum_imag += c_real * x_imag + c_imag * x_real;
        }
    }
    
    // ===== WARP-LEVEL REDUCTION =====
    // Sum all 32 lanes' partial results using shuffle
    // This is O(log 32) = 5 steps with NO atomics
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum_real += __shfl_down_sync(0xffffffff, sum_real, offset);
        sum_imag += __shfl_down_sync(0xffffffff, sum_imag, offset);
    }
    
    // ===== SINGLE WRITE (NO ATOMIC!) =====
    // Only lane 0 writes the final accumulated result
    if (lane_id == 0) {
        y[out_idx] = make_cuDoubleComplex(sum_real, sum_imag);
    }
}

// ============================================================================
// SYMMETRIZED MATVEC KERNEL
// ============================================================================

/**
 * @brief Hash table lookup using open addressing with linear probing
 *
 * Returns -1 if key not found (state outside this sector).
 */
__device__ __forceinline__ int hashLookup(
    uint64_t key,
    const GPUHashEntry* table,
    int table_size,
    cuDoubleComplex* out_projection)
{
    // Fibonacci hashing for good distribution
    uint64_t hash = key * 11400714819323198485ULL;  // golden ratio * 2^64
    int idx = static_cast<int>(hash % static_cast<uint64_t>(table_size));
    
    // Linear probing (max table_size probes for safety)
    for (int probe = 0; probe < table_size; ++probe) {
        const GPUHashEntry& entry = table[idx];
        if (entry.key == key) {
            *out_projection = entry.projection;
            return entry.value;
        }
        if (entry.key == UINT64_MAX) {
            return -1;  // Empty slot — key doesn't exist
        }
        idx = (idx + 1) % table_size;
    }
    return -1;  // Table full (shouldn't happen with proper sizing)
}

/**
 * @brief Binary search to find which basis state j owns global orbit index idx
 *
 * orbit_offsets[j] <= idx < orbit_offsets[j+1]
 * Returns j.  orbit_offsets has sector_dim+1 entries.
 */
__device__ __forceinline__ int findBasisState(
    int global_orbit_idx,
    const int* orbit_offsets,
    int sector_dim)
{
    int lo = 0, hi = sector_dim;
    while (lo < hi) {
        int mid = (lo + hi) / 2;
        if (orbit_offsets[mid + 1] <= global_orbit_idx) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    return lo;
}

/**
 * @brief Symmetrized matvec kernel — scatter pattern with hash-table projection
 *
 * 2D grid: x-dim over total_orbit_elements, y-dim over transforms.
 *
 * For each (orbit_element_idx, transform_idx):
 *   1. Find j = basis state owning this orbit element
 *   2. Load s = orbit_elements[idx], α = orbit_coefficients[idx], c_j = x[j]
 *   3. Apply transform to s → s', h
 *   4. Hash lookup s' → (k, projection_factor)
 *   5. Atomically accumulate weighted * h * projection_factor into y[k]
 *
 * Memory: O(sector_dim) for output only; all orbit/hash data read-only.
 */
__global__ void matVecSymmetrized(
    const cuDoubleComplex* __restrict__ x,
    cuDoubleComplex* __restrict__ y,
    const uint64_t* orbit_elements,
    const cuDoubleComplex* orbit_coefficients,
    const int* orbit_offsets,
    const double* orbit_norms,
    int sector_dim,
    const GPUTransformData* transforms, int num_transforms,
    const GPUHashEntry* hash_table, int hash_table_size,
    int n_sites, float spin_l,
    int total_orbit_elements)
{
    // 2D index: x = orbit element, y = transform
    int oe_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int t_idx  = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (oe_idx >= total_orbit_elements || t_idx >= num_transforms) return;
    
    // 1. Find which basis state j owns this orbit element
    int j = findBasisState(oe_idx, orbit_offsets, sector_dim);
    
    // 2. Load data
    cuDoubleComplex c_j = __ldg(&x[j]);
    // Skip zero input
    if (cuCreal(c_j) == 0.0 && cuCimag(c_j) == 0.0) return;
    
    uint64_t s = orbit_elements[oe_idx];
    cuDoubleComplex alpha_s = orbit_coefficients[oe_idx];
    double norm_j = orbit_norms[j];
    
    // weighted = c_j * α_s / norm_j
    cuDoubleComplex weighted = cuCdiv(cuCmul(c_j, alpha_s), make_cuDoubleComplex(norm_j, 0.0));
    
    // 3. Apply transform t to state s
    const GPUTransformData& t = transforms[t_idx];
    uint64_t s_prime = s;
    cuDoubleComplex h_element = t.coefficient;
    bool valid = true;
    
    if (!t.is_two_body) {
        // One-body operator
        if (t.op_type == 2) {
            // Sz: diagonal
            double sign = ((s >> t.site_index) & 1) ? -1.0 : 1.0;
            h_element = cuCmul(h_element, make_cuDoubleComplex(spin_l * sign, 0.0));
        } else {
            // S+ or S-: flip bit
            uint64_t bit = (s >> t.site_index) & 1;
            if (bit != t.op_type) {
                s_prime ^= (1ULL << t.site_index);
            } else {
                valid = false;
            }
        }
    } else {
        // Two-body operator
        uint64_t bit_i = (s >> t.site_index) & 1;
        uint64_t bit_j = (s >> t.site_index_2) & 1;
        
        if (t.op_type == 2 && t.op_type_2 == 2) {
            // Sz_i Sz_j: diagonal
            double sign_i = bit_i ? -1.0 : 1.0;
            double sign_j = bit_j ? -1.0 : 1.0;
            h_element = cuCmul(h_element, make_cuDoubleComplex(spin_l * spin_l * sign_i * sign_j, 0.0));
        } else {
            if (t.op_type != 2) {
                if (bit_i != t.op_type) {
                    s_prime ^= (1ULL << t.site_index);
                } else {
                    valid = false;
                }
            } else {
                double sign_i = bit_i ? -1.0 : 1.0;
                h_element = cuCmul(h_element, make_cuDoubleComplex(spin_l * sign_i, 0.0));
            }
            
            if (valid && t.op_type_2 != 2) {
                uint64_t new_bit_j = (s_prime >> t.site_index_2) & 1;
                if (new_bit_j != t.op_type_2) {
                    s_prime ^= (1ULL << t.site_index_2);
                } else {
                    valid = false;
                }
            } else if (valid) {
                uint64_t new_bit_j = (s_prime >> t.site_index_2) & 1;
                double sign_j = new_bit_j ? -1.0 : 1.0;
                h_element = cuCmul(h_element, make_cuDoubleComplex(spin_l * sign_j, 0.0));
            }
        }
    }
    
    if (!valid) return;
    
    // 4. Hash table lookup: s' → (k, projection_factor)
    cuDoubleComplex projection;
    int k = hashLookup(s_prime, hash_table, hash_table_size, &projection);
    if (k < 0) return;  // s' not in this sector
    
    // 5. Accumulate: y[k] += weighted * h_element * projection
    cuDoubleComplex contrib = cuCmul(cuCmul(weighted, h_element), projection);
    
    // Atomic add to real and imaginary parts
    atomicAddDouble(&((double*)&y[k])[0], cuCreal(contrib));
    atomicAddDouble(&((double*)&y[k])[1], cuCimag(contrib));
}

} // namespace GPUKernels

#endif // WITH_CUDA
