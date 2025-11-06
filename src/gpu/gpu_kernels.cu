#ifdef WITH_CUDA

#include "gpu_operator.cuh"
#include "bit_operations.cuh"

using namespace GPUBitOps;

// ============================================================================
// CUDA Kernels Implementation
// ============================================================================

namespace GPUKernels {

/**
 * Helper device function to apply a single operator to a state
 */
__device__ void applyOperator(uint64_t state, int site, char op,
                             uint64_t& new_state, cuDoubleComplex& amplitude) {
    switch(op) {
        case '+':  // S+
            apply_sp(state, site, new_state, amplitude);
            break;
        case '-':  // S-
            apply_sm(state, site, new_state, amplitude);
            break;
        case 'z':  // Sz
            apply_sz(state, site, new_state, amplitude);
            break;
        case 'x':  // Sx
            apply_sx(state, site, new_state, amplitude);
            break;
        case 'y':  // Sy
            apply_sy(state, site, new_state, amplitude);
            break;
        default:
            new_state = state;
            amplitude = make_cuDoubleComplex(0.0, 0.0);
    }
}

/**
 * Main matrix-vector product kernel
 * Computes y = H * x on-the-fly without storing the matrix
 */
__global__ void matVecKernel(const cuDoubleComplex* x, cuDoubleComplex* y,
                             int N, int n_sites,
                             const void* interactions_ptr, int num_interactions,
                             const void* single_site_ops_ptr, int num_single_site_ops) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    
    // Current basis state (represented as integer)
    uint64_t state = static_cast<uint64_t>(idx);
    
    // Accumulator for this row of H
    cuDoubleComplex result = make_cuDoubleComplex(0.0, 0.0);
    
    // Cast void pointers to appropriate types
    typedef struct {int site1, site2; char op1, op2; double coupling;} Interaction;
    typedef struct {int site; char op; double coupling;} SingleSiteOp;
    
    const Interaction* interactions = static_cast<const Interaction*>(interactions_ptr);
    const SingleSiteOp* single_site_ops = static_cast<const SingleSiteOp*>(single_site_ops_ptr);
    
    // Apply two-site interactions
    for (int i = 0; i < num_interactions; ++i) {
        const Interaction& inter = interactions[i];
        
        // Apply first operator
        uint64_t temp_state1;
        cuDoubleComplex amp1;
        applyOperator(state, inter.site1, inter.op1, temp_state1, amp1);
        
        if (cuCreal(amp1) == 0.0 && cuCimag(amp1) == 0.0) continue;
        
        // Apply second operator
        uint64_t final_state;
        cuDoubleComplex amp2;
        applyOperator(temp_state1, inter.site2, inter.op2, final_state, amp2);
        
        if (cuCreal(amp2) == 0.0 && cuCimag(amp2) == 0.0) continue;
        
        // Total amplitude
        cuDoubleComplex total_amp = cuCmul(amp1, amp2);
        total_amp = make_cuDoubleComplex(
            cuCreal(total_amp) * inter.coupling,
            cuCimag(total_amp) * inter.coupling
        );
        
        // Add contribution: result += total_amp * x[final_state]
        if (final_state < N) {
            cuDoubleComplex contrib = cuCmul(total_amp, x[final_state]);
            result = complex_add(result, contrib);
        }
    }
    
    // Apply single-site operators
    for (int i = 0; i < num_single_site_ops; ++i) {
        const SingleSiteOp& op = single_site_ops[i];
        
        uint64_t new_state;
        cuDoubleComplex amp;
        applyOperator(state, op.site, op.op, new_state, amp);
        
        if (cuCreal(amp) == 0.0 && cuCimag(amp) == 0.0) continue;
        
        cuDoubleComplex scaled_amp = complex_scale(amp, op.coupling);
        
        if (new_state < N) {
            cuDoubleComplex contrib = cuCmul(scaled_amp, x[new_state]);
            result = complex_add(result, contrib);
        }
    }
    
    // Write result
    y[idx] = result;
}

/**
 * Sparse matrix-vector product kernel (CSR format)
 */
__global__ void sparseMatVecKernel(const int* row_ptr, const int* col_ind,
                                   const cuDoubleComplex* values,
                                   const cuDoubleComplex* x, cuDoubleComplex* y,
                                   int N) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= N) return;
    
    cuDoubleComplex sum = make_cuDoubleComplex(0.0, 0.0);
    int row_start = row_ptr[row];
    int row_end = row_ptr[row + 1];
    
    for (int i = row_start; i < row_end; ++i) {
        int col = col_ind[i];
        cuDoubleComplex val = values[i];
        cuDoubleComplex x_val = x[col];
        sum = complex_add(sum, cuCmul(val, x_val));
    }
    
    y[row] = sum;
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
 * Build hash table for fast state lookup
 * Uses open addressing with linear probing
 */
__global__ void buildHashTableKernel(const uint64_t* basis_states, void* hash_table_ptr,
                                    int hash_size, int num_states) {
    typedef struct {uint64_t state; int index;} HashEntry;
    HashEntry* hash_table = static_cast<HashEntry*>(hash_table_ptr);
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_states) return;
    
    uint64_t state = basis_states[idx];
    
    // Hash function: simple modulo
    int hash = state % hash_size;
    
    // Linear probing to find empty slot
    while (true) {
        uint64_t old = atomicCAS(
            (unsigned long long*)&hash_table[hash].state,
            0ULL, state
        );
        
        if (old == 0ULL || old == state) {
            // Successfully inserted or already present
            hash_table[hash].index = idx;
            break;
        }
        
        // Try next slot
        hash = (hash + 1) % hash_size;
    }
}

/**
 * Device function to lookup state in hash table
 * Returns index in basis, or -1 if not found
 */
__device__ int lookupState(uint64_t state, const void* hash_table_ptr, int hash_size) {
    typedef struct {uint64_t state; int index;} HashEntry;
    const HashEntry* hash_table = static_cast<const HashEntry*>(hash_table_ptr);
    
    int hash = state % hash_size;
    
    for (int probe = 0; probe < hash_size; ++probe) {
        uint64_t stored_state = hash_table[hash].state;
        
        if (stored_state == state) {
            return hash_table[hash].index;
        }
        if (stored_state == 0) {
            return -1;  // Not found
        }
        
        hash = (hash + 1) % hash_size;
    }
    
    return -1;  // Not found
}

/**
 * Matrix-vector product kernel for fixed Sz sector
 */
__global__ void matVecFixedSzKernel(const cuDoubleComplex* x, cuDoubleComplex* y,
                                    const uint64_t* basis_states,
                                    const void* hash_table, int hash_size,
                                    int N, int n_sites,
                                    const void* interactions_ptr, int num_interactions,
                                    const void* single_site_ops_ptr, int num_single_site_ops) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    
    // Get basis state for this index
    uint64_t state = basis_states[idx];
    
    cuDoubleComplex result = make_cuDoubleComplex(0.0, 0.0);
    
    typedef struct {int site1, site2; char op1, op2; double coupling;} Interaction;
    typedef struct {int site; char op; double coupling;} SingleSiteOp;
    
    const Interaction* interactions = static_cast<const Interaction*>(interactions_ptr);
    const SingleSiteOp* single_site_ops = static_cast<const SingleSiteOp*>(single_site_ops_ptr);
    
    // Apply two-site interactions
    for (int i = 0; i < num_interactions; ++i) {
        const Interaction& inter = interactions[i];
        
        uint64_t temp_state1;
        cuDoubleComplex amp1;
        applyOperator(state, inter.site1, inter.op1, temp_state1, amp1);
        
        if (cuCreal(amp1) == 0.0 && cuCimag(amp1) == 0.0) continue;
        
        uint64_t final_state;
        cuDoubleComplex amp2;
        applyOperator(temp_state1, inter.site2, inter.op2, final_state, amp2);
        
        if (cuCreal(amp2) == 0.0 && cuCimag(amp2) == 0.0) continue;
        
        cuDoubleComplex total_amp = cuCmul(amp1, amp2);
        total_amp = complex_scale(total_amp, inter.coupling);
        
        // Look up final state in hash table
        int final_idx = lookupState(final_state, hash_table, hash_size);
        if (final_idx >= 0) {
            cuDoubleComplex contrib = cuCmul(total_amp, x[final_idx]);
            result = complex_add(result, contrib);
        }
    }
    
    // Apply single-site operators
    for (int i = 0; i < num_single_site_ops; ++i) {
        const SingleSiteOp& op = single_site_ops[i];
        
        uint64_t new_state;
        cuDoubleComplex amp;
        applyOperator(state, op.site, op.op, new_state, amp);
        
        if (cuCreal(amp) == 0.0 && cuCimag(amp) == 0.0) continue;
        
        cuDoubleComplex scaled_amp = complex_scale(amp, op.coupling);
        
        int new_idx = lookupState(new_state, hash_table, hash_size);
        if (new_idx >= 0) {
            cuDoubleComplex contrib = cuCmul(scaled_amp, x[new_idx]);
            result = complex_add(result, contrib);
        }
    }
    
    y[idx] = result;
}

} // namespace GPUKernels

#endif // WITH_CUDA
