#ifdef WITH_CUDA

#include <ed/gpu/gpu_operator.cuh>
#include <ed/gpu/bit_operations.cuh>

#include <vector>

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
__global__ void matVecKernelOptimized(cuDoubleComplex* y,
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
 * Constant-memory Pascal triangle for combinadic unrank.
 *
 * Indexed as d_pascal_shared[n][k] = C(n, k) for 0 <= n, k <= 64.
 * ~33.8 KiB, fits well within the 64 KiB constant-memory limit on
 * every CUDA arch we target.
 *
 * Phase E.1 of the "Kill the GPU State-Lookup Hash" plan moved this
 * symbol into the ``ed::gpu::combinadic`` namespace so that other
 * CUDA TUs (e.g. ``streaming_symmetry_gpu_mirror.cu``) can read the
 * SAME constant table via ``extern __device__ __constant__`` in
 * ``include/ed/gpu/combinadic.cuh``. Without sharing, each TU would
 * upload its own 33 KiB table and the device-link object would blow
 * the 64 KiB per-binary constant-memory budget.
 *
 * Populated once from the host via cudaMemcpyToSymbol when the first
 * fixed-Sz operator is constructed (see ensure_pascal_uploaded below).
 * Storage is uint64_t — C(64, 32) = 1.83e18 fits in 64 bits.
 */
// We are currently inside ``namespace GPUKernels { ... }``. Briefly
// close it so the constant-memory symbol lives at its true linkage
// home (``::ed::gpu::combinadic::d_pascal_shared``), then reopen
// ``GPUKernels`` to continue the original file.
}  // namespace GPUKernels (paused for cross-TU symbol)

namespace ed::gpu::combinadic {
__device__ __constant__ unsigned long long d_pascal_shared[65][65];
}  // namespace ed::gpu::combinadic

namespace GPUKernels {

static __device__ __forceinline__ unsigned long long binomial_dev(int n, int k) {
    if (k < 0 || k > n || n < 0 || n > 64) return 0ULL;
    return ::ed::gpu::combinadic::d_pascal_shared[n][k];
}

/**
 * Combinadic unrank: O(k) per thread, total O(N * k) for the whole basis.
 *
 * For colex-ordered subsets of size k drawn from {0, ..., n_bits - 1}
 * (Gosper's hack produces this order when interpreted as bit positions),
 * the rank-r combination has positions p_{k-1} > ... > p_0 satisfying
 *   r = sum_{i=0}^{k-1} C(p_i, i + 1).
 * We extract them top-down: at step i, find the largest p with
 * C(p, i+1) <= r, set bit p, subtract, recurse.
 *
 * Replaces the previous Gosper-walk implementation which cost O(idx)
 * per thread (i.e. O(N^2) total work). For C(20, 10) = 184756 this is a
 * ~5000x speedup at basis construction.
 */
static __device__ uint64_t unrank_combination_dev(uint64_t rank, int n_bits, int k) {
    uint64_t state = 0ULL;
    for (int i = k - 1; i >= 0; --i) {
        // Find largest p in [i, n_bits - 1] with C(p, i+1) <= rank.
        // Linear scan is O(n_bits) — fine for n_bits <= 64; binary
        // search would shave a factor of log(n_bits) but adds branch
        // divergence, which hurts on warps where all threads are in
        // lock-step.
        int p = i;
        while (p + 1 < n_bits && binomial_dev(p + 1, i + 1) <= rank) {
            ++p;
        }
        state |= (1ULL << p);
        rank -= binomial_dev(p, i + 1);
    }
    return state;
}


/**
 * Combinadic RANK: inverse of unrank_combination_dev.
 *
 * Phase A.1 of the "Kill the GPU State-Lookup Hash" plan (May 2026):
 * `unrank_combination_dev` already enumerates the fixed-Sz basis by
 * walking bit positions in colex order and consuming the rank top-down.
 * The inverse direction was never written, so every state -> idx lookup
 * was forced through ``d_state_hash_`` (8 - 32 GiB random-access HBM
 * probe table).
 *
 * The colex convention used by unrank: a combination with set-bit
 * positions ``p_{k-1} > ... > p_0`` has rank
 *
 *     rank = sum_{i=0}^{k-1} C(p_i, i + 1).
 *
 * Scanning ``state`` from low bit to high bit, the (seen)-th set bit
 * encountered is exactly ``p_{seen-1}`` (with index ``seen``) and
 * contributes ``C(p_{seen-1}, seen)``.  All reads land in
 * ``__constant__`` cache (the Pascal triangle); no HBM traffic. The
 * loop runs at most n_bits times (<= 64).
 */
static __device__ __forceinline__
int rank_combination_dev(uint64_t state, int n_bits, int k) {
    int rank = 0;
    int seen = 0;
    #pragma unroll
    for (int bit = 0; bit < 64; ++bit) {
        if (bit >= n_bits) break;
        if (seen >= k) break;
        if ((state >> bit) & 1ULL) {
            ++seen;
            // binomial_dev(bit, seen) reads d_pascal[bit][seen] from
            // __constant__ memory; bit < n_bits <= 64 and seen <= k <= 32
            // so we are inside the uploaded 65x65 triangle.
            rank += static_cast<int>(binomial_dev(bit, seen));
        }
    }
    return rank;
}


// =============================================================================
// Host-callable roundtrip test harness for rank_combination_dev.
//
// Used by tests/unit/test_gpu_fixed_sz_rank.cpp to pin
//     rank_combination_dev(unrank_combination_dev(r), N, k) == r
// for r in [0, C(N, k)). The kernel writes 1 into ``d_fail`` on the
// first mismatch; the host reads back ``h_fail`` to assert correctness.
//
// We accept a host-provided list of ranks (rather than scanning the
// whole [0, dim) range) so the test can keep memory bounded at large
// (N, k) like (32, 16) where C(32, 16) ~ 6e8.
// =============================================================================
__global__ void rankUnrankRoundtripKernel(const uint64_t* ranks_in,
                                          int num_ranks, int n_bits, int k,
                                          int* d_fail,
                                          uint64_t* d_first_fail_rank) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_ranks) return;

    uint64_t r = ranks_in[tid];
    uint64_t state = unrank_combination_dev(r, n_bits, k);
    int  popcount = __popcll(state);
    int  back = rank_combination_dev(state, n_bits, k);

    if (popcount != k || static_cast<uint64_t>(back) != r) {
        // Race is fine: any failing thread wins, the host just needs to
        // know SOMETHING failed at this (N, k).
        atomicExch(d_fail, 1);
        atomicExch(reinterpret_cast<unsigned long long*>(d_first_fail_rank),
                   static_cast<unsigned long long>(r));
    }
}

/**
 * Host-callable: upload Pascal triangle to constant memory. Idempotent
 * via a static flag — multiple GPUFixedSzOperator constructions share
 * the same uploaded table.
 */
void ensure_pascal_uploaded() {
    static bool uploaded = false;
    if (uploaded) return;
    unsigned long long h_pascal[65][65] = {};
    for (int n = 0; n <= 64; ++n) {
        h_pascal[n][0] = 1ULL;
        for (int k = 1; k <= n; ++k) {
            unsigned long long left  = (k - 1 >= 0) ? h_pascal[n - 1][k - 1] : 0ULL;
            unsigned long long right = (k < n)      ? h_pascal[n - 1][k]     : 0ULL;
            h_pascal[n][k] = left + right;
        }
    }
    cudaMemcpyToSymbol(::ed::gpu::combinadic::d_pascal_shared,
                       h_pascal, sizeof(h_pascal));
    uploaded = true;
}

}  // namespace GPUKernels (paused so we can define the cross-TU
   // ``ed::gpu::combinadic::upload_pascal_shared`` symbol at its true
   // linkage home; reopened immediately below)

// Phase E.1: expose the same upload through the public combinadic
// namespace so callers that only include ``ed/gpu/combinadic.cuh``
// (no internal ed_solvers_gpu deps) can populate the shared table.
namespace ed::gpu::combinadic {
void upload_pascal_shared() { GPUKernels::ensure_pascal_uploaded(); }
}  // namespace ed::gpu::combinadic

namespace GPUKernels {

/**
 * Host-callable roundtrip launcher for rank_combination_dev.
 *
 * Phase A.1 of the "Kill the GPU State-Lookup Hash" plan (May 2026).
 * The test harness builds a host-side list of ranks (random sample or
 * exhaustive [0, dim)) and asks the GPU whether
 *
 *     rank_combination_dev(unrank_combination_dev(r), N, k) == r
 *
 * for every entry. Returns:
 *   - ``true``: all ranks roundtripped.
 *   - ``false``: at least one failure; the offending rank is written
 *     to ``*first_fail_rank_out`` (host).
 *
 * The host stages the input ranks H -> D, launches the kernel, reads
 * back two scalars (failure flag + first failing rank), and frees.
 * This is a CORRECTNESS test, not a perf path, so the H2D + D2H + sync
 * cost is fine.
 */
bool gpu_rank_unrank_roundtrip(const std::vector<uint64_t>& ranks,
                               int n_bits, int k,
                               uint64_t* first_fail_rank_out) {
    if (first_fail_rank_out != nullptr) {
        *first_fail_rank_out = static_cast<uint64_t>(-1);
    }
    if (ranks.empty()) return true;
    if (n_bits <= 0 || n_bits > 64 || k < 0 || k > n_bits) return false;

    ensure_pascal_uploaded();

    uint64_t* d_ranks            = nullptr;
    int*      d_fail             = nullptr;
    uint64_t* d_first_fail_rank  = nullptr;
    size_t    bytes_in           = ranks.size() * sizeof(uint64_t);

    if (cudaMalloc(&d_ranks, bytes_in)               != cudaSuccess) return false;
    if (cudaMalloc(&d_fail, sizeof(int))             != cudaSuccess) { cudaFree(d_ranks); return false; }
    if (cudaMalloc(&d_first_fail_rank, sizeof(uint64_t)) != cudaSuccess) {
        cudaFree(d_ranks); cudaFree(d_fail); return false;
    }

    cudaMemcpy(d_ranks, ranks.data(), bytes_in, cudaMemcpyHostToDevice);
    cudaMemset(d_fail, 0, sizeof(int));
    cudaMemset(d_first_fail_rank, 0xFF, sizeof(uint64_t));  // -> UINT64_MAX

    const int threads = 256;
    const int blocks  = static_cast<int>((ranks.size() + threads - 1) / threads);
    rankUnrankRoundtripKernel<<<blocks, threads>>>(
        d_ranks, static_cast<int>(ranks.size()), n_bits, k,
        d_fail, d_first_fail_rank);

    int      h_fail = 0;
    uint64_t h_fail_rank = static_cast<uint64_t>(-1);
    cudaMemcpy(&h_fail, d_fail, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_fail_rank, d_first_fail_rank, sizeof(uint64_t), cudaMemcpyDeviceToHost);

    cudaFree(d_ranks);
    cudaFree(d_fail);
    cudaFree(d_first_fail_rank);

    if (h_fail != 0) {
        if (first_fail_rank_out != nullptr) *first_fail_rank_out = h_fail_rank;
        return false;
    }
    return true;
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
// MIXED-PRECISION CAST KERNELS (Phase 3a #3)
//
// These are tight, embarrassingly parallel element-wise casts. They run on
// the same stream as the surrounding cuSPARSE call, so no host-side sync
// is needed: GPU ordering on the stream guarantees the cast completes
// before cusparseSpMV reads the FP32 input vector.
// ============================================================================
__global__ void castDoubleToFloatComplex(const cuDoubleComplex* in,
                                         cuFloatComplex* out, int N) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    const cuDoubleComplex z = in[idx];
    out[idx] = make_cuFloatComplex(static_cast<float>(cuCreal(z)),
                                   static_cast<float>(cuCimag(z)));
}

__global__ void castFloatToDoubleComplex(const cuFloatComplex* in,
                                         cuDoubleComplex* out, int N) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    const cuFloatComplex z = in[idx];
    out[idx] = make_cuDoubleComplex(static_cast<double>(cuCrealf(z)),
                                    static_cast<double>(cuCimagf(z)));
}

// Same kernel as castDoubleToFloatComplex but with int64 length so we can
// process the full nnz (which can exceed INT_MAX on big sectors). Kept
// distinct so the per-vector launch (where N fits in int) doesn't pay
// the 64-bit index arithmetic on every thread.
__global__ void castDoubleToFloatComplexValues(const cuDoubleComplex* in,
                                               cuFloatComplex* out, int64_t nnz) {
    const int64_t idx =
        static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= nnz) return;
    const cuDoubleComplex z = in[idx];
    out[idx] = make_cuFloatComplex(static_cast<float>(cuCreal(z)),
                                   static_cast<float>(cuCimag(z)));
}

} // namespace GPUKernels

#endif // WITH_CUDA
