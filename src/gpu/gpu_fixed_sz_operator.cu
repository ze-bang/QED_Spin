#ifdef WITH_CUDA

// Prevent inclusion of CPU Operator class
#define CONSTRUCT_HAM_H

#include "gpu_operator.cuh"
#include <iostream>
#include <cmath>

using namespace GPUConfig;

// ============================================================================
// GPUFixedSzOperator Implementation
// ============================================================================

GPUFixedSzOperator::GPUFixedSzOperator(int n_sites, int n_up, float spin_l)
    : GPUOperator(n_sites, spin_l), n_up_(n_up),
      d_basis_states_(nullptr), d_hash_table_(nullptr) {
    
    // Calculate binomial coefficient C(n_sites, n_up) for dimension
    auto binomial = [](int n, int k) -> int64_t {
        if (k > n - k) k = n - k;
        int64_t result = 1;
        for (int i = 0; i < k; ++i) {
            result *= (n - i);
            result /= (i + 1);
        }
        return result;
    };
    
    fixed_sz_dim_ = binomial(n_sites, n_up);
    dimension_ = fixed_sz_dim_;  // Override full dimension
    
    std::cout << "GPU Fixed Sz Operator initialized (OPTIMIZED)\n";
    std::cout << "  Sites: " << n_sites << ", N_up: " << n_up << "\n";
    std::cout << "  Fixed Sz dimension: " << fixed_sz_dim_ << "\n";
    std::cout << "  Reduction factor: " << (1 << n_sites) / (double)fixed_sz_dim_ << "x\n";
    std::cout << "  State lookup: Binary search (warp-coherent, no hash table)\n";
    
    // Hash table not needed - using binary search instead
    hash_table_size_ = 0;
    
    // Build basis on GPU
    buildBasisOnGPU();
}

GPUFixedSzOperator::~GPUFixedSzOperator() {
    if (d_basis_states_) {
        cudaFree(d_basis_states_);
        d_basis_states_ = nullptr;
    }
    if (d_hash_table_) {
        cudaFree(d_hash_table_);
        d_hash_table_ = nullptr;
    }
}

void GPUFixedSzOperator::buildBasisOnGPU() {
    std::cout << "Building fixed Sz basis on GPU...\n";
    
    // Allocate memory for basis states
    CUDA_CHECK(cudaMalloc(&d_basis_states_, fixed_sz_dim_ * sizeof(uint64_t)));
    
    // Generate initial state: lowest n_up bits set
    uint64_t start_state = (1ULL << n_up_) - 1;
    
    // Launch kernel to generate basis
    int num_blocks = (fixed_sz_dim_ + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    GPUKernels::generateFixedSzBasisKernel<<<num_blocks, BLOCK_SIZE>>>(
        d_basis_states_, n_sites_, n_up_, start_state, fixed_sz_dim_);
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    std::cout << "  Basis generation complete (naturally sorted)\n";
    std::cout << "  State lookup optimized: Binary search O(log N) with no warp divergence\n";
    
    // Hash table construction REMOVED - using binary search instead
    // Binary search on sorted basis is faster and more warp-coherent
}

void GPUFixedSzOperator::buildHashTableOnGPU() {
    // DEPRECATED: Hash table no longer used
    // Binary search on sorted basis states is superior:
    // - No warp divergence (all threads follow same path)
    // - O(log N) complexity with better cache behavior
    // - No memory overhead for hash table
    std::cout << "  Note: Hash table construction skipped (using binary search)\n";
}

void GPUFixedSzOperator::matVecFixedSz(const cuDoubleComplex* d_x, cuDoubleComplex* d_y) {
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    
    int num_blocks = (fixed_sz_dim_ + BLOCK_SIZE - 1) / BLOCK_SIZE;
    num_blocks = std::min(num_blocks, MAX_BLOCKS);
    
    if (num_transforms_ > 0) {
        // Copy transform data to device if not already done
        if (d_transform_data_ == nullptr) {
            copyTransformDataToDevice();
        }
        
        // Auto-select kernel based on parallelism potential
        const int TRANSFORM_PARALLEL_THRESHOLD = 64;
        
        if (num_transforms_ > TRANSFORM_PARALLEL_THRESHOLD) {
            // GPU-NATIVE: Transform-parallel kernel (2D parallelism)
            // Zero output vector (required for atomic accumulation)
            CUDA_CHECK(cudaMemset(d_y, 0, fixed_sz_dim_ * sizeof(cuDoubleComplex)));
            
            // 2D grid: (N/16, T/16) with 16×16 blocks
            dim3 block(16, 16);
            dim3 grid((fixed_sz_dim_ + block.x - 1) / block.x,
                     (num_transforms_ + block.y - 1) / block.y);
            
            GPUKernels::matVecFixedSzTransformParallel<<<grid, block>>>(
                d_x, d_y, d_basis_states_,
                d_transform_data_, num_transforms_, fixed_sz_dim_, n_sites_, spin_l_);
        } else {
            // State-parallel kernel (better for small T)
            size_t shared_mem_size = std::min(num_transforms_, 4096) * sizeof(GPUTransformData);
            
            GPUKernels::matVecFixedSzKernelOptimized<<<num_blocks, BLOCK_SIZE, shared_mem_size>>>(
                d_x, d_y, d_basis_states_,
                fixed_sz_dim_, n_sites_, spin_l_,
                d_transform_data_, num_transforms_);
        }
    } else {
        // Fallback to legacy kernel
        GPUKernels::matVecFixedSzKernel<<<num_blocks, BLOCK_SIZE>>>(
            d_x, d_y,
            d_basis_states_,
            nullptr, 0,  // Hash table unused (binary search instead)
            fixed_sz_dim_, n_sites_,
            d_interactions_, num_interactions_,
            d_single_site_ops_, num_single_site_ops_);
    }
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    float milliseconds = 0;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    stats_.matVecTime = milliseconds / 1000.0;
    
    // Estimate throughput
    double flops = static_cast<double>(fixed_sz_dim_) * NNZ_PER_STATE_ESTIMATE * 8;
    stats_.throughput = flops / (stats_.matVecTime * 1e9);
    
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
}

// Override matVecGPU to use fixed Sz version
void GPUFixedSzOperator::matVecGPU(const cuDoubleComplex* d_x, cuDoubleComplex* d_y, int N) {
    if (N != fixed_sz_dim_) {
        throw std::runtime_error("GPUFixedSzOperator::matVecGPU: dimension mismatch");
    }
    matVecFixedSz(d_x, d_y);
}

// Override host-side matVec to use fixed Sz version
void GPUFixedSzOperator::matVec(const std::complex<double>* x, std::complex<double>* y, int N) {
    if (N != fixed_sz_dim_) {
        throw std::runtime_error("GPUFixedSzOperator::matVec: dimension mismatch");
    }
    
    if (!gpu_memory_allocated_) {
        allocateGPUMemory(N);
    }
    
    // Copy input to device
    CUDA_CHECK(cudaMemcpy(d_vector_in_, x, N * sizeof(cuDoubleComplex), 
                         cudaMemcpyHostToDevice));
    
    // Perform matrix-vector product
    matVecFixedSz(d_vector_in_, d_vector_out_);
    
    // Copy output to host
    CUDA_CHECK(cudaMemcpy(y, d_vector_out_, N * sizeof(cuDoubleComplex),
                         cudaMemcpyDeviceToHost));
}

#endif // WITH_CUDA
