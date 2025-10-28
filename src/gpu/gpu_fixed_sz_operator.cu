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
    
    std::cout << "GPU Fixed Sz Operator initialized\n";
    std::cout << "  Sites: " << n_sites << ", N_up: " << n_up << "\n";
    std::cout << "  Fixed Sz dimension: " << fixed_sz_dim_ << "\n";
    std::cout << "  Reduction factor: " << (1 << n_sites) / (double)fixed_sz_dim_ << "x\n";
    
    // Choose hash table size (should be prime and larger than basis size)
    hash_table_size_ = fixed_sz_dim_ * 2 + 1;
    // Find next prime
    while (true) {
        bool is_prime = true;
        for (int i = 2; i * i <= hash_table_size_; ++i) {
            if (hash_table_size_ % i == 0) {
                is_prime = false;
                break;
            }
        }
        if (is_prime) break;
        hash_table_size_++;
    }
    
    std::cout << "  Hash table size: " << hash_table_size_ << "\n";
    
    // Build basis on GPU
    buildBasisOnGPU();
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
    
    std::cout << "  Basis generation complete\n";
    
    // Build hash table
    buildHashTableOnGPU();
}

void GPUFixedSzOperator::buildHashTableOnGPU() {
    std::cout << "Building hash table on GPU...\n";
    
    // Allocate hash table
    CUDA_CHECK(cudaMalloc(&d_hash_table_, hash_table_size_ * sizeof(HashEntry)));
    
    // Initialize hash table to zero
    CUDA_CHECK(cudaMemset(d_hash_table_, 0, hash_table_size_ * sizeof(HashEntry)));
    
    // Build hash table
    int num_blocks = (fixed_sz_dim_ + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    GPUKernels::buildHashTableKernel<<<num_blocks, BLOCK_SIZE>>>(
        d_basis_states_, d_hash_table_, hash_table_size_, fixed_sz_dim_);
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    std::cout << "  Hash table construction complete\n";
}

void GPUFixedSzOperator::matVecFixedSz(const cuDoubleComplex* d_x, cuDoubleComplex* d_y) {
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    
    int num_blocks = (fixed_sz_dim_ + BLOCK_SIZE - 1) / BLOCK_SIZE;
    num_blocks = std::min(num_blocks, MAX_BLOCKS);
    
    GPUKernels::matVecFixedSzKernel<<<num_blocks, BLOCK_SIZE>>>(
        d_x, d_y,
        d_basis_states_,
        d_hash_table_, hash_table_size_,
        fixed_sz_dim_, n_sites_,
        d_interactions_, num_interactions_,
        d_single_site_ops_, num_single_site_ops_);
    
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

#endif // WITH_CUDA
