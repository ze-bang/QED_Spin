#ifdef WITH_CUDA

// Prevent inclusion of CPU Operator class
#define CONSTRUCT_HAM_H

#include <ed/gpu/gpu_operator.cuh>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <algorithm>

using namespace GPUConfig;

// ============================================================================
// GPUSymmetrizedOperator Implementation
// ============================================================================

GPUSymmetrizedOperator::GPUSymmetrizedOperator(int n_sites, float spin_l)
    : GPUOperator(n_sites, spin_l) {
    std::cout << "GPU Symmetrized Operator initialized\n";
    std::cout << "  Sites: " << n_sites << ", Spin: " << spin_l << "\n";
}

GPUSymmetrizedOperator::~GPUSymmetrizedOperator() {
    if (d_orbit_elements_)     cudaFree(d_orbit_elements_);
    if (d_orbit_coefficients_) cudaFree(d_orbit_coefficients_);
    if (d_orbit_offsets_)      cudaFree(d_orbit_offsets_);
    if (d_orbit_norms_)        cudaFree(d_orbit_norms_);
    if (d_hash_table_)         cudaFree(d_hash_table_);
}

void GPUSymmetrizedOperator::setSectorData(
    int sector_dim,
    const std::vector<uint64_t>& orbit_elements,
    const std::vector<std::complex<double>>& orbit_coefficients,
    const std::vector<int>& orbit_offsets,
    const std::vector<double>& orbit_norms,
    int group_size)
{
    sector_dim_ = sector_dim;
    group_size_ = group_size;
    total_orbit_elements_ = static_cast<int>(orbit_elements.size());
    dimension_ = sector_dim;  // Override base class dimension
    
    h_orbit_elements_ = orbit_elements;
    h_orbit_coefficients_ = orbit_coefficients;
    h_orbit_offsets_ = orbit_offsets;
    h_orbit_norms_ = orbit_norms;
    
    std::cout << "  Sector dimension: " << sector_dim_ << "\n";
    std::cout << "  Total orbit elements: " << total_orbit_elements_ << "\n";
    std::cout << "  Group size: " << group_size_ << "\n";
    std::cout << "  Avg orbit size: " << std::fixed << std::setprecision(1)
              << static_cast<double>(total_orbit_elements_) / sector_dim_ << "\n";
    
    // Copy orbit data to GPU
    copyOrbitDataToDevice();
    
    // Build hash table
    buildHashTable();
    
    sector_data_on_device_ = true;
}

void GPUSymmetrizedOperator::copyOrbitDataToDevice() {
    // Orbit elements
    CUDA_CHECK(cudaMalloc(&d_orbit_elements_, 
                          total_orbit_elements_ * sizeof(uint64_t)));
    CUDA_CHECK(cudaMemcpy(d_orbit_elements_, h_orbit_elements_.data(),
                          total_orbit_elements_ * sizeof(uint64_t),
                          cudaMemcpyHostToDevice));
    
    // Orbit coefficients: convert std::complex<double> to cuDoubleComplex
    std::vector<cuDoubleComplex> cu_coeffs(total_orbit_elements_);
    for (int i = 0; i < total_orbit_elements_; ++i) {
        cu_coeffs[i] = make_cuDoubleComplex(h_orbit_coefficients_[i].real(),
                                             h_orbit_coefficients_[i].imag());
    }
    CUDA_CHECK(cudaMalloc(&d_orbit_coefficients_,
                          total_orbit_elements_ * sizeof(cuDoubleComplex)));
    CUDA_CHECK(cudaMemcpy(d_orbit_coefficients_, cu_coeffs.data(),
                          total_orbit_elements_ * sizeof(cuDoubleComplex),
                          cudaMemcpyHostToDevice));
    
    // Orbit offsets (sector_dim + 1 entries)
    CUDA_CHECK(cudaMalloc(&d_orbit_offsets_,
                          (sector_dim_ + 1) * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_orbit_offsets_, h_orbit_offsets_.data(),
                          (sector_dim_ + 1) * sizeof(int),
                          cudaMemcpyHostToDevice));
    
    // Orbit norms
    CUDA_CHECK(cudaMalloc(&d_orbit_norms_, sector_dim_ * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d_orbit_norms_, h_orbit_norms_.data(),
                          sector_dim_ * sizeof(double),
                          cudaMemcpyHostToDevice));
    
    size_t orbit_mem = total_orbit_elements_ * (sizeof(uint64_t) + sizeof(cuDoubleComplex))
                     + (sector_dim_ + 1) * sizeof(int)
                     + sector_dim_ * sizeof(double);
    std::cout << "  Orbit data on GPU: " << orbit_mem / 1024.0 << " KB\n";
}

void GPUSymmetrizedOperator::buildHashTable() {
    // Size the hash table to ~2x the number of entries for low collision rate
    hash_table_size_ = static_cast<int>(total_orbit_elements_ * 2);
    // Round up to next power of 2 for fast modulo (optional, but helps)
    int p = 1;
    while (p < hash_table_size_) p *= 2;
    hash_table_size_ = p;
    
    // Build hash table on host
    std::vector<GPUHashEntry> h_table(hash_table_size_);
    // All entries initialized with key = UINT64_MAX (empty) by default constructor
    
    const double group_norm = 1.0 / static_cast<double>(group_size_);
    
    int collisions = 0;
    for (int j = 0; j < sector_dim_; ++j) {
        int start = h_orbit_offsets_[j];
        int end = h_orbit_offsets_[j + 1];
        double norm_j = h_orbit_norms_[j];
        
        for (int oe = start; oe < end; ++oe) {
            uint64_t state = h_orbit_elements_[oe];
            std::complex<double> coeff = h_orbit_coefficients_[oe];
            
            // Pre-compute projection factor: conj(coeff) * group_norm / norm_j
            std::complex<double> proj = std::conj(coeff) * group_norm / norm_j;
            
            // Insert into hash table with linear probing
            uint64_t hash = state * 11400714819323198485ULL;
            int idx = static_cast<int>(hash % static_cast<uint64_t>(hash_table_size_));
            
            int probes = 0;
            while (h_table[idx].key != UINT64_MAX) {
                // Duplicate key shouldn't happen in well-formed orbit data
                if (h_table[idx].key == state) {
                    std::cerr << "Warning: Duplicate state " << state 
                              << " in hash table (basis " << j << ")\n";
                    break;
                }
                idx = (idx + 1) % hash_table_size_;
                probes++;
                collisions++;
            }
            
            h_table[idx].key = state;
            h_table[idx].value = j;
            h_table[idx].projection = make_cuDoubleComplex(proj.real(), proj.imag());
        }
    }
    
    // Copy hash table to device
    CUDA_CHECK(cudaMalloc(&d_hash_table_, hash_table_size_ * sizeof(GPUHashEntry)));
    CUDA_CHECK(cudaMemcpy(d_hash_table_, h_table.data(),
                          hash_table_size_ * sizeof(GPUHashEntry),
                          cudaMemcpyHostToDevice));
    
    size_t hash_mem = hash_table_size_ * sizeof(GPUHashEntry);
    std::cout << "  Hash table: " << hash_table_size_ << " slots, "
              << hash_mem / 1024.0 << " KB, "
              << collisions << " collisions during build\n";
}

void GPUSymmetrizedOperator::matVec(
    const std::complex<double>* x, std::complex<double>* y, int N)
{
    if (!gpu_memory_allocated_) {
        allocateGPUMemory(N);
    }
    
    // Copy input to device
    CUDA_CHECK(cudaMemcpy(d_vector_in_, x, N * sizeof(cuDoubleComplex),
                          cudaMemcpyHostToDevice));
    
    // GPU matvec
    matVecGPU(d_vector_in_, d_vector_out_, N);
    
    // Copy result back
    CUDA_CHECK(cudaMemcpy(y, d_vector_out_, N * sizeof(cuDoubleComplex),
                          cudaMemcpyDeviceToHost));
}

void GPUSymmetrizedOperator::matVecGPU(
    const cuDoubleComplex* d_x, cuDoubleComplex* d_y, int N)
{
    if (N != sector_dim_) {
        std::cerr << "Error: GPUSymmetrizedOperator::matVecGPU dimension mismatch ("
                  << N << " vs " << sector_dim_ << ")\n";
        return;
    }
    
    if (!sector_data_on_device_) {
        std::cerr << "Error: Sector data not on device\n";
        return;
    }
    
    // Ensure transform data on device
    if (d_transform_data_ == nullptr && !transform_data_.empty()) {
        copyTransformDataToDevice();
    }
    
    if (num_transforms_ == 0) {
        std::cerr << "Error: No transforms loaded\n";
        CUDA_CHECK(cudaMemset(d_y, 0, N * sizeof(cuDoubleComplex)));
        return;
    }
    
    // Timing
    CUDA_CHECK(cudaEventRecord(timing_start_));
    
    // Zero output
    CUDA_CHECK(cudaMemset(d_y, 0, N * sizeof(cuDoubleComplex)));
    
    // Launch 2D kernel: (total_orbit_elements, num_transforms)
    dim3 block(16, 16);
    dim3 grid((total_orbit_elements_ + block.x - 1) / block.x,
              (num_transforms_ + block.y - 1) / block.y);
    
    GPUKernels::matVecSymmetrized<<<grid, block>>>(
        d_x, d_y,
        d_orbit_elements_, d_orbit_coefficients_,
        d_orbit_offsets_, d_orbit_norms_,
        sector_dim_,
        d_transform_data_, num_transforms_,
        d_hash_table_, hash_table_size_,
        n_sites_, spin_l_,
        total_orbit_elements_);
    
    CUDA_CHECK(cudaGetLastError());
    
    // Timing
    CUDA_CHECK(cudaEventRecord(timing_stop_));
    CUDA_CHECK(cudaEventSynchronize(timing_stop_));
    
    float milliseconds = 0;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, timing_start_, timing_stop_));
    stats_.matVecTime = milliseconds / 1000.0;
}

void GPUSymmetrizedOperator::matVecGPUAsync(
    const cuDoubleComplex* d_x, cuDoubleComplex* d_y, int N, cudaStream_t stream)
{
    // Synchronous fallback — hash table + atomic scatter not safe for multi-stream
    (void)stream;
    matVecGPU(d_x, d_y, N);
}

#endif // WITH_CUDA
