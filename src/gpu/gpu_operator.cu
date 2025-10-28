#ifdef WITH_CUDA

// Prevent inclusion of CPU Operator class that has CUDA-incompatible code
#define CONSTRUCT_HAM_H  

#include "gpu_operator.cuh"
#include <iostream>
#include <cmath>
#include <algorithm>

using namespace GPUConfig;

// ============================================================================
// GPUOperator Implementation
// ============================================================================

GPUOperator::GPUOperator(int n_sites, float spin_l)
    : n_sites_(n_sites), spin_l_(spin_l), dimension_(1 << n_sites),
      d_vector_in_(nullptr), d_vector_out_(nullptr), d_temp_(nullptr),
      d_csr_row_ptr_(nullptr), d_csr_col_ind_(nullptr), d_csr_values_(nullptr),
      nnz_(0), d_interactions_(nullptr), d_single_site_ops_(nullptr),
      num_interactions_(0), num_single_site_ops_(0),
      gpu_memory_allocated_(false), sparse_matrix_built_(false) {
    
    if (n_sites > MAX_SITES) {
        throw std::runtime_error("Number of sites exceeds maximum supported (" 
                               + std::to_string(MAX_SITES) + ")");
    }
    
    // Get available GPU memory
    size_t free_mem, total_mem;
    CUDA_CHECK(cudaMemGetInfo(&free_mem, &total_mem));
    available_gpu_memory_ = free_mem;
    
    std::cout << "GPU Operator initialized for " << n_sites << " sites\n";
    std::cout << "Hilbert space dimension: " << dimension_ << "\n";
    std::cout << "Available GPU memory: " << free_mem / (1024.0 * 1024.0 * 1024.0) 
              << " GB\n";
    
    // Initialize CUDA libraries
    initializeCUSPARSE();
    initializeCUBLAS();
    
    // Initialize stats
    stats_.matVecTime = 0.0;
    stats_.memoryUsed = 0.0;
    stats_.numChunks = 0;
    stats_.throughput = 0.0;
}

GPUOperator::~GPUOperator() {
    freeGPUMemory();
    
    if (cusparse_handle_) {
        cusparseDestroy(cusparse_handle_);
    }
    if (cublas_handle_) {
        cublasDestroy(cublas_handle_);
    }
}

void GPUOperator::initializeCUSPARSE() {
    CUSPARSE_CHECK(cusparseCreate(&cusparse_handle_));
}

void GPUOperator::initializeCUBLAS() {
    CUBLAS_CHECK(cublasCreate(&cublas_handle_));
}

void GPUOperator::setInteraction(int site1, int site2, char op1, char op2, double coupling) {
    interactions_.push_back({site1, site2, op1, op2, coupling});
}

void GPUOperator::setSingleSite(int site, char op, double coupling) {
    single_site_ops_.push_back({site, op, coupling});
}

size_t GPUOperator::estimateMemoryRequirement(int N) const {
    size_t vector_size = N * sizeof(cuDoubleComplex);
    size_t sparse_matrix_size = 0;
    
    // Estimate non-zeros per row
    int nnz_per_row = NNZ_PER_STATE_ESTIMATE;
    size_t nnz_estimate = static_cast<size_t>(N) * nnz_per_row;
    
    sparse_matrix_size = (N + 1) * sizeof(int) +  // row pointers
                        nnz_estimate * sizeof(int) +  // column indices
                        nnz_estimate * sizeof(cuDoubleComplex);  // values
    
    size_t total = 3 * vector_size + sparse_matrix_size;
    
    return total;
}

bool GPUOperator::allocateGPUMemory(int N) {
    if (gpu_memory_allocated_) {
        freeGPUMemory();
    }
    
    size_t required_memory = estimateMemoryRequirement(N);
    
    if (required_memory > available_gpu_memory_ * 0.9) {
        std::cout << "Warning: Required memory (" << required_memory / (1024.0*1024.0*1024.0)
                  << " GB) exceeds available GPU memory. Using chunked processing.\n";
        setupChunks(N);
        
        // Allocate memory for one chunk only
        size_t chunk_size = chunks_[0].size;
        CUDA_CHECK(cudaMalloc(&d_vector_in_, chunk_size * sizeof(cuDoubleComplex)));
        CUDA_CHECK(cudaMalloc(&d_vector_out_, chunk_size * sizeof(cuDoubleComplex)));
        CUDA_CHECK(cudaMalloc(&d_temp_, chunk_size * sizeof(cuDoubleComplex)));
        
        stats_.memoryUsed = 3 * chunk_size * sizeof(cuDoubleComplex);
    } else {
        // Allocate full vectors
        CUDA_CHECK(cudaMalloc(&d_vector_in_, N * sizeof(cuDoubleComplex)));
        CUDA_CHECK(cudaMalloc(&d_vector_out_, N * sizeof(cuDoubleComplex)));
        CUDA_CHECK(cudaMalloc(&d_temp_, N * sizeof(cuDoubleComplex)));
        
        stats_.memoryUsed = 3 * N * sizeof(cuDoubleComplex);
    }
    
    // Copy interaction data to device
    copyInteractionsToDevice();
    
    gpu_memory_allocated_ = true;
    return true;
}

void GPUOperator::freeGPUMemory() {
    if (d_vector_in_) cudaFree(d_vector_in_);
    if (d_vector_out_) cudaFree(d_vector_out_);
    if (d_temp_) cudaFree(d_temp_);
    if (d_csr_row_ptr_) cudaFree(d_csr_row_ptr_);
    if (d_csr_col_ind_) cudaFree(d_csr_col_ind_);
    if (d_csr_values_) cudaFree(d_csr_values_);
    if (d_interactions_) cudaFree(d_interactions_);
    if (d_single_site_ops_) cudaFree(d_single_site_ops_);
    
    d_vector_in_ = nullptr;
    d_vector_out_ = nullptr;
    d_temp_ = nullptr;
    d_csr_row_ptr_ = nullptr;
    d_csr_col_ind_ = nullptr;
    d_csr_values_ = nullptr;
    d_interactions_ = nullptr;
    d_single_site_ops_ = nullptr;
    
    gpu_memory_allocated_ = false;
    sparse_matrix_built_ = false;
}

void GPUOperator::copyInteractionsToDevice() {
    num_interactions_ = interactions_.size();
    num_single_site_ops_ = single_site_ops_.size();
    
    if (num_interactions_ > 0) {
        CUDA_CHECK(cudaMalloc(&d_interactions_, num_interactions_ * sizeof(Interaction)));
        CUDA_CHECK(cudaMemcpy(d_interactions_, interactions_.data(),
                            num_interactions_ * sizeof(Interaction),
                            cudaMemcpyHostToDevice));
    }
    
    if (num_single_site_ops_ > 0) {
        CUDA_CHECK(cudaMalloc(&d_single_site_ops_, 
                            num_single_site_ops_ * sizeof(SingleSiteOp)));
        CUDA_CHECK(cudaMemcpy(d_single_site_ops_, single_site_ops_.data(),
                            num_single_site_ops_ * sizeof(SingleSiteOp),
                            cudaMemcpyHostToDevice));
    }
}

void GPUOperator::setupChunks(int N) {
    chunks_.clear();
    
    size_t max_chunk_size = CHUNK_SIZE;
    int num_chunks = (N + max_chunk_size - 1) / max_chunk_size;
    
    for (int i = 0; i < num_chunks; ++i) {
        ChunkInfo chunk;
        chunk.start_idx = i * max_chunk_size;
        chunk.size = std::min(static_cast<size_t>(N - chunk.start_idx), max_chunk_size);
        chunk.start_state = chunk.start_idx;
        chunk.end_state = chunk.start_state + chunk.size;
        chunks_.push_back(chunk);
    }
    
    stats_.numChunks = num_chunks;
    std::cout << "Using " << num_chunks << " chunks for processing\n";
}

void GPUOperator::matVec(const std::complex<double>* x, std::complex<double>* y, int N) {
    if (!gpu_memory_allocated_) {
        allocateGPUMemory(N);
    }
    
    // Copy input vector to device
    CUDA_CHECK(cudaMemcpy(d_vector_in_, x, N * sizeof(cuDoubleComplex),
                        cudaMemcpyHostToDevice));
    
    // Perform matrix-vector product on GPU
    matVecGPU(d_vector_in_, d_vector_out_, N);
    
    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(y, d_vector_out_, N * sizeof(cuDoubleComplex),
                        cudaMemcpyDeviceToHost));
}

void GPUOperator::matVecGPU(const cuDoubleComplex* d_x, cuDoubleComplex* d_y, int N) {
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    
    if (sparse_matrix_built_) {
        // Use cuSPARSE for sparse matrix-vector product
        cuDoubleComplex alpha = make_cuDoubleComplex(1.0, 0.0);
        cuDoubleComplex beta = make_cuDoubleComplex(0.0, 0.0);
        
        size_t buffer_size = 0;
        void* d_buffer = nullptr;
        
        CUSPARSE_CHECK(cusparseSpMV_bufferSize(
            cusparse_handle_, CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha, mat_descriptor_, vec_x_descriptor_,
            &beta, vec_y_descriptor_, CUDA_C_64F,
            CUSPARSE_SPMV_ALG_DEFAULT, &buffer_size));
        
        if (buffer_size > 0) {
            CUDA_CHECK(cudaMalloc(&d_buffer, buffer_size));
        }
        
        CUSPARSE_CHECK(cusparseSpMV(
            cusparse_handle_, CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha, mat_descriptor_, vec_x_descriptor_,
            &beta, vec_y_descriptor_, CUDA_C_64F,
            CUSPARSE_SPMV_ALG_DEFAULT, d_buffer));
        
        if (d_buffer) {
            cudaFree(d_buffer);
        }
    } else {
        // Use on-the-fly computation kernel
        int num_blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
        num_blocks = std::min(num_blocks, MAX_BLOCKS);
        
        GPUKernels::matVecKernel<<<num_blocks, BLOCK_SIZE>>>(
            d_x, d_y, N, n_sites_,
            d_interactions_, num_interactions_,
            d_single_site_ops_, num_single_site_ops_);
        
        CUDA_CHECK(cudaGetLastError());
    }
    
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    float milliseconds = 0;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    stats_.matVecTime = milliseconds / 1000.0;
    
    // Estimate throughput (rough estimate)
    double flops = static_cast<double>(N) * NNZ_PER_STATE_ESTIMATE * 8; // multiply-add per element
    stats_.throughput = flops / (stats_.matVecTime * 1e9);
    
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
}

void GPUOperator::processChunk(const ChunkInfo& chunk, const cuDoubleComplex* d_x,
                              cuDoubleComplex* d_y) {
    // Process one chunk of the matrix-vector product
    int num_blocks = (chunk.size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    GPUKernels::matVecKernel<<<num_blocks, BLOCK_SIZE>>>(
        d_x + chunk.start_idx, d_y + chunk.start_idx,
        chunk.size, n_sites_,
        d_interactions_, num_interactions_,
        d_single_site_ops_, num_single_site_ops_);
    
    CUDA_CHECK(cudaGetLastError());
}

bool GPUOperator::buildSparseMatrix(int N) {
    std::cout << "Building sparse matrix on GPU (this may take a while)...\n";
    
    // This is a placeholder - actual implementation would build CSR matrix
    // For now, we use on-the-fly computation
    sparse_matrix_built_ = false;
    
    return sparse_matrix_built_;
}

bool GPUOperator::loadCSR(int N, const std::vector<int>& row_ptr,
                         const std::vector<int>& col_ind,
                         const std::vector<std::complex<double>>& values) {
    if (row_ptr.size() != static_cast<size_t>(N + 1)) {
        std::cerr << "loadCSR: row_ptr size mismatch\n";
        return false;
    }

    size_t nnz = values.size();
    nnz_ = nnz;

    // Allocate device CSR arrays
    CUDA_CHECK(cudaMalloc(&d_csr_row_ptr_, (N + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_csr_col_ind_, nnz_ * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_csr_values_, nnz_ * sizeof(cuDoubleComplex)));

    // Copy row ptr and col ind
    CUDA_CHECK(cudaMemcpy(d_csr_row_ptr_, row_ptr.data(), (N + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_csr_col_ind_, col_ind.data(), nnz_ * sizeof(int), cudaMemcpyHostToDevice));

    // Convert values to cuDoubleComplex temporary buffer
    std::vector<cuDoubleComplex> tmp(nnz_);
    for (size_t i = 0; i < nnz_; ++i) {
        tmp[i] = make_cuDoubleComplex(values[i].real(), values[i].imag());
    }
    CUDA_CHECK(cudaMemcpy(d_csr_values_, tmp.data(), nnz_ * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));

    // Create cuSPARSE CSR matrix descriptor
    cusparseIndexType_t idxType = CUSPARSE_INDEX_32I;
    cusparseIndexBase_t idxBase = CUSPARSE_INDEX_BASE_ZERO;
    CUSPARSE_CHECK(cusparseCreateCsr(&mat_descriptor_,
                                    static_cast<int64_t>(N), static_cast<int64_t>(N), static_cast<int64_t>(nnz_),
                                    d_csr_row_ptr_, d_csr_col_ind_, d_csr_values_,
                                    idxType, idxType, idxBase, CUDA_C_64F));

    // Create dense vector descriptors (d_vector_in_/out_ must be allocated)
    CUSPARSE_CHECK(cusparseCreateDnVec(&vec_x_descriptor_, static_cast<int64_t>(N), reinterpret_cast<void*>(d_vector_in_), CUDA_C_64F));
    CUSPARSE_CHECK(cusparseCreateDnVec(&vec_y_descriptor_, static_cast<int64_t>(N), reinterpret_cast<void*>(d_vector_out_), CUDA_C_64F));

    sparse_matrix_built_ = true;
    return true;
}

#endif // WITH_CUDA
