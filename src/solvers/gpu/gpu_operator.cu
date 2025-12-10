#ifdef WITH_CUDA

// Prevent inclusion of CPU Operator class that has CUDA-incompatible code
#define CONSTRUCT_HAM_H  

#include <ed/gpu/gpu_operator.cuh>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
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
      d_transform_data_(nullptr), num_transforms_(0), 
      d_three_body_data_(nullptr), num_three_body_(0),
      tex_input_vector_(0),
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
    destroyTextureObject();
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

// OPTIMIZED: Direct data population methods
void GPUOperator::addOneBodyTerm(uint8_t op_type, uint32_t site, const std::complex<double>& coeff) {
    GPUTransformData tdata;
    tdata.op_type = op_type;
    tdata.site_index = site;
    tdata.coefficient = make_cuDoubleComplex(coeff.real(), coeff.imag());
    tdata.is_two_body = 0;
    transform_data_.push_back(tdata);
}

void GPUOperator::addTwoBodyTerm(uint8_t op1, uint32_t site1, uint8_t op2, uint32_t site2,
                                const std::complex<double>& coeff) {
    GPUTransformData tdata;
    tdata.op_type = op1;
    tdata.site_index = site1;
    tdata.op_type_2 = op2;
    tdata.site_index_2 = site2;
    tdata.coefficient = make_cuDoubleComplex(coeff.real(), coeff.imag());
    tdata.is_two_body = 1;
    transform_data_.push_back(tdata);
}

void GPUOperator::addThreeBodyTerm(uint8_t op1, uint32_t site1, uint8_t op2, uint32_t site2,
                                  uint8_t op3, uint32_t site3, const std::complex<double>& coeff) {
    GPUThreeBodyTransformData tdata;
    tdata.op_type_1 = op1;
    tdata.site_index_1 = site1;
    tdata.op_type_2 = op2;
    tdata.site_index_2 = site2;
    tdata.op_type_3 = op3;
    tdata.site_index_3 = site3;
    tdata.coefficient = make_cuDoubleComplex(coeff.real(), coeff.imag());
    three_body_data_.push_back(tdata);
}

void GPUOperator::loadThreeBodyFile(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open three-body file: " + filename);
    }
    
    std::string line;
    std::getline(file, line);  // "==================="
    std::getline(file, line);  // "num       352"
    std::istringstream iss(line);
    std::string label;
    int numLines;
    iss >> label >> numLines;
    
    // Skip separator lines
    for (int i = 0; i < 3; ++i) std::getline(file, line);
    
    int lineCount = 0;
    while (std::getline(file, line) && lineCount < numLines) {
        std::istringstream lineStream(line);
        int op_type_1, site_1, op_type_2, op_type_3, op_type_4, site_2;
        double real_part, imag_part;
        
        if (!(lineStream >> op_type_1 >> site_1 >> op_type_2 >> op_type_3 
                        >> op_type_4 >> site_2 >> real_part >> imag_part)) {
            continue;
        }
        
        std::complex<double> coeff(real_part, imag_part);
        if (std::abs(coeff) < 1e-15) continue;
        
        addThreeBodyTerm(static_cast<uint8_t>(op_type_1), site_1,
                        static_cast<uint8_t>(op_type_2), static_cast<uint32_t>(op_type_3),
                        static_cast<uint8_t>(op_type_4), site_2, coeff);
        
        lineCount++;
    }
    
    std::cout << "GPU: Loaded " << three_body_data_.size() << " three-body terms from " 
              << filename << std::endl;
    
    // Warn user that 3-body terms are not yet GPU-accelerated
    if (!three_body_data_.empty()) {
        std::cerr << "WARNING: Three-body terms loaded but GPU kernel not yet implemented.\n";
        std::cerr << "         These terms will be IGNORED in GPU calculations.\n";
        std::cerr << "         Consider using CPU solvers for Hamiltonians with 3-body interactions.\n";
    }
}

void GPUOperator::copyThreeBodyDataToDevice() {
    num_three_body_ = three_body_data_.size();
    
    if (num_three_body_ > 0) {
        CUDA_CHECK(cudaMalloc(&d_three_body_data_, num_three_body_ * sizeof(GPUThreeBodyTransformData)));
        CUDA_CHECK(cudaMemcpy(d_three_body_data_, three_body_data_.data(),
                            num_three_body_ * sizeof(GPUThreeBodyTransformData),
                            cudaMemcpyHostToDevice));
        
        std::cout << "GPU: Copied " << num_three_body_ << " three-body operations to device\n";
        
        // WARNING: Three-body terms are stored but NOT yet supported in GPU kernels
        std::cerr << "\n";
        std::cerr << "╔══════════════════════════════════════════════════════════════════════════╗\n";
        std::cerr << "║  WARNING: THREE-BODY TERMS NOT YET IMPLEMENTED IN GPU KERNELS           ║\n";
        std::cerr << "╠══════════════════════════════════════════════════════════════════════════╣\n";
        std::cerr << "║  " << num_three_body_ << " three-body terms loaded but will be IGNORED during GPU computation.  ║\n";
        std::cerr << "║  For Hamiltonians with three-body interactions, use CPU solvers instead.║\n";
        std::cerr << "║  GPU three-body kernel implementation is planned for a future release.  ║\n";
        std::cerr << "╚══════════════════════════════════════════════════════════════════════════╝\n";
        std::cerr << "\n";
    }
}

// Legacy methods (kept for compatibility with char-based interface)
// FIXED: Now also populates transform_data_ for optimized kernels
void GPUOperator::setInteraction(int site1, int site2, char op1, char op2, double coupling) {
    interactions_.push_back({site1, site2, op1, op2, coupling});
    
    // Map char operators to uint8_t: 0=S+, 1=S-, 2=Sz
    // Kernel uses: 0=S+ (raises spin), 1=S- (lowers spin), 2=Sz (diagonal)
    auto mapOp = [](char c) -> uint8_t {
        if (c == '+') return 0;  // S+ (raising operator)
        if (c == '-') return 1;  // S- (lowering operator)
        if (c == 'z' || c == 'Z') return 2;  // Sz (diagonal)
        throw std::runtime_error(std::string("Invalid operator '") + c + "': must be '+', '-', or 'z'");
    };
    
    addTwoBodyTerm(mapOp(op1), site1, mapOp(op2), site2, std::complex<double>(coupling, 0.0));
}

void GPUOperator::setSingleSite(int site, char op, double coupling) {
    single_site_ops_.push_back({site, op, coupling});
    
    // Map char operators to uint8_t: 0=S+, 1=S-, 2=Sz
    // Kernel uses: 0=S+ (raises spin), 1=S- (lowers spin), 2=Sz (diagonal)
    auto mapOp = [](char c) -> uint8_t {
        if (c == '+') return 0;  // S+ (raising operator)
        if (c == '-') return 1;  // S- (lowering operator)
        if (c == 'z' || c == 'Z') return 2;  // Sz (diagonal)
        throw std::runtime_error(std::string("Invalid operator '") + c + "': must be '+', '-', or 'z'");
    };
    
    addOneBodyTerm(mapOp(op), site, std::complex<double>(coupling, 0.0));
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
    if (d_transform_data_) cudaFree(d_transform_data_);
    if (d_three_body_data_) cudaFree(d_three_body_data_);
    
    d_vector_in_ = nullptr;
    d_vector_out_ = nullptr;
    d_temp_ = nullptr;
    d_csr_row_ptr_ = nullptr;
    d_csr_col_ind_ = nullptr;
    d_csr_values_ = nullptr;
    d_interactions_ = nullptr;
    d_single_site_ops_ = nullptr;
    d_transform_data_ = nullptr;
    d_three_body_data_ = nullptr;
    
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

void GPUOperator::copyTransformDataToDevice() {
    num_transforms_ = transform_data_.size();
    
    if (num_transforms_ > 0) {
        CUDA_CHECK(cudaMalloc(&d_transform_data_, num_transforms_ * sizeof(GPUTransformData)));
        CUDA_CHECK(cudaMemcpy(d_transform_data_, transform_data_.data(),
                            num_transforms_ * sizeof(GPUTransformData),
                            cudaMemcpyHostToDevice));
        
        std::cout << "Copied " << num_transforms_ << " transform operations to GPU\n";
    }
}

void GPUOperator::createTextureObject(cuDoubleComplex* d_data, int size) {
    if (tex_input_vector_ != 0) {
        destroyTextureObject();
    }
    
    // Create resource descriptor
    cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeLinear;
    resDesc.res.linear.devPtr = d_data;
    resDesc.res.linear.desc = cudaCreateChannelDesc<double2>();
    resDesc.res.linear.sizeInBytes = size * sizeof(cuDoubleComplex);
    
    // Create texture descriptor
    cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.readMode = cudaReadModeElementType;
    
    // Create texture object
    CUDA_CHECK(cudaCreateTextureObject(&tex_input_vector_, &resDesc, &texDesc, NULL));
}

void GPUOperator::destroyTextureObject() {
    if (tex_input_vector_ != 0) {
        cudaDestroyTextureObject(tex_input_vector_);
        tex_input_vector_ = 0;
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
        
        // Update the dense vector descriptors with the provided pointers
        CUSPARSE_CHECK(cusparseDnVecSetValues(vec_x_descriptor_, (void*)d_x));
        CUSPARSE_CHECK(cusparseDnVecSetValues(vec_y_descriptor_, (void*)d_y));
        
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
    } else if (!transform_data_.empty()) {
        // Copy transform data to device if not already done
        if (d_transform_data_ == nullptr) {
            copyTransformDataToDevice();
        }
        
        // Auto-select kernel based on parallelism potential
        // Transform-parallel benefits from high T (more parallel work)
        // Threshold: Use transform-parallel when T > 64 for maximum GPU utilization
        const int TRANSFORM_PARALLEL_THRESHOLD = 64;
        
        if (num_transforms_ > TRANSFORM_PARALLEL_THRESHOLD) {
            // GPU-NATIVE: Transform-parallel kernel (2D parallelism)
            // Zero output vector (required for atomic accumulation)
            CUDA_CHECK(cudaMemset(d_y, 0, N * sizeof(cuDoubleComplex)));
            
            // 2D grid: (N/16, T/16) with 16×16 blocks
            dim3 block(16, 16);  // 256 threads per block
            dim3 grid((N + block.x - 1) / block.x,
                     (num_transforms_ + block.y - 1) / block.y);
            
            GPUKernels::matVecTransformParallel<<<grid, block>>>(
                d_x, d_y, d_transform_data_, num_transforms_, N, n_sites_, spin_l_);
            
            CUDA_CHECK(cudaGetLastError());
        } else {
            // State-parallel kernel (better for small T)
            int num_blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
            num_blocks = std::min(num_blocks, MAX_BLOCKS);
            
            // Calculate shared memory size
            size_t shared_mem_size = std::min(num_transforms_, 4096) * sizeof(GPUTransformData);
            
            GPUKernels::matVecKernelOptimized<<<num_blocks, BLOCK_SIZE, shared_mem_size>>>(
                0, d_y, N, n_sites_, spin_l_,
                d_transform_data_, num_transforms_, d_x);
            
            CUDA_CHECK(cudaGetLastError());
        }
    } else {
        // Fallback to legacy kernel
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
