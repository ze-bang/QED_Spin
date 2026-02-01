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
      gpu_memory_allocated_(false), sparse_matrix_built_(false),
      events_initialized_(false), d_spmv_buffer_(nullptr), spmv_buffer_size_(0) {
    
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
    
    // OPTIMIZATION: Pre-allocate CUDA events for timing (avoid create/destroy per matVec)
    CUDA_CHECK(cudaEventCreate(&timing_start_));
    CUDA_CHECK(cudaEventCreate(&timing_stop_));
    events_initialized_ = true;
    
    // Initialize stats
    stats_.matVecTime = 0.0;
    stats_.memoryUsed = 0.0;
    stats_.numChunks = 0;
    stats_.throughput = 0.0;
}

GPUOperator::~GPUOperator() {
    destroyTextureObject();
    freeGPUMemory();
    
    // Clean up pre-allocated CUDA events
    if (events_initialized_) {
        cudaEventDestroy(timing_start_);
        cudaEventDestroy(timing_stop_);
    }
    
    // Clean up pre-allocated sparse buffer
    if (d_spmv_buffer_) {
        cudaFree(d_spmv_buffer_);
    }
    
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
    
    // Only estimate sparse matrix memory if we're going to build one
    // For matrix-free operators (using transform_data_), we don't need sparse storage
    bool use_matrix_free = (num_transforms_ > 0 || !transform_data_.empty());
    
    if (!use_matrix_free) {
        // Estimate non-zeros per row for sparse matrix
        int nnz_per_row = NNZ_PER_STATE_ESTIMATE;
        size_t nnz_estimate = static_cast<size_t>(N) * nnz_per_row;
        
        sparse_matrix_size = (N + 1) * sizeof(int) +  // row pointers
                            nnz_estimate * sizeof(int) +  // column indices
                            nnz_estimate * sizeof(cuDoubleComplex);  // values
    }
    
    size_t total = 3 * vector_size + sparse_matrix_size;
    
    return total;
}

bool GPUOperator::allocateGPUMemory(int N) {
    if (gpu_memory_allocated_) {
        freeGPUMemory();
    }
    
    size_t required_memory = estimateMemoryRequirement(N);
    
    // Log the operation mode
    bool use_matrix_free = (num_transforms_ > 0 || !transform_data_.empty());
    std::cout << "GPU Operator mode: " << (use_matrix_free ? "matrix-free (transform_data)" : "sparse matrix") << std::endl;
    std::cout << "Required memory: " << required_memory / (1024.0*1024.0*1024.0) << " GB" << std::endl;
    std::cout << "Available GPU memory: " << available_gpu_memory_ / (1024.0*1024.0*1024.0) << " GB" << std::endl;
    
    if (required_memory > available_gpu_memory_ * 0.9) {
        std::cout << "Warning: Required memory exceeds 90% of available GPU memory. Using chunked processing.\n";
        
        // Calculate chunk size based on available memory (leave 20% headroom)
        size_t usable_memory = static_cast<size_t>(available_gpu_memory_ * 0.8);
        size_t memory_per_element = 3 * sizeof(cuDoubleComplex);  // 3 vectors
        size_t max_chunk_by_memory = usable_memory / memory_per_element;
        size_t chunk_size = std::min(max_chunk_by_memory, static_cast<size_t>(N));
        
        // Round down to power of 2 for better performance
        size_t power_of_2_chunk = 1;
        while (power_of_2_chunk * 2 <= chunk_size) {
            power_of_2_chunk *= 2;
        }
        chunk_size = power_of_2_chunk;
        
        std::cout << "Adjusted chunk size: " << chunk_size << " states ("
                  << (chunk_size * memory_per_element) / (1024.0*1024.0*1024.0) << " GB for vectors)\n";
        
        setupChunks(N);
        
        // Override with memory-aware chunk size
        if (!chunks_.empty()) {
            chunks_[0].size = std::min(static_cast<size_t>(N), chunk_size);
        }
        
        size_t actual_chunk_size = chunks_.empty() ? chunk_size : chunks_[0].size;
        CUDA_CHECK(cudaMalloc(&d_vector_in_, actual_chunk_size * sizeof(cuDoubleComplex)));
        CUDA_CHECK(cudaMalloc(&d_vector_out_, actual_chunk_size * sizeof(cuDoubleComplex)));
        CUDA_CHECK(cudaMalloc(&d_temp_, actual_chunk_size * sizeof(cuDoubleComplex)));
        
        stats_.memoryUsed = 3 * actual_chunk_size * sizeof(cuDoubleComplex);
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

// ============================================================================
// Branch-Free Transform Separation for GPU
// ============================================================================

void GPUOperator::separateTransformsByType() {
    if (transforms_separated_) return;
    
    // Clear previous separations
    diag_one_body_.clear();
    offdiag_one_body_.clear();
    diag_two_body_.clear();
    mixed_two_body_.clear();
    offdiag_two_body_.clear();
    
    for (const auto& t : transform_data_) {
        if (t.is_two_body == 0) {
            // One-body term
            if (t.op_type == 2) {
                // Sz - diagonal
                GPUDiagonalOneBody d;
                d.site_index = t.site_index;
                d.coefficient = t.coefficient;
                diag_one_body_.push_back(d);
            } else {
                // S+ or S- - off-diagonal
                GPUOffDiagonalOneBody od;
                od.site_index = t.site_index;
                od.op_type = t.op_type;
                od.coefficient = t.coefficient;
                offdiag_one_body_.push_back(od);
            }
        } else {
            // Two-body term
            bool op1_diag = (t.op_type == 2);
            bool op2_diag = (t.op_type_2 == 2);
            
            if (op1_diag && op2_diag) {
                // Sz * Sz - fully diagonal
                GPUDiagonalTwoBody d;
                d.site_index_1 = t.site_index;
                d.site_index_2 = t.site_index_2;
                d.coefficient = t.coefficient;
                diag_two_body_.push_back(d);
            } else if (op1_diag || op2_diag) {
                // Mixed: one Sz, one S+/S-
                GPUMixedTwoBody m;
                if (op1_diag) {
                    m.sz_site = t.site_index;
                    m.flip_site = t.site_index_2;
                    m.flip_op_type = t.op_type_2;
                } else {
                    m.sz_site = t.site_index_2;
                    m.flip_site = t.site_index;
                    m.flip_op_type = t.op_type;
                }
                m.coefficient = t.coefficient;
                mixed_two_body_.push_back(m);
            } else {
                // Both S+/S- - fully off-diagonal
                GPUOffDiagonalTwoBody od;
                od.site_index_1 = t.site_index;
                od.site_index_2 = t.site_index_2;
                od.op_type_1 = t.op_type;
                od.op_type_2 = t.op_type_2;
                od.coefficient = t.coefficient;
                offdiag_two_body_.push_back(od);
            }
        }
    }
    
    transforms_separated_ = true;
    
    std::cout << "GPU transforms separated: "
              << diag_one_body_.size() << " diag-1B, "
              << offdiag_one_body_.size() << " offdiag-1B, "
              << diag_two_body_.size() << " diag-2B, "
              << mixed_two_body_.size() << " mixed-2B, "
              << offdiag_two_body_.size() << " offdiag-2B\n";
}

void GPUOperator::copySeparatedTransformsToDevice() {
    if (!transforms_separated_) {
        separateTransformsByType();
    }
    
    // Free any previously allocated device memory
    if (d_diag_one_body_) { cudaFree(d_diag_one_body_); d_diag_one_body_ = nullptr; }
    if (d_offdiag_one_body_) { cudaFree(d_offdiag_one_body_); d_offdiag_one_body_ = nullptr; }
    if (d_diag_two_body_) { cudaFree(d_diag_two_body_); d_diag_two_body_ = nullptr; }
    if (d_mixed_two_body_) { cudaFree(d_mixed_two_body_); d_mixed_two_body_ = nullptr; }
    if (d_offdiag_two_body_) { cudaFree(d_offdiag_two_body_); d_offdiag_two_body_ = nullptr; }
    
    // Copy each separated array to device
    num_diag_one_body_ = diag_one_body_.size();
    num_offdiag_one_body_ = offdiag_one_body_.size();
    num_diag_two_body_ = diag_two_body_.size();
    num_mixed_two_body_ = mixed_two_body_.size();
    num_offdiag_two_body_ = offdiag_two_body_.size();
    
    if (num_diag_one_body_ > 0) {
        CUDA_CHECK(cudaMalloc(&d_diag_one_body_, num_diag_one_body_ * sizeof(GPUDiagonalOneBody)));
        CUDA_CHECK(cudaMemcpy(d_diag_one_body_, diag_one_body_.data(),
                            num_diag_one_body_ * sizeof(GPUDiagonalOneBody),
                            cudaMemcpyHostToDevice));
    }
    
    if (num_offdiag_one_body_ > 0) {
        CUDA_CHECK(cudaMalloc(&d_offdiag_one_body_, num_offdiag_one_body_ * sizeof(GPUOffDiagonalOneBody)));
        CUDA_CHECK(cudaMemcpy(d_offdiag_one_body_, offdiag_one_body_.data(),
                            num_offdiag_one_body_ * sizeof(GPUOffDiagonalOneBody),
                            cudaMemcpyHostToDevice));
    }
    
    if (num_diag_two_body_ > 0) {
        CUDA_CHECK(cudaMalloc(&d_diag_two_body_, num_diag_two_body_ * sizeof(GPUDiagonalTwoBody)));
        CUDA_CHECK(cudaMemcpy(d_diag_two_body_, diag_two_body_.data(),
                            num_diag_two_body_ * sizeof(GPUDiagonalTwoBody),
                            cudaMemcpyHostToDevice));
    }
    
    if (num_mixed_two_body_ > 0) {
        CUDA_CHECK(cudaMalloc(&d_mixed_two_body_, num_mixed_two_body_ * sizeof(GPUMixedTwoBody)));
        CUDA_CHECK(cudaMemcpy(d_mixed_two_body_, mixed_two_body_.data(),
                            num_mixed_two_body_ * sizeof(GPUMixedTwoBody),
                            cudaMemcpyHostToDevice));
    }
    
    if (num_offdiag_two_body_ > 0) {
        CUDA_CHECK(cudaMalloc(&d_offdiag_two_body_, num_offdiag_two_body_ * sizeof(GPUOffDiagonalTwoBody)));
        CUDA_CHECK(cudaMemcpy(d_offdiag_two_body_, offdiag_two_body_.data(),
                            num_offdiag_two_body_ * sizeof(GPUOffDiagonalTwoBody),
                            cudaMemcpyHostToDevice));
    }
    
    separated_on_device_ = true;
    std::cout << "GPU separated transforms copied to device\n";
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
    
    // Free separated transform arrays
    if (d_diag_one_body_) cudaFree(d_diag_one_body_);
    if (d_offdiag_one_body_) cudaFree(d_offdiag_one_body_);
    if (d_diag_two_body_) cudaFree(d_diag_two_body_);
    if (d_mixed_two_body_) cudaFree(d_mixed_two_body_);
    if (d_offdiag_two_body_) cudaFree(d_offdiag_two_body_);
    
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
    d_diag_one_body_ = nullptr;
    d_offdiag_one_body_ = nullptr;
    d_diag_two_body_ = nullptr;
    d_mixed_two_body_ = nullptr;
    d_offdiag_two_body_ = nullptr;
    
    gpu_memory_allocated_ = false;
    sparse_matrix_built_ = false;
    separated_on_device_ = false;
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
    // OPTIMIZATION: Use pre-allocated events instead of create/destroy per call
    CUDA_CHECK(cudaEventRecord(timing_start_));
    
    if (sparse_matrix_built_) {
        // Use cuSPARSE for sparse matrix-vector product
        cuDoubleComplex alpha = make_cuDoubleComplex(1.0, 0.0);
        cuDoubleComplex beta = make_cuDoubleComplex(0.0, 0.0);
        
        // Update the dense vector descriptors with the provided pointers
        CUSPARSE_CHECK(cusparseDnVecSetValues(vec_x_descriptor_, (void*)d_x));
        CUSPARSE_CHECK(cusparseDnVecSetValues(vec_y_descriptor_, (void*)d_y));
        
        // OPTIMIZATION: Pre-allocate buffer on first use, reuse afterwards
        if (spmv_buffer_size_ == 0) {
            CUSPARSE_CHECK(cusparseSpMV_bufferSize(
                cusparse_handle_, CUSPARSE_OPERATION_NON_TRANSPOSE,
                &alpha, mat_descriptor_, vec_x_descriptor_,
                &beta, vec_y_descriptor_, CUDA_C_64F,
                CUSPARSE_SPMV_ALG_DEFAULT, &spmv_buffer_size_));
            
            if (spmv_buffer_size_ > 0) {
                CUDA_CHECK(cudaMalloc(&d_spmv_buffer_, spmv_buffer_size_));
            }
        }
        
        CUSPARSE_CHECK(cusparseSpMV(
            cusparse_handle_, CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha, mat_descriptor_, vec_x_descriptor_,
            &beta, vec_y_descriptor_, CUDA_C_64F,
            CUSPARSE_SPMV_ALG_DEFAULT, d_spmv_buffer_));
    } else if (!transform_data_.empty()) {
        // Copy transform data to device if not already done
        if (d_transform_data_ == nullptr) {
            copyTransformDataToDevice();
        }
        
        // Select kernel pathway once and cache it
        if (selected_pathway_ == KernelPathway::UNINITIALIZED || cached_N_ != N) {
            selectKernelPathway(N);
        }
        
        // Ensure transforms are separated and copied to device (for non-legacy paths)
        if (selected_pathway_ != KernelPathway::SHARED_MEMORY && 
            selected_pathway_ != KernelPathway::LEGACY && !separated_on_device_) {
            copySeparatedTransformsToDevice();
        }
        
        // Execute selected pathway (no branching within hot path)
        switch (selected_pathway_) {
        case KernelPathway::WARP_REDUCTION: {
            // V3: WARP-REDUCTION (GATHER) KERNEL - no atomics
            GPUKernels::matVecWarpReductionFused<<<launch_config_.num_blocks, launch_config_.threads_per_block>>>(
                d_x, d_y,
                d_diag_one_body_, num_diag_one_body_,
                d_diag_two_body_, num_diag_two_body_,
                d_offdiag_one_body_, num_offdiag_one_body_,
                d_mixed_two_body_, num_mixed_two_body_,
                d_offdiag_two_body_, num_offdiag_two_body_,
                N, spin_l_);
            CUDA_CHECK(cudaGetLastError());
            break;
        }
        
        case KernelPathway::BRANCH_FREE_SCATTER: {
            // V2: Branch-free separated kernels with atomics
            CUDA_CHECK(cudaMemset(d_y, 0, N * sizeof(cuDoubleComplex)));
            
            // Launch separate kernel for each transform type
            if (num_diag_one_body_ > 0) {
                dim3 grid((N + 15) / 16, (num_diag_one_body_ + 15) / 16);
                GPUKernels::matVecDiagonalOneBody<<<grid, launch_config_.block_2d>>>(
                    d_x, d_y, d_diag_one_body_, num_diag_one_body_, N, spin_l_);
            }
            if (num_offdiag_one_body_ > 0) {
                dim3 grid((N + 15) / 16, (num_offdiag_one_body_ + 15) / 16);
                GPUKernels::matVecOffDiagonalOneBody<<<grid, launch_config_.block_2d>>>(
                    d_x, d_y, d_offdiag_one_body_, num_offdiag_one_body_, N);
            }
            if (num_diag_two_body_ > 0) {
                dim3 grid((N + 15) / 16, (num_diag_two_body_ + 15) / 16);
                GPUKernels::matVecDiagonalTwoBody<<<grid, launch_config_.block_2d>>>(
                    d_x, d_y, d_diag_two_body_, num_diag_two_body_, N, spin_l_);
            }
            if (num_mixed_two_body_ > 0) {
                dim3 grid((N + 15) / 16, (num_mixed_two_body_ + 15) / 16);
                GPUKernels::matVecMixedTwoBody<<<grid, launch_config_.block_2d>>>(
                    d_x, d_y, d_mixed_two_body_, num_mixed_two_body_, N, spin_l_);
            }
            if (num_offdiag_two_body_ > 0) {
                dim3 grid((N + 15) / 16, (num_offdiag_two_body_ + 15) / 16);
                GPUKernels::matVecOffDiagonalTwoBody<<<grid, launch_config_.block_2d>>>(
                    d_x, d_y, d_offdiag_two_body_, num_offdiag_two_body_, N);
            }
            CUDA_CHECK(cudaGetLastError());
            break;
        }
        
        case KernelPathway::SHARED_MEMORY: {
            // V1: Shared memory kernel
            CUDA_CHECK(cudaMemset(d_y, 0, N * sizeof(cuDoubleComplex)));
            GPUKernels::matVecKernelOptimized<<<launch_config_.num_blocks, launch_config_.threads_per_block, launch_config_.shared_mem_size>>>(
                0, d_y, N, n_sites_, spin_l_,
                d_transform_data_, num_transforms_, d_x);
            CUDA_CHECK(cudaGetLastError());
            break;
        }
        
        default:
            // Should not reach here if selectKernelPathway was called
            CUDA_CHECK(cudaMemset(d_y, 0, N * sizeof(cuDoubleComplex)));
            GPUKernels::matVecKernelOptimized<<<launch_config_.num_blocks, launch_config_.threads_per_block, launch_config_.shared_mem_size>>>(
                0, d_y, N, n_sites_, spin_l_,
                d_transform_data_, num_transforms_, d_x);
            CUDA_CHECK(cudaGetLastError());
            break;
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
    
    // OPTIMIZATION: Use pre-allocated events
    CUDA_CHECK(cudaEventRecord(timing_stop_));
    CUDA_CHECK(cudaEventSynchronize(timing_stop_));
    
    float milliseconds = 0;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, timing_start_, timing_stop_));
    stats_.matVecTime = milliseconds / 1000.0;
    
    // Estimate throughput (rough estimate)
    double flops = static_cast<double>(N) * NNZ_PER_STATE_ESTIMATE * 8; // multiply-add per element
    stats_.throughput = flops / (stats_.matVecTime * 1e9);
}

void GPUOperator::selectKernelPathway(int N) {
    /**
     * Kernel Selection Criteria:
     * 
     * 1. WARP_REDUCTION (gather pattern):
     *    - Benefits: Zero atomic contention, direct memory writes
     *    - Overhead: Must compute inverse transforms, warp shuffle reductions
     *    - Use when: T >= 1024 AND N >= 8192
     *    - Reason: Warp overhead only worth it with massive atomic contention
     * 
     * 2. BRANCH_FREE_SCATTER (scatter pattern):
     *    - Benefits: No warp divergence, parallel over states × transforms
     *    - Overhead: Atomics for off-diagonal terms
     *    - Use when: T >= 64 (enough to saturate warps)
     * 
     * 3. SHARED_MEMORY (legacy optimized):
     *    - Benefits: Coalesced access, shared memory caching of transforms
     *    - Overhead: Warp divergence for mixed transform types
     *    - Use when: T < 64 (warp divergence less costly)
     */
    
    cached_N_ = N;
    
    // Thresholds tuned from empirical testing
    constexpr int WARP_REDUCTION_T_THRESHOLD = 1024;  // High T needed for atomic contention
    constexpr int WARP_REDUCTION_N_THRESHOLD = 8192;  // Enough warps to amortize overhead
    constexpr int BRANCH_FREE_THRESHOLD = 64;
    
    // Calculate off-diagonal ratio (indicator of atomic contention severity)
    int total_transforms = num_transforms_;
    int offdiag_count = num_offdiag_one_body_ + num_offdiag_two_body_ + num_mixed_two_body_;
    float offdiag_ratio = (total_transforms > 0) ? 
        static_cast<float>(offdiag_count) / total_transforms : 0.0f;
    
    // Selection logic
    if (total_transforms >= WARP_REDUCTION_T_THRESHOLD && 
        N >= WARP_REDUCTION_N_THRESHOLD &&
        offdiag_ratio > 0.3f) {
        // Heavy atomic contention expected - use gather pattern
        selected_pathway_ = KernelPathway::WARP_REDUCTION;
        
        // Cache launch config for warp reduction
        constexpr int WARPS_PER_BLOCK = 8;
        launch_config_.threads_per_block = WARPS_PER_BLOCK * 32;
        launch_config_.num_blocks = (N + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
        launch_config_.shared_mem_size = 0;
        launch_config_.block_2d = dim3(16, 16);
        
        std::cout << "Selected WARP_REDUCTION pathway: T=" << total_transforms 
                  << ", N=" << N << ", offdiag_ratio=" << offdiag_ratio << std::endl;
                  
    } else if (total_transforms >= BRANCH_FREE_THRESHOLD) {
        // Moderate T - use branch-free scatter kernels
        selected_pathway_ = KernelPathway::BRANCH_FREE_SCATTER;
        
        // Cache launch config for branch-free scatter
        launch_config_.threads_per_block = BLOCK_SIZE;
        launch_config_.num_blocks = std::min((N + BLOCK_SIZE - 1) / BLOCK_SIZE, MAX_BLOCKS);
        launch_config_.shared_mem_size = 0;
        launch_config_.block_2d = dim3(16, 16);
        
        std::cout << "Selected BRANCH_FREE_SCATTER pathway: T=" << total_transforms 
                  << ", N=" << N << std::endl;
                  
    } else {
        // Small T - use shared memory kernel
        selected_pathway_ = KernelPathway::SHARED_MEMORY;
        
        // Cache launch config for shared memory
        launch_config_.threads_per_block = BLOCK_SIZE;
        launch_config_.num_blocks = std::min((N + BLOCK_SIZE - 1) / BLOCK_SIZE, MAX_BLOCKS);
        launch_config_.shared_mem_size = std::min(total_transforms, 4096) * 
                                          static_cast<int>(sizeof(GPUTransformData));
        launch_config_.block_2d = dim3(16, 16);
        
        std::cout << "Selected SHARED_MEMORY pathway: T=" << total_transforms 
                  << ", N=" << N << std::endl;
    }
}

void GPUOperator::matVecGPUAsync(const cuDoubleComplex* d_x, cuDoubleComplex* d_y, int N, cudaStream_t stream) {
    // Async version for parallel block operations
    // Note: No event timing to avoid synchronization
    
    if (sparse_matrix_built_) {
        // Need a separate cuSPARSE handle for stream to avoid race conditions
        // For now, fall back to sequential for sparse matrices
        cusparseSetStream(cusparse_handle_, stream);
        
        cuDoubleComplex alpha = make_cuDoubleComplex(1.0, 0.0);
        cuDoubleComplex beta = make_cuDoubleComplex(0.0, 0.0);
        
        // Create temporary vector descriptors for this stream
        cusparseDnVecDescr_t vec_x_desc, vec_y_desc;
        cusparseCreateDnVec(&vec_x_desc, N, (void*)d_x, CUDA_C_64F);
        cusparseCreateDnVec(&vec_y_desc, N, (void*)d_y, CUDA_C_64F);
        
        size_t buffer_size = 0;
        void* d_buffer = nullptr;
        
        cusparseSpMV_bufferSize(
            cusparse_handle_, CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha, mat_descriptor_, vec_x_desc,
            &beta, vec_y_desc, CUDA_C_64F,
            CUSPARSE_SPMV_ALG_DEFAULT, &buffer_size);
        
        if (buffer_size > 0) {
            cudaMallocAsync(&d_buffer, buffer_size, stream);
        }
        
        cusparseSpMV(
            cusparse_handle_, CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha, mat_descriptor_, vec_x_desc,
            &beta, vec_y_desc, CUDA_C_64F,
            CUSPARSE_SPMV_ALG_DEFAULT, d_buffer);
        
        if (d_buffer) {
            cudaFreeAsync(d_buffer, stream);
        }
        
        cusparseDestroyDnVec(vec_x_desc);
        cusparseDestroyDnVec(vec_y_desc);
        
        // Reset stream to default
        cusparseSetStream(cusparse_handle_, 0);
    } else if (!transform_data_.empty()) {
        // Copy transform data to device if not already done
        if (d_transform_data_ == nullptr) {
            copyTransformDataToDevice();
        }
        
        // V2 OPTIMIZATION: Use branch-free kernels for larger transform counts
        const int BRANCH_FREE_THRESHOLD = 128;
        const bool USE_BRANCH_FREE = (num_transforms_ >= BRANCH_FREE_THRESHOLD);
        
        if (USE_BRANCH_FREE) {
            if (!separated_on_device_) {
                copySeparatedTransformsToDevice();
            }
            
            cudaMemsetAsync(d_y, 0, N * sizeof(cuDoubleComplex), stream);
            
            dim3 block(16, 16);
            
            // Launch separate kernels for each transform type
            if (num_diag_one_body_ > 0) {
                dim3 grid((N + block.x - 1) / block.x,
                         (num_diag_one_body_ + block.y - 1) / block.y);
                GPUKernels::matVecDiagonalOneBody<<<grid, block, 0, stream>>>(
                    d_x, d_y, d_diag_one_body_, num_diag_one_body_, N, spin_l_);
            }
            
            if (num_offdiag_one_body_ > 0) {
                dim3 grid((N + block.x - 1) / block.x,
                         (num_offdiag_one_body_ + block.y - 1) / block.y);
                GPUKernels::matVecOffDiagonalOneBody<<<grid, block, 0, stream>>>(
                    d_x, d_y, d_offdiag_one_body_, num_offdiag_one_body_, N);
            }
            
            if (num_diag_two_body_ > 0) {
                dim3 grid((N + block.x - 1) / block.x,
                         (num_diag_two_body_ + block.y - 1) / block.y);
                GPUKernels::matVecDiagonalTwoBody<<<grid, block, 0, stream>>>(
                    d_x, d_y, d_diag_two_body_, num_diag_two_body_, N, spin_l_);
            }
            
            if (num_mixed_two_body_ > 0) {
                dim3 grid((N + block.x - 1) / block.x,
                         (num_mixed_two_body_ + block.y - 1) / block.y);
                GPUKernels::matVecMixedTwoBody<<<grid, block, 0, stream>>>(
                    d_x, d_y, d_mixed_two_body_, num_mixed_two_body_, N, spin_l_);
            }
            
            if (num_offdiag_two_body_ > 0) {
                dim3 grid((N + block.x - 1) / block.x,
                         (num_offdiag_two_body_ + block.y - 1) / block.y);
                GPUKernels::matVecOffDiagonalTwoBody<<<grid, block, 0, stream>>>(
                    d_x, d_y, d_offdiag_two_body_, num_offdiag_two_body_, N);
            }
        } else {
            const int TRANSFORM_PARALLEL_THRESHOLD = 64;
            
            if (num_transforms_ > TRANSFORM_PARALLEL_THRESHOLD) {
                // Zero output vector with async memset
                cudaMemsetAsync(d_y, 0, N * sizeof(cuDoubleComplex), stream);
                
                dim3 block(16, 16);
                dim3 grid((N + block.x - 1) / block.x,
                         (num_transforms_ + block.y - 1) / block.y);
                
                GPUKernels::matVecTransformParallel<<<grid, block, 0, stream>>>(
                    d_x, d_y, d_transform_data_, num_transforms_, N, n_sites_, spin_l_);
            } else {
                cudaMemsetAsync(d_y, 0, N * sizeof(cuDoubleComplex), stream);
                
                int num_blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
                num_blocks = std::min(num_blocks, MAX_BLOCKS);
                
                size_t shared_mem_size = std::min(num_transforms_, 4096) * sizeof(GPUTransformData);
                
                GPUKernels::matVecKernelOptimized<<<num_blocks, BLOCK_SIZE, shared_mem_size, stream>>>(
                    0, d_y, N, n_sites_, spin_l_,
                    d_transform_data_, num_transforms_, d_x);
            }
        }
    } else {
        // Fallback to legacy kernel
        int num_blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
        num_blocks = std::min(num_blocks, MAX_BLOCKS);
        
        GPUKernels::matVecKernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(
            d_x, d_y, N, n_sites_,
            d_interactions_, num_interactions_,
            d_single_site_ops_, num_single_site_ops_);
    }
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
