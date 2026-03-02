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
      d_vector_in_(nullptr), d_vector_out_(nullptr),
      d_transform_data_(nullptr), num_transforms_(0), 
      d_three_body_data_(nullptr), num_three_body_(0),
      gpu_memory_allocated_(false),
      events_initialized_(false) {
    
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
    freeGPUMemory();
    
    // Clean up pre-allocated CUDA events
    if (events_initialized_) {
        cudaEventDestroy(timing_start_);
        cudaEventDestroy(timing_stop_);
    }
    
    if (cublas_handle_) {
        cublasDestroy(cublas_handle_);
    }
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

void GPUOperator::setInteraction(int site1, int site2, char op1, char op2, double coupling) {
    // Map char operators to uint8_t: 0=S+, 1=S-, 2=Sz
    auto mapOp = [](char c) -> uint8_t {
        if (c == '+') return 0;
        if (c == '-') return 1;
        if (c == 'z' || c == 'Z') return 2;
        throw std::runtime_error(std::string("Invalid operator '") + c + "': must be '+', '-', or 'z'");
    };
    
    addTwoBodyTerm(mapOp(op1), site1, mapOp(op2), site2, std::complex<double>(coupling, 0.0));
}

void GPUOperator::setSingleSite(int site, char op, double coupling) {
    auto mapOp = [](char c) -> uint8_t {
        if (c == '+') return 0;
        if (c == '-') return 1;
        if (c == 'z' || c == 'Z') return 2;
        throw std::runtime_error(std::string("Invalid operator '") + c + "': must be '+', '-', or 'z'");
    };
    
    addOneBodyTerm(mapOp(op), site, std::complex<double>(coupling, 0.0));
}

size_t GPUOperator::estimateMemoryRequirement(int N) const {
    // 2 vectors (input + output) for matrix-free operation
    size_t vector_size = N * sizeof(cuDoubleComplex);
    return 2 * vector_size;
}

bool GPUOperator::allocateGPUMemory(int N) {
    if (gpu_memory_allocated_) {
        freeGPUMemory();
    }
    
    size_t required_memory = estimateMemoryRequirement(N);
    
    std::cout << "GPU Operator mode: matrix-free (transform_data)" << std::endl;
    std::cout << "Required memory: " << required_memory / (1024.0*1024.0*1024.0) << " GB" << std::endl;
    std::cout << "Available GPU memory: " << available_gpu_memory_ / (1024.0*1024.0*1024.0) << " GB" << std::endl;
    
    if (required_memory > available_gpu_memory_ * 0.9) {
        std::cerr << "Error: Required memory (" << required_memory / (1024.0*1024.0*1024.0) 
                  << " GB) exceeds available GPU memory.\n";
        return false;
    }
    
    CUDA_CHECK(cudaMalloc(&d_vector_in_, N * sizeof(cuDoubleComplex)));
    CUDA_CHECK(cudaMalloc(&d_vector_out_, N * sizeof(cuDoubleComplex)));
    
    stats_.memoryUsed = 2 * N * sizeof(cuDoubleComplex);
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
    d_transform_data_ = nullptr;
    d_three_body_data_ = nullptr;
    d_diag_one_body_ = nullptr;
    d_offdiag_one_body_ = nullptr;
    d_diag_two_body_ = nullptr;
    d_mixed_two_body_ = nullptr;
    d_offdiag_two_body_ = nullptr;
    
    gpu_memory_allocated_ = false;
    separated_on_device_ = false;
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
    
    if (!transform_data_.empty()) {
        // Copy transform data to device if not already done
        if (d_transform_data_ == nullptr) {
            copyTransformDataToDevice();
        }
        
        // Select kernel pathway once and cache it
        if (selected_pathway_ == KernelPathway::UNINITIALIZED || cached_N_ != N) {
            selectKernelPathway(N);
        }
        
        // Ensure transforms are separated and copied to device (for non-legacy paths)
        if (selected_pathway_ != KernelPathway::SHARED_MEMORY && !separated_on_device_) {
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
        // No transform data available - this shouldn't happen in normal operation
        std::cerr << "Error: GPUOperator::matVecGPU called with no transform data" << std::endl;
        CUDA_CHECK(cudaMemset(d_y, 0, N * sizeof(cuDoubleComplex)));
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
    // Async version for parallel block operations - uses same pathway selection as matVecGPU
    // Note: No event timing to avoid synchronization
    
    if (!transform_data_.empty()) {
        // Copy transform data to device if not already done
        if (d_transform_data_ == nullptr) {
            copyTransformDataToDevice();
        }
        
        // Select kernel pathway once and cache it (same as matVecGPU)
        if (selected_pathway_ == KernelPathway::UNINITIALIZED || cached_N_ != N) {
            selectKernelPathway(N);
        }
        
        // Ensure separated transforms on device for non-SHARED_MEMORY paths
        if (selected_pathway_ != KernelPathway::SHARED_MEMORY && !separated_on_device_) {
            copySeparatedTransformsToDevice();
        }
        
        switch (selected_pathway_) {
        case KernelPathway::WARP_REDUCTION: {
            GPUKernels::matVecWarpReductionFused<<<launch_config_.num_blocks, launch_config_.threads_per_block, 0, stream>>>(
                d_x, d_y,
                d_diag_one_body_, num_diag_one_body_,
                d_diag_two_body_, num_diag_two_body_,
                d_offdiag_one_body_, num_offdiag_one_body_,
                d_mixed_two_body_, num_mixed_two_body_,
                d_offdiag_two_body_, num_offdiag_two_body_,
                N, spin_l_);
            break;
        }
        
        case KernelPathway::BRANCH_FREE_SCATTER: {
            cudaMemsetAsync(d_y, 0, N * sizeof(cuDoubleComplex), stream);
            
            if (num_diag_one_body_ > 0) {
                dim3 grid((N + 15) / 16, (num_diag_one_body_ + 15) / 16);
                GPUKernels::matVecDiagonalOneBody<<<grid, launch_config_.block_2d, 0, stream>>>(
                    d_x, d_y, d_diag_one_body_, num_diag_one_body_, N, spin_l_);
            }
            if (num_offdiag_one_body_ > 0) {
                dim3 grid((N + 15) / 16, (num_offdiag_one_body_ + 15) / 16);
                GPUKernels::matVecOffDiagonalOneBody<<<grid, launch_config_.block_2d, 0, stream>>>(
                    d_x, d_y, d_offdiag_one_body_, num_offdiag_one_body_, N);
            }
            if (num_diag_two_body_ > 0) {
                dim3 grid((N + 15) / 16, (num_diag_two_body_ + 15) / 16);
                GPUKernels::matVecDiagonalTwoBody<<<grid, launch_config_.block_2d, 0, stream>>>(
                    d_x, d_y, d_diag_two_body_, num_diag_two_body_, N, spin_l_);
            }
            if (num_mixed_two_body_ > 0) {
                dim3 grid((N + 15) / 16, (num_mixed_two_body_ + 15) / 16);
                GPUKernels::matVecMixedTwoBody<<<grid, launch_config_.block_2d, 0, stream>>>(
                    d_x, d_y, d_mixed_two_body_, num_mixed_two_body_, N, spin_l_);
            }
            if (num_offdiag_two_body_ > 0) {
                dim3 grid((N + 15) / 16, (num_offdiag_two_body_ + 15) / 16);
                GPUKernels::matVecOffDiagonalTwoBody<<<grid, launch_config_.block_2d, 0, stream>>>(
                    d_x, d_y, d_offdiag_two_body_, num_offdiag_two_body_, N);
            }
            break;
        }
        
        case KernelPathway::SHARED_MEMORY: {
            cudaMemsetAsync(d_y, 0, N * sizeof(cuDoubleComplex), stream);
            GPUKernels::matVecKernelOptimized<<<launch_config_.num_blocks, launch_config_.threads_per_block, launch_config_.shared_mem_size, stream>>>(
                0, d_y, N, n_sites_, spin_l_,
                d_transform_data_, num_transforms_, d_x);
            break;
        }
        
        default:
            cudaMemsetAsync(d_y, 0, N * sizeof(cuDoubleComplex), stream);
            GPUKernels::matVecKernelOptimized<<<launch_config_.num_blocks, launch_config_.threads_per_block, launch_config_.shared_mem_size, stream>>>(
                0, d_y, N, n_sites_, spin_l_,
                d_transform_data_, num_transforms_, d_x);
            break;
        }
    } else {
        // No transform data available - this shouldn't happen in normal operation
        cudaMemsetAsync(d_y, 0, N * sizeof(cuDoubleComplex), stream);
    }
}

#endif // WITH_CUDA
