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

    // cuSPARSE handle: shared across the assembled-CSR fast path and any
    // future BSR/COO experiments. Deferred allocation of the SpMat / DnVec
    // descriptors until we actually build the CSR (see buildCsrOnDevice).
    CUSPARSE_CHECK(cusparseCreate(&cusparse_handle_));

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
    freeCsrDeviceData();
    freeGPUMemory();

    // Clean up pre-allocated CUDA events
    if (events_initialized_) {
        cudaEventDestroy(timing_start_);
        cudaEventDestroy(timing_stop_);
    }

    if (cusparse_handle_) {
        cusparseDestroy(cusparse_handle_);
        cusparse_handle_ = nullptr;
    }
    if (cublas_handle_) {
        cublasDestroy(cublas_handle_);
    }
}

void GPUOperator::initializeCUBLAS() {
    CUBLAS_CHECK(cublasCreate(&cublas_handle_));
}

// OPTIMIZED: Direct data population methods
//
// Every mutation invalidates ALL derived caches: separated SoA, host-side
// SoA flag, the assembled CSR, and the kernel-pathway selection. Otherwise
// a stale CSR / pathway from a previous call would silently return wrong
// results once the operator is reused with new terms.
void GPUOperator::addOneBodyTerm(uint8_t op_type, uint32_t site, const std::complex<double>& coeff) {
    GPUTransformData tdata;
    tdata.op_type = op_type;
    tdata.site_index = site;
    tdata.coefficient = make_cuDoubleComplex(coeff.real(), coeff.imag());
    tdata.is_two_body = 0;
    transform_data_.push_back(tdata);
    invalidateDerivedCaches();
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
    invalidateDerivedCaches();
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
    invalidateDerivedCaches();
}

void GPUOperator::invalidateDerivedCaches() {
    // Host-side classification is now stale.
    transforms_separated_ = false;
    // Re-upload of separated SoA on next matVec is required.
    separated_on_device_ = false;
    // Re-select kernel pathway because nnz / off-diagonal ratio may change.
    selected_pathway_ = KernelPathway::UNINITIALIZED;
    // Drop the assembled CSR and any cuSPARSE descriptors that point into it.
    if (csr_assembled_) {
        freeCsrDeviceData();
    }
    // Drop cached transform_data_ device pointer (it'll get rebuilt on demand).
    if (d_transform_data_) {
        cudaFree(d_transform_data_);
        d_transform_data_ = nullptr;
        num_transforms_   = 0;
    }
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

    // Three-body terms are loaded but the GPU kernel only handles 1- and
    // 2-body parts of H. Silently dropping them would produce wrong-energy
    // results that look superficially fine — refuse to proceed unless the
    // user explicitly opts in via ED_GPU_ALLOW_DROPPED_THREEBODY=1, in
    // which case we still emit the loud warning so the situation is
    // recorded in run logs.
    if (!three_body_data_.empty()) {
        const char* opt_in = std::getenv("ED_GPU_ALLOW_DROPPED_THREEBODY");
        const bool user_acknowledged = (opt_in && opt_in[0] == '1');
        if (!user_acknowledged) {
            throw std::runtime_error(
                "GPUOperator: " + std::to_string(three_body_data_.size()) +
                " three-body terms were loaded from '" + filename +
                "', but the GPU matvec kernel does not implement them. "
                "Use a CPU solver for this Hamiltonian, or set "
                "ED_GPU_ALLOW_DROPPED_THREEBODY=1 to acknowledge that "
                "three-body interactions will be silently dropped on the GPU.");
        }
        std::cerr << "WARNING: Three-body terms loaded but GPU kernel not implemented.\n";
        std::cerr << "         ED_GPU_ALLOW_DROPPED_THREEBODY=1 is set — proceeding with\n";
        std::cerr << "         " << three_body_data_.size()
                  << " three-body term(s) IGNORED on the GPU.\n";
        std::cerr << "         Energies and observables will not match the full Hamiltonian.\n";
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

        // Reaching this point implies ED_GPU_ALLOW_DROPPED_THREEBODY=1 was
        // set (otherwise loadInterAllFile would have thrown). Re-emit a
        // compact warning so the run log makes the loss-of-physics obvious.
        std::cerr << "[GPUOperator] " << num_three_body_
                  << " three-body terms copied to device but IGNORED by the matvec kernel "
                  << "(ED_GPU_ALLOW_DROPPED_THREEBODY=1).\n";
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
    // OPTIMIZATION: Per-call event timing forces a host-side sync at the end
    // (cudaEventSynchronize) which serializes the GPU pipeline and can be
    // 30-50% of the wall time at small N. Make it opt-in via env var so the
    // hot path is fully asynchronous by default. CPU profilers like nsys /
    // ncu still expose per-kernel timings without the host sync.
    static const bool ed_gpu_timing = []{
        const char* e = std::getenv("ED_GPU_TIMING");
        return e && e[0] == '1';
    }();
    if (ed_gpu_timing) {
        CUDA_CHECK(cudaEventRecord(timing_start_));
    }

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
        // CUSPARSE_CSR doesn't need separated SoA on device — it uses its
        // own assembled CSR cache built in selectKernelPathway().
        if (selected_pathway_ != KernelPathway::SHARED_MEMORY &&
            selected_pathway_ != KernelPathway::CUSPARSE_CSR &&
            !separated_on_device_) {
            copySeparatedTransformsToDevice();
        }

        // Execute selected pathway (no branching within hot path)
        switch (selected_pathway_) {
        case KernelPathway::CUSPARSE_CSR: {
            // Single tuned cuSPARSE SpMV. cusparseSpMV writes directly into
            // d_y with beta=0, so no separate cudaMemset is needed.
            applyCusparse(d_x, d_y, N, /*stream=*/0);
            break;
        }

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
    
    // Per-call event timing only when explicitly enabled. The
    // cudaEventSynchronize stall would otherwise dominate at small N.
    // When disabled, callers that read stats_.matVecTime get 0 (the GPU
    // pipeline is fully asynchronous and per-call timing is meaningless
    // without a sync). Use ED_GPU_TIMING=1 for honest per-call numbers.
    if (ed_gpu_timing) {
        CUDA_CHECK(cudaEventRecord(timing_stop_));
        CUDA_CHECK(cudaEventSynchronize(timing_stop_));

        float milliseconds = 0;
        CUDA_CHECK(cudaEventElapsedTime(&milliseconds, timing_start_, timing_stop_));
        stats_.matVecTime = milliseconds / 1000.0;

        // Estimate throughput (rough estimate)
        double flops = static_cast<double>(N) * NNZ_PER_STATE_ESTIMATE * 8;
        stats_.throughput = flops / (stats_.matVecTime * 1e9);
    } else {
        stats_.matVecTime = 0.0;
        stats_.throughput = 0.0;
    }
}

void GPUOperator::selectKernelPathway(int N) {
    /**
     * Kernel Selection Criteria:
     *
     * 0. CUSPARSE_CSR (assembled CSR + cuSPARSE SpMV) -- DEFAULT when feasible:
     *    - Benefits: NVIDIA-tuned warp-vector SpMV, no per-call transform loop,
     *      no atomic contention, tiny per-iteration launch overhead.
     *    - Overhead: One-time host-side CSR build + HtoD copy (amortized over
     *      hundreds of Lanczos matvecs).
     *    - Use when: build succeeds (matrix fits in ~65% of free GPU memory)
     *      AND not disabled via ED_GPU_DISABLE_CUSPARSE=1.
     *    - Required for the GPU to match the CPU's assembled-CSR fast path
     *      that gives 1.3-7.8x speedup vs SciPy on N=12-18.
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

    // ---- Preferred path: cuSPARSE assembled CSR ------------------------
    // Three-body terms are not yet emitted into the CSR by buildCsrOnDevice,
    // so they would silently get dropped if we took this path. Skip cuSPARSE
    // for any operator that has them (and rely on the existing matrix-free
    // pathways, which also currently ignore three-body GPU contributions but
    // at least loudly warn). Use the host vector size, not num_three_body_,
    // because the latter is only set after copyThreeBodyDataToDevice().
    //
    // Empirically (H100, Heisenberg PBC chain), the matrix-free fused kernel
    // beats cuSPARSE for very small dimensions (<= ~16k rows) where the per-
    // call cuSPARSE launch + descriptor binding overhead dominates over the
    // raw SpMV work. The crossover is operator-dependent so we expose it via
    // ED_GPU_CUSPARSE_MIN_DIM (default 32768).
    const char* disable_env = std::getenv("ED_GPU_DISABLE_CUSPARSE");
    const char* min_dim_env = std::getenv("ED_GPU_CUSPARSE_MIN_DIM");
    const int   cusparse_min_dim =
        (min_dim_env && min_dim_env[0] != '\0') ? std::atoi(min_dim_env) : 32768;
    const bool cusparse_disabled =
        (disable_env && disable_env[0] == '1') ||
        (!three_body_data_.empty()) ||
        (N < cusparse_min_dim);

    if (!cusparse_disabled) {
        if (!transforms_separated_) separateTransformsByType();
        if (buildCsrOnDevice(N)) {
            selected_pathway_ = KernelPathway::CUSPARSE_CSR;
            std::cout << "Selected CUSPARSE_CSR pathway: T=" << num_transforms_
                      << ", N=" << N << ", nnz=" << csr_nnz_ << std::endl;
            return;
        }
        // If buildCsrOnDevice() returned false (insufficient memory or no
        // entries), fall through to matrix-free selection.
    }

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
        
        // Ensure separated transforms on device for non-SHARED_MEMORY paths.
        // CUSPARSE_CSR uses its own assembled CSR cache (no SoA needed).
        if (selected_pathway_ != KernelPathway::SHARED_MEMORY &&
            selected_pathway_ != KernelPathway::CUSPARSE_CSR &&
            !separated_on_device_) {
            copySeparatedTransformsToDevice();
        }

        switch (selected_pathway_) {
        case KernelPathway::CUSPARSE_CSR: {
            applyCusparse(d_x, d_y, N, stream);
            break;
        }

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

// ============================================================================
// cuSPARSE assembled-CSR fast path
//
// One-time host-side CSR assembly (mirrors the CPU's
// Operator::buildSparseMatrixFromData/buildRowMajorCSR), followed by an
// HtoD memcpy and cuSPARSE SpMV calls. Once built, every matVec is a single
// cusparseSpMV() launch — no atomics, no transform-loop overhead, and full
// use of NVIDIA's tuned warp-vector SpMV scheduling.
// ============================================================================

void GPUOperator::freeCsrDeviceData() {
    if (vec_x_descr_) { cusparseDestroyDnVec(vec_x_descr_); vec_x_descr_ = nullptr; }
    if (vec_y_descr_) { cusparseDestroyDnVec(vec_y_descr_); vec_y_descr_ = nullptr; }
    if (csr_descr_)   { cusparseDestroySpMat(csr_descr_);   csr_descr_   = nullptr; }
    if (cusparse_workspace_) {
        cudaFree(cusparse_workspace_);
        cusparse_workspace_ = nullptr;
        cusparse_workspace_bytes_ = 0;
    }
    if (d_csr_row_offsets_) { cudaFree(d_csr_row_offsets_); d_csr_row_offsets_ = nullptr; }
    if (d_csr_col_idx_)     { cudaFree(d_csr_col_idx_);     d_csr_col_idx_     = nullptr; }
    if (d_csr_values_)      { cudaFree(d_csr_values_);      d_csr_values_      = nullptr; }
    csr_nnz_ = 0;
    csr_dim_ = 0;
    csr_assembled_ = false;
}

bool GPUOperator::buildCsrOnDevice(int N) {
    // Caller is responsible for invalidating any stale CSR before mutating
    // transform_data_; we just check the cached flag here.
    if (csr_assembled_ && csr_dim_ == N) return true;
    freeCsrDeviceData();

    if (!transforms_separated_) separateTransformsByType();

    // ---- Memory feasibility check --------------------------------------
    // Estimate nnz pessimistically: each diagonal term contributes 1 entry per
    // row, each off-diagonal one or two-body term up to 1 entry per row. The
    // bound (terms_per_row) is exact for the operator types we support.
    // Total bytes: (N+1)*4 (row_offsets) + nnz*(4 + 16) (col_idx + value).
    //
    // NOTE: We deliberately use the host-side vector sizes (.size()) rather
    // than the num_* counters: those counters are only set by
    // copySeparatedTransformsToDevice(), which the cuSPARSE path does not
    // invoke. Using the counters here would make this function silently
    // believe the operator is empty and bail out to the matrix-free path.
    const size_t terms_per_row =
        diag_one_body_.size() + offdiag_one_body_.size() +
        diag_two_body_.size() + mixed_two_body_.size() +
        offdiag_two_body_.size() + three_body_data_.size();
    if (terms_per_row == 0) return false;

    const size_t est_nnz   = static_cast<size_t>(N) * terms_per_row;
    const size_t est_bytes = static_cast<size_t>(N + 1) * sizeof(int)
                           + est_nnz * (sizeof(int) + sizeof(cuDoubleComplex));

    size_t free_mem = 0, total_mem = 0;
    cudaMemGetInfo(&free_mem, &total_mem);
    // Reserve 35% of free memory for Lanczos vectors / cuSPARSE workspace /
    // future allocations. This is conservative but avoids OOM under typical
    // Lanczos workloads (4 working vectors + ~max_iter stored vectors).
    const size_t budget = static_cast<size_t>(free_mem * 0.65);
    if (est_bytes > budget) {
        std::cout << "GPUOperator: CSR fast path skipped (" 
                  << (est_bytes / (1024.0 * 1024.0)) << " MB > "
                  << (budget / (1024.0 * 1024.0)) << " MB budget)\n";
        return false;
    }

    // ---- Host-side CSR assembly ----------------------------------------
    // Build COO-style triplets first (row, col, val), then sort by row to
    // produce a sorted CSR. We use sequential build for now: it's a one-time
    // O(N * T) cost that's amortized over hundreds of SpMV calls.
    const float spin = spin_l_;
    const double spin_sq = static_cast<double>(spin) * static_cast<double>(spin);

    std::vector<int> rows;        rows.reserve(est_nnz);
    std::vector<int> cols;        cols.reserve(est_nnz);
    std::vector<cuDoubleComplex> vals;  vals.reserve(est_nnz);

    auto emit = [&](int r, int c, cuDoubleComplex v) {
        rows.push_back(r);
        cols.push_back(c);
        vals.push_back(v);
    };

    for (int basis = 0; basis < N; ++basis) {
        // Diagonal one-body (Sz)
        for (const auto& t : diag_one_body_) {
            const double sign = ((basis >> t.site_index) & 1) ? -1.0 : 1.0;
            cuDoubleComplex v = make_cuDoubleComplex(
                cuCreal(t.coefficient) * spin * sign,
                cuCimag(t.coefficient) * spin * sign);
            emit(basis, basis, v);
        }
        // Off-diagonal one-body (S+/S-)
        for (const auto& t : offdiag_one_body_) {
            const uint64_t bit = (basis >> t.site_index) & 1;
            if (bit != t.op_type) {
                const int new_basis = basis ^ (1 << t.site_index);
                if (new_basis < N) emit(new_basis, basis, t.coefficient);
            }
        }
        // Diagonal two-body (Sz Sz)
        for (const auto& t : diag_two_body_) {
            const double si = ((basis >> t.site_index_1) & 1) ? -1.0 : 1.0;
            const double sj = ((basis >> t.site_index_2) & 1) ? -1.0 : 1.0;
            const double s  = spin_sq * si * sj;
            cuDoubleComplex v = make_cuDoubleComplex(
                cuCreal(t.coefficient) * s,
                cuCimag(t.coefficient) * s);
            emit(basis, basis, v);
        }
        // Mixed two-body (Sz * S+/S-)
        for (const auto& t : mixed_two_body_) {
            const uint64_t flip_bit = (basis >> t.flip_site) & 1;
            if (flip_bit != t.flip_op_type) {
                const double sz_sign = ((basis >> t.sz_site) & 1) ? -1.0 : 1.0;
                const int new_basis = basis ^ (1 << t.flip_site);
                if (new_basis < N) {
                    const double s = spin * sz_sign;
                    cuDoubleComplex v = make_cuDoubleComplex(
                        cuCreal(t.coefficient) * s,
                        cuCimag(t.coefficient) * s);
                    emit(new_basis, basis, v);
                }
            }
        }
        // Off-diagonal two-body (S+/S- * S+/S-)
        for (const auto& t : offdiag_two_body_) {
            const uint64_t b1 = (basis >> t.site_index_1) & 1;
            const uint64_t b2 = (basis >> t.site_index_2) & 1;
            if (b1 != t.op_type_1 && b2 != t.op_type_2) {
                const int new_basis = basis ^ (1 << t.site_index_1) ^ (1 << t.site_index_2);
                if (new_basis < N) emit(new_basis, basis, t.coefficient);
            }
        }
    }

    if (rows.empty()) {
        std::cout << "GPUOperator: CSR fast path skipped (no entries to assemble)\n";
        return false;
    }

    // ---- COO -> CSR (sort by row, then merge duplicate (row,col) pairs) -
    // Indirect sort to avoid moving cuDoubleComplex (16-byte) values during
    // the comparison-heavy phase: build perm, then gather.
    std::vector<int> perm(rows.size());
    for (size_t i = 0; i < perm.size(); ++i) perm[i] = static_cast<int>(i);
    std::sort(perm.begin(), perm.end(), [&](int a, int b) {
        if (rows[a] != rows[b]) return rows[a] < rows[b];
        return cols[a] < cols[b];
    });

    std::vector<int> sorted_rows(rows.size());
    std::vector<int> sorted_cols(rows.size());
    std::vector<cuDoubleComplex> sorted_vals(rows.size());
    for (size_t i = 0; i < perm.size(); ++i) {
        sorted_rows[i] = rows[perm[i]];
        sorted_cols[i] = cols[perm[i]];
        sorted_vals[i] = vals[perm[i]];
    }

    // Merge consecutive entries with identical (row, col) — duplicates are
    // possible when two distinct transforms map basis -> new_basis with the
    // same coefficient slot (e.g., two equivalent Sz_i*Sz_j terms).
    std::vector<int> mrows;        mrows.reserve(sorted_rows.size());
    std::vector<int> mcols;        mcols.reserve(sorted_cols.size());
    std::vector<cuDoubleComplex> mvals;  mvals.reserve(sorted_vals.size());
    for (size_t i = 0; i < sorted_rows.size(); ++i) {
        if (!mrows.empty() && mrows.back() == sorted_rows[i] && mcols.back() == sorted_cols[i]) {
            mvals.back() = cuCadd(mvals.back(), sorted_vals[i]);
        } else {
            mrows.push_back(sorted_rows[i]);
            mcols.push_back(sorted_cols[i]);
            mvals.push_back(sorted_vals[i]);
        }
    }

    const int64_t nnz = static_cast<int64_t>(mrows.size());

    // Build row_offsets[N+1] from the (now sorted) row array.
    std::vector<int> row_offsets(N + 1, 0);
    for (int64_t i = 0; i < nnz; ++i) {
        row_offsets[mrows[i] + 1]++;
    }
    for (int r = 0; r < N; ++r) row_offsets[r + 1] += row_offsets[r];

    // ---- Upload to device ----------------------------------------------
    CUDA_CHECK(cudaMalloc(&d_csr_row_offsets_, (N + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_csr_col_idx_,     nnz * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_csr_values_,      nnz * sizeof(cuDoubleComplex)));
    CUDA_CHECK(cudaMemcpy(d_csr_row_offsets_, row_offsets.data(),
                          (N + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_csr_col_idx_, mcols.data(),
                          nnz * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_csr_values_, mvals.data(),
                          nnz * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));

    // ---- cuSPARSE descriptors + workspace ------------------------------
    CUSPARSE_CHECK(cusparseCreateCsr(&csr_descr_,
        /*rows=*/N, /*cols=*/N, nnz,
        d_csr_row_offsets_, d_csr_col_idx_, d_csr_values_,
        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_BASE_ZERO, CUDA_C_64F));

    // We create dense-vector descriptors with nullptr data pointers and
    // re-bind them to (d_x, d_y) on every applyCusparse() call via
    // cusparseDnVecSetValues. Cheaper than recreating the descriptor.
    CUSPARSE_CHECK(cusparseCreateDnVec(&vec_x_descr_, N, nullptr, CUDA_C_64F));
    CUSPARSE_CHECK(cusparseCreateDnVec(&vec_y_descr_, N, nullptr, CUDA_C_64F));

    cuDoubleComplex alpha = make_cuDoubleComplex(1.0, 0.0);
    cuDoubleComplex beta  = make_cuDoubleComplex(0.0, 0.0);

    CUSPARSE_CHECK(cusparseSpMV_bufferSize(cusparse_handle_,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, csr_descr_, vec_x_descr_, &beta, vec_y_descr_,
        CUDA_C_64F, CUSPARSE_SPMV_CSR_ALG2, &cusparse_workspace_bytes_));

    if (cusparse_workspace_bytes_ > 0) {
        CUDA_CHECK(cudaMalloc(&cusparse_workspace_, cusparse_workspace_bytes_));
    }

    csr_nnz_ = nnz;
    csr_dim_ = N;
    csr_assembled_ = true;

    std::cout << "GPUOperator: cuSPARSE CSR assembled, dim=" << N
              << ", nnz=" << nnz
              << " (" << (est_bytes / (1024.0 * 1024.0)) << " MB"
              << ", workspace=" << (cusparse_workspace_bytes_ / 1024.0) << " KB)\n";
    return true;
}

void GPUOperator::applyCusparse(const cuDoubleComplex* d_x, cuDoubleComplex* d_y,
                                int N, cudaStream_t stream) {
    // Bind I/O vectors to the cached descriptors. cusparseDnVecSetValues is
    // O(1) -- it just patches the pointer in the opaque descriptor.
    CUSPARSE_CHECK(cusparseDnVecSetValues(vec_x_descr_, const_cast<cuDoubleComplex*>(d_x)));
    CUSPARSE_CHECK(cusparseDnVecSetValues(vec_y_descr_, d_y));

    if (stream) {
        CUSPARSE_CHECK(cusparseSetStream(cusparse_handle_, stream));
    }

    cuDoubleComplex alpha = make_cuDoubleComplex(1.0, 0.0);
    cuDoubleComplex beta  = make_cuDoubleComplex(0.0, 0.0);
    CUSPARSE_CHECK(cusparseSpMV(cusparse_handle_,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, csr_descr_, vec_x_descr_, &beta, vec_y_descr_,
        CUDA_C_64F, CUSPARSE_SPMV_CSR_ALG2, cusparse_workspace_));

    if (stream) {
        // Restore the default stream so subsequent (synchronous) cuSPARSE
        // calls behave as expected.
        CUSPARSE_CHECK(cusparseSetStream(cusparse_handle_, 0));
    }
}

#endif // WITH_CUDA
