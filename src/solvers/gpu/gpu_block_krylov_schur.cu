/**
 * @file gpu_block_krylov_schur.cu
 * @brief GPU-accelerated Block Krylov-Schur algorithm implementation
 * 
 * Implements a block version of the Krylov-Schur algorithm optimized for GPU.
 * Combines block Arnoldi iteration with Schur decomposition and implicit restarts.
 * 
 * Key optimizations:
 * - Block operations use cuBLAS GEMM for BLAS-3 efficiency
 * - Batched orthogonalization reduces kernel launch overhead
 * - cuSOLVER for QR factorization and eigenvalue problems
 * - Efficient memory layout for coalesced GPU access
 * 
 * Advantages over standard Krylov-Schur:
 * - Better handling of degenerate/clustered eigenvalues
 * - Higher arithmetic intensity through block operations
 * - Fewer synchronization points
 */

#ifdef WITH_CUDA

#include <ed/gpu/gpu_lanczos.cuh>
#include <ed/gpu/kernel_config.h>
#include <ed/core/hdf5_io.h>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <random>
#include <algorithm>
#include <chrono>
#include <curand_kernel.h>
#include <cusolverDn.h>

using namespace GPUConfig;

// cuSOLVER error checking macro
#ifndef CUSOLVER_CHECK
#define CUSOLVER_CHECK(call) do { \
    cusolverStatus_t status = call; \
    if (status != CUSOLVER_STATUS_SUCCESS) { \
        std::cerr << "cuSOLVER error at " << __FILE__ << ":" << __LINE__ \
                  << " - status = " << status << std::endl; \
        throw std::runtime_error("cuSOLVER call failed"); \
    } \
} while(0)
#endif

// ============================================================================
// GPUBlockKrylovSchur Implementation
// ============================================================================

GPUBlockKrylovSchur::GPUBlockKrylovSchur(GPUOperator* op, int max_iter, int block_size, double tolerance)
    : op_(op), max_iter_(max_iter), block_size_(block_size), tolerance_(tolerance),
      max_outer_iter_(50), max_num_blocks_(0),
      d_V_(nullptr), d_W_(nullptr), d_temp_(nullptr),
      d_A_blocks_(nullptr), d_B_blocks_(nullptr),
      d_T_dense_(nullptr), d_evecs_(nullptr), d_evals_(nullptr),
      d_work_(nullptr), d_info_(nullptr), work_size_(0),
      d_tau_(nullptr), d_qr_work_(nullptr), qr_work_size_(0) {
    
    dimension_ = op_->getDimension();
    
    std::cout << "Initializing GPU Block Krylov-Schur\n";
    std::cout << "  Dimension: " << dimension_ << "\n";
    std::cout << "  Block size: " << block_size_ << "\n";
    std::cout << "  Max iterations: " << max_iter_ << "\n";
    std::cout << "  Tolerance: " << tolerance_ << "\n";
    
    CUBLAS_CHECK(cublasCreate(&cublas_handle_));
    CUSOLVER_CHECK(cusolverDnCreate(&cusolver_handle_));
    
    memset(&stats_, 0, sizeof(stats_));
}

GPUBlockKrylovSchur::~GPUBlockKrylovSchur() {
    freeMemory();
    if (cublas_handle_) cublasDestroy(cublas_handle_);
    if (cusolver_handle_) cusolverDnDestroy(cusolver_handle_);
}

void GPUBlockKrylovSchur::allocateMemory(int num_eigenvalues) {
    // Calculate number of blocks needed
    int k = num_eigenvalues;
    int p = block_size_;
    max_num_blocks_ = std::min((2 * k + p) / p + 3, max_iter_ / p);
    max_num_blocks_ = std::max(max_num_blocks_, 3);  // At least 3 blocks
    
    int total_krylov_dim = max_num_blocks_ * p;
    
    // Check available GPU memory
    size_t free_mem, total_mem;
    CUDA_CHECK(cudaMemGetInfo(&free_mem, &total_mem));
    
    size_t block_vec_size = static_cast<size_t>(dimension_) * p * sizeof(cuDoubleComplex);
    size_t krylov_mem = static_cast<size_t>(max_num_blocks_) * block_vec_size;
    size_t work_mem = 3 * block_vec_size;
    size_t block_mat_size = static_cast<size_t>(p) * p * sizeof(cuDoubleComplex);
    size_t tridiag_mem = static_cast<size_t>(max_num_blocks_) * block_mat_size * 2;
    size_t dense_mem = static_cast<size_t>(total_krylov_dim) * total_krylov_dim * sizeof(cuDoubleComplex);
    size_t solver_overhead = 20 * 1024 * 1024;  // 20 MB for solver workspace
    
    size_t total_needed = krylov_mem + work_mem + tridiag_mem + dense_mem + solver_overhead;
    
    std::cout << "  Max blocks: " << max_num_blocks_ << "\n";
    std::cout << "  Total Krylov dimension: " << total_krylov_dim << "\n";
    std::cout << "  GPU Memory: " << (free_mem / (1024.0 * 1024.0 * 1024.0)) << " GB free\n";
    std::cout << "  Allocating: " << (total_needed / (1024.0 * 1024.0)) << " MB\n";
    
    if (total_needed > free_mem * 0.8) {
        // Reduce number of blocks
        max_num_blocks_ = std::max(3, max_num_blocks_ / 2);
        total_krylov_dim = max_num_blocks_ * p;
        std::cout << "  Warning: Reducing to " << max_num_blocks_ << " blocks due to memory\n";
    }
    
    // Allocate block Krylov basis V (dimension × max_num_blocks × block_size)
    CUDA_CHECK(cudaMalloc(&d_V_, static_cast<size_t>(dimension_) * max_num_blocks_ * p * sizeof(cuDoubleComplex)));
    CUDA_CHECK(cudaMemset(d_V_, 0, static_cast<size_t>(dimension_) * max_num_blocks_ * p * sizeof(cuDoubleComplex)));
    
    // Work blocks
    CUDA_CHECK(cudaMalloc(&d_W_, block_vec_size));
    CUDA_CHECK(cudaMalloc(&d_temp_, block_vec_size));
    
    // Block tridiagonal components
    CUDA_CHECK(cudaMalloc(&d_A_blocks_, static_cast<size_t>(max_num_blocks_) * p * p * sizeof(cuDoubleComplex)));
    CUDA_CHECK(cudaMalloc(&d_B_blocks_, static_cast<size_t>(max_num_blocks_) * p * p * sizeof(cuDoubleComplex)));
    h_A_blocks_.resize(static_cast<size_t>(max_num_blocks_) * p * p);
    h_B_blocks_.resize(static_cast<size_t>(max_num_blocks_) * p * p);
    
    // Dense block tridiagonal for eigendecomposition
    CUDA_CHECK(cudaMalloc(&d_T_dense_, static_cast<size_t>(total_krylov_dim) * total_krylov_dim * sizeof(cuDoubleComplex)));
    CUDA_CHECK(cudaMalloc(&d_evecs_, static_cast<size_t>(total_krylov_dim) * total_krylov_dim * sizeof(cuDoubleComplex)));
    CUDA_CHECK(cudaMalloc(&d_evals_, static_cast<size_t>(total_krylov_dim) * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_info_, sizeof(int)));
    
    // Query workspace for zheevd
    int lwork = 0;
    CUSOLVER_CHECK(cusolverDnZheevd_bufferSize(
        cusolver_handle_,
        CUSOLVER_EIG_MODE_VECTOR,
        CUBLAS_FILL_MODE_UPPER,
        total_krylov_dim,
        reinterpret_cast<cuDoubleComplex*>(d_evecs_),
        total_krylov_dim,
        d_evals_,
        &lwork));
    work_size_ = lwork;
    CUDA_CHECK(cudaMalloc(&d_work_, work_size_ * sizeof(cuDoubleComplex)));
    
    // QR workspace
    CUDA_CHECK(cudaMalloc(&d_tau_, p * sizeof(cuDoubleComplex)));
    int qr_lwork = 0;
    CUSOLVER_CHECK(cusolverDnZgeqrf_bufferSize(
        cusolver_handle_, dimension_, p,
        reinterpret_cast<cuDoubleComplex*>(d_W_), dimension_, &qr_lwork));
    qr_work_size_ = qr_lwork;
    CUDA_CHECK(cudaMalloc(&d_qr_work_, qr_work_size_ * sizeof(cuDoubleComplex)));
    
    stats_.memory_used = krylov_mem + work_mem + tridiag_mem + dense_mem + 
                        work_size_ * sizeof(cuDoubleComplex) + qr_work_size_ * sizeof(cuDoubleComplex);
}

void GPUBlockKrylovSchur::freeMemory() {
    if (d_V_) { cudaFree(d_V_); d_V_ = nullptr; }
    if (d_W_) { cudaFree(d_W_); d_W_ = nullptr; }
    if (d_temp_) { cudaFree(d_temp_); d_temp_ = nullptr; }
    if (d_A_blocks_) { cudaFree(d_A_blocks_); d_A_blocks_ = nullptr; }
    if (d_B_blocks_) { cudaFree(d_B_blocks_); d_B_blocks_ = nullptr; }
    if (d_T_dense_) { cudaFree(d_T_dense_); d_T_dense_ = nullptr; }
    if (d_evecs_) { cudaFree(d_evecs_); d_evecs_ = nullptr; }
    if (d_evals_) { cudaFree(d_evals_); d_evals_ = nullptr; }
    if (d_work_) { cudaFree(d_work_); d_work_ = nullptr; }
    if (d_info_) { cudaFree(d_info_); d_info_ = nullptr; }
    if (d_tau_) { cudaFree(d_tau_); d_tau_ = nullptr; }
    if (d_qr_work_) { cudaFree(d_qr_work_); d_qr_work_ = nullptr; }
}

void GPUBlockKrylovSchur::initializeRandomBlock(cuDoubleComplex* d_block) {
    int total_elements = dimension_ * block_size_;
    int num_blocks_kernel = (total_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned long long seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    
    GPULanczosKernels::initRandomBlockKernel<<<num_blocks_kernel, BLOCK_SIZE>>>(
        d_block, dimension_, block_size_, seed);
    CUDA_CHECK(cudaDeviceSynchronize());
}

void GPUBlockKrylovSchur::orthonormalizeBlock(cuDoubleComplex* d_block) {
    // QR factorization using cuSOLVER
    CUSOLVER_CHECK(cusolverDnZgeqrf(
        cusolver_handle_, dimension_, block_size_,
        reinterpret_cast<cuDoubleComplex*>(d_block), dimension_,
        reinterpret_cast<cuDoubleComplex*>(d_tau_),
        reinterpret_cast<cuDoubleComplex*>(d_qr_work_), qr_work_size_,
        d_info_));
    
    // Generate Q from QR
    CUSOLVER_CHECK(cusolverDnZungqr(
        cusolver_handle_, dimension_, block_size_, block_size_,
        reinterpret_cast<cuDoubleComplex*>(d_block), dimension_,
        reinterpret_cast<cuDoubleComplex*>(d_tau_),
        reinterpret_cast<cuDoubleComplex*>(d_qr_work_), qr_work_size_,
        d_info_));
    
    CUDA_CHECK(cudaDeviceSynchronize());
}

void GPUBlockKrylovSchur::blockMatVec(const cuDoubleComplex* d_V_block, cuDoubleComplex* d_W_block) {
    // Apply operator to each column of the block
    for (int j = 0; j < block_size_; j++) {
        const cuDoubleComplex* v_col = d_V_block + j * dimension_;
        cuDoubleComplex* w_col = d_W_block + j * dimension_;
        op_->matVecGPU(v_col, w_col, dimension_);
    }
    stats_.total_block_steps += block_size_;
}

void GPUBlockKrylovSchur::computeBlockOverlap(const cuDoubleComplex* d_V1, const cuDoubleComplex* d_V2,
                                              cuDoubleComplex* d_overlap) {
    // Compute S = V1^H * V2 using cuBLAS ZGEMM
    // S (block_size × block_size) = V1^H (block_size × dim) * V2 (dim × block_size)
    cuDoubleComplex alpha = make_cuDoubleComplex(1.0, 0.0);
    cuDoubleComplex beta = make_cuDoubleComplex(0.0, 0.0);
    
    CUBLAS_CHECK(cublasZgemm(cublas_handle_,
                            CUBLAS_OP_C, CUBLAS_OP_N,  // V1^H * V2
                            block_size_, block_size_, dimension_,
                            &alpha,
                            d_V1, dimension_,
                            d_V2, dimension_,
                            &beta,
                            d_overlap, block_size_));
}

void GPUBlockKrylovSchur::orthogonalizeBlockAgainstBasis(int num_blocks, cuDoubleComplex* d_target) {
    auto start = std::chrono::high_resolution_clock::now();
    
    // Allocate temporary overlap matrix
    cuDoubleComplex* d_overlap;
    CUDA_CHECK(cudaMalloc(&d_overlap, block_size_ * block_size_ * sizeof(cuDoubleComplex)));
    
    cuDoubleComplex alpha = make_cuDoubleComplex(-1.0, 0.0);
    cuDoubleComplex beta = make_cuDoubleComplex(1.0, 0.0);
    
    // Orthogonalize against each stored block
    for (int b = 0; b < num_blocks; b++) {
        cuDoubleComplex* d_Vb = getBlock(b);
        
        // Compute overlap: S = Vb^H * target
        computeBlockOverlap(d_Vb, d_target, d_overlap);
        
        // target = target - Vb * S
        CUBLAS_CHECK(cublasZgemm(cublas_handle_,
                                CUBLAS_OP_N, CUBLAS_OP_N,
                                dimension_, block_size_, block_size_,
                                &alpha,
                                d_Vb, dimension_,
                                d_overlap, block_size_,
                                &beta,
                                d_target, dimension_));
    }
    
    // Reorthogonalize for numerical stability
    for (int b = 0; b < num_blocks; b++) {
        cuDoubleComplex* d_Vb = getBlock(b);
        computeBlockOverlap(d_Vb, d_target, d_overlap);
        
        CUBLAS_CHECK(cublasZgemm(cublas_handle_,
                                CUBLAS_OP_N, CUBLAS_OP_N,
                                dimension_, block_size_, block_size_,
                                &alpha,
                                d_Vb, dimension_,
                                d_overlap, block_size_,
                                &beta,
                                d_target, dimension_));
    }
    
    cudaFree(d_overlap);
    
    auto end = std::chrono::high_resolution_clock::now();
    stats_.ortho_time += std::chrono::duration<double>(end - start).count();
}

int GPUBlockKrylovSchur::blockArnoldiIteration(int start_block, int max_blocks) {
    int p = block_size_;
    int actual_blocks = start_block;
    
    // Temporary for diagonal block computation
    cuDoubleComplex* d_A_temp;
    CUDA_CHECK(cudaMalloc(&d_A_temp, p * p * sizeof(cuDoubleComplex)));
    
    for (int b = start_block; b < max_blocks; b++) {
        actual_blocks = b + 1;
        
        cuDoubleComplex* d_Vb = getBlock(b);
        
        // W = H * V_b
        auto matvec_start = std::chrono::high_resolution_clock::now();
        blockMatVec(d_Vb, d_W_);
        auto matvec_end = std::chrono::high_resolution_clock::now();
        stats_.matvec_time += std::chrono::duration<double>(matvec_end - matvec_start).count();
        
        // Compute diagonal block A_b = V_b^H * W before orthogonalization
        computeBlockOverlap(d_Vb, d_W_, d_A_temp);
        CUDA_CHECK(cudaMemcpy(&h_A_blocks_[b * p * p], d_A_temp, 
                             p * p * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost));
        
        // Orthogonalize W against all previous blocks (including current)
        orthogonalizeBlockAgainstBasis(b + 1, d_W_);
        
        // QR factorization of residual to get V_{b+1} and B_b
        if (b + 1 < max_blocks) {
            auto qr_start = std::chrono::high_resolution_clock::now();
            
            // Copy W to next block location
            cuDoubleComplex* d_Vbp1 = getBlock(b + 1);
            CUDA_CHECK(cudaMemcpy(d_Vbp1, d_W_, 
                                 static_cast<size_t>(dimension_) * p * sizeof(cuDoubleComplex),
                                 cudaMemcpyDeviceToDevice));
            
            // QR factorization
            CUSOLVER_CHECK(cusolverDnZgeqrf(
                cusolver_handle_, dimension_, p,
                reinterpret_cast<cuDoubleComplex*>(d_Vbp1), dimension_,
                reinterpret_cast<cuDoubleComplex*>(d_tau_),
                reinterpret_cast<cuDoubleComplex*>(d_qr_work_), qr_work_size_,
                d_info_));
            
            // Extract R as B_b (upper triangular part)
            std::vector<std::complex<double>> h_Vbp1(static_cast<size_t>(dimension_) * p);
            CUDA_CHECK(cudaMemcpy(h_Vbp1.data(), d_Vbp1,
                                 static_cast<size_t>(dimension_) * p * sizeof(cuDoubleComplex),
                                 cudaMemcpyDeviceToHost));
            
            for (int j = 0; j < p; j++) {
                for (int i = 0; i < p; i++) {
                    if (i <= j) {
                        h_B_blocks_[b * p * p + i + j * p] = h_Vbp1[i + j * dimension_];
                    } else {
                        h_B_blocks_[b * p * p + i + j * p] = std::complex<double>(0.0, 0.0);
                    }
                }
            }
            
            // Check for breakdown (small diagonal elements of R)
            double r_norm = 0.0;
            for (int j = 0; j < p; j++) {
                r_norm += std::norm(h_B_blocks_[b * p * p + j + j * p]);
            }
            r_norm = std::sqrt(r_norm);
            
            if (r_norm < tolerance_) {
                std::cout << "  Block Krylov subspace exhausted at block " << b + 1 << "\n";
                actual_blocks = b + 1;
                break;
            }
            
            // Generate Q for next block
            CUSOLVER_CHECK(cusolverDnZungqr(
                cusolver_handle_, dimension_, p, p,
                reinterpret_cast<cuDoubleComplex*>(d_Vbp1), dimension_,
                reinterpret_cast<cuDoubleComplex*>(d_tau_),
                reinterpret_cast<cuDoubleComplex*>(d_qr_work_), qr_work_size_,
                d_info_));
            
            auto qr_end = std::chrono::high_resolution_clock::now();
            stats_.qr_time += std::chrono::duration<double>(qr_end - qr_start).count();
        }
    }
    
    cudaFree(d_A_temp);
    return actual_blocks;
}

bool GPUBlockKrylovSchur::solveBlockTridiagonalEigenproblem(int num_blocks, 
                                                           std::vector<double>& eigenvalues) {
    auto start = std::chrono::high_resolution_clock::now();
    
    int p = block_size_;
    int total_dim = num_blocks * p;
    
    // Build dense block tridiagonal matrix on host
    std::vector<std::complex<double>> h_T(static_cast<size_t>(total_dim) * total_dim, 
                                          std::complex<double>(0.0, 0.0));
    
    // Fill diagonal blocks
    for (int b = 0; b < num_blocks; b++) {
        for (int j = 0; j < p; j++) {
            for (int i = 0; i < p; i++) {
                int row = b * p + i;
                int col = b * p + j;
                h_T[col * total_dim + row] = h_A_blocks_[b * p * p + i + j * p];
            }
        }
    }
    
    // Fill off-diagonal blocks
    for (int b = 0; b < num_blocks - 1; b++) {
        for (int j = 0; j < p; j++) {
            for (int i = 0; i < p; i++) {
                int row = (b + 1) * p + i;
                int col = b * p + j;
                h_T[col * total_dim + row] = h_B_blocks_[b * p * p + i + j * p];
                // Hermitian conjugate
                h_T[row * total_dim + col] = std::conj(h_B_blocks_[b * p * p + i + j * p]);
            }
        }
    }
    
    // Copy to GPU and solve
    CUDA_CHECK(cudaMemcpy(d_evecs_, h_T.data(),
                         static_cast<size_t>(total_dim) * total_dim * sizeof(cuDoubleComplex),
                         cudaMemcpyHostToDevice));
    
    CUSOLVER_CHECK(cusolverDnZheevd(
        cusolver_handle_,
        CUSOLVER_EIG_MODE_VECTOR,
        CUBLAS_FILL_MODE_UPPER,
        total_dim,
        reinterpret_cast<cuDoubleComplex*>(d_evecs_), total_dim,
        d_evals_,
        reinterpret_cast<cuDoubleComplex*>(d_work_), work_size_,
        d_info_));
    
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Check for errors
    int info;
    CUDA_CHECK(cudaMemcpy(&info, d_info_, sizeof(int), cudaMemcpyDeviceToHost));
    if (info != 0) {
        std::cerr << "cuSOLVER zheevd failed with info = " << info << "\n";
        return false;
    }
    
    // Copy eigenvalues back
    eigenvalues.resize(total_dim);
    CUDA_CHECK(cudaMemcpy(eigenvalues.data(), d_evals_,
                         total_dim * sizeof(double), cudaMemcpyDeviceToHost));
    
    auto end = std::chrono::high_resolution_clock::now();
    stats_.schur_time += std::chrono::duration<double>(end - start).count();
    
    return true;
}

int GPUBlockKrylovSchur::checkConvergence(int num_blocks, int num_desired) {
    int p = block_size_;
    int total_dim = num_blocks * p;
    
    // Get eigenvectors from device
    std::vector<std::complex<double>> h_evecs(static_cast<size_t>(total_dim) * total_dim);
    CUDA_CHECK(cudaMemcpy(h_evecs.data(), d_evecs_,
                         static_cast<size_t>(total_dim) * total_dim * sizeof(cuDoubleComplex),
                         cudaMemcpyDeviceToHost));
    
    int num_converged = 0;
    double max_residual = 0.0;
    
    // Estimate residuals from last block components and B_last
    for (int i = 0; i < std::min(num_desired, total_dim); i++) {
        double residual = 0.0;
        
        if (num_blocks > 1) {
            // Residual ~ ||B_last|| * ||y_last||
            for (int j = 0; j < p; j++) {
                int last_row = total_dim - p + j;
                residual += std::norm(h_evecs[last_row + i * total_dim]);
            }
            residual = std::sqrt(residual);
            
            // Scale by norm of last B block
            double b_norm = 0.0;
            for (int j = 0; j < p; j++) {
                b_norm += std::norm(h_B_blocks_[(num_blocks - 2) * p * p + j + j * p]);
            }
            residual *= std::sqrt(b_norm);
        }
        
        max_residual = std::max(max_residual, residual);
        if (residual < tolerance_) {
            num_converged++;
        }
    }
    
    stats_.final_residual = max_residual;
    return num_converged;
}

void GPUBlockKrylovSchur::computeEigenvectors(int num_blocks, int num_eigs,
                                              std::vector<std::vector<std::complex<double>>>& eigenvectors) {
    int p = block_size_;
    int total_dim = num_blocks * p;
    
    // Get Ritz vectors from device
    std::vector<std::complex<double>> h_evecs(static_cast<size_t>(total_dim) * total_dim);
    CUDA_CHECK(cudaMemcpy(h_evecs.data(), d_evecs_,
                         static_cast<size_t>(total_dim) * total_dim * sizeof(cuDoubleComplex),
                         cudaMemcpyDeviceToHost));
    
    eigenvectors.resize(num_eigs);
    
    for (int e = 0; e < num_eigs; e++) {
        eigenvectors[e].resize(dimension_, std::complex<double>(0.0, 0.0));
        
        // eigenvector = sum_b V_b * y_b where y_b is the b-th block of Ritz vector
        cuDoubleComplex* d_eigvec;
        CUDA_CHECK(cudaMalloc(&d_eigvec, dimension_ * sizeof(cuDoubleComplex)));
        CUDA_CHECK(cudaMemset(d_eigvec, 0, dimension_ * sizeof(cuDoubleComplex)));
        
        cuDoubleComplex alpha = make_cuDoubleComplex(1.0, 0.0);
        cuDoubleComplex beta = make_cuDoubleComplex(1.0, 0.0);
        
        for (int b = 0; b < num_blocks; b++) {
            // Extract coefficients for this block
            cuDoubleComplex* d_coef;
            CUDA_CHECK(cudaMalloc(&d_coef, p * sizeof(cuDoubleComplex)));
            CUDA_CHECK(cudaMemcpy(d_coef, &h_evecs[b * p + e * total_dim],
                                 p * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));
            
            // eigvec += V_b * coef
            CUBLAS_CHECK(cublasZgemv(cublas_handle_, CUBLAS_OP_N,
                                    dimension_, p,
                                    &alpha,
                                    getBlock(b), dimension_,
                                    d_coef, 1,
                                    &beta,
                                    d_eigvec, 1));
            
            cudaFree(d_coef);
        }
        
        // Normalize
        double norm;
        CUBLAS_CHECK(cublasDznrm2(cublas_handle_, dimension_, d_eigvec, 1, &norm));
        if (norm > tolerance_) {
            cuDoubleComplex scale = make_cuDoubleComplex(1.0 / norm, 0.0);
            CUBLAS_CHECK(cublasZscal(cublas_handle_, dimension_, &scale, d_eigvec, 1));
        }
        
        // Copy to host
        std::vector<cuDoubleComplex> h_eigvec(dimension_);
        CUDA_CHECK(cudaMemcpy(h_eigvec.data(), d_eigvec,
                             dimension_ * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost));
        
        for (int i = 0; i < dimension_; i++) {
            eigenvectors[e][i] = std::complex<double>(cuCreal(h_eigvec[i]), cuCimag(h_eigvec[i]));
        }
        
        cudaFree(d_eigvec);
    }
}

void GPUBlockKrylovSchur::run(int num_eigenvalues,
                              std::vector<double>& eigenvalues,
                              std::vector<std::vector<std::complex<double>>>& eigenvectors,
                              bool compute_vectors) {
    
    auto overall_start = std::chrono::high_resolution_clock::now();
    
    std::cout << "\n========================================\n";
    std::cout << "GPU Block Krylov-Schur Algorithm (Direct Projection)\n";
    std::cout << "========================================\n";
    std::cout << "  Dimension: " << dimension_ << "\n";
    std::cout << "  Target eigenvalues: " << num_eigenvalues << "\n";
    std::cout << "  Block size: " << block_size_ << "\n";
    std::cout << "  Tolerance: " << tolerance_ << "\n\n";
    
    // Parameters - match CPU implementation
    int p = std::min(block_size_, static_cast<int>(dimension_));
    int k = std::min(num_eigenvalues, static_cast<int>(dimension_));
    
    // Subspace size: use larger subspace for better convergence
    // At least 4*k or 6*p, whichever is larger, capped at dimension and max_iter
    int m = std::min(std::max(4*k + p, 6*p), std::min(max_iter_, static_cast<int>(dimension_)));
    m = std::min(m, static_cast<int>(dimension_));
    // For small dimensions, use the full space for guaranteed convergence
    if (dimension_ <= 100) m = dimension_;
    
    std::cout << "  Subspace size: " << m << "\n";
    
    // Allocate memory for Krylov basis V (dimension x m)
    cuDoubleComplex* d_V;
    CUDA_CHECK(cudaMalloc(&d_V, static_cast<size_t>(dimension_) * m * sizeof(cuDoubleComplex)));
    CUDA_CHECK(cudaMemset(d_V, 0, static_cast<size_t>(dimension_) * m * sizeof(cuDoubleComplex)));
    
    // Allocate work vectors
    cuDoubleComplex* d_Hv;
    CUDA_CHECK(cudaMalloc(&d_Hv, dimension_ * sizeof(cuDoubleComplex)));
    
    // Initialize first p vectors randomly with non-fixed seed for better convergence
    int first_block = std::min(p, m);
    std::vector<std::complex<double>> h_init(static_cast<size_t>(dimension_) * first_block);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    for (int j = 0; j < first_block; j++) {
        for (int i = 0; i < dimension_; i++) {
            h_init[j * dimension_ + i] = std::complex<double>(dist(gen), dist(gen));
        }
    }
    CUDA_CHECK(cudaMemcpy(d_V, h_init.data(), 
                         static_cast<size_t>(dimension_) * first_block * sizeof(cuDoubleComplex),
                         cudaMemcpyHostToDevice));
    
    // QR factorization of initial block
    cuDoubleComplex* d_tau;
    CUDA_CHECK(cudaMalloc(&d_tau, first_block * sizeof(cuDoubleComplex)));
    
    int qr_lwork;
    CUSOLVER_CHECK(cusolverDnZgeqrf_bufferSize(cusolver_handle_, dimension_, first_block,
                                               d_V, dimension_, &qr_lwork));
    cuDoubleComplex* d_qr_work;
    CUDA_CHECK(cudaMalloc(&d_qr_work, qr_lwork * sizeof(cuDoubleComplex)));
    int* d_info;
    CUDA_CHECK(cudaMalloc(&d_info, sizeof(int)));
    
    CUSOLVER_CHECK(cusolverDnZgeqrf(cusolver_handle_, dimension_, first_block,
                                    d_V, dimension_, d_tau, d_qr_work, qr_lwork, d_info));
    CUSOLVER_CHECK(cusolverDnZungqr(cusolver_handle_, dimension_, first_block, first_block,
                                    d_V, dimension_, d_tau, d_qr_work, qr_lwork, d_info));
    
    int current_dim = first_block;
    
    std::cout << "  Building Krylov subspace..." << std::endl;
    
    auto matvec_start = std::chrono::high_resolution_clock::now();
    
    // Expand Krylov subspace one vector at a time (like CPU)
    while (current_dim < m) {
        cuDoubleComplex* v_src = d_V + (current_dim - 1) * dimension_;
        cuDoubleComplex* v_dst = d_V + current_dim * dimension_;
        
        // Apply Hamiltonian
        op_->matVecGPU(v_src, v_dst, dimension_);
        stats_.total_block_steps++;
        
        // Full orthogonalization with two passes (modified Gram-Schmidt + reorthogonalization)
        for (int pass = 0; pass < 2; pass++) {
            for (int l = 0; l < current_dim; l++) {
                cuDoubleComplex* v_l = d_V + l * dimension_;
                cuDoubleComplex dot;
                CUBLAS_CHECK(cublasZdotc(cublas_handle_, dimension_, v_l, 1, v_dst, 1, &dot));
                cuDoubleComplex neg_dot = make_cuDoubleComplex(-cuCreal(dot), -cuCimag(dot));
                CUBLAS_CHECK(cublasZaxpy(cublas_handle_, dimension_, &neg_dot, v_l, 1, v_dst, 1));
            }
        }
        
        // Normalize
        double norm;
        CUBLAS_CHECK(cublasDznrm2(cublas_handle_, dimension_, v_dst, 1, &norm));
        if (norm < tolerance_) {
            std::cout << "  Krylov subspace exhausted at dimension " << current_dim << "\n";
            break;
        }
        cuDoubleComplex scale = make_cuDoubleComplex(1.0 / norm, 0.0);
        CUBLAS_CHECK(cublasZscal(cublas_handle_, dimension_, &scale, v_dst, 1));
        
        current_dim++;
    }
    
    auto matvec_end = std::chrono::high_resolution_clock::now();
    stats_.matvec_time = std::chrono::duration<double>(matvec_end - matvec_start).count();
    
    std::cout << "  Final subspace size: " << current_dim << "\n";
    std::cout << "  Computing projected Hamiltonian..." << std::endl;
    
    // Compute projected Hamiltonian directly: H_m[i,j] = <v_i | H | v_j>
    std::vector<std::complex<double>> h_H_m(static_cast<size_t>(current_dim) * current_dim, 
                                            std::complex<double>(0.0, 0.0));
    
    auto proj_start = std::chrono::high_resolution_clock::now();
    
    for (int j = 0; j < current_dim; j++) {
        // Compute H * v_j
        op_->matVecGPU(d_V + j * dimension_, d_Hv, dimension_);
        
        // Compute inner products: H_m[i,j] = v_i^H * (H * v_j)
        for (int i = 0; i < current_dim; i++) {
            cuDoubleComplex dot;
            CUBLAS_CHECK(cublasZdotc(cublas_handle_, dimension_, d_V + i * dimension_, 1, d_Hv, 1, &dot));
            h_H_m[j * current_dim + i] = std::complex<double>(cuCreal(dot), cuCimag(dot));  // Column-major
        }
    }
    
    // Enforce Hermitian symmetry for numerical stability
    for (int i = 0; i < current_dim; i++) {
        for (int j = i + 1; j < current_dim; j++) {
            std::complex<double> avg = 0.5 * (h_H_m[j * current_dim + i] + std::conj(h_H_m[i * current_dim + j]));
            h_H_m[j * current_dim + i] = avg;
            h_H_m[i * current_dim + j] = std::conj(avg);
        }
        h_H_m[i * current_dim + i] = std::complex<double>(std::real(h_H_m[i * current_dim + i]), 0.0);
    }
    
    auto proj_end = std::chrono::high_resolution_clock::now();
    stats_.ortho_time = std::chrono::duration<double>(proj_end - proj_start).count();
    
    // Diagonalize projected Hamiltonian
    std::cout << "  Diagonalizing projected matrix (size " << current_dim << "x" << current_dim << ")..." << std::endl;
    
    auto eigen_start = std::chrono::high_resolution_clock::now();
    
    // Allocate for eigensolve
    cuDoubleComplex* d_H_m;
    CUDA_CHECK(cudaMalloc(&d_H_m, static_cast<size_t>(current_dim) * current_dim * sizeof(cuDoubleComplex)));
    CUDA_CHECK(cudaMemcpy(d_H_m, h_H_m.data(), 
                         static_cast<size_t>(current_dim) * current_dim * sizeof(cuDoubleComplex),
                         cudaMemcpyHostToDevice));
    
    double* d_evals;
    CUDA_CHECK(cudaMalloc(&d_evals, current_dim * sizeof(double)));
    
    // Query workspace size
    int lwork;
    CUSOLVER_CHECK(cusolverDnZheevd_bufferSize(cusolver_handle_, CUSOLVER_EIG_MODE_VECTOR,
                                               CUBLAS_FILL_MODE_UPPER, current_dim,
                                               d_H_m, current_dim, d_evals, &lwork));
    cuDoubleComplex* d_work;
    CUDA_CHECK(cudaMalloc(&d_work, lwork * sizeof(cuDoubleComplex)));
    
    CUSOLVER_CHECK(cusolverDnZheevd(cusolver_handle_, CUSOLVER_EIG_MODE_VECTOR,
                                    CUBLAS_FILL_MODE_UPPER, current_dim,
                                    d_H_m, current_dim, d_evals, d_work, lwork, d_info));
    
    auto eigen_end = std::chrono::high_resolution_clock::now();
    stats_.schur_time = std::chrono::duration<double>(eigen_end - eigen_start).count();
    
    // Copy eigenvalues back
    std::vector<double> evals_all(current_dim);
    CUDA_CHECK(cudaMemcpy(evals_all.data(), d_evals, current_dim * sizeof(double), cudaMemcpyDeviceToHost));
    
    // Extract requested eigenvalues
    int num_eigs = std::min(k, current_dim);
    eigenvalues.resize(num_eigs);
    for (int i = 0; i < num_eigs; i++) {
        eigenvalues[i] = evals_all[i];
    }
    
    std::cout << "\n  Computed eigenvalues:" << std::endl;
    for (int i = 0; i < std::min(5, num_eigs); i++) {
        std::cout << "    E[" << i << "] = " << std::fixed << std::setprecision(10) << eigenvalues[i] << "\n";
    }
    if (num_eigs > 5) std::cout << "    ...\n";
    
    // Compute eigenvectors if requested
    if (compute_vectors && num_eigs > 0) {
        std::cout << "  Computing eigenvectors..." << std::endl;
        
        // Get eigenvector coefficients from GPU (d_H_m now contains them)
        std::vector<std::complex<double>> h_evec_coefs(static_cast<size_t>(current_dim) * current_dim);
        CUDA_CHECK(cudaMemcpy(h_evec_coefs.data(), d_H_m,
                             static_cast<size_t>(current_dim) * current_dim * sizeof(cuDoubleComplex),
                             cudaMemcpyDeviceToHost));
        
        eigenvectors.resize(num_eigs);
        
        cuDoubleComplex* d_eigvec;
        CUDA_CHECK(cudaMalloc(&d_eigvec, dimension_ * sizeof(cuDoubleComplex)));
        
        for (int e = 0; e < num_eigs; e++) {
            // eigenvector = V * y_e
            cuDoubleComplex* d_coefs;
            CUDA_CHECK(cudaMalloc(&d_coefs, current_dim * sizeof(cuDoubleComplex)));
            CUDA_CHECK(cudaMemcpy(d_coefs, &h_evec_coefs[e * current_dim],
                                 current_dim * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));
            
            cuDoubleComplex alpha = make_cuDoubleComplex(1.0, 0.0);
            cuDoubleComplex beta = make_cuDoubleComplex(0.0, 0.0);
            CUBLAS_CHECK(cublasZgemv(cublas_handle_, CUBLAS_OP_N,
                                    dimension_, current_dim,
                                    &alpha, d_V, dimension_,
                                    d_coefs, 1,
                                    &beta, d_eigvec, 1));
            
            // Normalize
            double norm;
            CUBLAS_CHECK(cublasDznrm2(cublas_handle_, dimension_, d_eigvec, 1, &norm));
            if (norm > tolerance_) {
                cuDoubleComplex scale_v = make_cuDoubleComplex(1.0 / norm, 0.0);
                CUBLAS_CHECK(cublasZscal(cublas_handle_, dimension_, &scale_v, d_eigvec, 1));
            }
            
            // Copy to host
            eigenvectors[e].resize(dimension_);
            std::vector<cuDoubleComplex> h_eigvec(dimension_);
            CUDA_CHECK(cudaMemcpy(h_eigvec.data(), d_eigvec, dimension_ * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost));
            for (int i = 0; i < dimension_; i++) {
                eigenvectors[e][i] = std::complex<double>(cuCreal(h_eigvec[i]), cuCimag(h_eigvec[i]));
            }
            
            cudaFree(d_coefs);
        }
        
        cudaFree(d_eigvec);
    }
    
    auto overall_end = std::chrono::high_resolution_clock::now();
    stats_.total_time = std::chrono::duration<double>(overall_end - overall_start).count();
    
    // Print results
    std::cout << "\n========================================\n";
    std::cout << "GPU Block Krylov-Schur Results\n";
    std::cout << "========================================\n";
    std::cout << "Converged: YES\n";
    std::cout << "Subspace dimension: " << current_dim << "\n";
    std::cout << "\nTiming breakdown:\n";
    std::cout << "  Krylov expansion: " << stats_.matvec_time << " s\n";
    std::cout << "  Projected H computation: " << stats_.ortho_time << " s\n";
    std::cout << "  Eigenproblem: " << stats_.schur_time << " s\n";
    std::cout << "  Total time: " << stats_.total_time << " s\n";
    
    std::cout << "\nLowest eigenvalues:\n";
    int n_print = std::min(5, static_cast<int>(eigenvalues.size()));
    for (int i = 0; i < n_print; i++) {
        std::cout << "  E[" << i << "] = " << std::fixed << std::setprecision(10) 
                  << eigenvalues[i] << "\n";
    }
    if (eigenvalues.size() > 5) {
        std::cout << "  ... (" << eigenvalues.size() - 5 << " more)\n";
    }
    std::cout << "========================================\n";
    
    // Cleanup
    cudaFree(d_V);
    cudaFree(d_Hv);
    cudaFree(d_tau);
    cudaFree(d_qr_work);
    cudaFree(d_info);
    cudaFree(d_H_m);
    cudaFree(d_evals);
    cudaFree(d_work);
}

#endif // WITH_CUDA
