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
    
    // Reuse d_temp_ as overlap scratch (it's dim*block_size, we only need block_size^2)
    // This avoids expensive cudaMalloc/cudaFree on every call which kills GPU performance
    cuDoubleComplex* d_overlap = d_temp_;
    
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
    
    // No cudaFree needed - d_overlap aliases pre-allocated d_temp_
    
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

void GPUBlockKrylovSchur::performRestart(int num_blocks, int num_keep) {
    // Thick restart is handled inline in run() using Ritz vector rotation.
    // This method is kept for API compatibility with the block tridiagonal interface.
    // See run() Phase 4 for the actual restart implementation.
}

void GPUBlockKrylovSchur::run(int num_eigenvalues,
                              std::vector<double>& eigenvalues,
                              std::vector<std::vector<std::complex<double>>>& eigenvectors,
                              bool compute_vectors) {
    
    auto overall_start = std::chrono::high_resolution_clock::now();
    
    std::cout << "\n========================================\n";
    std::cout << "GPU Block Krylov-Schur Algorithm\n";
    std::cout << "========================================\n";
    std::cout << "  Dimension: " << dimension_ << "\n";
    std::cout << "  Target eigenvalues: " << num_eigenvalues << "\n";
    std::cout << "  Block size: " << block_size_ << "\n";
    std::cout << "  Tolerance: " << tolerance_ << "\n\n";
    
    int k = std::min(num_eigenvalues, static_cast<int>(dimension_));
    int p = std::min(block_size_, static_cast<int>(dimension_));
    
    // Krylov subspace size: needs m > k for restart to work
    // Rule: m ≈ max(4k + p, 6p, 200), capped at dimension and max_iter
    int m = std::min(std::max({4*k + p, 6*p, 200}),
                     std::min(max_iter_, static_cast<int>(dimension_)));
    if (dimension_ <= 100) m = dimension_;
    if (m <= k) m = std::min(2 * k + 10, static_cast<int>(dimension_));
    
    std::cout << "  Krylov subspace size: " << m << "\n";
    std::cout << "  Max restart cycles: " << max_outer_iter_ << "\n";
    
    // ===================== ONE-TIME ALLOCATION =====================
    // All GPU buffers pre-allocated here. No cudaMalloc in any loop.
    
    cuDoubleComplex* d_V;
    CUDA_CHECK(cudaMalloc(&d_V, static_cast<size_t>(dimension_) * m * sizeof(cuDoubleComplex)));
    CUDA_CHECK(cudaMemset(d_V, 0, static_cast<size_t>(dimension_) * m * sizeof(cuDoubleComplex)));
    
    cuDoubleComplex* d_Hv;
    CUDA_CHECK(cudaMalloc(&d_Hv, dimension_ * sizeof(cuDoubleComplex)));
    
    cuDoubleComplex* d_h_overlaps;
    CUDA_CHECK(cudaMalloc(&d_h_overlaps, m * sizeof(cuDoubleComplex)));
    
    cuDoubleComplex* d_V_restart;
    CUDA_CHECK(cudaMalloc(&d_V_restart, static_cast<size_t>(dimension_) * k * sizeof(cuDoubleComplex)));
    
    cuDoubleComplex* d_H_proj;
    CUDA_CHECK(cudaMalloc(&d_H_proj, static_cast<size_t>(m) * m * sizeof(cuDoubleComplex)));
    
    double* d_evals;
    CUDA_CHECK(cudaMalloc(&d_evals, m * sizeof(double)));
    
    cuDoubleComplex* d_tau;
    CUDA_CHECK(cudaMalloc(&d_tau, p * sizeof(cuDoubleComplex)));
    
    int* d_info;
    CUDA_CHECK(cudaMalloc(&d_info, sizeof(int)));
    
    cuDoubleComplex* d_coefs;
    CUDA_CHECK(cudaMalloc(&d_coefs, m * sizeof(cuDoubleComplex)));
    
    cuDoubleComplex* d_eigvec;
    CUDA_CHECK(cudaMalloc(&d_eigvec, dimension_ * sizeof(cuDoubleComplex)));
    
    int qr_lwork;
    CUSOLVER_CHECK(cusolverDnZgeqrf_bufferSize(cusolver_handle_, dimension_, p,
                                               d_V, dimension_, &qr_lwork));
    cuDoubleComplex* d_qr_work;
    CUDA_CHECK(cudaMalloc(&d_qr_work, qr_lwork * sizeof(cuDoubleComplex)));
    
    int eigsolve_lwork;
    CUSOLVER_CHECK(cusolverDnZheevd_bufferSize(cusolver_handle_, CUSOLVER_EIG_MODE_VECTOR,
                                               CUBLAS_FILL_MODE_UPPER, m,
                                               d_H_proj, m, d_evals, &eigsolve_lwork));
    cuDoubleComplex* d_eigsolve_work;
    CUDA_CHECK(cudaMalloc(&d_eigsolve_work, eigsolve_lwork * sizeof(cuDoubleComplex)));
    
    // Host Hessenberg matrix (column-major, m × m)
    std::vector<std::complex<double>> h_H_m(static_cast<size_t>(m) * m,
                                            std::complex<double>(0.0, 0.0));
    
    // ===================== INITIALIZATION =====================
    // Generate p random orthonormal starting vectors via QR
    int init_block = std::min(p, m);
    {
        std::vector<std::complex<double>> h_init(static_cast<size_t>(dimension_) * init_block);
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<double> dist(-1.0, 1.0);
        for (int j = 0; j < init_block; j++) {
            for (int i = 0; i < dimension_; i++) {
                h_init[j * dimension_ + i] = std::complex<double>(dist(gen), dist(gen));
            }
        }
        CUDA_CHECK(cudaMemcpy(d_V, h_init.data(),
                             static_cast<size_t>(dimension_) * init_block * sizeof(cuDoubleComplex),
                             cudaMemcpyHostToDevice));
        CUSOLVER_CHECK(cusolverDnZgeqrf(cusolver_handle_, dimension_, init_block,
                                        d_V, dimension_, d_tau, d_qr_work, qr_lwork, d_info));
        CUSOLVER_CHECK(cusolverDnZungqr(cusolver_handle_, dimension_, init_block, init_block,
                                        d_V, dimension_, d_tau, d_qr_work, qr_lwork, d_info));
    }
    
    // Compute projected H for initial block explicitly: h_H_m[i,j] = <v_i, H*v_j>
    for (int j = 0; j < init_block; j++) {
        op_->matVecGPU(d_V + static_cast<size_t>(j) * dimension_, d_Hv, dimension_);
        stats_.total_block_steps++;
        for (int i = 0; i <= j; i++) {
            cuDoubleComplex dot;
            CUBLAS_CHECK(cublasZdotc(cublas_handle_, dimension_,
                                    d_V + static_cast<size_t>(i) * dimension_, 1, d_Hv, 1, &dot));
            h_H_m[i + static_cast<size_t>(j) * m] =
                std::complex<double>(cuCreal(dot), cuCimag(dot));
            if (i != j) {
                h_H_m[j + static_cast<size_t>(i) * m] =
                    std::conj(h_H_m[i + static_cast<size_t>(j) * m]);
            }
        }
    }
    
    int current_dim = init_block;
    bool converged = false;
    std::vector<double> prev_eigenvalues;
    
    // ===================== MAIN RESTART LOOP =====================
    for (int outer = 0; outer < max_outer_iter_ && !converged; outer++) {
        stats_.outer_iterations++;
        std::cout << "  Krylov-Schur cycle " << outer + 1
                  << " (starting dim = " << current_dim << ")\n";
        
        // === Phase 1: Expand Krylov subspace via Arnoldi ===
        // Uses batched GEMV for orthogonalization + DGKS criterion.
        // Builds Hessenberg H_m incrementally (avoids redundant mat-vecs for projection).
        auto matvec_start = std::chrono::high_resolution_clock::now();
        
        while (current_dim < m) {
            cuDoubleComplex* v_src = d_V + static_cast<size_t>(current_dim - 1) * dimension_;
            cuDoubleComplex* v_dst = d_V + static_cast<size_t>(current_dim) * dimension_;
            
            // w = H * v_{current_dim-1}
            op_->matVecGPU(v_src, v_dst, dimension_);
            stats_.total_block_steps++;
            
            // Norm before orthogonalization (for DGKS criterion)
            double norm_before;
            CUBLAS_CHECK(cublasDznrm2(cublas_handle_, dimension_, v_dst, 1, &norm_before));
            
            // --- Batched CGS pass 1: h = V^H * w, then w -= V * h ---
            {
                cuDoubleComplex alpha_cgs = make_cuDoubleComplex(1.0, 0.0);
                cuDoubleComplex beta_cgs = make_cuDoubleComplex(0.0, 0.0);
                
                // h = V[:,0:current_dim]^H * w   (single GEMV instead of current_dim dot products)
                CUBLAS_CHECK(cublasZgemv(cublas_handle_, CUBLAS_OP_C,
                                        dimension_, current_dim,
                                        &alpha_cgs, d_V, dimension_,
                                        v_dst, 1,
                                        &beta_cgs, d_h_overlaps, 1));
                
                // Store overlaps in Hessenberg column (current_dim - 1)
                std::vector<cuDoubleComplex> h_tmp(current_dim);
                CUDA_CHECK(cudaMemcpy(h_tmp.data(), d_h_overlaps,
                                     current_dim * sizeof(cuDoubleComplex),
                                     cudaMemcpyDeviceToHost));
                for (int i = 0; i < current_dim; i++) {
                    h_H_m[i + static_cast<size_t>(current_dim - 1) * m] =
                        std::complex<double>(cuCreal(h_tmp[i]), cuCimag(h_tmp[i]));
                }
                
                // w -= V * h
                alpha_cgs = make_cuDoubleComplex(-1.0, 0.0);
                beta_cgs = make_cuDoubleComplex(1.0, 0.0);
                CUBLAS_CHECK(cublasZgemv(cublas_handle_, CUBLAS_OP_N,
                                        dimension_, current_dim,
                                        &alpha_cgs, d_V, dimension_,
                                        d_h_overlaps, 1,
                                        &beta_cgs, v_dst, 1));
            }
            
            // --- DGKS criterion: second pass if norm dropped > 1/sqrt(2) ---
            double norm_after;
            CUBLAS_CHECK(cublasDznrm2(cublas_handle_, dimension_, v_dst, 1, &norm_after));
            
            static constexpr double DGKS_ETA = 0.7071067811865476;  // 1/sqrt(2)
            if (norm_after < DGKS_ETA * norm_before) {
                cuDoubleComplex alpha2 = make_cuDoubleComplex(1.0, 0.0);
                cuDoubleComplex beta2 = make_cuDoubleComplex(0.0, 0.0);
                
                CUBLAS_CHECK(cublasZgemv(cublas_handle_, CUBLAS_OP_C,
                                        dimension_, current_dim,
                                        &alpha2, d_V, dimension_,
                                        v_dst, 1,
                                        &beta2, d_h_overlaps, 1));
                
                // Accumulate corrections into Hessenberg
                std::vector<cuDoubleComplex> h_corr(current_dim);
                CUDA_CHECK(cudaMemcpy(h_corr.data(), d_h_overlaps,
                                     current_dim * sizeof(cuDoubleComplex),
                                     cudaMemcpyDeviceToHost));
                for (int i = 0; i < current_dim; i++) {
                    h_H_m[i + static_cast<size_t>(current_dim - 1) * m] +=
                        std::complex<double>(cuCreal(h_corr[i]), cuCimag(h_corr[i]));
                }
                
                alpha2 = make_cuDoubleComplex(-1.0, 0.0);
                beta2 = make_cuDoubleComplex(1.0, 0.0);
                CUBLAS_CHECK(cublasZgemv(cublas_handle_, CUBLAS_OP_N,
                                        dimension_, current_dim,
                                        &alpha2, d_V, dimension_,
                                        d_h_overlaps, 1,
                                        &beta2, v_dst, 1));
                
                CUBLAS_CHECK(cublasDznrm2(cublas_handle_, dimension_, v_dst, 1, &norm_after));
            }
            
            // Store beta (subdiagonal)
            h_H_m[current_dim + static_cast<size_t>(current_dim - 1) * m] =
                std::complex<double>(norm_after, 0.0);
            
            // Check for Krylov breakdown
            if (norm_after < tolerance_) {
                std::cout << "  Krylov breakdown at dim " << current_dim << "\n";
                break;
            }
            
            // Normalize new vector
            cuDoubleComplex scale = make_cuDoubleComplex(1.0 / norm_after, 0.0);
            CUBLAS_CHECK(cublasZscal(cublas_handle_, dimension_, &scale, v_dst, 1));
            current_dim++;
        }
        
        // The Arnoldi loop above computes column (current_dim-1) of the projected
        // matrix each iteration.  When it exits at current_dim == m, the last
        // column (m-1) was never filled: we created V[m-1] but never computed
        // h_H_m[i, m-1] = <V[i], H*V[m-1]>.  Fix that with one extra mat-vec.
        if (current_dim == m) {
            cuDoubleComplex* v_last = d_V + static_cast<size_t>(m - 1) * dimension_;
            op_->matVecGPU(v_last, d_Hv, dimension_);
            stats_.total_block_steps++;
            for (int i = 0; i < m; i++) {
                cuDoubleComplex dot;
                CUBLAS_CHECK(cublasZdotc(cublas_handle_, dimension_,
                                        d_V + static_cast<size_t>(i) * dimension_, 1,
                                        d_Hv, 1, &dot));
                h_H_m[i + static_cast<size_t>(m - 1) * m] =
                    std::complex<double>(cuCreal(dot), cuCimag(dot));
            }
        }
        
        auto matvec_end = std::chrono::high_resolution_clock::now();
        stats_.matvec_time += std::chrono::duration<double>(matvec_end - matvec_start).count();
        
        std::cout << "  Subspace size: " << current_dim << "\n";
        
        // === Phase 2: Solve projected eigenproblem ===
        // Copy upper triangle of Hessenberg to lower to form Hermitian
        // projected matrix.  The expansion uses CGS so upper-triangle entries
        // are exact inner products <v_i, H*v_j>.  After a thick restart the
        // lower triangle of the restart columns is zero, so averaging would
        // halve the cross-block coupling.  Copy instead.
        std::vector<double> evals_all(current_dim);
        {
            auto proj_start = std::chrono::high_resolution_clock::now();
            
            std::vector<cuDoubleComplex> h_proj(static_cast<size_t>(current_dim) * current_dim);
            for (int i = 0; i < current_dim; i++) {
                // Diagonal: force real
                h_proj[i + i * current_dim] = make_cuDoubleComplex(
                    h_H_m[i + static_cast<size_t>(i) * m].real(), 0.0);
                for (int j = i + 1; j < current_dim; j++) {
                    // Upper triangle entry: h_H_m(i, j) where i < j
                    auto h_ij = h_H_m[i + static_cast<size_t>(j) * m];
                    h_proj[i + j * current_dim] = make_cuDoubleComplex(h_ij.real(), h_ij.imag());
                    h_proj[j + i * current_dim] = make_cuDoubleComplex(h_ij.real(), -h_ij.imag());
                }
            }
            
            CUDA_CHECK(cudaMemcpy(d_H_proj, h_proj.data(),
                                 static_cast<size_t>(current_dim) * current_dim * sizeof(cuDoubleComplex),
                                 cudaMemcpyHostToDevice));
            
            CUSOLVER_CHECK(cusolverDnZheevd(cusolver_handle_, CUSOLVER_EIG_MODE_VECTOR,
                                            CUBLAS_FILL_MODE_UPPER, current_dim,
                                            d_H_proj, current_dim, d_evals,
                                            d_eigsolve_work, eigsolve_lwork, d_info));
            
            int info;
            CUDA_CHECK(cudaMemcpy(&info, d_info, sizeof(int), cudaMemcpyDeviceToHost));
            if (info != 0) {
                std::cerr << "  cuSOLVER zheevd failed (info=" << info << ")\n";
                break;
            }
            
            CUDA_CHECK(cudaMemcpy(evals_all.data(), d_evals,
                                 current_dim * sizeof(double), cudaMemcpyDeviceToHost));
            
            auto proj_end = std::chrono::high_resolution_clock::now();
            stats_.schur_time += std::chrono::duration<double>(proj_end - proj_start).count();
        }
        
        int num_eigs = std::min(k, current_dim);
        eigenvalues.resize(num_eigs);
        for (int i = 0; i < num_eigs; i++) eigenvalues[i] = evals_all[i];
        
        // === Phase 3: Check convergence ===
        int num_converged = 0;
        double max_change = 0.0;
        if (!prev_eigenvalues.empty()) {
            for (int i = 0; i < num_eigs && i < static_cast<int>(prev_eigenvalues.size()); i++) {
                double change = std::abs(eigenvalues[i] - prev_eigenvalues[i]);
                double s = std::max(1.0, std::abs(eigenvalues[i]));
                double rel = change / s;
                max_change = std::max(max_change, rel);
                if (rel < tolerance_) num_converged++;
            }
            std::cout << "  Converged: " << num_converged << "/" << k
                      << "  (max rel. change: " << std::scientific << max_change << ")\n";
        }
        
        for (int i = 0; i < std::min(3, num_eigs); i++) {
            std::cout << "    E[" << i << "] = " << std::fixed
                      << std::setprecision(12) << eigenvalues[i] << "\n";
        }
        
        if (current_dim >= dimension_ || num_converged >= k) {
            converged = true;
        }
        prev_eigenvalues = eigenvalues;
        
        if (converged || outer == max_outer_iter_ - 1) {
            if (!converged) {
                std::cout << "\n  WARNING: Max restart cycles (" << max_outer_iter_
                          << ") reached. " << num_converged << "/" << k << " converged.\n"
                          << "  Consider: increasing max_iter, loosening tolerance, "
                          << "or using symmetries.\n";
            }
            break;
        }
        
        // === Phase 4: Thick restart ===
        // Keep k Ritz vectors as new basis via V_new = V * Q[:,0:k], where
        // Q = eigenvectors of the projected matrix (stored in d_H_proj after zheevd).
        {
            auto restart_start = std::chrono::high_resolution_clock::now();
            std::cout << "  Thick restart: keeping " << k << " Ritz vectors\n";
            
            cuDoubleComplex gemm_alpha = make_cuDoubleComplex(1.0, 0.0);
            cuDoubleComplex gemm_beta = make_cuDoubleComplex(0.0, 0.0);
            
            // V_new[:,0:k] = V * Q[:,0:k]
            CUBLAS_CHECK(cublasZgemm(cublas_handle_, CUBLAS_OP_N, CUBLAS_OP_N,
                                    dimension_, k, current_dim,
                                    &gemm_alpha,
                                    d_V, dimension_,
                                    d_H_proj, current_dim,
                                    &gemm_beta,
                                    d_V_restart, dimension_));
            
            // Copy back to d_V[:,0:k]
            CUDA_CHECK(cudaMemcpy(d_V, d_V_restart,
                                 static_cast<size_t>(dimension_) * k * sizeof(cuDoubleComplex),
                                 cudaMemcpyDeviceToDevice));
            
            // Reorthonormalize for numerical stability (CGS on k vectors)
            for (int i = 0; i < k; i++) {
                cuDoubleComplex* v_i = d_V + static_cast<size_t>(i) * dimension_;
                for (int j = 0; j < i; j++) {
                    cuDoubleComplex* v_j = d_V + static_cast<size_t>(j) * dimension_;
                    cuDoubleComplex dot;
                    CUBLAS_CHECK(cublasZdotc(cublas_handle_, dimension_, v_j, 1, v_i, 1, &dot));
                    cuDoubleComplex neg = make_cuDoubleComplex(-cuCreal(dot), -cuCimag(dot));
                    CUBLAS_CHECK(cublasZaxpy(cublas_handle_, dimension_, &neg, v_j, 1, v_i, 1));
                }
                double nrm;
                CUBLAS_CHECK(cublasDznrm2(cublas_handle_, dimension_, v_i, 1, &nrm));
                if (nrm > 1e-15) {
                    cuDoubleComplex sc = make_cuDoubleComplex(1.0 / nrm, 0.0);
                    CUBLAS_CHECK(cublasZscal(cublas_handle_, dimension_, &sc, v_i, 1));
                }
            }
            
            // Reset Hessenberg: diagonal = eigenvalues of kept pairs
            std::fill(h_H_m.begin(), h_H_m.end(), std::complex<double>(0.0, 0.0));
            for (int i = 0; i < k; i++) {
                h_H_m[i + static_cast<size_t>(i) * m] =
                    std::complex<double>(evals_all[i], 0.0);
            }
            current_dim = k;
            
            auto restart_end = std::chrono::high_resolution_clock::now();
            stats_.restart_time += std::chrono::duration<double>(restart_end - restart_start).count();
        }
    }
    
    // ===================== EIGENVECTORS =====================
    int num_eigs = std::min(k, static_cast<int>(eigenvalues.size()));
    
    if (compute_vectors && num_eigs > 0) {
        std::cout << "  Computing eigenvectors...\n";
        
        std::vector<std::complex<double>> h_evec_coefs(
            static_cast<size_t>(current_dim) * current_dim);
        CUDA_CHECK(cudaMemcpy(h_evec_coefs.data(), d_H_proj,
                             static_cast<size_t>(current_dim) * current_dim * sizeof(cuDoubleComplex),
                             cudaMemcpyDeviceToHost));
        
        eigenvectors.resize(num_eigs);
        std::vector<cuDoubleComplex> h_eigvec(dimension_);
        
        for (int e = 0; e < num_eigs; e++) {
            CUDA_CHECK(cudaMemcpy(d_coefs,
                                 &h_evec_coefs[static_cast<size_t>(e) * current_dim],
                                 current_dim * sizeof(cuDoubleComplex),
                                 cudaMemcpyHostToDevice));
            
            cuDoubleComplex alp = make_cuDoubleComplex(1.0, 0.0);
            cuDoubleComplex bet = make_cuDoubleComplex(0.0, 0.0);
            CUBLAS_CHECK(cublasZgemv(cublas_handle_, CUBLAS_OP_N,
                                    dimension_, current_dim,
                                    &alp, d_V, dimension_,
                                    d_coefs, 1,
                                    &bet, d_eigvec, 1));
            
            double nrm;
            CUBLAS_CHECK(cublasDznrm2(cublas_handle_, dimension_, d_eigvec, 1, &nrm));
            if (nrm > tolerance_) {
                cuDoubleComplex sc = make_cuDoubleComplex(1.0 / nrm, 0.0);
                CUBLAS_CHECK(cublasZscal(cublas_handle_, dimension_, &sc, d_eigvec, 1));
            }
            
            eigenvectors[e].resize(dimension_);
            CUDA_CHECK(cudaMemcpy(h_eigvec.data(), d_eigvec,
                                 dimension_ * sizeof(cuDoubleComplex),
                                 cudaMemcpyDeviceToHost));
            for (int i = 0; i < dimension_; i++) {
                eigenvectors[e][i] = std::complex<double>(
                    cuCreal(h_eigvec[i]), cuCimag(h_eigvec[i]));
            }
        }
    }
    
    auto overall_end = std::chrono::high_resolution_clock::now();
    stats_.total_time = std::chrono::duration<double>(overall_end - overall_start).count();
    
    // ===================== PRINT RESULTS =====================
    std::cout << "\n========================================\n";
    std::cout << "GPU Block Krylov-Schur Results\n";
    std::cout << "========================================\n";
    std::cout << "Converged: " << (converged ? "YES" : "NO") << "\n";
    std::cout << "Restart cycles: " << stats_.outer_iterations << "\n";
    std::cout << "Total mat-vecs: " << stats_.total_block_steps << "\n";
    std::cout << "Final subspace: " << current_dim << "\n";
    std::cout << "\nTiming:\n";
    std::cout << "  Krylov expansion: " << stats_.matvec_time << " s\n";
    std::cout << "  Eigenproblem: " << stats_.schur_time << " s\n";
    std::cout << "  Restart: " << stats_.restart_time << " s\n";
    std::cout << "  Total: " << stats_.total_time << " s\n";
    std::cout << "\nLowest eigenvalues:\n";
    for (int i = 0; i < std::min(5, static_cast<int>(eigenvalues.size())); i++) {
        std::cout << "  E[" << i << "] = " << std::fixed << std::setprecision(10)
                  << eigenvalues[i] << "\n";
    }
    if (eigenvalues.size() > 5)
        std::cout << "  ... (" << eigenvalues.size() - 5 << " more)\n";
    std::cout << "========================================\n";
    
    // ===================== CLEANUP =====================
    cudaFree(d_V);
    cudaFree(d_Hv);
    cudaFree(d_h_overlaps);
    cudaFree(d_V_restart);
    cudaFree(d_H_proj);
    cudaFree(d_evals);
    cudaFree(d_eigsolve_work);
    cudaFree(d_tau);
    cudaFree(d_qr_work);
    cudaFree(d_info);
    cudaFree(d_coefs);
    cudaFree(d_eigvec);
}

#endif // WITH_CUDA
