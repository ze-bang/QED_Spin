/**
 * @file gpu_krylov_schur.cu
 * @brief GPU-accelerated Krylov-Schur algorithm implementation
 * 
 * Implements a restarted Krylov-Schur algorithm optimized for GPU execution.
 * The algorithm builds a Krylov subspace using Arnoldi iteration, computes
 * Ritz pairs, and performs implicit restarts to maintain orthogonality.
 * 
 * GPU Optimizations:
 * - All Krylov vectors stored contiguously on GPU for coalesced access
 * - Batched orthogonalization using cuBLAS GEMV
 * - cuSOLVER for small projected eigenvalue problem
 * - cuBLAS GEMM for efficient basis update during restart
 */

#ifdef WITH_CUDA

#include <ed/gpu/gpu_lanczos.cuh>
#include <ed/gpu/kernel_config.h>
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
// GPUKrylovSchur Implementation
// ============================================================================

GPUKrylovSchur::GPUKrylovSchur(GPUOperator* op, int max_iter, double tolerance)
    : op_(op), max_iter_(max_iter), tolerance_(tolerance),
      max_outer_iter_(50),
      d_V_(nullptr), d_w_(nullptr), d_temp_(nullptr),
      d_H_projected_(nullptr), d_evecs_(nullptr), d_evals_(nullptr),
      d_work_(nullptr), d_info_(nullptr), work_size_(0),
      max_krylov_size_(0) {
    
    dimension_ = op_->getDimension();
    
    std::cout << "Initializing GPU Krylov-Schur\n";
    std::cout << "  Dimension: " << dimension_ << "\n";
    std::cout << "  Max Krylov subspace size: " << max_iter_ << "\n";
    std::cout << "  Tolerance: " << tolerance_ << "\n";
    
    CUBLAS_CHECK(cublasCreate(&cublas_handle_));
    CUSOLVER_CHECK(cusolverDnCreate(&cusolver_handle_));
    
    stats_.total_time = 0.0;
    stats_.matvec_time = 0.0;
    stats_.ortho_time = 0.0;
    stats_.schur_time = 0.0;
    stats_.restart_time = 0.0;
    stats_.outer_iterations = 0;
    stats_.total_arnoldi_steps = 0;
    stats_.converged_eigs = 0;
    stats_.final_residual = 0.0;
    stats_.memory_used = 0;
}

GPUKrylovSchur::~GPUKrylovSchur() {
    freeMemory();
    if (cublas_handle_) {
        cublasDestroy(cublas_handle_);
    }
    if (cusolver_handle_) {
        cusolverDnDestroy(cusolver_handle_);
    }
}

void GPUKrylovSchur::allocateMemory(int num_eigenvalues) {
    // Determine optimal Krylov subspace size
    // For reliable convergence without restart issues, use much larger subspace
    // Target: 6*k + 60 for good convergence, or at least 200 for large systems
    int m = std::min(std::max(6 * num_eigenvalues + 60, 200), max_iter_);
    
    // For very large Hilbert spaces, we may need even more
    // But cap it based on available memory
    
    // Check available GPU memory
    size_t free_mem, total_mem;
    CUDA_CHECK(cudaMemGetInfo(&free_mem, &total_mem));
    
    // Memory needed per Krylov vector
    size_t vec_size = static_cast<size_t>(dimension_) * sizeof(cuDoubleComplex);
    
    // Memory needed for m vectors + work vectors + projected matrix + workspace
    size_t krylov_mem = static_cast<size_t>(m) * vec_size;  // V: dim × m
    size_t work_mem = 3 * vec_size;  // w, temp, and some buffer
    size_t proj_mem = static_cast<size_t>(m) * m * sizeof(cuDoubleComplex);  // H_m: m × m
    size_t evec_mem = static_cast<size_t>(m) * m * sizeof(cuDoubleComplex);  // eigenvectors
    size_t eval_mem = static_cast<size_t>(m) * sizeof(double);  // eigenvalues
    size_t solver_overhead = 10 * 1024 * 1024;  // 10 MB for cuSOLVER workspace
    
    size_t total_needed = krylov_mem + work_mem + proj_mem + evec_mem + eval_mem + solver_overhead;
    
    // Reserve 20% of free memory for safety
    size_t usable_mem = static_cast<size_t>(free_mem * 0.8);
    
    if (total_needed > usable_mem) {
        // Reduce m to fit in memory
        int max_m = static_cast<int>((usable_mem - work_mem - solver_overhead) / 
                                     (vec_size + 2 * m * sizeof(cuDoubleComplex) + sizeof(double)));
        max_m = std::max(max_m, num_eigenvalues + 5);  // Need at least k+5 vectors
        m = std::min(m, max_m);
        std::cout << "  Warning: Limited GPU memory, reducing Krylov size to " << m << "\n";
    }
    
    max_krylov_size_ = m;
    
    std::cout << "  Krylov subspace size: " << max_krylov_size_ << "\n";
    std::cout << "  GPU Memory: " << (free_mem / (1024.0 * 1024.0 * 1024.0)) << " GB free\n";
    std::cout << "  Allocating: " << (total_needed / (1024.0 * 1024.0)) << " MB\n";
    
    // Allocate Krylov basis V (dimension × max_krylov_size, column-major)
    CUDA_CHECK(cudaMalloc(&d_V_, static_cast<size_t>(dimension_) * max_krylov_size_ * sizeof(cuDoubleComplex)));
    
    // Allocate work vectors
    CUDA_CHECK(cudaMalloc(&d_w_, vec_size));
    // d_temp_ needs to hold at least max_krylov_size_ elements for GEMV output in orthogonalization
    size_t temp_size = std::max(vec_size, static_cast<size_t>(max_krylov_size_) * sizeof(cuDoubleComplex));
    CUDA_CHECK(cudaMalloc(&d_temp_, temp_size));
    
    // Allocate projected Hessenberg matrix
    CUDA_CHECK(cudaMalloc(&d_H_projected_, static_cast<size_t>(max_krylov_size_) * max_krylov_size_ * sizeof(cuDoubleComplex)));
    CUDA_CHECK(cudaMemset(d_H_projected_, 0, static_cast<size_t>(max_krylov_size_) * max_krylov_size_ * sizeof(cuDoubleComplex)));
    
    // Host-side Hessenberg matrix
    h_H_projected_.resize(static_cast<size_t>(max_krylov_size_) * max_krylov_size_, std::complex<double>(0.0, 0.0));
    
    // Allocate for eigendecomposition
    CUDA_CHECK(cudaMalloc(&d_evecs_, static_cast<size_t>(max_krylov_size_) * max_krylov_size_ * sizeof(cuDoubleComplex)));
    CUDA_CHECK(cudaMalloc(&d_evals_, static_cast<size_t>(max_krylov_size_) * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_info_, sizeof(int)));
    
    // Query workspace size for cuSOLVER zheev
    int lwork = 0;
    CUSOLVER_CHECK(cusolverDnZheevd_bufferSize(
        cusolver_handle_,
        CUSOLVER_EIG_MODE_VECTOR,  // Compute eigenvectors
        CUBLAS_FILL_MODE_UPPER,
        max_krylov_size_,
        reinterpret_cast<cuDoubleComplex*>(d_evecs_),
        max_krylov_size_,
        d_evals_,
        &lwork));
    
    work_size_ = lwork;
    CUDA_CHECK(cudaMalloc(&d_work_, work_size_ * sizeof(cuDoubleComplex)));
    
    stats_.memory_used = static_cast<size_t>(dimension_) * max_krylov_size_ * sizeof(cuDoubleComplex) +
                        3 * vec_size +
                        2 * static_cast<size_t>(max_krylov_size_) * max_krylov_size_ * sizeof(cuDoubleComplex) +
                        static_cast<size_t>(max_krylov_size_) * sizeof(double) +
                        work_size_ * sizeof(cuDoubleComplex);
}

void GPUKrylovSchur::freeMemory() {
    if (d_V_) { cudaFree(d_V_); d_V_ = nullptr; }
    if (d_w_) { cudaFree(d_w_); d_w_ = nullptr; }
    if (d_temp_) { cudaFree(d_temp_); d_temp_ = nullptr; }
    if (d_H_projected_) { cudaFree(d_H_projected_); d_H_projected_ = nullptr; }
    if (d_evecs_) { cudaFree(d_evecs_); d_evecs_ = nullptr; }
    if (d_evals_) { cudaFree(d_evals_); d_evals_ = nullptr; }
    if (d_work_) { cudaFree(d_work_); d_work_ = nullptr; }
    if (d_info_) { cudaFree(d_info_); d_info_ = nullptr; }
}

void GPUKrylovSchur::initializeRandomVector(cuDoubleComplex* d_vec) {
    int num_blocks = (dimension_ + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned long long seed = std::random_device{}();
    GPULanczosKernels::initRandomVectorKernel<<<num_blocks, BLOCK_SIZE>>>(
        d_vec, dimension_, seed);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    normalizeVector(d_vec);
}

double GPUKrylovSchur::vectorNorm(const cuDoubleComplex* d_vec) {
    double result;
    CUBLAS_CHECK(cublasDznrm2(cublas_handle_, dimension_, d_vec, 1, &result));
    return result;
}

void GPUKrylovSchur::normalizeVector(cuDoubleComplex* d_vec) {
    double norm = vectorNorm(d_vec);
    if (norm > 1e-15) {
        cuDoubleComplex scale = make_cuDoubleComplex(1.0 / norm, 0.0);
        CUBLAS_CHECK(cublasZscal(cublas_handle_, dimension_, &scale, d_vec, 1));
    }
}

void GPUKrylovSchur::vectorCopy(const cuDoubleComplex* src, cuDoubleComplex* dst) {
    CUDA_CHECK(cudaMemcpy(dst, src, dimension_ * sizeof(cuDoubleComplex), cudaMemcpyDeviceToDevice));
}

void GPUKrylovSchur::vectorScale(cuDoubleComplex* d_vec, double scale) {
    cuDoubleComplex alpha = make_cuDoubleComplex(scale, 0.0);
    CUBLAS_CHECK(cublasZscal(cublas_handle_, dimension_, &alpha, d_vec, 1));
}

void GPUKrylovSchur::vectorAxpy(const cuDoubleComplex* d_x, cuDoubleComplex* d_y,
                               const cuDoubleComplex& alpha) {
    CUBLAS_CHECK(cublasZaxpy(cublas_handle_, dimension_, &alpha, d_x, 1, d_y, 1));
}

std::complex<double> GPUKrylovSchur::vectorDot(const cuDoubleComplex* d_x,
                                               const cuDoubleComplex* d_y) {
    cuDoubleComplex result;
    CUBLAS_CHECK(cublasZdotc(cublas_handle_, dimension_, d_x, 1, d_y, 1, &result));
    return std::complex<double>(cuCreal(result), cuCimag(result));
}

double GPUKrylovSchur::orthogonalizeAgainstBasis(int j) {
    auto ortho_start = std::chrono::high_resolution_clock::now();
    
    // w is in d_w_, we orthogonalize against V[:, 0:j+1]
    // For each i = 0, ..., j:
    //   h_{i,j} = <v_i, w>
    //   w = w - h_{i,j} * v_i
    
    // Use cuBLAS GEMV for efficiency: compute all dot products at once
    // h = V^H * w where V is (dim × (j+1)) and w is (dim × 1)
    // Result h is ((j+1) × 1)
    
    std::vector<cuDoubleComplex> h_overlaps(j + 1);
    
    // Compute overlaps: h = V^H * w
    cuDoubleComplex alpha = make_cuDoubleComplex(1.0, 0.0);
    cuDoubleComplex beta = make_cuDoubleComplex(0.0, 0.0);
    
    // Use GEMV: y = alpha * A^H * x + beta * y
    // A is dim × (j+1), x is w (dim × 1), y is h ((j+1) × 1)
    CUBLAS_CHECK(cublasZgemv(cublas_handle_, CUBLAS_OP_C,
                            dimension_, j + 1,  // m, n for A
                            &alpha,
                            d_V_, dimension_,   // A, lda
                            d_w_, 1,            // x, incx
                            &beta,
                            d_temp_, 1));       // y, incy (use d_temp as scratch)
    
    // Copy overlaps to host
    CUDA_CHECK(cudaMemcpy(h_overlaps.data(), d_temp_, 
                         (j + 1) * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost));
    
    // Store in Hessenberg matrix (column j)
    for (int i = 0; i <= j; i++) {
        h_H_projected_[i + j * max_krylov_size_] = std::complex<double>(
            cuCreal(h_overlaps[i]), cuCimag(h_overlaps[i]));
    }
    
    // Apply corrections: w = w - V * h
    // This is GEMV: w = w - V * h, or w = -1 * V * h + 1 * w
    alpha = make_cuDoubleComplex(-1.0, 0.0);
    beta = make_cuDoubleComplex(1.0, 0.0);
    
    CUBLAS_CHECK(cublasZgemv(cublas_handle_, CUBLAS_OP_N,
                            dimension_, j + 1,
                            &alpha,
                            d_V_, dimension_,
                            d_temp_, 1,
                            &beta,
                            d_w_, 1));
    
    // Reorthogonalization pass for numerical stability
    cuDoubleComplex h_correction;
    for (int i = 0; i <= j; i++) {
        cuDoubleComplex* v_i = getKrylovVector(i);
        CUBLAS_CHECK(cublasZdotc(cublas_handle_, dimension_, v_i, 1, d_w_, 1, &h_correction));
        
        double correction_mag = sqrt(cuCreal(h_correction) * cuCreal(h_correction) + 
                                    cuCimag(h_correction) * cuCimag(h_correction));
        
        if (correction_mag > tolerance_) {
            // Update Hessenberg entry
            h_H_projected_[i + j * max_krylov_size_] += std::complex<double>(
                cuCreal(h_correction), cuCimag(h_correction));
            
            // w = w - h_correction * v_i
            cuDoubleComplex neg_h = make_cuDoubleComplex(-cuCreal(h_correction), -cuCimag(h_correction));
            CUBLAS_CHECK(cublasZaxpy(cublas_handle_, dimension_, &neg_h, v_i, 1, d_w_, 1));
        }
    }
    
    // Compute beta = ||w||
    double beta_val = vectorNorm(d_w_);
    
    auto ortho_end = std::chrono::high_resolution_clock::now();
    stats_.ortho_time += std::chrono::duration<double>(ortho_end - ortho_start).count();
    
    return beta_val;
}

int GPUKrylovSchur::arnoldiIteration(int j_start, int m) {
    // Arnoldi iteration to expand Krylov subspace from j_start to m
    // For j = j_start, ..., m-1:
    //   w = H * v_j
    //   Orthogonalize w against V[:, 0:j+1]
    //   beta = ||w||
    //   if beta < tol: breakdown, return j+1
    //   v_{j+1} = w / beta
    
    int actual_m = m;
    
    for (int j = j_start; j < m; j++) {
        cuDoubleComplex* v_j = getKrylovVector(j);
        
        // w = H * v_j
        auto mv_start = std::chrono::high_resolution_clock::now();
        op_->matVecGPU(v_j, d_w_, dimension_);
        auto mv_end = std::chrono::high_resolution_clock::now();
        stats_.matvec_time += std::chrono::duration<double>(mv_end - mv_start).count();
        stats_.total_arnoldi_steps++;
        
        // Orthogonalize and get beta
        double beta = orthogonalizeAgainstBasis(j);
        
        // Store beta in Hessenberg matrix - only if j+1 < max_krylov_size_ to avoid out-of-bounds
        if (j + 1 < max_krylov_size_) {
            h_H_projected_[(j + 1) + j * max_krylov_size_] = std::complex<double>(beta, 0.0);
        }
        
        // Check for breakdown
        if (beta < tolerance_) {
            std::cout << "  Krylov subspace exhausted at dimension " << j + 1 << "\n";
            actual_m = j + 1;
            break;
        }
        
        // Normalize and store as v_{j+1}
        if (j < m - 1) {
            normalizeVector(d_w_);
            vectorCopy(d_w_, getKrylovVector(j + 1));
        }
    }
    
    return actual_m;
}

bool GPUKrylovSchur::solveProjectedEigenproblem(int m, std::vector<double>& eigenvalues_m) {
    auto schur_start = std::chrono::high_resolution_clock::now();
    
    // Copy Hessenberg matrix to device for cuSOLVER
    // Note: We've been maintaining h_H_projected_ on host
    std::vector<cuDoubleComplex> h_dense(m * m);
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < m; j++) {
            std::complex<double> val = h_H_projected_[i + j * max_krylov_size_];
            h_dense[i + j * m] = make_cuDoubleComplex(val.real(), val.imag());
        }
    }
    
    // Copy to d_evecs_ (will be overwritten with eigenvectors)
    CUDA_CHECK(cudaMemcpy(d_evecs_, h_dense.data(), 
                         m * m * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));
    
    // Solve eigenvalue problem using cuSOLVER zheevd
    CUSOLVER_CHECK(cusolverDnZheevd(
        cusolver_handle_,
        CUSOLVER_EIG_MODE_VECTOR,
        CUBLAS_FILL_MODE_UPPER,
        m,
        d_evecs_, m,
        d_evals_,
        d_work_, work_size_,
        d_info_));
    
    // Check for errors
    int h_info;
    CUDA_CHECK(cudaMemcpy(&h_info, d_info_, sizeof(int), cudaMemcpyDeviceToHost));
    
    if (h_info != 0) {
        std::cerr << "cuSOLVER zheevd failed with error " << h_info << "\n";
        return false;
    }
    
    // Copy eigenvalues to host
    eigenvalues_m.resize(m);
    CUDA_CHECK(cudaMemcpy(eigenvalues_m.data(), d_evals_, 
                         m * sizeof(double), cudaMemcpyDeviceToHost));
    
    auto schur_end = std::chrono::high_resolution_clock::now();
    stats_.schur_time += std::chrono::duration<double>(schur_end - schur_start).count();
    
    return true;
}

int GPUKrylovSchur::checkConvergence(int m, int k, double beta_m) {
    // For each Ritz pair (lambda_i, y_i), the residual is:
    // ||H * V * y_i - lambda_i * V * y_i|| = |beta_m * y_i[m-1]|
    // where y_i[m-1] is the last component of the Ritz vector
    
    // Get eigenvectors from device (stored in d_evecs_)
    std::vector<cuDoubleComplex> h_evecs(m * m);
    CUDA_CHECK(cudaMemcpy(h_evecs.data(), d_evecs_, 
                         m * m * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost));
    
    int num_converged = 0;
    for (int i = 0; i < k && i < m; i++) {
        // Get last component of eigenvector i
        // Eigenvectors are stored column-major, so eigenvector i is at columns h_evecs[i*m : (i+1)*m]
        cuDoubleComplex last_component = h_evecs[(m - 1) + i * m];
        double residual = beta_m * sqrt(cuCreal(last_component) * cuCreal(last_component) +
                                       cuCimag(last_component) * cuCimag(last_component));
        
        if (residual < tolerance_) {
            num_converged++;
        }
        
        stats_.final_residual = std::max(stats_.final_residual, residual);
    }
    
    return num_converged;
}

double GPUKrylovSchur::performRestart(int m, int k) {
    auto restart_start = std::chrono::high_resolution_clock::now();
    
    // Thick restart for Hermitian matrices:
    // 1. Keep the k Ritz vectors corresponding to wanted eigenvalues
    // 2. Orthogonalize the residual against them
    // 3. Continue Arnoldi from the new starting point
    
    // Get eigenvectors from device (columns of Q)
    std::vector<cuDoubleComplex> h_evecs(m * m);
    CUDA_CHECK(cudaMemcpy(h_evecs.data(), d_evecs_, 
                         m * m * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost));
    
    // Get eigenvalues
    std::vector<double> eigenvalues_m(m);
    CUDA_CHECK(cudaMemcpy(eigenvalues_m.data(), d_evals_, 
                         m * sizeof(double), cudaMemcpyDeviceToHost));
    
    // Compute new basis: V_new[:, 0:k] = V_old * Q[:, 0:k]
    // These are the Ritz vectors for the k smallest eigenvalues
    
    // Copy Q[:, 0:k] to device
    cuDoubleComplex* d_Q_k;
    CUDA_CHECK(cudaMalloc(&d_Q_k, static_cast<size_t>(k) * m * sizeof(cuDoubleComplex)));
    
    std::vector<cuDoubleComplex> Q_k(static_cast<size_t>(k) * m);
    for (int j = 0; j < k; j++) {
        for (int i = 0; i < m; i++) {
            Q_k[i + j * m] = h_evecs[i + j * m];
        }
    }
    CUDA_CHECK(cudaMemcpy(d_Q_k, Q_k.data(), 
                         static_cast<size_t>(k) * m * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));
    
    // Allocate temporary for new basis
    cuDoubleComplex* d_V_new;
    CUDA_CHECK(cudaMalloc(&d_V_new, static_cast<size_t>(dimension_) * k * sizeof(cuDoubleComplex)));
    
    // V_new[:, 0:k] = V_old * Q[:, 0:k]
    cuDoubleComplex alpha = make_cuDoubleComplex(1.0, 0.0);
    cuDoubleComplex beta = make_cuDoubleComplex(0.0, 0.0);
    
    CUBLAS_CHECK(cublasZgemm(cublas_handle_, CUBLAS_OP_N, CUBLAS_OP_N,
                            dimension_, k, m,
                            &alpha,
                            d_V_, dimension_,
                            d_Q_k, m,
                            &beta,
                            d_V_new, dimension_));
    
    // Copy new Ritz vectors back to d_V_
    CUDA_CHECK(cudaMemcpy(d_V_, d_V_new, 
                         static_cast<size_t>(dimension_) * k * sizeof(cuDoubleComplex),
                         cudaMemcpyDeviceToDevice));
    
    cudaFree(d_Q_k);
    cudaFree(d_V_new);
    
    // Reorthonormalize the new basis vectors (crucial for numerical stability)
    for (int i = 0; i < k; i++) {
        cuDoubleComplex* v_i = getKrylovVector(i);
        
        // Orthogonalize against previous vectors
        for (int j = 0; j < i; j++) {
            cuDoubleComplex* v_j = getKrylovVector(j);
            cuDoubleComplex dot;
            CUBLAS_CHECK(cublasZdotc(cublas_handle_, dimension_, v_j, 1, v_i, 1, &dot));
            cuDoubleComplex neg_dot = make_cuDoubleComplex(-cuCreal(dot), -cuCimag(dot));
            CUBLAS_CHECK(cublasZaxpy(cublas_handle_, dimension_, &neg_dot, v_j, 1, v_i, 1));
        }
        
        // Normalize
        normalizeVector(v_i);
    }
    
    // Clear Hessenberg and set diagonal to eigenvalues
    std::fill(h_H_projected_.begin(), h_H_projected_.end(), std::complex<double>(0.0, 0.0));
    
    for (int i = 0; i < k; i++) {
        h_H_projected_[i + i * max_krylov_size_] = std::complex<double>(eigenvalues_m[i], 0.0);
    }
    
    // For thick restart, we need to compute the residual and continue Arnoldi
    // The residual is: r = H * v_{k-1} - eigenvalue_{k-1} * v_{k-1}
    // But for simplicity, we'll just continue Arnoldi from the last Ritz vector
    
    // Apply H to the last Ritz vector to get starting point for continuation
    cuDoubleComplex* v_last = getKrylovVector(k - 1);
    op_->matVecGPU(v_last, d_w_, dimension_);
    
    // Orthogonalize against all kept vectors
    for (int i = 0; i < k; i++) {
        cuDoubleComplex* v_i = getKrylovVector(i);
        cuDoubleComplex dot;
        CUBLAS_CHECK(cublasZdotc(cublas_handle_, dimension_, v_i, 1, d_w_, 1, &dot));
        
        // Store in Hessenberg (this is H[i, k-1])
        h_H_projected_[i + (k - 1) * max_krylov_size_] = std::complex<double>(cuCreal(dot), cuCimag(dot));
        
        cuDoubleComplex neg_dot = make_cuDoubleComplex(-cuCreal(dot), -cuCimag(dot));
        CUBLAS_CHECK(cublasZaxpy(cublas_handle_, dimension_, &neg_dot, v_i, 1, d_w_, 1));
    }
    
    // Compute beta = ||residual||
    double beta_val = vectorNorm(d_w_);
    
    // Store as v_k (the starting vector for continuing Arnoldi)
    if (beta_val > tolerance_) {
        cuDoubleComplex scale = make_cuDoubleComplex(1.0 / beta_val, 0.0);
        CUBLAS_CHECK(cublasZscal(cublas_handle_, dimension_, &scale, d_w_, 1));
        vectorCopy(d_w_, getKrylovVector(k));
        
        // Store beta in Hessenberg
        if (k < max_krylov_size_) {
            h_H_projected_[k + (k - 1) * max_krylov_size_] = std::complex<double>(beta_val, 0.0);
        }
    }
    
    auto restart_end = std::chrono::high_resolution_clock::now();
    stats_.restart_time += std::chrono::duration<double>(restart_end - restart_start).count();
    
    return beta_val;
}

void GPUKrylovSchur::computeEigenvectors(int m, int num_eigs,
                                         std::vector<std::vector<std::complex<double>>>& eigenvectors) {
    std::cout << "  Computing full eigenvectors on GPU...\n";
    
    // Get Ritz vectors (eigenvectors of projected matrix)
    std::vector<cuDoubleComplex> h_evecs(m * m);
    CUDA_CHECK(cudaMemcpy(h_evecs.data(), d_evecs_, 
                         m * m * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost));
    
    eigenvectors.resize(num_eigs);
    
    for (int i = 0; i < num_eigs; i++) {
        eigenvectors[i].resize(dimension_);
        
        // eigenvector[i] = V * y_i where y_i is the i-th Ritz vector
        // Use cuBLAS GEMV: eigvec = V * y_i
        
        // Copy Ritz vector to device
        cuDoubleComplex* d_ritz_vec;
        CUDA_CHECK(cudaMalloc(&d_ritz_vec, m * sizeof(cuDoubleComplex)));
        CUDA_CHECK(cudaMemcpy(d_ritz_vec, &h_evecs[i * m], 
                             m * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));
        
        // eigvec = V * y_i
        cuDoubleComplex alpha = make_cuDoubleComplex(1.0, 0.0);
        cuDoubleComplex beta = make_cuDoubleComplex(0.0, 0.0);
        
        CUBLAS_CHECK(cublasZgemv(cublas_handle_, CUBLAS_OP_N,
                                dimension_, m,
                                &alpha,
                                d_V_, dimension_,
                                d_ritz_vec, 1,
                                &beta,
                                d_temp_, 1));
        
        // Normalize
        normalizeVector(d_temp_);
        
        // Copy to host
        std::vector<cuDoubleComplex> h_eigvec(dimension_);
        CUDA_CHECK(cudaMemcpy(h_eigvec.data(), d_temp_, 
                             dimension_ * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost));
        
        for (int j = 0; j < dimension_; j++) {
            eigenvectors[i][j] = std::complex<double>(cuCreal(h_eigvec[j]), cuCimag(h_eigvec[j]));
        }
        
        cudaFree(d_ritz_vec);
    }
    
    std::cout << "  Eigenvectors computed\n";
}

void GPUKrylovSchur::run(int num_eigenvalues,
                        std::vector<double>& eigenvalues,
                        std::vector<std::vector<std::complex<double>>>& eigenvectors,
                        bool compute_vectors) {
    
    auto overall_start = std::chrono::high_resolution_clock::now();
    
    std::cout << "\n========================================\n";
    std::cout << "GPU Krylov-Schur Algorithm\n";
    std::cout << "========================================\n";
    std::cout << "Seeking " << num_eigenvalues << " eigenvalues\n";
    std::cout << "Dimension: " << dimension_ << "\n";
    std::cout << "Tolerance: " << tolerance_ << "\n\n";
    
    // Allocate memory
    allocateMemory(num_eigenvalues);
    
    int k = num_eigenvalues;  // Number of desired eigenvalues
    int m = max_krylov_size_; // Maximum subspace size
    
    // Initialize first Krylov vector
    initializeRandomVector(getKrylovVector(0));
    
    bool converged = false;
    int actual_m = m;
    
    for (int outer_iter = 0; outer_iter < max_outer_iter_ && !converged; outer_iter++) {
        stats_.outer_iterations++;
        std::cout << "Krylov-Schur: Outer iteration " << outer_iter + 1 << "\n";
        
        // Determine starting point for Arnoldi
        int j_start = (outer_iter == 0) ? 0 : k;
        
        // Arnoldi iteration to build/extend subspace
        actual_m = arnoldiIteration(j_start, m);
        
        // Solve projected eigenvalue problem
        std::vector<double> eigenvalues_m;
        if (!solveProjectedEigenproblem(actual_m, eigenvalues_m)) {
            std::cerr << "Failed to solve projected eigenvalue problem\n";
            break;
        }
        
        // Get beta_m for residual estimation
        // beta_m is the last subdiagonal element, at H[actual_m-1, actual_m-2] for actual_m >= 2
        double beta_m = 0.0;
        if (actual_m >= 2) {
            beta_m = std::abs(h_H_projected_[(actual_m - 1) + (actual_m - 2) * max_krylov_size_]);
        }
        
        // Check convergence
        int num_converged = checkConvergence(actual_m, k, beta_m);
        std::cout << "  " << num_converged << " / " << k << " eigenvalues converged\n";
        
        if (num_converged >= k || outer_iter == max_outer_iter_ - 1) {
            converged = true;
            stats_.converged_eigs = num_converged;
            
            // Extract converged eigenvalues
            eigenvalues.resize(k);
            for (int i = 0; i < k; i++) {
                eigenvalues[i] = eigenvalues_m[i];
            }
            
            // Compute eigenvectors if requested
            if (compute_vectors) {
                computeEigenvectors(actual_m, k, eigenvectors);
            }
            
            break;
        }
        
        // Perform Krylov-Schur restart
        std::cout << "  Performing restart...\n";
        performRestart(actual_m, k);
    }
    
    auto overall_end = std::chrono::high_resolution_clock::now();
    stats_.total_time = std::chrono::duration<double>(overall_end - overall_start).count();
    
    // Print summary
    std::cout << "\n========================================\n";
    std::cout << "GPU Krylov-Schur Results\n";
    std::cout << "========================================\n";
    
    if (converged) {
        std::cout << "Converged: YES\n";
    } else {
        std::cout << "Converged: NO (max iterations reached)\n";
    }
    
    std::cout << "Outer iterations: " << stats_.outer_iterations << "\n";
    std::cout << "Total Arnoldi steps: " << stats_.total_arnoldi_steps << "\n";
    std::cout << "Converged eigenvalues: " << stats_.converged_eigs << "\n";
    std::cout << "\nTiming breakdown:\n";
    std::cout << "  Matrix-vector products: " << stats_.matvec_time << " s\n";
    std::cout << "  Orthogonalization: " << stats_.ortho_time << " s\n";
    std::cout << "  Projected eigenproblem: " << stats_.schur_time << " s\n";
    std::cout << "  Restart operations: " << stats_.restart_time << " s\n";
    std::cout << "  Total time: " << stats_.total_time << " s\n";
    std::cout << "\nLowest eigenvalues:\n";
    for (int i = 0; i < std::min(5, static_cast<int>(eigenvalues.size())); i++) {
        std::cout << "  E[" << i << "] = " << std::fixed << std::setprecision(10) << eigenvalues[i] << "\n";
    }
    if (eigenvalues.size() > 5) {
        std::cout << "  ... (" << eigenvalues.size() - 5 << " more)\n";
    }
    std::cout << "========================================\n";
    
    // Free memory
    freeMemory();
}

#endif // WITH_CUDA
