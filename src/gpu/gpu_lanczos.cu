#ifdef WITH_CUDA

#include "gpu_lanczos.cuh"
#include "kernel_config.h"
#include <iostream>
#include <cmath>
#include <random>
#include <algorithm>
#include <chrono>
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <curand_kernel.h>

using namespace GPUConfig;

// ============================================================================
// GPU Lanczos Kernels
// ============================================================================

namespace GPULanczosKernels {

__global__ void initRandomVectorKernel(cuDoubleComplex* vec, int N, unsigned long long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    
    curandState state;
    curand_init(seed, idx, 0, &state);
    
    double real_part = curand_normal_double(&state);
    double imag_part = curand_normal_double(&state);
    
    vec[idx] = make_cuDoubleComplex(real_part, imag_part);
}

__global__ void vectorAddKernel(const cuDoubleComplex* x, const cuDoubleComplex* y,
                               cuDoubleComplex* result, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    
    result[idx] = cuCadd(x[idx], y[idx]);
}

__global__ void vectorSubKernel(const cuDoubleComplex* x, const cuDoubleComplex* y,
                               cuDoubleComplex* result, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    
    result[idx] = cuCsub(x[idx], y[idx]);
}

__global__ void vectorScaleKernel(cuDoubleComplex* vec, double scale, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    
    vec[idx] = make_cuDoubleComplex(
        cuCreal(vec[idx]) * scale,
        cuCimag(vec[idx]) * scale
    );
}

__global__ void vectorAxpyKernel(const cuDoubleComplex* x, cuDoubleComplex* y,
                                cuDoubleComplex alpha, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    
    cuDoubleComplex ax = cuCmul(alpha, x[idx]);
    y[idx] = cuCadd(y[idx], ax);
}

} // namespace GPULanczosKernels

// ============================================================================
// GPULanczos Implementation
// ============================================================================

GPULanczos::GPULanczos(GPUOperator* op, int max_iter, double tolerance)
    : op_(op), max_iter_(max_iter), tolerance_(tolerance),
      d_v_current_(nullptr), d_v_prev_(nullptr), d_w_(nullptr), d_temp_(nullptr),
      d_lanczos_vectors_(nullptr), num_stored_vectors_(0) {
    
    dimension_ = op_->getDimension();
    
    std::cout << "Initializing GPU Lanczos\n";
    std::cout << "  Dimension: " << dimension_ << "\n";
    std::cout << "  Max iterations: " << max_iter_ << "\n";
    
    CUBLAS_CHECK(cublasCreate(&cublas_handle_));
    
    allocateMemory();
    
    stats_.total_time = 0.0;
    stats_.matvec_time = 0.0;
    stats_.ortho_time = 0.0;
    stats_.iterations = 0;
    stats_.convergence_error = 0.0;
}

GPULanczos::~GPULanczos() {
    freeMemory();
    if (cublas_handle_) {
        cublasDestroy(cublas_handle_);
    }
}

void GPULanczos::allocateMemory() {
    size_t vec_size = dimension_ * sizeof(cuDoubleComplex);
    
    CUDA_CHECK(cudaMalloc(&d_v_current_, vec_size));
    CUDA_CHECK(cudaMalloc(&d_v_prev_, vec_size));
    CUDA_CHECK(cudaMalloc(&d_w_, vec_size));
    CUDA_CHECK(cudaMalloc(&d_temp_, vec_size));
    
    // Check if we can store all Lanczos vectors
    size_t free_mem, total_mem;
    CUDA_CHECK(cudaMemGetInfo(&free_mem, &total_mem));
    
    size_t required_mem = max_iter_ * vec_size;
    if (required_mem < free_mem * 0.5) {
        // Allocate array of pointers for Lanczos vectors
        d_lanczos_vectors_ = new cuDoubleComplex*[max_iter_];
        for (int i = 0; i < max_iter_; ++i) {
            CUDA_CHECK(cudaMalloc(&d_lanczos_vectors_[i], vec_size));
        }
        num_stored_vectors_ = max_iter_;
        std::cout << "  Storing all Lanczos vectors on GPU\n";
    } else {
        std::cout << "  Using partial reorthogonalization (insufficient GPU memory)\n";
        num_stored_vectors_ = 0;
    }
    
    alpha_.reserve(max_iter_);
    beta_.reserve(max_iter_);
}

void GPULanczos::freeMemory() {
    if (d_v_current_) cudaFree(d_v_current_);
    if (d_v_prev_) cudaFree(d_v_prev_);
    if (d_w_) cudaFree(d_w_);
    if (d_temp_) cudaFree(d_temp_);
    
    if (d_lanczos_vectors_) {
        for (int i = 0; i < num_stored_vectors_; ++i) {
            if (d_lanczos_vectors_[i]) {
                cudaFree(d_lanczos_vectors_[i]);
            }
        }
        delete[] d_lanczos_vectors_;
    }
}

void GPULanczos::initializeRandomVector(cuDoubleComplex* d_vec) {
    int num_blocks = (dimension_ + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    unsigned long long seed = std::random_device{}();
    GPULanczosKernels::initRandomVectorKernel<<<num_blocks, BLOCK_SIZE>>>(
        d_vec, dimension_, seed);
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    normalizeVector(d_vec);
}

double GPULanczos::vectorNorm(const cuDoubleComplex* d_vec) {
    cuDoubleComplex result;
    CUBLAS_CHECK(cublasZdotc(cublas_handle_, dimension_,
                            d_vec, 1,
                            d_vec, 1,
                            &result));
    double norm_squared = cuCreal(result);
    return std::sqrt(std::abs(norm_squared));
}

void GPULanczos::normalizeVector(cuDoubleComplex* d_vec) {
    double norm = vectorNorm(d_vec);
    if (norm > 1e-15) {
        vectorScale(d_vec, 1.0 / norm);
    }
}

void GPULanczos::vectorCopy(const cuDoubleComplex* src, cuDoubleComplex* dst) {
    CUDA_CHECK(cudaMemcpy(dst, src, dimension_ * sizeof(cuDoubleComplex),
                        cudaMemcpyDeviceToDevice));
}

void GPULanczos::vectorScale(cuDoubleComplex* d_vec, double scale) {
    int num_blocks = (dimension_ + BLOCK_SIZE - 1) / BLOCK_SIZE;
    GPULanczosKernels::vectorScaleKernel<<<num_blocks, BLOCK_SIZE>>>(
        d_vec, scale, dimension_);
    CUDA_CHECK(cudaGetLastError());
}

std::complex<double> GPULanczos::vectorDot(const cuDoubleComplex* d_x,
                                          const cuDoubleComplex* d_y) {
    cuDoubleComplex result;
    CUBLAS_CHECK(cublasZdotc(cublas_handle_, dimension_,
                            d_x, 1, d_y, 1, &result));
    return std::complex<double>(cuCreal(result), cuCimag(result));
}

void GPULanczos::vectorAxpy(const cuDoubleComplex* d_x, cuDoubleComplex* d_y,
                           const cuDoubleComplex& alpha) {
    CUBLAS_CHECK(cublasZaxpy(cublas_handle_, dimension_,
                            &alpha, d_x, 1, d_y, 1));
}

// Adaptive selective reorthogonalization using Parlett-Simon criterion
void GPULanczos::orthogonalize(cuDoubleComplex* d_vec, int iter, 
                               std::vector<std::vector<double>>& omega,
                               const std::vector<double>& alpha,
                               const std::vector<double>& beta,
                               double ortho_threshold) {
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    
    if (num_stored_vectors_ > 0 && iter > 0) {
        // Update omega estimates for selective reorthogonalization
        const double eps = 2.22e-16; // Machine epsilon for double precision
        
        // Resize omega for new iteration
        omega.resize(iter + 1);
        for (int i = 0; i <= iter - 1; i++) {
            omega[iter].push_back(0.0);
        }
        omega[iter].push_back(eps);
        
        // Compute omega[iter][i] using Parlett-Simon recurrence
        for (int i = 0; i < iter; i++) {
            double contrib = eps * (std::abs(alpha[iter - 1]) + std::abs(beta[iter - 1]));
            if (iter > 1) {
                contrib += eps * std::abs(beta[iter - 2]) + std::abs(beta[iter - 1]) * omega[iter - 1][i];
            }
            omega[iter][i] = contrib;
        }
        
        // Determine which vectors need reorthogonalization
        std::vector<int> reorth_indices;
        for (int i = 0; i < iter; i++) {
            if (omega[iter][i] > ortho_threshold) {
                reorth_indices.push_back(i);
            }
        }
        
        // Decide between selective and full reorthogonalization
        bool do_full_reorth = (reorth_indices.size() > iter * 0.3);
        
        if (do_full_reorth) {
            // Full reorthogonalization
            for (int i = 0; i < iter; ++i) {
                std::complex<double> dot = vectorDot(d_lanczos_vectors_[i], d_vec);
                cuDoubleComplex neg_dot = make_cuDoubleComplex(-dot.real(), -dot.imag());
                vectorAxpy(d_lanczos_vectors_[i], d_vec, neg_dot);
            }
            
            // Reset omega after full reorthogonalization
            for (int i = 0; i < iter; i++) {
                omega[iter][i] = eps;
            }
            
            stats_.full_reorth_count++;
            stats_.total_reorth_ops += iter;
        } else if (!reorth_indices.empty()) {
            // Selective reorthogonalization
            for (int idx : reorth_indices) {
                std::complex<double> dot = vectorDot(d_lanczos_vectors_[idx], d_vec);
                cuDoubleComplex neg_dot = make_cuDoubleComplex(-dot.real(), -dot.imag());
                vectorAxpy(d_lanczos_vectors_[idx], d_vec, neg_dot);
                
                // Reset omega for reorthogonalized vectors
                omega[iter][idx] = eps;
            }
            
            stats_.selective_reorth_count++;
            stats_.total_reorth_ops += reorth_indices.size();
        }
    }
    
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    float milliseconds = 0;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    stats_.ortho_time += milliseconds / 1000.0;
    
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
}

void GPULanczos::run(int num_eigenvalues,
                    std::vector<double>& eigenvalues,
                    std::vector<std::vector<std::complex<double>>>& eigenvectors,
                    bool compute_vectors) {
    
    auto overall_start = std::chrono::high_resolution_clock::now();
    
    std::cout << "\nRunning GPU Lanczos with Adaptive Selective Reorthogonalization...\n";
    
    alpha_.clear();
    beta_.clear();
    
    // Initialize first Lanczos vector
    initializeRandomVector(d_v_current_);
    
    if (num_stored_vectors_ > 0) {
        vectorCopy(d_v_current_, d_lanczos_vectors_[0]);
        std::cout << "  Storing Lanczos vectors on GPU for selective reorthogonalization\n";
    } else {
        std::cout << "  Warning: Insufficient GPU memory for vector storage - using local orthogonalization only\n";
    }
    
    // Initialize previous vector to zero
    CUDA_CHECK(cudaMemset(d_v_prev_, 0, dimension_ * sizeof(cuDoubleComplex)));
    
    // For eigenvalue convergence checking
    std::vector<double> prev_eigenvalues;
    int check_convergence_interval = 10;  // Check every 10 iterations
    bool eigenvalues_converged = false;
    
    // Adaptive selective reorthogonalization (Parlett-Simon)
    const double eps = 2.22e-16; // Machine epsilon
    const double sqrt_eps = std::sqrt(eps);
    const double ortho_threshold = sqrt_eps; // ~1.5e-8
    std::vector<std::vector<double>> omega; // omega[j][i] tracks loss of orthogonality
    omega.resize(1);
    omega[0].push_back(eps);
    
    // Statistics
    stats_.full_reorth_count = 0;
    stats_.selective_reorth_count = 0;
    stats_.total_reorth_ops = 0;
    
    std::cout << "  Reorthogonalization threshold: " << ortho_threshold << "\n";
    
    int m = 0;  // Number of iterations performed
    
    for (m = 0; m < max_iter_; ++m) {
        // w = H * v_current
        op_->matVecGPU(d_v_current_, d_w_, dimension_);
        stats_.matvec_time += op_->getStats().matVecTime;
        
        // alpha[m] = <v_current | w>
        std::complex<double> alpha_complex = vectorDot(d_v_current_, d_w_);
        alpha_.push_back(alpha_complex.real());
        
        // w = w - alpha[m] * v_current
        cuDoubleComplex neg_alpha = make_cuDoubleComplex(-alpha_complex.real(), 0.0);
        vectorAxpy(d_v_current_, d_w_, neg_alpha);
        
        // w = w - beta[m-1] * v_prev
        if (m > 0) {
            cuDoubleComplex neg_beta = make_cuDoubleComplex(-beta_[m-1], 0.0);
            vectorAxpy(d_v_prev_, d_w_, neg_beta);
        }
        
        // Adaptive selective reorthogonalization
        if (num_stored_vectors_ > 0 && m > 0) {
            orthogonalize(d_w_, m, omega, alpha_, beta_, ortho_threshold);
        }
        
        // beta[m] = ||w||
        double beta = vectorNorm(d_w_);
        beta_.push_back(beta);
        
        // Print progress
        if ((m + 1) % 10 == 0) {
            std::cout << "  Iteration " << m+1 << "/" << max_iter_ 
                     << ", beta = " << beta << "\n";
        }
        
        // ========== Breakdown Conditions ==========
        
        // 1. Beta breakdown: If beta is too small, Lanczos basis is complete
        if (beta < tolerance_) {
            std::cout << "  Beta breakdown at iteration " << m+1 << " (beta = " << beta << " < tol = " << tolerance_ << ")\n";
            std::cout << "  Lanczos basis is complete - invariant subspace found!\n";
            m++;
            break;
        }
        
        // 2. Near-breakdown: Warn if beta is getting dangerously small
        if (beta < 100.0 * tolerance_ && beta >= tolerance_) {
            std::cout << "  Warning: Near-breakdown at iteration " << m+1 << " (beta = " << beta << ")\n";
        }
        
        // 3. Eigenvalue convergence check (every check_convergence_interval iterations)
        if (m >= num_eigenvalues && (m + 1) % check_convergence_interval == 0) {
            // Solve tridiagonal problem with current Krylov space
            std::vector<double> current_eigenvalues;
            std::vector<std::vector<double>> temp_eigenvecs;
            solveTridiagonal(m + 1, num_eigenvalues, current_eigenvalues, temp_eigenvecs);
            
            // Check if eigenvalues have converged
            if (!prev_eigenvalues.empty() && prev_eigenvalues.size() >= num_eigenvalues) {
                double max_change = 0.0;
                for (int i = 0; i < num_eigenvalues && i < current_eigenvalues.size(); ++i) {
                    double change = std::abs(current_eigenvalues[i] - prev_eigenvalues[i]);
                    max_change = std::max(max_change, change);
                }
                
                if (max_change < tolerance_) {
                    std::cout << "  Eigenvalues converged at iteration " << m+1 
                             << " (max change = " << max_change << " < tol = " << tolerance_ << ")\n";
                    eigenvalues_converged = true;
                    m++;
                    break;
                }
            }
            
            prev_eigenvalues = current_eigenvalues;
        }
        
        // 4. Loss of orthogonality check (if full reorthogonalization is not used)
        if (num_stored_vectors_ == 0 && m > 0) {
            // Estimate loss of orthogonality using beta values
            // If beta suddenly increases significantly, we may have lost orthogonality
            if (m > 10 && beta > 10.0 * beta_[m-1] && beta_[m-1] < tolerance_ * 10.0) {
                std::cout << "  Warning: Possible loss of orthogonality at iteration " << m+1 << "\n";
                std::cout << "  Consider using full reorthogonalization for better accuracy\n";
            }
        }
        
        // ==========================================
        
        // v_next = w / beta
        vectorScale(d_w_, 1.0 / beta);
        
        // Cycle vectors: v_prev = v_current, v_current = w
        std::swap(d_v_prev_, d_v_current_);
        std::swap(d_v_current_, d_w_);
        
        // Store Lanczos vector if space available
        if (num_stored_vectors_ > 0 && m + 1 < num_stored_vectors_) {
            vectorCopy(d_v_current_, d_lanczos_vectors_[m + 1]);
        }
    }
    
    stats_.iterations = m;
    
    // Print completion message with reason for termination
    std::cout << "\nGPU Lanczos algorithm completed after " << m << " iterations\n";
    if (m >= max_iter_) {
        std::cout << "  Reason: Maximum iterations reached\n";
    } else if (eigenvalues_converged) {
        std::cout << "  Reason: Eigenvalues converged (within tolerance)\n";
    } else if (m > 0 && beta_[m-1] < tolerance_) {
        std::cout << "  Reason: Beta breakdown (invariant subspace found)\n";
    }
    
    // Print reorthogonalization statistics
    std::cout << "\n===== Reorthogonalization Statistics =====" << std::endl;
    std::cout << "Total Lanczos iterations: " << m << std::endl;
    std::cout << "Full reorthogonalizations: " << stats_.full_reorth_count << std::endl;
    std::cout << "Selective reorthogonalizations: " << stats_.selective_reorth_count << std::endl;
    std::cout << "Total inner products: " << stats_.total_reorth_ops << std::endl;
    if (m > 0) {
        std::cout << "Average reorth per iteration: " << (double)stats_.total_reorth_ops / m << std::endl;
        uint64_t theoretical_full = (m * (m + 1)) / 2;
        std::cout << "Theoretical full reorth cost: " << theoretical_full << std::endl;
        if (stats_.total_reorth_ops > 0) {
            std::cout << "Savings factor: " << (double)theoretical_full / stats_.total_reorth_ops << "x" << std::endl;
        }
    }
    std::cout << "==========================================\n" << std::endl;
    
    std::cout << "  Total matvec time: " << stats_.matvec_time << " s\n";
    std::cout << "  Total ortho time: " << stats_.ortho_time << " s\n";
    
    // Solve tridiagonal eigenvalue problem
    std::vector<std::vector<double>> tridiag_eigenvecs;
    solveTridiagonal(m, num_eigenvalues, eigenvalues, tridiag_eigenvecs);
    
    // Compute Ritz vectors if requested
    if (compute_vectors && num_stored_vectors_ > 0) {
        computeRitzVectors(tridiag_eigenvecs, num_eigenvalues, eigenvectors);
    }
    
    auto overall_end = std::chrono::high_resolution_clock::now();
    stats_.total_time = std::chrono::duration<double>(overall_end - overall_start).count();
    
    std::cout << "Total GPU Lanczos time: " << stats_.total_time << " s\n";
}

void GPULanczos::solveTridiagonal(int m, int num_eigs,
                                 std::vector<double>& eigenvalues,
                                 std::vector<std::vector<double>>& eigenvectors) {
    
    // Use Eigen to solve tridiagonal system
    Eigen::MatrixXd T = Eigen::MatrixXd::Zero(m, m);
    
    for (int i = 0; i < m; ++i) {
        T(i, i) = alpha_[i];
        if (i < m - 1) {
            T(i, i+1) = beta_[i];
            T(i+1, i) = beta_[i];
        }
    }
    
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(T);
    
    if (solver.info() != Eigen::Success) {
        std::cerr << "Eigenvalue computation failed!\n";
        return;
    }
    
    // Extract lowest eigenvalues
    int n_eigs = std::min(num_eigs, m);
    eigenvalues.resize(n_eigs);
    eigenvectors.resize(n_eigs);
    
    for (int i = 0; i < n_eigs; ++i) {
        eigenvalues[i] = solver.eigenvalues()(i);
        eigenvectors[i].resize(m);
        for (int j = 0; j < m; ++j) {
            eigenvectors[i][j] = solver.eigenvectors()(j, i);
        }
    }
    
    std::cout << "\nLowest " << n_eigs << " eigenvalues:\n";
    for (int i = 0; i < n_eigs; ++i) {
        std::cout << "  E[" << i << "] = " << eigenvalues[i] << "\n";
    }
}

void GPULanczos::computeRitzVectors(
    const std::vector<std::vector<double>>& tridiag_eigenvecs,
    int num_vecs,
    std::vector<std::vector<std::complex<double>>>& eigenvectors) {
    
    std::cout << "\nComputing Ritz vectors...\n";
    
    eigenvectors.resize(num_vecs);
    
    for (int i = 0; i < num_vecs; ++i) {
        eigenvectors[i].resize(dimension_);
        
        // Initialize to zero
        CUDA_CHECK(cudaMemset(d_temp_, 0, dimension_ * sizeof(cuDoubleComplex)));
        
        // Linear combination: eigenvec[i] = sum_j tridiag_eigenvecs[i][j] * lanczos_vectors[j]
        for (size_t j = 0; j < tridiag_eigenvecs[i].size(); ++j) {
            cuDoubleComplex coeff = make_cuDoubleComplex(tridiag_eigenvecs[i][j], 0.0);
            vectorAxpy(d_lanczos_vectors_[j], d_temp_, coeff);
        }
        
        // Copy back to host
        std::vector<cuDoubleComplex> temp_host(dimension_);
        CUDA_CHECK(cudaMemcpy(temp_host.data(), d_temp_,
                            dimension_ * sizeof(cuDoubleComplex),
                            cudaMemcpyDeviceToHost));
        
        for (int k = 0; k < dimension_; ++k) {
            eigenvectors[i][k] = std::complex<double>(
                cuCreal(temp_host[k]),
                cuCimag(temp_host[k])
            );
        }
    }
    
    std::cout << "  Ritz vectors computed\n";
}

#endif // WITH_CUDA
