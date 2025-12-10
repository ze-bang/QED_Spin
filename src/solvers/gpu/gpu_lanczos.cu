#ifdef WITH_CUDA

#include <ed/gpu/gpu_lanczos.cuh>
#include <ed/gpu/kernel_config.h>
#include <iostream>
#include <iomanip>
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

/**
 * @brief Batched modified Gram-Schmidt orthogonalization kernel
 * 
 * Computes multiple dot products in parallel and accumulates the
 * orthogonalization corrections. Each block handles one basis vector.
 * Uses shared memory for efficient reduction within each block.
 * 
 * @param basis Array of basis vector pointers
 * @param target Vector to orthogonalize (modified in-place)
 * @param overlaps Output array for computed overlaps (size = num_vecs)
 * @param num_vecs Number of basis vectors to orthogonalize against
 * @param N Vector dimension
 */
__global__ void batchedDotProductKernel(const cuDoubleComplex* const* basis,
                                        const cuDoubleComplex* target,
                                        cuDoubleComplex* overlaps,
                                        int num_vecs, int N) {
    extern __shared__ double shared[];
    double* shared_real = shared;
    double* shared_imag = shared + blockDim.x;
    
    int vec_idx = blockIdx.x;  // Each block handles one basis vector
    if (vec_idx >= num_vecs) return;
    
    const cuDoubleComplex* basis_vec = basis[vec_idx];
    
    // Each thread computes partial sum over its assigned elements
    double sum_real = 0.0;
    double sum_imag = 0.0;
    
    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        cuDoubleComplex b = basis_vec[i];
        cuDoubleComplex t = target[i];
        
        // Compute conjugate(b) * t
        double b_real = cuCreal(b);
        double b_imag = cuCimag(b);
        double t_real = cuCreal(t);
        double t_imag = cuCimag(t);
        
        sum_real += b_real * t_real + b_imag * t_imag;  // Re(conj(b) * t)
        sum_imag += b_real * t_imag - b_imag * t_real;  // Im(conj(b) * t)
    }
    
    shared_real[threadIdx.x] = sum_real;
    shared_imag[threadIdx.x] = sum_imag;
    __syncthreads();
    
    // Parallel reduction
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            shared_real[threadIdx.x] += shared_real[threadIdx.x + stride];
            shared_imag[threadIdx.x] += shared_imag[threadIdx.x + stride];
        }
        __syncthreads();
    }
    
    // Thread 0 writes result
    if (threadIdx.x == 0) {
        overlaps[vec_idx] = make_cuDoubleComplex(shared_real[0], shared_imag[0]);
    }
}

/**
 * @brief Apply orthogonalization corrections in batched manner
 * 
 * target = target - sum_i(overlaps[i] * basis[i])
 */
__global__ void batchedOrthogonalizeKernel(cuDoubleComplex* const* basis,
                                          cuDoubleComplex* target,
                                          const cuDoubleComplex* overlaps,
                                          int num_vecs, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    
    cuDoubleComplex correction = make_cuDoubleComplex(0.0, 0.0);
    
    for (int v = 0; v < num_vecs; ++v) {
        cuDoubleComplex overlap = overlaps[v];
        cuDoubleComplex basis_val = basis[v][idx];
        correction = cuCadd(correction, cuCmul(overlap, basis_val));
    }
    
    target[idx] = cuCsub(target[idx], correction);
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
    
    // IMPROVED: Smart memory allocation strategy
    // Check available GPU memory and allocate as many vectors as feasible
    size_t free_mem, total_mem;
    CUDA_CHECK(cudaMemGetInfo(&free_mem, &total_mem));
    
    // Reserve 20% of free memory for other operations and overhead
    size_t usable_mem = static_cast<size_t>(free_mem * 0.8);
    
    // We already allocated 4 working vectors, subtract their memory
    size_t working_mem = 4 * vec_size;
    size_t available_for_storage = (usable_mem > working_mem) ? (usable_mem - working_mem) : 0;
    
    // Calculate how many vectors we can store
    int max_storable = static_cast<int>(available_for_storage / vec_size);
    
    // For local reorthogonalization, we only need to store recent vectors
    // Limit to min(max_iter, max_storable, 50) - we don't need more than 50 for local reorth
    int target_storage = std::min({max_iter_, max_storable, 50});
    
    if (target_storage >= 10) {
        // Allocate array of pointers for Lanczos vectors
        d_lanczos_vectors_ = new cuDoubleComplex*[target_storage];
        for (int i = 0; i < target_storage; ++i) {
            CUDA_CHECK(cudaMalloc(&d_lanczos_vectors_[i], vec_size));
        }
        num_stored_vectors_ = target_storage;
        std::cout << "  Storing " << num_stored_vectors_ << " Lanczos vectors on GPU for local reorthogonalization\n";
        std::cout << "  GPU Memory: " << (free_mem / (1024.0 * 1024.0 * 1024.0)) << " GB free, "
                  << "using " << ((num_stored_vectors_ * vec_size) / (1024.0 * 1024.0 * 1024.0)) << " GB for basis storage\n";
    } else {
        std::cout << "  Warning: Insufficient GPU memory for vector storage\n";
        std::cout << "  GPU Memory: " << (free_mem / (1024.0 * 1024.0 * 1024.0)) << " GB free, "
                  << "need " << ((10 * vec_size) / (1024.0 * 1024.0 * 1024.0)) << " GB minimum\n";
        std::cout << "  Using no reorthogonalization (may produce less accurate results)\n";
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

// IMPROVED: Local reorthogonalization with batched operations for better GPU utilization
// Uses batched kernel when num_check >= 4 for better performance
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
        // Determine how many recent vectors to reorthogonalize against
        int num_check = std::min(iter, std::min(10, num_stored_vectors_));
        
        // Use batched approach when there are enough vectors (better GPU utilization)
        const int BATCH_THRESHOLD = 4;
        
        if (num_check >= BATCH_THRESHOLD) {
            // BATCHED ORTHOGONALIZATION: More efficient for multiple vectors
            
            // Allocate device memory for batch pointers and overlaps
            cuDoubleComplex** d_basis_ptrs = nullptr;
            cuDoubleComplex* d_overlaps = nullptr;
            CUDA_CHECK(cudaMalloc(&d_basis_ptrs, num_check * sizeof(cuDoubleComplex*)));
            CUDA_CHECK(cudaMalloc(&d_overlaps, num_check * sizeof(cuDoubleComplex)));
            
            // Collect pointers to the basis vectors we want to orthogonalize against
            std::vector<cuDoubleComplex*> h_basis_ptrs(num_check);
            for (int i = 0; i < num_check; ++i) {
                int src_idx = std::max(0, iter - num_check) + i;
                int buffer_idx = src_idx % num_stored_vectors_;
                h_basis_ptrs[i] = d_lanczos_vectors_[buffer_idx];
            }
            CUDA_CHECK(cudaMemcpy(d_basis_ptrs, h_basis_ptrs.data(), 
                                 num_check * sizeof(cuDoubleComplex*), cudaMemcpyHostToDevice));
            
            // Launch batched dot product kernel
            // Each block handles one basis vector
            int threads_per_block = 256;
            size_t shared_mem = 2 * threads_per_block * sizeof(double);
            GPULanczosKernels::batchedDotProductKernel<<<num_check, threads_per_block, shared_mem>>>(
                d_basis_ptrs, d_vec, d_overlaps, num_check, dimension_);
            CUDA_CHECK(cudaGetLastError());
            
            // Copy overlaps back to host to check threshold
            std::vector<cuDoubleComplex> h_overlaps(num_check);
            CUDA_CHECK(cudaMemcpy(h_overlaps.data(), d_overlaps, 
                                 num_check * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost));
            
            // Count significant overlaps and apply corrections
            int num_reorthed = 0;
            for (int i = 0; i < num_check; ++i) {
                double overlap_mag = sqrt(cuCreal(h_overlaps[i]) * cuCreal(h_overlaps[i]) + 
                                         cuCimag(h_overlaps[i]) * cuCimag(h_overlaps[i]));
                if (overlap_mag > ortho_threshold) {
                    // Apply correction: vec -= overlap * basis[i]
                    cuDoubleComplex neg_overlap = make_cuDoubleComplex(-cuCreal(h_overlaps[i]), 
                                                                       -cuCimag(h_overlaps[i]));
                    vectorAxpy(h_basis_ptrs[i], d_vec, neg_overlap);
                    num_reorthed++;
                }
            }
            
            if (num_reorthed > 0) {
                stats_.selective_reorth_count++;
                stats_.total_reorth_ops += num_reorthed;
            }
            
            // Cleanup
            cudaFree(d_basis_ptrs);
            cudaFree(d_overlaps);
            
        } else {
            // SEQUENTIAL APPROACH: More efficient for small number of vectors
            int num_reorthed = 0;
            
            for (int i = std::max(0, iter - num_check); i < iter; ++i) {
                int buffer_idx = i % num_stored_vectors_;
                std::complex<double> dot = vectorDot(d_lanczos_vectors_[buffer_idx], d_vec);
                double overlap_magnitude = std::abs(dot);
                
                if (overlap_magnitude > ortho_threshold) {
                    cuDoubleComplex neg_dot = make_cuDoubleComplex(-dot.real(), -dot.imag());
                    vectorAxpy(d_lanczos_vectors_[buffer_idx], d_vec, neg_dot);
                    num_reorthed++;
                }
            }
            
            if (num_reorthed > 0) {
                stats_.selective_reorth_count++;
                stats_.total_reorth_ops += num_reorthed;
            }
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
    
    std::cout << "\nRunning GPU Lanczos with Local Reorthogonalization...\n";
    
    alpha_.clear();
    beta_.clear();
    
    // Initialize first Lanczos vector
    initializeRandomVector(d_v_current_);
    
    if (num_stored_vectors_ > 0) {
        vectorCopy(d_v_current_, d_lanczos_vectors_[0]);
        std::cout << "  Storing " << num_stored_vectors_ << " Lanczos vectors on GPU\n";
        std::cout << "  Using local reorthogonalization (threshold-based)\n";
    } else {
        std::cout << "  Warning: Insufficient GPU memory for vector storage\n";
        std::cout << "  Running without reorthogonalization (may reduce accuracy)\n";
    }
    
    // Initialize previous vector to zero
    CUDA_CHECK(cudaMemset(d_v_prev_, 0, dimension_ * sizeof(cuDoubleComplex)));
    
    // For eigenvalue convergence checking
    std::vector<double> prev_eigenvalues;
    int check_convergence_interval = 10;  // Check every 10 iterations
    bool eigenvalues_converged = false;
    
    // Local reorthogonalization parameters
    const double eps = 2.22e-16; // Machine epsilon
    const double sqrt_eps = std::sqrt(eps);
    const double ortho_threshold = sqrt_eps; // ~1.5e-8
    std::vector<std::vector<double>> omega; // Placeholder for compatibility (not used with fixed version)
    
    // Statistics
    stats_.full_reorth_count = 0;
    stats_.selective_reorth_count = 0;
    stats_.total_reorth_ops = 0;
    
    if (num_stored_vectors_ > 0) {
        std::cout << "  Reorthogonalization threshold: " << ortho_threshold << "\n";
    }
    
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
        
        // Local reorthogonalization with stored vectors
        if (num_stored_vectors_ > 0 && m > 0) {
            orthogonalize(d_w_, m, omega, alpha_, beta_, ortho_threshold);
        }
        
        // beta[m] = ||w||
        double beta = vectorNorm(d_w_);
        beta_.push_back(beta);
        
        // Compute residual error for monitoring
        // Residual = ||H*v_j - alpha_j*v_j - beta_{j+1}*v_{j+1}|| / ||H*v_j||
        double residual_error = 0.0;
        if (m == 0) {
            // For first iteration, estimate ||H*v_j|| from alpha and beta
            residual_error = beta / (std::abs(alpha_[m]) + beta);
        } else {
            // Estimate from current iteration quantities
            residual_error = beta / (std::abs(alpha_[m]) + std::abs(beta_[m-1]) + beta);
        }
        
        // Print progress with residual error
        if ((m + 1) % 10 == 0 || m < 5) {
            std::cout << "  Iteration " << m+1 << "/" << max_iter_ 
                     << "  |  beta = " << std::scientific << std::setprecision(4) << beta
                     << "  |  residual = " << residual_error << std::defaultfloat << "\n";
        }
        
        // ========== Breakdown Conditions ==========
        
        // 1. Beta breakdown: If beta is too small, Lanczos basis is complete
        if (beta < tolerance_) {
            std::cout << "\n  === GPU Lanczos Breakdown Detected ===" << std::endl;
            std::cout << "  Iteration: " << m+1 << std::endl;
            std::cout << "  Beta = " << std::scientific << std::setprecision(4) << beta 
                     << " < tolerance = " << tolerance_ << std::endl;
            std::cout << "  Residual error: " << residual_error << std::endl;
            std::cout << "  Invariant subspace found - exact diagonalization complete!" << std::defaultfloat << std::endl;
            std::cout << "  ========================================\n" << std::endl;
            m++;
            break;
        }
        
        // 2. Near-breakdown: Warn if beta is getting dangerously small
        if (beta < 100.0 * tolerance_ && beta >= tolerance_) {
            std::cout << "  Warning: Near-breakdown at iteration " << m+1 
                     << " (beta = " << std::scientific << beta << ")" << std::defaultfloat << "\n";
        }
        
        // 3. Check for numerical issues with residual
        if (m > 10 && residual_error > 0.9) {
            std::cout << "\n  !!! WARNING: High residual error detected !!!" << std::endl;
            std::cout << "  Iteration " << m+1 << ": residual = " << residual_error << std::endl;
            std::cout << "  This may indicate loss of orthogonality or numerical issues." << std::endl;
            if (num_stored_vectors_ == 0) {
                std::cout << "  Recommendation: Increase GPU memory for vector storage.\n" << std::endl;
            } else {
                std::cout << "  Consider increasing stored vectors or using CPU Lanczos.\n" << std::endl;
            }
        }
        
        // 4. Eigenvalue convergence check (every check_convergence_interval iterations)
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
        
        // 5. Loss of orthogonality check (if full reorthogonalization is not used)
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
        
        // Store Lanczos vector using circular buffer indexing
        // For local reorthogonalization, we only need the most recent vectors
        if (num_stored_vectors_ > 0) {
            // Use modulo for circular buffer: always overwrite oldest vector
            int buffer_idx = (m + 1) % num_stored_vectors_;
            vectorCopy(d_v_current_, d_lanczos_vectors_[buffer_idx]);
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
    std::cout << "\n===== GPU Reorthogonalization Statistics =====" << std::endl;
    std::cout << "Total Lanczos iterations: " << m << std::endl;
    if (num_stored_vectors_ > 0) {
        std::cout << "Vectors stored on GPU: " << std::min(m, num_stored_vectors_) << " / " << num_stored_vectors_ << std::endl;
        std::cout << "Local reorthogonalizations: " << stats_.selective_reorth_count << std::endl;
        std::cout << "Total inner products: " << stats_.total_reorth_ops << std::endl;
        if (m > 0) {
            std::cout << "Average reorth per iteration: " << (double)stats_.total_reorth_ops / m << std::endl;
            uint64_t theoretical_full = (m * (m + 1)) / 2;
            std::cout << "Theoretical full reorth cost: " << theoretical_full << std::endl;
            if (stats_.total_reorth_ops > 0) {
                std::cout << "Savings factor: " << (double)theoretical_full / stats_.total_reorth_ops << "x" << std::endl;
            }
        }
    } else {
        std::cout << "No reorthogonalization performed (insufficient GPU memory)" << std::endl;
        std::cout << "WARNING: Results may be less accurate due to loss of orthogonality" << std::endl;
    }
    std::cout << "=============================================\n" << std::endl;
    
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
    
    // Safety check: can only compute Ritz vectors if all needed Lanczos vectors are stored
    size_t num_lanczos_vecs_needed = tridiag_eigenvecs.empty() ? 0 : tridiag_eigenvecs[0].size();
    if (num_lanczos_vecs_needed > static_cast<size_t>(num_stored_vectors_)) {
        std::cerr << "Warning: Cannot compute Ritz vectors - not enough Lanczos vectors stored\n";
        std::cerr << "  Need: " << num_lanczos_vecs_needed << ", have: " << num_stored_vectors_ << "\n";
        std::cerr << "  Increase GPU memory allocation or reduce max_iterations\n";
        return;
    }
    
    eigenvectors.resize(num_vecs);
    
    for (int i = 0; i < num_vecs; ++i) {
        eigenvectors[i].resize(dimension_);
        
        // Initialize to zero
        CUDA_CHECK(cudaMemset(d_temp_, 0, dimension_ * sizeof(cuDoubleComplex)));
        
        // Linear combination: eigenvec[i] = sum_j tridiag_eigenvecs[i][j] * lanczos_vectors[j]
        // Note: This assumes circular buffer hasn't wrapped (i.e., iterations <= num_stored_vectors_)
        for (size_t j = 0; j < tridiag_eigenvecs[i].size() && j < static_cast<size_t>(num_stored_vectors_); ++j) {
            cuDoubleComplex coeff = make_cuDoubleComplex(tridiag_eigenvecs[i][j], 0.0);
            int buffer_idx = j % num_stored_vectors_;  // Use modulo for safety
            vectorAxpy(d_lanczos_vectors_[buffer_idx], d_temp_, coeff);
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
