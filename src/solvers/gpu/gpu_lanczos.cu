#ifdef WITH_CUDA

#include <ed/gpu/gpu_lanczos.cuh>
#include <ed/gpu/kernel_config.h>
#include <ed/core/blas_lapack_wrapper.h>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <random>
#include <algorithm>
#include <chrono>
#include <stdexcept>
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <curand_kernel.h>

using namespace GPUConfig;

// ============================================================================
// GPU Lanczos Kernels
// ============================================================================

namespace GPULanczosKernels {

__global__ void initRandomVectorKernel(cuDoubleComplex* vec, int N, unsigned long long seed,
                                       int complex_seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    curandState state;
    curand_init(seed, idx, 0, &state);

    // Default: real-only seed, mirroring the CPU lanczos.cpp convention. For
    // a Hermitian, real H, a real Krylov space is mathematically sufficient
    // and lets cuSPARSE/cuBLAS stay on real arithmetic where applicable.
    // Set complex_seed=1 (host-side ED_LANCZOS_COMPLEX_SEED=1) to recover
    // the legacy fully-complex behavior, e.g. for testing.
    double real_part = curand_normal_double(&state);
    double imag_part = complex_seed ? curand_normal_double(&state) : 0.0;

    vec[idx] = make_cuDoubleComplex(real_part, imag_part);
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
      d_lanczos_vectors_(nullptr), num_stored_vectors_(0),
      d_ortho_basis_ptrs_(nullptr), d_ortho_overlaps_(nullptr),
      ortho_timing_events_created_(false) {
    
    dimension_ = op_->getDimension();
    
    // Cap max iterations well below the Hilbert space dimension.
    // When iterations approach the dimension, the Krylov subspace nearly spans
    // the full space. On GPU, floating-point non-determinism in
    // reorthogonalization can cause beta values to grow exponentially rather
    // than decaying to zero, corrupting the tridiagonal matrix.
    // Capping at 80% of dimension provides a safety margin while still
    // allowing the eigenvalue convergence check to trigger.
    int safe_max = std::max(1, (int)(0.8 * dimension_));
    if (max_iter_ > safe_max) {
        std::cout << "  Note: capping iterations from " << max_iter_ 
                 << " to " << safe_max << " (80% of dim=" << dimension_ 
                 << ") to prevent GPU numerical instability\n";
        max_iter_ = safe_max;
    }
    
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
    
    // Store as many vectors as possible up to max_iter
    // We need all Lanczos vectors to compute Ritz vectors (eigenvectors) at the end
    int target_storage = std::min(max_iter_, max_storable);
    
    if (target_storage >= 10) {
        // Allocate array of pointers for Lanczos vectors
        d_lanczos_vectors_ = new cuDoubleComplex*[target_storage];
        for (int i = 0; i < target_storage; ++i) {
            CUDA_CHECK(cudaMalloc(&d_lanczos_vectors_[i], vec_size));
        }
        num_stored_vectors_ = target_storage;
        const char* reorth_kind = (num_stored_vectors_ >= max_iter_) ? "FULL" : "WINDOWED";
        std::cout << "  Storing " << num_stored_vectors_ << " Lanczos vectors on GPU"
                  << " (max_iter=" << max_iter_ << ", reorth=" << reorth_kind << ")\n";
        std::cout << "  GPU Memory: " << (free_mem / (1024.0 * 1024.0 * 1024.0)) << " GB free, "
                  << "using " << ((num_stored_vectors_ * vec_size) / (1024.0 * 1024.0 * 1024.0)) << " GB for basis storage\n";

        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_ortho_basis_ptrs_),
                              static_cast<size_t>(num_stored_vectors_) * sizeof(cuDoubleComplex*)));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_ortho_overlaps_),
                              static_cast<size_t>(num_stored_vectors_) * sizeof(cuDoubleComplex)));
    } else {
        std::cout << "  Warning: Insufficient GPU memory for vector storage\n";
        std::cout << "  GPU Memory: " << (free_mem / (1024.0 * 1024.0 * 1024.0)) << " GB free, "
                  << "need " << ((10 * vec_size) / (1024.0 * 1024.0 * 1024.0)) << " GB minimum\n";
        std::cout << "  Using no reorthogonalization (may produce less accurate results)\n";
        num_stored_vectors_ = 0;
    }

    CUDA_CHECK(cudaEventCreate(&ortho_timing_start_));
    CUDA_CHECK(cudaEventCreate(&ortho_timing_stop_));
    ortho_timing_events_created_ = true;
    
    alpha_.reserve(max_iter_);
    beta_.reserve(max_iter_);
}

void GPULanczos::freeMemory() {
    if (ortho_timing_events_created_) {
        CUDA_CHECK(cudaEventDestroy(ortho_timing_start_));
        CUDA_CHECK(cudaEventDestroy(ortho_timing_stop_));
        ortho_timing_events_created_ = false;
    }
    if (d_v_current_) cudaFree(d_v_current_);
    if (d_v_prev_) cudaFree(d_v_prev_);
    if (d_w_) cudaFree(d_w_);
    if (d_temp_) cudaFree(d_temp_);
    if (d_ortho_basis_ptrs_) {
        cudaFree(d_ortho_basis_ptrs_);
        d_ortho_basis_ptrs_ = nullptr;
    }
    if (d_ortho_overlaps_) {
        cudaFree(d_ortho_overlaps_);
        d_ortho_overlaps_ = nullptr;
    }
    
    if (d_lanczos_vectors_) {
        for (int i = 0; i < num_stored_vectors_; ++i) {
            if (d_lanczos_vectors_[i]) {
                cudaFree(d_lanczos_vectors_[i]);
            }
        }
        delete[] d_lanczos_vectors_;
    }
}

void GPULanczos::initializeRandomVector(cuDoubleComplex* d_vec, unsigned long long seed) {
    int num_blocks = (dimension_ + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Use provided seed for reproducibility, or random seed if 0
    unsigned long long actual_seed = (seed == 0) ? std::random_device{}() : seed;

    // Honour the same real-only-seed convention as the CPU lanczos. The env
    // var ED_LANCZOS_COMPLEX_SEED=1 forces a fully complex starting vector
    // (legacy behaviour). Default is real-only so the real-arith fast path
    // can be taken throughout the Krylov space when H is real Hermitian.
    const char* cs = std::getenv("ED_LANCZOS_COMPLEX_SEED");
    const int complex_seed = (cs && cs[0] == '1') ? 1 : 0;

    GPULanczosKernels::initRandomVectorKernel<<<num_blocks, BLOCK_SIZE>>>(
        d_vec, dimension_, actual_seed, complex_seed);

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
    // cuBLAS Zdscal: x <- alpha * x with real alpha — same semantics as vectorScaleKernel
    CUBLAS_CHECK(cublasZdscal(cublas_handle_, dimension_, &scale, d_vec, 1));
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

// Reorthogonalization against the GPU-resident Lanczos basis.
//
// Window size: min(iter, num_stored_vectors_). When num_stored_vectors_ ==
// max_iter_ (the common case — see allocateMemory()) this IS full
// reorthogonalization; when GPU memory could only hold a prefix it degrades
// gracefully to "reorth against the last num_stored_vectors_ vectors". The
// runtime warning below fires once when that degradation is in effect, so
// users are not silently running with windowed reorth.
//
// Two-pass DGKS is applied below: project out overlaps above
// ortho_threshold, then re-project anything still above 1e-15.
void GPULanczos::orthogonalize(cuDoubleComplex* d_vec, int iter,
                               std::vector<std::vector<double>>& omega,
                               const std::vector<double>& alpha,
                               const std::vector<double>& beta,
                               double ortho_threshold) {
    // Per-call orthogonalization timing forces a host/GPU sync via
    // cudaEventSynchronize, which serializes the entire Lanczos pipeline.
    // Make it opt-in via ED_GPU_TIMING=1 (same convention as matVecGPU).
    static const bool timing_enabled = []() {
        const char* s = std::getenv("ED_GPU_TIMING");
        return (s && s[0] == '1');
    }();
    if (timing_enabled) {
        CUDA_CHECK(cudaEventRecord(ortho_timing_start_));
    }

    if (num_stored_vectors_ > 0 && iter > 0) {
        // Window size = min(iter, num_stored_vectors_). Equals 'iter' (full
        // reorth) when the basis fits in GPU memory; otherwise falls back to
        // the last num_stored_vectors_ vectors (windowed reorth).
        int num_check = std::min(iter, num_stored_vectors_);

        // Fire the windowed-reorth warning ONCE when we first wrap the
        // ring buffer, so users aren't surprised by reduced numerical
        // stability when iter > num_stored_vectors_.
        static bool warned_once = false;
        if (!warned_once && iter > num_stored_vectors_) {
            warned_once = true;
            std::cerr << "[GPULanczos] Note: iter=" << iter << " exceeds GPU basis buffer ("
                      << num_stored_vectors_ << "). Reorthogonalization is now WINDOWED "
                      << "against the last " << num_stored_vectors_ << " vectors. "
                      << "Increase GPU memory or reduce max_iter for full reorth.\n";
        }
        
        // Use batched approach when there are enough vectors (better GPU utilization)
        const int BATCH_THRESHOLD = 4;
        const bool use_batched = (num_check >= BATCH_THRESHOLD) &&
                                 (d_ortho_basis_ptrs_ != nullptr) &&
                                 (d_ortho_overlaps_ != nullptr);

        if (use_batched) {
            // BATCHED ON-DEVICE DGKS ORTHOGONALIZATION
            //
            // Previous version copied the m overlaps to the host, branched
            // on |overlap| > threshold per-vector, and issued m separate
            // axpy launches — forcing an implicit sync per call and
            // serializing the pipeline. The fused path keeps everything
            // on the device:
            //   1. batchedDotProductKernel computes all m overlaps.
            //   2. batchedOrthogonalizeKernel applies all m corrections
            //      in one launch (overlaps below 1e-15 contribute nothing
            //      meaningful, so we always apply both passes; this is
            //      what cuBLAS-based eigensolvers like SLEPc-CUDA do).
            // The threshold parameter is retained for API stability but
            // its host-side use has been removed.
            (void)ortho_threshold;
            (void)omega;

            std::vector<cuDoubleComplex*> h_basis_ptrs(num_check);
            for (int i = 0; i < num_check; ++i) {
                int src_idx = std::max(0, iter - num_check) + i;
                int buffer_idx = src_idx % num_stored_vectors_;
                h_basis_ptrs[i] = d_lanczos_vectors_[buffer_idx];
            }
            CUDA_CHECK(cudaMemcpy(d_ortho_basis_ptrs_, h_basis_ptrs.data(),
                                 static_cast<size_t>(num_check) * sizeof(cuDoubleComplex*),
                                 cudaMemcpyHostToDevice));

            const int threads_per_block = 256;
            const size_t shared_mem = 2 * threads_per_block * sizeof(double);
            const int axpy_blocks = (dimension_ + threads_per_block - 1) / threads_per_block;

            // ---- Pass 1 ----
            GPULanczosKernels::batchedDotProductKernel<<<num_check, threads_per_block, shared_mem>>>(
                d_ortho_basis_ptrs_, d_vec, d_ortho_overlaps_, num_check, dimension_);
            CUDA_CHECK(cudaGetLastError());
            GPULanczosKernels::batchedOrthogonalizeKernel<<<axpy_blocks, threads_per_block>>>(
                d_ortho_basis_ptrs_, d_vec, d_ortho_overlaps_, num_check, dimension_);
            CUDA_CHECK(cudaGetLastError());

            // ---- Pass 2 (DGKS) ----
            GPULanczosKernels::batchedDotProductKernel<<<num_check, threads_per_block, shared_mem>>>(
                d_ortho_basis_ptrs_, d_vec, d_ortho_overlaps_, num_check, dimension_);
            CUDA_CHECK(cudaGetLastError());
            GPULanczosKernels::batchedOrthogonalizeKernel<<<axpy_blocks, threads_per_block>>>(
                d_ortho_basis_ptrs_, d_vec, d_ortho_overlaps_, num_check, dimension_);
            CUDA_CHECK(cudaGetLastError());

            // Bookkeeping (approximate: we always apply both passes now)
            stats_.selective_reorth_count++;
            stats_.total_reorth_ops += 2 * num_check;

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
                
                // DOUBLE ORTHOGONALIZATION: Second pass for numerical stability
                for (int i = std::max(0, iter - num_check); i < iter; ++i) {
                    int buffer_idx = i % num_stored_vectors_;
                    std::complex<double> dot = vectorDot(d_lanczos_vectors_[buffer_idx], d_vec);
                    double overlap_magnitude = std::abs(dot);
                    
                    if (overlap_magnitude > 1e-15) {
                        cuDoubleComplex neg_dot = make_cuDoubleComplex(-dot.real(), -dot.imag());
                        vectorAxpy(d_lanczos_vectors_[buffer_idx], d_vec, neg_dot);
                        stats_.total_reorth_ops++;
                    }
                }
            }
        }
    }
    
    if (timing_enabled) {
        CUDA_CHECK(cudaEventRecord(ortho_timing_stop_));
        CUDA_CHECK(cudaEventSynchronize(ortho_timing_stop_));

        float milliseconds = 0;
        CUDA_CHECK(cudaEventElapsedTime(&milliseconds, ortho_timing_start_, ortho_timing_stop_));
        stats_.ortho_time += milliseconds / 1000.0;
    }
}

void GPULanczos::run(int num_eigenvalues,
                    std::vector<double>& eigenvalues,
                    std::vector<std::vector<std::complex<double>>>& eigenvectors,
                    bool compute_vectors) {
    
    auto overall_start = std::chrono::high_resolution_clock::now();
    
    std::cout << "\nRunning GPU Lanczos with Full Reorthogonalization...\n";
    
    alpha_.clear();
    beta_.clear();
    
    // Initialize first Lanczos vector with DETERMINISTIC seed for reproducibility
    // Using seed=42 ensures identical results between runs
    // Change to seed=0 for random starting vector if desired
    unsigned long long deterministic_seed = 42;
    initializeRandomVector(d_v_current_, deterministic_seed);
    std::cout << "  Using deterministic seed: " << deterministic_seed << " for reproducibility\n";
    
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
    int check_convergence_interval = 5;  // Check every 5 iterations (more frequent for early detection)
    bool eigenvalues_converged = false;
    double prev_max_change = std::numeric_limits<double>::max();
    
    // Local reorthogonalization parameters
    const double eps = 2.22e-16; // Machine epsilon
    const double sqrt_eps = std::sqrt(eps);
    const double ortho_threshold = sqrt_eps; // ~1.5e-8
    std::vector<std::vector<double>> omega; // Placeholder for compatibility (not used with fixed version)
    
    stats_.selective_reorth_count = 0;
    stats_.total_reorth_ops = 0;
    
    if (num_stored_vectors_ > 0) {
        std::cout << "  Reorthogonalization threshold: " << ortho_threshold << "\n";
    }
    
    int m = 0;  // Number of iterations performed
    int good_m = 0;  // Last known-good iteration count for tridiagonal solve
    double max_beta = 0.0;  // Track maximum beta for relative breakdown detection
    double early_max_beta = 0.0;  // Track max beta from first 20 iterations
    
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
        max_beta = std::max(max_beta, beta);
        if (m < 20) early_max_beta = std::max(early_max_beta, beta);
        
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
        
        // 1. Beta breakdown: absolute or relative
        // Absolute: beta < tolerance (invariant subspace found)
        // Relative: beta < max_beta * eps_rel (Krylov space nearly exhausted)
        // The relative check prevents numerical instability from dividing by
        // near-zero betas when the Krylov space is almost fully spanned.
        double relative_threshold = max_beta * 1e-8;
        double effective_tolerance = std::max(tolerance_, relative_threshold);
        
        if (beta < effective_tolerance) {
            std::cout << "\n  === GPU Lanczos Breakdown Detected ===" << std::endl;
            std::cout << "  Iteration: " << m+1 << std::endl;
            std::cout << "  Beta = " << std::scientific << std::setprecision(4) << beta 
                     << " < effective tolerance = " << effective_tolerance;
            if (relative_threshold > tolerance_) {
                std::cout << " (relative: max_beta=" << max_beta << " * 1e-8)";
            }
            std::cout << std::endl;
            std::cout << "  Residual error: " << residual_error << std::endl;
            std::cout << "  Invariant subspace found - exact diagonalization complete!" << std::defaultfloat << std::endl;
            std::cout << "  ========================================\n" << std::endl;
            m++;
            break;
        }
        
        // 2. Beta explosion: if beta grows by more than 1e6× from the previous
        // iteration in the second half of the run, numerical breakdown has occurred.
        // This catches the failure mode where near-zero beta normalization
        // amplifies GPU floating-point noise, producing garbage Krylov vectors.
        // Only checked in the second half of iterations to avoid false positives
        // from normal Lanczos beta fluctuations in the early/middle phase.
        if (m > max_iter_ / 2 && beta > 1e6 * beta_[m-1]) {
            std::cout << "\n  === GPU Lanczos Numerical Breakdown ===" << std::endl;
            std::cout << "  Iteration: " << m+1 << std::endl;
            std::cout << "  Beta jumped from " << std::scientific << std::setprecision(4) 
                     << beta_[m-1] << " to " << beta 
                     << " (ratio: " << std::fixed << std::setprecision(0) << beta / beta_[m-1] << "x)" << std::endl;
            std::cout << "  Krylov space exhausted - stopping to prevent instability." << std::defaultfloat << std::endl;
            std::cout << "  ========================================\n" << std::endl;
            m++;
            break;
        }
        
        // 2b. Beta growth beyond early range: if beta exceeds 10× the maximum
        // from the first 20 iterations, Krylov space exhaustion has begun and
        // the tridiagonal entries are becoming unreliable.
        if (m >= 20 && early_max_beta > 0 && beta > 10.0 * early_max_beta) {
            std::cout << "\n  === GPU Lanczos Beta Growth Breakdown ===" << std::endl;
            std::cout << "  Iteration: " << m+1 << std::endl;
            std::cout << "  Beta = " << std::scientific << std::setprecision(4) << beta 
                     << " exceeds 10× early max beta = " << early_max_beta << std::endl;
            std::cout << "  Krylov space exhausted - stopping to prevent instability." << std::defaultfloat << std::endl;
            std::cout << "  ========================================\n" << std::endl;
            m++;
            break;
        }
        
        // 3. Near-breakdown: Warn if beta is getting dangerously small
        if (beta < 100.0 * effective_tolerance && beta >= effective_tolerance) {
            std::cout << "  Warning: Near-breakdown at iteration " << m+1 
                     << " (beta = " << std::scientific << beta << ")" << std::defaultfloat << "\n";
        }
        
        // 4. Check for numerical issues with residual
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
        
        // 5. Eigenvalue convergence check (frequent after initial phase)
        // Check every 5 iterations early on, every iteration after half of max_iter
        bool should_check_convergence = false;
        if (m >= num_eigenvalues) {
            if (m >= max_iter_ / 2) {
                should_check_convergence = true;  // Every iteration in second half
            } else {
                should_check_convergence = ((m + 1) % check_convergence_interval == 0);
            }
        }
        if (should_check_convergence) {
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
                
                // Divergence detection: if eigenvalues were nearly converged
                // (prev_max_change was small) but now suddenly shifted significantly,
                // the tridiagonal matrix has been corrupted by loss of orthogonality.
                // Stop immediately and use the tridiagonal from the previous check point.
                // Conditions: (1) past initial convergence phase,
                // (2) large absolute change, (3) previously near-converged,
                // (4) change grew by 10× or more.
                if (m > 20 && max_change > 0.1 
                    && prev_max_change < 1e-3 
                    && max_change > 10.0 * prev_max_change) {
                    std::cout << "\n  === GPU Lanczos Eigenvalue Divergence Detected ===" << std::endl;
                    std::cout << "  Iteration: " << m+1 << std::endl;
                    std::cout << "  Eigenvalue change = " << std::scientific << max_change 
                             << " (prev = " << prev_max_change << ")" << std::endl;
                    std::cout << "  Using tridiagonal from iteration " << (m + 1 - check_convergence_interval) 
                             << " for final eigenvalues." << std::defaultfloat << std::endl;
                    std::cout << "  ================================================\n" << std::endl;
                    // Use the previous good checkpoint for the final tridiagonal solve
                    good_m = m + 1 - check_convergence_interval;
                    eigenvalues_converged = true;
                    m++;
                    break;
                }
                
                prev_max_change = max_change;
                
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
    
    // Use the last known-good iteration count for the final tridiagonal solve.
    // If divergence was detected, good_m was set to the pre-divergence checkpoint.
    // Otherwise, use all computed iterations.
    if (good_m == 0) good_m = m;
    
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
    
    // Solve tridiagonal eigenvalue problem using good_m iterations
    std::vector<std::vector<double>> tridiag_eigenvecs;
    solveTridiagonal(good_m, num_eigenvalues, eigenvalues, tridiag_eigenvecs);
    
    // Compute Ritz vectors if requested
    if (compute_vectors && num_stored_vectors_ > 0 && !eigenvalues.empty() && !tridiag_eigenvecs.empty()) {
        computeRitzVectors(tridiag_eigenvecs, num_eigenvalues, eigenvectors);
    }
    
    auto overall_end = std::chrono::high_resolution_clock::now();
    stats_.total_time = std::chrono::duration<double>(overall_end - overall_start).count();
    
    std::cout << "Total GPU Lanczos time: " << stats_.total_time << " s\n";
}

void GPULanczos::solveTridiagonal(int m, int num_eigs,
                                 std::vector<double>& eigenvalues,
                                 std::vector<std::vector<double>>& eigenvectors) {
    if (m <= 0) {
        eigenvalues.clear();
        eigenvectors.clear();
        return;
    }

    const int n_eigs = std::min(num_eigs, m);

    // For tridiagonal eigenproblems, dstemr (MRRR) is the SOTA partial-spectrum
    // solver: O(m * n_eigs) work and O(m) per-eigenvector storage instead of
    // O(m^3)/O(m^2) for dense Jacobi/QR via Eigen::SelfAdjointEigenSolver.
    // For small m (< 32) the dstemr setup constants don't pay off; fall back
    // to dstevd (D&C), which is what Eigen's solver effectively does, but
    // routed through tuned LAPACK rather than Eigen's portable implementation.
    const bool use_dstemr = (m >= 32) && (n_eigs * 2 < m);

    std::vector<double> diag(m);
    std::vector<double> offdiag(m);  // dstemr requires m entries (last is workspace)
    for (int i = 0; i < m; ++i) {
        diag[i] = alpha_[i];
        offdiag[i] = (i + 1 < m) ? beta_[i] : 0.0;
    }

    // Eigenvectors of the tridiagonal: column-major (m x n_eigs) for both paths
    // (dstemr writes only the first n_eigs columns; dstevd writes all m, of
    // which we discard the last m - n_eigs).
    const lapack_int ldz = m;
    eigenvalues.assign(n_eigs, 0.0);
    eigenvectors.assign(n_eigs, std::vector<double>(m, 0.0));

    if (use_dstemr) {
        std::vector<double> w(m, 0.0);
        std::vector<double> z(static_cast<size_t>(m) * n_eigs, 0.0);
        std::vector<lapack_int> isuppz(2 * std::max(1, n_eigs), 0);
        lapack_int m_found = 0;
        lapack_logical tryrac = 1;

        lapack_int info = LAPACKE_dstemr(
            LAPACK_COL_MAJOR, 'V', 'I', static_cast<lapack_int>(m),
            diag.data(), offdiag.data(),
            /*vl=*/0.0, /*vu=*/0.0,
            /*il=*/1, /*iu=*/static_cast<lapack_int>(n_eigs),
            &m_found, w.data(),
            z.data(), ldz, static_cast<lapack_int>(n_eigs),
            isuppz.data(), &tryrac);

        if (info != 0) {
            throw std::runtime_error(
                "GPULanczos::solveTridiagonal: LAPACKE_dstemr failed with info="
                + std::to_string(info));
        }

        for (int i = 0; i < n_eigs; ++i) {
            eigenvalues[i] = w[i];
            for (int j = 0; j < m; ++j) {
                eigenvectors[i][j] = z[j + static_cast<size_t>(i) * m];
            }
        }
    } else {
        std::vector<double> z(static_cast<size_t>(m) * m, 0.0);
        lapack_int info = LAPACKE_dstevd(
            LAPACK_COL_MAJOR, 'V', static_cast<lapack_int>(m),
            diag.data(), offdiag.data(), z.data(), ldz);

        if (info != 0) {
            throw std::runtime_error(
                "GPULanczos::solveTridiagonal: LAPACKE_dstevd failed with info="
                + std::to_string(info));
        }

        for (int i = 0; i < n_eigs; ++i) {
            eigenvalues[i] = diag[i];
            for (int j = 0; j < m; ++j) {
                eigenvectors[i][j] = z[j + static_cast<size_t>(i) * m];
            }
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
    
    // Hard fail (rather than silently producing zero vectors) when the
    // Lanczos vector buffer is too small. Returning empty / zero Ritz
    // vectors silently corrupts downstream observables and TPQ/FTLM
    // sampling without any signal to the caller. The caller MUST either
    // increase num_stored_vectors_ or reduce max_iterations to fit.
    const size_t num_lanczos_vecs_needed =
        tridiag_eigenvecs.empty() ? 0 : tridiag_eigenvecs[0].size();
    if (num_lanczos_vecs_needed > static_cast<size_t>(num_stored_vectors_)) {
        throw std::runtime_error(
            "GPULanczos::computeRitzVectors: Lanczos vector buffer overflow. "
            "Need " + std::to_string(num_lanczos_vecs_needed) +
            " Lanczos vectors but only " + std::to_string(num_stored_vectors_) +
            " are stored. Increase GPU memory allocation (num_stored_vectors_) "
            "or reduce max_iterations so the Krylov dimension fits.");
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
