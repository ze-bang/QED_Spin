// gpu_ftlm.cu - GPU-accelerated Finite Temperature Lanczos Method
#include "gpu_ftlm.cuh"

#ifdef WITH_CUDA

#include <curand_kernel.h>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <lapacke.h>
#include <fstream>
#include <iomanip>
#include <numeric>

// Error checking macros are already defined in kernel_config.h (included via gpu_operator.cuh)

// ============================================================================
// GPU Kernels for FTLM
// ============================================================================

namespace GPUFTLMKernels {

/**
 * @brief Initialize random vector with cuRAND
 */
__global__ void initRandomVectorKernel(cuDoubleComplex* vec, int N, 
                                      unsigned long long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < N) {
        // Initialize cuRAND state
        curandState state;
        curand_init(seed, idx, 0, &state);
        
        // Generate random complex number with uniform distribution in [-1, 1]
        double real = 2.0 * curand_uniform_double(&state) - 1.0;
        double imag = 2.0 * curand_uniform_double(&state) - 1.0;
        
        vec[idx] = make_cuDoubleComplex(real, imag);
    }
}

/**
 * @brief Normalize vector: vec = vec / norm
 */
__global__ void normalizeKernel(cuDoubleComplex* vec, int N, double norm) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < N) {
        double inv_norm = 1.0 / norm;
        vec[idx] = make_cuDoubleComplex(
            cuCreal(vec[idx]) * inv_norm,
            cuCimag(vec[idx]) * inv_norm
        );
    }
}

/**
 * @brief Vector AXPY: y = alpha*x + y
 */
__global__ void axpyKernel(const cuDoubleComplex* x, cuDoubleComplex* y,
                          cuDoubleComplex alpha, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < N) {
        cuDoubleComplex ax = cuCmul(alpha, x[idx]);
        y[idx] = cuCadd(ax, y[idx]);
    }
}

/**
 * @brief Vector scaling: x = alpha*x
 */
__global__ void scaleKernel(cuDoubleComplex* x, double alpha, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < N) {
        x[idx] = make_cuDoubleComplex(
            cuCreal(x[idx]) * alpha,
            cuCimag(x[idx]) * alpha
        );
    }
}

} // namespace GPUFTLMKernels

// ============================================================================
// GPUFTLMSolver Implementation
// ============================================================================

GPUFTLMSolver::GPUFTLMSolver(GPUOperator* op, int N, int krylov_dim, double tolerance)
    : op_(op), N_(N), krylov_dim_(krylov_dim), tolerance_(tolerance),
      d_v_current_(nullptr), d_v_prev_(nullptr), d_w_(nullptr), d_temp_(nullptr),
      d_lanczos_basis_(nullptr), num_stored_vectors_(0), store_basis_(false),
      gpu_memory_allocated_(false) {
    
    // Initialize cuBLAS
    CUBLAS_CHECK(cublasCreate(&cublas_handle_));
    
    // Initialize stats
    stats_.total_time = 0.0;
    stats_.lanczos_time = 0.0;
    stats_.diag_time = 0.0;
    stats_.thermo_time = 0.0;
    stats_.total_iterations = 0;
    stats_.num_samples_completed = 0;
    
    // Allocate GPU memory
    allocateMemory();
    
    std::cout << "GPU FTLM Solver initialized:\n";
    std::cout << "  Hilbert space dimension: " << N_ << "\n";
    std::cout << "  Krylov dimension: " << krylov_dim_ << "\n";
    std::cout << "  Tolerance: " << tolerance_ << "\n";
}

GPUFTLMSolver::~GPUFTLMSolver() {
    freeMemory();
    
    if (cublas_handle_) {
        cublasDestroy(cublas_handle_);
    }
}

void GPUFTLMSolver::allocateMemory() {
    if (gpu_memory_allocated_) return;
    
    std::cout << "Allocating GPU memory for FTLM...\n";
    
    // Allocate working vectors
    CUDA_CHECK(cudaMalloc(&d_v_current_, N_ * sizeof(cuDoubleComplex)));
    CUDA_CHECK(cudaMalloc(&d_v_prev_, N_ * sizeof(cuDoubleComplex)));
    CUDA_CHECK(cudaMalloc(&d_w_, N_ * sizeof(cuDoubleComplex)));
    CUDA_CHECK(cudaMalloc(&d_temp_, N_ * sizeof(cuDoubleComplex)));
    
    // Initialize to zero
    CUDA_CHECK(cudaMemset(d_v_current_, 0, N_ * sizeof(cuDoubleComplex)));
    CUDA_CHECK(cudaMemset(d_v_prev_, 0, N_ * sizeof(cuDoubleComplex)));
    CUDA_CHECK(cudaMemset(d_w_, 0, N_ * sizeof(cuDoubleComplex)));
    CUDA_CHECK(cudaMemset(d_temp_, 0, N_ * sizeof(cuDoubleComplex)));
    
    gpu_memory_allocated_ = true;
    
    std::cout << "  Allocated " << (4 * N_ * sizeof(cuDoubleComplex) / (1024.0 * 1024.0)) 
              << " MB for working vectors\n";
}

void GPUFTLMSolver::freeMemory() {
    if (!gpu_memory_allocated_) return;
    
    if (d_v_current_) cudaFree(d_v_current_);
    if (d_v_prev_) cudaFree(d_v_prev_);
    if (d_w_) cudaFree(d_w_);
    if (d_temp_) cudaFree(d_temp_);
    
    // Free stored Lanczos basis if allocated
    if (d_lanczos_basis_ && store_basis_) {
        for (int i = 0; i < num_stored_vectors_; i++) {
            if (d_lanczos_basis_[i]) cudaFree(d_lanczos_basis_[i]);
        }
        delete[] d_lanczos_basis_;
    }
    
    gpu_memory_allocated_ = false;
}

void GPUFTLMSolver::initializeRandomVector(cuDoubleComplex* d_vec, unsigned int seed) {
    int threads = 256;
    int blocks = (N_ + threads - 1) / threads;
    
    GPUFTLMKernels::initRandomVectorKernel<<<blocks, threads>>>(d_vec, N_, seed);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Normalize
    normalizeVector(d_vec);
}

double GPUFTLMSolver::vectorNorm(const cuDoubleComplex* d_vec) {
    double result;
    CUBLAS_CHECK(cublasDznrm2(cublas_handle_, N_, d_vec, 1, &result));
    return result;
}

void GPUFTLMSolver::normalizeVector(cuDoubleComplex* d_vec) {
    double norm = vectorNorm(d_vec);
    
    if (norm < 1e-14) {
        std::cerr << "Warning: Attempting to normalize near-zero vector (norm = " 
                  << norm << ")\n";
        return;
    }
    
    int threads = 256;
    int blocks = (N_ + threads - 1) / threads;
    GPUFTLMKernels::normalizeKernel<<<blocks, threads>>>(d_vec, N_, norm);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

void GPUFTLMSolver::vectorCopy(const cuDoubleComplex* src, cuDoubleComplex* dst) {
    CUBLAS_CHECK(cublasZcopy(cublas_handle_, N_, src, 1, dst, 1));
}

void GPUFTLMSolver::vectorScale(cuDoubleComplex* d_vec, double scale) {
    cuDoubleComplex alpha = make_cuDoubleComplex(scale, 0.0);
    CUBLAS_CHECK(cublasZscal(cublas_handle_, N_, &alpha, d_vec, 1));
}

void GPUFTLMSolver::vectorAxpy(const cuDoubleComplex* d_x, cuDoubleComplex* d_y,
                              const cuDoubleComplex& alpha) {
    CUBLAS_CHECK(cublasZaxpy(cublas_handle_, N_, &alpha, d_x, 1, d_y, 1));
}

std::complex<double> GPUFTLMSolver::vectorDot(const cuDoubleComplex* d_x,
                                             const cuDoubleComplex* d_y) {
    cuDoubleComplex result;
    CUBLAS_CHECK(cublasZdotc(cublas_handle_, N_, d_x, 1, d_y, 1, &result));
    return std::complex<double>(cuCreal(result), cuCimag(result));
}

void GPUFTLMSolver::orthogonalizeAgainstBasis(cuDoubleComplex* d_vec, 
                                             int num_basis_vecs) {
    if (!store_basis_ || !d_lanczos_basis_) return;
    
    // Modified Gram-Schmidt: project out all previous basis vectors
    for (int i = 0; i < num_basis_vecs; i++) {
        std::complex<double> overlap = vectorDot(d_lanczos_basis_[i], d_vec);
        cuDoubleComplex neg_overlap = make_cuDoubleComplex(-overlap.real(), -overlap.imag());
        vectorAxpy(d_lanczos_basis_[i], d_vec, neg_overlap);
    }
}

void GPUFTLMSolver::gramSchmidt(cuDoubleComplex* d_vec, int iter) {
    // Orthogonalize against previous two vectors (sufficient for Lanczos)
    if (iter > 0) {
        std::complex<double> overlap_prev = vectorDot(d_v_current_, d_vec);
        cuDoubleComplex neg_overlap = make_cuDoubleComplex(-overlap_prev.real(), 
                                                           -overlap_prev.imag());
        vectorAxpy(d_v_current_, d_vec, neg_overlap);
    }
    
    if (iter > 1) {
        std::complex<double> overlap_prev2 = vectorDot(d_v_prev_, d_vec);
        cuDoubleComplex neg_overlap = make_cuDoubleComplex(-overlap_prev2.real(), 
                                                           -overlap_prev2.imag());
        vectorAxpy(d_v_prev_, d_vec, neg_overlap);
    }
}

int GPUFTLMSolver::buildLanczosTridiagonal(unsigned int seed,
                                           bool full_reorth,
                                           int reorth_freq,
                                           std::vector<double>& alpha,
                                           std::vector<double>& beta) {
    alpha.clear();
    beta.clear();
    beta.push_back(0.0);  // β₀ is not used
    
    // Setup for full reorthogonalization if requested
    store_basis_ = full_reorth || (reorth_freq > 0);
    if (store_basis_) {
        // Allocate storage for basis vectors
        d_lanczos_basis_ = new cuDoubleComplex*[krylov_dim_];
        for (int i = 0; i < krylov_dim_; i++) {
            CUDA_CHECK(cudaMalloc(&d_lanczos_basis_[i], N_ * sizeof(cuDoubleComplex)));
        }
        num_stored_vectors_ = 0;
    }
    
    // Initialize random starting vector
    initializeRandomVector(d_v_current_, seed);
    
    // Store first basis vector if needed
    if (store_basis_) {
        vectorCopy(d_v_current_, d_lanczos_basis_[0]);
        num_stored_vectors_ = 1;
    }
    
    // Ensure v_prev is zero initially
    CUDA_CHECK(cudaMemset(d_v_prev_, 0, N_ * sizeof(cuDoubleComplex)));
    
    int max_iter = std::min(N_, krylov_dim_);
    
    // Lanczos iteration
    for (int j = 0; j < max_iter; j++) {
        // w = H * v_current
        op_->matVecGPU(d_v_current_, d_w_, N_);
        
        // α_j = ⟨v_current|w⟩
        std::complex<double> alpha_complex = vectorDot(d_v_current_, d_w_);
        double alpha_j = alpha_complex.real();  // Should be real for Hermitian H
        alpha.push_back(alpha_j);
        
        // w = w - α_j * v_current
        cuDoubleComplex neg_alpha = make_cuDoubleComplex(-alpha_j, 0.0);
        vectorAxpy(d_v_current_, d_w_, neg_alpha);
        
        // w = w - β_j * v_prev
        if (j > 0) {
            cuDoubleComplex neg_beta = make_cuDoubleComplex(-beta[j], 0.0);
            vectorAxpy(d_v_prev_, d_w_, neg_beta);
        }
        
        // Reorthogonalization if requested
        if (full_reorth) {
            orthogonalizeAgainstBasis(d_w_, num_stored_vectors_);
        } else if (reorth_freq > 0 && (j % reorth_freq == 0)) {
            orthogonalizeAgainstBasis(d_w_, num_stored_vectors_);
        } else {
            // Standard Lanczos: orthogonalize against previous two vectors
            gramSchmidt(d_w_, j);
        }
        
        // β_{j+1} = ||w||
        double beta_next = vectorNorm(d_w_);
        
        // Check for convergence or breakdown
        if (beta_next < tolerance_) {
            std::cout << "  Lanczos breakdown at iteration " << j + 1 
                     << " (beta = " << beta_next << ")\n";
            beta.push_back(0.0);
            return j + 1;
        }
        
        beta.push_back(beta_next);
        
        // v_next = w / β_{j+1}
        normalizeVector(d_w_);
        
        // Cycle vectors: v_prev <- v_current, v_current <- w
        vectorCopy(d_v_current_, d_v_prev_);
        vectorCopy(d_w_, d_v_current_);
        
        // Store basis vector if needed
        if (store_basis_ && num_stored_vectors_ < krylov_dim_) {
            vectorCopy(d_v_current_, d_lanczos_basis_[num_stored_vectors_]);
            num_stored_vectors_++;
        }
    }
    
    return max_iter;
}

void GPUFTLMSolver::diagonalizeTridiagonal(const std::vector<double>& alpha,
                                           const std::vector<double>& beta,
                                           std::vector<double>& ritz_values,
                                           std::vector<double>& weights) {
    int m = alpha.size();
    
    // Copy to working arrays (LAPACK modifies input)
    std::vector<double> diag = alpha;
    std::vector<double> offdiag(m - 1);
    for (int i = 0; i < m - 1; i++) {
        offdiag[i] = beta[i + 1];  // beta[0] is not used
    }
    
    // Allocate for eigenvectors
    std::vector<double> eigenvectors(m * m);
    
    // Diagonalize using LAPACKE (symmetric tridiagonal eigensolver)
    int info = LAPACKE_dstevd(LAPACK_COL_MAJOR, 'V', m, 
                             diag.data(), offdiag.data(), 
                             eigenvectors.data(), m);
    
    if (info != 0) {
        std::cerr << "Error: LAPACKE_dstevd failed with info = " << info << "\n";
        throw std::runtime_error("Tridiagonal diagonalization failed");
    }
    
    // Extract eigenvalues (Ritz values) and weights
    ritz_values.resize(m);
    weights.resize(m);
    
    for (int i = 0; i < m; i++) {
        ritz_values[i] = diag[i];
        // Weight = |first component of eigenvector|²
        // This gives the overlap with the initial random state
        weights[i] = eigenvectors[i * m] * eigenvectors[i * m];
    }
}

ThermodynamicData GPUFTLMSolver::computeThermodynamics(
    const std::vector<double>& alpha,
    const std::vector<double>& beta,
    const std::vector<double>& temperatures) {
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Diagonalize tridiagonal matrix
    std::vector<double> ritz_values;
    std::vector<double> weights;
    diagonalizeTridiagonal(alpha, beta, ritz_values, weights);
    
    int n_states = ritz_values.size();
    int n_temps = temperatures.size();
    
    // Initialize thermodynamic data
    ThermodynamicData thermo;
    thermo.temperatures = temperatures;
    thermo.energy.resize(n_temps, 0.0);
    thermo.specific_heat.resize(n_temps, 0.0);
    thermo.entropy.resize(n_temps, 0.0);
    thermo.free_energy.resize(n_temps, 0.0);
    
    // Find minimum energy for numerical stability
    double e_min = *std::min_element(ritz_values.begin(), ritz_values.end());
    
    // Compute thermodynamic quantities for each temperature
    for (int t = 0; t < n_temps; t++) {
        double T = temperatures[t];
        double beta = 1.0 / T;
        
        // Compute partition function and averages
        double Z = 0.0;
        double E_avg = 0.0;
        double E2_avg = 0.0;
        
        for (int i = 0; i < n_states; i++) {
            double E_shifted = ritz_values[i] - e_min;
            double boltzmann = std::exp(-beta * E_shifted);
            double w_boltz = weights[i] * boltzmann;
            
            Z += w_boltz;
            E_avg += ritz_values[i] * w_boltz;
            E2_avg += ritz_values[i] * ritz_values[i] * w_boltz;
        }
        
        if (Z < 1e-100) {
            std::cerr << "Warning: Partition function too small at T = " << T << "\n";
            continue;
        }
        
        // Normalize
        E_avg /= Z;
        E2_avg /= Z;
        
        // Compute thermodynamic quantities
        thermo.energy[t] = E_avg;
        
        // Specific heat: C = β² * (⟨E²⟩ - ⟨E⟩²)
        double variance = E2_avg - E_avg * E_avg;
        thermo.specific_heat[t] = beta * beta * variance;
        
        // Entropy: S = β⟨E⟩ + ln(Z) + βE_min
        double log_Z = std::log(Z) + beta * e_min;
        thermo.entropy[t] = beta * E_avg + log_Z;
        
        // Free energy: F = -T * ln(Z) + E_min
        thermo.free_energy[t] = -T * log_Z;
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;
    stats_.thermo_time += elapsed.count();
    
    return thermo;
}

int GPUFTLMSolver::runSingleSample(unsigned int seed,
                                  std::vector<double>& alpha,
                                  std::vector<double>& beta) {
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Build Lanczos tridiagonal (default: selective reorthogonalization)
    int iterations = buildLanczosTridiagonal(seed, false, 10, alpha, beta);
    
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;
    stats_.lanczos_time += elapsed.count();
    stats_.total_iterations += iterations;
    
    return iterations;
}

FTLMResults GPUFTLMSolver::run(int num_samples,
                              double temp_min, double temp_max, int num_temp_bins,
                              const std::string& output_dir,
                              bool full_reorth,
                              int reorth_freq,
                              unsigned int random_seed) {
    
    auto total_start = std::chrono::high_resolution_clock::now();
    
    std::cout << "\n==========================================\n";
    std::cout << "GPU Finite Temperature Lanczos Method\n";
    std::cout << "==========================================\n";
    std::cout << "Hilbert space dimension: " << N_ << "\n";
    std::cout << "Krylov dimension: " << krylov_dim_ << "\n";
    std::cout << "Number of samples: " << num_samples << "\n";
    std::cout << "Temperature range: [" << temp_min << ", " << temp_max << "]\n";
    std::cout << "Temperature bins: " << num_temp_bins << "\n";
    
    // Generate temperature grid (logarithmic spacing)
    std::vector<double> temperatures(num_temp_bins);
    double log_tmin = std::log(temp_min);
    double log_tmax = std::log(temp_max);
    double log_step = (log_tmax - log_tmin) / std::max(1, num_temp_bins - 1);
    
    for (int i = 0; i < num_temp_bins; i++) {
        temperatures[i] = std::exp(log_tmin + i * log_step);
    }
    
    // Initialize random seed
    unsigned int seed = random_seed;
    if (seed == 0) {
        seed = std::chrono::system_clock::now().time_since_epoch().count();
    }
    
    // Storage for results
    FTLMResults results;
    results.total_samples = num_samples;
    std::vector<ThermodynamicData> sample_data;
    std::vector<double> ground_state_estimates;
    
    // Run FTLM for each sample
    for (int sample = 0; sample < num_samples; sample++) {
        std::cout << "\n--- Sample " << (sample + 1) << "/" << num_samples << " ---\n";
        
        unsigned int sample_seed = seed + sample * 12345;
        
        // Build Lanczos tridiagonal
        std::vector<double> alpha, beta;
        int iterations = buildLanczosTridiagonal(sample_seed, full_reorth, 
                                                reorth_freq, alpha, beta);
        
        std::cout << "  Lanczos iterations: " << iterations << "\n";
        
        // Compute thermodynamics from tridiagonal matrix
        ThermodynamicData sample_thermo = computeThermodynamics(alpha, beta, temperatures);
        sample_data.push_back(sample_thermo);
        
        // Estimate ground state
        std::vector<double> ritz_values, weights;
        diagonalizeTridiagonal(alpha, beta, ritz_values, weights);
        double E0_estimate = *std::min_element(ritz_values.begin(), ritz_values.end());
        ground_state_estimates.push_back(E0_estimate);
        
        std::cout << "  Ground state estimate: " << E0_estimate << "\n";
        
        stats_.num_samples_completed++;
    }
    
    // Average over all samples using CPU function
    std::cout << "\n--- Averaging over " << sample_data.size() << " samples ---\n";
    average_ftlm_samples(sample_data, results);
    
    // Estimate ground state energy
    if (!ground_state_estimates.empty()) {
        double E0_min = *std::min_element(ground_state_estimates.begin(), 
                                         ground_state_estimates.end());
        double E0_max = *std::max_element(ground_state_estimates.begin(), 
                                         ground_state_estimates.end());
        double E0_avg = std::accumulate(ground_state_estimates.begin(), 
                                       ground_state_estimates.end(), 0.0) / 
                                       ground_state_estimates.size();
        
        std::cout << "Ground state energy estimates:\n";
        std::cout << "  Min: " << E0_min << "\n";
        std::cout << "  Max: " << E0_max << "\n";
        std::cout << "  Avg: " << E0_avg << "\n";
    }
    
    auto total_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> total_elapsed = total_end - total_start;
    stats_.total_time = total_elapsed.count();
    
    std::cout << "\n==========================================\n";
    std::cout << "GPU FTLM Calculation Complete\n";
    std::cout << "==========================================\n";
    std::cout << "Total time: " << stats_.total_time << " s\n";
    std::cout << "  Lanczos time: " << stats_.lanczos_time << " s\n";
    std::cout << "  Thermodynamics time: " << stats_.thermo_time << " s\n";
    std::cout << "Total iterations: " << stats_.total_iterations << "\n";
    std::cout << "Avg iterations per sample: " 
              << (double)stats_.total_iterations / num_samples << "\n";
    
    return results;
}

// ============================================================================
// Dynamical Response / Spectral Functions
// ============================================================================

int GPUFTLMSolver::buildLanczosTridiagonalFromVector(
    const cuDoubleComplex* d_start_vec,
    bool full_reorth,
    int reorth_freq,
    std::vector<double>& alpha,
    std::vector<double>& beta) {
    
    alpha.clear();
    beta.clear();
    beta.push_back(0.0);
    
    // Setup for full reorthogonalization if requested
    store_basis_ = full_reorth || (reorth_freq > 0);
    if (store_basis_) {
        d_lanczos_basis_ = new cuDoubleComplex*[krylov_dim_];
        for (int i = 0; i < krylov_dim_; i++) {
            CUDA_CHECK(cudaMalloc(&d_lanczos_basis_[i], N_ * sizeof(cuDoubleComplex)));
        }
        num_stored_vectors_ = 0;
    }
    
    // Copy starting vector to v_current and normalize
    vectorCopy(d_start_vec, d_v_current_);
    normalizeVector(d_v_current_);
    
    // Store first basis vector
    if (store_basis_) {
        vectorCopy(d_v_current_, d_lanczos_basis_[0]);
        num_stored_vectors_ = 1;
    }
    
    // Ensure v_prev is zero
    CUDA_CHECK(cudaMemset(d_v_prev_, 0, N_ * sizeof(cuDoubleComplex)));
    
    int max_iter = std::min(N_, krylov_dim_);
    
    // Lanczos iteration
    for (int j = 0; j < max_iter; j++) {
        // w = H * v_current
        op_->matVecGPU(d_v_current_, d_w_, N_);
        
        // α_j = ⟨v_current|w⟩
        std::complex<double> alpha_complex = vectorDot(d_v_current_, d_w_);
        double alpha_j = alpha_complex.real();
        alpha.push_back(alpha_j);
        
        // w = w - α_j * v_current
        cuDoubleComplex neg_alpha = make_cuDoubleComplex(-alpha_j, 0.0);
        vectorAxpy(d_v_current_, d_w_, neg_alpha);
        
        // w = w - β_j * v_prev
        if (j > 0) {
            cuDoubleComplex neg_beta = make_cuDoubleComplex(-beta[j], 0.0);
            vectorAxpy(d_v_prev_, d_w_, neg_beta);
        }
        
        // Reorthogonalization
        if (full_reorth) {
            orthogonalizeAgainstBasis(d_w_, num_stored_vectors_);
        } else if (reorth_freq > 0 && (j % reorth_freq == 0)) {
            orthogonalizeAgainstBasis(d_w_, num_stored_vectors_);
        } else {
            gramSchmidt(d_w_, j);
        }
        
        // β_{j+1} = ||w||
        double beta_next = vectorNorm(d_w_);
        
        if (beta_next < tolerance_) {
            beta.push_back(0.0);
            return j + 1;
        }
        
        beta.push_back(beta_next);
        
        // Normalize and cycle vectors
        normalizeVector(d_w_);
        vectorCopy(d_v_current_, d_v_prev_);
        vectorCopy(d_w_, d_v_current_);
        
        if (store_basis_ && num_stored_vectors_ < krylov_dim_) {
            vectorCopy(d_v_current_, d_lanczos_basis_[num_stored_vectors_]);
            num_stored_vectors_++;
        }
    }
    
    return max_iter;
}

void GPUFTLMSolver::computeSpectralFunction(
    const std::vector<double>& ritz_values,
    const std::vector<double>& weights,
    const std::vector<double>& frequencies,
    double broadening,
    double temperature,
    std::vector<double>& spectral_func) {
    
    int n_states = ritz_values.size();
    int n_omega = frequencies.size();
    spectral_func.resize(n_omega, 0.0);
    
    double e_min = *std::min_element(ritz_values.begin(), ritz_values.end());
    double beta = (temperature > 1e-14) ? 1.0 / temperature : 0.0;
    
    // Compute partition function if needed
    double Z = 0.0;
    if (temperature > 1e-14) {
        for (int i = 0; i < n_states; i++) {
            double e_shifted = ritz_values[i] - e_min;
            Z += weights[i] * std::exp(-beta * e_shifted);
        }
    }
    
    // Compute spectral function with Lorentzian broadening
    for (int iw = 0; iw < n_omega; iw++) {
        double omega = frequencies[iw];
        double sum = 0.0;
        
        for (int i = 0; i < n_states; i++) {
            double E = ritz_values[i];
            double w = weights[i];
            
            // Apply thermal weighting if T > 0
            if (temperature > 1e-14) {
                double e_shifted = E - e_min;
                double boltzmann = std::exp(-beta * e_shifted);
                w *= boltzmann / Z;
            }
            
            // Lorentzian: L(ω - E) = (η/π) / ((ω - E)² + η²)
            double delta = omega - E;
            double lorentzian = (broadening / M_PI) / (delta * delta + broadening * broadening);
            
            sum += w * lorentzian;
        }
        
        spectral_func[iw] = sum;
    }
}

std::pair<std::vector<double>, std::vector<double>>
GPUFTLMSolver::computeDynamicalResponse(
    const cuDoubleComplex* d_psi,
    GPUOperator* op_O,
    double omega_min,
    double omega_max,
    int num_omega_bins,
    double broadening,
    double temperature) {
    
    std::cout << "\n==========================================\n";
    std::cout << "GPU Dynamical Response: S(ω)\n";
    std::cout << "==========================================\n";
    std::cout << "Hilbert space dimension: " << N_ << "\n";
    std::cout << "Krylov dimension: " << krylov_dim_ << "\n";
    std::cout << "Frequency range: [" << omega_min << ", " << omega_max << "]\n";
    std::cout << "Broadening: " << broadening << "\n";
    
    // Generate frequency grid
    std::vector<double> frequencies(num_omega_bins);
    double omega_step = (omega_max - omega_min) / std::max(1, num_omega_bins - 1);
    for (int i = 0; i < num_omega_bins; i++) {
        frequencies[i] = omega_min + i * omega_step;
    }
    
    // Apply operator O to initial state: |φ⟩ = O|ψ⟩
    // If op_O is nullptr, use identity (just copy)
    cuDoubleComplex* d_phi = d_temp_;  // Reuse temp buffer
    
    if (op_O != nullptr) {
        op_O->matVecGPU(d_psi, d_phi, N_);
    } else {
        vectorCopy(d_psi, d_phi);
    }
    
    // Check norm
    double phi_norm = vectorNorm(d_phi);
    if (phi_norm < 1e-14) {
        std::cerr << "Warning: O|ψ⟩ has zero norm\n";
        std::vector<double> zero_spec(num_omega_bins, 0.0);
        return {frequencies, zero_spec};
    }
    
    std::cout << "Norm of O|ψ⟩: " << phi_norm << "\n";
    normalizeVector(d_phi);
    
    // Build Lanczos tridiagonal from |φ⟩
    std::vector<double> alpha, beta;
    int iterations = buildLanczosTridiagonalFromVector(d_phi, false, 10, alpha, beta);
    
    std::cout << "Lanczos iterations: " << iterations << "\n";
    
    // Diagonalize tridiagonal
    std::vector<double> ritz_values, weights;
    diagonalizeTridiagonal(alpha, beta, ritz_values, weights);
    
    // Scale weights by norm factor
    double norm_factor = phi_norm * phi_norm;
    for (auto& w : weights) {
        w *= norm_factor;
    }
    
    std::cout << "Ground state estimate: " << ritz_values[0] << "\n";
    
    // Compute spectral function
    std::vector<double> spectral_func;
    computeSpectralFunction(ritz_values, weights, frequencies, 
                           broadening, temperature, spectral_func);
    
    std::cout << "Dynamical response complete\n";
    
    return {frequencies, spectral_func};
}

std::tuple<std::vector<double>, std::vector<double>, std::vector<double>>
GPUFTLMSolver::computeDynamicalResponseThermal(
    int num_samples,
    GPUOperator* op_O,
    double omega_min,
    double omega_max,
    int num_omega_bins,
    double broadening,
    double temperature,
    unsigned int random_seed) {
    
    std::cout << "\n==========================================\n";
    std::cout << "GPU Thermal Dynamical Response (FTLM)\n";
    std::cout << "==========================================\n";
    std::cout << "Hilbert space dimension: " << N_ << "\n";
    std::cout << "Krylov dimension: " << krylov_dim_ << "\n";
    std::cout << "Number of samples: " << num_samples << "\n";
    std::cout << "Frequency range: [" << omega_min << ", " << omega_max << "]\n";
    std::cout << "Broadening: " << broadening << "\n";
    
    // Generate frequency grid
    std::vector<double> frequencies(num_omega_bins);
    double omega_step = (omega_max - omega_min) / std::max(1, num_omega_bins - 1);
    for (int i = 0; i < num_omega_bins; i++) {
        frequencies[i] = omega_min + i * omega_step;
    }
    
    // Initialize seed
    unsigned int seed = random_seed;
    if (seed == 0) {
        seed = std::chrono::system_clock::now().time_since_epoch().count();
    }
    
    // Storage for sample spectral functions
    std::vector<std::vector<double>> sample_spectra;
    
    // Loop over samples
    for (int sample = 0; sample < num_samples; sample++) {
        std::cout << "\n--- Sample " << (sample + 1) << "/" << num_samples << " ---\n";
        
        unsigned int sample_seed = seed + sample * 12345;
        
        // Generate random state |ψ⟩
        initializeRandomVector(d_v_current_, sample_seed);
        
        // Apply O: |φ⟩ = O|ψ⟩
        cuDoubleComplex* d_phi = d_temp_;
        if (op_O != nullptr) {
            op_O->matVecGPU(d_v_current_, d_phi, N_);
        } else {
            vectorCopy(d_v_current_, d_phi);
        }
        
        double phi_norm = vectorNorm(d_phi);
        if (phi_norm < 1e-14) {
            std::cout << "  Warning: O|ψ⟩ has zero norm, skipping\n";
            continue;
        }
        
        normalizeVector(d_phi);
        
        // Build Lanczos tridiagonal
        std::vector<double> alpha, beta;
        int iterations = buildLanczosTridiagonalFromVector(d_phi, false, 10, alpha, beta);
        
        std::cout << "  Lanczos iterations: " << iterations << "\n";
        
        // Diagonalize
        std::vector<double> ritz_values, weights;
        diagonalizeTridiagonal(alpha, beta, ritz_values, weights);
        
        // Scale weights
        double norm_factor = phi_norm * phi_norm;
        for (auto& w : weights) {
            w *= norm_factor;
        }
        
        // Compute spectral function for this sample
        std::vector<double> spec;
        computeSpectralFunction(ritz_values, weights, frequencies, 
                               broadening, temperature, spec);
        
        sample_spectra.push_back(spec);
    }
    
    // Average over samples and compute error bars
    std::vector<double> avg_spec(num_omega_bins, 0.0);
    std::vector<double> error_spec(num_omega_bins, 0.0);
    
    int valid_samples = sample_spectra.size();
    
    if (valid_samples == 0) {
        std::cerr << "Error: No valid samples\n";
        return {frequencies, avg_spec, error_spec};
    }
    
    // Compute average
    for (int iw = 0; iw < num_omega_bins; iw++) {
        for (int s = 0; s < valid_samples; s++) {
            avg_spec[iw] += sample_spectra[s][iw];
        }
        avg_spec[iw] /= valid_samples;
    }
    
    // Compute standard error
    if (valid_samples > 1) {
        for (int iw = 0; iw < num_omega_bins; iw++) {
            double variance = 0.0;
            for (int s = 0; s < valid_samples; s++) {
                double diff = sample_spectra[s][iw] - avg_spec[iw];
                variance += diff * diff;
            }
            error_spec[iw] = std::sqrt(variance / (valid_samples * (valid_samples - 1)));
        }
    }
    
    std::cout << "\n==========================================\n";
    std::cout << "Thermal dynamical response complete\n";
    std::cout << "Valid samples: " << valid_samples << "\n";
    std::cout << "==========================================\n";
    
    return {frequencies, avg_spec, error_spec};
}

std::tuple<std::vector<double>, std::vector<double>, std::vector<double>,
           std::vector<double>, std::vector<double>>
GPUFTLMSolver::computeDynamicalCorrelation(
    int num_samples,
    GPUOperator* op_O1,
    GPUOperator* op_O2,
    double omega_min,
    double omega_max,
    int num_omega_bins,
    double broadening,
    double temperature,
    unsigned int random_seed) {
    
    std::cout << "\n==========================================\n";
    std::cout << "GPU Dynamical Correlation: S_{O1,O2}(ω)\n";
    std::cout << "==========================================\n";
    std::cout << "Hilbert space dimension: " << N_ << "\n";
    std::cout << "Krylov dimension: " << krylov_dim_ << "\n";
    std::cout << "Number of samples: " << num_samples << "\n";
    std::cout << "Frequency range: [" << omega_min << ", " << omega_max << "]\n";
    std::cout << "Broadening: " << broadening << "\n";
    
    // For cross-correlation, we would need to compute complex weights
    // and keep track of both real and imaginary parts
    // This is a simplified version that assumes O1 = O2 (auto-correlation)
    
    if (op_O1 != op_O2 && op_O1 != nullptr && op_O2 != nullptr) {
        std::cerr << "Warning: Cross-correlation (O1 != O2) not yet fully implemented\n";
        std::cerr << "         Falling back to O2 auto-correlation\n";
    }
    
    // Use the simpler thermal response for now
    auto result = computeDynamicalResponseThermal(
        num_samples, op_O2, omega_min, omega_max, num_omega_bins, 
        broadening, temperature, random_seed);
    
    std::vector<double> frequencies = std::get<0>(result);
    std::vector<double> spec_real = std::get<1>(result);
    std::vector<double> error_real = std::get<2>(result);
    
    // For auto-correlation, imaginary part is zero
    std::vector<double> spec_imag(num_omega_bins, 0.0);
    std::vector<double> error_imag(num_omega_bins, 0.0);
    
    return std::make_tuple(frequencies, spec_real, spec_imag, error_real, error_imag);
}

#endif // WITH_CUDA
