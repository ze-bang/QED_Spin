// gpu_ftlm.cu - GPU-accelerated Finite Temperature Lanczos Method
#include <ed/gpu/gpu_ftlm.cuh>
#include <ed/core/hdf5_io.h>  // For HDF5 output

#ifdef WITH_CUDA

#include <curand_kernel.h>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <lapacke.h>
#include <cblas.h>
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

int GPUFTLMSolver::buildLanczosTridiagonalWithBasis(
    const cuDoubleComplex* d_start_vec,
    bool full_reorth,
    int reorth_freq,
    std::vector<double>& alpha,
    std::vector<double>& beta,
    cuDoubleComplex*** d_basis_out) {
    
    alpha.clear();
    beta.clear();
    beta.push_back(0.0);
    
    // Always store basis for this version
    cuDoubleComplex** d_basis = new cuDoubleComplex*[krylov_dim_];
    for (int i = 0; i < krylov_dim_; i++) {
        CUDA_CHECK(cudaMalloc(&d_basis[i], N_ * sizeof(cuDoubleComplex)));
    }
    int num_vecs = 0;
    
    // Copy starting vector to v_current and normalize
    vectorCopy(d_start_vec, d_v_current_);
    normalizeVector(d_v_current_);
    
    // Store first basis vector
    vectorCopy(d_v_current_, d_basis[0]);
    num_vecs = 1;
    
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
            // Full reorthogonalization against all stored vectors
            for (int k = 0; k < num_vecs; k++) {
                std::complex<double> overlap = vectorDot(d_basis[k], d_w_);
                cuDoubleComplex neg_overlap = make_cuDoubleComplex(-overlap.real(), -overlap.imag());
                vectorAxpy(d_basis[k], d_w_, neg_overlap);
            }
        } else if (reorth_freq > 0 && (j % reorth_freq == 0)) {
            // Periodic reorthogonalization
            for (int k = 0; k < num_vecs; k++) {
                std::complex<double> overlap = vectorDot(d_basis[k], d_w_);
                cuDoubleComplex neg_overlap = make_cuDoubleComplex(-overlap.real(), -overlap.imag());
                vectorAxpy(d_basis[k], d_w_, neg_overlap);
            }
        } else {
            // Standard Lanczos: orthogonalize against previous two vectors
            gramSchmidt(d_w_, j);
        }
        
        // β_{j+1} = ||w||
        double beta_next = vectorNorm(d_w_);
        
        if (beta_next < tolerance_) {
            beta.push_back(0.0);
            *d_basis_out = d_basis;
            return j + 1;
        }
        
        beta.push_back(beta_next);
        
        // Normalize and cycle vectors
        normalizeVector(d_w_);
        vectorCopy(d_v_current_, d_v_prev_);
        vectorCopy(d_w_, d_v_current_);
        
        // Store next basis vector
        if (num_vecs < krylov_dim_) {
            vectorCopy(d_v_current_, d_basis[num_vecs]);
            num_vecs++;
        }
    }
    
    *d_basis_out = d_basis;
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

void GPUFTLMSolver::computeSpectralFunctionComplex(
    const std::vector<double>& ritz_values,
    const std::vector<std::complex<double>>& complex_weights,
    const std::vector<double>& frequencies,
    double broadening,
    double temperature,
    std::vector<double>& spectral_func_real,
    std::vector<double>& spectral_func_imag) {
    
    int n_states = ritz_values.size();
    int n_omega = frequencies.size();
    
    spectral_func_real.resize(n_omega, 0.0);
    spectral_func_imag.resize(n_omega, 0.0);
    
    // Compute thermal weights if temperature > 0
    std::vector<std::complex<double>> thermal_weights = complex_weights;
    
    if (temperature > 1e-14) {
        double beta = 1.0 / temperature;
        
        // Find minimum energy for numerical stability
        double e_min = *std::min_element(ritz_values.begin(), ritz_values.end());
        
        // Compute partition function with shifted energies
        double Z = 0.0;
        for (int i = 0; i < n_states; i++) {
            double shifted_energy = ritz_values[i] - e_min;
            double boltzmann_factor = std::exp(-beta * shifted_energy);
            Z += boltzmann_factor;
        }
        
        // Apply thermal weights: w_n → w_n * exp(-βE_n) / Z
        if (Z > 1e-300) {
            for (int i = 0; i < n_states; i++) {
                double shifted_energy = ritz_values[i] - e_min;
                double boltzmann_factor = std::exp(-beta * shifted_energy);
                thermal_weights[i] = complex_weights[i] * (boltzmann_factor / Z);
            }
        } else {
            // Very low temperature - only ground state contributes
            thermal_weights.assign(n_states, std::complex<double>(0.0, 0.0));
            int gs_idx = std::distance(ritz_values.begin(),
                                      std::min_element(ritz_values.begin(), ritz_values.end()));
            thermal_weights[gs_idx] = complex_weights[gs_idx];
            
            // Normalize by complex magnitude
            std::complex<double> sum = std::complex<double>(0.0, 0.0);
            for (const auto& w : thermal_weights) sum += w;
            if (std::abs(sum) > 1e-300) {
                for (auto& w : thermal_weights) w /= sum;
            }
        }
    }
    
    // For each frequency, sum contributions from all states
    // S(ω,T) = Σ_i w_i * exp(-βE_i)/Z * δ(ω - E_i)
    // Using Lorentzian broadening: δ(ω - E) → (η/π) / ((ω - E)² + η²)
    double norm_factor = broadening / M_PI;
    
    for (int i_omega = 0; i_omega < n_omega; i_omega++) {
        double omega = frequencies[i_omega];
        
        for (int i = 0; i < n_states; i++) {
            double delta = omega - ritz_values[i];
            double lorentzian = norm_factor / (delta * delta + broadening * broadening);
            std::complex<double> contribution = thermal_weights[i] * lorentzian;
            spectral_func_real[i_omega] += contribution.real();
            spectral_func_imag[i_omega] += contribution.imag();
        }
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
    double energy_shift,
    unsigned int random_seed,
    const std::string& output_dir,
    bool store_intermediate) {
    
    std::cout << "\n==========================================\n";
    std::cout << "GPU Dynamical Correlation: S(ω) = <O₁†δ(ω-H)O₂>\n";
    std::cout << "==========================================\n";
    std::cout << "Hilbert space dimension: " << N_ << "\n";
    std::cout << "Krylov dimension: " << krylov_dim_ << "\n";
    std::cout << "Number of samples: " << num_samples << "\n";
    std::cout << "Frequency range: [" << omega_min << ", " << omega_max << "]\n";
    std::cout << "Broadening: " << broadening << "\n";
    if (temperature > 1e-14) {
        std::cout << "Temperature: " << temperature << "\n";
    } else {
        std::cout << "Temperature: 0 (no thermal weighting)\n";
    }
    
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
    
    // Storage for per-sample spectral functions (real and imaginary parts)
    std::vector<std::vector<double>> sample_spectra_real;
    std::vector<std::vector<double>> sample_spectra_imag;
    
    // Create output directory if needed
    if (!output_dir.empty() && store_intermediate) {
        std::string cmd = "mkdir -p " + output_dir + "/dynamical_correlation_samples";
        system(cmd.c_str());
    }
    
    // Allocate temporary device vectors for operator applications
    cuDoubleComplex* d_phi = nullptr;
    cuDoubleComplex* d_O1_psi = nullptr;
    CUDA_CHECK(cudaMalloc(&d_phi, N_ * sizeof(cuDoubleComplex)));
    CUDA_CHECK(cudaMalloc(&d_O1_psi, N_ * sizeof(cuDoubleComplex)));
    
    // Host vectors for data transfer
    std::vector<cuDoubleComplex> h_psi(N_);
    std::vector<cuDoubleComplex> h_O1_psi(N_);
    
    // Loop over random samples
    for (int sample = 0; sample < num_samples; sample++) {
        std::cout << "\n--- Sample " << (sample + 1) << " / " << num_samples << " ---\n";
        
        unsigned int sample_seed = seed + sample * 12345;
        
        // Generate random initial state |ψ⟩
        initializeRandomVector(d_v_current_, sample_seed);
        
        // Apply operator O2: |φ⟩ = O₂|ψ⟩
        if (op_O2 != nullptr) {
            op_O2->matVecGPU(d_v_current_, d_phi, N_);
        } else {
            vectorCopy(d_v_current_, d_phi);
        }
        
        // Get norm of |φ⟩
        double phi_norm = vectorNorm(d_phi);
        if (phi_norm < 1e-14) {
            std::cout << "  Warning: O₂|ψ⟩ has zero norm, skipping sample\n";
            continue;
        }
        
        std::cout << "  Norm of O₂|ψ⟩: " << phi_norm << "\n";
        
        // Normalize |φ⟩
        normalizeVector(d_phi);
        
        // Build Lanczos tridiagonal and store basis vectors
        std::vector<double> alpha, beta;
        cuDoubleComplex** d_lanczos_basis = nullptr;
        
        int iterations = buildLanczosTridiagonalWithBasis(
            d_phi, false, 10, alpha, beta, &d_lanczos_basis);
        
        int m = alpha.size();
        std::cout << "  Lanczos iterations: " << m << "\n";
        
        // Diagonalize tridiagonal (need eigenvectors for weight computation)
        std::vector<double> ritz_values(m);
        std::vector<double> dummy_weights(m);
        std::vector<double> evecs(m * m);
        
        // Copy alpha/beta for diagonalization
        std::vector<double> diag = alpha;
        std::vector<double> offdiag(m - 1);
        for (int i = 0; i < m - 1; i++) {
            offdiag[i] = beta[i + 1];
        }
        
        // Diagonalize using LAPACKE
        int info = LAPACKE_dstevd(LAPACK_COL_MAJOR, 'V', m, 
                                 diag.data(), offdiag.data(), 
                                 evecs.data(), m);
        
        if (info != 0) {
            std::cerr << "  Warning: Tridiagonal diagonalization failed\n";
            
            // Free basis vectors
            for (int i = 0; i < m; i++) {
                cudaFree(d_lanczos_basis[i]);
            }
            delete[] d_lanczos_basis;
            continue;
        }
        
        // Extract eigenvalues
        for (int i = 0; i < m; i++) {
            ritz_values[i] = diag[i];
        }
        
        // Apply energy shift
        if (sample == 0) {
            double E_shift;
            if (std::abs(energy_shift) > 1e-14) {
                E_shift = energy_shift;
                std::cout << "  Using provided ground state energy shift: " << E_shift << "\n";
            } else {
                E_shift = *std::min_element(ritz_values.begin(), ritz_values.end());
                std::cout << "  Ground state energy (auto-detected from Krylov): " << E_shift << "\n";
            }
            std::cout << "  Shifting to excitation energies (E_gs = 0)\n";
        }
        
        double E_shift = (std::abs(energy_shift) > 1e-14) ? 
                         energy_shift : 
                         *std::min_element(ritz_values.begin(), ritz_values.end());
        
        for (int i = 0; i < m; i++) {
            ritz_values[i] -= E_shift;
        }
        
        // Compute complex weights ⟨ψ|O₁†|n⟩⟨n|O₂|ψ⟩
        // Apply O1 to original state: |O₁ψ⟩
        if (op_O1 != nullptr) {
            op_O1->matVecGPU(d_v_current_, d_O1_psi, N_);
        } else {
            vectorCopy(d_v_current_, d_O1_psi);
        }
        
        // Transfer to host for inner product computations
        CUDA_CHECK(cudaMemcpy(h_O1_psi.data(), d_O1_psi, 
                             N_ * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost));
        
        std::vector<std::complex<double>> complex_weights(m);
        
        for (int n = 0; n < m; n++) {
            // Compute ⟨ψ|O₁†|n⟩ = ⟨O₁ψ|n⟩ = Σⱼ evecs[n,j] ⟨O₁ψ|vⱼ⟩
            std::complex<double> overlap_O1(0.0, 0.0);
            
            for (int j = 0; j < m; j++) {
                // Transfer basis vector to host
                std::vector<cuDoubleComplex> h_basis_j(N_);
                CUDA_CHECK(cudaMemcpy(h_basis_j.data(), d_lanczos_basis[j],
                                     N_ * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost));
                
                // Compute ⟨O₁ψ|vⱼ⟩
                cuDoubleComplex bracket;
                cblas_zdotc_sub(N_, h_O1_psi.data(), 1, h_basis_j.data(), 1, &bracket);
                
                std::complex<double> bracket_cpp(cuCreal(bracket), cuCimag(bracket));
                overlap_O1 += evecs[n * m + j] * bracket_cpp;
            }
            
            // Compute ⟨n|O₂|ψ⟩ = evecs[n,0] × ||O₂|ψ|| (since |v₀⟩ = O₂|ψ⟩/||O₂|ψ||)
            std::complex<double> overlap_O2(evecs[n * m + 0] * phi_norm, 0.0);
            
            // Weight is ⟨ψ|O₁†|n⟩⟨n|O₂|ψ⟩ = conj(⟨O₁ψ|n⟩) × ⟨n|O₂|ψ⟩
            complex_weights[n] = std::conj(overlap_O1) * overlap_O2;
        }
        
        // Free basis vectors for this sample
        for (int i = 0; i < m; i++) {
            cudaFree(d_lanczos_basis[i]);
        }
        delete[] d_lanczos_basis;
        
        // Compute spectral function for this sample (both real and imaginary parts)
        std::vector<double> sample_spectrum_real, sample_spectrum_imag;
        computeSpectralFunctionComplex(ritz_values, complex_weights, frequencies,
                                      broadening, temperature, 
                                      sample_spectrum_real, sample_spectrum_imag);
        
        sample_spectra_real.push_back(sample_spectrum_real);
        sample_spectra_imag.push_back(sample_spectrum_imag);
        
        // Save intermediate data if requested (to HDF5)
        if (store_intermediate && !output_dir.empty()) {
            std::string h5_file = output_dir + "/ed_results.h5";
            if (!HDF5IO::fileExists(h5_file)) {
                HDF5IO::createOrOpenFile(output_dir);
            }
            
            HDF5IO::FTLMDynamicalSample h5_sample;
            h5_sample.frequencies = frequencies;
            h5_sample.spectral_real = sample_spectrum_real;
            h5_sample.spectral_imag = sample_spectrum_imag;
            
            HDF5IO::saveFTLMDynamicalSample(h5_file, sample, h5_sample, true);  // is_correlation=true
        }
    }
    
    // Free temporary device memory
    cudaFree(d_phi);
    cudaFree(d_O1_psi);
    
    // Average over all samples
    int n_valid_samples = sample_spectra_real.size();
    std::cout << "\n--- Averaging over " << n_valid_samples << " samples ---\n";
    
    std::vector<double> avg_spec_real(num_omega_bins, 0.0);
    std::vector<double> avg_spec_imag(num_omega_bins, 0.0);
    std::vector<double> error_real(num_omega_bins, 0.0);
    std::vector<double> error_imag(num_omega_bins, 0.0);
    
    if (n_valid_samples == 0) {
        std::cerr << "Error: No valid samples obtained\n";
        return std::make_tuple(frequencies, avg_spec_real, avg_spec_imag, error_real, error_imag);
    }
    
    // Compute mean (real and imaginary parts)
    for (int s = 0; s < n_valid_samples; s++) {
        for (int i = 0; i < num_omega_bins; i++) {
            avg_spec_real[i] += sample_spectra_real[s][i];
            avg_spec_imag[i] += sample_spectra_imag[s][i];
        }
    }
    
    for (int i = 0; i < num_omega_bins; i++) {
        avg_spec_real[i] /= n_valid_samples;
        avg_spec_imag[i] /= n_valid_samples;
    }
    
    // Compute standard error (real and imaginary parts)
    if (n_valid_samples > 1) {
        for (int s = 0; s < n_valid_samples; s++) {
            for (int i = 0; i < num_omega_bins; i++) {
                double diff_real = sample_spectra_real[s][i] - avg_spec_real[i];
                double diff_imag = sample_spectra_imag[s][i] - avg_spec_imag[i];
                error_real[i] += diff_real * diff_real;
                error_imag[i] += diff_imag * diff_imag;
            }
        }
        
        double norm_factor = std::sqrt(static_cast<double>(n_valid_samples * (n_valid_samples - 1)));
        for (int i = 0; i < num_omega_bins; i++) {
            error_real[i] = std::sqrt(error_real[i]) / norm_factor;
            error_imag[i] = std::sqrt(error_imag[i]) / norm_factor;
        }
    }
    
    std::cout << "\n==========================================\n";
    std::cout << "GPU Dynamical Correlation Complete\n";
    std::cout << "==========================================\n";
    
    return std::make_tuple(frequencies, avg_spec_real, avg_spec_imag, error_real, error_imag);
}

// ============================================================================
// STATIC RESPONSE FUNCTIONS
// ============================================================================

/**
 * @brief Compute thermal expectation value on GPU
 */
std::tuple<std::vector<double>, std::vector<double>, std::vector<double>>
GPUFTLMSolver::computeThermalExpectation(
    int num_samples,
    GPUOperator* op_O,
    double temp_min,
    double temp_max,
    int num_temp_bins,
    unsigned int random_seed,
    const std::string& output_dir,
    bool store_intermediate) {
    
    std::cout << "\n==========================================\n";
    std::cout << "GPU Thermal Expectation Value (FTLM)\n";
    std::cout << "==========================================\n";
    std::cout << "Hilbert space dimension: " << N_ << "\n";
    std::cout << "Krylov dimension: " << krylov_dim_ << "\n";
    std::cout << "Number of samples: " << num_samples << "\n";
    std::cout << "Temperature range: [" << temp_min << ", " << temp_max << "]\n";
    
    // Generate temperature grid (logarithmic spacing)
    std::vector<double> temperatures(num_temp_bins);
    double log_tmin = std::log(temp_min);
    double log_tmax = std::log(temp_max);
    double log_step = (log_tmax - log_tmin) / std::max(1, num_temp_bins - 1);
    
    for (int i = 0; i < num_temp_bins; i++) {
        temperatures[i] = std::exp(log_tmin + i * log_step);
    }
    
    // Initialize seed
    unsigned int seed = random_seed;
    if (seed == 0) {
        seed = std::chrono::system_clock::now().time_since_epoch().count();
    }
    
    // Storage for per-sample results
    std::vector<std::vector<double>> sample_expectations(num_samples);
    std::vector<std::vector<double>> sample_variances(num_samples);
    
    // Create output directory if needed
    if (!output_dir.empty() && store_intermediate) {
        std::string cmd = "mkdir -p " + output_dir + "/static_samples";
        system(cmd.c_str());
    }
    
    // Allocate temporary device vector for O|ψ⟩
    cuDoubleComplex* d_O_psi = nullptr;
    CUDA_CHECK(cudaMalloc(&d_O_psi, N_ * sizeof(cuDoubleComplex)));
    
    // Host vectors for data transfer
    std::vector<cuDoubleComplex> h_psi(N_);
    std::vector<cuDoubleComplex> h_O_psi(N_);
    
    // Loop over samples
    for (int sample = 0; sample < num_samples; sample++) {
        std::cout << "\n--- Sample " << (sample + 1) << " / " << num_samples << " ---\n";
        
        unsigned int sample_seed = seed + sample * 12345;
        
        // Generate random initial state
        initializeRandomVector(d_v_current_, sample_seed);
        
        // Build Lanczos tridiagonal and store basis vectors
        std::vector<double> alpha, beta;
        cuDoubleComplex** d_lanczos_basis = nullptr;
        
        int iterations = buildLanczosTridiagonalWithBasis(
            d_v_current_, false, 10, alpha, beta, &d_lanczos_basis);
        
        int m = alpha.size();
        std::cout << "  Lanczos iterations: " << m << "\n";
        
        if (m == 0) {
            std::cerr << "  Warning: Failed to build Krylov subspace\n";
            continue;
        }
        
        // Diagonalize tridiagonal
        std::vector<double> ritz_values, weights;
        diagonalizeTridiagonal(alpha, beta, ritz_values, weights);
        
        // Diagonalize with eigenvectors for computing expectation values
        std::vector<double> evecs(m * m);
        std::vector<double> diag = alpha;
        std::vector<double> offdiag(m - 1);
        for (int i = 0; i < m - 1; i++) {
            offdiag[i] = beta[i + 1];
        }
        
        int info = LAPACKE_dstevd(LAPACK_COL_MAJOR, 'V', m,
                                 diag.data(), offdiag.data(),
                                 evecs.data(), m);
        
        if (info != 0) {
            std::cerr << "  Warning: Tridiagonal diagonalization failed\n";
            for (int i = 0; i < m; i++) {
                cudaFree(d_lanczos_basis[i]);
            }
            delete[] d_lanczos_basis;
            continue;
        }
        
        for (int i = 0; i < m; i++) {
            ritz_values[i] = diag[i];
        }
        
        // Compute ⟨n|O|n⟩ for each eigenstate |n⟩
        std::vector<double> expectation_values(m);
        
        for (int n = 0; n < m; n++) {
            // Reconstruct |n⟩ in full Hilbert space
            CUDA_CHECK(cudaMemset(d_temp_, 0, N_ * sizeof(cuDoubleComplex)));
            
            for (int j = 0; j < m; j++) {
                double coeff = evecs[n * m + j];
                cuDoubleComplex alpha_gpu = make_cuDoubleComplex(coeff, 0.0);
                vectorAxpy(d_lanczos_basis[j], d_temp_, alpha_gpu);
            }
            
            // Apply O to |n⟩
            if (op_O != nullptr) {
                op_O->matVecGPU(d_temp_, d_O_psi, N_);
            } else {
                vectorCopy(d_temp_, d_O_psi);
            }
            
            // Compute ⟨n|O|n⟩
            std::complex<double> exp_val = vectorDot(d_temp_, d_O_psi);
            expectation_values[n] = exp_val.real();
        }
        
        // Free basis vectors
        for (int i = 0; i < m; i++) {
            cudaFree(d_lanczos_basis[i]);
        }
        delete[] d_lanczos_basis;
        
        // Compute thermal averages for this sample
        sample_expectations[sample].resize(num_temp_bins);
        sample_variances[sample].resize(num_temp_bins);
        
        double e_min = *std::min_element(ritz_values.begin(), ritz_values.end());
        
        for (int t = 0; t < num_temp_bins; t++) {
            double T = temperatures[t];
            double beta = 1.0 / T;
            
            double Z = 0.0;
            double O_avg = 0.0;
            double O2_avg = 0.0;
            
            for (int i = 0; i < m; i++) {
                double shifted_energy = ritz_values[i] - e_min;
                double boltzmann = weights[i] * std::exp(-beta * shifted_energy);
                Z += boltzmann;
            }
            
            if (Z > 1e-300) {
                for (int i = 0; i < m; i++) {
                    double shifted_energy = ritz_values[i] - e_min;
                    double prob = weights[i] * std::exp(-beta * shifted_energy) / Z;
                    O_avg += prob * expectation_values[i];
                    O2_avg += prob * expectation_values[i] * expectation_values[i];
                }
                
                sample_expectations[sample][t] = O_avg;
                sample_variances[sample][t] = O2_avg - O_avg * O_avg;
            } else {
                int gs_idx = std::distance(ritz_values.begin(),
                                          std::min_element(ritz_values.begin(), ritz_values.end()));
                sample_expectations[sample][t] = expectation_values[gs_idx];
                sample_variances[sample][t] = 0.0;
            }
        }
        
        // Save intermediate data if requested (to HDF5)
        if (store_intermediate && !output_dir.empty()) {
            std::string h5_file = output_dir + "/ed_results.h5";
            if (!HDF5IO::fileExists(h5_file)) {
                HDF5IO::createOrOpenFile(output_dir);
            }
            
            HDF5IO::FTLMStaticSample h5_sample;
            h5_sample.temperatures = temperatures;
            h5_sample.expectation = sample_expectations[sample];
            h5_sample.variance = sample_variances[sample];
            
            HDF5IO::saveFTLMStaticSample(h5_file, sample, h5_sample);
        }
    }
    
    // Free temporary memory
    cudaFree(d_O_psi);
    
    // Average over samples
    int n_valid_samples = 0;
    for (int s = 0; s < num_samples; s++) {
        if (!sample_expectations[s].empty()) n_valid_samples++;
    }
    
    std::cout << "\n--- Averaging over " << n_valid_samples << " samples ---\n";
    
    std::vector<double> expectation(num_temp_bins, 0.0);
    std::vector<double> susceptibility(num_temp_bins, 0.0);
    std::vector<double> error(num_temp_bins, 0.0);
    
    if (n_valid_samples == 0) {
        std::cerr << "Error: No valid samples\n";
        return std::make_tuple(temperatures, expectation, error);
    }
    
    // Compute means
    for (int s = 0; s < num_samples; s++) {
        if (sample_expectations[s].empty()) continue;
        for (int t = 0; t < num_temp_bins; t++) {
            expectation[t] += sample_expectations[s][t];
            susceptibility[t] += sample_variances[s][t];
        }
    }
    
    for (int t = 0; t < num_temp_bins; t++) {
        expectation[t] /= n_valid_samples;
        susceptibility[t] /= n_valid_samples;
        double beta = 1.0 / temperatures[t];
        susceptibility[t] *= beta;  // χ = β * variance
    }
    
    // Compute standard errors
    if (n_valid_samples > 1) {
        for (int s = 0; s < num_samples; s++) {
            if (sample_expectations[s].empty()) continue;
            for (int t = 0; t < num_temp_bins; t++) {
                double diff = sample_expectations[s][t] - expectation[t];
                error[t] += diff * diff;
            }
        }
        
        double norm = std::sqrt(static_cast<double>(n_valid_samples * (n_valid_samples - 1)));
        for (int t = 0; t < num_temp_bins; t++) {
            error[t] = std::sqrt(error[t]) / norm;
        }
    }
    
    std::cout << "\n==========================================\n";
    std::cout << "GPU Static Response Complete\n";
    std::cout << "==========================================\n";
    
    return std::make_tuple(temperatures, expectation, error);
}

/**
 * @brief Compute static correlation function on GPU
 */
std::tuple<std::vector<double>, std::vector<double>, std::vector<double>>
GPUFTLMSolver::computeStaticCorrelation(
    int num_samples,
    GPUOperator* op_O1,
    GPUOperator* op_O2,
    double temp_min,
    double temp_max,
    int num_temp_bins,
    unsigned int random_seed,
    const std::string& output_dir,
    bool store_intermediate) {
    
    std::cout << "\n==========================================\n";
    std::cout << "GPU Static Correlation: ⟨O₁†O₂⟩\n";
    std::cout << "==========================================\n";
    std::cout << "Hilbert space dimension: " << N_ << "\n";
    std::cout << "Krylov dimension: " << krylov_dim_ << "\n";
    std::cout << "Number of samples: " << num_samples << "\n";
    std::cout << "Temperature range: [" << temp_min << ", " << temp_max << "]\n";
    
    // Generate temperature grid
    std::vector<double> temperatures(num_temp_bins);
    double temp_step = (temp_max - temp_min) / std::max(1, num_temp_bins - 1);
    for (int i = 0; i < num_temp_bins; i++) {
        temperatures[i] = temp_min + i * temp_step;
    }
    
    // Initialize seed
    unsigned int seed = random_seed;
    if (seed == 0) {
        seed = std::chrono::system_clock::now().time_since_epoch().count();
    }
    
    // Storage for per-sample results
    std::vector<std::vector<double>> sample_expectations;
    
    // Create output directory if needed
    if (!output_dir.empty() && store_intermediate) {
        std::string cmd = "mkdir -p " + output_dir + "/static_correlation_samples";
        system(cmd.c_str());
    }
    
    // Allocate temporary device vectors
    cuDoubleComplex* d_O1_psi = nullptr;
    cuDoubleComplex* d_O2_psi = nullptr;
    CUDA_CHECK(cudaMalloc(&d_O1_psi, N_ * sizeof(cuDoubleComplex)));
    CUDA_CHECK(cudaMalloc(&d_O2_psi, N_ * sizeof(cuDoubleComplex)));
    
    // Loop over random samples
    for (int sample = 0; sample < num_samples; sample++) {
        std::cout << "\n--- Sample " << (sample + 1) << " / " << num_samples << " ---\n";
        
        unsigned int sample_seed = seed + sample * 12345;
        
        // Generate random initial state
        initializeRandomVector(d_v_current_, sample_seed);
        
        // Build Lanczos tridiagonal and store basis vectors
        std::vector<double> alpha, beta;
        cuDoubleComplex** d_lanczos_basis = nullptr;
        
        int iterations = buildLanczosTridiagonalWithBasis(
            d_v_current_, false, 10, alpha, beta, &d_lanczos_basis);
        
        int m = alpha.size();
        std::cout << "  Lanczos iterations: " << m << "\n";
        
        // Diagonalize tridiagonal with eigenvectors
        std::vector<double> ritz_values, weights;
        std::vector<double> evecs(m * m);
        std::vector<double> diag = alpha;
        std::vector<double> offdiag(m - 1);
        for (int i = 0; i < m - 1; i++) {
            offdiag[i] = beta[i + 1];
        }
        
        int info = LAPACKE_dstevd(LAPACK_COL_MAJOR, 'V', m,
                                 diag.data(), offdiag.data(),
                                 evecs.data(), m);
        
        if (info != 0) {
            std::cerr << "  Warning: Tridiagonal diagonalization failed\n";
            for (int i = 0; i < m; i++) {
                cudaFree(d_lanczos_basis[i]);
            }
            delete[] d_lanczos_basis;
            continue;
        }
        
        diagonalizeTridiagonal(alpha, beta, ritz_values, weights);
        
        // Compute ⟨n|O₁†O₂|n⟩ for each eigenstate
        std::vector<double> correlation_values(m);
        
        for (int n = 0; n < m; n++) {
            // Reconstruct |n⟩
            CUDA_CHECK(cudaMemset(d_temp_, 0, N_ * sizeof(cuDoubleComplex)));
            
            for (int j = 0; j < m; j++) {
                double coeff = evecs[n * m + j];
                cuDoubleComplex alpha_gpu = make_cuDoubleComplex(coeff, 0.0);
                vectorAxpy(d_lanczos_basis[j], d_temp_, alpha_gpu);
            }
            
            // Apply O₁ and O₂
            if (op_O1 != nullptr) {
                op_O1->matVecGPU(d_temp_, d_O1_psi, N_);
            } else {
                vectorCopy(d_temp_, d_O1_psi);
            }
            
            if (op_O2 != nullptr) {
                op_O2->matVecGPU(d_temp_, d_O2_psi, N_);
            } else {
                vectorCopy(d_temp_, d_O2_psi);
            }
            
            // Compute ⟨O₁n|O₂n⟩ = ⟨n|O₁†O₂|n⟩
            std::complex<double> corr = vectorDot(d_O1_psi, d_O2_psi);
            correlation_values[n] = corr.real();
        }
        
        // Free basis vectors
        for (int i = 0; i < m; i++) {
            cudaFree(d_lanczos_basis[i]);
        }
        delete[] d_lanczos_basis;
        
        // Compute thermal averages
        std::vector<double> sample_exp(num_temp_bins);
        
        for (int t = 0; t < num_temp_bins; t++) {
            double T = temperatures[t];
            double beta = 1.0 / T;
            
            double Z = 0.0;
            for (int i = 0; i < m; i++) {
                Z += weights[i] * std::exp(-beta * ritz_values[i]);
            }
            
            double expectation = 0.0;
            for (int i = 0; i < m; i++) {
                double boltzmann = std::exp(-beta * ritz_values[i]);
                expectation += weights[i] * correlation_values[i] * boltzmann / Z;
            }
            
            sample_exp[t] = expectation;
        }
        
        sample_expectations.push_back(sample_exp);
    }
    
    // Free temporary memory
    cudaFree(d_O1_psi);
    cudaFree(d_O2_psi);
    
    // Average over samples
    int n_valid_samples = sample_expectations.size();
    std::cout << "\n--- Averaging over " << n_valid_samples << " samples ---\n";
    
    std::vector<double> expectation(num_temp_bins, 0.0);
    std::vector<double> error(num_temp_bins, 0.0);
    
    if (n_valid_samples == 0) {
        std::cerr << "Error: No valid samples\n";
        return std::make_tuple(temperatures, expectation, error);
    }
    
    // Compute means
    for (int s = 0; s < n_valid_samples; s++) {
        for (int t = 0; t < num_temp_bins; t++) {
            expectation[t] += sample_expectations[s][t];
        }
    }
    
    for (int t = 0; t < num_temp_bins; t++) {
        expectation[t] /= n_valid_samples;
    }
    
    // Compute standard errors
    if (n_valid_samples > 1) {
        for (int s = 0; s < n_valid_samples; s++) {
            for (int t = 0; t < num_temp_bins; t++) {
                double diff = sample_expectations[s][t] - expectation[t];
                error[t] += diff * diff;
            }
        }
        
        double norm = std::sqrt(static_cast<double>(n_valid_samples * (n_valid_samples - 1)));
        for (int t = 0; t < num_temp_bins; t++) {
            error[t] = std::sqrt(error[t]) / norm;
        }
    }
    
    std::cout << "\n==========================================\n";
    std::cout << "GPU Static Correlation Complete\n";
    std::cout << "==========================================\n";
    
    return std::make_tuple(temperatures, expectation, error);
}

// ============================================================================
// SINGLE-STATE DYNAMICAL CORRELATION
// ============================================================================

/**
 * @brief Compute dynamical correlation for a given state (GPU version)
 * 
 * GPU-accelerated version of compute_dynamical_correlation_state.
 * Computes S(ω) = Σₙ ⟨ψ|O₁†|n⟩⟨n|O₂|ψ⟩ δ(ω - Eₙ) for a specific state |ψ⟩.
 * 
 * This is the single-state version (no random sampling). Use this when you
 * have a specific quantum state (e.g., ground state or excited state).
 */
std::tuple<std::vector<double>, std::vector<double>, std::vector<double>>
GPUFTLMSolver::computeDynamicalCorrelationState(
    const cuDoubleComplex* d_psi,
    GPUOperator* op_O1,
    GPUOperator* op_O2,
    double omega_min,
    double omega_max,
    int num_omega_bins,
    double broadening,
    double temperature,
    double energy_shift) {
    
    std::cout << "\n==========================================\n";
    std::cout << "GPU Dynamical Correlation (Given State): S(ω) = ⟨O₁†(ω)O₂⟩\n";
    std::cout << "==========================================\n";
    std::cout << "Hilbert space dimension: " << N_ << "\n";
    std::cout << "Krylov dimension: " << krylov_dim_ << "\n";
    std::cout << "Frequency range: [" << omega_min << ", " << omega_max << "]\n";
    std::cout << "Broadening: " << broadening << "\n";
    if (temperature > 1e-14) {
        std::cout << "Temperature: " << temperature << "\n";
    } else {
        std::cout << "Temperature: 0 (no thermal weighting)\n";
    }
    
    // Generate frequency grid
    std::vector<double> frequencies(num_omega_bins);
    double omega_step = (omega_max - omega_min) / std::max(1, num_omega_bins - 1);
    for (int i = 0; i < num_omega_bins; i++) {
        frequencies[i] = omega_min + i * omega_step;
    }
    
    // Verify state is normalized
    double state_norm = vectorNorm(d_psi);
    if (std::abs(state_norm - 1.0) > 1e-10) {
        std::cout << "  Warning: Input state norm = " << state_norm << " (expected 1.0)\n";
        std::cout << "  Normalizing state...\n";
    }
    
    // Copy and normalize state
    vectorCopy(d_psi, d_v_current_);
    if (std::abs(state_norm - 1.0) > 1e-10) {
        normalizeVector(d_v_current_);
    }
    
    // Allocate temporary device vector
    cuDoubleComplex* d_phi = nullptr;
    cuDoubleComplex* d_O1_psi = nullptr;
    CUDA_CHECK(cudaMalloc(&d_phi, N_ * sizeof(cuDoubleComplex)));
    CUDA_CHECK(cudaMalloc(&d_O1_psi, N_ * sizeof(cuDoubleComplex)));
    
    // Apply operator O2: |φ⟩ = O₂|ψ⟩
    if (op_O2 != nullptr) {
        op_O2->matVecGPU(d_v_current_, d_phi, N_);
    } else {
        vectorCopy(d_v_current_, d_phi);
    }
    
    // Get norm of |φ⟩
    double phi_norm = vectorNorm(d_phi);
    if (phi_norm < 1e-14) {
        std::cerr << "  Error: O₂|ψ⟩ has zero norm\n";
        cudaFree(d_phi);
        cudaFree(d_O1_psi);
        
        std::vector<double> zero_spec_real(num_omega_bins, 0.0);
        std::vector<double> zero_spec_imag(num_omega_bins, 0.0);
        return std::make_tuple(frequencies, zero_spec_real, zero_spec_imag);
    }
    
    std::cout << "  Norm of O₂|ψ⟩: " << phi_norm << "\n";
    
    // Normalize |φ⟩
    normalizeVector(d_phi);
    
    // Build Lanczos tridiagonal and store basis vectors
    std::vector<double> alpha, beta;
    cuDoubleComplex** d_lanczos_basis = nullptr;
    
    int iterations = buildLanczosTridiagonalWithBasis(
        d_phi, false, 10, alpha, beta, &d_lanczos_basis);
    
    int m = alpha.size();
    std::cout << "  Lanczos iterations: " << m << "\n";
    
    // Diagonalize tridiagonal (need eigenvectors for weight computation)
    std::vector<double> ritz_values(m);
    std::vector<double> dummy_weights(m);
    std::vector<double> evecs(m * m);
    
    // Copy alpha/beta for diagonalization
    std::vector<double> diag = alpha;
    std::vector<double> offdiag(m - 1);
    for (int i = 0; i < m - 1; i++) {
        offdiag[i] = beta[i + 1];
    }
    
    // Diagonalize using LAPACKE
    int info = LAPACKE_dstevd(LAPACK_COL_MAJOR, 'V', m,
                             diag.data(), offdiag.data(),
                             evecs.data(), m);
    
    if (info != 0) {
        std::cerr << "  Error: Tridiagonal diagonalization failed\n";
        
        // Free memory
        for (int i = 0; i < m; i++) {
            cudaFree(d_lanczos_basis[i]);
        }
        delete[] d_lanczos_basis;
        cudaFree(d_phi);
        cudaFree(d_O1_psi);
        
        std::vector<double> zero_spec_real(num_omega_bins, 0.0);
        std::vector<double> zero_spec_imag(num_omega_bins, 0.0);
        return std::make_tuple(frequencies, zero_spec_real, zero_spec_imag);
    }
    
    // Extract eigenvalues
    for (int i = 0; i < m; i++) {
        ritz_values[i] = diag[i];
    }
    
    // Apply energy shift
    double E_shift;
    if (std::abs(energy_shift) > 1e-14) {
        E_shift = energy_shift;
        std::cout << "  Using provided ground state energy shift: " << E_shift << "\n";
    } else {
        E_shift = *std::min_element(ritz_values.begin(), ritz_values.end());
        std::cout << "  Ground state energy (auto-detected from Krylov): " << E_shift << "\n";
    }
    
    for (int i = 0; i < m; i++) {
        ritz_values[i] -= E_shift;
    }
    std::cout << "  Shifted to excitation energies (E_gs = 0)\n";
    
    // Compute complex weights ⟨ψ|O₁†|n⟩⟨n|O₂|ψ⟩
    
    // Apply O1 to original state: |O₁ψ⟩
    if (op_O1 != nullptr) {
        op_O1->matVecGPU(d_v_current_, d_O1_psi, N_);
    } else {
        vectorCopy(d_v_current_, d_O1_psi);
    }
    
    // Transfer to host for inner product computations
    std::vector<cuDoubleComplex> h_O1_psi(N_);
    CUDA_CHECK(cudaMemcpy(h_O1_psi.data(), d_O1_psi,
                         N_ * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost));
    
    std::vector<std::complex<double>> complex_weights(m);
    
    for (int n = 0; n < m; n++) {
        // Compute ⟨ψ|O₁†|n⟩ = ⟨O₁ψ|n⟩ = Σⱼ evecs[n,j] ⟨O₁ψ|vⱼ⟩
        std::complex<double> overlap_O1(0.0, 0.0);
        
        for (int j = 0; j < m; j++) {
            // Transfer basis vector to host
            std::vector<cuDoubleComplex> h_basis_j(N_);
            CUDA_CHECK(cudaMemcpy(h_basis_j.data(), d_lanczos_basis[j],
                                 N_ * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost));
            
            // Compute ⟨O₁ψ|vⱼ⟩
            cuDoubleComplex bracket;
            cblas_zdotc_sub(N_, h_O1_psi.data(), 1, h_basis_j.data(), 1, &bracket);
            
            std::complex<double> bracket_cpp(cuCreal(bracket), cuCimag(bracket));
            overlap_O1 += evecs[n * m + j] * bracket_cpp;
        }
        
        // Compute ⟨n|O₂|ψ⟩ = evecs[n,0] × ||O₂|ψ|| (since |v₀⟩ = O₂|ψ⟩/||O₂|ψ||)
        std::complex<double> overlap_O2(evecs[n * m + 0] * phi_norm, 0.0);
        
        // Weight is ⟨ψ|O₁†|n⟩⟨n|O₂|ψ⟩ = conj(⟨O₁ψ|n⟩) × ⟨n|O₂|ψ⟩
        complex_weights[n] = std::conj(overlap_O1) * overlap_O2;
    }
    
    // Free basis vectors
    for (int i = 0; i < m; i++) {
        cudaFree(d_lanczos_basis[i]);
    }
    delete[] d_lanczos_basis;
    
    // Free temporary memory
    cudaFree(d_phi);
    cudaFree(d_O1_psi);
    
    // Compute spectral function (both real and imaginary parts)
    std::vector<double> spectral_func_real, spectral_func_imag;
    computeSpectralFunctionComplex(ritz_values, complex_weights, frequencies,
                                   broadening, temperature,
                                   spectral_func_real, spectral_func_imag);
    
    std::cout << "\n==========================================\n";
    std::cout << "GPU Dynamical Correlation (Given State) Complete\n";
    std::cout << "==========================================\n";
    
    return std::make_tuple(frequencies, spectral_func_real, spectral_func_imag);
}

#endif // WITH_CUDA
