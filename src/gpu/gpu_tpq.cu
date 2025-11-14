#include "gpu_tpq.cuh"
#include "gpu_operator.cuh"
#include "../core/system_utils.h"
#include <iostream>
#include <fstream>
#include <iomanip>
#include <chrono>
#include <cmath>
#include <sys/stat.h>

// Kernel to generate random complex numbers for initial state
__global__ void initRandomState(cuDoubleComplex* state, int N, unsigned long long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        curandState rand_state;
        curand_init(seed, idx, 0, &rand_state);
        
        double real = curand_normal_double(&rand_state);
        double imag = curand_normal_double(&rand_state);
        state[idx] = make_cuDoubleComplex(real, imag);
    }
}

// Kernel to compute |state|^2 element-wise for reduction
__global__ void computeNormSquared(const cuDoubleComplex* state, double* result, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        cuDoubleComplex val = state[idx];
        result[idx] = cuCreal(val) * cuCreal(val) + cuCimag(val) * cuCimag(val);
    }
}

// Kernel to scale vector by a real scalar
__global__ void scaleVector(cuDoubleComplex* state, double scale, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        cuDoubleComplex val = state[idx];
        state[idx] = make_cuDoubleComplex(cuCreal(val) * scale, cuCimag(val) * scale);
    }
}

// Constructor
GPUTPQSolver::GPUTPQSolver(GPUOperator* gpu_op, int N)
    : gpu_op_(gpu_op), N_(N), d_state_(nullptr), d_temp_(nullptr), 
      d_h_state_(nullptr), d_real_scratch_(nullptr) {
    
    // Create cuBLAS handle
    cublasStatus_t stat = cublasCreate(&cublas_handle_);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        std::cerr << "cuBLAS initialization failed!" << std::endl;
        throw std::runtime_error("cuBLAS init failed");
    }
    
    // Create cuRAND generator
    curandStatus_t rand_stat = curandCreateGenerator(&curand_gen_, CURAND_RNG_PSEUDO_DEFAULT);
    if (rand_stat != CURAND_STATUS_SUCCESS) {
        std::cerr << "cuRAND initialization failed!" << std::endl;
        throw std::runtime_error("cuRAND init failed");
    }
    
    allocateMemory();
    
    // Initialize stats
    stats_.total_time = 0.0;
    stats_.matvec_time = 0.0;
    stats_.normalize_time = 0.0;
    stats_.iterations = 0;
    stats_.throughput = 0.0;
}

// Destructor
GPUTPQSolver::~GPUTPQSolver() {
    freeMemory();
    cublasDestroy(cublas_handle_);
    curandDestroyGenerator(curand_gen_);
}

void GPUTPQSolver::allocateMemory() {
    cudaMalloc(&d_state_, N_ * sizeof(cuDoubleComplex));
    cudaMalloc(&d_temp_, N_ * sizeof(cuDoubleComplex));
    cudaMalloc(&d_h_state_, N_ * sizeof(cuDoubleComplex));
    cudaMalloc(&d_real_scratch_, N_ * sizeof(double));
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "GPU memory allocation failed: " << cudaGetErrorString(err) << std::endl;
        throw std::runtime_error("GPU malloc failed");
    }
}

void GPUTPQSolver::freeMemory() {
    if (d_state_) cudaFree(d_state_);
    if (d_temp_) cudaFree(d_temp_);
    if (d_h_state_) cudaFree(d_h_state_);
    if (d_real_scratch_) cudaFree(d_real_scratch_);
}

void GPUTPQSolver::generateRandomState(unsigned int seed) {
    int blockSize = 256;
    int numBlocks = (N_ + blockSize - 1) / blockSize;
    
    initRandomState<<<numBlocks, blockSize>>>(d_state_, N_, seed);
    cudaDeviceSynchronize();
    
    // Normalize
    normalizeState();
}

void GPUTPQSolver::normalizeState() {
    auto start = std::chrono::high_resolution_clock::now();
    
    double norm = computeNorm();
    if (norm < 1e-14) {
        std::cerr << "Warning: TPQ state has near-zero norm" << std::endl;
        return;
    }
    
    double scale = 1.0 / norm;
    int blockSize = 256;
    int numBlocks = (N_ + blockSize - 1) / blockSize;
    scaleVector<<<numBlocks, blockSize>>>(d_state_, scale, N_);
    cudaDeviceSynchronize();
    
    auto end = std::chrono::high_resolution_clock::now();
    stats_.normalize_time += std::chrono::duration<double>(end - start).count();
}

double GPUTPQSolver::computeNorm() {
    // Compute ||state||^2
    int blockSize = 256;
    int numBlocks = (N_ + blockSize - 1) / blockSize;
    computeNormSquared<<<numBlocks, blockSize>>>(d_state_, d_real_scratch_, N_);
    
    // Sum reduction using cuBLAS
    double norm_squared = 0.0;
    double* h_scratch = new double[N_];
    cudaMemcpy(h_scratch, d_real_scratch_, N_ * sizeof(double), cudaMemcpyDeviceToHost);
    for (int i = 0; i < N_; ++i) {
        norm_squared += h_scratch[i];
    }
    delete[] h_scratch;
    
    return std::sqrt(norm_squared);
}

std::pair<double, double> GPUTPQSolver::computeEnergyAndVariance() {
    // Compute H|state> -> d_h_state_
    gpu_op_->matVecGPU(d_state_, d_h_state_, N_);
    
    // Energy = <state|H|state>
    cuDoubleComplex energy_complex;
    cublasZdotc(cublas_handle_, N_, 
                reinterpret_cast<cuDoubleComplex*>(d_state_), 1,
                reinterpret_cast<cuDoubleComplex*>(d_h_state_), 1,
                &energy_complex);
    double energy = cuCreal(energy_complex);
    
    // Compute H^2|state> -> d_temp_
    gpu_op_->matVecGPU(d_h_state_, d_temp_, N_);
    
    // <H^2> = <state|H^2|state>
    cuDoubleComplex h2_complex;
    cublasZdotc(cublas_handle_, N_,
                reinterpret_cast<cuDoubleComplex*>(d_state_), 1,
                reinterpret_cast<cuDoubleComplex*>(d_temp_), 1,
                &h2_complex);
    double h2 = cuCreal(h2_complex);
    
    // Variance = <H^2> - <H>^2
    double variance = h2 - energy * energy;
    
    return {energy, variance};
}

void GPUTPQSolver::imaginaryTimeEvolve(double delta_beta, int taylor_order) {
    // Implement e^{-delta_beta * H} |state> using Taylor expansion
    // |new_state> = sum_{k=0}^{n_max} (-delta_beta)^k / k! * H^k |state>
    
    // Copy state to temp
    cudaMemcpy(d_temp_, d_state_, N_ * sizeof(cuDoubleComplex), cudaMemcpyDeviceToDevice);
    
    // Zero out state (will accumulate series)
    cudaMemset(d_state_, 0, N_ * sizeof(cuDoubleComplex));
    
    // Add zeroth term: |state>
    cuDoubleComplex alpha = make_cuDoubleComplex(1.0, 0.0);
    cuDoubleComplex beta_blas = make_cuDoubleComplex(1.0, 0.0);
    cublasZaxpy(cublas_handle_, N_, &alpha, d_temp_, 1, d_state_, 1);
    
    // Taylor series
    double factorial = 1.0;
    double power = 1.0;
    
    for (int k = 1; k <= taylor_order; ++k) {
        // H^k term: d_h_state_ = H * d_temp_
        gpu_op_->matVecGPU(d_temp_, d_h_state_, N_);
        
        // Update coefficients
        factorial *= k;
        power *= -delta_beta;
        double coeff = power / factorial;
        
        // Add contribution: |state> += coeff * H^k |state>
        cuDoubleComplex alpha_k = make_cuDoubleComplex(coeff, 0.0);
        cublasZaxpy(cublas_handle_, N_, &alpha_k, d_h_state_, 1, d_state_, 1);
        
        // Prepare for next iteration: d_temp_ = H^k |state>
        cudaMemcpy(d_temp_, d_h_state_, N_ * sizeof(cuDoubleComplex), cudaMemcpyDeviceToDevice);
        
        // Check for convergence
        if (std::abs(coeff) < 1e-14) break;
    }
    
    // Normalize
    normalizeState();
}

void GPUTPQSolver::writeTPQData(const std::string& filename, double inv_temp, 
                                 double energy, double variance, double norm, int step) {
    std::ofstream file;
    if (step == 0) {
        file.open(filename, std::ios::out);
        file << "# Step    InvTemp    Energy    Variance    Norm    SpecificHeat" << std::endl;
    } else {
        file.open(filename, std::ios::app);
    }
    
    // Estimate specific heat from variance
    double cv = variance * inv_temp * inv_temp;
    
    file << std::setw(8) << step << " "
         << std::setw(15) << std::scientific << std::setprecision(8) << inv_temp << " "
         << std::setw(15) << energy << " "
         << std::setw(15) << variance << " "
         << std::setw(15) << norm << " "
         << std::setw(15) << cv << std::endl;
    
    file.close();
}

bool GPUTPQSolver::saveTPQState(const std::string& filename) {
    std::vector<std::complex<double>> h_state(N_);
    cudaMemcpy(h_state.data(), d_state_, N_ * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);
    
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) return false;
    
    size_t size = N_;
    file.write(reinterpret_cast<const char*>(&size), sizeof(size_t));
    file.write(reinterpret_cast<const char*>(h_state.data()), N_ * sizeof(std::complex<double>));
    file.close();
    
    return true;
}

void GPUTPQSolver::runMicrocanonicalTPQ(
    int max_iter,
    int num_samples,
    int temp_interval,
    std::vector<double>& eigenvalues,
    const std::string& dir,
    double large_value
) {
    auto total_start = std::chrono::high_resolution_clock::now();
    
    std::cout << "\n=== GPU Microcanonical TPQ ===" << std::endl;
    std::cout << "Hilbert space dimension: " << N_ << std::endl;
    std::cout << "Number of samples: " << num_samples << std::endl;
    std::cout << "Max iterations: " << max_iter << std::endl;
    std::cout << "Large value (energy shift): " << large_value << std::endl;
    std::cout << "Algorithm: Power method on (L-H) to find ground state" << std::endl;
    
    // Create output directory
    if (!dir.empty()) {
        std::string cmd = "mkdir -p " + dir;
        safe_system_call(cmd);
    }
    
    eigenvalues.clear();
    
    // Calculate dimension entropy S = log2(N)
    double D_S = std::log2(static_cast<double>(N_));
    
    // Define measurement temperatures (similar to CPU version)
    const int num_temp_points = 20;
    std::vector<double> measure_inv_temp(num_temp_points);
    double log_min = std::log10(1);   // Start from β = 1
    double log_max = std::log10(1000); // End at β = 1000
    for (int i = 0; i < num_temp_points; ++i) {
        measure_inv_temp[i] = std::pow(10.0, log_min + (log_max - log_min) * i / (num_temp_points - 1));
    }
    
    for (int sample = 0; sample < num_samples; ++sample) {
        std::cout << "\nSample " << sample + 1 << "/" << num_samples << std::endl;
        
        // Track which measurement temperatures have been saved
        std::vector<bool> temp_measured(num_temp_points, false);
        
        // Generate random initial state |v1⟩
        generateRandomState(12345 + sample * 67890);
        
        // Apply H|v1⟩ -> d_temp_
        gpu_op_->matVecGPU(d_state_, d_temp_, N_);
        
        // Compute |v0⟩ = (L*D_S - H)|v1⟩ = L*D_S*|v1⟩ - H|v1⟩
        // Save H|v1⟩ in d_h_state_ first
        cudaMemcpy(d_h_state_, d_temp_, N_ * sizeof(cuDoubleComplex), cudaMemcpyDeviceToDevice);
        
        // d_state_ *= L*D_S
        cuDoubleComplex alpha = make_cuDoubleComplex(large_value * D_S, 0.0);
        cublasZscal(cublas_handle_, N_, &alpha, d_state_, 1);
        
        // d_state_ -= H|v1⟩
        cuDoubleComplex minus_one = make_cuDoubleComplex(-1.0, 0.0);
        cublasZaxpy(cublas_handle_, N_, &minus_one, d_h_state_, 1, d_state_, 1);
        
        // Normalize |v0⟩
        double first_norm = computeNorm();
        cuDoubleComplex scale = make_cuDoubleComplex(1.0 / first_norm, 0.0);
        cublasZscal(cublas_handle_, N_, &scale, d_state_, 1);
        
        // Output file for this sample
        std::string sample_file = dir + "/tpq_sample_" + std::to_string(sample) + ".dat";
        
        // Initial measurements (step 1)
        std::pair<double, double> energy_var_pair = computeEnergyAndVariance();
        double energy = energy_var_pair.first;
        double variance = energy_var_pair.second;
        
        // Compute inverse temperature: β = 2*step / (L*D_S - E)
        double inv_temp = 2.0 / (large_value * D_S - energy);
        
        writeTPQData(sample_file, inv_temp, energy, variance, first_norm, 1);
        
        std::cout << "Step 1: E = " << energy << ", β = " << inv_temp << std::endl;
        
        // Main TPQ loop - applies (L-H) repeatedly
        for (int step = 2; step <= max_iter; ++step) {
            // Apply H|v0⟩ -> d_temp_
            auto matvec_start = std::chrono::high_resolution_clock::now();
            gpu_op_->matVecGPU(d_state_, d_temp_, N_);
            auto matvec_end = std::chrono::high_resolution_clock::now();
            stats_.matvec_time += std::chrono::duration<double>(matvec_end - matvec_start).count();
            
            // Compute |v0_new⟩ = (L*D_S - H)|v0⟩ = L*D_S*|v0⟩ - H|v0⟩
            // Save H|v0⟩ in d_h_state_ first
            cudaMemcpy(d_h_state_, d_temp_, N_ * sizeof(cuDoubleComplex), cudaMemcpyDeviceToDevice);
            
            // d_state_ *= L*D_S
            alpha = make_cuDoubleComplex(large_value * D_S, 0.0);
            cublasZscal(cublas_handle_, N_, &alpha, d_state_, 1);
            
            // d_state_ -= H|v0⟩
            cuDoubleComplex minus_one = make_cuDoubleComplex(-1.0, 0.0);
            cublasZaxpy(cublas_handle_, N_, &minus_one, d_h_state_, 1, d_state_, 1);
            
            // Normalize
            double current_norm = computeNorm();
            scale = make_cuDoubleComplex(1.0 / current_norm, 0.0);
            cublasZscal(cublas_handle_, N_, &scale, d_state_, 1);
            
            // Measurements every temp_interval steps
            if (step % temp_interval == 0 || step == max_iter) {
                std::pair<double, double> E_var_pair = computeEnergyAndVariance();
                double E = E_var_pair.first;
                double var = E_var_pair.second;
                
                // Update inverse temperature
                inv_temp = (2.0 * step) / (large_value * D_S - E);
                
                writeTPQData(sample_file, inv_temp, E, var, current_norm, step);
                
                if (step % (temp_interval * 10) == 0 || step == max_iter) {
                    std::cout << "Step " << step << ": E = " << E 
                              << ", var = " << var 
                              << ", β = " << inv_temp << std::endl;
                }
                
                energy = E;
            }
            
            // Save TPQ state at specified measurement temperatures
            for (int i = 0; i < num_temp_points; ++i) {
                if (!temp_measured[i] && std::abs(inv_temp - measure_inv_temp[i]) < 4e-3) {
                    std::cout << "  Saving TPQ state at β = " << inv_temp << std::endl;
                    std::string state_file = dir + "/tpq_state_" + std::to_string(sample) + 
                                           "_beta=" + std::to_string(inv_temp) + ".dat";
                    saveTPQState(state_file);
                    temp_measured[i] = true; // Mark this temperature as saved
                }
            }
            
            stats_.iterations++;
        }
        
        
        // Save final energy
        std::pair<double, double> final_pair = computeEnergyAndVariance();
        double final_energy = final_pair.first;
        double final_var = final_pair.second;
        eigenvalues.push_back(final_energy);
        
        std::cout << "Final energy: " << final_energy << " (variance: " << final_var << ")" << std::endl;
    }
    
    auto total_end = std::chrono::high_resolution_clock::now();
    stats_.total_time = std::chrono::duration<double>(total_end - total_start).count();
    stats_.throughput = (stats_.iterations * 2.0 * N_) / stats_.matvec_time / 1e9; // GFLOPS
    
    std::cout << "\n=== GPU TPQ Statistics ===" << std::endl;
    std::cout << "Total time: " << stats_.total_time << " s" << std::endl;
    std::cout << "MatVec time: " << stats_.matvec_time << " s" << std::endl;
    std::cout << "Normalize time: " << stats_.normalize_time << " s" << std::endl;
    std::cout << "Throughput: " << stats_.throughput << " GFLOPS" << std::endl;
}

void GPUTPQSolver::runCanonicalTPQ(
    double beta_max,
    int num_samples,
    int temp_interval,
    std::vector<double>& energies,
    const std::string& dir,
    double delta_beta,
    int taylor_order
) {
    auto total_start = std::chrono::high_resolution_clock::now();
    
    std::cout << "\n=== GPU Canonical TPQ ===" << std::endl;
    std::cout << "Hilbert space dimension: " << N_ << std::endl;
    std::cout << "Number of samples: " << num_samples << std::endl;
    std::cout << "Beta max: " << beta_max << std::endl;
    std::cout << "Delta beta: " << delta_beta << std::endl;
    
    // Create output directory
    if (!dir.empty()) {
        std::string cmd = "mkdir -p " + dir;
        safe_system_call(cmd);
    }
    
    energies.clear();
    
    int num_steps = static_cast<int>(beta_max / delta_beta);
    
    // Define measurement temperatures (similar to microcanonical version)
    const int num_temp_points = 20;
    std::vector<double> measure_inv_temp(num_temp_points);
    double log_min = std::log10(1);   // Start from β = 1
    double log_max = std::log10(1000); // End at β = 1000
    for (int i = 0; i < num_temp_points; ++i) {
        measure_inv_temp[i] = std::pow(10.0, log_min + (log_max - log_min) * i / (num_temp_points - 1));
    }
    
    for (int sample = 0; sample < num_samples; ++sample) {
        std::cout << "\nSample " << sample + 1 << "/" << num_samples << std::endl;
        
        // Track which measurement temperatures have been saved
        std::vector<bool> temp_measured(num_temp_points, false);
        
        // Generate random initial state
        generateRandomState(98765 + sample * 43210);
        
        // Output file
        std::string sample_file = dir + "/ctpq_sample_" + std::to_string(sample) + ".dat";
        
        // Initial measurements at beta=0
        std::pair<double, double> energy_var_pair = computeEnergyAndVariance();
        double energy = energy_var_pair.first;
        double variance = energy_var_pair.second;
        double norm = computeNorm();
        writeTPQData(sample_file, 0.0, energy, variance, norm, 0);
        
        // Imaginary time evolution
        for (int step = 1; step <= num_steps; ++step) {
            double beta = step * delta_beta;
            
            // Evolve: |state> -> e^{-delta_beta H} |state>
            auto evolve_start = std::chrono::high_resolution_clock::now();
            imaginaryTimeEvolve(delta_beta, taylor_order);
            auto evolve_end = std::chrono::high_resolution_clock::now();
            stats_.matvec_time += std::chrono::duration<double>(evolve_end - evolve_start).count();
            
            // Measurements
            if (step % temp_interval == 0 || step == num_steps) {
                std::pair<double, double> E_var_pair = computeEnergyAndVariance();
                double E = E_var_pair.first;
                double var = E_var_pair.second;
                norm = computeNorm();
                
                writeTPQData(sample_file, beta, E, var, norm, step);
                
                // Save state periodically
                if (step % (temp_interval * 5) == 0) {
                    std::string state_file = dir + "/ctpq_state_" + std::to_string(sample) + 
                                           "_beta=" + std::to_string(beta) + ".dat";
                    saveTPQState(state_file);
                }
            }
            
            // Save TPQ state at specified measurement temperatures
            for (int i = 0; i < num_temp_points; ++i) {
                if (!temp_measured[i] && std::abs(beta - measure_inv_temp[i]) < 4e-3) {
                    std::cout << "  Saving TPQ state at β = " << beta << std::endl;
                    std::string state_file = dir + "/tpq_state_" + std::to_string(sample) + 
                                           "_beta=" + std::to_string(beta) + ".dat";
                    saveTPQState(state_file);
                    temp_measured[i] = true; // Mark this temperature as saved
                }
            }
            
            stats_.iterations++;
        }
        
        // Final energy
        std::pair<double, double> final_pair = computeEnergyAndVariance();
        double final_energy = final_pair.first;
        // double final_var = final_pair.second;  // Currently unused
        energies.push_back(final_energy);
        
        std::cout << "Final energy at beta=" << beta_max << ": " << final_energy << std::endl;
    }
    
    auto total_end = std::chrono::high_resolution_clock::now();
    stats_.total_time = std::chrono::duration<double>(total_end - total_start).count();
    
    std::cout << "\n=== GPU Canonical TPQ Statistics ===" << std::endl;
    std::cout << "Total time: " << stats_.total_time << " s" << std::endl;
    std::cout << "Evolution time: " << stats_.matvec_time << " s" << std::endl;
}
