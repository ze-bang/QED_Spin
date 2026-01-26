#include <ed/gpu/gpu_tpq.cuh>
#include <ed/gpu/gpu_operator.cuh>
#include <ed/core/system_utils.h>
#include <ed/core/hdf5_io.h>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <chrono>
#include <cmath>
#include <sys/stat.h>
#include <dirent.h>
#include <regex>
#include <sstream>
#include <algorithm>

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
      d_h_state_(nullptr), d_real_scratch_(nullptr),
      compute_stream_(nullptr), transfer_stream_(nullptr), streams_initialized_(false) {
    
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
    
    // Initialize CUDA streams for pipelining
    cudaError_t stream_err1 = cudaStreamCreate(&compute_stream_);
    cudaError_t stream_err2 = cudaStreamCreate(&transfer_stream_);
    if (stream_err1 == cudaSuccess && stream_err2 == cudaSuccess) {
        streams_initialized_ = true;
        // Set cuBLAS to use compute stream
        cublasSetStream(cublas_handle_, compute_stream_);
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
    
    // Destroy CUDA streams
    if (streams_initialized_) {
        cudaStreamDestroy(compute_stream_);
        cudaStreamDestroy(transfer_stream_);
    }
    
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
    // OPTIMIZED: Use cuBLAS Zdotc for efficient GPU-side norm computation
    // This avoids the expensive host-to-device copy and CPU reduction
    // that was previously causing a performance bottleneck
    cuDoubleComplex result;
    cublasZdotc(cublas_handle_, N_, d_state_, 1, d_state_, 1, &result);
    return std::sqrt(cuCreal(result));
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
    
    // Copy state to temp (will hold H^k |state> during iteration)
    cudaMemcpy(d_temp_, d_state_, N_ * sizeof(cuDoubleComplex), cudaMemcpyDeviceToDevice);
    
    // Zero out state (will accumulate series)
    cudaMemset(d_state_, 0, N_ * sizeof(cuDoubleComplex));
    
    // Add zeroth term: |state>
    cuDoubleComplex alpha = make_cuDoubleComplex(1.0, 0.0);
    cublasZaxpy(cublas_handle_, N_, &alpha, d_temp_, 1, d_state_, 1);
    
    // Taylor series with optimized ping-pong buffer strategy
    // Avoids redundant cudaMemcpy by swapping pointers
    double factorial = 1.0;
    double power = 1.0;
    
    // Use d_temp_ and d_h_state_ as ping-pong buffers
    cuDoubleComplex* d_current = d_temp_;    // H^{k-1} |state>
    cuDoubleComplex* d_next = d_h_state_;    // Will hold H^k |state>
    
    for (int k = 1; k <= taylor_order; ++k) {
        // H^k term: d_next = H * d_current
        gpu_op_->matVecGPU(d_current, d_next, N_);
        
        // Update coefficients
        factorial *= k;
        power *= -delta_beta;
        double coeff = power / factorial;
        
        // Add contribution: |state> += coeff * H^k |state>
        cuDoubleComplex alpha_k = make_cuDoubleComplex(coeff, 0.0);
        cublasZaxpy(cublas_handle_, N_, &alpha_k, d_next, 1, d_state_, 1);
        
        // OPTIMIZED: Swap pointers instead of copying memory
        // This avoids the expensive cudaMemcpy(d_temp_, d_h_state_, ...)
        std::swap(d_current, d_next);
        
        // Check for convergence (early termination)
        if (std::abs(coeff) < 1e-14) break;
    }
    
    // Normalize
    normalizeState();
}

void GPUTPQSolver::writeTPQDataHDF5(const std::string& h5_file, size_t sample,
                                    double inv_temp, double energy, double variance, 
                                    double doublon, uint64_t step) {
    // Write to HDF5 only (text file output removed - data already in HDF5)
    if (!h5_file.empty() && HDF5IO::fileExists(h5_file)) {
        try {
            HDF5IO::TPQThermodynamicPoint point;
            point.beta = inv_temp;
            point.energy = energy;
            point.variance = variance;
            point.doublon = doublon;
            point.step = step;
            HDF5IO::appendTPQThermodynamics(h5_file, sample, point);
        } catch (const std::exception& e) {
            std::cerr << "Warning: Failed to write TPQ thermodynamics to HDF5: " << e.what() << std::endl;
        }
    }
}

bool GPUTPQSolver::saveTPQStateHDF5(const std::string& dir, size_t sample, double beta, GPUFixedSzOperator* fixed_sz_op) {
    try {
        // Copy state from GPU to host
        std::vector<std::complex<double>> h_state(N_);
        cudaMemcpy(h_state.data(), d_state_, N_ * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);
        
        // MPI-safe: determine the correct HDF5 file path
        std::string hdf5_path;
        #ifdef WITH_MPI
        int mpi_rank = 0;
        MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
        hdf5_path = HDF5IO::getPerRankFilePath(dir, mpi_rank, "ed_results.h5");
        // Ensure file exists
        if (!HDF5IO::fileExists(hdf5_path)) {
            HDF5IO::createPerRankFile(dir, mpi_rank, "ed_results.h5");
        }
        #else
        hdf5_path = HDF5IO::createOrOpenFile(dir, "ed_results.h5");
        #endif
        
        // Ensure sample group exists
        HDF5IO::ensureTPQSampleGroup(hdf5_path, sample);
        
        // Transform to full basis if using fixed-Sz
        if (fixed_sz_op != nullptr) {
            std::vector<std::complex<double>> full_state = fixed_sz_op->embedToFull(h_state);
            HDF5IO::saveTPQState(hdf5_path, sample, beta, full_state);
            std::cout << "  [GPU Fixed-Sz] Transformed state from dim " << N_ 
                      << " to full space dim " << full_state.size() << " before saving to HDF5" << std::endl;
        } else {
            HDF5IO::saveTPQState(hdf5_path, sample, beta, h_state);
        }
        
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Error saving GPU TPQ state to HDF5: " << e.what() << std::endl;
        return false;
    }
}

bool GPUTPQSolver::loadTPQStateFromHDF5(const std::string& h5_file, 
                                        const std::string& dataset_name,
                                        GPUFixedSzOperator* fixed_sz_op) {
    try {
        std::vector<std::complex<double>> state;
        if (!HDF5IO::loadTPQStateByName(h5_file, dataset_name, state)) {
            std::cerr << "Error: Could not load TPQ state from HDF5: " << dataset_name << std::endl;
            return false;
        }
        
        // If fixed_sz_op is provided and state is in full basis, project to reduced
        if (fixed_sz_op != nullptr) {
            size_t full_dim = fixed_sz_op->getFullDim();
            
            if (state.size() == full_dim) {
                // State is in full basis - project to reduced
                std::vector<std::complex<double>> reduced_state = fixed_sz_op->projectToReduced(state);
                
                if (reduced_state.size() != static_cast<size_t>(N_)) {
                    std::cerr << "Error: Projected state dimension mismatch. Expected " << N_ 
                              << ", got " << reduced_state.size() << std::endl;
                    return false;
                }
                
                cudaMemcpy(d_state_, reduced_state.data(), N_ * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);
                std::cout << "  [GPU Fixed-Sz] Projected from full basis (dim=" << full_dim 
                          << ") to reduced basis (dim=" << N_ << ")" << std::endl;
            } else if (state.size() == static_cast<size_t>(N_)) {
                // Already in reduced basis
                cudaMemcpy(d_state_, state.data(), N_ * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);
            } else {
                std::cerr << "Error: State dimension " << state.size() << " doesn't match expected dimensions" << std::endl;
                return false;
            }
        } else {
            if (state.size() != static_cast<size_t>(N_)) {
                std::cerr << "Error: State dimension mismatch. Expected " << N_ 
                          << ", got " << state.size() << std::endl;
                return false;
            }
            cudaMemcpy(d_state_, state.data(), N_ * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);
        }
        
        std::cout << "Loaded TPQ state from HDF5: " << dataset_name << std::endl;
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "Error loading TPQ state from HDF5: " << e.what() << std::endl;
        return false;
    }
}

// Find the lowest energy (highest beta) TPQ state in HDF5
HDF5IO::TPQStateInfo GPUTPQSolver::findLowestEnergyTPQStateHDF5(const std::string& dir, int sample_filter) {
    std::string h5_file = dir + "/ed_results.h5";
    
    if (!HDF5IO::fileExists(h5_file)) {
        std::cerr << "Error: HDF5 file not found: " << h5_file << std::endl;
        return HDF5IO::TPQStateInfo{0, -1.0, ""};
    }
    
    // List all TPQ states in the HDF5 file
    auto states = HDF5IO::listTPQStates(h5_file, sample_filter);
    
    if (states.empty()) {
        std::cerr << "Error: No TPQ states found in HDF5: " << h5_file << std::endl;
        return HDF5IO::TPQStateInfo{0, -1.0, ""};
    }
    
    // Find the state with highest beta (lowest energy)
    HDF5IO::TPQStateInfo best_state = states[0];
    for (const auto& state : states) {
        if (state.beta > best_state.beta) {
            best_state = state;
        }
    }
    
    std::cout << "Found lowest energy TPQ state in HDF5:" << std::endl;
    std::cout << "  Sample: " << best_state.sample_index << std::endl;
    std::cout << "  Beta: " << best_state.beta << std::endl;
    
    return best_state;
}

// Legacy function for backward compatibility - now uses HDF5
std::string GPUTPQSolver::findLowestEnergyTPQState(const std::string& dir, int sample, 
                                                   double& beta_out, int& step_out) {
    auto state_info = findLowestEnergyTPQStateHDF5(dir, sample);
    
    if (state_info.beta < 0) {
        return "";  // Not found
    }
    
    beta_out = state_info.beta;
    
    // Look up step from thermodynamics data
    std::string h5_file = dir + "/ed_results.h5";
    step_out = -1;
    try {
        auto thermo = HDF5IO::loadTPQThermodynamics(h5_file, state_info.sample_index);
        double closest_diff = 1e10;
        for (const auto& point : thermo) {
            double diff = std::abs(point.beta - state_info.beta);
            if (diff < closest_diff) {
                closest_diff = diff;
                step_out = point.step;
            }
        }
    } catch (...) {
        // Step lookup failed, but we can still continue
    }
    
    std::cout << "  Step: " << step_out << std::endl;
    
    // Return HDF5 file path (not a .dat file) as indicator that HDF5 should be used
    return h5_file;
}

void GPUTPQSolver::runMicrocanonicalTPQ(
    int max_iter,
    int num_samples,
    int temp_interval,
    std::vector<double>& eigenvalues,
    const std::string& dir,
    double large_value,
    GPUFixedSzOperator* fixed_sz_op,
    bool continue_quenching,
    int continue_sample,
    double continue_beta,
    bool save_thermal_states,
    double target_beta
) {
    auto total_start = std::chrono::high_resolution_clock::now();
    
    std::cout << "\n=== GPU Microcanonical TPQ ===" << std::endl;
    std::cout << "Hilbert space dimension: " << N_ << std::endl;
    std::cout << "Number of samples: " << num_samples << std::endl;
    std::cout << "Max iterations: " << max_iter << std::endl;
    std::cout << "Large value (energy shift): " << large_value << std::endl;
    std::cout << "Target beta: " << target_beta << std::endl;
    std::cout << "Algorithm: Power method on (L-H) to find ground state" << std::endl;
    
    if (continue_quenching) {
        std::cout << "Continue-quenching mode enabled:" << std::endl;
        std::cout << "  Sample: " << (continue_sample == 0 ? "auto-detect" : std::to_string(continue_sample)) << std::endl;
        std::cout << "  Beta: " << (continue_beta == 0.0 ? "auto-detect" : std::to_string(continue_beta)) << std::endl;
    }
    
    // Create output directory
    if (!dir.empty()) {
        std::string cmd = "mkdir -p " + dir;
        safe_system_call(cmd);
    }
    
    eigenvalues.clear();
    
    // Initialize MPI-safe HDF5 file for TPQ data
    std::string h5_file;
    #ifdef WITH_MPI
    int mpi_rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    h5_file = HDF5IO::getPerRankFilePath(dir, mpi_rank, "ed_results.h5");
    try {
        if (!HDF5IO::fileExists(h5_file)) {
            HDF5IO::createPerRankFile(dir, mpi_rank, "ed_results.h5");
        }
    } catch (const std::exception& e) {
        std::cerr << "Warning: Could not initialize HDF5 TPQ storage: " << e.what() << std::endl;
        h5_file = "";  // Disable HDF5 writing
    }
    #else
    h5_file = dir + "/ed_results.h5";
    try {
        if (!HDF5IO::fileExists(h5_file)) {
            HDF5IO::createOrOpenFile(dir, "ed_results.h5");
        }
    } catch (const std::exception& e) {
        std::cerr << "Warning: Could not initialize HDF5 TPQ storage: " << e.what() << std::endl;
        h5_file = "";  // Disable HDF5 writing
    }
    #endif
    
    // Calculate dimension entropy S = log2(N)
    double D_S = std::log2(static_cast<double>(N_));
    
    // MPI sample distribution - each rank processes only its share of samples
    int start_sample = 0;
    int end_sample = num_samples;
    #ifdef WITH_MPI
    int mpi_size = 1;
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
    
    int samples_per_rank = num_samples / mpi_size;
    int remainder = num_samples % mpi_size;
    
    // Ranks with index < remainder get one extra sample
    start_sample = mpi_rank * samples_per_rank + std::min(mpi_rank, remainder);
    end_sample = start_sample + samples_per_rank + (mpi_rank < remainder ? 1 : 0);
    
    if (mpi_rank == 0) {
        std::cout << "\n==========================================\n";
        std::cout << "MPI-Parallel GPU TPQ Calculation\n";
        std::cout << "==========================================\n";
        std::cout << "Total MPI ranks: " << mpi_size << "\n";
        std::cout << "Total samples: " << num_samples << "\n";
        std::cout << "Samples per rank: " << samples_per_rank << " (+ " << remainder << " remainder)\n";
        std::cout << "==========================================\n";
    }
    
    std::cout << "Rank " << mpi_rank << " processing samples [" 
              << start_sample << ", " << end_sample << ")" << std::endl;
    
    MPI_Barrier(MPI_COMM_WORLD);
    #else
    int mpi_rank = 0;  // For non-MPI builds
    #endif
    
    // Define measurement temperatures (similar to CPU version)
    const int num_temp_points = 20;
    std::vector<double> measure_inv_temp(num_temp_points);
    double log_min = std::log10(1);   // Start from β = 1
    double log_max = std::log10(1000); // End at β = 1000
    for (int i = 0; i < num_temp_points; ++i) {
        measure_inv_temp[i] = std::pow(10.0, log_min + (log_max - log_min) * i / (num_temp_points - 1));
    }
    
    for (int sample = start_sample; sample < end_sample; ++sample) {
        std::cout << "\n[Rank " << mpi_rank << "] Sample " << sample << " of " << num_samples 
                  << " (local: " << (sample - start_sample + 1) << " of " << (end_sample - start_sample) << ")" << std::endl;
        
        // Track which measurement temperatures have been saved
        std::vector<bool> temp_measured(num_temp_points, false);
        
        // Variables for continuing from saved state
        int start_step = 1;
        double energy = 0.0;
        double variance = 0.0;
        double inv_temp = 0.0;
        bool loaded_from_file = false;
        
        // Check if we should continue from a saved state (only for first sample)
        if (continue_quenching && sample == 0) {
            std::cout << "Continue-quenching: Looking for saved states in HDF5..." << std::endl;
            
            // Find the lowest energy (highest beta) state in HDF5 (merged file)
            std::string merged_h5_file = dir + "/ed_results.h5";
            int sample_filter = (continue_sample == 0) ? -1 : continue_sample;
            auto state_info = findLowestEnergyTPQStateHDF5(dir, sample_filter);
            
            if (state_info.beta > 0 && !state_info.dataset_name.empty()) {
                // Found a valid state - load it from HDF5 using the exact dataset name
                // Note: Load from the merged file (ed_results.h5), not the per-rank file
                if (loadTPQStateFromHDF5(merged_h5_file, state_info.dataset_name, fixed_sz_op)) {
                    loaded_from_file = true;
                    
                    // Look up step from thermodynamics (also from merged file)
                    int found_step = -1;
                    try {
                        auto thermo = HDF5IO::loadTPQThermodynamics(merged_h5_file, state_info.sample_index);
                        double closest_diff = 1e10;
                        for (const auto& point : thermo) {
                            double diff = std::abs(point.beta - state_info.beta);
                            if (diff < closest_diff) {
                                closest_diff = diff;
                                found_step = point.step;
                            }
                        }
                    } catch (...) {
                        found_step = 1;  // Default if lookup fails
                    }
                    
                    start_step = found_step + 1;
                    inv_temp = state_info.beta;
                    
                    // Compute energy and variance from loaded state
                    std::pair<double, double> ev_pair = computeEnergyAndVariance();
                    energy = ev_pair.first;
                    variance = ev_pair.second;
                    
                    std::cout << "Resuming from HDF5:" << std::endl;
                    std::cout << "  Original sample: " << state_info.sample_index << std::endl;
                    std::cout << "  Continuing as sample: 0" << std::endl;
                    std::cout << "  Beta: " << state_info.beta << std::endl;
                    std::cout << "  Step: " << found_step << std::endl;
                    std::cout << "  Will run " << max_iter << " additional iterations" << std::endl;
                    std::cout << "  Target final step: " << (found_step + max_iter) << std::endl;
                    std::cout << "Continuing from step " << found_step 
                              << " (beta=" << state_info.beta << ", E=" << energy << ")" << std::endl;
                } else {
                    std::cout << "Warning: Could not load TPQ state from HDF5. Starting fresh." << std::endl;
                }
            } else {
                std::cout << "Warning: No TPQ states found in HDF5. Starting fresh." << std::endl;
            }
        }
        
        // If not continuing, initialize normally
        if (!loaded_from_file) {
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
            
            // Initial measurements (step 1) - for internal use only
            // Step 1 data is unphysical (not yet thermalized), so we don't write it
            std::pair<double, double> energy_var_pair = computeEnergyAndVariance();
            energy = energy_var_pair.first;
            variance = energy_var_pair.second;
            
            // Compute inverse temperature: β = 2*step / (L*D_S - E)
            inv_temp = 2.0 / (large_value * D_S - energy);
            
            // Skip writing step 1 - it contains unphysical initialization data
            // Physical data starts from step 2
            
            start_step = 2;
        }
        
        // Initialize HDF5 sample group
        if (!h5_file.empty()) {
            try {
                HDF5IO::ensureTPQSampleGroup(h5_file, sample);
            } catch (const std::exception& e) {
                std::cerr << "Warning: Could not initialize HDF5 sample group: " << e.what() << std::endl;
            }
        }
        
        // Determine final step: if continuing, run for additional max_iter iterations
        int final_step = loaded_from_file ? (start_step - 1 + max_iter) : max_iter;
        
        // Main TPQ loop - applies (L-H) repeatedly
        for (int step = start_step; step <= final_step; ++step) {
            // Apply H|v0⟩ -> d_temp_
            auto matvec_start = std::chrono::high_resolution_clock::now();
            gpu_op_->matVecGPU(d_state_, d_temp_, N_);
            auto matvec_end = std::chrono::high_resolution_clock::now();
            stats_.matvec_time += std::chrono::duration<double>(matvec_end - matvec_start).count();
            
            // Compute |v0_new⟩ = (L*D_S - H)|v0⟩ = L*D_S*|v0⟩ - H|v0⟩
            // Save H|v0⟩ in d_h_state_ first
            cudaMemcpy(d_h_state_, d_temp_, N_ * sizeof(cuDoubleComplex), cudaMemcpyDeviceToDevice);
            
            // d_state_ *= L*D_S
            cuDoubleComplex alpha = make_cuDoubleComplex(large_value * D_S, 0.0);
            cublasZscal(cublas_handle_, N_, &alpha, d_state_, 1);
            
            // d_state_ -= H|v0⟩
            cuDoubleComplex minus_one = make_cuDoubleComplex(-1.0, 0.0);
            cublasZaxpy(cublas_handle_, N_, &minus_one, d_h_state_, 1, d_state_, 1);
            
            // Normalize
            double current_norm = computeNorm();
            cuDoubleComplex scale = make_cuDoubleComplex(1.0 / current_norm, 0.0);
            cublasZscal(cublas_handle_, N_, &scale, d_state_, 1);
            
            // Check if we should measure observables at target temperatures
            // We need to check this at EVERY step to avoid missing temperature points
            bool should_measure_observables = false;
            int target_temp_idx = -1;
            
            // First, do a quick check using estimated temperature
            // Estimate current inverse temperature (using last known energy)
            double estimated_inv_temp = (2.0 * step) / (large_value * D_S - energy);
            
            // Check if we're potentially near any target temperature
            // Use a wider search window (5% instead of 1%) for the initial check
            for (int i = 0; i < num_temp_points; ++i) {
                if (!temp_measured[i]) {
                    double search_tolerance = 0.05 * measure_inv_temp[i];  // 5% search window
                    if (std::abs(estimated_inv_temp - measure_inv_temp[i]) < search_tolerance) {
                        // We're potentially close - need to compute actual energy to be sure
                        should_measure_observables = true;
                        target_temp_idx = i;
                        break;
                    }
                }
            }
            
            // Determine if we should do measurements this step
            bool do_regular_measurement = (step % temp_interval == 0 || step == max_iter);
            bool do_measurement = do_regular_measurement || should_measure_observables;
            
            // Measurements when needed
            if (do_measurement) {
                std::pair<double, double> E_var_pair = computeEnergyAndVariance();
                double E = E_var_pair.first;
                double var = E_var_pair.second;
                
                // Update inverse temperature with accurate energy
                inv_temp = (2.0 * step) / (large_value * D_S - E);
                
                // Update energy for next iteration's estimate
                energy = E;
                
                // Now check with accurate temperature if we're really at the target
                bool actually_at_target = false;
                if (should_measure_observables && target_temp_idx >= 0) {
                    double precise_tolerance = 0.01 * measure_inv_temp[target_temp_idx];  // 1% precise tolerance
                    if (std::abs(inv_temp - measure_inv_temp[target_temp_idx]) < precise_tolerance) {
                        actually_at_target = true;
                    }
                }
                
                // Write data (always write when we compute energy) - to HDF5
                writeTPQDataHDF5(h5_file, sample, inv_temp, E, var, 0.0, step);
                
                if (step % (temp_interval * 10) == 0 || step == final_step) {
                    std::cout << "Step " << step << ": E = " << E 
                              << ", var = " << var 
                              << ", β = " << inv_temp << std::endl;
                }
                
                // Save TPQ state at target temperatures (with accurate inv_temp) to HDF5
                // Only save if save_thermal_states flag is enabled
                if (actually_at_target && save_thermal_states) {
                    std::cout << "  *** Saving TPQ state at β = " << inv_temp 
                              << " (target: " << measure_inv_temp[target_temp_idx] << ") ***" << std::endl;
                    saveTPQStateHDF5(dir, sample, inv_temp, fixed_sz_op);
                    
                    temp_measured[target_temp_idx] = true;
                }
                
                // Check if we've reached the target beta - early termination
                // Note: This is different from continue_quenching - target_beta is a stopping condition
                // regardless of whether we're continuing from a previous run or starting fresh
                if (inv_temp >= target_beta) {
                    std::cout << "  *** Reached target beta " << target_beta 
                              << " (current β = " << inv_temp << ") - stopping iteration ***" << std::endl;
                    break;  // Exit the main TPQ loop for this sample
                }
            }
            
            stats_.iterations++;
        }
        
        // Compute final state metrics
        std::pair<double, double> final_pair = computeEnergyAndVariance();
        double final_energy = final_pair.first;
        double final_var = final_pair.second;
        double final_inv_temp = (2.0 * final_step) / (large_value * D_S - final_energy);
        
        // Save final state to HDF5 only if save_thermal_states is enabled
        if (save_thermal_states) {
            saveTPQStateHDF5(dir, sample, final_inv_temp, fixed_sz_op);
            std::cout << "Saved final TPQ state to HDF5: sample=" << sample << ", beta=" << final_inv_temp << std::endl;
        }
        
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
    int taylor_order,
    GPUFixedSzOperator* fixed_sz_op
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
    
    // Initialize MPI-safe HDF5 file for TPQ data
    std::string h5_file;
    #ifdef WITH_MPI
    int mpi_rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    h5_file = HDF5IO::getPerRankFilePath(dir, mpi_rank, "ed_results.h5");
    try {
        if (!HDF5IO::fileExists(h5_file)) {
            HDF5IO::createPerRankFile(dir, mpi_rank, "ed_results.h5");
        }
    } catch (const std::exception& e) {
        std::cerr << "Warning: Could not initialize HDF5 TPQ storage: " << e.what() << std::endl;
        h5_file = "";  // Disable HDF5 writing
    }
    #else
    h5_file = dir + "/ed_results.h5";
    try {
        if (!HDF5IO::fileExists(h5_file)) {
            HDF5IO::createOrOpenFile(dir, "ed_results.h5");
        }
    } catch (const std::exception& e) {
        std::cerr << "Warning: Could not initialize HDF5 TPQ storage: " << e.what() << std::endl;
        h5_file = "";  // Disable HDF5 writing
    }
    #endif
    
    int num_steps = static_cast<int>(beta_max / delta_beta);
    
    // MPI sample distribution - each rank processes only its share of samples
    int start_sample = 0;
    int end_sample = num_samples;
    #ifdef WITH_MPI
    int mpi_size = 1;
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
    
    int samples_per_rank = num_samples / mpi_size;
    int remainder = num_samples % mpi_size;
    
    // Ranks with index < remainder get one extra sample
    start_sample = mpi_rank * samples_per_rank + std::min(mpi_rank, remainder);
    end_sample = start_sample + samples_per_rank + (mpi_rank < remainder ? 1 : 0);
    
    if (mpi_rank == 0) {
        std::cout << "\n==========================================\n";
        std::cout << "MPI-Parallel GPU Canonical TPQ Calculation\n";
        std::cout << "==========================================\n";
        std::cout << "Total MPI ranks: " << mpi_size << "\n";
        std::cout << "Total samples: " << num_samples << "\n";
        std::cout << "Samples per rank: " << samples_per_rank << " (+ " << remainder << " remainder)\n";
        std::cout << "==========================================\n";
    }
    
    std::cout << "Rank " << mpi_rank << " processing samples [" 
              << start_sample << ", " << end_sample << ")" << std::endl;
    
    MPI_Barrier(MPI_COMM_WORLD);
    #else
    int mpi_rank = 0;  // For non-MPI builds
    #endif
    
    // Define measurement temperatures (similar to microcanonical version)
    const int num_temp_points = 20;
    std::vector<double> measure_inv_temp(num_temp_points);
    double log_min = std::log10(1);   // Start from β = 1
    double log_max = std::log10(1000); // End at β = 1000
    for (int i = 0; i < num_temp_points; ++i) {
        measure_inv_temp[i] = std::pow(10.0, log_min + (log_max - log_min) * i / (num_temp_points - 1));
    }
    
    for (int sample = start_sample; sample < end_sample; ++sample) {
        std::cout << "\n[Rank " << mpi_rank << "] Sample " << sample << " of " << num_samples 
                  << " (local: " << (sample - start_sample + 1) << " of " << (end_sample - start_sample) << ")" << std::endl;
        
        // Track which measurement temperatures have been saved
        std::vector<bool> temp_measured(num_temp_points, false);
        
        // Generate random initial state
        generateRandomState(98765 + sample * 43210);
        
        // Initialize HDF5 sample group
        if (!h5_file.empty()) {
            try {
                HDF5IO::ensureTPQSampleGroup(h5_file, sample);
            } catch (const std::exception& e) {
                std::cerr << "Warning: Could not initialize HDF5 sample group: " << e.what() << std::endl;
            }
        }
        
        // Initial measurements at beta=0
        std::pair<double, double> energy_var_pair = computeEnergyAndVariance();
        double energy = energy_var_pair.first;
        double variance = energy_var_pair.second;
        double norm = computeNorm();
        writeTPQDataHDF5(h5_file, sample, 0.0, energy, variance, 0.0, 0);
        
        // Imaginary time evolution
        for (int step = 1; step <= num_steps; ++step) {
            double beta = step * delta_beta;
            
            // Evolve: |state> -> e^{-delta_beta H} |state>
            auto evolve_start = std::chrono::high_resolution_clock::now();
            imaginaryTimeEvolve(delta_beta, taylor_order);
            auto evolve_end = std::chrono::high_resolution_clock::now();
            stats_.matvec_time += std::chrono::duration<double>(evolve_end - evolve_start).count();
            
            // Check if we should measure observables at target temperatures
            // In canonical TPQ, beta is known exactly, so we can check directly
            bool should_measure_observables = false;
            int target_temp_idx = -1;
            
            for (int i = 0; i < num_temp_points; ++i) {
                if (!temp_measured[i]) {
                    // Use relative tolerance
                    double tolerance = 0.01 * measure_inv_temp[i];  // 1% tolerance
                    if (std::abs(beta - measure_inv_temp[i]) < tolerance) {
                        should_measure_observables = true;
                        target_temp_idx = i;
                        break;
                    }
                }
            }
            
            // Determine if we should do measurements this step
            bool do_regular_measurement = (step % temp_interval == 0 || step == num_steps);
            bool do_measurement = do_regular_measurement || should_measure_observables;
            
            // Measurements when needed
            if (do_measurement) {
                std::pair<double, double> E_var_pair = computeEnergyAndVariance();
                double E = E_var_pair.first;
                double var = E_var_pair.second;
                norm = computeNorm();
                
                writeTPQDataHDF5(h5_file, sample, beta, E, var, 0.0, step);
                
                // Save TPQ state at target temperatures to HDF5
                if (should_measure_observables && target_temp_idx >= 0) {
                    std::cout << "  *** Saving TPQ state at β = " << beta 
                              << " (target: " << measure_inv_temp[target_temp_idx] << ") ***" << std::endl;
                    saveTPQStateHDF5(dir, sample, beta, fixed_sz_op);
                    temp_measured[target_temp_idx] = true;
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
