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

bool GPUTPQSolver::saveTPQState(const std::string& filename, GPUFixedSzOperator* fixed_sz_op) {
    // Copy state from GPU to host
    std::vector<std::complex<double>> h_state(N_);
    cudaMemcpy(h_state.data(), d_state_, N_ * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);
    
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) return false;
    
    // Transform to full basis if using fixed-Sz
    if (fixed_sz_op != nullptr) {
        std::vector<std::complex<double>> full_state = fixed_sz_op->embedToFull(h_state);
        size_t full_size = full_state.size();
        file.write(reinterpret_cast<const char*>(&full_size), sizeof(size_t));
        file.write(reinterpret_cast<const char*>(full_state.data()), full_size * sizeof(std::complex<double>));
        std::cout << "  [GPU Fixed-Sz] Transformed state from dim " << N_ 
                  << " to full space dim " << full_size << " before saving" << std::endl;
    } else {
        size_t size = N_;
        file.write(reinterpret_cast<const char*>(&size), sizeof(size_t));
        file.write(reinterpret_cast<const char*>(h_state.data()), N_ * sizeof(std::complex<double>));
    }
    
    file.close();
    return true;
}

bool GPUTPQSolver::loadTPQState(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open TPQ state file: " << filename << std::endl;
        return false;
    }
    
    // Read size
    size_t size;
    file.read(reinterpret_cast<char*>(&size), sizeof(size_t));
    
    if (size != static_cast<size_t>(N_)) {
        std::cerr << "Error: TPQ state dimension mismatch. Expected " << N_ 
                  << ", got " << size << std::endl;
        file.close();
        return false;
    }
    
    // Read state to host
    std::vector<std::complex<double>> h_state(N_);
    file.read(reinterpret_cast<char*>(h_state.data()), N_ * sizeof(std::complex<double>));
    file.close();
    
    // Copy to GPU
    cudaMemcpy(d_state_, h_state.data(), N_ * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);
    
    std::cout << "Loaded TPQ state from: " << filename << std::endl;
    return true;
}

std::string GPUTPQSolver::findLowestEnergyTPQState(const std::string& dir, int sample, 
                                                   double& beta_out, int& step_out) {
    // Check if directory exists using POSIX API
    DIR* directory = opendir(dir.c_str());
    if (!directory) {
        std::cerr << "Error: Directory does not exist: " << dir << std::endl;
        return "";
    }
    
    // Pattern: tpq_state_{sample}_beta={beta}_step={step}.dat (new format with step)
    // Also support legacy pattern: tpq_state_{sample}_beta={beta}.dat
    std::regex state_pattern_new("tpq_state_([0-9]+)_beta=([0-9.]+)_step=([0-9]+)\\.dat");
    std::regex state_pattern_legacy("tpq_state_([0-9]+)_beta=([0-9.]+)\\.dat");
    
    double max_beta = -1.0;
    int best_sample = -1;
    int best_step = -1;
    std::string best_file = "";
    
    struct dirent* entry;
    while ((entry = readdir(directory)) != nullptr) {
        std::string filename = entry->d_name;
        
        // Skip if not a regular file (check using stat)
        std::string filepath = dir + "/" + filename;
        struct stat file_stat;
        if (stat(filepath.c_str(), &file_stat) != 0 || !S_ISREG(file_stat.st_mode)) {
            continue;
        }
        
        std::smatch match;
        int file_sample = -1;
        double file_beta = -1.0;
        int file_step = -1;
        
        // Try new format first
        if (std::regex_match(filename, match, state_pattern_new)) {
            file_sample = std::stoi(match[1].str());
            file_beta = std::stod(match[2].str());
            file_step = std::stoi(match[3].str());
        } 
        // Fall back to legacy format
        else if (std::regex_match(filename, match, state_pattern_legacy)) {
            file_sample = std::stoi(match[1].str());
            file_beta = std::stod(match[2].str());
            file_step = -1; // Will need to look up from SS_rand file
        }
        else {
            continue;
        }
        
        // If sample is specified (non-zero), only consider that sample
        if (sample != 0 && file_sample != sample) continue;
        
        // Find the highest beta (lowest energy state)
        if (file_beta > max_beta) {
            max_beta = file_beta;
            best_sample = file_sample;
            best_step = file_step;
            best_file = filepath;
        }
    }
    
    closedir(directory);
    
    if (best_file.empty()) {
        std::cerr << "Error: No TPQ state files found in directory: " << dir << std::endl;
        return "";
    }
    
    beta_out = max_beta;
    
    // If step was not in filename (legacy format), look it up from HDF5 or SS_rand file
    if (best_step == -1) {
        std::string h5_file = dir + "/ed_results.h5";
        bool found_step = false;
        
        // Try HDF5 first
        if (HDF5IO::fileExists(h5_file)) {
            try {
                auto points = HDF5IO::loadTPQThermodynamics(h5_file, best_sample);
                double closest_beta_diff = 1e10;
                for (const auto& point : points) {
                    double beta_diff = std::abs(point.beta - max_beta);
                    if (beta_diff < closest_beta_diff) {
                        closest_beta_diff = beta_diff;
                        best_step = point.step;
                        found_step = true;
                    }
                }
            } catch (const std::exception& e) {
                // Fall back to text file
            }
        }
        
        // Fall back to SS_rand file if HDF5 lookup failed
        if (!found_step) {
            std::string ss_file = dir + "/SS_rand" + std::to_string(best_sample) + ".dat";
            std::ifstream ss_stream(ss_file);
            
            if (!ss_stream.is_open()) {
                std::cerr << "Warning: Could not find step info in HDF5 or SS_rand file" << std::endl;
                step_out = -1;
                return best_file;
            }
            
            // Find the step corresponding to this beta
            // Format: inv_temp energy variance norm doublon step
            double closest_beta_diff = 1e10;
            int closest_step = -1;
            
            std::string line;
            std::getline(ss_stream, line); // Skip header
            
            while (std::getline(ss_stream, line)) {
                if (line.empty() || line[0] == '#') continue;
                
                std::istringstream iss(line);
                double inv_temp, energy, variance, norm, doublon;
                int step;
                
                // Format: inv_temp energy variance norm doublon step
                if (iss >> inv_temp >> energy >> variance >> norm >> doublon >> step) {
                    double beta_diff = std::abs(inv_temp - max_beta);
                    if (beta_diff < closest_beta_diff) {
                        closest_beta_diff = beta_diff;
                        closest_step = step;
                    }
                }
            }
            
            ss_stream.close();
            best_step = closest_step;
        }
    }
    
    step_out = best_step;
    
    std::cout << "Found TPQ state: sample=" << best_sample 
              << ", beta=" << max_beta 
              << ", step=" << step_out << std::endl;
    
    return best_file;
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
    double continue_beta
) {
    auto total_start = std::chrono::high_resolution_clock::now();
    
    std::cout << "\n=== GPU Microcanonical TPQ ===" << std::endl;
    std::cout << "Hilbert space dimension: " << N_ << std::endl;
    std::cout << "Number of samples: " << num_samples << std::endl;
    std::cout << "Max iterations: " << max_iter << std::endl;
    std::cout << "Large value (energy shift): " << large_value << std::endl;
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
    
    // Initialize HDF5 file for TPQ data
    std::string h5_file = dir + "/ed_results.h5";
    try {
        if (!HDF5IO::fileExists(h5_file)) {
            HDF5IO::createOrOpenFile(dir, "ed_results.h5");
        }
    } catch (const std::exception& e) {
        std::cerr << "Warning: Could not initialize HDF5 TPQ storage: " << e.what() << std::endl;
        h5_file = "";  // Disable HDF5 writing
    }
    
    // Calculate dimension entropy S = log2(N)
    double D_S = std::log2(static_cast<double>(N_));
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
        
        // Variables for continuing from saved state
        int start_step = 1;
        double energy = 0.0;
        double variance = 0.0;
        double inv_temp = 0.0;
        bool loaded_from_file = false;
        
        // Check if we should continue from a saved state (only for first sample)
        if (continue_quenching && sample == 0) {
            double found_beta = 0.0;
            int found_step = -1;
            int found_sample = 0;
            std::string state_file;
            
            if (continue_sample == 0) {
                // Auto-detect lowest energy state (highest beta) from any sample
                std::cout << "Auto-detecting lowest energy state (highest beta)..." << std::endl;
                state_file = findLowestEnergyTPQState(dir, 0, found_beta, found_step);
                
                // Extract sample number from the state file name
                if (!state_file.empty()) {
                    size_t sample_pos = state_file.find("tpq_state_");
                    if (sample_pos != std::string::npos) {
                        size_t beta_pos = state_file.find("_beta=", sample_pos);
                        if (beta_pos != std::string::npos) {
                            std::string sample_str = state_file.substr(sample_pos + 10, beta_pos - (sample_pos + 10));
                            found_sample = std::stoi(sample_str);
                        }
                    }
                }
                
                if (state_file.empty()) {
                    std::cout << "Warning: Could not find saved state to continue from. Falling back to normal TPQ (starting fresh)." << std::endl;
                }
            } else {
                // Use specified sample
                found_sample = continue_sample;
                std::cout << "Continuing from sample " << continue_sample << std::endl;
                
                state_file = findLowestEnergyTPQState(dir, continue_sample, found_beta, found_step);
                
                if (state_file.empty()) {
                    std::cout << "Warning: Could not find state file for sample " << continue_sample 
                              << ". Falling back to normal TPQ (starting fresh)." << std::endl;
                }
            }
            
            // Try to load the state file if we found one
            if (!state_file.empty() && loadTPQState(state_file)) {
                loaded_from_file = true;
                start_step = found_step + 1;
                inv_temp = found_beta;
                
                // Compute energy and variance from loaded state
                std::pair<double, double> ev_pair = computeEnergyAndVariance();
                energy = ev_pair.first;
                variance = ev_pair.second;
                
                std::cout << "Resuming from:" << std::endl;
                std::cout << "  Original sample: " << found_sample << std::endl;
                std::cout << "  Continuing as sample: 0 (output to SS_rand0.dat)" << std::endl;
                std::cout << "  Beta: " << found_beta << std::endl;
                std::cout << "  Step: " << found_step << std::endl;
                std::cout << "  Will run " << max_iter << " additional iterations" << std::endl;
                std::cout << "  Target final step: " << (found_step + max_iter) << std::endl;
                std::cout << "Continuing from step " << found_step 
                          << " (beta=" << found_beta << ", E=" << energy << ")" << std::endl;
            } else if (!state_file.empty()) {
                std::cout << "Warning: Could not load TPQ state. Falling back to normal TPQ (starting fresh)." << std::endl;
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
                
                // Save TPQ state at target temperatures (with accurate inv_temp)
                if (actually_at_target) {
                    std::cout << "  *** Saving TPQ state at β = " << inv_temp 
                              << " (target: " << measure_inv_temp[target_temp_idx] << ") ***" << std::endl;
                    std::string state_file = dir + "/tpq_state_" + std::to_string(sample) + 
                                           "_beta=" + std::to_string(inv_temp) + 
                                           "_step=" + std::to_string(step) + ".dat";
                    saveTPQState(state_file, fixed_sz_op);
                    temp_measured[target_temp_idx] = true;
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
    
    // Initialize HDF5 file for TPQ data
    std::string h5_file = dir + "/ed_results.h5";
    try {
        if (!HDF5IO::fileExists(h5_file)) {
            HDF5IO::createOrOpenFile(dir, "ed_results.h5");
        }
    } catch (const std::exception& e) {
        std::cerr << "Warning: Could not initialize HDF5 TPQ storage: " << e.what() << std::endl;
        h5_file = "";  // Disable HDF5 writing
    }
    
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
                
                // Save state periodically
                if (step % (temp_interval * 5) == 0) {
                    std::string state_file = dir + "/ctpq_state_" + std::to_string(sample) + 
                                           "_beta=" + std::to_string(beta) + ".dat";
                    saveTPQState(state_file, fixed_sz_op);
                }
                
                // Save TPQ state at target temperatures
                if (should_measure_observables && target_temp_idx >= 0) {
                    std::cout << "  *** Saving TPQ state at β = " << beta 
                              << " (target: " << measure_inv_temp[target_temp_idx] << ") ***" << std::endl;
                    std::string state_file = dir + "/tpq_state_" + std::to_string(sample) + 
                                           "_beta=" + std::to_string(beta) + 
                                           "_step=" + std::to_string(step) + ".dat";
                    saveTPQState(state_file, fixed_sz_op);
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
