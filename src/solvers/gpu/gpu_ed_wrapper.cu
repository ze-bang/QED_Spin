#include <ed/gpu/gpu_ed_wrapper.h>
#include <ed/core/hdf5_io.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>
#include <sys/stat.h>
#include <sys/types.h>

#ifdef WITH_CUDA
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <curand.h>
#include <map>
#include <algorithm>
#include <ctime>
#include <ed/gpu/gpu_operator.cuh>
#include <ed/gpu/gpu_lanczos.cuh>
#include <ed/gpu/gpu_tpq.cuh>
#include <ed/gpu/gpu_cg.cuh>
#include <ed/gpu/gpu_ftlm.cuh>

bool GPUEDWrapper::isGPUAvailable() {
    int device_count = 0;
    cudaError_t error = cudaGetDeviceCount(&device_count);
    return (error == cudaSuccess && device_count > 0);
}

void GPUEDWrapper::printGPUInfo() {
    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    
    std::cout << "\n=== GPU Information ===\n";
    std::cout << "Number of CUDA devices: " << device_count << "\n";
    
    for (int i = 0; i < device_count; ++i) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        
        std::cout << "\nDevice " << i << ": " << prop.name << "\n";
        std::cout << "  Compute Capability: " << prop.major << "." << prop.minor << "\n";
        std::cout << "  Total Global Memory: " 
                  << prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0) << " GB\n";
        std::cout << "  Multiprocessors: " << prop.multiProcessorCount << "\n";
        std::cout << "  Max Threads per Block: " << prop.maxThreadsPerBlock << "\n";
        std::cout << "  Warp Size: " << prop.warpSize << "\n";
        std::cout << "  Memory Clock Rate: " << prop.memoryClockRate / 1000.0 << " MHz\n";
        std::cout << "  Memory Bus Width: " << prop.memoryBusWidth << " bits\n";
        std::cout << "  Peak Memory Bandwidth: " 
                  << 2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6 
                  << " GB/s\n";
    }
    std::cout << "=======================\n\n";
}

int GPUEDWrapper::getGPUCount() {
    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    return device_count;
}

size_t GPUEDWrapper::getAvailableGPUMemory(int device) {
    cudaSetDevice(device);
    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);
    return free_mem;
}

size_t GPUEDWrapper::estimateGPUMemory(int n_sites, bool fixed_sz, int n_up) {
    size_t dimension;
    
    if (fixed_sz) {
        // Calculate binomial coefficient
        auto binomial = [](int n, int k) -> size_t {
            if (k > n - k) k = n - k;
            size_t result = 1;
            for (int i = 0; i < k; ++i) {
                result *= (n - i);
                result /= (i + 1);
            }
            return result;
        };
        dimension = binomial(n_sites, n_up);
    } else {
        dimension = 1ULL << n_sites;
    }
    
    // Estimate memory for vectors and operators
    size_t vector_size = dimension * sizeof(std::complex<double>);
    size_t basis_size = fixed_sz ? dimension * sizeof(uint64_t) : 0;
    size_t hash_size = fixed_sz ? dimension * 2 * sizeof(void*) : 0;
    
    // Need at least 4 vectors for Lanczos (current, previous, work, temp)
    size_t total = 4 * vector_size + basis_size + hash_size;
    
    return total;
}

bool GPUEDWrapper::shouldUseGPU(int n_sites, bool fixed_sz) {
    if (!isGPUAvailable()) {
        return false;
    }
    
    // For small systems (< 20 sites), CPU might be faster due to overhead
    if (n_sites < 20) {
        return false;
    }
    
    // For very large systems (> 28 sites without symmetry), GPU is essential
    if (n_sites > 28 && !fixed_sz) {
        return true;
    }
    
    // Check if GPU has enough memory
    size_t required_mem = estimateGPUMemory(n_sites, fixed_sz);
    size_t available_mem = getAvailableGPUMemory(0);
    
    return (required_mem < available_mem * 0.8);
}

void* GPUEDWrapper::createGPUOperatorDirect(
    int n_sites,
    const std::vector<std::tuple<int, int, char, char, double>>& interactions,
    const std::vector<std::tuple<int, char, double>>& single_site_ops) {
    
    GPUOperator* gpu_op = new GPUOperator(n_sites);
    
    // Add interactions
    for (const auto& inter : interactions) {
        int site1, site2;
        char op1, op2;
        double coupling;
        std::tie(site1, site2, op1, op2, coupling) = inter;
        gpu_op->setInteraction(site1, site2, op1, op2, coupling);
    }
    
    // Add single-site operators
    for (const auto& op : single_site_ops) {
        int site;
        char op_type;
        double coupling;
        std::tie(site, op_type, coupling) = op;
        gpu_op->setSingleSite(site, op_type, coupling);
    }
    
    // Allocate GPU memory for vectors
    int N = static_cast<int>(1ULL << n_sites);
    gpu_op->allocateGPUMemory(N);
    
    return static_cast<void*>(gpu_op);
}

void* GPUEDWrapper::createGPUOperatorFromFiles(
    int n_sites,
    const std::string& interall_file,
    const std::string& trans_file) {
    
    // Create GPU operator and populate directly with integers (no char conversion)
    // File format: 0=S+, 1=S-, 2=Sz (matches kernel encoding exactly)
    GPUOperator* gpu_op = new GPUOperator(n_sites);
    
    int num_interactions = 0;
    int num_single_site = 0;
    
    // Load InterAll.dat (two-site interactions)
    std::ifstream interall(interall_file);
    if (!interall.is_open()) {
        std::cerr << "Warning: Could not open " << interall_file << "\n";
    } else {
        std::string line;
        std::getline(interall, line);  // Skip header
        std::getline(interall, line);  // Read num line
        
        std::istringstream iss(line);
        int numLines;
        std::string m;
        iss >> m >> numLines;
        
        // Skip 3 separator lines
        for (int i = 0; i < 3; ++i) std::getline(interall, line);
        
        // Read interactions and add directly
        int lineCount = 0;
        while (std::getline(interall, line) && lineCount < numLines) {
            std::istringstream lineStream(line);
            int Op_i, indx_i, Op_j, indx_j;
            double E, F;
            
            if (!(lineStream >> Op_i >> indx_i >> Op_j >> indx_j >> E >> F)) continue;
            if (std::abs(E) < 1e-12 && std::abs(F) < 1e-12) {
                lineCount++;
                continue;  // Skip zero couplings
            }
            
            // Validate operator codes
            if (Op_i < 0 || Op_i > 2 || Op_j < 0 || Op_j > 2) {
                std::cerr << "Warning: Invalid operator codes: Op_i=" << Op_i << ", Op_j=" << Op_j << "\n";
                lineCount++;
                continue;
            }
            
            // Add directly to transform_data_ - no char conversion!
            // IMPORTANT: Use both real (E) and imaginary (F) parts of the coefficient
            gpu_op->addTwoBodyTerm(Op_i, indx_i, Op_j, indx_j, std::complex<double>(E, F));
            num_interactions++;
            
            lineCount++;
        }
        interall.close();
    }
    
    // Load Trans.dat (single-site terms)
    std::ifstream trans(trans_file);
    if (!trans.is_open()) {
        std::cerr << "Warning: Could not open " << trans_file << "\n";
    } else {
        std::string line;
        std::getline(trans, line);  // Skip header
        std::getline(trans, line);  // Read num line
        
        std::istringstream iss(line);
        int numLines;
        std::string m;
        iss >> m >> numLines;
        
        // Skip 3 separator lines
        for (int i = 0; i < 3; ++i) std::getline(trans, line);
        
        // Read single-site terms and add directly
        int lineCount = 0;
        while (std::getline(trans, line) && lineCount < numLines) {
            std::istringstream lineStream(line);
            int Op, indx;
            double E, F;
            
            if (!(lineStream >> Op >> indx >> E >> F)) continue;
            
            // Only process if coupling is non-zero
            if (std::abs(E) > 1e-12 || std::abs(F) > 1e-12) {
                // Add directly to transform_data_ - no char conversion!
                // IMPORTANT: Use both real (E) and imaginary (F) parts of the coefficient
                gpu_op->addOneBodyTerm(Op, indx, std::complex<double>(E, F));
                num_single_site++;
            }
            
            lineCount++;
        }
        trans.close();
    }
    
    std::cout << "Loaded " << num_interactions << " interaction terms from " 
              << interall_file << "\n";
    std::cout << "Loaded " << num_single_site << " single-site terms from " 
              << trans_file << "\n";
    
    // Allocate GPU memory for vectors
    int N = static_cast<int>(1ULL << n_sites);
    gpu_op->allocateGPUMemory(N);
    
    return static_cast<void*>(gpu_op);
}

void* GPUEDWrapper::createGPUFixedSzOperatorDirect(
    int n_sites, int n_up, float spin_l,
    const std::vector<std::tuple<int, int, char, char, double>>& interactions,
    const std::vector<std::tuple<int, char, double>>& single_site_ops) {
    
    std::cout << "Creating GPU Fixed Sz Operator...\n";
    std::cout << "  Sites: " << n_sites << ", N_up: " << n_up << ", Spin: " << spin_l << "\n";
    
    GPUFixedSzOperator* gpu_op = new GPUFixedSzOperator(n_sites, n_up, spin_l);
    
    // Add interactions
    for (const auto& inter : interactions) {
        int site1, site2;
        char op1, op2;
        double coupling;
        std::tie(site1, site2, op1, op2, coupling) = inter;
        gpu_op->setInteraction(site1, site2, op1, op2, coupling);
    }
    
    // Add single-site operators
    for (const auto& op : single_site_ops) {
        int site;
        char op_type;
        double coupling;
        std::tie(site, op_type, coupling) = op;
        gpu_op->setSingleSite(site, op_type, coupling);
    }
    
    std::cout << "GPU Fixed Sz Operator created successfully\n";
    std::cout << "  Fixed Sz dimension: " << gpu_op->getFixedSzDimension() << "\n";
    
    return static_cast<void*>(gpu_op);
}

void* GPUEDWrapper::createGPUOperatorFromCSR(int n_sites,
                                            int N,
                                            const std::vector<int>& row_ptr,
                                            const std::vector<int>& col_ind,
                                            const std::vector<std::complex<double>>& values) {
    // Create GPUOperator and upload CSR arrays
    GPUOperator* gpu_op = new GPUOperator(n_sites);

    // Allocate GPU vectors (input/output) for size N
    gpu_op->allocateGPUMemory(N);

    // Convert and upload CSR
    bool ok = gpu_op->loadCSR(N, row_ptr, col_ind, values);
    if (!ok) {
        std::cerr << "Error: Failed to load CSR into GPUOperator" << std::endl;
        delete gpu_op;
        return nullptr;
    }

    return static_cast<void*>(gpu_op);
}

void GPUEDWrapper::destroyGPUOperator(void* gpu_op_handle) {
    if (gpu_op_handle) {
        GPUOperator* gpu_op = static_cast<GPUOperator*>(gpu_op_handle);
        delete gpu_op;
    }
}

void GPUEDWrapper::gpuMatVec(void* gpu_op_handle,
                            const std::complex<double>* x,
                            std::complex<double>* y,
                            int N) {
    if (!gpu_op_handle) {
        std::cerr << "Error: NULL GPU operator handle\n";
        return;
    }
    
    GPUOperator* gpu_op = static_cast<GPUOperator*>(gpu_op_handle);
    gpu_op->matVec(x, y, N);
}

void GPUEDWrapper::runGPULanczos(void* gpu_op_handle,
                                int N, int max_iter, int num_eigs,
                                double tol,
                                std::vector<double>& eigenvalues,
                                std::string dir,
                                bool eigenvectors) {
    if (!gpu_op_handle) {
        std::cerr << "Error: NULL GPU operator handle\n";
        return;
    }
    
    GPUOperator* gpu_op = static_cast<GPUOperator*>(gpu_op_handle);
    
    // Note: Do NOT call allocateGPUMemory here if already allocated
    // (it was already done in createGPUOperatorFromCSR or createGPUOperatorFromFiles)
    // Calling it again would clear the CSR data for symmetrized blocks
    
    // Create GPU Lanczos solver
    GPULanczos lanczos(gpu_op, max_iter, tol);
    
    // Run Lanczos
    std::vector<std::vector<std::complex<double>>> eigvecs;
    lanczos.run(num_eigs, eigenvalues, eigvecs, eigenvectors);
    
    // Save results to HDF5
    if (!dir.empty()) {
        if (eigenvectors && !eigvecs.empty()) {
            // Save both eigenvalues and eigenvectors
            HDF5IO::saveDiagonalizationResults(dir, eigenvalues, eigvecs, "GPU_LANCZOS");
            std::cout << "GPU Lanczos: Saved " << eigenvalues.size() << " eigenvalues and " 
                      << eigvecs.size() << " eigenvectors to " << dir << "/ed_results.h5" << std::endl;
        } else {
            // Save eigenvalues only
            try {
                std::string hdf5_file = HDF5IO::createOrOpenFile(dir);
                HDF5IO::saveEigenvalues(hdf5_file, eigenvalues);
                std::cout << "GPU Lanczos: Saved " << eigenvalues.size() << " eigenvalues to " << hdf5_file << std::endl;
            } catch (const std::exception& e) {
                std::cerr << "Warning: Failed to save eigenvalues to HDF5: " << e.what() << std::endl;
            }
        }
    }
    
    // Print statistics
    auto stats = lanczos.getStats();
    std::cout << "\nGPU Lanczos Statistics:\n";
    std::cout << "  Total time: " << stats.total_time << " s\n";
    std::cout << "  MatVec time: " << stats.matvec_time << " s\n";
    std::cout << "  Ortho time: " << stats.ortho_time << " s\n";
    std::cout << "  Iterations: " << stats.iterations << "\n";
    
    // Get operator performance stats
    auto op_stats = gpu_op->getStats();
    std::cout << "  Throughput: " << op_stats.throughput << " GFLOPS\n";
}

void GPUEDWrapper::runGPULanczosFixedSz(void* gpu_op_handle,
                                       int n_up,
                                       int max_iter, int num_eigs,
                                       double tol,
                                       std::vector<double>& eigenvalues,
                                       std::string dir,
                                       bool eigenvectors) {
    if (!gpu_op_handle) {
        std::cerr << "Error: NULL GPU operator handle\n";
        return;
    }
    
    // Cast to GPUFixedSzOperator
    GPUFixedSzOperator* gpu_op = static_cast<GPUFixedSzOperator*>(gpu_op_handle);
    int fixed_sz_dim = gpu_op->getFixedSzDimension();
    
    std::cout << "Running GPU Lanczos for fixed Sz sector (N_up=" << n_up 
              << ", dim=" << fixed_sz_dim << ")\n";
    
    // Allocate GPU memory for vectors
    gpu_op->allocateGPUMemory(fixed_sz_dim);
    
    // Create GPU Lanczos solver with fixed Sz operator
    GPULanczos lanczos(gpu_op, max_iter, tol);
    
    // Run Lanczos
    std::vector<std::vector<std::complex<double>>> eigvecs;
    lanczos.run(num_eigs, eigenvalues, eigvecs, eigenvectors);
    
    // Save results to HDF5
    if (!dir.empty()) {
        if (eigenvectors && !eigvecs.empty()) {
            // Save both eigenvalues and eigenvectors
            HDF5IO::saveDiagonalizationResults(dir, eigenvalues, eigvecs, "GPU_LANCZOS_FIXED_SZ");
            std::cout << "GPU Lanczos Fixed Sz: Saved " << eigenvalues.size() << " eigenvalues and " 
                      << eigvecs.size() << " eigenvectors to " << dir << "/ed_results.h5" << std::endl;
        } else {
            // Save eigenvalues only
            try {
                std::string hdf5_file = HDF5IO::createOrOpenFile(dir);
                HDF5IO::saveEigenvalues(hdf5_file, eigenvalues);
                std::cout << "GPU Lanczos Fixed Sz: Saved " << eigenvalues.size() << " eigenvalues to " << hdf5_file << std::endl;
            } catch (const std::exception& e) {
                std::cerr << "Warning: Failed to save eigenvalues to HDF5: " << e.what() << std::endl;
            }
        }
    }
    
    // Print statistics
    auto stats = lanczos.getStats();
    std::cout << "\nGPU Lanczos Fixed Sz Statistics:\n";
    std::cout << "  Total time: " << stats.total_time << " s\n";
    std::cout << "  MatVec time: " << stats.matvec_time << " s\n";
    std::cout << "  Ortho time: " << stats.ortho_time << " s\n";
    std::cout << "  Iterations: " << stats.iterations << "\n";
    
    // Get operator performance stats
    auto op_stats = gpu_op->getStats();
    std::cout << "  Throughput: " << op_stats.throughput << " GFLOPS\n";
}

void GPUEDWrapper::runGPUMicrocanonicalTPQ(void* gpu_op_handle,
                                           int N, int max_iter, int num_samples,
                                           int temp_interval,
                                           std::vector<double>& eigenvalues,
                                           std::string dir,
                                           double large_value,
                                           bool continue_quenching,
                                           int continue_sample,
                                           double continue_beta) {
    if (!gpu_op_handle) {
        std::cerr << "Error: NULL GPU operator handle\n";
        return;
    }
    
    GPUOperator* gpu_op = static_cast<GPUOperator*>(gpu_op_handle);
    GPUTPQSolver tpq_solver(gpu_op, N);
    
    tpq_solver.runMicrocanonicalTPQ(max_iter, num_samples, temp_interval,
                                    eigenvalues, dir, large_value, nullptr,
                                    continue_quenching, continue_sample, continue_beta);
}

void GPUEDWrapper::runGPUMicrocanonicalTPQFixedSz(void* gpu_op_handle,
                                                 int n_up,
                                                 int max_iter, int num_samples,
                                                 int temp_interval,
                                                 std::vector<double>& eigenvalues,
                                                 std::string dir,
                                                 double large_value,
                                                 bool continue_quenching,
                                                 int continue_sample,
                                                 double continue_beta) {
    if (!gpu_op_handle) {
        std::cerr << "Error: NULL GPU operator handle\n";
        return;
    }
    
    // Cast to GPUFixedSzOperator
    GPUFixedSzOperator* gpu_op = static_cast<GPUFixedSzOperator*>(gpu_op_handle);
    int fixed_sz_dim = gpu_op->getFixedSzDimension();
    
    std::cout << "Running GPU Microcanonical TPQ for fixed Sz sector (N_up=" << n_up 
              << ", dim=" << fixed_sz_dim << ")\n";
    
    // Allocate GPU memory for vectors
    gpu_op->allocateGPUMemory(fixed_sz_dim);
    
    // Create GPU TPQ solver with fixed Sz operator (pass pointer for embedding)
    GPUTPQSolver tpq_solver(gpu_op, fixed_sz_dim);
    
    tpq_solver.runMicrocanonicalTPQ(max_iter, num_samples, temp_interval,
                                    eigenvalues, dir, large_value, gpu_op,
                                    continue_quenching, continue_sample, continue_beta);
    
    std::cout << "\nGPU Microcanonical TPQ Fixed Sz complete\n";
}

void GPUEDWrapper::runGPUCanonicalTPQ(void* gpu_op_handle,
                                      int N, double beta_max, int num_samples,
                                      int temp_interval,
                                      std::vector<double>& energies,
                                      std::string dir,
                                      double delta_beta,
                                      int taylor_order) {
    if (!gpu_op_handle) {
        std::cerr << "Error: NULL GPU operator handle\n";
        return;
    }
    
    GPUOperator* gpu_op = static_cast<GPUOperator*>(gpu_op_handle);
    GPUTPQSolver tpq_solver(gpu_op, N);
    
    tpq_solver.runCanonicalTPQ(beta_max, num_samples, temp_interval,
                               energies, dir, delta_beta, taylor_order);
}

void GPUEDWrapper::runGPUCanonicalTPQFixedSz(void* gpu_op_handle,
                                            int n_up,
                                            double beta_max, int num_samples,
                                            int temp_interval,
                                            std::vector<double>& energies,
                                            std::string dir,
                                            double delta_beta,
                                            int taylor_order) {
    if (!gpu_op_handle) {
        std::cerr << "Error: NULL GPU operator handle\n";
        return;
    }
    
    // Cast to GPUFixedSzOperator
    GPUFixedSzOperator* gpu_op = static_cast<GPUFixedSzOperator*>(gpu_op_handle);
    int fixed_sz_dim = gpu_op->getFixedSzDimension();
    
    std::cout << "Running GPU Canonical TPQ for fixed Sz sector (N_up=" << n_up 
              << ", dim=" << fixed_sz_dim << ")\n";
    
    // Allocate GPU memory for vectors
    gpu_op->allocateGPUMemory(fixed_sz_dim);
    
    // Create GPU TPQ solver with fixed Sz operator (pass pointer for embedding)
    GPUTPQSolver tpq_solver(gpu_op, fixed_sz_dim);
    
    tpq_solver.runCanonicalTPQ(beta_max, num_samples, temp_interval,
                               energies, dir, delta_beta, taylor_order, gpu_op);
    
    std::cout << "\nGPU Canonical TPQ Fixed Sz complete\n";
}

void GPUEDWrapper::runGPUDavidson(void* gpu_op_handle,
                                  int N, int num_eigenvalues, int max_iter,
                                  int max_subspace, double tol,
                                  std::vector<double>& eigenvalues,
                                  std::string dir,
                                  bool compute_eigenvectors) {
    if (!gpu_op_handle) {
        std::cerr << "Error: NULL GPU operator handle\n";
        return;
    }
    
    GPUOperator* gpu_op = static_cast<GPUOperator*>(gpu_op_handle);
    GPUIterativeSolver solver(gpu_op, N);
    
    std::vector<std::vector<std::complex<double>>> eigenvectors;
    solver.runDavidson(num_eigenvalues, max_iter, max_subspace, tol,
                      eigenvalues, eigenvectors, dir, compute_eigenvectors);
}

void GPUEDWrapper::runGPUDavidsonFixedSz(void* gpu_op_handle,
                                        int n_up,
                                        int num_eigenvalues, int max_iter,
                                        int max_subspace, double tol,
                                        std::vector<double>& eigenvalues,
                                        std::string dir,
                                        bool compute_eigenvectors) {
    if (!gpu_op_handle) {
        std::cerr << "Error: NULL GPU operator handle\n";
        return;
    }
    
    // Cast to GPUFixedSzOperator
    GPUFixedSzOperator* gpu_op = static_cast<GPUFixedSzOperator*>(gpu_op_handle);
    int fixed_sz_dim = gpu_op->getFixedSzDimension();
    
    std::cout << "Running GPU Davidson for fixed Sz sector (N_up=" << n_up 
              << ", dim=" << fixed_sz_dim << ")\n";
    
    // Allocate GPU memory for vectors
    gpu_op->allocateGPUMemory(fixed_sz_dim);
    
    // Create GPU Davidson solver with fixed Sz operator
    GPUIterativeSolver solver(gpu_op, fixed_sz_dim);
    
    std::vector<std::vector<std::complex<double>>> eigenvectors;
    solver.runDavidson(num_eigenvalues, max_iter, max_subspace, tol,
                      eigenvalues, eigenvectors, dir, compute_eigenvectors);
    
    std::cout << "\nGPU Davidson Fixed Sz complete\n";
}

void GPUEDWrapper::runGPULOBPCG(void* gpu_op_handle,
                                int N, int num_eigenvalues, int max_iter,
                                double tol,
                                std::vector<double>& eigenvalues,
                                std::string dir,
                                bool compute_eigenvectors) {
    // LOBPCG_GPU is retired - redirect to Davidson GPU which is more robust
    std::cout << "Note: LOBPCG_GPU is deprecated. Using Davidson GPU instead.\n";
    
    // Use a reasonable max_subspace for Davidson (typically 2-3x num_eigenvalues)
    int max_subspace = std::max(50, 3 * num_eigenvalues);
    
    runGPUDavidson(gpu_op_handle, N, num_eigenvalues, max_iter, max_subspace,
                   tol, eigenvalues, dir, compute_eigenvectors);
}

void GPUEDWrapper::runGPULOBPCGFixedSz(void* gpu_op_handle,
                                      int n_up,
                                      int num_eigenvalues, int max_iter,
                                      double tol,
                                      std::vector<double>& eigenvalues,
                                      std::string dir,
                                      bool compute_eigenvectors) {
    // LOBPCG_GPU is retired - redirect to Davidson GPU which is more robust
    std::cout << "Note: LOBPCG_GPU Fixed Sz is deprecated. Using Davidson GPU Fixed Sz instead.\n";
    
    // Use a reasonable max_subspace for Davidson (typically 2-3x num_eigenvalues)
    int max_subspace = std::max(50, 3 * num_eigenvalues);
    
    runGPUDavidsonFixedSz(gpu_op_handle, n_up, num_eigenvalues, max_iter, max_subspace,
                         tol, eigenvalues, dir, compute_eigenvectors);
}

void GPUEDWrapper::runGPUFTLM(void* gpu_op_handle,
                             int N,
                             int krylov_dim,
                             int num_samples,
                             double temp_min,
                             double temp_max,
                             int num_temp_bins,
                             double tolerance,
                             std::string dir,
                             bool full_reorth,
                             int reorth_freq,
                             unsigned int random_seed) {
    if (!gpu_op_handle) {
        std::cerr << "Error: GPU operator handle is null\n";
        return;
    }
    
    GPUOperator* gpu_op = static_cast<GPUOperator*>(gpu_op_handle);
    
    // Create GPU FTLM solver
    GPUFTLMSolver ftlm_solver(gpu_op, N, krylov_dim, tolerance);
    
    // Run FTLM
    FTLMResults results = ftlm_solver.run(num_samples, temp_min, temp_max, 
                                         num_temp_bins, dir, full_reorth, 
                                         reorth_freq, random_seed);
    
    // Save results if directory provided
    if (!dir.empty()) {
        // Create thermo subdirectory if it doesn't exist
        std::string thermo_dir = dir + "/thermo";
        mkdir(thermo_dir.c_str(), 0755);
        
        std::string output_file = thermo_dir + "/ftlm_thermo.txt";
        save_ftlm_results(results, output_file);
    }
    
    // Print statistics
    auto stats = ftlm_solver.getStats();
    std::cout << "\nGPU FTLM Statistics:\n";
    std::cout << "  Total time: " << stats.total_time << " s\n";
    std::cout << "  Lanczos time: " << stats.lanczos_time << " s\n";
    std::cout << "  Thermodynamics time: " << stats.thermo_time << " s\n";
    std::cout << "  Total iterations: " << stats.total_iterations << "\n";
    std::cout << "  Samples completed: " << stats.num_samples_completed << "\n";
}

void GPUEDWrapper::runGPUFTLMFixedSz(void* gpu_op_handle,
                                    int n_up,
                                    int krylov_dim,
                                    int num_samples,
                                    double temp_min,
                                    double temp_max,
                                    int num_temp_bins,
                                    double tolerance,
                                    std::string dir,
                                    bool full_reorth,
                                    int reorth_freq,
                                    unsigned int random_seed) {
    if (!gpu_op_handle) {
        std::cerr << "Error: GPU operator handle is null\n";
        return;
    }
    
    // Cast to GPUFixedSzOperator
    GPUFixedSzOperator* gpu_op = static_cast<GPUFixedSzOperator*>(gpu_op_handle);
    int fixed_sz_dim = gpu_op->getFixedSzDimension();
    
    std::cout << "Running GPU FTLM for fixed Sz sector (N_up=" << n_up 
              << ", dimension=" << fixed_sz_dim << ")\n";
    
    // Create GPU FTLM solver for fixed Sz
    GPUFTLMSolver ftlm_solver(gpu_op, fixed_sz_dim, krylov_dim, tolerance);
    
    // Run FTLM
    FTLMResults results = ftlm_solver.run(num_samples, temp_min, temp_max, 
                                         num_temp_bins, dir, full_reorth, 
                                         reorth_freq, random_seed);
    
    // Save results if directory provided
    if (!dir.empty()) {
        // Create thermo subdirectory if it doesn't exist
        std::string thermo_dir = dir + "/thermo";
        mkdir(thermo_dir.c_str(), 0755);
        
        std::string output_file = thermo_dir + "/ftlm_thermo.txt";
        save_ftlm_results(results, output_file);
    }
    
    // Print statistics
    auto stats = ftlm_solver.getStats();
    std::cout << "\nGPU FTLM Fixed Sz Statistics:\n";
    std::cout << "  Total time: " << stats.total_time << " s\n";
    std::cout << "  Lanczos time: " << stats.lanczos_time << " s\n";
    std::cout << "  Thermodynamics time: " << stats.thermo_time << " s\n";
    std::cout << "  Total iterations: " << stats.total_iterations << "\n";
    std::cout << "  Samples completed: " << stats.num_samples_completed << "\n";
    
    std::cout << "\nGPU FTLM Fixed Sz complete\n";
}

std::pair<std::vector<double>, std::vector<double>>
GPUEDWrapper::runGPUDynamicalResponse(void* gpu_op_handle,
                                     void* gpu_obs_handle,
                                     void* d_psi_state,
                                     int N,
                                     int krylov_dim,
                                     double omega_min,
                                     double omega_max,
                                     int num_omega_bins,
                                     double broadening,
                                     double temperature,
                                     double ground_state_energy) {
    if (!gpu_op_handle) {
        std::cerr << "Error: GPU operator handle is null\n";
        return {{}, {}};
    }
    
    GPUOperator* gpu_op = static_cast<GPUOperator*>(gpu_op_handle);
    GPUOperator* gpu_obs = gpu_obs_handle ? static_cast<GPUOperator*>(gpu_obs_handle) : nullptr;
    cuDoubleComplex* d_psi = static_cast<cuDoubleComplex*>(d_psi_state);
    
    // Create GPU FTLM solver
    GPUFTLMSolver ftlm_solver(gpu_op, N, krylov_dim, 1e-10);
    
    // Shift frequencies by ground state energy
    double omega_min_shifted = omega_min + ground_state_energy;
    double omega_max_shifted = omega_max + ground_state_energy;
    
    // Compute dynamical response
    auto result = ftlm_solver.computeDynamicalResponse(
        d_psi, gpu_obs, omega_min_shifted, omega_max_shifted, 
        num_omega_bins, broadening, temperature
    );
    
    // Shift frequencies back for output
    for (auto& freq : result.first) {
        freq -= ground_state_energy;
    }
    
    return result;
}

std::tuple<std::vector<double>, std::vector<double>, std::vector<double>>
GPUEDWrapper::runGPUDynamicalResponseThermal(void* gpu_op_handle,
                                            void* gpu_obs_handle,
                                            int N,
                                            int num_samples,
                                            int krylov_dim,
                                            double omega_min,
                                            double omega_max,
                                            int num_omega_bins,
                                            double broadening,
                                            double temperature,
                                            unsigned int random_seed,
                                            double ground_state_energy) {
    if (!gpu_op_handle) {
        std::cerr << "Error: GPU operator handle is null\n";
        return {{}, {}, {}};
    }
    
    GPUOperator* gpu_op = static_cast<GPUOperator*>(gpu_op_handle);
    GPUOperator* gpu_obs = gpu_obs_handle ? static_cast<GPUOperator*>(gpu_obs_handle) : nullptr;
    
    // Create GPU FTLM solver
    GPUFTLMSolver ftlm_solver(gpu_op, N, krylov_dim, 1e-10);
    
    // Shift frequencies by ground state energy
    double omega_min_shifted = omega_min + ground_state_energy;
    double omega_max_shifted = omega_max + ground_state_energy;
    
    // Compute thermal dynamical response
    auto result = ftlm_solver.computeDynamicalResponseThermal(
        num_samples, gpu_obs, omega_min_shifted, omega_max_shifted,
        num_omega_bins, broadening, temperature, random_seed
    );
    
    // Shift frequencies back for output
    for (auto& freq : std::get<0>(result)) {
        freq -= ground_state_energy;
    }
    
    return result;
}

std::tuple<std::vector<double>, std::vector<double>, std::vector<double>,
          std::vector<double>, std::vector<double>>
GPUEDWrapper::runGPUDynamicalCorrelation(void* gpu_op_handle,
                                        void* gpu_obs1_handle,
                                        void* gpu_obs2_handle,
                                        int N,
                                        int num_samples,
                                        int krylov_dim,
                                        double omega_min,
                                        double omega_max,
                                        int num_omega_bins,
                                        double broadening,
                                        double temperature,
                                        unsigned int random_seed,
                                        double ground_state_energy) {
    if (!gpu_op_handle || !gpu_obs1_handle || !gpu_obs2_handle) {
        std::cerr << "Error: GPU operator handles are null\n";
        return {{}, {}, {}, {}, {}};
    }
    
    GPUOperator* gpu_op = static_cast<GPUOperator*>(gpu_op_handle);
    GPUOperator* gpu_obs1 = static_cast<GPUOperator*>(gpu_obs1_handle);
    GPUOperator* gpu_obs2 = static_cast<GPUOperator*>(gpu_obs2_handle);
    
    // Create GPU FTLM solver
    GPUFTLMSolver ftlm_solver(gpu_op, N, krylov_dim, 1e-10);
    
    // Compute dynamical correlation
    // Note: ground_state_energy is passed as energy_shift parameter
    auto result = ftlm_solver.computeDynamicalCorrelation(
        num_samples, gpu_obs1, gpu_obs2, omega_min, omega_max,
        num_omega_bins, broadening, temperature, ground_state_energy, 
        random_seed, "", false
    );
    
    return result;
}

std::tuple<std::vector<double>, std::vector<double>, std::vector<double>>
GPUEDWrapper::runGPUDynamicalCorrelationState(void* gpu_op_handle,
                                              void* gpu_obs1_handle,
                                              void* gpu_obs2_handle,
                                              void* d_psi_state,
                                              int N,
                                              int krylov_dim,
                                              double omega_min,
                                              double omega_max,
                                              int num_omega_bins,
                                              double broadening,
                                              double temperature,
                                              double ground_state_energy) {
    if (!gpu_op_handle || !gpu_obs1_handle || !gpu_obs2_handle || !d_psi_state) {
        std::cerr << "Error: GPU handles or state is null\n";
        return {{}, {}, {}};
    }
    
    GPUOperator* gpu_op = static_cast<GPUOperator*>(gpu_op_handle);
    GPUOperator* gpu_obs1 = static_cast<GPUOperator*>(gpu_obs1_handle);
    GPUOperator* gpu_obs2 = static_cast<GPUOperator*>(gpu_obs2_handle);
    cuDoubleComplex* d_psi = static_cast<cuDoubleComplex*>(d_psi_state);
    
    // Create GPU FTLM solver
    GPUFTLMSolver ftlm_solver(gpu_op, N, krylov_dim, 1e-10);
    
    // Compute dynamical correlation for specific state
    auto result = ftlm_solver.computeDynamicalCorrelationState(
        d_psi, gpu_obs1, gpu_obs2, omega_min, omega_max,
        num_omega_bins, broadening, temperature, ground_state_energy
    );
    
    return result;
}

std::tuple<std::vector<double>, std::vector<double>, std::vector<double>,
          std::vector<double>, std::vector<double>>
GPUEDWrapper::runGPUThermalExpectation(void* gpu_op_handle,
                                      void* gpu_obs_handle,
                                      int N,
                                      int num_samples,
                                      int krylov_dim,
                                      double temp_min,
                                      double temp_max,
                                      int num_temp_bins,
                                      unsigned int random_seed) {
    if (!gpu_op_handle || !gpu_obs_handle) {
        std::cerr << "Error: GPU operator handles are null\n";
        return {{}, {}, {}, {}, {}};
    }
    
    GPUOperator* gpu_op = static_cast<GPUOperator*>(gpu_op_handle);
    GPUOperator* gpu_obs = static_cast<GPUOperator*>(gpu_obs_handle);
    
    // Create GPU FTLM solver
    GPUFTLMSolver ftlm_solver(gpu_op, N, krylov_dim, 1e-10);
    
    // Compute thermal expectation values
    // Returns (temperatures, expectations, errors)
    auto result = ftlm_solver.computeThermalExpectation(
        num_samples, gpu_obs, temp_min, temp_max, num_temp_bins, random_seed
    );
    
    auto temps = std::get<0>(result);
    auto exps = std::get<1>(result);
    auto errs = std::get<2>(result);
    
    // Compute susceptibility χ = β(⟨O²⟩ - ⟨O⟩²) = βσ²
    // Note: This is a simplified version; full implementation would need ⟨O²⟩
    std::vector<double> susceptibility(temps.size());
    std::vector<double> sus_error(temps.size());
    for (size_t i = 0; i < temps.size(); ++i) {
        double beta = (temps[i] > 1e-10) ? (1.0 / temps[i]) : 0.0;
        susceptibility[i] = beta * errs[i] * errs[i];  // Simplified: βσ²
        sus_error[i] = 0.0;  // Placeholder
    }
    
    return std::make_tuple(temps, exps, susceptibility, errs, sus_error);
}

std::tuple<std::vector<double>, std::vector<double>, std::vector<double>,
          std::vector<double>, std::vector<double>>
GPUEDWrapper::runGPUStaticCorrelation(void* gpu_op_handle,
                                     void* gpu_obs1_handle,
                                     void* gpu_obs2_handle,
                                     int N,
                                     int num_samples,
                                     int krylov_dim,
                                     double temp_min,
                                     double temp_max,
                                     int num_temp_bins,
                                     unsigned int random_seed) {
    if (!gpu_op_handle || !gpu_obs1_handle || !gpu_obs2_handle) {
        std::cerr << "Error: GPU operator handles are null\n";
        return {{}, {}, {}, {}, {}};
    }
    
    GPUOperator* gpu_op = static_cast<GPUOperator*>(gpu_op_handle);
    GPUOperator* gpu_obs1 = static_cast<GPUOperator*>(gpu_obs1_handle);
    GPUOperator* gpu_obs2 = static_cast<GPUOperator*>(gpu_obs2_handle);
    
    // Create GPU FTLM solver
    GPUFTLMSolver ftlm_solver(gpu_op, N, krylov_dim, 1e-10);
    
    // Compute static correlation
    // Returns (temperatures, correlations, errors)
    auto result = ftlm_solver.computeStaticCorrelation(
        num_samples, gpu_obs1, gpu_obs2, temp_min, temp_max, num_temp_bins, random_seed
    );
    
    auto temps = std::get<0>(result);
    auto corr = std::get<1>(result);
    auto errs = std::get<2>(result);
    
    // Static correlations are real-valued for Hermitian operators
    // Return real part in first vector, imaginary (zero) in second
    std::vector<double> corr_imag(corr.size(), 0.0);
    std::vector<double> err_imag(errs.size(), 0.0);
    
    return std::make_tuple(temps, corr, corr_imag, errs, err_imag);
}

std::map<double, std::tuple<std::vector<double>, std::vector<double>, std::vector<double>>>
GPUEDWrapper::runGPUDynamicalCorrelationMultiTemp(void* gpu_op_handle,
                                                 void* gpu_obs1_handle,
                                                 void* gpu_obs2_handle,
                                                 int N,
                                                 int num_samples,
                                                 int krylov_dim,
                                                 double omega_min,
                                                 double omega_max,
                                                 int num_omega_bins,
                                                 double broadening,
                                                 const std::vector<double>& temperatures,
                                                 unsigned int random_seed,
                                                 double ground_state_energy) {
    std::cout << "\n==========================================" << std::endl;
    std::cout << "GPU MULTI-SAMPLE MULTI-TEMPERATURE FTLM (CORRECT)" << std::endl;
    std::cout << "==========================================" << std::endl;
    std::cout << "Samples: " << num_samples << std::endl;
    std::cout << "Temperatures: " << temperatures.size() << std::endl;
    std::cout << "Using correct FTLM formulation matching CPU implementation" << std::endl;
    std::cout << "==========================================" << std::endl;
    
    if (!gpu_op_handle || !gpu_obs1_handle || !gpu_obs2_handle) {
        std::cerr << "Error: GPU operator handles are null\n";
        return {};
    }
    
    GPUOperator* gpu_op = static_cast<GPUOperator*>(gpu_op_handle);
    GPUOperator* gpu_obs1 = static_cast<GPUOperator*>(gpu_obs1_handle);
    GPUOperator* gpu_obs2 = static_cast<GPUOperator*>(gpu_obs2_handle);
    
    // Create GPU FTLM solver
    GPUFTLMSolver ftlm_solver(gpu_op, N, krylov_dim, 1e-10);
    
    // Call the CORRECT FTLM multi-temperature spectral function
    auto full_results = ftlm_solver.computeDynamicalCorrelationMultiTemp(
        num_samples,
        gpu_obs1,
        gpu_obs2,
        omega_min,
        omega_max,
        num_omega_bins,
        broadening,
        temperatures,
        ground_state_energy,
        random_seed
    );
    
    // Convert to the expected return format (without errors for backward compatibility)
    std::map<double, std::tuple<std::vector<double>, std::vector<double>, std::vector<double>>> results;
    
    for (const auto& [T, data] : full_results) {
        auto& [freqs, S_real, S_imag, err_real, err_imag] = data;
        results[T] = std::make_tuple(freqs, S_real, S_imag);
    }
    
    std::cout << "\nGPU multi-temperature FTLM complete!\n";
    return results;
}

bool GPUEDWrapper::createGPUOperatorFromCPU(const Operator& cpu_op,
                                           void** gpu_op_handle,
                                           int n_sites) {
    // This would extract interactions from CPU operator and create GPU version
    // Placeholder for now - requires access to Operator internals
    std::cout << "CPU to GPU operator conversion not yet implemented\n";
    return false;
}

#else // !WITH_CUDA

// Stub implementations when CUDA is not available
bool GPUEDWrapper::isGPUAvailable() { return false; }
void GPUEDWrapper::printGPUInfo() { 
    std::cout << "CUDA support not compiled\n"; 
}
int GPUEDWrapper::getGPUCount() { return 0; }
size_t GPUEDWrapper::getAvailableGPUMemory(int device) { return 0; }
size_t GPUEDWrapper::estimateGPUMemory(int n_sites, bool fixed_sz, int n_up) { return 0; }
bool GPUEDWrapper::shouldUseGPU(int n_sites, bool fixed_sz) { return false; }

void* GPUEDWrapper::createGPUOperatorDirect(
    int n_sites,
    const std::vector<std::tuple<int, int, char, char, double>>& interactions,
    const std::vector<std::tuple<int, char, double>>& single_site_ops) {
    return nullptr;
}

void* GPUEDWrapper::createGPUOperatorFromFiles(
    int n_sites,
    const std::string& interall_file,
    const std::string& trans_file) {
    return nullptr;
}

void* GPUEDWrapper::createGPUOperatorFromCSR(int n_sites,
                                            int N,
                                            const std::vector<int>& row_ptr,
                                            const std::vector<int>& col_ind,
                                            const std::vector<std::complex<double>>& values) {
    return nullptr;
}

void GPUEDWrapper::destroyGPUOperator(void* gpu_op_handle) {}

void GPUEDWrapper::gpuMatVec(void* gpu_op_handle,
                            const std::complex<double>* x,
                            std::complex<double>* y,
                            int N) {}

void GPUEDWrapper::runGPULanczos(void* gpu_op_handle,
                                int N, int max_iter, int num_eigs,
                                double tol,
                                std::vector<double>& eigenvalues,
                                std::string dir,
                                bool eigenvectors) {}

void GPUEDWrapper::runGPULanczosFixedSz(void* gpu_op_handle,
                                       int n_up,
                                       int max_iter, int num_eigs,
                                       double tol,
                                       std::vector<double>& eigenvalues,
                                       std::string dir,
                                       bool eigenvectors) {}

void GPUEDWrapper::runGPUDavidsonFixedSz(void* gpu_op_handle,
                                        int n_up,
                                        int num_eigenvalues, int max_iter,
                                        int max_subspace, double tol,
                                        std::vector<double>& eigenvalues,
                                        std::string dir,
                                        bool compute_eigenvectors) {}

void GPUEDWrapper::runGPUMicrocanonicalTPQFixedSz(void* gpu_op_handle,
                                                 int n_up,
                                                 int max_iter, int num_samples,
                                                 int temp_interval,
                                                 std::vector<double>& eigenvalues,
                                                 std::string dir,
                                                 double large_value) {}

void GPUEDWrapper::runGPUCanonicalTPQFixedSz(void* gpu_op_handle,
                                            int n_up,
                                            double beta_max, int num_samples,
                                            int temp_interval,
                                            std::vector<double>& energies,
                                            std::string dir,
                                            double delta_beta,
                                            int taylor_order) {}

void* GPUEDWrapper::createGPUFixedSzOperatorDirect(
    int n_sites, int n_up, float spin_l,
    const std::vector<std::tuple<int, int, char, char, double>>& interactions,
    const std::vector<std::tuple<int, char, double>>& single_site_ops) {
    return nullptr;
}

void GPUEDWrapper::runGPUMicrocanonicalTPQ(void* gpu_op_handle,
                                           int N, int max_iter, int num_samples,
                                           int temp_interval,
                                           std::vector<double>& eigenvalues,
                                           std::string dir,
                                           double large_value) {}

void GPUEDWrapper::runGPUCanonicalTPQ(void* gpu_op_handle,
                                      int N, double beta_max, int num_samples,
                                      int temp_interval,
                                      std::vector<double>& energies,
                                      std::string dir,
                                      double delta_beta,
                                      int taylor_order) {}

void GPUEDWrapper::runGPUDavidson(void* gpu_op_handle,
                                  int N, int num_eigenvalues, int max_iter,
                                  int max_subspace, double tol,
                                  std::vector<double>& eigenvalues,
                                  std::string dir,
                                  bool compute_eigenvectors) {}

void GPUEDWrapper::runGPULOBPCG(void* gpu_op_handle,
                                int N, int num_eigenvalues, int max_iter,
                                double tol,
                                std::vector<double>& eigenvalues,
                                std::string dir,
                                bool compute_eigenvectors) {
    // Stub: LOBPCG_GPU redirects to Davidson GPU
    std::cerr << "CUDA not available - cannot run GPU methods\n";
}

void GPUEDWrapper::runGPULOBPCGFixedSz(void* gpu_op_handle,
                                      int n_up,
                                      int num_eigenvalues, int max_iter,
                                      double tol,
                                      std::vector<double>& eigenvalues,
                                      std::string dir,
                                      bool compute_eigenvectors) {
    // Stub: LOBPCG_GPU Fixed Sz redirects to Davidson GPU
    std::cerr << "CUDA not available - cannot run GPU methods\n";
}

void GPUEDWrapper::runGPUFTLM(void* gpu_op_handle,
                             int N,
                             int krylov_dim,
                             int num_samples,
                             double temp_min,
                             double temp_max,
                             int num_temp_bins,
                             double tolerance,
                             std::string dir,
                             bool full_reorth,
                             int reorth_freq,
                             unsigned int random_seed) {
    std::cerr << "CUDA not available - cannot run GPU FTLM\n";
}

void GPUEDWrapper::runGPUFTLMFixedSz(void* gpu_op_handle,
                                    int n_up,
                                    int krylov_dim,
                                    int num_samples,
                                    double temp_min,
                                    double temp_max,
                                    int num_temp_bins,
                                    double tolerance,
                                    std::string dir,
                                    bool full_reorth,
                                    int reorth_freq,
                                    unsigned int random_seed) {
    std::cerr << "CUDA not available - cannot run GPU FTLM Fixed Sz\n";
}

std::pair<std::vector<double>, std::vector<double>>
GPUEDWrapper::runGPUDynamicalResponse(void* gpu_op_handle,
                                     void* gpu_obs_handle,
                                     void* d_psi_state,
                                     int N,
                                     int krylov_dim,
                                     double omega_min,
                                     double omega_max,
                                     int num_omega_bins,
                                     double broadening,
                                     double temperature,
                                     double ground_state_energy) {
    std::cerr << "CUDA not available - cannot run GPU dynamical response\n";
    return {{}, {}};
}

std::tuple<std::vector<double>, std::vector<double>, std::vector<double>>
GPUEDWrapper::runGPUDynamicalResponseThermal(void* gpu_op_handle,
                                            void* gpu_obs_handle,
                                            int N,
                                            int num_samples,
                                            int krylov_dim,
                                            double omega_min,
                                            double omega_max,
                                            int num_omega_bins,
                                            double broadening,
                                            double temperature,
                                            unsigned int random_seed,
                                            double ground_state_energy) {
    std::cerr << "CUDA not available - cannot run GPU thermal dynamical response\n";
    return {{}, {}, {}};
}

std::tuple<std::vector<double>, std::vector<double>, std::vector<double>,
          std::vector<double>, std::vector<double>>
GPUEDWrapper::runGPUDynamicalCorrelation(void* gpu_op_handle,
                                        void* gpu_obs1_handle,
                                        void* gpu_obs2_handle,
                                        int N,
                                        int num_samples,
                                        int krylov_dim,
                                        double omega_min,
                                        double omega_max,
                                        int num_omega_bins,
                                        double broadening,
                                        double temperature,
                                        unsigned int random_seed,
                                        double ground_state_energy) {
    std::cerr << "CUDA not available - cannot run GPU dynamical correlation\n";
    return {{}, {}, {}, {}, {}};
}

std::tuple<std::vector<double>, std::vector<double>, std::vector<double>>
GPUEDWrapper::runGPUDynamicalCorrelationState(void* gpu_op_handle,
                                              void* gpu_obs1_handle,
                                              void* gpu_obs2_handle,
                                              void* d_psi_state,
                                              int N,
                                              int krylov_dim,
                                              double omega_min,
                                              double omega_max,
                                              int num_omega_bins,
                                              double broadening,
                                              double temperature,
                                              double ground_state_energy) {
    std::cerr << "CUDA not available - cannot run GPU dynamical correlation state\n";
    return {{}, {}, {}};
}

std::tuple<std::vector<double>, std::vector<double>, std::vector<double>,
          std::vector<double>, std::vector<double>>
GPUEDWrapper::runGPUThermalExpectation(void* gpu_op_handle,
                                      void* gpu_obs_handle,
                                      int N,
                                      int num_samples,
                                      int krylov_dim,
                                      double temp_min,
                                      double temp_max,
                                      int num_temp_bins,
                                      unsigned int random_seed) {
    std::cerr << "CUDA not available - cannot run GPU thermal expectation\n";
    return {{}, {}, {}, {}, {}};
}

std::tuple<std::vector<double>, std::vector<double>, std::vector<double>,
          std::vector<double>, std::vector<double>>
GPUEDWrapper::runGPUStaticCorrelation(void* gpu_op_handle,
                                     void* gpu_obs1_handle,
                                     void* gpu_obs2_handle,
                                     int N,
                                     int num_samples,
                                     int krylov_dim,
                                     double temp_min,
                                     double temp_max,
                                     int num_temp_bins,
                                     unsigned int random_seed) {
    std::cerr << "CUDA not available - cannot run GPU static correlation\n";
    return {{}, {}, {}, {}, {}};
}

std::map<double, std::tuple<std::vector<double>, std::vector<double>, std::vector<double>>>
GPUEDWrapper::runGPUDynamicalCorrelationMultiTemp(void* gpu_op_handle,
                                                 void* gpu_obs1_handle,
                                                 void* gpu_obs2_handle,
                                                 int N,
                                                 int num_samples,
                                                 int krylov_dim,
                                                 double omega_min,
                                                 double omega_max,
                                                 int num_omega_bins,
                                                 double broadening,
                                                 const std::vector<double>& temperatures,
                                                 unsigned int random_seed,
                                                 double ground_state_energy) {
    std::cerr << "CUDA not available - cannot run GPU multi-temperature dynamical correlation\n";
    return {};
}

bool GPUEDWrapper::createGPUOperatorFromCPU(const Operator& cpu_op,
                                           void** gpu_op_handle,
                                           int n_sites) {
    return false;
}

#endif // WITH_CUDA
