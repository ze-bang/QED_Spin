#include "gpu_ed_wrapper.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>

#ifdef WITH_CUDA
#include <cuda_runtime.h>
#include "gpu_operator.cuh"
#include "gpu_lanczos.cuh"
#include "gpu_tpq.cuh"
#include "gpu_cg.cuh"

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
    
    return static_cast<void*>(gpu_op);
}

void* GPUEDWrapper::createGPUOperatorFromFiles(
    int n_sites,
    const std::string& interall_file,
    const std::string& trans_file) {
    
    std::vector<std::tuple<int, int, char, char, double>> interactions;
    std::vector<std::tuple<int, char, double>> single_site_ops;
    
    // Lambda to convert operator code to character
    auto op_to_char = [](int op_code) -> char {
        switch(op_code) {
            case 0: return '+';  // S+
            case 1: return '-';  // S-
            case 2: return 'z';  // Sz
            default: return '?';
        }
    };
    
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
        
        // Read interactions
        int lineCount = 0;
        while (std::getline(interall, line) && lineCount < numLines) {
            std::istringstream lineStream(line);
            int Op_i, indx_i, Op_j, indx_j;
            double E, F;
            
            if (!(lineStream >> Op_i >> indx_i >> Op_j >> indx_j >> E >> F)) continue;
            
            // Convert to our format
            // For Sz-Sz interactions (Op_i=2, Op_j=2), we need special handling
            if (Op_i == 2 && Op_j == 2) {
                // Sz*Sz interaction
                interactions.push_back(std::make_tuple(indx_i, indx_j, 'z', 'z', E));
            } else if (Op_i == 2) {
                // Sz * (S+ or S-)
                char op_j = op_to_char(Op_j);
                // For mixed terms, we need to handle Sx and Sy components
                if (Op_j == 0) {  // S+
                    // Sz*S+ = Sz*(Sx+iSy)
                    interactions.push_back(std::make_tuple(indx_i, indx_j, 'z', 'x', E * 0.5));
                    interactions.push_back(std::make_tuple(indx_i, indx_j, 'z', 'y', E * 0.5));
                } else if (Op_j == 1) {  // S-
                    // Sz*S- = Sz*(Sx-iSy)
                    interactions.push_back(std::make_tuple(indx_i, indx_j, 'z', 'x', E * 0.5));
                    interactions.push_back(std::make_tuple(indx_i, indx_j, 'z', 'y', -E * 0.5));
                }
            } else if (Op_j == 2) {
                // (S+ or S-) * Sz
                char op_i = op_to_char(Op_i);
                if (Op_i == 0) {  // S+
                    interactions.push_back(std::make_tuple(indx_i, indx_j, 'x', 'z', E * 0.5));
                    interactions.push_back(std::make_tuple(indx_i, indx_j, 'y', 'z', E * 0.5));
                } else if (Op_i == 1) {  // S-
                    interactions.push_back(std::make_tuple(indx_i, indx_j, 'x', 'z', E * 0.5));
                    interactions.push_back(std::make_tuple(indx_i, indx_j, 'y', 'z', -E * 0.5));
                }
            } else {
                // Both are S+ or S- operators
                // S+*S+ = (Sx+iSy)*(Sx+iSy) = Sx*Sx - Sy*Sy + i(Sx*Sy + Sy*Sx)
                // S+*S- = (Sx+iSy)*(Sx-iSy) = Sx*Sx + Sy*Sy
                // S-*S+ = (Sx-iSy)*(Sx+iSy) = Sx*Sx + Sy*Sy
                // S-*S- = (Sx-iSy)*(Sx-iSy) = Sx*Sx - Sy*Sy - i(Sx*Sy + Sy*Sx)
                
                if (Op_i == 0 && Op_j == 1) {  // S+ * S-
                    interactions.push_back(std::make_tuple(indx_i, indx_j, 'x', 'x', E));
                    interactions.push_back(std::make_tuple(indx_i, indx_j, 'y', 'y', E));
                } else if (Op_i == 1 && Op_j == 0) {  // S- * S+
                    interactions.push_back(std::make_tuple(indx_i, indx_j, 'x', 'x', E));
                    interactions.push_back(std::make_tuple(indx_i, indx_j, 'y', 'y', E));
                }
                // S+*S+ and S-*S- create complex terms that need careful handling
            }
            
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
        
        // Read single-site terms
        int lineCount = 0;
        while (std::getline(trans, line) && lineCount < numLines) {
            std::istringstream lineStream(line);
            int Op, indx;
            double E, F;
            
            if (!(lineStream >> Op >> indx >> E >> F)) continue;
            
            // Only process if coupling is non-zero
            if (std::abs(E) > 1e-12 || std::abs(F) > 1e-12) {
                char op_char = op_to_char(Op);
                
                // For S+ and S-, convert to Sx and Sy components
                if (Op == 0) {  // S+ = Sx + iSy
                    single_site_ops.push_back(std::make_tuple(indx, 'x', E));
                    single_site_ops.push_back(std::make_tuple(indx, 'y', E));
                } else if (Op == 1) {  // S- = Sx - iSy
                    single_site_ops.push_back(std::make_tuple(indx, 'x', E));
                    single_site_ops.push_back(std::make_tuple(indx, 'y', -E));
                } else if (Op == 2) {  // Sz
                    single_site_ops.push_back(std::make_tuple(indx, 'z', E));
                }
            }
            
            lineCount++;
        }
        trans.close();
    }
    
    std::cout << "Loaded " << interactions.size() << " interaction terms from " 
              << interall_file << "\n";
    std::cout << "Loaded " << single_site_ops.size() << " single-site terms from " 
              << trans_file << "\n";
    
    // Create GPU operator with loaded interactions
    return createGPUOperatorDirect(n_sites, interactions, single_site_ops);
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
    
    // Allocate GPU memory
    gpu_op->allocateGPUMemory(N);
    
    // Create GPU Lanczos solver
    GPULanczos lanczos(gpu_op, max_iter, tol);
    
    // Run Lanczos
    std::vector<std::vector<std::complex<double>>> eigvecs;
    lanczos.run(num_eigs, eigenvalues, eigvecs, eigenvectors);
    
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
    // This would require casting to GPUFixedSzOperator
    // Placeholder for now
    std::cout << "GPU Fixed Sz Lanczos not yet fully implemented\n";
}

void GPUEDWrapper::runGPUMicrocanonicalTPQ(void* gpu_op_handle,
                                           int N, int max_iter, int num_samples,
                                           int temp_interval,
                                           std::vector<double>& eigenvalues,
                                           std::string dir,
                                           double large_value) {
    if (!gpu_op_handle) {
        std::cerr << "Error: NULL GPU operator handle\n";
        return;
    }
    
    GPUOperator* gpu_op = static_cast<GPUOperator*>(gpu_op_handle);
    GPUTPQSolver tpq_solver(gpu_op, N);
    
    tpq_solver.runMicrocanonicalTPQ(max_iter, num_samples, temp_interval,
                                    eigenvalues, dir, large_value);
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

void GPUEDWrapper::runGPULOBPCG(void* gpu_op_handle,
                                int N, int num_eigenvalues, int max_iter,
                                double tol,
                                std::vector<double>& eigenvalues,
                                std::string dir,
                                bool compute_eigenvectors) {
    if (!gpu_op_handle) {
        std::cerr << "Error: NULL GPU operator handle\n";
        return;
    }
    
    GPUOperator* gpu_op = static_cast<GPUOperator*>(gpu_op_handle);
    GPUIterativeSolver solver(gpu_op, N);
    
    solver.runLOBPCG(num_eigenvalues, max_iter, tol,
                    eigenvalues, dir, compute_eigenvectors);
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
                                bool compute_eigenvectors) {}

bool GPUEDWrapper::createGPUOperatorFromCPU(const Operator& cpu_op,
                                           void** gpu_op_handle,
                                           int n_sites) {
    return false;
}

#endif // WITH_CUDA
