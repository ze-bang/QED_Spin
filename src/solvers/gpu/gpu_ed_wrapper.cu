#include <ed/gpu/gpu_ed_wrapper.h>
#include <ed/gpu/gpu_solvers.h>  // ed::matvec::gpu::run_lanczos_eigenvalues_kernel_facade
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
#include <ed/gpu/gpu_ftlm.cuh>

// Forward declaration from gpu_full_diag.cu
extern void gpuFullDiagonalization(
    GPUOperator* gpu_op, int N, int num_eigs,
    std::vector<double>& eigenvalues,
    std::vector<std::vector<std::complex<double>>>& eigenvectors,
    bool compute_eigenvectors);

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

void* GPUEDWrapper::createGPUSymmetrizedOperator(
    int n_sites, float spin_l,
    int sector_dim,
    const std::vector<uint64_t>& orbit_elements,
    const std::vector<std::complex<double>>& orbit_coefficients,
    const std::vector<int>& orbit_offsets,
    const std::vector<double>& orbit_norms,
    int group_size,
    const std::string& interall_file,
    const std::string& trans_file)
{
    std::cout << "Creating GPU Symmetrized Operator for sector dim=" << sector_dim << "\n";
    
    GPUSymmetrizedOperator* gpu_op = new GPUSymmetrizedOperator(n_sites, spin_l);
    
    // Load Hamiltonian terms from files (same as regular GPU operator)
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
        
        for (int i = 0; i < 3; ++i) std::getline(interall, line);  // Skip separators
        
        int lineCount = 0;
        while (std::getline(interall, line) && lineCount < numLines) {
            std::istringstream lineStream(line);
            int Op_i, indx_i, Op_j, indx_j;
            double E, F;
            
            if (!(lineStream >> Op_i >> indx_i >> Op_j >> indx_j >> E >> F)) { lineCount++; continue; }
            if (std::abs(E) < 1e-12 && std::abs(F) < 1e-12) { lineCount++; continue; }
            if (Op_i < 0 || Op_i > 2 || Op_j < 0 || Op_j > 2) { lineCount++; continue; }
            
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
        std::getline(trans, line);
        std::getline(trans, line);
        
        std::istringstream iss(line);
        int numLines;
        std::string m;
        iss >> m >> numLines;
        
        for (int i = 0; i < 3; ++i) std::getline(trans, line);
        
        int lineCount = 0;
        while (std::getline(trans, line) && lineCount < numLines) {
            std::istringstream lineStream(line);
            int Op, indx;
            double E, F;
            
            if (!(lineStream >> Op >> indx >> E >> F)) { lineCount++; continue; }
            if (std::abs(E) > 1e-12 || std::abs(F) > 1e-12) {
                gpu_op->addOneBodyTerm(Op, indx, std::complex<double>(E, F));
                num_single_site++;
            }
            lineCount++;
        }
        trans.close();
    }
    
    std::cout << "  Loaded " << num_interactions << " interactions, "
              << num_single_site << " single-site terms\n";
    
    // Set sector orbit data (copies to GPU, builds hash table)
    gpu_op->setSectorData(sector_dim, orbit_elements, orbit_coefficients,
                          orbit_offsets, orbit_norms, group_size);
    
    // Allocate GPU memory for input/output vectors
    gpu_op->allocateGPUMemory(sector_dim);
    
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

void GPUEDWrapper::destroyGPUOperator(void* gpu_op_handle) {
    if (gpu_op_handle) {
        GPUOperator* gpu_op = static_cast<GPUOperator*>(gpu_op_handle);
        delete gpu_op;
    }
}

void GPUEDWrapper::runGPULanczos(void* gpu_op_handle,
                                int N, int max_iter, int num_eigs,
                                double tol,
                                std::vector<double>& eigenvalues,
                                std::string dir,
                                bool eigenvectors,
                                unsigned long long seed) {
    if (!gpu_op_handle) {
        std::cerr << "Error: NULL GPU operator handle\n";
        return;
    }

    GPUOperator* gpu_op = static_cast<GPUOperator*>(gpu_op_handle);

    // -----------------------------------------------------------------
    // Both branches (eigvals only / eigvals+eigvecs) now route through
    // the unified `lanczos_kernel<CudaBackend>` facade (May 2026).
    // The legacy `GPULanczos::run` is kept ONLY as a defensive fallback
    // if the facade throws -- e.g. if the basis won't fit in device
    // memory and the kernel's `keep_basis = true` assertion fires.
    // The fallback path uses the legacy class's windowed-reorth +
    // on-disk basis-spill regime, which the unified kernel doesn't
    // implement yet.
    // -----------------------------------------------------------------
    std::vector<std::vector<std::complex<double>>> eigvecs;
    bool used_facade = false;
    try {
        if (eigenvectors) {
            ed::matvec::gpu::run_lanczos_eigenpairs_kernel_facade(
                *gpu_op, N, max_iter, num_eigs, tol, seed,
                eigenvalues, eigvecs);
        } else {
            ed::matvec::gpu::run_lanczos_eigenvalues_kernel_facade(
                *gpu_op, N, max_iter, num_eigs, tol, seed, eigenvalues);
        }
        used_facade = true;
    } catch (const std::exception& e) {
        std::cerr << "GPU Lanczos kernel-facade path failed ("
                  << e.what()
                  << "); falling back to legacy GPULanczos.\n";
        GPULanczos lanczos_legacy(gpu_op, max_iter, tol);
        lanczos_legacy.setSeed(seed);
        lanczos_legacy.run(num_eigs, eigenvalues, eigvecs,
                           /*compute_vectors=*/eigenvectors);
    }

    if (!dir.empty()) {
        if (eigenvectors && !eigvecs.empty()) {
            HDF5IO::saveDiagonalizationResults(dir, eigenvalues, eigvecs,
                                               "GPU_LANCZOS");
            std::cout << "GPU Lanczos: Saved " << eigenvalues.size()
                      << " eigenvalues and " << eigvecs.size()
                      << " eigenvectors to " << dir << "/ed_results.h5"
                      << std::endl;
        } else {
            try {
                std::string hdf5_file = HDF5IO::createOrOpenFile(dir);
                HDF5IO::saveEigenvalues(hdf5_file, eigenvalues);
                std::cout << "GPU Lanczos: Saved " << eigenvalues.size()
                          << " eigenvalues to " << hdf5_file << std::endl;
            } catch (const std::exception& e) {
                std::cerr << "Warning: Failed to save eigenvalues to HDF5: "
                          << e.what() << std::endl;
            }
        }
    }

    if (used_facade) {
        // Facade already printed its own run summary; nothing more to
        // surface to the user beyond the matvec throughput.
        auto op_stats = gpu_op->getStats();
        std::cout << "  GPU SpMV throughput: " << op_stats.throughput
                  << " GFLOPS\n";
    }
}

void GPUEDWrapper::runGPULanczosFixedSz(void* gpu_op_handle,
                                       int n_up,
                                       int max_iter, int num_eigs,
                                       double tol,
                                       std::vector<double>& eigenvalues,
                                       std::string dir,
                                       bool eigenvectors,
                                       unsigned long long seed) {
    if (!gpu_op_handle) {
        std::cerr << "Error: NULL GPU operator handle\n";
        return;
    }

    GPUFixedSzOperator* gpu_op = static_cast<GPUFixedSzOperator*>(gpu_op_handle);
    const int fixed_sz_dim = gpu_op->getFixedSzDimension();

    std::cout << "Running GPU Lanczos for fixed Sz sector (N_up=" << n_up
              << ", dim=" << fixed_sz_dim << ")\n";

    gpu_op->allocateGPUMemory(fixed_sz_dim);

    // -----------------------------------------------------------------
    // Unified facade for both branches (May 2026 day 7). The fixed-Sz
    // path is identical to the full-Hilbert one apart from the
    // dimension argument: `GPUFixedSzOperator` derives from
    // `GPUOperator`, so the matvec callback's polymorphic `matVecGPU`
    // dispatches into the fixed-Sz override automatically.
    // -----------------------------------------------------------------
    std::vector<std::vector<std::complex<double>>> eigvecs;
    bool used_facade = false;
    try {
        if (eigenvectors) {
            ed::matvec::gpu::run_lanczos_eigenpairs_kernel_facade(
                *gpu_op, fixed_sz_dim, max_iter, num_eigs, tol, seed,
                eigenvalues, eigvecs);
        } else {
            ed::matvec::gpu::run_lanczos_eigenvalues_kernel_facade(
                *gpu_op, fixed_sz_dim, max_iter, num_eigs, tol, seed,
                eigenvalues);
        }
        used_facade = true;
    } catch (const std::exception& e) {
        std::cerr << "GPU Lanczos (fixed-Sz) kernel-facade path failed ("
                  << e.what()
                  << "); falling back to legacy GPULanczos.\n";
        GPULanczos lanczos_legacy(gpu_op, max_iter, tol);
        lanczos_legacy.setSeed(seed);
        lanczos_legacy.run(num_eigs, eigenvalues, eigvecs,
                           /*compute_vectors=*/eigenvectors);
    }

    if (!dir.empty()) {
        if (eigenvectors && !eigvecs.empty()) {
            HDF5IO::saveDiagonalizationResults(dir, eigenvalues, eigvecs,
                                               "GPU_LANCZOS_FIXED_SZ");
            std::cout << "GPU Lanczos Fixed Sz: Saved " << eigenvalues.size()
                      << " eigenvalues and " << eigvecs.size()
                      << " eigenvectors to " << dir << "/ed_results.h5"
                      << std::endl;
        } else {
            try {
                std::string hdf5_file = HDF5IO::createOrOpenFile(dir);
                HDF5IO::saveEigenvalues(hdf5_file, eigenvalues);
                std::cout << "GPU Lanczos Fixed Sz: Saved " << eigenvalues.size()
                          << " eigenvalues to " << hdf5_file << std::endl;
            } catch (const std::exception& e) {
                std::cerr << "Warning: Failed to save eigenvalues to HDF5: "
                          << e.what() << std::endl;
            }
        }
    }

    if (used_facade) {
        auto op_stats = gpu_op->getStats();
        std::cout << "  GPU SpMV throughput: " << op_stats.throughput
                  << " GFLOPS\n";
    }
}

void GPUEDWrapper::runGPUBlockLanczos(void* gpu_op_handle,
                                     int N, int max_iter, int num_eigs,
                                     int block_size,
                                     double tol,
                                     std::vector<double>& eigenvalues,
                                     std::string dir,
                                     bool eigenvectors) {
    if (!gpu_op_handle) {
        std::cerr << "Error: NULL GPU operator handle\n";
        return;
    }
    
    GPUOperator* gpu_op = static_cast<GPUOperator*>(gpu_op_handle);
    
    // Create GPU Block Lanczos solver
    GPUBlockLanczos block_lanczos(gpu_op, max_iter, block_size, tol);
    
    // Run Block Lanczos
    std::vector<std::vector<std::complex<double>>> eigvecs;
    block_lanczos.run(num_eigs, eigenvalues, eigvecs, eigenvectors);
    
    // Save results to HDF5
    if (!dir.empty()) {
        if (eigenvectors && !eigvecs.empty()) {
            HDF5IO::saveDiagonalizationResults(dir, eigenvalues, eigvecs, "GPU_BLOCK_LANCZOS");
            std::cout << "GPU Block Lanczos: Saved " << eigenvalues.size() << " eigenvalues and " 
                      << eigvecs.size() << " eigenvectors to " << dir << "/ed_results.h5" << std::endl;
        } else {
            try {
                std::string hdf5_file = HDF5IO::createOrOpenFile(dir);
                HDF5IO::saveEigenvalues(hdf5_file, eigenvalues);
                std::cout << "GPU Block Lanczos: Saved " << eigenvalues.size() << " eigenvalues to " << hdf5_file << std::endl;
            } catch (const std::exception& e) {
                std::cerr << "Warning: Failed to save eigenvalues to HDF5: " << e.what() << std::endl;
            }
        }
    }
    
    // Print statistics
    auto stats = block_lanczos.getStats();
    std::cout << "\nGPU Block Lanczos Statistics:\n";
    std::cout << "  Total time: " << stats.total_time << " s\n";
    std::cout << "  MatVec time: " << stats.matvec_time << " s (" << stats.total_matvecs << " matvecs)\n";
    std::cout << "  Ortho time: " << stats.ortho_time << " s\n";
    std::cout << "  QR time: " << stats.qr_time << " s\n";
    std::cout << "  Diag time: " << stats.diag_time << " s\n";
    std::cout << "  Block iterations: " << stats.block_iterations << "\n";
    std::cout << "  Reorthogonalizations: " << stats.reorth_count << "\n";
    
    auto op_stats = gpu_op->getStats();
    std::cout << "  Throughput: " << op_stats.throughput << " GFLOPS\n";
}

void GPUEDWrapper::runGPUBlockLanczosFixedSz(void* gpu_op_handle,
                                            int n_up,
                                            int max_iter, int num_eigs,
                                            int block_size,
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
    
    std::cout << "Running GPU Block Lanczos for fixed Sz sector (N_up=" << n_up 
              << ", dim=" << fixed_sz_dim << ", block_size=" << block_size << ")\n";
    
    // Allocate GPU memory for vectors
    gpu_op->allocateGPUMemory(fixed_sz_dim);
    
    // Create GPU Block Lanczos solver with fixed Sz operator
    GPUBlockLanczos block_lanczos(gpu_op, max_iter, block_size, tol);
    
    // Run Block Lanczos
    std::vector<std::vector<std::complex<double>>> eigvecs;
    block_lanczos.run(num_eigs, eigenvalues, eigvecs, eigenvectors);
    
    // Save results to HDF5
    if (!dir.empty()) {
        if (eigenvectors && !eigvecs.empty()) {
            HDF5IO::saveDiagonalizationResults(dir, eigenvalues, eigvecs, "GPU_BLOCK_LANCZOS_FIXED_SZ");
            std::cout << "GPU Block Lanczos Fixed Sz: Saved " << eigenvalues.size() << " eigenvalues and " 
                      << eigvecs.size() << " eigenvectors to " << dir << "/ed_results.h5" << std::endl;
        } else {
            try {
                std::string hdf5_file = HDF5IO::createOrOpenFile(dir);
                HDF5IO::saveEigenvalues(hdf5_file, eigenvalues);
                std::cout << "GPU Block Lanczos Fixed Sz: Saved " << eigenvalues.size() << " eigenvalues to " << hdf5_file << std::endl;
            } catch (const std::exception& e) {
                std::cerr << "Warning: Failed to save eigenvalues to HDF5: " << e.what() << std::endl;
            }
        }
    }
    
    // Print statistics
    auto stats = block_lanczos.getStats();
    std::cout << "\nGPU Block Lanczos Fixed Sz Statistics:\n";
    std::cout << "  Total time: " << stats.total_time << " s\n";
    std::cout << "  MatVec time: " << stats.matvec_time << " s (" << stats.total_matvecs << " matvecs)\n";
    std::cout << "  Ortho time: " << stats.ortho_time << " s\n";
    std::cout << "  QR time: " << stats.qr_time << " s\n";
    std::cout << "  Diag time: " << stats.diag_time << " s\n";
    std::cout << "  Block iterations: " << stats.block_iterations << "\n";
    
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
                                           double continue_beta,
                                           bool save_thermal_states,
                                           double target_beta,
                                           int num_measure_points,
                                           double measure_beta_min,
                                           double measure_beta_max) {
    if (!gpu_op_handle) {
        std::cerr << "Error: NULL GPU operator handle\n";
        return;
    }
    
    GPUOperator* gpu_op = static_cast<GPUOperator*>(gpu_op_handle);
    GPUTPQSolver tpq_solver(gpu_op, N);
    
    tpq_solver.runMicrocanonicalTPQ(max_iter, num_samples, temp_interval,
                                    eigenvalues, dir, large_value, nullptr,
                                    continue_quenching, continue_sample, continue_beta,
                                    save_thermal_states, target_beta,
                                    num_measure_points, measure_beta_min, measure_beta_max);
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
                                                 double continue_beta,
                                                 bool save_thermal_states,
                                                 double target_beta,
                                                 int num_measure_points,
                                                 double measure_beta_min,
                                                 double measure_beta_max) {
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
                                    continue_quenching, continue_sample, continue_beta,
                                    save_thermal_states, target_beta,
                                    num_measure_points, measure_beta_min, measure_beta_max);
    
    std::cout << "\nGPU Microcanonical TPQ Fixed Sz complete\n";
}

void GPUEDWrapper::runGPUCanonicalTPQ(void* gpu_op_handle,
                                      int N, double beta_max, int num_samples,
                                      int temp_interval,
                                      std::vector<double>& energies,
                                      std::string dir,
                                      double delta_beta,
                                      int taylor_order,
                                      int num_measure_points,
                                      double measure_beta_min,
                                      double measure_beta_max) {
    if (!gpu_op_handle) {
        std::cerr << "Error: NULL GPU operator handle\n";
        return;
    }
    
    GPUOperator* gpu_op = static_cast<GPUOperator*>(gpu_op_handle);
    GPUTPQSolver tpq_solver(gpu_op, N);
    
    tpq_solver.runCanonicalTPQ(beta_max, num_samples, temp_interval,
                               energies, dir, delta_beta, taylor_order, nullptr,
                               num_measure_points, measure_beta_min, measure_beta_max);
}

void GPUEDWrapper::runGPUCanonicalTPQFixedSz(void* gpu_op_handle,
                                            int n_up,
                                            double beta_max, int num_samples,
                                            int temp_interval,
                                            std::vector<double>& energies,
                                            std::string dir,
                                            double delta_beta,
                                            int taylor_order,
                                            int num_measure_points,
                                            double measure_beta_min,
                                            double measure_beta_max) {
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
                               energies, dir, delta_beta, taylor_order, gpu_op,
                               num_measure_points, measure_beta_min, measure_beta_max);
    
    std::cout << "\nGPU Canonical TPQ Fixed Sz complete\n";
}

void GPUEDWrapper::runGPUKrylovSchur(void* gpu_op_handle,
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
    
    std::cout << "\n========================================\n";
    std::cout << "GPU Krylov-Schur Algorithm\n";
    std::cout << "========================================\n";
    std::cout << "  Dimension: " << N << "\n";
    std::cout << "  Target eigenvalues: " << num_eigenvalues << "\n";
    std::cout << "  Max Krylov size: " << max_iter << "\n";
    std::cout << "  Tolerance: " << tol << "\n\n";
    
    // Create and run GPU Krylov-Schur solver
    GPUKrylovSchur solver(gpu_op, max_iter, tol);
    
    std::vector<std::vector<std::complex<double>>> eigenvectors;
    solver.run(num_eigenvalues, eigenvalues, eigenvectors, compute_eigenvectors);
    
    // Save results if directory specified
    if (!dir.empty() && !eigenvalues.empty()) {
        try {
            HDF5IO::saveDiagonalizationResults(dir, eigenvalues, eigenvectors, "GPU-Krylov-Schur");
            std::cout << "Saved results to " << dir << "/ed_results.h5\n";
        } catch (const std::exception& e) {
            std::cerr << "Warning: Failed to save results: " << e.what() << "\n";
        }
    }
    
    std::cout << "\nGPU Krylov-Schur complete\n";
}

void GPUEDWrapper::runGPUKrylovSchurFixedSz(void* gpu_op_handle,
                                           int n_up,
                                           int num_eigenvalues, int max_iter,
                                           double tol,
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
    
    std::cout << "\n========================================\n";
    std::cout << "GPU Krylov-Schur (Fixed Sz)\n";
    std::cout << "========================================\n";
    std::cout << "  N_up: " << n_up << "\n";
    std::cout << "  Dimension: " << fixed_sz_dim << "\n";
    std::cout << "  Target eigenvalues: " << num_eigenvalues << "\n";
    std::cout << "  Max Krylov size: " << max_iter << "\n";
    std::cout << "  Tolerance: " << tol << "\n\n";
    
    // Allocate GPU memory for vectors
    gpu_op->allocateGPUMemory(fixed_sz_dim);
    
    // Create and run GPU Krylov-Schur solver
    GPUKrylovSchur solver(gpu_op, max_iter, tol);
    
    std::vector<std::vector<std::complex<double>>> eigenvectors;
    solver.run(num_eigenvalues, eigenvalues, eigenvectors, compute_eigenvectors);
    
    // Save results if directory specified
    if (!dir.empty() && !eigenvalues.empty()) {
        try {
            HDF5IO::saveDiagonalizationResults(dir, eigenvalues, eigenvectors, "GPU-Krylov-Schur-FixedSz");
            std::cout << "Saved results to " << dir << "/ed_results.h5\n";
        } catch (const std::exception& e) {
            std::cerr << "Warning: Failed to save results: " << e.what() << "\n";
        }
    }
    
    std::cout << "\nGPU Krylov-Schur Fixed Sz complete\n";
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

// `runGPUFTLMFixedSz`, `runGPUDynamicalResponseThermal`, `runGPUDynamicalCorrelation`,
// `runGPUDynamicalCorrelationStateCF` and `runGPUThermalExpectation` were retired in
// the minimalist-architecture rev (May 2026): the live GPU DSSF entry points are
// `runGPUDynamicalCorrelationMultiTemp` (cross-correlator, thermal, multi-T) and
// `runGPUStaticCorrelation` (static thermal correlator), reached from `workflows.cpp`.

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
    
    // Check for large systems that may cause memory issues
    bool large_system = (static_cast<uint64_t>(N) > (1ULL << 24));
    if (large_system) {
        std::cerr << "\n*** WARNING: LARGE SYSTEM DETECTED ***" << std::endl;
        std::cerr << "N = " << N << " states (>" << (1 << 24) << ")" << std::endl;
        std::cerr << "State vector size: " << (N * 16.0 / (1024*1024*1024)) << " GB" << std::endl;
        std::cerr << "This method requires basis storage for eigenvector reconstruction." << std::endl;
        std::cerr << "GPU memory may be insufficient. Consider:" << std::endl;
        std::cerr << "  - Using 'spectral' method with O1=O2 (uses continued fraction)" << std::endl;
        std::cerr << "  - Running on CPU with memory optimization" << std::endl;
        std::cerr << "  - Reducing krylov_dim (currently: " << krylov_dim << ")" << std::endl;
        std::cerr << "Continuing with potentially limited basis storage...\n" << std::endl;
    }
    
    if (!gpu_op_handle || !gpu_obs1_handle || !gpu_obs2_handle) {
        std::cerr << "Error: GPU operator handles are null\n";
        return {};
    }
    
    GPUOperator* gpu_op = static_cast<GPUOperator*>(gpu_op_handle);
    GPUOperator* gpu_obs1 = static_cast<GPUOperator*>(gpu_obs1_handle);
    GPUOperator* gpu_obs2 = static_cast<GPUOperator*>(gpu_obs2_handle);
    
    // Create GPU FTLM solver (constructor will auto-skip basis pool for large systems)
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

void GPUEDWrapper::runGPUFullDiag(void* gpu_op_handle,
                                  int N, int num_eigenvalues,
                                  std::vector<double>& eigenvalues,
                                  std::string dir,
                                  bool compute_eigenvectors) {
    if (!gpu_op_handle) {
        std::cerr << "Error: NULL GPU operator handle\n";
        return;
    }

    GPUOperator* gpu_op = static_cast<GPUOperator*>(gpu_op_handle);

    std::vector<std::vector<std::complex<double>>> eigvecs;
    gpuFullDiagonalization(gpu_op, N,
                           num_eigenvalues > 0 ? num_eigenvalues : N,
                           eigenvalues, eigvecs, compute_eigenvectors);

    // Save results to HDF5
    if (!dir.empty()) {
        if (compute_eigenvectors && !eigvecs.empty()) {
            HDF5IO::saveDiagonalizationResults(dir, eigenvalues, eigvecs, "GPU_FULL");
            std::cout << "GPU Full Diag: Saved " << eigenvalues.size() << " eigenvalues and "
                      << eigvecs.size() << " eigenvectors to " << dir << "/ed_results.h5" << std::endl;
        } else {
            try {
                std::string hdf5_file = HDF5IO::createOrOpenFile(dir);
                HDF5IO::saveEigenvalues(hdf5_file, eigenvalues);
                std::cout << "GPU Full Diag: Saved " << eigenvalues.size()
                          << " eigenvalues to " << hdf5_file << std::endl;
            } catch (const std::exception& e) {
                std::cerr << "Warning: Failed to save eigenvalues to HDF5: " << e.what() << std::endl;
            }
        }
    }
}

#else  // !WITH_CUDA

// The ~180-line `#else` block of "CUDA not available" stub
// implementations that used to live here (covering every public
// `GPUEDWrapper::*` entry point) was retired in the minimalist-
// architecture rev (May 2026): it was unreachable. `gpu_ed_wrapper.cu`
// is only added to the build inside `if(WITH_CUDA)`
// (`cmake/EDLibraries.cmake:566`), and every `GPUEDWrapper::*`
// callsite in the GPU kernel facades (e.g.
// `gpu_lanczos_kernel_facade.cu`, `gpu_block_lanczos_kernel.cu`) is
// itself gated by `#ifdef WITH_CUDA`. An #error guard catches accidental
// inclusion in a CPU-only build instead of silently providing
// no-op symbols.
#error "gpu_ed_wrapper.cu must only be compiled in WITH_CUDA builds (see cmake/EDLibraries.cmake)."

#endif // WITH_CUDA
