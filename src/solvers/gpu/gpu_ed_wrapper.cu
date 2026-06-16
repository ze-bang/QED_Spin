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

// ============================================================================
// GPUEDWrapper live surface (operator-collapse Phase 2b, Jun 2026).
//
// The bulk of the historical `GPUEDWrapper::runGPU*` forwarders were retired
// once the orchestrator's `CudaBackend` lane and the unified host operators'
// `bind_cuda()` device matvec (`CudaMatVecBackend`) became the production GPU
// path. What remains here is just the legacy GPU Lanczos entry point (reached
// from `tests/unit/test_cpu_gpu_equivalence.cpp` via `ed/gpu/gpu_solvers.h`)
// plus a few device-info utilities. The retired entry points and their
// bespoke GPU solver classes (GPUFixedSzOperator, GPUBlockLanczos,
// GPUKrylovSchur, GPUTPQSolver, gpuFullDiagonalization, GPUFTLMSolver via the
// `void*` wrappers) were:
//   * minimalist-architecture rev (May 2026): runGPUFTLMFixedSz,
//     runGPUDynamicalResponse[Thermal], runGPUDynamicalCorrelation[State,
//     StateCF], runGPUThermalExpectation.
//   * operator-collapse Phase 2b (Jun 2026): runGPULanczosFixedSz,
//     runGPUBlockLanczos[FixedSz], runGPUMicrocanonicalTPQ[FixedSz],
//     runGPUCanonicalTPQ[FixedSz], runGPUKrylovSchur[FixedSz], runGPUFTLM,
//     runGPUStaticCorrelation, runGPUDynamicalCorrelationMultiTemp,
//     runGPUFullDiag. GPUFTLMSolver is now constructed directly from
//     `src/cli/workflows.cpp` off `Operator`/`FixedSzOperator::bind_cuda`.
// ============================================================================

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

#else  // !WITH_CUDA

// The ~180-line `#else` block of "CUDA not available" stub
// implementations that used to live here (covering every public
// `GPUEDWrapper::*` entry point) was retired in the minimalist-
// architecture rev (May 2026): it was unreachable. `gpu_ed_wrapper.cu`
// is only added to the build inside `if(WITH_CUDA)`
// (`cmake/EDLibraries.cmake`), and every `GPUEDWrapper::*`
// callsite in the GPU kernel facades (e.g.
// `gpu_lanczos_kernel_facade.cu`) is itself gated by `#ifdef WITH_CUDA`.
// An #error guard catches accidental inclusion in a CPU-only build
// instead of silently providing no-op symbols.
#error "gpu_ed_wrapper.cu must only be compiled in WITH_CUDA builds (see cmake/EDLibraries.cmake)."

#endif // WITH_CUDA
