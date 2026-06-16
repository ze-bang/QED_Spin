#pragma once

// Forward declaration to avoid including construct_ham.h which has CUDA-incompatible code
class Operator;

#include <vector>
#include <complex>
#include <string>
#include <cstdint>

// Forward declarations only - don't include CUDA headers in this header
// They will be included in the .cu implementation file

/**
 * Thin GPU-solver surface kept after the operator-collapse cleanup.
 *
 * Most of the historical `runGPU*` forwarders were retired once the
 * orchestrator's `CudaBackend` lane plus the unified host operators'
 * `bind_cuda()` device matvec (`CudaMatVecBackend`) became the production GPU
 * path, and `GPUFTLMSolver` moved to being constructed directly from
 * `src/cli/workflows.cpp`. What remains is the legacy GPU Lanczos entry point
 * (reached from `tests/unit/test_cpu_gpu_equivalence.cpp` via
 * `ed/gpu/gpu_solvers.h`) plus device-info / cleanup utilities. See the
 * banner in `src/solvers/gpu/gpu_ed_wrapper.cu` for the full retirement list.
 */
class GPUEDWrapper {
public:
    /**
     * Run Lanczos algorithm on GPU
     * Compatible with existing Lanczos interface.
     * ``seed`` (defaulted 0) sets the GPU Lanczos starting-vector RNG;
     * 0 keeps the legacy deterministic seed (42), nonzero passes
     * through verbatim. Closes the audit S1 #21 silent CPU/GPU
     * divergence gap.
     */
    static void runGPULanczos(void* gpu_op_handle,
                             int N, int max_iter, int num_eigs,
                             double tol,
                             std::vector<double>& eigenvalues,
                             std::string dir = "",
                             bool eigenvectors = false,
                             unsigned long long seed = 0ULL);

    /**
     * Clean up GPU resources
     */
    static void destroyGPUOperator(void* gpu_op_handle);

    /**
     * Check if GPU is available and ready
     */
    static bool isGPUAvailable();

    /**
     * Get GPU device information
     */
    static void printGPUInfo();
};
