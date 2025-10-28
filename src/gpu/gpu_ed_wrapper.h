#ifndef GPU_ED_WRAPPER_H
#define GPU_ED_WRAPPER_H

// Forward declaration to avoid including construct_ham.h which has CUDA-incompatible code
class Operator;

#include <vector>
#include <complex>
#include <functional>
#include <string>
#include <tuple>

// Forward declarations only - don't include CUDA headers in this header
// They will be included in the .cu implementation file

/**
 * Wrapper class to integrate GPU operators with existing ED code
 * Provides a unified interface that works with both CPU and GPU implementations
 */
class GPUEDWrapper {
public:
    /**
     * Create GPU operator from CPU Operator
     * Transfers interaction data to GPU
     */
    static bool createGPUOperatorFromCPU(const Operator& cpu_op, 
                                        void** gpu_op_handle,
                                        int n_sites);
    
    /**
     * Run Lanczos algorithm on GPU
     * Compatible with existing Lanczos interface
     */
    static void runGPULanczos(void* gpu_op_handle,
                             int N, int max_iter, int num_eigs,
                             double tol,
                             std::vector<double>& eigenvalues,
                             std::string dir = "",
                             bool eigenvectors = false);
    
    /**
     * Run GPU Lanczos for Fixed Sz sector
     */
    static void runGPULanczosFixedSz(void* gpu_op_handle,
                                    int n_up,
                                    int max_iter, int num_eigs,
                                    double tol,
                                    std::vector<double>& eigenvalues,
                                    std::string dir = "",
                                    bool eigenvectors = false);
    
    /**
     * Matrix-vector product using GPU
     * y = H * x
     */
    static void gpuMatVec(void* gpu_op_handle,
                         const std::complex<double>* x,
                         std::complex<double>* y,
                         int N);
    
    /**
     * Create GPU operator directly from interaction lists
     */
    static void* createGPUOperatorDirect(int n_sites,
                                        const std::vector<std::tuple<int, int, char, char, double>>& interactions,
                                        const std::vector<std::tuple<int, char, double>>& single_site_ops);
    
    /**
     * Create GPU operator from InterAll.dat and Trans.dat files
     * Standard format used by the ED pipeline
     */
    static void* createGPUOperatorFromFiles(int n_sites,
                                           const std::string& interall_file,
                                           const std::string& trans_file);
    
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
    
    /**
     * Estimate memory requirements for GPU computation
     */
    static size_t estimateGPUMemory(int n_sites, bool fixed_sz = false, int n_up = 0);
    
    /**
     * Automatic decision: use GPU or CPU based on problem size and available resources
     */
    static bool shouldUseGPU(int n_sites, bool fixed_sz = false);
    
private:
    static int getGPUCount();
    static size_t getAvailableGPUMemory(int device = 0);
};

#endif // GPU_ED_WRAPPER_H
