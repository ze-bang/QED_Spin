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
     * Run GPU Davidson for Fixed Sz sector
     */
    static void runGPUDavidsonFixedSz(void* gpu_op_handle,
                                     int n_up,
                                     int num_eigenvalues, int max_iter,
                                     int max_subspace, double tol,
                                     std::vector<double>& eigenvalues,
                                     std::string dir = "",
                                     bool compute_eigenvectors = false);
    
    /**
     * Run GPU TPQ for Fixed Sz sector (microcanonical)
     */
    static void runGPUMicrocanonicalTPQFixedSz(void* gpu_op_handle,
                                              int n_up,
                                              int max_iter, int num_samples,
                                              int temp_interval,
                                              std::vector<double>& eigenvalues,
                                              std::string dir = "",
                                              double large_value = 1e5);
    
    /**
     * Run GPU TPQ for Fixed Sz sector (canonical)
     */
    static void runGPUCanonicalTPQFixedSz(void* gpu_op_handle,
                                         int n_up,
                                         double beta_max, int num_samples,
                                         int temp_interval,
                                         std::vector<double>& energies,
                                         std::string dir = "",
                                         double delta_beta = 0.1,
                                         int taylor_order = 50);
    
    /**
     * Create GPU Fixed Sz operator directly from interaction lists
     */
    static void* createGPUFixedSzOperatorDirect(int n_sites, int n_up, float spin_l,
                                               const std::vector<std::tuple<int, int, char, char, double>>& interactions,
                                               const std::vector<std::tuple<int, char, double>>& single_site_ops);
    
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
     * Create GPU operator from CSR arrays (host-side)
     * row_ptr size = N+1, col_ind size = nnz, values size = nnz
     */
    static void* createGPUOperatorFromCSR(int n_sites,
                                         int N,
                                         const std::vector<int>& row_ptr,
                                         const std::vector<int>& col_ind,
                                         const std::vector<std::complex<double>>& values);
    
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
    
    /**
     * Run GPU-accelerated microcanonical TPQ
     */
    static void runGPUMicrocanonicalTPQ(void* gpu_op_handle,
                                        int N, int max_iter, int num_samples,
                                        int temp_interval,
                                        std::vector<double>& eigenvalues,
                                        std::string dir = "",
                                        double large_value = 1e5);
    
    /**
     * Run GPU-accelerated canonical TPQ
     */
    static void runGPUCanonicalTPQ(void* gpu_op_handle,
                                   int N, double beta_max, int num_samples,
                                   int temp_interval,
                                   std::vector<double>& energies,
                                   std::string dir = "",
                                   double delta_beta = 0.1,
                                   int taylor_order = 50);
    
    /**
     * Run GPU-accelerated Davidson method
     */
    static void runGPUDavidson(void* gpu_op_handle,
                              int N, int num_eigenvalues, int max_iter,
                              int max_subspace, double tol,
                              std::vector<double>& eigenvalues,
                              std::string dir = "",
                              bool compute_eigenvectors = false);
    
    /**
     * Run GPU-accelerated LOBPCG method
     * @deprecated This method now redirects to Davidson GPU for better stability
     */
    static void runGPULOBPCG(void* gpu_op_handle,
                            int N, int num_eigenvalues, int max_iter,
                            double tol,
                            std::vector<double>& eigenvalues,
                            std::string dir = "",
                            bool compute_eigenvectors = false);
    
    /**
     * Run GPU-accelerated LOBPCG method for Fixed Sz sector
     * @deprecated This method now redirects to Davidson GPU for better stability
     */
    static void runGPULOBPCGFixedSz(void* gpu_op_handle,
                                   int n_up,
                                   int num_eigenvalues, int max_iter,
                                   double tol,
                                   std::vector<double>& eigenvalues,
                                   std::string dir = "",
                                   bool compute_eigenvectors = false);
    
    /**
     * Run GPU-accelerated Finite Temperature Lanczos Method (FTLM)
     */
    static void runGPUFTLM(void* gpu_op_handle,
                          int N,
                          int krylov_dim,
                          int num_samples,
                          double temp_min,
                          double temp_max,
                          int num_temp_bins,
                          double tolerance,
                          std::string dir = "",
                          bool full_reorth = false,
                          int reorth_freq = 10,
                          unsigned int random_seed = 0);
    
    /**
     * Run GPU-accelerated FTLM for Fixed Sz sector
     */
    static void runGPUFTLMFixedSz(void* gpu_op_handle,
                                 int n_up,
                                 int krylov_dim,
                                 int num_samples,
                                 double temp_min,
                                 double temp_max,
                                 int num_temp_bins,
                                 double tolerance,
                                 std::string dir = "",
                                 bool full_reorth = false,
                                 int reorth_freq = 10,
                                 unsigned int random_seed = 0);
    
private:
    static int getGPUCount();
    static size_t getAvailableGPUMemory(int device = 0);
};

#endif // GPU_ED_WRAPPER_H
