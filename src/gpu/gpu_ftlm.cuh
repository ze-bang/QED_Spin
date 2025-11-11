#ifndef GPU_FTLM_CUH
#define GPU_FTLM_CUH

#ifdef WITH_CUDA

#include <cuda_runtime.h>
#include <cuComplex.h>
#include <cublas_v2.h>
#include <vector>
#include <complex>
#include <functional>
#include "gpu_operator.cuh"
#include "../core/thermal_types.h"

// Forward declare CPU functions for averaging (defined in ftlm.cpp)
void average_ftlm_samples(const std::vector<ThermodynamicData>& sample_data, FTLMResults& results);
void save_ftlm_results(const FTLMResults& results, const std::string& filename);

/**
 * @brief GPU-accelerated Finite Temperature Lanczos Method (FTLM)
 * 
 * Implements FTLM on GPU for computing finite-temperature thermodynamic properties.
 * Uses Lanczos method to build Krylov subspace and extract thermal properties from
 * the microcanonical spectrum approximated by Ritz values.
 */
class GPUFTLMSolver {
public:
    /**
     * @brief Constructor
     * @param op GPU operator (Hamiltonian)
     * @param N Hilbert space dimension
     * @param krylov_dim Maximum Krylov subspace dimension
     * @param tolerance Convergence tolerance for Lanczos
     */
    GPUFTLMSolver(GPUOperator* op, int N, int krylov_dim = 100, double tolerance = 1e-10);
    
    /**
     * @brief Destructor - cleanup GPU memory
     */
    ~GPUFTLMSolver();
    
    /**
     * @brief Run FTLM calculation
     * @param num_samples Number of random initial states to sample
     * @param temp_min Minimum temperature
     * @param temp_max Maximum temperature
     * @param num_temp_bins Number of temperature points
     * @param output_dir Output directory for results
     * @param full_reorth Use full reorthogonalization (slower but more accurate)
     * @param reorth_freq Selective reorthogonalization frequency (0 = none)
     * @param random_seed Random seed (0 = use system clock)
     * @return FTLMResults structure with thermodynamic data
     */
    FTLMResults run(int num_samples,
                   double temp_min, double temp_max, int num_temp_bins,
                   const std::string& output_dir = "",
                   bool full_reorth = false,
                   int reorth_freq = 10,
                   unsigned int random_seed = 0);
    
    /**
     * @brief Run FTLM for a single random sample
     * @param seed Random seed for initial state
     * @param alpha Output: diagonal elements of tridiagonal matrix
     * @param beta Output: off-diagonal elements of tridiagonal matrix
     * @return Number of Lanczos iterations performed
     */
    int runSingleSample(unsigned int seed,
                       std::vector<double>& alpha,
                       std::vector<double>& beta);
    
    /**
     * @brief Compute thermodynamics from Lanczos tridiagonal matrix
     * @param alpha Diagonal elements
     * @param beta Off-diagonal elements
     * @param temperatures Temperature grid
     * @return ThermodynamicData structure
     */
    ThermodynamicData computeThermodynamics(
        const std::vector<double>& alpha,
        const std::vector<double>& beta,
        const std::vector<double>& temperatures);
    
    /**
     * @brief Get performance statistics
     */
    struct Stats {
        double total_time;
        double lanczos_time;
        double diag_time;
        double thermo_time;
        int total_iterations;
        int num_samples_completed;
    };
    
    Stats getStats() const { return stats_; }
    
private:
    GPUOperator* op_;
    int N_;  // Hilbert space dimension
    int krylov_dim_;
    double tolerance_;
    
    // GPU memory for Lanczos vectors
    cuDoubleComplex* d_v_current_;   // Current Lanczos vector
    cuDoubleComplex* d_v_prev_;      // Previous Lanczos vector
    cuDoubleComplex* d_w_;           // Work vector (H*v)
    cuDoubleComplex* d_temp_;        // Temporary vector
    
    // Stored Lanczos vectors for reorthogonalization (if needed)
    cuDoubleComplex** d_lanczos_basis_;
    int num_stored_vectors_;
    bool store_basis_;
    
    // cuBLAS handle
    cublasHandle_t cublas_handle_;
    
    // Performance statistics
    Stats stats_;
    
    // Memory management
    bool gpu_memory_allocated_;
    void allocateMemory();
    void freeMemory();
    
    // Lanczos iteration helpers
    void initializeRandomVector(cuDoubleComplex* d_vec, unsigned int seed);
    void normalizeVector(cuDoubleComplex* d_vec);
    double vectorNorm(const cuDoubleComplex* d_vec);
    void vectorCopy(const cuDoubleComplex* src, cuDoubleComplex* dst);
    void vectorScale(cuDoubleComplex* d_vec, double scale);
    void vectorAxpy(const cuDoubleComplex* d_x, cuDoubleComplex* d_y,
                   const cuDoubleComplex& alpha);
    std::complex<double> vectorDot(const cuDoubleComplex* d_x,
                                   const cuDoubleComplex* d_y);
    
    // Orthogonalization
    void orthogonalizeAgainstBasis(cuDoubleComplex* d_vec, int num_basis_vecs);
    void gramSchmidt(cuDoubleComplex* d_vec, int iter);
    
    // Build Lanczos tridiagonal matrix
    int buildLanczosTridiagonal(unsigned int seed,
                               bool full_reorth,
                               int reorth_freq,
                               std::vector<double>& alpha,
                               std::vector<double>& beta);
    
    // Diagonalize tridiagonal matrix (on CPU)
    void diagonalizeTridiagonal(const std::vector<double>& alpha,
                               const std::vector<double>& beta,
                               std::vector<double>& ritz_values,
                               std::vector<double>& weights);
};

/**
 * @brief GPU kernels for FTLM operations
 */
namespace GPUFTLMKernels {

/**
 * @brief Initialize random vector on GPU
 */
__global__ void initRandomVectorKernel(cuDoubleComplex* vec, int N, 
                                      unsigned long long seed);

/**
 * @brief Normalize vector kernel
 */
__global__ void normalizeKernel(cuDoubleComplex* vec, int N, double norm);

/**
 * @brief Vector AXPY: y = alpha*x + y
 */
__global__ void axpyKernel(const cuDoubleComplex* x, cuDoubleComplex* y,
                          cuDoubleComplex alpha, int N);

/**
 * @brief Vector scaling: x = alpha*x
 */
__global__ void scaleKernel(cuDoubleComplex* x, double alpha, int N);

} // namespace GPUFTLMKernels

#endif // WITH_CUDA

#endif // GPU_FTLM_CUH
