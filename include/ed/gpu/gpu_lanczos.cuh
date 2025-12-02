#ifndef GPU_LANCZOS_CUH
#define GPU_LANCZOS_CUH

#ifdef WITH_CUDA

#include <cuda_runtime.h>
#include <cuComplex.h>
#include <cublas_v2.h>
#include <vector>
#include <functional>
#include <complex>
#include <ed/gpu/gpu_operator.cuh>

/**
 * GPU-accelerated Lanczos algorithm for large-scale eigenvalue problems
 * Optimized for systems with up to 32 sites
 */
class GPULanczos {
public:
    GPULanczos(GPUOperator* op, int max_iter, double tolerance);
    ~GPULanczos();
    
    // Run Lanczos algorithm to find lowest eigenvalues
    void run(int num_eigenvalues, std::vector<double>& eigenvalues,
            std::vector<std::vector<std::complex<double>>>& eigenvectors,
            bool compute_vectors = false);
    
    // Run Lanczos with custom starting vector
    void runWithStartVector(const std::vector<std::complex<double>>& start_vec,
                           int num_eigenvalues,
                           std::vector<double>& eigenvalues,
                           std::vector<std::vector<std::complex<double>>>& eigenvectors,
                           bool compute_vectors = false);
    
    // Get performance statistics
    struct Stats {
        double total_time;
        double matvec_time;
        double ortho_time;
        int iterations;
        double convergence_error;
        uint64_t full_reorth_count;
        uint64_t selective_reorth_count;
        uint64_t total_reorth_ops;
    };
    
    Stats getStats() const { return stats_; }
    
private:
    GPUOperator* op_;
    int max_iter_;
    double tolerance_;
    int dimension_;
    
    // GPU memory
    cuDoubleComplex* d_v_current_;    // Current Lanczos vector
    cuDoubleComplex* d_v_prev_;       // Previous Lanczos vector
    cuDoubleComplex* d_w_;            // Work vector (H*v)
    cuDoubleComplex* d_temp_;         // Temporary vector
    
    // Lanczos vectors stored on GPU (if memory allows)
    cuDoubleComplex** d_lanczos_vectors_;
    int num_stored_vectors_;
    
    // Tridiagonal matrix elements (on host)
    std::vector<double> alpha_;  // Diagonal
    std::vector<double> beta_;   // Off-diagonal
    
    // cuBLAS handle
    cublasHandle_t cublas_handle_;
    
    // Statistics
    Stats stats_;
    
    // Helper functions
    void allocateMemory();
    void freeMemory();
    void initializeRandomVector(cuDoubleComplex* d_vec);
    
    // Adaptive selective reorthogonalization (Parlett-Simon)
    void orthogonalize(cuDoubleComplex* d_vec, int iter,
                      std::vector<std::vector<double>>& omega,
                      const std::vector<double>& alpha,
                      const std::vector<double>& beta,
                      double ortho_threshold);
    
    void normalizeVector(cuDoubleComplex* d_vec);
    double vectorNorm(const cuDoubleComplex* d_vec);
    void vectorCopy(const cuDoubleComplex* src, cuDoubleComplex* dst);
    void vectorScale(cuDoubleComplex* d_vec, double scale);
    void vectorAxpy(const cuDoubleComplex* d_x, cuDoubleComplex* d_y,
                   const cuDoubleComplex& alpha);
    std::complex<double> vectorDot(const cuDoubleComplex* d_x,
                                   const cuDoubleComplex* d_y);
    
    // Tridiagonal solver
    void solveTridiagonal(int m, int num_eigs,
                         std::vector<double>& eigenvalues,
                         std::vector<std::vector<double>>& eigenvectors);
    
    // Ritz vector computation
    void computeRitzVectors(const std::vector<std::vector<double>>& tridiag_eigenvecs,
                           int num_vecs,
                           std::vector<std::vector<std::complex<double>>>& eigenvectors);
};

/**
 * GPU-accelerated Block Lanczos for finding multiple eigenvalues with degeneracies
 */
class GPUBlockLanczos {
public:
    GPUBlockLanczos(GPUOperator* op, int max_iter, int block_size, double tolerance);
    ~GPUBlockLanczos();
    
    void run(int num_eigenvalues,
            std::vector<double>& eigenvalues,
            std::vector<std::vector<std::complex<double>>>& eigenvectors,
            bool compute_vectors = false);
    
private:
    GPUOperator* op_;
    int max_iter_;
    int block_size_;
    double tolerance_;
    int dimension_;
    
    cublasHandle_t cublas_handle_;
    
    // Block vectors on GPU
    cuDoubleComplex** d_block_vectors_;
    
    void allocateMemory();
    void freeMemory();
    void orthogonalizeBlock(cuDoubleComplex** block, int size);
    void qrFactorization(cuDoubleComplex** block, int size);
};

// Kernel declarations for Lanczos helpers
namespace GPULanczosKernels {

__global__ void initRandomVectorKernel(cuDoubleComplex* vec, int N, unsigned long long seed);

__global__ void vectorAddKernel(const cuDoubleComplex* x, const cuDoubleComplex* y,
                               cuDoubleComplex* result, int N);

__global__ void vectorSubKernel(const cuDoubleComplex* x, const cuDoubleComplex* y,
                               cuDoubleComplex* result, int N);

__global__ void vectorScaleKernel(cuDoubleComplex* vec, double scale, int N);

__global__ void vectorAxpyKernel(const cuDoubleComplex* x, cuDoubleComplex* y,
                                cuDoubleComplex alpha, int N);

} // namespace GPULanczosKernels

#endif // WITH_CUDA

#endif // GPU_LANCZOS_CUH
