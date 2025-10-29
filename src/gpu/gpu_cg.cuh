#ifndef GPU_CG_CUH
#define GPU_CG_CUH

#include <cuda_runtime.h>
#include <cuComplex.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <vector>
#include <complex>
#include <string>

// Forward declarations
class GPUOperator;

/**
 * @brief GPU-accelerated iterative eigensolvers
 * 
 * Implements Davidson and LOBPCG methods on GPU for finding
 * a few extreme eigenvalues and eigenvectors.
 */
class GPUIterativeSolver {
public:
    /**
     * @brief Constructor
     * @param gpu_op Pointer to GPU operator (Hamiltonian)
     * @param N Hilbert space dimension
     */
    GPUIterativeSolver(GPUOperator* gpu_op, int N);
    
    /**
     * @brief Destructor - frees GPU resources
     */
    ~GPUIterativeSolver();
    
    /**
     * @brief Run Davidson method
     * @param num_eigenvalues Number of eigenvalues to find
     * @param max_iter Maximum iterations
     * @param max_subspace Maximum subspace size
     * @param tol Convergence tolerance
     * @param eigenvalues Output eigenvalues
     * @param eigenvectors Output eigenvectors (optional)
     * @param dir Output directory for eigenvectors
     * @param compute_eigenvectors Whether to compute eigenvectors
     */
    void runDavidson(
        int num_eigenvalues,
        int max_iter,
        int max_subspace,
        double tol,
        std::vector<double>& eigenvalues,
        std::vector<std::vector<std::complex<double>>>& eigenvectors,
        const std::string& dir = "",
        bool compute_eigenvectors = false
    );
    
    /**
     * @brief Run LOBPCG method
     * @param num_eigenvalues Number of eigenvalues to find
     * @param max_iter Maximum iterations
     * @param tol Convergence tolerance
     * @param eigenvalues Output eigenvalues
     * @param dir Output directory for eigenvectors
     * @param compute_eigenvectors Whether to compute eigenvectors
     */
    void runLOBPCG(
        int num_eigenvalues,
        int max_iter,
        double tol,
        std::vector<double>& eigenvalues,
        const std::string& dir = "",
        bool compute_eigenvectors = false
    );
    
    /**
     * @brief Get statistics about GPU solver execution
     */
    struct SolverStats {
        double total_time;
        double matvec_time;
        double ortho_time;
        double subspace_time;
        int iterations;
        double throughput;
    };
    
    SolverStats getStats() const { return stats_; }

private:
    GPUOperator* gpu_op_;
    int N_;
    cublasHandle_t cublas_handle_;
    cusolverDnHandle_t cusolver_handle_;
    
    // Device vectors
    cuDoubleComplex* d_V_;          // Subspace vectors
    cuDoubleComplex* d_AV_;         // H*V
    cuDoubleComplex* d_work_;       // Working space
    cuDoubleComplex* d_subspace_H_; // Projected Hamiltonian
    double* d_subspace_eigs_;       // Subspace eigenvalues
    double* d_residual_norms_;      // Residual norms
    
    int* d_info_;
    int lwork_;
    
    // Statistics
    SolverStats stats_;
    
    // Helper functions
    void allocateMemory(int max_subspace);
    void freeMemory();
    void initializeRandomVectors(cuDoubleComplex* vectors, int num_vecs);
    void orthogonalize(cuDoubleComplex* vectors, int num_vecs, int vec_offset = 0);
    void gramSchmidt(cuDoubleComplex* vec, cuDoubleComplex* basis, int num_basis);
    void projectSubspaceHamiltonian(cuDoubleComplex* V, cuDoubleComplex* AV, 
                                   int subspace_dim, cuDoubleComplex* H_sub);
    void solveSubspaceProblem(cuDoubleComplex* H_sub, int subspace_dim,
                             double* eigs, cuDoubleComplex* evecs);
    void computeRitzVectors(cuDoubleComplex* V, cuDoubleComplex* evecs,
                           int subspace_dim, int num_eigs,
                           cuDoubleComplex* ritz_vecs);
    double computeResidualNorm(cuDoubleComplex* vec, cuDoubleComplex* Avec,
                              double eigenvalue, cuDoubleComplex* d_temp);
    void saveEigenvector(const std::string& filename, cuDoubleComplex* vec);
    void solveGeneralizedEigenvalueProblem(
        const std::vector<std::complex<double>>& h_hsub,
        const std::vector<std::complex<double>>& h_ovlp,
        int nsub, int block_size,
        std::vector<std::complex<double>>& eigenvector_coeffs,
        std::vector<double>& eigenvalues_out);
};

#endif // GPU_CG_CUH
