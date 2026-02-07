#ifndef GPU_LANCZOS_CUH
#define GPU_LANCZOS_CUH

#ifdef WITH_CUDA

#include <cuda_runtime.h>
#include <cuComplex.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
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
 * 
 * Optimized for:
 * - Degenerate eigenvalue problems where standard Lanczos may miss multiplicities
 * - Better parallelism through block operations (BLAS-3 efficiency)
 * - Improved convergence for clustered eigenvalues
 * - Efficient GPU memory access patterns via column-major block storage
 * 
 * Architecture follows industry-standard patterns:
 * - Uses cuBLAS batched/strided GEMM for block operations
 * - cuSOLVER for QR factorization and band matrix diagonalization
 * - Streaming for overlapped computation and transfer
 * - Selective reorthogonalization with batched inner products
 */
class GPUBlockLanczos {
public:
    /**
     * @brief Construct GPU Block Lanczos solver
     * @param op Pointer to GPU operator (Hamiltonian)
     * @param max_iter Maximum number of block iterations
     * @param block_size Number of vectors in each block (typically 4-16)
     * @param tolerance Convergence tolerance for eigenvalues
     */
    GPUBlockLanczos(GPUOperator* op, int max_iter, int block_size, double tolerance);
    ~GPUBlockLanczos();
    
    /**
     * @brief Run Block Lanczos to find lowest eigenvalues
     * @param num_eigenvalues Number of eigenvalues to compute
     * @param eigenvalues Output: computed eigenvalues
     * @param eigenvectors Output: computed eigenvectors (if requested)
     * @param compute_vectors Whether to compute eigenvectors
     */
    void run(int num_eigenvalues,
            std::vector<double>& eigenvalues,
            std::vector<std::vector<std::complex<double>>>& eigenvectors,
            bool compute_vectors = false);
    
    /**
     * @brief Run with custom starting block
     * @param start_block Initial block of vectors (column-major, dimension × block_size)
     */
    void runWithStartBlock(const std::vector<std::complex<double>>& start_block,
                          int num_eigenvalues,
                          std::vector<double>& eigenvalues,
                          std::vector<std::vector<std::complex<double>>>& eigenvectors,
                          bool compute_vectors = false);
    
    // Performance statistics
    struct Stats {
        double total_time;
        double matvec_time;
        double ortho_time;
        double qr_time;
        double diag_time;
        int block_iterations;
        int total_matvecs;
        double convergence_error;
        uint64_t reorth_count;
        size_t memory_used;
    };
    
    Stats getStats() const { return stats_; }
    
    /**
     * @brief Set reorthogonalization strategy
     * @param strategy 0=none, 1=local (last few blocks), 2=periodic, 3=full
     */
    void setReorthStrategy(int strategy) { reorth_strategy_ = strategy; }
    
private:
    GPUOperator* op_;
    int max_iter_;
    int block_size_;
    double tolerance_;
    int dimension_;
    int reorth_strategy_;  // 0=none, 1=local, 2=periodic, 3=full
    
    // CUDA handles
    cublasHandle_t cublas_handle_;
    cusolverDnHandle_t cusolver_handle_;
    cudaStream_t compute_stream_;
    cudaStream_t transfer_stream_;
    
    // ========== GPU Memory - Block Storage (Column-Major) ==========
    // Block vectors stored as contiguous column-major matrices for BLAS-3 efficiency
    // V_j is stored at d_block_basis_[j] with layout: dimension_ × block_size_
    
    cuDoubleComplex* d_V_current_;      // Current block V_j: dim × block_size
    cuDoubleComplex* d_V_prev_;         // Previous block V_{j-1}: dim × block_size
    cuDoubleComplex* d_W_;              // Work block H*V_j: dim × block_size
    cuDoubleComplex* d_temp_block_;     // Temporary block: dim × block_size
    
    // Stored Lanczos blocks for reorthogonalization and Ritz vector computation
    cuDoubleComplex** d_block_basis_;   // Array of pointers to stored blocks
    int num_stored_blocks_;             // How many blocks we can store
    int blocks_computed_;               // How many blocks have been computed
    
    // ========== Block Tridiagonal Matrix Coefficients ==========
    // Block tridiagonal: T_jk = V_j^H * H * V_k
    // Stored as: A_j (diagonal blocks), B_j (off-diagonal blocks)
    // A_j = V_j^H * H * V_j (block_size × block_size, Hermitian)
    // B_j = V_{j+1}^H * W_j = R_j from QR(W_j - V_j*A_j - V_{j-1}*B_{j-1}^H)
    
    std::vector<std::vector<std::complex<double>>> alpha_blocks_;  // Diagonal blocks A_j
    std::vector<std::vector<std::complex<double>>> beta_blocks_;   // Off-diagonal blocks B_j
    
    // ========== cuSOLVER Workspace ==========
    cuDoubleComplex* d_qr_work_;        // QR factorization workspace
    cuDoubleComplex* d_tau_;            // Householder reflectors
    int* d_info_;                       // cuSOLVER info
    int qr_lwork_;                      // QR workspace size
    
    // ========== Overlap/Projection Matrices ==========
    cuDoubleComplex* d_overlap_;        // block_size × block_size overlap matrix
    cuDoubleComplex* d_projection_;     // For block orthogonalization
    
    // Statistics
    Stats stats_;
    
    // ========== Memory Management ==========
    void allocateMemory();
    void freeMemory();
    size_t estimateMemoryUsage() const;
    
    // ========== Initialization ==========
    void initializeRandomBlock(cuDoubleComplex* d_block);
    void orthonormalizeBlock(cuDoubleComplex* d_block);  // QR with column normalization
    
    // ========== Block Operations (BLAS-3 Optimized) ==========
    
    /**
     * @brief Batched matrix-vector product: W = H * V (block version)
     * Each column of V is multiplied by H independently
     */
    void blockMatVec(const cuDoubleComplex* d_V, cuDoubleComplex* d_W);
    
    /**
     * @brief Compute block inner product: C = V^H * W
     * Uses cuBLAS ZGEMM for BLAS-3 efficiency
     * @param d_V First block (dim × block_size)
     * @param d_W Second block (dim × block_size)
     * @param d_C Output (block_size × block_size)
     */
    void blockInnerProduct(const cuDoubleComplex* d_V, const cuDoubleComplex* d_W,
                          cuDoubleComplex* d_C);
    
    /**
     * @brief Block AXPY: W = W - V * C
     * Uses cuBLAS ZGEMM: W = W - V @ C where C is block_size × block_size
     */
    void blockAxpy(cuDoubleComplex* d_W, const cuDoubleComplex* d_V,
                  const cuDoubleComplex* d_C, bool subtract = true);
    
    /**
     * @brief QR factorization of block: V = Q * R
     * Uses cuSOLVER for GPU-accelerated Householder QR
     * @param d_block Input/Output: on exit contains Q
     * @param d_R Output: upper triangular R (block_size × block_size)
     * @return true if all columns are linearly independent
     */
    bool qrFactorization(cuDoubleComplex* d_block, cuDoubleComplex* d_R);
    
    /**
     * @brief Block copy: dst = src
     */
    void blockCopy(const cuDoubleComplex* d_src, cuDoubleComplex* d_dst);
    
    /**
     * @brief Check for deflation (near-zero columns after QR)
     * @param d_R R factor from QR
     * @param deflation_indices Output: indices of deflated columns
     * @return Number of deflated vectors
     */
    int checkDeflation(const cuDoubleComplex* d_R, std::vector<int>& deflation_indices);
    
    // ========== Reorthogonalization ==========
    
    /**
     * @brief Reorthogonalize current block against stored blocks
     * Uses batched operations for efficiency when many blocks are stored
     */
    void reorthogonalizeBlock(cuDoubleComplex* d_block, int current_iter);
    
    /**
     * @brief Full reorthogonalization against all stored blocks
     * Uses streamed computation for large number of blocks
     */
    void fullReorthogonalization(cuDoubleComplex* d_block);
    
    // ========== Block Tridiagonal Solver ==========
    
    /**
     * @brief Solve block tridiagonal eigenvalue problem
     * Constructs full block tridiagonal matrix and solves with cuSOLVER
     * @param num_blocks Number of block iterations completed
     * @param num_eigs Number of eigenvalues to extract
     * @param eigenvalues Output: eigenvalues
     * @param tridiag_eigenvecs Output: eigenvectors in block tridiagonal basis
     */
    void solveBlockTridiagonal(int num_blocks, int num_eigs,
                              std::vector<double>& eigenvalues,
                              std::vector<std::vector<std::complex<double>>>& tridiag_eigenvecs);
    
    /**
     * @brief Compute Ritz vectors from block tridiagonal eigenvectors
     */
    void computeBlockRitzVectors(
        const std::vector<std::vector<std::complex<double>>>& tridiag_eigenvecs,
        int num_vecs,
        std::vector<std::vector<std::complex<double>>>& eigenvectors);
    
    // ========== Convergence Checking ==========
    
    /**
     * @brief Check eigenvalue convergence using residual bounds
     * Uses last block's contribution to estimate residuals
     */
    bool checkConvergence(int iter, const std::vector<double>& prev_eigenvalues,
                         double& max_change);
};

/**
 * GPU-accelerated Krylov-Schur algorithm for restarted eigenvalue computation
 * 
 * Krylov-Schur is a restarted variant of Arnoldi/Lanczos that maintains
 * a partial Schur decomposition and can efficiently restart the iteration.
 * 
 * GPU Architecture Optimizations:
 * - All Arnoldi/Krylov vectors stored on GPU (no disk I/O)
 * - Batched orthogonalization using custom CUDA kernels
 * - cuSOLVER for small projected eigenvalue problem
 * - cuBLAS ZGEMM for efficient basis update V_new = V * Q
 * - Memory-efficient: adapts to available GPU memory
 * 
 * Advantages over standard Lanczos:
 * - Implicit restart maintains orthogonality
 * - Better for computing many eigenvalues
 * - Can target interior eigenvalues with shift-invert
 */
class GPUKrylovSchur {
public:
    /**
     * @brief Construct GPU Krylov-Schur solver
     * @param op Pointer to GPU operator (Hamiltonian)
     * @param max_iter Maximum Krylov subspace size per restart cycle
     * @param tolerance Convergence tolerance for eigenvalues
     */
    GPUKrylovSchur(GPUOperator* op, int max_iter, double tolerance);
    ~GPUKrylovSchur();
    
    /**
     * @brief Run Krylov-Schur to find lowest eigenvalues
     * @param num_eigenvalues Number of eigenvalues to compute
     * @param eigenvalues Output: computed eigenvalues (sorted ascending)
     * @param eigenvectors Output: computed eigenvectors (if requested)
     * @param compute_vectors Whether to compute eigenvectors
     */
    void run(int num_eigenvalues,
            std::vector<double>& eigenvalues,
            std::vector<std::vector<std::complex<double>>>& eigenvectors,
            bool compute_vectors = false);
    
    // Performance statistics
    struct Stats {
        double total_time;       // Total wall time
        double matvec_time;      // Time in matrix-vector products
        double ortho_time;       // Time in orthogonalization
        double schur_time;       // Time solving projected eigenproblem
        double restart_time;     // Time in restart operations
        int outer_iterations;    // Number of restart cycles
        int total_arnoldi_steps; // Total Arnoldi iterations
        int converged_eigs;      // Number of converged eigenvalues
        double final_residual;   // Maximum residual at convergence
        size_t memory_used;      // GPU memory used in bytes
    };
    
    Stats getStats() const { return stats_; }
    
    /**
     * @brief Set maximum outer iterations (restart cycles)
     */
    void setMaxOuterIterations(int max_outer) { max_outer_iter_ = max_outer; }
    
private:
    GPUOperator* op_;
    int max_iter_;         // Maximum Krylov subspace size
    double tolerance_;
    int dimension_;        // Hilbert space dimension N
    int max_outer_iter_;   // Maximum restart cycles
    
    // CUDA handles
    cublasHandle_t cublas_handle_;
    cusolverDnHandle_t cusolver_handle_;
    
    // ========== GPU Memory ==========
    // Arnoldi/Krylov basis vectors V = [v_0, v_1, ..., v_{m-1}]
    // Stored as single contiguous array: dimension_ × max_krylov_size
    cuDoubleComplex* d_V_;           // Krylov basis (dim × m)
    int max_krylov_size_;            // How many vectors we can store
    
    cuDoubleComplex* d_w_;           // Work vector for H*v
    cuDoubleComplex* d_temp_;        // Temporary vector
    
    // Projected Hessenberg/tridiagonal matrix H_m
    // For Hermitian case, this is tridiagonal
    cuDoubleComplex* d_H_projected_; // m × m projected matrix on GPU
    
    // Workspace for cuSOLVER eigendecomposition
    cuDoubleComplex* d_evecs_;       // Eigenvectors of projected matrix
    double* d_evals_;                // Eigenvalues of projected matrix
    cuDoubleComplex* d_work_;        // cuSOLVER workspace
    int* d_info_;                    // cuSOLVER info
    int work_size_;                  // Workspace size
    
    // Host-side Hessenberg matrix (for easier manipulation)
    std::vector<std::complex<double>> h_H_projected_;
    
    // Statistics
    Stats stats_;
    
    // ========== Memory Management ==========
    void allocateMemory(int num_eigenvalues);
    void freeMemory();
    
    // ========== Core Operations ==========
    
    /**
     * @brief Initialize random starting vector and normalize
     */
    void initializeRandomVector(cuDoubleComplex* d_vec);
    
    /**
     * @brief Arnoldi iteration: expand Krylov subspace from j_start to m
     * @param j_start Starting index (0 for first iteration, k after restart)
     * @param m Target subspace size
     * @return Actual subspace size achieved (may be < m if breakdown)
     */
    int arnoldiIteration(int j_start, int m);
    
    /**
     * @brief Orthogonalize w against columns 0..j of V using Modified Gram-Schmidt
     * Updates H_projected[0:j+1, j] with the overlaps
     * @param j Current column index
     * @return The norm ||w|| after orthogonalization (beta_{j+1})
     */
    double orthogonalizeAgainstBasis(int j);
    
    /**
     * @brief Solve eigenvalue problem for projected matrix
     * @param m Current subspace size
     * @param eigenvalues_m Output: eigenvalues (sorted)
     * @param eigenvectors_m Output: eigenvectors (column-major m×m)
     * @return true if successful
     */
    bool solveProjectedEigenproblem(int m, std::vector<double>& eigenvalues_m);
    
    /**
     * @brief Check convergence of Ritz pairs
     * @param m Current subspace size
     * @param k Number of desired eigenvalues
     * @param beta_m The beta value at position m (residual scale)
     * @return Number of converged eigenvalues
     */
    int checkConvergence(int m, int k, double beta_m);
    
    /**
     * @brief Perform Krylov-Schur restart
     * Updates V and H_projected to keep only k converged Ritz vectors
     * @param m Current subspace size
     * @param k Number of vectors to keep
     * @return The new beta value for continuing
     */
    double performRestart(int m, int k);
    
    /**
     * @brief Compute full eigenvectors from Ritz vectors
     * eigenvector[i] = V * y_i where y_i is the i-th Ritz vector
     */
    void computeEigenvectors(int m, int num_eigs,
                            std::vector<std::vector<std::complex<double>>>& eigenvectors);
    
    // ========== Vector Operations (use cuBLAS) ==========
    double vectorNorm(const cuDoubleComplex* d_vec);
    void normalizeVector(cuDoubleComplex* d_vec);
    void vectorCopy(const cuDoubleComplex* src, cuDoubleComplex* dst);
    void vectorScale(cuDoubleComplex* d_vec, double scale);
    void vectorAxpy(const cuDoubleComplex* d_x, cuDoubleComplex* d_y,
                   const cuDoubleComplex& alpha);
    std::complex<double> vectorDot(const cuDoubleComplex* d_x,
                                   const cuDoubleComplex* d_y);
    
    /**
     * @brief Get pointer to j-th Krylov vector in the basis
     */
    cuDoubleComplex* getKrylovVector(int j) {
        return d_V_ + static_cast<size_t>(j) * dimension_;
    }
    
    const cuDoubleComplex* getKrylovVector(int j) const {
        return d_V_ + static_cast<size_t>(j) * dimension_;
    }
};

/**
 * GPU-accelerated Block Krylov-Schur algorithm for finding multiple eigenvalues
 * 
 * Combines the benefits of block methods (handling degeneracies, BLAS-3 efficiency)
 * with Krylov-Schur's robust restart mechanism.
 * 
 * Key features:
 * - Block Arnoldi iteration builds subspace with multiple vectors per step
 * - Schur decomposition for implicit restarts
 * - Efficient for degenerate or clustered eigenvalues
 * - Uses cuBLAS GEMM for block operations
 * - cuSOLVER for QR and eigenvalue problems
 */
class GPUBlockKrylovSchur {
public:
    GPUBlockKrylovSchur(GPUOperator* op, int max_iter, int block_size, double tolerance);
    ~GPUBlockKrylovSchur();
    
    void run(int num_eigenvalues,
            std::vector<double>& eigenvalues,
            std::vector<std::vector<std::complex<double>>>& eigenvectors,
            bool compute_vectors = false);
    
    struct Stats {
        double total_time;
        double matvec_time;
        double ortho_time;
        double qr_time;
        double schur_time;
        double restart_time;
        int outer_iterations;
        int total_block_steps;
        int converged_eigs;
        double final_residual;
        size_t memory_used;
    };
    
    Stats getStats() const { return stats_; }
    
private:
    GPUOperator* op_;
    int max_iter_;
    int block_size_;
    double tolerance_;
    int dimension_;
    int max_outer_iter_;
    int max_num_blocks_;
    
    // CUDA handles
    cublasHandle_t cublas_handle_;
    cusolverDnHandle_t cusolver_handle_;
    
    // GPU memory for block Krylov basis (dimension × num_blocks × block_size)
    cuDoubleComplex* d_V_;          // Block Krylov basis (column-major blocks)
    cuDoubleComplex* d_W_;          // Work block (dimension × block_size)
    cuDoubleComplex* d_temp_;       // Temporary block
    
    // Block tridiagonal matrix components
    cuDoubleComplex* d_A_blocks_;   // Diagonal blocks (block_size × block_size each)
    cuDoubleComplex* d_B_blocks_;   // Off-diagonal blocks
    std::vector<std::complex<double>> h_A_blocks_;
    std::vector<std::complex<double>> h_B_blocks_;
    
    // For eigendecomposition of block tridiagonal
    cuDoubleComplex* d_T_dense_;    // Dense block tridiagonal matrix
    cuDoubleComplex* d_evecs_;
    double* d_evals_;
    cuDoubleComplex* d_work_;
    int* d_info_;
    int work_size_;
    
    // QR factorization workspace
    cuDoubleComplex* d_tau_;
    cuDoubleComplex* d_qr_work_;
    int qr_work_size_;
    
    Stats stats_;
    
    void allocateMemory(int num_eigenvalues);
    void freeMemory();
    
    void initializeRandomBlock(cuDoubleComplex* d_block);
    void orthonormalizeBlock(cuDoubleComplex* d_block);
    
    cuDoubleComplex* getBlock(int block_idx) {
        return d_V_ + static_cast<size_t>(block_idx) * block_size_ * dimension_;
    }
    
    void blockMatVec(const cuDoubleComplex* d_V_block, cuDoubleComplex* d_W_block);
    void computeBlockOverlap(const cuDoubleComplex* d_V1, const cuDoubleComplex* d_V2,
                            cuDoubleComplex* d_overlap);
    void orthogonalizeBlockAgainstBasis(int num_blocks, cuDoubleComplex* d_target);
    
    int blockArnoldiIteration(int start_block, int max_blocks);
    bool solveBlockTridiagonalEigenproblem(int num_blocks, std::vector<double>& eigenvalues);
    int checkConvergence(int num_blocks, int num_desired);
    void performRestart(int num_blocks, int num_keep);
    void computeEigenvectors(int num_blocks, int num_eigs,
                            std::vector<std::vector<std::complex<double>>>& eigenvectors);
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

/**
 * @brief Batched dot product kernel for efficient orthogonalization
 * 
 * Computes multiple inner products in parallel using one block per vector.
 * More efficient than sequential cuBLAS calls for multiple vectors.
 */
__global__ void batchedDotProductKernel(const cuDoubleComplex* const* basis,
                                        const cuDoubleComplex* target,
                                        cuDoubleComplex* overlaps,
                                        int num_vecs, int N);

/**
 * @brief Batched orthogonalization correction kernel
 */
__global__ void batchedOrthogonalizeKernel(cuDoubleComplex* const* basis,
                                          cuDoubleComplex* target,
                                          const cuDoubleComplex* overlaps,
                                          int num_vecs, int N);

// ========== Block Lanczos Kernels ==========

/**
 * @brief Initialize random block with cuRAND
 * Initializes a block of vectors (dim × block_size) with random complex values
 * 
 * @param block Output block (column-major, dim × block_size)
 * @param dim Vector dimension
 * @param block_size Number of columns
 * @param seed Random seed
 */
__global__ void initRandomBlockKernel(cuDoubleComplex* block, int dim, int block_size,
                                     unsigned long long seed);

/**
 * @brief Batched block inner product kernel
 * Computes multiple V_i^H * W overlaps for block reorthogonalization
 * 
 * Each CUDA block handles one (block_basis, target_block) pair
 * Output: overlap matrices of size block_size × block_size for each basis block
 * 
 * @param basis_blocks Array of pointers to stored blocks
 * @param target_block Block to orthogonalize (dim × block_size)
 * @param overlaps Output: array of overlap matrices (num_blocks × block_size × block_size)
 * @param num_blocks Number of basis blocks
 * @param dim Vector dimension
 * @param block_size Block size
 */
__global__ void batchedBlockInnerProductKernel(const cuDoubleComplex* const* basis_blocks,
                                               const cuDoubleComplex* target_block,
                                               cuDoubleComplex* overlaps,
                                               int num_blocks, int dim, int block_size);

/**
 * @brief Apply block orthogonalization corrections
 * target = target - sum_i(basis[i] @ overlaps[i])
 * Uses shared memory for overlap matrices
 */
__global__ void batchedBlockOrthogonalizeKernel(const cuDoubleComplex* const* basis_blocks,
                                                cuDoubleComplex* target_block,
                                                const cuDoubleComplex* overlaps,
                                                int num_blocks, int dim, int block_size);

/**
 * @brief Column norm computation kernel
 * Computes the 2-norm of each column in a block
 * 
 * @param block Input block (dim × block_size, column-major)
 * @param norms Output: array of column norms (block_size)
 * @param dim Vector dimension
 * @param block_size Number of columns
 */
__global__ void columnNormsKernel(const cuDoubleComplex* block, double* norms,
                                  int dim, int block_size);

/**
 * @brief Normalize columns of a block
 * Each column is divided by its norm
 */
__global__ void normalizeColumnsKernel(cuDoubleComplex* block, const double* norms,
                                       int dim, int block_size);

/**
 * @brief Check diagonal elements of R for deflation
 * Counts how many diagonal elements are below threshold
 * 
 * @param R Upper triangular R from QR (block_size × block_size)
 * @param threshold Deflation threshold
 * @param block_size Matrix dimension
 * @param deflation_flags Output: 1 if deflated, 0 otherwise
 * @param num_deflated Output: total number of deflated columns
 */
__global__ void checkDeflationKernel(const cuDoubleComplex* R, double threshold,
                                     int block_size, int* deflation_flags, int* num_deflated);

} // namespace GPULanczosKernels

#endif // WITH_CUDA

#endif // GPU_LANCZOS_CUH
