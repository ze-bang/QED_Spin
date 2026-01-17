// ScaLAPACK-based distributed diagonalization with mixed precision support
// filepath: /home/pc_linux/exact_diagonalization_cpp/include/ed/solvers/scalapack_diag.h
#ifndef SCALAPACK_DIAG_H
#define SCALAPACK_DIAG_H

#include <complex>
#include <vector>
#include <functional>
#include <string>
#include <cstdint>

// Type definitions
using Complex = std::complex<double>;
using ComplexFloat = std::complex<float>;
using ComplexVector = std::vector<Complex>;

// ============================================================================
// SCALAPACK CONFIGURATION
// ============================================================================

/**
 * @brief Configuration for ScaLAPACK distributed diagonalization
 */
struct ScaLAPACKConfig {
    // Process grid configuration
    int nprow = 0;              // Number of process rows (0 = auto-detect)
    int npcol = 0;              // Number of process columns (0 = auto-detect)
    
    // Block sizes for distribution
    int mb = 64;                // Row block size
    int nb = 64;                // Column block size
    
    // Mixed precision settings
    bool use_mixed_precision = true;     // Use single precision for intermediate calculations
    double refinement_tol = 1e-12;       // Tolerance for iterative refinement
    int max_refinement_iter = 5;         // Maximum refinement iterations
    
    // Memory optimization
    bool compute_eigenvectors = true;    // Compute eigenvectors
    bool partial_spectrum = false;       // Compute only subset of eigenvalues
    uint64_t num_eigenvalues = 0;        // Number of eigenvalues (0 = all)
    double eigenvalue_lower = 0.0;       // Lower bound for eigenvalue range
    double eigenvalue_upper = 0.0;       // Upper bound for eigenvalue range
    
    // Performance tuning
    bool use_elpa = false;               // Use ELPA if available (faster for large matrices)
    int elpa_solver = 1;                 // ELPA solver (1 = 1-stage, 2 = 2-stage)
    
    // Output settings
    std::string output_dir = "";         // Directory for output files
    bool verbose = true;                 // Print progress information
};

/**
 * @brief Results from ScaLAPACK diagonalization
 */
struct ScaLAPACKResults {
    std::vector<double> eigenvalues;     // Computed eigenvalues
    std::vector<ComplexVector> eigenvectors;  // Eigenvectors (if requested)
    
    // Performance metrics
    double construction_time = 0.0;      // Time for matrix construction
    double diagonalization_time = 0.0;   // Time for diagonalization
    double refinement_time = 0.0;        // Time for iterative refinement
    double total_time = 0.0;             // Total wall time
    
    // Memory usage
    size_t peak_memory_bytes = 0;        // Peak memory usage
    size_t distributed_memory_bytes = 0; // Memory per process
    
    // Quality metrics
    double max_residual = 0.0;           // Maximum eigenvalue residual
    int refinement_iterations = 0;       // Number of refinement iterations
    bool converged = false;              // Did refinement converge
};

// ============================================================================
// MAIN API FUNCTIONS
// ============================================================================

/**
 * @brief Distributed full diagonalization using ScaLAPACK with mixed precision
 * 
 * This function distributes a large Hermitian matrix across MPI processes and
 * uses ScaLAPACK's parallel eigensolvers. Mixed precision is used to reduce
 * memory and computation time:
 * 
 * 1. Matrix construction in single precision (half memory)
 * 2. Initial diagonalization in single precision (faster)
 * 3. Iterative refinement to double precision accuracy
 * 
 * Memory requirements:
 * - Single precision: 8 bytes per complex element (vs 16 for double)
 * - Distributed: N²/P elements per process (P = total processes)
 * - Example: N=100,000 with 100 processes: ~80 GB total, ~0.8 GB per process
 * 
 * @param H Hamiltonian matrix-vector product function
 * @param N Hilbert space dimension
 * @param config Configuration parameters
 * @return ScaLAPACKResults containing eigenvalues and performance metrics
 */
ScaLAPACKResults scalapack_diagonalization(
    std::function<void(const Complex*, Complex*, int)> H,
    uint64_t N,
    const ScaLAPACKConfig& config = ScaLAPACKConfig()
);

/**
 * @brief Distributed full diagonalization for very large matrices (out-of-core capable)
 * 
 * Uses a combination of techniques for matrices too large to fit in memory:
 * 1. Tiled/blocked matrix construction
 * 2. Spectrum slicing to compute eigenvalues in windows
 * 3. Optional out-of-core storage for eigenvectors
 * 
 * @param H Hamiltonian matrix-vector product function
 * @param N Hilbert space dimension  
 * @param config Configuration parameters
 * @param num_spectral_slices Number of spectrum slices (for parallelization)
 * @return ScaLAPACKResults containing eigenvalues
 */
ScaLAPACKResults scalapack_diagonalization_outofcore(
    std::function<void(const Complex*, Complex*, int)> H,
    uint64_t N,
    const ScaLAPACKConfig& config,
    int num_spectral_slices = 0
);

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

/**
 * @brief Initialize ScaLAPACK/BLACS process grid
 * 
 * Must be called before any ScaLAPACK operations. Sets up the 2D process grid.
 * If MPI is not initialized, this function will initialize it.
 * 
 * @param nprow Requested number of process rows (0 = auto)
 * @param npcol Requested number of process columns (0 = auto)
 * @return BLACS context handle (-1 on failure)
 */
int initialize_scalapack_grid(int nprow = 0, int npcol = 0);

/**
 * @brief Finalize ScaLAPACK/BLACS and cleanup resources
 * 
 * @param context BLACS context to release
 */
void finalize_scalapack_grid(int context);

/**
 * @brief Get optimal block size for given matrix dimension and process grid
 * 
 * @param N Matrix dimension
 * @param nprow Number of process rows
 * @param npcol Number of process columns
 * @return Recommended block size
 */
int get_optimal_block_size(uint64_t N, int nprow, int npcol);

/**
 * @brief Estimate memory requirements for distributed diagonalization
 * 
 * @param N Matrix dimension
 * @param nprow Number of process rows
 * @param npcol Number of process columns
 * @param use_mixed_precision Use single precision storage
 * @param compute_eigenvectors Include eigenvector storage
 * @return Memory estimate per process in bytes
 */
size_t estimate_distributed_memory(
    uint64_t N, 
    int nprow, 
    int npcol,
    bool use_mixed_precision = true,
    bool compute_eigenvectors = true
);

/**
 * @brief Check if ScaLAPACK/BLACS is available
 * 
 * @return true if ScaLAPACK is linked and functional
 */
bool is_scalapack_available();

/**
 * @brief Check if ELPA is available (faster eigensolver)
 * 
 * @return true if ELPA is linked and functional
 */
bool is_elpa_available();

// ============================================================================
// MIXED PRECISION UTILITIES
// ============================================================================

/**
 * @brief Convert double precision complex vector to single precision
 * 
 * @param double_vec Input double precision vector
 * @return Single precision vector
 */
std::vector<ComplexFloat> double_to_single_precision(const ComplexVector& double_vec);

/**
 * @brief Convert single precision complex vector to double precision
 * 
 * @param single_vec Input single precision vector
 * @return Double precision vector
 */
ComplexVector single_to_double_precision(const std::vector<ComplexFloat>& single_vec);

/**
 * @brief Perform iterative refinement on eigenpairs
 * 
 * Starting from approximate eigenpairs (λ̃, ṽ) computed in single precision,
 * refine to double precision accuracy using:
 *   r = H*v - λ*v
 *   Solve (H - λI) δv = r
 *   v ← v + δv
 *   λ ← v†Hv / v†v
 * 
 * @param H Hamiltonian matrix-vector product
 * @param eigenvalue Initial eigenvalue estimate (updated in place)
 * @param eigenvector Initial eigenvector estimate (updated in place)
 * @param N Dimension
 * @param tol Convergence tolerance
 * @param max_iter Maximum iterations
 * @return Number of iterations performed (-1 if failed to converge)
 */
int refine_eigenpair(
    std::function<void(const Complex*, Complex*, int)> H,
    double& eigenvalue,
    ComplexVector& eigenvector,
    uint64_t N,
    double tol = 1e-12,
    int max_iter = 10
);

// ============================================================================
// INTERNAL DECLARATIONS (for testing/advanced use)
// ============================================================================

namespace scalapack_internal {

/**
 * @brief Construct local portion of distributed matrix
 * 
 * Each process constructs its local tile of the global matrix using
 * the Hamiltonian matrix-vector product.
 */
template<typename Scalar>
void construct_local_matrix(
    std::function<void(const Complex*, Complex*, int)> H,
    Scalar* local_A,
    int64_t N,
    int mb, int nb,
    int myrow, int mycol,
    int nprow, int npcol,
    int local_rows, int local_cols
);

/**
 * @brief Wrapper for pzheev (parallel Hermitian eigenvalue solver)
 */
int call_pzheev(
    char jobz, char uplo,
    int n,
    std::complex<double>* A, int ia, int ja, const int* descA,
    double* w,
    std::complex<double>* Z, int iz, int jz, const int* descZ
);

/**
 * @brief Wrapper for pcheev (single precision parallel Hermitian eigenvalue solver)
 */
int call_pcheev(
    char jobz, char uplo,
    int n,
    std::complex<float>* A, int ia, int ja, const int* descA,
    float* w,
    std::complex<float>* Z, int iz, int jz, const int* descZ
);

} // namespace scalapack_internal

#endif // SCALAPACK_DIAG_H
