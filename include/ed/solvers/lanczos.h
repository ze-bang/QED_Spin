// Lanczos algorithm implementation for exact diagonalization
// filepath: /home/pc_linux/exact_diagonalization_cpp/src/lanczos.h
#ifndef LANCZOS_H
#define LANCZOS_H
#if defined(WITH_MKL)
#define EIGEN_USE_MKL_ALL
#endif

// Define M_PI if not already defined (non-standard but commonly needed)
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#include <iostream>
#include <complex>
#include <vector>
#include <functional>
#include <random>
#include <cmath>
#include <ed/core/blas_lapack_wrapper.h>
#include <ed/core/construct_ham.h>
#include <iomanip>
#include <algorithm>
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <stack>
#include <fstream>
#include <set>
#include <thread>
#include <chrono>
#include <mutex>
#include <numeric>
#include <map>

// Type definition for complex vector and matrix operations
using Complex = std::complex<double>;
using ComplexVector = std::vector<Complex>;
using ComplexMatrix = std::vector<ComplexVector>;

ComplexVector generateRandomVector(int N, std::mt19937& gen, std::uniform_real_distribution<double>& dist);

// Generate a random complex vector that is orthogonal to all vectors in the provided set
ComplexVector generateOrthogonalVector(int N, const std::vector<ComplexVector>& vectors, std::mt19937& gen, std::uniform_real_distribution<double>& dist);

void refine_eigenvector_with_cg(std::function<void(const Complex*, Complex*, int)> H,
                               ComplexVector& v, double& lambda, uint64_t N, double tol);
                               
// Helper function to refine a set of degenerate eigenvectors
void refine_degenerate_eigenvectors(std::function<void(const Complex*, Complex*, int)> H,
                                  std::vector<ComplexVector>& vectors, double lambda, uint64_t N, double tol);
ComplexVector read_basis_vector(const std::string& temp_dir, uint64_t index, uint64_t N);

// Helper function to write a basis vector to file
bool write_basis_vector(const std::string& temp_dir, uint64_t index, const ComplexVector& vec, uint64_t N);

// Helper function to solve tridiagonal eigenvalue problem
int solve_tridiagonal_matrix(const std::vector<double>& alpha, const std::vector<double>& beta, 
                            uint64_t m, uint64_t exct, std::vector<double>& eigenvalues, 
                            const std::string& temp_dir, const std::string& evec_dir, 
                            bool eigenvectors, uint64_t N);

/**
 * @brief Diagonalize tridiagonal matrix and extract Ritz values and weights
 * 
 * This is a lightweight helper for FTLM-style calculations that just need
 * the Ritz values and weights (squared first component) without full eigenvector reconstruction.
 * 
 * @param alpha Diagonal elements of tridiagonal matrix
 * @param beta Off-diagonal elements (beta[0] should be 0)
 * @param ritz_values Output: eigenvalues sorted in ascending order
 * @param weights Output: squared first component of each eigenvector (for FTLM weighting)
 * @param evecs Optional output: eigenvectors in column-major order (m x m)
 */
void diagonalize_tridiagonal_ritz(
    const std::vector<double>& alpha,
    const std::vector<double>& beta,
    std::vector<double>& ritz_values,
    std::vector<double>& weights,
    std::vector<double>* evecs = nullptr
);

/**
 * @brief Build Lanczos tridiagonal with optional in-memory basis vector storage
 * 
 * Similar to build_lanczos_tridiagonal in ftlm.cpp, but with option to store
 * basis vectors in memory for later use (e.g., computing expectation values).
 * 
 * @param H Hamiltonian matrix-vector product
 * @param v0 Initial vector (should be normalized)
 * @param N Hilbert space dimension
 * @param max_iter Maximum Lanczos iterations
 * @param tol Convergence tolerance
 * @param full_reorth Use full reorthogonalization
 * @param reorth_freq Reorthogonalization frequency
 * @param alpha Output: diagonal elements
 * @param beta Output: off-diagonal elements
 * @param basis_vectors Optional: store basis vectors in memory
 * @return Number of iterations performed
 */
int build_lanczos_tridiagonal_with_basis(
    std::function<void(const Complex*, Complex*, int)> H,
    const ComplexVector& v0,
    uint64_t N,
    uint64_t max_iter,
    double tol,
    bool full_reorth,
    uint64_t reorth_freq,
    std::vector<double>& alpha,
    std::vector<double>& beta,
    std::vector<ComplexVector>* basis_vectors = nullptr
);

void lanczos_no_ortho(std::function<void(const Complex*, Complex*, int)> H, uint64_t N, uint64_t max_iter, uint64_t exct, 
             double tol, std::vector<double>& eigenvalues, std::string dir = "",
             bool eigenvectors = false);

// Lanczos algorithm with selective reorthogonalization
void lanczos_selective_reorth(std::function<void(const Complex*, Complex*, int)> H, uint64_t N, uint64_t max_iter, uint64_t exct, 
             double tol, std::vector<double>& eigenvalues, std::string dir = "",
             bool eigenvectors = false);

// Lanczos algorithm implementation with basis vectors stored on disk
void lanczos(std::function<void(const Complex*, Complex*, int)> H, uint64_t N, uint64_t max_iter, uint64_t exct, 
             double tol, std::vector<double>& eigenvalues, std::string dir = "",
             bool eigenvectors = false);

// Block Lanczos algorithm for finding eigenvalues with degeneracies
void block_lanczos(std::function<void(const Complex*, Complex*, int)> H, uint64_t N, uint64_t max_iter, 
                   uint64_t num_eigs, uint64_t block_size, double tol, std::vector<double>& eigenvalues, 
                   std::string dir = "", bool compute_eigenvectors = false);

// Chebyshev Filtered Lanczos algorithm with automatic spectrum range estimation
void chebyshev_filtered_lanczos(std::function<void(const Complex*, Complex*, int)> H, uint64_t N, 
                              uint64_t max_iter, uint64_t num_eigs,
                              double tol, std::vector<double>& eigenvalues, std::string dir = "",
                              bool compute_eigenvectors = false, double target_lower=0, double target_upper=0);

// Shift-Invert Lanczos algorithm - state-of-the-art implementation
// Finds eigenvalues near a target shift σ by solving (H - σI)^{-1} eigenproblem
void shift_invert_lanczos(std::function<void(const Complex*, Complex*, int)> H, uint64_t N, 
                         uint64_t max_iter, uint64_t num_eigs, double sigma, double tol, 
                         std::vector<double>& eigenvalues, std::string dir = "",
                         bool compute_eigenvectors = false);

// Full diagonalization algorithm optimized for sparse matrices
void full_diagonalization(std::function<void(const Complex*, Complex*, int)> H, uint64_t N, uint64_t num_eigs, 
                       std::vector<double>& eigenvalues, std::string dir = "",
                       bool compute_eigenvectors = true);

// Krylov-Schur algorithm implementation
void krylov_schur(std::function<void(const Complex*, Complex*, int)> H, uint64_t N, uint64_t max_iter, 
                  uint64_t num_eigs, double tol, std::vector<double>& eigenvalues, std::string dir = "",
                  bool compute_eigenvectors = false);

// Block Krylov-Schur algorithm for computing multiple eigenvalues with degeneracies
// Combines block Arnoldi iteration with Schur decomposition and implicit restarts
void block_krylov_schur(std::function<void(const Complex*, Complex*, int)> H, uint64_t N, uint64_t max_iter, 
                        uint64_t num_eigs, uint64_t block_size, double tol, std::vector<double>& eigenvalues, 
                        std::string dir = "", bool compute_eigenvectors = false);

// Implicitly Restarted Lanczos algorithm implementation
void implicitly_restarted_lanczos(std::function<void(const Complex*, Complex*, int)> H, uint64_t N, 
                                 uint64_t max_iter, uint64_t num_eigs, double tol, 
                                 std::vector<double>& eigenvalues, std::string dir = "",
                                 bool compute_eigenvectors = false);

// Thick Restart Lanczos algorithm implementation
// Combines the benefits of implicit restart with retention of converged vectors
void thick_restart_lanczos(std::function<void(const Complex*, Complex*, int)> H, uint64_t N, 
                           uint64_t max_iter, uint64_t num_eigs, double tol, 
                           std::vector<double>& eigenvalues, std::string dir = "",
                           bool compute_eigenvectors = false);

// Helper function to estimate number of eigenvalues in an interval
int estimate_eigenvalue_count(std::function<void(const Complex*, Complex*, int)> H, uint64_t N, 
                            double lower_bound, double upper_bound);

// Helper function to orthogonalize degenerate eigenvector subspace
void orthogonalize_degenerate_subspace(std::vector<ComplexVector>& vectors, double eigenvalue,
                                     std::function<void(const Complex*, Complex*, int)> H, uint64_t N);

// Adaptive Spectrum Slicing Full Diagonalization with Degeneracy Preservation
void optimal_spectrum_solver(std::function<void(const Complex*, Complex*, int)> H, uint64_t N, uint64_t max_iter,
                                             std::vector<double>& eigenvalues, std::string dir = "",
                                             bool compute_eigenvectors = true);
#endif // LANCZOS_H