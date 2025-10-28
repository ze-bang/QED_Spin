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
#include "../core/blas_lapack_wrapper.h"
#include "../core/construct_ham.h"
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

ComplexVector generateRandomVector(int N, std::mt19937& gen, std::uniform_real_distribution<double>& dist);

// Generate a random complex vector that is orthogonal to all vectors in the provided set
ComplexVector generateOrthogonalVector(int N, const std::vector<ComplexVector>& vectors, std::mt19937& gen, std::uniform_real_distribution<double>& dist);

void refine_eigenvector_with_cg(std::function<void(const Complex*, Complex*, int)> H,
                               ComplexVector& v, double& lambda, int N, double tol);
                               
// Helper function to refine a set of degenerate eigenvectors
void refine_degenerate_eigenvectors(std::function<void(const Complex*, Complex*, int)> H,
                                  std::vector<ComplexVector>& vectors, double lambda, int N, double tol);
ComplexVector read_basis_vector(const std::string& temp_dir, int index, int N);

// Helper function to write a basis vector to file
bool write_basis_vector(const std::string& temp_dir, int index, const ComplexVector& vec, int N);

// Helper function to solve tridiagonal eigenvalue problem
int solve_tridiagonal_matrix(const std::vector<double>& alpha, const std::vector<double>& beta, 
                            int m, int exct, std::vector<double>& eigenvalues, 
                            const std::string& temp_dir, const std::string& evec_dir, 
                            bool eigenvectors, int N);

void lanczos_no_ortho(std::function<void(const Complex*, Complex*, int)> H, int N, int max_iter, int exct, 
             double tol, std::vector<double>& eigenvalues, std::string dir = "",
             bool eigenvectors = false);

// Lanczos algorithm with selective reorthogonalization
void lanczos_selective_reorth(std::function<void(const Complex*, Complex*, int)> H, int N, int max_iter, int exct, 
             double tol, std::vector<double>& eigenvalues, std::string dir = "",
             bool eigenvectors = false);

// Lanczos algorithm implementation with basis vectors stored on disk
void lanczos(std::function<void(const Complex*, Complex*, int)> H, int N, int max_iter, int exct, 
             double tol, std::vector<double>& eigenvalues, std::string dir = "",
             bool eigenvectors = false);

// Block Lanczos algorithm for finding eigenvalues with degeneracies
void block_lanczos(std::function<void(const Complex*, Complex*, int)> H, int N, int max_iter, 
                   int num_eigs, int block_size, double tol, std::vector<double>& eigenvalues, 
                   std::string dir = "", bool compute_eigenvectors = false);

// Chebyshev Filtered Lanczos algorithm with automatic spectrum range estimation
void chebyshev_filtered_lanczos(std::function<void(const Complex*, Complex*, int)> H, int N, 
                              int max_iter, int num_eigs,
                              double tol, std::vector<double>& eigenvalues, std::string dir = "",
                              bool compute_eigenvectors = false, double target_lower=0, double target_upper=0);

// Shift-Invert Lanczos algorithm - state-of-the-art implementation
// Finds eigenvalues near a target shift σ by solving (H - σI)^{-1} eigenproblem
void shift_invert_lanczos(std::function<void(const Complex*, Complex*, int)> H, int N, 
                         int max_iter, int num_eigs, double sigma, double tol, 
                         std::vector<double>& eigenvalues, std::string dir = "",
                         bool compute_eigenvectors = false);

// Full diagonalization algorithm optimized for sparse matrices
void full_diagonalization(std::function<void(const Complex*, Complex*, int)> H, int N, int num_eigs, 
                       std::vector<double>& eigenvalues, std::string dir = "",
                       bool compute_eigenvectors = true);

// Krylov-Schur algorithm implementation
void krylov_schur(std::function<void(const Complex*, Complex*, int)> H, int N, int max_iter, 
                  int num_eigs, double tol, std::vector<double>& eigenvalues, std::string dir = "",
                  bool compute_eigenvectors = false);

// Implicitly Restarted Lanczos algorithm implementation
void implicitly_restarted_lanczos(std::function<void(const Complex*, Complex*, int)> H, int N, 
                                 int max_iter, int num_eigs, double tol, 
                                 std::vector<double>& eigenvalues, std::string dir = "",
                                 bool compute_eigenvectors = false);

// Thick Restart Lanczos algorithm implementation
// Combines the benefits of implicit restart with retention of converged vectors
void thick_restart_lanczos(std::function<void(const Complex*, Complex*, int)> H, int N, 
                           int max_iter, int num_eigs, double tol, 
                           std::vector<double>& eigenvalues, std::string dir = "",
                           bool compute_eigenvectors = false);

// Helper function to estimate number of eigenvalues in an interval
int estimate_eigenvalue_count(std::function<void(const Complex*, Complex*, int)> H, int N, 
                            double lower_bound, double upper_bound);

// Helper function to orthogonalize degenerate eigenvector subspace
void orthogonalize_degenerate_subspace(std::vector<ComplexVector>& vectors, double eigenvalue,
                                     std::function<void(const Complex*, Complex*, int)> H, int N);

// Adaptive Spectrum Slicing Full Diagonalization with Degeneracy Preservation
void optimal_spectrum_solver(std::function<void(const Complex*, Complex*, int)> H, int N, int max_iter,
                                             std::vector<double>& eigenvalues, std::string dir = "",
                                             bool compute_eigenvectors = true);
#endif // LANCZOS_H