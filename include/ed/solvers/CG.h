#ifndef CG_H
#define CG_H

#pragma once

#include <vector>
#include <complex>
#include <functional>
#include <random>
#include <cmath>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <cstdlib>
#include <ed/core/blas_lapack_wrapper.h>
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <string>

// Define types for convenience
using Complex = std::complex<double>;
using ComplexVector = std::vector<Complex>;


// Davidson method for finding lowest eigenvalues
void davidson_method(
    std::function<void(const Complex*, Complex*, int)> H,  // Hamiltonian operator
    uint64_t N,                                                 // Hilbert space dimension
    uint64_t max_iter,                                          // Maximum iterations
    uint64_t max_subspace,                                      // Maximum subspace size
    uint64_t num_eigenvalues,                                   // Number of eigenvalues to find
    double tol,                                            // Tolerance for convergence
    std::vector<double>& eigenvalues,                      // Output eigenvalues
    std::vector<ComplexVector>& eigenvectors,              // Output eigenvectors
    std::string dir = ""                                   // Directory for temporary files
);

void lobpcg_method(
    std::function<void(const Complex*, Complex*, int)> H,  // Hamiltonian operator
    uint64_t N,                                                 // Hilbert space dimension
    uint64_t max_iter,                                          // Maximum iterations
    uint64_t num_eigenvalues,                                   // Number of eigenvalues to find
    double tol,                                           // Tolerance for convergence
    std::vector<double>& eigenvalues,                     // Output eigenvalues
    std::vector<ComplexVector>& eigenvectors,             // Output eigenvectors
    std::string dir = "",                                 // Directory for temporary files
    bool use_preconditioning = false                      // Whether to use preconditioning
);

// Function with same interface as cg_diagonalization
void lobpcg_diagonalization(
    std::function<void(const Complex*, Complex*, int)> H, 
    uint64_t N, 
    uint64_t max_iter, 
    uint64_t exct, 
    double tol, 
    std::vector<double>& eigenvalues, 
    std::string dir = "",
    bool compute_eigenvectors = false,
    bool use_preconditioning = false
);

#endif  // CG_H