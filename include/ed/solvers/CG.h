#ifndef CG_H
#define CG_H

#pragma once

// =============================================================================
// IMPORTANT — naming history
//
// This header is named "CG.h" for legacy reasons. It does NOT implement
// the resolvent / linear-solver Conjugate Gradient (which would solve
// (A - sigma*I) x = b for a fixed shift sigma). The Conjugate Gradient
// _eigen_solver path lived here historically, has long since been removed,
// and the two functions that remain are:
//
//   * davidson_method      — preconditioned Davidson eigensolver
//   * lobpcg_method        — Locally Optimal Block Preconditioned CG
//                            EIGENSOLVER (the "CG" in the name) for the
//                            lowest few eigenpairs
//   * lobpcg_diagonalization — back-compat wrapper around lobpcg_method
//
// Both LOBPCG and Davidson are *eigenvalue* solvers, not linear solvers,
// despite the file name. If you are looking for a resolvent CG (used by
// e.g. the continued-fraction Green's function path), see
// src/solvers/cpu/lanczos.cpp's continued-fraction routines or the
// shift-invert path in arpack.h. (D-11 in the modernization audit.)
//
// TODO(Phase 2): rename this header to ed/solvers/eigen_cg.h and add a
// shim that #includes the new name; keep the legacy "CG.h" as a
// deprecated alias for one release cycle.
// =============================================================================

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

// =============================================================================
// Phase 4 (matvec-unification): MatVecOperator-taking overloads.
// =============================================================================
#include <ed/matvec/matvec.h>

inline void davidson_method(const ed::matvec::MatVecOperator& H_op,
                            uint64_t N, uint64_t max_iter, uint64_t max_subspace,
                            uint64_t num_eigenvalues, double tol,
                            std::vector<double>& eigenvalues,
                            std::vector<ComplexVector>& eigenvectors,
                            std::string dir = "")
{
    davidson_method(ed::matvec::as_apply_function(H_op),
                    N, max_iter, max_subspace, num_eigenvalues, tol,
                    eigenvalues, eigenvectors, std::move(dir));
}

inline void lobpcg_method(const ed::matvec::MatVecOperator& H_op,
                          uint64_t N, uint64_t max_iter, uint64_t num_eigenvalues,
                          double tol, std::vector<double>& eigenvalues,
                          std::vector<ComplexVector>& eigenvectors,
                          std::string dir = "", bool use_preconditioning = false)
{
    lobpcg_method(ed::matvec::as_apply_function(H_op),
                  N, max_iter, num_eigenvalues, tol, eigenvalues, eigenvectors,
                  std::move(dir), use_preconditioning);
}

inline void lobpcg_diagonalization(const ed::matvec::MatVecOperator& H_op,
                                   uint64_t N, uint64_t max_iter, uint64_t exct,
                                   double tol, std::vector<double>& eigenvalues,
                                   std::string dir = "",
                                   bool compute_eigenvectors = false,
                                   bool use_preconditioning = false)
{
    lobpcg_diagonalization(ed::matvec::as_apply_function(H_op),
                           N, max_iter, exct, tol, eigenvalues, std::move(dir),
                           compute_eigenvectors, use_preconditioning);
}

#endif  // CG_H