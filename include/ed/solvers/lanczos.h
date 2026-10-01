// Lanczos algorithm implementation for exact diagonalization
#pragma once
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

// Type definitions for complex vectors
using Complex = std::complex<double>;
using ComplexVector = std::vector<Complex>;

namespace ed { class LinearOperator; }

// Dense full diagonalization (LAPACK) of a block inside the dense window
// (dimension <= 120000); larger blocks throw.
//
// `op_for_dense` (optional): when non-null AND it supports it
// (`try_build_dense_columns`), the dense matrix is assembled DIRECTLY from the
// operator's sparse term structure in O(nnz) -- vs the O(dim) full matvecs (=
// O(dim*nnz)) of the `H` column build, which dominates for large sparse H. Pass
// the LinearOperator being solved; nullptr keeps the matvec column build (used
// by GPU / wrapped-matvec callers that have no operator handle).
// ``eigenvectors_out`` (optional): receives the requested eigenvectors in
// memory (one std::vector<Complex> per eigenvalue, in the operator's basis)
// when ``compute_eigenvectors`` is set.
void full_diagonalization(std::function<void(const Complex*, Complex*, int)> H, uint64_t N, uint64_t num_eigs,
                       std::vector<double>& eigenvalues,
                       bool compute_eigenvectors = true,
                       const ed::LinearOperator* op_for_dense = nullptr,
                       std::vector<std::vector<Complex>>* eigenvectors_out = nullptr);

// LinearOperator overload: the matvec column build through op.apply() (one virtual call
// per matvec; no dense fast path).
void full_diagonalization(const ed::LinearOperator& H_op,
                          uint64_t N, uint64_t num_eigs,
                          std::vector<double>& eigenvalues,
                          bool compute_eigenvectors = true);
