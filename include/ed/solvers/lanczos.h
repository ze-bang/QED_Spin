// Lanczos algorithm implementation for exact diagonalization
// filepath: /home/pc_linux/exact_diagonalization_cpp/src/lanczos.h
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
#include <ed/matvec/matvec.h>            // MatVecOperator + as_apply_function (Phase 4)
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

/**
 * @brief Generate a random complex vector with i.i.d. complex Gaussian components.
 *
 * Each component has independent N(0,1) real and imaginary parts; the result is
 * then L2-normalised. This is the canonical Hutchinson-style trace estimator
 * (Jaklic & Prelovsek, PRB 49, 5065 (1994); Skilling 1989) and is statistically
 * isotropic on the unit sphere, unlike normalised uniform-cube samples.
 *
 * Use this for FTLM / TPQ-style finite-temperature random sampling.
 */
ComplexVector generateGaussianRandomVector(int N, std::mt19937& gen);

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
 * @brief Estimate the extreme eigenvalues [e_min, e_max] of a Hermitian H.
 *
 * Blocks with dim <= 512 are assembled densely and the exact extremes are
 * returned; larger blocks run a Lanczos sweep without reorthogonalization
 * (krylov_dim clamped to dim, stopping when ||w|| < tol) from a Gaussian
 * random start and return the extreme Ritz values. `gen` is in/out: a
 * deterministically seeded `gen` gives a reproducible estimate.
 * Used by the mTPQ auto-tune to place the shift L above the spectrum.
 */
void estimate_spectral_bounds(
    std::function<void(const Complex*, Complex*, int)> H,
    uint64_t dim,
    int krylov_dim,
    double tol,
    std::mt19937& gen,
    double& e_min,
    double& e_max);

// Default Lanczos with three-vector LOCAL reorthogonalization (DGKS-style),
// basis vectors kept in RAM by default (use ED_LANCZOS_DISK=1 for disk).
//
// Best for small-to-medium Krylov spaces where the three-term recurrence
// stays numerically clean.
void lanczos(std::function<void(const Complex*, Complex*, int)> H, uint64_t N, uint64_t max_iter, uint64_t exct,
             double tol, std::vector<double>& eigenvalues, std::string dir = "",
             bool eigenvectors = false);

// -----------------------------------------------------------------------------
// Real-arithmetic Lanczos (eigenvalues only).                Phase 6 #7
//
// When the Hamiltonian is real and we use a real starting vector, the entire
// Krylov basis stays real in exact arithmetic and to machine precision in
// finite arithmetic. ``lanczos()`` above always uses ``std::complex<double>``
// storage, which doubles every BLAS-1 call's memory traffic and FLOP count
// over the strictly-needed amount.
//
// This entry point uses real (double) storage end-to-end:
//   * 4x4 working set: v_prev, v_current, v_next, w  (each ``N * 8`` bytes)
//   * BLAS-1: cblas_daxpy / cblas_ddot / cblas_dnrm2 / cblas_dscal
//   * H is a real-arithmetic matrix-vector product: ``f(in_re, out_re, N)``
//
// Eigenvalue-only by contract (no basis I/O, no Ritz reconstruction). For
// eigenvector reconstruction, fall back to the complex ``lanczos()``.
//
// Algorithmic choices match ``lanczos()``: 3-vector ring-buffer DGKS local
// reorth, periodic eigenvalue convergence check on the Lanczos tridiagonal
// every 10 iters, breakdown on beta < tol.
// -----------------------------------------------------------------------------
//
// ``iters_out`` / ``converged_out`` (optional): number of Lanczos steps taken
// and whether the Ritz-value test fired before ``max_iter`` (audit 2026-09:
// the orchestrator used to report ``converged = true`` unconditionally).
//
// ``LanczosRealExtras`` (audit 2026-09, optional): deterministic start vector,
// a per-iteration basis-vector hook (two-pass eigenvector reconstruction),
// a fixed-iteration mode (pass 2 replays pass 1 exactly), and the final
// tridiagonal + Ritz residual bounds |beta_m| |z_{m,i}| on output.
struct LanczosRealExtras {
    // ---- inputs ----
    const double* v0 = nullptr;          ///< start vector (length N), nullptr => random
    bool fixed_iterations = false;       ///< run exactly max_iter steps, no convergence test
    /// Called at the top of iteration j with V_j (unit norm, length N),
    /// before the vector is consumed; j = 0, 1, ..., m-1.
    std::function<void(uint64_t j, const double* v_j)> on_basis_vector;
    bool want_ritz = false;              ///< fill ritz_bounds / ritz_vectors below
    bool converge_vectors = false;       ///< also stop only when every requested level's residual bound <= tol*max(1,|E0|)
    // ---- outputs ----
    std::vector<double> alpha;           ///< tridiagonal diagonal (m)
    std::vector<double> beta;            ///< off-diagonal, beta[0] = 0 (m or m+1 entries)
    double beta_last = 0.0;              ///< |beta_m| (norm after the last step)
    std::vector<double> ritz_bounds;     ///< |beta_m| |z_{m,i}|, i < n_eig
    std::vector<double> ritz_vectors;    ///< column-major m x n_eig tridiag eigenvectors
};

void lanczos_real(std::function<void(const double*, double*, int)> H_real,
                  uint64_t N, uint64_t max_iter, uint64_t exct,
                  double tol, std::vector<double>& eigenvalues,
                  uint64_t* iters_out = nullptr, bool* converged_out = nullptr,
                  LanczosRealExtras* extras = nullptr);

// Full diagonalization algorithm optimized for sparse matrices.
//
// `op_for_dense` (optional): when non-null AND it supports it
// (`try_build_dense_columns`), the dense matrix is assembled DIRECTLY from the
// operator's sparse term structure in O(nnz) -- vs the O(dim) full matvecs (=
// O(dim*nnz)) of the `H` column build, which dominates for large sparse H. Pass
// the LinearOperator being solved; nullptr keeps the matvec column build (used
// by GPU / wrapped-matvec callers that have no operator handle).
// ``eigenvectors_out`` (optional): receives the requested eigenvectors in
// memory (one std::vector<Complex> per eigenvalue, in the operator's basis)
// when ``compute_eigenvectors`` is set -- independent of whether ``dir``
// persists them to HDF5.
void full_diagonalization(std::function<void(const Complex*, Complex*, int)> H, uint64_t N, uint64_t num_eigs,
                       std::vector<double>& eigenvalues, std::string dir = "",
                       bool compute_eigenvectors = true,
                       const ed::matvec::MatVecOperator* op_for_dense = nullptr,
                       std::vector<std::vector<Complex>>* eigenvectors_out = nullptr);

// MatVecOperator-taking convenience overload: adapts `op.apply(...)` through
// ed::matvec::as_apply_function (one virtual call per matvec).
inline void full_diagonalization(const ed::matvec::MatVecOperator& H_op,
                                 uint64_t N, uint64_t num_eigs,
                                 std::vector<double>& eigenvalues,
                                 std::string dir = "",
                                 bool compute_eigenvectors = true)
{
    full_diagonalization(ed::matvec::as_apply_function(H_op),
                         N, num_eigs, eigenvalues, std::move(dir),
                         compute_eigenvectors);
}
