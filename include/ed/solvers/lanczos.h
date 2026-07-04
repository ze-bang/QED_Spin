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

ComplexVector generateRandomVector(int N, std::mt19937& gen, std::uniform_real_distribution<double>& dist);

/**
 * @brief Generate a random complex vector with i.i.d. complex Gaussian components.
 *
 * Each component has independent N(0,1) real and imaginary parts; the result is
 * then L2-normalised. This is the canonical Hutchinson-style trace estimator
 * (Jaklic & Prelovsek, PRB 49, 5065 (1994); Skilling 1989) and is statistically
 * isotropic on the unit sphere, unlike normalised uniform-cube samples.
 *
 * Use this for FTLM/LTLM/TPQ-style finite-temperature random sampling.
 * For Lanczos starting vectors (where the distribution does not matter after
 * normalisation), generateRandomVector is equally valid.
 */
ComplexVector generateGaussianRandomVector(int N, std::mt19937& gen);

// generateOrthogonalVector was deleted in the debt-cleanup sweep
// (Jul 2026): zero callers.

// refine_eigenvector_with_cg / refine_degenerate_eigenvectors were
// retired together with orthogonalize_degenerate_subspace in the
// minimalist-architecture rev (May 2026): no callers and no remaining
// internal use. See lanczos.cpp for the retirement note.

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
 * @deprecated Phase 5.3 of the Krylov-unification gap-fill (May 2026 day 12+).
 * For new code, prefer `ed::krylov::lanczos_tridiag(...)` in
 * `<ed/krylov/lanczos_tridiag.h>`, which calls
 * `ed::krylov::lanczos_kernel<CpuBackend>` directly without the
 * `std::function` adapter or the `UniqueVec` -> `ComplexVector` copy.
 * The legacy wrapper is retained because the no-reorth / periodic-reorth
 * branches and the in-memory-basis ABI shape of FTLM / LTLM downstream
 * consumers have not been individually ported yet; once those are
 * migrated this function will be removed in a follow-up version bump.
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
// Phase 5.3 of the Krylov-unification gap-fill (May 2026 day 12+):
// the `[[deprecated]]` attribute is gated by `ED_BUILDING_INTERNAL` so
// the library's own legacy callsites (FTLM / LTLM / KPM_DOS / TPQ_DYNAMICAL,
// each waiting on its own per-call migration to `ed::krylov::lanczos_tridiag`)
// do not produce a wall of internal warnings during a clean build. The
// attribute is visible to ALL EXTERNAL CALLERS (Python pybind11 layer,
// downstream C++ apps): for them every use of this function will emit a
// diagnostic pointing at the kernel-direct replacement.
#ifndef ED_BUILDING_INTERNAL
[[deprecated("Use ed::krylov::lanczos_tridiag(...) in "
             "<ed/krylov/lanczos_tridiag.h> -- see Phase 5.3 of the "
             "Krylov-unification gap-fill rollout (May 2026)")]]
#endif
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
void lanczos_real(std::function<void(const double*, double*, int)> H_real,
                  uint64_t N, uint64_t max_iter, uint64_t exct,
                  double tol, std::vector<double>& eigenvalues);

// Block Lanczos algorithm for finding eigenvalues with degeneracies
void block_lanczos(std::function<void(const Complex*, Complex*, int)> H, uint64_t N, uint64_t max_iter, 
                   uint64_t num_eigs, uint64_t block_size, double tol, std::vector<double>& eigenvalues, 
                   std::string dir = "", bool compute_eigenvectors = false);

// Full diagonalization algorithm optimized for sparse matrices.
//
// `op_for_dense` (optional): when non-null AND it supports it
// (`try_build_dense_columns`), the dense matrix is assembled DIRECTLY from the
// operator's sparse term structure in O(nnz) -- vs the O(dim) full matvecs (=
// O(dim*nnz)) of the `H` column build, which dominates for large sparse H. Pass
// the LinearOperator being solved; nullptr keeps the matvec column build (used
// by distributed / GPU / wrapped-matvec callers that have no operator handle).
void full_diagonalization(std::function<void(const Complex*, Complex*, int)> H, uint64_t N, uint64_t num_eigs,
                       std::vector<double>& eigenvalues, std::string dir = "",
                       bool compute_eigenvectors = true,
                       const ed::matvec::MatVecOperator* op_for_dense = nullptr);

// Krylov-Schur algorithm implementation
void krylov_schur(std::function<void(const Complex*, Complex*, int)> H, uint64_t N, uint64_t max_iter, 
                  uint64_t num_eigs, double tol, std::vector<double>& eigenvalues, std::string dir = "",
                  bool compute_eigenvectors = false);

// orthogonalize_degenerate_subspace was retired in the
// minimalist-architecture rev (May 2026): no callers. The Krylov-Schur
// / Block-Lanczos paths get orthonormal degenerate subspaces directly
// from LAPACK on the projected matrix.

// =============================================================================
// Phase 4 (matvec-unification): MatVecOperator-taking convenience overloads.
//
// Every solver above keeps its legacy std::function signature for back-compat,
// and gets a thin inline overload that takes `const ed::matvec::MatVecOperator&`
// directly. The overload uses ed::matvec::as_apply_function(op) -- a one-line
// std::function adapter that captures `&op` and calls op.apply(...) -- so the
// per-matvec cost is one virtual call (already implicit in MatVecOperator).
// No solver internals had to change; the bridge collapses to identity when
// the optimiser inlines through it.
//
// Call sites that already build a std::function continue to compile unchanged;
// new call sites can pass any Operator / FixedSzOperator / GPUOperator /
// StreamingSymmetrySectorView / Distributed{,Symmetry}Operator directly.
// =============================================================================
inline void lanczos(const ed::matvec::MatVecOperator& H_op,
                    uint64_t N, uint64_t max_iter, uint64_t exct, double tol,
                    std::vector<double>& eigenvalues, std::string dir = "",
                    bool eigenvectors = false)
{
    lanczos(ed::matvec::as_apply_function(H_op),
            N, max_iter, exct, tol, eigenvalues, std::move(dir), eigenvectors);
}

inline void block_lanczos(const ed::matvec::MatVecOperator& H_op,
                          uint64_t N, uint64_t max_iter, uint64_t num_eigs,
                          uint64_t block_size, double tol,
                          std::vector<double>& eigenvalues, std::string dir = "",
                          bool compute_eigenvectors = false)
{
    block_lanczos(ed::matvec::as_apply_function(H_op),
                  N, max_iter, num_eigs, block_size, tol, eigenvalues,
                  std::move(dir), compute_eigenvectors);
}

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

inline void krylov_schur(const ed::matvec::MatVecOperator& H_op,
                         uint64_t N, uint64_t max_iter, uint64_t num_eigs,
                         double tol, std::vector<double>& eigenvalues,
                         std::string dir = "",
                         bool compute_eigenvectors = false)
{
    krylov_schur(ed::matvec::as_apply_function(H_op),
                 N, max_iter, num_eigs, tol, eigenvalues, std::move(dir),
                 compute_eigenvectors);
}
