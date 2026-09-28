// ftlm.h - Finite Temperature Lanczos Method implementation
// Computes thermodynamic properties without full spectrum diagonalization

#pragma once

#include <ed/core/solver_defaults.h>

#include <iostream>
#include <complex>
#include <vector>
#include <functional>
#include <random>
#include <cmath>
#include <algorithm>
#include <map>
#include <ed/core/blas_lapack_wrapper.h>
#include <ed/core/construct_ham.h>
#include <ed/matvec/matvec.h>            // MatVecOperator + as_apply_function (Phase 4)

using Complex = std::complex<double>;
using ComplexVector = std::vector<Complex>;

/**
 * @brief Parameters for FTLM calculation
 */
struct FTLMParameters {
    uint64_t krylov_dim = 100;              // Dimension of Krylov subspace per sample
    uint64_t num_samples = 10;              // Number of random initial states
    uint64_t max_iterations = 1000;         // Maximum Lanczos iterations
    double tolerance = 1e-10;          // Convergence tolerance for Lanczos
    bool full_reorthogonalization = ed::defaults::kThermalFullReorth;
    uint64_t reorth_frequency = 10;         // Frequency of reorthogonalization (if not full)
    uint64_t random_seed = 0;      // Random seed (0 = use random_device)
    bool store_intermediate = false;   // Store per-sample intermediate data for debugging
    bool compute_error_bars = true;    // Compute standard error across samples
};

/**
 * @brief Complete static response results
 */
struct StaticResponseResults {
    std::vector<double> temperatures;        // Temperature grid
    std::vector<double> expectation;         // ⟨O⟩_T at each temperature
    std::vector<double> expectation_error;   // Standard error in ⟨O⟩
    std::vector<double> variance;            // ⟨O²⟩ - ⟨O⟩² (fluctuations)
    std::vector<double> variance_error;      // Standard error in variance
    std::vector<double> susceptibility;      // χ = β(⟨O²⟩ - ⟨O⟩²)
    std::vector<double> susceptibility_error;  // Standard error in χ
    uint64_t total_samples;                       // Number of samples used
};

/**
 * @brief Build Krylov subspace and extract tridiagonal matrix coefficients
 * 
 * This is a helper function that runs Lanczos iterations to build a Krylov subspace
 * and returns the tridiagonal matrix elements (alpha, beta) without expanding eigenvectors
 * back to the full Hilbert space.
 * 
 * @param H Hamiltonian matrix-vector product function
 * @param v0 Initial vector
 * @param N Hilbert space dimension
 * @param max_iter Maximum number of iterations
 * @param tol Convergence tolerance
 * @param full_reorth Use full reorthogonalization
 * @param reorth_freq Frequency of reorthogonalization steps
 * @param alpha Output: diagonal elements of tridiagonal matrix
 * @param beta Output: off-diagonal elements of tridiagonal matrix
 * @return Number of iterations performed
 */
int build_lanczos_tridiagonal(
    std::function<void(const Complex*, Complex*, int)> H,
    const ComplexVector& v0,
    uint64_t N,
    uint64_t max_iter,
    double tol,
    bool full_reorth,
    uint64_t reorth_freq,
    std::vector<double>& alpha,
    std::vector<double>& beta
);

/**
 * @brief Compute thermodynamic observables from a single FTLM sample
 * 
 * Given Ritz values and weights from a Krylov subspace, compute thermodynamic
 * quantities at specified temperatures.
 * 
 * @param ritz_values Eigenvalues from tridiagonal diagonalization
 * @param weights Statistical weights (squared first component of eigenvectors)
 * @param temperatures Temperature points to evaluate
 * @param hilbert_dim Hilbert space dimension (needed for proper entropy normalization)
 * @return ThermodynamicData structure with energy, entropy, specific heat, free energy
 */
ThermodynamicData compute_ftlm_thermodynamics(
    const std::vector<double>& ritz_values,
    const std::vector<double>& weights,
    const std::vector<double>& temperatures,
    uint64_t hilbert_dim = 0
);

/**
 * @brief Average thermodynamic data across multiple samples with error estimation
 * 
 * @param sample_data Vector of per-sample thermodynamic data
 * @param results Output structure to store averaged data and error bars
 */
void average_ftlm_samples(
    const std::vector<ThermodynamicData>& sample_data,
    FTLMResults& results
);

/**
 * @brief Combine FTLM results from multiple symmetry sectors
 * 
 * When using symmetrized or fixed-Sz bases, FTLM is run independently on each
 * symmetry sector. This function properly combines the thermodynamic results
 * from all sectors by:
 * 1. Computing the total partition function: Z_total = Σ_α Z_α
 * 2. Weighting each sector's contribution: weight_α = Z_α / Z_total
 * 3. Combining observables: <O> = Σ_α weight_α * <O>_α
 * 
 * This ensures correct thermal averages across the full Hilbert space.
 * 
 * @param sector_results FTLM results for each symmetry sector
 * @param sector_dims Dimension of each sector (for validation)
 * @return Combined thermodynamic data representing the full system
 */
ThermodynamicData combine_ftlm_sector_results(
    const std::vector<FTLMResults>& sector_results,
    const std::vector<uint64_t>& sector_dims
);

/**
 * Spectral function S(w) = -Im G(w + i*eta) / pi of the continued fraction
 * G(z) = norm_sq / (z - a0 - b1^2 / (z - a1 - ...)) built from a Lanczos tridiagonal, evaluated bottom-up
 * (stable, O(M) per frequency). beta[0] is unused; norm_sq = ||O|psi>||^2.
 */
std::vector<double> continued_fraction_spectral_function(
    const std::vector<double>& alpha,
    const std::vector<double>& beta,
    const std::vector<double>& omega_grid,
    double broadening,
    double norm_sq
);
