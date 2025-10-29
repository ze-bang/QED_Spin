// ftlm.h - Finite Temperature Lanczos Method implementation
// Computes thermodynamic properties without full spectrum diagonalization

#ifndef FTLM_H
#define FTLM_H

#include <iostream>
#include <complex>
#include <vector>
#include <functional>
#include <random>
#include <cmath>
#include <algorithm>
#include "../core/blas_lapack_wrapper.h"
#include "../core/construct_ham.h"

using Complex = std::complex<double>;
using ComplexVector = std::vector<Complex>;

/**
 * @brief Parameters for FTLM calculation
 */
struct FTLMParameters {
    int krylov_dim = 100;              // Dimension of Krylov subspace per sample
    int num_samples = 10;              // Number of random initial states
    int max_iterations = 1000;         // Maximum Lanczos iterations
    double tolerance = 1e-10;          // Convergence tolerance for Lanczos
    bool full_reorthogonalization = false;  // Use full reorthogonalization
    int reorth_frequency = 10;         // Frequency of reorthogonalization (if not full)
    unsigned int random_seed = 0;      // Random seed (0 = use random_device)
    bool store_intermediate = false;   // Store per-sample intermediate data for debugging
    bool compute_error_bars = true;    // Compute standard error across samples
};

/**
 * @brief Results from a single FTLM sample
 */
struct FTLMSampleResult {
    std::vector<double> ritz_values;   // Eigenvalues from Krylov subspace
    std::vector<double> weights;       // Statistical weights (squared overlap with initial state)
    double ground_state_estimate;      // Lowest Ritz value
    int lanczos_iterations;            // Actual number of Lanczos iterations performed
};

/**
 * @brief Complete FTLM results across all samples
 */
struct FTLMResults {
    ThermodynamicData thermo_data;           // Averaged thermodynamic properties
    std::vector<ThermodynamicData> per_sample_data;  // Per-sample data (if stored)
    std::vector<double> energy_error;        // Standard error in energy
    std::vector<double> specific_heat_error; // Standard error in specific heat
    std::vector<double> entropy_error;       // Standard error in entropy
    std::vector<double> free_energy_error;   // Standard error in free energy
    double ground_state_estimate;            // Best estimate of ground state energy
    int total_samples;                       // Number of samples used
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
    int N,
    int max_iter,
    double tol,
    bool full_reorth,
    int reorth_freq,
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
 * @return ThermodynamicData structure with energy, entropy, specific heat, free energy
 */
ThermodynamicData compute_ftlm_thermodynamics(
    const std::vector<double>& ritz_values,
    const std::vector<double>& weights,
    const std::vector<double>& temperatures
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
 * @brief Main FTLM driver function
 * 
 * Performs Finite Temperature Lanczos Method calculation:
 * 1. Generate R random initial states
 * 2. For each state, build Krylov subspace via Lanczos
 * 3. Diagonalize small tridiagonal matrix
 * 4. Compute thermodynamic observables with proper statistical weights
 * 5. Average over all samples
 * 
 * @param H Hamiltonian matrix-vector product function
 * @param N Hilbert space dimension
 * @param params FTLM parameters
 * @param temp_min Minimum temperature
 * @param temp_max Maximum temperature
 * @param num_temp_bins Number of temperature points
 * @param output_dir Directory for output files
 * @return FTLMResults containing thermodynamic properties vs temperature
 */
FTLMResults finite_temperature_lanczos(
    std::function<void(const Complex*, Complex*, int)> H,
    int N,
    const FTLMParameters& params,
    double temp_min,
    double temp_max,
    int num_temp_bins,
    const std::string& output_dir = ""
);

/**
 * @brief Save FTLM results to file
 * 
 * @param results FTLM results to save
 * @param filename Output filename
 */
void save_ftlm_results(
    const FTLMResults& results,
    const std::string& filename
);

#endif // FTLM_H
