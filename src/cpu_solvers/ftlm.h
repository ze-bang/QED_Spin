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
    uint64_t krylov_dim = 100;              // Dimension of Krylov subspace per sample
    uint64_t num_samples = 10;              // Number of random initial states
    uint64_t max_iterations = 1000;         // Maximum Lanczos iterations
    double tolerance = 1e-10;          // Convergence tolerance for Lanczos
    bool full_reorthogonalization = false;  // Use full reorthogonalization
    uint64_t reorth_frequency = 10;         // Frequency of reorthogonalization (if not full)
    uint64_t random_seed = 0;      // Random seed (0 = use random_device)
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
    uint64_t lanczos_iterations;            // Actual number of Lanczos iterations performed
};

// Note: FTLMResults is now defined in thermal_types.h for CPU/GPU compatibility

/**
 * @brief Parameters for dynamical response calculation
 */
struct DynamicalResponseParameters {
    uint64_t krylov_dim = 200;              // Dimension of Krylov subspace
    uint64_t num_samples = 10;              // Number of random initial states
    double tolerance = 1e-10;          // Convergence tolerance for Lanczos
    bool full_reorthogonalization = false;  // Use full reorthogonalization
    uint64_t reorth_frequency = 10;         // Frequency of reorthogonalization
    uint64_t random_seed = 0;      // Random seed (0 = use random_device)
    double broadening = 0.1;           // Lorentzian broadening parameter (eta)
    bool store_intermediate = false;   // Store per-sample data
};

/**
 * @brief Dynamical response results for a single sample
 */
struct DynamicalResponseSample {
    std::vector<double> ritz_values;   // Eigenvalues from Krylov subspace
    std::vector<double> weights;       // Spectral weights |<psi_i|O|0>|^2 (real, for self-correlation)
    std::vector<Complex> complex_weights;  // Complex spectral weights (for cross-correlation)
    uint64_t lanczos_iterations;            // Actual iterations performed
};

/**
 * @brief Complete dynamical response results
 */
struct DynamicalResponseResults {
    std::vector<double> frequencies;         // Frequency grid (ω)
    std::vector<double> spectral_function;   // Averaged Re[S(ω)]
    std::vector<double> spectral_function_imag;  // Averaged Im[S(ω)] (for cross-correlation)
    std::vector<double> spectral_error;      // Standard error in Re[S(ω)]
    std::vector<double> spectral_error_imag; // Standard error in Im[S(ω)]
    std::vector<DynamicalResponseSample> per_sample_data;  // Per-sample results
    uint64_t total_samples;                       // Number of samples used
    double omega_min;                        // Minimum frequency
    double omega_max;                        // Maximum frequency
};

/**
 * @brief Parameters for static response calculation
 */
struct StaticResponseParameters {
    uint64_t krylov_dim = 100;              // Dimension of Krylov subspace per sample
    uint64_t num_samples = 10;              // Number of random initial states
    double tolerance = 1e-10;          // Convergence tolerance for Lanczos
    bool full_reorthogonalization = false;  // Use full reorthogonalization
    uint64_t reorth_frequency = 10;         // Frequency of reorthogonalization
    uint64_t random_seed = 0;      // Random seed (0 = use random_device)
    bool store_intermediate = false;   // Store per-sample data
    bool compute_error_bars = true;    // Compute standard error across samples
};

/**
 * @brief Static response data for a single sample
 */
struct StaticResponseSample {
    std::vector<double> ritz_values;   // Eigenvalues from Krylov subspace
    std::vector<double> weights;       // Statistical weights
    std::vector<double> expectation_values;  // <n|O|n> for each Ritz state
    uint64_t lanczos_iterations;            // Actual iterations performed
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
    std::vector<StaticResponseSample> per_sample_data;  // Per-sample results
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
    uint64_t N,
    const FTLMParameters& params,
    double temp_min,
    double temp_max,
    uint64_t num_temp_bins,
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
 * @brief Compute dynamical response S(ω) for operator O using Lanczos method
 * 
 * Computes the spectral function S(ω) = <ψ|O†δ(ω - H)O|ψ> where:
 * - H is the Hamiltonian
 * - O is an operator (applied via matrix-vector product)
 * - |ψ> is an initial state (typically ground state or thermal state)
 * 
 * The calculation proceeds by:
 * 1. Apply O to initial state: |φ> = O|ψ>
 * 2. Build Krylov subspace from |φ> using Lanczos
 * 3. Diagonalize tridiagonal matrix to get eigenvalues and weights
 * 4. Construct spectral function with Lorentzian broadening
 * 
 * @param H Hamiltonian matrix-vector product function
 * @param O Operator matrix-vector product function
 * @param psi Initial state vector
 * @param N Hilbert space dimension
 * @param params Parameters for dynamical response calculation
 * @param omega_min Minimum frequency
 * @param omega_max Maximum frequency
 * @param num_omega_bins Number of frequency points
 * @param output_dir Directory for output files
 * @return DynamicalResponseResults containing S(ω) vs frequency
 */
DynamicalResponseResults compute_dynamical_response(
    std::function<void(const Complex*, Complex*, int)> H,
    std::function<void(const Complex*, Complex*, int)> O,
    const ComplexVector& psi,
    uint64_t N,
    const DynamicalResponseParameters& params,
    double omega_min,
    double omega_max,
    uint64_t num_omega_bins,
    double temperature = 0.0,  // Temperature (0 = no thermal weighting)
    const std::string& output_dir = ""
);

/**
 * @brief Compute dynamical response with random initial states (finite temperature)
 * 
 * Similar to compute_dynamical_response but averages over multiple random initial states
 * to approximate finite temperature response. This is useful when the initial state
 * is not known or when computing thermal averages.
 * 
 * @param H Hamiltonian matrix-vector product function
 * @param O Operator matrix-vector product function
 * @param N Hilbert space dimension
 * @param params Parameters for dynamical response calculation
 * @param omega_min Minimum frequency
 * @param omega_max Maximum frequency
 * @param num_omega_bins Number of frequency points
 * @param temperature Temperature for thermal weighting (0 = no thermal weighting)
 * @param output_dir Directory for output files
 * @return DynamicalResponseResults containing averaged S(ω) vs frequency
 */
DynamicalResponseResults compute_dynamical_response_thermal(
    std::function<void(const Complex*, Complex*, int)> H,
    std::function<void(const Complex*, Complex*, int)> O,
    uint64_t N,
    const DynamicalResponseParameters& params,
    double omega_min,
    double omega_max,
    uint64_t num_omega_bins,
    double temperature = 0.0,
    const std::string& output_dir = ""
);

/**
 * @brief Compute dynamical correlation S_{O1,O2}(ω) = ⟨O₁†(t)O₂⟩
 * 
 * Computes the time-dependent correlation function in the frequency domain.
 * This is the Fourier transform of ⟨O₁†(t)O₂⟩ where time evolution is via H.
 * 
 * For the same operator (O1=O2=O), this gives the auto-correlation/spectral density.
 * For different operators, it gives cross-correlations useful for:
 * - Dynamical structure factors S(q,ω)
 * - Current-current correlations
 * - Conductivity σ(ω)
 * 
 * The calculation proceeds by:
 * 1. Apply O2 to random initial states: |φ⟩ = O2|ψ⟩
 * 2. Build Krylov subspace from |φ⟩ using Hamiltonian H
 * 3. Compute matrix elements ⟨v_k|O1|v_l⟩ in Krylov basis
 * 4. Transform to energy eigenbasis to get spectral weights |⟨n|O1|φ⟩|²
 * 5. Construct S(ω) = Σ_n |⟨n|O1|O2ψ⟩|² · Lorentzian(ω - E_n)
 * 6. Average over random samples for thermal ensemble
 * 
 * @param H Hamiltonian matrix-vector product function
 * @param O1 First operator matrix-vector product function
 * @param O2 Second operator matrix-vector product function
 * @param N Hilbert space dimension
 * @param params Parameters for dynamical response calculation
 * @param omega_min Minimum frequency
 * @param omega_max Maximum frequency
 * @param num_omega_bins Number of frequency points
 * @param temperature Temperature for thermal weighting (0 = no thermal weighting)
 * @param output_dir Directory for output files
 * @return DynamicalResponseResults containing S_{O1,O2}(ω) vs frequency
 */
DynamicalResponseResults compute_dynamical_correlation(
    std::function<void(const Complex*, Complex*, int)> H,
    std::function<void(const Complex*, Complex*, int)> O1,
    std::function<void(const Complex*, Complex*, int)> O2,
    uint64_t N,
    const DynamicalResponseParameters& params,
    double omega_min,
    double omega_max,
    uint64_t num_omega_bins,
    double temperature = 0.0,
    const std::string& output_dir = ""
);

/**
 * @brief Compute dynamical correlation S_{O1,O2}(ω) = ⟨O₁†(ω)O₂⟩ for a given state
 * 
 * Computes the spectral function S(ω) = Σₙ ⟨ψ|O₁†|n⟩⟨n|O₂|ψ⟩ δ(ω - Eₙ)
 * where |n⟩ are eigenstates of H with energy Eₙ, for a specific state |ψ⟩.
 * 
 * This is the single-state version of compute_dynamical_correlation.
 * Use this when you have a specific quantum state (e.g., ground state, 
 * excited state, or thermal state) rather than averaging over random samples.
 * 
 * The calculation uses the Lehmann representation:
 * - Applies O₂ to the given state: |φ⟩ = O₂|ψ⟩
 * - Builds Krylov subspace from |φ⟩ using H
 * - Diagonalizes H in Krylov basis to get approximate eigenstates |n⟩
 * - Computes ⟨ψ|O₁†|n⟩ and ⟨n|O₂|ψ⟩
 * - Constructs S(ω) with Lorentzian broadening
 * 
 * For the same operator (O1=O2=O), this gives the spectral density.
 * For different operators, it gives cross-correlations.
 * 
 * @param H Hamiltonian matrix-vector product function
 * @param O1 First operator (O₁) matrix-vector product function
 * @param O2 Second operator (O₂) matrix-vector product function
 * @param state Input quantum state |ψ⟩ (should be normalized)
 * @param N Hilbert space dimension
 * @param params Parameters for dynamical response calculation
 * @param omega_min Minimum frequency
 * @param omega_max Maximum frequency
 * @param num_omega_bins Number of frequency points
 * @param temperature Temperature for Boltzmann weighting (0 = no weighting)
 * @param energy_shift Energy shift to apply (typically ground state energy, 0 = auto-detect from Krylov)
 * @return DynamicalResponseResults containing S_{O1,O2}(ω) vs frequency
 */
DynamicalResponseResults compute_dynamical_correlation_state(
    std::function<void(const Complex*, Complex*, int)> H,
    std::function<void(const Complex*, Complex*, int)> O1,
    std::function<void(const Complex*, Complex*, int)> O2,
    const ComplexVector& state,
    uint64_t N,
    const DynamicalResponseParameters& params,
    double omega_min,
    double omega_max,
    uint64_t num_omega_bins,
    double temperature = 0.0,
    double energy_shift = 0.0
);

/**
 * @brief Save dynamical response results to file
 * 
 * @param results Dynamical response results to save
 * @param filename Output filename
 */
void save_dynamical_response_results(
    const DynamicalResponseResults& results,
    const std::string& filename
);

/**
 * @brief Compute thermal expectation value ⟨O⟩_T and susceptibility
 * 
 * Computes thermal averages of a single operator O at various temperatures:
 * - ⟨O⟩_T = Tr(O exp(-βH)) / Z
 * - ⟨O²⟩_T = Tr(O² exp(-βH)) / Z
 * - χ_T = β(⟨O²⟩ - ⟨O⟩²)  [generalized susceptibility]
 * 
 * The calculation uses FTLM approach:
 * 1. Build Krylov subspace for random states using Lanczos
 * 2. Diagonalize to get approximate eigenvalues and eigenvectors
 * 3. Compute matrix elements ⟨n|O|n⟩ in the Krylov basis
 * 4. Calculate thermal averages with proper Boltzmann weights
 * 5. Average over multiple random samples
 * 
 * @param H Hamiltonian matrix-vector product function
 * @param O Operator matrix-vector product function (can be same as H for energy)
 * @param N Hilbert space dimension
 * @param params Parameters for static response calculation
 * @param temp_min Minimum temperature
 * @param temp_max Maximum temperature
 * @param num_temp_bins Number of temperature points
 * @param output_dir Directory for output files
 * @return StaticResponseResults containing ⟨O⟩_T, fluctuations, and χ vs T
 */
StaticResponseResults compute_thermal_expectation_value(
    std::function<void(const Complex*, Complex*, int)> H,
    std::function<void(const Complex*, Complex*, int)> O,
    uint64_t N,
    const StaticResponseParameters& params,
    double temp_min,
    double temp_max,
    uint64_t num_temp_bins,
    const std::string& output_dir = ""
);

/**
 * @brief Compute static response function ⟨O₁†O₂⟩_T (default two-point correlation)
 * 
 * Computes the static correlation function between two operators at finite temperature.
 * The correlation is computed as ⟨O₁†O₂⟩ = ⟨(O₁|n⟩)† · (O₂|n⟩)⟩ averaged over thermal ensemble.
 * 
 * This is the default static response function, analogous to the dynamical response S(ω).
 * For single-operator expectation values, use compute_thermal_expectation_value() instead.
 * 
 * Useful for computing correlation functions, structure factors at q=0, etc.
 * 
 * Note: This computes the full correlation ⟨O₁†O₂⟩, not the connected part.
 * To get the connected correlation, subtract ⟨O₁⟩*⟨O₂⟩* from the result.
 * 
 * @param H Hamiltonian matrix-vector product function
 * @param O1 First operator matrix-vector product function
 * @param O2 Second operator matrix-vector product function
 * @param N Hilbert space dimension
 * @param params Parameters for static response calculation
 * @param temp_min Minimum temperature
 * @param temp_max Maximum temperature
 * @param num_temp_bins Number of temperature points
 * @param output_dir Directory for output files
 * @return StaticResponseResults containing full correlation function vs T
 */
StaticResponseResults compute_static_response(
    std::function<void(const Complex*, Complex*, int)> H,
    std::function<void(const Complex*, Complex*, int)> O1,
    std::function<void(const Complex*, Complex*, int)> O2,
    uint64_t N,
    const StaticResponseParameters& params,
    double temp_min,
    double temp_max,
    uint64_t num_temp_bins,
    const std::string& output_dir = ""
);

/**
 * @brief Save static response results to file
 * 
 * @param results Static response results to save
 * @param filename Output filename
 */
void save_static_response_results(
    const StaticResponseResults& results,
    const std::string& filename
);

#endif // FTLM_H
