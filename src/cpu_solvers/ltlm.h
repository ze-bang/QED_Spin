// ltlm.h - Low Temperature Lanczos Method implementation
// Specialized for low temperature thermodynamics using ground state projection

#ifndef LTLM_H
#define LTLM_H

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
 * @brief Parameters for LTLM calculation
 * 
 * LTLM differs from FTLM by:
 * 1. First finding the ground state via Lanczos
 * 2. Building Krylov subspace from ground state
 * 3. More accurate at low temperatures
 */
struct LTLMParameters {
    int krylov_dim = 200;              // Dimension of Krylov subspace for thermodynamics
    int ground_state_krylov = 100;     // Krylov dimension for finding ground state
    int num_samples = 1;               // Usually 1 for LTLM (ground state is deterministic)
    int max_iterations = 1000;         // Maximum Lanczos iterations
    double tolerance = 1e-12;          // Convergence tolerance for Lanczos
    bool full_reorthogonalization = false;  // Use full reorthogonalization
    int reorth_frequency = 10;         // Frequency of reorthogonalization (if not full)
    unsigned int random_seed = 0;      // Random seed (0 = use random_device) for initial state
    bool store_intermediate = false;   // Store intermediate data for debugging
    bool compute_error_bars = false;   // Compute standard error (only useful if num_samples > 1)
    bool use_exact_ground_state = false; // If true and ground state eigenvector provided, use it
};

/**
 * @brief Results from LTLM calculation
 */
struct LTLMResults {
    ThermodynamicData thermo_data;           // Thermodynamic properties
    std::vector<ThermodynamicData> per_sample_data;  // Per-sample data (if stored)
    std::vector<double> energy_error;        // Standard error in energy
    std::vector<double> specific_heat_error; // Standard error in specific heat
    std::vector<double> entropy_error;       // Standard error in entropy
    std::vector<double> free_energy_error;   // Standard error in free energy
    double ground_state_energy;              // Ground state energy
    std::vector<double> low_lying_spectrum;  // Low-lying excitation energies
    int total_samples;                       // Number of samples used
    int krylov_dimension;                    // Actual Krylov dimension achieved
};

/**
 * @brief Find ground state using Lanczos iteration
 * 
 * This is the first step of LTLM - find the ground state accurately.
 * Returns the ground state energy and eigenvector.
 * 
 * @param H Hamiltonian matrix-vector product function
 * @param N Hilbert space dimension
 * @param krylov_dim Krylov subspace dimension
 * @param tolerance Convergence tolerance
 * @param full_reorth Use full reorthogonalization
 * @param reorth_freq Reorthogonalization frequency
 * @param ground_state Output: ground state eigenvector
 * @return Ground state energy
 */
double find_ground_state_lanczos(
    std::function<void(const Complex*, Complex*, int)> H,
    int N,
    int krylov_dim,
    double tolerance,
    bool full_reorth,
    int reorth_freq,
    ComplexVector& ground_state
);

/**
 * @brief Build Krylov subspace from ground state for low-lying excitations
 * 
 * After finding the ground state, build a Krylov subspace to capture
 * low-lying excitations. This gives accurate thermodynamics at low T.
 * 
 * @param H Hamiltonian matrix-vector product function
 * @param ground_state Ground state vector
 * @param ground_energy Ground state energy
 * @param N Hilbert space dimension
 * @param krylov_dim Maximum Krylov dimension
 * @param tolerance Convergence tolerance
 * @param full_reorth Use full reorthogonalization
 * @param reorth_freq Reorthogonalization frequency
 * @param excitation_energies Output: excitation energies (relative to ground state)
 * @param weights Output: statistical weights
 * @return Number of excitations found
 */
int build_excitation_spectrum(
    std::function<void(const Complex*, Complex*, int)> H,
    const ComplexVector& ground_state,
    double ground_energy,
    int N,
    int krylov_dim,
    double tolerance,
    bool full_reorth,
    int reorth_freq,
    std::vector<double>& excitation_energies,
    std::vector<double>& weights
);

/**
 * @brief Compute thermodynamics from ground state and low-lying excitations
 * 
 * Uses the ground state and excitation spectrum to compute thermodynamic
 * properties. More accurate than FTLM at low temperatures.
 * 
 * @param ground_energy Ground state energy
 * @param excitation_energies Excitation energies (relative to ground state)
 * @param weights Statistical weights
 * @param temperatures Temperature points to evaluate
 * @return ThermodynamicData structure with thermodynamic properties
 */
ThermodynamicData compute_ltlm_thermodynamics(
    double ground_energy,
    const std::vector<double>& excitation_energies,
    const std::vector<double>& weights,
    const std::vector<double>& temperatures
);

/**
 * @brief Main LTLM driver function
 * 
 * Low Temperature Lanczos Method for thermodynamics:
 * 1. Find ground state via Lanczos
 * 2. Build Krylov subspace from ground state to get low-lying excitations
 * 3. Compute thermodynamics using ground state + excitations
 * 4. More accurate than FTLM at low temperatures
 * 
 * @param H Hamiltonian matrix-vector product function
 * @param N Hilbert space dimension
 * @param params LTLM parameters
 * @param temp_min Minimum temperature
 * @param temp_max Maximum temperature
 * @param num_temp_bins Number of temperature points
 * @param ground_state Optional: pre-computed ground state vector (if available)
 * @param output_dir Directory for output files
 * @return LTLMResults containing thermodynamic properties vs temperature
 */
LTLMResults low_temperature_lanczos(
    std::function<void(const Complex*, Complex*, int)> H,
    int N,
    const LTLMParameters& params,
    double temp_min,
    double temp_max,
    int num_temp_bins,
    const ComplexVector* ground_state = nullptr,
    const std::string& output_dir = ""
);

/**
 * @brief Save LTLM results to file
 * 
 * @param results LTLM results to save
 * @param filename Output filename
 */
void save_ltlm_results(
    const LTLMResults& results,
    const std::string& filename
);

#endif // LTLM_H
