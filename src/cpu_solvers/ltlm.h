// ltlm.h - Low Temperature Lanczos Method implementation
// Computes exact low-energy eigenspectrum for accurate low-T thermodynamics

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
 */
struct LTLMParameters {
    int num_eigenstates = 50;          // Number of low-energy eigenstates to compute
    int krylov_dim = 200;              // Dimension of Krylov subspace (should be > num_eigenstates)
    double tolerance = 1e-12;          // Eigenvalue convergence tolerance
    bool full_reorthogonalization = true;   // Use full reorthogonalization (recommended)
    int reorth_frequency = 1;          // Reorthogonalization frequency (if not full)
    unsigned int random_seed = 0;      // Random seed (0 = use random_device)
    bool store_eigenvectors = false;   // Store eigenvectors (memory intensive)
    bool verify_eigenvalues = true;    // Verify eigenvalues using residual test
    double residual_tolerance = 1e-10; // Tolerance for residual verification
    double degeneracy_threshold = 1e-10; // Threshold for detecting degenerate eigenvalues
    bool verbose = false;              // Print detailed progress information
};

/**
 * @brief Results from eigenspectrum computation
 */
struct LTLMEigenResults {
    std::vector<double> eigenvalues;         // Low-energy eigenvalues (sorted)
    std::vector<int> degeneracies;           // Degeneracy of each eigenvalue
    std::vector<ComplexVector> eigenvectors; // Eigenvectors (if stored)
    std::vector<double> residual_norms;      // Residual norms ||H|ψ⟩ - E|ψ⟩||
    int lanczos_iterations;                  // Actual number of Lanczos iterations
    bool converged;                          // Whether eigenvalues converged
    int converged_states;                    // Number of converged states
    bool all_converged;                      // Whether all requested states converged
};

/**
 * @brief Complete LTLM results including thermodynamics
 */
struct LTLMResults {
    LTLMEigenResults eigen_results;    // Eigenspectrum data
    ThermodynamicData thermo_data;     // Thermodynamic properties vs temperature
    double ground_state_energy;        // Ground state energy (E_0)
    int num_states_used;               // Number of eigenstates used
    double max_valid_temperature;      // Estimated maximum valid temperature
    bool success;                      // Whether calculation succeeded
    std::string error_message;         // Error message if failed
};

/**
 * @brief Compute low-energy eigenspectrum using Lanczos method
 * 
 * This function computes the lowest num_states eigenvalues and (optionally)
 * eigenvectors of a Hamiltonian using the Lanczos algorithm. It employs
 * reorthogonalization to maintain numerical stability and can verify
 * eigenvalue accuracy using residual tests.
 * 
 * Algorithm:
 * 1. Generate random initial vector |v_0⟩
 * 2. Build Krylov subspace {|v_0⟩, H|v_0⟩, H²|v_0⟩, ...} via Lanczos iteration
 * 3. Construct tridiagonal matrix T from Lanczos coefficients
 * 4. Diagonalize T to obtain Ritz values (approximate eigenvalues)
 * 5. Optionally reconstruct eigenvectors in full Hilbert space
 * 6. Verify eigenvalues using residual ||H|ψ⟩ - E|ψ⟩||
 * 
 * @param H Hamiltonian matrix-vector product function: H(in, out, N)
 * @param N Hilbert space dimension
 * @param num_states Number of lowest eigenstates to compute
 * @param params LTLM parameters
 * @return LTLMEigenResults containing eigenvalues, degeneracies, and metadata
 */
LTLMEigenResults compute_low_energy_spectrum(
    std::function<void(const Complex*, Complex*, int)> H,
    int N,
    int num_states,
    const LTLMParameters& params
);

/**
 * @brief Compute thermodynamic properties from exact eigenspectrum
 * 
 * Computes thermodynamic observables using the canonical ensemble:
 * - Partition function: Z(β) = Σ_i g_i exp(-β E_i)
 * - Energy: E(T) = ⟨H⟩ = Tr(H e^(-βH)) / Z
 * - Specific heat: C_v(T) = β²(⟨H²⟩ - ⟨H⟩²)
 * - Entropy: S(T) = (E - F) / T
 * - Free energy: F(T) = -k_B T ln Z
 * 
 * where g_i is the degeneracy of eigenstate i.
 * 
 * @param eigenvalues Sorted list of eigenvalues
 * @param degeneracies Degeneracy of each eigenvalue
 * @param temperatures Temperature grid
 * @return ThermodynamicData structure with E(T), C_v(T), S(T), F(T)
 */
ThermodynamicData compute_ltlm_thermodynamics(
    const std::vector<double>& eigenvalues,
    const std::vector<double>& degeneracies,
    const std::vector<double>& temperatures
);

/**
 * @brief Compute thermodynamics with automatic degeneracy detection
 * 
 * Automatically groups nearly-degenerate eigenvalues within a threshold
 * and computes thermodynamic properties with proper degeneracy factors.
 * 
 * Degeneracy detection:
 * - Eigenvalues E_i and E_j are considered degenerate if |E_i - E_j| < threshold
 * - Consecutive degenerate eigenvalues are grouped together
 * - Each group is assigned degeneracy = group_size
 * 
 * @param eigenvalues Sorted list of eigenvalues
 * @param temperatures Temperature grid
 * @param degeneracy_threshold Threshold for degeneracy detection (default: 1e-10)
 * @return ThermodynamicData with automatic degeneracy handling
 */
ThermodynamicData compute_ltlm_thermodynamics_auto_degeneracy(
    const std::vector<double>& eigenvalues,
    const std::vector<double>& temperatures,
    double degeneracy_threshold = 1e-10
);

/**
 * @brief Main LTLM driver function
 * 
 * Performs complete Low-Temperature Lanczos Method calculation:
 * 1. Compute low-energy eigenspectrum via Lanczos
 * 2. Detect degeneracies automatically
 * 3. Compute thermodynamic properties across temperature range
 * 4. Estimate validity range of LTLM approximation
 * 5. Optionally save results to files
 * 
 * The LTLM is most accurate when:
 * - k_B T << (E_max - E_0) where E_max is the highest computed eigenstate
 * - The computed eigenstates capture >99% of the partition function weight
 * 
 * @param H Hamiltonian matrix-vector product function
 * @param N Hilbert space dimension
 * @param params LTLM parameters
 * @param temp_min Minimum temperature
 * @param temp_max Maximum temperature
 * @param num_temp_bins Number of temperature points
 * @param output_dir Directory for output files (empty = no file output)
 * @return LTLMResults containing eigenspectrum and thermodynamics
 */
LTLMResults low_temperature_lanczos(
    std::function<void(const Complex*, Complex*, int)> H,
    int N,
    const LTLMParameters& params,
    double temp_min = 0.01,
    double temp_max = 1.0,
    int num_temp_bins = 100,
    const std::string& output_dir = ""
);

/**
 * @brief Save LTLM results to file
 * 
 * Saves thermodynamic data to a text file with columns:
 * Temperature | Energy | Specific_Heat | Entropy | Free_Energy
 * 
 * @param results LTLM results to save
 * @param filename Output filename
 */
void save_ltlm_results(
    const LTLMResults& results,
    const std::string& filename
);

/**
 * @brief Save eigenspectrum to file
 * 
 * Saves eigenvalues and associated metadata to a text file.
 * Includes degeneracies and residuals (if computed).
 * 
 * @param eigen_results Eigenspectrum data to save
 * @param filename Output filename
 */
void save_eigenspectrum(
    const LTLMEigenResults& eigen_results,
    const std::string& filename
);

#endif // LTLM_H
