#ifndef OBSERVABLES_H
#define OBSERVABLES_H

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
#include <mutex>
// #include "lanczos.h"

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//// Observables

// Type definition for complex vector and matrix operations
using Complex = std::complex<double>;
using ComplexVector = std::vector<Complex>;


// Calculate thermodynamic quantities directly from eigenvalues
ThermodynamicData calculate_thermodynamics_from_spectrum(
    const std::vector<double>& eigenvalues,
    double T_min = 0.01,        // Minimum temperature
    double T_max = 10.0,        // Maximum temperature
    uint64_t num_points = 100        // Number of temperature points
);
// Calculate thermal expectation value of operator A using eigenvalues and eigenvectors
// <A> = (1/Z) * ∑_i exp(-β*E_i) * <ψ_i|A|ψ_i>
Complex calculate_thermal_expectation(
    std::function<void(const Complex*, Complex*, int)> A,  // Observable operator
    uint64_t N,                                               // Hilbert space dimension
    double beta,                                         // Inverse temperature β = 1/kT
    const std::string& eig_dir                           // Directory with eigenvector files
);

// Calculate matrix element <ψ₁|A|ψ₂> between two state vectors
Complex calculate_matrix_element(
    std::function<void(const Complex*, Complex*, int)> A,  // Operator A
    const ComplexVector& psi1,                           // First state vector |ψ₁⟩
    const ComplexVector& psi2,                           // Second state vector |ψ₂⟩
    uint64_t N                                               // Dimension of Hilbert space
);


SpectralFunctionData calculate_spectral_function(
    std::function<void(const Complex*, Complex*, int)> O,  // Operator O
    uint64_t N,                                                // Hilbert space dimension
    const std::string& eig_dir,                           // Directory with eigenvector files
    double omega_min = -10.0,                            // Minimum frequency
    double omega_max = 10.0,                             // Maximum frequency
    uint64_t num_points = 1000,                               // Number of frequency points
    double eta = 0.1,                                    // Broadening parameter
    double temperature = 0.0,                            // Temperature (0 for ground state only)
    bool use_lorentzian = false                          // Use Lorentzian (true) or Gaussian (false) broadening
);

DynamicalSusceptibilityData calculate_dynamical_susceptibility(
    std::function<void(const Complex*, Complex*, int)> A,  // Operator A
    uint64_t N,                                                // Hilbert space dimension
    const std::string& eig_dir,                           // Directory with eigenvector files
    double omega_min = -10.0,                            // Minimum frequency
    double omega_max = 10.0,                             // Maximum frequency
    uint64_t num_points = 1000,                               // Number of frequency points
    double eta = 0.1,                                    // Broadening parameter
    double temperature = 1.0                             // Temperature (in energy units)
);

// Calculate quantum Fisher information for operator A at temperature T
double calculate_quantum_fisher_information(
    std::function<void(const Complex*, Complex*, int)> A,  // Observable operator
    uint64_t N,                                               // Hilbert space dimension
    double temperature,                                  // Temperature (in energy units)
    const std::string& eig_dir                           // Directory with eigenvector files
);

// Compute thermal expectation values of S^+, S^-, S^z operators at each site
void compute_spin_expectations(
    const std::string& eigdir,  // Directory with eigenvalues and eigenvectors
    const std::string output_dir, // Directory for output files
    uint64_t num_sites,              // Number of sites
    float spin_l,              // Spin length (e.g., 0.5 for spin-1/2)
    double temperature,         // Temperature T (in energy units)
    bool print_output = true    // Whether to print the results to console
);

// Load eigenstate from file with format like eigenvector_block0_0.dat
ComplexVector load_eigenstate_from_file(const std::string& filename, uint64_t expected_dimension = -1);

// Load classical eigenstate (basis state with Nth largest amplitude) from file
ComplexVector load_classical_eigenstate_from_file(
    const std::string& filename, 
    uint64_t expected_dimension = -1,
    uint64_t nth_state = 1            // Select the nth most probable state (default: most probable)
);

// Calculate spin expectations for a single eigenstate
std::vector<std::vector<Complex>> compute_eigenstate_spin_expectations(
    const ComplexVector& eigenstate,   // Eigenstate as complex vector
    uint64_t num_sites,                     // Number of sites
    float spin_l,                      // Spin length (e.g., 0.5 for spin-1/2)
    const std::string& output_file = "",  // Optional: output file path
    bool print_output = true           // Whether to print the results to console
);


// Compute two-site correlations (Sz*Sz and S+*S-) for a single eigenstate
std::vector<std::vector<std::vector<Complex>>> compute_eigenstate_spin_correlations(
    const ComplexVector& eigenstate,   // Eigenstate as complex vector
    uint64_t num_sites,                     // Number of sites
    float spin_l,                      // Spin length (e.g., 0.5 for spin-1/2)
    const std::string& output_file = "",  // Optional: output file path
    bool print_output = true           // Whether to print the results to console
);

// Compute spin expectations for a specific eigenstate loaded from a file
std::vector<std::vector<Complex>> compute_eigenstate_spin_expectations_from_file(
    const std::string& eigenstate_file, // File containing the eigenstate
    uint64_t num_sites,                     // Number of sites
    float spin_l,                      // Spin length (e.g., 0.5 for spin-1/2)
    const std::string& output_file = "",  // Optional: output file path
    bool print_output = true,           // Whether to print the results to console
    bool classical = false,               // Whether to load a classical eigenstate
    uint64_t nth_state = 1                   // Select the nth most probable state (default: most probable)
);

std::vector<std::vector<std::vector<Complex>>> compute_eigenstate_spin_correlations_from_file(
    const std::string& eigenstate_file, // File containing the eigenstate
    uint64_t num_sites,                     // Number of sites
    float spin_l,                      // Spin length (e.g., 0.5 for spin-1/2)
    const std::string& output_file = "",  // Optional: output file path
    bool print_output = true,           // Whether to print the results to console
    bool classical = false,              // Whether to load a classical eigenstate
    uint64_t nth_state = 1                    // Select the nth most probable state (default: most probable)
);

// Compute thermal expectation values of two-site correlators (Sz*Sz and S+*S-)
void compute_spin_correlations(
    const std::string& eigdir,  // Directory with eigenvalues and eigenvectors
    const std::string output_dir, // Directory for output files
    uint64_t num_sites,              // Number of sites
    float spin_l,              // Spin length (e.g., 0.5 for spin-1/2)
    double temperature,         // Temperature T (in energy units)
    bool print_output = true    // Whether to print the results to console
);

#endif // OBSERVABLES_H