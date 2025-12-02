// dynamics.h - General quantum dynamics computation module
// This module provides time evolution and dynamical correlation function computation
// that can be used with any quantum state (not exclusive to TPQ)

#ifndef DYNAMICS_H
#define DYNAMICS_H

#include <iostream>
#include <complex>
#include <vector>
#include <functional>
#include <random>
#include <cmath>
#include <ed/core/blas_lapack_wrapper.h>
#include <fstream>
#include <iomanip>
#include <algorithm>
#include <string>
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <ed/solvers/observables.h>
#include <memory>

// ============================================================================
// TIME EVOLUTION METHODS
// ============================================================================

/**
 * Time evolution using Taylor expansion of exp(-iH*delta_t)
 * 
 * @param H Hamiltonian operator function
 * @param state Current state vector (will be modified in-place)
 * @param N Dimension of the Hilbert space
 * @param delta_t Time step
 * @param n_max Maximum order of Taylor expansion
 * @param normalize Whether to normalize the state after evolution
 */
void time_evolve_taylor(
    std::function<void(const Complex*, Complex*, int)> H,
    ComplexVector& state,
    uint64_t N,
    double delta_t,
    uint64_t n_max = 100,
    bool normalize = true
);

/**
 * Imaginary-time evolution using Taylor expansion of exp(-beta*H)
 * Used for canonical ensemble preparation
 * 
 * @param H Hamiltonian operator function
 * @param state Current state vector (will be modified in-place)
 * @param N Dimension of the Hilbert space
 * @param delta_beta Inverse temperature step
 * @param n_max Maximum order of Taylor expansion
 * @param normalize Whether to normalize the state after evolution
 */
void imaginary_time_evolve_taylor(
    std::function<void(const Complex*, Complex*, int)> H,
    ComplexVector& state,
    uint64_t N,
    double delta_beta,
    uint64_t n_max = 50,
    bool normalize = true
);

/**
 * Krylov-based time evolution using Lanczos method
 * This is much more accurate and stable than Taylor expansion
 * 
 * @param H Hamiltonian operator function
 * @param state Current state vector (will be modified in-place)
 * @param N Dimension of the Hilbert space
 * @param delta_t Time step
 * @param krylov_dim Dimension of Krylov subspace (typically 20-50)
 * @param normalize Whether to normalize the state after evolution
 */
void time_evolve_krylov(
    std::function<void(const Complex*, Complex*, int)> H,
    ComplexVector& state,
    uint64_t N,
    double delta_t,
    uint64_t krylov_dim = 30,
    bool normalize = true
);

/**
 * Chebyshev polynomial-based time evolution
 * Excellent for systems with bounded spectra
 * 
 * @param H Hamiltonian operator function
 * @param state Current state vector (will be modified in-place)
 * @param N Dimension of the Hilbert space
 * @param delta_t Time step
 * @param E_min Minimum eigenvalue of H (estimate)
 * @param E_max Maximum eigenvalue of H (estimate)
 * @param num_terms Number of Chebyshev polynomials to use
 * @param normalize Whether to normalize the state after evolution
 */
void time_evolve_chebyshev(
    std::function<void(const Complex*, Complex*, int)> H,
    ComplexVector& state,
    uint64_t N,
    double delta_t,
    double E_min,
    double E_max,
    uint64_t num_terms = 100,
    bool normalize = true
);

/**
 * 4th-order Runge-Kutta time evolution
 * Best for time-dependent Hamiltonians or when high accuracy is needed
 * 
 * @param H Hamiltonian operator function
 * @param state Current state vector (will be modified in-place)
 * @param N Dimension of the Hilbert space
 * @param delta_t Time step
 * @param normalize Whether to normalize the state after evolution
 */
void time_evolve_rk4(
    std::function<void(const Complex*, Complex*, int)> H,
    ComplexVector& state,
    uint64_t N,
    double delta_t,
    bool normalize = true
);

/**
 * Adaptive time evolution with automatic method selection
 * Chooses the best method based on system size and accuracy requirements
 * 
 * @param H Hamiltonian operator function
 * @param state Current state vector (will be modified in-place)
 * @param N Dimension of the Hilbert space
 * @param delta_t Time step
 * @param accuracy_level Accuracy level: 1=fast, 2=balanced, 3=high accuracy
 * @param normalize Whether to normalize the state after evolution
 */
void time_evolve_adaptive(
    std::function<void(const Complex*, Complex*, int)> H,
    ComplexVector& state,
    uint64_t N,
    double delta_t,
    uint64_t accuracy_level = 2,
    bool normalize = true
);

/**
 * Create a time evolution operator using Taylor expansion of exp(-iH*delta_t)
 * 
 * @param H Hamiltonian operator function
 * @param delta_t Time step
 * @param n_max Maximum order of Taylor expansion
 * @param normalize Whether to normalize the state after evolution
 * @return Function that applies time evolution to a complex vector
 */
std::function<void(const Complex*, Complex*, int)> create_time_evolution_operator(
    std::function<void(const Complex*, Complex*, int)> H,
    double delta_t,
    uint64_t n_max = 10,
    bool normalize = true
);

// ============================================================================
// DYNAMICAL CORRELATION FUNCTION COMPUTATION
// ============================================================================

/**
 * Compute time-dependent correlation function C(t) = ⟨ψ|O_1†(0) O_2(t)|ψ⟩
 * Uses real-time evolution to compute correlations
 * 
 * @param H Hamiltonian operator function
 * @param O1 First operator (applied at t=0)
 * @param O2 Second operator (applied at time t)
 * @param state Initial quantum state
 * @param N Dimension of the Hilbert space
 * @param t_max Maximum evolution time
 * @param dt Time step
 * @param time_evolution_method Method for time evolution (0=Taylor, 1=Krylov, 2=RK4)
 * @param taylor_order Order of Taylor expansion (if using Taylor method)
 * @param krylov_dim Dimension of Krylov subspace (if using Krylov method)
 * @return Vector of time correlation values
 */
std::vector<Complex> compute_time_correlation(
    std::function<void(const Complex*, Complex*, int)> H,
    std::function<void(const Complex*, Complex*, int)> O1,
    std::function<void(const Complex*, Complex*, int)> O2,
    const ComplexVector& state,
    uint64_t N,
    double t_max,
    double dt,
    uint64_t time_evolution_method = 1,
    uint64_t taylor_order = 100,
    uint64_t krylov_dim = 30
);

/**
 * Compute time-dependent correlation functions for multiple operators simultaneously
 * More efficient than calling compute_time_correlation multiple times
 * 
 * @param H Hamiltonian operator function
 * @param operators_1 Vector of first operators (applied at t=0)
 * @param operators_2 Vector of second operators (applied at time t)
 * @param state Initial quantum state
 * @param N Dimension of the Hilbert space
 * @param t_max Maximum evolution time
 * @param dt Time step
 * @param time_evolution_method Method for time evolution (0=Taylor, 1=Krylov, 2=RK4)
 * @param taylor_order Order of Taylor expansion (if using Taylor method)
 * @param krylov_dim Dimension of Krylov subspace (if using Krylov method)
 * @return Vector of vectors containing time correlation values for each operator pair
 */
std::vector<std::vector<Complex>> compute_multiple_time_correlations(
    std::function<void(const Complex*, Complex*, int)> H,
    const std::vector<std::function<void(const Complex*, Complex*, int)>>& operators_1,
    const std::vector<std::function<void(const Complex*, Complex*, int)>>& operators_2,
    const ComplexVector& state,
    uint64_t N,
    double t_max,
    double dt,
    uint64_t time_evolution_method = 1,
    uint64_t taylor_order = 100,
    uint64_t krylov_dim = 30
);

/**
 * Compute time-dependent correlations using pre-evolved time evolution operator
 * Useful when the time evolution operator is already constructed
 * 
 * @param U_t Time evolution operator U(t) = exp(-iHt)
 * @param operators_1 Vector of first operators (applied at t=0)
 * @param operators_2 Vector of second operators (applied at time t)
 * @param state Initial quantum state
 * @param N Dimension of the Hilbert space
 * @param num_steps Number of time steps
 * @return Vector of vectors containing time correlation values for each operator pair
 */
std::vector<std::vector<Complex>> compute_time_correlations_with_U_t(
    std::function<void(const Complex*, Complex*, int)> U_t,
    const std::vector<std::function<void(const Complex*, Complex*, int)>>& operators_1,
    const std::vector<std::function<void(const Complex*, Complex*, int)>>& operators_2,
    const ComplexVector& state,
    uint64_t N,
    uint64_t num_steps
);

/**
 * Incremental version that writes time correlation data during computation
 * More memory efficient for long time evolutions
 * 
 * @param U_t Time evolution operator U(t) = exp(-iHt)
 * @param operators_1 Vector of first operators (applied at t=0)
 * @param operators_2 Vector of second operators (applied at time t)
 * @param state Initial quantum state
 * @param N Dimension of the Hilbert space
 * @param num_steps Number of time steps
 * @param dt Time step
 * @param output_files Vector of open output file streams for writing results
 */
void compute_time_correlations_incremental(
    std::function<void(const Complex*, Complex*, int)> U_t,
    const std::vector<std::function<void(const Complex*, Complex*, int)>>& operators_1,
    const std::vector<std::function<void(const Complex*, Complex*, int)>>& operators_2,
    const ComplexVector& state,
    uint64_t N,
    uint64_t num_steps,
    double dt,
    std::vector<std::ofstream>& output_files
);

/**
 * Compute dynamical susceptibility (spectral function) from time correlation
 * S(ω) = ∫ dt e^(iωt) C(t)
 * 
 * @param time_correlation Time-dependent correlation function C(t)
 * @param dt Time step
 * @param omega_min Minimum frequency
 * @param omega_max Maximum frequency
 * @param num_omega Number of frequency points
 * @param eta Broadening parameter
 * @param use_lorentzian Use Lorentzian (true) or Gaussian (false) broadening
 * @return SpectralFunctionData structure with frequencies and spectral function
 */
SpectralFunctionData compute_spectral_function(
    const std::vector<Complex>& time_correlation,
    double dt,
    double omega_min,
    double omega_max,
    uint64_t num_omega,
    double eta = 0.1,
    bool use_lorentzian = false
);

/**
 * Compute dynamical correlation using Operator objects (higher-level interface)
 * 
 * @param H Hamiltonian operator function
 * @param state Initial quantum state
 * @param operators_1 Vector of first Operator objects (applied at t=0)
 * @param operators_2 Vector of second Operator objects (applied at time t)
 * @param operator_names Names corresponding to each operator pair
 * @param N Dimension of the Hilbert space
 * @param output_dir Output directory for results
 * @param label Label for output files (e.g., sample number, temperature)
 * @param t_max Maximum evolution time
 * @param dt Time step
 * @param krylov_dim Dimension of Krylov subspace for time evolution
 */
void compute_operator_dynamics(
    std::function<void(const Complex*, Complex*, int)> H,
    const ComplexVector& state,
    const std::vector<Operator>& operators_1,
    const std::vector<Operator>& operators_2,
    const std::vector<std::string>& operator_names,
    uint64_t N,
    const std::string& output_dir,
    const std::string& label,
    double t_max = 100.0,
    double dt = 0.01,
    uint64_t krylov_dim = 30
);

#endif // DYNAMICS_H
