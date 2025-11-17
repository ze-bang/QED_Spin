// TPQ.h - Thermal Pure Quantum state implementation

#ifndef TPQ_H
#define TPQ_H

#include <iostream>
#include <complex>
#include <vector>
#include <functional>
#include <random>
#include <cmath>
#include "../core/blas_lapack_wrapper.h"
#include <fstream>
#include <iomanip>
#include <algorithm>
#include <string>
#include <ctime>
#include <chrono>
#include <sys/stat.h>
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <Eigen/Sparse>
#include <unsupported/Eigen/MatrixFunctions>
#include "observables.h"
#include "../core/blas_lapack_wrapper.h"
#include "dynamics.h"
#include <memory>


#define GET_VARIABLE_NAME(Variable) (#Variable)

/**
 * Generate a random normalized vector for TPQ initial state
 * 
 * @param N Dimension of the Hilbert space
 * @param seed Random seed to use
 * @return Random normalized vector
 */
ComplexVector generateTPQVector(int N, uint64_t seed);
/**
 * Create directory if it doesn't exist
 */
bool ensureDirectoryExists(const std::string& path);
/**
 * Calculate energy and variance for a TPQ state
 * 
 * @param H Hamiltonian operator function
 * @param v Current TPQ state vector
 * @param N Dimension of the Hilbert space
 * @return Pair of energy and variance
 */
std::pair<double, double> calculateEnergyAndVariance(
    std::function<void(const Complex*, Complex*, int)> H,
    const ComplexVector& v,
    uint64_t N
);
std::vector<SingleSiteOperator> createSzOperators(int num_sites, float spin_length);
std::vector<SingleSiteOperator> createSxOperators(int num_sites, float spin_length);
std::vector<SingleSiteOperator> createSyOperators(int num_sites, float spin_length);

std::pair<std::vector<Complex>, std::vector<Complex>> calculateSzandSz2(
    const ComplexVector& tpq_state,
    uint64_t num_sites,
    float spin_length,
    const std::vector<SingleSiteOperator>& Sz_ops,
    uint64_t sublattice_size
);

Complex calculateSpm_onsite(
    const ComplexVector& tpq_state,
    uint64_t num_sites,
    float spin_length,
    const std::vector<SingleSiteOperator>& Spm_ops,
    uint64_t sublattice_size
);

std::pair<std::vector<DoubleSiteOperator>, std::vector<DoubleSiteOperator>> createDoubleSiteOperators(int num_sites, float spin_length);

std::pair<std::vector<SingleSiteOperator>, std::vector<SingleSiteOperator>> createSingleOperators_pair(int num_sites, float spin_length);


std::pair<std::vector<Complex>, std::vector<Complex>> calculateSzzSpm(
    const ComplexVector& tpq_state,
    uint64_t num_sites,
    float spin_length,
    std::pair<std::vector<DoubleSiteOperator>, std::vector<DoubleSiteOperator>> double_site_ops,
    uint64_t sublattice_size
);

std::tuple<std::vector<Complex>, std::vector<Complex>, std::vector<Complex>, std::vector<Complex>> calculateSzzSpm(
    const ComplexVector& tpq_state,
    uint64_t num_sites,
    float spin_length,
    std::pair<std::vector<SingleSiteOperator>, std::vector<SingleSiteOperator>> double_site_ops,
    uint64_t sublattice_size
);

/**
 * Write TPQ data to file
 */
void writeTPQData(const std::string& filename, double inv_temp, double energy, 
                 double variance, double norm, uint64_t step);

/**
 * Read TPQ data from file
 */
bool readTPQData(const std::string& filename, uint64_t step, double& energy, 
                double& temp, double& specificHeat);

/**
 * Save the current TPQ state to a file
 * 
 * @param tpq_state TPQ state vector to save (in fixed-Sz or full basis)
 * @param filename Name of the file to save to
 * @param fixed_sz_op Optional FixedSzOperator - if provided, transforms to full basis before saving
 * @return True if successful
 */
bool save_tpq_state(const ComplexVector& tpq_state, const std::string& filename, 
                    class FixedSzOperator* fixed_sz_op = nullptr);

/**
 * Load a TPQ state from a file
 * 
 * @param tpq_state TPQ state vector to load into
 * @param filename Name of the file to load from
 * @return True if successful
 */
bool load_tpq_state(ComplexVector& tpq_state, const std::string& filename);

/**
 * Load eigenvector data from a raw binary file
 * 
 * @param tpq_state TPQ state vector to load into
 * @param filename Name of the file to load from
 * @param N Expected size of the vector
 * @return True if successful
 */
bool load_raw_data(ComplexVector& tpq_state, const std::string& filename, uint64_t N);

/**
 * Compute spin expectations (S^+, S^-, S^z) at each site using a TPQ state
 * 
 * @param tpq_state The TPQ state vector
 * @param num_sites Number of lattice sites
 * @param spin_l Spin value (e.g., 0.5 for spin-1/2)
 * @param output_file Output file path
 * @param print_output Whether to print results to console
 * @return Vector of spin expectation values organized as [site][S+,S-,Sz]
 */
std::vector<std::vector<Complex>> compute_spin_expectations_from_tpq(
    const ComplexVector& tpq_state,
    uint64_t num_sites,
    float spin_l,
    const std::string& output_file = "",
    bool print_output = true
);

void writeFluctuationData(
    const std::string& flct_file,
    const std::vector<std::string>& spin_corr,
    double inv_temp,
    const ComplexVector& tpq_state,
    uint64_t num_sites,
    float spin_length,
    const std::vector<SingleSiteOperator>& Sx_ops,
    const std::vector<SingleSiteOperator>& Sy_ops,
    const std::vector<SingleSiteOperator>& Sz_ops,
    const std::pair<std::vector<SingleSiteOperator>, std::vector<SingleSiteOperator>>& double_site_ops,
    uint64_t sublattice_size,
    uint64_t step
);

/**
 * Get a TPQ state at a specific inverse temperature by loading the closest available state
 * 
 * @param tpq_dir Directory containing TPQ data
 * @param sample TPQ sample index
 * @param target_beta Target inverse temperature
 * @param N Dimension of Hilbert space
 * @return TPQ state vector at the specified temperature
 */
ComplexVector get_tpq_state_at_temperature(
    const std::string& tpq_dir,
    uint64_t sample,
    double target_beta,
    uint64_t N
);

/**
 * Initialize TPQ output files with appropriate headers
 * 
 * @param dir Directory for output files
 * @param sample Current sample index
 * @param sublattice_size Size of sublattice for measurements
 * @return Tuple of filenames (ss_file, norm_file, flct_file, spin_corr)
 */
std::tuple<std::string, std::string, std::string, std::vector<std::string>> initializeTPQFiles(
    const std::string& dir,
    uint64_t sample,
    uint64_t sublattice_size
);

/**
 * Standard TPQ (microcanonical) implementation
 * 
 * @param H Hamiltonian operator function
 * @param N Dimension of the Hilbert space (fixed-Sz dimension if using fixed-Sz)
 * @param max_iter Maximum number of iterations
 * @param num_samples Number of random samples
 * @param temp_interval Interval for calculating physical quantities
 * @param eigenvalues Optional output vector for final state energies
 * @param dir Output directory
 * @param compute_spectrum Whether to compute spectrum
 * @param fixed_sz_op Optional FixedSzOperator - if provided, transforms states to full basis before saving
 */
void microcanonical_tpq(
    std::function<void(const Complex*, Complex*, int)> H,
    uint64_t N, 
    uint64_t max_iter,
    uint64_t num_samples,
    uint64_t temp_interval,
    std::vector<double>& eigenvalues,
    std::string dir = "",
    bool compute_spectrum = false,
    double LargeValue = 1e5,
    bool compute_observables = false,
    std::vector<Operator> observables = {},
    std::vector<std::string> observable_names = {},
    double omega_min = -20.0,
    double omega_max = 20.0,
    uint64_t num_points = 10000,
    double t_end = 50.0,
    double dt = 0.01,
    float spin_length = 0.5,
    bool measure_sz = false,
    uint64_t sublattice_size = 1,
    uint64_t num_sites = 16,
    class FixedSzOperator* fixed_sz_op = nullptr
);

// Canonical TPQ using imaginary-time propagation e^{-βH} |r>
inline void imaginary_time_evolve_tpq_taylor(
    std::function<void(const Complex*, Complex*, int)> H,
    ComplexVector& state,
    uint64_t N,
    double delta_beta,
    uint64_t n_max = 50,
    bool normalize = true
);

void canonical_tpq(
    std::function<void(const Complex*, Complex*, int)> H,
    uint64_t N,
    double beta_max,
    uint64_t num_samples,
    uint64_t temp_interval,
    std::vector<double>& energies,
    std::string dir = "",
    double delta_beta = 0.1,
    uint64_t taylor_order = 50,
    bool compute_observables = false,
    std::vector<Operator> observables = {},
    std::vector<std::string> observable_names = {},
    double omega_min = -20.0,
    double omega_max = 20.0,
    uint64_t num_points = 10000,
    double t_end = 50.0,
    double dt = 0.01,
    float spin_length = 0.5,
    bool measure_sz = false,
    uint64_t sublattice_size = 1,
    uint64_t num_sites = 16,
    class FixedSzOperator* fixed_sz_op = nullptr
);

/**
 * Compute dynamical correlations for TPQ using Krylov method
 * Delegates to the general dynamics module with TPQ-specific file naming
 */
void computeDynamicCorrelationsKrylov(
    std::function<void(const Complex*, Complex*, int)> H,
    const ComplexVector& tpq_state,
    const std::vector<Operator>& operators_1,
    const std::vector<Operator>& operators_2,
    const std::vector<std::string>& operator_names,
    uint64_t N,
    const std::string& dir,
    uint64_t sample,
    double inv_temp,
    double t_end,
    double dt,
    uint64_t krylov_dim
);

/**
 * Compute observable dynamics for TPQ with legacy interface
 */
void computeObservableDynamics_U_t(
    std::function<void(const Complex*, Complex*, int)> U_t,
    const ComplexVector& tpq_state,
    const std::vector<Operator>& observables_1,
    const std::vector<Operator>& observables_2,
    const std::vector<std::string>& observable_names,
    uint64_t N,
    const std::string& dir,
    uint64_t sample,
    double inv_temp,
    double t_end,
    double dt
);

#endif // TPQ_H