// TPQ.h - Thermal Pure Quantum state implementation

#pragma once

#include <iostream>
#include <complex>
#include <vector>
#include <functional>
#include <random>
#include <cmath>
#include <ed/core/blas_lapack_wrapper.h>
#include <ed/core/thermal_types.h>       // ThermodynamicData (for compute_tpq_unified_thermo)
#include <ed/matvec/matvec.h>            // MatVecOperator + as_apply_function (Phase 4)
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
#include <ed/solvers/observables.h>
#include <ed/core/blas_lapack_wrapper.h>
#include <ed/solvers/dynamics.h>
#include <memory>


#define GET_VARIABLE_NAME(Variable) (#Variable)

#include <ed/solvers/tpq_seeding.h>
using ed::tpq_per_sample_seed;

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

// writeTPQData / readTPQData / readTPQDataHDF5 / save_tpq_state (the
// binary-sidecar / SS_rand*.dat reader-writer family) were retired in
// the minimalist-architecture rev (May 2026). All TPQ persistence now
// goes through HDF5IO::saveTPQState / loadTPQState (state vectors) and
// HDF5IO::appendTPQThermodynamics / loadTPQThermodynamics
// (thermodynamic trace). See src/solvers/cpu/TPQ.cpp for details.

/**
 * Load a TPQ state from a file
 * 
 * @param tpq_state TPQ state vector to load into
 * @param filename Name of the file to load from
 * @return True if successful
 */
bool load_tpq_state(ComplexVector& tpq_state, const std::string& filename);

/**
 * Load a TPQ state from a file with optional projection from full to reduced basis
 * 
 * @param tpq_state TPQ state vector to load into (will be in reduced basis if fixed_sz_op provided)
 * @param filename Name of the file to load from
 * @param fixed_sz_op Optional FixedSzOperator - if provided, projects from full to reduced basis
 * @param expected_reduced_dim Expected dimension of reduced basis
 * @return True if successful
 */
bool load_tpq_state(ComplexVector& tpq_state, const std::string& filename, 
                    class FixedSzOperator* fixed_sz_op, uint64_t expected_reduced_dim);

// load_raw_data was retired in the minimalist-architecture rev (May 2026):
// no callers; state persistence is HDF5-only.

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

// get_tpq_state_at_temperature was retired in the minimalist-architecture
// rev (May 2026): no callers; nearest-beta lookup is done in-place by the
// HDF5 loader.

/**
 * Find the lowest energy state from saved TPQ state files
 * Searches through tpq_state_*_beta=*.dat files for the highest beta value
 * 
 * @param tpq_dir Directory containing TPQ data
 * @param N Dimension of Hilbert space
 * @param out_sample Output parameter for sample number with highest beta (lowest energy)
 * @param out_beta Output parameter for beta value of the saved state
 * @param out_step Output parameter for step number (determined from SS_rand file)
 * @return True if a valid state was found
 */
bool find_lowest_energy_tpq_state(
    const std::string& tpq_dir,
    uint64_t N,
    uint64_t& out_sample,
    double& out_beta,
    uint64_t& out_step
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
 * @param continue_quenching If true, continue from a saved state instead of starting fresh
 * @param continue_sample Sample number to continue from (0 = auto-detect lowest energy)
 * @param continue_beta Beta value to continue from (0.0 = use saved beta from state file)
 * @param target_beta Target inverse temperature at which to stop iteration (default 1000.0)
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
    class FixedSzOperator* fixed_sz_op = nullptr,
    bool continue_quenching = false,
    uint64_t continue_sample = 0,
    double continue_beta = 0.0,
    double target_beta = 1000.0,
    uint64_t num_measure_points = 20,
    double measure_beta_min = 1.0,
    double measure_beta_max = 1000.0
);

// Phase 4 (matvec-unification): MatVecOperator-taking overload.
inline void microcanonical_tpq(
    const ed::matvec::MatVecOperator& H_op,
    uint64_t N, uint64_t max_iter, uint64_t num_samples,
    uint64_t temp_interval, std::vector<double>& eigenvalues,
    std::string dir = "", bool compute_spectrum = false,
    double LargeValue = 1e5, bool compute_observables = false,
    std::vector<Operator> observables = {},
    std::vector<std::string> observable_names = {},
    double omega_min = -20.0, double omega_max = 20.0,
    uint64_t num_points = 10000, double t_end = 50.0,
    double dt = 0.01, float spin_length = 0.5,
    bool measure_sz = false, uint64_t sublattice_size = 1,
    uint64_t num_sites = 16,
    class FixedSzOperator* fixed_sz_op = nullptr,
    bool continue_quenching = false, uint64_t continue_sample = 0,
    double continue_beta = 0.0, double target_beta = 1000.0,
    uint64_t num_measure_points = 20,
    double measure_beta_min = 1.0, double measure_beta_max = 1000.0)
{
    microcanonical_tpq(
        ed::matvec::as_apply_function(H_op),
        N, max_iter, num_samples, temp_interval, eigenvalues,
        std::move(dir), compute_spectrum, LargeValue, compute_observables,
        std::move(observables), std::move(observable_names),
        omega_min, omega_max, num_points, t_end, dt, spin_length,
        measure_sz, sublattice_size, num_sites, fixed_sz_op,
        continue_quenching, continue_sample, continue_beta, target_beta,
        num_measure_points, measure_beta_min, measure_beta_max);
}

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
    class FixedSzOperator* fixed_sz_op = nullptr,
    uint64_t num_measure_points = 20,
    double measure_beta_min = 1.0,
    double measure_beta_max = 1000.0
);

// Phase 4 (matvec-unification): MatVecOperator-taking overload.
inline void canonical_tpq(
    const ed::matvec::MatVecOperator& H_op,
    uint64_t N, double beta_max, uint64_t num_samples,
    uint64_t temp_interval, std::vector<double>& energies,
    std::string dir = "", double delta_beta = 0.1, uint64_t taylor_order = 50,
    bool compute_observables = false, std::vector<Operator> observables = {},
    std::vector<std::string> observable_names = {},
    double omega_min = -20.0, double omega_max = 20.0,
    uint64_t num_points = 10000, double t_end = 50.0,
    double dt = 0.01, float spin_length = 0.5,
    bool measure_sz = false, uint64_t sublattice_size = 1,
    uint64_t num_sites = 16,
    class FixedSzOperator* fixed_sz_op = nullptr,
    uint64_t num_measure_points = 20,
    double measure_beta_min = 1.0, double measure_beta_max = 1000.0)
{
    canonical_tpq(
        ed::matvec::as_apply_function(H_op),
        N, beta_max, num_samples, temp_interval, energies, std::move(dir),
        delta_beta, taylor_order, compute_observables,
        std::move(observables), std::move(observable_names),
        omega_min, omega_max, num_points, t_end, dt, spin_length,
        measure_sz, sublattice_size, num_sites, fixed_sz_op,
        num_measure_points, measure_beta_min, measure_beta_max);
}

// computeDynamicCorrelationsKrylov, computeObservableDynamics_U_t, and
// the older time_evolve_tpq_* / calculate_spectral_function_from_tpq*
// wrappers were retired in the minimalist-architecture rev (May 2026):
// none had external callers. The live TPQ dynamics path is
// ed::observables::time_evolution_correlator and the workflow drivers
// in src/cli/workflows.cpp.

/**
 * Convert TPQ results (SS_rand*.dat files) to unified thermodynamic format
 * 
 * Reads raw TPQ data from multiple sample files, computes averages and error bars,
 * then writes to the unified text format consistent with FTLM/LTLM/Hybrid methods.
 * 
 * @param tpq_dir Directory containing TPQ output files (SS_rand*.dat)
 * @param output_file Output file path for unified thermodynamic data
 * @param temp_min Minimum temperature for output (default 0.01)
 * @param temp_max Maximum temperature for output (default 100.0)
 * @param num_temp_bins Number of temperature bins (default 200)
 * @return true if successful, false otherwise
 */
bool convert_tpq_to_unified_thermo(
    const std::string& tpq_dir,
    const std::string& output_file,
    double temp_min = 0.01,
    double temp_max = 100.0,
    uint64_t num_temp_bins = 200
);

/**
 * @brief In-memory variant of ``convert_tpq_to_unified_thermo``.
 *
 * Reads the per-sample TPQ trajectory (β, E, σ²) from
 * ``<tpq_dir>/ed_results.h5`` (falling back to legacy
 * ``SS_rand*.dat`` text files), interpolates onto a common
 * temperature grid, averages across samples, and integrates to obtain
 * entropy and free energy. Returns the resulting
 * :class:`ThermodynamicData` directly so callers (notably
 * ``exact_diagonalization_core``'s mTPQ / cTPQ branches) can attach it
 * to ``EDResults`` without going through disk.
 *
 * Audit follow-up: this is the single source of truth for
 * post-processing TPQ trajectories into ``ThermodynamicData``. Both
 * the HDF5-writing variant (``convert_tpq_to_unified_thermo``) and the
 * unified ``thermal()`` entry point delegate to this function.
 *
 * Returns an empty ``ThermodynamicData`` (all vectors empty) when no
 * sample trajectories could be located -- callers should treat that
 * as "TPQ produced no usable data" and skip the sector.
 */
ThermodynamicData compute_tpq_unified_thermo(
    const std::string& tpq_dir,
    double temp_min = 0.01,
    double temp_max = 100.0,
    std::uint64_t num_temp_bins = 200
);

/**
 * Convenience wrapper for convert_tpq_to_unified_thermo that auto-generates output path
 * 
 * @param tpq_dir Directory containing TPQ output files (SS_rand*.dat)
 * @param num_samples Number of TPQ samples (used in output filename)
 * @param temp_min Minimum temperature for output (default 0.01)
 * @param temp_max Maximum temperature for output (default 100.0)
 * @param num_temp_points Number of temperature points (default 200)
 */
void convert_tpq_to_unified_thermodynamics(
    const std::string& tpq_dir,
    uint64_t num_samples,
    double temp_min = 0.01,
    double temp_max = 100.0,
    uint64_t num_temp_points = 200
);
