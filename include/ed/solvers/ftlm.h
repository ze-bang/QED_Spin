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
    uint64_t krylov_dim = 400;              // Dimension of Krylov subspace
    uint64_t num_samples = 40;              // Number of random initial states
    double tolerance = 1e-10;          // Convergence tolerance for Lanczos
    bool full_reorthogonalization = ed::defaults::kThermalFullReorth;
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
    bool full_reorthogonalization = ed::defaults::kThermalFullReorth;
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
 * @brief Explicit-temperatures FTLM driver.
 *
 * Identical to the (temp_min, temp_max, num_temp_bins) overload but the
 * temperature grid is supplied verbatim by the caller, so the returned
 * curves are index-aligned with whatever axis (linear / logarithmic /
 * arbitrary) the caller requested. This is the overload the unified
 * thermal backend lane forwards to with ``1.0 / beta`` so the reported
 * temperatures match the computed observables exactly.
 *
 * @param H Hamiltonian matrix-vector product function
 * @param N Hilbert space dimension
 * @param params FTLM parameters
 * @param temperatures Temperature grid (must be non-empty and all > 0)
 * @param output_dir Directory for output files
 */
FTLMResults finite_temperature_lanczos(
    std::function<void(const Complex*, Complex*, int)> H,
    uint64_t N,
    const FTLMParameters& params,
    const std::vector<double>& temperatures,
    const std::string& output_dir = ""
);

// Phase 4 (matvec-unification): MatVecOperator-taking overload. The body
// is a single ed::matvec::as_apply_function() forward; see lanczos.h for
// the design notes.
inline FTLMResults finite_temperature_lanczos(
    const ed::matvec::MatVecOperator& H_op,
    uint64_t N,
    const FTLMParameters& params,
    double temp_min,
    double temp_max,
    uint64_t num_temp_bins,
    const std::string& output_dir = "")
{
    return finite_temperature_lanczos(
        ed::matvec::as_apply_function(H_op),
        N, params, temp_min, temp_max, num_temp_bins, output_dir);
}

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

// Family-3 retirement (audit 2026-07-31): the Gen-1 dynamical-DSSF entry
// points (compute_dynamical_response_thermal, compute_dynamical_correlation,
// compute_dynamical_correlation_state_cf, and the earlier single-state
// compute_dynamical_response overload) are deleted. Use the backend-generic
// ed::observables::ftlm_dynamical_kernel_via_backend[_multitemp]
// (cf_dynamical.h) for random-sample S(q, omega), and
// ed::observables::cf_spectral_from_vector (cf_spectral_kernel.h) for the
// |psi> -> S(omega) continued-fraction primitive.

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
 * The two-operator entry point `compute_dynamical_correlation_state(O1, O2, ...)`
 * was retired in the minimalist-architecture rev (May 2026); use
 * `ed::observables::cf_dynamical_correlator` for the self-correlator case
 * (O1==O2) and call it twice for the cross-correlator case.
 */

// save_dynamical_response_results was retired in the minimalist-
// architecture rev (May 2026): no external callers. Persist directly via
// HDF5IO::saveDynamicalResponseFull instead.

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
 * @brief Compute the thermal-expansion covariance β²⟨δO δH⟩.
 *
 * The returned ``expectation`` dataset is
 *     (⟨OH⟩ - ⟨O⟩⟨H⟩) / T² = ∂_T⟨O⟩,
 * evaluated within each FTLM sample on a log-spaced temperature grid.
 * The returned ``variance`` dataset stores the raw connected covariance
 * ⟨δO δH⟩ before division by T².
 */
StaticResponseResults compute_connected_qh_response(
    std::function<void(const Complex*, Complex*, int)> H,
    std::function<void(const Complex*, Complex*, int)> O,
    uint64_t N,
    const StaticResponseParameters& params,
    double temp_min,
    double temp_max,
    uint64_t num_temp_bins,
    const std::string& output_dir = ""
);

// save_static_response_results was retired in the minimalist-
// architecture rev (May 2026): no external callers. Persist directly via
// HDF5IO::saveStaticResponse instead.

// ============================================================================
// TEMPERATURE-INDEPENDENT SPECTRAL DECOMPOSITION -- RETIRED (Jul 2026)
// ============================================================================
//
// `LanczosSpectralData`, `compute_lanczos_spectral_data`,
// `compute_spectral_function_from_lanczos_data`,
// `compute_dynamical_correlation_state_multi_temperature`, and the plain
// `compute_dynamical_correlation_multi_sample_multi_temperature` wrapper were
// deleted in the consolidation sweep (Family 3 follow-up): all multi-T DSSF
// traffic now flows through the backend-generic
// `ed::observables::ftlm_dynamical_kernel_via_backend` (in-memory API) or the
// multi-operator core below (CLI workflow). The MPI `_comm` variants in
// `ftlm_dist.h` remain the distributed entry points.

/**
 * @brief Multi-operator extension of FTLM dynamical correlation.
 *
 * Computes spectral functions for many operator pairs (O1[p], O2[p]) sharing
 * the SAME Hamiltonian Lanczos chain per random sample. The expensive outer
 * Lanczos run on H from |r> is performed exactly once per sample (instead of
 * once per (sample, operator) pair as in the per-pair entry point), and the
 * reconstructed Ritz eigenstates |psi_i> are cached and reused across pairs.
 *
 * Each pair p still triggers its own per-Ritz-state inner Lanczos (starting
 * from O2_p|psi_i>) and per-pair spectral accumulation; that work is not
 * shareable because it depends on the operators.
 *
 * Result: results[p] is the temperature->spectral map for the p-th pair,
 * structurally identical to what `compute_dynamical_correlation_multi_sample_multi_temperature`
 * would produce if called individually for that pair (modulo non-determinism
 * from differently-seeded sub-Lanczos calls; the outer sample seed and Krylov
 * basis are bit-identical to a single per-pair invocation).
 *
 * @param H Hamiltonian matvec
 * @param O1_list Operators on the bra side (size = number of pairs)
 * @param O2_list Operators on the ket side (same size)
 * @param N Hilbert dimension
 * @param params FTLM parameters
 * @param omega_min,omega_max,num_omega_bins Frequency grid
 * @param temperatures Temperatures
 * @param energy_shift Ground state energy shift (0 -> auto)
 * @param output_dir Optional debug output directory
 * @return Vector indexed by operator pair, each element a per-T results map
 */
std::vector<std::map<double, DynamicalResponseResults>>
compute_dynamical_correlation_multi_operator_multi_temperature(
    std::function<void(const Complex*, Complex*, int)> H,
    const std::vector<std::function<void(const Complex*, Complex*, int)>>& O1_list,
    const std::vector<std::function<void(const Complex*, Complex*, int)>>& O2_list,
    uint64_t N,
    const DynamicalResponseParameters& params,
    double omega_min,
    double omega_max,
    uint64_t num_omega_bins,
    const std::vector<double>& temperatures,
    double energy_shift = 0.0,
    const std::string& output_dir = ""
);

// ============================================================================
// GROUND STATE DYNAMICAL STRUCTURE FACTOR (CONTINUED FRACTION METHOD)
// ============================================================================

/**
 * @brief Parameters for ground state DSSF calculation
 */
struct GroundStateDSSFParameters {
    uint64_t krylov_dim = 500;              // Krylov dimension for spectral decomposition
    double tolerance = 1e-12;               // Lanczos convergence tolerance
    bool full_reorthogonalization = true;   // Use full reorthogonalization (recommended for accuracy)
    uint64_t reorth_frequency = 5;          // Reorthogonalization frequency (if not full)
    double broadening = 0.05;               // Lorentzian broadening η
    double omega_min = -5.0;                // Minimum frequency
    double omega_max = 10.0;                // Maximum frequency
    uint64_t num_omega_points = 2000;       // Number of frequency points
    bool use_continued_fraction = true;     // Use continued fraction (faster) vs eigendecomposition
    bool compute_both_directions = true;    // Compute both positive and negative frequency parts
};

/**
 * @brief Evaluate spectral function using continued fraction representation
 * 
 * This is the OPTIMAL method for ground state (T=0) dynamical correlations.
 * The Green's function G(z) = ⟨φ|(z-H)⁻¹|φ⟩ is computed as a continued fraction:
 * 
 *   G(z) = ||φ||² / (z - α₀ - β₁²/(z - α₁ - β₂²/(z - α₂ - ...)))
 * 
 * where α_n, β_n are the Lanczos coefficients starting from |φ⟩ = O|ψ₀⟩.
 * 
 * The spectral function is then S(ω) = -Im[G(ω + iη)] / π
 * 
 * Advantages over eigenvalue decomposition:
 * - No need to store/diagonalize tridiagonal matrix
 * - Numerically stable bottom-up evaluation
 * - O(M) per frequency point vs O(M²) for eigendecomposition
 * 
 * @param alpha Diagonal elements of Lanczos tridiagonal matrix
 * @param beta Off-diagonal elements (β[0] = 0 is not used, β[1]...β[M-1])
 * @param omega_grid Frequency grid for evaluation
 * @param broadening Lorentzian broadening parameter η > 0
 * @param norm_sq Squared norm ||O|ψ₀⟩||² for proper normalization
 * @return Spectral function values at each frequency
 */
std::vector<double> continued_fraction_spectral_function(
    const std::vector<double>& alpha,
    const std::vector<double>& beta,
    const std::vector<double>& omega_grid,
    double broadening,
    double norm_sq
);

/**
 * @brief Compute ground state dynamical structure factor S(q,ω)
 * 
 * Computes the zero-temperature dynamical correlation function:
 *   S(ω) = -1/π Im⟨0|O†(ω + E₀ - H + iη)⁻¹O|0⟩
 * 
 * This is the gold standard method for ground state dynamics in ED:
 * 1. Apply operator O to ground state: |φ⟩ = O|0⟩
 * 2. Run Lanczos starting from |φ⟩ to get tridiagonal matrix
 * 3. Evaluate continued fraction for each frequency ω
 * 
 * Memory efficient: Only stores 2-3 vectors at a time.
 * No random sampling: Exact result for given ground state.
 * 
 * @param H Hamiltonian matrix-vector product function
 * @param O Operator matrix-vector product function (e.g., S(q))
 * @param ground_state The ground state |0⟩ (must be normalized)
 * @param ground_state_energy Ground state energy E₀
 * @param N Hilbert space dimension
 * @param params Parameters for calculation
 * @return DynamicalResponseResults with spectral function S(ω)
 */
DynamicalResponseResults compute_ground_state_dssf(
    std::function<void(const Complex*, Complex*, int)> H,
    std::function<void(const Complex*, Complex*, int)> O,
    const ComplexVector& ground_state,
    double ground_state_energy,
    uint64_t N,
    const GroundStateDSSFParameters& params
);

/**
 * @brief Compute ground state two-operator correlation S_{O1,O2}(ω)
 * 
 * Generalizes compute_ground_state_dssf to different operators:
 *   S_{O1,O2}(ω) = -1/π Im⟨0|O₁†(ω + E₀ - H + iη)⁻¹O₂|0⟩
 * 
 * For structure factor: O₁ = O₂ = S(q) gives standard S(q,ω)
 * For cross-correlations: Different O₁, O₂ give off-diagonal responses
 * 
 * @param H Hamiltonian matrix-vector product function
 * @param O1 First operator O₁ (conjugated)
 * @param O2 Second operator O₂ (applied to ground state)
 * @param ground_state The ground state |0⟩
 * @param ground_state_energy Ground state energy E₀
 * @param N Hilbert space dimension
 * @param params Parameters for calculation
 * @return DynamicalResponseResults with cross-correlation spectral function
 */
DynamicalResponseResults compute_ground_state_cross_correlation(
    std::function<void(const Complex*, Complex*, int)> H,
    std::function<void(const Complex*, Complex*, int)> O1,
    std::function<void(const Complex*, Complex*, int)> O2,
    const ComplexVector& ground_state,
    double ground_state_energy,
    uint64_t N,
    const GroundStateDSSFParameters& params
);

/**
 * @brief Compute ground state DSSF in a *different* Hilbert sector than |0>.
 *
 * Audit item #1 (full): for fixed-Sz Hamiltonians the standard scalar
 * ground-state DSSF
 *
 *     S_{O1, O2}(omega) = -1/pi Im <0| O1^dag (omega + E0 - H + i eta)^{-1} O2 |0>
 *
 * vanishes whenever O1, O2 change the magnetisation by +-1 (e.g. S+, S-),
 * because the resolvent (omega - H)^{-1} only has matrix elements within
 * the same Sz sector and O2|0> sits in a different sector than the
 * source <0|. This routine evaluates the legitimate cross-sector
 * spectrum by:
 *   * applying O2 to |0> to obtain |phi> = O2|0> in the *destination*
 *     sector (dim = dim_inner);
 *   * applying O1^dag to |0> in the same destination sector to obtain |chi>;
 *   * running Lanczos with H_inner (the Hamiltonian restricted to the
 *     destination sector) starting from a normalised |phi>;
 *   * forming S(omega) from the Krylov tridiagonal projection of |phi>
 *     and |chi>, with a frequency shift by E0 to align onto the
 *     standard energy axis.
 *
 * For O1 == O2 this reduces to a self-correlator and the implementation
 * collapses to the cheaper continued-fraction path used by
 * `compute_ground_state_dssf`. For O1 != O2 the spectral weight is
 * evaluated via the Ritz eigendecomposition of the tridiagonal so that
 * the cross-coefficient <chi|n_ritz><n_ritz|phi> can be reconstructed.
 *
 * @param H_inner   Hamiltonian matvec on the *destination* sector, dim = dim_inner.
 * @param O1_dagger_apply  applies O1^dag to a vector of size dim_outer
 *                          (the source sector), producing dim_inner output.
 * @param O2_apply         applies O2 to a vector of size dim_outer,
 *                          producing dim_inner output.
 * @param ground_state     Normalised ground state |0> in the source sector
 *                          (size dim_outer).
 * @param ground_state_energy E0.
 * @param dim_outer        Source sector dimension (size of ground_state).
 * @param dim_inner        Destination sector dimension.
 * @param params           Parameters mirror compute_ground_state_dssf.
 *
 * The result fills `spectral_function` (real part) and `spectral_function_imag`
 * (imaginary part of the cross-correlation, zero in the diagonal case).
 */
DynamicalResponseResults compute_ground_state_dssf_cross_sector(
    std::function<void(const Complex*, Complex*, int)> H_inner,
    std::function<void(const Complex*, Complex*, int)> O1_dagger_apply,
    std::function<void(const Complex*, Complex*, int)> O2_apply,
    const ComplexVector& ground_state,
    double ground_state_energy,
    uint64_t dim_outer,
    uint64_t dim_inner,
    const GroundStateDSSFParameters& params
);

// load_ground_state_from_file was retired in the minimalist-architecture
// rev (May 2026): no external callers. Ground states are read from
// the unified HDF5 store via HDF5IO::loadEigenvector(/0/) directly.
