// ltlm.h - Low Temperature Lanczos Method implementation
// Specialized for low temperature thermodynamics using ground state projection

#pragma once

#include <ed/core/solver_defaults.h>

#include <iostream>
#include <complex>
#include <vector>
#include <functional>
#include <random>
#include <cmath>
#include <algorithm>
#include <ed/core/blas_lapack_wrapper.h>
#include <ed/core/construct_ham.h>
#include <ed/solvers/ftlm.h>

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
    uint64_t krylov_dim = 200;              // Dimension of Krylov subspace for thermodynamics
    uint64_t ground_state_krylov = 100;     // Krylov dimension for finding ground state
    uint64_t num_samples = 1;               // Usually 1 for LTLM (ground state is deterministic)
    uint64_t max_iterations = 1000;         // Maximum Lanczos iterations
    double tolerance = 1e-12;          // Convergence tolerance for Lanczos
    bool full_reorthogonalization = ed::defaults::kThermalFullReorth;
    uint64_t reorth_frequency = 10;         // Frequency of reorthogonalization (if not full)
    uint64_t random_seed = 0;      // Random seed (0 = use random_device) for initial state
    bool store_intermediate = false;   // Store intermediate data for debugging
    bool compute_error_bars = false;   // Compute standard error (only useful if num_samples > 1)
    bool use_exact_ground_state = false; // If true and ground state eigenvector provided, use it
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
    uint64_t N,
    uint64_t krylov_dim,
    double tolerance,
    bool full_reorth,
    uint64_t reorth_freq,
    ComplexVector& ground_state
);

// NOTE (Consolidation Family 1): the LTLM *thermodynamics* driver
// ``low_temperature_lanczos`` was removed. Its estimator seeded a second
// Lanczos from |0> and summed the GS-local density of states, not the
// thermal trace, so it stayed pinned near E0 at every T. Since LTLM
// thermodynamics reduces exactly to the FTLM trace for any function of H,
// all thermodynamics now routes through ``finite_temperature_lanczos`` /
// ``ftlm_kernel``. The connected static response below (⟨OH⟩-⟨O⟩⟨H⟩), which
// probes an operator that does NOT commute with H, is genuinely LTLM-only
// and is retained. See CONSOLIDATION_PLAN.md Family 1.

/**
 * @brief Compute the connected thermal-expansion covariance with LTLM.
 *
 * Evaluates the low-temperature estimator
 *     (⟨OH⟩ - ⟨O⟩⟨H⟩) / T²
 * by retaining the lowest LTLM Ritz states from a single outer Lanczos run.
 *
 * The returned ``expectation`` dataset stores the thermal-expansion
 * coefficient itself, while ``variance`` stores the raw connected covariance
 * ⟨δO δH⟩ and ``susceptibility`` stores ⟨δO δH⟩ / T.
 *
 * Parameter mapping for the legacy LTLM knobs:
 * - ``ground_state_krylov`` sets the outer Lanczos dimension floor.
 * - ``krylov_dim`` sets how many lowest Ritz states are retained.
 */
StaticResponseResults compute_connected_qh_response_ltlm(
    std::function<void(const Complex*, Complex*, int)> H,
    std::function<void(const Complex*, Complex*, int)> O,
    uint64_t N,
    const LTLMParameters& params,
    double temp_min,
    double temp_max,
    uint64_t num_temp_bins,
    const std::string& output_dir = ""
);
