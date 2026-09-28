// ltlm.h - the LTLM parameter block (bound in Python; LTLM thermodynamics runs through the FTLM trace).

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

