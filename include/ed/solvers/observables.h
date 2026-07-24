#pragma once

#include <iostream>
#include <complex>
#include <vector>
#include <functional>
#include <cmath>
#include <ed/core/blas_lapack_wrapper.h>
#include <ed/core/construct_ham.h>
#include <ed/solvers/ftlm.h>
#include <algorithm>

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
    uint64_t num_points = 100   // Number of temperature points
);

/**
 * @brief Connected Q–H thermal expansion from an exact energy spectrum.
 *
 * For each exact eigenpair (E_n, q_n) with q_n = ⟨n|Q|n⟩ in the H eigenbasis,
 * forms the canonical averages and returns
 *     α_Q(T) = (⟨QH⟩ − ⟨Q⟩⟨H⟩) / T²
 * with zero sample errors (``total_samples = 1``).
 */
StaticResponseResults compute_connected_qh_from_spectrum(
    const std::vector<double>& eigenvalues,
    const std::vector<double>& q_expectations,
    double temp_min,
    double temp_max,
    uint64_t num_temp_bins
);
