// ftlm.h - Finite Temperature Lanczos Method implementation
// Computes thermodynamic properties without full spectrum diagonalization

#pragma once


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

using Complex = std::complex<double>;
using ComplexVector = std::vector<Complex>;

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
 * Spectral function S(w) = -Im G(w + i*eta) / pi of the continued fraction
 * G(z) = norm_sq / (z - a0 - b1^2 / (z - a1 - ...)) built from a Lanczos tridiagonal, evaluated bottom-up
 * (stable, O(M) per frequency). beta[0] is unused; norm_sq = ||O|psi>||^2.
 */
std::vector<double> continued_fraction_spectral_function(
    const std::vector<double>& alpha,
    const std::vector<double>& beta,
    const std::vector<double>& omega_grid,
    double broadening,
    double norm_sq
);
