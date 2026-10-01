// ftlm.h - the continued-fraction spectral function of a Lanczos tridiagonal (P2.5 moves it
// to dynamics/cf.h; the FTLM thermodynamics live in include/ed/thermal/ftlm_kernel.h).

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
