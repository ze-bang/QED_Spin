// dynamics.h - General quantum dynamics computation module
// This module provides time evolution and dynamical correlation function computation
// that can be used with any quantum state (not exclusive to TPQ)

#pragma once

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
 * Krylov-based time evolution using the short-iteration Lanczos method.
 *
 * Only integrator kept after the minimalist-architecture rev (May 2026):
 * the alternative integrators (Taylor / Chebyshev / RK4 / adaptive /
 * imaginary-time Taylor) and their lambda-factory wrapper
 * `create_time_evolution_operator` had no remaining callers outside the
 * equally-retired legacy TPQ spectral-function family. All live dynamical
 * spectra flow through `ed::observables::time_evolution_correlator`
 * (Krylov-only) or the continued-fraction / KPM spectral functions in
 * ftlm.cpp. To re-introduce, copy the implementations from git history.
 *
 * @param H Hamiltonian operator function
 * @param state Current state vector (modified in-place)
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
