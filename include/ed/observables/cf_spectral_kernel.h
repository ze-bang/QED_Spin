#pragma once
// =============================================================================
// include/ed/observables/cf_spectral_kernel.h
//
// Continued-fraction spectral-function kernel --- `template<Backend,
// MatvecHFn, MatvecOFn>`. Wraps the unified Lanczos kernel in its
// "no basis storage" mode (keep_basis=false) plus the analytic
// continued-fraction evaluator. Produces S(omega) = -Im[G(omega + i*eta)] / pi
// where G(z) = <phi| (z - H + E_shift)^-1 |phi> and |phi> = O|psi> / ||O|psi>||.
//
// Same algorithmic content as the legacy
// `::compute_dynamical_correlation_state_cf` in
// `src/solvers/cpu/ftlm.cpp:1530`, but lifted out so any Backend can drive
// it: the only Backend-specific operations are the Lanczos tridiag build
// (already templated via `lanczos_kernel<Backend>`) and the |phi>
// preparation (axpy / nrm2 / scale / matvec --- all in `Backend`).
//
// Phase 2.5 of the Minimalist ED Collapse (May 2026).
// =============================================================================

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <vector>

#include <ed/core/blas_lapack_wrapper.h>
#include <ed/krylov/lanczos_kernel.h>
#include <ed/matvec/backend.h>
#include <ed/solvers/ftlm.h>   // for continued_fraction_spectral_function

namespace ed::observables {

using Complex = std::complex<double>;

struct CfSpectralOptions {
    std::size_t krylov_dim       = 200;
    double      broadening       = 0.05;
    /// If 0 (default) the kernel auto-detects via the smallest tridiag
    /// eigenvalue (matches the legacy ftlm.cpp behaviour).
    double      energy_shift     = 0.0;
    /// Convergence tolerance for the Lanczos tridiag build.
    double      tolerance        = 1e-12;
    /// Global problem dimension for the per-cycle Lanczos cap on
    /// distributed backends. 0 means "use local_n".
    std::uint64_t global_n       = 0;
    bool        verbose          = false;
};

struct CfSpectralResult {
    std::vector<double> frequencies;
    std::vector<double> spectral_function;
    std::size_t         tridiag_size = 0;
    double              phi_norm     = 0.0;
    double              energy_shift = 0.0;
};

// ----------------------------------------------------------------------------
// Main entry point. The two operator callables follow the matvec convention
// used by `lanczos_kernel`: `op(in, out, n)` performs out += / = O * in
// on backend memory of length n.
// ----------------------------------------------------------------------------
template <typename Backend, typename ApplyH, typename ApplyO>
CfSpectralResult cf_spectral_kernel(Backend&                   be,
                                    ApplyH&&                   apply_H,
                                    ApplyO&&                   apply_O,
                                    std::size_t                local_n,
                                    const Complex*             psi_seed,
                                    const std::vector<double>& omega_grid,
                                    const CfSpectralOptions&   opts)
{
    if (local_n == 0) {
        throw std::invalid_argument("cf_spectral_kernel: local_n == 0");
    }
    if (omega_grid.empty()) {
        throw std::invalid_argument("cf_spectral_kernel: empty frequency grid");
    }

    CfSpectralResult R;
    R.frequencies = omega_grid;

    // ------------------------------------------------------------------
    // |phi> = O|psi> / ||O|psi>||.
    // ------------------------------------------------------------------
    auto psi = be.make_zero_vector(local_n);
    be.copy(psi_seed, psi.get(), local_n);
    {
        const double n0 = be.nrm2(psi.get(), local_n);
        if (n0 < 1e-14) {
            R.spectral_function.assign(omega_grid.size(), 0.0);
            return R;
        }
        be.scale(Complex(1.0 / n0, 0.0), psi.get(), local_n);
    }

    auto phi = be.make_zero_vector(local_n);
    apply_O(psi.get(), phi.get(), local_n);
    const double phi_norm = be.nrm2(phi.get(), local_n);
    if (phi_norm < 1e-14) {
        R.spectral_function.assign(omega_grid.size(), 0.0);
        return R;
    }
    R.phi_norm = phi_norm;
    be.scale(Complex(1.0 / phi_norm, 0.0), phi.get(), local_n);

    // ------------------------------------------------------------------
    // Lanczos tridiag (no basis storage).
    // ------------------------------------------------------------------
    ed::krylov::LanczosKernelOptions kopts;
    kopts.max_iter      = opts.krylov_dim;
    kopts.reorth        = ed::krylov::ReorthPolicy::None;
    kopts.keep_basis    = false;
    kopts.breakdown_tol = opts.tolerance;
    kopts.dim_cap       = (opts.global_n > 0)
        ? static_cast<std::size_t>(opts.global_n)
        : local_n;
    auto kres = ed::krylov::lanczos_kernel(be, apply_H, local_n,
                                           phi.get(), kopts);
    std::vector<double> alpha = kres.alpha;
    std::vector<double> beta  = kres.beta;
    const std::size_t m = alpha.size();
    R.tridiag_size = m;
    if (m == 0) {
        R.spectral_function.assign(omega_grid.size(), 0.0);
        return R;
    }

    // ------------------------------------------------------------------
    // Energy shift: auto-detect via the smallest tridiag eigenvalue, or
    // honour the caller's override.
    // ------------------------------------------------------------------
    double E_shift = opts.energy_shift;
    if (std::abs(E_shift) < 1e-14) {
        std::vector<double> diag_copy = alpha;
        std::vector<double> offdiag(m > 1 ? m - 1 : 1, 0.0);
        for (std::size_t i = 0; i + 1 < m; ++i) offdiag[i] = beta[i + 1];
        const lapack_int info = LAPACKE_dstevd(LAPACK_COL_MAJOR, 'N',
            static_cast<lapack_int>(m),
            diag_copy.data(), offdiag.data(),
            nullptr, 1);
        if (info == 0 && !diag_copy.empty()) E_shift = diag_copy[0];
    }
    for (auto& a : alpha) a -= E_shift;
    R.energy_shift = E_shift;

    // ------------------------------------------------------------------
    // Continued-fraction evaluation. Reuses the existing host function
    // (omega grid is small; the loop is pure scalar arithmetic).
    // ------------------------------------------------------------------
    R.spectral_function = ::continued_fraction_spectral_function(
        alpha, beta, omega_grid, opts.broadening, phi_norm * phi_norm);
    return R;
}

// ----------------------------------------------------------------------------
// cf_spectral_from_vector --- continued-fraction kernel starting from a
// pre-built phi vector (no random seed, no |phi> = O|psi> step).
//
// Use when phi was computed externally (e.g. by applying a cross-sector
// observable to a ground state stored in a *different* sector basis): just
// pass phi in target-sector memory and the spectral weight is folded in via
// ||phi||^2, exactly as in the legacy ::compute_ground_state_dssf path. This
// is the kernel underneath the cross-irrep spectral walk in
// `workflows_spectral_streaming_symmetry_directory`.
//
// Phi is normalised before the Lanczos build, but ||phi||^2 is preserved as
// the spectral-function weight so the absolute amplitude of S(omega) is
// physically meaningful.
//
// ApplyH must matvec on the *target* sector (same dim as ``phi_seed``).
// ``E_shift`` is the energy reference (usually E_0 from the source-sector
// ground-state solve, so omega-axes line up with the standard
// S(Q, omega) convention). If ``opts.energy_shift == 0`` the kernel falls
// back to the auto-detect smallest tridiag eigenvalue, matching
// `cf_spectral_kernel`.
// ----------------------------------------------------------------------------
template <typename Backend, typename ApplyH>
CfSpectralResult cf_spectral_from_vector(Backend&                   be,
                                         ApplyH&&                   apply_H,
                                         std::size_t                local_n,
                                         const Complex*             phi_seed,
                                         const std::vector<double>& omega_grid,
                                         const CfSpectralOptions&   opts)
{
    if (local_n == 0) {
        throw std::invalid_argument(
            "cf_spectral_from_vector: local_n == 0");
    }
    if (omega_grid.empty()) {
        throw std::invalid_argument(
            "cf_spectral_from_vector: empty frequency grid");
    }

    CfSpectralResult R;
    R.frequencies = omega_grid;

    auto phi = be.make_zero_vector(local_n);
    // ``phi_seed`` is documented as a host-memory pointer: a vector
    // computed outside this kernel by applying a (possibly
    // cross-sector) observable to a state stored in another sector
    // basis. Use ``copy_from_host`` so the CUDA path actually stages
    // the seed H2D once and runs the rest of the CF Lanczos
    // device-resident; the CPU specialization is a plain memcpy.
    be.copy_from_host(phi_seed, phi.get(), local_n);
    const double phi_norm = be.nrm2(phi.get(), local_n);
    if (phi_norm < 1e-14) {
        R.spectral_function.assign(omega_grid.size(), 0.0);
        R.phi_norm = phi_norm;
        return R;
    }
    R.phi_norm = phi_norm;
    be.scale(Complex(1.0 / phi_norm, 0.0), phi.get(), local_n);

    ed::krylov::LanczosKernelOptions kopts;
    kopts.max_iter      = opts.krylov_dim;
    kopts.reorth        = ed::krylov::ReorthPolicy::None;
    kopts.keep_basis    = false;
    kopts.breakdown_tol = opts.tolerance;
    kopts.dim_cap       = (opts.global_n > 0)
        ? static_cast<std::size_t>(opts.global_n)
        : local_n;
    auto kres = ed::krylov::lanczos_kernel(be, apply_H, local_n,
                                           phi.get(), kopts);
    std::vector<double> alpha = kres.alpha;
    std::vector<double> beta  = kres.beta;
    const std::size_t m = alpha.size();
    R.tridiag_size = m;
    if (m == 0) {
        R.spectral_function.assign(omega_grid.size(), 0.0);
        return R;
    }

    double E_shift = opts.energy_shift;
    if (std::abs(E_shift) < 1e-14) {
        std::vector<double> diag_copy = alpha;
        std::vector<double> offdiag(m > 1 ? m - 1 : 1, 0.0);
        for (std::size_t i = 0; i + 1 < m; ++i) offdiag[i] = beta[i + 1];
        const lapack_int info = LAPACKE_dstevd(LAPACK_COL_MAJOR, 'N',
            static_cast<lapack_int>(m),
            diag_copy.data(), offdiag.data(),
            nullptr, 1);
        if (info == 0 && !diag_copy.empty()) E_shift = diag_copy[0];
    }
    for (auto& a : alpha) a -= E_shift;
    R.energy_shift = E_shift;

    R.spectral_function = ::continued_fraction_spectral_function(
        alpha, beta, omega_grid, opts.broadening, phi_norm * phi_norm);
    return R;
}

}  // namespace ed::observables
