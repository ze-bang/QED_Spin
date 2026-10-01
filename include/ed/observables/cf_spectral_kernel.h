#pragma once
// =============================================================================
// include/ed/observables/cf_spectral_kernel.h
//
// Continued-fraction spectral-function kernel --- `template<Backend,
// MatvecHFn>`. Wraps the unified Lanczos kernel in its "no basis storage"
// mode (keep_basis=false) plus the analytic continued-fraction evaluator.
// Produces S(omega) = -Im[G(omega + i*eta)] / pi where
// G(z) = <phi| (z - H + E_shift)^-1 |phi> for a caller-built |phi>
// (typically O|psi_0> assembled across sectors), weighted by ||phi||^2.
//
// Templated so any Backend can drive it: the only Backend-specific
// operations are the Lanczos tridiag build (`lanczos_kernel<Backend>`)
// and the |phi> staging (copy / nrm2 / scale --- all in `Backend`).
// =============================================================================

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <optional>
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
    /// The energy omega is measured from (E_0 of the source); unset, the
    /// smallest eigenvalue of the tridiagonal matrix. Zero is a shift like any other.
    std::optional<double> energy_shift;
    /// Convergence tolerance for the Lanczos tridiag build.
    double      tolerance        = 1e-12;
    /// Global problem dimension, forwarded as the Lanczos dimension cap.
    /// 0 means "use local_n".
    std::uint64_t global_n       = 0;
};

struct CfSpectralResult {
    std::vector<double> frequencies;
    std::vector<double> spectral_function;
    /// max |S_m - S_{m/2}| / max S_m -- the change of the continued
    /// fraction between half and full Krylov depth. Above ~0.05 the spectrum
    /// is not converged at this krylov_dim (an unconverged CF at eta = 0.05 on a
    /// 2.7e6-state block varied by 30 % between otherwise identical runs).
    double              convergence_change = 0.0;
    double              energy_shift = 0.0;
};

// ----------------------------------------------------------------------------
// cf_spectral_from_vector --- continued-fraction kernel starting from a
// pre-built phi vector (no random seed, no |phi> = O|psi> step).
//
// Use when phi was computed externally (e.g. by applying a cross-sector
// observable to a ground state stored in a *different* sector basis): just
// pass phi in target-sector memory and the spectral weight is folded in via
// ||phi||^2. This is the kernel underneath the ed::sectors dynamics verb
// (src/solvers/little_group/lg_sectors_dynamics.cpp).
//
// Phi is normalised before the Lanczos build, but ||phi||^2 is preserved as
// the spectral-function weight so the absolute amplitude of S(omega) is
// physically meaningful.
//
// ApplyH must matvec on the *target* sector (same dim as ``phi_seed``).
// ``E_shift`` is the energy reference (usually E_0 from the source-sector
// ground-state solve, so omega-axes line up with the standard
// S(Q, omega) convention). Without ``opts.energy_shift`` the kernel uses the
// smallest eigenvalue of the tridiagonal matrix.
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
        return R;
    }
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
    if (m == 0) {
        R.spectral_function.assign(omega_grid.size(), 0.0);
        return R;
    }

    double E_shift = opts.energy_shift.value_or(0.0);
    if (!opts.energy_shift) {
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
    if (alpha.size() >= 4) {
        const std::size_t h = alpha.size() / 2;
        std::vector<double> a2(alpha.begin(), alpha.begin() + h);
        std::vector<double> b2(beta.begin(), beta.begin() + std::min(beta.size(), h + 1));
        const auto S_half = ::continued_fraction_spectral_function(
            a2, b2, omega_grid, opts.broadening, phi_norm * phi_norm);
        double smax = 0.0, dmax = 0.0;
        for (std::size_t i = 0; i < R.spectral_function.size() && i < S_half.size(); ++i) {
            smax = std::max(smax, std::abs(R.spectral_function[i]));
            dmax = std::max(dmax, std::abs(R.spectral_function[i] - S_half[i]));
        }
        R.convergence_change = (smax > 0.0) ? dmax / smax : 0.0;
    }
    return R;
}

}  // namespace ed::observables
