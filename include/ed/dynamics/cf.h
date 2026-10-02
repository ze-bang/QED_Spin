#pragma once
// =============================================================================
// include/ed/dynamics/cf.h
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

#include <ed/krylov/lanczos.h>
#include <ed/krylov/tridiag.h>
#include <ed/matvec/backend.h>

namespace ed::observables {

using Complex = std::complex<double>;

/// S(w) = -Im G(w + i eta) / pi of the continued fraction
/// G(z) = norm_sq / (z - a_0 - b_1^2 / (z - a_1 - b_2^2 / ...)) of a Lanczos tridiagonal
/// (alpha[0..M), beta[n] coupling steps n-1 and n, beta[0] unused), evaluated bottom-up: stable
/// and O(M) per frequency. norm_sq = ||O|psi>||^2.
inline std::vector<double> continued_fraction(const std::vector<double>& alpha,
                                              const std::vector<double>& beta,
                                              const std::vector<double>& omega,
                                              double eta, double norm_sq) {
    std::vector<double> S(omega.size(), 0.0);
    const std::size_t M = alpha.size();
    if (M == 0) return S;
    constexpr double kPi = 3.14159265358979323846;
    #pragma omp parallel for schedule(static)
    for (std::int64_t iw = 0; iw < static_cast<std::int64_t>(omega.size()); ++iw) {
        const Complex z(omega[static_cast<std::size_t>(iw)], eta);
        Complex G(0.0, 0.0);                       // G_M = 0; G_n = b_n^2 / (z - a_n - G_{n+1})
        for (std::size_t n = M - 1; n >= 1; --n) {
            const double b2 = n < beta.size() ? beta[n] * beta[n] : 0.0;
            const Complex d = z - Complex(alpha[n], 0.0) - G;
            G = std::abs(d) > 1e-300 ? Complex(b2, 0.0) / d : Complex(0.0, 0.0);
        }
        const Complex d = z - Complex(alpha[0], 0.0) - G;
        const Complex G0 = std::abs(d) > 1e-300 ? Complex(norm_sq, 0.0) / d : Complex(0.0, 0.0);
        S[static_cast<std::size_t>(iw)] = -G0.imag() / kPi;
    }
    return S;
}

struct CfSpectralOptions {
    std::size_t krylov_dim       = 200;
    double      broadening       = 0.05;
    /// The energy omega is measured from (E_0 of the source); unset, the
    /// smallest eigenvalue of the tridiagonal matrix. Zero is a shift like any other.
    std::optional<double> energy_shift;
    /// Convergence tolerance for the Lanczos tridiag build.
    // scale-free: a default for C++ callers; the engine passes relative values (numerics.h)
    double      tolerance        = 1e-12;
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
// (src/engine/dynamics.cpp).
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
    // scale-free: unit-vector norm
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
    if (!opts.energy_shift)
        E_shift = ed::krylov::tridiag_eig(alpha, beta, m, /*vectors=*/false).values.front();
    for (auto& a : alpha) a -= E_shift;
    R.energy_shift = E_shift;

    R.spectral_function = continued_fraction(
        alpha, beta, omega_grid, opts.broadening, phi_norm * phi_norm);
    if (alpha.size() >= 4) {
        const std::size_t h = alpha.size() / 2;
        std::vector<double> a2(alpha.begin(), alpha.begin() + h);
        std::vector<double> b2(beta.begin(), beta.begin() + std::min(beta.size(), h + 1));
        const auto S_half = continued_fraction(
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

/// The cross spectral function S_AB(w) = sum_n <a|n><n|b> L_eta(w - E_n + E_shift) of two vectors
/// of one sector (a = A|psi>, b = B|psi>; complex in general), by one Lanczos run from b: with the
/// tridiagonal's eigenpairs (theta_k, z_k) and c_j = <V_j|a> taken as each Krylov vector V_j is
/// formed (LanczosKernelOptions::on_vector, no basis kept), <n|b> ~ ||b|| z_0k and
/// <a|n> ~ sum_j conj(c_j) z_jk -- the Krylov resolvent <a|(z - H)^-1|b> = ||b|| c^H (z - T)^-1 e_0
/// in pole form. a = b gives the continued fraction's poles; the polarisation identity
/// S_AB = (1/4) sum_k i^-k S_{A + i^k B} relates it to autocorrelations.
template <typename Backend, typename ApplyH>
std::vector<Complex> cross_spectral_from_vectors(Backend& be, ApplyH&& apply_H, std::size_t local_n,
                                                 const Complex* b_seed, const Complex* a_vec,
                                                 const std::vector<double>& omega_grid,
                                                 const CfSpectralOptions& opts) {
    if (local_n == 0) throw std::invalid_argument("cross_spectral_from_vectors: local_n == 0");
    if (omega_grid.empty()) throw std::invalid_argument("cross_spectral_from_vectors: empty frequency grid");
    std::vector<Complex> S(omega_grid.size(), Complex(0, 0));
    auto b = be.make_zero_vector(local_n);
    auto a = be.make_zero_vector(local_n);
    be.copy_from_host(b_seed, b.get(), local_n);
    be.copy_from_host(a_vec, a.get(), local_n);
    const double b_norm = be.nrm2(b.get(), local_n);
    // scale-free: unit-vector norm
    if (b_norm < 1e-14 || be.nrm2(a.get(), local_n) < 1e-14) return S;
    be.scale(Complex(1.0 / b_norm, 0.0), b.get(), local_n);
    std::vector<Complex> c;   // c_j = <V_j|a>
    ed::krylov::LanczosKernelOptions kopts;
    kopts.max_iter      = opts.krylov_dim;
    kopts.reorth        = ed::krylov::ReorthPolicy::None;
    kopts.keep_basis    = false;
    kopts.breakdown_tol = opts.tolerance;
    kopts.on_vector     = [&](std::size_t, const Complex* v) { c.push_back(be.dot(v, a.get(), local_n)); };
    const auto kres = ed::krylov::lanczos_kernel(be, apply_H, local_n, b.get(), kopts);
    const std::size_t m = kres.alpha.size();
    if (m == 0) return S;
    const ed::krylov::TridiagEig t = ed::krylov::tridiag_eig(kres.alpha, kres.beta, m, /*vectors=*/true);
    const double E_shift = opts.energy_shift.value_or(t.values.front());
    constexpr double kInvPi = 0.3183098861837907;
    const double eta = opts.broadening;
    for (std::size_t k = 0; k < m; ++k) {
        Complex an(0, 0);                                   // <a|n_k> ~ sum_j conj(c_j) z_jk
        for (std::size_t j = 0; j < m; ++j) an += std::conj(c[j]) * t.z(j, k);
        const Complex w = an * (b_norm * t.z(0, k));
        if (std::abs(w) < 1e-300) continue;
        const double E = t.values[k] - E_shift;
        for (std::size_t iw = 0; iw < omega_grid.size(); ++iw) {
            const double d = omega_grid[iw] - E;
            S[iw] += w * ((eta * kInvPi) / (d * d + eta * eta));
        }
    }
    return S;
}

}  // namespace ed::observables
