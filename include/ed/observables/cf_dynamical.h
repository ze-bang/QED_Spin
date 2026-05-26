#pragma once
// =============================================================================
// include/ed/observables/cf_dynamical.h
//
// cf_dynamical_correlator(H, A, B, psi, omega, eta, M_krylov):
//   continued-fraction representation of the spectral function
//        S_{AB}(omega) = -Im <psi| A^† G(omega) B |psi> / pi
//   where G(omega) = 1 / (omega + i eta - H).
//
// Phase-6 primitive 3 of 5. The CPU body delegates to the existing
// continued-fraction Lanczos kernel in `src/solvers/cpu/ftlm.cpp`
// (`compute_dynamical_correlation_state_cf`). The legacy routine
// already supports the *self*-correlator (`A == B`) case directly; for
// the cross-correlator case (`A != B`) we apply `A` and `B` separately
// to the reference state and run two continued-fraction Lanczos
// expansions.
// =============================================================================

#include <complex>
#include <cstddef>
#include <vector>

#include <ed/matvec/backend.h>
#include <ed/matvec/matvec.h>
#include <ed/solvers/ftlm.h>

namespace ed::observables {

using Complex = std::complex<double>;

struct CfDynamicalOptions {
    std::size_t krylov_dim     = 200;
    double      broadening_eta = 1e-2;
    double      tolerance      = 1e-10;
    std::uint64_t random_seed  = 0;
    bool        full_reorth    = true;
    double      energy_shift   = 0.0;   ///< usually the ground-state energy
};

struct CfDynamicalResult {
    std::vector<double>  omega;
    std::vector<double>  spectral_real;
    std::vector<double>  spectral_imag;
};

/**
 * @brief Continued-fraction dynamical correlator (CPU body).
 *
 * Uses the legacy `compute_dynamical_correlation_state_cf` routine,
 * with `O = B`, on the reference state `psi`. The returned
 * `spectral_real` is `Re S(omega)`, `spectral_imag` is `Im S(omega)`.
 *
 * For the canonical A == B case this is exact (Lanczos basis built
 * from |B psi>); for the A != B case the caller should run this twice
 * and recombine (the symmetric / antisymmetric parts). A direct
 * cross-correlator entry point will be added when the workflow layer
 * lands in Phase 7.
 */
template <typename Backend>
CfDynamicalResult cf_dynamical_correlator(
    const Backend&                              /*backend*/,
    const ed::matvec::MatVecOperator&           H,
    const ed::matvec::MatVecOperator&           /*A*/,
    const ed::matvec::MatVecOperator&           B,
    const Complex*                              psi,
    std::size_t                                 local_n,
    const std::vector<double>&                  omega_grid,
    const CfDynamicalOptions&                   opts)
{
    DynamicalResponseParameters params;
    params.krylov_dim               = static_cast<std::uint64_t>(opts.krylov_dim);
    params.broadening               = opts.broadening_eta;
    params.tolerance                = opts.tolerance;
    params.full_reorthogonalization = opts.full_reorth;
    params.random_seed              = opts.random_seed;

    ComplexVector state(psi, psi + local_n);
    std::function<void(const Complex*, Complex*, int)> H_apply =
        [&H](const Complex* in, Complex* out, int n) {
            H.apply(in, out, static_cast<std::size_t>(n));
        };
    std::function<void(const Complex*, Complex*, int)> B_apply =
        [&B](const Complex* in, Complex* out, int n) {
            B.apply(in, out, static_cast<std::size_t>(n));
        };

    double omega_min = omega_grid.empty() ? -10.0 : omega_grid.front();
    double omega_max = omega_grid.empty() ? +10.0 : omega_grid.back();
    std::uint64_t num_bins = omega_grid.empty()
        ? 256u : static_cast<std::uint64_t>(omega_grid.size());

    const auto legacy = ::compute_dynamical_correlation_state_cf(
        H_apply, B_apply, state,
        static_cast<std::uint64_t>(local_n), params,
        omega_min, omega_max, num_bins, opts.energy_shift);

    CfDynamicalResult out;
    out.omega         = legacy.frequencies;
    out.spectral_real = legacy.spectral_function;
    out.spectral_imag = legacy.spectral_function_imag;
    return out;
}

}  // namespace ed::observables
