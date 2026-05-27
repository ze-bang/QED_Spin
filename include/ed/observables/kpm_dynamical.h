#pragma once
// =============================================================================
// include/ed/observables/kpm_dynamical.h
//
// kpm_dynamical_correlator(H, A, B, psi, omega, M_moments, kernel):
//   Chebyshev expansion of the spectral function
//        S_{AB}(omega) = -Im <psi| A^† delta(omega - H) B |psi>
//   in the Chebyshev basis of the rescaled Hamiltonian.
//
// Phase-6 primitive 4 of 5. The CPU body delegates to
// `ed::kpm::compute_kpm_ltlm_from_states` with a single reference
// eigenstate (taken to be the supplied `psi`) and a single inverse
// temperature beta = 0 to recover the |psi>-only spectral weight.
// Backend specialisations (Cuda, MpiCpu, MpiCuda) land alongside their
// Backend implementations.
// =============================================================================

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <random>
#include <vector>

#include <ed/matvec/backend.h>
#include <ed/matvec/matvec.h>
#include <ed/solvers/ftlm_kpm.h>
#include <ed/solvers/kpm_dos.h>  // estimate_spectral_bounds

namespace ed::observables {

using Complex = std::complex<double>;

enum class KpmKernel : unsigned { Jackson, Lorentz };

struct KpmDynamicalOptions {
    std::size_t num_moments         = 512;
    KpmKernel   kernel              = KpmKernel::Jackson;
    double      lorentz_lambda      = 4.0;
    double      spectral_bound_buffer = 0.05;
    int         spectral_bounds_krylov = 150;
    std::uint64_t random_seed       = 0;
};

struct KpmDynamicalResult {
    std::vector<double> omega;
    std::vector<double> spectral_real;
    std::vector<double> spectral_imag;
    double              kpm_a = 1.0;
    double              kpm_b = 0.0;
};

template <typename Backend>
KpmDynamicalResult kpm_dynamical_correlator(
    const Backend&                              /*backend*/,
    const ed::matvec::MatVecOperator&           H,
    const ed::matvec::MatVecOperator&           A,
    const ed::matvec::MatVecOperator&           B,
    const Complex*                              psi,
    std::size_t                                 local_n,
    const std::vector<double>&                  omega_grid,
    const KpmDynamicalOptions&                  opts)
{
    using ComplexVector = std::vector<Complex>;

    ed::types::MatVec H_apply = [&H](const Complex* in, Complex* out, int n) {
        H.apply(in, out, static_cast<std::size_t>(n));
    };
    ed::types::MatVec A_apply = [&A](const Complex* in, Complex* out, int n) {
        A.apply(in, out, static_cast<std::size_t>(n));
    };
    ed::types::MatVec B_apply = [&B](const Complex* in, Complex* out, int n) {
        B.apply(in, out, static_cast<std::size_t>(n));
    };

    std::vector<ComplexVector> states;
    states.emplace_back(psi, psi + local_n);

    // Reference energy: <psi|H|psi>. The KPM weights drop out at beta=0
    // (single state), but supplying the energy lets the spectral grid
    // be centred correctly when the caller later re-weights.
    ComplexVector Hpsi(local_n);
    H.apply(psi, Hpsi.data(), local_n);
    Complex e_ref{0.0, 0.0};
    for (std::size_t i = 0; i < local_n; ++i) {
        e_ref += std::conj(psi[i]) * Hpsi[i];
    }
    std::vector<double> energies{e_ref.real()};

    // -----------------------------------------------------------------
    // Pillar 4 of the "Save and DSSF Upgrades" plan (May 2026): when
    // ``compute_kpm_ltlm_from_states`` is invoked with a SINGLE outer
    // state, its internal Chebyshev rescaling ``a, b`` is collapsed
    // (``E_min == E_max``, so the fall-back ``BW = 1`` kicks in and the
    // rescaled spectrum sits outside ``[-1, 1]``). We restore the
    // correct rescaling by estimating ``[E_lo, E_hi]`` of H via the
    // shared ``kpm_dos::estimate_spectral_bounds`` Lanczos sweep, then
    // setting ``spectral_bound_buffer`` so that
    //     ``a = buffer >= max(E_n - E_lo, E_hi - E_n)``
    // and ``b = E_n``; this guarantees the eigenvalues of H land in
    // ``[-1, 1]`` after the ``(H - b) / a`` shift used by
    // ``accumulate_kpm_inner``.
    // -----------------------------------------------------------------
    double e_lo = e_ref.real();
    double e_hi = e_ref.real();
    {
        std::mt19937 gen;
        if (opts.random_seed == 0) {
            std::random_device rd; gen.seed(rd());
        } else {
            gen.seed(static_cast<std::uint32_t>(opts.random_seed));
        }
        ed::kpm_dos::estimate_spectral_bounds(
            H_apply, static_cast<std::uint64_t>(local_n),
            opts.spectral_bounds_krylov,
            /*full_reorth=*/true, /*reorth_freq=*/0,
            /*tol=*/1e-10, gen,
            e_lo, e_hi);
    }
    const double half_width = std::max(e_ref.real() - e_lo,
                                        e_hi - e_ref.real());
    const double effective_buffer = (1.0 + opts.spectral_bound_buffer)
                                     * std::max(half_width, 1e-6);

    ed::kpm::KPMParameters params;
    params.num_moments            = static_cast<int>(opts.num_moments);
    params.num_lowest_states      = 1;
    params.outer_krylov_dim       = opts.spectral_bounds_krylov;
    params.use_jackson_kernel     = (opts.kernel == KpmKernel::Jackson);
    params.lorentz_lambda         = opts.lorentz_lambda;
    params.random_seed            = opts.random_seed;
    // The kernel computes ``buffer = params.spectral_bound_buffer * BW``,
    // with ``BW = max(E_max - E_min, 1.0) = 1.0`` for our single-state
    // input. So setting this field directly to ``effective_buffer``
    // gives ``a = effective_buffer, b = E_n`` -- exactly the rescaling
    // we need to keep H's spectrum inside ``[-1, 1]``.
    params.spectral_bound_buffer  = effective_buffer;

    const double omega_min = omega_grid.empty() ? -10.0 : omega_grid.front();
    const double omega_max = omega_grid.empty() ? +10.0 : omega_grid.back();
    const int    n_omega   = omega_grid.empty()
        ? 256 : static_cast<int>(omega_grid.size());

    const auto legacy = ed::kpm::compute_kpm_ltlm_from_states(
        H_apply, A_apply, B_apply,
        static_cast<std::uint64_t>(local_n),
        static_cast<std::uint64_t>(local_n),
        states, energies,
        omega_min, omega_max, n_omega,
        /*betas=*/std::vector<double>{0.0}, params);

    KpmDynamicalResult out;
    out.omega         = legacy.frequencies;
    out.spectral_real = legacy.spectral_real;
    out.spectral_imag = legacy.spectral_imag;
    out.kpm_a         = legacy.kpm_a;
    out.kpm_b         = legacy.kpm_b;
    return out;
}

}  // namespace ed::observables
