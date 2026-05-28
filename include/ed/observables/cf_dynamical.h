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
//
// Phase F of the "Close CPU/GPU Gaps" plan (May 2026):
// ``detail::ftlm_dynamical_kernel_via_backend<Backend>`` ships the
// multi-sample finite-T FTLM dynamical kernel that the orchestrator
// previously routed exclusively to ``::compute_dynamical_correlation``
// (host-only). It mirrors the LTLM dual-backend pattern: host-seeded
// random vectors, device-resident O2 / O1 / Lanczos, host-side
// tridiagonal diagonalisation, host-side Lorentzian sum.
// =============================================================================

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include <ed/core/blas_lapack_wrapper.h>
#include <ed/krylov/lanczos_kernel.h>
#include <ed/matvec/backend.h>
#include <ed/matvec/matvec.h>
#include <ed/solvers/ftlm.h>
#include <ed/solvers/lanczos.h>   // diagonalize_tridiagonal_ritz

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

// ---------------------------------------------------------------------------
// Phase F of the "Close CPU/GPU Gaps" plan (May 2026):
// multi-sample finite-T FTLM dynamical correlator, Backend-templated.
// ---------------------------------------------------------------------------

/// Options struct for the multi-sample FTLM dynamical kernel. Mirrors
/// the legacy ``DynamicalResponseParameters`` fields exposed by
/// ``ed::workflows::SpectralOptions``; constructed inside
/// ``orchestrator.cpp::spectral`` from the user-facing options.
struct FtlmDynamicalOptions {
    std::size_t   krylov_dim       = 200;
    std::size_t   num_samples      = 1;
    double        broadening       = 0.05;
    double        temperature      = 0.0;     ///< T=0 -> no thermal weighting
    double        energy_shift     = 0.0;     ///< 0 -> auto-detect per sample
    double        tolerance        = 1e-12;
    std::uint64_t random_seed      = 0;
    /// Global problem dimension for the per-cycle Lanczos cap on
    /// distributed backends. 0 means "use local_n".
    std::uint64_t global_n         = 0;
};

struct FtlmDynamicalResult {
    std::vector<double> omega;
    std::vector<double> spectral_real;
    std::vector<double> spectral_imag;
    std::vector<double> spectral_error_real;
    std::vector<double> spectral_error_imag;
    std::size_t         total_samples = 0;
};

namespace detail {

/// Host-side Lorentzian sum over Krylov-projected weights. Lifted
/// straight from the legacy ``compute_spectral_function_complex`` body
/// in ``src/solvers/cpu/ftlm.cpp`` so the GPU lane produces the same
/// per-omega numbers as the CPU lane (modulo Lanczos noise from
/// independent random seeds).
inline void ftlm_dynamical_sample_spectrum(
    const std::vector<double>&  ritz_values,
    const std::vector<Complex>& complex_weights,
    const std::vector<double>&  frequencies,
    double                      broadening,
    double                      temperature,
    std::vector<double>&        out_real,
    std::vector<double>&        out_imag)
{
    const std::size_t n_omega  = frequencies.size();
    const std::size_t n_states = ritz_values.size();
    out_real.assign(n_omega, 0.0);
    out_imag.assign(n_omega, 0.0);

    // Optional temperature-weight remap. Matches the legacy host
    // implementation byte-for-byte.
    std::vector<Complex> thermal_w = complex_weights;
    if (temperature > 1e-14 && n_states > 0) {
        const double beta = 1.0 / temperature;
        const double e_min = *std::min_element(
            ritz_values.begin(), ritz_values.end());
        double Z = 0.0;
        for (std::size_t i = 0; i < n_states; ++i) {
            Z += std::exp(-beta * (ritz_values[i] - e_min));
        }
        if (Z > 1e-300) {
            for (std::size_t i = 0; i < n_states; ++i) {
                const double bf = std::exp(
                    -beta * (ritz_values[i] - e_min)) / Z;
                thermal_w[i] = complex_weights[i] * bf;
            }
        } else {
            // Pathological low-T branch: collapse to the lowest
            // Ritz value's weight and renormalise. Matches legacy
            // host code's fall-back exactly.
            thermal_w.assign(n_states, Complex(0.0, 0.0));
            const std::size_t gs_idx = std::distance(
                ritz_values.begin(),
                std::min_element(ritz_values.begin(),
                                 ritz_values.end()));
            thermal_w[gs_idx] = complex_weights[gs_idx];
            Complex sum(0.0, 0.0);
            for (const auto& w : thermal_w) sum += w;
            if (std::abs(sum) > 1e-300) {
                for (auto& w : thermal_w) w /= sum;
            }
        }
    }

    const double norm_factor = broadening / M_PI;
    for (std::size_t io = 0; io < n_omega; ++io) {
        const double w_o = frequencies[io];
        Complex acc(0.0, 0.0);
        for (std::size_t i = 0; i < n_states; ++i) {
            const double delta = w_o - ritz_values[i];
            const double lor =
                norm_factor / (delta * delta + broadening * broadening);
            acc += thermal_w[i] * lor;
        }
        out_real[io] = acc.real();
        out_imag[io] = acc.imag();
    }
}

/// Backend-templated multi-sample FTLM dynamical kernel.
///
/// Algorithm per sample (mirrors the legacy
/// ``::compute_dynamical_correlation`` in ``src/solvers/cpu/ftlm.cpp``):
///   1. Host-side Gaussian random ``|psi>``, copy to backend.
///   2. ``|phi> = O_2 |psi>`` on backend, ``phi_norm = ||phi||``,
///      normalise ``|phi>``.
///   3. ``lanczos_kernel<Backend>(H, phi, keep_basis=true)`` ->
///      tridiagonal ``(alpha, beta)`` + Lanczos basis ``V`` (device-
///      resident).
///   4. Host-side ``diagonalize_tridiagonal_ritz`` -> Ritz values +
///      eigenvectors ``V_n``.
///   5. Energy shift (auto-detect smallest Ritz, or caller override).
///   6. ``|O1_psi> = O_1 |psi>`` on backend.
///   7. ``p[j] = <O1_psi | v_j>`` for ``j = 0 .. m-1``, computed via
///      ``backend.dot_many(basis, m, O1_psi, n, ...)`` (single cuBLAS
///      gemv on CUDA / single OpenMP region on CPU) plus a host-side
///      conjugation to flip the slot order.
///   8. Host-side: ``overlap_O1[n] = sum_j V[j,n] * p[j]`` via two
///      real ``cblas_dgemv`` calls (one for ``p_re``, one for
///      ``p_im``); ``weight[n] = overlap_O1[n] * V[0,n] * phi_norm``.
///   9. Host-side Lorentzian sum (``ftlm_dynamical_sample_spectrum``).
/// 10. Accumulate ``S_real`` / ``S_imag`` over samples.
template <typename Backend, typename ApplyH, typename ApplyO>
FtlmDynamicalResult ftlm_dynamical_kernel_via_backend(
    Backend&                   be,
    ApplyH&&                   apply_H,
    ApplyO&&                   apply_O1,
    ApplyO&&                   apply_O2,
    std::size_t                local_n,
    const std::vector<double>& omega_grid,
    const FtlmDynamicalOptions& opts)
{
    if (local_n == 0) {
        throw std::invalid_argument(
            "ftlm_dynamical_kernel_via_backend: local_n == 0");
    }
    if (omega_grid.empty()) {
        throw std::invalid_argument(
            "ftlm_dynamical_kernel_via_backend: empty frequency grid");
    }
    if (opts.num_samples == 0) {
        throw std::invalid_argument(
            "ftlm_dynamical_kernel_via_backend: num_samples == 0");
    }
    if (opts.krylov_dim < 2) {
        throw std::invalid_argument(
            "ftlm_dynamical_kernel_via_backend: krylov_dim must be >= 2");
    }

    const std::size_t n_omega = omega_grid.size();
    FtlmDynamicalResult R;
    R.omega = omega_grid;
    R.spectral_real.assign(n_omega, 0.0);
    R.spectral_imag.assign(n_omega, 0.0);
    R.spectral_error_real.assign(n_omega, 0.0);
    R.spectral_error_imag.assign(n_omega, 0.0);

    // Per-sample storage so we can compute the std error after
    // averaging.
    std::vector<std::vector<double>> sample_real;
    std::vector<std::vector<double>> sample_imag;
    sample_real.reserve(opts.num_samples);
    sample_imag.reserve(opts.num_samples);

    const std::uint64_t base_seed = (opts.random_seed != 0)
        ? opts.random_seed
        : 0xBADDCAFEULL;

    for (std::size_t s = 0; s < opts.num_samples; ++s) {
        // -------- 1. Host-side Gaussian seed, copy to backend --------
        std::mt19937_64 rng(base_seed + 0x9E3779B97F4A7C15ULL * s);
        std::normal_distribution<double> gauss(0.0, 1.0);

        std::vector<Complex> psi_host(local_n);
        for (auto& c : psi_host) {
            c = Complex(gauss(rng), gauss(rng));
        }
        double sumsq = 0.0;
        for (const auto& c : psi_host) sumsq += std::norm(c);
        const double psi_n0 = std::sqrt(sumsq);
        if (!(psi_n0 > 0.0)) {
            continue;   // pathologically small sample, skip
        }
        const double inv_n0 = 1.0 / psi_n0;
        for (auto& c : psi_host) c *= inv_n0;

        auto psi = be.make_zero_vector(local_n);
        be.copy_from_host(psi_host.data(), psi.get(), local_n);

        // -------- 2. |phi> = O_2 |psi>, normalise -------------------
        auto phi = be.make_zero_vector(local_n);
        apply_O2(psi.get(), phi.get(), local_n);
        const double phi_norm = be.nrm2(phi.get(), local_n);
        if (phi_norm < 1e-14) {
            continue;   // zero-norm sample
        }
        be.scale(Complex(1.0 / phi_norm, 0.0), phi.get(), local_n);

        // -------- 3. Lanczos with basis storage ---------------------
        ed::krylov::LanczosKernelOptions kopts;
        kopts.max_iter      = opts.krylov_dim;
        kopts.keep_basis    = true;
        kopts.breakdown_tol = opts.tolerance;
        kopts.reorth        = ed::krylov::ReorthPolicy::FullCGS2;
        kopts.dim_cap       = (opts.global_n > 0)
            ? static_cast<std::size_t>(opts.global_n)
            : local_n;

        auto kres = ed::krylov::lanczos_kernel(
            be, apply_H, local_n, phi.get(), kopts);
        const std::size_t m = kres.alpha.size();
        if (m == 0) continue;

        // -------- 4. Host-side tridiag diag (ritz + evecs) ----------
        std::vector<double> ritz_values, dummy_weights, evecs;
        diagonalize_tridiagonal_ritz(
            kres.alpha, kres.beta, ritz_values, dummy_weights, &evecs);
        if (ritz_values.empty()) continue;

        // -------- 5. Energy shift (auto-detect or override) ---------
        double E_shift = opts.energy_shift;
        if (std::abs(E_shift) < 1e-14) {
            E_shift = *std::min_element(
                ritz_values.begin(), ritz_values.end());
        }
        for (auto& r : ritz_values) r -= E_shift;

        // -------- 6. |O1_psi> = O_1 |psi> ---------------------------
        auto O1_psi = be.make_zero_vector(local_n);
        apply_O1(psi.get(), O1_psi.get(), local_n);

        // -------- 7. p[j] = <O1_psi | v_j> --------------------------
        // ``backend.dot_many(basis, m, O1_psi, n, out)`` computes
        //   ``out[k] = <basis[k] | O1_psi>`` = sum_i conj(basis[k][i]) * O1_psi[i].
        // We want ``p[j] = <O1_psi | v_j>`` = conj(<v_j | O1_psi>),
        // so flip on the host (cheap, m ~ krylov_dim ~ 100-200).
        std::vector<const Complex*> basis_ptrs(m, nullptr);
        for (std::size_t j = 0; j < m; ++j) {
            basis_ptrs[j] = kres.basis[j].get();
        }
        std::vector<Complex> p(m, Complex(0.0, 0.0));
        be.dot_many(basis_ptrs.data(), m, O1_psi.get(), local_n,
                    p.data());
        for (auto& c : p) c = std::conj(c);

        // -------- 8. Host-side overlap + weights --------------------
        std::vector<double> p_re(m), p_im(m);
        for (std::size_t j = 0; j < m; ++j) {
            p_re[j] = p[j].real();
            p_im[j] = p[j].imag();
        }
        std::vector<double> overlap_re(m), overlap_im(m);
        // ``evecs`` is column-major, stride m, so ``evecs[n*m + j]``
        // is V[j,n]. We need overlap[n] = sum_j V[j,n] * p[j], which is
        // V^T * p with V (m x m) column-major. Use CblasTrans on V
        // (handled by ``evecs``).
        cblas_dgemv(CblasColMajor, CblasTrans,
                    static_cast<int>(m), static_cast<int>(m),
                    1.0, evecs.data(), static_cast<int>(m),
                    p_re.data(), 1,
                    0.0, overlap_re.data(), 1);
        cblas_dgemv(CblasColMajor, CblasTrans,
                    static_cast<int>(m), static_cast<int>(m),
                    1.0, evecs.data(), static_cast<int>(m),
                    p_im.data(), 1,
                    0.0, overlap_im.data(), 1);

        std::vector<Complex> weights(m);
        for (std::size_t n = 0; n < m; ++n) {
            const Complex overlap_O1(overlap_re[n], overlap_im[n]);
            // V[0,n] = evecs[n*m + 0]. ``|v_0> = O_2 |psi> / phi_norm``,
            // so <n | O_2 | psi> = V[0,n] * phi_norm (real).
            const Complex overlap_O2(evecs[n * m + 0] * phi_norm, 0.0);
            weights[n] = overlap_O1 * overlap_O2;
        }

        // -------- 9. Lorentzian sum (host) --------------------------
        std::vector<double> samp_re, samp_im;
        ftlm_dynamical_sample_spectrum(
            ritz_values, weights, omega_grid,
            opts.broadening, opts.temperature, samp_re, samp_im);

        sample_real.push_back(std::move(samp_re));
        sample_imag.push_back(std::move(samp_im));
    }

    // ------ 10. Average over valid samples + std-error band --------
    const std::size_t n_valid = sample_real.size();
    R.total_samples = n_valid;
    if (n_valid == 0) {
        return R;   // all samples degenerate; spectrum stays zero
    }
    for (const auto& sr : sample_real) {
        for (std::size_t i = 0; i < n_omega; ++i) R.spectral_real[i] += sr[i];
    }
    for (const auto& si : sample_imag) {
        for (std::size_t i = 0; i < n_omega; ++i) R.spectral_imag[i] += si[i];
    }
    const double inv_n = 1.0 / static_cast<double>(n_valid);
    for (std::size_t i = 0; i < n_omega; ++i) {
        R.spectral_real[i] *= inv_n;
        R.spectral_imag[i] *= inv_n;
    }
    if (n_valid > 1) {
        for (std::size_t s = 0; s < n_valid; ++s) {
            for (std::size_t i = 0; i < n_omega; ++i) {
                const double dr = sample_real[s][i] - R.spectral_real[i];
                const double di = sample_imag[s][i] - R.spectral_imag[i];
                R.spectral_error_real[i] += dr * dr;
                R.spectral_error_imag[i] += di * di;
            }
        }
        const double denom = std::sqrt(
            static_cast<double>(n_valid * (n_valid - 1)));
        for (std::size_t i = 0; i < n_omega; ++i) {
            R.spectral_error_real[i] =
                std::sqrt(R.spectral_error_real[i]) / denom;
            R.spectral_error_imag[i] =
                std::sqrt(R.spectral_error_imag[i]) / denom;
        }
    }
    return R;
}

}  // namespace detail

}  // namespace ed::observables
