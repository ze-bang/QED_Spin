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
// Multi-temperature FTLM dynamical kernel (Consolidation Family 3). The
// Krylov basis, Ritz values and O1/O2 overlap weights (steps 1-8) are
// temperature-INDEPENDENT, so they are computed once per random sample and the
// final Lorentzian reweighting (step 9, ``ftlm_dynamical_sample_spectrum``,
// which already takes a temperature) is looped over ``temperatures``. This
// reproduces the Krylov-reuse efficiency of the legacy
// ``GPUFTLMSolver::computeDynamicalCorrelationMultiTemp`` on any Backend.
// Returns one FtlmDynamicalResult per temperature (same order as the input).
template <typename Backend, typename ApplyH, typename ApplyO>
std::vector<FtlmDynamicalResult> ftlm_dynamical_kernel_via_backend_multitemp(
    Backend&                   be,
    ApplyH&&                   apply_H,
    ApplyO&&                   apply_O1,
    ApplyO&&                   apply_O2,
    std::size_t                local_n,
    const std::vector<double>& omega_grid,
    const std::vector<double>& temperatures,
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
    if (temperatures.empty()) {
        throw std::invalid_argument(
            "ftlm_dynamical_kernel_via_backend: empty temperature list");
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
    const std::size_t n_temp  = temperatures.size();
    std::vector<FtlmDynamicalResult> Rs(n_temp);
    for (std::size_t t = 0; t < n_temp; ++t) {
        Rs[t].omega = omega_grid;
        Rs[t].spectral_real.assign(n_omega, 0.0);
        Rs[t].spectral_imag.assign(n_omega, 0.0);
        Rs[t].spectral_error_real.assign(n_omega, 0.0);
        Rs[t].spectral_error_imag.assign(n_omega, 0.0);
    }

    // Per-(temperature, sample) storage so we can compute the std error after
    // averaging. sample_real[t][s] is the s-th sample's spectrum at temp t.
    std::vector<std::vector<std::vector<double>>> sample_real(n_temp);
    std::vector<std::vector<std::vector<double>>> sample_imag(n_temp);
    for (std::size_t t = 0; t < n_temp; ++t) {
        sample_real[t].reserve(opts.num_samples);
        sample_imag[t].reserve(opts.num_samples);
    }

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

        // -------- 9. Lorentzian sum (host), per temperature ---------
        // Steps 1-8 above are temperature-independent; only this final
        // reweighting varies with T, so loop it (Krylov basis reused).
        for (std::size_t t = 0; t < n_temp; ++t) {
            std::vector<double> samp_re, samp_im;
            ftlm_dynamical_sample_spectrum(
                ritz_values, weights, omega_grid,
                opts.broadening, temperatures[t], samp_re, samp_im);
            sample_real[t].push_back(std::move(samp_re));
            sample_imag[t].push_back(std::move(samp_im));
        }
    }

    // ------ 10. Average over valid samples + std-error band, per T -----
    for (std::size_t t = 0; t < n_temp; ++t) {
        FtlmDynamicalResult& R = Rs[t];
        const auto& s_real = sample_real[t];
        const auto& s_imag = sample_imag[t];
        const std::size_t n_valid = s_real.size();
        R.total_samples = n_valid;
        if (n_valid == 0) {
            continue;   // all samples degenerate; spectrum stays zero
        }
        for (const auto& sr : s_real) {
            for (std::size_t i = 0; i < n_omega; ++i) R.spectral_real[i] += sr[i];
        }
        for (const auto& si : s_imag) {
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
                    const double dr = s_real[s][i] - R.spectral_real[i];
                    const double di = s_imag[s][i] - R.spectral_imag[i];
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
    }
    return Rs;
}

// Single-temperature wrapper (unchanged public API): delegates to the
// multi-temperature core with a one-element temperature list.
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
    auto Rs = ftlm_dynamical_kernel_via_backend_multitemp(
        be, apply_H, apply_O1, apply_O2, local_n, omega_grid,
        std::vector<double>{opts.temperature}, opts);
    return std::move(Rs.front());
}

// ---------------------------------------------------------------------------
// Static two-operator correlation <O1^dag O2>(T) via the standard FTLM trace
// estimator (Consolidation Family 3). DISTINCT from the dynamical kernel: the
// Krylov basis is seeded from the random vector |r> (NOT O2|psi>); each Ritz
// state |n> is reconstructed from the basis; the DIAGONAL element
// <n|O1^dag O2|n> is formed; and the FTLM weights w_n = |<r|n>|^2 thermal-
// average it (e_min-shifted for low-T stability):
//   <O1^dag O2>(T) = Sum_n w_n e^{-b(E_n-e_min)} <n|O1^dag O2|n>
//                  / Sum_n w_n e^{-b(E_n-e_min)}.
// Mirrors the trusted host ::compute_static_response; also emits the variance
// <O^2>-<O>^2 and susceptibility = variance / T. Backend-generic (CPU / CUDA):
// the single implementation replacing GPUFTLMSolver::computeStaticCorrelation
// and the host compute_static_response.
// ---------------------------------------------------------------------------
struct FtlmStaticOptions {
    std::size_t   krylov_dim  = 200;
    std::size_t   num_samples = 1;
    double        tolerance   = 1e-12;
    std::uint64_t random_seed = 0;
    std::uint64_t global_n    = 0;   ///< 0 -> use local_n for the per-cycle cap
};

struct FtlmStaticResult {
    std::vector<double> temperatures;
    std::vector<double> expectation;
    std::vector<double> variance;
    std::vector<double> susceptibility;
    std::vector<double> expectation_error;
    std::vector<double> variance_error;
    std::vector<double> susceptibility_error;
    std::size_t         total_samples = 0;
};

template <typename Backend, typename ApplyH, typename ApplyO>
FtlmStaticResult ftlm_static_correlation_via_backend_multitemp(
    Backend&                   be,
    ApplyH&&                   apply_H,
    ApplyO&&                   apply_O1,
    ApplyO&&                   apply_O2,
    std::size_t                local_n,
    const std::vector<double>& temperatures,
    const FtlmStaticOptions&   opts)
{
    if (local_n == 0)
        throw std::invalid_argument("ftlm_static_correlation: local_n == 0");
    if (temperatures.empty())
        throw std::invalid_argument("ftlm_static_correlation: empty temperature list");
    for (double T : temperatures)
        if (!(T > 0.0))
            throw std::invalid_argument("ftlm_static_correlation: temperatures must be > 0");
    if (opts.num_samples == 0)
        throw std::invalid_argument("ftlm_static_correlation: num_samples == 0");
    if (opts.krylov_dim < 2)
        throw std::invalid_argument("ftlm_static_correlation: krylov_dim must be >= 2");

    const std::size_t nT = temperatures.size();
    FtlmStaticResult R;
    R.temperatures = temperatures;
    R.expectation.assign(nT, 0.0);
    R.variance.assign(nT, 0.0);
    R.susceptibility.assign(nT, 0.0);
    R.expectation_error.assign(nT, 0.0);
    R.variance_error.assign(nT, 0.0);
    R.susceptibility_error.assign(nT, 0.0);

    std::vector<std::vector<double>> sample_exp;   // [valid_sample][T]
    std::vector<std::vector<double>> sample_var;
    sample_exp.reserve(opts.num_samples);
    sample_var.reserve(opts.num_samples);

    const std::uint64_t base_seed = (opts.random_seed != 0)
        ? opts.random_seed : 0xBADDCAFEULL;

    for (std::size_t s = 0; s < opts.num_samples; ++s) {
        // 1. Gaussian random seed (host), normalise, copy to backend.
        std::mt19937_64 rng(base_seed + 0x9E3779B97F4A7C15ULL * s);
        std::normal_distribution<double> gauss(0.0, 1.0);
        std::vector<Complex> r_host(local_n);
        for (auto& c : r_host) c = Complex(gauss(rng), gauss(rng));
        double sumsq = 0.0;
        for (const auto& c : r_host) sumsq += std::norm(c);
        const double r0 = std::sqrt(sumsq);
        if (!(r0 > 0.0)) continue;
        const double inv0 = 1.0 / r0;
        for (auto& c : r_host) c *= inv0;

        auto rvec = be.make_zero_vector(local_n);
        be.copy_from_host(r_host.data(), rvec.get(), local_n);

        // 2. Lanczos with basis storage (seeded from |r>).
        ed::krylov::LanczosKernelOptions kopts;
        kopts.max_iter      = opts.krylov_dim;
        kopts.keep_basis    = true;
        kopts.breakdown_tol = opts.tolerance;
        kopts.reorth        = ed::krylov::ReorthPolicy::FullCGS2;
        kopts.dim_cap       = (opts.global_n > 0)
            ? static_cast<std::size_t>(opts.global_n) : local_n;
        auto kres = ed::krylov::lanczos_kernel(
            be, apply_H, local_n, rvec.get(), kopts);
        const std::size_t m = kres.alpha.size();
        if (m == 0) continue;

        // 3. Tridiagonal diagonalisation: Ritz values, FTLM weights, evecs.
        std::vector<double> ritz_values, weights, evecs;
        diagonalize_tridiagonal_ritz(
            kres.alpha, kres.beta, ritz_values, weights, &evecs);
        if (ritz_values.empty()) continue;

        // 4. corr[n] = Re <n|O1^dag O2|n>, reconstructing
        //    |n> = Sum_j evecs[n*m+j] |v_j> in the backend.
        std::vector<double> corr(m, 0.0);
        auto psi_n = be.make_zero_vector(local_n);
        auto O1_n  = be.make_zero_vector(local_n);
        auto O2_n  = be.make_zero_vector(local_n);
        for (std::size_t n = 0; n < m; ++n) {
            be.scale(Complex(0.0, 0.0), psi_n.get(), local_n);   // zero
            for (std::size_t j = 0; j < m; ++j) {
                be.axpy(Complex(evecs[n * m + j], 0.0),
                        kres.basis[j].get(), psi_n.get(), local_n);
            }
            apply_O1(psi_n.get(), O1_n.get(), local_n);
            apply_O2(psi_n.get(), O2_n.get(), local_n);
            corr[n] = be.dot(O1_n.get(), O2_n.get(), local_n).real();
        }

        // 5. Thermal averages per temperature (e_min-shifted for stability).
        const double e_min =
            *std::min_element(ritz_values.begin(), ritz_values.end());
        std::vector<double> s_exp(nT), s_var(nT);
        for (std::size_t t = 0; t < nT; ++t) {
            const double beta = 1.0 / temperatures[t];
            double Z = 0.0, sO = 0.0, sO2 = 0.0;
            for (std::size_t i = 0; i < m; ++i) {
                const double bw =
                    weights[i] * std::exp(-beta * (ritz_values[i] - e_min));
                Z += bw; sO += bw * corr[i]; sO2 += bw * corr[i] * corr[i];
            }
            const double invZ = (Z > 1e-300) ? (1.0 / Z) : 0.0;
            const double mean = sO * invZ;
            s_exp[t] = mean;
            s_var[t] = sO2 * invZ - mean * mean;
        }
        sample_exp.push_back(std::move(s_exp));
        sample_var.push_back(std::move(s_var));
    }

    // 6. Average over valid samples + std error on the mean.
    const std::size_t nv = sample_exp.size();
    R.total_samples = nv;
    if (nv == 0) return R;
    for (const auto& se : sample_exp)
        for (std::size_t t = 0; t < nT; ++t) R.expectation[t] += se[t];
    for (const auto& sv : sample_var)
        for (std::size_t t = 0; t < nT; ++t) R.variance[t] += sv[t];
    const double invn = 1.0 / static_cast<double>(nv);
    for (std::size_t t = 0; t < nT; ++t) {
        R.expectation[t]    *= invn;
        R.variance[t]       *= invn;
        R.susceptibility[t]  = R.variance[t] / temperatures[t];
    }
    if (nv > 1) {
        std::vector<double> acc_e(nT, 0.0), acc_v(nT, 0.0), acc_s(nT, 0.0);
        for (std::size_t s = 0; s < nv; ++s)
            for (std::size_t t = 0; t < nT; ++t) {
                const double de = sample_exp[s][t] - R.expectation[t];
                const double dv = sample_var[s][t] - R.variance[t];
                const double ds = (sample_var[s][t] / temperatures[t])
                                  - R.susceptibility[t];
                acc_e[t] += de * de; acc_v[t] += dv * dv; acc_s[t] += ds * ds;
            }
        const double denom =
            std::sqrt(static_cast<double>(nv * (nv - 1)));
        for (std::size_t t = 0; t < nT; ++t) {
            R.expectation_error[t]    = std::sqrt(acc_e[t]) / denom;
            R.variance_error[t]       = std::sqrt(acc_v[t]) / denom;
            R.susceptibility_error[t] = std::sqrt(acc_s[t]) / denom;
        }
    }
    return R;
}

}  // namespace detail

}  // namespace ed::observables
