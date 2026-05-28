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
//
// Phase G of the "Close CPU/GPU Gaps" plan (May 2026):
// ``detail::kpm_dynamical_kernel_via_backend<Backend>`` ships a fully
// device-resident Chebyshev recursion (3-term ``v_k = 2 H_sc v_{k-1}
// - v_{k-2}`` + per-moment ``mu_k = <left | v_k>``) that uses
// ``Backend::dot`` / ``axpy`` / ``scale`` and a Backend-bound
// ``apply_H`` callable. The legacy host body in
// ``::compute_kpm_ltlm_from_states`` is retained for the CPU lane to
// preserve HDF5 logging + console diagnostics; the new templated body
// is the CUDA lane.
// =============================================================================

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <random>
#include <stdexcept>
#include <utility>
#include <vector>

#include <ed/krylov/lanczos_kernel.h>
#include <ed/matvec/backend.h>
#include <ed/matvec/matvec.h>
#include <ed/solvers/ftlm_kpm.h>
#include <ed/solvers/kpm_dos.h>  // estimate_spectral_bounds (host helper)
#include <ed/solvers/lanczos.h>  // diagonalize_tridiagonal_ritz

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

namespace detail {

/// Backend-templated spectral bounds estimator. Mirrors the host
/// ``ed::kpm_dos::estimate_spectral_bounds`` but uses
/// ``lanczos_kernel<Backend>`` so the matvec runs on the same device
/// the rest of the KPM kernel uses. Returns the smallest and largest
/// Ritz values of a short Lanczos sweep starting from a host-seeded
/// Gaussian random vector.
template <typename Backend, typename ApplyH>
void estimate_spectral_bounds_via_backend(
    const Backend& be,
    ApplyH&&       apply_H,
    std::size_t    local_n,
    int            krylov_dim,
    std::uint64_t  seed,
    double&        e_min,
    double&        e_max)
{
    if (local_n == 0 || krylov_dim < 2) {
        e_min = 0.0;
        e_max = 1.0;
        return;
    }
    std::mt19937_64 rng(seed != 0 ? seed : 0xBABEFADEULL);
    std::normal_distribution<double> gauss(0.0, 1.0);

    std::vector<Complex> v0_host(local_n);
    for (auto& c : v0_host) c = Complex(gauss(rng), gauss(rng));
    double sumsq = 0.0;
    for (const auto& c : v0_host) sumsq += std::norm(c);
    const double nrm = std::sqrt(sumsq);
    if (!(nrm > 0.0)) { e_min = 0.0; e_max = 1.0; return; }
    const double inv = 1.0 / nrm;
    for (auto& c : v0_host) c *= inv;

    auto v0 = be.make_zero_vector(local_n);
    be.copy_from_host(v0_host.data(), v0.get(), local_n);

    ed::krylov::LanczosKernelOptions kopts;
    kopts.max_iter      = static_cast<std::size_t>(krylov_dim);
    kopts.keep_basis    = false;
    // Spectral-bound estimation only needs a faithful tridiagonal;
    // ``LocalDGKS3`` keeps Lanczos numerically stable without
    // requiring the basis to be kept around (``FullCGS2`` does).
    kopts.reorth        = ed::krylov::ReorthPolicy::LocalDGKS3;
    kopts.breakdown_tol = 1e-10;
    auto k = ed::krylov::lanczos_kernel(
        be, apply_H, local_n, v0.get(), kopts);
    if (k.alpha.empty()) { e_min = 0.0; e_max = 1.0; return; }

    std::vector<double> ritz, w;
    diagonalize_tridiagonal_ritz(k.alpha, k.beta, ritz, w);
    if (ritz.empty()) { e_min = 0.0; e_max = 1.0; return; }
    e_min = ritz.front();
    e_max = ritz.back();
}

/// Inline copies of the Jackson / Lorentz kernel coefficients
/// (matches ``src/solvers/cpu/ftlm_kpm.cpp``::make_jackson_kernel and
/// ::make_lorentz_kernel byte-for-byte). Pulled inline so the
/// header-only backend-templated path doesn't depend on the
/// ed_solvers_cpu translation unit.
inline std::vector<double> make_jackson_kernel(int M) {
    std::vector<double> g(M);
    const double Mp1 = static_cast<double>(M + 1);
    const double cot_term = 1.0 / std::tan(M_PI / Mp1);
    for (int k = 0; k < M; ++k) {
        const double kd = static_cast<double>(k);
        const double phi = M_PI * kd / Mp1;
        g[k] = ((Mp1 - kd) * std::cos(phi)
                + std::sin(phi) * cot_term) / Mp1;
    }
    return g;
}

inline std::vector<double> make_lorentz_kernel(int M, double lambda) {
    std::vector<double> g(M);
    const double sh_lambda = std::sinh(lambda);
    for (int k = 0; k < M; ++k) {
        const double x = lambda * (1.0 - static_cast<double>(k) / M);
        g[k] = std::sinh(x) / sh_lambda;
    }
    return g;
}

/// Backend-templated multi-moment KPM dynamical correlator. Computes
/// ``S(omega) = -Im <psi| A^† delta(omega + E_ref - H_sc) B |psi> / pi``
/// via Chebyshev expansion of the rescaled Hamiltonian
/// ``H_sc = (H - b) / a``, where ``a`` covers the spectral half-width
/// around the reference energy ``b = E_ref`` (the configurable buffer
/// pad guards against eigenvalues hitting the ``[-1, 1]`` boundary).
///
/// Algorithm (single-seed, beta = 0 -- the only mode the orchestrator
/// drives today via ``SpectralOptions::Method::KpmDynamical``):
///   1. Stage ``|psi>`` on the backend.
///   2. ``E_ref = <psi | H | psi>`` via one matvec + dot.
///   3. Spectral bounds ``[e_lo, e_hi]`` via
///      ``estimate_spectral_bounds_via_backend`` (a short device-resident
///      Lanczos sweep), then ``a = (1 + buffer) * max(E_ref - e_lo,
///      e_hi - E_ref)``, ``b = E_ref``.
///   4. ``|left> = A |psi>``, ``|right> = B |psi>``.
///   5. Chebyshev recursion on device:
///        ``T_0 = |right>``
///        ``T_1 = H_sc T_0``
///        ``T_k = 2 H_sc T_{k-1} - T_{k-2}`` for k = 2..M-1
///      and ``mu_k = <left | T_k>`` via ``backend.dot``.
///   6. Host-side kernel coefficients (Jackson or Lorentz) +
///      Chebyshev evaluation of S(omega) on the user grid.
template <typename Backend, typename ApplyH, typename ApplyA,
          typename ApplyB>
KpmDynamicalResult kpm_dynamical_kernel_via_backend(
    const Backend&             be,
    ApplyH&&                   apply_H,
    ApplyA&&                   apply_A,
    ApplyB&&                   apply_B,
    const Complex*             psi_host,
    std::size_t                local_n,
    const std::vector<double>& omega_grid,
    const KpmDynamicalOptions& opts)
{
    if (local_n == 0) {
        throw std::invalid_argument(
            "kpm_dynamical_kernel_via_backend: local_n == 0");
    }
    if (omega_grid.empty()) {
        throw std::invalid_argument(
            "kpm_dynamical_kernel_via_backend: empty frequency grid");
    }
    const int M = static_cast<int>(opts.num_moments);
    if (M < 1) {
        throw std::invalid_argument(
            "kpm_dynamical_kernel_via_backend: num_moments must be >= 1");
    }

    // -------- 1. Stage psi on device --------------------------------
    auto psi = be.make_zero_vector(local_n);
    be.copy_from_host(psi_host, psi.get(), local_n);
    const double psi_nrm = be.nrm2(psi.get(), local_n);
    if (psi_nrm > 1e-14) {
        be.scale(Complex(1.0 / psi_nrm, 0.0), psi.get(), local_n);
    }

    // -------- 2. E_ref = <psi | H | psi> ----------------------------
    auto Hpsi = be.make_zero_vector(local_n);
    apply_H(psi.get(), Hpsi.get(), local_n);
    const Complex e_ref_c = be.dot(psi.get(), Hpsi.get(), local_n);
    const double  E_ref   = e_ref_c.real();

    // -------- 3. Spectral bounds + rescaling ------------------------
    double e_lo = E_ref;
    double e_hi = E_ref;
    estimate_spectral_bounds_via_backend(
        be, apply_H, local_n,
        opts.spectral_bounds_krylov,
        opts.random_seed,
        e_lo, e_hi);
    const double half_width = std::max(E_ref - e_lo, e_hi - E_ref);
    const double a = (1.0 + opts.spectral_bound_buffer)
                       * std::max(half_width, 1e-6);
    const double b = E_ref;

    // -------- 4. |left> = A |psi>, |right> = B |psi> ----------------
    auto left  = be.make_zero_vector(local_n);
    auto right = be.make_zero_vector(local_n);
    apply_A(psi.get(), left.get(),  local_n);
    apply_B(psi.get(), right.get(), local_n);

    // -------- 5. Chebyshev recursion on device ----------------------
    // Working buffers: v_prev (T_{k-2}), v_curr (T_{k-1}), v_next (T_k),
    // Hv (scratch for H * v_curr).
    auto v_prev = be.make_zero_vector(local_n);
    auto v_curr = be.make_zero_vector(local_n);
    auto v_next = be.make_zero_vector(local_n);
    auto Hv     = be.make_zero_vector(local_n);
    be.copy(right.get(), v_curr.get(), local_n);

    std::vector<Complex> mu(M, Complex(0.0, 0.0));
    mu[0] = be.dot(left.get(), v_curr.get(), local_n);

    if (M > 1) {
        // v_next = H_sc * v_curr = (H*v_curr - b*v_curr) / a
        apply_H(v_curr.get(), v_next.get(), local_n);
        be.axpy(Complex(-b, 0.0), v_curr.get(), v_next.get(), local_n);
        be.scale(Complex(1.0 / a, 0.0), v_next.get(), local_n);
        mu[1] = be.dot(left.get(), v_next.get(), local_n);
        std::swap(v_prev, v_curr);
        std::swap(v_curr, v_next);
    }

    for (int k = 2; k < M; ++k) {
        // v_next = 2 * H_sc * v_curr - v_prev
        //        = 2 * ((H*v_curr - b*v_curr) / a) - v_prev
        apply_H(v_curr.get(), Hv.get(), local_n);
        be.axpy(Complex(-b, 0.0), v_curr.get(), Hv.get(), local_n);
        be.scale(Complex(1.0 / a, 0.0), Hv.get(), local_n);
        be.copy(Hv.get(), v_next.get(), local_n);
        be.scale(Complex(2.0, 0.0), v_next.get(), local_n);
        be.axpy(Complex(-1.0, 0.0), v_prev.get(), v_next.get(), local_n);

        mu[k] = be.dot(left.get(), v_next.get(), local_n);
        std::swap(v_prev, v_curr);
        std::swap(v_curr, v_next);
    }

    // -------- 6. Host-side kernel + spectral function ---------------
    const std::vector<double> g = (opts.kernel == KpmKernel::Jackson)
        ? make_jackson_kernel(M)
        : make_lorentz_kernel(M, opts.lorentz_lambda);

    KpmDynamicalResult out;
    out.omega         = omega_grid;
    out.spectral_real.assign(omega_grid.size(), 0.0);
    out.spectral_imag.assign(omega_grid.size(), 0.0);
    out.kpm_a         = a;
    out.kpm_b         = b;

    // Convention matches ``accumulate_kpm_inner``:
    // ``omega`` is the energy transfer (excited state above E_ref),
    // so the absolute energy is ``eps = omega + E_ref`` and we
    // sample T_k(x) at x = (eps - b) / a = (omega + E_ref - E_ref) / a
    // = omega / a. (b == E_ref by construction here.) We keep the
    // explicit form for symmetry with the CPU path and to support a
    // future per-state buffer pad if E_ref is overridden.
    for (std::size_t i = 0; i < omega_grid.size(); ++i) {
        const double eps = omega_grid[i] + E_ref;
        const double x   = (eps - b) / a;
        if (x <= -1.0 + 1e-10 || x >= 1.0 - 1e-10) continue;

        double Tk_prev = 1.0;
        double Tk_curr = x;
        double sum     = g[0] * mu[0].real();
        if (M > 1) {
            sum += 2.0 * g[1] * mu[1].real() * Tk_curr;
        }
        for (int k = 2; k < M; ++k) {
            const double Tk_next = 2.0 * x * Tk_curr - Tk_prev;
            sum += 2.0 * g[k] * mu[k].real() * Tk_next;
            Tk_prev = Tk_curr;
            Tk_curr = Tk_next;
        }
        const double denom = M_PI * a * std::sqrt(1.0 - x * x);
        out.spectral_real[i] = sum / denom;
    }
    return out;
}

}  // namespace detail

}  // namespace ed::observables
