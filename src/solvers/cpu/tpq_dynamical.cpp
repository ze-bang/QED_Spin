// =============================================================================
// src/solvers/cpu/tpq_dynamical.cpp
//
// TPQ dynamical correlator implementation.
// See include/ed/solvers/tpq_dynamical.h for mathematical derivation.
// =============================================================================

#include <ed/solvers/tpq_dynamical.h>
#include <ed/solvers/lanczos.h>   // build_lanczos_tridiagonal_with_basis,
                                   // diagonalize_tridiagonal_ritz,
                                   // generateGaussianRandomVector

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace ed::tpq::dynamical {

namespace {

// ---------------------------------------------------------------------------
// Verbose flag
// ---------------------------------------------------------------------------
inline bool tpq_verbose() {
    static const bool v = []() {
        const char* e = std::getenv("ED_TPQ_DYN_VERBOSE");
        return e && e[0] == '1';
    }();
    return v;
}

// ---------------------------------------------------------------------------
// Modified Bessel sequence I_0..I_{M-1}(s) using C++17 std::cyl_bessel_i.
// Truncates when terms are negligible relative to I_0.
// ---------------------------------------------------------------------------
std::vector<double> modified_bessel_sequence(int M, double s) {
    if (M <= 0 || s < 0.0) return {};
    std::vector<double> out(M, 0.0);
    if (s < 1e-10) {
        out[0] = 1.0;  // I_0(0) = 1, I_k(0) = 0 for k >= 1
        return out;
    }
    const double I0 = std::cyl_bessel_i(0, s);
    out[0] = I0;
    for (int k = 1; k < M; ++k) {
        const double Ik = std::cyl_bessel_i(static_cast<double>(k), s);
        out[k] = Ik;
        // Early termination: once terms are negligible vs I0, stop.
        if (k > static_cast<int>(s) + 10 && std::abs(Ik) < 1e-15 * std::abs(I0))
            break;
    }
    return out;
}

// ---------------------------------------------------------------------------
// Chebyshev expansion of e^{-s H_sc} |v⟩ for s = a*beta/2
//
//   e^{-s H_sc}|v⟩ ≈ I_0(s) T_0 + 2 Σ_{k=1}^{M-1} (-1)^k I_k(s) T_k(H_sc)|v⟩
//
// Input:  v        — initial vector
//         H_sc     — rescaled Hamiltonian MatVec
//         s        — expansion parameter (positive real)
//         M        — number of moments
// Output: result — e^{-s H_sc}|v⟩ (approximate)
//         norm_sq — ‖result‖² (for Z estimation)
// ---------------------------------------------------------------------------
void cheby_boltzmann(
    MatVec H_sc,
    const ComplexVector& v,
    double s,
    int M,
    std::uint64_t dim,
    ComplexVector& result,
    double& norm_sq)
{
    const int d = static_cast<int>(dim);
    auto Ik = modified_bessel_sequence(M, s);

    // Three-vector Chebyshev recursion.
    ComplexVector v_prev(dim, 0.0), v_curr(v), v_next(dim, 0.0), Hv(dim, 0.0);

    // result = I_0(s) * T_0|v⟩ = I_0(s) * v
    result.assign(dim, Complex(0.0, 0.0));
    const Complex c0(Ik[0], 0.0);
    cblas_zaxpy(d, &c0, v_curr.data(), 1, result.data(), 1);

    for (int k = 1; k < M; ++k) {
        if (k == 1) {
            H_sc(v_curr.data(), v_next.data(), d);
        } else {
            H_sc(v_curr.data(), Hv.data(), d);
            const Complex c2(2.0, 0.0), cm1(-1.0, 0.0);
            cblas_zcopy(d, Hv.data(), 1, v_next.data(), 1);
            cblas_zscal(d, &c2, v_next.data(), 1);
            cblas_zaxpy(d, &cm1, v_prev.data(), 1, v_next.data(), 1);
        }
        // coefficient: 2 * (-1)^k * I_k(s)
        const double sign = (k % 2 == 0) ? 1.0 : -1.0;
        const Complex ck(2.0 * sign * Ik[k], 0.0);
        cblas_zaxpy(d, &ck, v_next.data(), 1, result.data(), 1);

        std::swap(v_prev, v_curr);
        std::swap(v_curr, v_next);
    }

    // ‖result‖²
    norm_sq = 0.0;
    for (const auto& c : result) norm_sq += std::norm(c);
}

// ---------------------------------------------------------------------------
// Chebyshev expansion of e^{i s H_sc} |v⟩ for real s (time propagation).
//
//   e^{is H_sc}|v⟩ ≈ J_0(s) T_0 + 2 Σ_{k=1}^{M-1} i^k J_k(s) T_k(H_sc)|v⟩
//
// Combined with the energy shift: e^{iHt}|v⟩ = e^{ibt} e^{iat H_sc}|v⟩.
// Computes ONLY the first step (without the e^{ibt} phase); caller multiplies.
// ---------------------------------------------------------------------------
void cheby_time_evolve(
    MatVec H_sc,
    const ComplexVector& v,
    double s,   // = a * t  (rescaled time)
    int M,
    std::uint64_t dim,
    ComplexVector& result)
{
    const int d = static_cast<int>(dim);

    // Bessel coefficients (standard, not modified).
    // J_k(s) computed using the standard forward recursion (safe for |s| < k).
    // Use downward Miller for safety:
    const int K_start = M + static_cast<int>(2.0 * std::abs(s) + 40);
    std::vector<double> Jk(K_start + 2, 0.0);
    Jk[K_start]     = 0.0;
    Jk[K_start + 1] = 0.0;
    // Seed with small value and normalise via J_0 + 2*Σ(-1)^m J_{2m} = cos(s)... 
    // Actually use the simpler forward J recursion for small M:
    //   J_{k+1} = (2k/s) J_k - J_{k-1}  (valid for all k < s, unstable for k > s)
    // For |s| large, use asymptotic.  For safety, always use Boost-style backward.
    // Compute Bessel coefficients J_k(s) using C++17 std::cyl_bessel_j.
    // This is numerically stable for all s and k.
    Jk.assign(M, 0.0);
    if (std::abs(s) < 1e-12) {
        // e^{i*0}|v⟩ = |v⟩
        Jk[0] = 1.0;
    } else {
        const double J0 = std::cyl_bessel_j(0, s);
        Jk[0] = J0;
        for (int k = 1; k < M; ++k) {
            Jk[k] = std::cyl_bessel_j(static_cast<double>(k), s);
            // Early termination: truncate once terms are negligible.
            if (k > static_cast<int>(std::abs(s)) + 10 &&
                std::abs(Jk[k]) < 1e-15 * std::abs(J0))
                break;
        }
    }

    // Three-vector Chebyshev recursion, accumulating i^k J_k(s).
    ComplexVector v_prev(dim, 0.0), v_curr(v), v_next(dim, 0.0), Hv(dim, 0.0);

    result.assign(dim, Complex(0.0, 0.0));
    // k=0: J_0(s) * T_0|v⟩
    const Complex c0(Jk[0], 0.0);
    cblas_zaxpy(d, &c0, v_curr.data(), 1, result.data(), 1);

    // i^k sequence: 1, i, -1, -i, 1, i, -1, -i ...
    static const Complex ik_table[4] = {
        {1.0, 0.0}, {0.0, 1.0}, {-1.0, 0.0}, {0.0, -1.0}
    };

    for (int k = 1; k < M; ++k) {
        if (k == 1) {
            H_sc(v_curr.data(), v_next.data(), d);
        } else {
            H_sc(v_curr.data(), Hv.data(), d);
            const Complex c2(2.0, 0.0), cm1(-1.0, 0.0);
            cblas_zcopy(d, Hv.data(), 1, v_next.data(), 1);
            cblas_zscal(d, &c2, v_next.data(), 1);
            cblas_zaxpy(d, &cm1, v_prev.data(), 1, v_next.data(), 1);
        }
        // coeff = 2 * i^k * J_k(s)   (factor 2 for k≥1)
        const Complex ck = 2.0 * Jk[k] * ik_table[k % 4];
        cblas_zaxpy(d, &ck, v_next.data(), 1, result.data(), 1);

        std::swap(v_prev, v_curr);
        std::swap(v_curr, v_next);
    }
}

// ---------------------------------------------------------------------------
// Window function w(j, n_t) at time point j out of n_t total.
// ---------------------------------------------------------------------------
double window_value(WindowFunction wf, int j, int n_t) {
    if (n_t <= 1) return 1.0;
    const double x = static_cast<double>(j) / (n_t - 1);
    switch (wf) {
        case WindowFunction::Hann:
            return std::cos(0.5 * M_PI * x) * std::cos(0.5 * M_PI * x);
        case WindowFunction::Linear:
            return 1.0 - x;
        case WindowFunction::None:
        default:
            return 1.0;
    }
}

// ---------------------------------------------------------------------------
// Estimate energy bounds via a short Lanczos.
// Returns {E_min, E_max} (lowest and highest eigenvalue estimates).
// ---------------------------------------------------------------------------
std::pair<double,double> estimate_energy_bounds(
    MatVec H, std::uint64_t dim, int lanczos_dim, std::mt19937& gen)
{
    ComplexVector v0 = generateGaussianRandomVector(static_cast<int>(dim), gen);
    std::vector<double> alpha, beta;
    std::vector<ComplexVector> dummy;
    build_lanczos_tridiagonal_with_basis(
        H, v0, dim, lanczos_dim, 1e-10, false, 10, alpha, beta, &dummy);
    if (alpha.empty()) return {0.0, 1.0};
    std::vector<double> E, w, U;
    diagonalize_tridiagonal_ritz(alpha, beta, E, w, &U);
    const double lo = E.empty() ? 0.0 : E.front();
    const double hi = E.empty() ? 1.0 : E.back();
    return {lo, hi};
}

// ---------------------------------------------------------------------------
// Core TPQ dynamical computation for one sample |r⟩ at multiple β.
// Accumulates G(t_j, β) into G_acc and Z into Z_acc.
// ---------------------------------------------------------------------------
void compute_tpq_sample(
    MatVec H,
    MatVec O1, MatVec O2,
    std::uint64_t dim,
    const std::vector<double>& betas,
    double a, double b,           // KPM rescaling: H_sc = (H-b)/a
    const TPQParameters& params,
    const ComplexVector& r,       // raw random vector
    bool is_self,
    // Output accumulators: G_acc[beta_idx][time_idx], Z_acc[beta_idx]
    std::vector<std::vector<Complex>>& G_acc,
    std::vector<double>& Z_acc)
{
    const int d    = static_cast<int>(dim);
    const int n_T  = static_cast<int>(betas.size());
    const int n_t  = params.n_time;
    const double dt = params.t_max / (n_t - 1);

    // H_sc MatVec.
    auto H_sc = [&H, a, b](const Complex* in, Complex* out, int n) {
        H(in, out, n);
        for (int i = 0; i < n; ++i) out[i] = (out[i] - b * in[i]) / a;
    };

    for (int tidx = 0; tidx < n_T; ++tidx) {
        const double beta  = betas[tidx];
        const double s_b   = a * beta / 2.0;  // parameter for e^{-βH/2}|r⟩

        // Compute |ψ(β)⟩ = e^{-βH/2}|r⟩ (unnormalised).
        ComplexVector psi(dim), psi_right(dim);
        double psi_norm2 = 0.0;
        cheby_boltzmann(H_sc, r, s_b, params.boltzmann_moments, dim, psi, psi_norm2);

        // Partition function estimate: Z ≈ D * ‖e^{-βH/2}|r⟩‖² * e^{-βb}
        // (the e^{-βb} factor from the unrescaled Boltzmann weight)
        const double Z_est = static_cast<double>(dim) * psi_norm2
            * std::exp(-beta * b);
        Z_acc[tidx] += Z_est;

        // The Boltzmann weight for this sample at this β is psi_norm2 * exp(-β*b).
        // When we normalise |ψ⟩, we divide out the weight, so we must multiply
        // G back by psi_norm2 * exp(-β*b) to get the weighted G contribution.
        // (Factor D cancels when G is divided by Z = D * Σ_r psi_norm2 * e^{-βb}.)
        const double weight = psi_norm2 * std::exp(-beta * b);

        // Normalise |ψ(β)⟩.
        const double psi_norm = std::sqrt(psi_norm2);
        if (psi_norm < 1e-14) continue;
        const Complex inv_norm(1.0 / psi_norm, 0.0);
        cblas_zscal(d, &inv_norm, psi.data(), 1);

        // |φ_R⟩ = O2|ψ⟩,  |φ_L⟩ = O1|ψ⟩ (or same as φ_R for self-corr).
        O2(psi.data(), psi_right.data(), d);
        ComplexVector psi_left(dim);
        if (is_self) {
            psi_left = psi_right;
        } else {
            O1(psi.data(), psi_left.data(), d);
        }

        // G(t=0) = weight * ⟨φ_L|φ_R⟩
        Complex G0;
        cblas_zdotc_sub(d, psi_left.data(), 1, psi_right.data(), 1, &G0);
        const double w0 = window_value(params.window, 0, n_t);
        G_acc[tidx][0] += w0 * G0 * weight;

        // Time-evolve |φ_R(t)⟩ = e^{iHt}|φ_R⟩ step by step.
        // We accumulate G(t_j) = ⟨φ_L|φ_R(t_j)⟩.
        //
        // For efficiency: use single Chebyshev expansion at each time step
        // rather than re-expanding from scratch.  We propagate by Δt each step.
        //
        // e^{iHΔt}|v⟩ = e^{ibΔt} * e^{iaΔt H_sc}|v⟩
        // The e^{iΔt H_sc} is computed once and applied M times.

        const double s_dt = a * dt;   // rescaled time step
        const double phase_per_step = b * dt;  // global phase per step

        ComplexVector v_t(psi_right);  // |φ_R(t=0)⟩
        ComplexVector v_evolved(dim);

        for (int j = 1; j < n_t; ++j) {
            cheby_time_evolve(H_sc, v_t, s_dt, params.time_moments, dim, v_evolved);
            // Apply global phase: e^{ib*(j*dt)}
            const double phase = phase_per_step * j;
            // Note: cheby_time_evolve produces e^{ias H_sc}|v⟩ only; multiply by
            // the phase accumulated over ALL previous steps, not just this step.
            // The correct phase for t = j*dt is e^{ib*j*dt}.
            // Since we propagate step-by-step, v_evolved = e^{ia*dt*H_sc}|v_t⟩,
            // and the accumulated phase is e^{ib*(step)*dt} per step.
            // We track the cumulative phase separately.

            // Actually: v_t carries no phase; we track phase externally.
            // G(t_j) = e^{ib*j*dt} * ⟨φ_L| v_evolved_from_phi_R(t=0)⟩
            // But since we propagate step-by-step, v_evolved is
            //   e^{ia*dt*H_sc}|v_{t-1}⟩  (one step from the previous state).
            // So v_t after j steps = e^{ij*a*dt*H_sc}|φ_R⟩, and the full
            // propagated vector is e^{ij*b*dt} * v_t.
            std::swap(v_t, v_evolved);

            // G(t_j) = ⟨φ_L| e^{i*j*H*dt} |φ_R⟩
            //        = e^{i*j*b*dt} * ⟨φ_L|v_t⟩
            Complex G_inner;
            cblas_zdotc_sub(d, psi_left.data(), 1, v_t.data(), 1, &G_inner);
            const Complex phase_factor(std::cos(phase), std::sin(phase));
            const double wj = window_value(params.window, j, n_t);
            G_acc[tidx][j] += wj * phase_factor * G_inner * weight;
        }
    }
}

// ---------------------------------------------------------------------------
// Fourier transform G(t) → S(ω)
//
//   S(ω) = (1/π) Re[ ∫_0^{t_max} G(t) e^{iωt} dt ]
//         ≈ (Δt/π) Re[ Σ_j G(t_j) e^{iω t_j} ]   (already windowed in G)
// ---------------------------------------------------------------------------
void fourier_transform_G(
    const std::vector<Complex>& G_t,
    double dt,
    const std::vector<double>& omega_grid,
    std::vector<double>& S_out)
{
    const int n_t  = static_cast<int>(G_t.size());
    const int n_om = static_cast<int>(omega_grid.size());
    S_out.assign(n_om, 0.0);

    for (int j = 0; j < n_t; ++j) {
        const double t = j * dt;
        const Complex Gj = G_t[j];
        for (int i = 0; i < n_om; ++i) {
            const double phi = omega_grid[i] * t;
            const Complex phase(std::cos(phi), std::sin(phi));
            S_out[i] += (phase * Gj).real();
        }
    }
    const double scale = dt / M_PI;
    for (auto& s : S_out) s *= scale;
}

// ---------------------------------------------------------------------------
// Linspace helper
// ---------------------------------------------------------------------------
std::vector<double> linspace_tpq(double lo, double hi, int N) {
    std::vector<double> v(N);
    for (int i = 0; i < N; ++i)
        v[i] = lo + (hi - lo) * i / (N - 1);
    return v;
}

} // anonymous namespace

// =============================================================================
// Internal shared implementation
// =============================================================================
static TPQDynamicalResult compute_tpq_impl(
    MatVec H,
    MatVec O1,
    MatVec O2,
    std::uint64_t dim,
    const std::vector<double>& betas,
    double omega_min,
    double omega_max,
    const TPQParameters& params)
{
    if (dim == 0)    throw std::invalid_argument("tpq: dim must be > 0");
    if (betas.empty()) throw std::invalid_argument("tpq: betas must be non-empty");
    if (params.n_time < 2) throw std::invalid_argument("tpq: n_time must be >= 2");
    if (params.n_omega < 2) throw std::invalid_argument("tpq: n_omega must be >= 2");

    const int n_T  = static_cast<int>(betas.size());
    const int n_t  = params.n_time;
    const int n_om = params.n_omega;
    const double dt = params.t_max / (n_t - 1);

    // Detect self-correlation.
    const bool is_self = [&]() {
        if (dim == 0) return false;
        ComplexVector probe(dim, 0.0); probe[0] = 1.0;
        ComplexVector a(dim), b(dim);
        O1(probe.data(), a.data(), static_cast<int>(dim));
        O2(probe.data(), b.data(), static_cast<int>(dim));
        double sc = 0.0; Complex diff(0.0);
        for (std::size_t i = 0; i < dim; ++i) {
            diff += a[i] - b[i]; sc += std::norm(a[i]) + std::norm(b[i]);
        }
        return sc > 0.0 && std::abs(diff) < 1e-14 * std::sqrt(sc);
    }();

    // Energy bounds.
    std::mt19937_64 gen64;
    {
        if (params.random_seed == 0) {
            std::random_device rd; gen64.seed(rd());
        } else {
            gen64.seed(params.random_seed);
        }
    }

    double E_min = params.E_min, E_max = params.E_max;
    if (std::abs(E_min) < 1e-14 && std::abs(E_max) < 1e-14) {
        std::mt19937 g32(static_cast<std::uint32_t>(gen64()));
        auto [lo, hi] = estimate_energy_bounds(H, dim, params.bound_lanczos_dim, g32);
        E_min = lo; E_max = hi;
    }

    const double BW     = E_max - E_min;
    const double buffer = params.spectral_bound_buffer * std::max(BW, 1.0);
    const double kpm_lo = E_min - buffer;
    const double kpm_hi = E_max + buffer;
    const double a      = (kpm_hi - kpm_lo) / 2.0;
    const double b_kpm  = (kpm_hi + kpm_lo) / 2.0;

    const double energy_shift = (std::abs(params.energy_shift) > 0.0)
        ? params.energy_shift : E_min;

    const std::vector<double> omega_grid = linspace_tpq(omega_min, omega_max, n_om);

    // Accumulators: G_acc[beta_idx][time_idx], Z_acc[beta_idx].
    std::vector<std::vector<Complex>> G_acc(n_T, std::vector<Complex>(n_t, 0.0));
    std::vector<double> Z_acc(n_T, 0.0);

    // Main sample loop.
    for (int s = 0; s < params.num_samples; ++s) {
        std::mt19937 g32(static_cast<std::uint32_t>(
            gen64() ^ (0xDEADBEEF12345678ULL * (s + 1))));
        const ComplexVector r = generateGaussianRandomVector(
            static_cast<int>(dim), g32);

        if (tpq_verbose()) {
            std::cerr << "[TPQ] sample " << s+1 << "/" << params.num_samples
                      << " a=" << a << " b=" << b_kpm << "\n";
        }

        compute_tpq_sample(H, O1, O2, dim, betas, a, b_kpm, params,
                           r, is_self, G_acc, Z_acc);
    }

    // Build result.
    TPQDynamicalResult out;
    out.frequencies          = omega_grid;
    out.betas                = betas;
    out.ground_state_estimate = E_min;
    out.energy_shift_used    = energy_shift;
    out.total_samples        = params.num_samples;
    out.kpm_a                = a;
    out.kpm_b                = b_kpm;
    out.t_grid               = linspace_tpq(0.0, params.t_max, n_t);

    // Average G over samples and Fourier transform.
    out.partition_function.resize(n_T);
    out.spectral_real.resize(n_T * n_om, 0.0);
    out.spectral_imag.resize(n_T * n_om, 0.0);
    out.static_correlator.resize(n_T, 0.0);

    const double R = static_cast<double>(params.num_samples);
    for (int tidx = 0; tidx < n_T; ++tidx) {
        // Average G(t) and Z.
        const double Z = Z_acc[tidx] / R;
        out.partition_function[tidx] = Z;

        std::vector<Complex> G_avg(n_t);
        for (int j = 0; j < n_t; ++j)
            G_avg[j] = G_acc[tidx][j] / R;

        // Fourier transform → S(ω,β).
        // S(ω,β) = D * FT[ G_avg ] / Z_avg
        // where G_avg = (1/R) Σ_r weight_r * G_r(ω)
        //       Z_avg  = (1/R) Σ_r D * weight_r  = D * (1/R) Σ weight_r
        // So D * G_avg / Z_avg = D * G_avg / (D * W_avg) = G_avg / W_avg,
        // where W_avg = (1/R) Σ weight_r = Z_avg / D.
        // => divide FT[G_avg] by (Z / D).
        if (Z > 0.0) {
            const double W_avg = Z / static_cast<double>(dim);  // Z/D
            std::vector<double> S_omega;
            fourier_transform_G(G_avg, dt, omega_grid, S_omega);
            // S = (1/Z) * FT[G]
            const double dw = (n_om > 1)
                ? (omega_grid.back() - omega_grid.front()) / (n_om - 1) : 1.0;
            double integral = 0.0;
            for (int i = 0; i < n_om; ++i) {
                const double v = S_omega[i] / W_avg;
                out.spectral_real[tidx * n_om + i] = v;
                integral += v;
            }
            out.static_correlator[tidx] = integral * dw;
        }
    }

    // Save last-sample G for diagnostics.
    out.G_time.resize(n_T * n_t);
    for (int tidx = 0; tidx < n_T; ++tidx)
        for (int j = 0; j < n_t; ++j)
            out.G_time[tidx * n_t + j] = G_acc[tidx][j] / R;

    return out;
}

// =============================================================================
// Public API
// =============================================================================

TPQDynamicalResult compute_tpq_dynamical(
    MatVec H, MatVec O1, MatVec O2,
    std::uint64_t dim,
    double beta,
    double omega_min, double omega_max,
    const TPQParameters& params)
{
    return compute_tpq_impl(H, O1, O2, dim, {beta}, omega_min, omega_max, params);
}

TPQDynamicalResult compute_tpq_dynamical_multi_beta(
    MatVec H, MatVec O1, MatVec O2,
    std::uint64_t dim,
    const std::vector<double>& betas,
    double omega_min, double omega_max,
    const TPQParameters& params)
{
    if (betas.empty())
        throw std::invalid_argument("tpq multi-beta: betas must be non-empty");
    return compute_tpq_impl(H, O1, O2, dim, betas, omega_min, omega_max, params);
}

TPQDynamicalResult compute_tpq_dynamical_from_state(
    MatVec H, MatVec O1, MatVec O2,
    const ComplexVector& psi,
    double beta,
    double Z_estimate,
    double omega_min, double omega_max,
    const TPQParameters& params)
{
    const std::uint64_t dim = psi.size();
    if (dim == 0) throw std::invalid_argument("tpq from_state: psi is empty");

    const int n_t  = params.n_time;
    const int n_om = params.n_omega;
    const double dt = params.t_max / (n_t - 1);

    // Detect self-correlation.
    const bool is_self = [&]() {
        ComplexVector probe(dim, 0.0); probe[0] = 1.0;
        ComplexVector a(dim), b(dim);
        O1(probe.data(), a.data(), static_cast<int>(dim));
        O2(probe.data(), b.data(), static_cast<int>(dim));
        double sc = 0.0; Complex diff(0.0);
        for (std::size_t i = 0; i < dim; ++i) {
            diff += a[i] - b[i]; sc += std::norm(a[i]) + std::norm(b[i]);
        }
        return sc > 0.0 && std::abs(diff) < 1e-14 * std::sqrt(sc);
    }();

    // Energy bounds.
    double E_min = params.E_min, E_max = params.E_max;
    if (std::abs(E_min) < 1e-14 && std::abs(E_max) < 1e-14) {
        std::mt19937 g;
        if (params.random_seed != 0) g.seed(static_cast<uint32_t>(params.random_seed));
        else { std::random_device rd; g.seed(rd()); }
        auto [lo, hi] = estimate_energy_bounds(H, dim, params.bound_lanczos_dim, g);
        E_min = lo; E_max = hi;
    }

    const double BW     = E_max - E_min;
    const double buffer = params.spectral_bound_buffer * std::max(BW, 1.0);
    const double a      = ((E_max + buffer) - (E_min - buffer)) / 2.0;
    const double b_kpm  = ((E_max + buffer) + (E_min - buffer)) / 2.0;

    auto H_sc = [&H, a, b_kpm](const Complex* in, Complex* out, int n) {
        H(in, out, n);
        for (int i = 0; i < n; ++i) out[i] = (out[i] - b_kpm * in[i]) / a;
    };

    const int d = static_cast<int>(dim);
    const std::vector<double> omega_grid = linspace_tpq(omega_min, omega_max, n_om);

    // Compute O2|ψ⟩ and O1|ψ⟩.
    ComplexVector phi_R(dim), phi_L(dim);
    O2(psi.data(), phi_R.data(), d);
    if (is_self) {
        phi_L = phi_R;
    } else {
        O1(psi.data(), phi_L.data(), d);
    }

    // Time-evolve and accumulate G(t_j).
    std::vector<Complex> G_t(n_t);
    Complex G0; cblas_zdotc_sub(d, phi_L.data(), 1, phi_R.data(), 1, &G0);
    G_t[0] = window_value(params.window, 0, n_t) * G0;

    ComplexVector v_t(phi_R), v_evolved(dim);
    for (int j = 1; j < n_t; ++j) {
        cheby_time_evolve(H_sc, v_t, a * dt, params.time_moments, dim, v_evolved);
        std::swap(v_t, v_evolved);
        Complex G_inner; cblas_zdotc_sub(d, phi_L.data(), 1, v_t.data(), 1, &G_inner);
        const double phase = b_kpm * j * dt;
        const Complex pf(std::cos(phase), std::sin(phase));
        G_t[j] = window_value(params.window, j, n_t) * pf * G_inner;
    }

    // Fourier transform.
    std::vector<double> S_omega;
    fourier_transform_G(G_t, dt, omega_grid, S_omega);

    TPQDynamicalResult out;
    out.frequencies           = omega_grid;
    out.betas                 = {beta};
    out.partition_function    = {Z_estimate};
    out.ground_state_estimate = E_min;
    out.energy_shift_used     = E_min;
    out.total_samples         = 1;
    out.kpm_a                 = a;
    out.kpm_b                 = b_kpm;
    out.t_grid                = linspace_tpq(0.0, params.t_max, n_t);
    out.G_time                = G_t;

    const double dw = (n_om > 1)
        ? (omega_grid.back() - omega_grid.front()) / (n_om - 1) : 1.0;
    out.spectral_real.resize(n_om);
    out.spectral_imag.resize(n_om, 0.0);
    out.static_correlator.resize(1, 0.0);

    if (Z_estimate > 0.0) {
        double integral = 0.0;
        for (int i = 0; i < n_om; ++i) {
            out.spectral_real[i] = S_omega[i] / Z_estimate;
            integral += out.spectral_real[i];
        }
        out.static_correlator[0] = integral * dw;
    }
    return out;
}

} // namespace ed::tpq::dynamical
