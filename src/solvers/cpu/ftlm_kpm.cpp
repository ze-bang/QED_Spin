// =============================================================================
// src/solvers/cpu/ftlm_kpm.cpp
//
// KPM (Chebyshev moment) kernel for finite-T dynamical correlators.
// See include/ed/solvers/ftlm_kpm.h for the full mathematical specification.
// =============================================================================

#include <ed/solvers/ftlm_kpm.h>
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

namespace ed::kpm {

namespace {

// ---------------------------------------------------------------------------
// Verbose logging
// ---------------------------------------------------------------------------
inline bool kpm_verbose() {
    static const bool v = []() {
        const char* e = std::getenv("ED_KPM_VERBOSE");
        return e && e[0] == '1';
    }();
    return v;
}

// ---------------------------------------------------------------------------
// Kernel coefficients
// ---------------------------------------------------------------------------

/// Jackson kernel: gₖ = [(M+1−k) cos(πk/(M+1)) + sin(πk/(M+1)) cot(π/(M+1))] / (M+1)
/// Guarantees positive-definite spectral function.
std::vector<double> make_jackson_kernel(int M) {
    std::vector<double> g(M);
    const double Mp1 = static_cast<double>(M + 1);
    const double cot_term = 1.0 / std::tan(M_PI / Mp1);
    for (int k = 0; k < M; ++k) {
        const double kd = static_cast<double>(k);
        const double phi = M_PI * kd / Mp1;
        g[k] = ((Mp1 - kd) * std::cos(phi) + std::sin(phi) * cot_term) / Mp1;
    }
    return g;
}

/// Lorentz kernel: gₖ = sinh(λ(1 − k/M)) / sinh(λ)
/// Gives Lorentzian broadening η = a λ / M.
std::vector<double> make_lorentz_kernel(int M, double lambda) {
    std::vector<double> g(M);
    const double sh_lambda = std::sinh(lambda);
    for (int k = 0; k < M; ++k) {
        const double x = lambda * (1.0 - static_cast<double>(k) / M);
        g[k] = std::sinh(x) / sh_lambda;
    }
    return g;
}

// ---------------------------------------------------------------------------
// Reconstruct the j-th Ritz vector from the Lanczos basis (same as in
// ftlm_ltlm_dyn.cpp and ftlm_sssf.cpp — local copy to keep files independent).
// ---------------------------------------------------------------------------
void reconstruct_ritz(const std::vector<ComplexVector>& V,
                      const std::vector<double>& evecs,
                      std::size_t M, std::size_t j,
                      ComplexVector& out)
{
    const int N = static_cast<int>(V.front().size());
    std::fill(out.begin(), out.end(), Complex(0.0, 0.0));
    for (std::size_t k = 0; k < M; ++k) {
        const double u = evecs[k + j * M];
        if (u == 0.0) continue;
        const Complex c(u, 0.0);
        cblas_zaxpy(N, &c, V[k].data(), 1, out.data(), 1);
    }
}

// ---------------------------------------------------------------------------
// Self-correlation detection via probe vector heuristic.
// ---------------------------------------------------------------------------
bool detect_self_corr(MatVec O1, MatVec O2, std::uint64_t dim) {
    if (dim == 0) return false;
    ComplexVector probe(dim, Complex(0.0, 0.0));
    probe[0] = Complex(1.0, 0.0);
    ComplexVector a(dim), b(dim);
    O1(probe.data(), a.data(), static_cast<int>(dim));
    O2(probe.data(), b.data(), static_cast<int>(dim));
    double scale = 0.0;
    Complex diff(0.0, 0.0);
    for (std::size_t i = 0; i < dim; ++i) {
        diff  += a[i] - b[i];
        scale += std::norm(a[i]) + std::norm(b[i]);
    }
    return scale > 0.0 && std::abs(diff) < 1e-14 * std::sqrt(scale);
}

// ---------------------------------------------------------------------------
// Core inner KPM loop:
//   Given outer state |n⟩ with energy E_n, compute Chebyshev moments
//   μₖ = ⟨g_n|Tₖ(H_sc)|f_n⟩  (k = 0..M-1)
//   and accumulate into S_acc[t][i] for all temperatures and frequencies.
//
// Parameters
// ----------
//   H_sc      : H rescaled to [-1,1]: H_sc(v) = (H(v) - b*v) / a
//   left      : ⟨g_n| = ⟨O1 n|  (may equal right for self-correlation)
//   right     : |f_n⟩ = |O2 n⟩
//   E_n       : energy of outer state |n⟩
//   energy_shift : Boltzmann shift
//   a, b      : KPM rescaling params: H_sc = (H-b)/a
//   betas     : inverse temperatures
//   omega_grid: output frequency grid (transfer energies ω)
//   kernel    : g_0..g_{M-1}
//   S_acc     : [n_T][n_omega] accumulator (modified in place)
//   Z_acc     : [n_T] partition function accumulator (modified in place)
// ---------------------------------------------------------------------------
void accumulate_kpm_inner(
    MatVec H_sc,
    const ComplexVector& left,
    const ComplexVector& right,
    double E_n,
    double energy_shift,
    std::uint64_t dim,
    double a,
    double b,
    const std::vector<double>& betas,
    const std::vector<double>& omega_grid,
    const std::vector<double>& kernel,
    std::vector<std::vector<double>>& S_acc,
    std::vector<double>& Z_acc)
{
    const int M     = static_cast<int>(kernel.size());
    const int d     = static_cast<int>(dim);
    const int n_omega = static_cast<int>(omega_grid.size());
    const int n_T   = static_cast<int>(betas.size());

    // Boltzmann factors (computed once per outer state).
    std::vector<double> boltz(n_T);
    for (int t = 0; t < n_T; ++t) {
        boltz[t] = std::exp(-betas[t] * (E_n - energy_shift));
        Z_acc[t] += boltz[t];
    }

    // Compute Chebyshev moments via three-term recursion.
    // Stores only v_prev, v_curr (two working vectors + H application buffer).
    std::vector<Complex> mu(M, Complex(0.0, 0.0));

    // Allocate working vectors (only 3 × dim complex doubles ≈ 24 × dim bytes).
    ComplexVector v_prev(dim, Complex(0.0, 0.0));
    ComplexVector v_curr(right);                   // v_0 = |f_n⟩
    ComplexVector v_next(dim, Complex(0.0, 0.0));
    ComplexVector Hv(dim, Complex(0.0, 0.0));

    // μ₀ = ⟨g_n|v_0⟩
    cblas_zdotc_sub(d, left.data(), 1, v_curr.data(), 1, &mu[0]);

    if (M > 1) {
        // v_1 = H_sc |v_0⟩
        H_sc(v_curr.data(), v_next.data(), d);
        cblas_zdotc_sub(d, left.data(), 1, v_next.data(), 1, &mu[1]);
        std::swap(v_prev, v_curr);
        std::swap(v_curr, v_next);
    }

    for (int k = 2; k < M; ++k) {
        // v_k = 2 H_sc v_{k-1} - v_{k-2}
        H_sc(v_curr.data(), Hv.data(), d);
        const Complex c2(2.0, 0.0), c_m1(-1.0, 0.0);
        // v_next = 2*Hv - v_prev  (reuse v_next buffer)
        cblas_zcopy(d, Hv.data(), 1, v_next.data(), 1);
        cblas_zscal(d, &c2, v_next.data(), 1);
        cblas_zaxpy(d, &c_m1, v_prev.data(), 1, v_next.data(), 1);

        cblas_zdotc_sub(d, left.data(), 1, v_next.data(), 1, &mu[k]);

        std::swap(v_prev, v_curr);
        std::swap(v_curr, v_next);
    }

    // Evaluate spectral function on the output grid.
    // Pre-compute T_k(x) for all k using Chebyshev recursion at each ω.
    for (int i = 0; i < n_omega; ++i) {
        // Absolute energy of the excited state.
        const double eps = omega_grid[i] + E_n;
        const double x   = (eps - b) / a;   // rescaled to [-1, 1]

        // Skip points outside the KPM domain.
        // Use a small margin to avoid √(1-x²) divergence.
        if (x <= -1.0 + 1e-10 || x >= 1.0 - 1e-10) continue;

        // Chebyshev series: g₀ Re(μ₀) T₀ + 2 Σ_{k≥1} gₖ Re(μₖ) Tₖ(x)
        // We evaluate only the real part because the spectral function is real.
        double Tk_prev = 1.0;  // T_0
        double Tk_curr = x;    // T_1
        double sum = kernel[0] * mu[0].real();  // k=0 term
        if (M > 1) {
            sum += 2.0 * kernel[1] * mu[1].real() * Tk_curr;
        }
        for (int k = 2; k < M; ++k) {
            const double Tk_next = 2.0 * x * Tk_curr - Tk_prev;
            sum += 2.0 * kernel[k] * mu[k].real() * Tk_next;
            Tk_prev = Tk_curr;
            Tk_curr = Tk_next;
        }

        // KPM spectral density at this frequency for this outer state.
        const double denom = M_PI * a * std::sqrt(1.0 - x * x);
        const double Sn_i  = sum / denom;

        // Accumulate with Boltzmann weights (all temperatures at once).
        for (int t = 0; t < n_T; ++t)
            S_acc[t][i] += boltz[t] * Sn_i;
    }
}

// ---------------------------------------------------------------------------
// Finalise KPMResult from accumulators.
// ---------------------------------------------------------------------------
void finalise_kpm_result(
    const std::vector<double>& omega_grid,
    const std::vector<double>& betas,
    const std::vector<std::vector<double>>& S_acc,
    const std::vector<double>& Z_acc,
    double gs_estimate,
    double energy_shift,
    double a, double b,
    int M,
    bool jackson,
    std::uint64_t K,
    KPMResult& out)
{
    const int n_T     = static_cast<int>(betas.size());
    const int n_omega = static_cast<int>(omega_grid.size());
    const double dw   = (n_omega > 1)
        ? (omega_grid.back() - omega_grid.front()) / (n_omega - 1)
        : 1.0;

    out.frequencies        = omega_grid;
    out.betas              = betas;
    out.ground_state_estimate = gs_estimate;
    out.energy_shift_used  = energy_shift;
    out.kpm_a              = a;
    out.kpm_b              = b;
    out.num_moments_used   = M;
    out.jackson_kernel_used = jackson;
    out.total_outer_states  = K;
    out.partition_function  = Z_acc;

    out.spectral_real.resize(n_T * n_omega);
    out.spectral_imag.resize(n_T * n_omega, 0.0);
    out.static_correlator.resize(n_T, 0.0);

    for (int t = 0; t < n_T; ++t) {
        const double Z = Z_acc[t];
        if (Z <= 0.0) continue;
        double integral = 0.0;
        for (int i = 0; i < n_omega; ++i) {
            const double v = S_acc[t][i] / Z;
            out.spectral_real[t * n_omega + i] = v;
            integral += v;
        }
        out.static_correlator[t] = integral * dw;
    }
}

// ---------------------------------------------------------------------------
// LTLM-KPM core loop (used by both compute_kpm_ltlm and from_states path).
// Processes K outer states, each with energy E_n and state vector ritz_n.
//
// On input:
//   ritz_states  : (may be empty) pre-computed states; if empty, outer
//                  Lanczos Ritz vectors are used.
//   energies     : (may be empty) pre-computed energies; if empty, E_out used.
//   V_outer, U_outer, E_out: outer Lanczos Ritz data (used if ritz_states empty).
// ---------------------------------------------------------------------------
void run_kpm_outer_loop(
    MatVec H_inner,
    MatVec O1, MatVec O2,
    std::uint64_t dim_outer, std::uint64_t dim_inner,
    double energy_shift,
    double a, double b,
    const std::vector<double>& betas,
    const std::vector<double>& omega_grid,
    const std::vector<double>& kernel,
    // Outer state source (exactly one of the two groups should be non-empty):
    // Group A: LTLM Lanczos output
    const std::vector<ComplexVector>* V_outer,
    const std::vector<double>* U_outer,
    const std::vector<double>* E_out,
    std::size_t K,
    // Group B: pre-computed eigenstates
    const std::vector<ComplexVector>* eigenstates,
    const std::vector<double>* energies,
    // Output:
    KPMResult& out)
{
    const bool use_eigenstates = (eigenstates != nullptr && !eigenstates->empty());
    const int n_T     = static_cast<int>(betas.size());
    const int n_omega = static_cast<int>(omega_grid.size());
    const bool is_self = detect_self_corr(O1, O2, dim_inner);

    // Accumulators [n_T][n_omega] and [n_T].
    std::vector<std::vector<double>> S_acc(n_T,
        std::vector<double>(n_omega, 0.0));
    std::vector<double> Z_acc(n_T, 0.0);

    ComplexVector ritz_n(dim_outer);
    ComplexVector left_v(dim_inner), right_v(dim_inner);

    double gs_estimate = std::numeric_limits<double>::infinity();

    // Build H_sc MatVec: H_sc(v) = (H_inner(v) - b*v) / a
    auto H_sc = [&H_inner, a, b, dim_inner]
        (const Complex* in, Complex* out_v, int n) {
        H_inner(in, out_v, n);
        const int d = static_cast<int>(dim_inner);
        for (int i = 0; i < d; ++i)
            out_v[i] = (out_v[i] - b * in[i]) / a;
    };

    for (std::size_t n = 0; n < K; ++n) {
        double E_n = 0.0;
        const ComplexVector* state_ptr = nullptr;

        if (use_eigenstates) {
            E_n       = (*energies)[n];
            state_ptr = &(*eigenstates)[n];
            // ritz_n points to eigenstate directly via const ref.
            ritz_n.assign((*eigenstates)[n].begin(), (*eigenstates)[n].end());
        } else {
            E_n = (*E_out)[n];
            // Reconstruct Ritz vector from Lanczos basis.
            reconstruct_ritz(*V_outer, *U_outer, K, n, ritz_n);
        }

        gs_estimate = std::min(gs_estimate, E_n);

        // Compute O2|n> and O1|n>.
        O2(ritz_n.data(), right_v.data(), static_cast<int>(dim_inner));
        if (is_self) {
            left_v = right_v;
        } else {
            O1(ritz_n.data(), left_v.data(), static_cast<int>(dim_inner));
        }

        if (kpm_verbose()) {
            double nr = 0.0;
            for (const auto& c : right_v) nr += std::norm(c);
            std::cerr << "  [KPM] state " << n << " E=" << E_n
                      << " ||O|n>||=" << std::sqrt(nr) << "\n";
        }

        accumulate_kpm_inner(H_sc, left_v, right_v, E_n, energy_shift,
                             dim_inner, a, b, betas, omega_grid, kernel,
                             S_acc, Z_acc);
    }

    finalise_kpm_result(omega_grid, betas, S_acc, Z_acc,
                        std::isfinite(gs_estimate) ? gs_estimate : 0.0,
                        energy_shift, a, b,
                        static_cast<int>(kernel.size()),
                        true, K, out);
}

// ---------------------------------------------------------------------------
// Frequency grid helper.
// ---------------------------------------------------------------------------
std::vector<double> make_linspace(double lo, double hi, int N) {
    std::vector<double> v(N);
    for (int i = 0; i < N; ++i)
        v[i] = lo + (hi - lo) * i / (N - 1);
    return v;
}

} // anonymous namespace

// =============================================================================
// Public API
// =============================================================================

KPMResult compute_kpm_ltlm(
    MatVec H,
    MatVec O1,
    MatVec O2,
    std::uint64_t dim,
    double omega_min,
    double omega_max,
    int n_omega,
    const std::vector<double>& betas,
    const KPMParameters& params)
{
    if (dim == 0)    throw std::invalid_argument("kpm: dim must be > 0");
    if (n_omega < 2) throw std::invalid_argument("kpm: n_omega must be >= 2");
    if (params.num_moments < 1)
        throw std::invalid_argument("kpm: num_moments must be >= 1");

    // ------------------------------------------------------------------
    // Step 1: outer Lanczos to get K lowest Ritz states and energy window.
    // ------------------------------------------------------------------
    std::mt19937 gen;
    if (params.random_seed == 0) {
        std::random_device rd; gen.seed(rd());
    } else {
        gen.seed(static_cast<std::uint32_t>(params.random_seed));
    }
    ComplexVector v0 = generateGaussianRandomVector(static_cast<int>(dim), gen);

    std::vector<double> alpha_o, beta_o;
    std::vector<ComplexVector> V_outer;
    const int M_lanc = build_lanczos_tridiagonal_with_basis(
        H, v0, dim, params.outer_krylov_dim,
        params.tolerance, params.full_reorthogonalization,
        params.reorth_frequency, alpha_o, beta_o, &V_outer);

    if (M_lanc == 0)
        throw std::runtime_error("kpm: outer Lanczos produced 0 iterations");

    std::vector<double> E_out, w_dummy, U_outer;
    diagonalize_tridiagonal_ritz(alpha_o, beta_o, E_out, w_dummy, &U_outer);
    if (E_out.empty())
        throw std::runtime_error("kpm: outer Ritz returned 0 eigenpairs");

    const std::size_t K = std::min(
        static_cast<std::size_t>(params.num_lowest_states), E_out.size());

    // ------------------------------------------------------------------
    // Step 2: determine KPM energy window.
    // ------------------------------------------------------------------
    const double E_min  = E_out.front();
    const double E_max  = E_out.back();
    const double BW     = E_max - E_min;
    const double buffer = params.spectral_bound_buffer * BW;
    const double kpm_lo = E_min - buffer;
    const double kpm_hi = E_max + buffer;
    const double a      = (kpm_hi - kpm_lo) / 2.0;
    const double b      = (kpm_hi + kpm_lo) / 2.0;

    const double energy_shift = (std::abs(params.energy_shift) > 0.0)
        ? params.energy_shift : E_min;

    // ------------------------------------------------------------------
    // Step 3: build kernel and frequency grid.
    // ------------------------------------------------------------------
    const int M = params.num_moments;
    const std::vector<double> kernel = params.use_jackson_kernel
        ? make_jackson_kernel(M) : make_lorentz_kernel(M, params.lorentz_lambda);

    const std::vector<double> omega_grid = make_linspace(omega_min, omega_max, n_omega);

    KPMResult out;
    run_kpm_outer_loop(H, O1, O2, dim, dim,
                       energy_shift, a, b, betas, omega_grid, kernel,
                       &V_outer, &U_outer, &E_out, K,
                       nullptr, nullptr,
                       out);
    out.jackson_kernel_used = params.use_jackson_kernel;
    return out;
}

KPMResult compute_kpm_ltlm_from_states(
    MatVec H_inner,
    MatVec O1,
    MatVec O2,
    std::uint64_t dim_outer,
    std::uint64_t dim_inner,
    const std::vector<ComplexVector>& eigenstates,
    const std::vector<double>& energies,
    double omega_min,
    double omega_max,
    int n_omega,
    const std::vector<double>& betas,
    const KPMParameters& params)
{
    if (eigenstates.empty())
        throw std::invalid_argument("kpm from_states: eigenstates must be non-empty");
    if (eigenstates.size() != energies.size())
        throw std::invalid_argument("kpm from_states: eigenstates/energies size mismatch");
    if (n_omega < 2)
        throw std::invalid_argument("kpm: n_omega must be >= 2");

    const std::size_t K = eigenstates.size();
    const double E_min  = *std::min_element(energies.begin(), energies.end());
    const double E_max  = *std::max_element(energies.begin(), energies.end());
    const double BW     = std::max(E_max - E_min, 1.0);  // guard against 0 BW
    const double buffer = params.spectral_bound_buffer * BW;
    const double kpm_lo = E_min - buffer;
    const double kpm_hi = E_max + buffer;
    const double a      = (kpm_hi - kpm_lo) / 2.0;
    const double b      = (kpm_hi + kpm_lo) / 2.0;

    const double energy_shift = (std::abs(params.energy_shift) > 0.0)
        ? params.energy_shift : E_min;

    const int M = params.num_moments;
    const std::vector<double> kernel = params.use_jackson_kernel
        ? make_jackson_kernel(M) : make_lorentz_kernel(M, params.lorentz_lambda);

    const std::vector<double> omega_grid = make_linspace(omega_min, omega_max, n_omega);

    KPMResult out;
    run_kpm_outer_loop(H_inner, O1, O2, dim_outer, dim_inner,
                       energy_shift, a, b, betas, omega_grid, kernel,
                       nullptr, nullptr, nullptr, K,
                       &eigenstates, &energies,
                       out);
    out.jackson_kernel_used = params.use_jackson_kernel;
    return out;
}

KPMResult combine_sector_kpm(
    const std::vector<KPMResult>& per_sector,
    const std::vector<std::uint64_t>& sector_dims)
{
    if (per_sector.empty())
        throw std::invalid_argument("kpm: per_sector must be non-empty");
    if (per_sector.size() != sector_dims.size())
        throw std::invalid_argument("kpm: per_sector / sector_dims size mismatch");

    const int n_T     = static_cast<int>(per_sector.front().betas.size());
    const int n_omega = static_cast<int>(per_sector.front().frequencies.size());

    KPMResult out = per_sector.front();  // copy layout and metadata
    std::fill(out.spectral_real.begin(), out.spectral_real.end(), 0.0);
    std::fill(out.spectral_imag.begin(), out.spectral_imag.end(), 0.0);
    std::fill(out.partition_function.begin(), out.partition_function.end(), 0.0);
    std::fill(out.static_correlator.begin(), out.static_correlator.end(), 0.0);
    out.ground_state_estimate = std::numeric_limits<double>::infinity();

    // Compute combined Z = Σ_s d_s * Z_s (effective partition function).
    for (std::size_t s = 0; s < per_sector.size(); ++s) {
        const double d = static_cast<double>(sector_dims[s]);
        out.ground_state_estimate = std::min(
            out.ground_state_estimate, per_sector[s].ground_state_estimate);
        for (int t = 0; t < n_T; ++t)
            out.partition_function[t] += d * per_sector[s].partition_function[t];
    }

    // Accumulate spectral weight with d_s * Z_s / Z_total weighting.
    const double dw = (n_omega > 1)
        ? (out.frequencies.back() - out.frequencies.front()) / (n_omega - 1)
        : 1.0;
    for (int t = 0; t < n_T; ++t) {
        const double Z_total = out.partition_function[t];
        if (Z_total <= 0.0) continue;
        double integral = 0.0;
        for (int i = 0; i < n_omega; ++i) {
            double v = 0.0;
            for (std::size_t s = 0; s < per_sector.size(); ++s) {
                const double d = static_cast<double>(sector_dims[s]);
                const double w = d * per_sector[s].partition_function[t] / Z_total;
                v += w * per_sector[s].spectral_real[t * n_omega + i];
            }
            out.spectral_real[t * n_omega + i] = v;
            integral += v;
        }
        out.static_correlator[t] = integral * dw;
    }
    return out;
}

} // namespace ed::kpm
