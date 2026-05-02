// =============================================================================
// src/solvers/cpu/kpm_dos.cpp
//
// KPM density-of-states + thermodynamics kernel.
// See include/ed/solvers/kpm_dos.h for the full mathematical specification.
// =============================================================================

#include <ed/solvers/kpm_dos.h>
#include <ed/solvers/lanczos.h>   // build_lanczos_tridiagonal_with_basis,
                                   // diagonalize_tridiagonal_ritz,
                                   // generateGaussianRandomVector

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <random>
#include <stdexcept>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace ed::kpm_dos {

namespace {

inline bool kpm_dos_verbose() {
    static const bool v = []() {
        const char* e = std::getenv("ED_KPM_DOS_VERBOSE");
        return e && e[0] == '1';
    }();
    return v;
}

// ---------------------------------------------------------------------------
// Kernel coefficients (Jackson, Lorentz)
// ---------------------------------------------------------------------------

/// Jackson kernel: g_k = [(M+1−k) cos(πk/(M+1)) + sin(πk/(M+1)) cot(π/(M+1))] / (M+1)
/// Guarantees positive-definite reconstructed ρ(E).
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

/// Lorentz kernel: g_k = sinh(λ(1 − k/M)) / sinh(λ).
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
// Spectral-bound estimator: a single high-quality Lanczos sweep gives both
// extreme Ritz values to ~1e-10 relative accuracy with 100–200 iterations
// (Kaniel-Paige; the extreme eigenvalues converge first).
// ---------------------------------------------------------------------------
void estimate_spectral_bounds(
    MatVec H,
    std::uint64_t dim,
    int krylov_dim,
    bool full_reorth,
    int reorth_freq,
    double tol,
    std::mt19937& gen,
    double& e_min,
    double& e_max)
{
    ComplexVector v0 = generateGaussianRandomVector(static_cast<int>(dim), gen);

    std::vector<double> alpha, beta;
    const int M_lanc = build_lanczos_tridiagonal_with_basis(
        H, v0, dim,
        static_cast<std::uint64_t>(krylov_dim),
        tol, full_reorth,
        static_cast<std::uint64_t>(reorth_freq),
        alpha, beta, /*basis_vectors=*/nullptr);

    if (M_lanc == 0)
        throw std::runtime_error("kpm_dos: spectral-bound Lanczos produced 0 iterations");

    std::vector<double> ritz, weights;
    diagonalize_tridiagonal_ritz(alpha, beta, ritz, weights, /*evecs=*/nullptr);
    if (ritz.empty())
        throw std::runtime_error("kpm_dos: spectral-bound Ritz returned 0 eigenpairs");

    e_min = ritz.front();
    e_max = ritz.back();
}

// ---------------------------------------------------------------------------
// Chebyshev DOS-moment accumulator for a single random vector.
//
// Computes μ_k^{(r)} = ⟨r| T_k(H_sc) |r⟩ for k = 0..M-1 using the three-term
// recursion v_0 = |r⟩, v_1 = H_sc |r⟩, v_k = 2 H_sc v_{k-1} - v_{k-2}.
// μ_k is real because H_sc is Hermitian and the moment is a real bilinear form.
//
// Stores only 3 working vectors of length D.
// ---------------------------------------------------------------------------
void accumulate_dos_moments_one_vector(
    MatVec H,                      // physical H (we apply (H - b)/a per step)
    const ComplexVector& r_vec,
    std::uint64_t dim,
    double a, double b,
    int M,
    std::vector<double>& mu_acc)   // length M, accumulated in place
{
    const int d = static_cast<int>(dim);

    ComplexVector v_prev(dim, Complex(0.0, 0.0));
    ComplexVector v_curr(r_vec);                        // v_0 = |r⟩
    ComplexVector v_next(dim, Complex(0.0, 0.0));
    ComplexVector Hv(dim, Complex(0.0, 0.0));

    // μ_0 = ⟨r|r⟩ (should be 1 for a unit vector — but compute for safety).
    {
        Complex z;
        cblas_zdotc_sub(d, v_curr.data(), 1, v_curr.data(), 1, &z);
        mu_acc[0] += z.real();
    }

    auto apply_Hsc = [&](const ComplexVector& in, ComplexVector& out) {
        // out = (H |in⟩ - b |in⟩) / a
        H(in.data(), out.data(), d);
        const double inv_a = 1.0 / a;
        const Complex c_neg_boa(-b / a, 0.0);
        // out = (1/a) * out + (-b/a) * in
        const Complex c_inv_a(inv_a, 0.0);
        cblas_zscal(d, &c_inv_a, out.data(), 1);
        cblas_zaxpy(d, &c_neg_boa, in.data(), 1, out.data(), 1);
    };

    if (M > 1) {
        apply_Hsc(v_curr, v_next);  // v_1 = H_sc |v_0⟩
        Complex z;
        cblas_zdotc_sub(d, r_vec.data(), 1, v_next.data(), 1, &z);
        mu_acc[1] += z.real();
        std::swap(v_prev, v_curr);
        std::swap(v_curr, v_next);
    }

    for (int k = 2; k < M; ++k) {
        // v_next = 2 H_sc v_curr - v_prev
        apply_Hsc(v_curr, Hv);
        const Complex c2(2.0, 0.0), c_m1(-1.0, 0.0);
        cblas_zcopy(d, Hv.data(), 1, v_next.data(), 1);
        cblas_zscal(d, &c2, v_next.data(), 1);
        cblas_zaxpy(d, &c_m1, v_prev.data(), 1, v_next.data(), 1);

        Complex z;
        cblas_zdotc_sub(d, r_vec.data(), 1, v_next.data(), 1, &z);
        mu_acc[k] += z.real();

        std::swap(v_prev, v_curr);
        std::swap(v_curr, v_next);
    }
}

// ---------------------------------------------------------------------------
// Reconstruct DOS at given energies E_i:  ρ(E_i) using kernel-weighted moments.
//   ρ(E) = (1 / (π a √(1-x²))) [g_0 μ_0 + 2 Σ_{k≥1} g_k μ_k T_k(x)]
// ---------------------------------------------------------------------------
std::vector<double> reconstruct_dos(
    const std::vector<double>& mu_weighted,
    double a, double b,
    const std::vector<double>& energies)
{
    const int M = static_cast<int>(mu_weighted.size());
    std::vector<double> rho(energies.size(), 0.0);

    for (std::size_t i = 0; i < energies.size(); ++i) {
        const double x = (energies[i] - b) / a;
        if (x <= -1.0 + 1e-12 || x >= 1.0 - 1e-12) continue;

        double Tk_prev = 1.0;             // T_0
        double Tk_curr = x;               // T_1
        double sum = mu_weighted[0];
        if (M > 1) sum += 2.0 * mu_weighted[1] * Tk_curr;
        for (int k = 2; k < M; ++k) {
            const double Tk_next = 2.0 * x * Tk_curr - Tk_prev;
            sum += 2.0 * mu_weighted[k] * Tk_next;
            Tk_prev = Tk_curr;
            Tk_curr = Tk_next;
        }
        const double sqrt_factor = std::sqrt(1.0 - x * x);
        rho[i] = sum / (M_PI * a * sqrt_factor);
    }
    return rho;
}

// ---------------------------------------------------------------------------
// Chebyshev–Gauss quadrature of f(E) against ρ(E) using kernel-weighted
// moments {g_k μ_k}.  At nodes x_i = cos((i+0.5)π/N), the √(1-x²) jacobian
// of ρ cancels the quadrature weight π/N, giving an unbiased estimator that
// uses *no* divisions by √(1-x²) (so we never lose precision near the band
// edges).
// ---------------------------------------------------------------------------
struct ChebQuadCache {
    int N;
    std::vector<double> x;          // nodes x_i = cos((i+0.5)π/N), length N
    std::vector<double> energy;     // E_i = b + a x_i
    std::vector<double> bracket;    // [g_0 μ_0 + 2 Σ_{k≥1} g_k μ_k T_k(x_i)]
};

ChebQuadCache build_cheb_quad_cache(
    const std::vector<double>& mu_weighted,
    double a, double b,
    int N_quad)
{
    const int M = static_cast<int>(mu_weighted.size());
    ChebQuadCache cache;
    cache.N = N_quad;
    cache.x.resize(N_quad);
    cache.energy.resize(N_quad);
    cache.bracket.assign(N_quad, 0.0);

    for (int i = 0; i < N_quad; ++i) {
        const double xi = std::cos(M_PI * (i + 0.5) / N_quad);
        cache.x[i] = xi;
        cache.energy[i] = b + a * xi;

        double Tk_prev = 1.0;
        double Tk_curr = xi;
        double sum = mu_weighted[0];
        if (M > 1) sum += 2.0 * mu_weighted[1] * Tk_curr;
        for (int k = 2; k < M; ++k) {
            const double Tk_next = 2.0 * xi * Tk_curr - Tk_prev;
            sum += 2.0 * mu_weighted[k] * Tk_next;
            Tk_prev = Tk_curr;
            Tk_curr = Tk_next;
        }
        cache.bracket[i] = sum;
    }
    return cache;
}

}  // anonymous namespace

// ---------------------------------------------------------------------------
// Public driver
// ---------------------------------------------------------------------------
KPMDOSResult compute_kpm_dos(
    MatVec H,
    std::uint64_t dim,
    const std::vector<double>& betas,
    const std::vector<double>& dos_energies,
    const KPMDOSParameters& params)
{
    if (dim == 0)
        throw std::invalid_argument("kpm_dos: dim must be > 0");
    if (params.num_moments < 4)
        throw std::invalid_argument("kpm_dos: num_moments must be >= 4");
    if (params.num_random_vectors < 1)
        throw std::invalid_argument("kpm_dos: num_random_vectors must be >= 1");

    const std::uint64_t seed = (params.random_seed != 0)
        ? params.random_seed
        : std::random_device{}();
    std::mt19937 gen(seed);

    // -----------------------------------------------------------------
    // Step 1: spectral-bound Lanczos to get (a, b).
    // -----------------------------------------------------------------
    double e_min = 0.0, e_max = 0.0;
    estimate_spectral_bounds(
        H, dim, params.spectral_bounds_krylov,
        params.full_reorthogonalization, params.reorth_frequency,
        params.tolerance, gen, e_min, e_max);

    if (e_max <= e_min) {
        // Degenerate spectrum (one point or numerical failure): nudge bounds.
        const double eps = 1.0;
        e_max = e_min + eps;
    }

    const double BW     = e_max - e_min;
    const double buffer = std::max(params.spectral_bound_buffer, 1e-6) * BW;
    const double kpm_lo = e_min - buffer;
    const double kpm_hi = e_max + buffer;
    const double a      = (kpm_hi - kpm_lo) / 2.0;
    const double b      = (kpm_hi + kpm_lo) / 2.0;
    const double shift  = e_min;  // Boltzmann shift for numerical stability

    if (kpm_dos_verbose()) {
        std::fprintf(stderr,
            "[kpm_dos] dim=%llu  E_min=%.6e  E_max=%.6e  a=%.6e  b=%.6e\n",
            static_cast<unsigned long long>(dim), e_min, e_max, a, b);
    }

    // -----------------------------------------------------------------
    // Step 2: Chebyshev moments via Hutchinson trace.
    // -----------------------------------------------------------------
    const int M = params.num_moments;
    const int R = params.num_random_vectors;

    std::vector<double> mu_avg(M, 0.0);   // running sum of ⟨r|T_k|r⟩

    for (int r = 0; r < R; ++r) {
        // Random complex Gaussian vector, normalised to unit norm.
        ComplexVector r_vec = generateGaussianRandomVector(static_cast<int>(dim), gen);
        const double norm = cblas_dznrm2(static_cast<int>(dim), r_vec.data(), 1);
        if (norm <= 0.0) {
            throw std::runtime_error("kpm_dos: random vector has zero norm");
        }
        const Complex inv_norm(1.0 / norm, 0.0);
        cblas_zscal(static_cast<int>(dim), &inv_norm, r_vec.data(), 1);

        accumulate_dos_moments_one_vector(H, r_vec, dim, a, b, M, mu_avg);

        if (kpm_dos_verbose() && (r % 5 == 0 || r == R - 1)) {
            std::fprintf(stderr, "[kpm_dos] sample %d/%d  μ_0/r = %.6e\n",
                         r + 1, R, mu_avg[0] / (r + 1));
        }
    }

    // Hutchinson normalisation: μ_k = (D / R) * Σ_r ⟨r|T_k|r⟩.
    // For a unit random vector, Σ_r ⟨r|T_0|r⟩ = R, so μ_0 reduces to D exactly.
    std::vector<double> mu_raw(M);
    const double Dscale = static_cast<double>(dim) / R;
    for (int k = 0; k < M; ++k) mu_raw[k] = Dscale * mu_avg[k];

    // -----------------------------------------------------------------
    // Step 3: kernel-weighted moments.
    // -----------------------------------------------------------------
    const std::vector<double> kernel = params.use_jackson_kernel
        ? make_jackson_kernel(M)
        : make_lorentz_kernel(M, params.lorentz_lambda);

    std::vector<double> mu_w(M);
    for (int k = 0; k < M; ++k) mu_w[k] = kernel[k] * mu_raw[k];

    // -----------------------------------------------------------------
    // Step 4: Chebyshev–Gauss quadrature for thermodynamics.
    // -----------------------------------------------------------------
    const int N_quad = (params.num_quadrature_nodes > 0)
        ? params.num_quadrature_nodes
        : 2 * M;

    ChebQuadCache cache = build_cheb_quad_cache(mu_w, a, b, N_quad);

    KPMDOSResult result;
    result.betas = betas;
    result.partition_function.assign(betas.size(), 0.0);
    result.energy.assign(betas.size(), 0.0);
    result.specific_heat.assign(betas.size(), 0.0);
    result.entropy.assign(betas.size(), 0.0);
    result.free_energy.assign(betas.size(), 0.0);

    // Quadrature: ∫ ρ(E) f(E) dE = (1/N_quad) Σ_i bracket_i * f(E_i)
    // We'll compute Z, ⟨E⟩, ⟨E²⟩ in one sweep per β (using Boltzmann shift).
    for (std::size_t t = 0; t < betas.size(); ++t) {
        const double beta = betas[t];
        // For numerical safety we factor out e^{-β shift}; it cancels in
        // ratios but enters logarithmically into F.
        double Z_shift = 0.0;
        double E_shift = 0.0;
        double E2_shift = 0.0;
        for (int i = 0; i < cache.N; ++i) {
            const double E_phys = cache.energy[i];
            const double w = std::exp(-beta * (E_phys - shift));  // safe
            const double br = cache.bracket[i];
            Z_shift  += br * w;
            E_shift  += br * w * E_phys;
            E2_shift += br * w * E_phys * E_phys;
        }
        const double inv_N = 1.0 / cache.N;
        Z_shift  *= inv_N;
        E_shift  *= inv_N;
        E2_shift *= inv_N;

        // Guard against pathological negative DOS (Lorentz kernel can dip
        // very slightly below zero near band edges).
        const double Z_safe = std::max(Z_shift, 1e-300);

        const double E_mean   = E_shift / Z_safe;
        const double E2_mean  = E2_shift / Z_safe;
        const double C_val    = (E2_mean - E_mean * E_mean) * beta * beta;

        // Z_phys = Z_shift * e^{-β shift};  log Z_phys = log Z_shift - β shift.
        const double log_Z = std::log(Z_safe) - beta * shift;
        const double F_val = -log_Z / beta;
        const double S_val = (E_mean - F_val) * beta;

        result.partition_function[t] = Z_safe * std::exp(-beta * shift);
        result.energy[t]             = E_mean;
        result.specific_heat[t]      = C_val;
        result.entropy[t]            = S_val;
        result.free_energy[t]        = F_val;
    }

    // -----------------------------------------------------------------
    // Step 5: optional reconstructed DOS on caller-provided E grid.
    // -----------------------------------------------------------------
    if (!dos_energies.empty()) {
        result.dos_grid_energies = dos_energies;
        result.dos_grid_values   = reconstruct_dos(mu_w, a, b, dos_energies);
    }

    result.moments_weighted        = std::move(mu_w);
    result.moments_raw             = std::move(mu_raw);
    result.kpm_a                   = a;
    result.kpm_b                   = b;
    result.e_min_estimate          = e_min;
    result.e_max_estimate          = e_max;
    result.energy_shift_used       = shift;
    result.hilbert_dim             = dim;
    result.num_moments_used        = M;
    result.num_random_vectors_used = R;
    result.jackson_kernel_used     = params.use_jackson_kernel;
    return result;
}

}  // namespace ed::kpm_dos
