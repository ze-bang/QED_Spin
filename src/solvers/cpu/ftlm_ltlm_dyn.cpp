// =============================================================================
// src/solvers/cpu/ftlm_ltlm_dyn.cpp
//
// Implementation of the Low-Temperature Lanczos Method (LTLM) dynamical
// two-point correlator.  See include/ed/solvers/ftlm_ltlm_dyn.h for the
// mathematical specification.
//
// Design notes
// ------------
// The inner-Lanczos body is identical to that of the JP kernel
// (ftlm_jp.cpp).  The only difference is in the outer loop:
//
//   JP:   random initial vector -> Ritz pairs with weight w = U[0,n]^2
//   LTLM: single outer Lanczos -> reconstruct K Ritz vectors; weight = 1
//
// Because reconstruct_ritz_vector and project_onto_basis are in the
// anonymous namespace of ftlm_jp.cpp we cannot link to them; we replicate
// the trivial helpers here (each 10 lines).
// =============================================================================

#include <ed/solvers/ftlm_ltlm_dyn.h>
#include <ed/solvers/lanczos.h>   // build_lanczos_tridiagonal_with_basis,
                                   // diagonalize_tridiagonal_ritz,
                                   // generateGaussianRandomVector

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace ed::ltlm {

namespace {

inline bool ltlm_verbose() {
    static const bool v = []() {
        const char* e = std::getenv("ED_LTLM_VERBOSE");
        return e && e[0] == '1';
    }();
    return v;
}

/// Reconstruct the j-th Ritz vector:
///   |n_j> = sum_k U[k, j] * V_basis[k]
/// evecs is column-major M x M: evecs[k + j*M] = U[k,j].
void reconstruct_ritz_vector(const std::vector<ComplexVector>& V_basis,
                             const std::vector<double>& evecs,
                             std::size_t M,
                             std::size_t j,
                             ComplexVector& out)
{
    const std::size_t N = V_basis.front().size();
    std::fill(out.begin(), out.end(), Complex(0.0, 0.0));
    for (std::size_t k = 0; k < M; ++k) {
        const double u = evecs[k + j * M];
        if (u == 0.0) continue;
        const Complex c(u, 0.0);
        cblas_zaxpy(static_cast<int>(N), &c,
                    V_basis[k].data(), 1, out.data(), 1);
    }
}

/// Project a vector onto an inner Krylov basis:
///   proj[l] = <V_inner[l] | x>   (conjugate-linear in first arg)
void project_onto_basis(const std::vector<ComplexVector>& V_inner,
                        const ComplexVector& x,
                        std::vector<Complex>& proj)
{
    const std::size_t M = V_inner.size();
    const int N = static_cast<int>(x.size());
    proj.assign(M, Complex(0.0, 0.0));
    for (std::size_t l = 0; l < M; ++l) {
        Complex z;
        cblas_zdotc_sub(N, V_inner[l].data(), 1, x.data(), 1, &z);
        proj[l] = z;
    }
}

/// Shared inner-Lanczos accumulation loop (called for each outer eigenstate n).
/// @param H_inner  Hamiltonian in the inner sector (= outer when intra-sector)
/// @param O1,O2    Operators (O2 seeds inner Krylov; O1 provides M1 overlaps)
/// @param ritz_n   The n-th outer Ritz vector (pre-reconstructed)
/// @param E_n      The n-th outer Ritz energy
/// @param dim_inner Inner sector dimension
/// @param betas    Inverse temperatures
/// @param omega_grid Frequency grid
/// @param eta      Lorentzian broadening
/// @param energy_shift Shift applied in Boltzmann weights
/// @param is_self  True iff O1 = O2 (fast path: skip O1 application)
/// @param p        LTLM parameters (inner_krylov_dim, tolerance, ...)
/// Z accumulation is NOT done here — callers must add exp(-beta*(E_n-shift))
/// to Z *before* calling this function, so that states where ||O|n>||=0 (e.g.
/// Sz=0 states for a transverse operator) still contribute to the partition
/// function.
///
/// @param stat     Accumulator for static correlator (size n_T)
/// @param S_real   Accumulator for Re[S(w,T)] (size n_T * n_omega)
/// @param S_imag   Accumulator for Im[S(w,T)] (size n_T * n_omega)
void accumulate_inner(
    MatVec H_inner,
    MatVec O1,
    MatVec O2,
    const ComplexVector& ritz_n,
    double E_n,
    std::uint64_t dim_inner,
    const std::vector<double>& betas,
    const std::vector<double>& omega_grid,
    double eta,
    double energy_shift,
    bool is_self,
    const LTLMParameters& p,
    std::vector<Complex>& stat,
    std::vector<double>& S_real,
    std::vector<double>& S_imag)
{
    const std::size_t n_T     = betas.size();
    const std::size_t n_omega = omega_grid.size();
    const double inv_pi = 1.0 / M_PI;
    const double eta2   = eta * eta;

    // phi = O2 |n>; phi /= ||phi||
    ComplexVector phi(dim_inner);
    O2(ritz_n.data(), phi.data(), static_cast<int>(dim_inner));
    const double norm_phi = cblas_dznrm2(static_cast<int>(dim_inner),
                                         phi.data(), 1);
    if (norm_phi < p.tolerance) return;  // ||O|n>||=0; Z already updated by caller.
    {
        Complex inv(1.0 / norm_phi, 0.0);
        cblas_zscal(static_cast<int>(dim_inner), &inv, phi.data(), 1);
    }

    // Inner Lanczos from phi.
    std::vector<double> alpha_i, beta_i;
    std::vector<ComplexVector> V_inner;
    const int inner_iters = build_lanczos_tridiagonal_with_basis(
        H_inner, phi, dim_inner, p.inner_krylov_dim,
        p.tolerance, p.full_reorthogonalization, p.reorth_frequency,
        alpha_i, beta_i, &V_inner);
    if (inner_iters == 0 || alpha_i.empty()) return;

    std::vector<double> E_in, w_dummy, U_inner;
    diagonalize_tridiagonal_ritz(alpha_i, beta_i, E_in, w_dummy, &U_inner);
    const std::size_t M_i = E_in.size();
    if (M_i == 0) return;

    // chi = O1 |n> projected onto inner basis (skipped for self-correlation).
    ComplexVector chi;
    std::vector<Complex> proj_O1;
    if (!is_self) {
        chi.resize(dim_inner);
        O1(ritz_n.data(), chi.data(), static_cast<int>(dim_inner));
        project_onto_basis(V_inner, chi, proj_O1);
    }

    for (std::size_t m = 0; m < M_i; ++m) {
        // <m|O2|n> = norm_phi * U_inner[0, m]  (real; inner Ritz from Hermitian H)
        const double M2_re = norm_phi * U_inner[0 + m * M_i];

        // <m|O1|n>
        Complex M1;
        if (is_self) {
            M1 = Complex(M2_re, 0.0);
        } else {
            M1 = Complex(0.0, 0.0);
            for (std::size_t l = 0; l < M_i; ++l) {
                const double u_lm = U_inner[l + m * M_i];
                M1 += std::conj(proj_O1[l]) * u_lm;
            }
        }

        // Contribution: conj(<m|O1|n>) * <m|O2|n>  (no w_outer; LTLM has w=1)
        const Complex prefac = std::conj(M1) * Complex(M2_re, 0.0);
        const double dE = E_in[m] - E_n;

        for (std::size_t t = 0; t < n_T; ++t) {
            const double boltz = std::exp(-betas[t] * (E_n - energy_shift));
            stat[t] += boltz * prefac;
        }

        for (std::size_t i = 0; i < n_omega; ++i) {
            const double dw  = omega_grid[i] - dE;
            const double lor = (eta * inv_pi) / (dw * dw + eta2);
            for (std::size_t t = 0; t < n_T; ++t) {
                const double boltz = std::exp(-betas[t] * (E_n - energy_shift));
                const std::size_t idx = t * n_omega + i;
                S_real[idx] += boltz * prefac.real() * lor;
                S_imag[idx] += boltz * prefac.imag() * lor;
            }
        }
    }
}

std::vector<double> linspace_ltlm(double lo, double hi, std::size_t n) {
    std::vector<double> v(n);
    if (n == 1) { v[0] = lo; return v; }
    const double step = (hi - lo) / static_cast<double>(n - 1);
    for (std::size_t i = 0; i < n; ++i) v[i] = lo + step * i;
    return v;
}

/// Core LTLM engine shared by all entry points.
/// @param dim_outer  Outer sector dimension (same as inner for intra-sector)
/// @param dim_inner  Inner sector dimension
LTLMDynamicalResult run_ltlm_loop(
    MatVec H_outer,
    MatVec H_inner,
    MatVec O1,
    MatVec O2,
    std::uint64_t dim_outer,
    std::uint64_t dim_inner,
    const std::vector<double>& omega_grid,
    const std::vector<double>& betas,
    double eta,
    bool is_self,
    const LTLMParameters& p)
{
    const std::size_t n_T     = betas.size();
    const std::size_t n_omega = omega_grid.size();

    // -------------------- Outer Lanczos --------------------
    // Generate deterministic initial vector.
    std::mt19937 gen;
    if (p.random_seed == 0) {
        std::random_device rd;
        gen.seed(rd());
    } else {
        gen.seed(static_cast<std::uint32_t>(p.random_seed));
    }
    // generateGaussianRandomVector takes int N and mt19937&
    ComplexVector v0 = generateGaussianRandomVector(
        static_cast<int>(dim_outer), gen);

    std::vector<double> alpha_o, beta_o;
    std::vector<ComplexVector> V_outer;
    const int outer_iters = build_lanczos_tridiagonal_with_basis(
        H_outer, v0, dim_outer, p.outer_krylov_dim,
        p.tolerance, p.full_reorthogonalization, p.reorth_frequency,
        alpha_o, beta_o, &V_outer);

    if (outer_iters == 0 || alpha_o.empty()) {
        throw std::runtime_error("LTLM: outer Lanczos produced 0 iterations");
    }

    std::vector<double> E_out, w_dummy, U_outer;
    diagonalize_tridiagonal_ritz(alpha_o, beta_o, E_out, w_dummy, &U_outer);
    const std::size_t M_o = E_out.size();
    if (M_o == 0) {
        throw std::runtime_error("LTLM: outer Ritz diagonalisation returned 0 pairs");
    }

    // Determine energy shift.
    const double gs_estimate = E_out.front();  // sorted ascending by Ritz
    const double energy_shift =
        (std::abs(p.energy_shift) > 0.0) ? p.energy_shift : gs_estimate;

    // Number of outer Ritz states to include.
    const std::size_t K = std::min(
        static_cast<std::size_t>(p.num_lowest_states), M_o);

    if (ltlm_verbose()) {
        std::cerr << "[LTLM] outer Lanczos: M=" << M_o
                  << "  K=" << K
                  << "  E[0]=" << E_out[0]
                  << "  shift=" << energy_shift << "\n";
    }

    // -------------------- Accumulation --------------------
    std::vector<double>  Z_acc(n_T, 0.0);
    std::vector<Complex> stat_acc(n_T, Complex(0.0, 0.0));
    std::vector<double>  S_real_acc(n_T * n_omega, 0.0);
    std::vector<double>  S_imag_acc(n_T * n_omega, 0.0);

    ComplexVector ritz_n(dim_outer);

    for (std::size_t n = 0; n < K; ++n) {
        reconstruct_ritz_vector(V_outer, U_outer, M_o, n, ritz_n);

        if (ltlm_verbose()) {
            std::cerr << "  [LTLM] eigenstate " << n
                      << "  E=" << E_out[n] << "\n";
        }

        // Always add this state's Boltzmann weight to Z, even if ||O|n>||=0.
        for (std::size_t t = 0; t < n_T; ++t)
            Z_acc[t] += std::exp(-betas[t] * (E_out[n] - energy_shift));

        accumulate_inner(H_inner, O1, O2, ritz_n, E_out[n],
                         dim_inner,
                         betas, omega_grid, eta, energy_shift, is_self, p,
                         stat_acc, S_real_acc, S_imag_acc);
    }

    // -------------------- Finalise --------------------
    LTLMDynamicalResult out;
    out.frequencies = omega_grid;
    out.betas = betas;
    out.spectral_real.assign(n_T * n_omega, 0.0);
    out.spectral_imag.assign(n_T * n_omega, 0.0);
    out.static_correlator.assign(n_T, Complex(0.0, 0.0));
    out.partition_function.assign(n_T, 0.0);

    for (std::size_t t = 0; t < n_T; ++t) {
        const double Z = Z_acc[t];
        out.partition_function[t] = Z;
        if (Z <= 0.0) continue;
        const double inv_Z = 1.0 / Z;
        out.static_correlator[t] = stat_acc[t] * inv_Z;
        for (std::size_t i = 0; i < n_omega; ++i) {
            const std::size_t idx = t * n_omega + i;
            out.spectral_real[idx] = S_real_acc[idx] * inv_Z;
            out.spectral_imag[idx] = S_imag_acc[idx] * inv_Z;
        }
    }

    out.ground_state_estimate = gs_estimate;
    out.energy_shift_used     = energy_shift;
    out.inner_lanczos_passes  = K;
    out.total_samples         = 1;   // single outer Lanczos pass

    return out;
}

} // anonymous namespace

// =============================================================================
// Public API
// =============================================================================

LTLMDynamicalResult compute_ltlm_dynamical_correlation(
    MatVec H,
    MatVec O1,
    MatVec O2,
    std::uint64_t dim,
    double omega_min,
    double omega_max,
    std::uint64_t n_omega,
    const std::vector<double>& betas,
    double eta,
    const LTLMParameters& params)
{
    if (dim == 0) throw std::invalid_argument("ltlm: dim must be > 0");
    if (n_omega == 0) throw std::invalid_argument("ltlm: n_omega must be > 0");
    if (eta <= 0.0) throw std::invalid_argument("ltlm: eta must be > 0");

    const auto omega_grid = linspace_ltlm(omega_min, omega_max, n_omega);

    // Self-correlation detection: probe a canonical basis vector.
    bool is_self = false;
    {
        ComplexVector probe(dim, Complex(0.0, 0.0));
        if (dim > 0) probe[0] = Complex(1.0, 0.0);
        ComplexVector a(dim), b(dim);
        O1(probe.data(), a.data(), static_cast<int>(dim));
        O2(probe.data(), b.data(), static_cast<int>(dim));
        double scale = 0.0;
        Complex diff(0.0, 0.0);
        for (std::size_t i = 0; i < dim; ++i) {
            diff  += a[i] - b[i];
            scale += std::norm(a[i]) + std::norm(b[i]);
        }
        if (scale > 0.0 &&
            std::abs(diff) < 1.0e-14 * std::sqrt(scale)) {
            is_self = true;
        }
    }

    return run_ltlm_loop(H, H, O1, O2, dim, dim,
                         omega_grid, betas, eta, is_self, params);
}

LTLMDynamicalResult compute_ltlm_dynamical_correlation_cross_sector(
    MatVec H_outer,
    MatVec H_inner,
    MatVec O1,
    MatVec O2,
    std::uint64_t dim_outer,
    std::uint64_t dim_inner,
    double omega_min,
    double omega_max,
    std::uint64_t n_omega,
    const std::vector<double>& betas,
    double eta,
    const LTLMParameters& params)
{
    if (dim_outer == 0 || dim_inner == 0)
        throw std::invalid_argument("ltlm: sector dims must be > 0");
    if (n_omega == 0) throw std::invalid_argument("ltlm: n_omega must be > 0");
    if (eta <= 0.0) throw std::invalid_argument("ltlm: eta must be > 0");

    const auto omega_grid = linspace_ltlm(omega_min, omega_max, n_omega);
    // Cross-sector O1 != O2 by construction (different spin sectors).
    return run_ltlm_loop(H_outer, H_inner, O1, O2, dim_outer, dim_inner,
                         omega_grid, betas, eta,
                         /*is_self=*/false, params);
}

LTLMDynamicalResult compute_ltlm_dynamical_correlation_from_states(
    MatVec H_inner,
    MatVec O1,
    MatVec O2,
    std::uint64_t dim_outer,
    std::uint64_t dim_inner,
    const std::vector<ComplexVector>& eigenstates,
    const std::vector<double>& energies,
    double omega_min,
    double omega_max,
    std::uint64_t n_omega,
    const std::vector<double>& betas,
    double eta,
    std::uint64_t inner_krylov_dim,
    double tolerance,
    bool full_reorth,
    std::uint64_t reorth_freq)
{
    if (eigenstates.empty() || energies.empty())
        throw std::invalid_argument("ltlm: eigenstates must be non-empty");
    if (eigenstates.size() != energies.size())
        throw std::invalid_argument("ltlm: eigenstates / energies size mismatch");
    if (n_omega == 0) throw std::invalid_argument("ltlm: n_omega must be > 0");
    if (eta <= 0.0) throw std::invalid_argument("ltlm: eta must be > 0");

    const auto omega_grid = linspace_ltlm(omega_min, omega_max, n_omega);
    const std::size_t n_T     = betas.size();
    const std::size_t n_omega_ = omega_grid.size();
    const std::size_t K = eigenstates.size();

    const double energy_shift = energies.front();  // assume sorted

    // Build a temporary LTLMParameters for the inner call.
    LTLMParameters p;
    p.inner_krylov_dim         = inner_krylov_dim;
    p.tolerance                = tolerance;
    p.full_reorthogonalization = full_reorth;
    p.reorth_frequency         = reorth_freq;
    p.energy_shift             = energy_shift;

    std::vector<double>  Z_acc(n_T, 0.0);
    std::vector<Complex> stat_acc(n_T, Complex(0.0, 0.0));
    std::vector<double>  S_real_acc(n_T * n_omega_, 0.0);
    std::vector<double>  S_imag_acc(n_T * n_omega_, 0.0);

    // Self-correlation detection via probe vector.
    bool is_self = false;
    if (!eigenstates.empty()) {
        ComplexVector probe(dim_outer, Complex(0.0, 0.0));
        if (dim_outer > 0) probe[0] = Complex(1.0, 0.0);
        ComplexVector a(dim_inner), b(dim_inner);
        O1(probe.data(), a.data(), static_cast<int>(dim_inner));
        O2(probe.data(), b.data(), static_cast<int>(dim_inner));
        double scale = 0.0;
        Complex diff(0.0, 0.0);
        for (std::size_t i = 0; i < dim_inner; ++i) {
            diff  += a[i] - b[i];
            scale += std::norm(a[i]) + std::norm(b[i]);
        }
        if (scale > 0.0 && std::abs(diff) < 1.0e-14 * std::sqrt(scale))
            is_self = true;
    }

    for (std::size_t n = 0; n < K; ++n) {
        // Always add this state's Boltzmann weight to Z.
        for (std::size_t t = 0; t < n_T; ++t)
            Z_acc[t] += std::exp(-betas[t] * (energies[n] - energy_shift));
        accumulate_inner(H_inner, O1, O2, eigenstates[n], energies[n],
                         dim_inner,
                         betas, omega_grid, eta, energy_shift, is_self, p,
                         stat_acc, S_real_acc, S_imag_acc);
    }

    LTLMDynamicalResult out;
    out.frequencies = omega_grid;
    out.betas = betas;
    out.spectral_real.assign(n_T * n_omega_, 0.0);
    out.spectral_imag.assign(n_T * n_omega_, 0.0);
    out.static_correlator.assign(n_T, Complex(0.0, 0.0));
    out.partition_function.assign(n_T, 0.0);

    for (std::size_t t = 0; t < n_T; ++t) {
        const double Z = Z_acc[t];
        out.partition_function[t] = Z;
        if (Z <= 0.0) continue;
        const double inv_Z = 1.0 / Z;
        out.static_correlator[t] = stat_acc[t] * inv_Z;
        for (std::size_t i = 0; i < n_omega_; ++i) {
            const std::size_t idx = t * n_omega_ + i;
            out.spectral_real[idx] = S_real_acc[idx] * inv_Z;
            out.spectral_imag[idx] = S_imag_acc[idx] * inv_Z;
        }
    }
    out.ground_state_estimate = energy_shift;
    out.energy_shift_used     = energy_shift;
    out.inner_lanczos_passes  = K;
    out.total_samples         = 1;
    return out;
}

} // namespace ed::ltlm
