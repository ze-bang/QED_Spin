// =============================================================================
// src/solvers/cpu/ftlm_sssf.cpp
//
// Static Spectral Structure Factor via JP (stochastic) and LTLM (deterministic).
// See include/ed/solvers/ftlm_sssf.h for the full mathematical specification.
// =============================================================================

#include <ed/solvers/ftlm_sssf.h>
#include <ed/solvers/lanczos.h>    // build_lanczos_tridiagonal_with_basis,
                                    // diagonalize_tridiagonal_ritz,
                                    // generateGaussianRandomVector

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <random>
#include <stdexcept>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace ed::sssf {

namespace {

inline bool sssf_verbose() {
    static const bool v = []() {
        const char* e = std::getenv("ED_SSSF_VERBOSE");
        return e && e[0] == '1';
    }();
    return v;
}

/// Reconstruct the j-th Ritz vector from the Lanczos basis.
/// evecs is column-major M×M: evecs[k + j*M] = U[k,j].
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

/// Core SSSF accumulation for one outer Ritz state |ñ_n>.
/// Computes <O1 ñ_n | O2 ñ_n> and accumulates into S and Z.
/// Z accumulation happens BEFORE this call (see callers).
///
/// @param w_outer  Statistical weight (U_outer[0,n]^2 for JP; 1.0 for LTLM)
void accumulate_sssf_state(
    MatVec O1, MatVec O2,
    const ComplexVector& ritz_n,
    double E_n,
    double w_outer,
    std::uint64_t dim_inner,
    const std::vector<double>& betas,
    double energy_shift,
    double tolerance,
    std::vector<Complex>& S_acc)
{
    const std::size_t n_T = betas.size();
    ComplexVector phi(dim_inner), chi(dim_inner);

    // phi = O2 |ñ_n>
    O2(ritz_n.data(), phi.data(), static_cast<int>(dim_inner));

    // chi = O1 |ñ_n>  (may be same as phi for self-correlation)
    O1(ritz_n.data(), chi.data(), static_cast<int>(dim_inner));

    // <O1 ñ_n | O2 ñ_n> = zdotc(chi, phi)
    Complex overlap;
    cblas_zdotc_sub(static_cast<int>(dim_inner),
                    chi.data(), 1, phi.data(), 1, &overlap);

    // S_acc[t] += w_outer * exp(-beta*(E_n-shift)) * <chi|phi>
    for (std::size_t t = 0; t < n_T; ++t) {
        const double boltz = w_outer *
            std::exp(-betas[t] * (E_n - energy_shift));
        S_acc[t] += boltz * overlap;
    }
    (void)tolerance;
}

/// Allocate and return a zeroed SSSFResult for n_T temperatures.
SSSFResult make_result(const std::vector<double>& betas) {
    SSSFResult r;
    r.betas = betas;
    const std::size_t n_T = betas.size();
    r.S_static_real.assign(n_T, 0.0);
    r.S_static_imag.assign(n_T, 0.0);
    r.partition_function.assign(n_T, 0.0);
    r.ground_state_estimate = std::numeric_limits<double>::infinity();
    r.energy_shift_used = 0.0;
    r.total_samples = 0;
    r.total_outer_ritz_processed = 0;
    return r;
}

/// Self-correlation detection via a probe vector.
bool detect_self_correlation(MatVec O1, MatVec O2, std::uint64_t dim) {
    if (dim == 0) return false;
    ComplexVector probe(dim, Complex(0.0, 0.0));
    probe[0] = Complex(1.0, 0.0);
    ComplexVector a(dim), b(dim);
    O1(probe.data(), a.data(), static_cast<int>(dim));
    O2(probe.data(), b.data(), static_cast<int>(dim));
    double scale = 0.0; Complex diff(0.0,0.0);
    for (std::size_t i = 0; i < dim; ++i) {
        diff  += a[i] - b[i];
        scale += std::norm(a[i]) + std::norm(b[i]);
    }
    return scale > 0.0 && std::abs(diff) < 1.0e-14 * std::sqrt(scale);
}

// ---------------------------------------------------------------------------
// JP core loop: outer Lanczos from random samples
// ---------------------------------------------------------------------------
void run_sssf_jp_loop(
    MatVec H_outer, MatVec O1, MatVec O2,
    std::uint64_t dim_outer, std::uint64_t dim_inner,
    const std::vector<double>& betas,
    const ed::ftlm::jp::JPParameters& params,
    SSSFResult& out)
{
    const std::size_t n_T = betas.size();
    const double cutoff = params.outer_boltzmann_cutoff;
    const double beta_cold = betas.empty()
        ? 0.0 : *std::max_element(betas.begin(), betas.end());

    std::mt19937_64 gen;
    if (params.random_seed == 0) {
        std::random_device rd; gen.seed(rd());
    } else {
        gen.seed(params.random_seed);
    }

    std::vector<Complex> S_acc(n_T, Complex(0.0, 0.0));
    std::vector<double>  Z_acc(n_T, 0.0);
    double gs_estimate = std::numeric_limits<double>::infinity();
    double energy_shift = params.energy_shift;
    bool have_shift = std::abs(params.energy_shift) > 0.0;
    std::uint64_t ritz_processed = 0;

    ComplexVector ritz_n(dim_outer);

    for (std::uint64_t r = 0; r < params.num_samples; ++r) {
        std::mt19937 sg(static_cast<std::uint32_t>(
            gen() ^ (0x9E3779B97F4A7C15ULL * (r + 1))));
        ComplexVector v0 = generateGaussianRandomVector(
            static_cast<int>(dim_outer), sg);

        std::vector<double> alpha_o, beta_o;
        std::vector<ComplexVector> V_outer;
        const int M = build_lanczos_tridiagonal_with_basis(
            H_outer, v0, dim_outer, params.outer_krylov_dim,
            params.tolerance, params.full_reorthogonalization,
            params.reorth_frequency, alpha_o, beta_o, &V_outer);

        if (M == 0 || alpha_o.empty()) continue;

        std::vector<double> E_out, w_dummy, U_outer;
        diagonalize_tridiagonal_ritz(alpha_o, beta_o, E_out, w_dummy, &U_outer);
        const std::size_t M_o = E_out.size();
        if (M_o == 0) continue;

        const double e_min_s = E_out.front();
        if (e_min_s < gs_estimate) gs_estimate = e_min_s;
        if (!have_shift) energy_shift = gs_estimate;

        double max_w = 0.0;
        for (std::size_t n = 0; n < M_o; ++n) {
            const double u0 = U_outer[0 + n * M_o];
            max_w = std::max(max_w, u0 * u0);
        }
        const double cutoff_thr = cutoff * max_w *
            std::exp(-beta_cold * (e_min_s - energy_shift));

        for (std::size_t n = 0; n < M_o; ++n) {
            const double u0 = U_outer[0 + n * M_o];
            const double w_outer = u0 * u0;
            const double E_n = E_out[n];

            if (cutoff > 0.0 && n_T > 0) {
                const double bc = w_outer *
                    std::exp(-beta_cold * (E_n - energy_shift));
                if (bc < cutoff_thr) continue;
            }

            // Always accumulate Z.
            for (std::size_t t = 0; t < n_T; ++t)
                Z_acc[t] += w_outer *
                    std::exp(-betas[t] * (E_n - energy_shift));

            reconstruct_ritz(V_outer, U_outer, M_o, n, ritz_n);
            accumulate_sssf_state(O1, O2, ritz_n, E_n, w_outer,
                                  dim_inner, betas, energy_shift,
                                  params.tolerance, S_acc);
            ++ritz_processed;
        }
        ++out.total_samples;

        if (sssf_verbose()) {
            std::cerr << "  [SSSF-JP] sample " << r+1 << "/"
                      << params.num_samples
                      << " gs=" << gs_estimate << "\n";
        }
    }

    // Finalise.
    out.partition_function = Z_acc;
    out.ground_state_estimate = std::isfinite(gs_estimate) ? gs_estimate : 0.0;
    out.energy_shift_used     = energy_shift;
    out.total_outer_ritz_processed = ritz_processed;

    for (std::size_t t = 0; t < n_T; ++t) {
        const double Z = Z_acc[t];
        if (Z <= 0.0) continue;
        const Complex c = S_acc[t] / Z;
        out.S_static_real[t] = c.real();
        out.S_static_imag[t] = c.imag();
    }
}

// ---------------------------------------------------------------------------
// LTLM core loop: outer Lanczos, take K lowest Ritz states
// ---------------------------------------------------------------------------
void run_sssf_ltlm_loop(
    MatVec H_outer, MatVec O1, MatVec O2,
    std::uint64_t dim_outer, std::uint64_t dim_inner,
    const std::vector<double>& betas,
    const ed::ltlm::LTLMParameters& params,
    SSSFResult& out)
{
    const std::size_t n_T = betas.size();

    std::mt19937 gen;
    if (params.random_seed == 0) {
        std::random_device rd; gen.seed(rd());
    } else {
        gen.seed(static_cast<std::uint32_t>(params.random_seed));
    }
    ComplexVector v0 = generateGaussianRandomVector(
        static_cast<int>(dim_outer), gen);

    std::vector<double> alpha_o, beta_o;
    std::vector<ComplexVector> V_outer;
    const int M = build_lanczos_tridiagonal_with_basis(
        H_outer, v0, dim_outer, params.outer_krylov_dim,
        params.tolerance, params.full_reorthogonalization,
        params.reorth_frequency, alpha_o, beta_o, &V_outer);

    if (M == 0 || alpha_o.empty())
        throw std::runtime_error("SSSF LTLM: outer Lanczos produced 0 iterations");

    std::vector<double> E_out, w_dummy, U_outer;
    diagonalize_tridiagonal_ritz(alpha_o, beta_o, E_out, w_dummy, &U_outer);
    const std::size_t M_o = E_out.size();
    if (M_o == 0)
        throw std::runtime_error("SSSF LTLM: outer Ritz returned 0 pairs");

    const double gs_estimate = E_out.front();
    const double energy_shift =
        (std::abs(params.energy_shift) > 0.0) ? params.energy_shift : gs_estimate;
    const std::size_t K = std::min(
        static_cast<std::size_t>(params.num_lowest_states), M_o);

    std::vector<Complex> S_acc(n_T, Complex(0.0, 0.0));
    std::vector<double>  Z_acc(n_T, 0.0);
    ComplexVector ritz_n(dim_outer);

    for (std::size_t n = 0; n < K; ++n) {
        const double E_n = E_out[n];
        // Always add to Z.
        for (std::size_t t = 0; t < n_T; ++t)
            Z_acc[t] += std::exp(-betas[t] * (E_n - energy_shift));
        reconstruct_ritz(V_outer, U_outer, M_o, n, ritz_n);
        accumulate_sssf_state(O1, O2, ritz_n, E_n, /*w_outer=*/1.0,
                              dim_inner, betas, energy_shift,
                              params.tolerance, S_acc);
    }

    out.partition_function    = Z_acc;
    out.ground_state_estimate = gs_estimate;
    out.energy_shift_used     = energy_shift;
    out.total_samples         = 1;
    out.total_outer_ritz_processed = K;

    for (std::size_t t = 0; t < n_T; ++t) {
        const double Z = Z_acc[t];
        if (Z <= 0.0) continue;
        const Complex c = S_acc[t] / Z;
        out.S_static_real[t] = c.real();
        out.S_static_imag[t] = c.imag();
    }
}

} // anonymous namespace

// =============================================================================
// Public API
// =============================================================================

SSSFResult combine_sector_sssf(
    const std::vector<SSSFResult>& per_sector,
    const std::vector<std::uint64_t>& sector_dims)
{
    if (per_sector.empty())
        throw std::invalid_argument("sssf: per_sector must be non-empty");
    if (per_sector.size() != sector_dims.size())
        throw std::invalid_argument("sssf: per_sector / sector_dims size mismatch");

    const std::size_t n_T = per_sector.front().betas.size();
    SSSFResult out = make_result(per_sector.front().betas);
    out.ground_state_estimate = std::numeric_limits<double>::infinity();

    // Effective partition function per sector: Z_eff_s = d_s * Z_s.
    for (std::size_t s = 0; s < per_sector.size(); ++s) {
        const double d = static_cast<double>(sector_dims[s]);
        const auto& sr = per_sector[s];
        out.ground_state_estimate =
            std::min(out.ground_state_estimate, sr.ground_state_estimate);
        for (std::size_t t = 0; t < n_T; ++t)
            out.partition_function[t] += d * sr.partition_function[t];
    }

    for (std::size_t t = 0; t < n_T; ++t) {
        const double Z_total = out.partition_function[t];
        if (Z_total <= 0.0) continue;
        double S_r = 0.0, S_i = 0.0;
        for (std::size_t s = 0; s < per_sector.size(); ++s) {
            const double d = static_cast<double>(sector_dims[s]);
            const double Z_s = per_sector[s].partition_function[t];
            const double w = d * Z_s / Z_total;
            S_r += w * per_sector[s].S_static_real[t];
            S_i += w * per_sector[s].S_static_imag[t];
        }
        out.S_static_real[t] = S_r;
        out.S_static_imag[t] = S_i;
    }
    return out;
}

SSSFResult compute_sssf_jp(
    MatVec H, MatVec O1, MatVec O2,
    std::uint64_t dim,
    const std::vector<double>& betas,
    const ed::ftlm::jp::JPParameters& params)
{
    if (dim == 0) throw std::invalid_argument("sssf: dim must be > 0");
    auto out = make_result(betas);
    // For self-correlation, O1 and O2 produce the same vector; the dot-product
    // <phi|phi> = norm^2 is real-positive, so the imaginary part is zero.
    run_sssf_jp_loop(H, O1, O2, dim, dim, betas, params, out);
    return out;
}

SSSFResult compute_sssf_jp_cross_sector(
    MatVec H_outer, MatVec O1, MatVec O2,
    std::uint64_t dim_outer, std::uint64_t dim_inner,
    const std::vector<double>& betas,
    const ed::ftlm::jp::JPParameters& params)
{
    if (dim_outer == 0 || dim_inner == 0)
        throw std::invalid_argument("sssf: sector dims must be > 0");
    auto out = make_result(betas);
    run_sssf_jp_loop(H_outer, O1, O2, dim_outer, dim_inner, betas, params, out);
    return out;
}

SSSFResult compute_sssf_ltlm(
    MatVec H, MatVec O1, MatVec O2,
    std::uint64_t dim,
    const std::vector<double>& betas,
    const ed::ltlm::LTLMParameters& params)
{
    if (dim == 0) throw std::invalid_argument("sssf: dim must be > 0");
    auto out = make_result(betas);
    run_sssf_ltlm_loop(H, O1, O2, dim, dim, betas, params, out);
    return out;
}

SSSFResult compute_sssf_ltlm_from_states(
    MatVec O1, MatVec O2,
    std::uint64_t dim_outer, std::uint64_t dim_inner,
    const std::vector<ComplexVector>& eigenstates,
    const std::vector<double>& energies,
    const std::vector<double>& betas)
{
    if (eigenstates.empty() || energies.empty())
        throw std::invalid_argument("sssf: eigenstates must be non-empty");
    if (eigenstates.size() != energies.size())
        throw std::invalid_argument("sssf: eigenstates / energies size mismatch");

    const std::size_t n_T = betas.size();
    const std::size_t K   = eigenstates.size();
    const double energy_shift = energies.front();   // assume sorted ascending

    auto out = make_result(betas);
    out.ground_state_estimate = energy_shift;
    out.energy_shift_used     = energy_shift;

    std::vector<Complex> S_acc(n_T, Complex(0.0, 0.0));
    std::vector<double>  Z_acc(n_T, 0.0);

    for (std::size_t n = 0; n < K; ++n) {
        const double E_n = energies[n];
        for (std::size_t t = 0; t < n_T; ++t)
            Z_acc[t] += std::exp(-betas[t] * (E_n - energy_shift));
        accumulate_sssf_state(O1, O2, eigenstates[n], E_n, /*w_outer=*/1.0,
                              dim_inner, betas, energy_shift,
                              /*tolerance=*/1e-30, S_acc);
    }

    out.partition_function         = Z_acc;
    out.total_samples              = 1;
    out.total_outer_ritz_processed = K;

    for (std::size_t t = 0; t < n_T; ++t) {
        const double Z = Z_acc[t];
        if (Z <= 0.0) continue;
        const Complex c = S_acc[t] / Z;
        out.S_static_real[t] = c.real();
        out.S_static_imag[t] = c.imag();
    }
    return out;
}

SSSFResult compute_sssf_ltlm_cross_sector(
    MatVec H_outer, MatVec O1, MatVec O2,
    std::uint64_t dim_outer, std::uint64_t dim_inner,
    const std::vector<double>& betas,
    const ed::ltlm::LTLMParameters& params)
{
    if (dim_outer == 0 || dim_inner == 0)
        throw std::invalid_argument("sssf: sector dims must be > 0");
    auto out = make_result(betas);
    run_sssf_ltlm_loop(H_outer, O1, O2, dim_outer, dim_inner, betas, params, out);
    return out;
}

} // namespace ed::sssf
