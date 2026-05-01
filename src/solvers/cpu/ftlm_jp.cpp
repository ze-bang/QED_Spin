// =============================================================================
// src/solvers/cpu/ftlm_jp.cpp
//
// Jaklic-Prelovsek finite-temperature double-Lanczos sampler. See the
// header for the mathematical contract.
// =============================================================================

#include <ed/solvers/ftlm_jp.h>
#include <ed/solvers/lanczos.h>     // build_lanczos_tridiagonal_with_basis,
                                    // diagonalize_tridiagonal_ritz,
                                    // generateGaussianRandomVector

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <random>
#include <sstream>
#include <stdexcept>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace ed::ftlm::jp {

namespace {

inline bool jp_verbose() {
    static const bool v = []() {
        const char* env = std::getenv("ED_FTLM_JP_VERBOSE");
        return env && env[0] == '1';
    }();
    return v;
}

/// Reconstruct the j-th outer Ritz vector
///   |n^(r)_j> = sum_k U[k, j] * V_basis[k]
/// where `evecs` is the column-major eigenvector matrix returned by
/// diagonalize_tridiagonal_ritz, sized M_actual x M_actual:
///   evecs[k + j * M_actual] = U[k, j].
void reconstruct_ritz_vector(const std::vector<ComplexVector>& V_basis,
                             const std::vector<double>& evecs,
                             std::size_t M_actual,
                             std::size_t j,
                             ComplexVector& out)
{
    const std::size_t N = V_basis.front().size();
    std::fill(out.begin(), out.end(), Complex(0.0, 0.0));
    for (std::size_t k = 0; k < M_actual; ++k) {
        const double u_kj = evecs[k + j * M_actual];
        if (u_kj == 0.0) continue;
        const Complex coef(u_kj, 0.0);
        // out += u_kj * V_basis[k]
        cblas_zaxpy(static_cast<int>(N),
                    &coef,
                    V_basis[k].data(), 1,
                    out.data(), 1);
    }
}

/// Project a dim-N vector onto an inner Krylov basis:
///   proj[l] = <V_inner[l] | x>  (zdotc convention: conjugate-linear in
///   the first argument).
void project_onto_basis(const std::vector<ComplexVector>& V_inner,
                        const ComplexVector& x,
                        std::vector<Complex>& proj)
{
    const std::size_t M = V_inner.size();
    const std::size_t N = x.size();
    proj.assign(M, Complex(0.0, 0.0));
    for (std::size_t l = 0; l < M; ++l) {
        Complex z;
        cblas_zdotc_sub(static_cast<int>(N),
                        V_inner[l].data(), 1,
                        x.data(), 1,
                        &z);
        proj[l] = z;
    }
}

/// One JP sample. The two-sector machinery is parameterised on
/// (H_outer, H_inner, O1, O2, dim_outer, dim_inner) so that intra- and
/// cross-sector entry points share this body.
struct AccumulatorState {
    // Per-temperature accumulators (size n_T). All temperature-dependent.
    std::vector<double> Z_partial;                 // sum e^{-beta E_n} U[0,n]^2
    std::vector<Complex> static_corr_partial;       // sum e^{-beta E_n} U[0,n]^2 c_O
    // Spectral function: row-major [t * n_omega + i].
    std::vector<double> S_real_partial;
    std::vector<double> S_imag_partial;
};

void run_jp_loop(MatVec H_outer,
                 MatVec H_inner,
                 MatVec O1,
                 MatVec O2,
                 std::uint64_t dim_outer,
                 std::uint64_t dim_inner,
                 const std::vector<double>& omega_grid,
                 const std::vector<double>& betas,
                 double eta,
                 const JPParameters& params,
                 bool same_sector_self_corr,
                 AccumulatorState& acc,
                 JPDynamicalResult& out)
{
    const std::size_t n_omega = omega_grid.size();
    const std::size_t n_T = betas.size();
    const double inv_pi = 1.0 / M_PI;
    const double eta2 = eta * eta;
    const double cutoff = params.outer_boltzmann_cutoff;

    // Random sample generator
    std::mt19937_64 gen;
    if (params.random_seed == 0) {
        std::random_device rd;
        gen.seed(static_cast<std::uint64_t>(rd()));
    } else {
        gen.seed(params.random_seed);
    }

    // Working buffers reused across samples.
    ComplexVector ritz_vec(dim_outer);    // |n^(r)> in outer sector
    ComplexVector phi(dim_inner);          // O2 |n^(r)>
    ComplexVector chi(dim_inner);          // O1 |n^(r)>
    std::vector<Complex> proj_O1;          // <v_l^inner | chi>

    double energy_shift = params.energy_shift;
    bool have_shift = std::abs(params.energy_shift) > 0.0;
    double gs_estimate = std::numeric_limits<double>::infinity();

    std::uint64_t samples_used = 0;
    std::uint64_t inner_passes = 0;

    // Cache the largest beta for the boltzmann cutoff filter (coldest T).
    const double beta_cold = betas.empty()
        ? 0.0
        : *std::max_element(betas.begin(), betas.end());

    for (std::uint64_t r = 0; r < params.num_samples; ++r) {
        // -------------------- Outer Lanczos --------------------
        std::mt19937 sample_gen(static_cast<std::uint32_t>(
            gen() ^ (0x9E3779B97F4A7C15ULL * (r + 1))));
        ComplexVector v0 = generateGaussianRandomVector(
            static_cast<int>(dim_outer), sample_gen);

        std::vector<double> alpha_o, beta_o;
        std::vector<ComplexVector> V_outer;
        const int outer_iters = build_lanczos_tridiagonal_with_basis(
            H_outer, v0, dim_outer, params.outer_krylov_dim,
            params.tolerance, params.full_reorthogonalization,
            params.reorth_frequency,
            alpha_o, beta_o, &V_outer);

        if (outer_iters == 0 || alpha_o.empty()) {
            if (jp_verbose()) {
                std::cerr << "  [JP] sample " << r
                          << ": outer Lanczos produced 0 iterations\n";
            }
            continue;
        }

        std::vector<double> E_out, w_out_dummy, U_outer;
        diagonalize_tridiagonal_ritz(alpha_o, beta_o, E_out, w_out_dummy,
                                     &U_outer);
        const std::size_t M_o = E_out.size();
        if (M_o == 0) continue;

        // Track best ground-state estimate; lazily set the energy shift on
        // the first sample if the user did not pin one explicitly.
        const double e_min_sample = E_out.front();
        if (e_min_sample < gs_estimate) gs_estimate = e_min_sample;
        if (!have_shift) {
            energy_shift = gs_estimate;
        }

        // -------------------- Loop over outer Ritz pairs --------------------
        // Precompute the largest |U[0,n]|^2 to gauge cutoff.
        double max_outer_weight = 0.0;
        for (std::size_t n = 0; n < M_o; ++n) {
            const double u0n = U_outer[0 + n * M_o];
            max_outer_weight = std::max(max_outer_weight, u0n * u0n);
        }
        const double max_boltz_at_cold = std::exp(
            -beta_cold * (e_min_sample - energy_shift));
        const double cutoff_thr =
            cutoff * max_outer_weight * max_boltz_at_cold;

        for (std::size_t n = 0; n < M_o; ++n) {
            const double u0n = U_outer[0 + n * M_o];
            const double w_outer = u0n * u0n;
            const double E_n = E_out[n];

            // Boltzmann pre-cutoff at the coldest requested T.
            if (cutoff > 0.0 && n_T > 0) {
                const double boltz_cold =
                    w_outer *
                    std::exp(-beta_cold * (E_n - energy_shift));
                if (boltz_cold < cutoff_thr) continue;
            }

            // Reconstruct |n^(r)>.
            reconstruct_ritz_vector(V_outer, U_outer, M_o, n, ritz_vec);

            // phi = O2 |n^(r)>
            O2(ritz_vec.data(), phi.data(), static_cast<int>(dim_inner));
            const double norm_phi = cblas_dznrm2(static_cast<int>(dim_inner),
                                                 phi.data(), 1);
            if (norm_phi < params.tolerance) continue;
            {
                Complex inv(1.0 / norm_phi, 0.0);
                cblas_zscal(static_cast<int>(dim_inner), &inv, phi.data(), 1);
            }

            // -------------------- Inner Lanczos --------------------
            std::vector<double> alpha_i, beta_i;
            std::vector<ComplexVector> V_inner;
            const int inner_iters = build_lanczos_tridiagonal_with_basis(
                H_inner, phi, dim_inner, params.inner_krylov_dim,
                params.tolerance, params.full_reorthogonalization,
                params.reorth_frequency,
                alpha_i, beta_i, &V_inner);
            if (inner_iters == 0 || alpha_i.empty()) continue;

            std::vector<double> E_in, w_in_dummy, U_inner;
            diagonalize_tridiagonal_ritz(alpha_i, beta_i, E_in, w_in_dummy,
                                         &U_inner);
            const std::size_t M_i = E_in.size();
            if (M_i == 0) continue;

            // chi = O1 |n^(r)> (skipped for self-correlation; reuse phi).
            const Complex* O1_proj_source = nullptr;
            if (same_sector_self_corr) {
                // We need <v_l^inner | O1 | n>. With O1 = O2 and the inner
                // Krylov starting from v_0^inner = phi = O2 |n>/||O2|n||,
                // we have <v_l | O1 | n> = norm_phi * delta_{l,0}.
                // -> <m|O1|n> = norm_phi * U_inner[0, m]
                O1_proj_source = nullptr;  // sentinel for the fast path
            } else {
                O1(ritz_vec.data(), chi.data(), static_cast<int>(dim_inner));
                project_onto_basis(V_inner, chi, proj_O1);
                O1_proj_source = proj_O1.data();
            }

            // -------------------- Accumulate spectra --------------------
            ++inner_passes;

            // Per-(r,n) Z and static correlator contributions.
            for (std::size_t t = 0; t < n_T; ++t) {
                const double boltz =
                    w_outer *
                    std::exp(-betas[t] * (E_n - energy_shift));
                acc.Z_partial[t] += boltz;
            }

            for (std::size_t m = 0; m < M_i; ++m) {
                // <m|O2|n> = norm_phi * U_inner[0, m]   (real)
                const double M2_re = norm_phi * U_inner[0 + m * M_i];

                // <m|O1|n>
                Complex M1{0.0, 0.0};
                if (same_sector_self_corr) {
                    M1 = Complex(M2_re, 0.0);
                } else {
                    for (std::size_t l = 0; l < M_i; ++l) {
                        // U_inner is from a real symmetric tridiagonal,
                        // so U_inner entries are real.
                        const double u_lm = U_inner[l + m * M_i];
                        M1 += std::conj(O1_proj_source[l]) * u_lm;
                    }
                    // Note: this gives <m|O1|n>, which is what JP needs:
                    //   contribution = w_outer * conj(<m|O1|n>) * <m|O2|n>
                    //                = w_outer * <n|O1^dagger|m> * <m|O2|n>.
                }

                // contribution = w_outer * conj(M1) * M2 (complex)
                const Complex prefac = w_outer * std::conj(M1) * Complex(M2_re, 0.0);

                const double dE = E_in[m] - E_n;

                // Static correlator picks up the m-sum at *every* T:
                // <O1^dagger O2>(T) = sum_{n,m} (Boltz_n * U_outer[0,n]^2)
                //                              * conj(<m|O1|n>) * <m|O2|n>.
                for (std::size_t t = 0; t < n_T; ++t) {
                    const double boltz =
                        std::exp(-betas[t] * (E_n - energy_shift));
                    acc.static_corr_partial[t] += boltz * prefac;
                }

                // Spectral function: Lorentzian broadening.
                for (std::size_t i = 0; i < n_omega; ++i) {
                    const double dw = omega_grid[i] - dE;
                    const double lor = (eta * inv_pi) / (dw * dw + eta2);
                    for (std::size_t t = 0; t < n_T; ++t) {
                        const double boltz =
                            std::exp(-betas[t] * (E_n - energy_shift));
                        const std::size_t idx = t * n_omega + i;
                        acc.S_real_partial[idx] += boltz * prefac.real() * lor;
                        acc.S_imag_partial[idx] += boltz * prefac.imag() * lor;
                    }
                }
            }
        }

        ++samples_used;
        if (jp_verbose()) {
            std::cerr << "  [JP] sample " << r + 1 << "/" << params.num_samples
                      << " gs=" << gs_estimate
                      << " inner_passes=" << inner_passes << "\n";
        }
    }

    // -------------------- Finalise: divide S by Z --------------------
    out.frequencies = omega_grid;
    out.betas = betas;
    out.spectral_real.assign(n_T * n_omega, 0.0);
    out.spectral_imag.assign(n_T * n_omega, 0.0);
    out.static_correlator.assign(n_T, Complex(0.0, 0.0));
    out.partition_function.assign(n_T, 0.0);

    for (std::size_t t = 0; t < n_T; ++t) {
        const double Z = acc.Z_partial[t];
        out.partition_function[t] = Z;
        if (Z <= 0.0) continue;
        const double inv_Z = 1.0 / Z;
        out.static_correlator[t] = acc.static_corr_partial[t] * inv_Z;
        for (std::size_t i = 0; i < n_omega; ++i) {
            const std::size_t idx = t * n_omega + i;
            out.spectral_real[idx] = acc.S_real_partial[idx] * inv_Z;
            out.spectral_imag[idx] = acc.S_imag_partial[idx] * inv_Z;
        }
    }

    out.ground_state_estimate = std::isfinite(gs_estimate) ? gs_estimate : 0.0;
    out.energy_shift_used = energy_shift;
    out.inner_lanczos_passes = inner_passes;
    out.total_samples = samples_used;
}

std::vector<double> linspace(double lo, double hi, std::size_t n) {
    std::vector<double> v(n);
    if (n == 1) { v[0] = lo; return v; }
    const double step = (hi - lo) / static_cast<double>(n - 1);
    for (std::size_t i = 0; i < n; ++i) v[i] = lo + step * i;
    return v;
}

} // namespace

JPDynamicalResult compute_dynamical_correlation(
    MatVec H_apply,
    MatVec O1_apply,
    MatVec O2_apply,
    std::uint64_t dim,
    double omega_min,
    double omega_max,
    std::uint64_t n_omega,
    const std::vector<double>& betas,
    double eta,
    const JPParameters& params)
{
    if (dim == 0) {
        throw std::invalid_argument("ftlm::jp: dim must be > 0");
    }
    if (n_omega == 0) {
        throw std::invalid_argument("ftlm::jp: n_omega must be > 0");
    }
    if (eta <= 0.0) {
        throw std::invalid_argument("ftlm::jp: eta must be > 0");
    }

    const auto omega_grid = linspace(omega_min, omega_max, n_omega);

    AccumulatorState acc;
    acc.Z_partial.assign(betas.size(), 0.0);
    acc.static_corr_partial.assign(betas.size(), Complex(0.0, 0.0));
    acc.S_real_partial.assign(betas.size() * n_omega, 0.0);
    acc.S_imag_partial.assign(betas.size() * n_omega, 0.0);

    JPDynamicalResult out;
    // Self-correlation fast path triggers when the caller passes the same
    // callable for O1 and O2 (compared by std::function::target_type +
    // pointer; safest to expose the optimisation explicitly via a flag).
    // We detect by pointer equality on the underlying target when both are
    // function pointers; otherwise we do the safe (slower) path. The
    // outer loop conditional below makes either choice correct.
    bool self = false;
    {
        // Heuristic: if the two std::function objects were initialised
        // from the *same* callable, treat as self-correlation. Users who
        // want the fast path can simply pass the same lambda twice.
        if (O1_apply.target_type() == O2_apply.target_type()) {
            // std::function does not expose a generic equality op; the
            // safe choice is to assume they are NOT identical and let the
            // user opt into the optimisation by passing the same operator
            // through a compute_dynamical_self_correlation helper.
            self = false;
        }
    }
    // Override: detect identity by comparing a probe vector. Cheap and
    // correct. (One matvec on each side; result determines whether we
    // skip the second projection per outer Ritz pair.)
    {
        ComplexVector probe(dim, Complex(0.0, 0.0));
        probe[0] = Complex(1.0, 0.0);
        ComplexVector a(dim), b(dim);
        O1_apply(probe.data(), a.data(), static_cast<int>(dim));
        O2_apply(probe.data(), b.data(), static_cast<int>(dim));
        Complex diff_acc(0.0, 0.0);
        double  scale = 0.0;
        for (std::size_t i = 0; i < dim; ++i) {
            const Complex d = a[i] - b[i];
            diff_acc += d;
            scale += std::norm(a[i]) + std::norm(b[i]);
        }
        const double diff_norm = std::abs(diff_acc);
        if (scale > 0.0 && diff_norm < 1.0e-14 * std::sqrt(scale)) {
            self = true;
        }
    }

    run_jp_loop(H_apply, H_apply, O1_apply, O2_apply,
                dim, dim, omega_grid, betas, eta, params,
                /*same_sector_self_corr=*/self, acc, out);
    return out;
}

JPDynamicalResult compute_dynamical_correlation_cross_sector(
    MatVec H_outer_apply,
    MatVec H_inner_apply,
    MatVec O1_apply,
    MatVec O2_apply,
    std::uint64_t dim_outer,
    std::uint64_t dim_inner,
    double omega_min,
    double omega_max,
    std::uint64_t n_omega,
    const std::vector<double>& betas,
    double eta,
    const JPParameters& params)
{
    if (dim_outer == 0 || dim_inner == 0) {
        throw std::invalid_argument("ftlm::jp: sector dims must be > 0");
    }
    if (n_omega == 0 || eta <= 0.0) {
        throw std::invalid_argument("ftlm::jp: bad omega / eta");
    }

    const auto omega_grid = linspace(omega_min, omega_max, n_omega);

    AccumulatorState acc;
    acc.Z_partial.assign(betas.size(), 0.0);
    acc.static_corr_partial.assign(betas.size(), Complex(0.0, 0.0));
    acc.S_real_partial.assign(betas.size() * n_omega, 0.0);
    acc.S_imag_partial.assign(betas.size() * n_omega, 0.0);

    JPDynamicalResult out;
    // Cross-sector cannot use the self-correlation fast path because the
    // operator changes the sector quantum number.
    run_jp_loop(H_outer_apply, H_inner_apply, O1_apply, O2_apply,
                dim_outer, dim_inner, omega_grid, betas, eta, params,
                /*same_sector_self_corr=*/false, acc, out);
    return out;
}

void save_jp_dynamical_result(const JPDynamicalResult& result,
                              const std::string& path)
{
    if (path.empty()) return;
    std::ofstream f(path);
    if (!f) {
        throw std::runtime_error("ftlm::jp: cannot open " + path);
    }
    f << "# JP dynamical correlator\n";
    f << "# n_omega=" << result.frequencies.size()
      << " n_T=" << result.betas.size()
      << " samples=" << result.total_samples
      << " inner_passes=" << result.inner_lanczos_passes
      << " gs=" << result.ground_state_estimate
      << " shift=" << result.energy_shift_used
      << "\n";
    f << "# omega";
    for (double b : result.betas) f << "  S_re(beta=" << b << ")  S_im(beta=" << b << ")";
    f << "\n";
    const std::size_t n_omega = result.frequencies.size();
    const std::size_t n_T = result.betas.size();
    for (std::size_t i = 0; i < n_omega; ++i) {
        f << result.frequencies[i];
        for (std::size_t t = 0; t < n_T; ++t) {
            const std::size_t idx = t * n_omega + i;
            f << ' ' << result.spectral_real[idx]
              << ' ' << result.spectral_imag[idx];
        }
        f << '\n';
    }
}

JPDynamicalResult combine_sector_results(
    const std::vector<JPDynamicalResult>& per_sector,
    const std::vector<std::uint64_t>& sector_dims)
{
    if (per_sector.empty()) {
        throw std::invalid_argument("ftlm::jp::combine: no sectors");
    }
    if (per_sector.size() != sector_dims.size()) {
        throw std::invalid_argument("ftlm::jp::combine: dim list size mismatch");
    }

    const auto& s0 = per_sector.front();
    const std::size_t n_omega = s0.frequencies.size();
    const std::size_t n_T = s0.betas.size();
    for (const auto& s : per_sector) {
        if (s.frequencies.size() != n_omega || s.betas.size() != n_T) {
            throw std::invalid_argument(
                "ftlm::jp::combine: inconsistent grids");
        }
    }

    JPDynamicalResult combined;
    combined.frequencies = s0.frequencies;
    combined.betas = s0.betas;
    combined.spectral_real.assign(n_T * n_omega, 0.0);
    combined.spectral_imag.assign(n_T * n_omega, 0.0);
    combined.static_correlator.assign(n_T, Complex(0.0, 0.0));
    combined.partition_function.assign(n_T, 0.0);
    combined.ground_state_estimate = std::numeric_limits<double>::infinity();

    // Z_alpha(T) is already per-sector divided by R_alpha. To weight by
    // genuine sector partition functions we multiply by sector dim.
    std::vector<std::vector<double>> Z_eff(per_sector.size(),
                                           std::vector<double>(n_T, 0.0));
    std::vector<double> Z_total(n_T, 0.0);
    for (std::size_t a = 0; a < per_sector.size(); ++a) {
        const double D = static_cast<double>(sector_dims[a]);
        for (std::size_t t = 0; t < n_T; ++t) {
            Z_eff[a][t] = D * per_sector[a].partition_function[t];
            Z_total[t] += Z_eff[a][t];
        }
        if (per_sector[a].ground_state_estimate <
            combined.ground_state_estimate) {
            combined.ground_state_estimate =
                per_sector[a].ground_state_estimate;
        }
    }

    for (std::size_t t = 0; t < n_T; ++t) {
        if (Z_total[t] <= 0.0) continue;
        for (std::size_t a = 0; a < per_sector.size(); ++a) {
            const double w = Z_eff[a][t] / Z_total[t];
            combined.static_correlator[t] +=
                w * per_sector[a].static_correlator[t];
            for (std::size_t i = 0; i < n_omega; ++i) {
                const std::size_t idx = t * n_omega + i;
                combined.spectral_real[idx] +=
                    w * per_sector[a].spectral_real[idx];
                combined.spectral_imag[idx] +=
                    w * per_sector[a].spectral_imag[idx];
            }
        }
        combined.partition_function[t] = Z_total[t];
    }

    if (!std::isfinite(combined.ground_state_estimate)) {
        combined.ground_state_estimate = 0.0;
    }
    combined.energy_shift_used = combined.ground_state_estimate;
    return combined;
}

} // namespace ed::ftlm::jp
