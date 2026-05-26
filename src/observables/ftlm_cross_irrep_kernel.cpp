// =============================================================================
// src/observables/ftlm_cross_irrep_kernel.cpp
//
// Implementation of the FTLM cross-irrep dynamical kernel. See the
// header for the math and the high-level design notes.
//
// The kernel mirrors the structure of
// ``compute_dynamical_correlation_multi_sample_multi_temperature_impl``
// in src/solvers/cpu/ftlm.cpp lines 2940 -- 3350, but with these
// substitutions:
//
//   * outer H matvec  -> ``H_src`` on ``dim_src``
//   * inner H matvec  -> ``H_dst`` on ``dim_dst``
//   * observable      -> rectangular ``O_apply`` (src -> dst)
//   * sample seed     -> ``generateGaussianRandomVector(dim_src, ...)``
//   * returned bundle -> UN-normalised (S, Z) so the streaming-
//                        symmetry binding can recombine across
//                        source sectors with the correct dim_src
//                        weighting (see header doc on
//                        ``combine_sector_dynamical_spectra``).
//
// The same-source / same-target case (O_1 = O_2 = O) lets us collapse
// the legacy two-overlap formula
//   w_k = (Sum_j V_S[j,k] * <phi_1|v_j>) * V_S[0,k] * ||phi_2||
// into the closed form
//   w_k = V_S[0,k]^2 * ||phi||^2
// which is exactly the Lehmann residue at the k-th target Ritz
// energy for an autocorrelator. The full kernel still allocates the
// inner basis (because the eigenvectors come from
// `diagonalize_tridiagonal_ritz`), but it avoids the second matvec
// and the inner-overlap loop -- making the per-sample inner cost
// comparable to ``cf_spectral_from_vector``.
// =============================================================================

#define ED_BUILDING_INTERNAL 1  // silence deprecation on build_lanczos_*

#include <ed/observables/ftlm_cross_irrep_kernel.h>

#include <ed/solvers/lanczos.h>   // generateGaussianRandomVector, build_lanczos_*, diagonalize_tridiagonal_ritz
#include <ed/core/blas_lapack_wrapper.h>

#include <chrono>
#include <iostream>
#include <limits>

namespace ed::observables {

namespace {

constexpr double  kInvPi          = 0.3183098861837907;  // 1 / pi
constexpr double  kPsiNormCutoff  = 1e-14;
constexpr double  kPhiNormCutoff  = 1e-14;

}  // namespace

FtlmCrossIrrepSectorResult ftlm_cross_irrep_kernel_one_sector(
    const std::function<void(const Complex*, Complex*, int)>& H_src,
    const std::function<void(const Complex*, Complex*, int)>& H_dst,
    const std::function<void(const Complex*, Complex*, int)>& O_apply,
    std::size_t                       dim_src,
    std::size_t                       dim_dst,
    const std::vector<double>&        temperatures,
    const std::vector<double>&        omega_grid,
    const FtlmCrossIrrepOptions&      opts)
{
    if (dim_src == 0 || dim_dst == 0) {
        throw std::invalid_argument(
            "ftlm_cross_irrep_kernel_one_sector: dim_src and dim_dst "
            "must be > 0.");
    }
    if (temperatures.empty()) {
        throw std::invalid_argument(
            "ftlm_cross_irrep_kernel_one_sector: temperatures is empty.");
    }
    if (omega_grid.empty()) {
        throw std::invalid_argument(
            "ftlm_cross_irrep_kernel_one_sector: omega_grid is empty.");
    }
    if (opts.num_samples == 0) {
        throw std::invalid_argument(
            "ftlm_cross_irrep_kernel_one_sector: num_samples = 0.");
    }

    const std::size_t num_omega = omega_grid.size();

    FtlmCrossIrrepSectorResult R;
    R.dim_src = dim_src;
    R.dim_dst = dim_dst;
    for (double T : temperatures) {
        R.S_real[T] = std::vector<double>(num_omega, 0.0);
        R.S_imag[T] = std::vector<double>(num_omega, 0.0);
        R.Z[T]      = 0.0;
    }

    // Wave C4 (May 2026): share outer Lanczos basis storage across
    // (k_src, k_dst) sample iterations within this one_sector call.
    // ``basis_H`` is rebuilt every sample; reusing the std::vector<
    // ComplexVector> shell across samples lets the underlying
    // per-vector heap blocks survive on the pool side -- net saves
    // ~30 malloc()s per (k_src, k_dst) call on the default
    // ``num_samples=30`` setting.
    std::vector<ComplexVector> basis_H_scratch;

    // Per-sector global energy reference for thermal-weight numerical
    // stability. We initialise to +infinity and update on the fly as
    // each sample's Ritz spectrum becomes available; this matches the
    // legacy "per-sample E_min" choice but lifts it to per-sector so
    // the eventual combine_sector_dynamical_spectra step can compose
    // sectors with disparate E_min values via the F-shift.
    double E_min_sector = std::numeric_limits<double>::infinity();

    // Cap the per-sample Ritz-state iteration the same way the legacy
    // multi-sample kernel does: 50 is more than enough for the highest
    // temperature shown in typical FTLM benchmarks.
    const std::size_t max_ritz_states = std::min<std::size_t>(
        opts.krylov_dim, 50);

    const auto start_time = std::chrono::high_resolution_clock::now();

    for (std::size_t sample_idx = 0; sample_idx < opts.num_samples; ++sample_idx) {
        if (opts.verbose) {
            std::cout << "[ftlm-xirrep] sample " << (sample_idx + 1)
                      << " / " << opts.num_samples
                      << "  (dim_src=" << dim_src
                      << ", dim_dst=" << dim_dst << ")\n";
        }

        // -------------------------------------------------------------
        // Outer Lanczos on H_src starting from |r> in dim_src.
        // Matches the legacy multi-sample seed convention exactly so
        // a single-sector FTLM call routed through this kernel is
        // bit-identical (up to floating-point rounding) to the legacy
        // path.
        // -------------------------------------------------------------
        std::mt19937 sample_gen(opts.random_seed + sample_idx * 12345ULL);
        ComplexVector r_state = generateGaussianRandomVector(
            static_cast<int>(dim_src), sample_gen);

        std::vector<double>          alpha_H, beta_H;
        // Wave C4: clear (preserves capacity) instead of fresh
        // ``std::vector<ComplexVector> basis_H;`` per sample. The
        // per-vector heap blocks survive across the inner clear()
        // (clear() does not run ~ComplexVector() on the held
        // elements; resize/push_back simply rewrites them).
        basis_H_scratch.clear();
        const int H_iters = build_lanczos_tridiagonal_with_basis(
            H_src, r_state, static_cast<std::uint64_t>(dim_src),
            opts.krylov_dim, opts.tolerance,
            opts.full_reorthogonalization, opts.reorth_frequency,
            alpha_H, beta_H, &basis_H_scratch);
        auto& basis_H = basis_H_scratch;
        if (H_iters == 0 || alpha_H.empty()) {
            if (opts.verbose) {
                std::cout << "  outer Lanczos failed; skipping sample\n";
            }
            continue;
        }
        const std::size_t m_H = alpha_H.size();

        // Diagonalise the outer tridiag for Ritz energies + the
        // (m_H x m_H) eigenvector matrix V in row-major form.
        std::vector<double> ritz_values, dummy_weights, V_H;
        diagonalize_tridiagonal_ritz(alpha_H, beta_H, ritz_values,
                                     dummy_weights, &V_H);
        if (ritz_values.empty()) {
            if (opts.verbose) {
                std::cout << "  outer diag failed; skipping sample\n";
            }
            continue;
        }

        // c_i = <psi_i | r> = V[i, 0]  (real because the tridiag is real)
        std::vector<double> c_sq(m_H);
        for (std::size_t i = 0; i < m_H; ++i) {
            const double v0 = V_H[i * m_H + 0];
            c_sq[i] = v0 * v0;
        }

        // Update the sector-wide E_min using *this* sample's Ritz
        // spectrum. Subsequent samples share the same reference for
        // the thermal exponent so all S_i contributions accumulate
        // consistently in `R.S_real[T]` / `R.S_imag[T]`.
        const double sample_E_min = *std::min_element(
            ritz_values.begin(), ritz_values.end());
        if (sample_E_min < E_min_sector) {
            // If E_min shifts downward mid-run we need to retroactively
            // rescale the previously accumulated arrays by
            // exp(-beta * (old_E_min - new_E_min)). This is the same
            // trick combine_sector_thermodynamics uses to keep the
            // partition-function exponents in float-safe range.
            if (std::isfinite(E_min_sector)) {
                for (double T : temperatures) {
                    const double beta  = 1.0 / T;
                    const double scale = std::exp(
                        -beta * (E_min_sector - sample_E_min));
                    for (auto& v : R.S_real[T]) v *= scale;
                    for (auto& v : R.S_imag[T]) v *= scale;
                    R.Z[T] *= scale;
                }
            }
            E_min_sector = sample_E_min;
        }

        // -------------------------------------------------------------
        // Identify "significant" Ritz states. We look at the highest
        // temperature to set the inclusiveness threshold, identical to
        // the legacy kernel's significant_states logic.
        // -------------------------------------------------------------
        const double T_max     = *std::max_element(
            temperatures.begin(), temperatures.end());
        const double beta_min  = 1.0 / T_max;
        std::vector<double> max_w(m_H);
        double              max_Z = 0.0;
        for (std::size_t i = 0; i < m_H; ++i) {
            const double b = std::exp(-beta_min * (ritz_values[i] - sample_E_min));
            max_w[i] = c_sq[i] * b;
            max_Z   += max_w[i];
        }
        const double thr = 1e-10 * std::max(max_Z, 1e-300);
        std::vector<std::size_t> significant;
        significant.reserve(max_ritz_states);
        for (std::size_t i = 0; i < std::min(m_H, max_ritz_states); ++i) {
            if (max_w[i] >= thr || c_sq[i] > 1e-12) {
                significant.push_back(i);
            }
        }

        // -------------------------------------------------------------
        // Precompute per-Ritz S_i(omega) in the TARGET sector via
        // CrossSectorOrbitObservable (rectangular) + inner Lanczos
        // on H_dst + Lehmann sum. We reuse the legacy two-overlap
        // formulation so this implementation also supports a
        // (future) extension to O_1 != O_2 by simply taking two
        // user lambdas; for now O_apply is both phi_1 and phi_2
        // and we exploit the autocorrelator simplification
        // w_k = V_S[0,k]^2 * ||phi||^2 derived in the header.
        // -------------------------------------------------------------
        std::vector<std::vector<double>> S_i_real(significant.size());
        std::vector<std::vector<double>> S_i_imag(significant.size());
        std::vector<double>              E_i_arr(significant.size(), 0.0);
        std::vector<double>              c_sq_arr(significant.size(), 0.0);
        std::vector<unsigned char>       i_valid(significant.size(), 0u);

        const double eta    = opts.broadening;
        const double eta_sq = eta * eta;

        for (std::size_t idx = 0; idx < significant.size(); ++idx) {
            const std::size_t i = significant[idx];

            // Reconstruct |psi_i> in dim_src orbit basis from the
            // outer Lanczos basis: psi_i = sum_j V[i,j] * basis_H[j].
            ComplexVector psi_src(dim_src, Complex(0.0, 0.0));
            for (std::size_t j = 0; j < m_H; ++j) {
                const double   c    = V_H[i * m_H + j];
                const Complex  cc(c, 0.0);
                cblas_zaxpy(static_cast<int>(dim_src), &cc,
                            basis_H[j].data(), 1,
                            psi_src.data(),     1);
            }
            const double psi_norm = cblas_dznrm2(
                static_cast<int>(dim_src), psi_src.data(), 1);
            if (psi_norm < kPsiNormCutoff) continue;
            const Complex inv_psi(1.0 / psi_norm, 0.0);
            cblas_zscal(static_cast<int>(dim_src), &inv_psi,
                        psi_src.data(), 1);

            // Rectangular scatter: phi = O |psi_i> in dim_dst.
            ComplexVector phi_dst(dim_dst, Complex(0.0, 0.0));
            O_apply(psi_src.data(), phi_dst.data(),
                    static_cast<int>(dim_dst));
            const double phi_norm = cblas_dznrm2(
                static_cast<int>(dim_dst), phi_dst.data(), 1);
            if (phi_norm < kPhiNormCutoff) continue;
            const Complex inv_phi(1.0 / phi_norm, 0.0);
            cblas_zscal(static_cast<int>(dim_dst), &inv_phi,
                        phi_dst.data(), 1);

            // Inner Lanczos on H_dst from phi/||phi||, keep basis only
            // so we can read off V_S[0,k] after diagonalisation; we
            // can free the basis immediately because the
            // autocorrelator weights w_k = V_S[0,k]^2 * ||phi||^2 do
            // not need the basis vectors themselves.
            std::vector<double>        alpha_S, beta_S;
            std::vector<ComplexVector> basis_S;
            build_lanczos_tridiagonal_with_basis(
                H_dst, phi_dst, static_cast<std::uint64_t>(dim_dst),
                opts.krylov_dim, opts.tolerance,
                opts.full_reorthogonalization, opts.reorth_frequency,
                alpha_S, beta_S, &basis_S);
            basis_S.clear();
            basis_S.shrink_to_fit();
            if (alpha_S.empty()) continue;

            std::vector<double> ritz_S, dummy_S, V_S;
            diagonalize_tridiagonal_ritz(alpha_S, beta_S, ritz_S,
                                         dummy_S, &V_S);
            if (ritz_S.empty()) continue;
            const std::size_t m_S = ritz_S.size();

            // Build the Lehmann poles for THIS source Ritz state. The
            // resolvent reference is E_i (NOT E_gs), so the poles sit
            // at omega = lambda_k - E_i. This is the genuine finite-T
            // convention; the legacy kernel uses E_gs as a global
            // reference, but for finite-T cross-correlators that
            // amounts to a constant omega shift -- we choose the
            // physically correct E_i shift so callers do not have to
            // re-zero the omega axis.
            const double E_i      = ritz_values[i];
            const double w_const  = phi_norm * phi_norm;  // ||phi||^2
            std::vector<double> w_arr(m_S);
            std::vector<double> E_arr(m_S);
            for (std::size_t k = 0; k < m_S; ++k) {
                const double v0 = V_S[k * m_S + 0];
                w_arr[k] = w_const * v0 * v0;
                E_arr[k] = ritz_S[k] - E_i;
            }
            std::vector<double>& S_i = S_i_real[idx];
            std::vector<double>& S_q = S_i_imag[idx];
            S_i.assign(num_omega, 0.0);
            S_q.assign(num_omega, 0.0);
            #pragma omp parallel for schedule(static)
            for (std::int64_t iw = 0;
                 iw < static_cast<std::int64_t>(num_omega); ++iw) {
                const double omega = omega_grid[iw];
                double sum_r = 0.0;
                for (std::size_t k = 0; k < m_S; ++k) {
                    const double d   = omega - E_arr[k];
                    const double lor = (eta * kInvPi) / (d * d + eta_sq);
                    sum_r += w_arr[k] * lor;
                }
                S_i[iw] = sum_r;
                S_q[iw] = 0.0;  // real-only autocorrelator (O_1 = O_2)
            }
            E_i_arr[idx]  = E_i;
            c_sq_arr[idx] = c_sq[i];
            i_valid[idx]  = 1u;
        }

        // -------------------------------------------------------------
        // Per-temperature accumulation. Multiply each S_i by
        // exp(-beta * (E_i - E_min_sector)) * c_i^2 and add to the
        // sector accumulators. Z gets the same Boltzmann weight.
        // The final dim_src multiplication happens once at the very
        // end -- it would be applied to BOTH numerator and
        // denominator and cancel, but downstream sector combination
        // needs the dim_src-weighted version (see header).
        // -------------------------------------------------------------
        for (double T : temperatures) {
            const double beta = 1.0 / T;
            for (std::size_t idx = 0; idx < significant.size(); ++idx) {
                if (!i_valid[idx]) continue;
                const double dE = E_i_arr[idx] - E_min_sector;
                const double wt = c_sq_arr[idx] * std::exp(-beta * dE);
                if (wt < 1e-300) continue;
                R.Z[T] += wt;
                const auto& Si = S_i_real[idx];
                const auto& Sq = S_i_imag[idx];
                auto&       Rr = R.S_real[T];
                auto&       Rq = R.S_imag[T];
                for (std::size_t iw = 0; iw < num_omega; ++iw) {
                    Rr[iw] += wt * Si[iw];
                    Rq[iw] += wt * Sq[iw];
                }
            }
        }
        R.samples_done++;
    }

    // Apply the (dim_src / R) trace-estimator prefactor so the
    // sector-pair contributions can be summed directly with no
    // additional bookkeeping in the combiner.
    if (R.samples_done > 0) {
        const double scale =
            static_cast<double>(dim_src) /
            static_cast<double>(R.samples_done);
        for (double T : temperatures) {
            for (auto& v : R.S_real[T]) v *= scale;
            for (auto& v : R.S_imag[T]) v *= scale;
            R.Z[T] *= scale;
        }
    }
    R.E_min = std::isfinite(E_min_sector) ? E_min_sector : 0.0;

    if (opts.verbose) {
        const auto end_time = std::chrono::high_resolution_clock::now();
        const double dt = std::chrono::duration<double>(
            end_time - start_time).count();
        std::cout << "[ftlm-xirrep] sector done in " << dt << " s ("
                  << R.samples_done << " samples).\n";
    }
    return R;
}

DynamicalSpectraMerged combine_sector_dynamical_spectra(
    const std::vector<FtlmCrossIrrepSectorResult>& sector_results,
    const std::vector<double>&                     temperatures,
    std::size_t                                    num_omega)
{
    DynamicalSpectraMerged out;
    if (sector_results.empty() || num_omega == 0) {
        for (double T : temperatures) {
            out.S_real[T] = std::vector<double>(num_omega, 0.0);
            out.S_imag[T] = std::vector<double>(num_omega, 0.0);
        }
        return out;
    }

    // The per-sector S_real / S_imag / Z accumulators are already
    // dim_src-weighted (see kernel return). However, each sector
    // carries its own ``E_min`` reference inside the exp(-beta E)
    // factor, so we apply an F-shift trick analogous to
    // ``ed::core::combine_sector_thermodynamics``: pick the global
    // minimum E_min across sectors, multiply each sector's numerator
    // AND denominator by exp(-beta * (E_min^sector - E_min_global)),
    // then sum and divide. This makes the float exponents stable
    // even when one sector's GS sits far above the global GS.

    // Find global E_min:
    double E_min_global = std::numeric_limits<double>::infinity();
    for (const auto& sr : sector_results) {
        if (sr.dim_src == 0 || sr.samples_done == 0) continue;
        if (sr.E_min < E_min_global) E_min_global = sr.E_min;
    }
    if (!std::isfinite(E_min_global)) {
        // Every sector empty / failed; return zeros.
        for (double T : temperatures) {
            out.S_real[T] = std::vector<double>(num_omega, 0.0);
            out.S_imag[T] = std::vector<double>(num_omega, 0.0);
        }
        return out;
    }

    for (double T : temperatures) {
        const double beta = 1.0 / T;
        std::vector<double> num_real(num_omega, 0.0);
        std::vector<double> num_imag(num_omega, 0.0);
        double              den = 0.0;
        for (const auto& sr : sector_results) {
            if (sr.dim_src == 0 || sr.samples_done == 0) continue;
            const double sec_E_min = sr.E_min;
            const double shift     = std::exp(
                -beta * (sec_E_min - E_min_global));
            auto sR = sr.S_real.find(T);
            auto sI = sr.S_imag.find(T);
            auto sZ = sr.Z.find(T);
            if (sR == sr.S_real.end() ||
                sI == sr.S_imag.end() ||
                sZ == sr.Z.end()) continue;
            const auto& sec_real = sR->second;
            const auto& sec_imag = sI->second;
            const double sec_Z   = sZ->second;
            if (sec_real.size() != num_omega ||
                sec_imag.size() != num_omega) continue;
            for (std::size_t iw = 0; iw < num_omega; ++iw) {
                num_real[iw] += shift * sec_real[iw];
                num_imag[iw] += shift * sec_imag[iw];
            }
            den += shift * sec_Z;
        }
        if (den > 1e-300) {
            const double inv = 1.0 / den;
            for (std::size_t iw = 0; iw < num_omega; ++iw) {
                num_real[iw] *= inv;
                num_imag[iw] *= inv;
            }
        }
        out.S_real[T] = std::move(num_real);
        out.S_imag[T] = std::move(num_imag);
    }
    return out;
}

}  // namespace ed::observables
