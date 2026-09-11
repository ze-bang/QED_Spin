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

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <vector>

namespace ed::observables {

namespace {

constexpr double  kInvPi          = 0.3183098861837907;  // 1 / pi
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

    const auto start_time = std::chrono::high_resolution_clock::now();

    // ---------------------------------------------------------------------
    // Audit 2026-09: standard FTLM dynamical estimator (Jaklic & Prelovsek).
    //
    //   S(w) = (1/Z) sum_r sum_{i,j} e^{-beta eps_i} <r|psi_i> <psi_i|O^+|phi_j>
    //                                 <phi_j|O|r>  L_eta(w - (eps~_j - eps_i)),
    //   Z    = sum_r sum_i e^{-beta eps_i} |<r|psi_i>|^2,
    //
    // |psi_i> (eps_i) are the Ritz pairs of a Lanczos run on H_src started at
    // |r>, |phi_j> (eps~_j) those of a run on H_dst started at O|r>. The
    // estimator is exact in expectation over r for ANY block dimension.
    //
    // The previous body used, per source Ritz state, an inner Lanczos from
    // O|psi_i> and the weight |<r|psi_i>|^2 -- O(M^3 D) instead of O(M^2 D),
    // and BIASED: it drops the i != i' cross terms of O|r> = sum_i' O|psi_i'>
    // <psi_i'|r>, which only vanish as D -> infinity. Measured on the N = 8
    // Heisenberg ring at T = 1 against the dense Lehmann sum: peak 2x too high,
    // unchanged from 4 to 128 samples.
    //
    // Both Lanczos runs keep their basis and are FULLY reorthogonalised: the
    // Ritz vectors enter the overlap matrix explicitly, so ghost copies from
    // a local-reorth run would double-count weight.
    // ---------------------------------------------------------------------
    std::vector<ComplexVector> basis_S_scratch;
    const double eta    = opts.broadening;
    const double eta_sq = eta * eta;

    for (std::size_t sample_idx = 0; sample_idx < opts.num_samples; ++sample_idx) {
        if (opts.verbose) {
            std::cout << "[ftlm-xirrep] sample " << (sample_idx + 1)
                      << " / " << opts.num_samples
                      << "  (dim_src=" << dim_src
                      << ", dim_dst=" << dim_dst << ")\n";
        }
        std::mt19937 sample_gen(opts.random_seed + sample_idx * 12345ULL);
        ComplexVector r_state = generateGaussianRandomVector(
            static_cast<int>(dim_src), sample_gen);   // unit norm

        // ---- outer Lanczos on H_src from |r> ----
        std::vector<double> alpha_H, beta_H;
        basis_H_scratch.clear();
        const int H_iters = build_lanczos_tridiagonal_with_basis(
            H_src, r_state, static_cast<std::uint64_t>(dim_src),
            opts.krylov_dim, opts.tolerance,
            /*full_reorth=*/true, opts.reorth_frequency,
            alpha_H, beta_H, &basis_H_scratch);
        auto& basis_H = basis_H_scratch;
        if (H_iters == 0 || alpha_H.empty()) {
            if (opts.verbose) std::cout << "  outer Lanczos failed; skipping sample\n";
            continue;
        }
        const std::size_t m_H = alpha_H.size();
        if (basis_H.size() < m_H) {
            if (opts.verbose) std::cout << "  outer basis short; skipping sample\n";
            continue;
        }
        std::vector<double> ritz_values, dummy_weights, V_H;   // V_H[i*m_H + a]
        diagonalize_tridiagonal_ritz(alpha_H, beta_H, ritz_values, dummy_weights, &V_H);
        if (ritz_values.empty()) {
            if (opts.verbose) std::cout << "  outer diag failed; skipping sample\n";
            continue;
        }

        // ---- sector-wide E_min bookkeeping (rescale earlier accumulators) ----
        const double sample_E_min = *std::min_element(ritz_values.begin(), ritz_values.end());
        if (sample_E_min < E_min_sector) {
            if (std::isfinite(E_min_sector)) {
                for (double T : temperatures) {
                    const double beta  = 1.0 / T;
                    const double scale = std::exp(-beta * (E_min_sector - sample_E_min));
                    for (auto& v : R.S_real[T]) v *= scale;
                    for (auto& v : R.S_imag[T]) v *= scale;
                    R.Z[T] *= scale;
                }
            }
            E_min_sector = sample_E_min;
        }

        // ---- Z contribution: sum_i e^{-beta dE_i} c_i^2, c_i = <r|psi_i> = V_H[i,0] ----
        std::vector<double> c_i(m_H);
        for (std::size_t i = 0; i < m_H; ++i) c_i[i] = V_H[i * m_H + 0];
        for (double T : temperatures) {
            const double beta = 1.0 / T;
            double z = 0.0;
            for (std::size_t i = 0; i < m_H; ++i)
                z += c_i[i] * c_i[i] * std::exp(-beta * (ritz_values[i] - E_min_sector));
            R.Z[T] += z;
        }

        // ---- phi0 = O|r> in the target sector ----
        ComplexVector phi0(dim_dst, Complex(0.0, 0.0));
        O_apply(r_state.data(), phi0.data(), static_cast<int>(dim_dst));
        const double nphi = cblas_dznrm2(static_cast<int>(dim_dst), phi0.data(), 1);
        if (nphi < kPhiNormCutoff) {          // O annihilates |r>: no spectral weight
            R.samples_done++;
            continue;
        }
        {
            const Complex inv(1.0 / nphi, 0.0);
            cblas_zscal(static_cast<int>(dim_dst), &inv, phi0.data(), 1);
        }

        // ---- inner Lanczos on H_dst from O|r>/||O r|| ----
        std::vector<double> alpha_S, beta_S;
        basis_S_scratch.clear();
        build_lanczos_tridiagonal_with_basis(
            H_dst, phi0, static_cast<std::uint64_t>(dim_dst),
            opts.krylov_dim, opts.tolerance,
            /*full_reorth=*/true, opts.reorth_frequency,
            alpha_S, beta_S, &basis_S_scratch);
        auto& basis_S = basis_S_scratch;
        if (alpha_S.empty() || basis_S.size() < alpha_S.size()) {
            R.samples_done++;
            continue;
        }
        const std::size_t m_S = alpha_S.size();
        std::vector<double> ritz_S, dummy_S, V_S;              // V_S[j*m_S + b]
        diagonalize_tridiagonal_ritz(alpha_S, beta_S, ritz_S, dummy_S, &V_S);
        if (ritz_S.empty()) {
            R.samples_done++;
            continue;
        }

        // ---- W[a,b] = <O v_a | u_b>  (m_H x m_S), one zgemm ----
        //      A = [O v_0 ... O v_{m_H-1}] (dim_dst x m_H), B = [u_0 ... u_{m_S-1}]
        std::vector<Complex> A(static_cast<std::size_t>(dim_dst) * m_H);
        std::vector<Complex> B(static_cast<std::size_t>(dim_dst) * m_S);
        for (std::size_t a = 0; a < m_H; ++a)
            O_apply(basis_H[a].data(), A.data() + a * dim_dst, static_cast<int>(dim_dst));
        for (std::size_t b = 0; b < m_S; ++b)
            std::copy(basis_S[b].begin(), basis_S[b].begin() + static_cast<std::ptrdiff_t>(dim_dst),
                      B.begin() + static_cast<std::ptrdiff_t>(b * dim_dst));
        std::vector<Complex> W(m_H * m_S);                      // column-major: W[a + b*m_H]
        {
            const Complex one(1.0, 0.0), zero(0.0, 0.0);
            cblas_zgemm(CblasColMajor, CblasConjTrans, CblasNoTrans,
                        static_cast<int>(m_H), static_cast<int>(m_S), static_cast<int>(dim_dst),
                        &one, A.data(), static_cast<int>(dim_dst),
                        B.data(), static_cast<int>(dim_dst),
                        &zero, W.data(), static_cast<int>(m_H));
        }
        std::vector<Complex>().swap(A);
        std::vector<Complex>().swap(B);

        // ---- Obar[i,j] = sum_a V_H[i,a] W[a,b] V_S[j,b] ----
        std::vector<Complex> Tm(m_H * m_S, Complex(0.0, 0.0));  // Tm[i*m_S + b] = sum_a V_H[i,a] W[a,b]
        for (std::size_t i = 0; i < m_H; ++i)
            for (std::size_t b = 0; b < m_S; ++b) {
                Complex acc(0.0, 0.0);
                for (std::size_t a = 0; a < m_H; ++a) acc += V_H[i * m_H + a] * W[a + b * m_H];
                Tm[i * m_S + b] = acc;
            }
        // ---- per-source-Ritz spectral rows s_i(w) = sum_j Re/Im(w_ij) L_eta(w - E_ij) ----
        //      w_ij = c_i * Obar[i,j] * ||O r|| * V_S[j,0]
        std::vector<double> s_re(m_H * num_omega, 0.0), s_im(m_H * num_omega, 0.0);
        #pragma omp parallel for schedule(static)
        for (std::int64_t ii = 0; ii < static_cast<std::int64_t>(m_H); ++ii) {
            const std::size_t i = static_cast<std::size_t>(ii);
            for (std::size_t j = 0; j < m_S; ++j) {
                Complex obar(0.0, 0.0);
                for (std::size_t b = 0; b < m_S; ++b) obar += Tm[i * m_S + b] * V_S[j * m_S + b];
                const Complex w_ij = c_i[i] * obar * (nphi * V_S[j * m_S + 0]);
                if (std::abs(w_ij) < 1e-300) continue;
                const double E_ij = ritz_S[j] - ritz_values[i];
                for (std::size_t iw = 0; iw < num_omega; ++iw) {
                    const double d   = omega_grid[iw] - E_ij;
                    const double lor = (eta * kInvPi) / (d * d + eta_sq);
                    s_re[i * num_omega + iw] += w_ij.real() * lor;
                    s_im[i * num_omega + iw] += w_ij.imag() * lor;
                }
            }
        }
        // ---- thermal accumulation ----
        for (double T : temperatures) {
            const double beta = 1.0 / T;
            auto& Rr = R.S_real[T];
            auto& Rq = R.S_imag[T];
            for (std::size_t i = 0; i < m_H; ++i) {
                const double wt = std::exp(-beta * (ritz_values[i] - E_min_sector));
                if (wt < 1e-300) continue;
                for (std::size_t iw = 0; iw < num_omega; ++iw) {
                    Rr[iw] += wt * s_re[i * num_omega + iw];
                    Rq[iw] += wt * s_im[i * num_omega + iw];
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
