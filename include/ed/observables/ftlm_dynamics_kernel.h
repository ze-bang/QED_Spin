#pragma once
// =============================================================================
// include/ed/observables/ftlm_dynamics_kernel.h
//
// Finite-temperature Lanczos for a dynamical correlation between two sectors, on any
// Backend. The Jaklic-Prelovsek estimator of ftlm_cross_irrep_kernel_one_sector (same
// random vectors, same accumulation, same result struct), with every O(dim) step on the
// backend: both Lanczos runs keep their bases in backend memory, O is applied to each
// source Krylov vector there, and the overlaps W = (O V_H)^dag V_S come from one GEMM.
// Only the m_H x m_S overlap matrix and the tridiagonals reach the host.
//
//   H_src, H_dst, O_apply: (in, out, n) callables over backend pointers; O maps a source
//   vector (dim_src) to a target vector (dim_dst).
// =============================================================================

#include <ed/krylov/lanczos_kernel.h>
#include <ed/observables/ftlm_cross_irrep_kernel.h>
#include <ed/solvers/lanczos.h>   // generateGaussianRandomVector, diagonalize_tridiagonal_ritz

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <random>
#include <stdexcept>
#include <vector>

namespace ed::observables {

template <class Backend, class HSrc, class HDst, class OApply>
FtlmCrossIrrepSectorResult ftlm_dynamics_kernel(Backend& be, HSrc&& H_src, HDst&& H_dst, OApply&& O_apply,
                                                std::size_t dim_src, std::size_t dim_dst,
                                                const std::vector<double>& temperatures,
                                                const std::vector<double>& omega,
                                                const FtlmCrossIrrepOptions& opts) {
    if (dim_src == 0 || dim_dst == 0) throw std::invalid_argument("ftlm_dynamics_kernel: empty sector");
    if (temperatures.empty() || omega.empty() || opts.num_samples == 0)
        throw std::invalid_argument("ftlm_dynamics_kernel: no temperatures, frequencies or samples");
    constexpr double kInvPi = 0.3183098861837907;
    const std::size_t nW = omega.size();
    const double eta = opts.broadening;

    FtlmCrossIrrepSectorResult R;
    R.dim_src = dim_src;
    R.dim_dst = dim_dst;
    for (double T : temperatures) { R.S_real[T].assign(nW, 0.0); R.S_imag[T].assign(nW, 0.0); R.Z[T] = 0.0; }
    double E_min = std::numeric_limits<double>::infinity();

    auto lanczos = [&](auto&& H, const Complex* v0, std::size_t n) {
        ed::krylov::LanczosKernelOptions lo;
        lo.max_iter   = std::min(n, opts.krylov_dim);
        lo.reorth     = ed::krylov::ReorthPolicy::FullCGS2;
        lo.keep_basis = true;
        auto mv = [&H](const Complex* in, Complex* out, std::size_t nn) { H(in, out, nn); };
        return ed::krylov::lanczos_kernel(be, mv, n, v0, lo);
    };

    for (std::size_t s = 0; s < opts.num_samples; ++s) {
        std::mt19937 gen(opts.random_seed + s * 12345ULL);
        const ComplexVector r_host = generateGaussianRandomVector(static_cast<int>(dim_src), gen);
        auto r = be.make_zero_vector(dim_src);
        be.copy_from_host(r_host.data(), r.get(), dim_src);

        auto kh = lanczos(H_src, r.get(), dim_src);
        if (kh.alpha.empty() || kh.basis.size() < kh.alpha.size()) continue;
        const std::size_t mH = kh.alpha.size();
        std::vector<double> ritz, wts, VH;                       // VH[i * mH + a]
        diagonalize_tridiagonal_ritz(kh.alpha, kh.beta, ritz, wts, &VH);
        if (ritz.empty()) continue;

        const double smin = *std::min_element(ritz.begin(), ritz.end());
        if (smin < E_min) {
            if (std::isfinite(E_min))
                for (double T : temperatures) {
                    const double f = std::exp(-(1.0 / T) * (E_min - smin));
                    for (auto& v : R.S_real[T]) v *= f;
                    for (auto& v : R.S_imag[T]) v *= f;
                    R.Z[T] *= f;
                }
            E_min = smin;
        }
        std::vector<double> c(mH);
        for (std::size_t i = 0; i < mH; ++i) c[i] = VH[i * mH];
        for (double T : temperatures) {
            const double beta = 1.0 / T;
            double z = 0.0;
            for (std::size_t i = 0; i < mH; ++i) z += c[i] * c[i] * std::exp(-beta * (ritz[i] - E_min));
            R.Z[T] += z;
        }

        auto phi = be.make_zero_vector(dim_dst);
        O_apply(r.get(), phi.get(), dim_dst);
        const double nphi = be.nrm2(phi.get(), dim_dst);
        if (nphi < 1e-14) { R.samples_done++; continue; }  // O annihilates |r>: no spectral weight
        be.scale(Complex(1.0 / nphi, 0.0), phi.get(), dim_dst);

        auto ks = lanczos(H_dst, phi.get(), dim_dst);
        if (ks.alpha.empty() || ks.basis.size() < ks.alpha.size()) { R.samples_done++; continue; }
        const std::size_t mS = ks.alpha.size();
        std::vector<double> ritzS, wtsS, VS;                     // VS[j * mS + b]
        diagonalize_tridiagonal_ritz(ks.alpha, ks.beta, ritzS, wtsS, &VS);
        if (ritzS.empty()) { R.samples_done++; continue; }

        // W[a + b mH] = <O v_a | w_b> from one GEMM over backend-resident blocks.
        auto A = be.make_zero_vector(dim_dst * mH);
        auto B = be.make_zero_vector(dim_dst * mS);
        for (std::size_t a = 0; a < mH; ++a) O_apply(kh.basis[a].get(), A.get() + a * dim_dst, dim_dst);
        for (std::size_t b = 0; b < mS; ++b) be.copy(ks.basis[b].get(), B.get() + b * dim_dst, dim_dst);
        auto Wd = be.make_zero_vector(mH * mS);
        be.gemm('C', 'N', mH, mS, dim_dst, Complex(1.0, 0.0), A.get(), dim_dst, B.get(), dim_dst,
                Complex(0.0, 0.0), Wd.get(), mH);
        std::vector<Complex> W(mH * mS);
        be.copy_to_host(Wd.get(), W.data(), mH * mS);

        std::vector<Complex> Tm(mH * mS, Complex(0, 0));         // Tm[i mS + b] = sum_a VH[i,a] W[a,b]
        for (std::size_t i = 0; i < mH; ++i)
            for (std::size_t b = 0; b < mS; ++b) {
                Complex acc(0, 0);
                for (std::size_t a = 0; a < mH; ++a) acc += VH[i * mH + a] * W[a + b * mH];
                Tm[i * mS + b] = acc;
            }
        // Per-source-Ritz rows s_i(w) = sum_j w_ij L_eta(w - E_ij), w_ij = c_i Obar_ij |O r| VS[j,0].
        std::vector<double> s_re(mH * nW, 0.0), s_im(mH * nW, 0.0);
        #pragma omp parallel for schedule(static)
        for (std::int64_t ii = 0; ii < static_cast<std::int64_t>(mH); ++ii) {
            const std::size_t i = static_cast<std::size_t>(ii);
            for (std::size_t j = 0; j < mS; ++j) {
                Complex obar(0, 0);
                for (std::size_t b = 0; b < mS; ++b) obar += Tm[i * mS + b] * VS[j * mS + b];
                const Complex w_ij = c[i] * obar * (nphi * VS[j * mS]);
                if (std::abs(w_ij) < 1e-300) continue;
                const double E_ij = ritzS[j] - ritz[i];
                for (std::size_t iw = 0; iw < nW; ++iw) {
                    const double d = omega[iw] - E_ij;
                    const double lor = (eta * kInvPi) / (d * d + eta * eta);
                    s_re[i * nW + iw] += w_ij.real() * lor;
                    s_im[i * nW + iw] += w_ij.imag() * lor;
                }
            }
        }
        for (double T : temperatures) {
            const double beta = 1.0 / T;
            auto& Rr = R.S_real[T];
            auto& Rq = R.S_imag[T];
            for (std::size_t i = 0; i < mH; ++i) {
                const double wt = std::exp(-beta * (ritz[i] - E_min));
                if (wt < 1e-300) continue;
                for (std::size_t iw = 0; iw < nW; ++iw) {
                    Rr[iw] += wt * s_re[i * nW + iw];
                    Rq[iw] += wt * s_im[i * nW + iw];
                }
            }
        }
        R.samples_done++;
    }
    if (R.samples_done > 0) {
        const double scale = static_cast<double>(dim_src) / static_cast<double>(R.samples_done);
        for (double T : temperatures) {
            for (auto& v : R.S_real[T]) v *= scale;
            for (auto& v : R.S_imag[T]) v *= scale;
            R.Z[T] *= scale;
        }
    }
    R.E_min = std::isfinite(E_min) ? E_min : 0.0;
    return R;
}

}  // namespace ed::observables
