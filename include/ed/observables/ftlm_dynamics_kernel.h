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
#include <ed/matvec/matvec_batcher.h>
#include <ed/observables/ftlm_cross_irrep_kernel.h>
#include <ed/solvers/lanczos.h>   // generateGaussianRandomVector, diagonalize_tridiagonal_ritz

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <random>
#include <stdexcept>
#include <vector>
#include <type_traits>

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

    // One sample: the source Ritz data and, when O reaches the target, the per-source-Ritz
    // spectral rows s_i(w). Samples are independent and are combined below in sample order.
    struct Sample {
        enum class Kind { Skip, ZOnly, Full } kind = Kind::Skip;
        std::vector<double> ritz, c;                             // source Ritz values, first components
        std::vector<double> s_re, s_im;                          // [i * nW + w]
    };
    auto sample = [&](auto& bk, auto&& Hs, auto&& Hd, std::size_t s) {
        Sample out;
        auto lanczos = [&](auto&& H, const Complex* v0, std::size_t n) {
            ed::krylov::LanczosKernelOptions lo;
            lo.max_iter   = std::min(n, opts.krylov_dim);
            lo.reorth     = ed::krylov::ReorthPolicy::FullCGS2;
            lo.keep_basis = true;
            auto mv = [&H](const Complex* in, Complex* o, std::size_t nn) { H(in, o, nn); };
            return ed::krylov::lanczos_kernel(bk, mv, n, v0, lo);
        };
        std::mt19937 gen(opts.random_seed + s * 12345ULL);
        ComplexVector r_host = generateGaussianRandomVector(static_cast<int>(dim_src), gen);
        if (opts.seed_transform) {
            opts.seed_transform(r_host.data(), dim_src);
            double n2 = 0.0;
            for (const auto& z : r_host) n2 += std::norm(z);
            if (n2 < 1e-24) return out;                           // no weight in the image: redraw
            const double inv = 1.0 / std::sqrt(n2);
            for (auto& z : r_host) z *= inv;
        }
        auto r = bk.make_zero_vector(dim_src);
        bk.copy_from_host(r_host.data(), r.get(), dim_src);

        auto kh = lanczos(Hs, r.get(), dim_src);
        if (kh.alpha.empty() || kh.basis.size() < kh.alpha.size()) return out;
        const std::size_t mH = kh.alpha.size();
        std::vector<double> wts, VH;                             // VH[i * mH + a]
        diagonalize_tridiagonal_ritz(kh.alpha, kh.beta, out.ritz, wts, &VH);
        if (out.ritz.empty()) return out;
        out.kind = Sample::Kind::ZOnly;
        out.c.resize(mH);
        for (std::size_t i = 0; i < mH; ++i) out.c[i] = VH[i * mH];

        auto phi = bk.make_zero_vector(dim_dst);
        O_apply(r.get(), phi.get(), dim_dst);
        const double nphi = bk.nrm2(phi.get(), dim_dst);
        if (nphi < 1e-14) return out;                             // O annihilates |r>: no spectral weight
        bk.scale(Complex(1.0 / nphi, 0.0), phi.get(), dim_dst);

        auto ks = lanczos(Hd, phi.get(), dim_dst);
        if (ks.alpha.empty() || ks.basis.size() < ks.alpha.size()) return out;
        const std::size_t mS = ks.alpha.size();
        std::vector<double> ritzS, wtsS, VS;                     // VS[j * mS + b]
        diagonalize_tridiagonal_ritz(ks.alpha, ks.beta, ritzS, wtsS, &VS);
        if (ritzS.empty()) return out;

        // W[a + b mH] = <O v_a | w_b> from one GEMM over backend-resident blocks.
        auto A = bk.make_zero_vector(dim_dst * mH);
        auto B = bk.make_zero_vector(dim_dst * mS);
        for (std::size_t a = 0; a < mH; ++a) O_apply(kh.basis[a].get(), A.get() + a * dim_dst, dim_dst);
        for (std::size_t b = 0; b < mS; ++b) bk.copy(ks.basis[b].get(), B.get() + b * dim_dst, dim_dst);
        auto Wd = bk.make_zero_vector(mH * mS);
        bk.gemm('C', 'N', mH, mS, dim_dst, Complex(1.0, 0.0), A.get(), dim_dst, B.get(), dim_dst,
                Complex(0.0, 0.0), Wd.get(), mH);
        std::vector<Complex> W(mH * mS);
        bk.copy_to_host(Wd.get(), W.data(), mH * mS);

        std::vector<Complex> Tm(mH * mS, Complex(0, 0));         // Tm[i mS + b] = sum_a VH[i,a] W[a,b]
        for (std::size_t i = 0; i < mH; ++i)
            for (std::size_t b = 0; b < mS; ++b) {
                Complex acc(0, 0);
                for (std::size_t a = 0; a < mH; ++a) acc += VH[i * mH + a] * W[a + b * mH];
                Tm[i * mS + b] = acc;
            }
        // Per-source-Ritz rows s_i(w) = sum_j w_ij L_eta(w - E_ij), w_ij = c_i Obar_ij |O r| VS[j,0].
        out.s_re.assign(mH * nW, 0.0);
        out.s_im.assign(mH * nW, 0.0);
        #pragma omp parallel for schedule(static)
        for (std::int64_t ii = 0; ii < static_cast<std::int64_t>(mH); ++ii) {
            const std::size_t i = static_cast<std::size_t>(ii);
            for (std::size_t j = 0; j < mS; ++j) {
                Complex obar(0, 0);
                for (std::size_t b = 0; b < mS; ++b) obar += Tm[i * mS + b] * VS[j * mS + b];
                const Complex w_ij = out.c[i] * obar * (nphi * VS[j * mS]);
                if (std::abs(w_ij) < 1e-300) continue;
                const double E_ij = ritzS[j] - out.ritz[i];
                for (std::size_t iw = 0; iw < nW; ++iw) {
                    const double d = omega[iw] - E_ij;
                    const double lor = (eta * kInvPi) / (d * d + eta * eta);
                    out.s_re[i * nW + iw] += w_ij.real() * lor;
                    out.s_im[i * nW + iw] += w_ij.imag() * lor;
                }
            }
        }
        out.kind = Sample::Kind::Full;
        return out;
    };

    std::vector<Sample> samples(opts.num_samples);
    bool batched = false;
#ifdef WITH_CUDA
    if constexpr (std::is_same_v<std::decay_t<Backend>, ed::matvec::CudaBackend>) {
        // Up to batch_width samples in lockstep, each on its own thread and backend, sharing every
        // source and target H apply (see MatvecBatcher); O is applied per sample.
        if (opts.batch_src && opts.batch_dst && opts.batch_width > 1 && opts.num_samples > 1) {
            batched = true;
            for (std::size_t s0 = 0; s0 < opts.num_samples; s0 += opts.batch_width) {
                const std::size_t k = std::min(opts.batch_width, opts.num_samples - s0);
                ed::matvec::MatvecBatcher b;
                const auto Hs = b.wrap(opts.batch_src);
                const auto Hd = b.wrap(opts.batch_dst);
                b.run(k, [&](std::size_t i) {
                    ed::matvec::CudaBackend bk;
                    samples[s0 + i] = sample(bk, Hs, Hd, s0 + i);
                });
            }
        }
    }
#endif
    if (!batched)
        for (std::size_t s = 0; s < opts.num_samples; ++s) samples[s] = sample(be, H_src, H_dst, s);

    for (const Sample& smp : samples) {
        if (smp.kind == Sample::Kind::Skip) continue;
        const std::size_t mH = smp.ritz.size();
        const double smin = *std::min_element(smp.ritz.begin(), smp.ritz.end());
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
        for (double T : temperatures) {
            const double beta = 1.0 / T;
            double z = 0.0;
            for (std::size_t i = 0; i < mH; ++i) z += smp.c[i] * smp.c[i] * std::exp(-beta * (smp.ritz[i] - E_min));
            R.Z[T] += z;
        }
        if (smp.kind == Sample::Kind::Full)
            for (double T : temperatures) {
                const double beta = 1.0 / T;
                auto& Rr = R.S_real[T];
                auto& Rq = R.S_imag[T];
                for (std::size_t i = 0; i < mH; ++i) {
                    const double wt = std::exp(-beta * (smp.ritz[i] - E_min));
                    if (wt < 1e-300) continue;
                    for (std::size_t iw = 0; iw < nW; ++iw) {
                        Rr[iw] += wt * smp.s_re[i * nW + iw];
                        Rq[iw] += wt * smp.s_im[i * nW + iw];
                    }
                }
            }
        R.samples_done++;
    }
    if (R.samples_done > 0) {
        const std::size_t tr = opts.trace_dim ? opts.trace_dim : dim_src;
        const double scale = static_cast<double>(tr) / static_cast<double>(R.samples_done);
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
