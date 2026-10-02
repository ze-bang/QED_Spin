// =============================================================================
// src/engine/oftlm.cpp
//
// Orthogonalized Finite-Temperature Lanczos Method (OFTLM).
// See OftlmOptions in include/ed/thermal/ftlm.h for the estimator and references.
// =============================================================================

#include <ed/thermal/sample_seed.h>
#include <ed/thermal/ftlm.h>

#include <ed/krylov/lanczos.h>
#include <ed/krylov/tridiag.h>
#include <ed/matvec/cpu_backend.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <limits>
#include <random>
#include <stdexcept>
#include <vector>

namespace ed::thermal {

using Complex       = std::complex<double>;
using ComplexVector = std::vector<Complex>;

namespace {

// One random sample's Lanczos spectrum: Ritz values eps~_j and weights
// w_j = |<r~|psi_j>|^2 (r~ normalized). Shared e_min shift is applied at combine.
struct SampleSpectrum {
    std::vector<double> ritz;
    std::vector<double> weights;
};

double norm2(const ComplexVector& v) {
    double s = 0.0;
    for (const auto& c : v) s += std::norm(c);
    return s;
}

// Lanczos tridiagonal (and optionally the Krylov basis) of apply_H from v0 on
// the default CPU backend.
ed::krylov::LanczosKernelResult run_lanczos(
    const std::function<void(const Complex*, Complex*, int)>& apply_H,
    const ComplexVector& v0,
    std::uint64_t N,
    const ed::krylov::LanczosKernelOptions& opts)
{
    auto matvec = [&apply_H](const Complex* in, Complex* out, std::size_t n) {
        apply_H(in, out, static_cast<int>(n));
    };
    return ed::krylov::lanczos_kernel(
        ed::matvec::default_cpu_backend(), matvec,
        static_cast<std::size_t>(N), v0.data(), opts);
}

}  // namespace

Curves oftlm_cpu(
    const std::function<void(const Complex*, Complex*, int)>& apply_H,
    std::uint64_t          N,
    const OftlmOptions&    opts)
{
    if (N == 0)
        throw std::invalid_argument("oftlm_cpu: N must be > 0");
    if (opts.betas.empty())
        throw std::invalid_argument("oftlm_cpu: opts.betas must be non-empty");

    const std::size_t nT = opts.betas.size();
    const std::size_t R  = std::max<std::size_t>(opts.num_samples, 1);
    const std::size_t M  = std::max<std::size_t>(opts.krylov_dim, 2);
    std::size_t       Nv = std::min<std::uint64_t>(
        opts.num_exact, (N > 1 ? N - 1 : 0));

    // seed == 0 == NONDETERMINISTIC (random_device), as for FTLM, so
    // independent default runs draw independent samples; explicit seeds
    // keep bit-reproducibility.
    const std::uint64_t base_seed = ed::thermal::resolve_base_seed(opts.random_seed);

    // -------------------------------------------------------------------------
    // 1. N_V lowest exact eigenpairs via one long full-reorthogonalized Lanczos
    //    with the basis retained; reconstruct the Ritz vectors on the host.
    // -------------------------------------------------------------------------
    std::vector<double>        exact_eigs;   // eps_i, i < Nv (ascending)
    std::vector<ComplexVector> exact_vecs;   // |i>, i < Nv
    if (Nv > 0) {
        const std::uint64_t Mex = std::min<std::uint64_t>(
            std::max<std::uint64_t>(
                opts.exact_krylov ? opts.exact_krylov : (2 * Nv + 30),
                static_cast<std::uint64_t>(4)),
            N);

        std::mt19937 gen(static_cast<std::mt19937::result_type>(
            base_seed ^ 0x9E3779B97F4A7C15ULL));
        ComplexVector v0 = gaussian_vector(N, gen);
        const double n0 = std::sqrt(norm2(v0));
        if (n0 > 0.0) for (auto& c : v0) c /= n0;

        ed::krylov::LanczosKernelOptions lopts;
        lopts.max_iter   = static_cast<std::size_t>(Mex);
        lopts.reorth     = ed::krylov::ReorthPolicy::FullCGS2;
        lopts.keep_basis = true;
        auto lres = run_lanczos(apply_H, v0, N, lopts);
        const auto& basis = lres.basis;

        const ed::krylov::TridiagEig t =
            ed::krylov::tridiag_eig(lres.alpha, lres.beta, lres.alpha.size(), /*vectors=*/true);

        const std::size_t m = t.m;
        Nv = std::min<std::size_t>(Nv, m);
        exact_eigs.reserve(Nv);
        exact_vecs.reserve(Nv);
        for (std::size_t i = 0; i < Nv; ++i) {
            exact_eigs.push_back(t.values[i]);
            // |i> = sum_k basis[k] * z(k, i)   (Ritz vector, column i)
            ComplexVector vi(N, Complex(0.0, 0.0));
            const std::size_t kmax = std::min<std::size_t>(m, basis.size());
            for (std::size_t k = 0; k < kmax; ++k) {
                const double ck = t.z(k, i);
                const Complex* bk = basis[k].get();
                for (std::uint64_t n = 0; n < N; ++n) vi[n] += ck * bk[n];
            }
            const double vn = std::sqrt(norm2(vi));
            if (vn > 0.0) for (auto& c : vi) c /= vn;
            exact_vecs.push_back(std::move(vi));
        }
    }
    Nv = exact_vecs.size();

    // -------------------------------------------------------------------------
    // 2. R random samples, each orthogonalized against the N_V exact vectors.
    // -------------------------------------------------------------------------
    std::vector<SampleSpectrum> samples;
    samples.reserve(R);
    for (std::size_t s = 0; s < R; ++s) {
        std::mt19937 gen = ed::thermal::sample_engine(base_seed, s);

        ComplexVector v = gaussian_vector(N, gen);
        // Gram-Schmidt against the exact eigenvectors: v -= sum_i |i><i|v>.
        for (const auto& ev : exact_vecs) {
            Complex ov(0.0, 0.0);
            for (std::uint64_t n = 0; n < N; ++n) ov += std::conj(ev[n]) * v[n];
            for (std::uint64_t n = 0; n < N; ++n) v[n] -= ov * ev[n];
        }
        const double vn = std::sqrt(norm2(v));
        if (!(vn > 0.0)) continue;   // degenerate (v fell entirely in the exact span)
        for (auto& c : v) c /= vn;

        // Plain three-term recurrence, DELIBERATELY without
        // reorthogonalization: storing the M-vector basis needed for reorth
        // costs M*N*16 bytes (~50 GB at N = 2^25), and standard FTLM
        // practice (Jaklic-Prelovsek; Schnack-Richter-Steinigeweg PRR 2,
        // 013186) runs the stochastic samples bare -- ghost Ritz duplicates
        // redistribute the sample weight but leave the trace estimator
        // consistent. The run stops early when ||w|| <= opts.breakdown_tol.
        ed::krylov::LanczosKernelOptions lopts;
        lopts.max_iter      = static_cast<std::size_t>(
            std::min<std::uint64_t>(N, M));
        lopts.reorth        = ed::krylov::ReorthPolicy::None;
        lopts.keep_basis    = false;
        lopts.breakdown_tol = opts.breakdown_tol;
        auto lres = run_lanczos(apply_H, v, N, lopts);

        ed::krylov::TridiagEig t =
            ed::krylov::tridiag_eig(lres.alpha, lres.beta, lres.alpha.size(), /*vectors=*/true);
        SampleSpectrum sp;
        sp.weights = t.weights();
        sp.ritz    = std::move(t.values);
        if (!sp.ritz.empty()) samples.push_back(std::move(sp));
    }

    const std::size_t R_eff = samples.size();

    // -------------------------------------------------------------------------
    // 3. Common ground-state shift e_min (numerical stability).
    // -------------------------------------------------------------------------
    double e_min = std::numeric_limits<double>::infinity();
    for (double e : exact_eigs) e_min = std::min(e_min, e);
    for (const auto& sp : samples)
        if (!sp.ritz.empty()) e_min = std::min(e_min, sp.ritz.front());
    if (!std::isfinite(e_min)) e_min = 0.0;

    // (D - N_V)/R prefactor for the stochastic (complement-space) part.
    const double pref = (R_eff > 0)
        ? (static_cast<double>(N) - static_cast<double>(Nv))
              / static_cast<double>(R_eff)
        : 0.0;

    // -------------------------------------------------------------------------
    // 4. Combine per beta.
    // -------------------------------------------------------------------------
    Curves out;
    out.lnZ.assign(nT, 0.0);
    out.E.assign(nT, 0.0);
    out.V.assign(nT, 0.0);

    for (std::size_t t = 0; t < nT; ++t) {
        const double beta = opts.betas[t];

        // Exact part.
        double Z = 0.0, EZ = 0.0, E2Z = 0.0;
        for (double e : exact_eigs) {
            const double bw = std::exp(-beta * (e - e_min));
            Z   += bw;                         // moments about e_min: no E^2 cancellation
            EZ  += (e - e_min) * bw;
            E2Z += (e - e_min) * (e - e_min) * bw;
        }

        // Stochastic part (complement space).
        double Zr = 0.0, EZr = 0.0, E2Zr = 0.0;
        for (const auto& sp : samples) {
            for (std::size_t j = 0; j < sp.ritz.size(); ++j) {
                const double e  = sp.ritz[j];
                const double bw = sp.weights[j] * std::exp(-beta * (e - e_min));
                Zr   += bw;
                EZr  += (e - e_min) * bw;
                E2Zr += (e - e_min) * (e - e_min) * bw;
            }
        }
        Z   += pref * Zr;
        EZ  += pref * EZr;
        E2Z += pref * E2Zr;

        if (Z > 1e-300) {
            const double dE  = EZ / Z;
            const double dE2 = E2Z / Z;
            // Z here is the full (shifted) trace Tr e^{-beta(H-e_min)}: ln Z_full = ln Z - beta e_min.
            out.lnZ[t] = std::log(Z) - beta * e_min;
            out.E[t]   = e_min + dE;
            out.V[t]   = std::max(dE2 - dE * dE, 0.0);
        } else {
            out.lnZ[t] = -beta * e_min;
            out.E[t]   = e_min;
            out.V[t]   = 0.0;
        }
    }

    return out;
}

}  // namespace ed::thermal
