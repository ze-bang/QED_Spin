// =============================================================================
// src/solvers/cpu/oftlm.cpp
//
// Orthogonalized Finite-Temperature Lanczos Method (OFTLM).
// See include/ed/thermal/oftlm_kernel.h for the estimator + references.
// =============================================================================

#include <ed/thermal/oftlm_kernel.h>

#include <ed/solvers/lanczos.h>   // generateGaussianRandomVector,
                                  // build_lanczos_tridiagonal_with_basis,
                                  // diagonalize_tridiagonal_ritz

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

}  // namespace

FtlmResult oftlm_cpu(
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

    const std::uint64_t base_seed =
        opts.random_seed != 0 ? opts.random_seed : 0xFEEDFACEULL;

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
        ComplexVector v0 = generateGaussianRandomVector(static_cast<int>(N), gen);
        const double n0 = std::sqrt(norm2(v0));
        if (n0 > 0.0) for (auto& c : v0) c /= n0;

        std::vector<double> alpha, beta;
        std::vector<ComplexVector> basis;
        build_lanczos_tridiagonal_with_basis(
            apply_H, v0, N, Mex, /*tol=*/1e-12,
            /*full_reorth=*/true, /*reorth_freq=*/1,
            alpha, beta, &basis);

        std::vector<double> ritz, weights, tri_evecs;   // tri_evecs: m*m col-major
        diagonalize_tridiagonal_ritz(alpha, beta, ritz, weights, &tri_evecs);

        const std::size_t m = ritz.size();
        Nv = std::min<std::size_t>(Nv, m);
        exact_eigs.reserve(Nv);
        exact_vecs.reserve(Nv);
        for (std::size_t i = 0; i < Nv; ++i) {
            exact_eigs.push_back(ritz[i]);
            // |i> = sum_k basis[k] * tri_evecs[k + i*m]   (Ritz vector, column i)
            ComplexVector vi(N, Complex(0.0, 0.0));
            const std::size_t kmax = std::min<std::size_t>(m, basis.size());
            for (std::size_t k = 0; k < kmax; ++k) {
                const double ck = tri_evecs[k + i * m];
                const ComplexVector& bk = basis[k];
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
        std::uint64_t z = base_seed + 0x9E3779B97F4A7C15ULL * (s + 1);
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
        z =  z ^ (z >> 31);
        std::mt19937 gen(static_cast<std::mt19937::result_type>(z));

        ComplexVector v = generateGaussianRandomVector(static_cast<int>(N), gen);
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
        // consistent. (Previously this passed full_reorth=true with a null
        // basis, which the legacy shim silently downgraded to None while
        // printing a spurious per-sample warning.)
        std::vector<double> alpha, beta;
        build_lanczos_tridiagonal_with_basis(
            apply_H, v, N, M, /*tol=*/1e-10,
            /*full_reorth=*/false, /*reorth_freq=*/0,
            alpha, beta, /*basis=*/nullptr);

        SampleSpectrum sp;
        diagonalize_tridiagonal_ritz(alpha, beta, sp.ritz, sp.weights);
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
    FtlmResult out;
    out.betas = opts.betas;
    out.partition_function.assign(nT, 0.0);
    out.energy.assign(nT, 0.0);
    out.heat_capacity.assign(nT, 0.0);
    out.entropy.assign(nT, 0.0);

    for (std::size_t t = 0; t < nT; ++t) {
        const double beta = opts.betas[t];

        // Exact part.
        double Z = 0.0, EZ = 0.0, E2Z = 0.0;
        for (double e : exact_eigs) {
            const double bw = std::exp(-beta * (e - e_min));
            Z   += bw;
            EZ  += e * bw;
            E2Z += e * e * bw;
        }

        // Stochastic part (complement space).
        double Zr = 0.0, EZr = 0.0, E2Zr = 0.0;
        for (const auto& sp : samples) {
            for (std::size_t j = 0; j < sp.ritz.size(); ++j) {
                const double e  = sp.ritz[j];
                const double bw = sp.weights[j] * std::exp(-beta * (e - e_min));
                Zr   += bw;
                EZr  += e * bw;
                E2Zr += e * e * bw;
            }
        }
        Z   += pref * Zr;
        EZ  += pref * EZr;
        E2Z += pref * E2Zr;

        out.partition_function[t] = Z;
        if (Z > 1e-300) {
            const double E   = EZ / Z;
            const double E2  = E2Z / Z;
            out.energy[t]        = E;
            out.heat_capacity[t] = beta * beta * (E2 - E * E);
            // Z here is the full (shifted) trace Tr e^{-beta(H-e_min)}, so
            // S = ln Z_true + beta<E> = ln Z + beta(<E> - e_min).
            out.entropy[t]       = std::log(Z) + beta * (E - e_min);
        } else {
            out.energy[t]        = e_min;
            out.heat_capacity[t] = 0.0;
            out.entropy[t]       = 0.0;
        }
    }

    return out;
}

}  // namespace ed::thermal
