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
#include <ed/matvec/backend.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace ed::thermal {

using Complex = std::complex<double>;
using ComplexVector = std::vector<Complex>;

namespace {

// One random sample's Lanczos spectrum: Ritz values eps~_j and weights
// w_j = |<r~|psi_j>|^2 (r~ normalized). Shared e_min shift is applied at combine. With observables,
// obs[t][o] = <phi_t|O_o|phi_t> about `ref` (the sample's lowest kept Ritz value).
struct SampleSpectrum {
    std::vector<double> ritz;
    std::vector<double> weights;
    double ref = 0.0;
    std::vector<std::vector<Complex>> obs;
};

}  // namespace

Curves oftlm(const ed::matvec::Backend& be, const std::function<void(const Complex*, Complex*, std::size_t)>& apply_H,
             std::uint64_t N, const OftlmOptions& opts) {
    if (N == 0) throw std::invalid_argument("oftlm: N must be > 0");
    if (opts.betas.empty()) throw std::invalid_argument("oftlm: opts.betas must be non-empty");

    const std::size_t nT = opts.betas.size();
    const std::size_t R = std::max<std::size_t>(opts.num_samples, 1);
    const std::size_t M = std::max<std::size_t>(opts.krylov_dim, 2);
    const std::size_t n_obs = opts.n_observables;
    if (n_obs > 0 && !opts.observe) throw std::invalid_argument("oftlm: n_observables > 0 without observe");

    // seed == 0 == NONDETERMINISTIC (random_device), as for FTLM, so
    // independent default runs draw independent samples; explicit seeds
    // keep bit-reproducibility.
    const std::uint64_t base_seed = ed::thermal::resolve_base_seed(opts.random_seed);

    // -------------------------------------------------------------------------
    // 1. The N_V exact eigenpairs, certified by the caller.
    // -------------------------------------------------------------------------
    const std::vector<double>& exact_eigs = opts.exact_values;
    const std::vector<ComplexVector>& exact_vecs = opts.exact_vectors;
    if (exact_vecs.size() != exact_eigs.size())
        throw std::invalid_argument("oftlm: " + std::to_string(exact_eigs.size()) + " exact values but "
                                    + std::to_string(exact_vecs.size()) + " exact vectors");
    for (const auto& v : exact_vecs)
        if (v.size() != N)
            throw std::invalid_argument("oftlm: an exact vector has length " + std::to_string(v.size()) + ", the block "
                                        + std::to_string(N));
    const std::size_t Nv = exact_vecs.size();
    if (Nv > N) throw std::invalid_argument("oftlm: more exact states than the block holds");

    // -------------------------------------------------------------------------
    // 2. R random samples, each orthogonalized against the N_V exact vectors.
    // -------------------------------------------------------------------------
    // The exact vectors in backend memory, uploaded once.
    std::vector<ed::matvec::Backend::UniqueVec> ev;
    ev.reserve(Nv);
    for (const auto& x : exact_vecs) {
        ev.push_back(be.make_zero_vector(N));
        be.copy_from_host(x.data(), ev.back().get(), N);
    }
    auto v = be.make_zero_vector(N);
    std::vector<SampleSpectrum> samples;
    samples.reserve(R);
    for (std::size_t s = 0; s < R; ++s) {
        std::mt19937 gen = ed::thermal::sample_engine(base_seed, s);

        // Drawn (and transformed) on the host, then on the backend.
        ComplexVector h = gaussian_vector(N, gen);
        if (opts.seed_transform) opts.seed_transform(h.data(), N);
        be.copy_from_host(h.data(), v.get(), N);
        // Gram-Schmidt against the exact eigenvectors, one at a time: v -= |i><i|v>.
        for (const auto& e : ev) be.axpy(-be.dot(e.get(), v.get(), N), e.get(), v.get(), N);
        const double vn = be.nrm2(v.get(), N);
        if (!(vn > 0.0)) continue;   // degenerate (v fell entirely in the exact span)
        be.scale(Complex(1.0 / vn, 0.0), v.get(), N);

        // Plain three-term recurrence, DELIBERATELY without
        // reorthogonalization: storing the M-vector basis needed for reorth
        // costs M*N*16 bytes (~50 GB at N = 2^25), and standard FTLM
        // practice (Jaklic-Prelovsek; Schnack-Richter-Steinigeweg PRR 2,
        // 013186) runs the stochastic samples bare -- ghost Ritz duplicates
        // redistribute the sample weight but leave the trace estimator
        // consistent. The run stops early when ||w|| <= opts.breakdown_tol.
        // With observables the basis is kept and fully reorthogonalised (phi is built from it).
        ed::krylov::LanczosKernelOptions lopts;
        lopts.max_iter = static_cast<std::size_t>(std::min<std::uint64_t>(N, M));
        lopts.reorth = n_obs > 0 ? ed::krylov::ReorthPolicy::FullCGS2 : ed::krylov::ReorthPolicy::None;
        lopts.keep_basis = n_obs > 0;
        lopts.breakdown_tol = opts.breakdown_tol;
        auto lres = ed::krylov::lanczos_kernel(be, apply_H, static_cast<std::size_t>(N), v.get(), lopts);

        ed::krylov::TridiagEig t = ed::krylov::tridiag_eig(lres.alpha, lres.beta, lres.alpha.size(), /*vectors=*/true);
        SampleSpectrum sp;
        const std::vector<double> w = t.weights();
        for (std::size_t j = 0; j < w.size(); ++j) {
            if (!(w[j] < opts.min_weight)) {   // a roundoff copy outside the seed's subspace: dropped
                sp.weights.push_back(w[j]);
                sp.ritz.push_back(t.values[j]);
            } else {
                t.vectors[j * t.m] = 0.0;   // and without weight in phi
            }
        }
        if (sp.ritz.empty()) continue;
        if (n_obs > 0) {
            const std::size_t m = t.m;
            if (lres.basis.size() < m) continue;   // the basis was not kept: the sample is dropped
            // phi_t = sum_a c_a v_a, c_a = sum_j g_j Y[j m + a], g_j = e^{-beta (eps_j - ref) / 2} Y[j m]
            // (FtlmOptions::observe), a few temperatures at a time.
            constexpr std::size_t kPhiChunk = 4;
            const std::size_t width = std::min(kPhiChunk, nT);
            sp.ref = sp.ritz.front();
            std::vector<const Complex*> V(m);
            for (std::size_t a = 0; a < m; ++a) V[a] = lres.basis[a];
            auto phi = be.make_zero_vector(N);
            std::vector<ComplexVector> host(width, ComplexVector(N));
            std::vector<Complex> c(m);
            sp.obs.assign(nT, {});
            for (std::size_t t0 = 0; t0 < nT; t0 += width) {
                const std::size_t nc = std::min(width, nT - t0);
                std::vector<const Complex*> ptrs(nc);
                for (std::size_t i = 0; i < nc; ++i) {
                    const double beta = opts.betas[t0 + i];
                    std::fill(c.begin(), c.end(), Complex(0, 0));
                    for (std::size_t j = 0; j < m; ++j) {
                        const double g = std::exp(-0.5 * beta * (t.values[j] - sp.ref)) * t.vectors[j * m];
                        for (std::size_t a = 0; a < m; ++a) c[a] += g * t.vectors[j * m + a];
                    }
                    be.scale(Complex(0, 0), phi.get(), N);
                    be.axpy_many(c.data(), V.data(), m, phi.get(), N);
                    be.copy_to_host(phi.get(), host[i].data(), N);
                    ptrs[i] = host[i].data();
                }
                auto vals = opts.observe(ptrs);
                if (vals.size() != nc) throw std::logic_error("oftlm: observe returned the wrong count");
                for (std::size_t i = 0; i < nc; ++i) sp.obs[t0 + i] = std::move(vals[i]);
            }
        }
        samples.push_back(std::move(sp));
    }
    // The exact states' own values, one sweep for all of them.
    std::vector<std::vector<Complex>> exact_obs;
    if (n_obs > 0 && Nv > 0) {
        std::vector<const Complex*> ptrs;
        ptrs.reserve(Nv);
        for (const auto& x : exact_vecs) ptrs.push_back(x.data());
        exact_obs = opts.observe(ptrs);
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

    // (D - N_V)/R prefactor for the stochastic (complement-space) part; D is the space the
    // random starts span (a spin tower's dimension when they are projected onto it).
    const double D = static_cast<double>(opts.trace_dim > 0 ? opts.trace_dim : N);
    const double pref = (R_eff > 0) ? (D - static_cast<double>(Nv)) / static_cast<double>(R_eff) : 0.0;

    // -------------------------------------------------------------------------
    // 4. Combine per beta.
    // -------------------------------------------------------------------------
    Curves out;
    out.e_min = e_min;   // the lowest certified eigenvalue, or the lowest Ritz value below it
    out.lnZ.assign(nT, 0.0);
    out.E.assign(nT, 0.0);
    out.V.assign(nT, 0.0);
    out.O.assign(n_obs, std::vector<Complex>(nT, Complex(0, 0)));

    for (std::size_t t = 0; t < nT; ++t) {
        const double beta = opts.betas[t];

        // Exact part.
        double Z = 0.0, EZ = 0.0, E2Z = 0.0;
        for (double e : exact_eigs) {
            const double bw = std::exp(-beta * (e - e_min));
            Z += bw;                         // moments about e_min: no E^2 cancellation
            EZ += (e - e_min) * bw;
            E2Z += (e - e_min) * (e - e_min) * bw;
        }

        // Stochastic part (complement space).
        double Zr = 0.0, EZr = 0.0, E2Zr = 0.0;
        for (const auto& sp : samples) {
            for (std::size_t j = 0; j < sp.ritz.size(); ++j) {
                const double e = sp.ritz[j];
                const double bw = sp.weights[j] * std::exp(-beta * (e - e_min));
                Zr += bw;
                EZr += (e - e_min) * bw;
                E2Zr += (e - e_min) * (e - e_min) * bw;
            }
        }
        Z += pref * Zr;
        EZ += pref * EZr;
        E2Z += pref * E2Zr;

        // Observables: the exact states' diagonal and each sample's phi value, against the same Z.
        for (std::size_t o = 0; o < n_obs && Z > 1e-300; ++o) {
            Complex num(0, 0);
            for (std::size_t i = 0; i < Nv; ++i) num += std::exp(-beta * (exact_eigs[i] - e_min)) * exact_obs[i].at(o);
            for (const auto& sp : samples) num += pref * std::exp(-beta * (sp.ref - e_min)) * sp.obs[t].at(o);
            out.O[o][t] = num / Z;
        }

        if (Z > 1e-300) {
            const double dE = EZ / Z;
            const double dE2 = E2Z / Z;
            // Z here is the full (shifted) trace Tr e^{-beta(H-e_min)}: ln Z_full = ln Z - beta e_min.
            out.lnZ[t] = std::log(Z) - beta * e_min;
            out.E[t] = e_min + dE;
            out.V[t] = std::max(dE2 - dE * dE, 0.0);
        } else {
            out.lnZ[t] = -beta * e_min;
            out.E[t] = e_min;
            out.V[t] = 0.0;
        }
    }

    return out;
}

}  // namespace ed::thermal
