#pragma once
// =============================================================================
// include/ed/thermal/tpq_thermo.h -- canonical thermodynamics from mTPQ
// trajectories.
//
// A microcanonical TPQ trajectory psi_k = (L - H)^k psi_0 / ||.|| (psi_0 a
// unit random vector, L above the spectrum) records E_k = <psi_k|H|psi_k> and
// the growth factors n_k = ||(L - H) psi_{k-1}||. With Q_k = prod_{i<=k} n_i^2
// they are the moments of the start vector,
//
//   mu_{2k}   = <psi_0|(L - H)^{2k}|psi_0>   = Q_k,
//   mu_{2k+1} = <psi_0|(L - H)^{2k+1}|psi_0> = Q_k (L - E_k),
//
// and the canonical TPQ state follows from them at every beta (Sugiura and
// Shimizu, PRL 111, 010401 (2013)):
//
//   S_m(beta) = sum_j beta^j mu_{j+m} / j!      (= <psi_0|(L-H)^m e^{beta (L-H)}|psi_0>),
//   ln Z      = ln D - beta L + ln S_0,
//   E         = L - S_1 / S_0,
//   Var(H)    = S_2 / S_0 - (S_1 / S_0)^2,   C = beta^2 Var(H),
//
// with each S_m averaged over the samples first (E[<psi_0|A|psi_0>] = Tr A / D).
// Exact in expectation at every beta: no interpolation, no temperature grid in
// ln Z, no microcanonical variance bias. The sums run in log space. A beta is
// usable when every sample's series has decayed by its last term; colder
// targets are reported (the caller extends the run or refuses), never clamped.
// =============================================================================

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include <ed/thermal/curves.h>

namespace ed::thermal {

struct MtpqThermo {
    Curves curves;                     ///< ln Z, E, V on the requested betas (all of them)
    /// Indices of the betas whose series did not converge in some sample (too cold for the
    /// trajectory length); their values are not to be used.
    std::vector<std::size_t> unconverged;
};

namespace detail {

inline double log_sum_exp(const std::vector<double>& x, std::size_t n) {
    double mx = -std::numeric_limits<double>::infinity();
    for (std::size_t i = 0; i < n; ++i) mx = std::max(mx, x[i]);
    if (!std::isfinite(mx)) return mx;
    double s = 0.0;
    for (std::size_t i = 0; i < n; ++i) s += std::exp(x[i] - mx);
    return mx + std::log(s);
}

}  // namespace detail

/**
 * Canonical thermodynamics from mTPQ trajectories (see the file comment).
 *
 * @param sample_energies   per sample, E_k for k = 0..K
 * @param sample_log_norms  per sample, ln n_k for k = 1..K
 * @param L                 the shift the trajectories were run with (above the spectrum)
 * @param betas             target inverse temperatures (> 0)
 * @param dim               dimension D of the space the start vectors fill
 */
inline MtpqThermo mtpq_canonical_thermo(const std::vector<std::vector<double>>& sample_energies,
                                        const std::vector<std::vector<double>>& sample_log_norms, double L,
                                        const std::vector<double>& betas, double dim) {
    MtpqThermo out;
    Curves& c = out.curves;
    const std::size_t nT = betas.size(), R = sample_energies.size();
    c.lnZ.assign(nT, 0.0);
    c.E.assign(nT, 0.0);
    c.V.assign(nT, 0.0);
    if (nT == 0 || R == 0) return out;

    // ln mu_j per sample, j = 0..2K+1.
    std::vector<std::vector<double>> ln_mu(R);
    std::size_t j_max = 0;
    for (std::size_t r = 0; r < R; ++r) {
        const auto& E = sample_energies[r];
        const auto& n = sample_log_norms[r];
        const std::size_t K = n.size();
        if (E.size() != K + 1)
            throw std::invalid_argument("mtpq_canonical_thermo: a sample has " + std::to_string(E.size())
                                        + " energies for " + std::to_string(K) + " steps");
        auto& m = ln_mu[r];
        m.resize(2 * K + 2);
        double lnQ = 0.0;
        for (std::size_t k = 0; k <= K; ++k) {
            if (k > 0) lnQ += 2.0 * n[k - 1];
            if (!(E[k] < L))
                throw std::invalid_argument("mtpq_canonical_thermo: E_k >= L (L must exceed the spectrum)");
            m[2 * k] = lnQ;
            m[2 * k + 1] = lnQ + std::log(L - E[k]);
        }
        j_max = std::max(j_max, m.size());
    }
    std::vector<double> ln_fact(j_max + 1, 0.0);              // ln j!
    for (std::size_t j = 1; j <= j_max; ++j) ln_fact[j] = ln_fact[j - 1] + std::log(static_cast<double>(j));

    // A series has converged when its last term is below 1e-15 of the sum and the terms decrease.
    constexpr double kTailLog = -34.5;
    const double lnD = std::log(std::max(dim, 1.0));
    std::vector<double> terms(j_max), per_sample(R);
    for (std::size_t t = 0; t < nT; ++t) {
        const double beta = betas[t];
        if (!(beta > 0.0)) throw std::invalid_argument("mtpq_canonical_thermo: betas must be positive");
        const double ln_beta = std::log(beta);
        double lnS[3];
        bool converged = true;
        for (int m = 0; m < 3; ++m) {
            for (std::size_t r = 0; r < R; ++r) {
                const auto& mu = ln_mu[r];
                const std::size_t n_terms = mu.size() - static_cast<std::size_t>(m);
                for (std::size_t j = 0; j < n_terms; ++j)
                    terms[j] = static_cast<double>(j) * ln_beta + mu[j + static_cast<std::size_t>(m)] - ln_fact[j];
                const double s = detail::log_sum_exp(terms, n_terms);
                if (n_terms >= 2 && (terms[n_terms - 1] - s > kTailLog || terms[n_terms - 1] > terms[n_terms - 2]))
                    converged = false;
                per_sample[r] = s;
            }
            lnS[m] = detail::log_sum_exp(per_sample, R) - std::log(static_cast<double>(R));
        }
        if (!converged) out.unconverged.push_back(t);
        const double a = std::exp(lnS[1] - lnS[0]);          // L - E
        const double var = std::max(std::exp(lnS[2] - lnS[0]) - a * a, 0.0);
        c.lnZ[t] = lnD - beta * L + lnS[0];
        c.E[t] = L - a;
        c.V[t] = var;
    }
    return out;
}

/**
 * Canonical <O>(beta) from mTPQ trajectories and the observables' values along them (Sugiura and
 * Shimizu): <beta|O|beta> = sum_j beta^j nu_j / j! with nu_{2k} = Q_k m_k and nu_{2k+1} = Q_k n_{k+1} c_k,
 * where m_k = <psi_k|O|psi_k> and c_k = (<psi_k|O|psi_k+1> + <psi_k+1|O|psi_k>) / 2: the j-th power of
 * (L - H) split as evenly as it goes on both sides of O, exact for an O that commutes with H (the
 * standard approximation otherwise, accurate where the series is sharply peaked: large systems).
 * Normalised by the same series of O = 1 (nu_j = mu_j), both truncated at j = 2K and averaged over
 * the samples first. Each term is weighted relative to mu_j (the H moments), so signed values never
 * enter a logarithm: o_{2k} = m_k, o_{2k+1} = n_{k+1} c_k / (L - E_k).
 *
 * @return out[o][t] for every observable and beta
 */
inline std::vector<std::vector<std::complex<double>>> mtpq_canonical_observables(
    const std::vector<std::vector<double>>& sample_energies, const std::vector<std::vector<double>>& sample_log_norms,
    const std::vector<std::vector<std::vector<std::complex<double>>>>& diag,
    const std::vector<std::vector<std::vector<std::complex<double>>>>& cross, std::size_t n_obs, double L,
    const std::vector<double>& betas) {
    using C = std::complex<double>;
    const std::size_t nT = betas.size(), R = sample_energies.size();
    std::vector<std::vector<C>> out(n_obs, std::vector<C>(nT, C(0.0, 0.0)));
    if (n_obs == 0 || R == 0) return out;
    // Per sample: ln mu_j (j = 0..2K) and the ratio o_j[o].
    std::vector<std::vector<double>> ln_mu(R);
    std::vector<std::vector<const std::vector<C>*>> num(R);
    std::vector<std::vector<double>> odd_factor(R);   // n_{k+1} / (L - E_k), for o_{2k+1}
    std::size_t j_max = 0;
    for (std::size_t r = 0; r < R; ++r) {
        const auto& E = sample_energies[r];
        const auto& n = sample_log_norms[r];
        const std::size_t K = n.size();
        if (diag.at(r).size() != K + 1 || cross.at(r).size() != K)
            throw std::invalid_argument("mtpq_canonical_observables: a sample's observable values do not match "
                                        "its steps");
        auto& m = ln_mu[r];
        m.resize(2 * K + 1);
        num[r].resize(2 * K + 1);
        odd_factor[r].assign(2 * K + 1, 1.0);
        double lnQ = 0.0;
        for (std::size_t k = 0; k <= K; ++k) {
            if (k > 0) lnQ += 2.0 * n[k - 1];
            m[2 * k] = lnQ;
            num[r][2 * k] = &diag[r][k];
            if (k < K) {
                m[2 * k + 1] = lnQ + std::log(L - E[k]);
                num[r][2 * k + 1] = &cross[r][k];
                odd_factor[r][2 * k + 1] = std::exp(n[k]) / (L - E[k]);
            }
        }
        j_max = std::max(j_max, m.size());
    }
    std::vector<double> ln_fact(j_max + 1, 0.0);
    for (std::size_t j = 1; j <= j_max; ++j) ln_fact[j] = ln_fact[j - 1] + std::log(static_cast<double>(j));
    for (std::size_t t = 0; t < nT; ++t) {
        const double ln_beta = std::log(betas[t]);
        double w_max = -std::numeric_limits<double>::infinity();
        for (std::size_t r = 0; r < R; ++r)
            for (std::size_t j = 0; j < ln_mu[r].size(); ++j)
                w_max = std::max(w_max, static_cast<double>(j) * ln_beta + ln_mu[r][j] - ln_fact[j]);
        double den = 0.0;
        std::vector<C> acc(n_obs, C(0.0, 0.0));
        for (std::size_t r = 0; r < R; ++r)
            for (std::size_t j = 0; j < ln_mu[r].size(); ++j) {
                const double w = std::exp(static_cast<double>(j) * ln_beta + ln_mu[r][j] - ln_fact[j] - w_max);
                den += w;
                const double f = w * odd_factor[r][j];
                const auto& v = *num[r][j];
                for (std::size_t o = 0; o < n_obs; ++o) acc[o] += f * v.at(o);
            }
        for (std::size_t o = 0; o < n_obs; ++o) out[o][t] = acc[o] / den;
    }
    return out;
}

/// The steps a trajectory needs for the canonical series to converge down to beta_max: its terms
/// beta^j mu_j / j! peak near j* = beta_max (L - E_min) with a width of about sqrt(j*).
inline std::size_t mtpq_steps_for(double beta_max, double L, double e_min) {
    const double j_star = std::max(0.0, beta_max * (L - e_min));
    return static_cast<std::size_t>(std::ceil(0.5 * (j_star + 8.0 * std::sqrt(j_star + 1.0) + 40.0)));
}

}  // namespace ed::thermal
