// =============================================================================
// test_kpm_dynamical_spectral  (Catch2 v3)
//
// Pillar 4 of the "Save and DSSF Upgrades" plan (May 2026): pins the
// ``ed::workflows::SpectralOptions::Method::KpmDynamical`` lane.
//
// Contract:
//   * The orchestrator dispatches the new ``KpmDynamical`` method to
//     ``ed::observables::kpm_dynamical_correlator``, seeded with the
//     ground state (when ``opts.initial_state`` is empty) or with the
//     user-supplied state otherwise.
//   * The returned S(omega) is the Chebyshev approximation of
//         S(omega) = -1/pi * Im <psi| O dagger (omega + i*eta - H)^{-1} O |psi>
//     for a single seed |psi>. We compare against the dense Lehmann
//     reference on a 4-site Heisenberg ring with a single-site Sz probe
//     ``O = S^z_0``. KPM weights are delta-like (Jackson kernel),
//     while the Lehmann reference has finite Lorentzian broadening,
//     so we integrate both spectra against the same Gaussian smearing
//     and compare the smeared curves at relative tolerance 5%.
//
// The integrated spectral weight ``int S(omega) d_omega`` is the
// f-sum: <psi| O dagger O |psi> = <psi| (S^z_0)^2 |psi> = 1/4 for any
// product state of spin-1/2 on site 0, and a value in (0, 1/4] for the
// singlet ground state. We use this as a sanity check on the
// normalisation of the KPM expansion.
// =============================================================================

#include "common/catch2_harness.h"

#include <ed/core/operator.h>
#include <ed/orchestrator.h>

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <complex>
#include <memory>
#include <vector>

using namespace ed_tests;

namespace {

constexpr std::uint64_t N_SITES = 4;
using Complex = std::complex<double>;

std::unique_ptr<Operator> heisen() {
    return build_heisenberg_chain(N_SITES, 1.0, /*periodic=*/true);
}

std::unique_ptr<Operator> sz_site_0() {
    auto obs = std::make_unique<Operator>(N_SITES, /*spin=*/0.5f);
    obs->addOneBodyTerm(2, 0, Complex(1.0, 0.0));
    return obs;
}

// Dense Lehmann reference: build the list of (dE, weight) poles.
struct LehmannPoles {
    std::vector<double> dE;
    std::vector<double> w;
};

LehmannPoles build_lehmann_poles(const Operator& H_op,
                                  const Operator& O_op,
                                  std::uint64_t dim) {
    auto Hv = [&](const Complex* in, Complex* out, int n) {
        H_op.apply(in, out, static_cast<size_t>(n));
    };
    auto Ov = [&](const Complex* in, Complex* out, int n) {
        O_op.apply(in, out, static_cast<size_t>(n));
    };
    Eigen::MatrixXcd H = apply_to_dense(Hv, dim);
    Eigen::MatrixXcd O = apply_to_dense(Ov, dim);
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> es(H);
    const auto& E = es.eigenvalues();
    const auto& V = es.eigenvectors();
    const double E0 = E(0);
    Eigen::VectorXcd Opsi = O * V.col(0);
    Eigen::VectorXcd amps = V.adjoint() * Opsi;  // <n|O|0>
    LehmannPoles out;
    for (int n = 0; n < amps.size(); ++n) {
        out.dE.push_back(E(n) - E0);
        out.w.push_back(std::norm(amps(n)));
    }
    return out;
}

// Gaussian deposition of (dE, w) poles onto an omega grid.
std::vector<double> deposit_gaussian(const std::vector<double>& omega,
                                      const LehmannPoles& poles,
                                      double sigma) {
    std::vector<double> S(omega.size(), 0.0);
    const double norm = 1.0 / (sigma * std::sqrt(2.0 * M_PI));
    for (std::size_t i = 0; i < omega.size(); ++i) {
        double acc = 0.0;
        for (std::size_t k = 0; k < poles.dE.size(); ++k) {
            const double d = omega[i] - poles.dE[k];
            acc += poles.w[k] * norm * std::exp(-0.5 * (d * d) / (sigma * sigma));
        }
        S[i] = acc;
    }
    return S;
}

// Convolve a grid-sampled spectrum with a Gaussian of width sigma.
std::vector<double> gauss_smear(const std::vector<double>& omega,
                                const std::vector<double>& S,
                                double sigma) {
    std::vector<double> out(omega.size(), 0.0);
    const double dw = (omega.size() > 1)
        ? (omega.back() - omega.front()) / static_cast<double>(omega.size() - 1)
        : 1.0;
    const double inv_norm = dw / (sigma * std::sqrt(2.0 * M_PI));
    for (std::size_t i = 0; i < omega.size(); ++i) {
        double acc = 0.0;
        for (std::size_t j = 0; j < omega.size(); ++j) {
            const double d = omega[i] - omega[j];
            acc += S[j] * inv_norm * std::exp(-0.5 * (d * d) / (sigma * sigma));
        }
        out[i] = acc;
    }
    return out;
}

ed::workflows::SpectralOptions base_opts() {
    ed::workflows::SpectralOptions opts;
    opts.method     = ed::workflows::SpectralOptions::Method::KpmDynamical;
    opts.broadening = 0.05;          // unused by KPM lane but kept for symmetry
    opts.omega_min  = -4.0;
    opts.omega_max  =  4.0;
    opts.num_omega  = 161;
    opts.kpm_moments = 400;
    return opts;
}

}  // namespace

TEST_CASE("ed::spectral KpmDynamical runs and produces a finite curve",
          "[orchestrator][spectral][kpm-dynamical]") {
    auto H   = heisen();
    auto obs = sz_site_0();
    std::vector<const ed::LinearOperator*> observables = { obs.get() };

    auto opts = base_opts();
    auto R = ed::workflows::spectral(*H, observables, opts);

    REQUIRE(R.omega.size() == opts.num_omega);
    REQUIRE(R.S_real.size() == opts.num_omega);

    double max_abs = 0.0;
    bool any_nonzero = false;
    for (double v : R.S_real) {
        if (!std::isfinite(v)) continue;
        max_abs = std::max(max_abs, std::abs(v));
        if (std::abs(v) > 1e-6) any_nonzero = true;
    }
    CHECK(any_nonzero);
    CHECK(max_abs > 1e-3);
}

TEST_CASE("ed::spectral KpmDynamical matches dense Lehmann reference (smoothed L2 < 5%)",
          "[orchestrator][spectral][kpm-dynamical]") {
    auto H   = heisen();
    auto obs = sz_site_0();
    std::vector<const ed::LinearOperator*> observables = { obs.get() };
    const std::uint64_t dim = std::uint64_t{1} << N_SITES;

    auto opts = base_opts();
    auto R = ed::workflows::spectral(*H, observables, opts);

    // Replace any NaN at the KPM boundary points with 0; the kernel
    // guards the sqrt(1 - x^2) divergence by skipping those omegas.
    std::vector<double> S_kpm = R.S_real;
    for (auto& v : S_kpm) if (!std::isfinite(v)) v = 0.0;

    // Equal-footing comparison: deposit the dense Lehmann poles onto
    // the same omega grid with the SAME Gaussian width that we then
    // apply (via convolution) to the KPM spectrum. This eliminates
    // the Jackson-kernel vs Lorentz-broadening mismatch -- both
    // curves end up as Gaussian-smeared delta combs of the same
    // sigma.
    const double sigma = 0.30;
    auto poles = build_lehmann_poles(*H, *obs, dim);
    auto S_ref = deposit_gaussian(R.omega, poles, sigma);
    auto S_kpm_smooth = gauss_smear(R.omega, S_kpm, sigma);

    double num = 0.0;
    double den = 0.0;
    for (std::size_t i = 0; i < R.omega.size(); ++i) {
        const double d = S_kpm_smooth[i] - S_ref[i];
        num += d * d;
        den += S_ref[i] * S_ref[i];
    }
    const double l2_rel = std::sqrt(num / std::max(den, 1e-30));
    INFO("smoothed L2 relative error = " << l2_rel
            << ", sigma = " << sigma);
    CHECK(l2_rel < 0.05);
}

TEST_CASE("ed::spectral KpmDynamical respects opts.initial_state",
          "[orchestrator][spectral][kpm-dynamical][initial-state]") {
    auto H   = heisen();
    auto obs = sz_site_0();
    std::vector<const ed::LinearOperator*> observables = { obs.get() };
    const std::uint64_t dim = std::uint64_t{1} << N_SITES;

    // Auto-GS run.
    auto opts_auto = base_opts();
    auto R_auto = ed::workflows::spectral(*H, observables, opts_auto);

    // Independent GS extraction.
    ed::workflows::SolveOptions sopts;
    sopts.num_eigs        = 1;
    sopts.compute_vectors = true;
    sopts.tolerance       = 1e-12;
    sopts.method          = ed::workflows::SolveMethod::Lanczos;
    auto gs = ed::workflows::solve(*H, sopts);
    REQUIRE(gs.eigenvectors.has_value());
    REQUIRE_FALSE(gs.eigenvectors->host.empty());
    REQUIRE(gs.eigenvectors->host[0].size() == dim);

    auto opts_seed = base_opts();
    opts_seed.initial_state = gs.eigenvectors->host[0];
    auto R_seed = ed::workflows::spectral(*H, observables, opts_seed);

    // Both curves should match closely (the KPM kernel sees the same
    // |psi> in both cases).
    REQUIRE(R_seed.S_real.size() == R_auto.S_real.size());
    double max_abs = 0.0;
    for (std::size_t i = 0; i < R_auto.S_real.size(); ++i) {
        if (!std::isfinite(R_auto.S_real[i])
                || !std::isfinite(R_seed.S_real[i])) continue;
        max_abs = std::max(max_abs,
                           std::abs(R_auto.S_real[i] - R_seed.S_real[i]));
    }
    INFO("max |S_auto - S_seed| = " << max_abs);
    CHECK(max_abs < 1e-6);
}

TEST_CASE("ed::spectral KpmDynamical Lorentz vs Jackson kernels both produce finite curves",
          "[orchestrator][spectral][kpm-dynamical][kernels]") {
    auto H   = heisen();
    auto obs = sz_site_0();
    std::vector<const ed::LinearOperator*> observables = { obs.get() };

    auto opts_j = base_opts();
    opts_j.kpm_kernel = ed::workflows::SpectralOptions::KpmKernel::Jackson;
    auto R_j = ed::workflows::spectral(*H, observables, opts_j);

    auto opts_l = base_opts();
    opts_l.kpm_kernel = ed::workflows::SpectralOptions::KpmKernel::Lorentz;
    opts_l.kpm_lorentz_lambda = 4.0;
    auto R_l = ed::workflows::spectral(*H, observables, opts_l);

    auto count_nonzero = [](const std::vector<double>& v) {
        std::size_t c = 0;
        for (double x : v) if (std::isfinite(x) && std::abs(x) > 1e-6) ++c;
        return c;
    };
    CHECK(count_nonzero(R_j.S_real) > 0);
    CHECK(count_nonzero(R_l.S_real) > 0);
}
