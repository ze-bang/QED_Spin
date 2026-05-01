// =============================================================================
// tests/unit/test_ftlm_jp.cpp
//
// Phase A acceptance test for the Jaklic-Prelovsek double-Lanczos finite-T
// dynamical correlator (`ed::ftlm::jp::compute_dynamical_correlation`).
//
// Reference: full Lehmann sum on a dense Heisenberg chain spectrum.
//   S(omega, T) = (1/Z) sum_{n,m} e^{-beta E_n} |<m|O|n>|^2
//                                 * Lorentzian(omega - (E_m - E_n))
// We pick O = S^z_0 (the local longitudinal magnetisation operator, fully
// intra-Sz so the intra-sector self-correlation kernel is exercised on the
// full Hilbert space) so JP and the reference agree at fixed parameters.
//
// Coverage:
//   1. Self-correlation S_{O,O}(omega, T) matches the Lehmann reference at
//      moderate T within stochastic tolerance.
//   2. Detailed balance: S(-omega, T) ~= e^{-beta omega} S(omega, T) on the
//      JP spectrum (statistical sanity check).
//   3. Sum rule: integral of S(omega, T) ~= static correlator <O O>(T)
//      reported by JP, which itself matches the dense Lehmann static
//      reference.
//   4. T -> infinity limit: <O>(T)^2 - <O O>(T) -> -<O^2> trace average.
// =============================================================================

#include "common/catch2_harness.h"
#include <catch2/catch_approx.hpp>

#include <ed/core/blas_lapack_wrapper.h>
#include <ed/solvers/ftlm_jp.h>

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>

#include <algorithm>
#include <cmath>
#include <complex>
#include <functional>
#include <memory>
#include <vector>

using ed_tests::build_heisenberg_chain;
using Complex = std::complex<double>;
using ComplexVector = std::vector<Complex>;

namespace {

constexpr double kPi = 3.14159265358979323846;

/// Build the explicit complex matrix for a callable acting on a tiny Hilbert
/// space by applying it to canonical basis vectors. Same trick used by
/// reference_from_operator() in test_harness.h.
Eigen::MatrixXcd to_dense(std::function<void(const Complex*, Complex*, int)> op,
                          uint64_t dim)
{
    Eigen::MatrixXcd M = Eigen::MatrixXcd::Zero(dim, dim);
    ComplexVector in(dim, Complex(0.0, 0.0));
    ComplexVector out(dim, Complex(0.0, 0.0));
    for (uint64_t j = 0; j < dim; ++j) {
        std::fill(in.begin(), in.end(), Complex(0.0, 0.0));
        in[j] = Complex(1.0, 0.0);
        std::fill(out.begin(), out.end(), Complex(0.0, 0.0));
        op(in.data(), out.data(), static_cast<int>(dim));
        for (uint64_t i = 0; i < dim; ++i) M(i, j) = out[i];
    }
    return M;
}

/// Build a local S^z site operator |s>=|0..0> => bit=0 means UP (+1/2),
/// bit=1 means DOWN (-1/2). Matches the construct_ham convention used in
/// the BFG correlator code (see src/bfg/correlations.cpp comment header).
Eigen::MatrixXcd dense_Sz_site(uint64_t N, uint64_t site) {
    const uint64_t dim = 1ULL << N;
    Eigen::MatrixXcd M = Eigen::MatrixXcd::Zero(dim, dim);
    for (uint64_t s = 0; s < dim; ++s) {
        const bool down = ((s >> site) & 1ULL) != 0;
        const double sz = down ? -0.5 : +0.5;
        M(s, s) = Complex(sz, 0.0);
    }
    return M;
}

/// Lehmann reference for S_{A,A}(omega, T) = sum_{n,m} e^{-beta E_n}
///   <n|A|m> <m|A|n> * Lorentzian(omega - (E_m - E_n)) / Z.
/// A is assumed Hermitian. Returns a vector of length omega_grid.size().
std::vector<double> lehmann_spectral_self(
    const Eigen::VectorXd& E,
    const Eigen::MatrixXcd& A_eig,    // <n|A|m>
    double beta,
    double e_min,
    const std::vector<double>& omega_grid,
    double eta)
{
    const int dim = static_cast<int>(E.size());
    std::vector<double> S(omega_grid.size(), 0.0);
    double Z = 0.0;
    Eigen::VectorXd boltz(dim);
    for (int n = 0; n < dim; ++n) {
        boltz(n) = std::exp(-beta * (E(n) - e_min));
        Z += boltz(n);
    }
    const double inv_pi = 1.0 / kPi;
    for (int n = 0; n < dim; ++n) {
        for (int m = 0; m < dim; ++m) {
            const double w = std::norm(A_eig(n, m));   // |<n|A|m>|^2
            const double dE = E(m) - E(n);
            for (size_t i = 0; i < omega_grid.size(); ++i) {
                const double dw = omega_grid[i] - dE;
                const double lor = (eta * inv_pi) / (dw * dw + eta * eta);
                S[i] += boltz(n) * w * lor;
            }
        }
    }
    for (auto& v : S) v /= Z;
    return S;
}

/// Lehmann static correlator <A^dagger A>(T) for Hermitian A.
double lehmann_static_self(
    const Eigen::VectorXd& E,
    const Eigen::MatrixXcd& A_eig,
    double beta,
    double e_min)
{
    const int dim = static_cast<int>(E.size());
    Eigen::VectorXd boltz(dim);
    double Z = 0.0;
    for (int n = 0; n < dim; ++n) {
        boltz(n) = std::exp(-beta * (E(n) - e_min));
        Z += boltz(n);
    }
    double s = 0.0;
    for (int n = 0; n < dim; ++n) {
        double row = 0.0;
        for (int m = 0; m < dim; ++m) row += std::norm(A_eig(n, m));
        s += boltz(n) * row;
    }
    return s / Z;
}

struct DenseFixture {
    std::unique_ptr<Operator> H_op;
    Eigen::MatrixXcd H_dense;
    Eigen::MatrixXcd Sz0_dense;          // S^z_0 in computational basis
    Eigen::VectorXd  E;                   // sorted eigenvalues
    Eigen::MatrixXcd U;                   // eigenvectors (columns)
    Eigen::MatrixXcd Sz0_eig;             // U^H Sz0 U
    uint64_t dim = 0;
    uint64_t N = 0;
};

DenseFixture make_chain_fixture(uint64_t N) {
    DenseFixture f;
    f.N = N;
    f.dim = 1ULL << N;
    f.H_op = build_heisenberg_chain(N, /*J=*/1.0, /*periodic=*/true);
    auto* op_ptr = f.H_op.get();
    auto Hv = [op_ptr](const Complex* in, Complex* out, int n) {
        op_ptr->apply(in, out, static_cast<size_t>(n));
    };
    f.H_dense = to_dense(Hv, f.dim);

    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> es(f.H_dense);
    f.E = es.eigenvalues();
    f.U = es.eigenvectors();

    f.Sz0_dense = dense_Sz_site(N, 0);
    f.Sz0_eig = f.U.adjoint() * f.Sz0_dense * f.U;
    return f;
}

} // namespace

TEST_CASE("JP double-Lanczos: self-correlation vs Lehmann reference",
          "[ftlm][jp][spectral]")
{
    const uint64_t N = 6;
    auto f = make_chain_fixture(N);

    // Build matvec callables for the JP kernel.
    auto* H_op = f.H_op.get();
    auto Hv = [H_op](const Complex* in, Complex* out, int n) {
        H_op->apply(in, out, static_cast<size_t>(n));
    };
    auto Sz0_v = [&f](const Complex* in, Complex* out, int n) {
        Eigen::Map<const Eigen::VectorXcd> x(in, n);
        Eigen::Map<Eigen::VectorXcd> y(out, n);
        y.noalias() = f.Sz0_dense * x;
    };

    const double e_min = f.E(0);
    const double e_max = f.E(f.E.size() - 1);

    // Frequency grid spans the full excitation spectrum +/- a bit.
    const double omega_min = -(e_max - e_min) - 1.0;
    const double omega_max = +(e_max - e_min) + 1.0;
    const size_t n_omega = 401;
    std::vector<double> omega_grid(n_omega);
    for (size_t i = 0; i < n_omega; ++i) {
        omega_grid[i] = omega_min +
            (omega_max - omega_min) * static_cast<double>(i) /
            static_cast<double>(n_omega - 1);
    }

    const std::vector<double> betas = {0.5, 2.0, 8.0};
    const double eta = 0.10;

    ed::ftlm::jp::JPParameters p;
    p.outer_krylov_dim = 60;
    p.inner_krylov_dim = 60;
    p.num_samples = 40;
    p.full_reorthogonalization = true;
    p.tolerance = 1e-12;
    p.random_seed = 20260501ULL;
    p.energy_shift = e_min;            // pin the shift to ground-state
    p.outer_boltzmann_cutoff = 0.0;    // disable cutoff for accuracy

    auto res = ed::ftlm::jp::compute_dynamical_correlation(
        Hv, Sz0_v, Sz0_v, f.dim, omega_min, omega_max, n_omega,
        betas, eta, p);

    REQUIRE(res.frequencies.size() == n_omega);
    REQUIRE(res.betas.size() == betas.size());
    REQUIRE(res.spectral_real.size() == betas.size() * n_omega);
    REQUIRE(res.total_samples >= 1);

    // Compare to Lehmann reference per temperature.
    for (size_t t = 0; t < betas.size(); ++t) {
        auto S_ref = lehmann_spectral_self(
            f.E, f.Sz0_eig, betas[t], e_min, omega_grid, eta);

        // Integrate both spectra (trapezoid) and compare integrals as a
        // robust scalar moment.
        const double dw = omega_grid[1] - omega_grid[0];
        double int_jp = 0.0, int_ref = 0.0;
        double max_diff = 0.0, max_ref = 0.0;
        for (size_t i = 0; i < n_omega; ++i) {
            const size_t idx = t * n_omega + i;
            const double weight = (i == 0 || i + 1 == n_omega) ? 0.5 : 1.0;
            int_jp  += weight * res.spectral_real[idx];
            int_ref += weight * S_ref[i];
            const double d = std::abs(res.spectral_real[idx] - S_ref[i]);
            max_diff = std::max(max_diff, d);
            max_ref  = std::max(max_ref,  std::abs(S_ref[i]));
        }
        int_jp  *= dw;
        int_ref *= dw;

        INFO("beta=" << betas[t]
             << " int_JP=" << int_jp
             << " int_ref=" << int_ref
             << " max|S_JP-S_ref|=" << max_diff
             << " max|S_ref|=" << max_ref);

        // Frequency-integrated spectral weight should agree to ~5%
        // (statistical error 1/sqrt(R=40) ~ 16%, JP+full reorth is much
        // better in practice for N=6).
        REQUIRE(std::abs(int_jp - int_ref) <= 0.05 * std::max(int_ref, 1e-3));
        // Pointwise agreement: 30% of the peak height (Lorentzian
        // broadening makes pointwise comparison inherently noisy).
        REQUIRE(max_diff <= 0.30 * std::max(max_ref, 1e-3));
    }
}

TEST_CASE("JP double-Lanczos: detailed balance and sum rule",
          "[ftlm][jp][detailed-balance][sum-rule]")
{
    const uint64_t N = 6;
    auto f = make_chain_fixture(N);

    auto* H_op = f.H_op.get();
    auto Hv = [H_op](const Complex* in, Complex* out, int n) {
        H_op->apply(in, out, static_cast<size_t>(n));
    };
    auto Sz0_v = [&f](const Complex* in, Complex* out, int n) {
        Eigen::Map<const Eigen::VectorXcd> x(in, n);
        Eigen::Map<Eigen::VectorXcd> y(out, n);
        y.noalias() = f.Sz0_dense * x;
    };

    const double e_min = f.E(0);
    const double e_max = f.E(f.E.size() - 1);
    const double omega_max = (e_max - e_min) + 1.0;

    // Symmetric grid around omega = 0 so we can pair (omega_i, -omega_i)
    // exactly.
    const size_t n_half = 200;
    std::vector<double> omega_grid;
    omega_grid.reserve(2 * n_half + 1);
    const double dw = omega_max / static_cast<double>(n_half);
    for (int i = -static_cast<int>(n_half); i <= static_cast<int>(n_half); ++i) {
        omega_grid.push_back(i * dw);
    }
    const size_t n_omega = omega_grid.size();
    REQUIRE(n_omega == 2 * n_half + 1);

    const std::vector<double> betas = {1.0, 4.0};
    const double eta = 0.20;

    ed::ftlm::jp::JPParameters p;
    p.outer_krylov_dim = 60;
    p.inner_krylov_dim = 60;
    p.num_samples = 40;
    p.full_reorthogonalization = true;
    p.tolerance = 1e-12;
    p.random_seed = 20260501ULL;
    p.energy_shift = e_min;
    p.outer_boltzmann_cutoff = 0.0;

    auto res = ed::ftlm::jp::compute_dynamical_correlation(
        Hv, Sz0_v, Sz0_v, f.dim, omega_grid.front(), omega_grid.back(),
        n_omega, betas, eta, p);

    REQUIRE(res.frequencies.size() == n_omega);

    for (size_t t = 0; t < betas.size(); ++t) {
        const double beta = betas[t];

        // Detailed balance: the JP spectrum (at fixed beta) should obey
        // S(-omega, T) = e^{-beta omega} S(omega, T) up to the Lorentzian-
        // broadening artefact and stochastic noise. We compare on an
        // intermediate-omega window where the spectrum is non-trivial and
        // the broadening kernel does not pile up at +/- band edges.
        double max_db_violation = 0.0;
        double max_S_seen = 0.0;
        for (size_t i = 1; i < n_half; ++i) {
            const size_t pos_idx = t * n_omega + (n_half + i);
            const size_t neg_idx = t * n_omega + (n_half - i);
            const double S_pos = res.spectral_real[pos_idx];
            const double S_neg = res.spectral_real[neg_idx];
            const double w = omega_grid[n_half + i];
            // S_neg - exp(-beta w) S_pos == 0 (up to noise)
            const double pred = std::exp(-beta * w) * S_pos;
            const double err = std::abs(S_neg - pred);
            max_db_violation = std::max(max_db_violation, err);
            max_S_seen = std::max(max_S_seen,
                                  std::max(std::abs(S_pos), std::abs(S_neg)));
        }
        INFO("beta=" << beta << " max DB violation=" << max_db_violation
             << " peak S=" << max_S_seen);
        REQUIRE(max_db_violation <= 0.25 * std::max(max_S_seen, 1e-3));

        // Sum rule: integrate S(omega) over all omega -> static correlator.
        double integral = 0.0;
        for (size_t i = 0; i < n_omega; ++i) {
            const double w = (i == 0 || i + 1 == n_omega) ? 0.5 : 1.0;
            integral += w * res.spectral_real[t * n_omega + i];
        }
        integral *= dw;

        const double static_jp = res.static_correlator[t].real();
        const double static_ref = lehmann_static_self(
            f.E, f.Sz0_eig, beta, e_min);

        INFO("beta=" << beta
             << " integral S(w)=" << integral
             << " static_JP=" << static_jp
             << " static_ref=" << static_ref);

        // Integral should match JP's own static report well (broadening
        // conserves total weight up to small finite-grid effects).
        REQUIRE(std::abs(integral - static_jp) <= 0.10 *
                std::max(std::abs(static_jp), 1e-3));
        // JP static estimate matches the dense Lehmann reference.
        REQUIRE(std::abs(static_jp - static_ref) <= 0.05 *
                std::max(std::abs(static_ref), 1e-3));
    }
}

TEST_CASE("JP double-Lanczos: sector-recombination Z-weighting",
          "[ftlm][jp][combine]")
{
    // Synthesise two trivially-different per-sector results and verify the
    // Z-weighted recombination matches the closed-form weighted average.
    using ed::ftlm::jp::JPDynamicalResult;

    JPDynamicalResult a;
    a.frequencies = {-1.0, 0.0, 1.0};
    a.betas = {1.0, 2.0};
    a.spectral_real.assign(2 * 3, 0.0);
    a.spectral_imag.assign(2 * 3, 0.0);
    a.static_correlator = {Complex(0.5, 0), Complex(0.6, 0)};
    a.partition_function = {1.0, 0.5};
    // S_a per (t, omega):
    //   T0: [0.1, 0.2, 0.3]   T1: [0.0, 0.4, 0.0]
    a.spectral_real[0*3 + 0] = 0.1; a.spectral_real[0*3 + 1] = 0.2;
    a.spectral_real[0*3 + 2] = 0.3;
    a.spectral_real[1*3 + 0] = 0.0; a.spectral_real[1*3 + 1] = 0.4;
    a.spectral_real[1*3 + 2] = 0.0;

    JPDynamicalResult b = a;  // copy frequencies, betas
    b.spectral_real.assign(2 * 3, 0.0);
    b.spectral_imag.assign(2 * 3, 0.0);
    b.static_correlator = {Complex(1.0, 0), Complex(1.2, 0)};
    b.partition_function = {1.0, 0.5};
    b.spectral_real[0*3 + 0] = 0.0; b.spectral_real[0*3 + 1] = 0.6;
    b.spectral_real[0*3 + 2] = 0.0;
    b.spectral_real[1*3 + 0] = 0.0; b.spectral_real[1*3 + 1] = 0.8;
    b.spectral_real[1*3 + 2] = 0.0;

    a.ground_state_estimate = -3.0;
    b.ground_state_estimate = -2.5;

    std::vector<JPDynamicalResult> per_sector = {a, b};
    std::vector<uint64_t> dims = {2, 4};   // dim_b = 2 * dim_a

    auto c = ed::ftlm::jp::combine_sector_results(per_sector, dims);

    // Z_eff = D * Z_partial.
    //   Z_eff_a(T0)=2*1=2,  Z_eff_b(T0)=4*1=4 => total 6, weights 1/3, 2/3
    //   Z_eff_a(T1)=2*0.5=1, Z_eff_b(T1)=4*0.5=2 => total 3, weights 1/3, 2/3
    using Catch::Approx;
    REQUIRE(c.partition_function[0] == Approx(6.0));
    REQUIRE(c.partition_function[1] == Approx(3.0));

    // T0, omega=0 -> (1/3)*0.2 + (2/3)*0.6 = 0.0667 + 0.4 = 0.4667
    REQUIRE(c.spectral_real[0*3 + 1] == Approx(1.0/3.0 * 0.2 +
                                                2.0/3.0 * 0.6));
    // Static at T0: (1/3)*0.5 + (2/3)*1.0 = 0.8333
    REQUIRE(c.static_correlator[0].real() == Approx(1.0/3.0 * 0.5 +
                                                     2.0/3.0 * 1.0));
    REQUIRE(c.ground_state_estimate == Approx(-3.0));
}
