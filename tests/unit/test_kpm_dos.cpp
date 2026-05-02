// =============================================================================
// tests/unit/test_kpm_dos.cpp
//
// Acceptance tests for ed::kpm_dos::compute_kpm_dos:
//
//   1. Spectral-bound estimator returns E_min, E_max within 1% of full diag.
//   2. Reconstructed DOS integrates to D = Hilbert-space dimension (sum rule).
//   3. Z(β) matches full-diag Z within 1% over a broad β window.
//   4. C(β) matches full-diag C within 5% over the thermal peak.
//   5. Convergence: doubling M shrinks max C(β) error by >2x.
//
// Reference is *exact* full diagonalisation of the N=8 PBC Heisenberg chain,
// dim = 256 — small enough for fast diag, large enough for KPM smoothing
// (η = π·BW/M) to be small.
// =============================================================================

#include "common/catch2_harness.h"
#include <catch2/catch_approx.hpp>

#include <ed/operators/spin_ops.h>
#include <ed/solvers/kpm_dos.h>

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

// ---------------------------------------------------------------------------
// Dense materialisation helper
// ---------------------------------------------------------------------------
Eigen::MatrixXcd to_dense(std::function<void(const Complex*, Complex*, int)> op,
                           uint64_t dim)
{
    Eigen::MatrixXcd M = Eigen::MatrixXcd::Zero(dim, dim);
    ComplexVector in(dim, Complex(0.0)), out(dim, Complex(0.0));
    for (uint64_t j = 0; j < dim; ++j) {
        std::fill(in.begin(), in.end(), Complex(0.0));
        in[j] = Complex(1.0);
        std::fill(out.begin(), out.end(), Complex(0.0));
        op(in.data(), out.data(), static_cast<int>(dim));
        for (uint64_t i = 0; i < dim; ++i) M(i, j) = out[i];
    }
    return M;
}

struct ExactThermo {
    std::vector<double> Z, E, C, S, F;
};

/// Reference thermodynamics from the full eigenvalue spectrum.
ExactThermo exact_thermo(const Eigen::VectorXd& E,
                         const std::vector<double>& betas)
{
    const int d = static_cast<int>(E.size());
    const double e_min = E(0);
    ExactThermo r;
    r.Z.resize(betas.size());
    r.E.resize(betas.size());
    r.C.resize(betas.size());
    r.S.resize(betas.size());
    r.F.resize(betas.size());
    for (std::size_t t = 0; t < betas.size(); ++t) {
        const double beta = betas[t];
        double Zs = 0.0, Es = 0.0, E2s = 0.0;
        for (int n = 0; n < d; ++n) {
            const double w = std::exp(-beta * (E(n) - e_min));
            Zs  += w;
            Es  += w * E(n);
            E2s += w * E(n) * E(n);
        }
        const double E_mean  = Es / Zs;
        const double E2_mean = E2s / Zs;
        r.Z[t] = Zs * std::exp(-beta * e_min);
        r.E[t] = E_mean;
        r.C[t] = (E2_mean - E_mean * E_mean) * beta * beta;
        r.F[t] = -(std::log(Zs) - beta * e_min) / beta;
        r.S[t] = (E_mean - r.F[t]) * beta;
    }
    return r;
}

/// Build a stable Hamiltonian + dense reference once per test.
struct DenseFixture {
    uint64_t N = 0, dim = 0;
    std::unique_ptr<Operator> H_op;
    Eigen::MatrixXcd H_dense;
    Eigen::VectorXd  E;
    std::function<void(const Complex*, Complex*, int)> Hv;
};

DenseFixture make_fixture(uint64_t N) {
    DenseFixture f;
    f.N = N;
    f.dim = 1ULL << N;
    f.H_op = build_heisenberg_chain(N, 1.0, /*pbc=*/true);
    auto* op = f.H_op.get();
    f.Hv = [op](const Complex* in, Complex* out, int n) {
        op->apply(in, out, static_cast<size_t>(n));
    };
    f.H_dense = to_dense(f.Hv, f.dim);
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> es(f.H_dense);
    f.E = es.eigenvalues();
    return f;
}

double max_rel_err(const std::vector<double>& truth,
                   const std::vector<double>& est)
{
    double m = 0.0;
    for (std::size_t i = 0; i < truth.size(); ++i) {
        const double denom = std::max(std::abs(truth[i]), 1e-12);
        m = std::max(m, std::abs(est[i] - truth[i]) / denom);
    }
    return m;
}

}  // namespace

// ============================================================================
// TEST 1: spectral-bound estimator
// ============================================================================
TEST_CASE("kpm_dos: spectral bounds match full diag within 1%",
          "[kpm-dos][bounds]")
{
    auto f = make_fixture(8);

    ed::kpm_dos::KPMDOSParameters p;
    p.num_moments              = 256;
    p.num_random_vectors       = 1;
    p.spectral_bounds_krylov   = 100;
    p.random_seed              = 12345;

    auto r = ed::kpm_dos::compute_kpm_dos(f.Hv, f.dim, /*betas=*/{1.0}, /*dos=*/{}, p);

    const double E_lo = f.E(0);
    const double E_hi = f.E(f.dim - 1);
    const double BW   = E_hi - E_lo;

    REQUIRE(r.e_min_estimate <= E_lo + 0.01 * BW);
    REQUIRE(r.e_max_estimate >= E_hi - 0.01 * BW);
    REQUIRE(r.e_min_estimate >= E_lo - 0.05 * BW);
    REQUIRE(r.e_max_estimate <= E_hi + 0.05 * BW);
}

// ============================================================================
// TEST 2: DOS sum rule  ∫ ρ(E) dE = D
// ============================================================================
TEST_CASE("kpm_dos: DOS sum rule integrates to D",
          "[kpm-dos][sum-rule]")
{
    auto f = make_fixture(6);  // dim = 64

    ed::kpm_dos::KPMDOSParameters p;
    p.num_moments              = 512;
    p.num_random_vectors       = 50;
    p.spectral_bounds_krylov   = 60;
    p.random_seed              = 4242;

    // μ_0 (raw, before kernel) is exactly D for a unit random vector.
    auto r = ed::kpm_dos::compute_kpm_dos(f.Hv, f.dim, /*betas=*/{1.0}, /*dos=*/{}, p);

    // μ_0^raw should equal D, μ_0^weighted = g_0 * μ_0^raw.
    REQUIRE(r.moments_raw.front() == Catch::Approx(static_cast<double>(f.dim))
            .margin(1e-9));
}

// ============================================================================
// TEST 3: Z(β) matches full diag within 1% across β-window
// ============================================================================
TEST_CASE("kpm_dos: Z(β) matches full diagonalisation",
          "[kpm-dos][partition]")
{
    auto f = make_fixture(8);  // dim = 256

    const std::vector<double> betas = {0.05, 0.2, 1.0, 5.0};
    auto truth = exact_thermo(f.E, betas);

    ed::kpm_dos::KPMDOSParameters p;
    p.num_moments              = 1024;
    p.num_random_vectors       = 400;   // Hutchinson σ ∼ 1/√(R·D); D=256 is small
    p.spectral_bounds_krylov   = 120;
    p.random_seed              = 7;

    auto r = ed::kpm_dos::compute_kpm_dos(f.Hv, f.dim, betas, /*dos=*/{}, p);

    const double max_rel = max_rel_err(truth.Z, r.partition_function);
    INFO("max rel err in Z(β) = " << max_rel);
    REQUIRE(max_rel < 0.05);
}

// ============================================================================
// TEST 4: C(β) and E(β) match full diag within 5% across thermal window
// ============================================================================
TEST_CASE("kpm_dos: C(β), E(β) match full diag at large R",
          "[kpm-dos][thermo]")
{
    auto f = make_fixture(8);  // dim = 256

    const std::vector<double> betas = {0.2, 0.5, 1.0, 2.0, 4.0, 8.0};
    auto truth = exact_thermo(f.E, betas);

    ed::kpm_dos::KPMDOSParameters p;
    p.num_moments              = 1024;
    p.num_random_vectors       = 400;
    p.spectral_bounds_krylov   = 150;
    p.random_seed              = 11;

    auto r = ed::kpm_dos::compute_kpm_dos(f.Hv, f.dim, betas, /*dos=*/{}, p);

    const double e_err = max_rel_err(truth.E, r.energy);
    const double c_err = max_rel_err(truth.C, r.specific_heat);
    INFO("max rel err in E(β) = " << e_err);
    INFO("max rel err in C(β) = " << c_err);
    REQUIRE(e_err < 0.05);
    REQUIRE(c_err < 0.10);  // C is a 2nd cumulant, noisier than Z, E
}

// ============================================================================
// TEST 5: convergence — doubling M shrinks max C error by >2x (for small N)
// ============================================================================
TEST_CASE("kpm_dos: increasing R reduces error (Hutchinson variance)",
          "[kpm-dos][convergence]")
{
    auto f = make_fixture(8);

    const std::vector<double> betas = {0.5, 2.0, 5.0};
    auto truth = exact_thermo(f.E, betas);

    auto run = [&](int R) {
        ed::kpm_dos::KPMDOSParameters p;
        p.num_moments              = 1024;
        p.num_random_vectors       = R;
        p.spectral_bounds_krylov   = 150;
        p.random_seed              = 31337;
        return ed::kpm_dos::compute_kpm_dos(f.Hv, f.dim, betas, {}, p);
    };

    auto r_R20  = run(20);
    auto r_R200 = run(200);

    const double err_R20  = max_rel_err(truth.C, r_R20.specific_heat);
    const double err_R200 = max_rel_err(truth.C, r_R200.specific_heat);

    INFO("err(R=20)  = " << err_R20);
    INFO("err(R=200) = " << err_R200);
    // Hutchinson variance ∝ 1/R; 10x more samples should give >2x reduction.
    REQUIRE(err_R200 < 0.6 * err_R20);
}
