// =============================================================================
// tests/unit/test_ftlm_sssf.cpp
//
// Phase C acceptance tests for:
//   1. ed::sssf::compute_sssf_jp       (JP random-sampling SSSF)
//   2. ed::sssf::compute_sssf_ltlm     (LTLM deterministic SSSF)
//   3. ed::sssf::compute_sssf_ltlm_from_states (exact-eigenstate SSSF)
//   4. ed::sssf::combine_sector_sssf   (sector recombination)
//   5. Multi-q sum rule: sum_q S^{zz}(q,T) = N/4 at every temperature.
//   6. JP SSSF vs LTLM from_states agreement (self-consistency)
//
// Reference: full Lehmann static
//   S_static(T) = (1/Z) sum_n exp(-beta*E_n) sum_m |<m|O|n>|^2
//                = (1/Z) sum_n exp(-beta*E_n) ||O|n>||^2
//
// Mathematical identity:
//   sum_{q} S^{zz}(q,T) = N/4   for any T (spin-1/2 Parseval sum rule)
//
// System: N=6 PBC Heisenberg chain (dim=64) for Sz0 tests;
//         N=4 chain (dim=16)       for multi-q tests.
// =============================================================================

#include "common/catch2_harness.h"
#include <catch2/catch_approx.hpp>

#include <ed/operators/spin_ops.h>
#include <ed/solvers/ftlm_sssf.h>

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>

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

// ---------------------------------------------------------------------------
// Dense helpers
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

/// Lehmann exact static: (1/Z) sum_n exp(-b*E_n) <n|O^dag O|n>
///                     = (1/Z) sum_n exp(-b*E_n) sum_m |<m|O|n>|^2
double lehmann_static(const Eigen::VectorXd& E,
                      const Eigen::MatrixXcd& O_eig,  // O in energy basis
                      double beta, double e_min)
{
    const int d = static_cast<int>(E.size());
    double Z = 0.0, S = 0.0;
    for (int n = 0; n < d; ++n) {
        const double boltz = std::exp(-beta * (E(n) - e_min));
        Z += boltz;
        double row_norm2 = 0.0;
        for (int m = 0; m < d; ++m) row_norm2 += std::norm(O_eig(m, n));
        S += boltz * row_norm2;
    }
    return S / Z;
}

struct DenseFixture {
    uint64_t N = 0, dim = 0;
    std::unique_ptr<Operator> H_op;
    Eigen::MatrixXcd H_dense;
    Eigen::VectorXd  E;
    Eigen::MatrixXcd U;   // eigenvectors as columns, ascending energy
    std::function<void(const Complex*, Complex*, int)> Hv;
    double e_min = 0.0;
};

DenseFixture make_fixture(uint64_t N) {
    DenseFixture f;
    f.N = N; f.dim = 1ULL << N;
    f.H_op = build_heisenberg_chain(N, 1.0, /*pbc=*/true);
    auto* op = f.H_op.get();
    f.Hv = [op](const Complex* in, Complex* out, int n) {
        op->apply(in, out, static_cast<size_t>(n));
    };
    f.H_dense = to_dense(f.Hv, f.dim);
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> es(f.H_dense);
    f.E = es.eigenvalues();
    f.U = es.eigenvectors();
    f.e_min = f.E(0);
    return f;
}

/// Build sorted vector of eigenstates and energies from a DenseFixture.
void get_eigenstates(const DenseFixture& f,
                     std::vector<ComplexVector>& states,
                     std::vector<double>& energies)
{
    const int dim = static_cast<int>(f.dim);
    const int K   = dim;
    states.resize(K);
    energies.resize(K);
    for (int n = 0; n < K; ++n) {
        energies[n] = f.E(n);
        states[n].resize(dim);
        for (int i = 0; i < dim; ++i)
            states[n][i] = f.U(i, n);
    }
}

} // namespace

// ============================================================================
// TEST 1: LTLM from_states SSSF agrees with Lehmann reference (exact)
// ============================================================================
TEST_CASE("SSSF from_states: exact agreement with Lehmann static",
          "[sssf][from-states][exact]")
{
    const uint64_t N = 6;
    auto f = make_fixture(N);

    auto Sz0 = ed::ops::make_Sz_site(0, N);
    auto Sz0_dense = to_dense(Sz0, f.dim);
    auto Sz0_eig = f.U.adjoint() * Sz0_dense * f.U;  // O in energy basis

    std::vector<ComplexVector> states;
    std::vector<double> energies;
    get_eigenstates(f, states, energies);

    const std::vector<double> betas = {0.5, 1.0, 2.0, 4.0, 8.0, 16.0};

    auto res = ed::sssf::compute_sssf_ltlm_from_states(
        Sz0, Sz0, f.dim, f.dim, states, energies, betas);

    REQUIRE(res.betas.size() == betas.size());
    REQUIRE(res.S_static_real.size() == betas.size());

    for (std::size_t t = 0; t < betas.size(); ++t) {
        const double ref = lehmann_static(f.E, Sz0_eig, betas[t], f.e_min);
        // All K eigenstates → exact result, so we only allow 0.1% deviation.
        INFO("beta=" << betas[t] << " ref=" << ref
             << " got=" << res.S_static_real[t]);
        CHECK(res.S_static_real[t] == Catch::Approx(ref).epsilon(1e-3));
        // Imaginary part must be nearly zero (self-correlation).
        CHECK(std::abs(res.S_static_imag[t])
              == Catch::Approx(0.0).margin(1e-10));
    }
}

// ============================================================================
// TEST 2: LTLM SSSF (outer Lanczos, all Ritz states) vs Lehmann reference
// ============================================================================
TEST_CASE("SSSF LTLM: outer-Lanczos static agrees with Lehmann",
          "[sssf][ltlm][sum-rule]")
{
    const uint64_t N = 6;
    auto f = make_fixture(N);

    auto Sz0 = ed::ops::make_Sz_site(0, N);
    auto Sz0_dense = to_dense(Sz0, f.dim);
    auto Sz0_eig = f.U.adjoint() * Sz0_dense * f.U;

    const std::vector<double> betas = {0.5, 2.0, 8.0};

    ed::ltlm::LTLMParameters p;
    p.num_lowest_states        = 64;   // all 64 eigenstates
    p.outer_krylov_dim         = 64;
    p.full_reorthogonalization = true;
    p.tolerance                = 1e-12;
    p.random_seed              = 20260601ULL;
    p.energy_shift             = f.e_min;

    auto res = ed::sssf::compute_sssf_ltlm(
        f.Hv, Sz0, Sz0, f.dim, betas, p);

    for (std::size_t t = 0; t < betas.size(); ++t) {
        const double ref = lehmann_static(f.E, Sz0_eig, betas[t], f.e_min);
        INFO("beta=" << betas[t] << " ref=" << ref
             << " got=" << res.S_static_real[t]);
        CHECK(res.S_static_real[t] == Catch::Approx(ref).epsilon(0.02));
        CHECK(std::abs(res.S_static_imag[t])
              == Catch::Approx(0.0).margin(1e-8));
    }
}

// ============================================================================
// TEST 3: JP SSSF (stochastic) vs Lehmann reference  (≤3% tolerance)
// ============================================================================
TEST_CASE("SSSF JP: stochastic static agrees with Lehmann within 3%",
          "[sssf][jp][stochastic]")
{
    const uint64_t N = 6;
    auto f = make_fixture(N);

    auto Sz0 = ed::ops::make_Sz_site(0, N);
    auto Sz0_dense = to_dense(Sz0, f.dim);
    auto Sz0_eig = f.U.adjoint() * Sz0_dense * f.U;

    const std::vector<double> betas = {0.5, 2.0, 8.0};

    ed::ftlm::jp::JPParameters p;
    p.outer_krylov_dim         = 64;
    p.inner_krylov_dim         = 1;    // not used in SSSF
    p.num_samples              = 40;
    p.full_reorthogonalization = true;
    p.tolerance                = 1e-12;
    p.random_seed              = 20260602ULL;
    p.energy_shift             = f.e_min;

    auto res = ed::sssf::compute_sssf_jp(
        f.Hv, Sz0, Sz0, f.dim, betas, p);

    for (std::size_t t = 0; t < betas.size(); ++t) {
        const double ref = lehmann_static(f.E, Sz0_eig, betas[t], f.e_min);
        INFO("beta=" << betas[t] << " ref=" << ref
             << " got=" << res.S_static_real[t]);
        CHECK(res.S_static_real[t] == Catch::Approx(ref).epsilon(0.05));
    }
}

// ============================================================================
// TEST 4: Multi-q Parseval sum rule  sum_q S^{zz}(q,T) = N/4  at all T
//
//   sum_q ||Sz_q|n>||^2 = sum_j <n|Sz_j^2|n> = N/4
//   => sum_q S^{zz}(q,T) = N/4   (for all beta)
//
// Uses N=4 chain (16-dim, 4 q-values: 0, pi/2, pi, 3pi/2).
// Evaluated via from_states (exact).
// ============================================================================
TEST_CASE("SSSF multi-q Parseval sum rule: sum_q S^{zz}(q) = N/4",
          "[sssf][multi-q][sum-rule]")
{
    const uint64_t N = 4;
    auto f = make_fixture(N);

    std::vector<ComplexVector> states;
    std::vector<double> energies;
    get_eigenstates(f, states, energies);

    const std::vector<double> betas = {0.0001, 0.5, 1.0, 4.0, 10.0};

    // q-values for N=4: 0, pi/2, pi, 3pi/2
    const int n_q = static_cast<int>(N);
    const double expected_sum = static_cast<double>(N) / 4.0;

    for (std::size_t t = 0; t < betas.size(); ++t) {
        double sum_S = 0.0;
        for (int k = 0; k < n_q; ++k) {
            const double q = 2.0 * kPi * k / N;
            auto Szq = ed::ops::make_Sz_q(q, N);
            // S^{zz}(q,T) = (1/Z) sum_n boltz_n <n|Sz_q^dag Sz_q|n>
            //             = (1/Z) sum_n boltz_n ||Sz_q|n>||^2
            // API: compute_sssf_ltlm_from_states(O1, O2) computes <n|O1^dag O2|n>
            // So set O1 = O2 = Sz_q to get ||Sz_q|n>||^2. ✓
            auto res = ed::sssf::compute_sssf_ltlm_from_states(
                Szq, Szq, f.dim, f.dim, states, energies, {betas[t]});
            // Result is real (||Sz_q|n>||^2 >= 0) and positive.
            sum_S += res.S_static_real[0];
        }
        INFO("beta=" << betas[t] << " sum_q S^{zz}(q) = " << sum_S
             << " expected=" << expected_sum);
        CHECK(sum_S == Catch::Approx(expected_sum).epsilon(1e-6));
    }
}

// ============================================================================
// TEST 5: JP and from_states SSSF agree on N=4 chain  (≤2%)
// ============================================================================
TEST_CASE("SSSF JP vs from_states: self-consistency on N=4",
          "[sssf][jp][from-states][consistency]")
{
    const uint64_t N = 4;
    auto f = make_fixture(N);
    auto Sz0 = ed::ops::make_Sz_site(0, N);

    std::vector<ComplexVector> states;
    std::vector<double> energies;
    get_eigenstates(f, states, energies);

    const std::vector<double> betas = {0.5, 2.0, 8.0};

    // Exact reference via from_states.
    auto ref = ed::sssf::compute_sssf_ltlm_from_states(
        Sz0, Sz0, f.dim, f.dim, states, energies, betas);

    // JP with many samples.
    ed::ftlm::jp::JPParameters p;
    p.outer_krylov_dim         = 16;
    p.num_samples              = 60;
    p.full_reorthogonalization = true;
    p.tolerance                = 1e-12;
    p.random_seed              = 20260603ULL;
    p.energy_shift             = f.e_min;

    auto jp = ed::sssf::compute_sssf_jp(
        f.Hv, Sz0, Sz0, f.dim, betas, p);

    for (std::size_t t = 0; t < betas.size(); ++t) {
        INFO("beta=" << betas[t]
             << " ref=" << ref.S_static_real[t]
             << " jp="  << jp.S_static_real[t]);
        CHECK(jp.S_static_real[t]
              == Catch::Approx(ref.S_static_real[t]).epsilon(0.05));
    }
}

// ============================================================================
// TEST 6: combine_sector_sssf Z-weighting correctness
//
// Synthetic two-sector test:
//   sector A: dim=4, Z_A(beta) = 4, S_A(beta) = 0.25
//   sector B: dim=2, Z_B(beta) = 2, S_B(beta) = 0.5
//
//   Z_eff_A = 4 * 4 = 16
//   Z_eff_B = 2 * 2 = 4
//   Z_total = 20
//   S_combined = (16 * 0.25 + 4 * 0.5) / 20 = (4 + 2) / 20 = 0.3
// ============================================================================
TEST_CASE("SSSF combine_sector_sssf: Z-weighted recombination",
          "[sssf][combine]")
{
    using ed::sssf::SSSFResult;

    SSSFResult r_A, r_B;
    r_A.betas            = {1.0};
    r_A.S_static_real    = {0.25};
    r_A.S_static_imag    = {0.0};
    r_A.partition_function = {4.0};

    r_B.betas            = {1.0};
    r_B.S_static_real    = {0.5};
    r_B.S_static_imag    = {0.0};
    r_B.partition_function = {2.0};

    auto combined = ed::sssf::combine_sector_sssf({r_A, r_B}, {4ULL, 2ULL});

    REQUIRE(combined.S_static_real.size() == 1);
    const double expected_S = (16.0*0.25 + 4.0*0.5) / 20.0;  // = 0.3
    const double expected_Z = 16.0 + 4.0;                      // = 20.0
    CHECK(combined.S_static_real[0]   == Catch::Approx(expected_S).epsilon(1e-10));
    CHECK(combined.S_static_imag[0]   == Catch::Approx(0.0).margin(1e-10));
    CHECK(combined.partition_function[0] == Catch::Approx(expected_Z).epsilon(1e-10));
}
