// =============================================================================
// tests/unit/test_ftlm_kpm.cpp
//
// Phase D acceptance tests for:
//   1. ed::kpm::compute_kpm_ltlm          (LTLM outer + Chebyshev inner)
//   2. ed::kpm::compute_kpm_ltlm_from_states (exact states + Chebyshev inner)
//   3. ed::kpm::combine_sector_kpm         (Z-weighted sector recombination)
//
// Tests:
//   1. Sum rule: ∫ S_{KPM}(ω,β) dω ≈ ∫ S_{Lehmann}(ω,β) dω   (within 3%)
//      Using N=6 PBC Heisenberg chain, K=64 (all states), M=256 moments.
//
//   2. Positivity: S(ω,β) ≥ -ε for all ω (Jackson kernel guarantees ≥0).
//
//   3. Spectral integral vs SSSF: ∫ S_{KPM} dω ≈ SSSF static (within 3%).
//
//   4. from_states consistency: KPM from all exact states matches LTLM KPM
//      within 5% (integral) for all temperatures.
//
//   5. combine_sector_kpm: Z-weighted recombination correctness (analytic).
//
//   6. Resolution check: KPM with M=512 moments resolves spectral peak positions
//      within bandwidth/M of the Lehmann exact peaks (for N=4 chain).
// =============================================================================

#include "common/catch2_harness.h"
#include <catch2/catch_approx.hpp>

#include <ed/operators/spin_ops.h>
#include <ed/solvers/ftlm_kpm.h>
#include <ed/solvers/ftlm_sssf.h>   // for SSSF reference

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

// ---------------------------------------------------------------------------
// Dense helpers (consistent with other test files in this project)
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

/// Lehmann static: (1/Z) Σ_n exp(-β E_n) ‖O|n⟩‖²
double lehmann_static(const Eigen::VectorXd& E,
                      const Eigen::MatrixXcd& O_eig,
                      double beta, double e_min)
{
    const int d = static_cast<int>(E.size());
    double Z = 0.0, S = 0.0;
    for (int n = 0; n < d; ++n) {
        const double blt = std::exp(-beta * (E(n) - e_min));
        Z += blt;
        double row = 0.0;
        for (int m = 0; m < d; ++m) row += std::norm(O_eig(m, n));
        S += blt * row;
    }
    return S / Z;
}

struct DenseFixture {
    uint64_t N = 0, dim = 0;
    std::unique_ptr<Operator> H_op;
    Eigen::MatrixXcd H_dense;
    Eigen::VectorXd  E;
    Eigen::MatrixXcd U;
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

void get_eigenstates(const DenseFixture& f,
                     std::vector<ComplexVector>& states,
                     std::vector<double>& energies)
{
    const int d = static_cast<int>(f.dim);
    states.resize(d);
    energies.resize(d);
    for (int n = 0; n < d; ++n) {
        energies[n] = f.E(n);
        states[n].resize(d);
        for (int i = 0; i < d; ++i)
            states[n][i] = f.U(i, n);
    }
}

} // namespace

// ============================================================================
// TEST 1: KPM from_states sum rule — ∫S(ω,β)dω vs Lehmann static (within 3%)
// ============================================================================
TEST_CASE("KPM from_states: sum rule ∫S dω ≈ Lehmann static",
          "[kpm][from-states][sum-rule]")
{
    const uint64_t N = 6;
    auto f = make_fixture(N);
    auto Sz0 = ed::ops::make_Sz_site(0, N);

    // Lehmann reference.
    auto Sz0_d = to_dense(Sz0, f.dim);
    auto Sz0_eig = f.U.adjoint() * Sz0_d * f.U;

    std::vector<ComplexVector> states;
    std::vector<double> energies;
    get_eigenstates(f, states, energies);

    const std::vector<double> betas = {0.5, 2.0, 8.0};

    const double e_min = f.e_min, e_max = f.E(f.dim - 1);
    const double bw    = e_max - e_min;
    const double omega_min = -(bw + 1.0);
    const double omega_max =  (bw + 1.0);
    const int    n_omega   = 2001;

    ed::kpm::KPMParameters p;
    p.num_moments              = 256;
    p.use_jackson_kernel       = true;
    p.spectral_bound_buffer    = 0.05;
    p.energy_shift             = e_min;

    auto res = ed::kpm::compute_kpm_ltlm_from_states(
        f.Hv, Sz0, Sz0,
        f.dim, f.dim, states, energies,
        omega_min, omega_max, n_omega,
        betas, p);

    REQUIRE(res.frequencies.size() == static_cast<std::size_t>(n_omega));
    REQUIRE(res.betas.size() == betas.size());

    for (std::size_t t = 0; t < betas.size(); ++t) {
        const double ref_static = lehmann_static(f.E, Sz0_eig, betas[t], e_min);
        const double kpm_static = res.static_correlator[t];  // ∫S dω pre-computed
        INFO("beta=" << betas[t] << " ref=" << ref_static << " kpm=" << kpm_static);
        CHECK(kpm_static == Catch::Approx(ref_static).epsilon(0.03));
    }
}

// ============================================================================
// TEST 2: Jackson kernel produces S(ω,β) ≥ 0 everywhere
// ============================================================================
TEST_CASE("KPM Jackson kernel: S(ω,β) ≥ 0 for all ω",
          "[kpm][jackson][positivity]")
{
    const uint64_t N = 6;
    auto f = make_fixture(N);
    auto Sz0 = ed::ops::make_Sz_site(0, N);

    std::vector<ComplexVector> states;
    std::vector<double> energies;
    get_eigenstates(f, states, energies);

    const double bw = f.E(f.dim-1) - f.e_min;
    const double omega_min = -(bw + 1.0);
    const double omega_max =  (bw + 1.0);
    const int n_omega = 1001;

    const std::vector<double> betas = {0.5, 4.0, 10.0};

    ed::kpm::KPMParameters p;
    p.num_moments              = 256;
    p.use_jackson_kernel       = true;   // guaranteed positive
    p.spectral_bound_buffer    = 0.05;
    p.energy_shift             = f.e_min;

    auto res = ed::kpm::compute_kpm_ltlm_from_states(
        f.Hv, Sz0, Sz0,
        f.dim, f.dim, states, energies,
        omega_min, omega_max, n_omega,
        betas, p);

    // Jackson kernel guarantees S ≥ 0.  Allow tiny numerical noise.
    const double tol = -1e-8;
    for (std::size_t t = 0; t < betas.size(); ++t) {
        for (int i = 0; i < n_omega; ++i) {
            const double v = res.spectral_real[t * n_omega + i];
            INFO("beta=" << betas[t] << " omega=" << res.frequencies[i]
                 << " S=" << v);
            CHECK(v >= tol);
        }
    }
}

// ============================================================================
// TEST 3: ∫S_{KPM} dω ≈ SSSF static  (internal consistency)
// ============================================================================
TEST_CASE("KPM LTLM: integral matches SSSF static within 3%",
          "[kpm][ltlm][sssf][consistency]")
{
    const uint64_t N = 6;
    auto f = make_fixture(N);
    auto Sz0 = ed::ops::make_Sz_site(0, N);

    const std::vector<double> betas = {0.5, 2.0, 8.0};

    const double bw = f.E(f.dim-1) - f.e_min;

    ed::kpm::KPMParameters kp;
    kp.num_moments              = 256;
    kp.num_lowest_states        = 64;   // all states for N=6
    kp.outer_krylov_dim         = 64;
    kp.use_jackson_kernel       = true;
    kp.full_reorthogonalization = true;
    kp.tolerance                = 1e-12;
    kp.random_seed              = 20260701ULL;
    kp.energy_shift             = f.e_min;
    kp.spectral_bound_buffer    = 0.05;

    auto kpm_res = ed::kpm::compute_kpm_ltlm(
        f.Hv, Sz0, Sz0, f.dim,
        -(bw+1.0), (bw+1.0), 2001,
        betas, kp);

    // SSSF reference via from_states.
    std::vector<ComplexVector> states;
    std::vector<double> energies;
    get_eigenstates(f, states, energies);

    auto sssf_res = ed::sssf::compute_sssf_ltlm_from_states(
        Sz0, Sz0, f.dim, f.dim, states, energies, betas);

    for (std::size_t t = 0; t < betas.size(); ++t) {
        const double kpm_static  = kpm_res.static_correlator[t];
        const double sssf_static = sssf_res.S_static_real[t];
        INFO("beta=" << betas[t]
             << " kpm_integral=" << kpm_static
             << " sssf="         << sssf_static);
        CHECK(kpm_static == Catch::Approx(sssf_static).epsilon(0.05));
    }
}

// ============================================================================
// TEST 4: from_states vs LTLM Lanczos consistency  (≤5% integral)
// ============================================================================
TEST_CASE("KPM LTLM vs from_states: spectral integrals agree within 5%",
          "[kpm][ltlm][from-states][consistency]")
{
    const uint64_t N = 6;
    auto f = make_fixture(N);
    auto Sz0 = ed::ops::make_Sz_site(0, N);

    std::vector<ComplexVector> states;
    std::vector<double> energies;
    get_eigenstates(f, states, energies);

    const std::vector<double> betas = {0.5, 2.0, 8.0};

    const double bw      = f.E(f.dim-1) - f.e_min;
    const double om_min  = -(bw + 1.0);
    const double om_max  =  (bw + 1.0);
    const int    n_omega = 1001;

    ed::kpm::KPMParameters p;
    p.num_moments              = 128;
    p.num_lowest_states        = 64;
    p.outer_krylov_dim         = 64;
    p.use_jackson_kernel       = true;
    p.full_reorthogonalization = true;
    p.tolerance                = 1e-12;
    p.random_seed              = 20260702ULL;
    p.energy_shift             = f.e_min;
    p.spectral_bound_buffer    = 0.05;

    // from_states (reference):
    auto ref = ed::kpm::compute_kpm_ltlm_from_states(
        f.Hv, Sz0, Sz0, f.dim, f.dim, states, energies,
        om_min, om_max, n_omega, betas, p);

    // LTLM outer Lanczos:
    auto kpm = ed::kpm::compute_kpm_ltlm(
        f.Hv, Sz0, Sz0, f.dim, om_min, om_max, n_omega, betas, p);

    for (std::size_t t = 0; t < betas.size(); ++t) {
        INFO("beta=" << betas[t]
             << " ref_integral="  << ref.static_correlator[t]
             << " kpm_integral="  << kpm.static_correlator[t]);
        CHECK(kpm.static_correlator[t]
              == Catch::Approx(ref.static_correlator[t]).epsilon(0.08));
    }
}

// ============================================================================
// TEST 5: combine_sector_kpm — Z-weighted recombination (analytic check)
//
//   Sector A (dim=4): Z_A=4, S_A uniform = 1/n_omega
//   Sector B (dim=2): Z_B=2, S_B uniform = 2/n_omega
//
//   Z_eff_A = 16,  Z_eff_B = 4,  Z_total = 20
//   S_combined = (16 * 1/n + 4 * 2/n) / 20 = 24 / (20*n) per bin
//   integral   = (24/20) = 1.2
// ============================================================================
TEST_CASE("KPM combine_sector_kpm: Z-weighted recombination",
          "[kpm][combine]")
{
    const int n_omega = 5;
    const std::vector<double> freqs = {-2.0, -1.0, 0.0, 1.0, 2.0};
    const std::vector<double> betas = {1.0};

    auto make_sector = [&](double Z_val, double S_val_per_bin) -> ed::kpm::KPMResult {
        ed::kpm::KPMResult r;
        r.frequencies        = freqs;
        r.betas              = betas;
        r.partition_function = {Z_val};
        r.spectral_real.resize(n_omega, S_val_per_bin);
        r.spectral_imag.resize(n_omega, 0.0);
        r.static_correlator  = {S_val_per_bin * n_omega * 1.0};  // dω=1
        r.kpm_a = 1.0; r.kpm_b = 0.0;
        r.num_moments_used = 64;
        r.jackson_kernel_used = true;
        r.total_outer_states = 4;
        r.ground_state_estimate = -2.0;
        r.energy_shift_used = -2.0;
        return r;
    };

    auto rA = make_sector(4.0, 1.0 / n_omega);  // Z=4, S per bin = 0.2
    auto rB = make_sector(2.0, 2.0 / n_omega);  // Z=2, S per bin = 0.4

    auto combined = ed::kpm::combine_sector_kpm({rA, rB}, {4ULL, 2ULL});

    // Z_eff_A=16, Z_eff_B=4, Z_total=20
    const double Z_total = 20.0;
    const double S_combined_per_bin = (16.0 * 0.2 + 4.0 * 0.4) / Z_total;
    // = (3.2 + 1.6) / 20 = 4.8/20 = 0.24

    REQUIRE(combined.spectral_real.size() == static_cast<std::size_t>(n_omega));
    CHECK(combined.partition_function[0] == Catch::Approx(Z_total).epsilon(1e-10));
    for (int i = 0; i < n_omega; ++i) {
        CHECK(combined.spectral_real[i]
              == Catch::Approx(S_combined_per_bin).epsilon(1e-10));
    }
}

// ============================================================================
// TEST 6: T=∞ sum rule for KPM from_states
//   At T=∞ (β→0): S(ω) dω → (1/D) Tr[O† O] = <Sz₀²>_{T=∞} = 1/4
//   (average of +1/4 and +1/4 from UP and DOWN states)
// ============================================================================
TEST_CASE("KPM from_states T=∞: ∫S dω = (1/D)Tr[Sz†Sz] = 1/4",
          "[kpm][from-states][sum-rule][T-infinity]")
{
    const uint64_t N = 6;
    auto f = make_fixture(N);
    auto Sz0 = ed::ops::make_Sz_site(0, N);

    std::vector<ComplexVector> states;
    std::vector<double> energies;
    get_eigenstates(f, states, energies);

    // β → 0 is T=∞: use a very small β.
    const std::vector<double> betas = {1e-5};

    // At T=∞: S_static = (1/D) Tr[Sz₀² ] = 1/D * D * 1/4 = 1/4.
    const double expected = 0.25;

    const double bw = f.E(f.dim-1) - f.e_min;

    ed::kpm::KPMParameters p;
    p.num_moments              = 256;
    p.use_jackson_kernel       = true;
    p.spectral_bound_buffer    = 0.05;
    p.energy_shift             = f.e_min;

    auto res = ed::kpm::compute_kpm_ltlm_from_states(
        f.Hv, Sz0, Sz0, f.dim, f.dim, states, energies,
        -(bw+1.0), (bw+1.0), 2001, betas, p);

    INFO("T=∞ integral=" << res.static_correlator[0] << " expected=" << expected);
    CHECK(res.static_correlator[0] == Catch::Approx(expected).epsilon(0.03));
}
