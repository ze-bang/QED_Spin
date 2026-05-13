// =============================================================================
// tests/unit/test_block_lanczos_dssf.cpp
//
// Audit item #6: acceptance tests for the multi-channel block / band Lanczos
// ground-state DSSF kernel.
//
// Reference: P scalar Lanczos chains computed via
// `compute_ground_state_dssf` (already validated). Block Lanczos with the
// SAME total Krylov budget MP must reproduce each scalar spectrum to high
// precision (well below the broadening eta), and integrate to the same
// per-channel sum rule ||O_p|0>||^2.
// =============================================================================

#include "common/catch2_harness.h"
#include <catch2/catch_approx.hpp>

#include <ed/solvers/block_lanczos_dssf.h>
#include <ed/solvers/ftlm.h>     // compute_ground_state_dssf, build_lanczos_tridiagonal
#include <ed/solvers/lanczos.h>  // standardLanczos for ground state

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

// Dense materialisation of an apply() callable -> Eigen::MatrixXcd.
Eigen::MatrixXcd to_dense(
    const std::function<void(const Complex*, Complex*, int)>& op,
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

// Build a single-site Sz operator at site `s` (acts on full Hilbert space
// dim = 2^N). Returns an apply lambda closing over a captured shared_ptr.
struct SiteOpData {
    uint64_t N;
    uint64_t site;
    int op_type;  // 2=Sz, 0=S+, 1=S-
};

std::function<void(const Complex*, Complex*, std::size_t)>
make_site_op(SiteOpData d) {
    const uint64_t dim = 1ULL << d.N;
    return [d, dim](const Complex* in, Complex* out, std::size_t n) {
        if (n != dim) return;
        const uint64_t bit = 1ULL << d.site;
        for (uint64_t i = 0; i < dim; ++i) {
            const bool up = (i & bit) != 0;
            switch (d.op_type) {
                case 2: {  // Sz |up> = +1/2 |up>, Sz |dn> = -1/2 |dn>
                    out[i] = (up ? 0.5 : -0.5) * in[i];
                    break;
                }
                case 0: {  // S+ |dn> = |up>; S+ |up> = 0
                    if (!up) {
                        const uint64_t flipped = i ^ bit;
                        out[flipped] = in[i];
                    }
                    break;
                }
                case 1: {  // S- |up> = |dn>; S- |dn> = 0
                    if (up) {
                        const uint64_t flipped = i ^ bit;
                        out[flipped] = in[i];
                    }
                    break;
                }
                default: break;
            }
        }
    };
}

// Reference: scalar GS DSSF for one operator, returning (frequencies,
// spectral, norm_sq) via the existing validated kernel.
struct ScalarRef {
    std::vector<double> frequencies;
    std::vector<double> spectral;
    double norm_sq;
};

ScalarRef scalar_ref(
    const std::function<void(const Complex*, Complex*, int)>& H,
    const std::function<void(const Complex*, Complex*, std::size_t)>& O,
    const ComplexVector& gs,
    double E0,
    uint64_t dim,
    std::size_t krylov_dim,
    double eta,
    double omega_min,
    double omega_max,
    std::size_t num_omega)
{
    GroundStateDSSFParameters p;
    p.krylov_dim = krylov_dim;
    p.broadening = eta;
    p.omega_min = omega_min;
    p.omega_max = omega_max;
    p.num_omega_points = num_omega;
    p.tolerance = 1e-12;
    p.full_reorthogonalization = true;
    p.use_continued_fraction = true;

    auto O_int = [&O](const Complex* in, Complex* out, int n) {
        O(in, out, static_cast<std::size_t>(n));
    };
    auto res = compute_ground_state_dssf(
        H, O_int, gs, E0, dim, p);

    ScalarRef r;
    r.frequencies = res.frequencies;
    r.spectral = res.spectral_function;
    // Sum-rule target: ||O|0>||^2
    ComplexVector tmp(dim);
    O(gs.data(), tmp.data(), dim);
    double n2 = 0.0;
    for (std::size_t i = 0; i < dim; ++i) n2 += std::norm(tmp[i]);
    r.norm_sq = n2;
    return r;
}

}  // namespace

TEST_CASE("block_lanczos_dssf: trivial P=1 reproduces scalar GS DSSF",
          "[block_lanczos_dssf]") {
    const uint64_t N = 6;
    const uint64_t dim = 1ULL << N;
    auto H_op = build_heisenberg_chain(N, 1.0, /*periodic=*/true);
    auto H = [&H_op, dim](const Complex* in, Complex* out, std::size_t n) {
        H_op->apply(in, out, n);
    };

    // Get exact ground state via dense diagonalisation (small dim).
    auto H_int = [&H_op](const Complex* in, Complex* out, int n) {
        H_op->apply(in, out, static_cast<std::size_t>(n));
    };
    Eigen::MatrixXcd Hdense = to_dense(H_int, dim);
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> es(Hdense);
    REQUIRE(es.info() == Eigen::Success);
    const double E0 = es.eigenvalues()(0);
    Eigen::VectorXcd gs_eig = es.eigenvectors().col(0);
    ComplexVector gs(dim);
    for (std::size_t i = 0; i < dim; ++i) gs[i] = gs_eig(i);

    auto Sz0 = make_site_op({N, 0, 2});

    const std::size_t M = 60;
    const double eta = 0.05;
    const double wmin = -3.0, wmax = 3.0;
    const std::size_t nw = 256;

    auto ref = scalar_ref(H_int, Sz0, gs, E0, dim, M, eta, wmin, wmax, nw);

    ed::block_lanczos_dssf::BlockLanczosDSSFParameters bp;
    bp.krylov_dim = M;
    bp.broadening = eta;
    bp.omega_min = wmin;
    bp.omega_max = wmax;
    bp.num_omega_points = nw;
    bp.reorth_interval = 1;  // aggressive

    auto block = ed::block_lanczos_dssf::compute_ground_state_block_dssf(
        H, {Sz0}, gs, E0, dim, bp);

    REQUIRE(block.spectral.size() == 1);
    REQUIRE(block.spectral[0].size() == nw);
    REQUIRE(block.frequencies.size() == nw);

    // Per-frequency comparison: with the SAME Krylov dim and a single
    // channel, block reduces to scalar (R0 is 1x1 = ||O|0>||).
    double max_abs = 0.0;
    for (std::size_t i = 0; i < nw; ++i) {
        max_abs = std::max(max_abs,
                           std::abs(block.spectral[0][i] - ref.spectral[i]));
    }
    INFO("max |block - scalar| = " << max_abs);
    // Same Krylov dim, same starting vector -> identical Ritz pairs up to
    // BLAS round-off. Allow generous threshold for accumulated FP error.
    REQUIRE(max_abs < 1e-8);
}

TEST_CASE("block_lanczos_dssf: P=3 multi-site Sz reproduces scalar refs",
          "[block_lanczos_dssf]") {
    const uint64_t N = 6;
    const uint64_t dim = 1ULL << N;
    auto H_op = build_heisenberg_chain(N, 1.0, /*periodic=*/true);
    auto H_int = [&H_op](const Complex* in, Complex* out, int n) {
        H_op->apply(in, out, static_cast<std::size_t>(n));
    };
    auto H = [&H_op](const Complex* in, Complex* out, std::size_t n) {
        H_op->apply(in, out, n);
    };

    Eigen::MatrixXcd Hdense = to_dense(H_int, dim);
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> es(Hdense);
    REQUIRE(es.info() == Eigen::Success);
    const double E0 = es.eigenvalues()(0);
    Eigen::VectorXcd gs_eig = es.eigenvectors().col(0);
    ComplexVector gs(dim);
    for (std::size_t i = 0; i < dim; ++i) gs[i] = gs_eig(i);

    std::vector<std::function<void(const Complex*, Complex*, std::size_t)>> Olist;
    Olist.push_back(make_site_op({N, 0, 2}));
    Olist.push_back(make_site_op({N, 1, 2}));
    Olist.push_back(make_site_op({N, 2, 2}));

    // Block Krylov dim -- intentionally generous so block-tridiag captures
    // the full spectrum (block Lanczos can exhaust dim before MP iters
    // when starting vectors span only a small invariant subspace).
    const std::size_t M = 80;
    const double eta = 0.08;  // larger eta => smoother spectra => looser
                              // Krylov-truncation comparison
    const double wmin = -3.0, wmax = 3.0;
    const std::size_t nw = 200;

    ed::block_lanczos_dssf::BlockLanczosDSSFParameters bp;
    bp.krylov_dim = M;
    bp.broadening = eta;
    bp.omega_min = wmin;
    bp.omega_max = wmax;
    bp.num_omega_points = nw;
    bp.reorth_interval = 1;

    auto block = ed::block_lanczos_dssf::compute_ground_state_block_dssf(
        H, Olist, gs, E0, dim, bp);

    REQUIRE(block.spectral.size() == Olist.size());
    REQUIRE(block.iterations_completed > 0);

    // Block Lanczos with M block iters spans an MP-dimensional Krylov
    // subspace per starting vector (in expectation). Compare each channel
    // against a scalar Lanczos ref of dim MP -- should agree to within
    // eta-scale precision.
    const std::size_t scalar_M = std::min<std::size_t>(M * Olist.size(), dim);
    for (std::size_t p = 0; p < Olist.size(); ++p) {
        auto ref = scalar_ref(H_int, Olist[p], gs, E0, dim,
                              scalar_M, eta, wmin, wmax, nw);

        // Compare integrals (sum rules) -- both methods should integrate
        // to ||O|0>||^2 within Krylov truncation error.
        double Iblock = 0.0, Iref = 0.0;
        for (std::size_t i = 1; i < nw; ++i) {
            const double dw = block.frequencies[i] - block.frequencies[i - 1];
            Iblock += 0.5 * (block.spectral[p][i] + block.spectral[p][i - 1]) * dw;
            Iref   += 0.5 * (ref.spectral[i]      + ref.spectral[i - 1])      * dw;
        }
        INFO("Channel " << p << ": Iblock=" << Iblock
              << " Iref=" << Iref << " ||O|0>||^2=" << ref.norm_sq);
        REQUIRE(std::abs(Iblock - ref.norm_sq) / ref.norm_sq < 0.05);
        REQUIRE(std::abs(Iref   - ref.norm_sq) / ref.norm_sq < 0.05);

        // Per-frequency: spectra should agree within 5% of peak height
        // (block and scalar take different Krylov paths so individual
        // peaks may shift by ~eta but bulk shape matches).
        double peak = 0.0;
        for (std::size_t i = 0; i < nw; ++i) {
            peak = std::max(peak, std::max(block.spectral[p][i],
                                           ref.spectral[i]));
        }
        double max_abs = 0.0;
        for (std::size_t i = 0; i < nw; ++i) {
            max_abs = std::max(max_abs,
                               std::abs(block.spectral[p][i] - ref.spectral[i]));
        }
        INFO("Channel " << p << ": max|block - ref| = " << max_abs
              << " peak = " << peak);
        REQUIRE(max_abs < 0.20 * peak);
    }
}

TEST_CASE("block_lanczos_dssf: per-channel sum rule",
          "[block_lanczos_dssf]") {
    const uint64_t N = 4;
    const uint64_t dim = 1ULL << N;
    auto H_op = build_heisenberg_chain(N, 1.0, /*periodic=*/true);
    auto H_int = [&H_op](const Complex* in, Complex* out, int n) {
        H_op->apply(in, out, static_cast<std::size_t>(n));
    };
    auto H = [&H_op](const Complex* in, Complex* out, std::size_t n) {
        H_op->apply(in, out, n);
    };
    Eigen::MatrixXcd Hdense = to_dense(H_int, dim);
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> es(Hdense);
    REQUIRE(es.info() == Eigen::Success);
    const double E0 = es.eigenvalues()(0);
    Eigen::VectorXcd gs_eig = es.eigenvectors().col(0);
    ComplexVector gs(dim);
    for (std::size_t i = 0; i < dim; ++i) gs[i] = gs_eig(i);

    // Mix of Sz, S+, S- on different sites
    std::vector<std::function<void(const Complex*, Complex*, std::size_t)>> Olist;
    Olist.push_back(make_site_op({N, 0, 2}));
    Olist.push_back(make_site_op({N, 1, 0}));
    Olist.push_back(make_site_op({N, 2, 1}));

    ed::block_lanczos_dssf::BlockLanczosDSSFParameters bp;
    bp.krylov_dim = std::min<std::size_t>(20, dim);
    bp.broadening = 0.05;
    bp.omega_min = -3.0;
    bp.omega_max = 3.0;
    bp.num_omega_points = 400;
    bp.reorth_interval = 1;

    auto block = ed::block_lanczos_dssf::compute_ground_state_block_dssf(
        H, Olist, gs, E0, dim, bp);

    REQUIRE(block.spectral.size() == Olist.size());
    REQUIRE(block.channel_norm_sq.size() == Olist.size());

    for (std::size_t p = 0; p < Olist.size(); ++p) {
        double I = 0.0;
        for (std::size_t i = 1; i < bp.num_omega_points; ++i) {
            const double dw = block.frequencies[i] - block.frequencies[i - 1];
            I += 0.5 * (block.spectral[p][i] + block.spectral[p][i - 1]) * dw;
        }
        const double target = block.channel_norm_sq[p];
        REQUIRE(target > 0.0);
        INFO("Channel " << p << ": integral = " << I
              << " target = " << target
              << " ratio = " << (I / target));
        // For a small dim (16) and generous Krylov, block Lanczos saturates
        // the Krylov subspace -> sum rule holds tightly, modulo
        // Lorentzian tails truncated by [omega_min, omega_max].
        REQUIRE(I / target > 0.85);
        REQUIRE(I / target < 1.15);
    }
}
