// =============================================================================
// tests/unit/test_cross_sector_dssf.cpp
//
// Audit item #1 (FULL): acceptance tests for the cross-sector observable
// `ed::dssf::CrossSectorObservable` and the associated continued-fraction
// kernel `compute_ground_state_dssf_cross_sector`. These together unblock
// the legitimate (S+, S-) / (S-, S+) channels that the
// audit-#1-partial guard rail in src/cli/workflows.cpp currently drops to
// avoid silent zeros in fixed-Sz DSSF runs.
//
// References:
//   * Apply correctness:   direct bit-flip computation against a
//     hand-built dense rectangular operator (dim_dst x dim_src).
//   * Spectrum correctness: dense Lehmann decomposition of the
//     destination-sector Hamiltonian, S(S-, S+)(omega) computed two
//     ways and compared.
// =============================================================================

#include "common/catch2_harness.h"
#include <catch2/catch_approx.hpp>

#include <ed/dssf/cross_sector_observable.h>
#include <ed/solvers/ftlm.h>
#include <ed/solvers/lanczos.h>
#include <ed/core/basis_utils.h>

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>

#include <algorithm>
#include <cmath>
#include <complex>
#include <memory>
#include <vector>

using ed_tests::build_heisenberg_chain_fixed_sz;
using Complex = std::complex<double>;
using ComplexVector = std::vector<Complex>;

namespace {

// Build a dense (dim_dst x dim_src) rectangular matrix representing the
// CrossSectorObservable, by applying it to each canonical src basis
// vector.
Eigen::MatrixXcd materialise_dense(const ed::dssf::CrossSectorObservable& O) {
    const std::size_t dim_src = O.dim_src();
    const std::size_t dim_dst = O.dim_dst();
    Eigen::MatrixXcd M = Eigen::MatrixXcd::Zero(dim_dst, dim_src);
    ComplexVector in(dim_src), out(dim_dst);
    for (std::size_t j = 0; j < dim_src; ++j) {
        std::fill(in.begin(), in.end(), Complex(0.0, 0.0));
        in[j] = Complex(1.0, 0.0);
        std::fill(out.begin(), out.end(), Complex(0.0, 0.0));
        O.apply(in.data(), out.data(), dim_dst);
        for (std::size_t i = 0; i < dim_dst; ++i) M(i, j) = out[i];
    }
    return M;
}

// Build a dense single-site spin operator on the FULL Hilbert space
// (size 2^N), then project to the (src, dst) fixed-Sz blocks for use as
// reference. op_type: 0 = S+, 1 = S-, 2 = Sz. Spin-1/2.
Eigen::MatrixXcd build_dense_single_site(
    uint64_t N, std::uint8_t op_type, uint64_t site,
    const std::vector<uint64_t>& src_basis,
    const std::vector<uint64_t>& dst_basis)
{
    const std::size_t dim_src = src_basis.size();
    const std::size_t dim_dst = dst_basis.size();
    Eigen::MatrixXcd M = Eigen::MatrixXcd::Zero(dim_dst, dim_src);
    // Map dst basis state -> index for O(1) lookup
    std::vector<std::pair<uint64_t, std::size_t>> dst_lookup;
    dst_lookup.reserve(dim_dst);
    for (std::size_t i = 0; i < dim_dst; ++i) {
        dst_lookup.emplace_back(dst_basis[i], i);
    }
    std::sort(dst_lookup.begin(), dst_lookup.end());

    auto find_dst = [&](uint64_t b) -> long {
        auto it = std::lower_bound(
            dst_lookup.begin(), dst_lookup.end(),
            std::make_pair(b, std::size_t{0}),
            [](const auto& a, const auto& c) { return a.first < c.first; });
        if (it == dst_lookup.end() || it->first != b) return -1;
        return static_cast<long>(it->second);
    };

    const double S = 0.5;
    for (std::size_t j = 0; j < dim_src; ++j) {
        const uint64_t b = src_basis[j];
        const uint64_t bit = (b >> site) & 1ULL;
        if (op_type == 2) {
            const double sgn = bit ? -1.0 : 1.0;
            const long i = find_dst(b);
            if (i >= 0) M(i, j) = Complex(S * sgn, 0.0);
        } else {
            // S+ requires bit==0; S- requires bit==1
            if (bit != op_type) {
                const uint64_t b2 = b ^ (1ULL << site);
                const long i = find_dst(b2);
                if (i >= 0) M(i, j) = Complex(1.0, 0.0);
            }
        }
        (void)N;
    }
    return M;
}

}  // namespace

TEST_CASE("CrossSectorObservable: single-site S+ matches dense reference",
          "[dssf][cross_sector]") {
    // NOTE on convention: in this codebase bit=1 carries the popcount,
    // and op_type=0 ("S+" in the legacy naming) flips bit=1 -> 0, which
    // *decreases* popcount by 1. Conversely op_type=1 ("S-") flips
    // bit=0 -> 1 and *increases* popcount by 1. The fixed-Sz sector is
    // labelled by n_up = popcount of basis states, so:
    //   op_type=0 -> n_up_dst = n_up_src - 1
    //   op_type=1 -> n_up_dst = n_up_src + 1
    constexpr uint64_t N = 6;
    constexpr int64_t n_up_src = 3;
    constexpr int64_t n_up_dst = 2;  // op_type=0 lowers popcount by 1
    constexpr double J = 1.0;

    auto src = build_heisenberg_chain_fixed_sz(N, J, n_up_src, /*periodic=*/true);
    auto dst = build_heisenberg_chain_fixed_sz(N, J, n_up_dst, /*periodic=*/true);
    std::shared_ptr<FixedSzOperator> src_sp(std::move(src));
    std::shared_ptr<FixedSzOperator> dst_sp(std::move(dst));

    // T = (op_type=0)_2 with coefficient 1.0
    std::vector<Operator::TransformData> tlist(1);
    tlist[0].op_type = 0;
    tlist[0].site_index = 2;
    tlist[0].coefficient = Complex(1.0, 0.0);
    tlist[0].is_two_body = false;

    ed::dssf::CrossSectorObservable O(src_sp, dst_sp, tlist, 0.5f);

    REQUIRE(O.delta_n_up() == -1);
    REQUIRE(O.dim_src() == src_sp->getFixedSzDim());
    REQUIRE(O.dim_dst() == dst_sp->getFixedSzDim());

    auto M_ours = materialise_dense(O);
    auto M_ref  = build_dense_single_site(N, /*S+*/0, /*site=*/2,
                                          src_sp->getBasisStates(),
                                          dst_sp->getBasisStates());

    REQUIRE(M_ours.rows() == M_ref.rows());
    REQUIRE(M_ours.cols() == M_ref.cols());
    const double diff = (M_ours - M_ref).cwiseAbs().maxCoeff();
    INFO("max |M_ours - M_ref| = " << diff);
    REQUIRE(diff < 1e-12);
}

TEST_CASE("CrossSectorObservable: linear-combination S- matches dense reference",
          "[dssf][cross_sector]") {
    constexpr uint64_t N = 6;
    constexpr int64_t n_up_src = 3;
    constexpr int64_t n_up_dst = 4;  // op_type=1 raises popcount by 1
    constexpr double J = 1.0;

    auto src = build_heisenberg_chain_fixed_sz(N, J, n_up_src, /*periodic=*/true);
    auto dst = build_heisenberg_chain_fixed_sz(N, J, n_up_dst, /*periodic=*/true);
    std::shared_ptr<FixedSzOperator> src_sp(std::move(src));
    std::shared_ptr<FixedSzOperator> dst_sp(std::move(dst));

    // T = sum_i c_i (op_type=1)_i with arbitrary complex c_i
    // (uniform delta_n_up = +1).
    std::vector<Complex> coeffs = {
        Complex(0.7, -0.2), Complex(-1.1, 0.4), Complex(0.3, 0.0),
        Complex(0.0, 1.0),  Complex(-0.5, -0.5), Complex(0.9, 0.1)};
    REQUIRE(coeffs.size() == N);

    std::vector<Operator::TransformData> tlist;
    for (uint64_t s = 0; s < N; ++s) {
        Operator::TransformData t;
        t.op_type = 1;
        t.site_index = s;
        t.coefficient = coeffs[s];
        t.is_two_body = false;
        tlist.push_back(t);
    }

    ed::dssf::CrossSectorObservable O(src_sp, dst_sp, tlist, 0.5f);
    auto M_ours = materialise_dense(O);

    Eigen::MatrixXcd M_ref =
        Eigen::MatrixXcd::Zero(dst_sp->getFixedSzDim(),
                               src_sp->getFixedSzDim());
    for (uint64_t s = 0; s < N; ++s) {
        M_ref += coeffs[s] * build_dense_single_site(
            N, /*op_type=1*/1, s,
            src_sp->getBasisStates(), dst_sp->getBasisStates());
    }
    const double diff = (M_ours - M_ref).cwiseAbs().maxCoeff();
    INFO("max |M_ours - M_ref| = " << diff);
    REQUIRE(diff < 1e-12);
}

TEST_CASE("compute_ground_state_dssf_cross_sector: S(S-, S+) matches dense Lehmann",
          "[dssf][cross_sector][spectrum]") {
    // N=6 Heisenberg, GS in sector n_up=3 (Sz=0). Apply S+_0 to lift to
    // n_up=4. Compute S_{S-_0, S+_0}(omega) two ways and compare on a
    // single representative frequency window.
    constexpr uint64_t N = 6;
    constexpr int64_t n_up_src = 3;
    constexpr int64_t n_up_dst = 2;  // op_type=0 lowers popcount by 1
    constexpr double J = 1.0;
    constexpr uint64_t site = 0;

    auto src = build_heisenberg_chain_fixed_sz(N, J, n_up_src, /*periodic=*/true);
    auto dst = build_heisenberg_chain_fixed_sz(N, J, n_up_dst, /*periodic=*/true);
    std::shared_ptr<FixedSzOperator> src_sp(std::move(src));
    std::shared_ptr<FixedSzOperator> dst_sp(std::move(dst));

    const std::size_t dim_src = src_sp->getFixedSzDim();
    const std::size_t dim_dst = dst_sp->getFixedSzDim();

    // ---- Source-sector dense diagonalisation for ground state ----
    Eigen::MatrixXcd H_src = Eigen::MatrixXcd::Zero(dim_src, dim_src);
    {
        ComplexVector in(dim_src), out(dim_src);
        for (std::size_t j = 0; j < dim_src; ++j) {
            std::fill(in.begin(), in.end(), Complex(0.0, 0.0));
            in[j] = Complex(1.0, 0.0);
            std::fill(out.begin(), out.end(), Complex(0.0, 0.0));
            src_sp->apply(in.data(), out.data(), dim_src);
            for (std::size_t i = 0; i < dim_src; ++i) H_src(i, j) = out[i];
        }
    }
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> es_src(H_src);
    const double E0 = es_src.eigenvalues()[0];
    Eigen::VectorXcd gs_eig = es_src.eigenvectors().col(0);
    ComplexVector gs(dim_src);
    for (std::size_t i = 0; i < dim_src; ++i) gs[i] = gs_eig(i);

    // ---- Destination-sector dense diagonalisation for Lehmann reference ----
    Eigen::MatrixXcd H_dst = Eigen::MatrixXcd::Zero(dim_dst, dim_dst);
    {
        ComplexVector in(dim_dst), out(dim_dst);
        for (std::size_t j = 0; j < dim_dst; ++j) {
            std::fill(in.begin(), in.end(), Complex(0.0, 0.0));
            in[j] = Complex(1.0, 0.0);
            std::fill(out.begin(), out.end(), Complex(0.0, 0.0));
            dst_sp->apply(in.data(), out.data(), dim_dst);
            for (std::size_t i = 0; i < dim_dst; ++i) H_dst(i, j) = out[i];
        }
    }
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> es_dst(H_dst);

    // ---- Build CrossSectorObservable for O = S+_site (raise) ----
    std::vector<Operator::TransformData> tlist(1);
    tlist[0].op_type = 0;
    tlist[0].site_index = site;
    tlist[0].coefficient = Complex(1.0, 0.0);
    tlist[0].is_two_body = false;
    auto O_raise = std::make_shared<ed::dssf::CrossSectorObservable>(
        src_sp, dst_sp, tlist, 0.5f);

    // ---- Reference: dense Lehmann sum at T=0 ----
    // |phi> = O |gs> in dst sector
    ComplexVector phi(dim_dst);
    O_raise->apply(gs.data(), phi.data(), dim_dst);
    Eigen::VectorXcd phi_eig(dim_dst);
    for (std::size_t i = 0; i < dim_dst; ++i) phi_eig(i) = phi[i];

    const auto& Vdst = es_dst.eigenvectors();
    const auto& Edst = es_dst.eigenvalues();
    Eigen::VectorXcd c = Vdst.adjoint() * phi_eig;          // <n|phi>
    // For the S(S-, S+) self-correlator: weight_n = |<n|S+|0>|^2
    std::vector<double> weight(dim_dst);
    for (std::size_t n = 0; n < dim_dst; ++n) {
        weight[n] = std::norm(c(n));
    }

    // ---- Cross-sector kernel evaluation ----
    auto H_inner_apply = [&](const Complex* in, Complex* out, int n) {
        dst_sp->apply(in, out, static_cast<std::size_t>(n));
    };
    auto O_apply = [&](const Complex* in, Complex* out, int n) {
        O_raise->apply(in, out, static_cast<std::size_t>(n));
    };
    GroundStateDSSFParameters params;
    params.omega_min = -1.0;
    params.omega_max = 8.0;
    params.num_omega_points = 60;
    params.broadening = 0.10;
    params.krylov_dim = std::min<uint64_t>(dim_dst, 80);
    params.tolerance = 1e-12;
    params.full_reorthogonalization = true;
    params.reorth_frequency = 1;

    auto res = compute_ground_state_dssf_cross_sector(
        H_inner_apply, /*O1=*/O_apply, /*O2=*/O_apply,
        gs, E0, dim_src, dim_dst, params);

    REQUIRE(res.frequencies.size() == params.num_omega_points);
    REQUIRE(res.spectral_function.size() == params.num_omega_points);

    // Compare against dense Lehmann
    const double eta = params.broadening;
    double max_abs_diff = 0.0;
    double ref_peak = 0.0;
    for (std::size_t k = 0; k < params.num_omega_points; ++k) {
        const double w = res.frequencies[k];
        double S_ref = 0.0;
        for (std::size_t n = 0; n < dim_dst; ++n) {
            const double E = Edst[n] - E0;
            const double L = (eta / M_PI) /
                             ((w - E) * (w - E) + eta * eta);
            S_ref += weight[n] * L;
        }
        const double S_lan = res.spectral_function[k];
        max_abs_diff = std::max(max_abs_diff, std::abs(S_ref - S_lan));
        ref_peak = std::max(ref_peak, std::abs(S_ref));
    }
    INFO("max |S_lan - S_ref| = " << max_abs_diff
         << "  (ref peak = " << ref_peak << ")");
    // 1% of peak is comfortably within Lanczos+Lorentzian agreement
    REQUIRE(max_abs_diff < 0.01 * std::max(ref_peak, 1e-6));

    // Sum-rule: integral over omega of S(omega) = ||O|0>||^2
    double integral = 0.0;
    const double dw = (params.omega_max - params.omega_min) /
                       (params.num_omega_points - 1);
    for (std::size_t k = 0; k < params.num_omega_points; ++k) {
        integral += res.spectral_function[k] * dw;
    }
    const double phi_norm_sq = phi_eig.squaredNorm();
    INFO("integral = " << integral << "  norm^2 = " << phi_norm_sq);
    REQUIRE(std::abs(integral - phi_norm_sq) < 0.05 * phi_norm_sq);
}
