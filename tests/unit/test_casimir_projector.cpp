// =============================================================================
// tests/unit/test_casimir_projector.cpp
//
// Stage 12d of the SU(2) rollout: the Lowdin total-spin projector
// (include/ed/symmetry/casimir_projector.h).
//
// Pinned:
//   * allowed_two_S_in_block: full space, fixed-Sz floor, Sz-parity S=0
//     rule, flip-parity mask + admissibility guard;
//   * P_S lands in the S^2 eigenspace (residual <= 1e-10), is idempotent,
//     and P_S P_{S'} = 0;
//   * trace(P_S) equals the exact tower dimension -- (2S+1)*M(N,S) in the
//     full space, M(N,S) in a fixed-Sz sector (each multiplet contributes
//     exactly one state per admissible Sz);
//   * `project` restores the exact (unnormalised) P_S v against a dense
//     eigenbasis reference;
//   * CasimirProjectedOperator preserves H's action on the targeted tower
//     and scrubs off-tower drift.
// =============================================================================
#include "common/catch2_harness.h"

#include <Eigen/Dense>

#include <ed/core/fixed_sz_operator.h>
#include <ed/core/operator.h>
#include <ed/operators/casimir.h>
#include <ed/symmetry/casimir_projector.h>
#include <ed/symmetry/su2_dims.h>

#include <complex>
#include <cstdint>
#include <memory>
#include <random>
#include <vector>

using Cx = std::complex<double>;
using ed::ops::make_S2_carrier;
using ed::symmetry::allowed_two_S_in_block;
using ed::symmetry::CasimirProjectedOperator;
using ed::symmetry::LowdinS2Projector;
using ed::symmetry::multiplet_count;

namespace {

std::vector<Cx> random_vector(std::uint64_t dim, unsigned seed) {
    std::mt19937 gen(seed);
    std::normal_distribution<double> dist(0.0, 1.0);
    std::vector<Cx> v(dim);
    for (auto& x : v) x = Cx(dist(gen), dist(gen));
    return v;
}

double norm_of(const std::vector<Cx>& v) {
    double n2 = 0.0;
    for (const auto& x : v) n2 += std::norm(x);
    return std::sqrt(n2);
}

// ||S^2 v - lam v|| / ||v||
double eigen_residual(const ed::matvec::MatVecOperator& s2,
                      const std::vector<Cx>& v, double lam) {
    std::vector<Cx> w(v.size());
    s2.apply(v.data(), w.data(), v.size());
    double r2 = 0.0, n2 = 0.0;
    for (std::size_t i = 0; i < v.size(); ++i) {
        r2 += std::norm(w[i] - lam * v[i]);
        n2 += std::norm(v[i]);
    }
    return std::sqrt(r2 / n2);
}

// Heisenberg ring J = 1 on N sites (terms only; caller picks the operator).
template <class Op>
void add_heisenberg_ring(Op& op, std::uint64_t N) {
    for (std::uint64_t i = 0; i < N; ++i) {
        const std::uint64_t j = (i + 1) % N;
        op.addTwoBodyTerm(2, i, 2, j, Cx(1.0, 0.0));
        op.addTwoBodyTerm(0, i, 1, j, Cx(0.5, 0.0));
        op.addTwoBodyTerm(1, i, 0, j, Cx(0.5, 0.0));
    }
}

}  // namespace

TEST_CASE("allowed_two_S_in_block: floors, parities, guards", "[casimir_proj]") {
    // Full space, N = 6.
    REQUIRE(allowed_two_S_in_block(6) == std::vector<int>{0, 2, 4, 6});
    // Fixed Sz: N = 8, n_up = 3 -> |2*3-8| = 2.
    REQUIRE(allowed_two_S_in_block(8, 3) == std::vector<int>{2, 4, 6, 8});
    // Sz parity: N = 6, N/2 = 3 odd -> S = 0 lives in the ODD half.
    REQUIRE(allowed_two_S_in_block(6, -1, /*sz_parity=*/0) ==
            std::vector<int>{2, 4, 6});
    REQUIRE(allowed_two_S_in_block(6, -1, /*sz_parity=*/1) ==
            std::vector<int>{0, 2, 4, 6});
    // Flip parity at half filling, N = 6: (N - 2S)/2 % 2 selects.
    REQUIRE(allowed_two_S_in_block(6, 3, -1, /*flip=*/0) ==
            std::vector<int>{2, 6});
    REQUIRE(allowed_two_S_in_block(6, 3, -1, /*flip=*/1) ==
            std::vector<int>{0, 4});
    // Flip in a non-admissible block throws (n_up != N/2).
    REQUIRE_THROWS(allowed_two_S_in_block(6, 2, -1, 0));
}

TEST_CASE("Lowdin projector: eigenspace, idempotence, orthogonality, trace",
          "[casimir_proj]") {
    const std::uint64_t N = 6, dim = 1ULL << N;
    auto s2 = std::static_pointer_cast<const ed::matvec::MatVecOperator>(
        std::shared_ptr<::Operator>(make_S2_carrier(N)));
    const auto towers = allowed_two_S_in_block(static_cast<int>(N));

    for (int ts : towers) {
        LowdinS2Projector P(s2, ts, towers);
        const double lam = 0.25 * ts * (ts + 2);

        auto v = random_vector(dim, 7 + ts);
        P.project(v.data(), dim);
        REQUIRE(norm_of(v) > 1e-8);  // random vectors overlap every tower
        REQUIRE(eigen_residual(*s2, v, lam) < 1e-10);

        // Idempotence.
        auto w = v;
        P.project(w.data(), dim);
        double d2 = 0.0;
        for (std::uint64_t i = 0; i < dim; ++i) d2 += std::norm(w[i] - v[i]);
        REQUIRE(std::sqrt(d2) / norm_of(v) < 1e-10);

        // Orthogonality against every other tower.
        for (int ts2 : towers) {
            if (ts2 == ts) continue;
            LowdinS2Projector P2(s2, ts2, towers);
            auto u = v;
            P2.project(u.data(), dim);
            REQUIRE(norm_of(u) < 1e-9 * norm_of(v));
        }

        // trace(P_S) == (2S+1) * M(N, S): project unit vectors, sum the
        // diagonal.
        double trace = 0.0;
        std::vector<Cx> e(dim);
        for (std::uint64_t j = 0; j < dim; ++j) {
            std::fill(e.begin(), e.end(), Cx(0.0, 0.0));
            e[j] = Cx(1.0, 0.0);
            P.project(e.data(), dim);
            trace += e[j].real();
        }
        const double expected =
            static_cast<double>((ts + 1) *
                                multiplet_count(static_cast<int>(N), ts));
        REQUIRE(std::abs(trace - expected) < 1e-8);
    }
}

TEST_CASE("Lowdin projector matches the dense eigenbasis projector",
          "[casimir_proj]") {
    const std::uint64_t N = 5, dim = 1ULL << N;
    auto carrier = make_S2_carrier(N);
    auto s2 = std::static_pointer_cast<const ed::matvec::MatVecOperator>(
        std::shared_ptr<::Operator>(carrier));

    Eigen::MatrixXcd M(dim, dim);
    {
        std::vector<Cx> e(dim), col(dim);
        for (std::uint64_t j = 0; j < dim; ++j) {
            std::fill(e.begin(), e.end(), Cx(0.0, 0.0));
            e[j] = Cx(1.0, 0.0);
            s2->apply(e.data(), col.data(), dim);
            for (std::uint64_t i = 0; i < dim; ++i) M(i, j) = col[i];
        }
    }
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> es(M);

    const auto towers = allowed_two_S_in_block(static_cast<int>(N));
    const int ts = 1;  // S = 1/2 (odd N)
    const double lam = 0.25 * ts * (ts + 2);
    LowdinS2Projector P(s2, ts, towers);

    auto v = random_vector(dim, 42);
    auto pv = v;
    P.project(pv.data(), dim);

    // Dense reference: sum of |u><u| over eigenvectors with eigenvalue lam.
    Eigen::VectorXcd vin(dim), pref = Eigen::VectorXcd::Zero(dim);
    for (std::uint64_t i = 0; i < dim; ++i) vin(i) = v[i];
    for (Eigen::Index c = 0; c < es.eigenvalues().size(); ++c) {
        if (std::abs(es.eigenvalues()[c] - lam) < 1e-8) {
            const auto u = es.eigenvectors().col(c);
            pref += u * (u.adjoint() * vin);
        }
    }
    double d2 = 0.0;
    for (std::uint64_t i = 0; i < dim; ++i) d2 += std::norm(pv[i] - pref(i));
    REQUIRE(std::sqrt(d2) / pref.norm() < 1e-9);
}

TEST_CASE("fixed-Sz composition: trace(P_S) == M(N,S) per Sz sector",
          "[casimir_proj]") {
    const std::uint64_t N = 8;
    const int n_up = 3;  // Sz = -1, dim = C(8,3) = 56
    auto s2sz = std::make_shared<FixedSzOperator>(N, 0.5f, n_up);
    s2sz->copyTermsFrom(*make_S2_carrier(N));
    const std::uint64_t dim = s2sz->dim();
    REQUIRE(dim == 56);

    const auto towers = allowed_two_S_in_block(static_cast<int>(N), n_up);
    auto s2 =
        std::static_pointer_cast<const ed::matvec::MatVecOperator>(s2sz);
    for (int ts : towers) {
        LowdinS2Projector P(s2, ts, towers);
        const double lam = 0.25 * ts * (ts + 2);
        auto v = random_vector(dim, 100 + ts);
        P.project(v.data(), dim);
        REQUIRE(eigen_residual(*s2, v, lam) < 1e-10);

        double trace = 0.0;
        std::vector<Cx> e(dim);
        for (std::uint64_t j = 0; j < dim; ++j) {
            std::fill(e.begin(), e.end(), Cx(0.0, 0.0));
            e[j] = Cx(1.0, 0.0);
            P.project(e.data(), dim);
            trace += e[j].real();
        }
        // One state per multiplet at any admissible Sz.
        const double expected = static_cast<double>(
            multiplet_count(static_cast<int>(N), ts));
        REQUIRE(std::abs(trace - expected) < 1e-8);
    }
}

TEST_CASE("CasimirProjectedOperator preserves H on the tower and scrubs "
          "drift", "[casimir_proj]") {
    const std::uint64_t N = 6;
    const int n_up = 3;
    auto h = std::make_shared<FixedSzOperator>(N, 0.5f, n_up);
    add_heisenberg_ring(*h, N);
    auto s2sz = std::make_shared<FixedSzOperator>(N, 0.5f, n_up);
    s2sz->copyTermsFrom(*make_S2_carrier(N));
    const std::uint64_t dim = h->dim();

    const auto towers = allowed_two_S_in_block(static_cast<int>(N), n_up);
    auto s2 =
        std::static_pointer_cast<const ed::matvec::MatVecOperator>(s2sz);
    auto proj = std::make_shared<const LowdinS2Projector>(s2, 0, towers);
    CasimirProjectedOperator wrapped(
        std::static_pointer_cast<const ed::matvec::MatVecOperator>(h), proj,
        /*reproject_freq=*/1);

    // Seed preparation lands in the S = 0 tower.
    auto v = random_vector(dim, 5);
    const double w0 = wrapped.prepare_start_vector(v.data(), dim);
    REQUIRE(w0 > 1e-8);
    REQUIRE(eigen_residual(*s2, v, 0.0) < 1e-10);

    // On the tower the wrapper action equals plain H (freq = 1 projects
    // every apply; [H, P] = 0 makes that a no-op up to roundoff).
    std::vector<Cx> hv(dim), wv(dim);
    h->apply(v.data(), hv.data(), dim);
    wrapped.apply(v.data(), wv.data(), dim);
    double d2 = 0.0, n2 = 0.0;
    for (std::uint64_t i = 0; i < dim; ++i) {
        d2 += std::norm(wv[i] - hv[i]);
        n2 += std::norm(hv[i]);
    }
    REQUIRE(std::sqrt(d2 / n2) < 1e-10);

    // Drift handling (ghost-shift contract, audit 2026-07-30): contaminate
    // the input with an S = 1 component. The wrapper maps the off-tower
    // part to mu * (that part) -- NOT to zero: annihilating it left the
    // complement as an exact eigenvalue-0 kernel, and Lanczos converged a
    // ghost 0 below any tower whose true minimum is positive. Subtracting
    // mu * dirt from the output must land back in the S = 0 eigenspace,
    // and mu must sit above the block's spectral radius so no
    // lowest-eigenvalue lane can mistake a ghost for physics.
    auto dirt = random_vector(dim, 6);
    LowdinS2Projector P1(s2, 2, towers);
    P1.project(dirt.data(), dim);
    for (std::uint64_t i = 0; i < dim; ++i) v[i] += 0.05 * dirt[i];
    wrapped.apply(v.data(), wv.data(), dim);
    const double mu = wrapped.ghost_shift();
    REQUIRE(mu > 0.0);
    std::vector<Cx> tower_part(dim);
    for (std::uint64_t i = 0; i < dim; ++i) {
        tower_part[i] = wv[i] - mu * 0.05 * dirt[i];
    }
    REQUIRE(eigen_residual(*s2, tower_part, 0.0) < 1e-8);
    h->apply(v.data(), hv.data(), dim);
    REQUIRE(eigen_residual(*s2, hv, 0.0) > 1e-4);

    // The ghost sits ABOVE the tower minimum for the N=6 Heisenberg ring
    // (E0(S=0) ~ -2.803, ||H|| ~ a few): a positively shifted spectrum
    // stays below mu too, by the 2x margin on the power-iteration bound.
    REQUIRE(mu > 2.0);
}
