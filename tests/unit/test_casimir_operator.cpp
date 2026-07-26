// =============================================================================
// tests/unit/test_casimir_operator.cpp
//
// Stage 12a of the SU(2) rollout: the S^2_tot Casimir in the TermStorage
// schema (include/ed/operators/casimir.h).
//
// Pinned:
//   * dense S^2 from the carrier == the algebraic reference
//     Sz^2 + Sz + S-_tot S+_tot, element by element (N = 4, 5);
//   * spectrum is exactly { S(S+1) } with multiplicity (2S+1) * M(N,S),
//     M(N,S) = C(N, N/2-S) - C(N, N/2-S-1)  (multiplet counting);
//   * snap_two_S: parity of N, |Sz| floor, flip-parity mask, tolerance;
//   * s2_expectation certifies eigenvectors (tiny residual) and refuses
//     a 50/50 mix of different-S eigenvectors (large residual).
// =============================================================================
#include "common/catch2_harness.h"

#include <Eigen/Dense>

#include <ed/core/operator.h>
#include <ed/operators/casimir.h>

#include <cmath>
#include <complex>
#include <cstdint>
#include <map>
#include <vector>

using Cx = std::complex<double>;
using ed::ops::make_S2_carrier;
using ed::ops::s2_eigenvalue_of_two_S;
using ed::ops::s2_expectation;
using ed::ops::snap_two_S;

namespace {

// Dense matrix of an operator by applying it to unit vectors.
Eigen::MatrixXcd dense_of(const ed::matvec::MatVecOperator& op) {
    const std::uint64_t dim = op.dim();
    Eigen::MatrixXcd M(dim, dim);
    std::vector<Cx> e(dim), col(dim);
    for (std::uint64_t j = 0; j < dim; ++j) {
        std::fill(e.begin(), e.end(), Cx(0.0, 0.0));
        e[j] = Cx(1.0, 0.0);
        op.apply(e.data(), col.data(), dim);
        for (std::uint64_t i = 0; i < dim; ++i) M(i, j) = col[i];
    }
    return M;
}

// Algebraic reference: S^2 = Sz^2 + Sz + S^-_tot S^+_tot, built directly
// from the bit convention (bit 0 = UP).
Eigen::MatrixXcd dense_S2_reference(std::uint64_t N) {
    const std::uint64_t dim = 1ULL << N;
    // S+_tot: for each DOWN site, clear the bit (amplitude 1).
    Eigen::MatrixXcd Sp = Eigen::MatrixXcd::Zero(dim, dim);
    for (std::uint64_t s = 0; s < dim; ++s) {
        for (std::uint64_t j = 0; j < N; ++j) {
            if ((s >> j) & 1ULL) Sp(s ^ (1ULL << j), s) += 1.0;
        }
    }
    Eigen::MatrixXcd Sz = Eigen::MatrixXcd::Zero(dim, dim);
    for (std::uint64_t s = 0; s < dim; ++s) {
        const int n_dn = __builtin_popcountll(s);
        Sz(s, s) = 0.5 * static_cast<double>(N - n_dn) - 0.5 * n_dn;
    }
    return Sz * Sz + Sz + Sp.adjoint() * Sp;
}

std::uint64_t binom(std::uint64_t n, std::int64_t k) {
    if (k < 0 || k > static_cast<std::int64_t>(n)) return 0;
    std::uint64_t r = 1;
    for (std::int64_t i = 0; i < k; ++i) r = r * (n - i) / (i + 1);
    return r;
}

// Multiplet count M(N, S) for two_S = 2S.
std::uint64_t multiplet_count_ref(std::uint64_t N, int two_S) {
    const std::int64_t k = (static_cast<std::int64_t>(N) - two_S) / 2;
    return binom(N, k) - binom(N, k - 1);
}

}  // namespace

TEST_CASE("S^2 carrier matches the algebraic reference", "[casimir]") {
    for (std::uint64_t N : {2ULL, 4ULL, 5ULL}) {
        auto op = make_S2_carrier(N);
        const Eigen::MatrixXcd A = dense_of(*op);
        const Eigen::MatrixXcd R = dense_S2_reference(N);
        REQUIRE((A - R).cwiseAbs().maxCoeff() < 1e-12);
    }
}

TEST_CASE("S^2 spectrum is {S(S+1)} with multiplet-counting multiplicities",
          "[casimir]") {
    for (std::uint64_t N : {4ULL, 6ULL}) {
        auto op = make_S2_carrier(N);
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> es(dense_of(*op));
        REQUIRE(es.info() == Eigen::Success);

        std::map<int, std::uint64_t> mult;  // two_S -> count
        for (Eigen::Index i = 0; i < es.eigenvalues().size(); ++i) {
            const int ts = snap_two_S(es.eigenvalues()[i],
                                      static_cast<int>(N));
            REQUIRE(ts >= 0);  // every eigenvalue snaps
            ++mult[ts];
        }
        std::uint64_t total = 0;
        for (const auto& [ts, count] : mult) {
            const std::uint64_t expected =
                (ts + 1) * multiplet_count_ref(N, ts);  // (2S+1) * M(N,S)
            REQUIRE(count == expected);
            total += count;
        }
        REQUIRE(total == (1ULL << N));
    }
}

TEST_CASE("append_S2_total and the carrier agree term-for-term", "[casimir]") {
    const std::uint64_t N = 5;
    ed::matvec::TermStorage t;
    ed::ops::append_S2_total(t, N);
    auto op = make_S2_carrier(N);
    const auto& ct = op->getTerms();
    REQUIRE(t.diag_two_body.size() == ct.diag_two_body.size());
    REQUIRE(t.offdiag_two_body.size() == ct.offdiag_two_body.size());
    REQUIRE(t.diag_one_body.empty());
    REQUIRE(t.offdiag_one_body.empty());
    REQUIRE(t.mixed_two_body.empty());
    REQUIRE(t.three_body.empty());
    // N identity shifts + N(N-1)/2 zz pairs; 2 ladder terms per pair.
    REQUIRE(t.diag_two_body.size() == N + N * (N - 1) / 2);
    REQUIRE(t.offdiag_two_body.size() == N * (N - 1));
}

TEST_CASE("snap_two_S: parity, |Sz| floor, flip mask, tolerance", "[casimir]") {
    // N = 6: allowed two_S in {0, 2, 4, 6}; S(S+1) in {0, 2, 6, 12}.
    REQUIRE(snap_two_S(0.0, 6) == 0);
    REQUIRE(snap_two_S(2.0 + 5e-7, 6) == 2);
    REQUIRE(snap_two_S(2.0 + 5e-3, 6) == -1);      // outside tol
    // Odd N = 5: allowed two_S in {1, 3, 5}; S(S+1) in {0.75, 3.75, 8.75}.
    REQUIRE(snap_two_S(0.75, 5) == 1);
    REQUIRE(snap_two_S(0.0, 5) == -1);             // 0 not allowed for odd N
    // |Sz| floor: N = 6, n_up = 1 -> Sz = -2 -> two_S >= 4.
    REQUIRE(snap_two_S(6.0, 6, /*n_up=*/1) == 4);
    REQUIRE(snap_two_S(2.0, 6, /*n_up=*/1) == -1);  // S = 1 < |Sz|
    // Flip parity at half filling, N = 6: X eigenvalue (-1)^{N/2 - S}.
    // two_S = 6 -> (6-6)/2 = 0 even -> parity 0; two_S = 4 -> odd -> parity 1.
    REQUIRE(snap_two_S(12.0, 6, 3, /*flip_parity=*/0) == 6);
    REQUIRE(snap_two_S(12.0, 6, 3, /*flip_parity=*/1) == -1);
    REQUIRE(snap_two_S(6.0, 6, 3, /*flip_parity=*/1) == 4);
}

TEST_CASE("s2_expectation certifies pure-S vectors and flags mixtures",
          "[casimir]") {
    const std::uint64_t N = 4;
    auto op = make_S2_carrier(N);
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> es(dense_of(*op));
    const auto& evals = es.eigenvalues();
    const auto& evecs = es.eigenvectors();
    const std::uint64_t dim = 1ULL << N;

    // Pure eigenvector: expectation == eigenvalue, residual ~ 0.
    {
        std::vector<Cx> v(dim);
        for (std::uint64_t i = 0; i < dim; ++i) v[i] = evecs(i, 0);
        double res = -1.0;
        const double s2 = s2_expectation(*op, v.data(), dim, &res);
        REQUIRE(std::abs(s2 - evals[0]) < 1e-10);
        REQUIRE(res < ed::ops::kS2CertifyTol);
    }
    // 50/50 mixture of the lowest (S=0) and highest (S=N/2) eigenvectors:
    // expectation is between the S(S+1) points and the residual is O(1).
    {
        const Eigen::Index last = evals.size() - 1;
        REQUIRE(std::abs(evals[0] - evals[last]) > 1.0);
        std::vector<Cx> v(dim);
        const double inv = 1.0 / std::sqrt(2.0);
        for (std::uint64_t i = 0; i < dim; ++i)
            v[i] = inv * (evecs(i, 0) + evecs(i, last));
        double res = -1.0;
        const double s2 = s2_expectation(*op, v.data(), dim, &res);
        REQUIRE(res > 0.1);
        REQUIRE(snap_two_S(s2, static_cast<int>(N)) == -1);
    }
}
