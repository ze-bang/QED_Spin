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

// Snap an S^2 eigenvalue to 2S when it is an allowed S(S+1) for n_sites
// spin-1/2 sites (2S of the parity of N, 2S <= N); -1 otherwise.
int snap_two_S(double s2, int n_sites) {
    for (int ts = n_sites % 2; ts <= n_sites; ts += 2)
        if (std::abs(s2 - 0.25 * ts * (ts + 2)) <= 1e-6) return ts;
    return -1;
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

