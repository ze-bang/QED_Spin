// =============================================================================
// test_fixed_sz_operator (Catch2 v3, P1.8 / audit Q12)
//
// Verifies that FixedSzOperator:
//   * has dimension C(N, n_up) for each allowed n_up,
//   * apply() on the reduced basis matches the full Operator::apply()
//     restricted to the same sector (embedding consistency),
//   * the union of sector spectra reproduces the full-Hilbert-space spectrum.
// =============================================================================

#include "common/catch2_harness.h"

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <complex>
#include <memory>
#include <vector>

using namespace ed_tests;

namespace {

uint64_t binom(uint64_t n, uint64_t k) {
    if (k > n) return 0;
    if (k > n - k) k = n - k;
    uint64_t r = 1;
    for (uint64_t i = 1; i <= k; ++i) { r = r * (n - k + i) / i; }
    return r;
}

} // namespace

TEST_CASE("FixedSzOperator dim equals C(N, n_up)", "[fixed_sz][dim]") {
    for (uint64_t N : {2u, 3u, 4u, 6u}) {
        for (int64_t nu = 0; nu <= static_cast<int64_t>(N); ++nu) {
            auto op = build_heisenberg_chain_fixed_sz(N, 1.0, nu);
            uint64_t expected = binom(N, static_cast<uint64_t>(nu));
            INFO("N=" << N << " n_up=" << nu
                 << " expected C(N,n_up)=" << expected
                 << " got=" << op->getFixedSzDim());
            REQUIRE(op->getFixedSzDim() == expected);
        }
    }
}

TEST_CASE("FixedSzOperator sector matches restriction of full H",
          "[fixed_sz][embedding]") {
    const uint64_t N = 4;
    auto full = build_heisenberg_chain(N, 1.0);
    const uint64_t full_dim = 1ULL << N;
    auto full_ref = reference_from_operator(*full, full_dim);

    std::vector<double> union_eigs;
    for (int64_t nu = 0; nu <= static_cast<int64_t>(N); ++nu) {
        auto op = build_heisenberg_chain_fixed_sz(N, 1.0, nu);
        const uint64_t d = op->getFixedSzDim();
        if (d == 0) continue;
        const auto& basis = op->getBasisStates();

        auto sec_ref = reference_from_fixed_sz_operator(*op, d);

        Eigen::MatrixXcd restricted = Eigen::MatrixXcd::Zero(d, d);
        for (uint64_t i = 0; i < d; ++i)
            for (uint64_t j = 0; j < d; ++j)
                restricted(i, j) = full_ref.H(basis[i], basis[j]);

        double err = (sec_ref.H - restricted).norm();
        INFO("nu=" << nu << " ||Hsec - P H P|| = " << err);
        REQUIRE(err < 1e-10);

        for (double e : sec_ref.eigs) union_eigs.push_back(e);
    }
    require_eigs_close(union_eigs, full_ref.eigs, full_ref.eigs.size(),
                       1e-9, "Union of sector spectra");
}

TEST_CASE("FixedSzOperator sector is Hermitian (N=6, Sz=0)",
          "[fixed_sz][hermiticity]") {
    const uint64_t N = 6;
    const int64_t nu = 3;
    auto op = build_heisenberg_chain_fixed_sz(N, 1.0, nu);
    const uint64_t d = op->getFixedSzDim();
    auto r = reference_from_fixed_sz_operator(*op, d);
    double herm = (r.H - r.H.adjoint()).norm() / std::max(r.H.norm(), 1e-30);
    INFO("||H - H^+||/||H|| = " << herm);
    REQUIRE(herm < 1e-12);
}
