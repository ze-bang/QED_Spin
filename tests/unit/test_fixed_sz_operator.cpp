// =============================================================================
// test_fixed_sz_operator
//
// Verifies that FixedSzOperator:
//   * has dimension C(N, n_up) for each allowed n_up,
//   * apply() on the reduced basis matches the full Operator::apply()
//     restricted to the same sector (embedding consistency),
//   * the union of sector spectra reproduces the full-Hilbert-space spectrum.
// =============================================================================

#include "common/test_harness.h"

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <complex>
#include <memory>
#include <vector>

using namespace ed_tests;

static uint64_t binom(uint64_t n, uint64_t k) {
    if (k > n) return 0;
    if (k > n - k) k = n - k;
    uint64_t r = 1;
    for (uint64_t i = 1; i <= k; ++i) { r = r * (n - k + i) / i; }
    return r;
}

static void test_dimension(TestContext& ctx) {
    for (uint64_t N : {2u, 3u, 4u, 6u}) {
        for (int64_t nu = 0; nu <= static_cast<int64_t>(N); ++nu) {
            auto op = build_heisenberg_chain_fixed_sz(N, 1.0, nu);
            uint64_t expected = binom(N, static_cast<uint64_t>(nu));
            check(ctx, op->getFixedSzDim() == expected,
                  "FixedSz dim = C(" + std::to_string(N) + "," +
                      std::to_string(nu) + ") = " + std::to_string(expected),
                  "got " + std::to_string(op->getFixedSzDim()));
        }
    }
}

static void test_sector_matches_full(TestContext& ctx) {
    const uint64_t N = 4;
    auto full = build_heisenberg_chain(N, 1.0);
    const uint64_t full_dim = 1ULL << N;

    // Full reference from apply-to-dense.
    auto full_ref = reference_from_operator(*full, full_dim);

    // For each sector, build both the sector-restricted dense matrix from
    // FixedSzOperator::apply and the same restriction extracted from the
    // full dense matrix via the sector's basis states. They must agree.
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
        check(ctx, err < 1e-10,
              "N=" + std::to_string(N) + " sector nu=" +
                  std::to_string(nu) + " matches restriction of full H",
              "|Hsec - P H P| = " + std::to_string(err));

        for (double e : sec_ref.eigs) union_eigs.push_back(e);
    }
    check_eigs_close(ctx, union_eigs, full_ref.eigs, full_ref.eigs.size(),
                     1e-9, "Union of sector spectra = full spectrum");
}

static void test_sector_hermiticity(TestContext& ctx) {
    // Exercise a slightly larger problem to stress the radix-sort + binary
    // search paths inside FixedSzOperator::apply().
    const uint64_t N = 6;
    const int64_t nu = 3;
    auto op = build_heisenberg_chain_fixed_sz(N, 1.0, nu);
    const uint64_t d = op->getFixedSzDim();
    auto r = reference_from_fixed_sz_operator(*op, d);
    double herm = (r.H - r.H.adjoint()).norm() / std::max(r.H.norm(), 1e-30);
    check(ctx, herm < 1e-12,
          "N=6 Sz=0 Heisenberg sector is Hermitian",
          "||H - H^+||/||H|| = " + std::to_string(herm));
}

int main() {
    TestContext ctx("test_fixed_sz_operator");
    test_dimension(ctx);
    test_sector_matches_full(ctx);
    test_sector_hermiticity(ctx);
    return ctx.summary_exit_code();
}
