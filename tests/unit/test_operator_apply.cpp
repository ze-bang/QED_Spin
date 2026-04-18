// =============================================================================
// test_operator_apply
//
// Sanity tests for the matrix-free Operator::apply() path on tiny Heisenberg
// chains. We cross-check against:
//   * analytic spin-1/2 dimer spectrum {-3/4, 1/4, 1/4, 1/4} for N=2,
//   * explicit dense construction via apply-on-basis-states for N=4/6,
//   * hermiticity of the dense matrix,
//   * matrix-vector consistency (apply(v) == Hdense * v) on random v.
// =============================================================================

#include "common/test_harness.h"

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <complex>
#include <memory>
#include <vector>

using namespace ed_tests;

static void test_N2_analytic(TestContext& ctx) {
    auto op = build_heisenberg_chain(/*N=*/2, /*J=*/1.0);
    const uint64_t dim = 1ULL << 2;
    auto ref = reference_from_operator(*op, dim);

    // Analytic: singlet at -3/4, triplet at +1/4 (three-fold).
    std::vector<double> expected = {-0.75, 0.25, 0.25, 0.25};
    check_eigs_close(ctx, ref.eigs, expected, expected.size(), 1e-12,
                     "N=2 Heisenberg dimer spectrum");
}

static void test_N4_dense_consistency(TestContext& ctx) {
    auto op = build_heisenberg_chain(/*N=*/4, /*J=*/1.0);
    const uint64_t dim = 1ULL << 4;
    auto ref = reference_from_operator(*op, dim);

    // Hermiticity: ||H - H^dagger||_F / ||H||_F < eps
    double num = (ref.H - ref.H.adjoint()).norm();
    double den = std::max(ref.H.norm(), 1e-30);
    check(ctx, num / den < 1e-12,
          "N=4 Heisenberg dense H is Hermitian",
          "||H - H^+||/||H|| = " + std::to_string(num / den));

    // trace(H) = sum of Sz_i Sz_j on all basis states averaged over
    // computational basis is deterministic and equals 0 for the pure
    // exchange Hamiltonian on N=4 open chain. Independent check:
    Complex tr = ref.H.trace();
    check(ctx, std::abs(tr) < 1e-12,
          "N=4 Heisenberg trace is 0",
          "tr(H) = " + std::to_string(tr.real()) + " + " +
              std::to_string(tr.imag()) + "i");

    // Matrix-vector consistency on random vectors.
    for (uint64_t seed : {1u, 42u, 7777u}) {
        auto v = random_unit_vector(dim, seed);
        ComplexVector out(dim);
        op->apply(v.data(), out.data(), dim);
        Eigen::VectorXcd vref(dim);
        for (uint64_t i = 0; i < dim; ++i) vref[i] = v[i];
        Eigen::VectorXcd w = ref.H * vref;
        double err = 0.0;
        for (uint64_t i = 0; i < dim; ++i)
            err += std::norm(out[i] - w[i]);
        err = std::sqrt(err);
        check(ctx, err < 1e-10,
              "apply(v) == H_dense * v (seed=" + std::to_string(seed) + ")",
              "||apply(v) - Hv|| = " + std::to_string(err));
    }
}

static void test_N4_pbc_has_more_symmetry(TestContext& ctx) {
    auto op_open = build_heisenberg_chain(/*N=*/4, /*J=*/1.0, /*periodic=*/false);
    auto op_pbc  = build_heisenberg_chain(/*N=*/4, /*J=*/1.0, /*periodic=*/true);
    const uint64_t dim = 1ULL << 4;
    auto ref_open = reference_from_operator(*op_open, dim);
    auto ref_pbc  = reference_from_operator(*op_pbc, dim);

    // The PBC ground state of the 4-site S=1/2 Heisenberg antiferromagnet
    // at J=1 is the singlet with E_0 = -2 exactly (classic textbook result).
    check_near(ctx, ref_pbc.eigs.front(), -2.0, 1e-10,
               "N=4 PBC Heisenberg ground state = -2");

    // OBC is strictly higher in energy than PBC (fewer bonds of the same
    // sign); verify the ordering rather than a memorized closed form.
    check(ctx, ref_open.eigs.front() > ref_pbc.eigs.front() - 1e-12,
          "N=4 OBC ground state >= N=4 PBC ground state",
          "open=" + std::to_string(ref_open.eigs.front()) +
              " pbc=" + std::to_string(ref_pbc.eigs.front()));

    // Total-Sz block structure implies doubly-degenerate multiplets show up.
    // At a minimum, the ground-state degeneracy for PBC N=4 Heisenberg is 1
    // (non-degenerate singlet). Cheap sanity check: gap > 0.
    double gap = ref_pbc.eigs[1] - ref_pbc.eigs[0];
    check(ctx, gap > 1e-10,
          "N=4 PBC has a positive gap above the ground state",
          "gap = " + std::to_string(gap));
}

static void test_zero_coeff_skipped(TestContext& ctx) {
    // A transform with |coefficient| == 0 must not contribute. Build a
    // Hamiltonian and then append a bunch of tiny transforms and verify the
    // spectrum is unchanged up to float precision.
    auto base = build_heisenberg_chain(/*N=*/4, /*J=*/1.0);
    auto modified = build_heisenberg_chain(/*N=*/4, /*J=*/1.0);
    Operator::TransformData t;
    t.op_type = 2; t.site_index = 0; t.op_type_2 = 2;
    t.site_index_2 = 1; t.coefficient = Complex(0.0, 0.0);
    t.is_two_body = true;
    modified->transform_data_.push_back(t);

    const uint64_t dim = 1ULL << 4;
    auto eb = reference_from_operator(*base, dim).eigs;
    auto em = reference_from_operator(*modified, dim).eigs;
    check_eigs_close(ctx, em, eb, eb.size(), 1e-12,
                     "adding a zero-coefficient term does not change spectrum");
}

int main() {
    TestContext ctx("test_operator_apply");
    test_N2_analytic(ctx);
    test_N4_dense_consistency(ctx);
    test_N4_pbc_has_more_symmetry(ctx);
    test_zero_coeff_skipped(ctx);
    return ctx.summary_exit_code();
}
