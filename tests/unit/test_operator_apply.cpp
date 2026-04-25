// =============================================================================
// test_operator_apply (Catch2 v3, P1.8 / audit Q12)
//
// Sanity tests for the matrix-free Operator::apply() path on tiny Heisenberg
// chains. We cross-check against:
//   * analytic spin-1/2 dimer spectrum {-3/4, 1/4, 1/4, 1/4} for N=2,
//   * explicit dense construction via apply-on-basis-states for N=4,
//   * hermiticity of the dense matrix,
//   * matrix-vector consistency (apply(v) == Hdense * v) on random v,
//   * OBC vs PBC ground-state ordering for N=4,
//   * adding a zero-coefficient term does not perturb the spectrum.
// =============================================================================

#include "common/catch2_harness.h"

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <complex>
#include <memory>
#include <vector>

using namespace ed_tests;

TEST_CASE("Operator::apply: N=2 Heisenberg dimer matches analytic spectrum",
          "[operator_apply][analytic]") {
    auto op = build_heisenberg_chain(/*N=*/2, /*J=*/1.0);
    const uint64_t dim = 1ULL << 2;
    auto ref = reference_from_operator(*op, dim);

    // Singlet at -3/4, triplet at +1/4 (three-fold).
    std::vector<double> expected = {-0.75, 0.25, 0.25, 0.25};
    require_eigs_close(ref.eigs, expected, expected.size(), 1e-12,
                       "N=2 dimer spectrum");
}

TEST_CASE("Operator::apply: N=4 OBC dense reference is consistent",
          "[operator_apply][dense]") {
    auto op = build_heisenberg_chain(/*N=*/4, /*J=*/1.0);
    const uint64_t dim = 1ULL << 4;
    auto ref = reference_from_operator(*op, dim);

    SECTION("hermiticity") {
        double num = (ref.H - ref.H.adjoint()).norm();
        double den = std::max(ref.H.norm(), 1e-30);
        INFO("||H - H^+||/||H|| = " << (num / den));
        REQUIRE(num / den < 1e-12);
    }

    SECTION("trace is zero") {
        Complex tr = ref.H.trace();
        INFO("tr(H) = " << tr.real() << " + " << tr.imag() << "i");
        REQUIRE(std::abs(tr) < 1e-12);
    }

    SECTION("matrix-vector consistency on random vectors") {
        for (uint64_t seed : {1u, 42u, 7777u}) {
            auto v = random_unit_vector(dim, seed);
            ComplexVector out(dim);
            op->apply(v.data(), out.data(), dim);
            Eigen::VectorXcd vref(dim);
            for (uint64_t i = 0; i < dim; ++i) vref[i] = v[i];
            Eigen::VectorXcd outref = ref.H * vref;
            ComplexVector outref_v(dim);
            for (uint64_t i = 0; i < dim; ++i) outref_v[i] = outref[i];
            INFO("seed=" << seed
                 << "  ||apply - Hdense*v|| = " << l2_diff(out, outref_v));
            REQUIRE(l2_diff(out, outref_v) < 1e-10);
        }
    }
}

TEST_CASE("Operator::apply: N=4 PBC ground state below OBC ground state",
          "[operator_apply][pbc]") {
    auto op_open = build_heisenberg_chain(/*N=*/4, /*J=*/1.0,
                                          /*periodic=*/false);
    auto op_pbc  = build_heisenberg_chain(/*N=*/4, /*J=*/1.0,
                                          /*periodic=*/true);
    const uint64_t dim = 1ULL << 4;
    auto ref_open = reference_from_operator(*op_open, dim);
    auto ref_pbc  = reference_from_operator(*op_pbc,  dim);

    INFO("open=" << ref_open.eigs.front()
         << " pbc=" << ref_pbc.eigs.front());
    REQUIRE(ref_open.eigs.front() > ref_pbc.eigs.front() - 1e-12);

    double gap = ref_pbc.eigs[1] - ref_pbc.eigs[0];
    INFO("PBC gap = " << gap);
    REQUIRE(gap > 1e-10);
}

TEST_CASE("Operator::apply_real: matches Operator::apply on real Heisenberg",
          "[operator_apply][apply_real][audit-2.1-phase-1]") {
    // Audit §2.1 Phase 1: the real-typed SpMV must be byte-equivalent to the
    // complex SpMV when the operator is real and the input vector is real.
    // We use the dim>=1024 threshold from apply()'s dispatch, so N=10 (dim=1024)
    // exercises both the apply_real direct path and the apply() dispatch path.
    constexpr int N = 10;
    constexpr uint64_t dim = 1ULL << N;
    auto op = build_heisenberg_chain(N, /*J=*/1.0, /*periodic=*/true);

    SECTION("isReal classifies a real Heisenberg chain as real") {
        REQUIRE(op->isReal());
    }

    SECTION("apply_real(re) == real(apply(complex(re)))") {
        for (uint64_t seed : {1u, 31415u, 2718281u}) {
            auto v_complex = random_unit_vector(dim, seed);
            std::vector<double> v_real(dim);
            for (uint64_t i = 0; i < dim; ++i) v_real[i] = v_complex[i].real();
            for (uint64_t i = 0; i < dim; ++i) v_complex[i] = Complex(v_real[i], 0.0);

            std::vector<double> out_real(dim, 0.0);
            op->apply_real(v_real.data(), out_real.data(), dim);

            ComplexVector out_complex(dim);
            op->apply(v_complex.data(), out_complex.data(), dim);

            double diff_sq = 0.0;
            for (uint64_t i = 0; i < dim; ++i) {
                double dr = out_complex[i].real() - out_real[i];
                double di = out_complex[i].imag();
                diff_sq += dr * dr + di * di;
            }
            INFO("seed=" << seed << "  ||apply_real - apply||_2 = " << std::sqrt(diff_sq));
            REQUIRE(std::sqrt(diff_sq) < 1e-12);
        }
    }
}

TEST_CASE("Operator::apply: zero-coefficient term does not change spectrum",
          "[operator_apply][regression]") {
    auto base = build_heisenberg_chain(/*N=*/4, /*J=*/1.0);
    auto modified = build_heisenberg_chain(/*N=*/4, /*J=*/1.0);
    Operator::TransformData t;
    t.op_type = 2; t.site_index = 0; t.op_type_2 = 2;
    t.site_index_2 = 1; t.coefficient = Complex(0.0, 0.0);
    t.is_two_body = true;
    modified->transform_data_.push_back(t);

    const uint64_t dim = 1ULL << 4;
    auto eb = reference_from_operator(*base,     dim).eigs;
    auto em = reference_from_operator(*modified, dim).eigs;
    require_eigs_close(em, eb, eb.size(), 1e-12, "zero-coeff term invariance");
}

TEST_CASE("Operator::apply_row_range matches full apply on row slabs (N=6 PBC)",
          "[operator_apply][row_range][phase3b]") {
    auto op = build_heisenberg_chain(/*N=*/6, /*J=*/1.0, /*periodic=*/true);
    const uint64_t dim = 1ULL << 6;
    ComplexVector v(dim);
    for (uint64_t i = 0; i < dim; ++i) {
        v[i] = Complex(0.01 * static_cast<double>(i % 11),
                       0.02 * static_cast<double>(i % 7));
    }
    ComplexVector y_full(dim);
    op->apply(v.data(), y_full.data(), dim);

    for (uint64_t rb : {uint64_t{0}, uint64_t{13}, uint64_t{31}}) {
        for (uint64_t n : {uint64_t{1}, uint64_t{17}, uint64_t{32}}) {
            if (rb + n > dim) continue;
            ComplexVector y_part(n);
            op->apply_row_range(v.data(), y_part.data(), dim, rb, n);
            for (uint64_t k = 0; k < n; ++k) {
                const double d =
                    std::abs(y_part[k].real() - y_full[rb + k].real()) +
                    std::abs(y_part[k].imag() - y_full[rb + k].imag());
                INFO("rb=" << rb << " n=" << n << " k=" << k << " err=" << d);
                REQUIRE(d < 1e-10);
            }
        }
    }
}
