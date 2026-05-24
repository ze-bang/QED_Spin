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
#include <random>
#include <vector>

#include <ed/matvec/term_kernels_gather.h>
#include <ed/matvec/term_storage.h>

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

TEST_CASE("Operator: isReal() cache invalidates when a complex coefficient "
          "is added between calls",
          "[operator_apply][regression][s0]") {
    // Regression test mirroring the structural-audit Python-binding
    // finding: ``isReal()`` caches its first answer in ``real_check_done_``,
    // so a real-only operator that gets queried once and then has a
    // complex coefficient pushed in would keep claiming real -- routing
    // a subsequent lanczos() call through the lanczos_real fast path
    // with the wrong matvec. The Python bindings now call
    // ``invalidateMatrixCaches()`` from every ``op_add_*`` helper, but
    // this test exercises the underlying invariant: an explicit
    // ``invalidateMatrixCaches()`` after a direct AoS push must reset
    // the isReal() cache too.
    auto op = build_heisenberg_chain(/*N=*/4, /*J=*/1.0);
    REQUIRE(op->isReal());

    Operator::TransformData t;
    t.op_type      = 2;
    t.site_index   = 0;
    t.coefficient  = Complex(0.0, 0.5);  // pure imaginary
    t.is_two_body  = false;
    op->transform_data_.push_back(t);
    op->invalidateMatrixCaches();

    INFO("After pushing an imaginary coefficient, isReal() must return false.");
    REQUIRE_FALSE(op->isReal());
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

// =============================================================================
// S0 regression: AoS cache invalidation (audit S0 #2, May 2026).
//
// Sequence:
//   1. build the operator
//   2. call apply() (sets terms_fresh_ = true, populates SoA cache)
//   3. push a new term directly into transform_data_ (bypassing the
//      typed setters and any explicit invalidateMatrixCaches() call)
//   4. call apply() again
//
// Before the May 2026 size-tracking fix, step (4) returned the SoA cache
// from step (2), silently dropping the term added in step (3). The fix
// records the AoS sizes at every commit and rebuilds the SoA cache when
// they diverge from the live sizes. This regression test asserts that
// the new term participates in the second apply.
// =============================================================================
TEST_CASE("Operator: direct AoS push between applies is honoured "
          "(size-tracking cache invalidation)",
          "[operator_apply][regression][s0]") {
    auto op = build_heisenberg_chain(/*N=*/4, /*J=*/1.0);
    const uint64_t dim = 1ULL << 4;

    std::vector<Complex> x(dim, Complex(0.0, 0.0));
    x[0] = Complex(1.0, 0.0);  // |0000>
    std::vector<Complex> y_before(dim, Complex(0.0, 0.0));
    op->apply(x.data(), y_before.data(), dim);

    // Add a non-trivial diagonal term on site 0; |0000> picks up
    // +spin * coeff under Sz_0.
    Operator::TransformData t;
    t.op_type      = 2;     // Sz
    t.site_index   = 0;
    t.coefficient  = Complex(3.14159, 0.0);
    t.is_two_body  = false;
    op->transform_data_.push_back(t);

    std::vector<Complex> y_after(dim, Complex(0.0, 0.0));
    op->apply(x.data(), y_after.data(), dim);

    // The contributions on the diagonal entry y[|0000>] must differ by
    // exactly +spin * 3.14159 = +0.5 * 3.14159 = +1.57080 (spin-1/2).
    const double expected_delta = 0.5 * 3.14159;
    const double actual_delta   = std::real(y_after[0] - y_before[0]);
    INFO("expected delta " << expected_delta << ", got " << actual_delta);
    REQUIRE(std::abs(actual_delta - expected_delta) < 1e-12);
}

// =============================================================================
// S0 regression: FixedSzOperator::apply_real dispatch through Operator&
// (audit S0 #4, May 2026).
//
// Before the May 2026 fix, Operator::apply_real was non-virtual, so a
// FixedSzOperator bound through an Operator& reference would slice to the
// base apply_real that checks the FULL 2^N dim. Marking apply_real
// virtual on Operator (and override on FixedSzOperator) ensures that
// virtual dispatch picks up the projected-Sz check and SoA backend.
//
// This test asserts that a vector sized to the projected dim works
// through the base reference; before the fix it would throw "input/output
// vector size mismatch" (size != 2^N).
// =============================================================================
TEST_CASE("FixedSzOperator::apply_real dispatches virtually through "
          "Operator& (no slicing)",
          "[operator_apply][regression][s0][fixed_sz]") {
    constexpr uint64_t N = 4;
    auto base = build_heisenberg_chain(N, /*J=*/1.0);
    // Need a real-coupling Hamiltonian for apply_real; Heisenberg is real.
    REQUIRE(base->isReal());

    // Project to Sz = N/2 sector. binom(4, 2) = 6.
    // FixedSzOperator ctor signature: (n_bits, spin_l, n_up).
    FixedSzOperator fz(N, /*spin=*/0.5f, /*n_up=*/2);
    fz.transform_data_  = base->transform_data_;
    fz.three_body_data_ = base->three_body_data_;
    fz.invalidateMatrixCaches();
    const std::uint64_t sec_dim = fz.getFixedSzDim();
    REQUIRE(sec_dim == 6);

    std::vector<double> x(sec_dim, 0.0), y(sec_dim, 0.0);
    x[0] = 1.0;
    Operator& base_ref = static_cast<Operator&>(fz);
    // Pre-fix: this would throw because base apply_real checks
    // size == 2^N == 16, not the projected sec_dim == 6.
    REQUIRE_NOTHROW(base_ref.apply_real(x.data(), y.data(), sec_dim));
}

// =============================================================================
// S0 regression: GATHER three-body kernel respects complex coefficients
// (audit S0 #5, May 2026).
//
// Before the May 2026 fix, ``gather_row`` collapsed a three-body
// coupling to ``coefficient.real()``, silently dropping the imaginary
// part. The distributed CPU SpMV reaches this kernel via
// ``DistributedOperator::apply``, so any Hamiltonian with a complex
// three-body term gave a different answer between the serial CPU
// (SCATTER) and distributed CPU (GATHER) paths.
//
// We exercise gather_row directly against the SCATTER ``apply_terms``
// for a tiny 3-site Hamiltonian with an imaginary-only 3-body coupling:
// S+_0 S-_1 Sz_2 with i. Both kernels must produce the same y[r] for
// every r.
// =============================================================================
TEST_CASE("matvec::kernel::gather_row: complex three-body matches SCATTER",
          "[matvec][kernel][regression][s0][three_body]") {
    using namespace ed::matvec;
    constexpr std::uint64_t N   = 3;
    constexpr std::uint64_t dim = 1ULL << N;
    const double spin           = 0.5;

    TermStorage T;
    // i * S+_0 S-_1 Sz_2 -- intentionally pure imaginary so the .real()
    // bug zeros out the entire term.
    T.add_three_body(/*op1=*/0, /*site1=*/0,
                     /*op2=*/1, /*site2=*/1,
                     /*op3=*/2, /*site3=*/2,
                     /*coeff=*/Complex(0.0, 1.0));

    // Random input vector.
    std::mt19937 gen(12345);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    std::vector<Complex> v(dim);
    for (auto& z : v) z = Complex(dist(gen), dist(gen));

    // ----- GATHER path -----------------------------------------------
    std::vector<Complex> y_gather(dim, Complex(0.0, 0.0));
    auto get_v = [&](std::uint64_t c) -> Complex { return v[c]; };
    for (std::uint64_t r = 0; r < dim; ++r) {
        y_gather[r] = kernel::gather_row(r, v[r], T, spin, get_v);
    }

    // ----- SCATTER path (drive ``Operator::apply``) ------------------
    auto op = std::make_unique<Operator>(N, /*spin_l=*/0.5f);
    op->addThreeBodyTerm(/*op1=*/0, /*site1=*/0,
                         /*op2=*/1, /*site2=*/1,
                         /*op3=*/2, /*site3=*/2,
                         /*coeff=*/Complex(0.0, 1.0));
    std::vector<Complex> y_scatter(dim, Complex(0.0, 0.0));
    op->apply(v.data(), y_scatter.data(), dim);

    double diff_sq = 0.0;
    for (std::uint64_t r = 0; r < dim; ++r) {
        const Complex d = y_gather[r] - y_scatter[r];
        diff_sq += std::norm(d);
    }
    INFO("||y_gather - y_scatter||_2 = " << std::sqrt(diff_sq));
    REQUIRE(std::sqrt(diff_sq) < 1e-12);

    // Also assert the imaginary part actually carried through (pre-fix
    // y_gather would have been identically zero for this pure-imag
    // coupling).
    double scatter_norm_sq = 0.0;
    for (std::uint64_t r = 0; r < dim; ++r) {
        scatter_norm_sq += std::norm(y_scatter[r]);
    }
    REQUIRE(std::sqrt(scatter_norm_sq) > 1e-12);
}
