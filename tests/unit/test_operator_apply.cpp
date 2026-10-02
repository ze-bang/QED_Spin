// =============================================================================
// test_operator_apply (Catch2 v3)
//
// Sanity tests for the matrix-free Operator::apply() path on tiny Heisenberg
// chains. We cross-check against:
//   * analytic spin-1/2 dimer spectrum {-3/4, 1/4, 1/4, 1/4} for N=2,
//   * explicit dense construction via apply-on-basis-states for N=4,
//   * hermiticity of the dense matrix,
//   * matrix-vector consistency (apply(v) == Hdense * v) on random v,
//   * OBC vs PBC ground-state ordering for N=4,
//   * adding a zero-coefficient term does not perturb the spectrum,
//   * the walk == the assembled CSR (the two lanes of Operator::apply), real inputs
//     staying real, and the cache-invalidation invariants of the term storage.
// =============================================================================

#include "common/catch2_harness.h"

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdlib>
#include <memory>
#include <random>
#include <vector>

using namespace ed_tests;

// =============================================================================
// Walk == CSR. Operator::apply walks the rows of H (row_walk.h) per apply, or assembles
// them once into a CSR (ED_CSR_FORCE=1, or a dimension below ED_CSR_DIM_MAX); the switch is
// read when the lane is built, so each run below uses a freshly built operator. Both must
// agree across {complex, real} x {1/2/3-body}. Every lane against a dense reference is
// test_row_walk.
// =============================================================================
namespace {

// A rich term mix with nonzero imaginary parts: one-body diagonal and off-diagonal,
// two-body diagonal, mixed and off-diagonal, three-body.
inline void add_rich_complex_terms(Operator& op) {
    // one-body diagonal (Sz)
    op.addOneBodyTerm(/*Sz*/ 2, /*site*/ 0, Complex(0.37, 0.0));
    // one-body off-diagonal (S+/S-) with complex weight
    op.addOneBodyTerm(/*S+*/ 0, /*site*/ 1, Complex(0.2, 0.5));
    op.addOneBodyTerm(/*S-*/ 1, /*site*/ 1, Complex(0.2, -0.5));
    // two-body diagonal (SzSz)
    op.addTwoBodyTerm(2, 0, 2, 1, Complex(0.91, 0.0));
    // mixed two-body (Sz S+/-) complex
    op.addTwoBodyTerm(2, 0, 0, 2, Complex(0.1, 0.3));
    op.addTwoBodyTerm(2, 0, 1, 2, Complex(0.1, -0.3));
    // off-diagonal two-body (S+ S- / S- S+) complex -- Sz-conserving
    op.addTwoBodyTerm(0, 0, 1, 1, Complex(0.45, 0.22));
    op.addTwoBodyTerm(1, 0, 0, 1, Complex(0.45, -0.22));
    // three-body complex (S+_0 S-_1 Sz_2) -- Sz-conserving
    op.addThreeBodyTerm(0, 0, 1, 1, 2, 2, Complex(0.0, 0.7));
    op.addThreeBodyTerm(1, 0, 0, 1, 2, 2, Complex(0.0, -0.7));
}

// The same shapes with real coefficients only.
inline void add_rich_real_terms(Operator& op) {
    op.addOneBodyTerm(2, 0, Complex(0.37, 0.0));
    op.addTwoBodyTerm(2, 0, 2, 1, Complex(0.91, 0.0));
    op.addTwoBodyTerm(0, 0, 1, 1, Complex(0.45, 0.0));
    op.addTwoBodyTerm(1, 0, 0, 1, Complex(0.45, 0.0));
    op.addThreeBodyTerm(0, 0, 1, 1, 2, 2, Complex(0.5, 0.0));
    op.addThreeBodyTerm(1, 0, 0, 1, 2, 2, Complex(0.5, 0.0));
}

// Run op.apply() on the CSR or the walk. ``build`` must return a freshly constructed
// operator, so its lane (and the ED_CSR_FORCE it reads) is built inside this scope.
template <class Build>
inline ComplexVector apply_under_lane(bool csr, Build&& build, const ComplexVector& v) {
    ::setenv("ED_CSR_FORCE", csr ? "1" : "0", 1);
    auto op = build();
    ComplexVector out(v.size(), Complex(0.0, 0.0));
    op->apply(v.data(), out.data(), v.size());
    REQUIRE(std::string(op->full_space_lane()) == (csr ? "csr" : "walk"));
    ::unsetenv("ED_CSR_FORCE");
    return out;
}

template <class Build>
inline std::vector<double> apply_to_real_under_lane(bool csr, Build&& build, const std::vector<double>& v) {
    ComplexVector vc(v.size());
    for (std::size_t i = 0; i < v.size(); ++i) vc[i] = Complex(v[i], 0.0);
    const ComplexVector out = apply_under_lane(csr, build, vc);
    std::vector<double> re(v.size());
    for (std::size_t i = 0; i < v.size(); ++i) re[i] = out[i].real();
    return re;
}

}  // namespace

TEST_CASE("Operator::apply: the walk equals the CSR (complex, 1/2/3-body)",
          "[operator_apply][equivalence]") {
    constexpr uint64_t N   = 10;
    constexpr uint64_t dim = 1ULL << N;
    auto build = [] {
        auto op = build_heisenberg_chain(N, /*J=*/1.0, /*periodic=*/true);
        add_rich_complex_terms(*op);
        return op;
    };
    for (uint64_t seed : {1u, 42u, 90210u}) {
        auto v = random_unit_vector(dim, seed);
        auto y_csr  = apply_under_lane(/*csr=*/true,  build, v);
        auto y_walk = apply_under_lane(/*csr=*/false, build, v);
        INFO("seed=" << seed << "  ||walk - csr|| = " << l2_diff(y_walk, y_csr));
        REQUIRE(l2_diff(y_walk, y_csr) < 1e-12);
    }
}

TEST_CASE("Operator::apply: the walk equals the CSR on real input (Full)",
          "[operator_apply][equivalence][real_input]") {
    SECTION("Full basis") {
        constexpr uint64_t N   = 10;
        constexpr uint64_t dim = 1ULL << N;
        auto build = [] {
            auto op = build_heisenberg_chain(N, 1.0, /*periodic=*/true);
            add_rich_real_terms(*op);
            return op;
        };
        std::vector<double> v(dim);
        std::mt19937_64 g(2024);
        std::normal_distribution<double> nd(0, 1);
        for (auto& x : v) x = nd(g);
        auto y_csr  = apply_to_real_under_lane(true,  build, v);
        auto y_walk = apply_to_real_under_lane(false, build, v);
        double s = 0.0;
        for (uint64_t i = 0; i < dim; ++i) {
            double d = y_walk[i] - y_csr[i];
            s += d * d;
        }
        REQUIRE(std::sqrt(s) < 1e-12);
    }
}

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

TEST_CASE("Operator::apply: a real H keeps real vectors real",
          "[operator_apply][real_input][audit-2.1-phase-1]") {
    // For real H, H(v + iu) = Hv + iHu, and Hv has no imaginary part at all.
    constexpr int N = 10;
    constexpr uint64_t dim = 1ULL << N;
    auto op = build_heisenberg_chain(N, /*J=*/1.0, /*periodic=*/true);

    SECTION("isReal classifies a real Heisenberg chain as real") {
        REQUIRE(op->isReal());
    }

    SECTION("H(v + iu) == Hv + i Hu") {
        for (uint64_t seed : {1u, 31415u, 2718281u}) {
            const auto w = random_unit_vector(dim, seed);
            ComplexVector v(dim), u(dim);
            for (uint64_t i = 0; i < dim; ++i) {
                v[i] = Complex(w[i].real(), 0.0);
                u[i] = Complex(w[i].imag(), 0.0);
            }
            ComplexVector hw(dim), hv(dim), hu(dim);
            op->apply(w.data(), hw.data(), dim);
            op->apply(v.data(), hv.data(), dim);
            op->apply(u.data(), hu.data(), dim);

            double diff_sq = 0.0, imag_sq = 0.0;
            for (uint64_t i = 0; i < dim; ++i) {
                diff_sq += std::norm(hw[i] - (hv[i] + Complex(0.0, 1.0) * hu[i]));
                imag_sq += hv[i].imag() * hv[i].imag() + hu[i].imag() * hu[i].imag();
            }
            INFO("seed=" << seed << "  ||H(v+iu) - (Hv + iHu)||_2 = " << std::sqrt(diff_sq));
            REQUIRE(std::sqrt(diff_sq) < 1e-12);
            REQUIRE(imag_sq == 0.0);
        }
    }
}

TEST_CASE("Operator: isReal() cache invalidates when a complex coefficient "
          "is added between calls",
          "[operator_apply][regression][s0]") {
    // ``isReal()`` caches its first answer in ``real_check_done_``, so a
    // real-only operator that is queried once and then gets a complex
    // coefficient would keep claiming real -- routing a subsequent
    // solve through the real fast path with the wrong matvec. Adding a
    // record must reset the isReal() cache too.
    auto op = build_heisenberg_chain(/*N=*/4, /*J=*/1.0);
    REQUIRE(op->isReal());

    Operator::TransformData t;
    t.op_type      = 2;
    t.site_index   = 0;
    t.coefficient  = Complex(0.0, 0.5);  // pure imaginary
    t.is_two_body  = false;
    op->add_record(t);
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
    modified->add_record(t);

    const uint64_t dim = 1ULL << 4;
    auto eb = reference_from_operator(*base,     dim).eigs;
    auto em = reference_from_operator(*modified, dim).eigs;
    require_eigs_close(em, eb, eb.size(), 1e-12, "zero-coeff term invariance");
}

// =============================================================================
// Cache invalidation: a record added between two applies (the first one built the SoA
// cache) must take part in the second.
// =============================================================================
TEST_CASE("Operator: a record added between applies is honoured",
          "[operator_apply][regression][s0]") {
    auto op = build_heisenberg_chain(/*N=*/4, /*J=*/1.0);
    const uint64_t dim = 1ULL << 4;

    std::vector<Complex> x(dim, Complex(0.0, 0.0));
    x[dim - 1] = Complex(1.0, 0.0);  // |1111>: every spin up (a set bit is up)
    std::vector<Complex> y_before(dim, Complex(0.0, 0.0));
    op->apply(x.data(), y_before.data(), dim);

    // Add a non-trivial diagonal term on site 0; |1111> picks up
    // +spin * coeff under Sz_0.
    Operator::TransformData t;
    t.op_type      = 2;     // Sz
    t.site_index   = 0;
    t.coefficient  = Complex(3.14159, 0.0);
    t.is_two_body  = false;
    op->add_record(t);

    std::vector<Complex> y_after(dim, Complex(0.0, 0.0));
    op->apply(x.data(), y_after.data(), dim);

    // The contributions on the diagonal entry y[|1111>] must differ by
    // exactly +spin * 3.14159 = +0.5 * 3.14159 = +1.57080 (spin-1/2).
    const double expected_delta = 0.5 * 3.14159;
    const double actual_delta   = std::real(y_after[dim - 1] - y_before[dim - 1]);
    INFO("expected delta " << expected_delta << ", got " << actual_delta);
    REQUIRE(std::abs(actual_delta - expected_delta) < 1e-12);
}

// =============================================================================
// Complex three-body coefficients survive both lanes: taking only ``.real()`` would
// silently drop i S+_0 S-_1 Sz_2. Applied by the walk and the CSR.
// =============================================================================
TEST_CASE("Operator::apply: a three-body term keeps its complex coefficient",
          "[matvec][kernel][regression][s0][three_body]") {
    constexpr std::uint64_t N   = 3;
    constexpr std::uint64_t dim = 1ULL << N;
    auto build = [] {
        auto op = std::make_unique<Operator>(std::uint64_t{N}, /*spin_l=*/0.5f);
        op->addThreeBodyTerm(/*op1=*/0, /*site1=*/0, /*op2=*/1, /*site2=*/1,
                             /*op3=*/2, /*site3=*/2, /*coeff=*/Complex(0.0, 1.0));
        return op;
    };
    std::mt19937 gen(12345);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    ComplexVector v(dim);
    for (auto& z : v) z = Complex(dist(gen), dist(gen));

    const auto y_csr  = apply_under_lane(/*csr=*/true,  build, v);
    const auto y_gather = apply_under_lane(/*csr=*/false, build, v);
    INFO("||walk - csr||_2 = " << l2_diff(y_gather, y_csr));
    REQUIRE(l2_diff(y_gather, y_csr) < 1e-12);

    // The pure-imaginary coupling carried through (a .real() truncation would
    // leave the result identically zero).
    double norm_sq = 0.0;
    for (const auto& z : y_gather) norm_sq += std::norm(z);
    REQUIRE(std::sqrt(norm_sq) > 1e-12);
}
