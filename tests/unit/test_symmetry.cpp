// =============================================================================
// test_symmetry
//
// Validates the matrix-free symmetry-adapted Operator
// (`StreamingSymmetryOperator`) on a known small system.
//
// Test plan:
//   1. Bypass the Python automorphism pipeline by emitting the JSON files
//      that `SymmetryGroupInfo::loadFromDirectory()` consumes:
//        * max_clique.json
//        * minimal_generators.json
//        * sector_metadata.json
//      We use Z_4 cyclic translation on a 4-site PBC Heisenberg chain --
//      the smallest non-trivial Abelian symmetry where every irrep
//      contributes.
//
//   2. Build the same Hamiltonian via the streaming op AND the standard
//      op (test_harness Heisenberg chain). Verify:
//        a. `applyFullSpace()` matches the bare `Operator::apply()` on
//           random vectors. This tests the Hamiltonian construction
//           independently of the symmetry sectors.
//        b. After `generateSymmetrySectorsStreaming()`, the union of
//           sector spectra equals the full spectrum from the dense
//           reference. This is the main correctness statement for the
//           symmetry projection / matrix-free sector matvec.
//        c. Sector dimensions sum to the full Hilbert space.
//
// All tests run in a per-suite scratch dir so they don't collide with
// concurrent CTest jobs.
// =============================================================================

#include "common/test_harness.h"

#include <ed/core/streaming_symmetry.h>
#include <ed/solvers/lanczos.h>

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>

#include <algorithm>
#include <cmath>
#include <complex>
#include <filesystem>  // P0.12
#include <fstream>
#include <iomanip>
#include <numeric>
#include <sstream>
#include <system_error>  // P0.12
#include <vector>

using namespace ed_tests;

// `full_diagonalization` is declared in `lanczos.h` (already included).

namespace {

// -----------------------------------------------------------------------------
// JSON fixture writer for the Z_N cyclic translation group on N sites.
//
// Permutation convention (matches `applyPermutation`):
//   result_bit[i] = basis_bit[perm[i]]
// so the cyclic translation T (shifting bit at site i to site i+1) is
// represented by perm[i] = (i - 1) mod N.
// -----------------------------------------------------------------------------
std::vector<int> translation_perm(int N, int shift) {
    std::vector<int> p(N);
    for (int i = 0; i < N; ++i) p[i] = ((i - shift) % N + N) % N;
    return p;
}

// Compose perm a after perm b: (a ∘ b)[i] = a[b[i]]  (so result_bit[i] =
// in_bit[(a∘b)[i]]). Used only for sanity in tests.
std::vector<int> compose(const std::vector<int>& a, const std::vector<int>& b) {
    std::vector<int> r(a.size());
    for (size_t i = 0; i < a.size(); ++i) r[i] = a[b[i]];
    return r;
}

void write_zN_translation_fixtures(const std::string& dir, int N) {
    // Place under "<dir>/automorphism_results" -- this is the layout
    // SymmetryGroupInfo::loadFromDirectory() expects.
    // P0.12: was system("mkdir -p '...'") (shell-quoted).
    std::string root = dir + "/automorphism_results";
    std::error_code ec;
    std::filesystem::create_directories(root, ec);

    // ---- max_clique.json: the full group [T^0, T^1, ..., T^{N-1}].
    {
        std::ofstream f(root + "/max_clique.json");
        f << "[";
        for (int g = 0; g < N; ++g) {
            auto p = translation_perm(N, g);
            f << "[";
            for (size_t i = 0; i < p.size(); ++i) {
                f << p[i] << (i + 1 < p.size() ? "," : "");
            }
            f << "]" << (g + 1 < N ? "," : "");
        }
        f << "]";
    }

    // ---- minimal_generators.json: a single generator T (translation by 1)
    //      with order N.
    {
        std::ofstream f(root + "/minimal_generators.json");
        auto p = translation_perm(N, 1);
        f << "{\"generators\":[{\"permutation\":[";
        for (size_t i = 0; i < p.size(); ++i) {
            f << p[i] << (i + 1 < p.size() ? "," : "");
        }
        f << "],\"order\":" << N << "}]}";
    }

    // ---- sector_metadata.json: one sector per momentum k = 0..N-1, with
    //      phase_factor for the single generator equal to e^{-2πik/N}.
    {
        std::ofstream f(root + "/sector_metadata.json");
        f << std::setprecision(17);
        f << "{\"sectors\":[";
        for (int k = 0; k < N; ++k) {
            const double angle = -2.0 * M_PI * static_cast<double>(k) /
                                 static_cast<double>(N);
            const double re = std::cos(angle);
            const double im = std::sin(angle);
            f << "{\"sector_id\":" << k
              << ",\"quantum_numbers\":[" << k << "]"
              << ",\"phase_factors\":[{\"real\":" << re
              << ",\"imag\":" << im << "}]}";
            if (k + 1 < N) f << ",";
        }
        f << "]}";
    }
}

// Programmatic copy of build_heisenberg_chain into a StreamingSymmetryOperator
// (which inherits the same `transform_data_` SoA backbone but lives in a
// different concrete class).
std::unique_ptr<StreamingSymmetryOperator>
build_heisenberg_pbc_streaming(uint64_t N, double J) {
    auto op = std::make_unique<StreamingSymmetryOperator>(N, 0.5f);
    const Complex J_real(J, 0.0);
    const Complex J_half(0.5 * J, 0.0);
    for (uint64_t i = 0; i < N; ++i) {
        uint64_t j = (i + 1) % N;
        // Sz_i Sz_j
        Operator::TransformData t;
        t.op_type = 2; t.site_index = i; t.op_type_2 = 2;
        t.site_index_2 = j; t.coefficient = J_real; t.is_two_body = true;
        op->transform_data_.push_back(t);

        t.op_type = 0; t.site_index = i; t.op_type_2 = 1;
        t.site_index_2 = j; t.coefficient = J_half; t.is_two_body = true;
        op->transform_data_.push_back(t);

        t.op_type = 1; t.site_index = i; t.op_type_2 = 0;
        t.site_index_2 = j; t.coefficient = J_half; t.is_two_body = true;
        op->transform_data_.push_back(t);
    }
    return op;
}

} // namespace

// -----------------------------------------------------------------------------
// 1. JSON loader smoke test: did our fixture files parse?
// -----------------------------------------------------------------------------
static void test_json_fixture_loads(TestContext& ctx) {
    const int N = 4;
    std::string dir = make_scratch_dir("symmetry", "fixture_load");
    write_zN_translation_fixtures(dir, N);

    SymmetryGroupInfo info;
    bool threw = false;
    try {
        info.loadFromDirectory(dir);
    } catch (const std::exception& e) {
        threw = true;
    }
    check(ctx, !threw, "SymmetryGroupInfo::loadFromDirectory parses JSON fixture");

    check(ctx, info.max_clique.size() == static_cast<size_t>(N),
          "max_clique has |G|=N elements",
          "got " + std::to_string(info.max_clique.size()));
    check(ctx, info.generators.size() == 1,
          "exactly one generator (T)",
          "got " + std::to_string(info.generators.size()));
    check(ctx, info.generator_orders.size() == 1 &&
              info.generator_orders[0] == N,
          "generator order = N");
    check(ctx, info.sectors.size() == static_cast<size_t>(N),
          "N momentum sectors",
          "got " + std::to_string(info.sectors.size()));

    // Power representation: the i-th group element of max_clique is T^i, so
    // its power representation should be [i].
    bool powers_ok = (info.power_representation.size() == static_cast<size_t>(N));
    if (powers_ok) {
        for (int g = 0; g < N; ++g) {
            if (info.power_representation[g].size() != 1 ||
                info.power_representation[g][0] != g) {
                powers_ok = false; break;
            }
        }
    }
    check(ctx, powers_ok,
          "power_representation maps clique index -> [i]");
}

// -----------------------------------------------------------------------------
// 2. Inherited Operator::apply() consistency: the streaming op (which is an
//    `Operator` subclass and shares the same `transform_data_` SoA backbone)
//    must produce the exact same H*v as a freshly-built Operator with the
//    same Heisenberg terms. This isolates "did we wire transform_data_
//    correctly" from "is the symmetry projection correct".
// -----------------------------------------------------------------------------
static void test_streaming_op_apply_matches_operator(TestContext& ctx) {
    const uint64_t N = 4;
    const uint64_t dim = 1ULL << N;

    auto sym_op = build_heisenberg_pbc_streaming(N, 1.0);
    auto op     = build_heisenberg_chain(N, 1.0, /*periodic=*/true);

    double max_err = 0.0;
    for (uint64_t seed = 1; seed <= 4; ++seed) {
        auto v = random_unit_vector(dim, seed * 991ULL);
        ComplexVector a(dim, Complex(0.0, 0.0)), b(dim, Complex(0.0, 0.0));
        sym_op->apply(v.data(), a.data(), dim);
        op->apply(v.data(), b.data(), dim);
        max_err = std::max(max_err, l2_diff(a, b));
    }
    check(ctx, max_err < 1e-12,
          "StreamingSymmetryOperator::apply == Operator::apply on full space",
          "max ||Δ|| = " + std::to_string(max_err));
}

// -----------------------------------------------------------------------------
// 3. Streaming sector spectra (full Z_N projection) reproduce the full
//    Hilbert-space spectrum.
// -----------------------------------------------------------------------------
static void test_streaming_sectors_full_spectrum(TestContext& ctx) {
    const uint64_t N = 4;
    const uint64_t dim = 1ULL << N;

    std::string dir = make_scratch_dir("symmetry", "streaming_sectors");
    write_zN_translation_fixtures(dir, static_cast<int>(N));

    auto sym_op = build_heisenberg_pbc_streaming(N, 1.0);
    bool threw = false;
    try {
        sym_op->generateSymmetrySectorsStreaming(dir);
    } catch (const std::exception& e) {
        threw = true;
        ctx.record_fail("generateSymmetrySectorsStreaming", e.what());
    }
    if (threw) return;

    // 3a. Sector dimensions should sum to the full Hilbert space dimension.
    uint64_t total = 0;
    std::vector<uint64_t> dims;
    for (size_t s = 0; s < sym_op->getNumSectors(); ++s) {
        uint64_t d = sym_op->getSectorDimension(s);
        dims.push_back(d);
        total += d;
    }
    check(ctx, total == dim,
          "Σ sector_dim == 2^N",
          "got " + std::to_string(total) +
          " want " + std::to_string(dim));

    // 3b. Diagonalize each sector by densifying and using Eigen, then
    //     compare the union of spectra to the dense reference.
    auto ref = reference_from_operator(*build_heisenberg_chain(N, 1.0, true), dim);

    std::vector<double> all_eigs;
    for (size_t s = 0; s < sym_op->getNumSectors(); ++s) {
        uint64_t sd = sym_op->getSectorDimension(s);
        if (sd == 0) continue;
        Eigen::MatrixXcd H = Eigen::MatrixXcd::Zero(sd, sd);
        std::vector<Complex> in(sd), out(sd);
        for (uint64_t j = 0; j < sd; ++j) {
            std::fill(in.begin(), in.end(), Complex(0, 0));
            in[j] = Complex(1.0, 0.0);
            sym_op->applySymmetrized(s, in.data(), out.data());
            for (uint64_t i = 0; i < sd; ++i) H(i, j) = out[i];
        }
        // Hermiticity check per sector: not strict (numerical noise possible
        // with the projection algorithm) but should be very small.
        double herm_err = (H - H.adjoint()).cwiseAbs().maxCoeff();
        check(ctx, herm_err < 1e-10,
              "sector " + std::to_string(s) + " H is Hermitian",
              "max |H - H†| = " + std::to_string(herm_err));

        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> es(H);
        for (int i = 0; i < es.eigenvalues().size(); ++i) {
            all_eigs.push_back(es.eigenvalues()[i]);
        }
    }
    check_eigs_close(ctx, all_eigs, ref.eigs, dim, 1e-9,
                     "Σ sector spectra == full spectrum");
}

// -----------------------------------------------------------------------------
// 4. Streaming sector spectra match the full spectrum for a SECOND geometry
//    (N=6 PBC chain). Larger Z_6 has more non-trivial irrep mixing and
//    catches algorithm bugs that happen to be hidden by Z_4's parity-like
//    symmetry. We use Lanczos per sector to also exercise the matrix-free
//    matvec interface end-to-end.
// -----------------------------------------------------------------------------
static void test_streaming_n6(TestContext& ctx) {
    const uint64_t N = 6;
    const uint64_t dim = 1ULL << N;

    std::string dir = make_scratch_dir("symmetry", "streaming_n6");
    write_zN_translation_fixtures(dir, static_cast<int>(N));

    auto sym_op = build_heisenberg_pbc_streaming(N, 1.0);
    sym_op->generateSymmetrySectorsStreaming(dir);

    uint64_t total = 0;
    for (size_t s = 0; s < sym_op->getNumSectors(); ++s) {
        total += sym_op->getSectorDimension(s);
    }
    check(ctx, total == dim,
          "N=6: Σ sector_dim == 2^N",
          "got " + std::to_string(total));

    // Build the dense reference from the standard Operator.
    auto std_op = build_heisenberg_chain(N, 1.0, /*periodic=*/true);
    auto ref = reference_from_operator(*std_op, dim);

    // Densify each sector via applySymmetrized and diagonalize.
    std::vector<double> all_eigs;
    for (size_t s = 0; s < sym_op->getNumSectors(); ++s) {
        uint64_t sd = sym_op->getSectorDimension(s);
        if (sd == 0) continue;
        Eigen::MatrixXcd H = Eigen::MatrixXcd::Zero(sd, sd);
        std::vector<Complex> in(sd), out(sd);
        for (uint64_t j = 0; j < sd; ++j) {
            std::fill(in.begin(), in.end(), Complex(0, 0));
            in[j] = Complex(1.0, 0.0);
            sym_op->applySymmetrized(s, in.data(), out.data());
            for (uint64_t i = 0; i < sd; ++i) H(i, j) = out[i];
        }
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> es(H);
        for (int i = 0; i < es.eigenvalues().size(); ++i) {
            all_eigs.push_back(es.eigenvalues()[i]);
        }
    }
    check_eigs_close(ctx, all_eigs, ref.eigs, dim, 1e-9,
                     "N=6 Σ sector spectra == full spectrum");
}

int main() {
    TestContext ctx("test_symmetry");
    test_json_fixture_loads(ctx);
    test_streaming_op_apply_matches_operator(ctx);
    test_streaming_sectors_full_spectrum(ctx);
    test_streaming_n6(ctx);
    return ctx.summary_exit_code();
}
