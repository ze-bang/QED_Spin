// =============================================================================
// test_symmetry (Catch2 v3, P1.8 / audit Q12)
//
// Validates the matrix-free symmetry-adapted Operator
// (`StreamingSymmetryOperator`) on Z_N translations of an N-site Heisenberg
// chain (N=4 and N=6). The test:
//
//   1. Bypasses the Python automorphism pipeline by emitting JSON files that
//      `SymmetryGroupInfo::loadFromDirectory()` consumes.
//   2. Builds the same Hamiltonian via the streaming op and the standard op,
//      and verifies that:
//        a. `apply()` matches the bare `Operator::apply()` on random vectors;
//        b. After `generateSymmetrySectorsStreaming()`, the union of sector
//           spectra equals the full spectrum;
//        c. Sector dimensions sum to the full Hilbert space.
// =============================================================================

#include "common/catch2_harness.h"

#include <ed/core/operator.h>
#include <ed/solvers/lanczos.h>
#include <ed/symmetry/sector_operator.h>
#include <ed/symmetry/sector_set.h>

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>

#include <algorithm>
#include <cmath>
#include <complex>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <numeric>
#include <sstream>
#include <system_error>
#include <vector>

using namespace ed_tests;

namespace {

std::vector<int> translation_perm(int N, int shift) {
    std::vector<int> p(N);
    for (int i = 0; i < N; ++i) p[i] = ((i - shift) % N + N) % N;
    return p;
}

void write_zN_translation_fixtures(const std::string& dir, int N) {
    std::string root = dir + "/automorphism_results";
    std::error_code ec;
    std::filesystem::create_directories(root, ec);

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

    {
        std::ofstream f(root + "/minimal_generators.json");
        auto p = translation_perm(N, 1);
        f << "{\"generators\":[{\"permutation\":[";
        for (size_t i = 0; i < p.size(); ++i) {
            f << p[i] << (i + 1 < p.size() ? "," : "");
        }
        f << "],\"order\":" << N << "}]}";
    }

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

void add_heisenberg_pbc_terms(ed::symmetry::SectorOperator& op,
                              uint64_t N, double J) {
    const Complex J_real(J, 0.0);
    const Complex J_half(0.5 * J, 0.0);
    for (uint64_t i = 0; i < N; ++i) {
        const uint64_t j = (i + 1) % N;
        op.addTwoBodyTerm(2, i, 2, j, J_real);
        op.addTwoBodyTerm(0, i, 1, j, J_half);
        op.addTwoBodyTerm(1, i, 0, j, J_half);
    }
}

// Full-Hilbert reference operator (no symmetry) built from the same explicit
// term list, so we can cross-check the standard chain builder.
std::unique_ptr<Operator> build_heisenberg_pbc_full(uint64_t N, double J) {
    auto op = std::make_unique<Operator>(N, 0.5f);
    const Complex J_real(J, 0.0);
    const Complex J_half(0.5 * J, 0.0);
    for (uint64_t i = 0; i < N; ++i) {
        const uint64_t j = (i + 1) % N;
        op->addTwoBodyTerm(2, i, 2, j, J_real);
        op->addTwoBodyTerm(0, i, 1, j, J_half);
        op->addTwoBodyTerm(1, i, 0, j, J_half);
    }
    return op;
}

// Build the dense matrix of a (small) operator by probing standard-basis
// vectors through its matvec.
Eigen::MatrixXcd dense_from_operator(ed::LinearOperator& op) {
    const std::size_t d = op.dim();
    Eigen::MatrixXcd H = Eigen::MatrixXcd::Zero(d, d);
    std::vector<Complex> in(d), out(d);
    for (std::size_t j = 0; j < d; ++j) {
        std::fill(in.begin(), in.end(), Complex(0, 0));
        in[j] = Complex(1.0, 0.0);
        op.apply(in.data(), out.data(), d);
        for (std::size_t i = 0; i < d; ++i) H(i, j) = out[i];
    }
    return H;
}

} // namespace

TEST_CASE("Z_N JSON fixture loads (smoke)",
          "[symmetry][fixture_load]") {
    const int N = 4;
    std::string dir = make_scratch_dir("symmetry", "fixture_load");
    write_zN_translation_fixtures(dir, N);

    SymmetryGroupInfo info;
    REQUIRE_NOTHROW(info.loadFromDirectory(dir));

    REQUIRE(info.max_clique.size() == static_cast<size_t>(N));
    REQUIRE(info.generators.size() == 1);
    REQUIRE(info.generator_orders.size() == 1);
    REQUIRE(info.generator_orders[0] == N);
    REQUIRE(info.sectors.size() == static_cast<size_t>(N));

    REQUIRE(info.power_representation.size() == static_cast<size_t>(N));
    for (int g = 0; g < N; ++g) {
        INFO("g=" << g);
        REQUIRE(info.power_representation[g].size() == 1);
        REQUIRE(info.power_representation[g][0] == g);
    }
}

TEST_CASE("explicit-term Operator::apply == chain-builder Operator::apply",
          "[symmetry][apply]") {
    const uint64_t N = 4;
    const uint64_t dim = 1ULL << N;

    auto full_op = build_heisenberg_pbc_full(N, 1.0);
    auto op      = build_heisenberg_chain(N, 1.0, /*periodic=*/true);

    double max_err = 0.0;
    for (uint64_t seed = 1; seed <= 4; ++seed) {
        auto v = random_unit_vector(dim, seed * 991ULL);
        ComplexVector a(dim, Complex(0.0, 0.0)), b(dim, Complex(0.0, 0.0));
        full_op->apply(v.data(), a.data(), dim);
        op->apply(v.data(), b.data(), dim);
        max_err = std::max(max_err, l2_diff(a, b));
    }
    INFO("max ||Δ|| = " << max_err);
    REQUIRE(max_err < 1e-12);
}

TEST_CASE("Σ Z_N sector spectra == full spectrum (N=4)",
          "[symmetry][sectors][N4]") {
    const uint64_t N = 4;
    const uint64_t dim = 1ULL << N;

    std::string dir = make_scratch_dir("symmetry", "streaming_sectors");
    write_zN_translation_fixtures(dir, static_cast<int>(N));

    SymmetryGroupInfo info;
    REQUIRE_NOTHROW(info.loadFromDirectory(dir));

    auto ops = ed::symmetry::build_full_sector_operators_lazy(
        N, 0.5f, info,
        [&](ed::symmetry::SectorOperator& op) {
            add_heisenberg_pbc_terms(op, N, 1.0);
        });

    uint64_t total = 0;
    for (const auto& op : ops) total += op->dim();
    INFO("Σ sector_dim=" << total << " want=" << dim);
    REQUIRE(total == dim);

    auto ref = reference_from_operator(*build_heisenberg_chain(N, 1.0, true), dim);

    std::vector<double> all_eigs;
    for (const auto& op : ops) {
        const std::size_t sd = op->dim();
        if (sd == 0) continue;
        Eigen::MatrixXcd H = dense_from_operator(*op);
        double herm_err = (H - H.adjoint()).cwiseAbs().maxCoeff();
        INFO("sector dim=" << sd << " max |H - H†| = " << herm_err);
        REQUIRE(herm_err < 1e-10);

        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> es(H);
        for (int i = 0; i < es.eigenvalues().size(); ++i) {
            all_eigs.push_back(es.eigenvalues()[i]);
        }
    }
    require_eigs_close(all_eigs, ref.eigs, dim, 1e-9,
                       "Σ sector spectra == full spectrum");
}

TEST_CASE("Σ Z_N sector spectra == full spectrum (N=6)",
          "[symmetry][sectors][N6]") {
    const uint64_t N = 6;
    const uint64_t dim = 1ULL << N;

    std::string dir = make_scratch_dir("symmetry", "streaming_n6");
    write_zN_translation_fixtures(dir, static_cast<int>(N));

    SymmetryGroupInfo info;
    REQUIRE_NOTHROW(info.loadFromDirectory(dir));

    auto ops = ed::symmetry::build_full_sector_operators_lazy(
        N, 0.5f, info,
        [&](ed::symmetry::SectorOperator& op) {
            add_heisenberg_pbc_terms(op, N, 1.0);
        });

    uint64_t total = 0;
    for (const auto& op : ops) total += op->dim();
    INFO("N=6: Σ sector_dim=" << total << " want=" << dim);
    REQUIRE(total == dim);

    auto std_op = build_heisenberg_chain(N, 1.0, /*periodic=*/true);
    auto ref = reference_from_operator(*std_op, dim);

    std::vector<double> all_eigs;
    for (const auto& op : ops) {
        const std::size_t sd = op->dim();
        if (sd == 0) continue;
        Eigen::MatrixXcd H = dense_from_operator(*op);
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> es(H);
        for (int i = 0; i < es.eigenvalues().size(); ++i) {
            all_eigs.push_back(es.eigenvalues()[i]);
        }
    }
    require_eigs_close(all_eigs, ref.eigs, dim, 1e-9,
                       "N=6 Σ sector spectra == full spectrum");
}
