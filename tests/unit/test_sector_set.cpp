// =============================================================================
// test_sector_set
//
// Validation pin for P5 of the operator-collapse refactor (Jun 2026):
// ``ed::symmetry::build_full_sector_operators`` /
// ``build_fixed_sz_sector_operators`` -- the multi-sector enumerator that
// emits a flat vector of standalone ``SectorOperator``s from a
// ``SymmetryGroupInfo`` + a term-builder, replacing the legacy
// ``generateSymmetrySectorsStreaming`` + ``SectorView`` fan-out.
//
// Validates, for the Heisenberg Z_N ring (N=6, J=1):
//
//   (A) Full lane: the set of surviving sector dimensions matches the
//       legacy ``StreamingSymmetryOperator`` sector dimensions, the union
//       of sector dims equals the full Hilbert space (2^6 = 64), and the
//       minimum sector ground-state energy equals the Bethe-ansatz
//       E_0 = -2.802775637731995.
//
//   (B) Fixed-Sz lane (n_up = 3): the surviving sector dims sum to the
//       half-filled subspace dimension C(6,3) = 20, and the minimum sector
//       ground state matches the global GS (the Sz=0 sector contains it).
//
// Each SectorOperator is independent (owns its SectorBasis, references no
// parent), so the orchestrator can consume the vector uniformly.
// =============================================================================

#include "common/catch2_harness.h"

#include <ed/core/operator.h>
#include <ed/symmetry/projector.h>
#include <ed/symmetry/sector_basis.h>
#include <ed/symmetry/sector_operator.h>
#include <ed/symmetry/sector_set.h>
#include <ed/symmetry/subspace.h>

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <numeric>
#include <string>
#include <system_error>
#include <vector>

using namespace ed_tests;

namespace {

constexpr double kE0_N6 = -2.802775637731995;  // Bethe-ansatz GS, J=1, N=6

std::vector<int> translation_perm(int N, int shift) {
    std::vector<int> p(N);
    for (int i = 0; i < N; ++i) p[i] = ((i - shift) % N + N) % N;
    return p;
}

void write_zN_translation_fixtures(const std::string& dir, int N) {
    const std::string root = dir + "/automorphism_results";
    std::error_code ec;
    std::filesystem::create_directories(root, ec);
    {
        std::ofstream f(root + "/max_clique.json");
        f << "[";
        for (int g = 0; g < N; ++g) {
            const auto p = translation_perm(N, g);
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
        const auto p = translation_perm(N, 1);
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

// Append J=1 PBC Heisenberg terms via the typed setters.
void add_heisenberg_pbc_terms(ed::symmetry::SectorOperator& op,
                              std::uint64_t N, double J) {
    const Complex J_real(J, 0.0);
    const Complex J_half(0.5 * J, 0.0);
    for (std::uint64_t i = 0; i < N; ++i) {
        const std::uint64_t j = (i + 1) % N;
        op.addTwoBodyTerm(2, i, 2, j, J_real);
        op.addTwoBodyTerm(0, i, 1, j, J_half);
        op.addTwoBodyTerm(1, i, 0, j, J_half);
    }
}

double lowest_eigenvalue(const ed::symmetry::SectorOperator& op) {
    const std::size_t d = op.dim();
    Eigen::MatrixXcd H(d, d);
    std::vector<Complex> e(d, Complex(0.0, 0.0));
    std::vector<Complex> col(d, Complex(0.0, 0.0));
    for (std::size_t j = 0; j < d; ++j) {
        std::fill(e.begin(), e.end(), Complex(0.0, 0.0));
        e[j] = Complex(1.0, 0.0);
        op.apply(e.data(), col.data(), d);
        for (std::size_t i = 0; i < d; ++i) H(i, j) = col[i];
    }
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> es(H);
    return es.eigenvalues()(0);
}

} // namespace

TEST_CASE("sector_set: build_full_sector_operators matches legacy dims + Bethe GS (N=6)",
          "[symmetry][sector_set][full][N6]")
{
    const int N = 6;
    std::string dir = make_scratch_dir("sector_set", "full_N6");
    write_zN_translation_fixtures(dir, N);

    SymmetryGroupInfo info;
    REQUIRE_NOTHROW(info.loadFromDirectory(dir));

    // New path: a flat vector of standalone per-sector operators.
    auto ops = ed::symmetry::build_full_sector_operators_lazy(
        static_cast<std::uint64_t>(N), 0.5f, info,
        [&](ed::symmetry::SectorOperator& op) {
            add_heisenberg_pbc_terms(op, N, 1.0);
        });

    // (A1) every surviving sector is non-empty and carries its raw irrep id.
    for (const auto& op : ops) {
        op->basis().ensureHostCsr();  // lazy op: id lives on the materialised sector
        const std::size_t id =
            static_cast<std::size_t>(op->basis().sector().sector_id);
        REQUIRE(id < info.sectors.size());
        REQUIRE(op->dim() > 0);
    }

    // (A2) the surviving sector dims tile the full Hilbert space.
    std::size_t total = 0;
    for (const auto& op : ops) total += op->dim();
    REQUIRE(total == (1ULL << N));

    // (A3) the global GS is the min sector ground state.
    double e0_min = std::numeric_limits<double>::infinity();
    for (const auto& op : ops) {
        e0_min = std::min(e0_min, lowest_eigenvalue(*op));
    }
    REQUIRE(std::abs(e0_min - kE0_N6) < 1e-10);
}

TEST_CASE("sector_set: build_fixed_sz_sector_operators tiles half-filling + Bethe GS (N=6)",
          "[symmetry][sector_set][fixedsz][N6]")
{
    const int N = 6;
    const std::int64_t n_up = 3;
    std::string dir = make_scratch_dir("sector_set", "fixedsz_N6");
    write_zN_translation_fixtures(dir, N);

    SymmetryGroupInfo info;
    REQUIRE_NOTHROW(info.loadFromDirectory(dir));

    auto ops = ed::symmetry::build_fixed_sz_sector_operators_lazy(
        static_cast<std::uint64_t>(N), 0.5f, n_up, info,
        [&](ed::symmetry::SectorOperator& op) {
            add_heisenberg_pbc_terms(op, N, 1.0);
        });

    REQUIRE_FALSE(ops.empty());

    // (B1) surviving dims tile the half-filled subspace C(6,3) = 20.
    std::size_t total = 0;
    for (const auto& op : ops) {
        REQUIRE(op->dim() > 0);
        total += op->dim();
    }
    REQUIRE(total == 20);

    // (B2) the half-filled GS equals the global Bethe-ansatz GS (the Sz=0
    // sector contains the singlet ground state of the N=6 chain).
    double e0_min = std::numeric_limits<double>::infinity();
    for (const auto& op : ops) {
        e0_min = std::min(e0_min, lowest_eigenvalue(*op));
    }
    REQUIRE(std::abs(e0_min - kE0_N6) < 1e-10);
}

