// =============================================================================
// test_make_sector_operators_e2e
//
// End-to-end acceptance pin for the option-1 wiring of the operator-collapse
// refactor (Jun 2026): ``ed::make_sector_operators(OperatorSpec)`` routes a
// directory-loaded Hamiltonian straight through the P5 sector-set enumerator
// (``build_{full,fixed_sz}_sector_operators``) -- building owning,
// standalone ``SectorOperator``s with NO ``StreamingSymmetryOperator``
// materialised -- and the resulting per-sector operators are then solved
// through the production ``ed::workflows::solve`` orchestrator.
//
// Reference: 6-site periodic AFM Heisenberg ring at J = 1.
//   * Bethe-ansatz ground state E_0 = -2.802775637731995.
//   * Full Hilbert space 2^6 = 64; the union of symmetry-sector spectra
//     reconstructs the entire spectrum.
//   * Fixed-Sz n_up = 3 subspace dimension C(6,3) = 20.
//
// Validates END TO END (factory -> sectors -> solver):
//   [A] Full lane: make_sector_operators yields sectors whose dims tile
//       2^6, every sector solves through ed::workflows::solve, the global
//       minimum equals the Bethe GS, and the FULL union spectrum is
//       byte-equal (sorted, 1e-9) to an INDEPENDENT full-Hilbert dense
//       diagonalisation of the same Hamiltonian (the carrier-free golden
//       reference after the operator-collapse Phase 3 removal).
//   [B] Fixed-Sz lane (n_up = 3): dims tile C(6,3) = 20, solver min over
//       sectors equals the Bethe GS.
// =============================================================================

#include "common/catch2_harness.h"

#include <ed/core/make_operator.h>
#include <ed/orchestrator.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <memory>
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

// Z_N translation automorphism metadata consumed by SymmetryGroupInfo::
// loadFromDirectory (mirrors test_sector_set.cpp's fixture writer).
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

// 6-site periodic Heisenberg chain: Trans.dat (empty single-site deck) +
// InterAll.dat (three terms per bond, N periodic bonds).
void write_heisenberg_directory(const std::string& dir, int N, double J = 1.0) {
    std::error_code ec;
    std::filesystem::create_directories(dir, ec);
    {
        std::ofstream f(dir + "/Trans.dat");
        f << "===================\n";
        f << "num       0\n";
        f << "===================\n";
        f << "===================\n";
        f << "===================\n";
    }
    {
        std::ofstream f(dir + "/InterAll.dat");
        const std::uint64_t nlines = 3ULL * static_cast<std::uint64_t>(N);
        f << "===================\n";
        f << "num       " << nlines << "\n";
        f << "===================\n";
        f << "===================\n";
        f << "===================\n";
        for (int i = 0; i < N; ++i) {
            const int j = (i + 1) % N;
            f << "        2         " << i
              << "           2         " << j
              << "    " << J << "    0.000000\n";
            f << "        0         " << i
              << "           1         " << j
              << "    " << (0.5 * J) << "    0.000000\n";
            f << "        1         " << i
              << "           0         " << j
              << "    " << (0.5 * J) << "    0.000000\n";
        }
    }
}

// Solve a single sector operator for its FULL spectrum via the production
// orchestrator, returning the (ascending) eigenvalues.
std::vector<double> solve_full_spectrum(const ed::LinearOperator& op) {
    ed::SolveOptions opts;
    opts.num_eigs = op.dim();
    opts.method   = ed::SolveMethod::FullDiag;
    auto r = ed::workflows::solve(op, opts);
    std::vector<double> e = r.eigenvalues;
    std::sort(e.begin(), e.end());
    return e;
}

ed::OperatorSpec heisenberg_spec(const std::string& dir, int N) {
    ed::OperatorSpec spec;
    spec.source             = ed::DirectoryPath{dir};
    spec.num_sites          = static_cast<std::uint64_t>(N);
    spec.spin_l             = 0.5f;
    spec.streaming_symmetry = true;
    return spec;
}

}  // namespace

// -----------------------------------------------------------------------------
TEST_CASE("make_sector_operators: full lane solves to Bethe GS + matches legacy",
          "[make_operator][sector_set][e2e][full][N6]") {
    const int N = 6;
    std::string dir = make_scratch_dir("make_sector_ops", "full_N6");
    write_zN_translation_fixtures(dir, N);
    write_heisenberg_directory(dir, N, 1.0);

    // --- NEW direct path: make_operator -> build_full_sector_operators ----
    auto ops = ed::make_sector_operators(heisenberg_spec(dir, N));
    REQUIRE_FALSE(ops.empty());

    std::size_t total_dim = 0;
    double e0_min = std::numeric_limits<double>::infinity();
    std::vector<double> new_spectrum;
    for (const auto& op : ops) {
        REQUIRE(op->dim() > 0);
        total_dim += op->dim();
        auto e = solve_full_spectrum(*op);
        REQUIRE(e.size() == op->dim());
        e0_min = std::min(e0_min, e.front());
        new_spectrum.insert(new_spectrum.end(), e.begin(), e.end());
    }

    // (A1) sectors tile the full Hilbert space.
    REQUIRE(total_dim == (1ULL << N));
    // (A2) global GS == Bethe ansatz.
    REQUIRE(std::abs(e0_min - kE0_N6) < 1e-9);

    // --- Independent reference: full-Hilbert dense diagonalisation -------
    // Build the SAME Hamiltonian on the full 2^N space (no symmetry) and
    // diagonalise it densely. Its spectrum is the carrier-free golden the
    // symmetry-sector union must reproduce.
    ed::OperatorSpec full_spec;
    full_spec.source             = ed::DirectoryPath{dir};
    full_spec.num_sites          = static_cast<std::uint64_t>(N);
    full_spec.spin_l             = 0.5f;
    full_spec.streaming_symmetry = false;
    auto full_op = ed::make_operator(std::move(full_spec));
    REQUIRE(full_op->dim() == (1ULL << N));
    std::vector<double> ref_spectrum = solve_full_spectrum(*full_op);

    // (A3) the union spectra agree element-wise once sorted.
    std::sort(new_spectrum.begin(), new_spectrum.end());
    REQUIRE(new_spectrum.size() == ref_spectrum.size());
    double max_diff = 0.0;
    for (std::size_t i = 0; i < new_spectrum.size(); ++i) {
        max_diff = std::max(max_diff,
                            std::abs(new_spectrum[i] - ref_spectrum[i]));
    }
    INFO("max |E_sym - E_full| over full spectrum = " << max_diff);
    REQUIRE(max_diff < 1e-9);
}

// -----------------------------------------------------------------------------
// Pins the CSR-free rep-walk DENSE-ASSEMBLY lane (SubspaceOperator::assembleDense
// for `needs_orbit_walk && has_coeff_modifier` producers -- the "orbit-walk
// symmetry dense-assembly lane"). Forcing ED_SYM_LAZY_SECTORS=1 routes
// make_sector_operators to the lazy rep-walk producers, so a FullDiag solve
// assembles each sector densely *through that lane* (vs the eager orbit-CSR path
// the N=6 default would otherwise take). The sector union must still reproduce
// the full-Hilbert dense reference -- byte-for-byte the same physics as the
// gather matvec, now built in one O(|G|*nnz) pass.
TEST_CASE("make_sector_operators: lazy rep-walk dense-assembly lane == full reference",
          "[make_operator][sector_set][e2e][dense][lazy][N6]") {
    const int N = 6;
    setenv("ED_SYM_LAZY_SECTORS", "1", /*overwrite=*/1);  // force rep-walk producers

    std::string dir = make_scratch_dir("make_sector_ops", "lazydense_full_N6");
    write_zN_translation_fixtures(dir, N);
    write_heisenberg_directory(dir, N, 1.0);

    // Symmetry sectors via the LAZY rep-walk producer; each FullDiag goes through
    // the dense-assembly lane under test.
    auto ops = ed::make_sector_operators(heisenberg_spec(dir, N));
    REQUIRE_FALSE(ops.empty());
    std::vector<double> sym_spectrum;
    for (const auto& op : ops) {
        REQUIRE(op->dim() > 0);
        auto e = solve_full_spectrum(*op);    // FullDiag -> assembleDense (the lane)
        sym_spectrum.insert(sym_spectrum.end(), e.begin(), e.end());
    }
    std::sort(sym_spectrum.begin(), sym_spectrum.end());

    // Reference: same Hamiltonian on the full 2^N space, dense-diagonalised.
    ed::OperatorSpec full_spec;
    full_spec.source             = ed::DirectoryPath{dir};
    full_spec.num_sites          = static_cast<std::uint64_t>(N);
    full_spec.spin_l             = 0.5f;
    full_spec.streaming_symmetry = false;
    auto full_op = ed::make_operator(std::move(full_spec));
    std::vector<double> ref_spectrum = solve_full_spectrum(*full_op);

    unsetenv("ED_SYM_LAZY_SECTORS");

    REQUIRE(sym_spectrum.size() == ref_spectrum.size());
    REQUIRE(sym_spectrum.size() == (1ULL << N));
    double max_diff = 0.0;
    for (std::size_t i = 0; i < sym_spectrum.size(); ++i)
        max_diff = std::max(max_diff,
                            std::abs(sym_spectrum[i] - ref_spectrum[i]));
    INFO("max |E_lazy_dense - E_full| = " << max_diff);
    REQUIRE(max_diff < 1e-9);
}

// -----------------------------------------------------------------------------
TEST_CASE("make_sector_operators: fixed-Sz lane tiles half-filling + Bethe GS",
          "[make_operator][sector_set][e2e][fixedsz][N6]") {
    const int N = 6;
    std::string dir = make_scratch_dir("make_sector_ops", "fixedsz_N6");
    write_zN_translation_fixtures(dir, N);
    write_heisenberg_directory(dir, N, 1.0);

    ed::OperatorSpec spec = heisenberg_spec(dir, N);
    spec.fixed_sz = 3;  // half-filled

    auto ops = ed::make_sector_operators(spec);
    REQUIRE_FALSE(ops.empty());

    std::size_t total_dim = 0;
    double e0_min = std::numeric_limits<double>::infinity();
    for (const auto& op : ops) {
        REQUIRE(op->dim() > 0);
        total_dim += op->dim();
        auto e = solve_full_spectrum(*op);
        e0_min = std::min(e0_min, e.front());
    }

    // (B1) surviving dims tile the half-filled subspace C(6,3) = 20.
    REQUIRE(total_dim == 20);
    // (B2) the half-filled GS equals the global Bethe-ansatz GS.
    REQUIRE(std::abs(e0_min - kE0_N6) < 1e-9);
}

// -----------------------------------------------------------------------------
TEST_CASE("make_sector_operators: rejects non-streaming spec",
          "[make_operator][sector_set][guard]") {
    ed::OperatorSpec spec;
    spec.source    = ed::DirectoryPath{"/tmp/does_not_matter"};
    spec.num_sites = 6;
    spec.spin_l    = 0.5f;
    spec.streaming_symmetry = false;  // wrong lane
    REQUIRE_THROWS_AS(ed::make_sector_operators(spec), std::runtime_error);
}
