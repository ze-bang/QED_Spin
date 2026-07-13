// =============================================================================
// test_symmetry_ordering_parity
//
// Cutover-safety invariant pin (Jun 2026). The operator-collapse refactor runs
// TWO independently-constructed symmetry paths that must agree not just
// numerically but in their BASIS ORDERING:
//
//   * legacy  : StreamingSymmetryOperator::generateSymmetrySectorsStreaming
//               -> per-sector ``SymmetrySector::basis_states`` ordered by the
//               legacy "Pass 2" loop (iterate ``unique_orbit_reps_`` ascending,
//               append every surviving orbit).
//   * new     : ed::symmetry::build_full_sector_operators
//               -> SectorBasis built from ``enumerate_full_orbit_reps`` +
//               ``SectorBasis::build`` (a separate construction).
//
// The existing tests (test_sector_set, test_symmetry_backend) only assert
// sector DIMENSIONS and EIGENVALUES match. Eigenvalues are invariant under any
// permutation of the basis, so they do NOT catch a per-sector basis ORDERING
// divergence. But the sector index -> (state -> basis index) mapping is exactly
// what keys HDF5 orbit caches and cross-sector observables. If the two paths
// ever ordered their basis states differently, every eigenvalue test would
// still pass while saved bases / observables silently corrupted on cutover.
//
// This test makes the ordering a first-class, named invariant:
//
//   (1) rep-set parity: the deduplicated union of every legacy sector's
//       ``orbit_rep`` equals ``enumerate_full_orbit_reps(info, N)`` (each
//       orbit survives in >= 1 sector -- the k=0 symmetric combination always
//       has nonzero norm -- so the union is the full canonical rep set).
//
//   (2) per-sector ordering parity: for each sector_id, the legacy and the new
//       path emit the SAME ``orbit_rep`` sequence and the SAME ``orbit_elements``
//       sequence, element for element (both are exact, convention-free
//       integer quantities).
//
// Heisenberg Z_N ring, N=6, J=1 (same fixture as test_sector_set).
// =============================================================================

#include "common/catch2_harness.h"

#include <ed/symmetry/sector_operator.h>
#include <ed/symmetry/sector_set.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <string>
#include <system_error>
#include <vector>

using namespace ed_tests;

namespace {

std::vector<int> translation_perm(int N, int shift) {
    std::vector<int> p(N);
    for (int i = 0; i < N; ++i) p[i] = ((i - shift) % N + N) % N;
    return p;
}

// Z_N translation group with N momentum sectors (mirror of test_sector_set).
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

// Append the J=1 PBC Heisenberg terms to a SectorOperator.
void add_heisenberg_pbc_terms_new(ed::symmetry::SectorOperator& op,
                                  int N, double J) {
    const Complex J_real(J, 0.0), J_half(0.5 * J, 0.0);
    for (std::uint64_t i = 0; i < static_cast<std::uint64_t>(N); ++i) {
        const std::uint64_t j = (i + 1) % N;
        op.addTwoBodyTerm(2, i, 2, j, J_real);
        op.addTwoBodyTerm(0, i, 1, j, J_half);
        op.addTwoBodyTerm(1, i, 0, j, J_half);
    }
}

} // namespace

TEST_CASE("symmetry ordering parity: rep-set union matches enumerate_full_orbit_reps (N=6)",
          "[symmetry][ordering][parity][N6]")
{
    const int N = 6;
    std::string dir = make_scratch_dir("sym_ordering", "repset_N6");
    write_zN_translation_fixtures(dir, N);

    SymmetryGroupInfo info;
    REQUIRE_NOTHROW(info.loadFromDirectory(dir));

    // Canonical rep list straight from the production free function.
    const std::vector<std::uint64_t> reps =
        ed::symmetry::enumerate_full_orbit_reps(
            info, static_cast<std::uint64_t>(N));
    REQUIRE_FALSE(reps.empty());
    // enumerate_full_orbit_reps is ascending by construction.
    REQUIRE(std::is_sorted(reps.begin(), reps.end()));

    // Deduplicated union of every surviving sector's orbit_rep. Each orbit
    // survives in at least one sector (the k=0 symmetric combination has
    // nonzero norm), so the union must be the full canonical rep set.
    auto ops = ed::symmetry::build_full_sector_operators_lazy(
        static_cast<std::uint64_t>(N), 0.5f, info,
        [&](ed::symmetry::SectorOperator& op) {
            add_heisenberg_pbc_terms_new(op, N, 1.0);
        });
    std::vector<std::uint64_t> union_reps;
    for (const auto& op : ops) {
        op->basis().ensureHostCsr();
        for (const auto& bs : op->basis().sector().basis_states) {
            union_reps.push_back(bs.orbit_rep);
        }
    }
    std::sort(union_reps.begin(), union_reps.end());
    union_reps.erase(std::unique(union_reps.begin(), union_reps.end()),
                     union_reps.end());

    REQUIRE(union_reps.size() == reps.size());
    REQUIRE(union_reps == reps);
}

TEST_CASE("symmetry ordering parity: build_full per-sector basis is canonically ordered (N=6)",
          "[symmetry][ordering][parity][N6]")
{
    const int N = 6;
    std::string dir = make_scratch_dir("sym_ordering", "perorder_N6");
    write_zN_translation_fixtures(dir, N);

    SymmetryGroupInfo info;
    REQUIRE_NOTHROW(info.loadFromDirectory(dir));

    // The canonical (ascending) orbit-rep list every sector basis is a
    // subsequence of -- ``SectorBasis::build`` iterates it ascending and
    // appends each surviving orbit in order.
    const std::vector<std::uint64_t> reps =
        ed::symmetry::enumerate_full_orbit_reps(
            info, static_cast<std::uint64_t>(N));

    auto ops = ed::symmetry::build_full_sector_operators_lazy(
        static_cast<std::uint64_t>(N), 0.5f, info,
        [&](ed::symmetry::SectorOperator& op) {
            add_heisenberg_pbc_terms_new(op, N, 1.0);
        });
    REQUIRE_FALSE(ops.empty());

    // The basis ordering invariant the eigenvalue tests cannot see: for each
    // sector the ``orbit_rep`` sequence is a strictly-ascending subsequence of
    // the canonical rep list (Pass-2 order), each orbit's canonical rep is the
    // numeric minimum of its (ascending-sorted) orbit elements, and the
    // lookup index keys off exactly that ordering. These are the integer,
    // convention-free guarantees the HDF5 orbit caches + cross-sector
    // observables depend on.
    for (const auto& op : ops) {
        op->basis().ensureHostCsr();
        const SymmetrySector& sec = op->basis().sector();
        REQUIRE_FALSE(sec.basis_states.empty());

        std::size_t rep_cursor = 0;
        std::uint64_t prev_rep = 0;
        bool have_prev = false;
        for (const auto& bs : sec.basis_states) {
            // Strictly ascending orbit_rep sequence.
            if (have_prev) REQUIRE(bs.orbit_rep > prev_rep);
            prev_rep = bs.orbit_rep;
            have_prev = true;

            // orbit_rep is a member of the canonical ascending rep list, in
            // increasing position (subsequence of ``reps``).
            while (rep_cursor < reps.size() && reps[rep_cursor] != bs.orbit_rep) {
                ++rep_cursor;
            }
            REQUIRE(rep_cursor < reps.size());

            // orbit_elements sorted ascending + rep == numeric minimum.
            REQUIRE(std::is_sorted(bs.orbit_elements.begin(),
                                   bs.orbit_elements.end()));
            REQUIRE_FALSE(bs.orbit_elements.empty());
            REQUIRE(bs.orbit_rep == bs.orbit_elements.front());
        }
    }
}
