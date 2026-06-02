// =============================================================================
// test_sector_basis
//
// First validation pin for the operator-collapse refactor (Jun 2026):
// ``ed::symmetry::SectorBasis`` -- the owning symmetry-basis type that
// completes the basis-owning triplet
//
//     FullSpaceSubspace / FixedSzSubspace  (subspace.h)
//     SectorBasis                          (sector_basis.h)   <-- new
//
// SectorBasis lifts the per-sector orbit data that used to live inside
// ``StreamingSymmetryOperator`` into a free-standing value that yields
// the POD ``SymmetryBasisPolicy`` the unified matvec kernels consume.
// This test exercises both construction routes and the lookup surface:
//
//   1. ``SectorBasis::build(FullSpaceSubspace, SpatialProjector, ...)``
//      against a Heisenberg-ring Z_N translation group on N=6 produces,
//      for every sector, orbit data whose per-rep ``norm`` and orbit
//      membership match the independent ``compute_orbit_for_state``
//      reference -- and whose state->orbit lookup round-trips for every
//      orbit element.
//   2. ``SectorBasis::adopt(SymmetrySector, |G|)`` reproduces an
//      identical lookup from an already-materialised sector.
//   3. ``policy()`` yields a non-owning view with group_norm = 1/|G|
//      and a sector pointer into the owning object.
// =============================================================================

#include "common/catch2_harness.h"

#include <ed/core/streaming_symmetry.h>
#include <ed/symmetry/projector.h>
#include <ed/symmetry/projector_chain.h>
#include <ed/symmetry/sector_basis.h>
#include <ed/symmetry/subspace.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <string>
#include <system_error>
#include <unordered_set>
#include <vector>

using namespace ed_tests;

namespace {

std::vector<int> translation_perm(int N, int shift) {
    std::vector<int> p(N);
    for (int i = 0; i < N; ++i) p[i] = ((i - shift) % N + N) % N;
    return p;
}

// Mirror of the fixture writer in test_projector_chain.cpp: a Z_N
// translation group with N momentum sectors.
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

// Deduplicated orbit representatives over the full Hilbert space: a
// computational state is a representative iff it is the numeric minimum
// of its translation orbit. Mirrors the legacy "Pass 1" rep enumeration.
std::vector<std::uint64_t>
enumerate_orbit_reps(const SymmetryGroupInfo& info, int N) {
    std::vector<std::uint64_t> reps;
    const std::uint64_t dim = (1ULL << N);
    for (std::uint64_t s = 0; s < dim; ++s) {
        std::uint64_t mn = s;
        for (std::size_t g = 0; g < info.max_clique.size(); ++g) {
            mn = std::min(mn, applyPermutation(s, info.max_clique[g]));
        }
        if (mn == s) reps.push_back(s);
    }
    return reps;
}

} // namespace

TEST_CASE("sector_basis: build(FullSpace) round-trips lookup for every sector (N=6)",
          "[symmetry][sector_basis][N6]")
{
    const int N = 6;
    std::string dir = make_scratch_dir("sector_basis", "build_N6");
    write_zN_translation_fixtures(dir, N);

    SymmetryGroupInfo info;
    REQUIRE_NOTHROW(info.loadFromDirectory(dir));

    const ed::symmetry::FullSpaceSubspace full(static_cast<std::uint64_t>(N));
    const ed::symmetry::SpatialProjector  spatial(info);
    const std::vector<std::uint64_t> reps = enumerate_orbit_reps(info, N);

    const std::size_t G = info.max_clique.size();
    REQUIRE(G == static_cast<std::size_t>(N));

    for (std::size_t s = 0; s < info.sectors.size(); ++s) {
        const auto& qn = info.sectors[s].quantum_numbers;
        const auto& pf = info.sectors[s].phase_factors;

        ed::symmetry::SectorBasis sb =
            ed::symmetry::SectorBasis::build(full, spatial, qn, pf, reps,
                                             /*sector_id=*/s);

        // Cross-check dim against an independent count of non-vanishing
        // orbits in this irrep.
        std::size_t expected_dim = 0;
        for (std::uint64_t rep : reps) {
            std::vector<std::uint64_t> el;
            std::vector<Complex>       co;
            double                     ns = 0.0;
            ed::symmetry::compute_orbit_for_state(full, spatial, rep, pf,
                                                  el, co, ns);
            if (!el.empty() && ns > ed::symmetry::SectorBasis::kOrbitNormSqEpsilon)
                ++expected_dim;
        }
        REQUIRE(sb.dim() == expected_dim);

        // Every orbit element must resolve back to its owning orbit
        // index; every representative must resolve to itself.
        for (std::uint64_t i = 0; i < sb.dim(); ++i) {
            const std::uint64_t rep = sb.state_of(i);
            REQUIRE(sb.index_of(rep) == static_cast<std::int64_t>(i));

            std::vector<std::uint64_t> el;
            std::vector<Complex>       co;
            double                     ns = 0.0;
            ed::symmetry::compute_orbit_for_state(full, spatial, rep, pf,
                                                  el, co, ns);
            for (std::uint64_t e : el) {
                REQUIRE(sb.index_of(e) == static_cast<std::int64_t>(i));
            }
            // Per-rep norm matches sqrt(norm_sq) verbatim.
            REQUIRE(std::abs(sb.sector().basis_states[i].norm -
                             std::sqrt(ns)) < 1e-12);
        }

        // States outside any orbit in this sector return -1.
        REQUIRE(sb.index_of((1ULL << N)) == -1);

        // policy() view: group_norm = 1/|G|, sector pointer valid.
        const auto pol = sb.policy();
        REQUIRE(std::abs(pol.group_norm - 1.0 / static_cast<double>(G)) < 1e-15);
        REQUIRE(pol.sector == &sb.sector());
    }
}

TEST_CASE("sector_basis: adopt reproduces build lookup (N=6)",
          "[symmetry][sector_basis][adopt]")
{
    const int N = 6;
    std::string dir = make_scratch_dir("sector_basis", "adopt_N6");
    write_zN_translation_fixtures(dir, N);

    SymmetryGroupInfo info;
    REQUIRE_NOTHROW(info.loadFromDirectory(dir));

    const ed::symmetry::FullSpaceSubspace full(static_cast<std::uint64_t>(N));
    const ed::symmetry::SpatialProjector  spatial(info);
    const std::vector<std::uint64_t> reps = enumerate_orbit_reps(info, N);
    const std::size_t G = info.max_clique.size();

    // Use sector 0 (trivial irrep) for the adopt round-trip.
    const auto& qn = info.sectors[0].quantum_numbers;
    const auto& pf = info.sectors[0].phase_factors;

    ed::symmetry::SectorBasis built =
        ed::symmetry::SectorBasis::build(full, spatial, qn, pf, reps, 0);

    // Snapshot the materialised sector and adopt a copy of it.
    SymmetrySector copy = built.sector();
    ed::symmetry::SectorBasis adopted =
        ed::symmetry::SectorBasis::adopt(std::move(copy), G);

    REQUIRE(adopted.dim() == built.dim());
    REQUIRE(adopted.group_size() == G);
    for (std::uint64_t i = 0; i < built.dim(); ++i) {
        const std::uint64_t rep = built.state_of(i);
        REQUIRE(adopted.index_of(rep) == built.index_of(rep));
    }
}
