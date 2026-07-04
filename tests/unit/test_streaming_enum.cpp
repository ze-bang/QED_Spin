// =============================================================================
// tests/unit/test_streaming_enum.cpp
//
// Bit-identity coverage for the streaming/tableless basis-construction campaign:
//   1. enumerate_fixed_sz_orbit_reps_streaming == legacy enumerate_fixed_sz_orbit_reps
//      (same reps, same ascending order) over a Z_N translation group + fixed Sz.
//   2. compute_orbit_for_state with FixedSzMembershipSubspace yields per-rep norms
//      bit-identical to the materialized FixedSzSubspace.
//   3. enumerate_full_orbit_reps (now OpenMP-parallel) == a serial reference.
//   4. combinadic rank/unrank round-trips over the whole fixed-Sz basis, and the
//      rank equals the ascending-order index (so it is index-compatible with the
//      materialized FixedSzBasisPolicy).
// =============================================================================

#include "common/catch2_harness.h"

#include <ed/core/basis_utils.h>
#include <ed/core/combinadic.h>
#include <ed/planner/basis_policy_hook.h>
#include <ed/symmetry/fixed_sz_membership.h>
#include <ed/symmetry/projector.h>
#include <ed/symmetry/projector_chain.h>
#include <ed/symmetry/sector_set.h>
#include <ed/symmetry/subspace.h>

#include <complex>
#include <cstdint>
#include <fstream>
#include <filesystem>
#include <unistd.h>
#include <iomanip>
#include <random>
#include <string>
#include <vector>

using Complex = std::complex<double>;

namespace {

std::vector<int> translation_perm(int N, int shift) {
    std::vector<int> p(N);
    for (int i = 0; i < N; ++i) p[i] = ((i - shift) % N + N) % N;
    return p;
}

// Minimal Z_N translation group fixtures (mirrors test_sector_set.cpp).
std::string write_zN(const std::string& base, int N) {
    const std::string dir = base + "/zN_" + std::to_string(N);
    const std::string root = dir + "/automorphism_results";
    std::error_code ec;
    std::filesystem::create_directories(root, ec);
    {
        std::ofstream f(root + "/max_clique.json");
        f << "[";
        for (int g = 0; g < N; ++g) {
            const auto p = translation_perm(N, g);
            f << "[";
            for (size_t i = 0; i < p.size(); ++i)
                f << p[i] << (i + 1 < p.size() ? "," : "");
            f << "]" << (g + 1 < N ? "," : "");
        }
        f << "]";
    }
    {
        std::ofstream f(root + "/minimal_generators.json");
        const auto p = translation_perm(N, 1);
        f << "{\"generators\":[{\"permutation\":[";
        for (size_t i = 0; i < p.size(); ++i)
            f << p[i] << (i + 1 < p.size() ? "," : "");
        f << "],\"order\":" << N << "}]}";
    }
    {
        std::ofstream f(root + "/sector_metadata.json");
        f << std::setprecision(17) << "{\"sectors\":[";
        for (int k = 0; k < N; ++k) {
            const double angle = -2.0 * M_PI * double(k) / double(N);
            f << "{\"sector_id\":" << k << ",\"quantum_numbers\":[" << k << "]"
              << ",\"phase_factors\":[{\"real\":" << std::cos(angle)
              << ",\"imag\":" << std::sin(angle) << "}]}"
              << (k + 1 < N ? "," : "");
        }
        f << "]}";
    }
    return dir;
}

SymmetryGroupInfo load_zN(int N) {
    // Per-PROCESS fixture dir: multiple test binaries in a parallel ctest
    // run call load_zN concurrently; a shared path races one binary's
    // rewrite against another's read (observed on CI as "attempting to
    // parse an empty input" for sector_metadata.json).
    const std::string base =
        (std::filesystem::temp_directory_path() /
         ("qed_stream_enum_" + std::to_string(::getpid()))).string();
    const std::string dir = write_zN(base, N);
    SymmetryGroupInfo info;
    info.loadFromDirectory(dir);
    return info;
}

}  // namespace

TEST_CASE("streaming fixed-Sz reps == legacy reps (ascending, bit-identical)",
          "[streaming][symmetry]") {
    for (int N : {8, 12, 16}) {
        const auto info = load_zN(N);
        const int n_up = N / 2;

        const auto fixed = ed::symmetry::FixedSzSubspace::build(N, n_up);
        const auto legacy = ed::symmetry::enumerate_fixed_sz_orbit_reps(fixed, info);
        const auto streamed =
            ed::symmetry::enumerate_fixed_sz_orbit_reps_streaming(N, n_up, info);

        REQUIRE(streamed.size() == legacy.size());
        REQUIRE(streamed == legacy);   // identical contents AND order
    }
}

TEST_CASE("membership-subspace orbit norms == materialized subspace norms",
          "[streaming][symmetry]") {
    const int N = 12, n_up = 6;
    const auto info = load_zN(N);
    const auto fixed = ed::symmetry::FixedSzSubspace::build(N, n_up);
    const ed::symmetry::FixedSzMembershipSubspace membership(N, n_up);
    const ed::symmetry::SpatialProjector projector(info);
    const auto reps = ed::symmetry::enumerate_fixed_sz_orbit_reps(fixed, info);

    const auto& phase = info.sectors[0].phase_factors;
    std::vector<std::uint64_t> e1, e2;
    std::vector<Complex>       c1, c2;
    for (std::uint64_t rep : reps) {
        double n_fixed = 0.0, n_mem = 0.0;
        ed::symmetry::compute_orbit_for_state(fixed, projector, rep, phase, e1, c1, n_fixed);
        ed::symmetry::compute_orbit_for_state(membership, projector, rep, phase, e2, c2, n_mem);
        REQUIRE(n_mem == n_fixed);     // exact bit-identity
        REQUIRE(e2 == e1);
    }
}

TEST_CASE("enumerate_full_orbit_reps parallel == serial reference",
          "[streaming][symmetry]") {
    const int N = 12;
    const auto info = load_zN(N);
    const auto reps = ed::symmetry::enumerate_full_orbit_reps(info, N);

    // Serial reference.
    std::vector<std::uint64_t> ref;
    const std::uint64_t dim = (1ull << N);
    for (std::uint64_t s = 0; s < dim; ++s) {
        bool is_rep = true;
        for (std::size_t g = 0; g < info.max_clique.size(); ++g)
            if (applyPermutation(s, info.max_clique[g]) < s) { is_rep = false; break; }
        if (is_rep) ref.push_back(s);
    }
    REQUIRE(reps == ref);
}

TEST_CASE("combinadic rank/unrank round-trips and matches ascending index",
          "[streaming][combinadic]") {
    for (int N : {8, 12, 16}) {
        const int n_up = N / 2;
        ed::core::combinadic::BinomialTable binom(N);
        const auto basis = generateFixedSzBasis(N, n_up);
        for (std::size_t idx = 0; idx < basis.size(); ++idx) {
            const std::uint64_t s = basis[idx];
            const std::int64_t r = ed::core::combinadic::rank_state(s, N, n_up, binom);
            REQUIRE(r == static_cast<std::int64_t>(idx));   // colex rank == ascending index
            REQUIRE(ed::core::combinadic::unrank_to_state(
                        static_cast<std::uint64_t>(r), N, n_up, binom) == s);
        }
    }
}

TEST_CASE("Track A: tableless combinadic fixed-Sz matvec == materialized matvec",
          "[streaming][combinadic][matvec]") {
    using namespace ed_tests;
    for (int N : {8, 12}) {
        const int n_up = N / 2;

        ed::planner::clear_basis_repr();
        auto mat = build_heisenberg_chain_fixed_sz(N, 1.0, n_up, /*periodic=*/true);

        ed::planner::set_basis_repr(ed::planner::BasisRepr::Tableless);
        auto tab = build_heisenberg_chain_fixed_sz(N, 1.0, n_up, /*periodic=*/true);
        ed::planner::clear_basis_repr();

        REQUIRE(mat->dim() == tab->dim());
        const std::size_t d = mat->dim();

        std::mt19937 gen(0xC0FFEE);
        std::uniform_real_distribution<double> dist(-1.0, 1.0);
        std::vector<Complex> x(d), ym(d), yt(d);
        for (auto& v : x) v = Complex(dist(gen), dist(gen));
        mat->apply(x.data(), ym.data(), d);
        tab->apply(x.data(), yt.data(), d);
        for (std::size_t i = 0; i < d; ++i)
            REQUIRE(std::abs(ym[i] - yt[i]) < 1e-12);
    }
}
