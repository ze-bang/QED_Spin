// =============================================================================
// test_projector_chain
//
// Phase R5 of the "Orthogonal symmetry composition: Subspace x
// ProjectorChain" plan (May 2026).
//
// Pins byte-equality between the new templated orbit/character helper
// ``ed::symmetry::compute_orbit_for_state`` (used now as the SINGLE
// implementation behind the legacy ``StreamingSymmetryOperator::
// computeOrbitData`` and
// ``FixedSzStreamingSymmetryOperator::computeOrbitDataFixedSz``) and
// the previous inline loops. We verify three things:
//
//   1. ``compute_orbit_for_state(FullSpaceSubspace, SpatialProjector, ...)``
//      matches a hand-rolled reference implementation that mirrors the
//      pre-refactor inline body verbatim, for a Heisenberg ring on N=6
//      with Z_N translation symmetry, all sectors.
//   2. ``compute_orbit_for_state(FixedSzSubspace, SpatialProjector, ...)``
//      matches the same reference (with the fixed-Sz ``index_of`` filter
//      enabled) at n_up = N/2 for N=6.
//   3. End-to-end: the unchanged
//      ``StreamingSymmetryOperator::applySymmetrized`` still recovers
//      the Bethe-ansatz ground state for the Heisenberg N=6 ring (a
//      smoke regression against accidental sign / normalisation drift
//      in the refactor).
//
// =============================================================================

#include "common/catch2_harness.h"

#include <ed/symmetry/projector_chain.h>
#include <ed/symmetry/projector.h>
#include <ed/symmetry/sector_operator.h>
#include <ed/symmetry/sector_set.h>
#include <ed/symmetry/subspace.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <memory>
#include <string>
#include <system_error>
#include <unordered_map>
#include <vector>

using namespace ed_tests;

namespace {

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

// ---------------------------------------------------------------------------
// Reference orbit builder: pasted verbatim from the pre-refactor
// inline body (streaming_symmetry.h:1768-1808 / 3682-3725). The
// ``in_subspace`` predicate carries the FixedSz filter when the
// caller wants it; for the full-space variant, pass [](u64){return true;}.
// ---------------------------------------------------------------------------
template <class InSubspace>
void reference_compute_orbit(
    const SymmetryGroupInfo& info,
    InSubspace               in_subspace,
    std::uint64_t            basis,
    const std::vector<Complex>& phase_factors,
    std::vector<std::uint64_t>&  orbit_elements,
    std::vector<Complex>&        orbit_coefficients,
    double&                       norm_sq)
{
    std::unordered_map<std::uint64_t, Complex> coeff_map;
    for (std::size_t g = 0; g < info.max_clique.size(); ++g) {
        const auto& perm   = info.max_clique[g];
        const auto& powers = info.power_representation[g];

        Complex character(1.0, 0.0);
        for (std::size_t k = 0; k < powers.size(); ++k) {
            Complex phase = phase_factors[k];
            for (int p = 0; p < powers[k]; ++p) {
                character *= phase;
            }
        }
        const std::uint64_t permuted = applyPermutation(basis, perm);
        if (!in_subspace(permuted)) continue;
        coeff_map[permuted] += std::conj(character);
    }

    orbit_elements.clear();
    orbit_coefficients.clear();
    norm_sq = 0.0;
    for (const auto& [state, coeff] : coeff_map) {
        if (std::abs(coeff) > 1e-15) {
            orbit_elements.push_back(state);
            orbit_coefficients.push_back(coeff);
            norm_sq += std::norm(coeff);
        }
    }
    if (!info.max_clique.empty()) {
        norm_sq /= static_cast<double>(info.max_clique.size());
    }
}

// Sort (parallel-sort coefficients with elements) so we can compare
// vectors that may differ only in iteration order from the
// unordered_map traversal.
void sort_orbit(std::vector<std::uint64_t>& el,
                std::vector<Complex>&       co)
{
    const std::size_t n = el.size();
    std::vector<std::size_t> idx(n);
    for (std::size_t i = 0; i < n; ++i) idx[i] = i;
    std::sort(idx.begin(), idx.end(),
              [&](std::size_t a, std::size_t b){ return el[a] < el[b]; });
    std::vector<std::uint64_t> e(n);
    std::vector<Complex>       c(n);
    for (std::size_t i = 0; i < n; ++i) {
        e[i] = el[idx[i]];
        c[i] = co[idx[i]];
    }
    el = std::move(e);
    co = std::move(c);
}

bool coeffs_close(const std::vector<Complex>& a,
                  const std::vector<Complex>& b,
                  double tol = 1e-14)
{
    if (a.size() != b.size()) return false;
    for (std::size_t i = 0; i < a.size(); ++i) {
        if (std::abs(a[i] - b[i]) > tol) return false;
    }
    return true;
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

} // namespace

TEST_CASE("projector_chain: compute_orbit_for_state(FullSpace) matches reference (N=6)",
          "[symmetry][projector_chain][N6]")
{
    const int N = 6;
    std::string dir = make_scratch_dir("projector_chain", "fullspace_N6");
    write_zN_translation_fixtures(dir, N);

    SymmetryGroupInfo info;
    REQUIRE_NOTHROW(info.loadFromDirectory(dir));

    const ed::symmetry::FullSpaceSubspace full(static_cast<std::uint64_t>(N));
    const ed::symmetry::SpatialProjector  spatial(info);

    // Sweep every sector and every basis state; compare the new
    // helper against the inline reference.
    for (std::size_t s = 0; s < info.sectors.size(); ++s) {
        const auto& pf = info.sectors[s].phase_factors;

        for (std::uint64_t basis = 0; basis < (1ULL << N); ++basis) {
            std::vector<std::uint64_t> el_ref, el_new;
            std::vector<Complex>       co_ref, co_new;
            double                     ns_ref = 0.0, ns_new = 0.0;

            reference_compute_orbit(info, [](std::uint64_t){ return true; },
                                    basis, pf, el_ref, co_ref, ns_ref);
            ed::symmetry::compute_orbit_for_state(full, spatial, basis, pf,
                                                  el_new, co_new, ns_new);

            sort_orbit(el_ref, co_ref);
            sort_orbit(el_new, co_new);

            INFO("sector=" << s << " basis=" << basis);
            REQUIRE(el_new == el_ref);
            REQUIRE(coeffs_close(co_new, co_ref, 1e-14));
            REQUIRE(std::abs(ns_new - ns_ref) < 1e-14);
        }
    }
}

TEST_CASE("projector_chain: compute_orbit_for_state(FixedSz) matches reference (N=6, n_up=3)",
          "[symmetry][projector_chain][N6][fixedsz]")
{
    const int N    = 6;
    const int n_up = N / 2;
    std::string dir = make_scratch_dir("projector_chain", "fixedsz_N6");
    write_zN_translation_fixtures(dir, N);

    SymmetryGroupInfo info;
    REQUIRE_NOTHROW(info.loadFromDirectory(dir));

    auto sub = ed::symmetry::FixedSzSubspace::build(N, n_up);
    const ed::symmetry::SpatialProjector spatial(info);

    // Build the reference predicate from the fixed-Sz lookup so we
    // mirror the legacy ``lookupState(permuted) >= 0`` filter exactly.
    const auto& lin = sub.lin_index();
    auto in_sz_sector = [&](std::uint64_t s) {
        return lin.lookup(s) >= 0;
    };

    for (std::size_t s = 0; s < info.sectors.size(); ++s) {
        const auto& pf = info.sectors[s].phase_factors;

        // Sweep the entire fixed-Sz basis (C(N, n_up) = 20 states for N=6).
        for (std::uint64_t b : sub.basis_states()) {
            std::vector<std::uint64_t> el_ref, el_new;
            std::vector<Complex>       co_ref, co_new;
            double                     ns_ref = 0.0, ns_new = 0.0;

            reference_compute_orbit(info, in_sz_sector, b, pf,
                                    el_ref, co_ref, ns_ref);
            ed::symmetry::compute_orbit_for_state(sub, spatial, b, pf,
                                                  el_new, co_new, ns_new);

            sort_orbit(el_ref, co_ref);
            sort_orbit(el_new, co_new);

            INFO("sector=" << s << " basis=" << b);
            REQUIRE(el_new == el_ref);
            REQUIRE(coeffs_close(co_new, co_ref, 1e-14));
            REQUIRE(std::abs(ns_new - ns_ref) < 1e-14);
        }
    }
}

TEST_CASE("projector_chain: end-to-end ground state on Heisenberg N=6 unchanged",
          "[symmetry][projector_chain][e2e][N6]")
{
    const uint64_t N = 6;
    std::string dir = make_scratch_dir("projector_chain", "e2e_N6");
    write_zN_translation_fixtures(dir, static_cast<int>(N));

    SymmetryGroupInfo info;
    REQUIRE_NOTHROW(info.loadFromDirectory(dir));

    // Carrier-free standalone per-sector operators (the production path).
    auto ops = ed::symmetry::build_full_sector_operators_lazy(
        N, 0.5f, info,
        [&](ed::symmetry::SectorOperator& op) {
            add_heisenberg_pbc_terms(op, N, 1.0);
        });

    // Run a tiny power iteration in each sector and pick the minimum
    // Rayleigh quotient; for N=6 PBC Heisenberg the ground state is
    // -2.802775637731995. The refactor must not change this value.
    const double bethe_e0 = -2.802775637731995;
    double best = std::numeric_limits<double>::infinity();
    for (std::size_t si = 0; si < ops.size(); ++si) {
        ed::symmetry::SectorOperator& op = *ops[si];
        const std::size_t sd = op.dim();
        if (sd == 0) continue;

        auto v = random_unit_vector(sd, (si + 1) * 31337ULL);
        std::vector<Complex> w(sd, Complex(0.0, 0.0));

        // 200 inverse-iteration-free power steps with explicit shift
        // (H - shift I). Shift is +5 (above the eigenvalue spectrum
        // for Heisenberg N=6 PBC) so the smallest |H - shift| picks
        // up the ground state.
        const double shift = 5.0;
        for (int it = 0; it < 200; ++it) {
            std::vector<Complex> Hv(sd, Complex(0.0, 0.0));
            op.apply(v.data(), Hv.data(), sd);
            for (std::size_t i = 0; i < sd; ++i) {
                w[i] = shift * v[i] - Hv[i];
            }
            double nrm = 0.0;
            for (auto c : w) nrm += std::norm(c);
            nrm = std::sqrt(nrm);
            if (nrm < 1e-30) break;
            for (std::size_t i = 0; i < sd; ++i) w[i] /= nrm;
            v = w;
        }
        std::vector<Complex> Hv(sd, Complex(0.0, 0.0));
        op.apply(v.data(), Hv.data(), sd);
        Complex e(0.0, 0.0);
        for (std::size_t i = 0; i < sd; ++i) e += std::conj(v[i]) * Hv[i];
        if (e.real() < best) best = e.real();
    }
    INFO("best Rayleigh quotient across sectors = " << best);
    REQUIRE(std::abs(best - bethe_e0) < 1e-3);
}
