// =============================================================================
// test_sector_thermo (Catch2 v3)
//
// Validates the matvec-unification audit follow-up:
//
//   1. `ed::core::combine_sector_thermodynamics` correctly recombines
//      per-sector ThermodynamicData into the full-Hilbert thermo via
//      free-energy Z-weighting. Reference is the dense thermodynamics
//      computed from the union of all sector spectra.
//
//   2. `ed::core::method_produces_sector_thermo` returns true exactly
//      for the finite-T methods that populate `EDResults::thermo_data`.
//
// The streaming-symmetry kernel calls (1) automatically after the
// per-sector loop for FTLM / LTLM / HYBRID / KPM_DOS / mTPQ, so a
// correct combiner is necessary for
// ``ed::exact_diagonalization(method=FTLM, use_symmetry=true)`` to
// produce the same thermo as the full-Hilbert FTLM call.
// =============================================================================

#include "common/catch2_harness.h"

#include <ed/core/sector_thermo.h>
#include <ed/core/thermal_types.h>
#include <ed/solvers/observables.h>

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

using namespace ed_tests;

namespace {

// Diagonalize the operator densely and bucket the spectrum into
// "sectors" by a coarse partition. We use this to fabricate per-sector
// thermo blocks that combine_sector_thermodynamics must merge into the
// total thermo block that calculate_thermodynamics_from_spectrum
// produces directly on the union of spectra.
struct PartitionedSpectra {
    std::vector<std::vector<double>> per_sector_eigs;
    std::vector<double> full_eigs;
};

inline PartitionedSpectra partition_spectrum(const std::vector<double>& eigs,
                                             std::size_t num_sectors) {
    PartitionedSpectra out;
    out.full_eigs = eigs;
    out.per_sector_eigs.assign(num_sectors, {});
    for (std::size_t i = 0; i < eigs.size(); ++i) {
        out.per_sector_eigs[i % num_sectors].push_back(eigs[i]);
    }
    return out;
}

inline std::vector<double> heisenberg_chain_spectrum(uint64_t N) {
    auto H = build_heisenberg_chain(N, /*J=*/1.0, /*periodic=*/true);
    auto ref = reference_from_operator(*H, 1ULL << N);
    return ref.eigs;
}

} // namespace

TEST_CASE("combine_sector_thermodynamics: full-spectrum partition matches "
          "dense thermo on N=4 Heisenberg PBC",
          "[sector_thermo]") {
    const auto eigs = heisenberg_chain_spectrum(/*N=*/4);
    const double T_min = 0.1, T_max = 10.0;
    const std::uint64_t num_bins = 24;

    const auto ref =
        calculate_thermodynamics_from_spectrum(eigs, T_min, T_max, num_bins);

    // Partition the spectrum into 3 sectors -- arbitrary, the combiner
    // must be invariant under partition choice.
    auto parts = partition_spectrum(eigs, /*num_sectors=*/3);

    std::vector<ThermodynamicData> per_sector_thermo;
    std::vector<std::uint64_t>     per_sector_dims;
    for (const auto& sub : parts.per_sector_eigs) {
        if (sub.empty()) continue;
        per_sector_thermo.push_back(
            calculate_thermodynamics_from_spectrum(sub, T_min, T_max, num_bins));
        per_sector_dims.push_back(sub.size());
    }

    const auto combined = ed::core::combine_sector_thermodynamics(
        per_sector_thermo, per_sector_dims);

    REQUIRE(combined.temperatures.size() == ref.temperatures.size());
    double max_err_E = 0.0, max_err_Cv = 0.0, max_err_F = 0.0;
    for (std::size_t t = 0; t < num_bins; ++t) {
        max_err_E  = std::max(max_err_E,
                              std::abs(combined.energy[t]        - ref.energy[t]));
        max_err_Cv = std::max(max_err_Cv,
                              std::abs(combined.specific_heat[t] - ref.specific_heat[t]));
        max_err_F  = std::max(max_err_F,
                              std::abs(combined.free_energy[t]   - ref.free_energy[t]));
    }
    INFO("max |ΔE|=" << max_err_E
         << "  max |ΔCv|=" << max_err_Cv
         << "  max |ΔF|=" << max_err_F);
    REQUIRE(max_err_E  < 1e-8);
    REQUIRE(max_err_Cv < 1e-8);
    REQUIRE(max_err_F  < 1e-8);
}

TEST_CASE("combine_sector_thermodynamics: partition invariance "
          "(2 vs 4 sectors on N=6)",
          "[sector_thermo]") {
    // Splitting the same 64-state spectrum into 2 buckets vs 4 buckets
    // should give the same combined thermo (the free-energy Z-weight
    // doesn't care about how the spectrum is bucketed, only about which
    // eigenvalues sit in which Z_s).
    const auto eigs = heisenberg_chain_spectrum(/*N=*/6);
    const double T_min = 0.05, T_max = 5.0;
    const std::uint64_t num_bins = 16;

    auto bucket = [&](std::size_t S) {
        auto parts = partition_spectrum(eigs, S);
        std::vector<ThermodynamicData> per;
        std::vector<std::uint64_t>     dims;
        for (const auto& sub : parts.per_sector_eigs) {
            if (sub.empty()) continue;
            per.push_back(calculate_thermodynamics_from_spectrum(
                sub, T_min, T_max, num_bins));
            dims.push_back(sub.size());
        }
        return ed::core::combine_sector_thermodynamics(per, dims);
    };

    auto c2 = bucket(2);
    auto c4 = bucket(4);
    double max_dE = 0.0, max_dCv = 0.0;
    for (std::size_t t = 0; t < num_bins; ++t) {
        max_dE  = std::max(max_dE,
                           std::abs(c2.energy[t]        - c4.energy[t]));
        max_dCv = std::max(max_dCv,
                           std::abs(c2.specific_heat[t] - c4.specific_heat[t]));
    }
    INFO("2-vs-4 partition diff: max |ΔE|=" << max_dE
         << "  max |ΔCv|=" << max_dCv);
    REQUIRE(max_dE  < 1e-8);
    REQUIRE(max_dCv < 1e-8);
}

TEST_CASE("combine_sector_thermodynamics: rejects mismatched input "
          "shapes",
          "[sector_thermo]") {
    ThermodynamicData a, b;
    a.temperatures = {1.0, 2.0};
    a.energy = {0.0, 0.0};
    a.specific_heat = {0.0, 0.0};
    a.free_energy = {0.0, 0.0};
    b.temperatures = {1.0, 2.0, 3.0};
    b.energy = {0.0, 0.0, 0.0};
    b.specific_heat = {0.0, 0.0, 0.0};
    b.free_energy = {0.0, 0.0, 0.0};

    REQUIRE_THROWS_AS(
        ed::core::combine_sector_thermodynamics({a, b}, {1, 1}),
        std::invalid_argument);
    REQUIRE_THROWS_AS(
        ed::core::combine_sector_thermodynamics({a}, {1, 1}),
        std::invalid_argument);
    REQUIRE_THROWS_AS(
        ed::core::combine_sector_thermodynamics({}, {}),
        std::invalid_argument);
}

TEST_CASE("method_produces_sector_thermo: classifies the finite-T methods "
          "exactly",
          "[sector_thermo]") {
    using ed::core::method_produces_sector_thermo;

    // Finite-temperature -- must require recombination.
    REQUIRE(method_produces_sector_thermo(DiagonalizationMethod::FTLM));
    REQUIRE(method_produces_sector_thermo(DiagonalizationMethod::LTLM));
    REQUIRE(method_produces_sector_thermo(DiagonalizationMethod::KPM_DOS));
    REQUIRE(method_produces_sector_thermo(DiagonalizationMethod::mTPQ));

    // Ground-state methods -- thermo block stays empty, no recombination.
    REQUIRE_FALSE(method_produces_sector_thermo(DiagonalizationMethod::FULL));
    REQUIRE_FALSE(method_produces_sector_thermo(DiagonalizationMethod::LANCZOS));
    REQUIRE_FALSE(method_produces_sector_thermo(DiagonalizationMethod::BLOCK_LANCZOS));
    REQUIRE_FALSE(method_produces_sector_thermo(DiagonalizationMethod::KRYLOV_SCHUR));
}
