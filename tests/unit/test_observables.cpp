// =============================================================================
// test_observables (Catch2 v3, P1.8 / audit Q12)
//
// Light-weight sanity checks for `calculate_thermodynamics_from_spectrum()`.
// We verify:
//   * sizes match num_points,
//   * E(T) is monotonic in T on a two-level system,
//   * the free-energy identity F = E - T*S holds,
//   * at T->0 the energy approaches the ground state,
//   * at T->infinity the energy approaches the unweighted average.
// =============================================================================

#include "common/catch2_harness.h"

#include <ed/core/thermal_types.h>
#include <ed/solvers/observables.h>

#include <algorithm>
#include <cmath>
#include <vector>

ThermodynamicData calculate_thermodynamics_from_spectrum(
    const std::vector<double>& eigenvalues,
    double T_min, double T_max, uint64_t num_points);

using namespace ed_tests;

TEST_CASE("Two-level system thermodynamics", "[observables][two_level]") {
    // H = diag(0, Δ) with degeneracies g0=1, g1=1. Then:
    //   E(T) = Δ / (1 + exp(Δ/T))
    //   at T=0      => E=0, S=0
    //   at T=inf    => E=Δ/2, S=ln(2)
    const double dE = 1.0;
    std::vector<double> eigs = {0.0, dE};

    const uint64_t N = 40;
    auto r = calculate_thermodynamics_from_spectrum(eigs, 0.01, 100.0, N);

    REQUIRE(r.temperatures.size() == N);
    REQUIRE(r.energy.size() == N);
    REQUIRE(r.specific_heat.size() == N);
    REQUIRE(r.entropy.size() == N);
    REQUIRE(r.free_energy.size() == N);

    SECTION("E(T) monotonically non-decreasing") {
        for (uint64_t i = 1; i < N; ++i) {
            INFO("i=" << i << " E[i-1]=" << r.energy[i - 1]
                 << " E[i]=" << r.energy[i]);
            REQUIRE(r.energy[i] + 1e-12 >= r.energy[i - 1]);
        }
    }

    SECTION("F = E - T*S identity") {
        double max_dev = 0.0;
        for (uint64_t i = 0; i < N; ++i) {
            double lhs = r.free_energy[i];
            double rhs = r.energy[i] - r.temperatures[i] * r.entropy[i];
            max_dev = std::max(max_dev, std::abs(lhs - rhs));
        }
        INFO("max |F - (E - T*S)| = " << max_dev);
        REQUIRE(max_dev < 1e-6);
    }

    SECTION("low-T limit: E(T_min) ~ ground state") {
        INFO("E(T_min) = " << r.energy.front());
        REQUIRE(std::abs(r.energy.front()) < 1e-6);
    }

    SECTION("high-T limit: E(T_max) ~ mean energy") {
        INFO("E(T_max) = " << r.energy.back());
        REQUIRE(std::abs(r.energy.back() - 0.5 * dE) < 1e-2);
    }
}

TEST_CASE("Heisenberg N=4 partition function bounds",
          "[observables][heisenberg]") {
    auto op = build_heisenberg_chain(4, 1.0);
    auto ref = reference_from_operator(*op, 16);

    auto r = calculate_thermodynamics_from_spectrum(ref.eigs, 0.05, 50.0, 30);

    double e_lo = ref.eigs.front();
    double e_hi = ref.eigs.back();
    double S_max = std::log(static_cast<double>(ref.eigs.size())) + 1e-9;

    for (uint64_t i = 0; i < r.temperatures.size(); ++i) {
        INFO("i=" << i << " T=" << r.temperatures[i]
             << " Cv=" << r.specific_heat[i]
             << " S=" << r.entropy[i]
             << " E=" << r.energy[i]);
        REQUIRE(r.specific_heat[i] >= -1e-9);
        REQUIRE(r.entropy[i] >= -1e-9);
        REQUIRE(r.entropy[i] <= S_max);
        REQUIRE(r.energy[i] >= e_lo - 1e-8);
        REQUIRE(r.energy[i] <= e_hi + 1e-8);
    }
}
