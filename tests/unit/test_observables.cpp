// =============================================================================
// test_observables
//
// Light-weight sanity checks for `calculate_thermodynamics_from_spectrum()`.
// We verify:
//   * sizes match num_points,
//   * E(T) is monotonic in T on a two-level system,
//   * the free-energy identity F = E - T*S holds,
//   * at T->0 the energy approaches the ground state,
//   * at T->infinity the energy approaches the unweighted average.
// =============================================================================

#include "common/test_harness.h"

#include <ed/core/thermal_types.h>
#include <ed/solvers/observables.h>

#include <algorithm>
#include <cmath>
#include <vector>

// Implemented in observables.cpp, declared (inconsistently) as a free
// function. Forward-declare to avoid pulling the full header.
ThermodynamicData calculate_thermodynamics_from_spectrum(
    const std::vector<double>& eigenvalues,
    double T_min, double T_max, uint64_t num_points);

using namespace ed_tests;

static void test_two_level(TestContext& ctx) {
    // H = diag(0, Δ) with degeneracies g0=1, g1=1. Then:
    //   E(T)    = Δ / (1 + exp(Δ/T))
    //   at T=0      => E=0, S=0
    //   at T=inf    => E=Δ/2, S=ln(2)
    const double dE = 1.0;
    std::vector<double> eigs = {0.0, dE};

    const uint64_t N = 40;
    auto r = calculate_thermodynamics_from_spectrum(eigs, 0.01, 100.0, N);

    check(ctx, r.temperatures.size() == N, "temperatures size = num_points");
    check(ctx, r.energy.size() == N, "energy size = num_points");
    check(ctx, r.specific_heat.size() == N, "specific_heat size = num_points");
    check(ctx, r.entropy.size() == N, "entropy size = num_points");
    check(ctx, r.free_energy.size() == N, "free_energy size = num_points");

    // Monotonic E(T).
    bool mono = true;
    for (uint64_t i = 1; i < N; ++i) {
        if (r.energy[i] + 1e-12 < r.energy[i - 1]) { mono = false; break; }
    }
    check(ctx, mono, "E(T) monotonically non-decreasing");

    // F = E - T*S within a reasonable tolerance.
    double max_dev = 0.0;
    for (uint64_t i = 0; i < N; ++i) {
        double lhs = r.free_energy[i];
        double rhs = r.energy[i] - r.temperatures[i] * r.entropy[i];
        max_dev = std::max(max_dev, std::abs(lhs - rhs));
    }
    check(ctx, max_dev < 1e-6,
          "F = E - T*S identity",
          "max |F - (E - T*S)| = " + std::to_string(max_dev));

    // Low-T and high-T limits.
    check(ctx, std::abs(r.energy.front()) < 1e-6,
          "E(T_min) ~ ground state",
          "E(T_min) = " + std::to_string(r.energy.front()));
    check(ctx, std::abs(r.energy.back() - 0.5 * dE) < 1e-2,
          "E(T_max) ~ mean energy",
          "E(T_max) = " + std::to_string(r.energy.back()));
}

static void test_heisenberg_N4_partition(TestContext& ctx) {
    // Full spectrum of the 4-site OBC Heisenberg chain, computed from the
    // same apply() path the main code uses. Verify that:
    //   * specific heat is non-negative,
    //   * entropy is bounded by ln(dim),
    //   * energy is bounded by min and max eigenvalue.
    auto op = build_heisenberg_chain(4, 1.0);
    auto ref = reference_from_operator(*op, 16);

    auto r = calculate_thermodynamics_from_spectrum(ref.eigs, 0.05, 50.0, 30);

    double e_lo = ref.eigs.front();
    double e_hi = ref.eigs.back();
    double S_max = std::log(static_cast<double>(ref.eigs.size())) + 1e-9;

    bool cv_ok = true, s_ok = true, e_ok = true;
    double worst_cv = 0, worst_s = 0, worst_e = 0;
    for (uint64_t i = 0; i < r.temperatures.size(); ++i) {
        if (r.specific_heat[i] < -1e-9) { cv_ok = false; worst_cv = std::min(worst_cv, r.specific_heat[i]); }
        if (r.entropy[i] < -1e-9 || r.entropy[i] > S_max) { s_ok = false; worst_s = r.entropy[i]; }
        if (r.energy[i] < e_lo - 1e-8 || r.energy[i] > e_hi + 1e-8) {
            e_ok = false; worst_e = r.energy[i];
        }
    }
    check(ctx, cv_ok, "N=4 Heisenberg Cv(T) >= 0 everywhere",
          "worst Cv = " + std::to_string(worst_cv));
    check(ctx, s_ok, "N=4 Heisenberg 0 <= S(T) <= ln(dim) everywhere",
          "worst S = " + std::to_string(worst_s) +
              " max allowed = " + std::to_string(S_max));
    check(ctx, e_ok, "N=4 Heisenberg E(T) within spectrum bounds",
          "worst E = " + std::to_string(worst_e));
}

int main() {
    TestContext ctx("test_observables");
    test_two_level(ctx);
    test_heisenberg_N4_partition(ctx);
    return ctx.summary_exit_code();
}
