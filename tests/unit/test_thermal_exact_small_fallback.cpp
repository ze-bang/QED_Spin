// =============================================================================
// test_thermal_exact_small_fallback  (Catch2 v3)
//
// Pins the orchestrator's small-dimension exact-thermal fallback:
// for D <= SMALL_THERMAL_DIM (512), every SAMPLING thermodynamics method
// (mTPQ / FTLM / LTLM / OFTLM) is answered by an exact eigensolve instead of
// its stochastic estimator.
//
// Why this file exists
// --------------------
// Until Jul 2026 the gate required mTPQ specifically, so FTLM/LTLM kept
// sampling in a regime where the exact solve is free AND machine precise --
// measured at dim=64: mTPQ 1.4e-15 vs FTLM/LTLM 2.3e-02, i.e. 13 orders for
// microseconds of eigensolve. The deliverable of all four methods here is
// identical (canonical E/C/S), so all four take the exact route.
//
// The fallback is invisible to a tolerance-based check -- it makes things
// MORE accurate -- so it needs its own pin: assert machine precision, which
// only the exact path can deliver. Conversely test_thermal_dense_ref sets
// ED_THERMAL_EXACT_SMALL=0 so it keeps gating the real kernels; that escape
// is pinned here too, since the two files' contracts are complementary and a
// regression in either direction should fail exactly one of them.
//
// KpmDos is deliberately NOT in the fallback: its deliverable includes the
// Chebyshev density of states, which the exact path does not produce (the
// same rationale as the probe_betas carve-out).
// =============================================================================

#include "common/catch2_harness.h"
#include "common/test_harness.h"

#include <ed/core/operator.h>
#include <ed/core/thermal_types.h>
#include <ed/orchestrator.h>

#include <cmath>
#include <cstdlib>
#include <string>
#include <vector>

using namespace ed_tests;
using ed::workflows::ThermalOptions;

namespace {

constexpr uint64_t N_SITES = 6;      // dim = 64 << SMALL_THERMAL_DIM = 512
constexpr double   J       = 1.0;

// A broad grid: the pre-Jul-2026 LTLM bug hid at low T, so a fallback pin that
// only looked at T -> 0 would repeat that mistake.
const std::vector<double> BETAS = [] {
    std::vector<double> b;
    for (double T : {0.05, 0.25, 1.0, 3.0, 10.0}) b.push_back(1.0 / T);
    return b;
}();

// Exact canonical E(T) from a dense spectrum of the N=6 Heisenberg ring.
std::vector<double> exact_energies() {
    auto H = build_heisenberg_chain(N_SITES, J, /*periodic=*/true);
    const auto eigs = reference_from_operator(*H, 1ULL << N_SITES).eigs;
    std::vector<double> E;
    for (double beta : BETAS) {
        double Z = 0.0, num = 0.0;
        for (double e : eigs) {
            const double w = std::exp(-beta * (e - eigs.front()));
            Z += w;
            num += e * w;
        }
        E.push_back(num / Z);
    }
    return E;
}

ThermalOptions base_opts(ThermalOptions::Method m) {
    ThermalOptions o;
    o.method      = m;
    o.betas       = BETAS;
    o.num_samples = 4;      // deliberately far too few to be accurate...
    o.krylov_dim  = 8;      // ...and a far too short Krylov space.
    o.random_seed = 12345ULL;
    o.backend.allow_gpu = false;
    return o;
}

struct EnvGuard {
    const char* name;
    std::string saved;
    bool had;
    EnvGuard(const char* n, const char* v) : name(n) {
        const char* cur = std::getenv(n);
        had = cur != nullptr;
        if (had) saved = cur;
        if (v) ::setenv(n, v, 1); else ::unsetenv(n);
    }
    ~EnvGuard() {
        if (had) ::setenv(name, saved.c_str(), 1); else ::unsetenv(name);
    }
};

}  // namespace

TEST_CASE("ed::thermal: D <= SMALL_THERMAL_DIM is exact for every sampling method",
          "[thermal][exact_fallback]") {
    // Default behaviour (gate unset) -- do not inherit a stray value.
    EnvGuard g("ED_THERMAL_EXACT_SMALL", nullptr);

    const auto E_exact = exact_energies();

    // The knobs above (4 samples, krylov 8) could not reach 1e-12 by sampling
    // at ANY temperature: machine precision here proves the exact path ran.
    for (auto m : {ThermalOptions::Method::mTPQ,
                   ThermalOptions::Method::FTLM,
                   ThermalOptions::Method::LTLM,
                   ThermalOptions::Method::OFTLM}) {
        auto H = build_heisenberg_chain(N_SITES, J, /*periodic=*/true);
        auto R = ed::workflows::thermal(*H, base_opts(m));

        REQUIRE(R.thermo.energy.size() == BETAS.size());
        for (std::size_t i = 0; i < BETAS.size(); ++i) {
            INFO("method index " << static_cast<int>(m)
                 << " at T=" << 1.0 / BETAS[i]);
            REQUIRE(std::abs(R.thermo.energy[i] - E_exact[i]) < 1e-10);
        }
    }
}

TEST_CASE("ed::thermal: ED_THERMAL_EXACT_SMALL=0 restores the sampling kernel",
          "[thermal][exact_fallback]") {
    // The escape exists so accuracy tests can still gate the estimators
    // (test_thermal_dense_ref). If it silently stopped working, that file
    // would go on "passing" while testing nothing.
    EnvGuard g("ED_THERMAL_EXACT_SMALL", "0");

    const auto E_exact = exact_energies();

    auto H = build_heisenberg_chain(N_SITES, J, /*periodic=*/true);
    auto R = ed::workflows::thermal(*H, base_opts(ThermalOptions::Method::FTLM));
    REQUIRE(R.thermo.energy.size() == BETAS.size());

    // With 4 samples / krylov 8 the estimator must MISS somewhere on this
    // grid; if every point were exact, the fallback ran despite the escape.
    double worst = 0.0;
    for (std::size_t i = 0; i < BETAS.size(); ++i)
        worst = std::max(worst, std::abs(R.thermo.energy[i] - E_exact[i]));
    INFO("worst |dE| under forced sampling = " << worst);
    REQUIRE(worst > 1e-10);
}
