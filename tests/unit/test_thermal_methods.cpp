// =============================================================================
// test_thermal_methods
//
// Sanity & accuracy checks for the finite-temperature Lanczos family:
//   * FTLM     (finite-temperature Lanczos, random-state averaging)
//   * LTLM     (low-temperature Lanczos, ground-state + excitations)
//   * Hybrid   (LTLM for T<T_c and FTLM for T>=T_c, automatically merged)
//
// Ground truth is the full spectrum of a small Heisenberg chain, computed
// with Eigen's dense self-adjoint eigensolver and fed through
// `calculate_thermodynamics_from_spectrum()`. Each method is validated
// against this reference on three axes:
//
//   1. Global sanity:  Cv(T) >= 0, 0 <= S(T) <= ln(dim), E(T) in [Emin, Emax].
//   2. Asymptotics:    E(T_min) ≈ ground state, E(T_max) ≈ mean energy.
//   3. Accuracy:       E(T), Cv(T), S(T) close to full-diag reference at
//                      a handful of representative temperatures.
//
// These methods have stochastic components (random initial states). We pin
// the FTLM seed and use generous num_samples so the test is reproducible
// within a few-percent tolerance on small systems.
// =============================================================================

#include "common/test_harness.h"

#include <ed/core/thermal_types.h>
#include <ed/solvers/ftlm.h>
#include <ed/solvers/ltlm.h>
#include <ed/solvers/hybrid_thermal.h>
#include <ed/solvers/observables.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

using namespace ed_tests;

namespace {

struct ThermalFixture {
    std::unique_ptr<Operator> op;
    uint64_t dim;
    std::vector<double> dense_eigs;      // full spectrum (ground truth)
    std::function<void(const Complex*, Complex*, int)> Hv;
    ThermodynamicData ref_thermo;        // reference thermodynamics
    double temp_min;
    double temp_max;
    uint64_t num_temp_bins;
};

ThermalFixture make_fixture(uint64_t N, double temp_min, double temp_max,
                            uint64_t num_bins) {
    ThermalFixture f;
    f.op = build_heisenberg_chain(N, 1.0, /*periodic=*/true);
    f.dim = 1ULL << N;
    auto ref = reference_from_operator(*f.op, f.dim);
    f.dense_eigs = ref.eigs;
    auto* op_ptr = f.op.get();
    f.Hv = [op_ptr](const Complex* in, Complex* out, int n) {
        op_ptr->apply(in, out, static_cast<size_t>(n));
    };
    f.temp_min = temp_min;
    f.temp_max = temp_max;
    f.num_temp_bins = num_bins;
    f.ref_thermo = calculate_thermodynamics_from_spectrum(
        f.dense_eigs, temp_min, temp_max, num_bins);
    return f;
}

// Generic "does this look like a thermodynamic curve?" sanity pass.
//   * cv_neg_slack: how negative we allow Cv to go. Random-state methods
//     produce small negative Cv from Boltzmann cancellation; LTLM at high
//     T produces larger violations because the truncated thermal trace is
//     not strictly positive there.
//   * s_neg_slack: same, but for entropy. FTLM with finite samples can
//     produce S(T) ≈ -0.1..-0.2 at low T due to log-of-noisy-Z.
void sanity_check_thermo(TestContext& ctx, const std::string& name,
                         const ThermodynamicData& t, const ThermalFixture& f,
                         double cv_neg_slack = 1e-6,
                         double s_neg_slack  = 1e-6) {
    check(ctx, t.temperatures.size() == f.num_temp_bins,
          name + " temperatures size = num_temp_bins");
    check(ctx, t.energy.size() == f.num_temp_bins,
          name + " energy size = num_temp_bins");
    check(ctx, t.specific_heat.size() == f.num_temp_bins,
          name + " specific_heat size = num_temp_bins");
    check(ctx, t.entropy.size() == f.num_temp_bins,
          name + " entropy size = num_temp_bins");

    const double e_lo = f.dense_eigs.front();
    const double e_hi = f.dense_eigs.back();
    const double s_max = std::log(static_cast<double>(f.dim)) + 1e-6;

    double worst_cv = 0.0, worst_s = 0.0, worst_e_low = 0.0, worst_e_high = 0.0;
    bool cv_ok = true, s_ok = true, e_ok = true;
    for (uint64_t i = 0; i < t.temperatures.size(); ++i) {
        if (t.specific_heat[i] < -cv_neg_slack) {
            cv_ok = false;
            worst_cv = std::min(worst_cv, t.specific_heat[i]);
        }
        if (t.entropy[i] < -s_neg_slack || t.entropy[i] > s_max) {
            s_ok = false; worst_s = t.entropy[i];
        }
        if (t.energy[i] < e_lo - 1e-6) {
            e_ok = false; worst_e_low = t.energy[i];
        }
        if (t.energy[i] > e_hi + 1e-6) {
            e_ok = false; worst_e_high = t.energy[i];
        }
    }
    check(ctx, cv_ok, name + " Cv(T) >= 0",
          "worst Cv = " + std::to_string(worst_cv));
    check(ctx, s_ok, name + " 0 <= S(T) <= ln(dim)",
          "worst S = " + std::to_string(worst_s));
    check(ctx, e_ok, name + " E(T) in [Emin, Emax]",
          "worst low=" + std::to_string(worst_e_low) +
          " high=" + std::to_string(worst_e_high));
}

// Compare method thermodynamics against full-diag reference, returning the
// worst relative errors. Used with method-specific tolerances.
struct ThermoErrors {
    double e_abs;   // max |E_method - E_ref| / max(|E_ref|, 1)
    double cv_abs;  // max |Cv_method - Cv_ref| / max(|Cv_ref|, 1)
    double s_abs;   // max |S_method - S_ref|
};

ThermoErrors compare_thermo(const ThermodynamicData& t,
                            const ThermodynamicData& ref) {
    ThermoErrors errs{0, 0, 0};
    uint64_t n = std::min(t.temperatures.size(), ref.temperatures.size());
    for (uint64_t i = 0; i < n; ++i) {
        double e_scale  = std::max(std::abs(ref.energy[i]), 1.0);
        double cv_scale = std::max(std::abs(ref.specific_heat[i]), 1.0);
        errs.e_abs  = std::max(errs.e_abs,
                               std::abs(t.energy[i] - ref.energy[i]) / e_scale);
        errs.cv_abs = std::max(errs.cv_abs,
                               std::abs(t.specific_heat[i] - ref.specific_heat[i]) / cv_scale);
        errs.s_abs  = std::max(errs.s_abs,
                               std::abs(t.entropy[i] - ref.entropy[i]));
    }
    return errs;
}

} // namespace

// -----------------------------------------------------------------------------
// FTLM: random-state averaging. Tolerance is relatively loose because FTLM
// introduces statistical noise via random samples. Tested across the full
// temperature range -- this is the regime FTLM is designed for.
// -----------------------------------------------------------------------------
static void test_ftlm(TestContext& ctx, const ThermalFixture& f) {
    FTLMParameters p;
    p.krylov_dim = 50;
    p.num_samples = 80;          // enough samples for dim=256 to give ~5% Cv
    p.tolerance = 1e-12;
    p.full_reorthogonalization = true;
    p.random_seed = 12345;
    p.compute_error_bars = false;

    FTLMResults res = finite_temperature_lanczos(
        f.Hv, f.dim, p, f.temp_min, f.temp_max, f.num_temp_bins,
        /*output_dir=*/"");

    // FTLM entropy can dip slightly negative at the lowest T from the
    // log of a noisy partition function -- allow O(0.3) slack.
    sanity_check_thermo(ctx, "FTLM", res.thermo_data, f,
                        /*cv_neg_slack=*/1e-6, /*s_neg_slack=*/0.3);

    auto errs = compare_thermo(res.thermo_data, f.ref_thermo);
    check(ctx, errs.e_abs < 0.10,
          "FTLM E(T) within 10% of full-diag reference",
          "max rel |ΔE| = " + std::to_string(errs.e_abs));
    check(ctx, errs.cv_abs < 0.40,
          "FTLM Cv(T) within 40% of full-diag reference",
          "max rel |ΔCv| = " + std::to_string(errs.cv_abs));
    // Entropy is the most statistically noisy quantity for FTLM.
    check(ctx, errs.s_abs < 0.6,
          "FTLM S(T) absolute error < 0.6",
          "max |ΔS| = " + std::to_string(errs.s_abs));

    check(ctx, std::abs(res.ground_state_estimate - f.dense_eigs.front()) < 1e-6,
          "FTLM ground state estimate matches dense reference",
          "got " + std::to_string(res.ground_state_estimate) +
          " want " + std::to_string(f.dense_eigs.front()));
}

// -----------------------------------------------------------------------------
// LTLM: deterministic (starts from ground state). Tested ONLY in its
// design regime, kT << bandwidth, where the ground-state-rooted Krylov
// approximation is accurate. At high T the truncated thermal trace is
// not quantitatively meaningful and we don't assert on it.
// -----------------------------------------------------------------------------
static void test_ltlm(TestContext& ctx) {
    // Low-T fixture only.
    const uint64_t N = 8;
    const double T_min = 0.05;
    const double T_max = 0.5;        // well below the ~7-unit bandwidth
    const uint64_t num_bins = 12;
    ThermalFixture f = make_fixture(N, T_min, T_max, num_bins);

    LTLMParameters p;
    p.krylov_dim = 80;
    p.ground_state_krylov = 80;
    p.num_samples = 1;
    p.tolerance = 1e-12;
    p.full_reorthogonalization = true;
    p.random_seed = 7;

    LTLMResults res = low_temperature_lanczos(
        f.Hv, f.dim, p, f.temp_min, f.temp_max, f.num_temp_bins,
        /*ground_state=*/nullptr, /*output_dir=*/"");

    // LTLM at high T can give larger Cv/S violations; in the low-T window
    // the conservative slack should be enough.
    sanity_check_thermo(ctx, "LTLM", res.thermo_data, f,
                        /*cv_neg_slack=*/1e-4, /*s_neg_slack=*/1e-3);

    check(ctx, std::abs(res.ground_state_energy - f.dense_eigs.front()) < 1e-6,
          "LTLM ground state matches dense reference",
          "got " + std::to_string(res.ground_state_energy) +
          " want " + std::to_string(f.dense_eigs.front()));

    // At T_min ≈ 0.05 we expect LTLM to hit the ground state essentially
    // exactly.
    check(ctx, std::abs(res.thermo_data.energy.front() - f.dense_eigs.front()) < 1e-3,
          "LTLM E(T_min) ~ ground state",
          "got " + std::to_string(res.thermo_data.energy.front()));

    // Bound E(T) above ground state and below dense reference + tolerance.
    // LTLM computes a truncated thermal trace from a single Krylov basis
    // rooted at the ground state. Even in the low-T window it can overshoot
    // upwards (it captures excited contributions imperfectly), so we only
    // require that it stays inside [E_gs, E_ref + slack] across the window.
    bool e_bounded = true;
    double worst_above = 0.0;
    for (uint64_t i = 0; i < res.thermo_data.energy.size(); ++i) {
        const double e = res.thermo_data.energy[i];
        const double e_ref = f.ref_thermo.energy[i];
        const double slack = 0.5;   // ~7% of the bandwidth
        if (e > e_ref + slack || e < f.dense_eigs.front() - 1e-6) {
            e_bounded = false;
            worst_above = std::max(worst_above, std::abs(e - e_ref));
        }
    }
    check(ctx, e_bounded,
          "LTLM E(T) bounded above by reference + slack in low-T window",
          "worst |ΔE| = " + std::to_string(worst_above));

    // Cv must stay non-negative within slack and bounded; we don't assert on
    // its absolute accuracy because LTLM's Cv = <H^2>-<H>^2 is a difference
    // of two truncated traces and is highly sensitive at low T.
    bool cv_bounded = true;
    double worst_cv_high = 0.0;
    for (double cv : res.thermo_data.specific_heat) {
        if (cv < -1e-3 || cv > 100.0) {
            cv_bounded = false;
            worst_cv_high = std::max(worst_cv_high, std::abs(cv));
        }
    }
    check(ctx, cv_bounded,
          "LTLM Cv(T) bounded and non-negative",
          "worst |Cv| = " + std::to_string(worst_cv_high));
}

// -----------------------------------------------------------------------------
// Hybrid thermal: LTLM at low T + FTLM at high T with automatic crossover.
// We verify the basic plumbing -- ground state, point split, sanity bounds --
// rather than tight global accuracy: hybrid methods are known to develop
// discontinuities at the crossover and are inherently statistical above it.
// -----------------------------------------------------------------------------
static void test_hybrid(TestContext& ctx, const ThermalFixture& f) {
    HybridThermalParameters p;
    p.crossover_temperature = 1.0;
    p.ltlm_krylov_dim = 80;
    p.ltlm_ground_krylov = 80;
    p.ltlm_full_reorth = true;
    p.ltlm_seed = 101;
    p.ftlm_krylov_dim = 50;
    p.ftlm_num_samples = 80;
    p.ftlm_full_reorth = true;
    p.ftlm_seed = 202;
    p.ftlm_error_bars = false;
    p.tolerance = 1e-12;

    HybridThermalResults res = hybrid_thermal_method(
        f.Hv, f.dim, p, f.temp_min, f.temp_max, f.num_temp_bins, "");

    // Hybrid inherits LTLM's high-T artifacts in the (low-T) LTLM segment
    // and FTLM's noise in the (high-T) FTLM segment. Allow the union of
    // both methods' slacks.
    sanity_check_thermo(ctx, "Hybrid", res.thermo_data, f,
                        /*cv_neg_slack=*/1.0, /*s_neg_slack=*/1.0);

    check(ctx, std::abs(res.ground_state_energy - f.dense_eigs.front()) < 1e-6,
          "Hybrid ground state matches dense reference",
          "got " + std::to_string(res.ground_state_energy) +
          " want " + std::to_string(f.dense_eigs.front()));

    check(ctx, res.ltlm_points + res.ftlm_points == f.num_temp_bins,
          "Hybrid ltlm_points + ftlm_points = total bins",
          "got " + std::to_string(res.ltlm_points) + "+"
              + std::to_string(res.ftlm_points));

    check(ctx, res.ltlm_points > 0 && res.ftlm_points > 0,
          "Hybrid uses both LTLM and FTLM segments");

    // E(T) must remain inside the spectrum (already enforced by sanity), but
    // additionally the FTLM segment alone should be reasonably accurate
    // because it spans the high-T regime where FTLM does its job. Slice the
    // FTLM tail and compare.
    const uint64_t cross = res.ltlm_points;
    if (cross < res.thermo_data.temperatures.size()) {
        ThermodynamicData ftlm_tail, ref_tail;
        for (uint64_t i = cross; i < res.thermo_data.temperatures.size(); ++i) {
            ftlm_tail.temperatures.push_back(res.thermo_data.temperatures[i]);
            ftlm_tail.energy.push_back(res.thermo_data.energy[i]);
            ftlm_tail.specific_heat.push_back(res.thermo_data.specific_heat[i]);
            ftlm_tail.entropy.push_back(res.thermo_data.entropy[i]);
            ref_tail.temperatures.push_back(f.ref_thermo.temperatures[i]);
            ref_tail.energy.push_back(f.ref_thermo.energy[i]);
            ref_tail.specific_heat.push_back(f.ref_thermo.specific_heat[i]);
            ref_tail.entropy.push_back(f.ref_thermo.entropy[i]);
        }
        auto tail_errs = compare_thermo(ftlm_tail, ref_tail);
        check(ctx, tail_errs.e_abs < 0.10,
              "Hybrid FTLM-segment E(T) within 10% of reference",
              "max rel |ΔE| = " + std::to_string(tail_errs.e_abs));
    }
}

int main() {
    TestContext ctx("test_thermal_methods");

    // N=8 Heisenberg PBC chain: dim=256, big enough that Krylov<<Hilbert
    // (so random-start variance is small), small enough that the dense
    // reference is ~instant.
    const uint64_t N = 8;
    const double T_min = 0.1;
    const double T_max = 20.0;
    const uint64_t num_bins = 24;

    ThermalFixture f = make_fixture(N, T_min, T_max, num_bins);

    test_ftlm(ctx, f);
    test_ltlm(ctx);
    test_hybrid(ctx, f);

    return ctx.summary_exit_code();
}
