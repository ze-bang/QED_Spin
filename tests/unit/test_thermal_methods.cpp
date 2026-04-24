// =============================================================================
// test_thermal_methods (Catch2 v3, P1.8 / audit Q12)
//
// Sanity & accuracy checks for the finite-temperature Lanczos family:
//   * FTLM     (finite-temperature Lanczos, random-state averaging)
//   * LTLM     (low-temperature Lanczos, ground-state + excitations)
//   * Hybrid   (LTLM for T<T_c and FTLM for T>=T_c, automatically merged)
//
// Ground truth is the full spectrum of a small Heisenberg chain, computed
// with Eigen's dense self-adjoint eigensolver and fed through
// `calculate_thermodynamics_from_spectrum()`.
// =============================================================================

#include "common/catch2_harness.h"

#include <ed/core/thermal_types.h>
#include <ed/solvers/ftlm.h>
#include <ed/solvers/hybrid_thermal.h>
#include <ed/solvers/ltlm.h>
#include <ed/solvers/observables.h>

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <memory>
#include <vector>

using namespace ed_tests;

namespace {

struct ThermalFixture {
    std::unique_ptr<Operator> op;
    uint64_t dim;
    std::vector<double> dense_eigs;
    std::function<void(const Complex*, Complex*, int)> Hv;
    ThermodynamicData ref_thermo;
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

void sanity_check_thermo(const std::string& name,
                         const ThermodynamicData& t, const ThermalFixture& f,
                         double cv_neg_slack = 1e-6,
                         double s_neg_slack  = 1e-6) {
    INFO("method=" << name);
    REQUIRE(t.temperatures.size() == f.num_temp_bins);
    REQUIRE(t.energy.size() == f.num_temp_bins);
    REQUIRE(t.specific_heat.size() == f.num_temp_bins);
    REQUIRE(t.entropy.size() == f.num_temp_bins);

    const double e_lo = f.dense_eigs.front();
    const double e_hi = f.dense_eigs.back();
    const double s_max = std::log(static_cast<double>(f.dim)) + 1e-6;

    for (uint64_t i = 0; i < t.temperatures.size(); ++i) {
        INFO("i=" << i << " T=" << t.temperatures[i]
             << " Cv=" << t.specific_heat[i]
             << " S=" << t.entropy[i]
             << " E=" << t.energy[i]);
        REQUIRE(t.specific_heat[i] >= -cv_neg_slack);
        REQUIRE(t.entropy[i] >= -s_neg_slack);
        REQUIRE(t.entropy[i] <= s_max);
        REQUIRE(t.energy[i] >= e_lo - 1e-6);
        REQUIRE(t.energy[i] <= e_hi + 1e-6);
    }
}

struct ThermoErrors {
    double e_abs;
    double cv_abs;
    double s_abs;
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

ThermalFixture make_full_fixture() {
    return make_fixture(/*N=*/8, /*T_min=*/0.1, /*T_max=*/20.0,
                        /*num_bins=*/24);
}

} // namespace

TEST_CASE("FTLM matches dense reference within statistical tolerances",
          "[thermal][ftlm]") {
    auto f = make_full_fixture();
    FTLMParameters p;
    p.krylov_dim = 50;
    p.num_samples = 80;
    p.tolerance = 1e-12;
    p.full_reorthogonalization = true;
    p.random_seed = 12345;
    p.compute_error_bars = false;

    FTLMResults res = finite_temperature_lanczos(
        f.Hv, f.dim, p, f.temp_min, f.temp_max, f.num_temp_bins,
        /*output_dir=*/"");

    sanity_check_thermo("FTLM", res.thermo_data, f,
                        /*cv_neg_slack=*/1e-6, /*s_neg_slack=*/0.3);

    auto errs = compare_thermo(res.thermo_data, f.ref_thermo);
    INFO("FTLM max rel |ΔE|=" << errs.e_abs
         << " max rel |ΔCv|=" << errs.cv_abs
         << " max |ΔS|=" << errs.s_abs);
    REQUIRE(errs.e_abs < 0.10);
    REQUIRE(errs.cv_abs < 0.40);
    REQUIRE(errs.s_abs < 0.6);

    INFO("FTLM ground state estimate=" << res.ground_state_estimate
         << " want=" << f.dense_eigs.front());
    REQUIRE(std::abs(res.ground_state_estimate - f.dense_eigs.front()) < 1e-6);
}

TEST_CASE("LTLM matches dense reference in low-T window",
          "[thermal][ltlm]") {
    const uint64_t N = 8;
    const double T_min = 0.05;
    const double T_max = 0.5;
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

    sanity_check_thermo("LTLM", res.thermo_data, f,
                        /*cv_neg_slack=*/1e-4, /*s_neg_slack=*/1e-3);

    INFO("LTLM gs=" << res.ground_state_energy
         << " want=" << f.dense_eigs.front());
    REQUIRE(std::abs(res.ground_state_energy - f.dense_eigs.front()) < 1e-6);

    INFO("LTLM E(T_min)=" << res.thermo_data.energy.front());
    REQUIRE(std::abs(res.thermo_data.energy.front() - f.dense_eigs.front()) < 1e-3);

    for (uint64_t i = 0; i < res.thermo_data.energy.size(); ++i) {
        const double e = res.thermo_data.energy[i];
        const double e_ref = f.ref_thermo.energy[i];
        const double slack = 0.5;
        INFO("i=" << i << " E=" << e << " E_ref=" << e_ref);
        REQUIRE(e <= e_ref + slack);
        REQUIRE(e >= f.dense_eigs.front() - 1e-6);
    }

    for (double cv : res.thermo_data.specific_heat) {
        INFO("Cv=" << cv);
        REQUIRE(cv >= -1e-3);
        REQUIRE(cv <= 100.0);
    }
}

TEST_CASE("Hybrid thermal method dispatches LTLM/FTLM correctly",
          "[thermal][hybrid]") {
    auto f = make_full_fixture();
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

    sanity_check_thermo("Hybrid", res.thermo_data, f,
                        /*cv_neg_slack=*/1.0, /*s_neg_slack=*/1.0);

    INFO("Hybrid gs=" << res.ground_state_energy
         << " want=" << f.dense_eigs.front());
    REQUIRE(std::abs(res.ground_state_energy - f.dense_eigs.front()) < 1e-6);

    INFO("ltlm_points=" << res.ltlm_points
         << " ftlm_points=" << res.ftlm_points
         << " total=" << f.num_temp_bins);
    REQUIRE(res.ltlm_points + res.ftlm_points == f.num_temp_bins);
    REQUIRE(res.ltlm_points > 0);
    REQUIRE(res.ftlm_points > 0);

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
        INFO("Hybrid FTLM-tail max rel |ΔE| = " << tail_errs.e_abs);
        REQUIRE(tail_errs.e_abs < 0.10);
    }
}
