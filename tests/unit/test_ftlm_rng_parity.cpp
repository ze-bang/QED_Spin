// =============================================================================
// tests/unit/test_ftlm_rng_parity.cpp
//
// WP10 C3: the Backend-templated FTLM body
// (``ed::thermal::detail::ftlm_kernel_via_backend``) draws its sample
// vectors with the CPU driver's recipe (``sample_engine`` +
// ``generateGaussianRandomVector``), so on ``CpuBackend`` it reproduces
// ``::finite_temperature_lanczos`` for the same seed / samples / Krylov
// dimension / temperature grid. Checked on
//   * an 8-site Heisenberg ring (dim 256), full reorthogonalisation off and on;
//   * a 14-site Heisenberg ring (dim 16384, above the 8192 thread-budget
//     threshold of ``auto_threads_for_dim``), full reorthogonalisation off
//     and on.
// Energy, specific heat, entropy and free energy must agree to 1e-12
// (relative); the test also reports whether the curves are bit-identical.
// =============================================================================

#include "common/catch2_harness.h"
#include "common/test_harness.h"

#include <ed/matvec/backends/cpu_backend.h>
#include <ed/matvec/matvec.h>
#include <ed/solvers/ftlm.h>
#include <ed/thermal/ftlm_kernel.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <string>
#include <vector>

using Complex = std::complex<double>;

namespace {

struct MatvecCallable {
    const ed::matvec::MatVecOperator* op;
    void operator()(const Complex* in, Complex* out, std::size_t n) const {
        op->apply(in, out, n);
    }
};

// Returns true iff the two curves are bitwise equal; REQUIREs 1e-12 relative.
bool compare_curve(const std::string& name,
                   const std::vector<double>& gen2,
                   const std::vector<double>& gen1) {
    INFO("curve " << name);
    REQUIRE(gen2.size() == gen1.size());
    bool identical = true;
    for (std::size_t t = 0; t < gen1.size(); ++t) {
        INFO("t index " << t << ": via_backend " << gen2[t]
             << " vs driver " << gen1[t]);
        REQUIRE(std::isfinite(gen2[t]));
        CHECK(std::abs(gen2[t] - gen1[t])
              <= 1e-12 * std::max(1.0, std::abs(gen1[t])));
        if (gen2[t] != gen1[t]) identical = false;
    }
    return identical;
}

void check_parity(std::uint64_t n_sites, std::size_t samples,
                  std::size_t krylov, bool full_reorth) {
    const std::size_t dim = std::size_t{1} << n_sites;
    auto H = ed_tests::build_heisenberg_chain(n_sites, 1.0, /*periodic=*/true);
    MatvecCallable apply{H.get()};

    const std::vector<double> betas = {0.05, 0.2, 0.5, 1.0, 2.0, 5.0, 20.0};
    std::vector<double> temperatures;
    for (double b : betas) temperatures.push_back(1.0 / b);

    ed::thermal::FtlmOptions opts;
    opts.num_samples              = samples;
    opts.krylov_dim               = krylov;
    opts.betas                    = betas;
    opts.random_seed              = 20260919;
    opts.full_reorthogonalization = full_reorth;

    FTLMParameters params;
    params.num_samples              = samples;
    params.krylov_dim               = krylov;
    params.random_seed              = opts.random_seed;
    params.full_reorthogonalization = full_reorth;

    const auto gen2 = ed::thermal::detail::ftlm_kernel_via_backend(
        ed::matvec::default_cpu_backend(), apply, dim,
        static_cast<std::uint64_t>(dim), opts);
    const FTLMResults gen1 = finite_temperature_lanczos(
        ed::matvec::as_apply_function(*H),
        static_cast<std::uint64_t>(dim), params, temperatures);

    INFO("n_sites " << n_sites << ", full_reorth " << full_reorth);
    REQUIRE(gen1.thermo_data.energy.size() == betas.size());

    bool identical = true;
    identical &= compare_curve("energy",        gen2.energy,        gen1.thermo_data.energy);
    identical &= compare_curve("specific_heat", gen2.heat_capacity, gen1.thermo_data.specific_heat);
    identical &= compare_curve("entropy",       gen2.entropy,       gen1.thermo_data.entropy);
    identical &= compare_curve("free_energy",   gen2.free_energy,   gen1.thermo_data.free_energy);
    CHECK(std::abs(gen2.ground_state_estimate - gen1.ground_state_estimate)
          <= 1e-12 * std::max(1.0, std::abs(gen1.ground_state_estimate)));
    identical &= (gen2.ground_state_estimate == gen1.ground_state_estimate);

    WARN("n_sites " << n_sites << ", full_reorth " << full_reorth
         << ": via_backend vs finite_temperature_lanczos "
         << (identical ? "bit-identical" : "equal to 1e-12, not bit-identical"));
}

}  // namespace

TEST_CASE("ftlm_kernel_via_backend reproduces finite_temperature_lanczos: "
          "8-site ring",
          "[ftlm][thermal][wp10]") {
    SECTION("local reorthogonalisation") { check_parity(8, 4, 40, false); }
    SECTION("full reorthogonalisation")  { check_parity(8, 4, 40, true); }
}

TEST_CASE("ftlm_kernel_via_backend reproduces finite_temperature_lanczos: "
          "14-site ring (dim > 8192)",
          "[ftlm][thermal][wp10]") {
    SECTION("local reorthogonalisation") { check_parity(14, 2, 25, false); }
    SECTION("full reorthogonalisation")  { check_parity(14, 2, 25, true); }
}
