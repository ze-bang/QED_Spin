// =============================================================================
// test_spectral_initial_state  (Catch2 v3)
//
// Pillar 3 of the "Save and DSSF Upgrades" plan (May 2026): pins the
// ``ed::workflows::spectral`` contract for user-supplied seed states on
// the GroundStateCF lane.
//
// Contract:
//   * When ``opts.initial_state`` is empty, the orchestrator runs an
//     inner Lanczos solve with ``compute_vectors=true``, then feeds the
//     actual ground-state eigenvector into ``cf_spectral_kernel``. The
//     legacy "random vector" shortcut is gone.
//   * When ``opts.initial_state`` is non-empty:
//       - it must have ``H.geometry().local_dim`` entries (otherwise
//         we throw ``std::invalid_argument``);
//       - it is renormalised inside the orchestrator before being
//         handed to ``cf_spectral_kernel``;
//       - passing the actual GS eigenvector as ``initial_state``
//         reproduces the auto-GS S(omega) curve to within numerical
//         noise (this is the "TPQ-to-CF" pipeline reduced to its
//         GS limit: same input vector -> same output).
//
// The TPQ-to-CF integration test is the companion example pair
// (`examples/spectral/ground_state_dssf/cpu_from_tpq_state.{cpp,py}`);
// here we pin the kernel-level contract.
// =============================================================================

#include "common/catch2_harness.h"

#include <ed/core/operator.h>
#include <ed/orchestrator.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <memory>
#include <random>
#include <stdexcept>
#include <vector>

using namespace ed_tests;

namespace {

constexpr std::uint64_t N_SITES = 4;
using Complex = std::complex<double>;

std::unique_ptr<Operator> heisen() {
    return build_heisenberg_chain(N_SITES, 1.0, /*periodic=*/true);
}

std::unique_ptr<Operator> sz_total() {
    auto obs = std::make_unique<Operator>(N_SITES, /*spin=*/0.5f);
    for (std::uint64_t i = 0; i < N_SITES; ++i) {
        obs->addOneBodyTerm(2, i, Complex(1.0, 0.0));
    }
    return obs;
}

ed::workflows::SpectralOptions base_opts() {
    ed::workflows::SpectralOptions opts;
    opts.method     = ed::workflows::SpectralOptions::Method::GroundStateCF;
    opts.krylov_dim = 60;
    opts.broadening = 0.1;
    opts.omega_min  = -5.0;
    opts.omega_max  =  5.0;
    opts.num_omega  = 11;
    return opts;
}

}  // namespace

TEST_CASE("ed::spectral GroundStateCF rejects initial_state of wrong size",
          "[orchestrator][spectral][initial-state]") {
    auto H   = heisen();
    auto obs = sz_total();
    std::vector<const ed::LinearOperator*> observables = { obs.get() };

    auto opts = base_opts();
    opts.initial_state.assign(7, Complex(1.0, 0.0));  // wrong size

    REQUIRE_THROWS_AS(ed::workflows::spectral(*H, observables, opts),
                      std::invalid_argument);
}

TEST_CASE("ed::spectral GroundStateCF auto-GS matches user-supplied GS seed",
          "[orchestrator][spectral][initial-state]") {
    auto H   = heisen();
    auto obs = sz_total();
    std::vector<const ed::LinearOperator*> observables = { obs.get() };

    // -- Reference: let the orchestrator extract the GS itself.
    auto opts_auto = base_opts();
    auto R_auto = ed::workflows::spectral(*H, observables, opts_auto);
    REQUIRE(R_auto.S_real.size() == opts_auto.num_omega);

    // -- Independently extract the GS via solve() and pass it as
    //    initial_state. The orchestrator should reproduce the same
    //    spectral curve.
    ed::workflows::SolveOptions sopts;
    sopts.num_eigs        = 1;
    sopts.compute_vectors = true;
    sopts.tolerance       = 1e-12;
    sopts.method          = ed::workflows::SolveMethod::Lanczos;
    auto gs = ed::workflows::solve(*H, sopts);
    REQUIRE(gs.eigenvalues.size() >= 1);
    REQUIRE(gs.eigenvectors.has_value());
    REQUIRE_FALSE(gs.eigenvectors->host.empty());
    REQUIRE(gs.eigenvectors->host[0].size()
            == H->geometry().local_dim);

    auto opts_seed = base_opts();
    opts_seed.initial_state = gs.eigenvectors->host[0];
    auto R_seed = ed::workflows::spectral(*H, observables, opts_seed);
    REQUIRE(R_seed.S_real.size() == opts_seed.num_omega);

    double max_abs = 0.0;
    double max_rel = 0.0;
    for (std::size_t i = 0; i < R_auto.S_real.size(); ++i) {
        const double a = std::abs(R_auto.S_real[i] - R_seed.S_real[i]);
        max_abs = std::max(max_abs, a);
        const double denom = std::max(1e-12, std::abs(R_auto.S_real[i]));
        max_rel = std::max(max_rel, a / denom);
    }
    INFO("max |S_auto - S_seed| = " << max_abs
            << ", max relative = " << max_rel);
    CHECK(max_abs < 1e-8);
}

TEST_CASE("ed::spectral GroundStateCF accepts a non-GS seed without throwing",
          "[orchestrator][spectral][initial-state]") {
    auto H   = heisen();
    auto obs = sz_total();
    std::vector<const ed::LinearOperator*> observables = { obs.get() };

    // Build a random complex unit vector of the right size: this mimics
    // a warm TPQ state, which is what Pillar 3 was designed to plumb
    // through.
    auto random_seed = [](std::uint64_t dim) {
        std::vector<Complex> v(dim);
        std::mt19937_64 rng(123);
        std::normal_distribution<double> nd(0.0, 1.0);
        double sumsq = 0.0;
        for (auto& z : v) {
            z = Complex(nd(rng), nd(rng));
            sumsq += std::norm(z);
        }
        const double inv = 1.0 / std::sqrt(sumsq);
        for (auto& z : v) z *= inv;
        return v;
    };
    auto opts = base_opts();
    opts.initial_state = random_seed(H->geometry().local_dim);

    auto R = ed::workflows::spectral(*H, observables, opts);
    REQUIRE(R.S_real.size() == opts.num_omega);
    // The CF spectrum of a random unit vector w.r.t. Sz is not
    // identically zero -- assert at least one frequency carries
    // non-trivial spectral weight to ensure the kernel ran.
    double max_abs = 0.0;
    for (double v : R.S_real) max_abs = std::max(max_abs, std::abs(v));
    CHECK(max_abs > 1e-6);
}
