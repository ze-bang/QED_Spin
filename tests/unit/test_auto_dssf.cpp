// =============================================================================
// test_auto_dssf  (Catch2 v3)
//
// Smoke-tests the unified `ed::workflows::spectral(H, observables, opts)`
// orchestrator (continued-fraction Lanczos for ground-state DSSF). The
// FtlmDynamical lane is not yet wired and is gated by a documented
// `runtime_error`; we lock in that contract here.
//
// Migrated from the legacy `ed::auto_pilot::dssf::compute(...)` API
// during the ED Cleanup Sweep Phase 2 (May 2026). The legacy DSSF
// auto-pilot was a router on top of `ed::dssf::run(...)` that picked
// SINGLE_EXPECTATION / GROUND_STATE_DSSF / STATIC_THERMAL /
// DYNAMICAL_THERMAL from the (has_temperature, has_frequency) tuple;
// that decision is now the caller's responsibility (and the static /
// static-thermal modes live outside `workflows::spectral`, which is
// dedicated to S(omega)-resolved spectra).
// =============================================================================

#include "common/catch2_harness.h"

#include <ed/core/operator.h>
#include <ed/operators/spin_ops.h>
#include <ed/orchestrator.h>

#include <cmath>
#include <complex>
#include <memory>
#include <vector>

using namespace ed_tests;
using ed::workflows::SpectralOptions;

namespace {

std::unique_ptr<Operator> build_heisen(uint64_t N, bool periodic = true) {
    return build_heisenberg_chain(N, 1.0, periodic);
}

// Single-site Sz observable. The CF-spectral kernel needs a
// LinearOperator that implements apply(); the simplest such observable
// is a single Sz term at site 0.
std::unique_ptr<Operator> build_single_site_sz(uint64_t N) {
    auto op = std::make_unique<Operator>(N, 0.5f);
    Operator::TransformData t{};
    t.is_two_body  = false;
    t.op_type      = 2;            // Sz
    t.site_index   = 0;
    t.coefficient  = Complex(1.0, 0.0);
    op->transform_data_.push_back(t);
    return op;
}

// Plain Operators advertise supports_device_matvec on WITH_CUDA builds
// (operator-collapse Phase 2a), so the CF-spectral Lanczos lane auto-dispatches
// to the GPU when a device is visible.
std::string expected_iterative_lane() {
    return ed::have_cuda() ? "gpu" : "cpu";
}

}  // namespace

TEST_CASE("workflows::spectral GroundStateCF produces a finite S(omega) grid",
          "[workflows][spectral][cf]") {
    const uint64_t N = 6;          // small but exercises the CF kernel
    auto H   = build_heisen(N);
    auto Sz0 = build_single_site_sz(N);

    SpectralOptions opts;
    opts.method      = SpectralOptions::Method::GroundStateCF;
    opts.krylov_dim  = 120;
    opts.broadening  = 0.05;
    opts.omega_min   = -4.0;
    opts.omega_max   =  4.0;
    opts.num_omega   = 64;

    std::vector<const ed::LinearOperator*> obs{ Sz0.get() };
    auto res = ed::workflows::spectral(*H, obs, opts);

    REQUIRE(res.omega.size() == opts.num_omega);
    REQUIRE(res.S_real.size() == opts.num_omega);
    REQUIRE(res.backend.lane == expected_iterative_lane());

    // Every value must be finite; the spectral function may be zero at
    // some grid points (e.g. inside the gap) but never NaN/inf.
    for (double v : res.S_real) {
        REQUIRE(std::isfinite(v));
    }
}

TEST_CASE("workflows::spectral throws when no observable is supplied",
          "[workflows][spectral][guard]") {
    auto H = build_heisen(4);
    SpectralOptions opts;
    std::vector<const ed::LinearOperator*> obs;
    REQUIRE_THROWS_AS(ed::workflows::spectral(*H, obs, opts),
                       std::invalid_argument);
}

TEST_CASE("workflows::spectral FtlmDynamical lane runs the FTLM "
          "continued-fraction body and returns a populated spectrum",
          "[workflows][spectral][ftlm_dynamical]") {
    // Wave A4 (Full unified-interface collapse, May 2026): the
    // FtlmDynamical lane is now wired through to the legacy
    // `compute_dynamical_correlation` body inside the orchestrator.
    // This test verifies the lane returns a sensibly-sized result on
    // a small Heisenberg chain.
    auto H   = build_heisen(4);
    auto Sz0 = build_single_site_sz(4);

    SpectralOptions opts;
    opts.method      = SpectralOptions::Method::FtlmDynamical;
    opts.krylov_dim  = 32;
    opts.num_omega   = 16;
    opts.num_samples = 4;
    opts.omega_min   = -3.0;
    opts.omega_max   = +3.0;
    opts.broadening  = 0.1;

    std::vector<const ed::LinearOperator*> obs{ Sz0.get() };
    ed::SpectralResult R;
    REQUIRE_NOTHROW(R = ed::workflows::spectral(*H, obs, opts));
    REQUIRE(R.omega.size() == opts.num_omega);
    REQUIRE(R.S_real.size() == R.omega.size());
    REQUIRE(R.S_imag.size() == R.omega.size());
}
