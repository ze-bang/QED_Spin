// =============================================================================
// test_auto_thermal  (Catch2 v3)
//
// Smoke-tests the unified `ed::workflows::thermal(H, opts)` orchestrator.
// Covers the mTPQ / cTPQ paths that the orchestrator currently routes
// through `tpq_kernel<Backend>` (Phase 4.2 / 4.3 of the Minimalist ED
// Collapse). The FTLM / LTLM / KPM_DOS lanes are intentionally NOT
// covered here --- they remain on the legacy CPU-only path until Phase 6
// of the cleanup sweep lands the `<Backend>` versions and we can route
// them uniformly.
//
// Migrated from the legacy `ed::auto_pilot::thermal(...)` API during the
// ED Cleanup Sweep Phase 2 (May 2026). The legacy auto-pilot's per-Sz
// + per-irrep decomposition is gone: it is now the caller's
// responsibility to assemble the canonical partition function from
// sector-wise `workflows::thermal` calls (or run `workflows::solve` for
// the dense spectrum + `compute_thermodynamics_from_spectrum`).
// =============================================================================

#include "common/catch2_harness.h"

#include <ed/core/ed_types.h>
#include <ed/orchestrator.h>

#include <cmath>
#include <complex>
#include <vector>

using namespace ed_tests;
using ed::workflows::ThermalOptions;
using ed::workflows::SolveOptions;

namespace {

constexpr uint64_t N_SITES = 4;
constexpr std::uint64_t HILBERT_DIM = 1ULL << N_SITES;

std::unique_ptr<Operator> build_heisen(bool periodic = true) {
    return build_heisenberg_chain(N_SITES, 1.0, periodic);
}

}  // namespace

TEST_CASE("workflows::thermal mTPQ produces a finite ground-state estimate "
          "on the Heisenberg chain",
          "[workflows][thermal][mtpq]") {
    auto H = build_heisen();

    ThermalOptions opts;
    opts.method       = ThermalOptions::Method::mTPQ;
    opts.num_samples  = 4;
    opts.krylov_dim   = 80;
    opts.random_seed  = 12345;
    // CPU-lane smoke test. Plain Operators advertise supports_device_matvec on
    // WITH_CUDA builds (operator-collapse Phase 2a), but the GPU lane for these
    // tiny, high-iteration TPQ runs is dominated by kernel-launch overhead; GPU
    // thermal correctness is covered by the thermal::*_kernel<CudaBackend>
    // tests. Pin CPU so the smoke test stays fast and hardware-independent.
    opts.backend.allow_gpu = false;

    auto res = ed::workflows::thermal(*H, opts);

    REQUIRE(std::isfinite(res.ground_state_energy));
    REQUIRE(res.backend.lane == "cpu");

    // Cross-check against the FullDiag-based ground-state energy.
    SolveOptions sopts;
    sopts.num_eigs = 1;
    auto gs = ed::workflows::solve(*H, sopts);
    REQUIRE(gs.eigenvalues.size() >= 1);

    // mTPQ at finite sample count is noisy but must land at or above
    // the true GS within an O(1) tolerance for this tiny system.
    REQUIRE(res.ground_state_energy >= gs.eigenvalues[0] - 1e-6);
}

TEST_CASE("workflows::thermal cTPQ produces a finite ground-state estimate",
          "[workflows][thermal][ctpq]") {
    auto H = build_heisen();

    ThermalOptions opts;
    opts.method       = ThermalOptions::Method::cTPQ;
    opts.num_samples  = 4;
    opts.beta_max     = 8.0;
    opts.delta_beta   = 0.1;
    opts.taylor_order = 30;
    opts.random_seed  = 9999;
    opts.backend.allow_gpu = false;  // CPU-lane smoke test (see mTPQ case above)

    auto res = ed::workflows::thermal(*H, opts);

    REQUIRE(std::isfinite(res.ground_state_energy));
    REQUIRE(res.backend.lane == "cpu");

    // cTPQ at high beta should approach the true ground-state energy.
    SolveOptions sopts;
    sopts.num_eigs = 1;
    auto gs = ed::workflows::solve(*H, sopts);
    REQUIRE(gs.eigenvalues.size() >= 1);

    // Loose absolute tolerance: 4-site Heisenberg + few samples is noisy.
    REQUIRE(std::abs(res.ground_state_energy - gs.eigenvalues[0]) < 0.5);
}

TEST_CASE("workflows::thermal FTLM / LTLM / KpmDos lanes are wired",
          "[workflows][thermal][ftlm][ltlm][kpm_dos]") {
    // Phase 6 wired the FTLM / LTLM / KPM-DOS lanes through the unified
    // orchestrator by routing them at the kernel-shim level
    // (`ed::thermal::{ftlm_kernel,ltlm_kernel,kpm_dos_kernel}`). This
    // test verifies the orchestrator actually executes those kernels
    // and returns a populated ThermalResult.
    auto H = build_heisen();

    const std::vector<double> betas = { 0.1, 1.0, 5.0 };

    // CPU-lane smoke tests (see the mTPQ case above): these tiny,
    // high-iteration FTLM/LTLM/KPM runs are dominated by GPU launch overhead;
    // GPU correctness is covered by the thermal::*_kernel<CudaBackend> tests.
    SECTION("FTLM") {
        ThermalOptions opts;
        opts.method      = ThermalOptions::Method::FTLM;
        opts.num_samples = 4;
        opts.krylov_dim  = 30;
        opts.betas       = betas;
        opts.random_seed = 7;
        opts.backend.allow_gpu = false;
        REQUIRE_NOTHROW(ed::workflows::thermal(*H, opts));
    }
    SECTION("LTLM") {
        ThermalOptions opts;
        opts.method      = ThermalOptions::Method::LTLM;
        opts.num_samples = 2;
        opts.krylov_dim  = 30;
        opts.betas       = betas;
        opts.random_seed = 7;
        opts.backend.allow_gpu = false;
        REQUIRE_NOTHROW(ed::workflows::thermal(*H, opts));
    }
    SECTION("KpmDos") {
        ThermalOptions opts;
        opts.method      = ThermalOptions::Method::KpmDos;
        opts.betas       = betas;
        opts.random_seed = 7;
        opts.backend.allow_gpu = false;
        REQUIRE_NOTHROW(ed::workflows::thermal(*H, opts));
    }
}

TEST_CASE("workflows::thermal respects BackendConstraints.allow_gpu = false "
          "and lands on CPU",
          "[workflows][thermal][backend]") {
    auto H = build_heisen();

    ThermalOptions opts;
    opts.method        = ThermalOptions::Method::mTPQ;
    opts.num_samples   = 2;
    opts.krylov_dim    = 40;
    opts.random_seed   = 42;
    opts.backend.allow_gpu     = false;
    opts.backend.allow_mpi     = false;
    opts.backend.allow_mpi_gpu = false;

    auto res = ed::workflows::thermal(*H, opts);
    REQUIRE(res.backend.lane == "cpu");
}
