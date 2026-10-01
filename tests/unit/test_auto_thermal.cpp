// =============================================================================
// test_auto_thermal  (Catch2 v3)
//
// Smoke-tests the unified `ed::workflows::thermal(H, opts)` orchestrator:
// the mTPQ path (routed through `mtpq_kernel<Backend>` / `tpq_kernel`),
// a check that the FTLM lane is wired, and backend-constraint handling.
//
// `workflows::thermal` runs on the operator it is given; it does not
// decompose into Sz or irrep sectors. Sector-resolved thermodynamics are
// assembled by the caller (see ed/sectors/thermal.h).
// =============================================================================

#include "common/catch2_harness.h"

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
    // WITH_CUDA builds, but the GPU lane for these
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

TEST_CASE("workflows::thermal FTLM lane is wired",
          "[workflows][thermal][ftlm][ltlm]") {
    // The orchestrator routes FTLM to `ed::thermal::ftlm_kernel`. This
    // test verifies the orchestrator actually executes the kernel
    // without throwing.
    auto H = build_heisen();

    const std::vector<double> betas = { 0.1, 1.0, 5.0 };

    // CPU-lane smoke tests (see the mTPQ case above): these tiny,
    // high-iteration FTLM runs are dominated by GPU launch overhead;
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

    auto res = ed::workflows::thermal(*H, opts);
    REQUIRE(res.backend.lane == "cpu");
}
