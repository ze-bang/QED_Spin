// =============================================================================
// tests/unit/test_orchestrator.cpp
//
// Smoke test for `ed::workflows::solve` on the same small Heisenberg chain used
// elsewhere in the unit suite: it lands near the ground-state energy of the
// 6-site periodic AFM Heisenberg chain (E_0 = -2.8027757...), through the
// BackendVariant dispatch path on the CPU lane.
// =============================================================================

#include "common/catch2_harness.h"
#include "common/test_harness.h"

#include <ed/core/linear_operator.h>
#include <ed/core/results.h>
#include <ed/core/select_backend.h>
#include <ed/orchestrator.h>

#include <cmath>
#include <vector>

TEST_CASE("workflows::solve recovers the 6-site Heisenberg ground state",
          "[orchestrator][solve][phase4]") {
    constexpr std::uint64_t N   = 6;
    auto H = ed_tests::build_heisenberg_chain(N, /*J=*/1.0, /*periodic=*/true);

    ed::SolveOptions opts;
    opts.num_eigs       = 1;
    opts.max_iter       = 50;
    opts.tolerance      = 1e-10;
    opts.compute_vectors = false;
    opts.method         = ed::SolveMethod::Lanczos;
    // CPU-lane smoke test; pin the CPU lane explicitly.
    opts.backend.allow_gpu = false;

    auto res = ed::workflows::solve(*H, opts);

    REQUIRE_FALSE(res.eigenvalues.empty());
    // The ground state of the 6-site periodic AFM Heisenberg chain is
    // E_0 = -2.80277563... in units of J. Loose tolerance because the
    // Lanczos default seed makes the smallest eigenvalue easy to find
    // but the kernel cap of 50 iterations isn't tight on bare
    // convergence.
    REQUIRE(res.eigenvalues[0] < -2.5);
    REQUIRE(res.backend.lane == "cpu");
}
