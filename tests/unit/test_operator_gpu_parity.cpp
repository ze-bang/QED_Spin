// =============================================================================
// tests/unit/test_operator_gpu_parity.cpp
//
// Orchestrator-level GPU parity for the operator-collapse Phase 2a lane:
// a plain host ``Operator`` / ``FixedSzOperator`` now advertises
// ``geometry().supports_device_matvec`` and its ``bind_cuda()`` builds a
// ``CudaMatVecBackend`` device mirror (the SOTA no-atomic gather kernel).
//
// These tests drive ``ed::workflows::solve`` twice on the SAME host
// operator -- once pinned to the CPU lane (``allow_gpu=false``) and once
// on the GPU lane (``allow_gpu=true``) -- and assert the ground-state
// eigenvalue agrees and that the GPU run actually dispatched to the
// CudaBackend lane (``res.backend.lane == "gpu"``). This is the
// end-to-end check that closes the gap left by ``test_cuda_matvec_backend``
// (which pins the gather kernel == CPU at the backend level): here the
// whole orchestrator -> select_backend -> bind_cuda -> device Lanczos loop
// is exercised.
//
// Like ``test_cpu_gpu_equivalence``, every TEST_CASE SKIPs at runtime when
// no CUDA device is visible, so the build-only CUDA CI lane stays green.
//
// Phase 2a of the "Lean operator architecture collapse" plan (Jun 2026).
// =============================================================================

#include "common/catch2_harness.h"
#include "common/test_harness.h"

#ifdef WITH_CUDA

#include <ed/core/linear_operator.h>
#include <ed/core/results.h>
#include <ed/core/select_backend.h>
#include <ed/orchestrator.h>

#include <cmath>

namespace {

ed::GroundStateResult solve_lane(const ed::LinearOperator& H, bool allow_gpu) {
    ed::SolveOptions opts;
    opts.num_eigs        = 1;
    opts.max_iter        = 200;
    opts.tolerance       = 1e-11;
    opts.compute_vectors = false;
    opts.method          = ed::SolveMethod::Lanczos;
    opts.backend.allow_gpu = allow_gpu;
    opts.backend.gpu_dim_floor = 0;  // explicit lane comparison at tiny dim
    opts.backend.allow_mpi = false;
    return ed::workflows::solve(H, opts);
}

}  // namespace

TEST_CASE("operator GPU parity: full Operator ground state matches CPU "
          "and dispatches to the GPU lane",
          "[operator_gpu_parity][solve]") {
    if (!ed::have_cuda()) {
        SKIP("No CUDA device available -- skipping operator GPU parity test.");
    }

    constexpr std::uint64_t N = 8;
    auto H = ed_tests::build_heisenberg_chain(N, /*J=*/1.0, /*periodic=*/true);

    const auto cpu = solve_lane(*H, /*allow_gpu=*/false);
    const auto gpu = solve_lane(*H, /*allow_gpu=*/true);

    REQUIRE_FALSE(cpu.eigenvalues.empty());
    REQUIRE_FALSE(gpu.eigenvalues.empty());
    REQUIRE(cpu.backend.lane == "cpu");
    REQUIRE(gpu.backend.lane == "gpu");

    INFO("E_cpu=" << cpu.eigenvalues[0] << "  E_gpu=" << gpu.eigenvalues[0]
         << "  |Δ|=" << std::abs(cpu.eigenvalues[0] - gpu.eigenvalues[0]));
    REQUIRE(std::abs(cpu.eigenvalues[0] - gpu.eigenvalues[0]) < 1e-8);
}

TEST_CASE("operator GPU parity: FixedSzOperator sector ground state matches "
          "CPU and dispatches to the GPU lane",
          "[operator_gpu_parity][solve][fixed_sz]") {
    if (!ed::have_cuda()) {
        SKIP("No CUDA device available -- skipping operator GPU parity test.");
    }

    constexpr std::uint64_t N    = 8;
    constexpr std::int64_t  n_up = 4;  // half-filled sector
    auto H = ed_tests::build_heisenberg_chain_fixed_sz(
        N, /*J=*/1.0, n_up, /*periodic=*/true);

    const auto cpu = solve_lane(*H, /*allow_gpu=*/false);
    const auto gpu = solve_lane(*H, /*allow_gpu=*/true);

    REQUIRE_FALSE(cpu.eigenvalues.empty());
    REQUIRE_FALSE(gpu.eigenvalues.empty());
    REQUIRE(cpu.backend.lane == "cpu");
    REQUIRE(gpu.backend.lane == "gpu");

    INFO("E_cpu=" << cpu.eigenvalues[0] << "  E_gpu=" << gpu.eigenvalues[0]
         << "  |Δ|=" << std::abs(cpu.eigenvalues[0] - gpu.eigenvalues[0]));
    REQUIRE(std::abs(cpu.eigenvalues[0] - gpu.eigenvalues[0]) < 1e-8);
}

#else  // !WITH_CUDA

TEST_CASE("operator GPU parity: WITH_CUDA=OFF -- no-op",
          "[operator_gpu_parity][skip]") {
    SUCCEED("Built without WITH_CUDA -- nothing to compare.");
}

#endif  // WITH_CUDA
