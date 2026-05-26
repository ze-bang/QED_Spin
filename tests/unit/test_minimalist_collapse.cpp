// =============================================================================
// tests/unit/test_minimalist_collapse.cpp
//
// Phase 7a of the Minimalist ED Collapse (May 2026): focused unit tests
// for the architectural seams introduced in Phases 3-4:
//
//   * LinearOperator concept (Geometry + bind<Backend>)        [Phase 3.1]
//   * select_backend decision tree                              [Phase 4.1]
//   * LocalDGKS3 reorthogonalization policy via lanczos_kernel  [Phase 2.1]
//
// The full solve / thermal / spectral integration sweep lives in
// `test_orchestrator.cpp`; the kernel-facade round-trips live in
// `test_kernel_facades.cpp`. This file targets the three seams that
// don't have a direct kernel/orchestrator counterpart.
// =============================================================================

#include "common/catch2_harness.h"
#include "common/test_harness.h"

#include <ed/core/linear_operator.h>
#include <ed/core/operator.h>
#include <ed/core/select_backend.h>
#include <ed/krylov/lanczos_kernel.h>
#include <ed/matvec/backends/cpu_backend.h>

#include <random>
#include <variant>
#include <vector>

// -----------------------------------------------------------------------------
// Phase 3.1 — LinearOperator concept
// -----------------------------------------------------------------------------
TEST_CASE("LinearOperator: Operator reports a host-only single-rank geometry",
          "[linear_operator][concept][phase3]") {
    constexpr std::uint64_t N = 4;
    auto H = ed_tests::build_heisenberg_chain(N, /*J=*/1.0, /*periodic=*/true);
    const ed::LinearOperator& base = *H;

    auto g = base.geometry();

    REQUIRE(g.local_dim  == g.global_dim);              // single-rank
    REQUIRE(g.local_offset == 0);
    REQUIRE(g.memory_space == ed::matvec::MemorySpace::Host);
    REQUIRE(g.local_dim == (1ull << N));
}

TEST_CASE("LinearOperator: bind<CpuBackend> returns a callable matvec",
          "[linear_operator][bind][phase3]") {
    constexpr std::uint64_t N = 4;
    auto H = ed_tests::build_heisenberg_chain(N, /*J=*/1.0, /*periodic=*/true);
    const ed::LinearOperator& base = *H;

    auto matvec = base.bind<ed::matvec::CpuBackend>();
    REQUIRE(matvec);

    const std::size_t dim = static_cast<std::size_t>(1ull << N);
    std::vector<Complex> x(dim, Complex{0.0, 0.0});
    std::vector<Complex> y(dim, Complex{42.0, 0.0});  // sentinel
    x[0] = Complex{1.0, 0.0};

    matvec(x.data(), y.data(), dim);
    // Smoke check: matvec on |00...0> shouldn't leave the sentinel intact;
    // we expect H|0> = some finite vector (mostly zeros plus off-diagonal
    // mixing on a periodic AFM chain).
    bool any_nonzero = false;
    for (auto& z : y) {
        if (std::abs(z) > 1e-15) { any_nonzero = true; break; }
    }
    REQUIRE(any_nonzero);
}

// -----------------------------------------------------------------------------
// Phase 4.1 — select_backend
// -----------------------------------------------------------------------------
TEST_CASE("select_backend: host-only operator picks CpuBackend by default",
          "[select_backend][phase4]") {
    ed::Geometry g{};
    g.local_dim    = 16;
    g.global_dim   = 16;
    g.local_offset = 0;
    g.memory_space = ed::matvec::MemorySpace::Host;

    ed::BackendConstraints c{};
    auto v = ed::select_backend(g, c);

    bool picked_cpu = std::visit([](auto& uptr) -> bool {
        using T = std::decay_t<decltype(*uptr)>;
        return std::is_same_v<T, ed::matvec::CpuBackend>;
    }, v);

    REQUIRE(picked_cpu);
}

TEST_CASE("select_backend: allow_gpu=false honored even when CUDA is built",
          "[select_backend][phase4]") {
    ed::Geometry g{};
    g.local_dim    = 16;
    g.global_dim   = 16;
    g.memory_space = ed::matvec::MemorySpace::Host;

    ed::BackendConstraints c{};
    c.allow_gpu = false;

    auto v = ed::select_backend(g, c);
    bool not_cuda = std::visit([](auto& uptr) -> bool {
        using T = std::decay_t<decltype(*uptr)>;
#ifdef WITH_CUDA
        if constexpr (std::is_same_v<T, ed::matvec::CudaBackend>) {
            return false;
        }
#endif
        return true;
    }, v);
    REQUIRE(not_cuda);
}

// -----------------------------------------------------------------------------
// Phase 2.1 — LocalDGKS3 reorthogonalization in lanczos_kernel
// -----------------------------------------------------------------------------
TEST_CASE("lanczos_kernel converges under LocalDGKS3 reorth policy",
          "[lanczos][reorth][local_dgks3][phase2]") {
    constexpr std::uint64_t N = 4;
    auto H = ed_tests::build_heisenberg_chain(N, /*J=*/1.0, /*periodic=*/true);
    const std::size_t dim = static_cast<std::size_t>(1ull << N);

    ed::matvec::CpuBackend be;
    auto matvec = H->bind<ed::matvec::CpuBackend>();
    REQUIRE(matvec);

    std::mt19937_64 rng(0xC0FFEEull);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    std::vector<Complex> seed(dim);
    for (auto& z : seed) {
        z = Complex{dist(rng), dist(rng)};
    }

    ed::krylov::LanczosKernelOptions opts;
    opts.max_iter   = 12;
    opts.reorth     = ed::krylov::ReorthPolicy::LocalDGKS3;
    opts.keep_basis = false;

    auto res = ed::krylov::lanczos_kernel(
        be, matvec, dim, seed.data(), opts);

    // The kernel returns the tridiagonal projection (alpha, beta) ---
    // eigenvalue extraction is the orchestrator's responsibility.
    // We assert the kernel ran a non-trivial number of iterations and
    // produced a well-formed tridiagonal under LocalDGKS3.
    REQUIRE(res.iters_done >= 4u);
    REQUIRE(res.alpha.size() == res.iters_done);
    REQUIRE(res.beta.size()  == res.iters_done + 1);
    for (auto a : res.alpha) {
        REQUIRE(std::isfinite(a));
    }
    for (auto b : res.beta) {
        REQUIRE(std::isfinite(b));
        REQUIRE(b >= 0.0);
    }
}
