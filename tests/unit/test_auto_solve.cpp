// =============================================================================
// test_auto_solve  (Catch2 v3)
//
// Smoke-tests the unified `ed::workflows::solve(H, opts)` orchestrator.
// Tests exercise the public Backend-agnostic surface: numerical
// correctness of eigenvalues over CPU lane, full-diag fallback for
// small problems, Lanczos / KrylovSchur / BlockLanczos kernels, and the
// sector-projection workflow when the caller passes a FixedSzOperator.
//
// Migrated from the legacy `ed::auto_pilot::solve(...)` API during the
// ED Cleanup Sweep Phase 2 (May 2026). The legacy auto-pilot's
// `auto_basis` / `sz` / `Device` heuristics are gone --- those decisions
// are now the caller's responsibility (build a FixedSzOperator yourself
// for sector projection; set `BackendConstraints{.allow_gpu = true}` to
// opt into the GPU lane).
// =============================================================================

#include "common/catch2_harness.h"

#include <ed/core/fixed_sz_operator.h>
#include <ed/orchestrator.h>
#include <ed/operators/spin_ops.h>

#include <memory>
#include <string>
#include <vector>

using namespace ed_tests;
using ed::workflows::SolveMethod;
using ed::workflows::SolveOptions;

namespace {

// Small Heisenberg chain (Sz-conserving; ground-state energy known from
// Bethe ansatz and from test_full_diagonalization for small N).
std::unique_ptr<Operator> build_heisen(uint64_t N, bool periodic = false) {
    return build_heisenberg_chain(N, 1.0, periodic);
}

} // namespace

TEST_CASE("workflows::solve picks FullDiag on small full Hilbert space "
          "(no sector projection)",
          "[workflows][full_diag]") {
    auto H = build_heisen(4);                 // dim = 16
    auto ref = reference_from_operator(*H, 16);

    SolveOptions opts;
    opts.num_eigs        = 4;
    opts.method          = SolveMethod::Auto;  // -> FullDiag for dim <= 2^12

    auto res = ed::workflows::solve(*H, opts);

    REQUIRE(res.eigenvalues.size() >= 4);
    require_eigs_close(res.eigenvalues, ref.eigs, 4, 1e-8,
                       "workflows::solve N=4 ground manifold (auto -> FullDiag)");
}

TEST_CASE("workflows::solve over a FixedSzOperator at n_up = N/2 lands "
          "on the global ground state for the Heisenberg chain",
          "[workflows][fixed_sz]") {
    // The legacy `auto_pilot::solve(..., auto_basis = On)` heuristic
    // auto-built a FixedSzOperator and called the underlying solver on
    // it. Under the new surface the caller does the projection
    // explicitly; correctness must match.
    const uint64_t N = 4;
    auto H_fixed = build_heisenberg_chain_fixed_sz(N, 1.0, /*n_up=*/N / 2);

    SolveOptions opts;
    opts.num_eigs = 1;

    auto res = ed::workflows::solve(*H_fixed, opts);
    REQUIRE(res.eigenvalues.size() >= 1);

    // The Heisenberg-chain GS lives in Sz=0 (n_up=N/2). Compare against
    // the unprojected full-Hilbert reference.
    auto H_full = build_heisen(N);
    auto ref    = reference_from_operator(*H_full, 1 << N);
    REQUIRE(std::abs(res.eigenvalues[0] - ref.eigs[0]) < 1e-8);
}

TEST_CASE("workflows::solve over the full Hilbert space (no Sz pinning) "
          "recovers the full spectrum",
          "[workflows][full_diag]") {
    // Adding a uniform Zeeman field breaks the trivial GS-sector
    // mapping. The legacy auto-pilot would refuse to auto-project in
    // that case; under the new surface the caller just hands the full
    // Operator over and lets `workflows::solve` run FullDiag.
    auto H = build_heisen(4);
    Operator::TransformData zeeman{};
    zeeman.is_two_body = false;
    zeeman.op_type     = 2;            // Sz
    zeeman.coefficient = Complex(0.3, 0.0);
    for (uint64_t i = 0; i < 4; ++i) {
        zeeman.site_index = i;
        H->transform_data_.push_back(zeeman);
    }
    H->invalidateMatrixCaches();

    SolveOptions opts;
    opts.num_eigs = 4;

    auto res = ed::workflows::solve(*H, opts);
    auto ref = reference_from_operator(*H, 16);

    require_eigs_close(res.eigenvalues, ref.eigs, 4, 1e-8,
                       "workflows::solve full Hilbert with Zeeman field");
}

TEST_CASE("workflows::solve at a user-pinned Sz sector matches the "
          "global ground-state energy for the Heisenberg chain",
          "[workflows][fixed_sz]") {
    const uint64_t N = 6;                          // full dim = 64
    auto H_fixed = build_heisenberg_chain_fixed_sz(N, 1.0, /*n_up=*/N / 2);

    SolveOptions opts;
    opts.num_eigs = 1;

    auto res = ed::workflows::solve(*H_fixed, opts);

    // Compare to the unprojected full-Hilbert run.
    auto H_full = build_heisen(N);
    auto res_full = ed::workflows::solve(*H_full, opts);

    REQUIRE(res.eigenvalues.size() >= 1);
    REQUIRE(std::abs(res.eigenvalues[0] - res_full.eigenvalues[0]) < 1e-8);
}

TEST_CASE("workflows::solve respects an explicit Lanczos method",
          "[workflows][lanczos]") {
    // 10-site PBC AFM Heisenberg ring. We set `method = Lanczos`
    // explicitly because the auto-heuristic picks FullDiag for dim
    // <= 2^12.
    auto H = build_heisen(10, /*periodic=*/true);

    SolveOptions opts;
    opts.num_eigs = 1;
    opts.method   = SolveMethod::Lanczos;
    opts.tolerance = 1e-10;

    auto res = ed::workflows::solve(*H, opts);

    REQUIRE(res.eigenvalues.size() >= 1);
    // Known GS energy of the 10-site PBC AFM Heisenberg ring (Bethe).
    const double gs_expected = -4.515446354492155;
    REQUIRE(std::abs(res.eigenvalues[0] - gs_expected) < 1e-8);
}

TEST_CASE("workflows::solve runs on CPU when the caller forbids the GPU "
          "lane (BackendConstraints.allow_gpu = false)",
          "[workflows][backend]") {
    auto H = build_heisen(4);

    SolveOptions opts;
    opts.num_eigs = 1;
    opts.backend.allow_gpu     = false;
    opts.backend.allow_mpi     = false;
    opts.backend.allow_mpi_gpu = false;

    auto res = ed::workflows::solve(*H, opts);
    REQUIRE(res.eigenvalues.size() >= 1);
    REQUIRE(res.backend.lane == "cpu");
}

TEST_CASE("workflows::solve does not throw when GPU is asked for but "
          "unavailable (silent CPU fallback)",
          "[workflows][backend]") {
    auto H = build_heisen(4);

    SolveOptions opts;
    opts.num_eigs = 1;
    opts.backend.allow_gpu = true;
    // The CPU lane will be picked when no GPU is present or the
    // operator is host-only (which `build_heisen` produces).
    REQUIRE_NOTHROW(ed::workflows::solve(*H, opts));
}
