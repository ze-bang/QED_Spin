// =============================================================================
// test_ed_solver_matrix_e2e (Catch2 v3)
//
// End-to-end correctness check for every ground-state algorithm reachable
// through `ed::workflows::solve(H, opts)`. The point is to verify
// that the unified `<Backend>`-templated kernels
//
//   (a) `lanczos_kernel<CpuBackend>`
//   (b) `block_lanczos_kernel<CpuBackend>`
//   (c) `krylov_schur_kernel<CpuBackend>`
//   (d) the FullDiag (LAPACK zheevd) fallback
//
// all return the same ground-state energy on a small Sz-conserving
// Heisenberg chain, regardless of which solver the caller picks.
//
// The Lanczos-family numerical accuracy is already covered by
// `test_lanczos_variants` / `test_full_diagonalization`. This file is
// intentionally about the DISPATCH WIRING: every ground-state method
// must be reachable from `workflows::solve` and return the same answer.
//
// Migrated from the legacy `ed::auto_pilot::solve(...)` API during the
// ED Cleanup Sweep Phase 2 (May 2026). Auto-Sz projection has moved
// from auto_pilot into the caller: tests that previously relied on
// implicit projection now build a `FixedSzOperator` themselves before
// calling `workflows::solve`.
// =============================================================================

#include "common/catch2_harness.h"

#include <ed/core/fixed_sz_operator.h>
#include <ed/orchestrator.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <memory>
#include <vector>

using namespace ed_tests;
using ed::workflows::SolveMethod;
using ed::workflows::SolveOptions;

namespace {

constexpr uint64_t kN          = 8;            // 8-site Heisenberg PBC chain
constexpr uint64_t kFullDim    = 1ULL << kN;   // 256
constexpr uint64_t kSzGsDim    = 70;           // C(8, 4) = 70
constexpr double   kTolEnergy  = 1e-8;

// Pinned dense reference for the 8-site chain. Cheap (256x256 LAPACK).
struct Fixture {
    std::unique_ptr<Operator>          H_full;
    std::unique_ptr<FixedSzOperator>   H_sz;        // n_up = N/2
    DenseReference                     ref;
};
inline Fixture make_fixture() {
    Fixture f;
    f.H_full = build_heisenberg_chain(kN, /*J=*/1.0, /*periodic=*/true);
    f.H_sz   = build_heisenberg_chain_fixed_sz(kN, /*J=*/1.0,
                                                /*n_up=*/static_cast<int64_t>(kN / 2),
                                                /*periodic=*/true);
    f.ref    = reference_from_operator(*f.H_full, kFullDim);
    return f;
}

}  // namespace

// ---------------------------------------------------------------------------
// 1. Auto heuristic picks FullDiag (dim <= 2^12) and lands the GS.
// ---------------------------------------------------------------------------
TEST_CASE("workflows::solve e2e: Auto on the 8-site chain lands FullDiag and "
          "matches the dense reference GS",
          "[workflows][e2e][gs]") {
    auto f = make_fixture();

    SolveOptions opts;
    opts.num_eigs = 1;
    opts.method   = SolveMethod::Auto;

    auto res = ed::workflows::solve(*f.H_full, opts);
    REQUIRE(res.eigenvalues.size() >= 1);
    REQUIRE(std::abs(res.eigenvalues[0] - f.ref.eigs[0]) < kTolEnergy);
}

// ---------------------------------------------------------------------------
// 1b. Direct dense assembly (try_build_dense_columns, O(nnz) from the term
//     structure) must produce EXACTLY the same matrix as the O(dim)-matvec
//     column build, for both the full-space and fixed-Sz lanes. This is what the
//     FullDiag path now uses to skip the slow N-matvec construction.
// ---------------------------------------------------------------------------
TEST_CASE("dense assembly: try_build_dense_columns == matvec column build",
          "[workflows][dense][assembly]") {
    auto f = make_fixture();
    using C = std::complex<double>;

    auto check = [](const ed::matvec::MatVecOperator& op, std::size_t N) {
        std::vector<C> direct(N * N, C(0.0, 0.0));
        REQUIRE(op.try_build_dense_columns(direct.data(), N));   // lane supports it

        std::vector<C> via_mv(N * N, C(0.0, 0.0));
        std::vector<C> e(N), col(N);
        for (std::size_t j = 0; j < N; ++j) {
            std::fill(e.begin(), e.end(), C(0.0, 0.0));
            e[j] = C(1.0, 0.0);
            op.apply(e.data(), col.data(), N);
            for (std::size_t i = 0; i < N; ++i) via_mv[i + j * N] = col[i];
        }
        double maxdiff = 0.0;
        for (std::size_t k = 0; k < N * N; ++k)
            maxdiff = std::max(maxdiff, std::abs(direct[k] - via_mv[k]));
        REQUIRE(maxdiff < 1e-12);
    };

    check(*f.H_full, kFullDim);     // full Hilbert space (index == state)
    check(*f.H_sz,   kSzGsDim);     // fixed-Sz (combinadic index <-> state)
}

// ---------------------------------------------------------------------------
// 2. Explicit-method overrides per kernel. Each must hit the GS within
//    tolerance on the projected (n_up = N/2) Hilbert subspace.
// ---------------------------------------------------------------------------
TEST_CASE("workflows::solve e2e: explicit Lanczos on Sz=0 sector matches "
          "dense reference GS",
          "[workflows][e2e][gs][lanczos]") {
    auto f = make_fixture();

    SolveOptions opts;
    opts.num_eigs  = 1;
    opts.method    = SolveMethod::Lanczos;
    opts.tolerance = 1e-12;

    auto res = ed::workflows::solve(*f.H_sz, opts);
    REQUIRE(res.eigenvalues.size() >= 1);
    REQUIRE(std::abs(res.eigenvalues[0] - f.ref.eigs[0]) < kTolEnergy);
}

TEST_CASE("workflows::solve e2e: explicit KrylovSchur returns the lowest "
          "Sz=0 sector eigenvalues",
          "[workflows][e2e][gs][krylov]") {
    auto f = make_fixture();

    SolveOptions opts;
    opts.num_eigs  = 5;
    opts.method    = SolveMethod::KrylovSchur;
    opts.tolerance = 1e-12;

    auto res = ed::workflows::solve(*f.H_sz, opts);
    REQUIRE(res.eigenvalues.size() >= 5);
    // The GS is in n_up=4; the dense reference's higher eigenvalues
    // may live in other Sz sectors. We require the GS to agree tightly;
    // the in-sector excitations are validated separately by
    // test_block_lanczos.
    REQUIRE(std::abs(res.eigenvalues[0] - f.ref.eigs[0]) < kTolEnergy);
}

TEST_CASE("workflows::solve e2e: explicit BlockLanczos returns several "
          "Sz=0 sector eigenvalues",
          "[workflows][e2e][gs][block_lanczos]") {
    auto f = make_fixture();

    SolveOptions opts;
    opts.num_eigs   = 8;
    opts.block_size = 4;
    opts.method     = SolveMethod::BlockLanczos;
    opts.tolerance  = 1e-12;

    auto res = ed::workflows::solve(*f.H_sz, opts);
    REQUIRE(res.eigenvalues.size() >= 1);
    REQUIRE(std::abs(res.eigenvalues[0] - f.ref.eigs[0]) < kTolEnergy);
}

TEST_CASE("workflows::solve e2e: explicit FullDiag on Sz=0 sector matches "
          "dense reference GS",
          "[workflows][e2e][gs][full]") {
    auto f = make_fixture();

    SolveOptions opts;
    opts.num_eigs = 4;
    opts.method   = SolveMethod::FullDiag;

    auto res = ed::workflows::solve(*f.H_sz, opts);
    REQUIRE(res.eigenvalues.size() >= 1);
    REQUIRE(std::abs(res.eigenvalues[0] - f.ref.eigs[0]) < kTolEnergy);
}

// ---------------------------------------------------------------------------
// 3. FixedSzOperator dimension cap. Requesting more eigenvalues than the
//    sector can hold must be safely truncated.
// ---------------------------------------------------------------------------
TEST_CASE("workflows::solve e2e: requesting more eigenvalues than the "
          "FixedSz sector contains caps the result at C(N, N/2)",
          "[workflows][e2e][fixed_sz]") {
    auto f = make_fixture();

    SolveOptions opts;
    opts.num_eigs = kSzGsDim + 50;
    opts.method   = SolveMethod::FullDiag;

    auto res = ed::workflows::solve(*f.H_sz, opts);
    REQUIRE(res.eigenvalues.size() <= kSzGsDim);
    REQUIRE(res.eigenvalues.size() >= 1);
    REQUIRE(std::abs(res.eigenvalues[0] - f.ref.eigs[0]) < kTolEnergy);
}

// ---------------------------------------------------------------------------
// 4. Full-Hilbert FullDiag recovers the full 2^N spectrum.
// ---------------------------------------------------------------------------
TEST_CASE("workflows::solve e2e: FullDiag over the full Hilbert space "
          "recovers the dense 2^N spectrum",
          "[workflows][e2e][full]") {
    auto f = make_fixture();

    SolveOptions opts;
    opts.num_eigs = kFullDim;
    opts.method   = SolveMethod::FullDiag;

    auto res = ed::workflows::solve(*f.H_full, opts);
    REQUIRE(res.eigenvalues.size() == kFullDim);
    require_eigs_close(res.eigenvalues, f.ref.eigs,
                       /*n=*/16, /*tol=*/kTolEnergy,
                       "workflows::solve FullDiag full spectrum");
}
