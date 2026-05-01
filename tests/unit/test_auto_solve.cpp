// =============================================================================
// test_auto_solve  (Catch2 v3)
//
// Smoke-tests the modern-C++ auto-pilot façade `ed::auto_pilot::solve(...)`.
// The façade itself is heuristic on top of `exact_diagonalization_core`, so
// these tests intentionally focus on the AUTO BEHAVIOUR (Sz guard, fixed-Sz
// projection, default-method picking, GPU-fallback) rather than re-testing
// numerical correctness of every underlying solver — those are covered by
// test_full_diagonalization, test_lanczos_variants, etc.
// =============================================================================

#include "common/catch2_harness.h"

#include <ed/auto/solve.h>
#include <ed/operators/spin_ops.h>

#include <memory>
#include <string>
#include <vector>

using namespace ed_tests;
using ed::auto_pilot::AutoSolveOptions;
using ed::auto_pilot::Device;
using ed::auto_pilot::solve;

namespace {

// Small Heisenberg chain (Sz-conserving; ground-state energy known from
// Bethe ansatz and from test_full_diagonalization for small N).
std::unique_ptr<Operator> build_heisen(uint64_t N) {
    return build_heisenberg_chain(N, 1.0);
}

} // namespace

TEST_CASE("auto_pilot::solve picks FULL on small full Hilbert space",
          "[auto_pilot]") {
    auto H = build_heisen(4);                 // dim = 16, ≤ small-dim threshold
    auto ref = reference_from_operator(*H, 16);

    AutoSolveOptions opts;
    opts.num_eigenvalues = 4;
    opts.verbose         = false;             // keep test output clean
    opts.device          = Device::CPU;       // pin device to avoid GPU promo

    auto res = solve(*H, opts);

    REQUIRE(res.eigenvalues.size() >= 4);
    require_eigs_close(res.eigenvalues, ref.eigs, 4, 1e-8,
                       "auto_pilot::solve N=4 ground manifold");
}

TEST_CASE("auto_pilot::solve auto-projects when sz= is given on a "
          "Sz-conserving Hamiltonian",
          "[auto_pilot]") {
    auto H = build_heisen(6);                 // full dim = 64

    AutoSolveOptions opts;
    opts.num_eigenvalues = 1;
    opts.verbose         = false;
    opts.device          = Device::CPU;
    opts.sz              = 3;                 // half-filled sector

    auto res = solve(*H, opts);

    // Ground state of XXX chain has total Sz=0 (n_up=N/2). Energy must
    // match the global ground-state energy from the unprojected solve.
    auto Hfull = build_heisen(6);
    AutoSolveOptions full;
    full.num_eigenvalues = 1;
    full.verbose         = false;
    full.device          = Device::CPU;
    auto res_full = solve(*Hfull, full);

    REQUIRE(res.eigenvalues.size() >= 1);
    REQUIRE(std::abs(res.eigenvalues[0] - res_full.eigenvalues[0]) < 1e-8);
}

TEST_CASE("auto_pilot::solve refuses sz= when Sz is broken",
          "[auto_pilot]") {
    // Inject a bare S+ term so that conserves_sz() must return false; the
    // auto-pilot must throw rather than silently project onto the wrong
    // sector.
    auto H = build_heisen(4);
    Operator::TransformData tdata{};
    tdata.is_two_body  = false;
    tdata.op_type      = 0;          // S+ → net Sz shift = +1, breaks Sz.
    tdata.site_index   = 0;
    tdata.coefficient  = Complex(0.5, 0.0);
    H->transform_data_.push_back(tdata);
    H->invalidateMatrixCaches();

    AutoSolveOptions opts;
    opts.num_eigenvalues = 1;
    opts.verbose         = false;
    opts.device          = Device::CPU;
    opts.sz              = 2;

    REQUIRE_THROWS_AS(solve(*H, opts), std::invalid_argument);
}

TEST_CASE("auto_pilot::solve falls back to CPU when GPU unavailable",
          "[auto_pilot]") {
    auto H = build_heisen(4);

    AutoSolveOptions opts;
    opts.num_eigenvalues = 1;
    opts.verbose         = false;
    opts.device          = Device::GPU;
    opts.allow_fallback  = true;

    // Should NOT throw regardless of build flags: either GPU runs, or we
    // silently fall back to CPU.
    REQUIRE_NOTHROW(solve(*H, opts));
}
