// =============================================================================
// tests/unit/test_execution_planner.cpp
//
// Decision tests for the capability-aware ED execution planner
// (ed::planner::plan_execution). SystemCapabilities are constructed by hand so
// the assertions are deterministic and independent of the test machine.
//
// Coverage:
//   1. small dim + big RAM           -> CSR, cpu lane, feasible
//   2. 36-site sz+sym + 2 TB node    -> CSR chosen (NO hard dim cap)
//   3. 36-site sz+sym + modest RAM   -> matrix-free + tableless binary-search reps
//   4. 36-site sz-only + modest RAM  -> memory-infeasible on cpu
//   5. 36-site sz-only + MPI host    -> escalates to the mpi lane
//   6. user matvec override          -> honored over the fit/amortize test
//   7. csr_policy_hook round-trip    -> apply_csr_decision drives csr_override()
//   8. probe_system()                -> returns sane values
// =============================================================================

#include "common/catch2_harness.h"

#include <ed/planner/execution_planner.h>
#include <ed/planner/system_capabilities.h>
#include <ed/planner/csr_policy_hook.h>

using namespace ed::planner;

namespace {

SystemCapabilities make_caps(double ram_gb, int n_gpus, double vram_gb,
                             int n_ranks, bool cuda, bool mpi, bool nccl) {
    SystemCapabilities c;
    c.ram_total_bytes = (std::uint64_t)(ram_gb * (1ull << 30));
    c.ram_avail_bytes = c.ram_total_bytes;
    c.n_cores = 16;
    c.n_gpus = n_gpus;
    c.vram_total_bytes = (std::uint64_t)(vram_gb * (1ull << 30));
    c.vram_avail_bytes = c.vram_total_bytes;
    c.n_mpi_ranks = n_ranks;
    c.has_cuda_build = cuda;
    c.has_mpi_build = mpi;
    c.has_nccl_build = nccl;
    return c;
}

// 36-site triangular-ish: ~108 bonds; XXZ has ~1 off-diagonal hop per bond.
TaskDescriptor task_36_symsz(std::uint64_t dim) {
    TaskDescriptor t;
    t.basis_dim = dim;
    t.n_terms = 108;
    t.n_offdiag = 72;
    t.method = Method::Lanczos;
    t.num_eigs = 1;
    t.kind = BasisKind::SymSz;
    t.N = 36;
    t.n_up = 18;
    t.group_size = 432;
    return t;
}

}  // namespace

TEST_CASE("planner: small dim + big RAM picks CSR on CPU", "[planner]") {
    TaskDescriptor t;
    t.basis_dim = 4096;
    t.n_terms = 24;
    t.n_offdiag = 16;
    t.method = Method::Lanczos;
    t.kind = BasisKind::Full;

    auto caps = make_caps(/*ram*/256, 0, 0, 1, false, false, false);
    auto p = plan_execution(t, caps, UserConstraints{});

    REQUIRE(p.feasible);
    REQUIRE(p.device == DeviceLane::Cpu);
    REQUIRE(p.matvec == MatvecStrategy::Csr);
}

TEST_CASE("planner: 36-site sz+sym on a 2 TB node picks CSR (no hard cap)",
          "[planner]") {
    auto t = task_36_symsz(/*dim*/21'000'000ull);   // ~ C(36,18)/432
    auto caps = make_caps(/*ram*/2048, 0, 0, 1, false, false, false);
    auto p = plan_execution(t, caps, UserConstraints{});

    REQUIRE(p.device == DeviceLane::Cpu);
    REQUIRE(p.matvec == MatvecStrategy::Csr);   // the headline: CSR at dim 2e7
    REQUIRE(p.feasible);
}

TEST_CASE("planner: 36-site sz+sym on modest RAM -> matrix-free + tableless reps",
          "[planner]") {
    auto t = task_36_symsz(/*dim*/21'000'000ull);
    auto caps = make_caps(/*ram*/32, 0, 0, 1, false, false, false);
    auto p = plan_execution(t, caps, UserConstraints{});

    REQUIRE(p.matvec == MatvecStrategy::MatrixFree);   // CSR ~37 GB doesn't fit
    REQUIRE(p.basis == BasisStrategy::BinarySearchReps);  // 36 GB rank table skipped
    REQUIRE(p.feasible);                               // working set (~1.3 GB) fits
}

TEST_CASE("planner: 36-site sz-only on modest CPU RAM is memory-infeasible",
          "[planner]") {
    TaskDescriptor t;
    t.basis_dim = 9'075'135'300ull;   // C(36,18)
    t.n_terms = 108; t.n_offdiag = 72;
    t.method = Method::Lanczos;
    t.kind = BasisKind::Sz; t.N = 36; t.n_up = 18;

    auto caps = make_caps(/*ram*/64, 0, 0, 1, false, false, false);
    auto p = plan_execution(t, caps, UserConstraints{});

    REQUIRE_FALSE(p.feasible);
    // Two independent walls both exceed the ~50 GB budget: the ~72 GB basis
    // array (enumeration) and the ~580 GB Lanczos working set. The planner
    // reports whichever it checks first; either is a correct infeasibility.
    REQUIRE((p.bottleneck == std::string("basis_construction")
             || p.bottleneck == std::string("memory")));
}

TEST_CASE("planner: 36-site sz-only escalates to MPI when available", "[planner]") {
    TaskDescriptor t;
    t.basis_dim = 9'075'135'300ull;
    t.n_terms = 108; t.n_offdiag = 72;
    t.method = Method::Lanczos;
    t.kind = BasisKind::Sz; t.N = 36; t.n_up = 18;

    // Many ranks, each with modest RAM -> auto should route to mpi.
    auto caps = make_caps(/*ram*/128, 0, 0, /*ranks*/256, false, true, false);
    auto p = plan_execution(t, caps, UserConstraints{});
    REQUIRE(p.device == DeviceLane::Mpi);
    REQUIRE(p.n_ranks == 256);
}

TEST_CASE("planner: user matvec override beats the fit/amortize test", "[planner]") {
    TaskDescriptor t;
    t.basis_dim = 4096; t.n_terms = 24; t.n_offdiag = 16;
    t.method = Method::Lanczos; t.kind = BasisKind::Full;
    auto caps = make_caps(256, 0, 0, 1, false, false, false);

    UserConstraints uc;
    uc.matvec = "matrix_free";
    auto p = plan_execution(t, caps, uc);
    REQUIRE(p.matvec == MatvecStrategy::MatrixFree);   // forced despite CSR fitting
}

TEST_CASE("planner: csr_policy_hook round-trips through apply_csr_decision",
          "[planner]") {
    clear_csr_override();
    REQUIRE(csr_override() == -1);

    ExecutionPlan p;
    p.matvec = MatvecStrategy::Csr;
    apply_csr_decision(p);
    REQUIRE(csr_override() == 1);

    p.matvec = MatvecStrategy::MatrixFree;
    apply_csr_decision(p);
    REQUIRE(csr_override() == 0);

    clear_csr_override();
    REQUIRE(csr_override() == -1);
}

TEST_CASE("planner: probe_system returns sane values", "[planner]") {
    auto c = probe_system(/*refresh*/true);
    REQUIRE(c.n_cores >= 1);
    REQUIRE(c.ram_total_bytes > 0);
    REQUIRE(c.ram_avail_bytes > 0);
    REQUIRE(c.n_mpi_ranks >= 1);
}
