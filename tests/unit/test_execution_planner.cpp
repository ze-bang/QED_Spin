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
#include <ed/planner/sym_matvec_policy_hook.h>

#include <cmath>
#include <limits>
#include <string>

using namespace ed::planner;

namespace {

// Invariants every plan must satisfy regardless of how adversarial the input is.
// A "goofy" planner (NaN/inf estimates, wrapped byte counts that flip an
// impossible problem to feasible, negative rank counts, inconsistent
// feasibility, garbage enum strings, or a crash) trips one of these.
void require_sane(const ExecutionPlan& p) {
    REQUIRE(std::isfinite(p.est_memory_gb));
    REQUIRE(p.est_memory_gb >= 0.0);
    REQUIRE(std::isfinite(p.est_seconds));
    REQUIRE(p.est_seconds >= 0.0);
    REQUIRE(p.n_ranks >= 1);
    // enum -> string accessors return a known, non-empty token (no fallthrough).
    REQUIRE(std::string(p.matvec_str()) != std::string());
    REQUIRE(std::string(p.device_str()) != std::string());
    REQUIRE(std::string(p.basis_str())  != std::string());
    REQUIRE(std::string(p.reorth_str()) != std::string());
    REQUIRE_FALSE(p.summary().empty());
    // Feasibility must be self-consistent.
    if (!p.feasible) {
        REQUIRE(p.bottleneck != std::string("ok"));
        REQUIRE_FALSE(p.suggestions.empty());
    } else {
        REQUIRE(p.bottleneck == std::string("ok"));
    }
}

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
    // orbit-CSR ~ 21e6 * 432 * 24 B ~= 203 GB does NOT fit 32 GB -> rep walk.
    REQUIRE_FALSE(p.sym_orbit_csr);
}

TEST_CASE("planner: symmetry rep-walk vs orbit-CSR is memory-gated", "[planner]") {
    // Large 36-site SymSz sector on a 2 TB node: orbit-CSR (~203 GB) fits the
    // budget, so the planner materializes it (faster apply).
    {
        auto t = task_36_symsz(/*dim*/21'000'000ull);
        auto caps = make_caps(/*ram*/2048, 0, 0, 1, false, false, false);
        auto p = plan_execution(t, caps, UserConstraints{});
        REQUIRE(p.sym_orbit_csr);
    }
    // Same sector on 32 GB: orbit-CSR can't fit -> matrix-free rep walk.
    {
        auto t = task_36_symsz(/*dim*/21'000'000ull);
        auto caps = make_caps(/*ram*/32, 0, 0, 1, false, false, false);
        auto p = plan_execution(t, caps, UserConstraints{});
        REQUIRE_FALSE(p.sym_orbit_csr);
    }
    // A small symmetry sector fits orbit-CSR even on a modest box.
    {
        auto t = task_36_symsz(/*dim*/50'000ull);     // 50k * 432 * 24 B ~= 0.5 GB
        auto caps = make_caps(/*ram*/8, 0, 0, 1, false, false, false);
        auto p = plan_execution(t, caps, UserConstraints{});
        REQUIRE(p.sym_orbit_csr);
    }
    // The non-symmetry lane never sets the symmetry flag.
    {
        TaskDescriptor t;
        t.basis_dim = 4096; t.n_terms = 20; t.n_offdiag = 12;
        t.method = Method::Lanczos; t.num_eigs = 1; t.kind = BasisKind::Full;
        auto caps = make_caps(/*ram*/256, 0, 0, 1, false, false, false);
        auto p = plan_execution(t, caps, UserConstraints{});
        REQUIRE_FALSE(p.sym_orbit_csr);
    }
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

// ============================ stress / fuzz =================================

TEST_CASE("planner: block-Lanczos memory accounts for block_size + recommends lean",
          "[planner]") {
    // 20M-dim sector, Block Lanczos, block_size=8, ~80 iterations. The stored
    // basis is ~80*8*20M*16B ~= 192 GiB -- the OLD estimate (ignoring block_size)
    // mis-reported ~32 GiB and let it OOM. On a 64 GB node:
    auto caps = make_caps(/*ram*/64, 0, 0, 1, false, false, false);
    TaskDescriptor t;
    t.basis_dim = 20'000'000ull; t.n_terms = 100; t.n_offdiag = 64;
    t.method = Method::BlockLanczos; t.num_eigs = 8; t.krylov_dim = 80;
    t.block_size = 8; t.kind = BasisKind::Sz; t.N = 36; t.n_up = 18;

    SECTION("eigenvalues-only -> lean recommended (and feasible lean)") {
        t.compute_vectors = false;
        auto p = plan_execution(t, caps, UserConstraints{});
        REQUIRE(p.block_lanczos_lean);                 // full basis doesn't fit -> lean
        REQUIRE(p.est_memory_gb < 50.0);               // lean footprint is small
    }
    SECTION("eigenvectors -> cannot go lean; steer to Block Krylov-Schur") {
        t.compute_vectors = true;
        auto p = plan_execution(t, caps, UserConstraints{});
        REQUIRE_FALSE(p.block_lanczos_lean);
        REQUIRE_FALSE(p.feasible);                     // ~192 GiB basis can't fit 64 GB
        bool mentions_bks = false;
        for (const auto& s : p.suggestions)
            if (s.find("BLOCK_KRYLOV_SCHUR") != std::string::npos) mentions_bks = true;
        REQUIRE(mentions_bks);
    }
}

TEST_CASE("planner stress: degenerate dimensions stay sane", "[planner][stress]") {
    auto caps = make_caps(/*ram*/64, 0, 0, 1, false, false, false);
    for (std::uint64_t dim : {std::uint64_t{0}, std::uint64_t{1}, std::uint64_t{2},
                              std::uint64_t{1023}, std::uint64_t{1024}}) {
        TaskDescriptor t;
        t.basis_dim = dim; t.n_terms = 10; t.n_offdiag = 6;
        t.method = Method::Lanczos; t.kind = BasisKind::Full;
        auto p = plan_execution(t, caps, UserConstraints{});
        require_sane(p);
        REQUIRE_FALSE(p.sym_orbit_csr);   // Full lane never sets it
    }
}

TEST_CASE("planner stress: astronomically large dims do NOT wrap to feasible",
          "[planner][stress]") {
    // The headline anti-goofy guard: a 2^62 basis must be reported infeasible on
    // a finite node, not silently wrapped (overflow) into a tiny working set.
    auto caps = make_caps(/*ram*/256, 0, 0, 1, false, false, false);
    for (std::uint64_t dim : {std::uint64_t{1} << 50, std::uint64_t{1} << 62,
                              std::numeric_limits<std::uint64_t>::max() / 32}) {
        TaskDescriptor t;
        t.basis_dim = dim; t.n_terms = 100; t.n_offdiag = 64;
        t.method = Method::Lanczos; t.kind = BasisKind::Sz; t.N = 36; t.n_up = 18;
        auto p = plan_execution(t, caps, UserConstraints{});
        require_sane(p);
        REQUIRE_FALSE(p.feasible);                 // must NOT claim feasible
        REQUIRE(p.est_memory_gb > 1.0);            // working set is genuinely huge
        REQUIRE(p.matvec == MatvecStrategy::MatrixFree);  // CSR can't fit either
    }
}

TEST_CASE("planner stress: N>=64 / N=0 do not invoke UB shifts", "[planner][stress]") {
    auto caps = make_caps(/*ram*/128, 0, 0, 1, false, false, false);
    for (int N : {0, 32, 63, 64, 100, -4}) {
        for (auto kind : {BasisKind::Symm, BasisKind::SymSz}) {
            TaskDescriptor t;
            t.basis_dim = 1u << 16; t.n_terms = 20; t.n_offdiag = 12;
            t.method = Method::Lanczos; t.kind = kind;
            t.N = N; t.n_up = (kind == BasisKind::SymSz ? std::max(0, N / 2) : -1);
            t.group_size = 48;
            auto p = plan_execution(t, caps, UserConstraints{});
            require_sane(p);
        }
    }
}

TEST_CASE("planner stress: degenerate symmetry params (group 0 / huge)",
          "[planner][stress]") {
    auto caps = make_caps(/*ram*/64, 0, 0, 1, false, false, false);
    // group_size = 0 must not divide-by-zero; treated as >=1.
    {
        TaskDescriptor t;
        t.basis_dim = 1u << 16; t.n_terms = 20; t.n_offdiag = 12;
        t.method = Method::Lanczos; t.kind = BasisKind::Symm; t.N = 20;
        t.group_size = 0;
        auto p = plan_execution(t, caps, UserConstraints{});
        require_sane(p);
    }
    // Absurd group_size with a small dim must NOT overflow sym_orbit_csr to true:
    // orbit-CSR estimate ~ dim*group*24 is astronomically large -> rep walk.
    {
        TaskDescriptor t;
        t.basis_dim = 4096; t.n_terms = 20; t.n_offdiag = 12;
        t.method = Method::Lanczos; t.kind = BasisKind::SymSz; t.N = 24; t.n_up = 12;
        t.group_size = std::uint64_t{1} << 50;
        auto p = plan_execution(t, caps, UserConstraints{});
        require_sane(p);
        REQUIRE_FALSE(p.sym_orbit_csr);            // does not fit -> rep walk
    }
}

TEST_CASE("planner stress: degenerate n_up (out of range / boundary)",
          "[planner][stress]") {
    auto caps = make_caps(/*ram*/64, 0, 0, 1, false, false, false);
    for (int n_up : {-5, 0, 18, 36, 50}) {     // 50 > N=36 is invalid
        TaskDescriptor t;
        t.basis_dim = 1u << 18; t.n_terms = 40; t.n_offdiag = 24;
        t.method = Method::FTLM; t.n_samples = 8;
        t.kind = BasisKind::SymSz; t.N = 36; t.n_up = n_up; t.group_size = 48;
        auto p = plan_execution(t, caps, UserConstraints{});
        require_sane(p);
    }
}

TEST_CASE("planner stress: extreme memory_safety / RAM / rank constraints",
          "[planner][stress]") {
    TaskDescriptor t;
    t.basis_dim = 1u << 20; t.n_terms = 40; t.n_offdiag = 24;
    t.method = Method::Lanczos; t.kind = BasisKind::Full;

    // memory_safety at the extremes (0 -> budget 0; large -> budget huge).
    for (double ms : {0.0, 1e-9, 1.0, 8.0}) {
        auto caps = make_caps(/*ram*/16, 0, 0, 1, false, false, false);
        UserConstraints uc; uc.memory_safety = ms;
        auto p = plan_execution(t, caps, uc);
        require_sane(p);
    }
    // Zero / tiny RAM node: budget floors at >=0, never negative, no crash.
    {
        auto caps = make_caps(/*ram*/0, 0, 0, 1, false, false, false);
        auto p = plan_execution(t, caps, UserConstraints{});
        require_sane(p);
    }
    // Negative / zero requested rank count must not yield n_ranks < 1.
    {
        auto caps = make_caps(/*ram*/128, 0, 0, /*ranks*/8, false, true, false);
        UserConstraints uc; uc.device = "mpi"; uc.n_ranks = -3;
        auto p = plan_execution(t, caps, uc);
        require_sane(p);
    }
}

TEST_CASE("planner stress: unknown/unsupported device requests fall back sanely",
          "[planner][stress]") {
    TaskDescriptor t;
    t.basis_dim = 1u << 16; t.n_terms = 20; t.n_offdiag = 12;
    t.method = Method::Lanczos; t.kind = BasisKind::Full;
    auto caps = make_caps(/*ram*/64, 0, 0, 1, false, false, false);  // no GPU/MPI build

    for (const char* dev : {"gpu", "mpi", "mpi_gpu", "banana", "", "CPU"}) {
        UserConstraints uc; uc.device = dev;
        auto p = plan_execution(t, caps, uc);
        require_sane(p);
        // With no CUDA/MPI build, no plan may end up on a GPU/MPI lane.
        REQUIRE(p.device == DeviceLane::Cpu);
    }
}

TEST_CASE("planner stress: every method x basis-kind combination is sane",
          "[planner][stress]") {
    auto caps = make_caps(/*ram*/128, 0, 0, 1, false, false, false);
    const Method methods[] = {Method::Lanczos, Method::KrylovSchur,
                              Method::BlockLanczos, Method::FTLM, Method::LTLM,
                              Method::TPQ, Method::KPM, Method::Full};
    const BasisKind kinds[] = {BasisKind::Full, BasisKind::Sz,
                               BasisKind::Symm, BasisKind::SymSz};
    for (auto m : methods) {
        for (auto k : kinds) {
            TaskDescriptor t;
            t.basis_dim = 1u << 17; t.n_terms = 30; t.n_offdiag = 18;
            t.method = m; t.num_eigs = 4; t.n_samples = 4; t.compute_vectors = true;
            t.kind = k; t.N = 24;
            t.n_up = (k == BasisKind::Sz || k == BasisKind::SymSz) ? 12 : -1;
            t.group_size = (k == BasisKind::Symm || k == BasisKind::SymSz) ? 48 : 1;
            auto p = plan_execution(t, caps, UserConstraints{});
            require_sane(p);
            if (k == BasisKind::Full || k == BasisKind::Sz)
                REQUIRE_FALSE(p.sym_orbit_csr);  // symmetry flag is symm-only
        }
    }
}

TEST_CASE("planner stress: forced matvec is honored even when it cannot fit",
          "[planner][stress]") {
    // A user force is a directive, not a suggestion: CSR forced on a problem
    // whose CSR clearly won't fit must still report CSR (intended, not goofy).
    TaskDescriptor t;
    t.basis_dim = 9'075'135'300ull; t.n_terms = 108; t.n_offdiag = 72;
    t.method = Method::Lanczos; t.kind = BasisKind::Sz; t.N = 36; t.n_up = 18;
    auto caps = make_caps(/*ram*/16, 0, 0, 1, false, false, false);
    UserConstraints uc; uc.matvec = "csr";
    auto p = plan_execution(t, caps, uc);
    require_sane(p);
    REQUIRE(p.matvec == MatvecStrategy::Csr);
}

TEST_CASE("planner stress: sym_matvec_policy_hook round-trips + scoped guard",
          "[planner][stress]") {
    clear_sym_matvec_repr();
    REQUIRE(sym_matvec_repr() == -1);              // Auto

    ExecutionPlan p;
    p.sym_orbit_csr = true;
    apply_sym_matvec_decision(p);
    REQUIRE(sym_matvec_repr() == static_cast<int>(SymMatvecRepr::OrbitCsr));

    p.sym_orbit_csr = false;
    apply_sym_matvec_decision(p);
    REQUIRE(sym_matvec_repr() == static_cast<int>(SymMatvecRepr::Rep));

    {
        ScopedSymMatvecRepr guard(SymMatvecRepr::OrbitCsr);
        REQUIRE(sym_matvec_repr() == static_cast<int>(SymMatvecRepr::OrbitCsr));
    }
    REQUIRE(sym_matvec_repr() == static_cast<int>(SymMatvecRepr::Rep));  // restored

    clear_sym_matvec_repr();
    REQUIRE(sym_matvec_repr() == -1);
}
