// =============================================================================
// test_method_canonicalize (Catch2 v3, Phase 7)
//
// Pins the contract of ed::canonicalize_method_and_flags() — the single point
// where deprecated `_GPU` / `_CUDA` / `_MPI` / `_FIXED_SZ` enum variants are
// collapsed onto the canonical (base_method, use_fixed_sz, use_gpu, use_mpi)
// tuple.
//
// Goals:
//   1. Every legacy "_GPU" enum value canonicalizes to (base, *, true, *).
//   2. Every legacy "_FIXED_SZ" enum value canonicalizes to (base, true, true, *).
//   3. mTPQ_CUDA is an alias for mTPQ + use_gpu=true (not a separate kernel).
//   4. mTPQ_MPI canonicalizes to mTPQ + use_mpi=true.
//   5. SCALAPACK / SCALAPACK_MIXED stay as-is (they are different solver
//      kernels, not "FULL with use_mpi=true"); flagged use_mpi=true.
//   6. The transformation is idempotent (canonicalize(canonicalize(x)) == canonicalize(x)).
//   7. Caller-supplied flags are preserved by OR-merge (never lost).
//
// Phase 7 / Q12.
// =============================================================================

// We deliberately reference deprecated enum values throughout this file —
// the whole point is to test that the canonicalizer handles them. Suppress
// the deprecation warning at file scope.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"

#include <catch2/catch_test_macros.hpp>

#include <ed/core/ed_method_traits.h>
#include <ed/core/ed_types.h>

using ed::CanonicalMethod;
using ed::canonicalize_method_and_flags;
using ed::legacy_method_for_dispatch;
using M = DiagonalizationMethod;

namespace {

// Helper for compact equality comparisons.
struct Tup {
    M    method;
    bool use_fixed_sz;
    bool use_gpu;
    bool use_mpi;
    constexpr bool operator==(const Tup& o) const noexcept {
        return method == o.method &&
               use_fixed_sz == o.use_fixed_sz &&
               use_gpu == o.use_gpu &&
               use_mpi == o.use_mpi;
    }
};

constexpr Tup canon(M m, bool fz = false, bool gpu = false, bool mpi = false) {
    auto c = canonicalize_method_and_flags(m, fz, gpu, mpi);
    return Tup{c.method, c.use_fixed_sz, c.use_gpu, c.use_mpi};
}

}  // namespace

TEST_CASE("canonicalize_method_and_flags: identity for canonical inputs", "[method][canonicalize]") {
    // CPU iterative + dense solvers without device flags should round-trip
    // unchanged.
    REQUIRE(canon(M::LANCZOS)              == (Tup{M::LANCZOS,             false, false, false}));
    REQUIRE(canon(M::LANCZOS_SELECTIVE)    == (Tup{M::LANCZOS_SELECTIVE,   false, false, false}));
    REQUIRE(canon(M::LANCZOS_NO_ORTHO)     == (Tup{M::LANCZOS_NO_ORTHO,    false, false, false}));
    REQUIRE(canon(M::BLOCK_LANCZOS)        == (Tup{M::BLOCK_LANCZOS,       false, false, false}));
    REQUIRE(canon(M::CHEBYSHEV_FILTERED)   == (Tup{M::CHEBYSHEV_FILTERED,  false, false, false}));
    REQUIRE(canon(M::SHIFT_INVERT)         == (Tup{M::SHIFT_INVERT,        false, false, false}));
    REQUIRE(canon(M::DAVIDSON)             == (Tup{M::DAVIDSON,            false, false, false}));
    REQUIRE(canon(M::BICG)                 == (Tup{M::BICG,                false, false, false}));
    REQUIRE(canon(M::LOBPCG)               == (Tup{M::LOBPCG,              false, false, false}));
    REQUIRE(canon(M::KRYLOV_SCHUR)         == (Tup{M::KRYLOV_SCHUR,        false, false, false}));
    REQUIRE(canon(M::BLOCK_KRYLOV_SCHUR)   == (Tup{M::BLOCK_KRYLOV_SCHUR,  false, false, false}));
    REQUIRE(canon(M::IMPLICIT_RESTART_LANCZOS) == (Tup{M::IMPLICIT_RESTART_LANCZOS, false, false, false}));
    REQUIRE(canon(M::THICK_RESTART_LANCZOS)    == (Tup{M::THICK_RESTART_LANCZOS,    false, false, false}));
    REQUIRE(canon(M::FULL)                 == (Tup{M::FULL,                false, false, false}));
    REQUIRE(canon(M::OSS)                  == (Tup{M::OSS,                 false, false, false}));

    // Thermal base methods.
    REQUIRE(canon(M::mTPQ)                 == (Tup{M::mTPQ,                false, false, false}));
    REQUIRE(canon(M::cTPQ)                 == (Tup{M::cTPQ,                false, false, false}));
    REQUIRE(canon(M::FTLM)                 == (Tup{M::FTLM,                false, false, false}));
    REQUIRE(canon(M::LTLM)                 == (Tup{M::LTLM,                false, false, false}));
    REQUIRE(canon(M::HYBRID)               == (Tup{M::HYBRID,              false, false, false}));

    // ARPACK variants.
    REQUIRE(canon(M::ARPACK_SM)            == (Tup{M::ARPACK_SM,           false, false, false}));
    REQUIRE(canon(M::ARPACK_LM)            == (Tup{M::ARPACK_LM,           false, false, false}));
    REQUIRE(canon(M::ARPACK_SHIFT_INVERT)  == (Tup{M::ARPACK_SHIFT_INVERT, false, false, false}));
    REQUIRE(canon(M::ARPACK_ADVANCED)      == (Tup{M::ARPACK_ADVANCED,     false, false, false}));
}

TEST_CASE("canonicalize_method_and_flags: legacy _GPU variants", "[method][canonicalize][gpu]") {
    // Each deprecated _GPU enum collapses onto its base method + use_gpu=true.
    REQUIRE(canon(M::LANCZOS_GPU)            == (Tup{M::LANCZOS,            false, true, false}));
    REQUIRE(canon(M::BLOCK_LANCZOS_GPU)      == (Tup{M::BLOCK_LANCZOS,      false, true, false}));
    REQUIRE(canon(M::DAVIDSON_GPU)           == (Tup{M::DAVIDSON,           false, true, false}));
    REQUIRE(canon(M::LOBPCG_GPU)             == (Tup{M::LOBPCG,             false, true, false}));
    REQUIRE(canon(M::KRYLOV_SCHUR_GPU)       == (Tup{M::KRYLOV_SCHUR,       false, true, false}));
    REQUIRE(canon(M::BLOCK_KRYLOV_SCHUR_GPU) == (Tup{M::BLOCK_KRYLOV_SCHUR, false, true, false}));
    REQUIRE(canon(M::mTPQ_GPU)               == (Tup{M::mTPQ,               false, true, false}));
    REQUIRE(canon(M::cTPQ_GPU)               == (Tup{M::cTPQ,               false, true, false}));
    REQUIRE(canon(M::FTLM_GPU)               == (Tup{M::FTLM,               false, true, false}));
    REQUIRE(canon(M::FULL_GPU)               == (Tup{M::FULL,               false, true, false}));
}

TEST_CASE("canonicalize_method_and_flags: legacy _FIXED_SZ variants", "[method][canonicalize][fixed_sz]") {
    // Each deprecated combined GPU+FIXED_SZ enum collapses onto base + both flags.
    REQUIRE(canon(M::LANCZOS_GPU_FIXED_SZ)        == (Tup{M::LANCZOS,       true, true, false}));
    REQUIRE(canon(M::BLOCK_LANCZOS_GPU_FIXED_SZ)  == (Tup{M::BLOCK_LANCZOS, true, true, false}));
    REQUIRE(canon(M::FTLM_GPU_FIXED_SZ)           == (Tup{M::FTLM,          true, true, false}));
}

TEST_CASE("canonicalize_method_and_flags: mTPQ_CUDA aliases mTPQ_GPU", "[method][canonicalize][tpq]") {
    // The user-visible point of Phase 7: mTPQ_CUDA is *not* a separate kernel,
    // it is mTPQ on GPU. canonicalize_method_and_flags() must collapse them
    // onto the same canonical form, so any downstream branch that switches
    // on the canonical (base, flags) tuple cannot accidentally diverge.
    const auto a = canon(M::mTPQ_CUDA);
    const auto b = canon(M::mTPQ_GPU);
    REQUIRE(a == b);
    REQUIRE(a == (Tup{M::mTPQ, false, true, false}));
}

TEST_CASE("canonicalize_method_and_flags: legacy _MPI variants", "[method][canonicalize][mpi]") {
    REQUIRE(canon(M::mTPQ_MPI) == (Tup{M::mTPQ, false, false, true}));
}

TEST_CASE("canonicalize_method_and_flags: SCALAPACK kept as a distinct kernel", "[method][canonicalize][scalapack]") {
    // SCALAPACK and SCALAPACK_MIXED are not collapsed to FULL: they go
    // through PDSYEVR / mixed-precision refinement, which is a different
    // dense LAPACK call than FULL. They are however *implicitly* MPI-backed,
    // so canonicalize_method_and_flags() flags use_mpi=true honestly.
    REQUIRE(canon(M::SCALAPACK)       == (Tup{M::SCALAPACK,       false, false, true}));
    REQUIRE(canon(M::SCALAPACK_MIXED) == (Tup{M::SCALAPACK_MIXED, false, false, true}));
}

TEST_CASE("canonicalize_method_and_flags: caller flags OR-merge", "[method][canonicalize][flags]") {
    // Caller-supplied flags must never be lost; canonicalization only
    // *adds* flags inferred from the legacy enum value.
    REQUIRE(canon(M::LANCZOS, /*fz=*/true)                     == (Tup{M::LANCZOS, true,  false, false}));
    REQUIRE(canon(M::LANCZOS, /*fz=*/false, /*gpu=*/true)      == (Tup{M::LANCZOS, false, true,  false}));
    REQUIRE(canon(M::LANCZOS, /*fz=*/false, /*gpu=*/false, /*mpi=*/true) == (Tup{M::LANCZOS, false, false, true}));
    // Already-set flags survive a deprecated input.
    REQUIRE(canon(M::LANCZOS_GPU, /*fz=*/true)                 == (Tup{M::LANCZOS, true,  true,  false}));
    REQUIRE(canon(M::FTLM_GPU,    /*fz=*/true)                 == (Tup{M::FTLM,    true,  true,  false}));
    // mTPQ + caller-supplied use_mpi=true (canonical "mTPQ on MPI").
    REQUIRE(canon(M::mTPQ, /*fz=*/false, /*gpu=*/false, /*mpi=*/true)
            == (Tup{M::mTPQ, false, false, true}));
    // mTPQ_GPU + use_mpi=true (canonical "mTPQ on GPU+MPI", i.e. multi-GPU
    // distributed). Ensures the GPU and MPI axes are truly orthogonal.
    REQUIRE(canon(M::mTPQ_GPU, /*fz=*/false, /*gpu=*/false, /*mpi=*/true)
            == (Tup{M::mTPQ, false, true, true}));
}

TEST_CASE("canonicalize_method_and_flags: idempotent", "[method][canonicalize][idempotent]") {
    // Applying canonicalize twice must be a no-op for every enum value.
    // We exercise a representative sample of canonical, _GPU, _FIXED_SZ,
    // _MPI, _CUDA, and SCALAPACK inputs.
    constexpr M sample[] = {
        M::LANCZOS, M::FULL, M::FTLM, M::mTPQ, M::OSS,
        M::SCALAPACK, M::SCALAPACK_MIXED,
        M::LANCZOS_GPU, M::BLOCK_LANCZOS_GPU, M::FTLM_GPU, M::mTPQ_GPU,
        M::cTPQ_GPU, M::FULL_GPU,
        M::LANCZOS_GPU_FIXED_SZ, M::BLOCK_LANCZOS_GPU_FIXED_SZ, M::FTLM_GPU_FIXED_SZ,
        M::mTPQ_MPI, M::mTPQ_CUDA,
    };
    for (M m : sample) {
        const auto c1 = canonicalize_method_and_flags(m, false, false, false);
        const auto c2 = canonicalize_method_and_flags(c1.method, c1.use_fixed_sz,
                                                      c1.use_gpu, c1.use_mpi);
        INFO("input=" << static_cast<int>(m));
        REQUIRE(c1.method       == c2.method);
        REQUIRE(c1.use_fixed_sz == c2.use_fixed_sz);
        REQUIRE(c1.use_gpu      == c2.use_gpu);
        REQUIRE(c1.use_mpi      == c2.use_mpi);
    }
}

TEST_CASE("legacy_method_for_dispatch: round-trip with canonicalize", "[method][canonicalize][dispatch]") {
    // legacy_method_for_dispatch() is the inverse half: given a canonical
    // (base, use_gpu) it returns the legacy `_GPU` enum value the existing
    // dispatcher branches on. For every legacy `_GPU` enum we expect:
    //
    //     legacy_method_for_dispatch(canonicalize(_GPU).method, /*use_gpu=*/true)
    //         == _GPU
    //
    // (modulo enum values that don't have a dedicated GPU dispatch tag.)
    const auto check = [](M legacy_gpu) {
        const auto c = canonicalize_method_and_flags(legacy_gpu, false, false, false);
        REQUIRE(c.use_gpu == true);
        REQUIRE(legacy_method_for_dispatch(c.method, /*use_gpu=*/true) == legacy_gpu);
    };
    check(M::LANCZOS_GPU);
    check(M::BLOCK_LANCZOS_GPU);
    check(M::DAVIDSON_GPU);
    check(M::LOBPCG_GPU);
    check(M::KRYLOV_SCHUR_GPU);
    check(M::BLOCK_KRYLOV_SCHUR_GPU);
    check(M::mTPQ_GPU);
    check(M::cTPQ_GPU);
    check(M::FTLM_GPU);
    check(M::FULL_GPU);

    // use_gpu=false is the identity.
    REQUIRE(legacy_method_for_dispatch(M::LANCZOS,        false) == M::LANCZOS);
    REQUIRE(legacy_method_for_dispatch(M::FULL,           false) == M::FULL);
    REQUIRE(legacy_method_for_dispatch(M::SCALAPACK,      false) == M::SCALAPACK);
    REQUIRE(legacy_method_for_dispatch(M::ARPACK_ADVANCED,false) == M::ARPACK_ADVANCED);

    // Solvers without a `_GPU` enum value pass through unchanged even when
    // use_gpu=true is requested. (Caller is expected to either fall back to
    // CPU or error out elsewhere.)
    REQUIRE(legacy_method_for_dispatch(M::LANCZOS_SELECTIVE, true) == M::LANCZOS_SELECTIVE);
    REQUIRE(legacy_method_for_dispatch(M::CHEBYSHEV_FILTERED, true) == M::CHEBYSHEV_FILTERED);
    REQUIRE(legacy_method_for_dispatch(M::OSS,                true) == M::OSS);
    REQUIRE(legacy_method_for_dispatch(M::SCALAPACK,          true) == M::SCALAPACK);
    REQUIRE(legacy_method_for_dispatch(M::LTLM,               true) == M::LTLM);
    REQUIRE(legacy_method_for_dispatch(M::HYBRID,             true) == M::HYBRID);
    REQUIRE(legacy_method_for_dispatch(M::ARPACK_ADVANCED,    true) == M::ARPACK_ADVANCED);
}

TEST_CASE("is_deprecated_axis_method: legacy enums are flagged", "[method][canonicalize][deprecated]") {
    // Quick smoke test of the deprecation-detection predicates that the
    // canonicalizer uses internally. New code should never need to test
    // these directly, but they are part of the public ed:: surface.
    REQUIRE(ed::is_deprecated_gpu_method(M::LANCZOS_GPU));
    REQUIRE(ed::is_deprecated_gpu_method(M::FTLM_GPU));
    REQUIRE(ed::is_deprecated_gpu_method(M::mTPQ_CUDA));
    REQUIRE(ed::is_deprecated_mpi_method(M::mTPQ_MPI));
    REQUIRE(ed::is_deprecated_fixed_sz_method(M::LANCZOS_GPU_FIXED_SZ));
    REQUIRE(ed::is_deprecated_axis_method(M::FTLM_GPU_FIXED_SZ));

    // Canonical forms are never flagged as deprecated.
    REQUIRE_FALSE(ed::is_deprecated_axis_method(M::LANCZOS));
    REQUIRE_FALSE(ed::is_deprecated_axis_method(M::FTLM));
    REQUIRE_FALSE(ed::is_deprecated_axis_method(M::mTPQ));
    REQUIRE_FALSE(ed::is_deprecated_axis_method(M::SCALAPACK));
    REQUIRE_FALSE(ed::is_deprecated_axis_method(M::SCALAPACK_MIXED));
}

#pragma GCC diagnostic pop

// =============================================================================
// Phase 7.1 — 5th orthogonal axis: symmetry projection
// =============================================================================
//
// `use_symmetry` is a flag-only axis on EDParameters: there are no
// corresponding `*_SYMMETRY` enum values in DiagonalizationMethod (the
// pre-Phase-7.1 surface embedded the choice in the *function name*, not
// the enum). So the canonicalize_method_and_flags() helper does NOT
// canonicalize anything for this axis -- we only need to verify that:
//   1. EDParameters::use_symmetry defaults to false,
//   2. it is mirrored onto EDConfig::SystemConfig::use_symmetry by the
//      adapter,
//   3. the EDConfig builder's useSymmetry() setter sets BOTH
//      system.use_symmetry AND workflow.run_symm_auto so the legacy
//      ed_main.cpp dispatch keeps firing run_streaming_symmetry_workflow,
//   4. setting use_symmetry does not perturb the canonical (method,
//      use_fixed_sz, use_gpu, use_mpi) tuple.
// =============================================================================

#include <ed/core/ed_config.h>
#include <ed/core/ed_config_adapter.h>
#include <ed/core/ed_parameters.h>
#include <string>
#include <vector>

TEST_CASE("EDParameters::use_symmetry default and round-trip", "[method][canonicalize][symmetry]") {
    EDParameters params;
    REQUIRE(params.use_symmetry == false);

    params.use_symmetry = true;
    REQUIRE(params.use_symmetry == true);

    // It is independent of the other axes -- you can mix-and-match.
    params.use_fixed_sz = true;
    params.use_gpu = true;
    REQUIRE(params.use_symmetry == true);
    REQUIRE(params.use_fixed_sz == true);
    REQUIRE(params.use_gpu == true);
}

TEST_CASE("EDConfig::useSymmetry() sets both canonical and legacy flags",
          "[method][canonicalize][symmetry][config]") {
    // Phase 7.1 keeps WorkflowConfig::run_symm_auto as a [legacy] mirror
    // of SystemConfig::use_symmetry so the existing ed_main.cpp
    // dispatch path (which fires run_streaming_symmetry_workflow when
    // run_symm_auto is set) keeps working without changes.
    EDConfig cfg(M::LANCZOS);
    REQUIRE(cfg.system.use_symmetry == false);
    REQUIRE(cfg.workflow.run_symm_auto == false);

    cfg.useSymmetry(true);
    REQUIRE(cfg.system.use_symmetry == true);
    REQUIRE(cfg.workflow.run_symm_auto == true);

    // The symm() / symmAuto() aliases mirror onto both fields too.
    EDConfig cfg2(M::LANCZOS);
    cfg2.symm(true);
    REQUIRE(cfg2.system.use_symmetry == true);
    REQUIRE(cfg2.workflow.run_symm_auto == true);

    EDConfig cfg3(M::LANCZOS);
    cfg3.symmAuto(true);
    REQUIRE(cfg3.system.use_symmetry == true);
    REQUIRE(cfg3.workflow.run_symm_auto == true);
}

TEST_CASE("ed_adapter mirrors EDParameters::use_symmetry through EDConfig",
          "[method][canonicalize][symmetry][adapter]") {
    EDConfig cfg(M::LANCZOS);
    cfg.system.use_symmetry = true;

    EDParameters params = ed_adapter::toEDParameters(cfg);
    REQUIRE(params.use_symmetry == true);

    // Reverse direction: legacy run_symm_auto-only configs (no
    // SystemConfig::use_symmetry) still set params.use_symmetry so
    // the dispatcher's 5-axis routing fires.
    EDConfig cfg_legacy(M::LANCZOS);
    cfg_legacy.workflow.run_symm_auto = true;
    EDParameters params_legacy = ed_adapter::toEDParameters(cfg_legacy);
    REQUIRE(params_legacy.use_symmetry == true);

    // Round-trip back: EDConfig <- EDParameters mirrors the flag onto
    // *both* the canonical SystemConfig field AND the legacy workflow
    // mirror so ed_main.cpp's dispatch keeps firing.
    EDParameters seed;
    seed.use_symmetry = true;
    EDConfig round_tripped = ed_adapter::fromEDParameters(seed, M::LANCZOS);
    REQUIRE(round_tripped.system.use_symmetry == true);
    REQUIRE(round_tripped.workflow.run_symm_auto == true);
}

TEST_CASE("use_symmetry does not interact with method canonicalization",
          "[method][canonicalize][symmetry]") {
    // The symmetry axis lives on EDParameters and is routed by
    // ed_dispatch_symmetry.h, not by canonicalize_method_and_flags().
    // Verify that the canonical (method, fz, gpu, mpi) tuple is
    // unchanged regardless of use_symmetry.
    const auto c = canonicalize_method_and_flags(
        M::LANCZOS, /*fz=*/true, /*gpu=*/true, /*mpi=*/false);
    REQUIRE(c.method       == M::LANCZOS);
    REQUIRE(c.use_fixed_sz == true);
    REQUIRE(c.use_gpu      == true);
    REQUIRE(c.use_mpi      == false);
    // (CanonicalMethod intentionally has no use_symmetry field.)
}

// Phase 8 #5: --scalapack-block-size CLI flag implicitly disables the
// new auto-block-size heuristic so explicit values are honoured by the
// solver. The default values flow through the EDConfig <-> EDParameters
// adapter unchanged.
TEST_CASE("scalapack_block_size_auto defaults and CLI override",
          "[method][canonicalize][scalapack][phase8]") {
    // Default: auto on, legacy block size = 64 (only consulted when
    // auto is off).
    EDParameters params;
    REQUIRE(params.scalapack_block_size_auto == true);
    REQUIRE(params.scalapack_block_size == 64);

    // EDConfig defaults mirror this.
    EDConfig cfg(M::SCALAPACK);
    REQUIRE(cfg.diag.scalapack_block_size_auto == true);
    REQUIRE(cfg.diag.scalapack_block_size == 64);

    // Adapter: EDConfig -> EDParameters preserves both fields.
    cfg.diag.scalapack_block_size = 128;
    cfg.diag.scalapack_block_size_auto = false;
    EDParameters params_from_cfg = ed_adapter::toEDParameters(cfg);
    REQUIRE(params_from_cfg.scalapack_block_size_auto == false);
    REQUIRE(params_from_cfg.scalapack_block_size == 128);

    // Adapter: EDParameters -> EDConfig round-trips.
    EDParameters seed;
    seed.scalapack_block_size_auto = false;
    seed.scalapack_block_size = 32;
    EDConfig round_tripped = ed_adapter::fromEDParameters(seed, M::SCALAPACK);
    REQUIRE(round_tripped.diag.scalapack_block_size_auto == false);
    REQUIRE(round_tripped.diag.scalapack_block_size == 32);
}

TEST_CASE("EDConfig::fromCommandLine wires --scalapack-block-size",
          "[method][canonicalize][scalapack][phase8][cli]") {
    {
        // No CLI flag -> auto stays on, block size keeps default.
        std::vector<std::string> args = {"prog"};
        std::vector<char*> argv;
        for (auto& s : args) argv.push_back(s.data());
        EDConfig cfg = EDConfig::fromCommandLine(static_cast<uint64_t>(argv.size()),
                                                  argv.data());
        REQUIRE(cfg.diag.scalapack_block_size_auto == true);
        REQUIRE(cfg.diag.scalapack_block_size == 64);
    }
    {
        // --scalapack-block-size=128 disables auto and sets the value.
        std::vector<std::string> args = {"prog", "--scalapack-block-size=128"};
        std::vector<char*> argv;
        for (auto& s : args) argv.push_back(s.data());
        EDConfig cfg = EDConfig::fromCommandLine(static_cast<uint64_t>(argv.size()),
                                                  argv.data());
        REQUIRE(cfg.diag.scalapack_block_size_auto == false);
        REQUIRE(cfg.diag.scalapack_block_size == 128);
    }
    {
        // Explicit --no-scalapack-block-size-auto without setting
        // block size: auto off, default block size kept.
        std::vector<std::string> args = {"prog", "--no-scalapack-block-size-auto"};
        std::vector<char*> argv;
        for (auto& s : args) argv.push_back(s.data());
        EDConfig cfg = EDConfig::fromCommandLine(static_cast<uint64_t>(argv.size()),
                                                  argv.data());
        REQUIRE(cfg.diag.scalapack_block_size_auto == false);
        REQUIRE(cfg.diag.scalapack_block_size == 64);
    }
}
