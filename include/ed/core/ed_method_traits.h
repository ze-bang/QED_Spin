#pragma once

// =============================================================================
// ed/core/ed_method_traits.h
//
// Single source of truth for "is this method a TPQ?", "is this a GPU
// method?", etc. Historically these predicates were duplicated in
// ed_wrapper.h (under namespace ed_internal::), workflows.cpp, and a few
// other spots, with each copy slowly diverging — most notably
// is_gpu_method() in workflows.cpp omitted KRYLOV_SCHUR_GPU and
// BLOCK_KRYLOV_SCHUR_GPU, so the disk-streaming and chunked-symmetry
// fallback warnings silently let those methods through. (D-4 in the
// modernization audit.)
//
// Keeping the helpers here, in a header that depends only on
// <ed/core/ed_types.h>, lets every translation unit pull in the
// classification without dragging in the entire ed_wrapper.h monolith.
// =============================================================================

#include <ed/core/ed_types.h>

namespace ed {

// ----------------------------------------------------------------------------
// Thermal-method predicates
// ----------------------------------------------------------------------------

/// True for any TPQ flavour (microcanonical / canonical, CPU / CUDA / GPU / MPI).
constexpr bool is_tpq_method(DiagonalizationMethod m) noexcept {
    return m == DiagonalizationMethod::mTPQ        ||
           m == DiagonalizationMethod::mTPQ_MPI    ||
           m == DiagonalizationMethod::mTPQ_CUDA   ||
           m == DiagonalizationMethod::cTPQ        ||
           m == DiagonalizationMethod::mTPQ_GPU    ||
           m == DiagonalizationMethod::cTPQ_GPU;
}

/// True for any FTLM flavour.
constexpr bool is_ftlm_method(DiagonalizationMethod m) noexcept {
    // Note: the deprecated _FIXED_SZ variants stay in the list because
    // legacy configs and pre-existing HDF5 metadata still reference them.
    // normalize_method() collapses them onto FTLM_GPU before dispatch.
    #pragma GCC diagnostic push
    #pragma GCC diagnostic ignored "-Wdeprecated-declarations"
    return m == DiagonalizationMethod::FTLM     ||
           m == DiagonalizationMethod::FTLM_GPU ||
           m == DiagonalizationMethod::FTLM_GPU_FIXED_SZ;
    #pragma GCC diagnostic pop
}

/// True for LTLM.
constexpr bool is_ltlm_method(DiagonalizationMethod m) noexcept {
    return m == DiagonalizationMethod::LTLM;
}

/// True for the hybrid LTLM/FTLM dispatcher.
constexpr bool is_hybrid_method(DiagonalizationMethod m) noexcept {
    return m == DiagonalizationMethod::HYBRID;
}

/// True for the KPM-DOS thermodynamics solver.
constexpr bool is_kpm_dos_method(DiagonalizationMethod m) noexcept {
    return m == DiagonalizationMethod::KPM_DOS;
}

/// True for any thermal method (TPQ / FTLM / LTLM / Hybrid / KPM-DOS).
constexpr bool is_thermal_method(DiagonalizationMethod m) noexcept {
    return is_tpq_method(m) || is_ftlm_method(m) ||
           is_ltlm_method(m) || is_hybrid_method(m) ||
           is_kpm_dos_method(m);
}

// ----------------------------------------------------------------------------
// GPU predicates
// ----------------------------------------------------------------------------

/// True for any GPU-backed method.
constexpr bool is_gpu_method(DiagonalizationMethod m) noexcept {
    #pragma GCC diagnostic push
    #pragma GCC diagnostic ignored "-Wdeprecated-declarations"
    return m == DiagonalizationMethod::LANCZOS_GPU                ||
           m == DiagonalizationMethod::LANCZOS_GPU_FIXED_SZ       ||
           m == DiagonalizationMethod::BLOCK_LANCZOS_GPU          ||
           m == DiagonalizationMethod::BLOCK_LANCZOS_GPU_FIXED_SZ ||
           m == DiagonalizationMethod::DAVIDSON_GPU               ||
           m == DiagonalizationMethod::LOBPCG_GPU                 ||
           m == DiagonalizationMethod::KRYLOV_SCHUR_GPU           ||
           m == DiagonalizationMethod::BLOCK_KRYLOV_SCHUR_GPU     ||
           m == DiagonalizationMethod::mTPQ_GPU                   ||
           m == DiagonalizationMethod::cTPQ_GPU                   ||
           m == DiagonalizationMethod::FTLM_GPU                   ||
           m == DiagonalizationMethod::FTLM_GPU_FIXED_SZ          ||
           m == DiagonalizationMethod::FULL_GPU;
    #pragma GCC diagnostic pop
}

/// True for the deprecated _FIXED_SZ enum variants. New code should use
/// the base method plus use_fixed_sz=true.
constexpr bool is_deprecated_fixed_sz_method(DiagonalizationMethod m) noexcept {
    #pragma GCC diagnostic push
    #pragma GCC diagnostic ignored "-Wdeprecated-declarations"
    return m == DiagonalizationMethod::LANCZOS_GPU_FIXED_SZ       ||
           m == DiagonalizationMethod::BLOCK_LANCZOS_GPU_FIXED_SZ ||
           m == DiagonalizationMethod::FTLM_GPU_FIXED_SZ;
    #pragma GCC diagnostic pop
}

// ----------------------------------------------------------------------------
// Phase 7: orthogonal device / parallelism axes
// ----------------------------------------------------------------------------
//
// The matrix we want is SOLVER × use_fixed_sz × use_gpu × use_mpi. The
// `_GPU` / `_CUDA` / `_MPI` enum variants are now legacy aliases that
// canonicalize_method_and_flags() collapses onto the base solver +
// flag. Keep them in the enum (and in is_gpu_method() / is_tpq_method())
// because (a) they appear in pre-existing HDF5 metadata, and (b) the
// CLI / Python still accept the strings.

/// True for an enum variant that encodes "GPU" in its name. New code
/// should pass the base method plus EDParameters::use_gpu=true instead.
/// (LOBPCG_GPU is included even though it currently redirects to
/// DAVIDSON_GPU; SCALAPACK is *not* — that's a distinct distributed
/// dense kernel, not a GPU backend.)
constexpr bool is_deprecated_gpu_method(DiagonalizationMethod m) noexcept {
    #pragma GCC diagnostic push
    #pragma GCC diagnostic ignored "-Wdeprecated-declarations"
    return m == DiagonalizationMethod::LANCZOS_GPU                ||
           m == DiagonalizationMethod::LANCZOS_GPU_FIXED_SZ       ||
           m == DiagonalizationMethod::BLOCK_LANCZOS_GPU          ||
           m == DiagonalizationMethod::BLOCK_LANCZOS_GPU_FIXED_SZ ||
           m == DiagonalizationMethod::DAVIDSON_GPU               ||
           m == DiagonalizationMethod::LOBPCG_GPU                 ||
           m == DiagonalizationMethod::KRYLOV_SCHUR_GPU           ||
           m == DiagonalizationMethod::BLOCK_KRYLOV_SCHUR_GPU     ||
           m == DiagonalizationMethod::mTPQ_GPU                   ||
           m == DiagonalizationMethod::mTPQ_CUDA                  ||
           m == DiagonalizationMethod::cTPQ_GPU                   ||
           m == DiagonalizationMethod::FTLM_GPU                   ||
           m == DiagonalizationMethod::FTLM_GPU_FIXED_SZ          ||
           m == DiagonalizationMethod::FULL_GPU;
    #pragma GCC diagnostic pop
}

/// True for an enum variant that encodes "MPI" in its name (and is *not*
/// a distributed dense kernel like SCALAPACK). New code should pass the
/// base method + EDParameters::use_mpi=true.
constexpr bool is_deprecated_mpi_method(DiagonalizationMethod m) noexcept {
    return m == DiagonalizationMethod::mTPQ_MPI;
}

/// True for any enum variant whose device / parallelism axis can be
/// expressed as flags on EDParameters instead. Includes the
/// is_deprecated_fixed_sz_method() / is_deprecated_gpu_method() /
/// is_deprecated_mpi_method() unions.
constexpr bool is_deprecated_axis_method(DiagonalizationMethod m) noexcept {
    return is_deprecated_fixed_sz_method(m) ||
           is_deprecated_gpu_method(m)      ||
           is_deprecated_mpi_method(m);
}

/// Result of canonicalize_method_and_flags(). Holds the base solver +
/// the orthogonal device / parallelism / basis flags.
struct CanonicalMethod {
    DiagonalizationMethod method;
    bool use_fixed_sz;
    bool use_gpu;
    bool use_mpi;
};

/// The inverse of canonicalize_method_and_flags(): given a base method
/// and `use_gpu` flag, return the legacy `_GPU` enum value the existing
/// dispatcher branches on internally. Phase 7 keeps the legacy enum
/// values as an *internal* dispatch tag so we don't have to rewrite the
/// CPU vs GPU branch structure -- the public API stays orthogonal
/// (base + flag), the implementation continues to switch on the
/// expanded enum for now.
///
/// Falls through to ``base`` when no GPU variant exists or when
/// ``use_gpu`` is false. The deprecated `_GPU` values are referenced
/// here intentionally; the surrounding pragma block silences the
/// deprecation warning at this single helper instead of every
/// dispatch site.
constexpr DiagonalizationMethod legacy_method_for_dispatch(
    DiagonalizationMethod base, bool use_gpu) noexcept {
    #pragma GCC diagnostic push
    #pragma GCC diagnostic ignored "-Wdeprecated-declarations"
    if (!use_gpu) return base;
    switch (base) {
        case DiagonalizationMethod::LANCZOS:
            return DiagonalizationMethod::LANCZOS_GPU;
        case DiagonalizationMethod::BLOCK_LANCZOS:
            return DiagonalizationMethod::BLOCK_LANCZOS_GPU;
        case DiagonalizationMethod::DAVIDSON:
            return DiagonalizationMethod::DAVIDSON_GPU;
        case DiagonalizationMethod::LOBPCG:
            return DiagonalizationMethod::LOBPCG_GPU;
        case DiagonalizationMethod::KRYLOV_SCHUR:
            return DiagonalizationMethod::KRYLOV_SCHUR_GPU;
        case DiagonalizationMethod::BLOCK_KRYLOV_SCHUR:
            return DiagonalizationMethod::BLOCK_KRYLOV_SCHUR_GPU;
        case DiagonalizationMethod::mTPQ:
            return DiagonalizationMethod::mTPQ_GPU;
        case DiagonalizationMethod::cTPQ:
            return DiagonalizationMethod::cTPQ_GPU;
        case DiagonalizationMethod::FTLM:
            return DiagonalizationMethod::FTLM_GPU;
        case DiagonalizationMethod::FULL:
            return DiagonalizationMethod::FULL_GPU;
        default:
            // No GPU variant for this solver (e.g. LANCZOS_SELECTIVE,
            // CHEBYSHEV_FILTERED, SHIFT_INVERT, BICG, IRL, TRL,
            // ARPACK_*, OSS, LTLM, HYBRID, SCALAPACK*). Caller will see
            // the unchanged base method and is expected to fall back
            // to CPU or to error out.
            return base;
    }
    #pragma GCC diagnostic pop
}

/// Collapses any deprecated `_FIXED_SZ` / `_GPU` / `_CUDA` / `_MPI` enum
/// variant onto the base solver + flag triple, and **OR**s the result
/// into the caller's existing flags. (The OR semantics let a caller
/// pass `LANCZOS_GPU_FIXED_SZ` together with an already-true
/// `use_fixed_sz` without losing information.)
///
/// SCALAPACK and SCALAPACK_MIXED are intentionally *not* canonicalized
/// to FULL + use_mpi: they are a different solver kernel.
constexpr CanonicalMethod canonicalize_method_and_flags(
    DiagonalizationMethod m,
    bool use_fixed_sz_in,
    bool use_gpu_in,
    bool use_mpi_in) noexcept {
    #pragma GCC diagnostic push
    #pragma GCC diagnostic ignored "-Wdeprecated-declarations"
    DiagonalizationMethod base = m;
    bool use_fixed_sz = use_fixed_sz_in;
    bool use_gpu      = use_gpu_in;
    bool use_mpi      = use_mpi_in;
    switch (m) {
        // ---- _FIXED_SZ family ----
        case DiagonalizationMethod::LANCZOS_GPU_FIXED_SZ:
            base = DiagonalizationMethod::LANCZOS;
            use_gpu = true;
            use_fixed_sz = true;
            break;
        case DiagonalizationMethod::BLOCK_LANCZOS_GPU_FIXED_SZ:
            base = DiagonalizationMethod::BLOCK_LANCZOS;
            use_gpu = true;
            use_fixed_sz = true;
            break;
        case DiagonalizationMethod::FTLM_GPU_FIXED_SZ:
            base = DiagonalizationMethod::FTLM;
            use_gpu = true;
            use_fixed_sz = true;
            break;
        // ---- _GPU family ----
        case DiagonalizationMethod::LANCZOS_GPU:
            base = DiagonalizationMethod::LANCZOS;
            use_gpu = true;
            break;
        case DiagonalizationMethod::BLOCK_LANCZOS_GPU:
            base = DiagonalizationMethod::BLOCK_LANCZOS;
            use_gpu = true;
            break;
        case DiagonalizationMethod::DAVIDSON_GPU:
            base = DiagonalizationMethod::DAVIDSON;
            use_gpu = true;
            break;
        case DiagonalizationMethod::LOBPCG_GPU:
            base = DiagonalizationMethod::LOBPCG;
            use_gpu = true;
            break;
        case DiagonalizationMethod::KRYLOV_SCHUR_GPU:
            base = DiagonalizationMethod::KRYLOV_SCHUR;
            use_gpu = true;
            break;
        case DiagonalizationMethod::BLOCK_KRYLOV_SCHUR_GPU:
            base = DiagonalizationMethod::BLOCK_KRYLOV_SCHUR;
            use_gpu = true;
            break;
        case DiagonalizationMethod::mTPQ_GPU:
        case DiagonalizationMethod::mTPQ_CUDA:
            base = DiagonalizationMethod::mTPQ;
            use_gpu = true;
            break;
        case DiagonalizationMethod::cTPQ_GPU:
            base = DiagonalizationMethod::cTPQ;
            use_gpu = true;
            break;
        case DiagonalizationMethod::FTLM_GPU:
            base = DiagonalizationMethod::FTLM;
            use_gpu = true;
            break;
        case DiagonalizationMethod::FULL_GPU:
            base = DiagonalizationMethod::FULL;
            use_gpu = true;
            break;
        // ---- _MPI family ----
        case DiagonalizationMethod::mTPQ_MPI:
            base = DiagonalizationMethod::mTPQ;
            use_mpi = true;
            break;
        // ---- distributed dense (kept as separate kernels) ----
        case DiagonalizationMethod::SCALAPACK:
        case DiagonalizationMethod::SCALAPACK_MIXED:
            // SCALAPACK is its own solver; it is *implicitly* MPI-backed
            // but uses a different LAPACK call than FULL, so we don't
            // collapse it. Mark use_mpi=true so introspection is honest.
            use_mpi = true;
            break;
        default:
            break;
    }
    return CanonicalMethod{base, use_fixed_sz, use_gpu, use_mpi};
    #pragma GCC diagnostic pop
}

/// Collapses the deprecated _FIXED_SZ variants onto their base method;
/// returns the input unchanged otherwise. Callers should also set
/// EDParameters::use_fixed_sz / EDConfig::system::use_fixed_sz to true
/// when the input was deprecated.
///
/// NOTE: this preserves the legacy behaviour in which `_GPU` enum
/// variants are kept as-is (the GPU dispatcher historically branched
/// on them). New code should call ``canonicalize_method_and_flags``
/// instead, which returns the full (base, fixed_sz, gpu, mpi) tuple.
constexpr DiagonalizationMethod normalize_method(DiagonalizationMethod m) noexcept {
    #pragma GCC diagnostic push
    #pragma GCC diagnostic ignored "-Wdeprecated-declarations"
    switch (m) {
        case DiagonalizationMethod::LANCZOS_GPU_FIXED_SZ:
            return DiagonalizationMethod::LANCZOS_GPU;
        case DiagonalizationMethod::BLOCK_LANCZOS_GPU_FIXED_SZ:
            return DiagonalizationMethod::BLOCK_LANCZOS_GPU;
        case DiagonalizationMethod::FTLM_GPU_FIXED_SZ:
            return DiagonalizationMethod::FTLM_GPU;
        default:
            return m;
    }
    #pragma GCC diagnostic pop
}

// ----------------------------------------------------------------------------
// Workflow capability predicates
// ----------------------------------------------------------------------------

/// True if the method needs to know which sector contains the global
/// ground state (currently only TPQ — its Z normalization depends on
/// shifting by the GS energy).
constexpr bool requires_ground_state_sector(DiagonalizationMethod m) noexcept {
    return is_tpq_method(m);
}

/// True if the method produces per-sector thermodynamic data that the
/// driver must combine (FTLM): each sector contributes a partition-
/// function summand and you have to add them with consistent ln D.
constexpr bool requires_sector_combination(DiagonalizationMethod m) noexcept {
    return is_ftlm_method(m);
}

}  // namespace ed

// -----------------------------------------------------------------------------
// Back-compat aliases under namespace ed_internal:: — the inline helpers in
// ed_wrapper.h used to live there. Keeping the names available avoids
// touching every call site while the migration finishes.
// -----------------------------------------------------------------------------
namespace ed_internal {
    using ed::is_tpq_method;
    using ed::is_ftlm_method;
    using ed::is_ltlm_method;
    using ed::is_hybrid_method;
    using ed::is_thermal_method;
    using ed::is_gpu_method;
    using ed::is_deprecated_fixed_sz_method;
    using ed::is_deprecated_gpu_method;
    using ed::is_deprecated_mpi_method;
    using ed::is_deprecated_axis_method;
    using ed::normalize_method;
    using ed::canonicalize_method_and_flags;
    using ed::CanonicalMethod;
    using ed::requires_ground_state_sector;
    using ed::requires_sector_combination;
}  // namespace ed_internal
