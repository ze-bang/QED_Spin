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

/// True for any thermal method (TPQ / FTLM / LTLM / Hybrid).
constexpr bool is_thermal_method(DiagonalizationMethod m) noexcept {
    return is_tpq_method(m) || is_ftlm_method(m) ||
           is_ltlm_method(m) || is_hybrid_method(m);
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

/// Collapses the deprecated _FIXED_SZ variants onto their base method;
/// returns the input unchanged otherwise. Callers should also set
/// EDParameters::use_fixed_sz / EDConfig::system::use_fixed_sz to true
/// when the input was deprecated.
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
    using ed::normalize_method;
    using ed::requires_ground_state_sector;
    using ed::requires_sector_combination;
}  // namespace ed_internal
