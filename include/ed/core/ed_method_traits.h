#pragma once

// =============================================================================
// ed/core/ed_method_traits.h
//
// Predicates / helpers for the trimmed-down `DiagonalizationMethod`. The
// legacy `*_GPU` / `*_MPI` / `*_FIXED_SZ` enum aliases were retired in the
// minimalist-architecture rev (May 2026): device and parallelism are now
// EDParameters flags. There is therefore no canonicalisation step any
// more -- `canonicalize_method_and_flags` is preserved as a pass-through
// for back-compat callers, but does nothing interesting.
// =============================================================================

#include <ed/core/ed_types.h>

namespace ed {

// ----------------------------------------------------------------------------
// Thermal-method predicates
// ----------------------------------------------------------------------------

constexpr bool is_tpq_method(DiagonalizationMethod m) noexcept {
    return m == DiagonalizationMethod::mTPQ ||
           m == DiagonalizationMethod::cTPQ;
}

constexpr bool is_ftlm_method(DiagonalizationMethod m) noexcept {
    return m == DiagonalizationMethod::FTLM;
}

constexpr bool is_ltlm_method(DiagonalizationMethod m) noexcept {
    return m == DiagonalizationMethod::LTLM;
}

constexpr bool is_kpm_dos_method(DiagonalizationMethod m) noexcept {
    return m == DiagonalizationMethod::KPM_DOS;
}

constexpr bool is_thermal_method(DiagonalizationMethod m) noexcept {
    return is_tpq_method(m) || is_ftlm_method(m) ||
           is_ltlm_method(m) || is_kpm_dos_method(m);
}

// ----------------------------------------------------------------------------
// Phase-7 device-axis helpers (now trivial post-retirement; kept as
// pass-throughs so back-compat callers keep compiling). All return false
// on the trimmed enum.
// ----------------------------------------------------------------------------

constexpr bool is_gpu_method(DiagonalizationMethod /*m*/) noexcept {
    return false;
}
constexpr bool is_deprecated_fixed_sz_method(DiagonalizationMethod /*m*/) noexcept {
    return false;
}
constexpr bool is_deprecated_gpu_method(DiagonalizationMethod /*m*/) noexcept {
    return false;
}
constexpr bool is_deprecated_mpi_method(DiagonalizationMethod /*m*/) noexcept {
    return false;
}
constexpr bool is_deprecated_axis_method(DiagonalizationMethod /*m*/) noexcept {
    return false;
}

struct CanonicalMethod {
    DiagonalizationMethod method;
    bool use_fixed_sz;
    bool use_gpu;
    bool use_mpi;
};

/// Pass-through. Kept so back-compat call sites keep compiling.
constexpr CanonicalMethod canonicalize_method_and_flags(
    DiagonalizationMethod m,
    bool use_fixed_sz_in,
    bool use_gpu_in,
    bool use_mpi_in) noexcept {
    return CanonicalMethod{m, use_fixed_sz_in, use_gpu_in, use_mpi_in};
}

/// Pass-through (no `_FIXED_SZ` aliases remain).
constexpr DiagonalizationMethod legacy_method_for_dispatch(
    DiagonalizationMethod base, bool /*use_gpu*/) noexcept {
    return base;
}

constexpr DiagonalizationMethod normalize_method(DiagonalizationMethod m) noexcept {
    return m;
}

// ----------------------------------------------------------------------------
// Workflow capability predicates
// ----------------------------------------------------------------------------

/// True if the method needs to know which sector contains the global
/// ground state (currently only TPQ -- its Z normalization depends on
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

namespace ed_internal {
    using ed::is_tpq_method;
    using ed::is_ftlm_method;
    using ed::is_ltlm_method;
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
