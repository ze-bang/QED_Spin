#pragma once

// =============================================================================
// ed/core/ed_types.h
//
// Single source of truth for the small, dependency-free types that need to be
// shared by every layer of the ED stack: the headers, the apps, the tests,
// and (eventually) the Python bindings.
//
// Historically `enum class DiagonalizationMethod` was defined in
// include/ed/core/ed_wrapper.h *and* duplicated verbatim in
// src/core/ed_config.cpp under a "MUST stay in sync with the one in
// ed_wrapper.h" comment. That comment is exactly the kind of latent bug
// modernization is meant to prevent. The enum now lives here, and is
// included by both call sites.
//
// P0.14 / audit Q6.
// =============================================================================

namespace ed {

/**
 * @brief Available diagonalization methods.
 *
 * IMPORTANT: the order of these values is part of the public ABI of any
 * downstream consumer that stores them as integers (e.g. HDF5 attributes,
 * pybind11 enum bindings). Append new values at the end of their group;
 * never reorder existing ones.
 */
enum class DiagonalizationMethod {
    // ------------------------------------------------------------------------
    // Minimalist-architecture rev (May 2026): keep only the canonical
    // Krylov / thermal / dense kernels we'll continue to support. Device
    // (CPU vs GPU) and parallelism (single-process vs MPI) are flags on
    // EDParameters (use_gpu, use_mpi), not enum values, mirroring the
    // use_fixed_sz convention. The `_GPU`, `_CUDA`, `_MPI`, and
    // `_FIXED_SZ` enum variants that used to live here were retired in
    // the same rev together with the legacy LANCZOS_SELECTIVE /
    // LANCZOS_NO_ORTHO / CHEBYSHEV_FILTERED / SHIFT_INVERT /
    // SHIFT_INVERT_ROBUST / DAVIDSON / BICG / LOBPCG /
    // BLOCK_KRYLOV_SCHUR / IMPLICIT_RESTART_LANCZOS /
    // THICK_RESTART_LANCZOS / OSS / SCALAPACK / SCALAPACK_MIXED /
    // ARPACK_* / HYBRID solvers. See docs/architecture/STRUCTURAL_AUDIT.md
    // Part IV.
    // ------------------------------------------------------------------------

    // ===== Ground-state / spectrum =====
    LANCZOS,                  // Standard Lanczos
    BLOCK_LANCZOS,            // Block Lanczos
    KRYLOV_SCHUR,             // Krylov-Schur (thick-restart Lanczos)
    BLOCK_KRYLOV_SCHUR,       // Block Krylov-Schur (thick-restart block Lanczos)
    FULL,                     // Full diagonalization (LAPACK D&C)

    // ===== Thermal =====
    mTPQ,                     // Microcanonical TPQ
    cTPQ,                     // Canonical TPQ
    FTLM,                     // Finite Temperature Lanczos Method
    LTLM,                     // Low-Temperature Lanczos Method
    KPM_DOS,                  // Kernel Polynomial Method DOS + thermo
    OFTLM,                    // Orthogonalized Finite-Temperature Lanczos Method
};

}  // namespace ed

// -----------------------------------------------------------------------------
// Compatibility shim: ed_wrapper.h, src/core/ed_config.cpp, and ~all of the
// existing apps refer to ::DiagonalizationMethod (no namespace). Until we do
// the full namespace migration in Phase 1, alias the new ed::DiagonalizationMethod
// up to the global scope so existing call sites keep compiling.
// Removing this 'using' is a Phase-2 task (P2 CLI/lib refactor).
//
// NOTE: a previous version of include/ed/core/ed_config.h had
//   enum class DiagonalizationMethod;  // forward declaration
// at global scope. That forward declaration was replaced by an
// `#include <ed/core/ed_types.h>` in the same P0.14 commit, so the
// `using` below is the only place ::DiagonalizationMethod gets introduced.
// -----------------------------------------------------------------------------
using DiagonalizationMethod = ::ed::DiagonalizationMethod;
