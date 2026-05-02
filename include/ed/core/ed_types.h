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
    // Phase 7: the canonical surface is "algorithmic_solver" only. Device
    // (CPU vs GPU) and parallelism (single-process vs MPI) are now flags
    // on EDParameters (use_gpu, use_mpi), not enum values, mirroring the
    // existing use_fixed_sz convention. The `_GPU`, `_CUDA`, `_MPI`, and
    // `_FIXED_SZ` enum variants below are kept for backwards compatibility
    // (HDF5 metadata, CLI strings, pre-Phase-7 Python code) and are
    // canonicalized at the dispatcher entry point via
    // ed::canonicalize_method_and_flags(). New code should use the base
    // method + flag.
    // ------------------------------------------------------------------------

    // ===== CPU iterative / dense / thermal / ARPACK base methods =====
    LANCZOS,                  // Standard Lanczos algorithm
    LANCZOS_SELECTIVE,        // Lanczos with selective reorthogonalization
    LANCZOS_NO_ORTHO,         // Lanczos without reorthogonalization
    BLOCK_LANCZOS,            // Block Lanczos
    CHEBYSHEV_FILTERED,       // Chebyshev filtered Lanczos for spectral slicing
    SHIFT_INVERT,             // Shift-invert Lanczos
    SHIFT_INVERT_ROBUST,      // Robust shift-invert Lanczos
    DAVIDSON,                 // Davidson method
    BICG,                     // Biconjugate gradient
    LOBPCG,                   // Locally optimal block preconditioned CG
    KRYLOV_SCHUR,             // Krylov-Schur algorithm
    BLOCK_KRYLOV_SCHUR,       // Block Krylov-Schur
    IMPLICIT_RESTART_LANCZOS, // Implicitly restarted Lanczos
    THICK_RESTART_LANCZOS,    // Thick-restart Lanczos with locking
    FULL,                     // Full diagonalization (LAPACK)
    OSS,                      // Optimal spectrum solver

    // Distributed dense kernels (kept as separate solvers because they
    // route through ScaLAPACK PDSYEVR / mixed-precision refinement,
    // which is a *different* dense LAPACK call than FULL — not just
    // "FULL with use_mpi=true"). Implicitly require MPI.
    SCALAPACK,                // ScaLAPACK distributed diagonalization
    SCALAPACK_MIXED,          // ScaLAPACK with mixed precision (single + refinement)

    // Thermal methods (base CPU forms)
    mTPQ,                     // Microcanonical TPQ
    mTPQ_MPI                  // [DEPRECATED Phase 7] use mTPQ + use_mpi=true
        [[deprecated("Use mTPQ with EDParameters::use_mpi=true instead")]],
    cTPQ,                     // Canonical TPQ
    mTPQ_CUDA                 // [DEPRECATED Phase 7] use mTPQ + use_gpu=true
        [[deprecated("Use mTPQ with EDParameters::use_gpu=true instead (alias of mTPQ_GPU)")]],
    FTLM,                     // Finite Temperature Lanczos Method
    LTLM,                     // Low-Temperature Lanczos Method
    HYBRID,                   // Hybrid (LTLM + FTLM with automatic crossover)
    KPM_DOS,                  // Kernel Polynomial Method density-of-states +
                              // thermodynamics (Chebyshev moments + Hutchinson
                              // trace; recommended for N >= 13).

    // ARPACK methods (algorithmic variants — *not* device variants)
    ARPACK_SM,                // ARPACK smallest magnitude
    ARPACK_LM,                // ARPACK largest magnitude
    ARPACK_SHIFT_INVERT,      // ARPACK in shift-invert mode
    ARPACK_ADVANCED,          // ARPACK advanced multi-attempt strategy

    // ===== Phase 7 deprecated GPU-axis variants =====
    // All collapse to base method + EDParameters::use_gpu=true via
    // ed::canonicalize_method_and_flags(). Kept in the enum for
    // backwards-compatible HDF5 metadata / CLI strings.
    LANCZOS_GPU
        [[deprecated("Use LANCZOS with EDParameters::use_gpu=true instead")]],
    BLOCK_LANCZOS_GPU
        [[deprecated("Use BLOCK_LANCZOS with EDParameters::use_gpu=true instead")]],
    DAVIDSON_GPU
        [[deprecated("Use DAVIDSON with EDParameters::use_gpu=true instead")]],
    LOBPCG_GPU
        [[deprecated("Use LOBPCG with EDParameters::use_gpu=true instead "
                     "(LOBPCG_GPU currently redirects to DAVIDSON_GPU internally)")]],
    KRYLOV_SCHUR_GPU
        [[deprecated("Use KRYLOV_SCHUR with EDParameters::use_gpu=true instead")]],
    BLOCK_KRYLOV_SCHUR_GPU
        [[deprecated("Use BLOCK_KRYLOV_SCHUR with EDParameters::use_gpu=true instead")]],
    mTPQ_GPU
        [[deprecated("Use mTPQ with EDParameters::use_gpu=true instead")]],
    cTPQ_GPU
        [[deprecated("Use cTPQ with EDParameters::use_gpu=true instead")]],
    FTLM_GPU
        [[deprecated("Use FTLM with EDParameters::use_gpu=true instead")]],
    FULL_GPU
        [[deprecated("Use FULL with EDParameters::use_gpu=true instead")]],

    // ===== Phase 7 deprecated combined GPU + FIXED_SZ variants =====
    // Collapse to base + use_gpu=true + use_fixed_sz=true.
    LANCZOS_GPU_FIXED_SZ
        [[deprecated("Use LANCZOS with use_gpu=true and use_fixed_sz=true instead")]],
    BLOCK_LANCZOS_GPU_FIXED_SZ
        [[deprecated("Use BLOCK_LANCZOS with use_gpu=true and use_fixed_sz=true instead")]],
    FTLM_GPU_FIXED_SZ
        [[deprecated("Use FTLM with use_gpu=true and use_fixed_sz=true instead")]],
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
