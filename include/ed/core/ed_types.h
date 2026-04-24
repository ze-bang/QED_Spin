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

    // Distributed/Parallel methods
    SCALAPACK,                // ScaLAPACK distributed diagonalization
    SCALAPACK_MIXED,          // ScaLAPACK with mixed precision (single + refinement)

    // Thermal methods
    mTPQ,                     // Microcanonical TPQ
    mTPQ_MPI,                 // MPI-parallel mTPQ
    cTPQ,                     // Canonical TPQ
    mTPQ_CUDA,                // CUDA microcanonical TPQ
    FTLM,                     // Finite Temperature Lanczos Method
    LTLM,                     // Low-Temperature Lanczos Method
    HYBRID,                   // Hybrid (LTLM + FTLM with automatic crossover)

    // ARPACK methods
    ARPACK_SM,                // ARPACK smallest magnitude
    ARPACK_LM,                // ARPACK largest magnitude
    ARPACK_SHIFT_INVERT,      // ARPACK in shift-invert mode
    ARPACK_ADVANCED,          // ARPACK advanced multi-attempt strategy

    // GPU methods
    LANCZOS_GPU,              // GPU-accelerated Lanczos
    BLOCK_LANCZOS_GPU,        // GPU-accelerated Block Lanczos
    DAVIDSON_GPU,             // GPU-accelerated Davidson
    LOBPCG_GPU,               // GPU-accelerated LOBPCG
    KRYLOV_SCHUR_GPU,         // GPU-accelerated Krylov-Schur
    BLOCK_KRYLOV_SCHUR_GPU,   // GPU-accelerated Block Krylov-Schur
    mTPQ_GPU,                 // GPU-accelerated microcanonical TPQ
    cTPQ_GPU,                 // GPU-accelerated canonical TPQ
    FTLM_GPU,                 // GPU-accelerated FTLM
    FULL_GPU,                 // GPU-accelerated full diag (cuSOLVER)

    // Deprecated: prefer the base method + --fixed-sz CLI flag instead.
    LANCZOS_GPU_FIXED_SZ
        [[deprecated("Use LANCZOS_GPU with --fixed-sz flag instead")]],
    BLOCK_LANCZOS_GPU_FIXED_SZ
        [[deprecated("Use BLOCK_LANCZOS_GPU with --fixed-sz flag instead")]],
    FTLM_GPU_FIXED_SZ
        [[deprecated("Use FTLM_GPU with --fixed-sz flag instead")]],
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
