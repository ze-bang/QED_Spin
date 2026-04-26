// =============================================================================
// python/quantum_ed/_bindings/dispatcher_bindings.h
//
// Phase 5 of the Python interface modernization (Apr 2026): expose the
// **single** high-level dispatcher and the directory- / streaming-symmetry
// drivers to Python so that one binding call unlocks the entire CPU iterative
// + dense + thermal + ARPACK + TPQ + per-sector GPU stack.
//
// `bind_dispatcher(m)` populates `quantum_ed._core` with:
//   * `DiagonalizationMethod` enum (every value the C++ enum carries)
//   * `EDParameters` mutable parameter bag
//   * `EDResults` immutable result envelope
//   * `exact_diagonalization_core(operator_, method, params)` for both
//     `Operator` and `FixedSzOperator` -- this single Python function
//     dispatches to ~30 CPU solver variants (LANCZOS family,
//     BLOCK_LANCZOS, KRYLOV_SCHUR, BLOCK_KRYLOV_SCHUR, DAVIDSON, LOBPCG,
//     CHEBYSHEV_FILTERED, SHIFT_INVERT[_ROBUST], IRL/TRL, BICG, ARPACK_*,
//     FULL/SCALAPACK, mTPQ/cTPQ, FTLM/LTLM/HYBRID).
//   * `exact_diagonalization_from_directory[_symmetrized]` and
//     `_fixed_sz_symmetrized` for directory-driven runs (also reaches
//     the GPU per-sector dispatch via `_GPU` methods).
//   * `exact_diagonalization_streaming_symmetry[_fixed_sz]` for the
//     in-memory streaming-symmetry path -- the canonical entry point for
//     symmetry-projected ED with optional GPU sector solves.
//   * `Operator.set_symmetry_info_from_dict()` /
//     `FixedSzOperator.set_symmetry_info_from_dict()` so callers can wire
//     the dict produced by `quantum_ed.symmetry.group_from_generators(...)`
//     straight onto an in-process Operator without going through the
//     legacy `automorphism_results/*.json` detour.
//   * `quantum_ed._core.has_cuda_build()` /
//     `has_mpi_build()` / `has_scalapack_build()` build introspection.
//
// The binding compiles against the same CMake target as the rest of
// `quantum_ed._core`, so `WITH_CUDA` / `WITH_MPI` / `WITH_SCALAPACK`
// macros are visible: GPU-only methods (`LANCZOS_GPU`, etc.) fall through
// to the C++ runtime's existing CPU-fallback warning when CUDA is off,
// and the introspection helpers report the exact build configuration.
// =============================================================================

#pragma once

#include <pybind11/pybind11.h>

void bind_dispatcher(pybind11::module_& m);
