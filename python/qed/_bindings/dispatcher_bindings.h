// =============================================================================
// python/qed/_bindings/dispatcher_bindings.h
//
// Expose the high-level ED dispatcher to Python: one binding call wires up the
// Krylov / dense / thermal solver surface plus the directory / streaming-
// symmetry drivers.
//
// `bind_dispatcher(m)` populates `qed._core` with:
//   * `DiagonalizationMethod` enum (9 values: LANCZOS, BLOCK_LANCZOS,
//     KRYLOV_SCHUR, FULL, mTPQ, cTPQ, FTLM, LTLM, KPM_DOS).
//   * `EDParameters` mutable parameter bag.
//   * `EDResults` immutable result envelope.
//   * `exact_diagonalization_core(operator_, method, params)` for both
//     `Operator` and `FixedSzOperator`. Device / parallelism / Sz /
//     symmetry are flags on `EDParameters` (`use_gpu`, `use_mpi`,
//     `use_fixed_sz`, `use_symmetry`).
//   * `exact_diagonalization_from_directory(...)` for directory-driven runs
//     -- the 5-axis dispatcher that also reaches the streaming-symmetry path.
//   * `exact_diagonalization_streaming_symmetry[_fixed_sz]` for the
//     in-memory streaming-symmetry path -- the canonical entry point for
//     symmetry-projected ED with optional GPU sector solves.
//   * `Operator.set_symmetry_info_from_dict()` /
//     `FixedSzOperator.set_symmetry_info_from_dict()` so callers can wire
//     the dict produced by `qed.symmetry.group_from_generators(...)`
//     straight onto an in-process Operator.
//   * `qed._core.has_cuda_build()` / `has_mpi_build()` build introspection.
//
// The binding compiles against the same CMake target as the rest of
// `qed._core`, so `WITH_CUDA` / `WITH_MPI` macros are visible: setting
// `params.use_gpu = True` on a build without CUDA falls back to the CPU
// code path with a runtime warning.
// =============================================================================

#pragma once

#include <pybind11/pybind11.h>

void bind_dispatcher(pybind11::module_& m);
