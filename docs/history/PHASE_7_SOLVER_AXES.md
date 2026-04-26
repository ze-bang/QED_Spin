---
orphan: true
---

# Phase 7 — solver organization on orthogonal axes

> Closes the audit item _"`LANCZOS_FIXED_SZ` should not be a separate
> solver, it's just LANCZOS on the fixed-Sz basis"_. Every solver is
> now picked along **four orthogonal axes** instead of along a hand-
> tangled enum:
>
> ```
>   solver       (DiagonalizationMethod)              ← algorithm only
>   use_fixed_sz (bool, EDParameters::use_fixed_sz)   ← basis flag
>   use_gpu      (bool, EDParameters::use_gpu)        ← device flag
>   use_mpi      (bool, EDParameters::use_mpi)        ← parallelism flag
> ```
>
> The legacy `_GPU` / `_CUDA` / `_MPI` / `_FIXED_SZ` enum variants are
> kept as deprecated aliases for backwards compatibility (HDF5 metadata,
> CLI strings, pre-Phase-7 user code) and collapsed onto the canonical
> tuple at every dispatcher entry point.
>
> **Phase 7.1 ([details](PHASE_7_1_SYMMETRY_AXIS.md))** adds a 5th
> orthogonal axis — `EDParameters::use_symmetry` — and consolidates
> all symmetry entry points onto a single canonical streaming kernel.

## Capability matrix

| solver                       | CPU | GPU | MPI | fixed-Sz | notes |
|------------------------------|-----|-----|-----|----------|-------|
| `LANCZOS`                    | yes | yes | (1) | yes      | default ground-state solver |
| `LANCZOS_SELECTIVE`          | yes |  -  | (1) | yes      | reorthogonalisation variant |
| `LANCZOS_NO_ORTHO`           | yes |  -  | (1) | yes      | reorthogonalisation variant |
| `BLOCK_LANCZOS`              | yes | yes | (1) | yes      | block size from `EDParameters::block_size` |
| `CHEBYSHEV_FILTERED`         | yes |  -  |  -  | yes      | spectral slicing |
| `SHIFT_INVERT`               | yes |  -  |  -  | yes      | algebraic interior eigs |
| `SHIFT_INVERT_ROBUST`        | yes |  -  |  -  | yes      | robust variant |
| `DAVIDSON`                   | yes | yes |  -  | yes      | preconditioned diag |
| `BICG`                       | yes |  -  |  -  | yes      | shift-invert helper |
| `LOBPCG`                     | yes | (2) |  -  | yes      | (2) `LOBPCG_GPU` currently redirects to `DAVIDSON_GPU` |
| `KRYLOV_SCHUR`               | yes | yes |  -  | yes      | restart-driven Krylov |
| `BLOCK_KRYLOV_SCHUR`         | yes | yes |  -  | yes      | blocked variant |
| `IMPLICIT_RESTART_LANCZOS`   | yes |  -  |  -  | yes      | IRLM |
| `THICK_RESTART_LANCZOS`      | yes |  -  |  -  | yes      | TRL with locking |
| `FULL`                       | yes | yes |  -  | yes (3)  | dense LAPACK; (3) auto-splits Sz sectors |
| `OSS`                        | yes |  -  |  -  | yes      | optimal spectrum solver |
| `SCALAPACK`                  |  -  |  -  | yes | yes (3)  | distributed dense (PDSYEVR), kept as separate kernel |
| `SCALAPACK_MIXED`            |  -  |  -  | yes | yes (3)  | mixed-precision SCALAPACK refinement |
| `mTPQ`                       | yes | yes | yes | yes      | microcanonical TPQ |
| `cTPQ`                       | yes | yes |  -  | yes      | canonical TPQ |
| `FTLM`                       | yes | yes |  -  | yes      | finite-temp Lanczos |
| `LTLM`                       | yes |  -  |  -  | yes      | low-temp Lanczos |
| `HYBRID`                     | yes |  -  |  -  | yes      | LTLM + FTLM auto-crossover |
| `ARPACK_SM` / `_LM` / …      | yes |  -  |  -  | yes      | ARPACK algorithmic variants |

Notes:

1. CPU-iterative solvers reach the MPI surface via the standalone
   `ed_distributed_main` binary (`quantum_ed.mpi.run_distributed(...)`),
   not by setting `use_mpi=true` on the in-process dispatcher. The
   `use_mpi` flag is consumed by the distributed driver and serialized
   into HDF5 metadata.
2. `LOBPCG_GPU` currently aliases the GPU Davidson kernel; future work
   will give LOBPCG its own GPU code path.
3. `FULL` and `SCALAPACK` *auto-detect* Sz conservation in the
   Hamiltonian and split the Hilbert space into sectors before calling
   the dense kernel — the user does not have to set `use_fixed_sz=true`
   for the auto-split to kick in. Setting `use_fixed_sz=true` (with
   `EDParameters::n_up`) restricts to one specific sector.

## Canonicalization

`ed::canonicalize_method_and_flags(method, use_fixed_sz, use_gpu, use_mpi)`
collapses every legacy `_GPU` / `_CUDA` / `_MPI` / `_FIXED_SZ` enum
variant onto the canonical `(base_method, flags)` tuple. The
collapse is OR-merged with caller-supplied flags so no information is
lost; the operation is idempotent and is exposed to Python as
`quantum_ed.canonicalize_method(...)` for testing and introspection.

| input enum                       | canonical method  | use_fixed_sz | use_gpu | use_mpi |
|----------------------------------|-------------------|--------------|---------|---------|
| `LANCZOS_GPU`                    | `LANCZOS`         | (caller)     | true    | (caller)|
| `LANCZOS_GPU_FIXED_SZ`           | `LANCZOS`         | true         | true    | (caller)|
| `BLOCK_LANCZOS_GPU_FIXED_SZ`     | `BLOCK_LANCZOS`   | true         | true    | (caller)|
| `FTLM_GPU_FIXED_SZ`              | `FTLM`            | true         | true    | (caller)|
| `mTPQ_CUDA`                      | `mTPQ`            | (caller)     | true    | (caller)|
| `mTPQ_GPU`                       | `mTPQ`            | (caller)     | true    | (caller)|
| `mTPQ_MPI`                       | `mTPQ`            | (caller)     | (caller)| true    |
| `SCALAPACK` / `SCALAPACK_MIXED`  | _unchanged_       | (caller)     | (caller)| true    |
| any canonical input              | _unchanged_       | (caller)     | (caller)| (caller)|

`SCALAPACK` and `SCALAPACK_MIXED` are *not* collapsed onto `FULL` —
they go through a different LAPACK call (`PDSYEVR` /
mixed-precision refinement) and should be selected as a distinct
solver, not as "FULL with use_mpi=true". They *are* implicitly
MPI-backed, so canonicalization honestly flags `use_mpi=true`.

`mTPQ_CUDA` was historically a no-op alias for `mTPQ_GPU`. Both
canonicalize to the same tuple and any new code should drop the
`_CUDA` spelling.

## Migration

| pre-Phase-7                                     | Phase 7 canonical                                                                                  |
|-------------------------------------------------|----------------------------------------------------------------------------------------------------|
| `method = LANCZOS_GPU`                          | `method = LANCZOS; params.use_gpu = True`                                                          |
| `method = LANCZOS_GPU_FIXED_SZ`                 | `method = LANCZOS; params.use_gpu = True; params.use_fixed_sz = True`                              |
| `method = FTLM_GPU_FIXED_SZ`                    | `method = FTLM; params.use_gpu = True; params.use_fixed_sz = True`                                 |
| `method = mTPQ_CUDA`                            | `method = mTPQ; params.use_gpu = True`                                                             |
| `method = mTPQ_MPI`                             | `method = mTPQ; params.use_mpi = True`                                                             |
| `method = SCALAPACK`                            | _unchanged_ (already canonical)                                                                    |

The legacy enum values still work: every dispatcher entry point
(`exact_diagonalization_core`, `*_from_files`,
`exact_diagonalization_fixed_sz`) calls
`ed::canonicalize_method_and_flags()` on entry, so callers that pass
`LANCZOS_GPU` end up dispatching exactly the same code path as
callers that pass `LANCZOS` + `use_gpu=true`. The compiler emits a
deprecation warning at the call site to nudge callers towards the
canonical form, except inside the dispatcher / CLI / Python binding,
where the deprecation warning is suppressed by a targeted
`#pragma GCC diagnostic` block (the dispatcher *needs* to know about
the legacy values to canonicalize them).

## CLI

The CLI gained `--gpu` / `--mpi` flags (next to the existing
`--fixed-sz` / `--n-up`). They set
`config.system.use_gpu` / `config.system.use_mpi`, which are
threaded into `EDParameters::use_gpu` / `use_mpi` by
`ed_config_adapter::toEDParameters()`. The `--method` argument
continues to accept all legacy strings (`LANCZOS_GPU`,
`LANCZOS_GPU_FIXED_SZ`, `mTPQ_CUDA`, `mTPQ_MPI`, etc.) and the
canonicalization happens on the dispatcher side.

## Tests

Phase 7 adds two lockdown test files:

| file | scope | sections |
|------|-------|----------|
| [`tests/unit/test_method_canonicalize.cpp`](../../tests/unit/test_method_canonicalize.cpp) | C++ canonicalize() contract: identity for canonical inputs; `_GPU` / `_FIXED_SZ` / `_MPI` / `_CUDA` collapse; SCALAPACK kept distinct; OR-merge of caller flags; idempotence; `legacy_method_for_dispatch()` round-trip. | 10 cases, 163 assertions |
| [`python/tests/test_canonicalize_method.py`](../../python/tests/test_canonicalize_method.py) | Mirrors the C++ contract through the Python binding (`quantum_ed.canonicalize_method`) plus `EDParameters::use_gpu` / `use_mpi` round-trip. | 61 parametrised cases |

Both files keep the canonicalize behaviour from accidentally diverging
between the C++ dispatcher and the Python introspection helper, and
between this Phase and the next refactor.
