---
orphan: true
---

# Phase 7.1 — symmetry projection as the 5th orthogonal axis

> Closes the audit item _"there are way too many ways to symmetrize;
> just decide on the best way and stick to this as the only symm flag."_
> Symmetry projection joins fixed-Sz, GPU, and MPI as a flag on
> `EDParameters`, the deprecated explicit-block kernel is marked
> `[[deprecated]]`, and the streaming kernel becomes the only canonical
> entry point.

The Phase 7 capability matrix grows one axis. New shape:

```
  solver       (DiagonalizationMethod)              ← algorithm only
  use_fixed_sz (bool, EDParameters::use_fixed_sz)   ← basis flag
  use_gpu      (bool, EDParameters::use_gpu)        ← device flag
  use_mpi      (bool, EDParameters::use_mpi)        ← parallelism flag
  use_symmetry (bool, EDParameters::use_symmetry)   ← projection flag (NEW)
```

## Audit — pre-Phase-7.1 symmetry surface

Eight-plus distinct entry points were doing essentially the same job:

| Entry point                                                    | Mode                       | Memory   | GPU | Fixed-Sz | Disk artifacts                | Verdict          |
|----------------------------------------------------------------|----------------------------|----------|-----|----------|-------------------------------|------------------|
| `exact_diagonalization_streaming_symmetry`                     | in-memory orbits, mat-free | low      | yes | via `_fixed_sz` | none (HDF5 cache opt-in) | **CANONICAL**    |
| `exact_diagonalization_streaming_symmetry_fixed_sz`            | streaming + fixed-Sz       | low      | yes | yes      | none                          | absorbed via flag |
| `exact_diagonalization_from_directory_symmetrized`             | explicit Eigen sparse blks | high     | no  | no       | block matrices                | **`[[deprecated]]`** |
| `exact_diagonalization_fixed_sz_symmetrized`                   | explicit blks + fixed-Sz   | high     | no  | yes      | block matrices                | **`[[deprecated]]`** |
| `exact_diagonalization_chunked_symmetry[_fixed_sz]`            | two-pass chunked builder   | very low | no  | yes      | sectors on disk               | CLI-only escape   |
| `exact_diagonalization_disk_chunked_symmetry`                  | per-sector disk            | minimum  | no  | yes      | full sector files             | CLI-only escape   |

**Decision**: streaming-symmetry is the only path that

* keeps orbit data in memory (no disk basis materialisation),
* supports `use_gpu` orthogonally (per-sector GPU kernel in
  `gpu_symmetrized_operator.cu`),
* supports `use_fixed_sz` orthogonally (the `_fixed_sz` overload
  branches on the same flag),
* scales to the largest tractable systems (32-site spin-1/2 with C2 + Tᵃ etc.),
* is what `--symm` already routes through in the CLI,
* doesn't require pre-existing `automorphism_results/` files (it
  generates them on the fly).

The chunked / disk-chunked variants stay as expert escape hatches
behind their own CLI flags (`--chunked-symm`, `--disk-streaming`) for
very-large-N memory-budget edge cases. They are **not** reachable via
`EDParameters::use_symmetry` and are not part of the orthogonal-axis
contract.

## Capability matrix (now 5-axis)

The Phase 7 capability matrix in
[`PHASE_7_SOLVER_AXES.md`](PHASE_7_SOLVER_AXES.md) is unchanged. It now
gains a single column: every solver that exposes a matrix-free `apply`
operator can be combined with `use_symmetry=true` (matrix-free is a
prerequisite because the streaming kernel feeds the per-sector
operator into the solver as a lambda).

| solver               | use_symmetry | notes |
|----------------------|--------------|-------|
| `LANCZOS`            | yes          | default symmetry-projected ground state |
| `BLOCK_LANCZOS`      | yes          | blocked variant |
| `DAVIDSON`           | yes          | preconditioned diag, GPU-capable per sector |
| `LOBPCG`             | yes          | per-sector |
| `KRYLOV_SCHUR`       | yes          | restart-driven |
| `BLOCK_KRYLOV_SCHUR` | yes          | blocked variant |
| `FULL` / dense       | no           | dense kernels need the explicit matrix; use the chunked CLI escape if needed |
| `SCALAPACK[_MIXED]`  | no           | distributed dense — CLI-only escape |
| `mTPQ` / `cTPQ`      | yes          | per-sector thermal Lanczos |
| `FTLM` / `LTLM`      | yes          | per-sector thermal observables |
| `HYBRID`             | yes          | inherits LTLM/FTLM coverage |
| `ARPACK_*`           | yes          | per-sector ARPACK (CPU) |

Setting `use_symmetry=true` with a non-matrix-free dense kernel
(`FULL`, `SCALAPACK`) is rejected at the dispatcher seam.

## Routing

Because `ed/core/ed_wrapper_streaming.h` includes `ed/core/ed_wrapper.h`
(not the other way round), the symmetry routing cannot live inside
`exact_diagonalization_from_files()` directly without a circular
dependency. Phase 7.1 adds a thin dispatcher header that includes
both:

[`ed/core/ed_dispatch_symmetry.h`](../../include/ed/core/ed_dispatch_symmetry.h)

```cpp
namespace ed_dispatch {
  EDResults exact_diagonalization_from_directory(
      const std::string& directory,
      DiagonalizationMethod method,
      const EDParameters& params,
      ...);
  EDResults exact_diagonalization_from_files(
      const std::string& interaction_file,
      const std::string& single_site_file,
      ...);
}
```

The Python binding for `qed.exact_diagonalization_from_directory`
forwards through `ed_dispatch::exact_diagonalization_from_directory`,
so the 5-axis contract is the *only* public Python entry point.

`ed_wrapper.h::exact_diagonalization_from_files()` raises a hard
runtime error if called with `params.use_symmetry=true` so a stale C++
caller fails loudly with a pointer to the dispatcher header.

## Migration

| pre-Phase-7.1                                                     | Phase 7.1 canonical                                                                                  |
|-------------------------------------------------------------------|------------------------------------------------------------------------------------------------------|
| `qed.exact_diagonalization_streaming_symmetry(dir, method, p)`    | `qed.exact_diagonalization_from_directory(dir, method, p)` with `p.use_symmetry = True`              |
| `qed.exact_diagonalization_streaming_symmetry_fixed_sz(dir,n,m,p)`| `from_directory(dir, m, p)` with `p.use_symmetry = True; p.use_fixed_sz = True; p.n_up = n`          |
| `qed.exact_diagonalization_from_directory_symmetrized(...)`       | same as above (use the streaming path; explicit blocks are slower & no GPU)                          |
| `qed.exact_diagonalization_fixed_sz_symmetrized(dir, n, m, p)`    | `from_directory(dir, m, p)` with `p.use_symmetry=True; p.use_fixed_sz=True; p.n_up = n`              |
| `./ED <dir> --symmetrized`                                        | `./ED <dir> --symm` (already an alias)                                                               |
| `./ED <dir> --streaming-symmetry`                                 | `./ED <dir> --symm` (already an alias)                                                               |
| `./ED <dir> --chunked-symm`                                       | _unchanged_ — escape hatch for very-large-N memory budgets                                           |
| `./ED <dir> --disk-streaming`                                     | _unchanged_ — escape hatch for OOM avoidance                                                         |

The deprecated bindings (`exact_diagonalization_*_symmetrized`)
continue to work but emit a `[[deprecated]]` note in their docstrings.

## CLI

`--symm` (and the deprecated aliases `--symmetrized` /
`--streaming-symmetry`) sets BOTH `system.use_symmetry = true` AND
the legacy `workflow.run_symm_auto = true`, so the existing
`ed_main.cpp` dispatch (which fires `run_streaming_symmetry_workflow`
when `run_symm_auto` is set) keeps working unchanged.

A new `--no-symm` flag explicitly disables symmetry projection. This
is useful when a config file enables it by default and you want to
override on a per-run basis.

## Tests

Phase 7.1 extends both lockdown files added in Phase 7:

| file                                                                                                | new content                                                                                                  |
|-----------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------|
| [`tests/unit/test_method_canonicalize.cpp`](../../tests/unit/test_method_canonicalize.cpp)         | 4 new C++ cases covering `EDParameters::use_symmetry` defaults, `EDConfig::useSymmetry()` mirroring, adapter round-trip, and orthogonality with `canonicalize_method_and_flags()`. |
| [`python/tests/test_canonicalize_method.py`](../../python/tests/test_canonicalize_method.py)        | 3 new Python cases mirroring the same contract through the binding plus deprecated-binding back-compat checks. |

Aggregate counts after Phase 7.1:

* C++ ctest suite: **148/148 passing** (was 144 in Phase 7).
* Python pytest suite: **196/196 passing** (was 193 in Phase 7), excluding the
  pre-existing `test_build_introspection_consistency` failure unrelated to
  Phase 7.1 (ScaLAPACK enabled in the wheel build without MPI).
