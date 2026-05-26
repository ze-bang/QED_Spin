# Migration Guide — Surface Unification (May 2026)

This guide explains how to port code from the legacy public surface
(`exact_diagonalization_*`, `ed::auto_pilot::*`, `ed_dispatch::*`,
`qed.diag`, `qed.dssf.compute`, `qed.workflows.*`, ...) to the new
unified entry points (`ed::workflows::*`, `qed.solve / thermal / spectral`).

## What changed in the final collapse (May 2026)

**The Python surface is now exactly three verbs plus one factory:**

* `qed.solve(H, **kwargs)`         — ground state / eigenvalues
* `qed.thermal(H, **kwargs)`       — finite-temperature thermodynamics
* `qed.spectral(H, observables, **kwargs)` — spectral / structure factors

All three take plain keyword arguments — no `SolveOptions` /
`ThermalOptions` / `SpectralOptions` objects in user code (those are
constructed internally from your kwargs).

**Hard-removed Python names:**

* `qed.workflows` (the entire module, ~156 LOC) — was the structured
  Python mirror of `_core.workflows_*` and is no longer needed.
* `qed.diag` — renamed to `qed.solve`.
* `qed.dssf.compute`, `qed.dssf.run_from_directory`, `qed.dssf.pick_method`
  — folded into `qed.spectral(directory, ...)`.
* `qed.exact_diagonalization_core`, `qed.exact_diagonalization_from_directory`,
  `qed.exact_diagonalization_streaming_symmetry`, and
  `qed.exact_diagonalization_streaming_symmetry_fixed_sz` — Python
  re-exports of the legacy C++ bindings removed from the package top
  level. (The underlying pybind11 entry points still exist on
  `qed._core` for one cycle while the in-tree streaming-symmetry path
  finishes its orchestrator migration.)

**Behavioral changes worth flagging:**

* `qed.solve` now auto-projects onto the half-filling Sz=N//2 sector
  by default when `H.conserves_sz()`. To preserve the old "full
  Hilbert space" behaviour, pass `auto_sz=False`.
* `qed.thermal` directory form no longer auto-detects symmetry. Pass
  `use_symmetry_if_available=True` to read `automorphism_results/`.

**Still in-flight (deferred to the follow-up commit):**

* `python/qed/_bindings/dispatcher_bindings.cpp`'s 5 ED-related
  `m.def(...)` registrations remain as deprecation-warning forwarders
  for the in-tree streaming-symmetry path.
* `include/ed/core/ed_wrapper.h` and `ed_wrapper_streaming.h` (the
  `ed::exact_diagonalization_*` C++ family) remain alive for one more
  cycle while the orchestrator gains `make_operator` pybind coverage
  and `workflows_thermal` gains full temperature-grid population for
  FTLM.

**Removed in the ED cleanup sweep (May 2026):**

* `include/ed/auto/solve.h` (`ed::auto_pilot::solve`) — use
  `ed::workflows::solve(op, ed::SolveOptions{...})`.
* `include/ed/auto/thermal.h` (`ed::auto_pilot::thermal`) — use
  `ed::workflows::thermal(op, ed::ThermalOptions{...})`.
* `include/ed/auto/dssf.h` (`ed::auto_pilot::dssf::compute`) — use
  `ed::workflows::spectral(op, observables, ed::SpectralOptions{...})`.
* `include/ed/auto/diag_tune.h` and `include/ed/auto/dssf_tune.h` —
  heuristic auto-tuners that are no longer needed; the orchestrator
  picks methods directly from `geometry().global_dim` and the requested
  `num_eigs`.
* `include/ed/core/ed_dispatch_symmetry.h` — was unreferenced.
* `include/ed/gpu/gpu_dynamics.cuh` + `src/solvers/gpu/gpu_dynamics.cu`
  (`GPUDynamicsSolver` class) — had zero callers.

**Removed in the Full Unified-Interface Collapse Wave F-partial (May 2026):**

* `include/ed/core/dispatch.h` (~312 LOC) — its only callers were the
  two pilot CLI workflows (`run_standard_workflow`,
  `run_streaming_symmetry_workflow`), the Python
  `exact_diagonalization_from_directory` binding, and the
  `test_dispatch_streaming_thermo` unit test. All of those have been
  migrated to `ed::make_operator + ed::workflows::solve` (CLI / Python)
  or removed (the unit test, which is now covered by
  `test_auto_thermal` exercising the unified entry).

**Still scheduled for removal (deferred to follow-up PR after the
remaining 5 heavy CLI workflows, distributed CLI/tests, and GPU
kernel-delegation inversion land):**

* `python/qed/_bindings/dispatcher_bindings.cpp`'s 5 ED-related
  `m.def(...)` registrations remain as **deprecation-warning forwarders**
  for out-of-tree consumers; they will be deleted once all known
  downstream callers are ported.
* `include/ed/core/ed_wrapper.h`, `ed_wrapper_streaming.h`.
* `include/ed/distributed/{distributed_lanczos,distributed_lanczos_gpu,
  distributed_krylov_schur,distributed_krylov_schur_gpu,distributed_ftlm,
  distributed_ftlm_gpu,distributed_tpq,distributed_tpq_gpu}.h`.
* `include/ed/gpu/{gpu_ed_wrapper.h,gpu_lanczos.cuh,gpu_solvers.h}`.
* `include/ed/solvers/*.h` shells once the kernel-shim layer in
  `include/ed/thermal/*_kernel.h` is rewritten to be self-contained.

---

## TL;DR

| Legacy (C++)                                              | New (C++)                                       |
|-----------------------------------------------------------|-------------------------------------------------|
| `exact_diagonalization_core(apply, dim, METHOD, params)`  | `ed::workflows::solve(op, {.method=...})`       |
| `exact_diagonalization_from_files(...)`                   | `ed::make_operator(...) + ed::workflows::solve` |
| `exact_diagonalization_from_directory(...)`               | `ed::make_operator(DirectoryPath{...}) + solve` |
| `ed::auto_pilot::solve(op, AutoSolveOptions{...})`        | `ed::workflows::solve(op, SolveOptions{...})`   |
| `ed::auto_pilot::thermal(op, AutoThermalOptions{...})`    | `ed::workflows::thermal(op, ThermalOptions{...})`|
| `ed::auto_pilot::dssf::compute(...)`                      | `ed::workflows::spectral(op, observables, ...)` |

| Legacy (Python)                              | New (Python)                                          |
|----------------------------------------------|-------------------------------------------------------|
| `qed.diag(H, ...)`                           | `qed.solve(H, ...)`                                   |
| `qed.workflows.solve(op, SolveOptions(...))` | `qed.solve(H, solver="LANCZOS", **kwargs)`            |
| `qed.workflows.thermal(op, ThermalOptions)`  | `qed.thermal(H, method="FTLM", **kwargs)`             |
| `qed.workflows.spectral(op, obs, ...)`       | `qed.spectral(H, observables, **kwargs)`              |
| `qed.dssf.compute(directory, ...)`           | `qed.spectral(directory, **kwargs)`                   |
| `qed.dssf.run_from_directory(...)`           | `qed.spectral(directory, method=..., **kwargs)`       |
| `qed.exact_diagonalization_core(op, ...)`    | `qed.solve(H, ...)` (or `_core.workflows_solve` for the raw orchestrator) |

**This is a hard rename for the public Python surface.** None of the
legacy names above resolve under `qed.*` any more — old notebooks must
be updated. The pybind11 internals on `qed._core` still expose
`workflows_solve / workflows_thermal / workflows_spectral` for power
users; you should not normally need them.

---

## C++ migration recipes

### 1. Ground-state ED from files

**Before**:

```cpp
#include <ed/core/ed_wrapper.h>

EDParameters params;
params.num_sites      = N;
params.spin_length    = 0.5f;
params.num_eigenvalues = 5;

EDResults r = exact_diagonalization_from_files(
    "/data/InterAll.dat", "/data/Trans.dat",
    DiagonalizationMethod::LANCZOS,
    params);
```

**After**:

```cpp
#include <ed/core/make_operator.h>
#include <ed/orchestrator.h>

auto op = ed::make_operator(ed::OperatorSpec{
    .source     = ed::FilePaths{"/data/InterAll.dat", "/data/Trans.dat"},
    .num_sites  = N,
    .spin_l     = 0.5f,
});

ed::GroundStateResult r = ed::workflows::solve(*op, ed::SolveOptions{
    .num_eigenvalues = 5,
    .method          = ed::SolveMethod::Lanczos,
});
```

### 2. Auto-pilot solve (let the library pick the backend + method)

**Before**:

```cpp
ed::auto_pilot::AutoSolveOptions opts;
opts.num_eigenvalues       = 1;
opts.compute_eigenvectors  = false;
auto r = ed::auto_pilot::solve(*op, opts);
```

**After** — the heuristics are absorbed into `SolveMethod::Auto`:

```cpp
auto r = ed::workflows::solve(*op, ed::SolveOptions{
    .num_eigenvalues      = 1,
    .compute_eigenvectors = false,
    .method               = ed::SolveMethod::Auto,
});
```

### 3. Finite-temperature TPQ

**Before**:

```cpp
ed::auto_pilot::AutoThermalOptions opts;
opts.method = ThermalMethod::MicroTPQ;
opts.temperatures = {0.1, 0.2, 0.5};
auto r = ed::auto_pilot::thermal(*op, opts);
```

**After**:

```cpp
auto r = ed::workflows::thermal(*op, ed::ThermalOptions{
    .method        = ed::ThermalMethod::MicroTPQ,
    .temperatures  = {0.1, 0.2, 0.5},
});
```

### 4. Spectral function (DSSF / continued-fraction Lanczos)

**Before**:

```cpp
ed::auto_pilot::dssf::Options opts;
opts.omega_grid = grid;
opts.method     = ed::auto_pilot::dssf::Method::CF_LANCZOS;
auto r = ed::auto_pilot::dssf::compute(/* op_files */, observables, opts);
```

**After**:

```cpp
auto r = ed::workflows::spectral(*op, observables, ed::SpectralOptions{
    .omega_grid = grid,
    .method     = ed::SpectralMethod::CF_Lanczos,
});
```

---

## Python migration recipes

### 1. Ground-state ED

```python
# Before
results = qed.diag(H, num_eigenvalues=5)
# or, via the structured surface that no longer exists:
opts = qed.workflows.SolveOptions(); opts.num_eigs = 5
results = qed.workflows.solve(H, opts)

# After
results = qed.solve(H, num_eigenvalues=5)
```

`qed.diag` was removed in this release; `qed.workflows.solve` was also
removed. Use `qed.solve` with plain kwargs. Note that `qed.solve` now
auto-projects onto the half-filling Sz sector by default when the
Hamiltonian conserves Sz; pass `auto_sz=False` to keep the full
Hilbert space.

### 2. Thermal

```python
# Before
ftlm = qed.finite_temperature_lanczos("./data", T=[0.1, 0.5, 1.0])

# After
thermal = qed.thermal("./data", method="ftlm", temperatures=[0.1, 0.5, 1.0])
```

### 3. Dynamical structure factor (DSSF)

```python
# Before -- directory form
res = qed.dssf.compute("./data", omega=ws, T=0.0)
# Before -- in-memory form via the removed structured surface
opts = qed.workflows.SpectralOptions(); opts.broadening = 0.05
res = qed.workflows.spectral(H, [Sz], opts)

# After -- one verb, polymorphic over (H, observables) vs (directory)
res = qed.spectral("./data", omega=ws, T=0.0)        # CLI form
res = qed.spectral(H, [Sz], omega=ws, eta=0.05)      # in-memory form
```

`qed.dssf` now only re-exports the **data helpers** (`OperatorSpec`,
`ObservablePairs`, `build_observable_pairs`,
`compute_transverse_bases`) — the actual workflow lives in
`qed.spectral`.

---

## Why the change?

The legacy API had ~30 entry points covering the cross-product of
(backend, fixed-Sz, streaming-symmetry, method, observable type) —
each one a long file with hand-rolled dispatch. The new API has three
entry points (`solve`, `thermal`, `spectral`), each of which dispatches
through a single `select_backend` + `std::visit` decision tree to one of
a small number of backend-templated kernels (`lanczos_kernel`,
`krylov_schur_kernel`, `block_lanczos_kernel`, `tpq_kernel`,
`cf_spectral_kernel`).

Net effect:

* **~3,500 LOC deleted** from the public surface (forwarder files and
  duplicated dispatch logic).
* **One** place to add a new backend (write the `Backend` impl, register
  it in `select_backend`).
* **One** place to add a new method (write the `*_kernel<Backend>`,
  register it in the orchestrator).
* **One** unified result shape (`GroundStateResult` / `ThermalResult` /
  `SpectralResult`) carrying `BackendMetadata` and `KrylovDiagnostics`
  for observability across all entry points.

For the architectural rationale see
`docs/architecture/STRUCTURAL_AUDIT.md` Part VI ("Minimalist collapse").
