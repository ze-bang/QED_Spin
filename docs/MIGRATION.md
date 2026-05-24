# Migration Guide — Minimalist ED Collapse (May 2026)

This guide explains how to port C++ and Python callers from the legacy
public surface (`exact_diagonalization_*`, `ed::auto_pilot::*`,
`ed_dispatch::*`, `qed.diag`, `qed.dssf.compute`, ...) to the new
unified entry points (`ed::workflows::*`, `qed.solve / thermal / spectral`).

The collapse is **source-compatible**: every legacy entry point remains
buildable during the migration window and routes through the new
orchestrators internally. New code MUST use the new entry points; the
legacy surface is being removed phase-by-phase as in-tree callers
(CLI binary, Python binding, unit/integration tests) are ported.

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

**Still scheduled for removal (deferred to follow-up PR after Phase 4
CLI migration completes):**

* `include/ed/core/dispatch.h`, `ed_wrapper.h`, `ed_wrapper_streaming.h`.
* `python/qed/_bindings/dispatcher_bindings.cpp` (collapse to
  deprecation aliases).
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

| Legacy (Python)                          | New (Python)                                          |
|------------------------------------------|-------------------------------------------------------|
| `qed.diag(directory, method=..., ...)`   | `qed.solve(directory, method=..., ...)`               |
| `qed.lanczos(directory, ...)`            | `qed.solve(directory, method="lanczos", ...)`         |
| `qed.finite_temperature_lanczos(...)`    | `qed.thermal(directory, method="ftlm", ...)`          |
| `qed.dssf.compute(directory, ...)`       | `qed.spectral(directory, observables=..., ...)`       |

The legacy Python names remain as thin aliases that route through the
new entry points; existing notebooks continue to work unchanged.

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
results = qed.diag("./data", method="lanczos", num_eigs=5)

# After
results = qed.solve("./data", method="lanczos", num_eigs=5)
```

`qed.diag` is preserved as a deprecation alias that emits a
`DeprecationWarning` and forwards to `qed.solve`.

### 2. Thermal

```python
# Before
ftlm = qed.finite_temperature_lanczos("./data", T=[0.1, 0.5, 1.0])

# After
thermal = qed.thermal("./data", method="ftlm", temperatures=[0.1, 0.5, 1.0])
```

### 3. Dynamical structure factor (DSSF)

```python
# Before
res = qed.dssf.compute("./data", omega_grid=ws, T=0.0)

# After
res = qed.spectral("./data", observables=[Sx, Sy, Sz], omega_grid=ws, T=0.0)
```

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
