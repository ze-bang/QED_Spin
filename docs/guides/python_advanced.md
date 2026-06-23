---
orphan: true
---

# Advanced Python usage

This guide is the **catalogue of advanced patterns** the Python API
supports. It complements the quickstart at
[`python_quickstart.md`](python_quickstart.md), the one-call reference
at [`one_call_api.md`](one_call_api.md), and the capability matrix at
[`python_api_coverage.md`](python_api_coverage.md).

If your task is straightforward (build a Hamiltonian, run Lanczos, look
at a few thermodynamic curves), use the quickstart. If you need:

* A **specific solver** other than Lanczos (`BLOCK_LANCZOS`,
  `KRYLOV_SCHUR`, `FTLM`, `LTLM`, `mTPQ`, `cTPQ`, `KPM_DOS`, `FULL`)
* The **GPU** path (`device='gpu'` — auto-selects the right GPU
  kernel for the requested solver)
* **Symmetry projection** in-process (without writing
  `automorphism_results/*.json` to disk first)
* **Streaming-symmetry** ED for the largest tractable clusters
* **MPI distributed** Lanczos / FTLM / TPQ runs from a Python script
* The **full `./ED dssf`** continued-fraction spectral driver
* To **introspect the build** (CUDA / MPI present?) at runtime so the
  same Python script works on a laptop and a cluster

…this is the right document. Every pattern below is exercised by
`python/tests/test_dispatcher.py`.

---

## 0. Choosing the right entry point

There are now **three Python entry points** — one for each kind of
computation. Everything else is reachable via kwargs on these three
verbs.

| You want… | …call this | Lives in |
|-----------|------------|----------|
| eigenvalues / ground state / a few low-lying states | `qed.solve(H, ...)` | `qed.workflow` |
| finite-temperature trajectories (mTPQ / cTPQ / FTLM / LTLM) | `qed.thermal(H, method=..., ...)` | `qed.thermal` |
| structure factors S(Q, ω) / S(Q, T) / KPM-DOS | `qed.spectral(dir, T=..., omega=..., ...)` | `qed.dssf` |
| MPI cluster + a directory | `qed.mpi.run_distributed(dir, method, n_ranks, ...)` | `qed.mpi` |

The legacy `qed.exact_diagonalization_*` family was deleted in the
May 2026 surface-unification collapse along with the C++
`ed::exact_diagonalization_*` and `ed::auto_pilot::*` entry points.
The Python wrappers now drop down to
`qed._core.workflows_{solve,thermal,spectral}` directly (Pybind11
bindings for `ed::workflows::*`), so the canonical Python and C++
surfaces are identical up to kwargs vs structured options.

---

## 1. `qed.solve` — the canonical eigenvalue / ground-state entry

```python
import qed

N = 12
b = qed.input.HamiltonianBuilder(num_sites=N)
b.heisenberg(bonds=[(i, (i + 1) % N) for i in range(N)], J=1.0)
op = b.to_operator()

result = qed.solve(
    op,
    num_eigenvalues=4,
    solver="KRYLOV_SCHUR",
    tolerance=1e-12,
    max_iterations=400,
)
print("E0..E3 =", sorted(result.eigenvalues)[:4])
```

`solver=` accepts either a `DiagonalizationMethod` enum value or its
string name (case-insensitive). Retained backends only:

| Family | Methods |
|--------|---------|
| Lanczos | `LANCZOS`, `BLOCK_LANCZOS` |
| Krylov-Schur | `KRYLOV_SCHUR` |
| Dense | `FULL` |
| Finite-temperature | `FTLM`, `LTLM`, `mTPQ`, `cTPQ`, `KPM_DOS` (use via `qed.thermal` / `qed.spectral`) |

The May 2026 minimalist-solver-matrix cleanup retired `ARPACK_*`,
`LOBPCG`, `DAVIDSON`, `CHEBYSHEV_FILTERED`, `SHIFT_INVERT*`, `IRL`,
`TRL`, `BICG`, `OSS`, `SCALAPACK*`, `HYBRID`, and every `_GPU` /
`_MPI` enum suffix (those axes are now flags carried through the
`device=` kwarg and the orchestrator's `BackendConstraints`).

`extra_params={...}` is the escape hatch for any niche `EDParameters`
field:

```python
result = qed.solve(
    op,
    solver="FTLM",
    sz=N // 2,
    extra_params={
        "ftlm_krylov_dim": 80,
        "ftlm_full_reorth": True,
        "ftlm_seed": 1234,
    },
)
```

`repr(qed.EDParameters())` summarises the headline fields, and
`qed.list_diag_parameters()` prints the full catalogue grouped by
category.

**Returns.** `EDResults` with `eigenvalues`, `eigenvectors_computed`,
`eigenvectors_path`, `thermo_data` (a `ThermodynamicData` instance with
`temperatures`, `energy`, `specific_heat`, `entropy`, `free_energy`),
and `ftlm_results`. Call `.to_dict()` for ergonomic JSON / pickle
serialization.

### 1.1 Fixed-Sz sectors

`qed.solve(..., sz=N//2)` auto-projects a Sz-conserving `Operator`
onto the half-filled sector. With `auto_sz=True` (the default), this
happens automatically when H conserves Sz and no explicit `sz=` is
given.

For full control, build the `FixedSzOperator` yourself:

```python
N = 14
b = qed.input.HamiltonianBuilder(num_sites=N)
b.heisenberg(bonds=[(i, (i + 1) % N) for i in range(N)], J=1.0)

fop = qed.FixedSzOperator(num_sites=N, n_up=N // 2, spin=0.5)
b.emit_into(fop)                     # populate the fixed-Sz operator

result = qed.solve(fop, num_eigenvalues=1, solver="LANCZOS")
```

If you prefer building term-by-term, `add_two_body` / `add_one_body` /
`add_three_body` are inherited from `Operator`:

```python
for i in range(N):
    j = (i + 1) % N
    fop.add_two_body(qed.OP_SZ,     i, qed.OP_SZ,     j, 1.0 + 0.0j)
    fop.add_two_body(qed.OP_SPLUS,  i, qed.OP_SMINUS, j, 0.5 + 0.0j)
    fop.add_two_body(qed.OP_SMINUS, i, qed.OP_SPLUS,  j, 0.5 + 0.0j)
```

The orchestrator detects the `FixedSzOperator` subclass and routes
through the fixed-Sz code paths automatically.

---

## 2. Symmetry projection in-process

The `qed.symmetry` DSL builds the same symmetry-group dict the C++
engine consumes, and Phase 5 lets you attach it directly to an
`Operator` without going through `automorphism_results/*.json`.

```python
import qed as qed

N = 6
g_t = qed.symmetry.translation(N, 1)        # cyclic shift by 1
g_r = qed.symmetry.reflection_1d(N)         # bond-centred reflection
info = qed.symmetry.group_from_generators(N, [g_t, g_r])

print("group size:", len(info["max_clique"]))
print("sectors:", [s["quantum_numbers"] for s in info["sectors"]])

b = qed.input.HamiltonianBuilder(num_sites=N)
b.heisenberg(bonds=[(i, (i + 1) % N) for i in range(N)], J=1.0)
op = b.to_operator()
op.set_symmetry_info_from_dict(info)

# Round-trip back to a dict for serialization / inspection.
info_back = op.get_symmetry_info_as_dict()
assert info_back["num_generators"] == info["num_generators"]
```

For an actual symmetry-projected solve, pass the `info` dict (or a
`GeneratorSet`) as the `symmetry=` kwarg on `qed.solve`. The Python
wrapper writes the operator and metadata to a temp directory and
invokes `_core.workflows_solve_streaming_symmetry_directory`, which
composes `ed::make_streaming_symmetry_operator(spec)` with a
per-sector `ed::workflows::solve` loop in C++.

```python
result = qed.solve(
    op,
    num_eigenvalues=4,
    symmetry=info,        # GeneratorSet or qed.symmetry.* dict
)
print("symmetry-projected E0..E3 =", sorted(result.eigenvalues)[:4])
```

Combine with `sz=` to request fixed-Sz × symmetry in one call:

```python
result = qed.solve(op, num_eigenvalues=2, sz=N//2, symmetry=info)
```

If you need to start from a directory deck (e.g. one written by
`HamiltonianBuilder.write_directory` plus an `automorphism_results/`
JSON tree from `edlib.automorphism_finder`), load it back into an
`Operator` and dispatch through the same entry point:

```python
b.write_directory("/tmp/heisenberg-6-chain", lattice=qed.input.lattice.chain(N))

# Run the legacy finder once to materialise automorphism_results/.
import subprocess, sys
subprocess.run([
    sys.executable, "-m", "edlib.automorphism_finder",
    "--data_dir", "/tmp/heisenberg-6-chain",
], check=True)

op = qed.input.load_from_directory("/tmp/heisenberg-6-chain")
info = op.get_symmetry_info_as_dict()
result = qed.solve(op, num_eigenvalues=4, symmetry=info)
```

---

## 3. GPU dispatch

When the project was built with `WITH_CUDA=ON` (check
`qed.has_cuda_build()`), every retained solver reaches its CUDA
kernel through the same `qed.solve` / `qed.thermal` / `qed.spectral`
entry points by passing `device="gpu"`:

```python
import qed

if not qed.has_cuda_build():
    print("This wheel was built without CUDA; GPU runs will fall back "
          "to CPU with a runtime warning.")

# Per-sector GPU Lanczos for a 24-site cluster with symmetry projection.
result = qed.solve(
    op,
    num_eigenvalues=4,
    tolerance=1e-10,
    device="gpu",
    symmetry=info,        # symmetry projection (streaming kernel)
)
print("GPU per-sector eigenvalues:", sorted(result.eigenvalues)[:4])
```

Fixed-Sz × symmetry × GPU all compose orthogonally:

```python
result = qed.solve(
    op,
    num_eigenvalues=2,
    sz=N // 2,
    symmetry=info,
    device="gpu",
)
```

Finite-temperature trajectories (mTPQ / cTPQ / FTLM / LTLM) on the
full Hilbert space go through `qed.thermal`:

```python
result = qed.thermal(
    op,
    method="mTPQ",
    num_samples=8,
    target_beta=20.0,
    num_T=200,
    device="gpu",
)
```

The orchestrator constructs the appropriate `GPUOperator` /
`StreamingSymmetryOperator` internally via `ed::make_operator(spec)`
and dispatches to the right CUDA kernel — exactly the same code path
`./ED --use-gpu` would take.

---

## 4. MPI distributed solvers

`qed.mpi.run_distributed` is a thin Python wrapper that locates
`mpiexec` (or your launcher of choice — `srun`, `mpirun`,
`jsrun`, …) and `ed_distributed_main`, then runs the right argv:

```python
import qed as qed
from qed import mpi as qed_mpi

if not qed.has_mpi_build():
    raise RuntimeError("This build was made without MPI; ed_distributed_main "
                       "does not exist.")

result = qed_mpi.run_distributed(
    directory="/scratch/runs/heisenberg-32-chain",
    method="lanczos",          # or "ftlm", "tpq", "lanczos_symmetry",
                               # "lanczos_gpu" (NCCL halo, requires WITH_CUDA)
    n_ranks=8,
    launcher="srun",           # default "mpiexec"; use "mpirun" for OpenMPI
    launcher_args=("--bind-to=core", "--map-by=numa"),
    extra_args=("--max-iter", "400", "--reorth", "1"),
    capture_output=True,
)
print(result.stdout)
```

Why a subprocess and not in-process bindings? A single Python
interpreter cannot host `MPI_Init` cleanly across all libraries it
loads (interaction with NumPy / NCCL / cuSPARSE etc. is brittle), and
the canonical SLURM workflow is one Python launcher per node anyway.
The wrapper just removes the `subprocess` boilerplate and validates
inputs (unknown method names raise `ValueError`; missing directory
raises `FileNotFoundError`; missing `mpiexec` / `ed_distributed_main`
raise `FileNotFoundError` with an actionable message).

---

## 5. The full `./ED dssf` driver

Mirrors §4: a thin Python wrapper around the canonical CLI. Reaches
the **continued-fraction S(Q,ω)** engine, the static-thermal driver,
and (when the build is GPU-enabled) the CUDA spectral kernels.

```python
from qed import dssf

result = dssf.run_from_directory(
    directory="/scratch/runs/heisenberg-16-chain",
    method="dynamical_thermal",   # or "static_thermal", "ground_state_dssf"
    extra_args=("--num-temps", "21"),
    capture_output=True,
)
print(result.stdout)
```

The DSSF observables themselves (the $(O_1, O_2, \text{name})$ pairs
the engine averages over) can still be assembled from Python via
`qed.dssf.build_observable_pairs(...)` — the
`run_from_directory` helper is for the *full pipeline* (operator
assembly + continued fractions + HDF5 output trees).

---

## 6. Build introspection

```python
import qed

print("CUDA build:    ", qed.has_cuda_build())
print("MPI build:     ", qed.has_mpi_build())
print("ScaLAPACK build:", qed.has_scalapack_build())

# Same script, two builds — pick the device kwarg at runtime.
device = "gpu" if qed.has_cuda_build() else "cpu"
result = qed.solve(op, num_eigenvalues=4, device=device)
```

These three helpers report the compile-time state of the
`WITH_CUDA` / `WITH_MPI` / `WITH_SCALAPACK` macros, so the same
Python script can run on a laptop CPU build *and* a CUDA + MPI cluster
build without if-else gymnastics around `import` failures.
`has_scalapack_build() == True` implies `has_mpi_build() == True`
(ScaLAPACK requires MPI).

---

## 7. Worked example: end-to-end on a 16-site Heisenberg chain

```python
import qed

N = 16
b = qed.input.HamiltonianBuilder(num_sites=N)
b.heisenberg(bonds=[(i, (i + 1) % N) for i in range(N)], J=1.0)

# 1. CPU ground state via Krylov-Schur.
op = b.to_operator()
gs = qed.solve(
    op,
    num_eigenvalues=6,
    solver="KRYLOV_SCHUR",
    max_iterations=500,
    tolerance=1e-12,
)
print("CPU low-energy spectrum:", sorted(gs.eigenvalues))

# 2. FTLM thermodynamics through the same surface — qed.thermal.
ftlm = qed.thermal(
    op,
    method="FTLM",
    num_samples=32,
    T_min=0.05, T_max=5.0, num_T=80,
    extra_params={"ftlm_krylov_dim": 80},
)
T = ftlm.temperatures
Cv = ftlm.specific_heat
print("FTLM Cv at T=1:", Cv[len(T) // 2])

# 3. Symmetry-projected solve.
g_t = qed.symmetry.translation(N, 1)
info = qed.symmetry.group_from_generators(N, [g_t])
op.set_symmetry_info_from_dict(info)

proj = qed.solve(op, num_eigenvalues=6, symmetry=info)
print("symmetry-projected spectrum:", sorted(proj.eigenvalues))

# 4. (Optional, requires WITH_MPI) MPI distributed Lanczos on a directory deck.
if qed.has_mpi_build():
    import tempfile
    tmp = tempfile.mkdtemp()
    b.write_directory(tmp, lattice=qed.input.lattice.chain(N))
    qed.mpi.run_distributed(
        tmp, "lanczos", n_ranks=2,
        extra_args=("--max-iter", "200"),
        capture_output=True,
    )

    # 5. (Optional, requires built ./ED on PATH) DSSF static thermal sweep.
    qed.spectral(tmp, T=[0.1, 0.3, 1.0], method="static_thermal")
```

This covers the canonical "single-script, multi-backend" pattern:
build the Hamiltonian once, route it through whichever solver fits
the cluster size and the available hardware, all without touching
the CLI directly.

---

## 8. Where to look for more

* **Authoritative bindings:**
  `python/qed/_bindings/workflow_bindings.cpp` (orchestrator —
  `_core.workflows_{solve,thermal,spectral}`),
  `python/qed/_bindings/qed_bindings.cpp` (`Operator` /
  `FixedSzOperator` / `EDParameters` / `EDResults` / utility types).
* **Capability matrix:** [`python_api_coverage.md`](python_api_coverage.md).
* **CLI counterparts:** [`usage.md`](usage.md) §2 (single-process CLI),
  §3 (MPI launcher), §6 (`./ED dssf`).
* **Tests:** `python/tests/test_workflow.py` exercises `qed.solve` /
  `qed.thermal` end-to-end on a 6-site Heisenberg ring (full Hilbert,
  fixed-Sz, symmetry, combined) and checks ground-state recovery to
  1e-9.


---

## See also — orchestrator abstraction & 32-site recipes

`qed.solve` / `qed.thermal` / `qed.spectral` are the **canonical**
three-verb surface; the page above documents their direct kwargs and
the underlying `EDParameters` carrier. They make the following
auto-decisions for you (Sz projection + guard, solver selection by
`(dim, num_eigenvalues)`, GPU promotion when `has_cuda_build()` and
`dim ≥ 2¹⁷`, MPI shell-out, thermal output-dir bookkeeping,
pre-flight planning) without taking away any of the control surface.

The escape hatches:

* `qed.solve(H, …, extra_params={...})` — sets any niche
  `EDParameters` field after the orchestrator has populated them.
* `qed._core.workflows_{solve,thermal,spectral}(spec, opts)` — drop
  down to the Pybind11 binding for `ed::workflows::*` and supply
  `OperatorSpec` / `SolveOptions` / `ThermalOptions` /
  `SpectralOptions` directly.
* `qed.list_diag_parameters("tpq" / "ftlm" / "ltlm" / "kpm" /
  "thermal" / "general")` — print the catalogue grouped by family.

End-to-end **32-site** worked examples for ground state, FTLM, DSSF,
and mTPQ live in
[`workflow.md`](workflow.md#worked-examples-32-site-spin-ed). The
abstraction layering (`qed.solve` decision tree → C++ `EDParameters`
fields) is documented in
[`workflow.md`](workflow.md#how-qedsolve-abstracts-the-dispatcher).
