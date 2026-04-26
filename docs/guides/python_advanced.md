---
orphan: true
---

# Advanced Python usage (`quantum_ed` Phase 5)

This guide is the **catalogue of advanced patterns** the Python API
supports as of `quantum_ed` 0.2.0 (Phase 5, Apr 2026). It complements the
quickstart at [`python_quickstart.md`](python_quickstart.md) and the
capability matrix at [`python_api_coverage.md`](python_api_coverage.md).

If your task is straightforward (build a Hamiltonian, run Lanczos, look
at a few thermodynamic curves), use the quickstart. If you need:

* A **specific solver** other than Lanczos (BLOCK_LANCZOS, KRYLOV_SCHUR,
  DAVIDSON, LOBPCG, ARPACK_*, IRL, TRL, Chebyshev-filtered, shift-invert,
  TPQ, …)
* The **GPU** path (`LANCZOS_GPU`, `FULL_GPU`, `mTPQ_GPU`, …)
* **ScaLAPACK** for distributed full-dense ED
* **Symmetry projection** in-process (without writing
  `automorphism_results/*.json` to disk first)
* **Streaming-symmetry** ED for the largest tractable clusters
* **MPI distributed** Lanczos / FTLM / TPQ runs from a Python script
* The **full `./ED dssf`** continued-fraction spectral driver
* To **introspect the build** (CUDA / MPI / ScaLAPACK present?) at
  runtime so the same Python script works on a laptop and a cluster

…this is the right document. Every pattern below is exercised by
`python/tests/test_dispatcher.py`.

---

## 0. Choosing the right entry point

There are now **three** entry points into the dispatcher (Phase 7.1
collapsed the symmetry zoo onto a single flag — see below); pick
whichever matches your inputs.

| You have… | …call this | Lives in |
|-----------|------------|----------|
| an in-memory `Operator` (matrix-free) | `exact_diagonalization_core(op, method, params)` | `quantum_ed._core` |
| an in-memory `FixedSzOperator` | `exact_diagonalization_core(fop, method, params)` (overload) | `quantum_ed._core` |
| a directory of `.dat` files (any combination of axes) | `exact_diagonalization_from_directory(dir, method, params, ...)` — set `params.use_symmetry`, `params.use_fixed_sz`, `params.use_gpu`, `params.use_mpi` orthogonally | `quantum_ed._core` |
| an MPI cluster + a directory | `quantum_ed.mpi.run_distributed(dir, method, n_ranks, ...)` | `quantum_ed.mpi` |
| a directory + want S(Q,ω) / S(Q) | `quantum_ed.dssf.run_from_directory(dir, method, ...)` | `quantum_ed.dssf` |

The legacy entry points
`exact_diagonalization_streaming_symmetry[_fixed_sz](...)` and
`exact_diagonalization_*_symmetrized(...)` remain for back-compat but
are no longer the recommended way to request symmetry projection.
Phase 7.1 routes everything through `from_directory(...)` based on the
`params.use_symmetry` flag — the explicit-block `_symmetrized`
variants are now `[[deprecated]]` (slower, no GPU, materialises blocks
on disk).

---

## 1. The single-call CPU dispatcher

```python
import quantum_ed as qed

N = 12
b = qed.input.HamiltonianBuilder(num_sites=N)
b.heisenberg(bonds=[(i, (i + 1) % N) for i in range(N)], J=1.0)
op = b.to_operator()

params = qed.EDParameters()
params.num_eigenvalues = 4
params.max_iterations = 400
params.tolerance = 1e-12

result = qed.exact_diagonalization_core(
    op,
    qed.DiagonalizationMethod.KRYLOV_SCHUR,
    params,
)
print("E0..E3 =", sorted(result.eigenvalues)[:4])
```

`DiagonalizationMethod` mirrors the C++ enum. Every one of these works
with `exact_diagonalization_core` on an `Operator`:

| Family | Methods |
|--------|---------|
| Lanczos | `LANCZOS`, `LANCZOS_SELECTIVE`, `LANCZOS_NO_ORTHO`, `BLOCK_LANCZOS`, `THICK_RESTART_LANCZOS`, `IMPLICIT_RESTART_LANCZOS` |
| Krylov-Schur | `KRYLOV_SCHUR`, `BLOCK_KRYLOV_SCHUR` |
| Davidson / LOBPCG | `DAVIDSON`, `LOBPCG` |
| Filter / shift | `CHEBYSHEV_FILTERED`, `SHIFT_INVERT`, `SHIFT_INVERT_ROBUST` |
| Iterative linear systems | `BICG` |
| ARPACK | `ARPACK_SM`, `ARPACK_LM`, `ARPACK_SHIFT_INVERT`, `ARPACK_ADVANCED` |
| Dense | `FULL`, `OSS` (and `SCALAPACK` / `SCALAPACK_MIXED` when `qed.has_scalapack_build()`) |
| Thermal | `FTLM`, `LTLM`, `HYBRID`, `mTPQ`, `cTPQ` |

`EDParameters` exposes every knob (block size, ARPACK NCV, FTLM Krylov
dim, TPQ Taylor order, ScaLAPACK process grid, output directory…) as
read/write Python attributes. `repr(params)` summarises the headline
fields.

**Returns.** `EDResults` with `eigenvalues`, `eigenvectors_computed`,
`eigenvectors_path`, `thermo_data` (a `ThermodynamicData` instance with
`temperatures`, `energy`, `specific_heat`, `entropy`, `free_energy`).
Call `.to_dict()` on either object for ergonomic JSON / pickle
serialization.

### 1.1 The `FixedSzOperator` overload

`FixedSzOperator` inherits from `Operator`, so `HamiltonianBuilder.emit_into`
accepts it directly -- the terms land in the chosen Sz sector with no
re-implementation:

```python
N = 14
b = qed.input.HamiltonianBuilder(num_sites=N)
b.heisenberg(bonds=[(i, (i + 1) % N) for i in range(N)], J=1.0)

# Restrict to total Sz=0 sector.
fop = qed.FixedSzOperator(num_sites=N, n_up=N // 2, spin=0.5)
b.emit_into(fop)                     # populate the fixed-Sz operator

params = qed.EDParameters()
params.num_eigenvalues = 1
result = qed.exact_diagonalization_core(
    fop, qed.DiagonalizationMethod.LANCZOS, params,
)
```

If you prefer building term-by-term (the path
`python/tests/test_dispatcher.py` exercises), `add_two_body` /
`add_one_body` / `add_three_body` are inherited from `Operator`:

```python
for i in range(N):
    j = (i + 1) % N
    fop.add_two_body(qed.OP_SZ,     i, qed.OP_SZ,     j, 1.0 + 0.0j)
    fop.add_two_body(qed.OP_SPLUS,  i, qed.OP_SMINUS, j, 0.5 + 0.0j)
    fop.add_two_body(qed.OP_SMINUS, i, qed.OP_SPLUS,  j, 0.5 + 0.0j)
```

The dispatcher detects the `FixedSzOperator` subclass and routes
through the fixed-Sz code paths automatically.

---

## 2. Symmetry projection in-process

The `quantum_ed.symmetry` DSL builds the same symmetry-group dict the C++
engine consumes, and Phase 5 lets you attach it directly to an
`Operator` without going through `automorphism_results/*.json`.

```python
import quantum_ed as qed

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

For an actual symmetry-projected solve, the streaming engine reads the
directory deck and (when present) the `automorphism_results/*.json`
tree that `edlib.automorphism_finder` writes -- this is still the
canonical entry point because it survives across Python ↔ CLI
boundaries:

```python
b.write_directory("/tmp/heisenberg-6-chain", lattice=qed.input.lattice.chain(N))

# Run the legacy finder once to materialise automorphism_results/.
import subprocess, sys
subprocess.run([
    sys.executable, "-m", "edlib.automorphism_finder",
    "--data_dir", "/tmp/heisenberg-6-chain",
], check=True)

params = qed.EDParameters()
params.num_sites = N
params.num_eigenvalues = 4
# Phase 7.1: symmetry projection is the 5th orthogonal axis. Setting
# this flag is *the* way to request symmetry-projected ED -- the
# dispatcher routes through the streaming kernel internally, which is
# the only path that supports GPU per-sector dispatch and avoids
# materialising the orbit basis on disk.
params.use_symmetry = True

result = qed.exact_diagonalization_from_directory(
    "/tmp/heisenberg-6-chain",
    qed.DiagonalizationMethod.LANCZOS,
    params,
)
print("symmetry-projected E0..E3 =", sorted(result.eigenvalues)[:4])
```

The legacy entry point `exact_diagonalization_streaming_symmetry(...)`
still works and routes through the same kernel — but new code should
prefer the orthogonal-flag form: it composes cleanly with
`params.use_fixed_sz`, `params.use_gpu`, and `params.use_mpi` without
needing a different function name per combination.

---

## 3. GPU per-sector dispatch via streaming symmetry

When the project was built with `WITH_CUDA=ON` (check
`qed.has_cuda_build()`), every GPU-flavoured `DiagonalizationMethod`
value reaches the appropriate CUDA kernel through the
streaming-symmetry or directory dispatcher:

```python
import quantum_ed as qed

if not qed.has_cuda_build():
    print("This wheel was built without CUDA; GPU methods will run on CPU "
          "with a runtime warning.")

params = qed.EDParameters()
params.num_sites = 24
params.num_eigenvalues = 4
params.tolerance = 1e-10

# Phase 7.1 canonical form: pick the *algorithm* on `method`, then
# turn on the orthogonal axes via flags. All four flags compose:
params.use_symmetry  = True   # symmetry projection (streaming kernel)
params.use_gpu       = True   # per-sector GPU dispatch
params.use_fixed_sz  = False  # full Hilbert space (set True + n_up to restrict)
params.use_mpi       = False  # single-process

# Per-sector GPU Lanczos for a 24-site cluster with C2 + translations.
result = qed.exact_diagonalization_from_directory(
    directory="/scratch/runs/kagome-24",
    method=qed.DiagonalizationMethod.LANCZOS,       # or BLOCK_LANCZOS,
                                                    # KRYLOV_SCHUR,
                                                    # BLOCK_KRYLOV_SCHUR,
                                                    # DAVIDSON, LOBPCG
    params=params,
)
print("GPU per-sector eigenvalues:", sorted(result.eigenvalues)[:4])
```

Equivalently — and this is the only form pre-Phase-7 code knows about
— pass the deprecated combined enum value and leave the flags
default-false:

```python
result = qed.exact_diagonalization_from_directory(
    directory="/scratch/runs/kagome-24",
    method=qed.DiagonalizationMethod.LANCZOS_GPU,   # legacy spelling
    # canonicalize_method() collapses LANCZOS_GPU -> (LANCZOS, use_gpu=True);
    # the symmetry flag still has to be set explicitly though.
    params=qed.EDParameters(use_symmetry=True),
)
```

The same dispatcher knows about the fixed-Sz × symmetry path —
Phase 7.1 canonical form: turn on **three** orthogonal flags on top
of plain `LANCZOS`:

```python
params = qed.EDParameters()
params.num_sites    = 24
params.use_symmetry = True
params.use_gpu      = True
params.use_fixed_sz = True
params.n_up         = 12

result = qed.exact_diagonalization_from_directory(
    directory="/scratch/runs/kagome-24",
    method=qed.DiagonalizationMethod.LANCZOS,    # algorithm only
    params=params,
)
```

The legacy `LANCZOS_GPU_FIXED_SZ` enum value still works (it gets
canonicalized to `LANCZOS` + `use_gpu=True` + `use_fixed_sz=True` on
entry) and the legacy
`exact_diagonalization_streaming_symmetry_fixed_sz(...)` entry point
still routes through the same kernel — but new code should prefer the
orthogonal-flag form on the canonical `from_directory` dispatcher.

For the **full-Hilbert-space** GPU methods (`mTPQ_GPU`, `cTPQ_GPU`,
`FULL_GPU`, `FTLM_GPU`, `LANCZOS_GPU` with no symmetry projection)
use the directory dispatcher:

```python
result = qed.exact_diagonalization_from_directory(
    directory="/scratch/runs/heisenberg-20-chain",
    method=qed.DiagonalizationMethod.mTPQ_GPU,
    params=params,
    interaction_filename="InterAll.dat",
    single_site_filename="Trans.dat",
)
```

The directory dispatcher constructs a `GPUOperator` from the .dat files
and routes through `GPUEDWrapper::runGPULanczos` / `runGPUMicrocanonicalTPQ`
/ `runGPUFullDiag` etc. — exactly the same kernels `./ED` would call
with the same flags.

---

## 4. MPI distributed solvers

`quantum_ed.mpi.run_distributed` is a thin Python wrapper that locates
`mpiexec` (or your launcher of choice — `srun`, `mpirun`,
`jsrun`, …) and `ed_distributed_main`, then runs the right argv:

```python
import quantum_ed as qed
from quantum_ed import mpi as qed_mpi

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
from quantum_ed import dssf

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
`quantum_ed.dssf.build_observable_pairs(...)` — the
`run_from_directory` helper is for the *full pipeline* (operator
assembly + continued fractions + HDF5 output trees).

---

## 6. Build introspection

```python
import quantum_ed as qed

print("CUDA build:    ", qed.has_cuda_build())
print("MPI build:     ", qed.has_mpi_build())
print("ScaLAPACK build:", qed.has_scalapack_build())

if not qed.has_cuda_build():
    method = qed.DiagonalizationMethod.LANCZOS
else:
    method = qed.DiagonalizationMethod.LANCZOS_GPU
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
import quantum_ed as qed

N = 16
b = qed.input.HamiltonianBuilder(num_sites=N)
b.heisenberg(bonds=[(i, (i + 1) % N) for i in range(N)], J=1.0)

# 1. CPU ground state via Krylov-Schur.
op = b.to_operator()
params = qed.EDParameters()
params.num_eigenvalues = 6
params.max_iterations = 500
params.tolerance = 1e-12
gs = qed.exact_diagonalization_core(
    op, qed.DiagonalizationMethod.KRYLOV_SCHUR, params,
)
print("CPU low-energy spectrum:", sorted(gs.eigenvalues))

# 2. FTLM thermodynamics (same dispatcher, just a different method).
fparams = qed.EDParameters()
fparams.num_samples = 32
fparams.ftlm_krylov_dim = 80
fparams.temp_min = 0.05
fparams.temp_max = 5.0
fparams.num_temp_bins = 80
ftlm = qed.exact_diagonalization_core(
    op, qed.DiagonalizationMethod.FTLM, fparams,
)
T = ftlm.thermo_data.temperatures
Cv = ftlm.thermo_data.specific_heat
print("FTLM Cv at T=1:", Cv[len(T) // 2])

# 3. Symmetry-projected: write a directory deck and call streaming.
import tempfile, subprocess, sys
tmp = tempfile.mkdtemp()
b.write_directory(tmp, lattice=qed.input.lattice.chain(N))

# Materialise automorphism_results/ via the legacy finder (one-off).
subprocess.run(
    [sys.executable, "-m", "edlib.automorphism_finder", "--data_dir", tmp],
    check=True,
)

sparams = qed.EDParameters()
sparams.num_sites = N
sparams.num_eigenvalues = 6
proj = qed.exact_diagonalization_streaming_symmetry(
    tmp, qed.DiagonalizationMethod.LANCZOS, sparams,
)
print("symmetry-projected spectrum:", sorted(proj.eigenvalues))

# 4. (Optional, requires WITH_MPI) MPI distributed Lanczos on the same deck.
if qed.has_mpi_build():
    qed.mpi.run_distributed(
        tmp, "lanczos", n_ranks=2,
        extra_args=("--max-iter", "200"),
        capture_output=True,
    )

# 5. (Optional, requires built ./ED on PATH) DSSF static thermal sweep.
qed.dssf.run_from_directory(tmp, "static_thermal", capture_output=True)
```

This covers the canonical "single-script, multi-backend" pattern:
build the Hamiltonian once, route it through whichever solver fits
the cluster size and the available hardware, all without touching
the CLI directly.

---

## 8. Where to look for more

* **Authoritative bindings:**
  `python/quantum_ed/_bindings/dispatcher_bindings.cpp` (Phase 5),
  `python/quantum_ed/_bindings/quantum_ed_bindings.cpp`
  (legacy thin wrappers).
* **Capability matrix:** [`python_api_coverage.md`](python_api_coverage.md).
* **CLI counterparts:** [`usage.md`](usage.md) §2 (single-process CLI),
  §3 (MPI launcher), §6 (`./ED dssf`).
* **Tests:** `python/tests/test_dispatcher.py` exercises every entry
  point listed in §0 above on a 6-site Heisenberg ring and checks
  ground-state recovery to 1e-5.
