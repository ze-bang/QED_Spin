# Stress-free workflow (`qed.solve` + `qed.find_symmetries`)

This page documents the recommended Python entry point for new code:
the unified `qed.workflow` API. It collapses the
"build → discover symmetries → pick sector → diagonalise" pipeline into
two function calls (`find_symmetries` and `diag`) with smart defaults
that match what an experienced ED user would tune by hand.

If you want the lower-level dispatcher with explicit method/parameter
control (FTLM, LTLM, ScaLAPACK, GPU streaming symmetry, etc.), see
[`python_advanced.md`](python_advanced.md). The new workflow is built
on top of the same C++ kernels — picking it up is purely a matter of
how much knob-twiddling you want to do yourself.

---

## TL;DR

```python
import qed as qed

# 1. Build a Hamiltonian.
N = 12
H = (qed.input.HamiltonianBuilder(N)
        .heisenberg([(i, (i + 1) % N) for i in range(N)], J=1.0)
        .to_operator())

# 2. (Optional) inspect what symmetries the engine sees.
report = qed.find_symmetries(H, verbose=False)
print(report.summary())   # tells you about U(1) Sz + lattice automorphisms

# 3. One-liner ground state.
print(qed.solve(H).eigenvalues[0])               # -5.387390917...

# 4. Bottom 4 eigenvalues in the Sz=0 sector.
print(qed.solve(H, num_eigenvalues=4, sz=N // 2).eigenvalues)

# 5. Bottom of the spectrum projected onto the full automorphism group +
#    Sz=0 sector. symmetry="auto" runs the search internally (pass a
#    GeneratorSet to pick a subgroup yourself); the spin-flip and
#    time-reversal mechanisms auto-compose on top, with per-call
#    toggles spin_flip= / time_reversal= in {"auto","on","off","require"}
#    ("on" = exploit + report, warn-and-continue when H lacks it).
print(qed.solve(H, num_eigenvalues=4,
               symmetry="auto",
               sz=N // 2).eigenvalues)

# 6. Memory safety: there is no pre-flight planner. If the dominant
#    allocation would not fit in available RAM, the workflow raises a
#    clean error (naming the workflow + estimated vs available bytes)
#    instead of OOM-killing the process. Shrink dim with sz + symmetry.
#    Bypass with ED_MEM_GUARD_OFF=1.
qed.solve(H_big, solver="FTLM", sz=16, symmetry="auto")
```

That's it. Everything below explains the knobs that the function picks
for you.

---

## Step 1 — Build the Hamiltonian

`qed.input.HamiltonianBuilder` is the fluent C++-backed builder
(see `python/qed/input.py`). The same model can also be loaded
from disk (`Operator()` + `op.load_inter_all` / `op.load_trans`) or
constructed by hand with `op.add_one_body` / `add_two_body` /
`add_three_body`. All three paths land in the same `Operator`
in-memory representation, so the workflow API works on whichever you
prefer.

```python
H = (qed.input.HamiltonianBuilder(num_sites=N)
        .heisenberg(bonds, J=1.0)        # SzSz + 0.5*(S+S- + S-S+)
        .zeeman_per_site(h_site_array)   # optional
        .to_operator())
```

## Step 2 — Inspect symmetries with `find_symmetries`

> **Background — orthogonal symmetry composition (May 2026).** Once
> you start passing both `sz=` and `symmetry=` you are using the
> orthogonal `(Subspace, ProjectorChain)` decomposition under the
> hood: `sz=` picks the `Subspace` (`FullSpaceSubspace` vs
> `FixedSzSubspace`), `symmetry=` populates the chain with a
> `SpatialProjector`. The four cells (`none` / `Sz` / `Symm` /
> `Sz+Symm`) are the Cartesian product, and future axes (spin-flip
> Z_2, time-reversal antiunitary, SU(2) total-S) extend the chain
> without breaking this signature. The internals are documented in
> [`docs/architecture/SYMMETRY.md`](../architecture/SYMMETRY.md) §6;
> the headers are in
> [`include/ed/symmetry/`](../../include/ed/symmetry/).


`find_symmetries(H, *, lattice=None, translation_only=False,
verbose=True) -> SymmetryReport` runs the colored-graph automorphism
pipeline (the same one that powers `automorphism_finder.py` on disk)
on `H` directly — no temp files, no CLI. It returns a
`SymmetryReport` with three pieces of information:

1. **U(1) Sz status.** `report.has_u1_sz` is `True` iff every term in
   the Hamiltonian commutes with total Sz, in which case
   `report.sz_sectors` is the list `[(n_up, dim), ...]`. The CLI
   would never tell you this — `qed.solve` will refuse a `sz=` argument
   on an operator that breaks Sz, with an actionable error.

2. **Generator sets.** `report.generator_sets` is a list of named
   `GeneratorSet` candidates. There is always a `"trivial"` entry
   (no symmetry projection); when the operator has non-trivial
   automorphisms, you also get `"full_automorphism"` (the largest
   commuting subgroup discovered by the colored-graph search). If you
   pass `lattice=` the report additionally exposes a `"translation"`
   entry that filters the automorphisms to pure lattice translations.

3. **Convenience attributes.** `report.full_set`, `report.translation_set`
   and `report.trivial_set` are short-hand pointers into
   `generator_sets`, suitable for direct passing as `symmetry=` to
   `diag`.

`report.summary()` returns a human-readable rendering. The default
`SymmetryReport.__repr__` is the same string, so you can just
`print(report)`.

```text
SymmetryReport(num_sites=6, has_u1_sz=True)

  U(1) Sz is conserved.  Available sectors (n_up: dimension):
    sz=  0   dim=1
    sz=  1   dim=6
    sz=  2   dim=15
    sz=  3   dim=20
    sz=  4   dim=15
    sz=  5   dim=6
    sz=  6   dim=1
  -> pass `sz=<n_up>` to qed.solve(...) to restrict to a sector.

  Generator sets (2):
    [             trivial]  group_size=   1  |generators|= 0   orders=[]
      No symmetry projection (full Hilbert space).
    [   full_automorphism]  group_size=   6  |generators|= 2   orders=[2, 3]
      Largest abelian subgroup of the lattice + Hamiltonian automorphism group.

  -> pass any GeneratorSet (or list[Permutation]) as `symmetry=...` to qed.solve(...).
```

You can also pass your own permutations to `qed.solve` as the
`symmetry=` argument; `find_symmetries` is purely advisory:

```python
T = [(i + 1) % N for i in range(N)]      # site i -> i+1 mod N
my_set = qed.GeneratorSet(name="Z6_T", description="cyclic translation",
                          generators=[T], orders=[N], group_size=N)
qed.solve(H, symmetry=my_set)
```

### Picking a subset of `full_set`

`report.full_set` typically has more than one generator (the example
above has `orders=[2, 3]` — a reflection and an order-3 rotation, whose
direct product is the full Z6 lattice symmetry of the ring). You rarely
want to use them all if you also care about wall-clock cost: a smaller
subgroup means fewer sectors and bigger blocks per solve. Two ergonomic
ways to pick a subgroup:

```python
# 1. Index/slice a GeneratorSet directly.
rot_only  = report.full_set[1]            # only generator #1 (order 3)
refl_only = report.full_set[0]            # only generator #0 (order 2)
first_two = report.full_set[:2]           # GeneratorSet with both gens

# 2. Or use .subgroup(indices) for explicit lists.
custom    = report.full_set.subgroup([1])
combo     = report.full_set.subgroup([0, 1])

# Either way, pass the sub-GeneratorSet straight into qed.solve(...):
eigs = qed.solve(H, num_eigenvalues=4, symmetry=rot_only).eigenvalues
```

For convenience, `find_symmetries` also automatically appends one
single-generator entry per generator into `report.generator_sets`, so
you can browse them by name without calling `subgroup()` yourself:

```text
Generator sets (4):
  [             trivial]   group_size=  1   ...
  [   full_automorphism]   group_size=  6   orders=[2, 3]   ...
  [full_automorphism[0]]   group_size=  2   orders=[2]      ...
  [full_automorphism[1]]   group_size=  3   orders=[3]      ...
```

```python
just_rotation = report.get("full_automorphism[1]")
qed.solve(H, symmetry=just_rotation)
```

The returned subgroup is always a fresh `GeneratorSet` whose
`group_size` is the product of the selected generators' orders
(correct because the parent `full_set` came from a minimal-generator
decomposition of an abelian group, so any subset is still independent).

## Step 3 — Diagonalise with `diag`

```python
qed.solve(H,
         num_eigenvalues=1,        # how many eigenpairs to converge
         tolerance=1e-10,          # solver convergence tolerance
         compute_eigenvectors=False,
         solver=None,              # auto: FULL / LANCZOS / KRYLOV_SCHUR
         device=None,              # auto / cpu / gpu / mpi / mpi_gpu
         symmetry=None,            # GeneratorSet | list[perm] | dict | None
         sector=None,              # restrict to one irrep when symmetry= set
         sz=None,                  # restrict to Sz=sz when conserved
         output_dir="",            # HDF5 output dir (empty = no writes)
         max_iterations=None,      # auto-tuned from num_eigenvalues
         block_size=None,          # only used by BLOCK_* solvers
         # Thermal-method shortcuts (mTPQ / cTPQ / FTLM / LTLM):
         num_samples=None, target_beta=None, num_temp_points=None,
         temp_min=None, temp_max=None,
         # (no pre-flight planner: a memory guard catches over-budget
         #  allocations at the point of use; ED_MEM_GUARD_OFF=1 bypasses)
         verbose=True,
         extra_params=None)        # forwarded to EDParameters as setattr
```

### Smart defaults

* **Krylov sizes.** `max_iterations` defaults to
  `max(200, 8*num_eigenvalues + 80)` (capped by the sector dimension
  minus one). These are the constants from the `bench_vs_xdiag`
  bake-off and converge the requested eigenvalues to `tolerance` with
  no further tuning at the sizes Python typically reaches in-process.
* **Solver.** `solver=None` picks the best backend for the matrix
  shape: `FULL` for dimensions ≤ 1024 (LAPACK is end-to-end faster
  than a Lanczos warmup at that scale), `KRYLOV_SCHUR` when
  `num_eigenvalues ≥ 16`, and `LANCZOS` otherwise. Pass an explicit
  enum value (`qed.DiagonalizationMethod.BLOCK_LANCZOS`, etc.) or its
  string name to override.
* **Device.** `device=None` uses GPU iff
  `qed.has_cuda_build()` is true and the matrix is large
  enough for cuSPARSE matvec to amortize H2D / D2H (rule of thumb:
  dim ≥ 2¹⁴). Pass `"cpu"` / `"gpu"` to force a backend.
  (`"mpi"` / `"mpi_gpu"` -- the retired subprocess launcher -- raise
  with guidance; MPI runs go through the CLI under `mpirun`, see
  "MPI jobs" below.)
* **Sz axis.** When `sz=` is passed and `H` is an `Operator`,
  `diag` materialises a `FixedSzOperator` from `H` for you (via the
  new `Operator.make_fixed_sz` binding) and runs the in-memory
  fixed-Sz kernel. When `H` is already a `FixedSzOperator`,
  `sz=` is only used as a sanity check.
* **Symmetry projection.** When `symmetry=` is passed,
  `qed.solve` writes the operator + symmetry metadata to a temp
  directory and invokes the unified
  `_core.workflows_solve_streaming_symmetry_directory` binding, which
  composes `ed::make_streaming_symmetry_operator(spec)` with a
  per-sector `ed::workflows::solve` loop. All accepted forms —
  `GeneratorSet`, raw `list[list[int]]` of permutations, or the dict
  produced by `qed.symmetry.group_from_generators` — are normalised to
  the same on-disk schema.
* **Combined Sz + symmetry.** Pass both `sz=` and `symmetry=`. The
  streaming-symmetry-fixed-Sz kernel handles the joint projection.

### What you get back

`qed.solve(...)` returns an `EDResults` object with:

* `eigenvalues` — sorted lowest-first.
* `eigenvectors` (when `compute_eigenvectors=True`) — packed as a
  numpy array of shape `(num_eigenvalues, dim)`.
* `eigenvectors_path` — when `output_dir` is non-empty, the HDF5
  file the solver wrote into. Useful for very large vectors that
  you don't want pulled back across the C++/Python boundary.

### Specifying simulation parameters

`qed.solve(...)` exposes the most common knobs as first-class keyword
arguments (listed in the signature above): `num_eigenvalues`,
`tolerance`, `compute_eigenvectors`, `solver`, `device`, `symmetry`,
`sector`, `sz`, `output_dir`, `max_iterations`, `block_size`, and
`verbose`. Anything else on `EDParameters` — FTLM / LTLM / TPQ /
KPM-DOS / observable settings — is reachable via `extra_params={...}`:

```python
res = qed.solve(
    H,
    solver="FTLM",
    sz=N // 2,
    num_samples=2,
    extra_params={
        "ftlm_krylov_dim": 80,
        "ftlm_full_reorth": True,
        "ftlm_seed": 1234,
    },
)
```

To discover what's available, call:

```python
qed.list_diag_parameters()         # full catalogue, grouped by category
qed.list_diag_parameters("ftlm")   # filter to one category (substring OK)
qed.list_diag_parameters(return_dict=True)   # programmatic access
```

The catalogue is bucketed into:

| category      | what it controls                                     |
| ------------- | ---------------------------------------------------- |
| `general`     | `num_eigenvalues`, `tolerance`, eigenvector toggles  |
| `krylov`      | Lanczos / Krylov-Schur subspace shape (`block_size`, ...) |
| `device`      | The orthogonal axes: `use_gpu`, `use_mpi`, `use_symmetry`, `use_fixed_sz` |
| `ftlm`        | Finite-Temperature Lanczos                           |
| `ltlm`        | Low-Temperature Lanczos                              |
| `tpq`         | Thermal Pure Quantum / mTPQ / cTPQ                   |
| `kpm`         | Kernel Polynomial Method DOS / thermodynamics        |
| `thermal`     | Temperature-grid post-processing                     |
| `observables` | Spectral / dynamical observables (`omega_*`, `dt`)   |
| `lattice`     | Lattice metadata                                     |

If you typo a key, the resulting `AttributeError` points back at
`list_diag_parameters()` so you can grep for the right name.

---

## Solver × path support matrix

`qed.solve` exposes every backend the C++ dispatcher knows about, but
not every (solver, basis) combination is meaningful. The table below
records what works through the unified Python entry point. "Path"
refers to the four basis choices the workflow can compose:

* **full** — the full 2ᴺ Hilbert space.
* **sz** — fixed-Sz block (pass `sz=`).
* **symm** — symmetry-projected basis (pass `symmetry=`).
* **symm + sz** — both at once.

| solver family             | full | sz | symm | symm + sz | returns                                |
| ------------------------- | :--: | :-: | :--: | :-------: | -------------------------------------- |
| `LANCZOS`                 | ✅ | ✅ | ✅ | ✅ | `EDResults.eigenvalues` |
| `KRYLOV_SCHUR`            | ✅ | ✅ | ✅ | ✅ | `EDResults.eigenvalues` |
| `BLOCK_LANCZOS`           | ✅ | ✅ | ✅ | ✅ | `EDResults.eigenvalues` |
| `FULL` (dense LAPACK)     | ✅ | ✅ | ✅ | ✅ | `EDResults.eigenvalues` |
| `mTPQ` / `cTPQ`           | ✅ | ✅ | ❌¹ / ✅³ | ❌¹ / ✅³ | trajectory in `eigenvalues`; thermo curve in `output_dir`/`thermo_data` |
| `FTLM`                    | ✅ | ✅ | ✅² | ✅² | `EDResults.thermo_data` (sectors are summed) |
| `LTLM`                    | ✅ | ✅ | ✅² | ✅² | `EDResults.thermo_data` |
| `KPM_DOS`                 | ✅ | ✅ | ✅² | ✅² | `EDResults.thermo_data` |

(The May-2026 minimalist-solver-matrix cleanup retired the
`ARPACK_*` / `DAVIDSON` / `LOBPCG` / `CHEBYSHEV_FILTERED` /
`SHIFT_INVERT*` / `IRL` / `TRL` / `BICG` / `OSS` / `SCALAPACK*` /
`HYBRID` families and every `_GPU` / `_MPI` enum suffix; backend
choice now goes through the `device=` kwarg.)

Notes:

¹ TPQ acts on a single random vector spread across the whole
sector; projecting onto each symmetry irrep destroys the Z
normalisation, so `solver='mTPQ' + symmetry=` raises a clear
`ValueError`. Pre-project to a fixed-Sz block instead (`sz=`), or
use FTLM/LTLM.

² FTLM/LTLM/KPM_DOS *do* combine across symmetry blocks correctly
because each block contributes an additive term to the partition
function; the dispatcher loops the sectors itself (and under
`mpirun`, SectorDistributor spreads the sectors across ranks).

```python
# Eigenvalue solver, all four paths:
qed.solve(H, solver="LANCZOS")                                 # full
qed.solve(H, solver="LANCZOS", sz=N // 2)                      # sz
qed.solve(H, solver="LANCZOS", symmetry=report.full_set)       # symm
qed.solve(H, solver="LANCZOS",
         symmetry=report.full_set, sz=N // 2)                 # symm + sz

# Thermal solver — case-insensitive name lookup:
res = qed.solve(H, solver="mTPQ", sz=N // 2,
               num_samples=8, target_beta=20.0,
               num_temp_points=40,
               output_dir="ed_runs/12site_thermal")
# res.eigenvalues  -> per-step / per-sample energy summary
# files in output_dir:
#   SS_rand*.dat   -> raw imaginary-time trajectories
#   ed_results.h5  -> unified E(T), C(T), S(T), F(T)
```

### Thermal-method first-class kwargs

| kwarg              | EDParameters slot                  | default | meaning                                       |
| ------------------ | ---------------------------------- | ------- | --------------------------------------------- |
| `num_samples`      | `num_samples`                      | 1       | random initial vectors averaged over          |
| `target_beta`      | `tpq_target_beta`                  | 1000    | lowest β = 1/T (TPQ only)                     |
| `num_temp_points`  | `tpq_num_measure_points` + `num_temp_bins` | 20 / 200 | size of the output thermodynamic grid |
| `temp_min`         | `temp_min` + `tpq_measure_beta_max` | 0.01   | lowest temperature in the output grid        |
| `temp_max`         | `temp_max` + `tpq_measure_beta_min` | 100.0  | highest temperature in the output grid        |

For everything else (delta_beta, Taylor order, observable list, …)
fall through to `extra_params={...}` and look it up via
`qed.list_diag_parameters("tpq")`.

If `output_dir=""` (the default) and the chosen solver is thermal, the
workflow auto-mints a fresh `./qed_thermal_<METHOD>_<timestamp>/`
directory and prints its path so the trajectory data isn't silently
lost. Pass an explicit `output_dir=` to keep artefacts where you want
them.

---

## Solver × device support matrix

Orthogonal to the basis (full / sz / symm / symm+sz) is the **device
axis**: where the matrix-vector products execute. Two in-process cells
remain — single-process CPU and single-GPU. (The distributed-operator
family and its `device='mpi'`/`'mpi_gpu'` subprocess launcher were
retired in Stage 11d, Jul 2026; see "MPI jobs" below for how MPI works
now.)

| solver family            |  cpu  |  gpu  | how to invoke                                        |
| ------------------------ | :---: | :---: | ---------------------------------------------------- |
| `LANCZOS`                |  ✅   |  ✅   | `qed.solve(H[, device='gpu'])`                        |
| `BLOCK_LANCZOS`          |  ✅   |  ✅   | `qed.solve(H, solver="BLOCK_LANCZOS"[, device='gpu'])` |
| `KRYLOV_SCHUR`           |  ✅   |  ✅   | `qed.solve(H, solver="KRYLOV_SCHUR"[, device='gpu'])` |
| `FULL`                   |  ✅   |  ✅   | `qed.solve(H, solver="FULL"[, device='gpu'])`         |
| `mTPQ`                   |  ✅   |  ✅   | `qed.thermal(H, method="mTPQ"[, device='gpu'])` (fp32 GPU lane via `tpq_fp32=True`) |
| `cTPQ`                   |  ✅   |  ✅   | `qed.thermal(H, method="cTPQ")`                       |
| `FTLM`                   |  ✅   |  ✅   | `qed.thermal(H, method="FTLM")`                       |
| `LTLM`                   |  ✅   |  ❌   | `qed.thermal(H, method="LTLM")`                       |
| `KPM_DOS`                |  ✅   |  ✅   | `qed.thermal(H, method="KPM_DOS"[, device='gpu'])`    |

### Path × device — cross-product caveats

The path matrix (`full` / `sz` / `symm` / `symm + sz`) and the device
matrix (`cpu` / `gpu`) are **almost** orthogonal:

* `full` and `sz` work on both devices for every solver that is ✅ in
  the table above.
* `symm` (symmetry-projected basis) and `symm + sz` work on **cpu**
  and **gpu** for every solver that supports them — the projection
  happens in the basis (CSR-free rep kernel on both devices), so the
  Lanczos / thermal kernel is none the wiser. The little-group
  projection lane (`point_group='auto'`) is CPU-side; an explicit
  `device='gpu'` request routes through the abelian rep lane's GPU
  mirror instead.
* `mTPQ`/`cTPQ` × `symmetry=` raises (footnote ¹ above): use `sz=`.

### Build-aware introspection

The matrix above is a static record of what kernels exist in C++. To
ask the *current build* whether a given (solver, device) cell is
reachable:

```python
qed.solver_device_support()                  # prints the table
m = qed.solver_device_support(return_dict=True)
m["LANCZOS"]["mpi"]
# {'kernel': False, 'available': False,
#  'note': 'retired (Stage 11d): run the CLI under mpirun -- '
#          'SectorDistributor + MpiBackend'}

qed.solver_device_support(solver="lanczos")  # filter to one solver family
```

Each cell carries `kernel` (does the C++ side have the code at all),
`available` (is *this* build wired for it), and a `note` string with
the rebuild flag to flip when `available=False`.

## No pre-flight planner — sensible defaults + a memory guard

There is **no** pre-flight planner — `qed.estimate_resources`,
`qed.suggest_workflow`, `qed.ResourceError`, and the `plan` / `dry_run` /
`force` arguments were all removed in favour of **sensible defaults**.
`qed.solve` / `qed.thermal` / `qed.spectral` pick the method from the problem
size (full diagonalization for `dim ≤ 1024`, Lanczos otherwise) and the
representation from static leaf-policy hooks (reduced-CSR symmetry matvec by
default; override with `ED_SYM_REDUCED_CSR` / `ED_SYM_REP`).

Instead of *predicting* feasibility, the orchestrator **guards the actual
allocation**: `ed::core::guard_working_set` checks the dominant working set
against available RAM (`/proc/meminfo` `MemAvailable`) immediately before it is
allocated and throws a clear error — naming the workflow and the estimated vs
available bytes — rather than letting the host OOM-kill the process. Set
`ED_MEM_GUARD_OFF=1` to bypass it (e.g. when the scheduler reserved more than
`/proc/meminfo` reports).

To size a job yourself: a dim-sized complex vector is `dim · 16 B`; Lanczos
keeps a handful live (plus the `m`-vector basis when `compute_eigenvectors=True`),
FTLM / TPQ keep O(few) per sample. Cut `dim` with `sz=N//2` (≈10× at half
filling) and `symmetry=qed.find_symmetries(H).full_set` (≈`|G|×`); the
CSR-free rep lane keeps the basis memory at O(#reps).

---

### Choosing a device

* `device=None` (the default) lets the workflow pick: CPU for small
  problems, single GPU when `WITH_CUDA=ON` and the dimension is large
  enough that PCIe transfer amortizes (current threshold: `dim ≥ 2¹⁴`).
* `device='cpu'` / `device='gpu'` force the choice. The GPU path
  routes through `_core.workflows_solve` with `OperatorSpec.distributed
  = false` and the GPU lane enabled in `BackendConstraints`; the
  orchestrator builds a `GPUOperator` via the same factory the
  CLI uses.
* `device='mpi'` / `device='mpi_gpu'` raise: the subprocess launcher
  was retired in Stage 11d. MPI runs go through the CLI under
  `mpirun` (see "MPI jobs" below).

---

## Recipes

### Ground state of an arbitrary spin Hamiltonian

```python
H = (qed.input.HamiltonianBuilder(N)
        .heisenberg(bonds, J=1.0)
        .to_operator())
e0 = qed.solve(H).eigenvalues[0]
```

### Bottom-of-spectrum within a target Sz sector

```python
report = qed.find_symmetries(H, verbose=False)
assert report.has_u1_sz, "Sz is not conserved -- pick a different model"
res = qed.solve(H, num_eigenvalues=4, sz=N // 2)
```

### Maximum projection: full automorphism group + fixed Sz

```python
report = qed.find_symmetries(H, lattice=lat, verbose=False)
res = qed.solve(H,
               num_eigenvalues=8,
               symmetry=report.full_set,
               sz=N // 2,
               compute_eigenvectors=True,
               output_dir="ed_runs/12site_J1")
```

### Force GPU even on a small problem (e.g. for benchmarking)

```python
res = qed.solve(H, device="gpu")
```

### Thermal trajectory with mTPQ

```python
res = qed.solve(H,
               solver="mTPQ",
               sz=N // 2,                # optional: project onto an Sz block
               num_samples=8,
               target_beta=20.0,
               num_temp_points=40,
               output_dir="ed_runs/12site_thermal")
# res.eigenvalues -- per-step / per-sample energy summary
# Look in output_dir for SS_rand*.dat (raw) and ed_results.h5
# (unified thermodynamic curve).
```

### MPI jobs

The `device='mpi'` subprocess launcher (`ed_distributed_main` +
`qed.mpi.run_distributed`) and the distributed-operator family behind
it were retired in Stage 11d (Jul 2026). MPI is now ONE story, driven
from the CLI:

```bash
# Across-sector MPI: each rank owns a dim-balanced subset of the
# symmetry sectors (SectorDistributor; Burnside-weighted greedy
# packing) and solves them rank-locally. Engages automatically for
# symmetry workloads under mpirun:
mpiexec -n 8 ./ED <input_dir> --use-symmetry --fixed-sz ...

# In-process MPI reductions (MpiBackend) engage automatically when
# the process runs under mpirun and the backend constraints allow it.
```

Single-node frontier runs (N = 32-36) do not need MPI at all: the
CSR-free rep lane keeps basis memory at O(#reps) and the fp32 GPU
mTPQ lane halves the vector footprint (`tpq_fp32=True`).

---

## Migration guide (from the legacy multi-step API)

| Old                                                        | New                                                |
| ---------------------------------------------------------- | -------------------------------------------------- |
| `op = qed.Operator(...)` + manual `add_*` calls           | `qed.input.HamiltonianBuilder(N).heisenberg(...)` |
| `qed.full_diagonalization(op)`                             | `qed.solve(op)`                                     |
| `qed.lanczos(op, exct=4, max_iter=200, tolerance=1e-10)`   | `qed.solve(op, num_eigenvalues=4, solver="LANCZOS")` |
| `qed.FixedSzOperator(N, n_up=k)` + manual rebuild          | `qed.solve(op, sz=k)`                               |
| `python -m edlib.automorphism_finder ...` + CLI ED run     | `qed.solve(op, symmetry=qed.find_symmetries(op).full_set)` |
| Read `include/ed/core/ed_parameters.h` to find `arpack_*`  | `qed.list_diag_parameters("arpack")`               |
| Manually slice `info["max_clique"]` to subgroup            | `report.full_set[1]` or `report.full_set.subgroup([0])` |
| `microcanonical_tpq(...)` driver + manual file plumbing    | `qed.solve(op, solver="mTPQ", target_beta=..., output_dir=...)` |
| Memorize that the enum is `mTPQ` (mixed case)              | `qed.solve(op, solver="mtpq")` works (case-insensitive) |
| Pass `--gpu` to `ED <dir> --method=LANCZOS_GPU`            | `qed.solve(op, device="gpu")` (auto temp-dir + from_directory) |
| `mpiexec ed_distributed_main ...` (the retired launcher)   | `mpiexec -n N ./ED <dir> --use-symmetry ...` (SectorDistributor + MpiBackend) |
| Look in `ed_method_traits.h` for solver/device wiring      | `qed.solver_device_support()` (build-aware (solver, device) matrix) |
| Compute "will this fit?" before running                    | size it by hand (`dim · 16 B` per vector); the workflow itself raises a clean error if the dominant allocation won't fit (no planner) |

The legacy entry points stay supported — they share the same C++
backend so behaviour is identical — but new code is encouraged to use
`qed.solve` as the single entry point.


---

## How `qed.solve` abstracts the dispatcher

`qed.solve(...)` is a **decision tree**, not a new solver. Every call
goes through the exact same C++ orchestrator
(`ed::workflows::{solve,thermal,spectral}` in
[`include/ed/orchestrator.h`](../../include/ed/orchestrator.h)) that
the `./ED` CLI uses; the Python wrapper just makes the choices for you:

```text
              qed.solve(H, num_eigenvalues=k, sz=…, solver=…, device=…, …)
                              │
   ┌──────────────────────────┴──────────────────────────┐
   │ 1. Sz axis                                          │
   │    sz=… given?                                      │
   │      → Operator.conserves_sz() guard                │
   │      → H.make_fixed_sz(sz)                          │
   │      → params.use_fixed_sz = True                   │
   │      → params.fixed_sz_op  = projected              │
   │    Else if Operator conserves Sz: print HINT.       │
   ├─────────────────────────────────────────────────────┤
   │ 2. Solver axis                                      │
   │    solver=None  → _resolve_solver(num_eig, dim):    │
   │      dim ≤ 2048           → FULL                    │
   │      k ≤ 5                → LANCZOS                 │
   │      k ≤ 20               → KRYLOV_SCHUR            │
   │      else                 → BLOCK_LANCZOS           │
   │    solver=str/enum → canonicalize_method(solver)    │
   ├─────────────────────────────────────────────────────┤
   │ 3. Device axis (orthogonal to solver)               │
   │    device='auto' (default):                         │
   │      has_cuda_build()  AND  dim ≥ 2¹⁷  → use_gpu    │
   │    device='gpu' / 'cpu' / 'mpi' / 'mpi_gpu'         │
   │       → params.use_gpu, params.use_mpi flags        │
   │    'mpi' / 'mpi_gpu' shells out to                  │
   │       mpiexec ed_distributed_main, then reads HDF5. │
   ├─────────────────────────────────────────────────────┤
   │ 4. Symmetry axis (orthogonal)                       │
   │    symmetry=… given?                                │
   │      → params.use_symmetry = True                   │
   │      → routes through streaming-symmetry kernel     │
   ├─────────────────────────────────────────────────────┤
   │ 5. Memory guard (no planner)                        │
   │    guard_working_set(...) checks the dominant        │
   │      allocation vs available RAM at the point of use │
   │      → raises a clear error instead of OOM-killing   │
   │      → ED_MEM_GUARD_OFF=1 bypasses                   │
   ├─────────────────────────────────────────────────────┤
   │ 6. Thermal-method bookkeeping                       │
   │    if solver ∈ {mTPQ, cTPQ, FTLM, LTLM, KPM_DOS}:   │
   │      auto-create output_dir if empty                │
   │      forward num_samples / target_beta / temp_*     │
   │        / num_temp_points to params.tpq_*/ftlm_*/…   │
   ├─────────────────────────────────────────────────────┤
   │ 7. Low-level escape hatch                           │
   │    extra_params={'ftlm_krylov_dim': 120,            │
   │                  'tpq_taylor_order': 200, ...}      │
   │      → setattr(params, key, value) for each pair    │
   │    list_diag_parameters() prints every field        │
   ├─────────────────────────────────────────────────────┤
   │ 8. Dispatch: _core.workflows_{solve,thermal} ...    │
   │    (or workflows_solve_streaming_symmetry_directory │
   │     when symmetry= is set)                          │
   └─────────────────────────────────────────────────────┘
```

### Layered API — pick the level of control you want

| Layer | Entry point | Use when |
|------:|------------|----------|
| 0 | `qed.solve(H)` / `qed.thermal(H, ...)` / `qed.spectral(dir, ...)` | **default** — common path, smart defaults |
| 1 | `qed.solve(H, solver=…, device=…, sz=…, …)` | override individual axes |
| 2 | `qed.solve(H, …, extra_params={...})` | tweak any niche `EDParameters` field |
| 3 | `qed._core.workflows_{solve,thermal,spectral}(spec, opts)` | drop down to the Pybind-bound orchestrator with raw `OperatorSpec` / `SolveOptions` / `ThermalOptions` / `SpectralOptions` |
| 4 | `qed.mpi.run_distributed(...)` / `qed.dssf.run_from_directory(...)` | shell out to the standalone `mpiexec` / `./ED dssf` binaries with custom flags |

Everything from layer 1 down maps **1-to-1** onto the C++
`EDParameters` fields documented in
[`include/ed/core/ed_parameters.h`](../../include/ed/core/ed_parameters.h).
Run `qed.list_diag_parameters()` to print them grouped by family
(general / thermal / TPQ / FTLM / LTLM / KPM / fixed-Sz / device /
symmetry / observables).

### C++ parity table

The C++ surface is identical, just one indirection lower:

| Python layer | C++ equivalent |
|-------------|----------------|
| `qed.solve(H, …)` | `ed::workflows::solve(H, SolveOptions{…})` (`include/ed/orchestrator.h`) |
| `qed.thermal(H, …)` | `ed::workflows::thermal(H, ThermalOptions{…})` (`include/ed/orchestrator.h`) |
| `qed.spectral(dir, T=…, omega=…)` | `ed::workflows::spectral(DSSFRequest{…}, SpectralOptions{…})` (`include/ed/orchestrator.h`) |
| `qed.solve(H, extra_params={'ftlm_krylov_dim': 80})` | `SolveOptions opts; opts.extra_params = [](EDParameters& p){ p.ftlm_krylov_dim = 80; };` |
| `qed._core.workflows_solve(spec, opts)` | `ed::workflows::solve(*ed::make_operator(spec), opts)` |

The C++ `extra_params` callback runs **after** the orchestrator has
populated `EDParameters` and **before** the kernel fires —
identical semantics to Python's `extra_params=`.

---

## Worked examples — smoke-tested at 12 sites (scale to 32 by changing `N`)

The examples below were exercised locally at `N=12` so they run quickly
and reproducibly on a laptop. To scale the same patterns to 32 sites,
set `N=32`, use `sz=16`, and keep symmetry projection on.

### Common setup

```python
import numpy as np
import qed as qed

N = 12
b = qed.input.HamiltonianBuilder(num_sites=N)
nn = [(i, (i + 1) % N) for i in range(N)]
b.heisenberg(bonds=nn, J=1.0)        # antiferromagnetic XXX
H = b.to_operator()

# Use the default automorphism search on the in-memory operator.
report = qed.find_symmetries(H, verbose=False)
print(report)
```

### 1 — Ground state (half-filled + symmetry)

```python
res = qed.solve(
    H,
    num_eigenvalues=2,
    sz=N // 2,
    symmetry=report.full_set,
    tolerance=1e-10,
    output_dir="ed_runs/heisenberg_N12_ground",
    compute_eigenvectors=True,
    extra_params={"block_size": 8},
)
print("E_0 =", res.eigenvalues[0])
print("Δ   =", res.eigenvalues[1] - res.eigenvalues[0])
```

For the C++ analog:

```cpp
#include <ed/orchestrator.h>
#include <ed/core/make_operator.h>

// Build the symmetry-projected operator via the unified factory.
ed::OperatorSpec spec;
spec.num_sites   = N;
spec.spin_length = 0.5f;
spec.sz          = N / 2;        // fixed-Sz × symmetry combination
spec.generators  = report_full_set;
// ... populate spec.bonds_heisenberg = J=1 PBC ring ...
auto H = ed::make_operator(spec);

ed::SolveOptions opts;
opts.num_eigenvalues       = 2;
opts.tolerance             = 1e-10;
opts.output_dir            = "ed_runs/heisenberg_N12_ground";
opts.compute_eigenvectors  = true;
opts.extra_params          = [](EDParameters& p) { p.block_size = 8; };
auto res = ed::workflows::solve(*H, opts);
```

### 2 — Finite-temperature Lanczos (FTLM)

```python
res = qed.solve(
    H,
    solver="FTLM",
    sz=N // 2,
    num_samples=2,
    temp_min=0.2,
    temp_max=1.0,
    num_temp_points=6,
    output_dir="ed_runs/heisenberg_N12_ftlm",
    extra_params={
        "ftlm_krylov_dim": 40,
        "ftlm_seed": 1234,
    },
)
```

For the equivalent **LTLM** run, swap `solver="FTLM"` to
`solver="LTLM"` and use `ltlm_*` tuning keys in `extra_params`.

### 3 — DSSF / SSSF (smoke-tested at N=12)

```python
import os, tempfile

# Write a deck to disk then call qed.spectral against it.
lat = qed.input.lattice.chain(N, pbc=True)
builder = qed.input.HamiltonianBuilder(lat.num_sites).heisenberg(lat.nn_pairs(), 1.0)

with tempfile.TemporaryDirectory() as td:
    builder.write_directory(td, lattice=lat)

    # Method auto-selection from (T, omega) axes:
    print(qed.dssf.pick_method(T=[0.5], omega=np.linspace(-1, 1, 64)))
    # -> "dynamical_thermal"

    # Static structure factor S(Q, T) — smoke-tested at N=12
    r_static = qed.spectral(
        td,
        T=[0.5, 1.0],
        method="static_thermal",
        ed_binary="/abs/path/to/ED",   # or put build dir on $PATH
        check=False,
        capture_output=True,
    )

    # Dynamical S(Q, ω, T) — smoke-tested at N=12
    r_dyn = qed.spectral(
        td,
        T=[0.5, 1.0],
        omega=np.linspace(-1.5, 1.5, 64),
        method="dynamical_thermal",
        ed_binary="/abs/path/to/ED",
        check=False,
        capture_output=True,
    )
    print("static rc:", r_static.returncode, "  dynamical rc:", r_dyn.returncode)
```

Scale to `N=32` by setting `N=32` in the common setup and removing the
`with tempfile.TemporaryDirectory()` block (use a real directory that
persists across runs).

### 4 — micro-canonical TPQ (mTPQ) trajectory

```python
res = qed.solve(
    H,
    solver="mTPQ",
    sz=N // 2,
    num_samples=2,
    target_beta=2.0,
    num_temp_points=6,
    output_dir="ed_runs/heisenberg_N12_mtpq",
    extra_params={
        "tpq_taylor_order": 40,
        "tpq_delta_beta": 0.05,
        "tpq_num_measure_points": 6,
        "tpq_measure_beta_min": 0.1,
        "tpq_measure_beta_max": 2.0,
    },
)
```

For the distributed variant, switch to `solver="cTPQ"` and
`device="mpi"` or `device="mpi_gpu"` with `mpi_n_ranks=...`.

---

## Cheat sheet — picking the right escape hatch

| You want to … | Use |
|--------------|-----|
| change `arpack_ncv`, `tpq_taylor_order`, `ftlm_seed`, …  | `qed.solve(…, extra_params={…})` |
| swap the **whole** parameter struct (e.g. copy from CLI) | `qed.solve(H, method, params)` |
| list every knob and which family it belongs to           | `qed.list_diag_parameters()` (or `('arpack')`, `('tpq')`, …) |
| inspect what the auto-pilot decided                      | `qed.solve(H, …, verbose=True)` (default) |
| keep an over-budget job from OOM-killing the host        | automatic (memory guard); `ED_MEM_GUARD_OFF=1` to bypass |
| chain low-level kernels yourself (Lanczos → CG → …)      | the dedicated solver modules in `qed.*` (see `python_advanced.md`) |


---

## DSSF / SSSF — `qed.spectral` auto-pilot + low-level overrides

Spectral / structure-factor calculations have their own one-call
auto-pilot, [`qed.spectral`](../../python/qed/dssf.py). It
mirrors `qed.solve` exactly: pass a directory + the axes you want, and
the wrapper picks the right kernel.

### Method auto-selection

The token forwarded to `./ED dssf` is decided by a 2×2 truth table on
`(T, omega)` — see `qed.dssf.pick_method(T=..., omega=...)`:

| `T` given? | `omega` given? | Method token | What it computes |
|:---:|:---:|---|---|
| no  | no  | `single_expectation`  | static observable ⟨ψ₀\|O\|ψ₀⟩ on the GS only |
| no  | yes | `ground_state_dssf`   | T=0 dynamical S(Q, ω) via Lanczos continued fractions |
| yes | no  | `static_thermal`      | thermal expectation ⟨O⟩(T) and equal-time S(Q, T) |
| yes | yes | `dynamical_thermal`   | full S(Q, ω, T) via FTLM/LTLM continued fractions |

Pass `method="ground_state_dssf"` (etc.) to override the auto-rule.
Bad tokens are rejected up-front with a helpful list.

### Common path — auto-pilot

```python
import numpy as np, qed as qed

# directory must contain a full ED deck (InterAll.dat, Trans.dat,
# positions.dat/lattice files) and must be compatible with your local
# `ED dssf` binary.

qed.spectral(
    "runs/heisenberg_N16",
    T=[0.1, 0.5],
    omega=np.linspace(-3.0, 3.0, 200),
    ed_binary="/abs/path/to/ED",   # or put ED on PATH
)
```

Output lands in `runs/<dir>/dssf/<momentum>/<observable>/<T>/` as the
unified `(omega, S, error)` HDF5 schema (Phase 8 — see
[`docs/architecture/CODEMAP.md`](../architecture/CODEMAP.md#dssf-output-schema)).

If your local build fails in `./ED dssf` on toy decks, validate first on
a known-good production directory (same call shape, just different
input deck).

### Low-level control — extra CLI flags

`compute(...)` forwards an arbitrary `extra_args=(...)` tuple to the
underlying `./ED dssf` invocation, giving you the full CLI surface of
[`src/cli/dssf_engine.cpp`](../../src/cli/dssf_engine.cpp). Common
flags:

| Flag (extra_args) | Knob | Notes |
|---|---|---|
| `--num_sites=<N>` | explicit site count | required when auto-detect misses |
| `--dyn-samples=<R>` | # random states for dynamical thermal | thermal averaging |
| `--dyn-krylov=<M>` | Krylov size for dynamical thermal | spectral resolution / cost |
| `--dyn-omega-min=<w0>` / `--dyn-omega-max=<w1>` / `--dyn-omega-points=<n>` | frequency grid | dynamical thermal / GS DSSF |
| `--dyn-temp-min=<T0>` / `--dyn-temp-max=<T1>` / `--dyn-temp-bins=<n>` | temperature grid | thermal methods |
| `--dyn-broadening=<eta>` | Lorentzian broadening | smoothness vs resolution |
| `--use-gpu` | enable GPU response kernels | requires CUDA build |

```python
qed.spectral(
    "runs/heisenberg_N16",
    T=[0.5],
    omega=np.linspace(-1.5, 1.5, 64),
    ed_binary="/abs/path/to/ED",
    extra_args=(
        "--num_sites=16",
        "--dyn-samples=2",
        "--dyn-krylov=40",
        "--dyn-omega-min=-1.5",
        "--dyn-omega-max=1.5",
        "--dyn-omega-points=64",
        "--dyn-temp-bins=1",
    ),
    capture_output=True,
)
```

### When to bypass `compute(...)`

For total control (custom env vars, custom binary path, distributed
DSSF), drop down to the lower layer:

| Use this | When |
|----------|------|
| `qed.dssf.run_from_directory(dir, method, ...)` | you already know the method token and want named-kwarg control |
| direct `subprocess.run(["/path/to/ED", "dssf", method, dir, ...])` | scripting around bespoke MPI launchers / SLURM |
| `ed::workflows::spectral(DSSFRequest{...}, SpectralOptions{...})` ([`include/ed/orchestrator.h`](../../include/ed/orchestrator.h)) | embedding DSSF in a C++ pipeline |
| `ed::dssf::run(DSSFRequest{...})` | full library-level control from C++ (no shell-out) |

### SSSF (equal-time structure factor)

SSSF is a **special case** of `static_thermal`: pass `T=...` without
`omega=`.

```python
print(qed.dssf.pick_method(T=[0.0, 0.5, 2.0], omega=None))
# -> "static_thermal"

qed.spectral(
    "runs/heisenberg_N16",
    T=[0.0, 0.5, 2.0],
    ed_binary="/abs/path/to/ED",
    extra_args=(
        "--num_sites=16",
        "--static-samples=4",
        "--static-krylov=120",
        "--static-temp-min=0.0",
        "--static-temp-max=2.0",
        "--static-temp-points=3",
    ),
)
```

`T=[0.0, ...]` is treated as the GS branch automatically by the C++
engine — no separate driver needed.

### C++ parity

| Python | C++ |
|--------|-----|
| `qed.spectral(dir, T=..., omega=...)` | `ed::workflows::spectral(req, opts)` |
| `qed.dssf.pick_method(T=..., omega=...)` | `ed::workflows::spectral_pick_method(has_T, has_w)` |
| `qed.dssf.run_from_directory(dir, method, ...)` | `ed::dssf::run(EDConfig*, req, method)` |

The same auto-rules apply on both sides — see
[`include/ed/orchestrator.h`](../../include/ed/orchestrator.h) and the
two-test smoke suite in `tests/unit/test_auto_dssf.cpp`.

---

## C++ mirrors for the workflow usages above

This page is Python-first (`qed.solve`, `qed.find_symmetries`,
`qed.spectral`). The same workflows are available from modern C++
through the auto-pilot façade and the core dispatcher.

### 0) Header set used by the snippets

```cpp
#include <ed/orchestrator.h>
#include <ed/orchestrator.h>
#include <ed/operators/spin_ops.h>

// Optional low-level entry points:
#include <ed/core/ed_wrapper.h>
#include <ed/core/ed_parameters.h>
```

### 1) TL;DR / Step 1 equivalent — build H and run a one-liner ground state

```cpp
ed::Operator H(/*N=*/12, /*S=*/0.5f);
ed::spin_ops::heisenberg_chain(H, /*N=*/12, /*J=*/1.0, /*pbc=*/true);

ed::SolveOptions opts;
opts.num_eigenvalues = 1;
auto gs = ed::workflows::solve(H, opts);
std::cout << "E0 = " << gs.eigenvalues.front() << "\n";
```

### 2) Step 3 equivalent — request low-lying states in a fixed-Sz sector

```cpp
ed::SolveOptions opts;
opts.num_eigenvalues = 4;
opts.sz = 6;                       // N/2 for N=12
opts.tolerance = 1e-10;
auto res = ed::workflows::solve(H, opts);
```

### 3) Maximum projection recipe (symmetry + Sz)

Symmetry projection in C++ is configured by populating
`OperatorSpec::generators` (and `OperatorSpec::sz` for fixed-Sz) on
`ed::make_operator`, which builds the streaming-symmetry operator
behind the scenes.

```cpp
#include <ed/core/make_operator.h>
#include <ed/core/streaming_symmetry.h>

ed::OperatorSpec spec;
spec.directory  = "ed_runs/12site_J1";   // hosts automorphism_results/
spec.num_sites  = 12;
spec.spin_length = 0.5f;
spec.sz         = 6;                     // fixed-Sz × symmetry
spec.generators = /* loaded from automorphism_results/ */ {};
auto H = ed::make_streaming_symmetry_operator(spec);

ed::SolveOptions opts;
opts.num_eigenvalues       = 1;
opts.compute_eigenvectors  = true;
opts.output_dir            = "ed_runs/12site_J1";
auto out = ed::workflows::solve(*H, opts);
```

### 4) Force GPU recipe

```cpp
ed::SolveOptions opts;
opts.device = ed::BackendConstraints::Device::GPU;
auto res = ed::workflows::solve(H, opts);
```

### 5) Thermal mTPQ recipe

```cpp
ed::SolveOptions opts;
opts.solver = DiagonalizationMethod::mTPQ;
opts.sz = 6;                       // optional fixed-Sz projection
opts.output_dir = "ed_runs/12site_thermal";
opts.extra_params = [](EDParameters& p) {
  p.num_samples = 8;
  p.target_beta = 20.0;
  p.num_temp_points = 40;
};
auto tpq = ed::workflows::solve(H, opts);
```

### 6) MPI / distributed recipe (library + launcher split)

The in-process C++ orchestrator routes a distributed run through
`OperatorSpec::distributed = true`; the caller is responsible for
already being inside an `mpiexec` launcher.

```cpp
#include <ed/core/make_operator.h>

ed::OperatorSpec spec;
spec.directory   = "ed_runs/24site_mpi";
spec.distributed = true;                 // build a DistributedOperator
auto H = ed::make_operator(spec);

ed::SolveOptions opts;
opts.num_eigenvalues = 4;
auto out = ed::workflows::solve(*H, opts);
```

For out-of-process rank launch (no `MPI_Init` in the calling
program), use the same `ed_distributed_main` binary as Python:

```bash
mpiexec -np 8 ./ed_distributed_main --mode lanczos --directory ed_runs/24site_mpi
mpiexec -np 4 ./ed_distributed_main --mode tpq --gpu --directory ed_runs/24site_mpi
```

### 7) Low-level escape hatch (Python extra_params equivalent)

```cpp
ed::SolveOptions opts;
opts.num_eigenvalues = 4;
opts.extra_params = [](EDParameters& p) {
  p.ftlm_krylov_dim   = 180;
  p.ltlm_krylov_dim   = 220;
  p.tpq_taylor_order  = 150;
};
auto out = ed::workflows::solve(H, opts);
```

Or drop down to a kernel directly:

```cpp
#include <ed/krylov/lanczos_kernel.h>
#include <ed/matvec/backends/cpu_backend.h>

ed::krylov::LanczosKernelOptions kopts;
kopts.num_eigenvalues = 4;
kopts.tolerance       = 1e-10;
kopts.max_iterations  = 300;

const std::uint64_t dim = H.dim();
auto apply = [&H](const Complex* in, Complex* out, int n) {
  H.apply(in, out, static_cast<size_t>(n));
};

auto out = ed::krylov::lanczos_kernel<ed::matvec::CpuBackend>(
    apply, dim, kopts);
```

### 8) DSSF / SSSF routine equivalents

Method selection is the same 2x2 rule as Python and can be queried by
`ed::workflows::spectral_pick_method(...)`.

```cpp
using ed::workflows::spectral_pick_method;
auto m0 = spectral_pick_method(/*has_temperature=*/false, /*has_frequency=*/false);
auto m1 = spectral_pick_method(/*has_temperature=*/false, /*has_frequency=*/true);
auto m2 = spectral_pick_method(/*has_temperature=*/true,  /*has_frequency=*/false);
auto m3 = spectral_pick_method(/*has_temperature=*/true,  /*has_frequency=*/true);
```

```cpp
ed::dssf::DSSFRequest req;
req.output_dir = "runs/heisenberg_N16";
req.config = &cfg;                  // required for non-single-expectation
// Fill req.operators (OperatorSpec) as needed.

ed::SpectralOptions o;
o.has_temperature = true;
o.has_frequency = true;
auto dssf = ed::workflows::spectral(req, o);
```

For direct CLI parity with the Python `extra_args` examples:

```bash
./ED dssf dynamical_thermal runs/heisenberg_N16 \
  --num_sites=16 --dyn-samples=2 --dyn-krylov=40 \
  --dyn-omega-min=-1.5 --dyn-omega-max=1.5 --dyn-omega-points=64 \
  --dyn-temp-min=0.5 --dyn-temp-max=0.5 --dyn-temp-bins=1
./ED dssf static_thermal runs/heisenberg_N16 \
  --num_sites=16 --static-samples=4 --static-krylov=120 \
  --static-temp-min=0.0 --static-temp-max=2.0 --static-temp-points=3
```

### 9) Smoke-tested examples on this page (ground / FTLM / DSSF / TPQ)

Use the same C++ patterns above with `N=12` and `opts.sz=6` for quick
checks, then scale to `N=32`, `opts.sz=16` for production:

1. Ground state: `OperatorSpec{sz=6}` + `SolveOptions{num_eigenvalues=2}`
   plus `extra_params = [](EDParameters& p){ p.block_size = 8; }`.
2. FTLM: `opts.solver = DiagonalizationMethod::FTLM`, set
   `num_samples`, `temp_min/max`, and `ftlm_krylov_dim` in
   `tune_params`.
3. DSSF: build `DSSFRequest` for your directory and call
  `ed::workflows::spectral(req, dssf_opts)` with
  `dssf_opts.has_temperature = true; dssf_opts.has_frequency = true;`.
4. mTPQ/cTPQ: `opts.solver = DiagonalizationMethod::mTPQ` (or `cTPQ`
   for MPI sample-splitting), set `target_beta` + TPQ knobs in
   `tune_params`.
