# Stress-free workflow (`qed.diag` + `qed.find_symmetries`)

This page documents the recommended Python entry point for new code:
the unified `quantum_ed.workflow` API. It collapses the
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
import quantum_ed as qed

# 1. Build a Hamiltonian.
N = 12
H = (qed.input.HamiltonianBuilder(N)
        .heisenberg([(i, (i + 1) % N) for i in range(N)], J=1.0)
        .to_operator())

# 2. (Optional) inspect what symmetries the engine sees.
report = qed.find_symmetries(H, verbose=False)
print(report.summary())   # tells you about U(1) Sz + lattice automorphisms

# 3. One-liner ground state.
print(qed.diag(H).eigenvalues[0])               # -5.387390917...

# 4. Bottom 4 eigenvalues in the Sz=0 sector.
print(qed.diag(H, num_eigenvalues=4, sz=N // 2).eigenvalues)

# 5. Bottom of the spectrum projected onto the full automorphism group +
#    Sz=0 sector. The streaming symmetry kernel runs under the hood.
print(qed.diag(H, num_eigenvalues=4,
               symmetry=report.full_set,
               sz=N // 2).eigenvalues)
```

That's it. Everything below explains the knobs that the function picks
for you.

---

## Step 1 — Build the Hamiltonian

`quantum_ed.input.HamiltonianBuilder` is the fluent C++-backed builder
(see `python/quantum_ed/input.py`). The same model can also be loaded
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

`find_symmetries(H, *, lattice=None, translation_only=False,
verbose=True) -> SymmetryReport` runs the colored-graph automorphism
pipeline (the same one that powers `automorphism_finder.py` on disk)
on `H` directly — no temp files, no CLI. It returns a
`SymmetryReport` with three pieces of information:

1. **U(1) Sz status.** `report.has_u1_sz` is `True` iff every term in
   the Hamiltonian commutes with total Sz, in which case
   `report.sz_sectors` is the list `[(n_up, dim), ...]`. The CLI
   would never tell you this — `qed.diag` will refuse a `sz=` argument
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
  -> pass `sz=<n_up>` to qed.diag(...) to restrict to a sector.

  Generator sets (2):
    [             trivial]  group_size=   1  |generators|= 0   orders=[]
      No symmetry projection (full Hilbert space).
    [   full_automorphism]  group_size=   6  |generators|= 2   orders=[2, 3]
      Largest abelian subgroup of the lattice + Hamiltonian automorphism group.

  -> pass any GeneratorSet (or list[Permutation]) as `symmetry=...` to qed.diag(...).
```

You can also pass your own permutations to `qed.diag` as the
`symmetry=` argument; `find_symmetries` is purely advisory:

```python
T = [(i + 1) % N for i in range(N)]      # site i -> i+1 mod N
my_set = qed.GeneratorSet(name="Z6_T", description="cyclic translation",
                          generators=[T], orders=[N], group_size=N)
qed.diag(H, symmetry=my_set)
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

# Either way, pass the sub-GeneratorSet straight into qed.diag(...):
eigs = qed.diag(H, num_eigenvalues=4, symmetry=rot_only).eigenvalues
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
qed.diag(H, symmetry=just_rotation)
```

The returned subgroup is always a fresh `GeneratorSet` whose
`group_size` is the product of the selected generators' orders
(correct because the parent `full_set` came from a minimal-generator
decomposition of an abelian group, so any subset is still independent).

## Step 3 — Diagonalise with `diag`

```python
qed.diag(H,
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
         max_subspace=None,        # auto-tuned from num_eigenvalues
         block_size=None,          # only used by BLOCK_* solvers
         # Thermal-method shortcuts (mTPQ / cTPQ / FTLM / LTLM):
         num_samples=None, target_beta=None, num_temp_points=None,
         temp_min=None, temp_max=None,
         # MPI launcher knobs (consulted only for device='mpi'/'mpi_gpu'):
         mpi_n_ranks=None,         # # of MPI ranks (default: 4)
         mpi_betas=None,           # β grid for distributed TPQ / FTLM
         mpi_compute_variance=False,
         mpi_binary=None,          # ed_distributed_main path override
         mpi_launcher="mpiexec",   # 'srun' on SLURM, etc.
         mpi_launcher_binary=None,
         verbose=True,
         extra_params=None)        # forwarded to EDParameters as setattr
```

### Smart defaults

* **Krylov sizes.** `max_iterations` and `max_subspace` default to
  `max(200, 8*num_eigenvalues + 80)` and
  `max(80, 4*num_eigenvalues + 40)` respectively (capped by the
  sector dimension minus one). These are the constants from the
  `bench_vs_xdiag` bake-off and converge the requested eigenvalues
  to `tolerance` with no further tuning at the sizes Python typically
  reaches in-process.
* **Solver.** `solver=None` picks the best backend for the matrix
  shape: `FULL` for dimensions ≤ 1024 (LAPACK is end-to-end faster
  than a Lanczos warmup at that scale), `KRYLOV_SCHUR` when
  `num_eigenvalues ≥ 16`, and `LANCZOS` otherwise. Pass an explicit
  enum value (`qed.DiagonalizationMethod.LOBPCG`, etc.) or its string
  name to override.
* **Device.** `device=None` uses GPU iff
  `quantum_ed.has_cuda_build()` is true and the matrix is large
  enough for cuSPARSE matvec to amortize H2D / D2H (rule of thumb:
  dim ≥ 2¹⁴). Pass `"cpu"` / `"gpu"` to force a backend.
  `"mpi"` and `"mpi_gpu"` are also accepted: the workflow writes
  the operator to a temp directory, shells out to `mpiexec
  ed_distributed_main` (with `--gpu` for `mpi_gpu`), and parses
  the HDF5 result back into an `EDResults`. Python never hosts
  `MPI_Init` itself; the launcher does. See "Distributed (MPI)
  jobs" below for what that buys you.
* **Sz axis.** When `sz=` is passed and `H` is an `Operator`,
  `diag` materialises a `FixedSzOperator` from `H` for you (via the
  new `Operator.make_fixed_sz` binding) and runs the in-memory
  fixed-Sz kernel. When `H` is already a `FixedSzOperator`,
  `sz=` is only used as a sanity check.
* **Symmetry projection.** When `symmetry=` is passed,
  `diag` writes the operator + symmetry metadata to a temp directory,
  invokes the C++ streaming-symmetry kernel
  (`exact_diagonalization_streaming_symmetry` /
  `exact_diagonalization_streaming_symmetry_fixed_sz`), and cleans up.
  All accepted forms — `GeneratorSet`, raw `list[list[int]]` of
  permutations, or the dict produced by
  `qed.symmetry.group_from_generators` — are normalised to the same
  on-disk schema.
* **Combined Sz + symmetry.** Pass both `sz=` and `symmetry=`. The
  streaming-symmetry-fixed-Sz kernel handles the joint projection.

### What you get back

`qed.diag(...)` returns an `EDResults` object with:

* `eigenvalues` — sorted lowest-first.
* `eigenvectors` (when `compute_eigenvectors=True`) — packed as a
  numpy array of shape `(num_eigenvalues, dim)`.
* `eigenvectors_path` — when `output_dir` is non-empty, the HDF5
  file (or the per-rank-slab directory for distributed runs) the
  solver wrote into. Useful for very large vectors that you don't
  want pulled back across the C++/Python boundary; for distributed
  runs see `qed.load_mpi_eigenvector(eigenvectors_path, k)`.

### Specifying simulation parameters

`qed.diag(...)` exposes the most common knobs as first-class keyword
arguments (listed in the signature above): `num_eigenvalues`,
`tolerance`, `compute_eigenvectors`, `solver`, `device`, `symmetry`,
`sector`, `sz`, `output_dir`, `max_iterations`, `max_subspace`,
`block_size`, and `verbose`. Anything else on `EDParameters` —
ARPACK / FTLM / LTLM / TPQ / ScaLAPACK / observable settings —
is reachable via `extra_params={...}`:

```python
res = qed.diag(
    H,
    num_eigenvalues=6,
    solver="ARPACK",
    extra_params={
        "arpack_which": "SA",     # smallest algebraic
        "arpack_ncv": 64,         # bigger Arnoldi NCV
        "arpack_max_restarts": 5,
    },
)
```

To discover what's available, call:

```python
qed.list_diag_parameters()         # full catalogue, grouped by category
qed.list_diag_parameters("arpack") # filter to one category (substring OK)
qed.list_diag_parameters(return_dict=True)   # programmatic access
```

The catalogue is bucketed into:

| category      | what it controls                                     |
| ------------- | ---------------------------------------------------- |
| `general`     | `num_eigenvalues`, `tolerance`, eigenvector toggles  |
| `krylov`      | Lanczos / Krylov-Schur subspace shape (`max_subspace`, `block_size`, ...) |
| `device`      | The orthogonal axes: `use_gpu`, `use_mpi`, `use_symmetry`, `use_fixed_sz` |
| `arpack`      | ARPACK iterative solver tunables                     |
| `scalapack`   | Distributed dense (`scalapack_*`)                   |
| `ftlm`        | Finite-Temperature Lanczos                           |
| `ltlm`        | Low-Temperature Lanczos + HYBRID crossover           |
| `tpq`         | Thermal Pure Quantum / mTPQ                          |
| `thermal`     | Temperature-grid post-processing                     |
| `observables` | Spectral / dynamical observables (`omega_*`, `dt`)   |
| `lattice`     | Lattice metadata                                     |

If you typo a key, the resulting `AttributeError` points back at
`list_diag_parameters()` so you can grep for the right name.

---

## Solver × path support matrix

`qed.diag` exposes every backend the C++ dispatcher knows about, but
not every (solver, basis) combination is meaningful. The table below
records what works through the unified Python entry point. "Path"
refers to the four basis choices the workflow can compose:

* **full** — the full 2ᴺ Hilbert space.
* **sz** — fixed-Sz block (pass `sz=`).
* **symm** — symmetry-projected basis (pass `symmetry=`).
* **symm + sz** — both at once.

| solver family             | full | sz | symm | symm + sz | returns                                |
| ------------------------- | :--: | :-: | :--: | :-------: | -------------------------------------- |
| `LANCZOS` (and NO_ORTHO / SELECTIVE / IRL / TRL variants) | ✅ | ✅ | ✅ | ✅ | `EDResults.eigenvalues` |
| `KRYLOV_SCHUR`            | ✅ | ✅ | ✅ | ✅ | `EDResults.eigenvalues` |
| `BLOCK_LANCZOS` / `BLOCK_KRYLOV_SCHUR` | ✅ | ✅ | ✅ | ✅ | `EDResults.eigenvalues` |
| `DAVIDSON` / `LOBPCG`     | ✅ | ✅ | ✅ | ✅ | `EDResults.eigenvalues` |
| `ARPACK_*` (`SM`, `LM`, `SHIFT_INVERT`, `ADVANCED`) | ✅ | ✅ | ✅ | ✅ | `EDResults.eigenvalues` |
| `CHEBYSHEV_FILTERED`, `SHIFT_INVERT[_ROBUST]`, `BICG`, `OSS` | ✅ | ✅ | ✅ | ✅ | `EDResults.eigenvalues` |
| `FULL`, `SCALAPACK[_MIXED]` | ✅ | ✅ | ✅ | ✅ | `EDResults.eigenvalues` |
| `mTPQ` / `cTPQ`           | ✅ | ✅ | ❌¹ | ❌¹ | trajectory in `eigenvalues`; thermo curve in `output_dir` |
| `FTLM`                    | ✅ | ✅ | ✅² | ✅² | `EDResults.thermo_data` (sectors are summed) |
| `LTLM` / `HYBRID`         | ✅ | ✅ | ✅² | ✅² | `EDResults.thermo_data` |

Notes:

¹ TPQ acts on a single random vector spread across the whole sector;
projecting onto each symmetry irrep destroys the Z normalisation. The
workflow raises a clear `ValueError` when this combination is
requested. Pre-project to a fixed-Sz block instead.

² FTLM/LTLM/HYBRID *do* combine across symmetry blocks correctly
because each block contributes an additive term to the partition
function. The dispatcher handles the per-sector aggregation.

```python
# Eigenvalue solver, all four paths:
qed.diag(H, solver="LANCZOS")                                 # full
qed.diag(H, solver="LANCZOS", sz=N // 2)                      # sz
qed.diag(H, solver="LANCZOS", symmetry=report.full_set)       # symm
qed.diag(H, solver="LANCZOS",
         symmetry=report.full_set, sz=N // 2)                 # symm + sz

# Thermal solver — case-insensitive name lookup:
res = qed.diag(H, solver="mTPQ", sz=N // 2,
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
axis**: where the matrix-vector products execute. The four cells are
single-process CPU, single-GPU, distributed CPU (MPI), and distributed
GPU (MPI + per-rank GPU + NCCL collectives). Not every solver has a
kernel for every cell -- the table below is the canonical record.

| solver family            |  cpu  |  gpu  |  mpi  | mpi+gpu | how to invoke                                        |
| ------------------------ | :---: | :---: | :---: | :-----: | ---------------------------------------------------- |
| `LANCZOS`                |  ✅   |  ✅   |  ✅   |   ✅¹   | `qed.diag(H[, device='gpu'/'mpi'/'mpi_gpu'])` / `run_distributed("lanczos"[, use_gpu=True])` |
| `BLOCK_LANCZOS`          |  ✅   |  ✅   |  ❌   |   ❌    | `qed.diag(H, solver="BLOCK_LANCZOS"[, device='gpu'])` |
| `KRYLOV_SCHUR`           |  ✅   |  ✅   |  ✅²  |   ❌²   | `qed.diag(H, solver="KRYLOV_SCHUR"[, device='gpu'/'mpi'])` |
| `BLOCK_KRYLOV_SCHUR`     |  ✅   |  ✅   |  ❌   |   ❌    | `qed.diag(H, solver="BLOCK_KRYLOV_SCHUR"[, device='gpu'])` |
| `DAVIDSON` / `LOBPCG`    |  ✅   |  ✅   |  ❌   |   ❌    | `qed.diag(H, solver="DAVIDSON"[, device='gpu'])`     |
| `ARPACK_*`               |  ✅   |  ❌   |  ❌   |   ❌    | `qed.diag(H, solver="ARPACK_SM")`                    |
| `FULL`                   |  ✅   |  ✅   |  ❌   |   ❌    | `qed.diag(H, solver="FULL"[, device='gpu'])`         |
| `SCALAPACK[_MIXED]`      |  ❌   |  ❌   |  ✅   |   ❌    | `run_distributed("lanczos", ...)` (under the hood)   |
| `mTPQ`                   |  ✅   |  ✅   |  ✅³  |   ✅⁴   | `qed.diag(H, solver="mTPQ"[, device='mpi'/'mpi_gpu'])` / `run_distributed("tpq"[, use_gpu=True], ...)` |
| `cTPQ`                   |  ✅   |  ✅   |  ✅³  |   ✅⁴   | `qed.diag(H, solver="cTPQ"[, device='mpi'/'mpi_gpu'])` / `run_distributed("tpq"[, use_gpu=True], ...)` |
| `FTLM`                   |  ✅   |  ✅   |  ✅   |   ❌    | `qed.diag(H, solver="FTLM")` / `run_distributed("ftlm", ...)` |
| `LTLM` / `HYBRID`        |  ✅   |  ❌   |  ❌   |   ❌    | `qed.diag(H, solver="LTLM")`                          |
| `SHIFT_INVERT[_ROBUST]` etc. | ✅ |  ❌   |  ❌   |   ❌    | `qed.diag(H, solver="SHIFT_INVERT")`                  |

Notes:

¹ Multi-GPU Lanczos is the `distributed_lanczos_gpu` kernel: per-rank
CUDA Krylov vectors with `ncclAllReduce` for the dot/norm reductions.
Built only when `WITH_MPI=ON && WITH_CUDA=ON && NCCL_FOUND`. With the
"stage 3" GPU-resident SpMV (`--gpu-resident-spmv`), the halo also
flows via `ncclSendRecv` between device buffers (GPU-Direct RDMA on
supported fabrics).

² Distributed Krylov-Schur is the `distributed_krylov_schur` kernel:
thick-restart Lanczos with Ritz-pair locking, full re-orthogonalisation
against locked vectors, and an orthogonalised re-seed each cycle. It
sits on the existing CPU `DistributedOperator`, so the mpi+gpu cell is
still ❌ -- the restart machinery isn't templated on
`DistributedGPUOperator` yet (use `device='mpi_gpu'` with
`solver='LANCZOS'` to get GPU + MPI for now).

³ Distributed mTPQ/cTPQ goes through `--mode tpq` of
`ed_distributed_main`, which invokes `distributed_tpq` (canonical TPQ
with MPI-over-samples). See "Distributed (MPI) jobs" below for the
recipe.

⁴ Multi-GPU mTPQ/cTPQ is the `distributed_tpq_gpu` kernel: device-
resident |ψ⟩, on-device SpMV via `DistributedGPUOperator` (NCCL halo +
CUDA SoA SpMV), Taylor accumulator with `cublasZaxpy`, and norm /
observable reductions through `cublasZdotc` +
`multi_gpu::all_reduce_sum_complex_double` (one NCCL allreduce per
scalar). MPI-over-samples mirrors the CPU path. Built only when
`WITH_MPI=ON && WITH_CUDA=ON && NCCL_FOUND`.

### Build-aware introspection

The matrix above is a static record of what kernels exist in C++. To
ask the *current build* whether a given (solver, device) cell is
reachable:

```python
qed.solver_device_support()                  # prints the table
m = qed.solver_device_support(return_dict=True)
m["LANCZOS"]["mpi_gpu"]
# {'kernel': True, 'available': False,
#  'note': 'needs WITH_MPI=ON and WITH_CUDA=ON'}

qed.solver_device_support(solver="lanczos")  # filter to one solver family
```

Each cell carries `kernel` (does the C++ side have the code at all),
`available` (is *this* build wired for it), and a `note` string with
the rebuild flag to flip when `available=False`.

### Choosing a device

* `device=None` (the default) lets the workflow pick: CPU for small
  problems, single GPU when `WITH_CUDA=ON` and the dimension is large
  enough that PCIe transfer amortizes (current threshold: `dim ≥ 2¹⁴`).
* `device='cpu'` / `device='gpu'` force the choice. The GPU path
  silently routes through a temp-dir + `exact_diagonalization_from_directory`
  because the in-process `exact_diagonalization_core` rejects GPU
  methods (they need a `GPUOperator` built from files via
  `GPUEDWrapper::createGPUOperatorFromFiles`).
* `device='mpi'` / `device='mpi_gpu'` write the operator to a temp
  directory and shell out to `mpiexec ed_distributed_main`. The
  helper picks the right `--mode` for the chosen solver
  (`lanczos` / `krylov_schur` / `ftlm` / `tpq`), forwards `--gpu`
  for `mpi_gpu`, and parses the HDF5 result back into an
  `EDResults` -- so the call shape is identical to the in-process
  CPU/GPU paths. Python never calls `MPI_Init` itself; the MPI
  launcher does, in a separate process tree.

---

## Recipes

### Ground state of an arbitrary spin Hamiltonian

```python
H = (qed.input.HamiltonianBuilder(N)
        .heisenberg(bonds, J=1.0)
        .to_operator())
e0 = qed.diag(H).eigenvalues[0]
```

### Bottom-of-spectrum within a target Sz sector

```python
report = qed.find_symmetries(H, verbose=False)
assert report.has_u1_sz, "Sz is not conserved -- pick a different model"
res = qed.diag(H, num_eigenvalues=4, sz=N // 2)
```

### Maximum projection: full automorphism group + fixed Sz

```python
report = qed.find_symmetries(H, lattice=lat, verbose=False)
res = qed.diag(H,
               num_eigenvalues=8,
               symmetry=report.full_set,
               sz=N // 2,
               compute_eigenvectors=True,
               output_dir="ed_runs/12site_J1")
```

### Force GPU even on a small problem (e.g. for benchmarking)

```python
res = qed.diag(H, device="gpu")
```

### Thermal trajectory with mTPQ

```python
res = qed.diag(H,
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

### Distributed (MPI) jobs

`qed.diag(H, device='mpi'[, ...])` is the "do what I mean" entry
point: it serialises `H` (and the symmetry directory if you pass
`symmetry=`), shells out to `mpiexec ed_distributed_main`, parses
the HDF5 result, and returns an `EDResults` indistinguishable in
shape from a CPU run:

```python
# Ground state of a 24-site chain on 8 CPU ranks:
res = qed.diag(H, device='mpi', mpi_n_ranks=8)          # --mode lanczos
print(res.eigenvalues[0])

# Same, but multi-GPU per rank (NCCL collectives + GPU-resident SpMV):
res = qed.diag(H, device='mpi_gpu', mpi_n_ranks=4)      # --mode lanczos --gpu

# Symmetry-projected on the lattice + Sz=N/2 sector, 8 ranks:
report = qed.find_symmetries(H, lattice=lat, verbose=False)
res = qed.diag(H, device='mpi', mpi_n_ranks=8,
               symmetry=report.full_set, sz=N // 2)

# Distributed Krylov-Schur (thick-restart Lanczos w/ Ritz-pair locking):
res = qed.diag(H, solver="KRYLOV_SCHUR",
               device='mpi', mpi_n_ranks=8, num_eigenvalues=8)

# Distributed canonical TPQ thermal trajectory, MPI-over-samples:
res = qed.diag(H, solver="cTPQ", device='mpi', mpi_n_ranks=8,
               num_samples=32, target_beta=20.0,
               mpi_betas=[0.1, 0.5, 1.0, 2.0])

# Multi-GPU canonical TPQ (distributed_tpq_gpu): device-resident |ψ⟩,
# cuBLAS axpys/dotcs, NCCL allreduces, MPI-over-samples:
res = qed.diag(H, solver="cTPQ", device='mpi_gpu', mpi_n_ranks=4,
               num_samples=32, mpi_betas=[0.1, 0.5, 1.0, 2.0])

# Eigenvectors over MPI: each rank dumps its slab to
# <output_dir>/eigvecs/rank_<r>.h5; reassemble in Python via
# qed.load_mpi_eigenvector(eigvec_dir, k).
res = qed.diag(H, device='mpi', mpi_n_ranks=8,
               num_eigenvalues=4, compute_eigenvectors=True,
               output_dir="ed_runs/24site_mpi")
psi0 = qed.load_mpi_eigenvector(res.eigenvectors_path, k=0)
```

If you want the lower-level launcher (e.g. for benchmarking with
hand-rolled `binary_args`), `qed.mpi.run_distributed(...)` is still
available and shells out to the same binary:

```python
# Distributed Lanczos on a chain Hamiltonian generated by the binary:
qed.mpi.run_distributed(
    method="lanczos", n_ranks=8,
    binary_args=("--N", "24", "--J", "1.0", "--periodic", "1",
                 "--max-iter", "400", "--reorth", "1"),
)

# Multi-GPU Lanczos with the stage-3 NCCL halo:
qed.mpi.run_distributed(
    method="lanczos", n_ranks=4, use_gpu=True,
    binary_args=("--N", "30", "--J", "1.0", "--periodic", "1",
                 "--max-iter", "200",
                 "--gpu-resident-spmv"),
)

# Multi-GPU canonical TPQ (distributed_tpq_gpu):
qed.mpi.run_distributed(
    method="tpq", n_ranks=4, use_gpu=True,
    binary_args=("--N", "24", "--betas", "0.1,0.5,1.0,2.0",
                 "--samples", "32", "--groups", "4",
                 "--delta-beta", "0.05", "--taylor-order", "30",
                 "--compute-variance"),
)
# Aliases: solver names from qed.diag work too --
# method='mtpq' / 'ctpq' map to --mode tpq, and
# method='ks' maps to --mode krylov_schur (no warning since Phase 9).
```

`run_distributed` shells out to the standalone `ed_distributed_main`
binary (built by the C++ side when `-DWITH_MPI=ON`). The binary
accepts `--mode {lanczos|krylov_schur|ftlm|tpq}` and a `--gpu`
switch; `--gpu` requires `WITH_CUDA=ON && NCCL_FOUND` and is
honoured by `lanczos` (`distributed_lanczos_gpu`) and `tpq`
(`distributed_tpq_gpu`).

---

## Migration guide (from the legacy multi-step API)

| Old                                                        | New                                                |
| ---------------------------------------------------------- | -------------------------------------------------- |
| `op = qed.Operator(...)` + manual `add_*` calls           | `qed.input.HamiltonianBuilder(N).heisenberg(...)` |
| `qed.full_diagonalization(op)`                             | `qed.diag(op)`                                     |
| `qed.lanczos(op, exct=4, max_iter=200, tolerance=1e-10)`   | `qed.diag(op, num_eigenvalues=4, solver="LANCZOS")` |
| `qed.FixedSzOperator(N, n_up=k)` + manual rebuild          | `qed.diag(op, sz=k)`                               |
| `python -m edlib.automorphism_finder ...` + CLI ED run     | `qed.diag(op, symmetry=qed.find_symmetries(op).full_set)` |
| Read `include/ed/core/ed_parameters.h` to find `arpack_*`  | `qed.list_diag_parameters("arpack")`               |
| Manually slice `info["max_clique"]` to subgroup            | `report.full_set[1]` or `report.full_set.subgroup([0])` |
| `microcanonical_tpq(...)` driver + manual file plumbing    | `qed.diag(op, solver="mTPQ", target_beta=..., output_dir=...)` |
| Memorize that the enum is `mTPQ` (mixed case)              | `qed.diag(op, solver="mtpq")` works (case-insensitive) |
| Pass `--gpu` to `ED <dir> --method=LANCZOS_GPU`            | `qed.diag(op, device="gpu")` (auto temp-dir + from_directory) |
| `mpiexec ed_distributed_main --mode lanczos` (no GPU flag) | `qed.diag(H, device='mpi'/'mpi_gpu', mpi_n_ranks=N)` (or `qed.mpi.run_distributed("lanczos", n_ranks=N, use_gpu=True, ...)`) |
| `mpiexec ed_distributed_main --mode tpq --gpu` for thermal | `qed.diag(H, solver='cTPQ', device='mpi_gpu', mpi_n_ranks=N, ...)` |
| Reassemble distributed eigenvectors by hand from rank_*.h5 | `qed.load_mpi_eigenvector(eigvec_dir, k)` (single) / `qed.load_mpi_eigenvectors(eigvec_dir)` (all) |
| Look in `ed_method_traits.h` for solver/device wiring      | `qed.solver_device_support()` (build-aware (solver, device) matrix) |

The legacy entry points stay supported — they share the same C++
backend so behaviour is identical — but new code is encouraged to use
`qed.diag` as the single entry point.
