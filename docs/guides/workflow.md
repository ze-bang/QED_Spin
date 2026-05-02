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

# 6. "Will my N=32 FTLM job fit on this host?" -- the pre-flight planner
#    is now ALWAYS on; it raises qed.ResourceError with ranked
#    suggestions when the answer is no. dry_run prints + exits.
qed.diag(H_big, solver="FTLM", sz=16, dry_run=True)
# [qed.diag.planner] verdict: INFEASIBLE (memory)
#   basis     : sz=n_up=16 sector dim=601_080_390
#   memory    : per-rank 752.37 GB / avail 123.02 GB
#   wall-time : ~106.86 d
#   suggestions:
#     - pass symmetry=qed.find_symmetries(H).full_set ...
#     - switch to device='mpi' with mpi_n_ranks>=7
#     - drop compute_eigenvectors=True ...

# 7. Goal-oriented planner: rank candidate workflows for a given intent.
print(qed.suggest_workflow(H_big, intent="thermal", n_samples=8).summary())
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
         # Pre-flight planner (Phase 9 / Layer 6):
         plan=True,                # always run the planner before dispatch
         dry_run=False,            # planner-only; do not run the kernel
         force=False,              # ignore ResourceError on infeasibility
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
| `KRYLOV_SCHUR`           |  ✅   |  ✅   |  ✅²  |   ✅⁶   | `qed.diag(H, solver="KRYLOV_SCHUR"[, device='gpu'/'mpi'/'mpi_gpu'])` |
| `BLOCK_KRYLOV_SCHUR`     |  ✅   |  ✅   |  ❌   |   ❌    | `qed.diag(H, solver="BLOCK_KRYLOV_SCHUR"[, device='gpu'])` |
| `DAVIDSON` / `LOBPCG`    |  ✅   |  ✅   |  ❌   |   ❌    | `qed.diag(H, solver="DAVIDSON"[, device='gpu'])`     |
| `ARPACK_*`               |  ✅   |  ❌   |  ❌   |   ❌    | `qed.diag(H, solver="ARPACK_SM")`                    |
| `FULL`                   |  ✅   |  ✅   |  ❌   |   ❌    | `qed.diag(H, solver="FULL"[, device='gpu'])`         |
| `SCALAPACK[_MIXED]`      |  ❌   |  ❌   |  ✅   |   ❌    | `run_distributed("lanczos", ...)` (under the hood)   |
| `mTPQ`                   |  ✅   |  ✅   |  ✅³  |   ✅⁴   | `qed.diag(H, solver="mTPQ"[, device='mpi'/'mpi_gpu'])` / `run_distributed("tpq"[, use_gpu=True], ...)` |
| `cTPQ`                   |  ✅   |  ✅   |  ✅³  |   ✅⁴   | `qed.diag(H, solver="cTPQ"[, device='mpi'/'mpi_gpu'])` / `run_distributed("tpq"[, use_gpu=True], ...)` |
| `FTLM`                   |  ✅   |  ✅   |  ✅   |   ✅⁵   | `qed.diag(H, solver="FTLM")` / `run_distributed("ftlm"[, use_gpu=True], ...)` |
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
against locked vectors, and an orthogonalised re-seed each cycle.
Built on the CPU `DistributedOperator`. See footnote ⁶ for the
multi-GPU sibling.

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

⁵ Multi-GPU FTLM is the `distributed_ftlm_gpu` kernel: per-sample
Lanczos with full modified Gram–Schmidt re-orthogonalisation runs
**entirely on device** (Krylov basis V[0..m-1] held contiguously in a
single device slab, `cublasZdotc` for the inner products, one NCCL
allreduce per scalar via `multi_gpu::all_reduce_sum_complex_double`,
`cublasZaxpy` for the recurrence, `DistributedGPUOperator` for the
SpMV). The (m × m) tridiagonal eigensolve is done redundantly on every
rank with Eigen (small problem; not a bottleneck). When an observable
is supplied the J&P contraction `q_j = ⟨V[j] | O V[0]⟩` reuses the
same on-device basis with one extra device SpMV per sample plus a
single NCCL allreduce of `2·m` doubles. MPI-over-samples mirrors the
CPU `distributed_ftlm`. Built only when `WITH_MPI=ON && WITH_CUDA=ON
&& NCCL_FOUND`.

⁶ Multi-GPU Krylov-Schur is the `distributed_krylov_schur_gpu` kernel:
the same thick-restart Lanczos with Ritz-pair locking as the CPU
sibling, but the in-cycle Krylov basis V[0..m-1] **and** the locked
Ritz vectors live in two contiguous device slabs ((m_max + |locked|)
× local_n × 16 B per rank). Twice-CGS re-orthogonalisation against
both sets is coalesced — each pass packs the |set| local
`cublasZdotc` results into a single NCCL allreduce of `2·|set|`
doubles, then runs |set| `cublasZaxpy` calls to subtract. SpMV goes
through `DistributedGPUOperator` (NCCL halo + on-device SoA SpMV).
The `(m × m)` tridiagonal eigensolve and the Ritz-residual /
locking sweep are host-side (replicated on every rank); reconstruction
of locked Ritz vectors `phi = Σ_j U[j,i] V[j]` and of the next-cycle
seed runs on device via `cublasZaxpy`. **Limitation**: the locked
Ritz vectors stay on device — `--compute-eigenvectors` /
`--eigenvector-dir` is rejected on the GPU path with an actionable
error (drop `--gpu` or skip the eigenvector dump). Built only when
`WITH_MPI=ON && WITH_CUDA=ON && NCCL_FOUND`.

### Path × device — cross-product caveats

The path matrix (`full` / `sz` / `symm` / `symm + sz`) and the device
matrix (`cpu` / `gpu` / `mpi` / `mpi+gpu`) are **almost** orthogonal
but not quite — a few combinations have known carve-outs that the
two tables above don't capture on their own:

* `full` and `sz` work on every device cell where the solver is ✅.
  This is the common case and what the `qed.diag(...)` quick-start
  examples use.
* `symm` (symmetry-projected basis) and `symm + sz` work on **cpu**
  and **gpu** for every solver that supports them (the symmetry
  projection happens before the Lanczos / TPQ kernel sees the
  vector, so the kernel itself is none the wiser).
* `symm` × **mpi** is wired through `DistributedSymmetryOperator`
  for `LANCZOS` and `FTLM` only at the time of writing. Other
  distributed solvers (`KRYLOV_SCHUR`, `BLOCK_*`, `DAVIDSON`,
  `LOBPCG`, `mTPQ`, `cTPQ`) currently fall back with an actionable
  error if `symm` and `mpi` are both requested. Phase D extends the
  dispatcher to wire them all uniformly once the GPU symmetry
  operator (Phase C) lands.
* `symm` × **mpi+gpu** has the on-device SpMV primitive in place
  (`DistributedSymmetryOperatorGPU` — Phase C: NCCL pairwise SendRecv
  halo + a single CSR sparse-matvec kernel reading the row slab that
  the wrapped CPU `DistributedSymmetryOperator` already built). It is
  not yet wired into individual solver dispatchers — Phase D walks
  through (solver × device × path) and routes every distributed
  GPU solver through this primitive when `--use-symmetry` is set.
  Until then, symmetry-projected runs on multi-GPU clusters should
  drop to `device='gpu'` per node.
* `mTPQ` / `cTPQ` × `symm`: see Phase E. The lift is "per-sector
  TPQ + FTLM-style Z-aggregation" (each sector gets its own random
  state and its own canonical TPQ trajectory; partition functions
  combine additively with the irrep multiplicities, exactly like
  FTLM does). Until Phase E ships, the workflow raises a clear
  `ValueError` for this combination.

The introspection helper `qed.solver_device_support()` only reports
the solver × device tensor; consult this subsection for the
path-axis carve-outs that aren't visible there.

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

## Pre-flight planner — `qed.estimate_resources` / `qed.suggest_workflow`

Before any kernel runs, `qed.diag` always asks the planner *"will this
fit on the host, and if not, what should I run instead?"*. The planner:

* counts the resident dim-sized vectors the chosen kernel keeps live,
  per the (solver, num\_eigenvalues, max\_subspace, block\_size,
  compute\_eigenvectors) tuple;
* multiplies by `dim * 16 B` for the chosen basis (full / sz / symm /
  sym+sz) and divides by `n_ranks` for the MPI / multi-GPU cells;
* probes the host (psutil + nvidia-smi, with `QED_HOST_*` env-var
  overrides) to learn how much CPU RAM / VRAM / MPI ranks are
  available;
* compares the two and emits a one-line **FEASIBLE** / **INFEASIBLE
  (memory|build|kernel)** verdict with a wall-time ballpark.

If the verdict is **INFEASIBLE**, `qed.diag` raises a
`qed.ResourceError` whose `.report` carries ranked, copy-pasteable
suggestions ("pass `sz=N//2` to cut dim 10×", "switch to
`device='mpi'` with `mpi_n_ranks≥7`", "the FULL dense path scales as
dim²; switch to `solver='LANCZOS'`", …). Pass `force=True` to ignore
the planner (e.g. when the planner can't see your scheduler-allocated
RAM), `dry_run=True` to print the report and stop, or `plan=False`
to skip the planner entirely.

```python
# 1. Predictive: "would this run? how big? how long?"
report = qed.estimate_resources(H, solver="FTLM", device="mpi",
                                 sz=N // 2, num_eigenvalues=1,
                                 n_samples=8, n_ranks=8)
print(report.summary())
# [qed.diag.planner] verdict: FEASIBLE
#   basis     : sz=n_up=16 sector dim=601_080_390
#   solver    : FTLM
#   device    : mpi  (n_ranks=8)
#   memory    : per-rank 94.05 GB / avail 14.0 GB    (total 752.4 GB across 8 ranks)
#   wall-time : ~26.7 h  (rough order of magnitude)

# 2. Goal-oriented: "I want X, what's my best shot on this host?"
sug = qed.suggest_workflow(H, intent="thermal", n_samples=8)
print(sug.summary(top=5))
print("Recommended:", sug.best().call_signature() if sug.best() else
      "no feasible plan; pass available_devices=[…] or scale up")

# 3. dry_run: planner-only mode (no kernel dispatch).
qed.diag(H, solver="FTLM", sz=N // 2, dry_run=True)

# 4. force=True: dispatch even when the planner says no.
res = qed.diag(H, solver="LANCZOS", device="mpi_gpu",
               mpi_n_ranks=4, force=True)
```

The verdict is also useful **after the fact** — `report.suggestions`
is a list of strings ranked by impact, so you can show them in a UI
or feed them to a notebook help cell.

### Honest scope

The planner is a feasibility filter, not a perf oracle. The wall-time
estimates use rough constants (100 ns / term-element for the CPU
SpMV, 5 ns for the GPU SpMV, AllReduce latency floor of 50 µs) tuned
from the Phase 6 / Phase 3c bake-offs. Memory estimates are
deliberately conservative — the planner exists to keep you from
OOM'ing, not to forecast within ±10 %. If you have a job script
that allocates more RAM than the planner detected (Slurm / PBS /
container limits), use `force=True` and trust the scheduler.

| host probe | source | override |
| --- | --- | --- |
| CPU RAM | `psutil.virtual_memory().total` -> `/proc/meminfo` -> 16 GB default | `QED_HOST_MEMORY_GB` |
| GPU VRAM | `nvidia-smi --query-gpu=memory.total` (smallest visible device) | `QED_HOST_GPU_MEMORY_GB`, `QED_HOST_N_GPUS` |
| MPI ranks | `SLURM_NTASKS` -> `PBS_NP` -> `OMPI_COMM_WORLD_SIZE` -> CPU count | `QED_HOST_N_MPI_RANKS` |
| build flags | `quantum_ed._core.has_{cuda,mpi,nccl}_build()` | (rebuild) |

---

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
| Compute "will this fit?" with back-of-envelope dim²/sample math | `qed.estimate_resources(H, solver=..., device=..., ...)` (FeasibilityReport with ranked suggestions) |
| "Which solver/device is best on this host for goal X?"     | `qed.suggest_workflow(H, intent="ground_state"/"low_lying"/"thermal"/"spectral")` (ranked candidates) |
| Run the kernel just to check it OOMs                       | `qed.diag(H, ..., dry_run=True)` (planner-only; no kernel dispatch) |

The legacy entry points stay supported — they share the same C++
backend so behaviour is identical — but new code is encouraged to use
`qed.diag` as the single entry point.


---

## How `qed.diag` abstracts the dispatcher

`qed.diag(...)` is a **decision tree**, not a new solver. Every call
goes through the exact same C++ entry point (`exact_diagonalization_*`
in [`include/ed/core/ed_wrapper.h`](../../include/ed/core/ed_wrapper.h))
that the `./ED` CLI uses; the wrapper just makes the choices for you:

```text
              qed.diag(H, num_eigenvalues=k, sz=…, solver=…, device=…, …)
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
   │ 5. Pre-flight planner                               │
   │    plan=True (default): estimate_resources(...)     │
   │      → raises ResourceError if infeasible           │
   │      → force=True overrides; dry_run=True returns   │
   │        the report and skips dispatch.               │
   ├─────────────────────────────────────────────────────┤
   │ 6. Thermal-method bookkeeping                       │
   │    if solver ∈ {mTPQ, cTPQ, FTLM, LTLM, HYBRID}:    │
   │      auto-create output_dir if empty                │
   │      forward num_samples / target_beta / temp_*     │
   │        / num_temp_points to params.tpq_*/ftlm_*/…   │
   ├─────────────────────────────────────────────────────┤
   │ 7. Low-level escape hatch                           │
   │    extra_params={'arpack_ncv': 64,                  │
   │                  'tpq_taylor_order': 200, ...}      │
   │      → setattr(params, key, value) for each pair    │
   │    list_diag_parameters() prints all ~70 fields     │
   ├─────────────────────────────────────────────────────┤
   │ 8. Dispatch: exact_diagonalization_core(...)        │
   └─────────────────────────────────────────────────────┘
```

### Layered API — pick the level of control you want

| Layer | Entry point | Use when |
|------:|------------|----------|
| 0 | `qed.diag(H)` / `qed.dssf.compute(dir)` | **default** — common path, smart defaults |
| 1 | `qed.diag(H, solver=…, device=…, sz=…, …)` | override individual axes |
| 2 | `qed.diag(H, …, extra_params={...})` | tweak any of the ~70 niche `EDParameters` fields |
| 3 | `qed.exact_diagonalization_core(H, method, params)` | full control; populate the `EDParameters` struct yourself |
| 4 | `qed.exact_diagonalization_from_directory(dir, method, params, …)` | start from on-disk Hamiltonian + custom `automorphism_results/` |
| 5 | `qed.mpi.run_distributed(...)` / `qed.dssf.run_from_directory(...)` | shell out to the standalone `mpiexec` / `./ED dssf` binaries with custom flags |

Everything from layer 1 down maps **1-to-1** onto the C++
`EDParameters` fields documented in
[`include/ed/core/ed_parameters.h`](../../include/ed/core/ed_parameters.h).
Run `qed.list_diag_parameters()` to print them grouped by family
(general / thermal / TPQ / FTLM / LTLM / ARPACK / ScaLAPACK /
fixed-Sz / device / symmetry / observables).

### C++ parity table

The C++ surface is identical, just one indirection lower:

| Python layer | C++ equivalent |
|-------------|----------------|
| `qed.diag(H, …)` | `ed::auto_pilot::solve(H, AutoSolveOptions{…})` (header-only, in `include/ed/auto/solve.h`) |
| `qed.dssf.compute(dir, T=…, omega=…)` | `ed::auto_pilot::dssf::compute(DSSFRequest{…}, AutoDSSFOptions{…})` (`include/ed/auto/dssf.h`) |
| `qed.diag(H, extra_params={'arpack_ncv': 64})` | `AutoSolveOptions opts; opts.tune_params = [](EDParameters& p){ p.arpack_ncv = 64; };` |
| `qed.exact_diagonalization_core(H, method, params)` | `exact_diagonalization_core(apply_fn, dim, method, params)` |

The C++ `tune_params` callback runs **after** the auto-pilot has
populated `EDParameters` and **before** the dispatcher fires —
identical semantics to Python's `extra_params=`.

---

## Worked examples — smoke-tested at 12 sites (scale to 32 by changing `N`)

The examples below were exercised locally at `N=12` so they run quickly
and reproducibly on a laptop. To scale the same patterns to 32 sites,
set `N=32`, use `sz=16`, and keep symmetry projection on.

### Common setup

```python
import numpy as np
import quantum_ed as qed

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
res = qed.diag(
    H,
    num_eigenvalues=2,
    sz=N // 2,
    symmetry=report.full_set,
    tolerance=1e-10,
    output_dir="ed_runs/heisenberg_N12_ground",
    compute_eigenvectors=True,
    extra_params={"max_subspace": 160},
)
print("E_0 =", res.eigenvalues[0])
print("Δ   =", res.eigenvalues[1] - res.eigenvalues[0])
```

For the C++ analog:

```cpp
#include <ed/auto/solve.h>
ed::auto_pilot::AutoSolveOptions opts;
opts.num_eigenvalues = 2;
opts.sz              = 6;
opts.tolerance       = 1e-10;
opts.output_dir      = "ed_runs/heisenberg_N12_ground";
opts.compute_eigenvectors = true;
opts.tune_params = [](EDParameters& p) { p.max_subspace = 160; };
auto res = ed::auto_pilot::solve(H, opts);
```

### 2 — Finite-temperature Lanczos (FTLM)

```python
res = qed.diag(
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

# Write a deck to disk then call qed.dssf.compute against it.
lat = qed.input.lattice.chain(N, pbc=True)
builder = qed.input.HamiltonianBuilder(lat.num_sites).heisenberg(lat.nn_pairs(), 1.0)

with tempfile.TemporaryDirectory() as td:
    builder.write_directory(td, lattice=lat)

    # Method auto-selection from (T, omega) axes:
    print(qed.dssf.pick_method(T=[0.5], omega=np.linspace(-1, 1, 64)))
    # -> "dynamical_thermal"

    # Static structure factor S(Q, T) — smoke-tested at N=12
    r_static = qed.dssf.compute(
        td,
        T=[0.5, 1.0],
        method="static_thermal",
        ed_binary="/abs/path/to/ED",   # or put build dir on $PATH
        check=False,
        capture_output=True,
    )

    # Dynamical S(Q, ω, T) — smoke-tested at N=12
    r_dyn = qed.dssf.compute(
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
res = qed.diag(
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
| change `arpack_ncv`, `tpq_taylor_order`, `ftlm_seed`, …  | `qed.diag(…, extra_params={…})` |
| swap the **whole** parameter struct (e.g. copy from CLI) | `qed.exact_diagonalization_core(H, method, params)` |
| list every knob and which family it belongs to           | `qed.list_diag_parameters()` (or `('arpack')`, `('tpq')`, …) |
| ask "would this fit on this host?" without running       | `qed.diag(H, …, dry_run=True)` |
| inspect what the auto-pilot decided                      | `qed.diag(H, …, verbose=True)` (default) |
| force a kernel through despite an `INFEASIBLE` verdict   | `qed.diag(H, …, force=True)` |
| chain low-level kernels yourself (Lanczos → CG → …)      | the dedicated solver modules in `quantum_ed.*` (see `python_advanced.md`) |


---

## DSSF / SSSF — `qed.dssf.compute` auto-pilot + low-level overrides

Spectral / structure-factor calculations have their own one-call
auto-pilot, [`qed.dssf.compute`](../../python/quantum_ed/dssf.py). It
mirrors `qed.diag` exactly: pass a directory + the axes you want, and
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
import numpy as np, quantum_ed as qed

# directory must contain a full ED deck (InterAll.dat, Trans.dat,
# positions.dat/lattice files) and must be compatible with your local
# `ED dssf` binary.

qed.dssf.compute(
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
qed.dssf.compute(
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
| `ed::auto_pilot::dssf::compute(DSSFRequest{...}, AutoDSSFOptions{...})` ([`include/ed/auto/dssf.h`](../../include/ed/auto/dssf.h)) | embedding DSSF in a C++ pipeline |
| `ed::dssf::run(DSSFRequest{...})` | full library-level control from C++ (no shell-out) |

### SSSF (equal-time structure factor)

SSSF is a **special case** of `static_thermal`: pass `T=...` without
`omega=`.

```python
print(qed.dssf.pick_method(T=[0.0, 0.5, 2.0], omega=None))
# -> "static_thermal"

qed.dssf.compute(
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
| `qed.dssf.compute(dir, T=..., omega=...)` | `ed::auto_pilot::dssf::compute(req, opts)` |
| `qed.dssf.pick_method(T=..., omega=...)` | `ed::auto_pilot::dssf::pick_method(has_T, has_w)` |
| `qed.dssf.run_from_directory(dir, method, ...)` | `ed::dssf::run(EDConfig*, req, method)` |

The same auto-rules apply on both sides — see
[`include/ed/auto/dssf.h`](../../include/ed/auto/dssf.h) and the
two-test smoke suite in `tests/unit/test_auto_dssf.cpp`.

---

## C++ mirrors for the workflow usages above

This page is Python-first (`qed.diag`, `qed.find_symmetries`,
`qed.dssf.compute`). The same workflows are available from modern C++
through the auto-pilot façade and the core dispatcher.

### 0) Header set used by the snippets

```cpp
#include <ed/auto/solve.h>
#include <ed/auto/dssf.h>
#include <ed/operators/spin_ops.h>

// Optional low-level entry points:
#include <ed/core/ed_wrapper.h>
#include <ed/core/ed_parameters.h>
```

### 1) TL;DR / Step 1 equivalent — build H and run a one-liner ground state

```cpp
ed::Operator H(/*N=*/12, /*S=*/0.5f);
ed::spin_ops::heisenberg_chain(H, /*N=*/12, /*J=*/1.0, /*pbc=*/true);

ed::auto_pilot::AutoSolveOptions opts;
opts.num_eigenvalues = 1;
auto gs = ed::auto_pilot::solve(H, opts);
std::cout << "E0 = " << gs.eigenvalues.front() << "\n";
```

### 2) Step 3 equivalent — request low-lying states in a fixed-Sz sector

```cpp
ed::auto_pilot::AutoSolveOptions opts;
opts.num_eigenvalues = 4;
opts.sz = 6;                       // N/2 for N=12
opts.tolerance = 1e-10;
auto res = ed::auto_pilot::solve(H, opts);
```

### 3) Maximum projection recipe (symmetry + Sz)

Symmetry projection in C++ is configured through the same dispatcher
axis as Python (`params.use_symmetry = true`) and runs via the
streaming-symmetry kernel when generators/metadata are present in the
workflow directory.

```cpp
EDParameters p;
p.use_fixed_sz = true;
p.n_up = 6;
p.use_symmetry = true;
p.compute_eigenvectors = true;
p.output_dir = "ed_runs/12site_J1";

// The symmetry metadata (automorphism_results/*) is read from directory.
auto out = exact_diagonalization_from_directory(
  "ed_runs/12site_J1",
  DiagonalizationMethod::LANCZOS,
  p);
```

### 4) Force GPU recipe

```cpp
ed::auto_pilot::AutoSolveOptions opts;
opts.device = ed::auto_pilot::Device::GPU;
auto res = ed::auto_pilot::solve(H, opts);
```

### 5) Thermal mTPQ recipe

```cpp
ed::auto_pilot::AutoSolveOptions opts;
opts.solver = DiagonalizationMethod::mTPQ;
opts.sz = 6;                       // optional fixed-Sz projection
opts.output_dir = "ed_runs/12site_thermal";
opts.tune_params = [](EDParameters& p) {
  p.num_samples = 8;
  p.target_beta = 20.0;
  p.num_temp_points = 40;
};
auto tpq = ed::auto_pilot::solve(H, opts);
```

### 6) MPI / distributed recipe (library + launcher split)

The in-process C++ API can mark MPI mode (`params.use_mpi=true`) when
already running under an MPI launcher. For out-of-process rank launch,
use the same `ed_distributed_main` binary as Python.

```cpp
EDParameters p;
p.use_mpi = true;
p.num_eigenvalues = 4;
auto out = exact_diagonalization_from_directory(
  "ed_runs/24site_mpi",
  DiagonalizationMethod::LANCZOS,
  p);
```

```bash
mpiexec -np 8 ./ed_distributed_main --mode lanczos --directory ed_runs/24site_mpi
mpiexec -np 4 ./ed_distributed_main --mode tpq --gpu --directory ed_runs/24site_mpi
```

### 7) Low-level escape hatch (Python extra_params equivalent)

```cpp
ed::auto_pilot::AutoSolveOptions opts;
opts.num_eigenvalues = 4;
opts.tune_params = [](EDParameters& p) {
  p.arpack_ncv = 96;
  p.max_subspace = 220;
  p.ftlm_krylov_dim = 180;
};
auto out = ed::auto_pilot::solve(H, opts);
```

Or bypass auto-pilot entirely:

```cpp
EDParameters p;
p.num_eigenvalues = 4;
p.tolerance = 1e-10;
p.max_iterations = 300;

const std::uint64_t dim = (1ULL << H.getNumBits());
auto apply = [&H](const Complex* in, Complex* out, int n) {
  H.apply(in, out, static_cast<size_t>(n));
};

auto out = exact_diagonalization_core(apply, dim,
                    DiagonalizationMethod::LANCZOS,
                    p);
```

### 8) DSSF / SSSF routine equivalents

Method selection is the same 2x2 rule as Python and can be queried by
`ed::auto_pilot::dssf::pick_method(...)`.

```cpp
using ed::auto_pilot::dssf::pick_method;
auto m0 = pick_method(/*has_temperature=*/false, /*has_frequency=*/false);
auto m1 = pick_method(/*has_temperature=*/false, /*has_frequency=*/true);
auto m2 = pick_method(/*has_temperature=*/true,  /*has_frequency=*/false);
auto m3 = pick_method(/*has_temperature=*/true,  /*has_frequency=*/true);
```

```cpp
ed::dssf::DSSFRequest req;
req.output_dir = "runs/heisenberg_N16";
req.config = &cfg;                  // required for non-single-expectation
// Fill req.operators (OperatorSpec) as needed.

ed::auto_pilot::dssf::AutoDSSFOptions o;
o.has_temperature = true;
o.has_frequency = true;
auto dssf = ed::auto_pilot::dssf::compute(req, o);
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

1. Ground state: `AutoSolveOptions{num_eigenvalues=2, sz=6}` plus
   `tune_params(max_subspace=...)`.
2. FTLM: `opts.solver = DiagonalizationMethod::FTLM`, set
   `num_samples`, `temp_min/max`, and `ftlm_krylov_dim` in
   `tune_params`.
3. DSSF: build `DSSFRequest` for your directory and call
  `ed::auto_pilot::dssf::compute(req, dssf_opts)` with
  `dssf_opts.has_temperature = true; dssf_opts.has_frequency = true;`.
4. mTPQ/cTPQ: `opts.solver = DiagonalizationMethod::mTPQ` (or `cTPQ`
   for MPI sample-splitting), set `target_beta` + TPQ knobs in
   `tune_params`.
