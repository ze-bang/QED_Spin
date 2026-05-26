# Python quickstart (`qed`)

The Python facade lives under the `qed` package. It is a `pybind11`
layer over the same `ed_solvers_*` static libraries that the C++ CLI
uses, so results are bit-identical. Since Phase 5 (Apr 2026) it reaches
**every backend** the CLI knows about (CPU iterative + dense, FTLM /
LTLM / TPQ, GPU per-method and per-sector, in-process symmetry
projection, ScaLAPACK) plus thin launcher helpers for the MPI
distributed solvers and the full `./ED dssf` continued-fraction engine.

This page covers the everyday entry points; for the full advanced
catalogue (single-call dispatcher across ~30 solver variants, GPU
streaming symmetry, MPI launcher, build introspection, end-to-end
worked example) jump to
[`python_advanced.md`](python_advanced.md).

## Heisenberg chain in 12 lines

```python
import numpy as np
import qed as qe

# 2-site Heisenberg dimer, S = 1/2
op = qe.Operator(num_sites=2, spin_length=0.5)
op.add_two_body(qe.OP_SPLUS,  0, qe.OP_SMINUS, 1, 0.5)
op.add_two_body(qe.OP_SMINUS, 0, qe.OP_SPLUS,  1, 0.5)
op.add_two_body(qe.OP_SZ,     0, qe.OP_SZ,     1, 1.0)

result = qe.full_diagonalization(op)
print("Eigenvalues:", result.eigenvalues)
# Expected: [-0.75, 0.25, 0.25, 0.25]
```

## Lanczos for large systems

```python
result = qe.lanczos(op, max_iter=200, exct=4, tolerance=1e-12)
print("Lowest 4 eigenvalues:", result.eigenvalues[:4])
```

## One-call auto-pilot — `qe.diag(...)` and `qe.dssf.compute(...)`

Phase 9 ships a **stress-free unified entry point** that picks the
solver, device, and Sz sector for you. Use it whenever you don't need
to override individual knobs:

```python
# Eigenvalues only — auto-picks FULL for tiny dim, LANCZOS / KRYLOV_SCHUR
# for medium / many eigenpairs, promotes to GPU if WITH_CUDA and
# sector_dim ≥ 2^17, and projects onto the Sz=N/2 sector when the
# Hamiltonian conserves total Sz.
res = qe.diag(op, num_eigenvalues=4)         # smart defaults

# Force CPU + explicit sector. The Sz guard refuses sz= for
# Sz-breaking Hamiltonians (transverse field, etc.) instead of
# silently giving the wrong answer.
res = qe.diag(op, num_eigenvalues=4, device="cpu", sz=N // 2)

# Thermal trajectory — auto-creates an output directory.
res = qe.diag(op, solver="mTPQ", sz=N // 2,
              num_samples=4, target_beta=20.0)
```

The DSSF pipeline has the same shape via
`qe.dssf.compute(directory, T=..., omega=...)`. The `(T, omega)` tuple
selects the kernel for you (the same rule as the C++ `DSSFMethod`
enum):

| `T` given? | `omega` given? | Method picked        |
|-----------:|---------------:|----------------------|
| no         | no             | `single_expectation` |
| no         | yes            | `ground_state_dssf`  |
| yes        | no             | `static_thermal`     |
| yes        | yes            | `dynamical_thermal`  |

```python
# Static structure factor at one T:
qe.dssf.compute("runs/heisenberg6", T=0.5)

# T = 0 dynamical S(Q, ω):
qe.dssf.compute("runs/heisenberg6", omega=np.linspace(-2, 2, 200))

# Full S(Q, ω, T):
qe.dssf.compute("runs/heisenberg6",
                T=[0.1, 0.3, 1.0],
                omega=np.linspace(-2, 2, 200))
```

For maximum control over per-method knobs (frequency window, Krylov
dim, num_random_states, …) keep using `qe.dssf.run_from_directory(...)`
with explicit `extra_args`.

## Fixed-Sz sector

```python
op = qe.FixedSzOperator(num_sites=8, spin_length=0.5, n_up=4)
# ... add terms as above (they will be projected onto the n_up=4 sector)
res = qe.full_diagonalization(op)
```

## Hamiltonian builder DSL

For textbook lattice models the `qed.hamiltonian` submodule wraps
the C++ `Operator` API in a fluent QuSpin-style builder. The 4-site
Heisenberg chain becomes one line:

```python
import qed as qe

H = (
    qe.hamiltonian.Hamiltonian(num_sites=4)
    .heisenberg([(0, 1), (1, 2), (2, 3)])
    .build()
)
print(qe.full_diagonalization(H).min())   # -1.6160254037844388
```

The builder accepts case-insensitive operator-token strings (`"x"`,
`"y"`, `"z"`, `"+"`, `"-"`, plus the verbose `"sx"`/`"sy"`/`"sz"` forms);
`Sx` and `Sy` are auto-expanded onto the underlying `S+`/`S-` primitives
at `build()` time. Common Hamiltonians get one-liner shortcuts
(`heisenberg`, `transverse_field_ising`, `xx_yy`, `zz`, `field`),
and arbitrary terms can always be added via `.add(("x", "z"), (0, 1),
coeff)`. The `n_up=` constructor kwarg returns a `FixedSzOperator`
restricted to the requested sector, so symmetry-resolved sweeps stay
ergonomic:

```python
H_singlet = (
    qe.hamiltonian.Hamiltonian(num_sites=4, n_up=2)
    .heisenberg([(0, 1), (1, 2), (2, 3)])
    .build()
)
H_singlet.dimension   # C(4, 2) == 6
```

## DSSF: building observable pairs

The `qed.dssf` submodule exposes the *same* `OperatorSpec` /
`build_observable_pairs` / `compute_transverse_bases` C++ functions used by
the `ED dssf` subcommand, so a Python script can produce a byte-identical
pair list (operators, names, ordering) and feed them into the solvers
without any glue code:

```python
import qed as qe

spec = qe.dssf.OperatorSpec()
spec.operator_type     = "transverse"
spec.basis             = "xyz"
spec.spin_combinations = [("x", "x"), ("y", "y")]
spec.momentum_points   = [[0.0, 0.0, 0.0], [3.14159, 0.0, 0.0]]
spec.polarization      = [0.0, 0.0, 1.0]
spec.unit_cell_size    = 4
spec.num_sites         = 4
spec.spin_length       = 0.5
spec.positions_file    = "/abs/path/to/positions.dat"

pairs = qe.dssf.build_observable_pairs(spec)
for name in pairs.names:
    print(name)
```

The returned `ObservablePairs` carries three parallel lists --- `obs_1`,
`obs_2`, `names` --- of equal length. The `Operator` instances inside are the
same C++ ones bound on the top-level facade, so `qe.lanczos(pair_obs_1)` /
`qe.finite_temperature_lanczos(...)` work directly on them.

## Single-call dispatcher (Phase 5)

The thin wrappers above (`qe.full_diagonalization`,
`qe.compute_thermodynamics_from_spectrum`, …) cover specific
low-level entry points. For the full backend matrix use the unified
three-verb orchestrator (`qe.solve` / `qe.thermal` / `qe.spectral`):

```python
import qed as qe

# Eigenvalue solvers: LANCZOS / BLOCK_LANCZOS / KRYLOV_SCHUR / FULL
res = qe.solve(op, num_eigenvalues=4, solver="KRYLOV_SCHUR",
               tolerance=1e-12)

# Finite-temperature trajectories: mTPQ / cTPQ / FTLM / LTLM
thermo = qe.thermal(op, method="FTLM",
                    num_samples=4,
                    temp_min=0.05, temp_max=10.0, num_temp_points=200)

# GPU per-sector with symmetry projection (large clusters):
if qe.has_cuda_build():
    res = qe.solve(op, symmetry=info, device="gpu",
                   num_eigenvalues=2)

# MPI distributed solvers (helper builds the mpiexec / srun argv):
if qe.has_mpi_build():
    qe.mpi.run_distributed("./my_dir", method="lanczos", n_ranks=8)

# Full continued-fraction S(Q,omega) engine:
qe.spectral("./my_dir", T=[0.1, 0.3, 1.0],
            omega=[-2, -1, 0, 1, 2])
```

See [`python_advanced.md`](python_advanced.md) for the full pattern
catalogue (every `EDParameters` knob via `extra_params={}`, GPU
per-method dispatch, in-process symmetry round-trip, choosing between
`qe.solve` / `qe.thermal` / `qe.spectral`).

## Backwards compatibility with `edlib`

The legacy `edlib` Python helpers (`hdf5_io`, `automorphism_finder`, …) are
still importable verbatim:

```python
from edlib import hdf5_io, automorphism_finder   # legacy code keeps working
```

New code should prefer the namespaced re-exports:

```python
from qed.helpers import hdf5_io, automorphism_finder
```

The `edlib` shim stays in place so that downstream notebooks and scripts
don't break, but every new feature lands under `qed.*`.
