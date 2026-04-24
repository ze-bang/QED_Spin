# Python quickstart (`quantum_ed`)

The Python facade lives under the `quantum_ed` package. It is a thin
`pybind11` layer over the same `ed_solvers_cpu` static library that the C++
CLI uses, so results are bit-identical.

## Heisenberg chain in 12 lines

```python
import numpy as np
import quantum_ed as qe

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

## Fixed-Sz sector

```python
op = qe.FixedSzOperator(num_sites=8, spin_length=0.5, n_up=4)
# ... add terms as above (they will be projected onto the n_up=4 sector)
res = qe.full_diagonalization(op)
```

## Hamiltonian builder DSL

For textbook lattice models the `quantum_ed.hamiltonian` submodule wraps
the C++ `Operator` API in a fluent QuSpin-style builder. The 4-site
Heisenberg chain becomes one line:

```python
import quantum_ed as qe

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

The `quantum_ed.dssf` submodule exposes the *same* `OperatorSpec` /
`build_observable_pairs` / `compute_transverse_bases` C++ functions used by
the `ED dssf` subcommand, so a Python script can produce a byte-identical
pair list (operators, names, ordering) and feed them into the solvers
without any glue code:

```python
import quantum_ed as qe

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

## Backwards compatibility with `edlib`

The legacy `edlib` Python helpers (`hdf5_io`, `automorphism_finder`, …) are
still importable verbatim:

```python
from edlib import hdf5_io, automorphism_finder   # legacy code keeps working
```

New code should prefer the namespaced re-exports:

```python
from quantum_ed.helpers import hdf5_io, automorphism_finder
```

The `edlib` shim stays in place so that downstream notebooks and scripts
don't break, but every new feature lands under `quantum_ed.*`.
