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
