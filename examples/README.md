# `examples/`

Six short scripts; each runs in seconds:

| script | verbs |
|---|---|
| [`01_levels.py`](01_levels.py) | `qed.eigs` (levels, labels, eigenvectors), `qed.expect` |
| [`02_thermal.py`](02_thermal.py) | `qed.thermal`: exact, FTLM, FTLM with exact low-lying states, mTPQ |
| [`03_dynamics.py`](03_dynamics.py) | `qed.dynamics` at T = 0 and finite T |
| [`04_symmetry.py`](04_symmetry.py) | `qed.find_symmetries`, `qed.Symmetry` options, sector selection |
| [`05_operator_algebra.py`](05_operator_algebra.py) | `qed.Operator` algebra: products, sums, adjoints, exact symmetry checks, derived observables |
| [`06_cross_dynamics.py`](06_cross_dynamics.py) | `qed.dynamics` cross-correlations: several probes, `B="all"` |

Every verb takes `sym=` (default `qed.Symmetry.auto()`) and `device=` (`"cpu"`, `"gpu"`,
`"auto"`). Correctness of every task x symmetry x backend combination is pinned by the
coverage grid in `tests/python/grid/`.
