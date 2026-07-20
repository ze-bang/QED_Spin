# QED — Quantum Exact Diagonalization

A modern C++17 / CUDA / MPI / Python toolkit for **exact diagonalization
(ED)** of spin Hamiltonians.

The user-facing surface is **three orchestrator verbs**, exposed in
parallel in C++ and in Python:

| C++                              | Python                                |
|----------------------------------|---------------------------------------|
| `ed::workflows::solve(op, opts)` | `qed.solve(H, **kw)`                  |
| `ed::workflows::thermal(op, opts)` | `qed.thermal(H, **kw)`              |
| `ed::workflows::spectral(op, opts)` | `qed.spectral(H, **kw)`            |

Operators are built once via `ed::make_operator(OperatorSpec)` (or
`qed.input.HamiltonianBuilder` in Python) and consumed by every
backend (CPU / single-GPU / MPI / MPI+GPU) without further surgery.
Symmetry projection is orthogonal: pick a `Subspace` (full Hilbert
space, fixed total Sz, or the Sz-parity half of a broken U(1)) and a
`ProjectorChain` (zero or more group representations: the lattice
space group, the global Z₂ spin-flip, time-reversal pairing, and the
non-abelian little-group projection); a future SU(2) total-S axis
would extend the chain through the same kwargs.

This site combines the **C++ API reference** (auto-extracted from the
headers under `include/ed/` via Doxygen + Breathe) with **prose
chapters** that walk through how the pieces fit together end-to-end.

```{toctree}
:maxdepth: 2
:caption: Getting started

guides/install
guides/quickstart
guides/python_quickstart
guides/one_call_api
guides/workflow
guides/python_advanced
guides/python_api_coverage
```

```{toctree}
:maxdepth: 2
:caption: Reference

api/cpp
api/python
```

```{toctree}
:maxdepth: 2
:caption: Architecture

architecture/ARCHITECTURE
architecture/SYMMETRY
architecture/CODEMAP
architecture/SCALING
architecture/ADD_NEW_BASIS_POLICY
architecture/ADD_NEW_GPU_CELL
architecture/ADD_NEW_MPI_CELL
```

```{toctree}
:maxdepth: 1
:caption: Benchmarks

benchmarks/BENCHMARKS
benchmarks/bench_vs_xdiag
benchmarks/ORTHOGONAL_SYMMETRY
```

## Project documents

The following live at the repository root and are rendered on GitHub:

- [`CHANGELOG.md`](https://github.com/ze-bang/QED/blob/main/CHANGELOG.md)
  — versioned release notes.
- [`CONTRIBUTING.md`](https://github.com/ze-bang/QED/blob/main/CONTRIBUTING.md)
  — how to set up a dev environment and submit changes.
- [`docs/architecture/ARCHITECTURE.md`](https://github.com/ze-bang/QED/blob/main/docs/architecture/ARCHITECTURE.md)
  — the post-collapse architectural picture (read first).
- [`docs/architecture/SYMMETRY.md`](https://github.com/ze-bang/QED/blob/main/docs/architecture/SYMMETRY.md)
  — Subspace × ProjectorChain math + workflows.
- [`docs/architecture/CODEMAP.md`](https://github.com/ze-bang/QED/blob/main/docs/architecture/CODEMAP.md)
  — directory-level tour.
- [`docs/architecture/SCALING.md`](https://github.com/ze-bang/QED/blob/main/docs/architecture/SCALING.md)
  — memory + N envelope, env-var knobs.
- [`docs/benchmarks/BENCHMARKS.md`](https://github.com/ze-bang/QED/blob/main/docs/benchmarks/BENCHMARKS.md)
  — head-to-head vs QuSpin / SciPy.
- [`docs/benchmarks/ORTHOGONAL_SYMMETRY.md`](https://github.com/ze-bang/QED/blob/main/docs/benchmarks/ORTHOGONAL_SYMMETRY.md)
  — full 4 × 6 sweep (four `(Subspace, ProjectorChain)` cells × six
  workflows) on CPU and GPU.
- [`examples/`](https://github.com/ze-bang/QED/tree/main/examples)
  — runnable C++ / Python / CLI examples (one per use case).
- [`docs/history/`](https://github.com/ze-bang/QED/tree/main/docs/history)
  — legacy phase summaries kept as time-capsules.

## Indices and tables

- {ref}`genindex`
- {ref}`search`
