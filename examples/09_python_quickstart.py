#!/usr/bin/env python3
"""examples/09_python_quickstart.py

The same 4-site Heisenberg ground state as `01_cpp_ground_state.cpp`, this
time through the Python bindings (`pip install -e .` from the repo root,
which installs the `qed` package).

Showcases the unified Python entry point :mod:`qed.workflows` introduced
by the Full Unified-Interface Collapse (May 2026):

    op -> qed.workflows.solve(op, opts)

This is the canonical "hello world" for the toolkit's Python surface.

Run::

    python3 examples/09_python_quickstart.py
    python3 examples/09_python_quickstart.py 8 1   # N=8, PBC
"""
from __future__ import annotations

import sys

import qed
from qed import workflows


def build_heisenberg_chain(N: int, periodic: bool, J: float = 1.0) -> qed.Operator:
    op = qed.Operator(N, 0.5)
    last = N if periodic else N - 1
    for i in range(last):
        j = (i + 1) % N
        op.add_two_body(qed.OP_SZ, i, qed.OP_SZ, j, J)
        op.add_two_body(qed.OP_SPLUS, i, qed.OP_SMINUS, j, 0.5 * J)
        op.add_two_body(qed.OP_SMINUS, i, qed.OP_SPLUS, j, 0.5 * J)
    return op


def main() -> int:
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    periodic = bool(int(sys.argv[2])) if len(sys.argv) > 2 else False

    op = build_heisenberg_chain(N, periodic)
    print(f"Heisenberg chain  N={N}  PBC={int(periodic)}  dim={2**N}")

    opts = workflows.SolveOptions()
    opts.num_eigs = 5
    opts.method = workflows.SolveMethod.Lanczos
    opts.tolerance = 1e-12
    opts.compute_vectors = False

    result = workflows.solve(op, opts)

    print(f"  backend lane = {result.backend.lane}  "
          f"(wall = {result.backend.wall_seconds:.4f} s)")
    print("Lowest 5 eigenvalues:")
    for k, e in enumerate(result.eigenvalues[:5]):
        print(f"  E[{k}] = {e:.10f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
