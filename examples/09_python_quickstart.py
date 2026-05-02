#!/usr/bin/env python3
"""examples/09_python_quickstart.py

The same 4-site Heisenberg ground state as `01_cpp_ground_state.cpp`, this
time through the Python bindings (`pip install -e .` from the repo root,
which installs the `qed` package).

Run::

    python3 examples/09_python_quickstart.py
    python3 examples/09_python_quickstart.py 8 1   # N=8, PBC

This is the canonical "hello world" for the toolkit's Python surface.
"""
from __future__ import annotations

import sys

import numpy as np

import qed as qed


def build_heisenberg_chain(N: int, periodic: bool, J: float = 1.0) -> qed.Operator:
    op = qed.Operator(num_sites=N, spin_length=0.5)
    last = N if periodic else N - 1
    for i in range(last):
        j = (i + 1) % N
        op.add_two_body(qed.OP_SZ, i, qed.OP_SZ, j, J)
        op.add_two_body(qed.OP_SPLUS,  i, qed.OP_SMINUS, j, 0.5 * J)
        op.add_two_body(qed.OP_SMINUS, i, qed.OP_SPLUS,  j, 0.5 * J)
    return op


def main() -> int:
    N        = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    periodic = bool(int(sys.argv[2])) if len(sys.argv) > 2 else False
    op = build_heisenberg_chain(N, periodic)

    print(f"Heisenberg chain  N={N}  PBC={int(periodic)}  dim={2**N}")
    eigs = qed.full_diagonalization(op)
    eigs = np.sort(np.asarray(eigs, dtype=float))
    print("Lowest 5 eigenvalues:")
    for k in range(min(5, len(eigs))):
        print(f"  E[{k}] = {eigs[k]:.10f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
