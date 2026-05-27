#!/usr/bin/env python3
"""examples/09_python_quickstart.py

The same 4-site Heisenberg ground state as ``01_cpp_ground_state.cpp``,
this time through the Python bindings (``pip install -e .`` from the
repo root, which installs the ``qed`` package).

Showcases the one canonical Python verb: :func:`qed.solve`.
The legacy ``qed.workflows`` module and ``qed.diag`` alias were removed
during the May-2026 surface unification.

Run::

    python3 examples/09_python_quickstart.py
    python3 examples/09_python_quickstart.py 8 1   # N=8, PBC
"""
from __future__ import annotations

import sys

import qed


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

    # One call. Auto-Sz projects to the half-filling sector when Sz is
    # conserved; auto-method picks LANCZOS/FULL/KRYLOV_SCHUR based on
    # dim + num_eigenvalues; auto-device picks CPU/GPU.
    result = qed.solve(
        op,
        num_eigenvalues=5,
        solver="LANCZOS",
        tolerance=1e-12,
        verbose=False,
        plan=False,
    )

    print("Lowest 5 eigenvalues:")
    for k, e in enumerate(result.eigenvalues[:5]):
        print(f"  E[{k}] = {e:.10f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
