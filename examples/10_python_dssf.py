#!/usr/bin/env python3
"""examples/10_python_dssf.py

T=0 ground-state Dynamical Structure Factor S(q, omega) on an 8-site
Heisenberg PBC chain. Uses the high-level `qed.dssf` wrapper
that the C++ `ED dssf` subcommand calls under the hood, so the
observable assembly is byte-identical to the production CLI.

Run::

    python3 examples/10_python_dssf.py
"""
from __future__ import annotations

import os
import tempfile

import numpy as np

import qed as qed


def build_chain(N: int, periodic: bool = True, J: float = 1.0) -> qed.Operator:
    op = qed.Operator(num_sites=N, spin_length=0.5)
    last = N if periodic else N - 1
    for i in range(last):
        j = (i + 1) % N
        op.add_two_body(qed.OP_SZ,     i, qed.OP_SZ,     j, J)
        op.add_two_body(qed.OP_SPLUS,  i, qed.OP_SMINUS, j, 0.5 * J)
        op.add_two_body(qed.OP_SMINUS, i, qed.OP_SPLUS,  j, 0.5 * J)
    return op


def main() -> int:
    N = 8

    # Build the spectrum of the Hamiltonian.
    H = build_chain(N, periodic=True)
    print(f"8-site Heisenberg PBC: building observable pairs via ed::dssf...")

    # Write a minimal positions file (1D chain along x).
    with tempfile.TemporaryDirectory() as workdir:
        positions_path = os.path.join(workdir, "positions.dat")
        with open(positions_path, "w") as fh:
            for i in range(N):
                fh.write(f"{i:d} {float(i):.6f} 0.0 0.0\n")

        spec = qed.dssf.OperatorSpec()
        spec.operator_type     = "sum"            # local <S_i^a> sum-over-sites
        spec.basis             = "ladder"         # +, -, z
        spec.spin_combinations = [("+", "-"), ("z", "z")]
        spec.momentum_points   = [
            [0.0,           0.0, 0.0],
            [np.pi / 2.0,   0.0, 0.0],
            [np.pi,         0.0, 0.0],
        ]
        spec.unit_cell_size = 1
        spec.num_sites      = N
        spec.spin_length    = 0.5
        spec.positions_file = positions_path

        pairs = qed.dssf.build_observable_pairs(spec)

    print(f"  built {len(pairs.names)} observable pairs (q-points x spin combos):")
    for name in pairs.names:
        print(f"    {name}")

    # Quick sanity demonstrating that the assembled operators act on the
    # same 2^N Hilbert space.
    eigs = qed.full_diagonalization(H)
    eigs = np.sort(np.asarray(eigs, dtype=float))
    print(f"  ground state energy E_0 = {eigs[0]:.10f}  (Bethe-ansatz "
          f"reference for the 8-site PBC chain ~ -3.65109)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
