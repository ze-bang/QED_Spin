#!/usr/bin/env python3
"""examples/10_python_dssf.py

T=0 ground-state Dynamical Structure Factor S(q, omega) on an 8-site
Heisenberg PBC chain.

Uses :mod:`qed.dssf` to assemble the momentum-resolved observable pairs
(byte-identical to the production CLI), then drives the unified Python
verbs :func:`qed.solve` (for the ground-state energy) and
:func:`qed.spectral` (for the dynamical correlator).

Run::

    python3 examples/10_python_dssf.py
"""
from __future__ import annotations

import os
import tempfile

import numpy as np

import qed


def build_chain(N: int, periodic: bool = True, J: float = 1.0) -> qed.Operator:
    op = qed.Operator(N, 0.5)
    last = N if periodic else N - 1
    for i in range(last):
        j = (i + 1) % N
        op.add_two_body(qed.OP_SZ, i, qed.OP_SZ, j, J)
        op.add_two_body(qed.OP_SPLUS, i, qed.OP_SMINUS, j, 0.5 * J)
        op.add_two_body(qed.OP_SMINUS, i, qed.OP_SPLUS, j, 0.5 * J)
    return op


def main() -> int:
    N = 8

    H = build_chain(N, periodic=True)
    print("8-site Heisenberg PBC: building observable pairs via qed.dssf...")

    with tempfile.TemporaryDirectory() as workdir:
        positions_path = os.path.join(workdir, "positions.dat")
        with open(positions_path, "w") as fh:
            for i in range(N):
                fh.write(f"{i:d} {float(i):.6f} 0.0 0.0\n")

        spec = qed.dssf.OperatorSpec()
        spec.operator_type = "sum"
        spec.basis = "ladder"
        spec.spin_combinations = [(qed.OP_SPLUS, qed.OP_SMINUS),
                                  (qed.OP_SZ, qed.OP_SZ)]
        spec.momentum_points = [
            [0.0, 0.0, 0.0],
            [np.pi / 2.0, 0.0, 0.0],
            [np.pi, 0.0, 0.0],
        ]
        spec.unit_cell_size = 1
        spec.num_sites = N
        spec.spin_length = 0.5
        spec.positions_file = positions_path

        pairs = qed.dssf.build_observable_pairs(spec)

    print(f"  built {len(pairs.names)} observable pairs:")
    for name in pairs.names:
        print(f"    {name}")

    print("\nRunning qed.solve(H, num_eigenvalues=3) to get the ground state...")
    gs = qed.solve(H,
                   num_eigenvalues=3,
                   solver="LANCZOS",
                   tolerance=1e-12,
                   verbose=False,
                   plan=False,
                   auto_sz=False)
    print(f"  E_0 = {gs.eigenvalues[0]:.10f}  "
          f"(Bethe-ansatz reference for 8-site PBC chain ~ -3.65109)")

    print("\nRunning qed.spectral(H, [obs0], ...) on the first observable pair...")
    obs0 = pairs.obs_1[0]
    spec_result = qed.spectral(
        H,
        [obs0],
        method="ground_state_cf",
        omega=np.linspace(-8.0, 8.0, 16),
        eta=0.1,
        krylov_dim=80,
        verbose=False,
    )
    print(f"  computed S(omega) at {len(spec_result.omega)} omega points")
    print(f"  omega range = [{spec_result.omega[0]:.3f}, "
          f"{spec_result.omega[-1]:.3f}]")
    print(f"  S_real[0] = {spec_result.S_real[0]:.6f}, "
          f"S_real[-1] = {spec_result.S_real[-1]:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
