#!/usr/bin/env python3
"""examples/10_python_dssf.py

T=0 ground-state Dynamical Structure Factor S(q, omega) on an 8-site
Heisenberg PBC chain.

Uses the high-level `qed.dssf` wrapper to assemble the momentum-resolved
operator pairs (byte-identical to the production CLI), then runs the
unified Python entry points :func:`qed.workflows.solve` for the
spectrum and :func:`qed.workflows.spectral` for the dynamical
correlator -- both introduced by the Full Unified-Interface Collapse
(May 2026).

Run::

    python3 examples/10_python_dssf.py
"""
from __future__ import annotations

import os
import tempfile

import numpy as np

import qed
from qed import workflows


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

    print("\nRunning qed.workflows.solve to get the ground-state energy...")
    solve_opts = workflows.SolveOptions()
    solve_opts.num_eigs = 3
    solve_opts.method = workflows.SolveMethod.Lanczos
    solve_opts.tolerance = 1e-12
    gs = workflows.solve(H, solve_opts)
    print(f"  E_0 = {gs.eigenvalues[0]:.10f}  "
          f"(Bethe-ansatz reference for 8-site PBC chain ~ -3.65109)")
    print(f"  backend lane = {gs.backend.lane}")

    print("\nRunning qed.workflows.spectral on the first observable pair...")
    obs0 = pairs.obs_1[0]
    sp_opts = workflows.SpectralOptions()
    sp_opts.method = workflows.SpectralMethod.GroundStateCF
    sp_opts.num_omega = 16
    sp_opts.omega_min = -8.0
    sp_opts.omega_max = 8.0
    sp_opts.broadening = 0.1
    sp_opts.krylov_dim = 80
    spec_result = workflows.spectral(H, [obs0], sp_opts)
    print(f"  computed S(omega) at {len(spec_result.omega)} omega points")
    print(f"  omega range = [{spec_result.omega[0]:.3f}, "
          f"{spec_result.omega[-1]:.3f}]")
    print(f"  S_real[0] = {spec_result.S_real[0]:.6f}, "
          f"S_real[-1] = {spec_result.S_real[-1]:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
