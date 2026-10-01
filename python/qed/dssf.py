"""``qed.dssf``: operators for dynamical structure factors.

Builds the momentum-resolved spin operators, one per (Q, component), with their names and
ordering, the way the C++ ``ed::dssf`` layer does; feed them to :func:`qed.dynamics`.

.. code-block:: python

    import numpy as np
    import qed

    spec = qed.dssf.OperatorSpec()
    spec.operator_type   = "transverse"
    spec.basis           = "xyz"
    spec.components      = [0, 1]          # Sx, Sy in the xyz basis
    spec.momentum_points = [[0.0, 0.0, 0.0], [3.14159, 0.0, 0.0]]
    spec.polarization    = [0.0, 0.0, 1.0]
    spec.num_sites       = 4
    spec.spin_length     = 0.5
    spec.positions_file  = "/abs/path/to/positions.dat"

    obs = qed.dssf.build_observables(spec)
    S = qed.dynamics(H, obs.operators[0], np.linspace(-2, 2, 200)).S[0]
"""

from __future__ import annotations

from ._core.dssf import (  # type: ignore[attr-defined]
    Observables,
    OperatorSpec,
    build_observables,
    compute_transverse_bases,
)

__all__ = [
    "Observables",
    "OperatorSpec",
    "build_observables",
    "compute_transverse_bases",
]
