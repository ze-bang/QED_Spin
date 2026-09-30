"""``qed.dssf``: operators for dynamical structure factors.

Builds the momentum-resolved spin operators (and their names and ordering) the way the
C++ ``ed::dssf`` layer does; feed them to :func:`qed.dynamics`.

.. code-block:: python

    import numpy as np
    import qed

    spec = qed.dssf.OperatorSpec()
    spec.operator_type     = "transverse"
    spec.basis             = "xyz"
    spec.spin_combinations = [("x", "x"), ("y", "y")]
    spec.momentum_points   = [[0.0, 0.0, 0.0], [3.14159, 0.0, 0.0]]
    spec.polarization      = [0.0, 0.0, 1.0]
    spec.unit_cell_size    = 4
    spec.num_sites         = 4
    spec.spin_length       = 0.5
    spec.positions_file    = "/abs/path/to/positions.dat"

    pairs = qed.dssf.build_observable_pairs(spec)
    S = qed.dynamics(H, pairs.obs_1[0], np.linspace(-2, 2, 200)).S[0]
"""

from __future__ import annotations

from ._core.dssf import (  # type: ignore[attr-defined]
    ObservablePairs,
    OperatorSpec,
    build_observable_pairs,
    compute_transverse_bases,
)

__all__ = [
    "ObservablePairs",
    "OperatorSpec",
    "build_observable_pairs",
    "compute_transverse_bases",
]
