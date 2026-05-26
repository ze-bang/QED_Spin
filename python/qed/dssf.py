"""``qed.dssf``: DSSF observable-pair data helpers.

This module re-exports the C++ ``ed::dssf`` observable-pair builder so
Python notebooks and downstream scripts produce **byte-identical**
observable names and ordering with the C++ ``./ED dssf`` workflow.

The actual workflow lives in :func:`qed.spectral`; the ``compute`` and
``run_from_directory`` helpers that used to live here were removed
during the May-2026 surface unification. Use:

* :func:`qed.spectral(H, observables, ...) <qed.spectral>` -- in-memory
  spectral / structure-factor calculation.

* :func:`qed.spectral(directory, ...) <qed.spectral>` -- directory-form
  CLI workflow (shells out to ``./ED dssf``).

Quick start (observable assembly):

.. code-block:: python

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
    for name in pairs.names:
        print(name)

    # Feed the pairs into qed.spectral or any in-process solver:
    res = qed.spectral(H, pairs.obs_1, omega=np.linspace(-2, 2, 200))
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
