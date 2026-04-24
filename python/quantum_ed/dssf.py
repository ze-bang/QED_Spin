"""High-level Python wrapper around ``ed::dssf`` (P2.8 / DSSF PR-G).

The C++ ``ED dssf`` subcommand calls
``ed::dssf::build_observable_pairs`` to assemble the list of observable
pairs ``(O_1, O_2, name)`` that get fed into the dynamical / static
structure-factor evaluator. This module re-exports the *same* builder so
Python notebooks and downstream scripts produce **byte-identical** observable
names and ordering, instead of hand-rolling
``Sum``/``Transverse``/``Sublattice``/``Experimental`` operator
constructors and risking drift.

Quick start
-----------

.. code-block:: python

    import quantum_ed as qed

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

The returned :class:`ObservablePairs` carries three parallel lists --
``obs_1``, ``obs_2``, ``names`` -- of equal length. The ``Operator`` objects
inside are the *same* C++ ``Operator`` / ``FixedSzOperator`` exposed via
:mod:`quantum_ed`, so they can be plugged directly into Lanczos / FTLM /
LTLM / Hybrid solvers without conversion.

See also
--------
:func:`compute_transverse_bases` -- helper that returns the ``(e1, e2)``
basis ``ed::dssf`` uses internally for transverse-component operators.
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
