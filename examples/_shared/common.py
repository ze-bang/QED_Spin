"""examples/_shared/common.py

Tiny helpers shared by every cell in the
``examples/{solve,thermal,spectral}/`` tree. Mirrors
``examples/_shared/common.h`` line-for-line so the C++ and Python
twins read identically.

Helpers:

* :func:`heisenberg_chain` -- J=1 spin-1/2 Heisenberg :class:`qed.Operator`.
* :func:`bethe_E0` -- exact ground-state energy of the Heisenberg ring
  (PBC) for canonical small-N example sizes (NaN otherwise).
* :func:`rank0_print` -- print only on MPI rank 0 (works on serial
  builds too; no-op on non-rank-0).

Author: ed-collapse, Phase B of the "mirror examples" plan (May 2026).
"""

from __future__ import annotations

import math
import os
from typing import Any

import qed


def heisenberg_chain(N: int, pbc: bool = True, J: float = 1.0) -> "qed.Operator":
    """Build a J=1 spin-1/2 Heisenberg chain.

    H = J * sum_<ij> S_i . S_j

    Parameters
    ----------
    N : int
        Number of sites.
    pbc : bool
        Periodic boundary conditions (True) or open (False).
    J : float
        Exchange coupling. Defaults to 1.0.
    """
    op = qed.Operator(N, 0.5)
    last = N if pbc else N - 1
    for i in range(last):
        j = (i + 1) % N
        op.add_two_body(qed.OP_SZ,     i, qed.OP_SZ,     j,         J)
        op.add_two_body(qed.OP_SPLUS,  i, qed.OP_SMINUS, j, 0.5 * J)
        op.add_two_body(qed.OP_SMINUS, i, qed.OP_SPLUS,  j, 0.5 * J)
    return op


# Bethe-ansatz roots for the J=1 spin-1/2 Heisenberg ring (PBC).
# Cross-checked against ``examples/_shared/common.h::bethe_E0`` and the
# closed-form roots tabulated in Mattis (1981).
_BETHE_E0_TABLE = {
    2:  -0.75,
    4:  -2.0,
    6:  -2.8027756377319946,
    8:  -3.6510934089371783,
    10: -4.515446354492385,
    12: -5.387390917796387,
    14: -6.263728685245183,
    16: -7.142296361092491,
}


def bethe_E0(N: int) -> float:
    """Exact ground-state energy of the Heisenberg ring (PBC) for the
    canonical small-N sizes; ``math.nan`` for N outside the table."""
    return _BETHE_E0_TABLE.get(int(N), math.nan)


def _mpi_rank() -> int:
    """Best-effort MPI rank query. Returns 0 if MPI is not active."""
    # The C++ side carries an MPI initialiser; the Python side just probes
    # env vars exported by the launcher (OpenMPI / mpich set these).
    for env_var in ("OMPI_COMM_WORLD_RANK", "PMI_RANK", "MV2_COMM_WORLD_RANK"):
        v = os.environ.get(env_var)
        if v is not None:
            try:
                return int(v)
            except ValueError:
                pass
    return 0


def is_rank0() -> bool:
    """True on the rank that should produce stdout."""
    return _mpi_rank() == 0


def rank0_print(*args: Any, **kwargs: Any) -> None:
    """``print`` that only fires on MPI rank 0 (always on serial)."""
    if is_rank0():
        print(*args, **kwargs)
