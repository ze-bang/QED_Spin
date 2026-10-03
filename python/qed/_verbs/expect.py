"""``qed.expect``: expectation values of operators in the lowest levels of H."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .. import _log
from .eigs import EigResult, eigs
from .symmetry import Symmetry


@dataclass
class ExpectResult:
    """One row per level (``levels``, not repeated by multiplicity).

    ``values[i, a]`` is <O_a> in level i averaged over the level's symmetry multiplet, so
    ``multiplicities[i] * values[i, a]`` is the level's contribution to Tr(P_E O_a).
    ``diagnostics``: those of the underlying :func:`qed.eigs` run.
    """

    energies: np.ndarray
    multiplicities: np.ndarray
    values: np.ndarray
    levels: list
    eigs: EigResult
    diagnostics: list = field(default_factory=list)


@_log.replays
def expect(H, ops, k: int = 1, *, sym: Optional[Symmetry] = None, device: str = "cpu", **eigs_kwargs) -> ExpectResult:
    """<O> for every operator in ``ops`` in each of the lowest levels of ``H``.

    Runs :func:`qed.eigs` with vectors (``eigs_kwargs`` pass through) and evaluates each
    operator in each level's own symmetry block. The value is averaged over the level's
    symmetry multiplet -- the quantity that does not depend on which partner the solver
    returned; for a non-degenerate level it is just <psi|O|psi>. Correlators and structure
    factors are operators like any other. With a total-spin restriction and an SU(2)-symmetric
    H the multiplet includes its 2S + 1 Sz members, so an operator that is not SU(2) invariant
    contributes its SU(2)-scalar part (its average over all spin rotations); in a uniform field
    each member is a level of its own.
    """
    single = not isinstance(ops, (list, tuple))
    ops = [ops] if single else list(ops)
    r = eigs(H, k, sym=sym, vectors=True, device=device, **eigs_kwargs)
    return ExpectResult(
        energies=np.array([L.energy for L in r.levels], float),
        multiplicities=np.array([L.multiplicity for L in r.levels], int),
        values=r.expect(ops),
        levels=r.levels,
        eigs=r,
        diagnostics=list(r.diagnostics),
    )
