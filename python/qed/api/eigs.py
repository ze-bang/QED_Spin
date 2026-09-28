"""``qed.eigs``: the lowest k levels of H over every symmetry sector, with vectors on demand."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .. import _core
from .symmetry import Symmetry


@dataclass
class EigResult:
    """Lowest levels of H. ``energies`` repeats each level by its multiplicity.

    ``levels`` holds one entry per distinct block eigenvalue (energy, multiplicity and
    the block's quantum numbers). :meth:`vectors` returns orthonormal eigenvectors of
    the lowest k energies, completing degenerate multiplets through the symmetry
    operations.
    """

    energies: np.ndarray
    levels: list
    k: int
    symmetry: Symmetry
    complete: bool
    _raw: object
    _spec: object
    _n_sites: int

    def vectors(self, basis: str = "full", n_up: Optional[int] = None) -> list:
        """Eigenvectors of the lowest k energies.

        ``basis="full"``: the 2^N computational basis (bit 0 of a state is site 0,
        a set bit is spin down). ``basis="sz"``: the Sz sector ``n_up`` (its states in
        ascending integer order); only the levels with a component there contribute.
        """
        if not any(l.vector >= 0 for l in self.levels):
            raise ValueError("no vectors: call qed.eigs(..., vectors=True)")
        if basis not in ("full", "sz"):
            raise ValueError("basis must be 'full' or 'sz'")
        if basis == "sz" and n_up is None:
            raise ValueError("basis='sz' needs n_up")
        want = -1 if basis == "full" else int(n_up)
        out = []
        for i, lvl in enumerate(self.levels):
            if len(out) >= self.k:
                break
            try:
                vs = self._raw.multiplet(self._spec, self._n_sites, i, want)
            except ValueError:
                continue                     # no component in this Sz sector
            out.extend(vs[: self.k - len(out)])
        return out


def eigs(H, k: int = 1, *, sym: Optional[Symmetry] = None, vectors: bool = False,
         block_size: int = 1, dense_max_dim: int = 64, allow_partial: bool = False) -> EigResult:
    """The lowest ``k`` eigenvalues of ``H`` (with multiplicity), resolved by symmetry.

    ``sym`` defaults to :meth:`Symmetry.auto`. Raises when a block cannot certify levels
    that may fall inside the window, unless ``allow_partial``.
    """
    sym = Symmetry.auto() if sym is None else sym
    spec = sym.resolve(H)
    n = int(H.num_sites)
    raw = _core.sectors.eigs(H, n, spec, k=int(k), vectors=bool(vectors),
                             dense_max_dim=int(dense_max_dim), block_size=int(block_size),
                             allow_partial=bool(allow_partial))
    return EigResult(energies=np.asarray(raw.energies(int(k)), float), levels=list(raw.levels),
                     k=int(k), symmetry=sym, complete=bool(raw.complete), _raw=raw, _spec=spec,
                     _n_sites=n)
