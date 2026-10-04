"""Operator families: the index axis of a measurement.

A :class:`Family` is a set of operators O_alpha with an index shape -- (N,) for S_i^z, (3, N) for the
three spin components on N sites, (n_bonds,) for bond energies -- and, for the last (site) axis,
positions. Its momentum transform

    O_q = N^-1/2 sum_r e^{-i q.r} O_r          (the package convention, see qed._geometry)

turns pair correlations <O_a^dag O_b> into the structure factor
S(q) = <O_q^dag O_q> = N^-1 sum_ab e^{+i q.(r_a - r_b)} <O_a^dag O_b>, and dynamics of O_q into S(q, w).
Every member is a qed.Operator built with the operator algebra, so a family can hold any operator:
spins, bonds S_i.S_j, chiralities, dimers, plaquette terms.
"""

from __future__ import annotations

from typing import Callable, Optional, Sequence

import numpy as np

from . import _core
from . import _geometry
from .errors import InvalidRequest

__all__ = ["Family", "MomentumFamily"]

_SPIN_LETTERS = {"x": "x", "y": "y", "z": "z", "+": "+", "-": "-"}


def _positions_of(where, n: Optional[int] = None):
    """(positions array or None, lattice or None) from a Lattice, an (n, 3) array or None."""
    if where is None:
        return None, None
    if hasattr(where, "positions") and hasattr(where, "num_sites"):
        return np.asarray(where.positions, float).reshape(-1, 3), where
    r = np.asarray(where, float)
    if r.ndim != 2 or r.shape[1] not in (2, 3):
        raise InvalidRequest(f"positions must be an (n, 3) array or a qed.input.Lattice, got shape {r.shape}")
    if r.shape[1] == 2:
        r = np.hstack([r, np.zeros((len(r), 1))])
    if n is not None and len(r) != n:
        raise InvalidRequest(f"{len(r)} positions for {n} sites")
    return r, None


class Family:
    """Operators ``ops`` (all on the same sites) with index ``shape`` (default ``(len(ops),)``,
    ``ops`` in C order over it) and, for the last axis, ``positions`` (an (shape[-1], 3) array or a
    ``qed.input.Lattice``). ``labels``: optional names, one per member; ``components``: the spin
    components of the first axis (set by :meth:`spins`, read by :meth:`StructureFactor.perp`)."""

    def __init__(
        self,
        ops: Sequence,
        *,
        shape: Optional[Sequence[int]] = None,
        positions=None,
        labels=None,
        components: Optional[str] = None,
    ):
        ops = list(ops)
        if not ops:
            raise InvalidRequest("Family: no operators")
        if not all(isinstance(o, _core.Operator) for o in ops):
            raise InvalidRequest("Family: every member must be a qed.Operator")
        n_sites = {int(o.num_sites) for o in ops}
        if len(n_sites) != 1:
            raise InvalidRequest(f"Family: the members act on different numbers of sites {sorted(n_sites)}")
        self.ops = ops
        self.num_sites = n_sites.pop()
        self.shape = tuple(int(s) for s in (shape if shape is not None else (len(ops),)))
        if int(np.prod(self.shape)) != len(ops):
            raise InvalidRequest(f"Family: shape {self.shape} holds {int(np.prod(self.shape))} members, got {len(ops)}")
        self.positions, self.lattice = _positions_of(positions, self.shape[-1])
        self.labels = list(labels) if labels is not None else None
        if self.labels is not None and len(self.labels) != len(ops):
            raise InvalidRequest(f"Family: {len(self.labels)} labels for {len(ops)} members")
        # the spin components of the first axis (Family.spins), e.g. "xyz"; None otherwise
        self.components = components

    def __len__(self) -> int:
        return len(self.ops)

    def __repr__(self) -> str:
        where = "" if self.positions is None else ", with positions"
        return f"<qed.Family of {len(self.ops)} operators on {self.num_sites} sites, shape {self.shape}{where}>"

    # ---- constructors ---------------------------------------------------------------------------
    @classmethod
    def spins(cls, sites, components: str = "xyz") -> "Family":
        """S_i^a on every site, shape (len(components), N): ``components`` letters from x, y, z, +, -.
        ``sites``: a qed.input.Lattice (its positions come along) or a number of sites."""
        r, lat = _positions_of(sites) if not isinstance(sites, (int, np.integer)) else (None, None)
        n = int(sites) if isinstance(sites, (int, np.integer)) else len(r)
        if not components or any(c not in _SPIN_LETTERS for c in components):
            raise InvalidRequest(f"Family.spins: components are letters of 'xyz+-', got {components!r}")
        ops = [_core.Operator.product(n, _SPIN_LETTERS[c], [i], 1.0) for c in components for i in range(n)]
        labels = [f"S{c}_{i}" for c in components for i in range(n)]
        return cls(
            ops,
            shape=(len(components), n),
            positions=lat if lat is not None else r,
            labels=labels,
            components=components,
        )

    @classmethod
    def sites(cls, sites, f: Callable[[int], object]) -> "Family":
        """f(i) for every site i (a qed.Operator each); ``sites``: a Lattice or a number of sites."""
        r, lat = _positions_of(sites) if not isinstance(sites, (int, np.integer)) else (None, None)
        n = int(sites) if isinstance(sites, (int, np.integer)) else len(r)
        return cls([f(i) for i in range(n)], positions=lat if lat is not None else r)

    @classmethod
    def bonds(cls, pairs, f: Callable[[int, int], object], positions=None) -> "Family":
        """f(i, j) for every pair (a qed.Operator each), e.g. ``lambda i, j: S(i) @ S(j)``. With
        site ``positions`` (an array or a Lattice) each bond sits at its midpoint (the minimum
        image is the caller's: pass the pairs a periodic generator returns, oriented as generated)."""
        pairs = [(int(i), int(j)) for i, j in pairs]
        ops = [f(i, j) for i, j in pairs]
        mids = None
        if positions is not None:
            r, _ = _positions_of(positions)
            mids = np.array([(r[i] + r[j]) / 2.0 for i, j in pairs])
        return cls(ops, positions=mids, labels=[f"b_{i}_{j}" for i, j in pairs])

    # ---- the momentum axis ----------------------------------------------------------------------
    def fourier(self, q) -> "MomentumFamily":
        """The momentum transform of the last axis: O_q = N^-1/2 sum_r e^{-i q.r} O_r for every q in
        ``q`` (an (n_q, 3) array in the units of the positions) or ``"cluster"`` (the momenta the
        periodic cluster allows, from the family's Lattice: qed.input.cluster_momenta)."""
        if self.positions is None:
            raise InvalidRequest("Family.fourier: the family has no positions")
        if isinstance(q, str):
            if q != "cluster":
                raise InvalidRequest(f"Family.fourier: q is an array or 'cluster', got {q!r}")
            if self.lattice is None:
                raise InvalidRequest(
                    "Family.fourier('cluster') needs a family built from a qed.input.Lattice; "
                    "pass q explicitly (qed.input.cluster_momenta)"
                )
            q = _geometry.cluster_momenta(self.lattice)
        return MomentumFamily(self, q)


class MomentumFamily:
    """A family's momentum transform: ``q`` (n_q, 3) and the weights ``phases[q, r]`` =
    N^-1/2 e^{-i q.r} over its last axis. Pair correlations of the base family contract with these
    weights into S(q); :meth:`operators` builds the O_q themselves (for dynamics probes)."""

    def __init__(self, base: Family, q):
        self.base = base
        Q = np.asarray(q, float)
        Q = Q.reshape(1, -1) if Q.ndim == 1 else Q
        if Q.shape[1] == 2:
            Q = np.hstack([Q, np.zeros((len(Q), 1))])
        if Q.ndim != 2 or Q.shape[1] != 3:
            raise InvalidRequest(f"q must be an (n_q, 3) array, got shape {np.asarray(q).shape}")
        self.q = Q
        n = base.shape[-1]
        self.phases = np.exp(-1j * (Q @ base.positions.T)) / np.sqrt(n)

    @property
    def shape(self) -> tuple:
        """(*base.shape[:-1], n_q)."""
        return (*self.base.shape[:-1], len(self.q))

    def operators(self) -> list:
        """The O_q as qed.Operators, in C order over :attr:`shape` (built with the algebra)."""
        n = self.base.shape[-1]
        lead = int(np.prod(self.base.shape[:-1])) if len(self.base.shape) > 1 else 1
        out = []
        for a in range(lead):
            members = self.base.ops[a * n : (a + 1) * n]
            for w in self.phases:
                O = members[0] * complex(w[0])
                for r in range(1, n):
                    O = O + members[r] * complex(w[r])
                out.append(O)
        return out
