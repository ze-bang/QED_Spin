"""``qed.eigs``: the lowest k levels of H over every symmetry sector, with vectors on demand."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from .. import _core
from . import _device
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
    device_blocks: int
    pruned_blocks: int
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

    def expect(self, ops: Sequence) -> np.ndarray:
        """<O> in each entry of ``levels``, averaged over the level's symmetry multiplet:
        a complex array [len(levels), len(ops)]. Needs ``vectors=True``."""
        single = not isinstance(ops, (list, tuple))
        ops = [ops] if single else list(ops)
        vals = np.asarray(self._raw.expect(self._spec, self._n_sites, ops), complex)
        return vals.reshape(len(self.levels), len(ops))

    def matrix_element(self, O, i: int, j: int) -> complex:
        """<v_i| O |v_j> between the vectors of ``levels[i]`` and ``levels[j]`` -- the
        partners the solver returned, from which each level's multiplet is expanded.
        ``O`` is arbitrary: it may change Sz and break every symmetry."""
        return complex(self._raw.matrix_element(self._n_sites, O, int(i), int(j)))

    def save(self, path) -> None:
        """Write the result to an ``.npz`` file: the levels with their block labels, the
        vectors in the sector basis they were solved in (each basis once), and the symmetry
        data. :func:`qed.load_eigs` restores it; :meth:`vectors`, :meth:`expect` and
        :meth:`matrix_element` then work without H."""
        arrays = {k: np.asarray(v) for k, v in
                  _core.sectors.eigs_to_arrays(self._raw, self._spec, self._n_sites).items()}
        np.savez_compressed(path, format_version=np.int64(1), k=np.int64(self.k),
                            energies=np.asarray(self.energies), **arrays)


def eigs(H, k: int = 1, *, sym: Optional[Symmetry] = None, vectors: bool = False,
         block_size: int = 1, dense_max_dim: int = 64, allow_partial: bool = False,
         device: str = "cpu", prune: bool = True, window: float = 0.0) -> EigResult:
    """The lowest ``k`` eigenvalues of ``H`` (with multiplicity), resolved by symmetry.

    ``sym`` defaults to :meth:`Symmetry.auto`. Raises when a block cannot certify levels
    that may fall inside the window, unless ``allow_partial``. ``prune`` solves only the blocks
    whose short Lanczos estimate lies near the window (``prune=False``: every block).
    ``window > 0`` also returns every block's lowest level within ``window`` above the k-th
    (the partners of a degenerate level in other blocks); ``energies`` then lists them all.
    """
    sym = Symmetry.auto() if sym is None else sym
    spec = sym.resolve(H)
    n = int(H.num_sites)
    raw = _core.sectors.eigs(H, n, spec, k=int(k), vectors=bool(vectors),
                             dense_max_dim=int(dense_max_dim), block_size=int(block_size),
                             allow_partial=bool(allow_partial), device=_device.resolve(device),
                             prune=bool(prune), window=float(window))
    rows = int(k) if window <= 0 else sum(int(L.multiplicity) for L in raw.levels)
    return EigResult(energies=np.asarray(raw.energies(rows), float), levels=list(raw.levels),
                     k=int(k), symmetry=sym, complete=bool(raw.complete),
                     device_blocks=int(raw.device_blocks), pruned_blocks=int(raw.pruned_blocks),
                     _raw=raw, _spec=spec,
                     _n_sites=n)


def load_eigs(path) -> "EigResult":
    """An :class:`EigResult` written by :meth:`EigResult.save` (``symmetry`` is None: the
    resolved group data travels in the file instead)."""
    with np.load(path) as f:
        d = {key: f[key] for key in f.files}
    if int(d.get("format_version", 0)) != 1:
        raise ValueError(f"{path}: not an EigResult file (format_version 1)")
    raw, spec, n = _core.sectors.eigs_from_arrays(d)
    return EigResult(energies=np.asarray(d["energies"], float), levels=list(raw.levels), k=int(d["k"]),
                     symmetry=None, complete=bool(raw.complete), device_blocks=int(raw.device_blocks),
                     pruned_blocks=int(raw.pruned_blocks), _raw=raw, _spec=spec, _n_sites=int(n))
