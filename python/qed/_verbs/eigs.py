"""``qed.eigs``: the lowest k levels of H over every symmetry sector, with vectors on demand."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .. import _core, _log
from ..errors import EmptySelection, InvalidRequest
from . import _device
from .symmetry import Labelled, Symmetry


@dataclass
class EigResult(Labelled):
    """Lowest levels of H. ``energies`` repeats each level by its multiplicity.

    ``levels`` holds one entry per distinct block eigenvalue (energy, multiplicity and
    the block's quantum numbers). :meth:`vectors` returns orthonormal eigenvectors of
    the lowest k energies, completing degenerate multiplets through the symmetry
    operations. :meth:`momentum` and :meth:`irrep_characters` name a level physically.
    ``diagnostics``: (code, message) pairs for fallbacks the run took (e.g. an incomplete
    window under ``allow_partial``).
    ``block_stats``: one dict per solved block -- dimension, the lane that applied H
    (dense, csr, csr-real, walk, gpu-gather, device-csr, device-gather), phase seconds (orbit
    tables, star build, CSR build, applies, the rest of the solve), nnz and the number of applies.
    ``placement``: how many solves ran as a Krylov or a dense solve on the device or the host
    (``device_krylov``, ``device_dense``, ``host_krylov``, ``host_dense``). Under
    ``device="gpu"`` no Krylov solve runs on the host: a block that cannot run on the device
    raises :class:`qed.errors.DeviceUnsupported` (or :class:`qed.errors.ResourceLimit` when it
    does not fit); small blocks may be solved densely there.
    ``time_reversal``: the antiunitary map that folded a returned level -- ``"K"`` (complex
    conjugation, a real H), ``"theta"`` (time reversal, an H that is not real) -- or None when no
    returned level was folded; each level's ``fold`` names its own.
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
    diagnostics: list = field(default_factory=list)
    block_stats: list = field(default_factory=list)
    placement: dict = field(default_factory=dict)
    time_reversal: Optional[str] = None

    @_log.replays
    def vectors(self, basis: str = "full", n_up: Optional[int] = None) -> list:
        """Eigenvectors of the lowest k energies.

        ``basis="full"``: the 2^N computational basis (bit 0 of a state is site 0,
        a set bit is spin up). ``basis="sz"``: the Sz sector ``n_up`` (its states in
        ascending integer order); only the levels with a component there contribute.
        """
        if not any(l.vector >= 0 for l in self.levels):
            raise InvalidRequest("no vectors: call qed.eigs(..., vectors=True)")
        if basis not in ("full", "sz"):
            raise InvalidRequest("basis must be 'full' or 'sz'")
        if basis == "sz" and n_up is None:
            raise InvalidRequest("basis='sz' needs n_up")
        want = -1 if basis == "full" else int(n_up)
        out = []
        for i, _lvl in enumerate(self.levels):
            if len(out) >= self.k:
                break
            try:
                vs = self._raw.multiplet(self._spec, i, want, self.k - len(out))
            except EmptySelection:
                continue  # no component in this Sz sector; anything else raises
            out.extend(vs[: self.k - len(out)])
        return out

    @_log.replays
    def expect(self, ops) -> np.ndarray:
        """<O> in each entry of ``levels``, averaged over the level's symmetry multiplet: a complex
        array [len(levels), *index], the index axes those of ``ops`` -- (len(ops),) for an Operator or
        a sequence, the shape of a :class:`qed.Family` or the momentum shape of a
        :class:`qed.MomentumFamily`. Needs ``vectors=True``."""
        from .measure import Expect, _evaluate

        return _evaluate(self, [Expect(ops)])[0].values

    @_log.replays
    def correlations(self, A, B=None):
        """<A_a^dag B_b> in each entry of ``levels`` for every pair (``B=None``: B = A), averaged as
        :meth:`expect` averages A_a^dag B_b: a :class:`qed.CorrelationResult`. All pairs come from one
        sweep of each level's basis, a symmetry orbit of pairs evaluated once."""
        from .measure import Correlations, _evaluate

        return _evaluate(self, [Correlations(A, B)])[0]

    def matrix_element(self, O, i: int, j: int) -> complex:
        """<v_i| O |v_j> between the vectors of ``levels[i]`` and ``levels[j]`` -- the
        partners the solver returned, from which each level's multiplet is expanded.
        ``O`` is arbitrary: it may change Sz and break every symmetry."""
        if not isinstance(O, _core.Operator):
            raise InvalidRequest(f"matrix_element: O must be a qed.Operator, got {type(O).__name__}")
        return complex(self._raw.matrix_element(O, int(i), int(j)))

    def save(self, path) -> None:
        """Write the result to an ``.npz`` file: the levels with their block labels, the
        vectors in the sector basis they were solved in (each basis once), and the symmetry
        data. :func:`qed.load_eigs` restores it; :meth:`vectors`, :meth:`expect` and
        :meth:`matrix_element` then work without H."""
        arrays = {k: np.asarray(v) for k, v in _core.sectors.eigs_to_arrays(self._raw, self._spec).items()}
        np.savez_compressed(
            path, format_version=np.int64(2), k=np.int64(self.k), energies=np.asarray(self.energies), **arrays
        )


@_log.replays
def eigs(
    H,
    k: int = 1,
    *,
    sym: Optional[Symmetry] = None,
    vectors: bool = False,
    dense_max_dim: Optional[int] = None,
    allow_partial: bool = False,
    device: str = "cpu",
    prune: bool = True,
    window: float = 0.0,
    per_block: Optional[int] = None,
) -> EigResult:
    """The lowest ``k`` eigenvalues of ``H`` (with multiplicity), resolved by symmetry.

    ``sym`` defaults to :meth:`Symmetry.auto`. Raises :class:`qed.errors.ConvergenceError` when a
    block cannot certify levels that may fall inside the window, unless ``allow_partial`` (then
    ``complete`` is False). ``prune`` solves only the blocks
    whose short Lanczos estimate lies near the window (``prune=False``: every block).
    ``window > 0`` also returns every block's lowest level within ``window`` above the k-th
    (the partners of a degenerate level in other blocks); ``energies`` then lists them all.
    ``dense_max_dim``: blocks up to this dimension are diagonalised densely (exact, and they
    resolve every copy of a degenerate level at once), larger ones by Krylov (the Lanczos scan,
    or the certified ground-state vector with ``vectors``, for one owed level; thick-restart
    Krylov-Schur for several); ``None``
    picks it from ``k`` (1600 for ``k <= 10``), 0 sends every block above dimension 2 to Krylov.
    ``per_block=m``: the lowest ``m`` levels of EVERY symmetry block instead of the lowest ``k``
    overall (``k`` and ``window`` are then not used; nothing is pruned) -- the excited states of each
    sector, e.g. for transitions between them.
    """
    if per_block is not None and int(per_block) < 1:
        raise InvalidRequest(f"per_block must be >= 1 or None, got {per_block}")
    if per_block is not None and window > 0:
        raise InvalidRequest("per_block and window exclude each other: per_block returns every block's levels")
    if dense_max_dim is not None and int(dense_max_dim) < 0:
        raise InvalidRequest(f"dense_max_dim must be >= 0 or None, got {dense_max_dim}")
    sym = Symmetry.auto() if sym is None else sym
    diagnostics: list = []
    spec = sym.resolve(H, diagnostics)
    raw = _core.sectors.eigs(
        H,
        spec,
        k=int(k),
        vectors=bool(vectors),
        dense_max_dim=-1 if dense_max_dim is None else int(dense_max_dim),
        allow_partial=bool(allow_partial),
        device=_device.resolve(device),
        prune=bool(prune),
        window=float(window),
        per_block=0 if per_block is None else int(per_block),
    )
    every = window > 0 or per_block is not None
    rows = sum(int(L.multiplicity) for L in raw.levels) if every else int(k)
    return EigResult(
        energies=np.asarray(raw.energies(rows), float),
        levels=list(raw.levels),
        k=rows if per_block is not None else int(k),
        symmetry=sym,
        complete=bool(raw.complete),
        device_blocks=int(raw.device_blocks),
        pruned_blocks=int(raw.pruned_blocks),
        _raw=raw,
        _spec=spec,
        _n_sites=int(raw.n_sites),
        diagnostics=diagnostics + [tuple(x) for x in raw.diagnostics],
        block_stats=list(raw.block_stats),
        placement=dict(raw.placement),
        time_reversal=raw.time_reversal,
    )


def load_eigs(path) -> "EigResult":
    """An :class:`EigResult` written by :meth:`EigResult.save` (``symmetry`` is None: the
    resolved group data travels in the file instead)."""
    with np.load(path) as f:
        d = {key: f[key] for key in f.files}
    version = int(d.get("format_version", 0))
    if version == 1:
        raise InvalidRequest(
            f"{path}: an EigResult file of format 1, written when a set bit meant spin down "
            "(qed < 0.6); its states and n_up labels mean the opposite now -- recompute it"
        )
    if version != 2:
        raise InvalidRequest(f"{path}: not an EigResult file (format_version 2)")
    raw, spec = _core.sectors.eigs_from_arrays(d)
    return EigResult(
        energies=np.asarray(d["energies"], float),
        levels=list(raw.levels),
        k=int(d["k"]),
        symmetry=None,
        complete=bool(raw.complete),
        device_blocks=int(raw.device_blocks),
        pruned_blocks=int(raw.pruned_blocks),
        _raw=raw,
        _spec=spec,
        _n_sites=int(raw.n_sites),
        time_reversal=raw.time_reversal,
    )
