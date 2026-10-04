"""``qed.measure`` and its one-request verbs: equal-time quantities of the lowest levels of H.

A measurement is (quantity) x (state) x (index axis):

* quantities -- :class:`Expect` (one-point <O>) and :class:`Correlations` (equal-time pairs
  <A_a^dag B_b>);
* states -- ``states="levels"`` (one row per returned level) or ``"ground"`` (one row, the ground
  manifold);
* index axes -- an operator, a sequence of operators, a :class:`qed.Family` (its index shape) or a
  :class:`qed.MomentumFamily` (a momentum axis on the family's last index).

Every request of one call is answered from one sweep of each level's basis: the products A_a^dag B_b
are formed exactly with the operator algebra, every operator is averaged over the level's symmetry
multiplet, and operators that average to the same one (a symmetry orbit of pairs) are evaluated once.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from typing import Optional

import numpy as np

from .. import _core, _log
from ..errors import InvalidRequest
from ..family import Family, MomentumFamily

__all__ = [
    "Expect",
    "Correlations",
    "ExpectResult",
    "CorrelationResult",
    "StructureFactor",
    "MeasureResult",
    "measure",
    "expect",
    "correlations",
]


# ---- requests ----------------------------------------------------------------------------------
@dataclass(frozen=True)
class Expect:
    """Request: <O> for ``ops`` -- an Operator, a sequence of Operators, a :class:`qed.Family` or a
    :class:`qed.MomentumFamily`."""

    ops: object


@dataclass(frozen=True)
class Correlations:
    """Request: <A_a^dag B_b> for every member a of ``A`` and b of ``B`` (``B=None``: B = A), each an
    Operator, a sequence, a :class:`qed.Family` or a :class:`qed.MomentumFamily`."""

    A: object
    B: object = None


# ---- operands: what the engine evaluates and the index axes of the answer ----------------------
@dataclass
class _Operand:
    ops: list
    shape: tuple                       # index shape of the evaluated operators
    family: Optional[Family] = None
    momentum: Optional[MomentumFamily] = None

    @property
    def index_shape(self) -> tuple:
        """The index shape of the answer (the momentum axis replaces the last one)."""
        return self.momentum.shape if self.momentum is not None else self.shape

    @property
    def source(self):
        return self.momentum if self.momentum is not None else self.family


def _operand(x, what: str) -> _Operand:
    if isinstance(x, MomentumFamily):
        # The pair values of the base family contract with the phases into the momentum axis:
        # cheaper and exact, against products of N-term Fourier operators.
        return _Operand(list(x.base.ops), x.base.shape, family=x.base, momentum=x)
    if isinstance(x, Family):
        return _Operand(list(x.ops), x.shape, family=x)
    if isinstance(x, _core.Operator):
        return _Operand([x], (1,))
    if isinstance(x, (list, tuple)):
        ops = list(x)
        for i, o in enumerate(ops):
            if o is None:
                raise InvalidRequest(f"{what}: observable {i} is None")
            if not isinstance(o, _core.Operator):
                raise InvalidRequest(f"{what}: observable {i} is a {type(o).__name__}, not a qed.Operator")
        return _Operand(ops, (len(ops),))
    raise InvalidRequest(
        f"{what}: expected a qed.Operator, a sequence of them, a qed.Family or a qed.MomentumFamily, "
        f"got {type(x).__name__}")


def _one_point(raw: np.ndarray, op: _Operand) -> np.ndarray:
    """[rows, n_ops] values -> [rows, *index_shape]."""
    v = raw.reshape(len(raw), *op.shape)
    if op.momentum is not None:
        v = v @ op.momentum.phases.T   # sum_r phi_qr <O_r>
    return v


def _pairs(raw: np.ndarray, a: _Operand, b: _Operand) -> np.ndarray:
    """[rows, n_a * n_b] values -> [rows, *a.index_shape, *b.index_shape]."""
    C = raw.reshape(len(raw), *a.shape, *b.shape)
    if b.momentum is not None:
        C = C @ b.momentum.phases.T                       # B_q = sum_s phi_qs B_s
    if a.momentum is not None:
        ax = len(a.shape)                                 # A's last axis (after the row axis)
        C = np.moveaxis(np.tensordot(C, np.conj(a.momentum.phases), axes=([ax], [1])), -1, ax)
    return C


def _ground(energies: np.ndarray, multiplicities: np.ndarray, degeneracy_tol: float):
    """(selected rows, weights): the rows within ``degeneracy_tol`` max(1, |E0|) of the lowest,
    weighted by multiplicity."""
    if len(energies) == 0:
        raise InvalidRequest("ground: there are no rows")
    if degeneracy_tol is None or float(degeneracy_tol) < 0:
        raise InvalidRequest(f"degeneracy_tol must be >= 0, got {degeneracy_tol}")
    e0 = float(np.min(energies))
    sel = np.flatnonzero(energies <= e0 + float(degeneracy_tol) * max(1.0, abs(e0)))
    w = multiplicities[sel].astype(float)
    return sel, w / w.sum()


def _reduce(x: np.ndarray, sel, w) -> np.ndarray:
    return np.tensordot(w, x[sel], axes=1)[None]


# ---- results -----------------------------------------------------------------------------------
@dataclass
class ExpectResult:
    """One row per level (``levels``, not repeated by multiplicity), or one row for the ground
    manifold (``rows == "ground"``).

    ``values[i, ...]`` is <O> in row i averaged over the level's symmetry multiplet, so
    ``multiplicities[i] * values[i, a]`` is the level's contribution to Tr(P_E O_a). The index axes
    after the row are those of the request: (len(ops),) for an operator or a sequence, the family's
    shape for a :class:`qed.Family`, its momentum shape for a :class:`qed.MomentumFamily`.
    ``diagnostics``: those of the underlying :func:`qed.eigs` run.
    """

    energies: np.ndarray
    multiplicities: np.ndarray
    values: np.ndarray
    levels: list
    eigs: object
    diagnostics: list = field(default_factory=list)
    rows: str = "levels"
    index: object = None

    def ground(self, degeneracy_tol: float = 1e-8) -> "ExpectResult":
        """One row: the levels within ``degeneracy_tol`` max(1, |E0|) of the lowest, averaged with
        their multiplicities."""
        sel, w = _ground(self.energies, self.multiplicities, degeneracy_tol)
        return replace(
            self,
            energies=np.array([self.energies[sel].min()]),
            multiplicities=np.array([int(self.multiplicities[sel].sum())]),
            values=_reduce(self.values, sel, w),
            levels=[self.levels[i] for i in sel],
            rows="ground",
        )


@dataclass
class StructureFactor:
    """S[row, *A_lead, *B_lead, q] = <A_q^dag B_q> with A_q = N^-1/2 sum_r e^{-i q.r} A_r (the
    package convention), i.e. N^-1 sum_ab e^{+i q.(r_a - r_b)} <A_a^dag B_b>; ``q`` is (n_q, 3)."""

    S: np.ndarray
    q: np.ndarray
    energies: np.ndarray
    multiplicities: np.ndarray
    rows: str = "levels"


@dataclass
class CorrelationResult:
    """<A_a^dag B_b> per row: ``C[row, *A_index, *B_index]`` (B_b acts first). A row is a level
    (``rows == "levels"``) or the ground manifold (``"ground"``), averaged over the symmetry
    multiplet as :class:`ExpectResult` averages A_a^dag B_b -- so it does not depend on which partner
    the solver returned. Pairs the level's sector cannot hold (an S^z change in a fixed-S^z sector)
    are exactly 0. ``mean_a`` and ``mean_b`` are the one-point values <A_a> and <B_b> per row, from
    the same sweep; :meth:`connected` subtracts them, :meth:`fourier` gives S(q)."""

    C: np.ndarray
    energies: np.ndarray
    multiplicities: np.ndarray
    mean_a: np.ndarray
    mean_b: np.ndarray
    levels: list
    eigs: object
    diagnostics: list = field(default_factory=list)
    rows: str = "levels"
    A: object = None
    B: object = None

    def ground(self, degeneracy_tol: float = 1e-8) -> "CorrelationResult":
        """One row: the levels within ``degeneracy_tol`` max(1, |E0|) of the lowest, averaged with
        their multiplicities (the ground manifold's average)."""
        sel, w = _ground(self.energies, self.multiplicities, degeneracy_tol)
        return replace(
            self,
            C=_reduce(self.C, sel, w),
            mean_a=_reduce(self.mean_a, sel, w),
            mean_b=_reduce(self.mean_b, sel, w),
            energies=np.array([self.energies[sel].min()]),
            multiplicities=np.array([int(self.multiplicities[sel].sum())]),
            levels=[self.levels[i] for i in sel],
            rows="ground",
        )

    def connected(self) -> np.ndarray:
        """<A_a^dag B_b> - <A_a>^* <B_b> per row (the one-point values averaged like the pairs)."""
        na, nb = self.mean_a.ndim - 1, self.mean_b.ndim - 1
        outer = np.conj(self.mean_a).reshape(*self.mean_a.shape, *([1] * nb)) * self.mean_b.reshape(
            len(self.mean_b), *([1] * na), *self.mean_b.shape[1:])
        return self.C - outer

    def fourier(self, q="cluster") -> StructureFactor:
        """The structure factor of two families with positions on their last axes:
        S[row, *A_lead, *B_lead, q] = sum_ab conj(phi_qa) phi_qb C[row, ..., a, ..., b], phi_qr =
        N^-1/2 e^{-i q.r}. ``q``: an (n_q, 3) array or ``"cluster"`` (from A's Lattice)."""
        A, B = self.A, self.B
        if not isinstance(A, Family) or not isinstance(B, Family):
            raise InvalidRequest("fourier: both operands must be qed.Family objects with positions "
                                 "(a MomentumFamily operand is already on its momentum axis)")
        mA = A.fourier(q)
        mB = MomentumFamily(B, mA.q)
        la, lb = len(A.shape) - 1, len(B.shape) - 1
        C = self.C
        # C[row, A_lead, a, B_lead, b] -> S[row, A_lead, B_lead, q]
        C = np.moveaxis(C, 1 + la, -1)                                  # a to the end
        S = np.einsum("...ba,qa,qb->...q", C, np.conj(mA.phases), mB.phases)
        return StructureFactor(S=S, q=mA.q, energies=self.energies, multiplicities=self.multiplicities,
                               rows=self.rows)


@dataclass
class MeasureResult(Sequence):
    """The answers of :func:`measure`, one per request in order (``result[i]``); ``energies``,
    ``multiplicities`` and ``rows`` describe the rows they share, ``eigs`` the eigensolve."""

    results: list
    energies: np.ndarray
    multiplicities: np.ndarray
    rows: str
    eigs: object
    diagnostics: list = field(default_factory=list)

    def __getitem__(self, i):
        return self.results[i]

    def __len__(self) -> int:
        return len(self.results)


# ---- the evaluation ----------------------------------------------------------------------------
def _evaluate(r, requests: Sequence) -> list:
    """Per-level answers (rows = r.levels) to every request, from one engine sweep."""
    if not any(L.vector >= 0 for L in r.levels):
        raise InvalidRequest("no vectors: call qed.eigs(..., vectors=True)")
    singles: list = []
    pairs: list = []
    plan = []
    for q in requests:
        if isinstance(q, Expect):
            op = _operand(q.ops, "Expect")
            plan.append(("expect", op, len(singles)))
            singles.extend(op.ops)
        elif isinstance(q, Correlations):
            a = _operand(q.A, "Correlations A")
            b = a if q.B is None else _operand(q.B, "Correlations B")
            plan.append(("pairs", (a, b), len(singles), len(pairs)))
            singles.extend(a.ops)
            if q.B is not None:
                singles.extend(b.ops)
            pairs.append((a.ops, b.ops))
        else:
            raise InvalidRequest(f"measure: unknown request {type(q).__name__} (Expect or Correlations)")
    n_levels = len(r.levels)
    raw = np.asarray(r._raw.evaluate(r._spec, singles, pairs), complex).reshape(n_levels, -1)
    starts = np.cumsum([len(singles)] + [len(A) * len(B) for A, B in pairs])
    energies = np.array([L.energy for L in r.levels], float)
    mult = np.array([L.multiplicity for L in r.levels], int)
    diags = list(r.diagnostics)
    out = []
    for entry in plan:
        if entry[0] == "expect":
            _, op, at = entry
            vals = _one_point(raw[:, at : at + len(op.ops)], op)
            out.append(ExpectResult(energies=energies, multiplicities=mult, values=vals, levels=list(r.levels),
                                    eigs=r, diagnostics=diags, index=op.source))
        else:
            _, (a, b), at, p = entry
            mean_a = _one_point(raw[:, at : at + len(a.ops)], a)
            mean_b = mean_a if b is a else _one_point(raw[:, at + len(a.ops) : at + len(a.ops) + len(b.ops)], b)
            C = _pairs(raw[:, starts[p] : starts[p + 1]], a, b)
            out.append(CorrelationResult(C=C, energies=energies, multiplicities=mult, mean_a=mean_a, mean_b=mean_b,
                                         levels=list(r.levels), eigs=r, diagnostics=diags, A=a.source, B=b.source))
    return out


def _states(states: str) -> str:
    if states not in ("levels", "ground"):
        raise InvalidRequest(f"states must be 'levels' or 'ground', got {states!r}")
    return states


def _solve(H, k: int, sym, device: str, eigs_kwargs: dict):
    """The EigResult to measure in: H itself when it is one (no solve), else qed.eigs with vectors."""
    from .eigs import EigResult, eigs

    if isinstance(H, EigResult):
        if eigs_kwargs or sym is not None:
            raise InvalidRequest("measure: an EigResult is measured as it is; eigs options do not apply")
        return H
    return eigs(H, k, sym=sym, vectors=True, device=device, **eigs_kwargs)


@_log.replays
def measure(H, requests: Sequence, k: int = 1, *, states: str = "levels", degeneracy_tol: float = 1e-8,
            sym=None, device: str = "cpu", **eigs_kwargs) -> MeasureResult:
    """Every request (:class:`Expect`, :class:`Correlations`) in the lowest levels of ``H`` from one
    eigensolve and one sweep of each level's basis.

    ``H``: the Hamiltonian (solved with :func:`qed.eigs` with vectors; ``k``, ``sym``, ``device`` and
    ``eigs_kwargs`` pass through) or an :class:`EigResult` with vectors to reuse.
    ``states="levels"``: one row per returned level; ``"ground"``: one row, the levels within
    ``degeneracy_tol`` max(1, |E0|) of the lowest averaged with their multiplicities (partners in
    other blocks are among them only if the solve returned them: raise ``k`` or pass ``window=``).
    """
    states = _states(states)
    requests = [requests] if isinstance(requests, (Expect, Correlations)) else list(requests)
    r = _solve(H, k, sym, device, eigs_kwargs)
    results = _evaluate(r, requests)
    energies = np.array([L.energy for L in r.levels], float)
    mult = np.array([L.multiplicity for L in r.levels], int)
    if states == "ground":
        results = [x.ground(degeneracy_tol) for x in results]
        sel, _ = _ground(energies, mult, degeneracy_tol)
        energies, mult = np.array([energies[sel].min()]), np.array([int(mult[sel].sum())])
    return MeasureResult(results=results, energies=energies, multiplicities=mult, rows=states, eigs=r,
                         diagnostics=list(r.diagnostics))


# ---- the one-request verbs ---------------------------------------------------------------------
def expect(H, ops, k: int = 1, *, states: str = "levels", degeneracy_tol: float = 1e-8, sym=None,
           device: str = "cpu", **eigs_kwargs) -> ExpectResult:
    """<O> for every operator in ``ops`` (an Operator, a sequence, a :class:`qed.Family` or a
    :class:`qed.MomentumFamily`) in each of the lowest levels of ``H``: :func:`measure` with one
    :class:`Expect` request.

    The value is averaged over the level's symmetry multiplet -- the quantity that does not depend
    on which partner the solver returned; for a non-degenerate level it is just <psi|O|psi>. With a
    total-spin restriction and an SU(2)-symmetric H the multiplet includes its 2S + 1 Sz members, so
    an operator that is not SU(2) invariant contributes its SU(2)-scalar part (its average over all
    spin rotations); in a uniform field each member is a level of its own.
    """
    return measure(H, [Expect(ops)], k, states=states, degeneracy_tol=degeneracy_tol, sym=sym, device=device,
                   **eigs_kwargs)[0]


def correlations(H, A, B=None, k: int = 1, *, states: str = "levels", degeneracy_tol: float = 1e-8, sym=None,
                 device: str = "cpu", **eigs_kwargs) -> CorrelationResult:
    """<A_a^dag B_b> for every pair (``B=None``: B = A) in each of the lowest levels of ``H``:
    :func:`measure` with one :class:`Correlations` request. ``A`` and ``B``: an Operator, a sequence, a
    :class:`qed.Family` (e.g. ``qed.Family.spins(lattice)``) or a :class:`qed.MomentumFamily`; the
    result's :meth:`CorrelationResult.fourier` gives the structure factor."""
    return measure(H, [Correlations(A, B)], k, states=states, degeneracy_tol=degeneracy_tol, sym=sym,
                   device=device, **eigs_kwargs)[0]
