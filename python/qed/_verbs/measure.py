"""``qed.measure`` and its one-request verbs: quantities of the lowest levels of H or at temperatures.

A measurement is (quantity) x (state) x (index axis):

* quantities -- :class:`Expect` (one-point <O>), :class:`Correlations` (equal-time pairs
  <A_a^dag B_b>), :class:`Transitions` (<m|O|n> between levels, as multiplet-invariant line
  strengths and pair matrices) and :class:`Dynamics` (S_AB(omega), :func:`qed.dynamics`);
* states -- ``states="levels"`` (one row per returned level), ``"ground"`` (one row, the ground
  manifold) or temperatures (``T=[...]``, one row each);
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
    "Transitions",
    "Dynamics",
    "ExpectResult",
    "CorrelationResult",
    "TransitionResult",
    "StructureFactor",
    "MeasureResult",
    "measure",
    "expect",
    "correlations",
    "transitions",
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


@dataclass(frozen=True)
class Dynamics:
    """Request: S_AB(omega) (:func:`qed.dynamics`) of ``A`` (an Operator, a sequence, a :class:`qed.Family`
    or a :class:`qed.MomentumFamily`) and ``B`` (None: autocorrelations; ``"all"``; an Operator, a sequence
    or family as long as A), at the measurement's temperatures (``T=None``: the ground state). ``eta``,
    ``krylov``, ``samples``, ``seed``: as in :func:`qed.dynamics`. At T > 0 under FTLM the dynamics,
    the thermodynamics and every equal-time request of the measurement share one pass (each sample's
    source Lanczos run)."""

    A: object
    omega: object
    B: object = None
    eta: float = 0.05
    krylov: int = 200
    samples: int = 40
    seed: int = 0


@dataclass(frozen=True)
class Transitions:
    """Request: transitions from every measured level to every measured level -- the line strengths
    of ``A`` and, with ``B`` or ``pairs=True`` (B = A), the pair matrices T_ab. ``raw=True`` keeps the
    member-resolved amplitudes (:meth:`TransitionResult.amplitudes`)."""

    A: object
    B: object = None
    pairs: bool = False
    raw: bool = False


# ---- operands: what the engine evaluates and the index axes of the answer ----------------------
@dataclass
class _Operand:
    ops: list
    shape: tuple  # index shape of the evaluated operators
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
        f"got {type(x).__name__}"
    )


def _one_point(raw: np.ndarray, op: _Operand) -> np.ndarray:
    """[rows, n_ops] values -> [rows, *index_shape]."""
    v = raw.reshape(len(raw), *op.shape)
    if op.momentum is not None:
        v = v @ op.momentum.phases.T  # sum_r phi_qr <O_r>
    return v


def _pairs(raw: np.ndarray, a: _Operand, b: _Operand) -> np.ndarray:
    """[rows, n_a * n_b] values -> [rows, *a.index_shape, *b.index_shape]."""
    C = raw.reshape(len(raw), *a.shape, *b.shape)
    if b.momentum is not None:
        C = C @ b.momentum.phases.T  # B_q = sum_s phi_qs B_s
    if a.momentum is not None:
        ax = len(a.shape)  # A's last axis (after the row axis)
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


def _level_rows(rows: str) -> None:
    if rows != "levels":
        raise InvalidRequest(f"ground() combines levels; these rows are {rows!r}")


# ---- results -----------------------------------------------------------------------------------
@dataclass
class ExpectResult:
    """One row per level (``levels``, not repeated by multiplicity), one row for the ground manifold
    (``rows == "ground"``), or one row per temperature (``rows == "T"``: <O>(T), ``T`` the temperatures,
    ``energies`` the thermal energy E(T)).

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
    T: Optional[np.ndarray] = None

    def ground(self, degeneracy_tol: float = 1e-8) -> "ExpectResult":
        """One row: the levels within ``degeneracy_tol`` max(1, |E0|) of the lowest, averaged with
        their multiplicities."""
        _level_rows(self.rows)
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
    components: Optional[str] = None  # the spin components of both lead axes (Family.spins)
    T: Optional[np.ndarray] = None

    def trace(self) -> np.ndarray:
        """sum_a S^aa(q) over the shared spin components: [row, q]."""
        if not self.components:
            raise InvalidRequest("trace: both operands must be Family.spins families with the same components")
        return np.einsum("raaq->rq", self.S)

    def perp(self) -> np.ndarray:
        """The neutron cross section's spin part sum_ab (delta_ab - q_a q_b / q^2) S^ab(q): [row, q], real
        up to roundoff. Needs Family.spins operands with components "xyz"; at q = 0, where the
        projector is undefined, the orientational average (2/3) sum_a S^aa."""
        if self.components != "xyz":
            raise InvalidRequest(f"perp: needs Family.spins(..., 'xyz') operands, got components {self.components!r}")
        norm = np.linalg.norm(self.q, axis=1)
        qhat = np.divide(self.q, norm[:, None], out=np.zeros_like(self.q), where=norm[:, None] > 1e-12)
        proj = np.eye(3)[:, :, None] - np.einsum("qa,qb->abq", qhat, qhat)
        out = np.einsum("abq,rabq->rq", proj, self.S)
        zero = norm <= 1e-12
        out[:, zero] = (2.0 / 3.0) * np.einsum("raaq->rq", self.S)[:, zero]
        return out


@dataclass
class CorrelationResult:
    """<A_a^dag B_b> per row: ``C[row, *A_index, *B_index]`` (B_b acts first). A row is a level
    (``rows == "levels"``) or the ground manifold (``"ground"``), or a temperature (``"T"``: thermal
    averages, ``T`` the temperatures), averaged over the symmetry
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
    T: Optional[np.ndarray] = None

    def ground(self, degeneracy_tol: float = 1e-8) -> "CorrelationResult":
        """One row: the levels within ``degeneracy_tol`` max(1, |E0|) of the lowest, averaged with
        their multiplicities (the ground manifold's average)."""
        _level_rows(self.rows)
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
            len(self.mean_b), *([1] * na), *self.mean_b.shape[1:]
        )
        return self.C - outer

    def fourier(self, q="cluster") -> StructureFactor:
        """The structure factor of two families with positions on their last axes:
        S[row, *A_lead, *B_lead, q] = sum_ab conj(phi_qa) phi_qb C[row, ..., a, ..., b], phi_qr =
        N^-1/2 e^{-i q.r}. ``q``: an (n_q, 3) array or ``"cluster"`` (from A's Lattice)."""
        A, B = self.A, self.B
        if not isinstance(A, Family) or not isinstance(B, Family):
            raise InvalidRequest(
                "fourier: both operands must be qed.Family objects with positions "
                "(a MomentumFamily operand is already on its momentum axis)"
            )
        mA = A.fourier(q)
        mB = MomentumFamily(B, mA.q)
        la = len(A.shape) - 1
        C = self.C
        # C[row, A_lead, a, B_lead, b] -> S[row, A_lead, B_lead, q]
        C = np.moveaxis(C, 1 + la, -1)  # a to the end
        S = np.einsum("...ba,qa,qb->...q", C, np.conj(mA.phases), mB.phases)
        comps = A.components if A.components == B.components else None
        return StructureFactor(
            S=S,
            q=mA.q,
            energies=self.energies,
            multiplicities=self.multiplicities,
            rows=self.rows,
            components=comps,
            T=self.T,
        )


@dataclass
class TransitionResult:
    """Transitions from initial level i to final level j (``initial_levels``, ``final_levels``), with
    ``omega[i, j] = E_j - E_i``. Sums over the members of the multiplets, which do not depend on the
    partners the solver returned:

    * ``strength[i, j, *A_index]`` = (1/d_i) sum_{n' in i, m' in j} |<m'|A_a|n'>|^2, averaged over the
      initial multiplet and summed over the final: the pole weight of A at omega in the T = 0
      dynamics from level i;
    * ``T[i, j, *A_index, *B_index]`` = (1/d_i) sum conj(<m'|A_a|n'>) <m'|B_b|n'> (with pairs): summed
      over a complete set of final levels it is <A_a^dag B_b> in level i.

    Transitions a selection rule forbids (momentum: O_q takes k to k - q; S^z) are exact zeros.
    :meth:`amplitudes` returns the member-resolved <m'|A_a|n'> when the request kept them (``raw``);
    those depend on the members chosen (the solver's vector first, then its symmetry images)."""

    omega: np.ndarray
    strength: np.ndarray
    T: Optional[np.ndarray]
    initial_energies: np.ndarray
    final_energies: np.ndarray
    initial_multiplicities: np.ndarray
    final_multiplicities: np.ndarray
    initial_levels: list
    final_levels: list
    A: object = None
    B: object = None
    rows: str = "levels"
    _amplitudes: Optional[np.ndarray] = field(default=None, repr=False)
    _offsets: tuple = field(default=(), repr=False)
    _a_shape: tuple = field(default=(), repr=False)

    def amplitudes(self, i: int, j: int) -> np.ndarray:
        """<m'|A_a|n'> for the members n' of initial level i and m' of final level j: a complex array
        [*A_index, d_j, d_i] (requires ``raw=True``)."""
        if self._amplitudes is None:
            raise InvalidRequest("amplitudes: request them with raw=True (and not after ground())")
        oi, of = self._offsets
        blk = self._amplitudes[:, of[j] : of[j + 1], oi[i] : oi[i + 1]]
        return blk.reshape(*self._a_shape, *blk.shape[1:])

    def ground(self, degeneracy_tol: float = 1e-8) -> "TransitionResult":
        """One initial row: the initial levels within ``degeneracy_tol`` max(1, |E0|) of the lowest,
        averaged with their multiplicities (transitions out of the ground manifold)."""
        sel, w = _ground(self.initial_energies, self.initial_multiplicities, degeneracy_tol)
        e0 = float(self.initial_energies[sel].min())
        return replace(
            self,
            omega=(self.final_energies - e0)[None],
            strength=_reduce(self.strength, sel, w),
            T=None if self.T is None else _reduce(self.T, sel, w),
            initial_energies=np.array([e0]),
            initial_multiplicities=np.array([int(self.initial_multiplicities[sel].sum())]),
            initial_levels=[self.initial_levels[i] for i in sel],
            rows="ground",
            _amplitudes=None,
        )


@dataclass
class MeasureResult(Sequence):
    """The answers of :func:`measure`, one per request in order (``result[i]``); ``energies``,
    ``multiplicities`` and ``rows`` describe the rows they share (levels, the ground manifold, or the
    temperatures ``T``), ``eigs`` the eigensolve, ``thermal`` the thermodynamics of a run with T."""

    results: list
    energies: np.ndarray
    multiplicities: np.ndarray
    rows: str
    eigs: object
    diagnostics: list = field(default_factory=list)
    T: Optional[np.ndarray] = None
    thermal: object = None

    def __getitem__(self, i):
        return self.results[i]

    def __len__(self) -> int:
        return len(self.results)


# ---- the evaluation ----------------------------------------------------------------------------
def _level_set(x, what: str):
    """(EigResult, level indices) from an EigResult (every level) or (EigResult, indices)."""
    from .eigs import EigResult

    if isinstance(x, EigResult):
        return x, list(range(len(x.levels)))
    if isinstance(x, tuple) and len(x) == 2 and isinstance(x[0], EigResult):
        r, idx = x
        idx = [int(i) for i in (idx if isinstance(idx, (list, tuple, range, np.ndarray)) else [idx])]
        for i in idx:
            if not 0 <= i < len(r.levels):
                raise IndexError(f"{what}: level {i} of a result with {len(r.levels)} levels")
        return r, idx
    raise InvalidRequest(f"{what}: an EigResult or (EigResult, level indices), got {type(x).__name__}")


def _transition_ops(x, what: str):
    """(operators, index shape, source). A momentum family enters as its O_q, so the selection rules
    of the lambda projection give exact zeros."""
    if isinstance(x, MomentumFamily):
        return x.operators(), x.shape, x
    op = _operand(x, what)
    return op.ops, op.shape, op.source


def _transitions(ri, ii: list, rf, jj: list, q: Transitions) -> TransitionResult:
    for r in (ri, rf):
        if not any(L.vector >= 0 for L in r.levels):
            raise InvalidRequest("no vectors: call qed.eigs(..., vectors=True)")
    a_ops, a_shape, a_src = _transition_ops(q.A, "Transitions A")
    with_pairs = bool(q.pairs) or q.B is not None
    b_ops, b_shape, b_src = (a_ops, a_shape, a_src) if q.B is None else _transition_ops(q.B, "Transitions B")
    ops = list(a_ops) + ([] if q.B is None else list(b_ops))
    d = _core.sectors.transition_amplitudes(ri._raw, ri._spec, ii, rf._raw, rf._spec, jj, ops)
    amps = np.asarray(d["amplitudes"], complex)
    oi = np.asarray(d["initial_offsets"], int)
    of = np.asarray(d["final_offsets"], int)
    na = len(a_ops)
    Aa = amps[:na]
    Bb = Aa if q.B is None else amps[na:]
    di = np.diff(oi).astype(float)  # members of each initial level: its multiplicity
    s = np.zeros((na, len(jj), len(ii)))
    if na:
        s = np.add.reduceat(np.add.reduceat(np.abs(Aa) ** 2, of[:-1], axis=1), oi[:-1], axis=2)
    strength = (s / di).transpose(2, 1, 0).reshape(len(ii), len(jj), *a_shape)
    T = None
    if with_pairs:
        T = np.empty((len(ii), len(jj), na, len(b_ops)), complex)
        for i in range(len(ii)):
            for j in range(len(jj)):
                bA = Aa[:, of[j] : of[j + 1], oi[i] : oi[i + 1]]
                bB = Bb[:, of[j] : of[j + 1], oi[i] : oi[i + 1]]
                T[i, j] = np.einsum("amn,bmn->ab", bA.conj(), bB) / di[i]
        T = T.reshape(len(ii), len(jj), *a_shape, *b_shape)
    Ei = np.array([ri.levels[i].energy for i in ii], float)
    Ef = np.array([rf.levels[j].energy for j in jj], float)
    return TransitionResult(
        omega=Ef[None, :] - Ei[:, None],
        strength=strength,
        T=T,
        initial_energies=Ei,
        final_energies=Ef,
        initial_multiplicities=np.array([ri.levels[i].multiplicity for i in ii], int),
        final_multiplicities=np.array([rf.levels[j].multiplicity for j in jj], int),
        initial_levels=[ri.levels[i] for i in ii],
        final_levels=[rf.levels[j] for j in jj],
        A=a_src,
        B=b_src if with_pairs else None,
        _amplitudes=Aa if q.raw else None,
        _offsets=(oi, of),
        _a_shape=tuple(a_shape),
    )


def _plan(requests: Sequence, thermal: bool = False):
    """(singles, pairs, plan): the operators every request needs, in one list for one pass."""
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
        elif isinstance(q, Transitions):
            if thermal:
                raise InvalidRequest("Transitions are between levels: measure them without T=")
            plan.append(("transitions", q))
        elif isinstance(q, Dynamics):
            plan.append(("dynamics", q))
        else:
            raise InvalidRequest(
                f"measure: unknown request {type(q).__name__} (Expect, Correlations, Transitions or Dynamics)"
            )
    return singles, pairs, plan


def _answers(plan: list, raw: np.ndarray, singles: list, pairs: list, rows: dict) -> list:
    """The answer to every request from raw[row, x] (singles, then each pair request a-major).
    ``rows``: energies, multiplicities, levels, eigs, diagnostics, rows ("levels" | "T"), T."""
    starts = np.cumsum([len(singles)] + [len(A) * len(B) for A, B in pairs])
    common = dict(
        energies=rows["energies"],
        multiplicities=rows["multiplicities"],
        levels=list(rows["levels"]),
        eigs=rows["eigs"],
        diagnostics=rows["diagnostics"],
        rows=rows["rows"],
        T=rows["T"],
    )
    out = []
    for entry in plan:
        if entry[0] == "dynamics":
            run = rows.get("dynamics")
            if run is None:
                raise InvalidRequest("Dynamics needs the Hamiltonian: measure H, not an EigResult")
            out.append(run(entry[1]))
        elif entry[0] == "transitions":
            r = rows["eigs"]
            every = list(range(len(r.levels)))
            out.append(_transitions(r, every, r, every, entry[1]))
        elif entry[0] == "expect":
            _, op, at = entry
            out.append(ExpectResult(values=_one_point(raw[:, at : at + len(op.ops)], op), index=op.source, **common))
        else:
            _, (a, b), at, p = entry
            mean_a = _one_point(raw[:, at : at + len(a.ops)], a)
            mean_b = mean_a if b is a else _one_point(raw[:, at + len(a.ops) : at + len(a.ops) + len(b.ops)], b)
            C = _pairs(raw[:, starts[p] : starts[p + 1]], a, b)
            out.append(CorrelationResult(C=C, mean_a=mean_a, mean_b=mean_b, A=a.source, B=b.source, **common))
    return out


def _evaluate(r, requests: Sequence, dynamics=None) -> list:
    """Per-level answers (rows = r.levels) to every request, from one engine sweep. ``dynamics``: runs a
    Dynamics request (it needs H)."""
    if not any(L.vector >= 0 for L in r.levels):
        raise InvalidRequest("no vectors: call qed.eigs(..., vectors=True)")
    singles, pairs, plan = _plan(requests)
    n_levels = len(r.levels)
    raw = np.zeros((n_levels, 0), complex)
    if singles or pairs:
        raw = np.asarray(r._raw.evaluate(r._spec, singles, pairs), complex).reshape(n_levels, -1)
    rows = dict(
        energies=np.array([L.energy for L in r.levels], float),
        multiplicities=np.array([L.multiplicity for L in r.levels], int),
        levels=r.levels,
        eigs=r,
        diagnostics=list(r.diagnostics),
        rows="levels",
        T=None,
        dynamics=dynamics,
    )
    return _answers(plan, raw, singles, pairs, rows)


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


def _dynamics_runner(H, T, sym, device: str):
    """Runs a Dynamics request at the measurement's temperatures (its own pass: qed.dynamics)."""
    from .dynamics import dynamics

    def run(q: Dynamics):
        return dynamics(
            H,
            q.A,
            q.omega,
            q.B,
            eta=q.eta,
            T=T,
            sym=sym,
            krylov=q.krylov,
            samples=q.samples,
            seed=q.seed,
            device=device,
        )

    return run


# qed.thermal's options, read by measure(T=...); dense_max_dim is also an eigs option.
_THERMAL_OPTIONS = ("method", "samples", "krylov", "steps", "exact_states", "seed", "dense_max_dim")


def _measure_shared(H, requests: list, T, sym, device: str, options: dict) -> MeasureResult:
    """T > 0 with Dynamics under FTLM: ONE pass -- the dynamics' source Lanczos runs also give the
    thermodynamics and every equal-time request (DynamicsSpec::thermodynamics)."""
    from .dynamics import _probes, _result, _spec
    from .symmetry import Symmetry
    from .thermal import ThermalResult, _thermal_rows

    dyn = [q for q in requests if isinstance(q, Dynamics)]
    first = dyn[0]
    key = lambda q: (tuple(np.asarray(q.omega, float).ravel()), q.eta, q.krylov, q.samples, q.seed)  # noqa: E731
    if any(key(q) != key(first) for q in dyn[1:]):
        raise InvalidRequest(
            "measure with T=: the Dynamics requests of one pass share omega, eta, krylov, samples "
            "and seed (they share each sample's source Lanczos)"
        )
    for name, value in (("samples", first.samples), ("krylov", first.krylov), ("seed", first.seed)):
        if name in options and options[name] is not None and int(options[name]) != int(value):
            raise InvalidRequest(
                f"measure with T= and Dynamics: {name}={options[name]} differs from the Dynamics "
                f"request's {value}; the pass shares one (set it on the Dynamics request)"
            )
    for name in ("steps",):
        if options.get(name) is not None:
            raise InvalidRequest(f"{name}= is an mTPQ option; the shared pass is FTLM")
    if int(options.get("exact_states", 0)) != 0:
        raise InvalidRequest("exact_states > 0 (OFTLM) is not shared with dynamics; measure Dynamics separately")
    singles, pairs, plan = _plan(requests, thermal=True)
    all_probes, layout = [], []
    for q in dyn:
        probes, axes, cross = _probes(q.A, q.B)
        layout.append((len(all_probes), len(probes), axes, cross, q))
        all_probes.extend(probes)
    sym = Symmetry.auto() if sym is None else sym
    d, temps, rows = _spec(
        first.omega,
        first.eta,
        T,
        first.krylov,
        first.samples,
        first.seed,
        1e-8,
        device,
        options.get("dense_max_dim"),
        True,
    )
    d.thermodynamics = True
    d.observables = list(singles)
    d.observable_pairs = [(list(A), list(B)) for A, B in pairs]
    diagnostics: list = []
    r = _core.sectors.dynamics(H, sym.resolve(H, diagnostics), all_probes, d)
    S = np.asarray(r.S, dtype=complex)
    answers = {
        id(q): _result(r, S[at : at + n], q.A, axes, cross, temps, rows, sym, diagnostics)
        for at, n, axes, cross, q in layout
    }
    beta = 1.0 / temps
    lnZ = np.asarray(r.lnZ, float)[rows]
    E = np.asarray(r.E, float)[rows]
    V = np.asarray(r.V, float)[rows]
    th = ThermalResult(
        T=temps,
        E=E,
        C=beta**2 * V,
        entropy=lnZ + beta * E,
        F=-temps * lnZ,
        lnZ=lnZ,
        M=None,
        chi=None,
        O=None,
        method="ftlm",
        e0=float(r.e0),
        blocks=0,
        device_blocks=int(r.device_blocks),
        symmetry=sym,
        diagnostics=diagnostics + [tuple(x) for x in r.diagnostics],
        placement=dict(r.placement),
    )
    n_x = len(singles) + sum(len(A) * len(B) for A, B in pairs)
    raw = np.asarray(r.O, complex).reshape(n_x, -1)[:, rows] if n_x else np.zeros((0, len(temps)), complex)
    rows_info = _thermal_rows(th)
    rows_info["dynamics"] = lambda q: answers[id(q)]
    results = _answers(plan, raw.T, singles, pairs, rows_info)
    th.measurements = results
    return MeasureResult(
        results=results,
        energies=th.E,
        multiplicities=np.ones(len(th.T), int),
        rows="T",
        eigs=None,
        diagnostics=list(th.diagnostics),
        T=th.T,
        thermal=th,
    )


def _measure_thermal(H, requests: list, T, sym, device: str, options: dict) -> MeasureResult:
    from .eigs import EigResult
    from .thermal import _thermal_rows, _thermal_run

    if isinstance(H, EigResult):
        raise InvalidRequest("measure with T= needs the Hamiltonian, not an EigResult")
    unknown = sorted(set(options) - set(_THERMAL_OPTIONS))
    if unknown:
        raise InvalidRequest(f"measure with T=: unknown option(s) {unknown}; the thermal ones are {_THERMAL_OPTIONS}")
    if any(isinstance(q, Dynamics) for q in requests) and str(options.get("method", "ftlm")).lower() == "ftlm":
        return _measure_shared(H, requests, T, sym, device, options)
    singles, pairs, plan = _plan(requests, thermal=True)
    th, raw = _thermal_run(
        H,
        T,
        method=options.get("method", "ftlm"),
        sym=sym,
        samples=options.get("samples", 40),
        krylov=options.get("krylov"),
        steps=options.get("steps"),
        exact_states=options.get("exact_states", 0),
        seed=options.get("seed", 0),
        device=device,
        dense_max_dim=options.get("dense_max_dim"),
        singles=singles,
        pairs=pairs,
    )
    rows = _thermal_rows(th)
    rows["dynamics"] = _dynamics_runner(H, T, sym, device)
    results = _answers(plan, raw.T, singles, pairs, rows)
    th.measurements = results
    return MeasureResult(
        results=results,
        energies=th.E,
        multiplicities=np.ones(len(th.T), int),
        rows="T",
        eigs=None,
        diagnostics=list(th.diagnostics),
        T=th.T,
        thermal=th,
    )


@_log.replays
def measure(
    H,
    requests: Sequence,
    k: int = 1,
    *,
    T=None,
    states: str = "levels",
    degeneracy_tol: float = 1e-8,
    sym=None,
    device: str = "cpu",
    **options,
) -> MeasureResult:
    """Every request (:class:`Expect`, :class:`Correlations`, :class:`Transitions`) from one pass.

    Without ``T``: in the lowest levels of ``H`` from one eigensolve, the equal-time requests sharing
    one sweep of each level's basis. ``H`` is the Hamiltonian (solved with :func:`qed.eigs` with
    vectors; ``k``, ``sym``, ``device`` and the eigs ``options`` pass through) or an
    :class:`EigResult` with vectors to reuse. ``states="levels"``: one row per returned level;
    ``"ground"``: one row, the levels within ``degeneracy_tol`` max(1, |E0|) of the lowest averaged
    with their multiplicities (partners in other blocks are among them only if the solve returned
    them: raise ``k`` or pass ``window=``).

    With ``T`` (temperatures): one thermal pass over every symmetry block (:func:`qed.thermal`'s
    ``method``, ``samples``, ``krylov``, ``steps``, ``exact_states``, ``seed`` and ``dense_max_dim`` in
    ``options``) gives every equal-time request <X>(T) = Tr(e^{-H/T} X) / Z, one row per temperature,
    and the thermodynamics in ``thermal``. Every operator of every request is measured in one sweep
    per block and sample (exact: the eigenvectors; FTLM: each sample's phi(T); mTPQ: each step, with the
    canonical series of Sugiura and Shimizu; OFTLM: its exact states and the samples' phi(T)).
    """
    requests = [requests] if isinstance(requests, (Expect, Correlations, Transitions, Dynamics)) else list(requests)
    if T is not None:
        if states != "levels":
            raise InvalidRequest("states= selects levels; with T= the rows are the temperatures")
        return _measure_thermal(H, requests, T, sym, device, options)
    thermal_only = sorted(set(options) & (set(_THERMAL_OPTIONS) - {"dense_max_dim"}))
    if thermal_only:
        raise InvalidRequest(f"{thermal_only}: thermal options, used with T=")
    states = _states(states)
    from .eigs import EigResult

    runner = None if isinstance(H, EigResult) else _dynamics_runner(H, None, sym, device)
    if requests and all(isinstance(q, Dynamics) for q in requests):  # no eigensolve needed
        if runner is None:
            raise InvalidRequest("Dynamics needs the Hamiltonian: measure H, not an EigResult")
        return MeasureResult(
            results=[runner(q) for q in requests],
            energies=np.zeros(0),
            multiplicities=np.zeros(0, int),
            rows="levels",
            eigs=None,
        )
    r = _solve(H, k, sym, device, options)
    results = _evaluate(r, requests, dynamics=runner)
    energies = np.array([L.energy for L in r.levels], float)
    mult = np.array([L.multiplicity for L in r.levels], int)
    if states == "ground":  # the T = 0 dynamics already averages over the ground manifold
        results = [x if isinstance(q, Dynamics) else x.ground(degeneracy_tol) for q, x in zip(requests, results)]
        sel, _ = _ground(energies, mult, degeneracy_tol)
        energies, mult = np.array([energies[sel].min()]), np.array([int(mult[sel].sum())])
    return MeasureResult(
        results=results, energies=energies, multiplicities=mult, rows=states, eigs=r, diagnostics=list(r.diagnostics)
    )


# ---- the one-request verbs ---------------------------------------------------------------------
def expect(
    H,
    ops,
    k: int = 1,
    *,
    T=None,
    states: str = "levels",
    degeneracy_tol: float = 1e-8,
    sym=None,
    device: str = "cpu",
    **options,
) -> ExpectResult:
    """<O> for every operator in ``ops`` (an Operator, a sequence, a :class:`qed.Family` or a
    :class:`qed.MomentumFamily`) in each of the lowest levels of ``H``, or at the temperatures ``T``:
    :func:`measure` with one :class:`Expect` request.

    The value in a level is averaged over the level's symmetry multiplet -- the quantity that does
    not depend on which partner the solver returned; for a non-degenerate level it is just
    <psi|O|psi>. With a total-spin restriction and an SU(2)-symmetric H the multiplet includes its
    2S + 1 Sz members, so an operator that is not SU(2) invariant contributes its SU(2)-scalar part
    (its average over all spin rotations); in a uniform field each member is a level of its own.
    """
    return measure(
        H, [Expect(ops)], k, T=T, states=states, degeneracy_tol=degeneracy_tol, sym=sym, device=device, **options
    )[0]


def correlations(
    H,
    A,
    B=None,
    k: int = 1,
    *,
    T=None,
    states: str = "levels",
    degeneracy_tol: float = 1e-8,
    sym=None,
    device: str = "cpu",
    **options,
) -> CorrelationResult:
    """<A_a^dag B_b> for every pair (``B=None``: B = A) in each of the lowest levels of ``H``, or at
    the temperatures ``T``: :func:`measure` with one :class:`Correlations` request. ``A`` and ``B``: an
    Operator, a sequence, a :class:`qed.Family` (e.g. ``qed.Family.spins(lattice)``) or a
    :class:`qed.MomentumFamily`; the result's :meth:`CorrelationResult.fourier` gives the structure
    factor."""
    return measure(
        H, [Correlations(A, B)], k, T=T, states=states, degeneracy_tol=degeneracy_tol, sym=sym, device=device, **options
    )[0]


@_log.replays
def transitions(A, initial, final=None, *, B=None, pairs: bool = False, raw: bool = False) -> TransitionResult:
    """Transitions <m|A|n> from the ``initial`` levels to the ``final`` levels (``final=None``: the
    initial ones), each an :class:`EigResult` with vectors or ``(EigResult, level indices)``; the two
    may come from different, targeted eigs calls over one spatial symmetry (e.g. the ground state and
    ``qed.eigs(H, per_block=4, vectors=True)``). Returns a :class:`TransitionResult`: line strengths
    of every member of ``A`` and, with ``B`` or ``pairs=True``, the pair matrices
    T_ab = (1/d_n) sum <n'|A_a^dag|m'><m'|B_b|n'>. ``A`` and ``B``: an Operator, a sequence, a
    :class:`qed.Family` or a :class:`qed.MomentumFamily`."""
    ri, ii = _level_set(initial, "transitions initial")
    rf, jj = (ri, ii) if final is None else _level_set(final, "transitions final")
    return _transitions(ri, ii, rf, jj, Transitions(A, B, pairs, raw))
