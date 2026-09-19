"""Block-resolved spectra and observables with physical labels.

The little-group engine splits a symmetric Hamiltonian into blocks labelled by a
momentum star, a little-co-group irrep and (at Sz = 0) the spin-flip parity. Its raw
verbs (``qed._core.little_group_*``) report those labels as ENGINE-INTERNAL indices:
``k_raw`` is a position in the engine's irrep decomposition, not a momentum, and
``irrep`` is an index into a per-star table. This module is the campaign-grade surface
over them. Every level carries

* ``momenta`` -- the star's momenta, each reduced against the caller's translation
  generators: ``kappa_j`` in [0, 1) with ``chi(T_j) = exp(-2 pi i kappa_j)``;
* ``characters`` -- the little-co-group irrep as ``{residue index: character}``
  (``-1`` is the identity), so the irrep is named by what it IS;
* optionally ``point`` / ``irrep_name`` from a lattice ``namer`` (e.g.
  :meth:`qed.lattice.TriangularSupercell.namer` gives 'K', 'A1');
* ``values`` -- ``<n|O_i|n>`` for the observables passed, computed in the sector's
  representative basis (never expanded to 2^N).

With ``O_i = dH/dlambda_i`` the values are Hellmann-Feynman derivatives
(:func:`dE_dlambda`). Non-convergence is never silent: every level has
``converged``/``residual`` and the result has ``unconverged_blocks``;
``strict=True`` (the default) raises instead of returning an incomplete result.
"""
from __future__ import annotations

import cmath
import dataclasses
import math
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Callable, Optional, Sequence

from . import _core

__all__ = ["BlockLevel", "BlockResult", "solve_blocks", "dE_dlambda", "dimer_zz_observables"]


@dataclass(frozen=True)
class BlockLevel:
    energy: float
    level: int                         # 0 = lowest level of its block
    momenta: tuple                     # ((kappa_1, kappa_2, ...), ...) of the star, Fractions
    characters: dict                   # {residue index or -1: complex}; {} if unprojected
    irrep_dim: int
    flip: int                          # -1 not engaged, 0 even, 1 odd under the spin flip
    multiplicity: int                  # degeneracy the block's energies carry in the full space
    converged: bool
    residual: Optional[float] = None
    values: tuple = ()                 # <n|O_i|n>, aligned with the observables
    diagonal_values: tuple = ()        # <n|D_j|n>, aligned with diagonal_observables
    point: Optional[str] = None
    irrep_name: Optional[str] = None
    k_raw: int = -1                    # engine-internal abelian irrep of the star representative
    k0: int = -1                       # engine-internal star index (with flip): pass to only_k0
    irrep_index: int = -1              # engine-internal

    @property
    def label(self) -> str:
        """'K.A1+' style label when a namer was given, else the first momentum."""
        if self.point is not None:
            s = "" if self.flip < 0 else ("+" if self.flip == 0 else "-")
            return f"{self.point}.{self.irrep_name}{s}"
        return str(tuple(str(x) for x in self.momenta[0])) if self.momenta else f"k_raw={self.k_raw}"


@dataclass
class BlockResult:
    levels: list
    observables: int = 0
    unconverged_blocks: int = 0
    flip_engaged: bool = False
    tr_engaged: bool = False
    stars: list = field(default_factory=list)

    def ground(self) -> BlockLevel:
        return min(self.levels, key=lambda l: l.energy)

    def select(self, point=None, irrep=None, flip=None, level=None) -> list:
        """Levels matching every given name (point / irrep need a namer)."""
        return [l for l in self.levels
                if (point is None or l.point == point)
                and (irrep is None or l.irrep_name == irrep)
                and (flip is None or l.flip == flip)
                and (level is None or l.level == level)]

    def lowest(self, point, irrep, flip=None) -> BlockLevel:
        hits = self.select(point, irrep, flip)
        if not hits:
            raise KeyError(f"no block {point}.{irrep} (flip={flip})")
        return min(hits, key=lambda l: l.energy)


def _kappa(row, A_index, gens):
    out = []
    for g in gens:
        a = A_index[tuple(g)]
        order, q = 1, list(g)
        ident = list(range(len(g)))
        while q != ident:
            q = [g[i] for i in q]
            order += 1
        n = int(round(-cmath.phase(complex(row[a])) * order / (2 * math.pi))) % order
        out.append(Fraction(n, order))
    return tuple(out)


def solve_blocks(H, abelian_group, residue_perms, *, k: int = 1,
                 observables: Sequence = (), diagonal_observables: Sequence = (),
                 momentum_generators: Sequence = None,
                 namer: Callable = None, n_up: int = -1, sz_parity: int = -1,
                 spin_flip: int = -1, time_reversal: int = -1,
                 dense_max_dim: int = 256, use_gpu: bool = False,
                 block_size: int = 1, only_k0: Sequence[int] = (),
                 only_irrep: Sequence[int] = (),
                 strict: bool = True) -> BlockResult:
    """Lowest ``k`` levels of every block of ``H``, labelled, with ``<n|O_i|n>``.

    ``abelian_group``: the closed abelian group (translations) as permutations.
    ``residue_perms``: point-group coset representatives (non-identity elements);
    observables must commute with all of them and with the translations.
    ``momentum_generators``: translations to reduce momenta against (default: none
    -> ``momenta`` is empty). ``namer``: callable ``BlockLevel -> (point, irrep)``.
    k = 1 without observables runs the engine's certified block-grounds lane; any
    other request runs the block-expectations lane (vectors in the rep basis).
    ``only_k0`` / ``only_irrep`` restrict the solve to star representatives / irrep
    indices (engine-internal: take them from ``BlockLevel.k0`` and ``irrep_index`` of
    a previous, e.g. cheaper, result -- never from a guess).
    """
    A = [list(map(int, a)) for a in abelian_group]
    R = [list(map(int, r)) for r in residue_perms]
    obs = list(observables)
    common = dict(n_up=n_up, sz_parity=sz_parity, dense_max_dim=dense_max_dim,
                  use_gpu=use_gpu, spin_flip=spin_flip, time_reversal=time_reversal,
                  only_k0=list(only_k0), only_irrep=list(only_irrep))
    diag = [[(float(w), [int(i) for i in sites]) for (w, sites) in d]
            for d in diagonal_observables]
    if k == 1 and not obs and not diag and block_size == 1:
        out = dict(_core.little_group_block_grounds(H, A, R, **common))
        energies = list(out["eigenvalues"])
        level_of = [0] * len(energies)
        residuals = [None] * len(energies)
        values = [()] * len(energies)
        dvalues = [()] * len(energies)
    else:
        out = dict(_core.little_group_block_expectations(H, obs, A, R, k=k,
                                                          block_size=block_size,
                                                          diagonal_observables=diag, **common))
        energies = list(out["energies"])
        level_of = list(out["level"])
        residuals = list(out["residuals"])
        values = [tuple(float(x) for x in row) for row in out["values"]]
        dvalues = [tuple(float(x) for x in row) for row in out["diagonal_values"]]
    if strict and int(out["unconverged_blocks"]) > 0:
        raise RuntimeError(f"{out['unconverged_blocks']} block(s) did not converge; "
                           "pass strict=False to inspect the certified part")

    gens = [list(map(int, g)) for g in (momentum_generators or [])]
    # irrep_characters columns follow the caller's order of abelian_group (pinned by
    # test_labels_do_not_depend_on_the_order_of_the_group).
    A_index = {tuple(a): i for i, a in enumerate(A)}
    for g in gens:
        if tuple(g) not in A_index:
            raise ValueError("a momentum generator is not an element of abelian_group")
    chars = out["irrep_characters"]
    n_a = len(A)
    stars = list(out["stars"])
    by_member = {}
    for st in stars:
        for m in list(st["members"]) + [st["k0"]]:
            by_member.setdefault(int(m), st)

    levels = []
    for i, e in enumerate(energies):
        k_raw, flip = int(out["k_raw"][i]), int(out["flip_parity"][i])
        irr = int(out["irrep"][i])
        st = by_member.get(k_raw + (flip if flip > 0 else 0) * n_a)
        members = [int(m) % n_a for m in st["members"]] if st is not None else [k_raw]
        momenta = tuple(sorted({_kappa(chars[m], A_index, gens) for m in members})) if gens else ()
        characters = {}
        if st is not None and irr >= 0 and st["little_characters"]:
            characters = {int(el): complex(c) for el, c in
                          zip(st["little_elems"], st["little_characters"][irr])}
        lv = BlockLevel(energy=float(e), level=int(level_of[i]), momenta=momenta,
                        characters=characters, irrep_dim=int(out["irrep_dim"][i]),
                        flip=flip, multiplicity=int(out["multiplicity"][i]),
                        converged=bool(out["converged"][i]),
                        residual=None if residuals[i] is None else float(residuals[i]),
                        values=values[i], diagonal_values=dvalues[i],
                        k_raw=k_raw, irrep_index=irr,
                        k0=int(st["k0"]) if st is not None else -1)
        if namer is not None:
            point, name = namer(lv)
            lv = dataclasses.replace(lv, point=point, irrep_name=name)
        levels.append(lv)
    return BlockResult(levels=levels, observables=len(obs),
                       unconverged_blocks=int(out["unconverged_blocks"]),
                       flip_engaged=bool(out["flip_engaged"]),
                       tr_engaged=bool(out["tr_engaged"]), stars=stars)


def dE_dlambda(H, dH: Sequence, abelian_group, residue_perms, **kw) -> BlockResult:
    """Hellmann-Feynman derivatives: ``values[i] = dE/dlambda_i`` for
    ``H(lambda) = H + sum_i lambda_i dH[i]``, per block level. Exact for a level that
    is non-degenerate inside its block; for a degenerate one it is the value in the
    solver's vector, and degenerate first-order theory needs the full multiplet."""
    return solve_blocks(H, abelian_group, residue_perms, observables=dH, **kw)


def dimer_zz_observables(bonds, cell_shift):
    """Diagonal observables for zz-dimer correlations, as ``diagonal_observables``.

    ``bonds[t][c] = (i, j)``: the bond of type ``t`` in cell ``c``, with
    ``B_{t,c} = S^z_i S^z_j``. ``cell_shift[d][c]`` is the cell ``c`` moves to under the
    ``d``-th displacement. Returns ``(observables, keys)``: ``("B", t)`` is
    ``sum_c B_{t,c}`` and ``("D", t, u, d)`` is ``sum_c B_{t,c} B_{u,shift[d][c]}``, both
    translation invariant when the bond table is translation covariant. Divide B by
    the number of cells for the mean bond value."""
    n_type, n_cell = len(bonds), len(bonds[0])
    obs, keys = [], []
    for t in range(n_type):
        obs.append([(1.0, list(bonds[t][c])) for c in range(n_cell)])
        keys.append(("B", t))
    for t in range(n_type):
        for u in range(n_type):
            for d, shift in enumerate(cell_shift):
                obs.append([(1.0, list(bonds[t][c]) + list(bonds[u][shift[c]]))
                            for c in range(n_cell)])
                keys.append(("D", t, u, d))
    return obs, keys
