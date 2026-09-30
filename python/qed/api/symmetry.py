"""``qed.Symmetry``: one object naming every symmetry a calculation may use.

A Symmetry is a request, resolved against a Hamiltonian by :meth:`Symmetry.resolve`
into the engine's :class:`qed._core.sectors.Spec`:

* ``spatial`` -- ``"auto"`` (the automorphisms of H that commute with it, found by
  :func:`qed.find_symmetries`), a ``GeneratorSet``, a list of site permutations, or
  ``None``. The group is split into a closed abelian part (the momenta) and one
  coset representative per point-group element (the little groups).
* ``sz`` -- ``"auto"`` (decompose by Sz, or by Sz parity when H only conserves that),
  an integer (one Sz sector, counted in set bits), ``"even"`` / ``"odd"`` (one parity
  half), or ``"off"``.
* ``spin_flip`` / ``time_reversal`` -- ``"auto"`` (use when H has it), ``"off"``,
  ``"require"`` (fail when H lacks it).
* ``point_group`` -- ``False`` keeps only the abelian part.
* ``total_spin`` -- a number S restricts to total spin S (H must be SU(2) symmetric);
  each level then counts 2S + 1 times.

:meth:`select` narrows the sectors (a momentum, a little-group irrep named by its character,
or the engine's star/irrep indices) without changing the symmetry.
"""
from __future__ import annotations

import cmath
import math
from dataclasses import dataclass, field, replace
from fractions import Fraction
from typing import Any, Optional, Sequence

from .. import _core

_TOGGLE = {"auto": -1, "off": 0, "require": 1}


def _toggle(value: str, name: str) -> int:
    try:
        return _TOGGLE[str(value).lower()]
    except KeyError:
        raise ValueError(f"{name} must be one of {sorted(_TOGGLE)}, got {value!r}") from None


@dataclass(frozen=True)
class Symmetry:
    spatial: Any = "auto"
    sz: Any = "auto"
    spin_flip: str = "auto"
    time_reversal: str = "auto"
    point_group: bool = True
    total_spin: Optional[float] = None
    only_k0: Sequence[int] = field(default_factory=tuple)
    only_irrep: Sequence[int] = field(default_factory=tuple)
    only_momentum: Sequence = field(default_factory=tuple)          # (((perm), Fraction), ...) per request
    only_irrep_character: Sequence = field(default_factory=tuple)   # (((perm), chi), ...) per request

    @classmethod
    def auto(cls) -> "Symmetry":
        return cls()

    @classmethod
    def none(cls) -> "Symmetry":
        return cls(spatial=None, sz="off", spin_flip="off", time_reversal="off")

    def select(self, *, sz: Any = None, k0: Optional[Sequence[int]] = None,
               irrep: Optional[Sequence[int]] = None, momentum: Any = None,
               irrep_character: Any = None) -> "Symmetry":
        """The same symmetry restricted to some sectors.

        ``momentum``: ``{T: theta}`` keeps the momenta with T|psi> = exp(-2 pi i theta)|psi>
        for each given translation T (a permutation in the abelian group; theta a fraction
        of a full turn, see :func:`momentum_of`). A list of such mappings keeps any of them.
        A level answers for its whole star, so a selected star also counts its partners.
        ``irrep_character``: ``{R: chi}`` keeps the little-co-group irreps with character
        chi on each given point-group element R (a coset representative, as listed by
        :meth:`groups`; the identity names the irrep dimension); a list keeps any of them.
        Blocks whose little group lacks some R are dropped. Time reversal is not folded
        under this selection, so each irrep is its own block.
        ``k0`` / ``irrep``: the engine's own star and irrep indices, as reported on levels.
        """
        out = self
        if sz is not None:
            out = replace(out, sz=sz)
        if k0 is not None:
            out = replace(out, only_k0=tuple(int(k) for k in k0))
        if irrep is not None:
            out = replace(out, only_irrep=tuple(int(i) for i in irrep))
        if momentum is not None:
            reqs = [momentum] if isinstance(momentum, dict) else list(momentum)
            out = replace(out, only_momentum=tuple(
                tuple((tuple(int(x) for x in T), Fraction(th).limit_denominator(1 << 20))
                      for T, th in r.items()) for r in reqs))
        if irrep_character is not None:
            reqs = [irrep_character] if isinstance(irrep_character, dict) else list(irrep_character)
            out = replace(out, only_irrep_character=tuple(
                tuple((tuple(int(x) for x in R), complex(c)) for R, c in r.items()) for r in reqs))
        return out

    # ------------------------------------------------------------------
    def groups(self, H) -> tuple[list[list[int]], list[list[int]]]:
        """(closed abelian group, point-group coset representatives) for H."""
        from .._groups import close_group, split_nonabelian

        n = int(H.num_sites)
        identity = [list(range(n))]
        spatial = self.spatial
        if spatial is None:
            return identity, []
        if isinstance(spatial, str):
            if spatial.lower() != "auto":
                raise ValueError(f"spatial must be 'auto', a GeneratorSet, a permutation list "
                                 f"or None, got {spatial!r}")
            from ..discovery import find_symmetries
            try:
                spatial = find_symmetries(H, verbose=False).full_set
            except ImportError as e:        # the graph-automorphism search needs pynauty
                import warnings
                warnings.warn(f"Symmetry(spatial='auto'): {e}; continuing without spatial "
                              "symmetry", RuntimeWarning, stacklevel=3)
                return identity, []
            if spatial is None:               # H has no spatial symmetry
                return identity, []
        gens = getattr(spatial, "generators", None)
        if gens is not None and not gens:
            return identity, []
        # Group arithmetic on a map that is not a bijection never closes (its powers never
        # return to the identity): refuse it before any.
        for p in (gens if gens is not None else spatial):
            if sorted(int(x) for x in p) != list(range(n)):
                raise ValueError(f"spatial symmetry: {list(p)} is not a permutation of the {n} sites")
        if self.point_group:
            split = split_nonabelian(spatial)
            if not isinstance(split, str):
                A, residues = split
                return [list(a) for a in A], [list(r) for r in residues]
        base = gens if gens is not None else spatial
        A = close_group([list(g) for g in base])
        if A is None:
            raise ValueError("the spatial group exceeds the closure cap")
        from .._groups import greedy_maximal_abelian
        if gens is None:            # a raw permutation list may be non-abelian
            A = greedy_maximal_abelian(A)
        return [list(a) for a in A], []

    def resolve(self, H) -> "_core.sectors.Spec":
        spec = _core.sectors.Spec()
        spec.abelian, spec.residues = self.groups(H)
        sz = self.sz
        if isinstance(sz, str):
            key = sz.lower()
            if key == "auto":
                pass
            elif key == "off":
                spec.use_sz = False
            elif key in ("even", "odd"):
                spec.sz_parity = 0 if key == "even" else 1
            else:
                raise ValueError(f"sz must be 'auto', 'off', 'even', 'odd' or an int, got {sz!r}")
        elif sz is not None:
            spec.n_up = int(sz)
        spec.spin_flip = _toggle(self.spin_flip, "spin_flip")
        spec.time_reversal = _toggle(self.time_reversal, "time_reversal")
        if self.total_spin is not None:
            two_s = round(2 * float(self.total_spin))
            if abs(two_s - 2 * float(self.total_spin)) > 1e-9 or two_s < 0:
                raise ValueError("total_spin must be a non-negative multiple of 1/2, "
                                 f"got {self.total_spin!r}")
            spec.two_S = int(two_s)
        spec.only_k0 = list(self.only_k0)
        spec.only_irrep = list(self.only_irrep)
        if self.only_momentum:
            index = {tuple(a): i for i, a in enumerate(spec.abelian)}
            reqs = []
            for req in self.only_momentum:
                c = []
                for T, th in req:
                    if T not in index:
                        raise ValueError(f"select(momentum=...): {list(T)} is not in the abelian group")
                    c.append((index[T], complex(cmath.exp(-2j * cmath.pi * float(th)))))
                reqs.append(c)
            spec.only_momentum = reqs
        if self.only_irrep_character:
            ident = tuple(range(int(H.num_sites)))
            index = {tuple(r): i for i, r in enumerate(spec.residues)}
            reqs = []
            for req in self.only_irrep_character:
                c = []
                for R, chi in req:
                    if R == ident:
                        c.append((-1, chi))
                    elif R in index:
                        c.append((index[R], chi))
                    else:
                        raise ValueError(f"select(irrep_character=...): {list(R)} is not a point-group "
                                         "coset representative (see Symmetry.groups)")
                reqs.append(c)
            spec.only_irrep_chars = reqs
        return spec


def _order(p) -> int:
    seen, order = set(), 1
    for s in range(len(p)):
        if s in seen:
            continue
        n, t = 0, s
        while t not in seen:
            seen.add(t)
            t = p[t]
            n += 1
        order = order * n // math.gcd(order, n)
    return order


def momentum_of(level, spec, translations) -> tuple:
    """The momentum of a level along each translation T: the fraction theta in [0, 1) with
    T|psi> = exp(-2 pi i theta)|psi>, where T acts on basis states as
    bit i of T|s> = bit T[i] of |s>. The level's star representative is reported."""
    index = {tuple(a): i for i, a in enumerate(spec.abelian)}
    out = []
    for T in translations:
        key = tuple(int(x) for x in T)
        if key not in index:
            raise ValueError(f"momentum_of: {list(T)} is not in the abelian group")
        chi = complex(level.momentum[index[key]])
        theta = (-cmath.phase(chi) / (2 * cmath.pi)) % 1.0
        out.append(Fraction(theta).limit_denominator(_order(key)) % 1)
    return tuple(out)


def irrep_characters_of(level, spec, n_sites: int) -> dict:
    """{R: chi_sigma(R)} over the level's little co-group, R the coset representatives
    (the identity included: its character is the irrep dimension); empty for a block
    without a co-group decomposition."""
    ident = tuple(range(n_sites))
    return {(ident if e < 0 else tuple(spec.residues[e])): complex(c) for e, c in level.irrep_characters}


class Labelled:
    """Physical labels of the levels of a result holding ``levels``, ``_spec``, ``_n_sites``."""

    def momentum(self, i: int, translations) -> tuple:
        """Momentum of ``levels[i]`` along each translation (see :func:`momentum_of`)."""
        return momentum_of(self.levels[i], self._spec, translations)

    def irrep_characters(self, i: int) -> dict:
        """Little-co-group characters of ``levels[i]`` (see :func:`irrep_characters_of`)."""
        return irrep_characters_of(self.levels[i], self._spec, self._n_sites)
