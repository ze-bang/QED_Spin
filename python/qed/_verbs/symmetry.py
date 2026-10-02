"""``qed.Symmetry``: one object naming every symmetry a calculation may use.

A Symmetry is a request, resolved against a Hamiltonian by :meth:`Symmetry.resolve`
into the engine's :class:`qed._core.sectors.Spec`:

* ``spatial`` -- ``"auto"`` (the automorphisms of H that commute with it, found by
  :func:`qed.find_symmetries`), a list of site permutations, a :class:`qed.Symmetries`, or
  ``None``. A list is closed and split into its largest normal abelian subgroup (the
  momenta) and one representative per coset of it (the point group); a ``Symmetries``
  is an explicit split: its ``abelian`` part (generators or the whole group) must be
  abelian and every residue must normalise it. With accidental symmetry (a cluster whose
  graph has more automorphisms than its lattice) the abelian part need not be the lattice
  translations: pass the translations and point group as a list to label by them.
* ``sz`` -- ``"auto"`` (decompose by Sz, or by Sz parity when H only conserves that),
  an integer n (one Sz sector: n spins up, Sz = n - N/2), ``"even"`` / ``"odd"`` (the half
  whose number of up spins has that parity), or ``"off"``.
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
from dataclasses import dataclass, field, replace
from fractions import Fraction
from typing import Any, Optional, Sequence

from .. import _core
from .._perm import order
from ..errors import InvalidRequest

_TOGGLE = {"auto": -1, "off": 0, "require": 1}


def _toggle(value: str, name: str) -> int:
    try:
        return _TOGGLE[str(value).lower()]
    except KeyError:
        raise InvalidRequest(f"{name} must be one of {sorted(_TOGGLE)}, got {value!r}") from None


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
        Blocks whose little group lacks some R are dropped. A star whose little co-group is
        trivial has the one-dimensional trivial irrep (character 1 on the identity); an
        element that acts on a small sector as a scalar c has character c there. Time
        reversal is not folded under this selection, so each irrep is its own block.
        ``k0`` / ``irrep``: the engine's own star and irrep indices, as reported on levels;
        ``irrep`` names projected blocks only.
        A selection that matches no block raises :class:`qed.errors.EmptySelection`.
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
    def groups(self, H, diagnostics: Optional[list] = None) -> tuple[list[list[int]], list[list[int]]]:
        """(closed abelian group, point-group coset representatives) for H: the abelian group is
        normal in the whole spatial group, the identity first. ``diagnostics``, when given,
        receives a (code, message) pair for each fallback taken."""
        from .._groups import spatial_split, split_generator_set
        from .._perm import close_group, is_permutation

        n = int(H.num_sites)
        identity = [list(range(n))]
        spatial = self.spatial
        if spatial is None:
            return identity, []
        if isinstance(spatial, str):
            if spatial.lower() != "auto":
                raise InvalidRequest(f"spatial must be 'auto', a permutation list, a Symmetries or None, "
                                     f"got {spatial!r}")
            from ..discovery import find_symmetries
            try:
                report = find_symmetries(H, verbose=False)
            except ImportError as e:        # the graph-automorphism search needs pynauty
                import warnings
                msg = f"Symmetry(spatial='auto'): {e}; continuing without spatial symmetry"
                warnings.warn(msg, RuntimeWarning, stacklevel=3)
                if diagnostics is not None:
                    diagnostics.append(("auto_spatial_skipped", msg))
                return identity, []
            if diagnostics is not None:
                diagnostics.extend(report.diagnostics)
            A, residues = report.abelian or identity, report.residues
        else:
            gens = getattr(spatial, "abelian", None)        # a Symmetries: an explicit split
            star = []
            if gens is not None and self.point_group:   # else its residues go unchecked
                sp = getattr(spatial, "residues", None)
                star = list(sp) if sp is not None else []
            perms = list(gens) + star if gens is not None else list(spatial)
            # Group arithmetic on a map that is not a bijection never closes (its powers never
            # return to the identity): refuse it before any.
            for p in perms:
                if not is_permutation(p, n):
                    raise InvalidRequest(f"spatial symmetry: {list(p)} is not a permutation of the {n} sites")
            if not perms:
                return identity, []
            if gens is not None:            # an explicit split: checked, not re-chosen
                A, residues = split_generator_set(gens, star, n)
            else:
                G = close_group(perms)
                if G is None:
                    raise InvalidRequest("the spatial group exceeds the 4096-element closure cap")
                A, residues, notes = spatial_split(G)
                if diagnostics is not None:
                    diagnostics.extend(notes)
        return [list(a) for a in A], ([list(r) for r in residues] if self.point_group else [])

    def resolve(self, H, diagnostics: Optional[list] = None) -> "_core.sectors.Spec":
        """The engine's Spec for H; ``diagnostics`` as in :meth:`groups`."""
        spec = _core.sectors.Spec()
        spec.abelian, spec.residues = self.groups(H, diagnostics)
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
                raise InvalidRequest(f"sz must be 'auto', 'off', 'even', 'odd' or an int, got {sz!r}")
        elif sz is not None:
            spec.n_up = int(sz)
        spec.spin_flip = _toggle(self.spin_flip, "spin_flip")
        spec.time_reversal = _toggle(self.time_reversal, "time_reversal")
        if self.total_spin is not None:
            two_s = round(2 * float(self.total_spin))
            if abs(two_s - 2 * float(self.total_spin)) > 1e-9 or two_s < 0:
                raise InvalidRequest("total_spin must be a non-negative multiple of 1/2, "
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
                        raise InvalidRequest(f"select(momentum=...): {list(T)} is not in the abelian group")
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
                        raise InvalidRequest(f"select(irrep_character=...): {list(R)} is not a point-group "
                                         "coset representative (see Symmetry.groups)")
                reqs.append(c)
            spec.only_irrep_chars = reqs
        return spec


def momentum_of(level, spec, translations) -> tuple:
    """The momentum of a level along each translation T: the fraction theta in [0, 1) with
    T|psi> = exp(-2 pi i theta)|psi>, where T acts on basis states as
    bit i of T|s> = bit T[i] of |s>. The level's star representative is reported."""
    index = {tuple(a): i for i, a in enumerate(spec.abelian)}
    out = []
    for T in translations:
        key = tuple(int(x) for x in T)
        if key not in index:
            raise InvalidRequest(f"momentum_of: {list(T)} is not in the abelian group")
        chi = complex(level.momentum[index[key]])
        theta = (-cmath.phase(chi) / (2 * cmath.pi)) % 1.0
        out.append(Fraction(theta).limit_denominator(order(key)) % 1)
    return tuple(out)


def irrep_characters_of(level, spec, n_sites: int) -> dict:
    """{R: chi_sigma(R)} over the level's little co-group, R the coset representatives
    (the identity included: its character is the irrep dimension), and over the residues
    that act on the level's sector as a multiple of a co-group element. A star with a trivial
    co-group reports the trivial irrep ({identity: 1}); empty when the co-group could not be
    projected (the block mixes irreps)."""
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
