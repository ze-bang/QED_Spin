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

:meth:`select` narrows the sectors (a star representative, a little-group irrep)
without changing the symmetry.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
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
    only_k0: Sequence[int] = field(default_factory=tuple)
    only_irrep: Sequence[int] = field(default_factory=tuple)

    @classmethod
    def auto(cls) -> "Symmetry":
        return cls()

    @classmethod
    def none(cls) -> "Symmetry":
        return cls(spatial=None, sz="off", spin_flip="off", time_reversal="off")

    def select(self, *, sz: Any = None, k0: Optional[Sequence[int]] = None,
               irrep: Optional[Sequence[int]] = None) -> "Symmetry":
        """The same symmetry restricted to some sectors."""
        out = self
        if sz is not None:
            out = replace(out, sz=sz)
        if k0 is not None:
            out = replace(out, only_k0=tuple(int(k) for k in k0))
        if irrep is not None:
            out = replace(out, only_irrep=tuple(int(i) for i in irrep))
        return out

    # ------------------------------------------------------------------
    def groups(self, H) -> tuple[list[list[int]], list[list[int]]]:
        """(closed abelian group, point-group coset representatives) for H."""
        from ..point_group_routing import close_group, split_nonabelian

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
            spatial = find_symmetries(H, verbose=False).full_set
        gens = getattr(spatial, "generators", None)
        if gens is not None and not gens:
            return identity, []
        if self.point_group:
            split = split_nonabelian(spatial)
            if not isinstance(split, str):
                A, residues = split
                return [list(a) for a in A], [list(r) for r in residues]
        base = gens if gens is not None else spatial
        A = close_group([list(g) for g in base])
        if A is None:
            raise ValueError("the spatial group exceeds the closure cap")
        from ..point_group_routing import greedy_maximal_abelian
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
        spec.only_k0 = list(self.only_k0)
        spec.only_irrep = list(self.only_irrep)
        return spec
