"""Symmetry discovery: the automorphisms of H's coloured interaction graph that commute
with H (``find_symmetries``), split into the largest normal abelian subgroup (the momenta)
and one representative per coset of it (the point group), as :class:`Symmetries`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from . import _core as _core
from . import _log
from ._core import Operator  # type: ignore[attr-defined]

Permutation = list[int]

_OP_CODE = {"+": 0, "-": 1, "z": 2}   # Operator.terms() factors as OP_SPLUS / OP_SMINUS / OP_SZ


def _operator_to_graph_records(
    operator: Operator,
) -> tuple[dict[int, tuple], list[dict[str, Any]], list[tuple]]:
    """Build (vertex_weights, edges, triples) records the
    ``automorphism_finder`` routines consume, from H's canonical terms (``Operator.terms()``):
    the graph does not depend on how H was written (cancelling records, same-site products,
    either order of a bond). A term on more than three sites does not enter the graph; the
    exact check after the search still sees it."""
    num_sites = int(operator.num_sites)
    onsite: dict[int, dict[int, complex]] = {i: {} for i in range(num_sites)}
    edges: list[dict[str, Any]] = []
    triples: list[tuple] = []
    for coeff, ops, sites in operator.terms():
        c = complex(coeff)
        codes = [_OP_CODE[o] for o in ops]
        sites = [int(s) for s in sites]
        if len(sites) == 1:      # every one-body term on the site colours it
            onsite[sites[0]][codes[0]] = onsite[sites[0]].get(codes[0], 0j) + c
        elif len(sites) == 2:
            edges.append({"vertex1": sites[0], "vertex2": sites[1], "type1": codes[0], "type2": codes[1],
                          "weight": (float(c.real), float(c.imag))})
        elif len(sites) == 3:
            triples.append((tuple(sites), tuple(codes), c))
    vertex_weights = {i: tuple(sorted((op, round(c.real, 8), round(c.imag, 8))
                                      for op, c in d.items() if abs(c) > 1e-12))
                      for i, d in onsite.items()}
    return vertex_weights, edges, triples


def _keep_hamiltonian_symmetries(operator: Any, autos: list[Permutation]) -> list[Permutation]:
    """Only the automorphisms that commute with the whole operator.

    The colored graph sees which pairs and triples of sites interact, but not the
    orientation of a coupling: a permutation that reverses a DM bond or a scalar-chirality
    triangle maps the term to minus itself and is still a graph automorphism. (Before the
    triples entered the graph, 1242 of the 1296 automorphisms of the 3x3 triangular torus
    with chirality on up-triangles were not symmetries, and the projection lane folded k
    with -k from them.) The term-level check is exact and sees every term; the survivors
    are the intersection of two groups, hence a group."""
    if not autos:
        return autos
    from . import _core
    keep = list(_core.check_generators_commute(operator, [list(map(int, p)) for p in autos]))
    return [p for p, ok in zip(autos, keep) if ok]




@dataclass
class Symmetries:
    """A split of a spatial symmetry group, the form ``Symmetry(spatial=...)`` takes it in:
    ``abelian`` generates (or lists) the abelian part, the momenta; ``residues`` holds one
    representative per coset of it, the point group. Every residue must normalise the abelian
    part. :func:`find_symmetries` returns the closed abelian group (sorted, the identity first)
    and the residues it chose; ``diagnostics`` holds a (code, message) pair for each cut it made:
    ``("aut_capped", ...)`` when the automorphism group was too large to enumerate and no
    spatial symmetry is used, ``("co_group_capped", ...)`` when the group was cut down to a
    subgroup whose co-group the engine can take (``abelian`` is then a maximal abelian
    subgroup, normal in the subgroup the residues span with it)."""

    abelian: list
    residues: list = field(default_factory=list)
    diagnostics: list = field(default_factory=list)

    def describe(self) -> str:
        """The group's order, its abelian part and residues, then one line per cut made."""
        from ._perm import close_group
        A = close_group(self.abelian) if self.abelian else None
        a, r = (len(A) if A is not None else len(self.abelian)), len(self.residues)
        lines = [f"{a * (r + 1)} spatial symmetries: an abelian part of {a} (the momenta) and "
                 f"{r} residue(s) (the point group)"]
        lines += [f"{code}: {msg}" for code, msg in self.diagnostics]
        return "\n".join(lines)


_FIND_SYM_MEMO: dict = {}
_FIND_SYM_MEMO_CAP = 32


def _find_symmetries_key(operator):
    """Content key for the find_symmetries memo: H's canonical terms (unique however H was
    written; complex coefficients enter as (re, im))."""
    terms = tuple((ops, tuple(int(s) for s in sites), complex(c).real, complex(c).imag)
                  for c, ops, sites in operator.terms())
    return (int(operator.num_sites), terms)


# The automorphism group is enumerated only up to this order: it must stay enumerable to be split
# into the abelian part and its cosets. nauty counts the group first, so a larger one (|Aut| = N!
# for field-only, empty or all-to-all H) costs nothing; such an H runs without spatial symmetry.
_AUT_ENUMERATION_CAP = 4096


def find_symmetries(operator: Operator, *, verbose: bool = True,
                    clique_budget: Optional[int] = None) -> Symmetries:
    """The spatial symmetries of ``operator``, split for the sector engine.

    The automorphisms of H's coloured interaction graph that commute with H are split into the
    largest normal abelian subgroup (``abelian``, the momenta) and one representative per coset
    of it (``residues``, the point group): the split ``Symmetry(spatial="auto")`` uses. A graph
    with more than 4096 automorphisms is not enumerated; the result then carries no spatial
    symmetry and says so in ``diagnostics``. A group whose co-group would exceed 128 elements is
    cut down to a subgroup, also with a diagnostic.

    The search is memoised on the operator's term content, so a ``spatial="auto"`` sweep that
    calls this repeatedly on the same H pays it once. ``verbose``: the cost notes are logged at
    Info, else at Debug (see :func:`qed.set_log_level`). ``clique_budget`` is accepted and
    ignored (deprecated): there is no clique search.
    """
    if clique_budget is not None:
        import warnings
        warnings.warn("find_symmetries(clique_budget=...) has no effect: the abelian part is the "
                      "largest normal abelian subgroup, found without a clique search",
                      DeprecationWarning, stacklevel=2)
    key = _find_symmetries_key(operator)
    if key is not None:
        hit = _FIND_SYM_MEMO.get(key)
        if hit is not None:
            return hit
    result = _find_symmetries_impl(operator, verbose=verbose)
    if key is not None:
        if len(_FIND_SYM_MEMO) >= _FIND_SYM_MEMO_CAP:
            _FIND_SYM_MEMO.pop(next(iter(_FIND_SYM_MEMO)))
        _FIND_SYM_MEMO[key] = result
    return result


def _find_symmetries_impl(operator: Operator, *, verbose: bool = True) -> Symmetries:
    num_sites = int(operator.num_sites)
    note = _log.INFO if verbose else _log.DEBUG
    if num_sites >= 20:
        _log.log(note, "[qed.find_symmetries] N=%d: searching the automorphism group", num_sites)
    vertex_weights, edges, triples = _operator_to_graph_records(operator)
    # Imported here so that pynauty is needed only by find_symmetries().
    try:
        from ._automorphism import automorphisms  # type: ignore
    except ImportError as e:  # pragma: no cover - environment-dependent
        raise ImportError(
            "find_symmetries() requires pynauty. Install it with `pip install pynauty` (or skip "
            "find_symmetries and pass your own permutations: qed.Symmetry(spatial=[...]))."
        ) from e
    from ._groups import spatial_split

    diagnostics: list[tuple[str, str]] = []
    identity = [list(range(num_sites))]
    autos, aut_order = automorphisms(vertex_weights, edges, triples, cap=_AUT_ENUMERATION_CAP)
    if autos is None:
        msg = (f"H's interaction graph has {aut_order:.6g} automorphisms, more than "
               f"{_AUT_ENUMERATION_CAP}: no spatial symmetry is used. Pass "
               "Symmetry(spatial=[...]) with generators of a subgroup to use one.")
        _log.log(note, "[qed.find_symmetries] %s", msg)
        diagnostics.append(("aut_capped", msg))
        autos = identity
    autos = _keep_hamiltonian_symmetries(operator, autos)
    if len(autos) <= 1:
        return Symmetries(abelian=identity, residues=[], diagnostics=diagnostics)
    abelian, residues, notes = spatial_split(autos)
    diagnostics.extend(notes)
    return Symmetries(abelian=[list(a) for a in abelian], residues=[list(r) for r in residues],
                      diagnostics=diagnostics)
