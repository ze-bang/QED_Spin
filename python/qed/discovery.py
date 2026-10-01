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


def _operator_to_graph_records(
    operator: Operator,
) -> tuple[dict[int, tuple], list[dict[str, Any]], list[tuple]]:
    """Build (vertex_weights, edges, triples) records the
    ``automorphism_finder`` routines consume."""
    num_sites = int(operator.num_sites)

    # One-body terms: each site's terms merged by operator, as a sorted tuple of
    # (op_type, real, imag) -- every term on the site colours it, not only the last one.
    onsite: dict[int, dict[int, complex]] = {i: {} for i in range(num_sites)}
    for op_type, site, coeff in operator.iter_one_body_terms():
        d = onsite[int(site)]
        d[int(op_type)] = d.get(int(op_type), 0j) + complex(coeff)
    vertex_weights = {i: tuple(sorted((op, round(c.real, 8), round(c.imag, 8))
                                      for op, c in d.items() if abs(c) > 1e-12))
                      for i, d in onsite.items()}

    # Two-body terms as edges: list of dicts.
    edges: list[dict[str, Any]] = []
    for op1, s1, op2, s2, coeff in operator.iter_two_body_terms():
        c = complex(coeff)
        edges.append({
            "vertex1": int(s1),
            "vertex2": int(s2),
            "type1": int(op1),
            "type2": int(op2),
            "weight": (float(c.real), float(c.imag)),
        })
    triples = [((int(s1), int(s2), int(s3)), (int(o1), int(o2), int(o3)), complex(c))
               for o1, s1, o2, s2, o3, s3, c in operator.iter_three_body_terms()]
    return vertex_weights, edges, triples


def _run_full_automorphism_pipeline(
    vertex_weights: dict[int, tuple],
    edges: list[dict[str, Any]],
    triples: list[tuple],
    construct_colored_graph,
    autgrp,
    cap: int,
) -> tuple[Optional[list[Permutation]], float]:
    """Run nauty: ``(permutations, |Aut|)``, the automorphisms of the coloured interaction graph
    on the original vertices and nauty's group order (a supergroup of H's symmetries: the exact
    term check, :func:`_keep_hamiltonian_symmetries`, follows). Above ``cap`` nothing is
    enumerated and the list is None (|Aut| reaches N! for field-only, empty or all-to-all H)."""
    graph, vertex_colors, idx_to_vid, vid_to_idx = construct_colored_graph(
        vertex_weights, edges, triples
    )
    aut = autgrp(graph)
    # One auxiliary vertex per interacting pair, so the group of the expanded graph is that
    # of the original vertices: nauty's count (grpsize1 * 10^grpsize2) is |Aut| itself.
    size = float(aut[1]) * 10.0 ** int(aut[2])
    if size > cap:
        return None, size
    from ._groups import close_group
    gens = [tuple(int(x) for x in g) for g in aut[0]]
    expanded = close_group(gens, cap=cap) if gens else [tuple(range(graph.number_of_vertices))]

    # Project back to original-vertex permutations and dedup.
    seen: set[tuple[int, ...]] = set()
    autos: list[Permutation] = []
    for perm in expanded:
        proj = [idx_to_vid[perm[vid_to_idx[vid]]] for vid in idx_to_vid]
        key = tuple(proj)
        if key not in seen:
            seen.add(key)
            autos.append(proj)

    return autos, size


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
        from ._groups import close_group
        A = close_group(self.abelian) if self.abelian else None
        a, r = (len(A) if A is not None else len(self.abelian)), len(self.residues)
        lines = [f"{a * (r + 1)} spatial symmetries: an abelian part of {a} (the momenta) and "
                 f"{r} residue(s) (the point group)"]
        lines += [f"{code}: {msg}" for code, msg in self.diagnostics]
        return "\n".join(lines)


_FIND_SYM_MEMO: dict = {}
_FIND_SYM_MEMO_CAP = 32


def _find_symmetries_key(operator):
    """Content key for the find_symmetries memo, or None to skip caching (any part that cannot
    be hashed => compute afresh, never cache wrong). Complex coefficients enter as (re, im):
    complex numbers do not order, and a sort over them failed for every H that has two terms
    on the same (op, site) -- nothing was ever cached for a J1-J2 or field H."""
    def flat(t):
        return tuple((x.real, x.imag) if isinstance(x, complex) else x for x in t)
    try:
        terms = tuple(sorted(flat(t) for t in operator.transform_tuples()))
        three = tuple(sorted(flat(t) for t in operator.iter_three_body_terms()))
        return (int(operator.num_sites), terms, three)
    except Exception:
        return None


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
        from ._automorphism import construct_colored_graph  # type: ignore
        from pynauty import autgrp  # type: ignore
    except ImportError as e:  # pragma: no cover - environment-dependent
        raise ImportError(
            "find_symmetries() requires pynauty. Install it with `pip install pynauty` (or skip "
            "find_symmetries and pass your own permutations: qed.Symmetry(spatial=[...]))."
        ) from e
    from ._groups import spatial_split

    diagnostics: list[tuple[str, str]] = []
    identity = [list(range(num_sites))]
    automorphisms, aut_order = _run_full_automorphism_pipeline(
        vertex_weights, edges, triples, construct_colored_graph, autgrp, cap=_AUT_ENUMERATION_CAP)
    if automorphisms is None:
        msg = (f"H's interaction graph has {aut_order:.6g} automorphisms, more than "
               f"{_AUT_ENUMERATION_CAP}: no spatial symmetry is used. Pass "
               "Symmetry(spatial=[...]) with generators of a subgroup to use one.")
        _log.log(note, "[qed.find_symmetries] %s", msg)
        diagnostics.append(("aut_capped", msg))
        automorphisms = identity
    automorphisms = _keep_hamiltonian_symmetries(operator, automorphisms)
    if len(automorphisms) <= 1:
        return Symmetries(abelian=identity, residues=[], diagnostics=diagnostics)
    abelian, residues, notes = spatial_split(automorphisms)
    diagnostics.extend(notes)
    return Symmetries(abelian=[list(a) for a in abelian], residues=[list(r) for r in residues],
                      diagnostics=diagnostics)
