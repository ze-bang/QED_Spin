"""Permutation groups: closure, a maximal abelian subgroup, and the split of a symmetry
group into its abelian part (the momenta) and coset representatives (the point group).

A greedy commuting subgroup is always safe: a smaller abelian part only means larger
stars (less momentum reduction), never wrong physics.
"""
from __future__ import annotations

__all__ = ["close_group", "greedy_maximal_abelian", "split_nonabelian"]

_GROUP_CLOSURE_CAP = 4096   # A (and the closed full group) must stay enumerable


def _compose(g, e):
    """U-composition convention (matches irreps.cpp): (g.e)[i] = e[g[i]]."""
    return tuple(e[g[i]] for i in range(len(g)))


def close_group(gens, cap=_GROUP_CLOSURE_CAP):
    """BFS closure of a permutation set. None when the group exceeds cap."""
    gens = [tuple(g) for g in gens]
    if not gens:
        return None
    n = len(gens[0])
    ident = tuple(range(n))
    elems = {ident}
    frontier = [ident]
    while frontier:
        nxt = []
        for e in frontier:
            for g in gens:
                c = _compose(g, e)
                if c not in elems:
                    if len(elems) >= cap:
                        return None
                    elems.add(c)
                    nxt.append(c)
        frontier = nxt
    return sorted(elems)


def _commute(a, b):
    return _compose(a, b) == _compose(b, a)


def greedy_maximal_abelian(elements, cap=_GROUP_CLOSURE_CAP):
    """Greedy maximal commuting subgroup of an already-enumerated
    permutation group.

    Seed with the HIGHEST-order elements first: long cycles (translations)
    build a large cyclic core, where naive sorted order tends to lock in an
    early involution (e.g. a reflection) and end up with a small Klein-type
    subgroup. Any commuting closure is *valid* -- smaller A only folds
    less -- this ordering just maximizes the reduction.

    Returns the CLOSED abelian subgroup as a sorted list of tuples (always
    contains the identity; ``[]`` for empty input). Shared by
    ``split_nonabelian``'s raw-list branch and ``find_symmetries``' clique
    budget (``clique_budget=``), which uses it to sidestep the NP-hard
    maximum-clique search on very large automorphism groups.
    """
    G = [tuple(e) for e in elements]
    if not G:
        return []
    n = len(G[0])

    def _order(e):
        k, c = 1, e
        ident = tuple(range(len(e)))
        while c != ident:
            c = _compose(e, c)
            k += 1
        return k

    A = [tuple(range(n))]
    Aset = set(A)
    for e in sorted(G, key=lambda e: (-_order(e), e)):
        if e in Aset:
            continue
        if all(_commute(e, a) for a in A):
            closed = close_group(A + [e])
            if closed is not None and len(closed) <= cap:
                A = [tuple(a) for a in closed]
                Aset = set(A)
    return sorted(Aset)


def split_nonabelian(symmetry_or_gens):
    """``(abelian_elements, residue_perms)`` for the factorized engine,
    or a ``str`` decline reason.

    * ``GeneratorSet``-like input (``generators`` + ``star_perms``): the
      abelian clique is closed; the retained residue is the coset set --
      the original ``_little_group_parts`` behaviour.
    * explicit permutation list: the FULL group is closed, a maximal
      abelian subgroup is chosen greedily (any commuting closure is
      valid -- smaller A just folds less), and one representative per
      A-coset of the remainder becomes the residue set.
    """
    gens = getattr(symmetry_or_gens, "generators", None)
    if gens is not None:
        star = list(getattr(symmetry_or_gens, "star_perms", None) or [])
        if not gens:
            return "the symmetry has no spatial generators"
        A = close_group([list(g) for g in gens])
        if A is None:
            return (f"the abelian group exceeds the {_GROUP_CLOSURE_CAP}-"
                    "element closure cap")
        if not star:
            return ("no point-group residue is retained on the symmetry "
                    "(symmetry='auto' or a find_symmetries GeneratorSet "
                    "carry one; pure-abelian input has nothing to project)")
        # Dedup the retained residues modulo A. find_symmetries carries the
        # FULL non-clique content as star_perms (e.g. all 30 elements of the
        # C2*A coset on a 5x6 torus; 396 on the 6x6 C6v torus). Same-coset
        # residues act as proportional monomials -- the engine validates and
        # discards each duplicate, paying the O(dim) monomial build per
        # element for zero extra reduction (measured 4x total at 30 sites).
        # One representative per A-coset is the complete input.
        Aset = {tuple(a) for a in A}
        residues, covered = [], set(Aset)
        for p in star:
            tp = tuple(p)
            if tp in covered:
                continue
            residues.append(list(tp))
            covered.update(_compose(a, tp) for a in Aset)
        if not residues:
            return ("every retained residue lies in the abelian group -- "
                    "nothing to project beyond the momentum sectors")
        return ([list(e) for e in A], residues)

    # Explicit raw permutation list.
    perms = [tuple(p) for p in (symmetry_or_gens or [])]
    if not perms:
        return "no generators given"
    G = close_group(perms)
    if G is None:
        return (f"the closed group exceeds the {_GROUP_CLOSURE_CAP}-element "
                "cap -- pass a GeneratorSet from find_symmetries instead")
    # Greedy maximal commuting subgroup (the closure of pairwise-commuting
    # elements is abelian by construction) -- see greedy_maximal_abelian.
    A = greedy_maximal_abelian(G)
    Aset = set(A)
    if len(A) <= 1:
        return "no non-trivial abelian subgroup found in the closed group"
    # One representative per A-coset of the remainder.
    residues, covered = [], set(Aset)
    for e in G:
        if e in covered:
            continue
        residues.append(list(e))
        covered.update(_compose(a, e) for a in Aset)
    if not residues:
        return ("the input group is abelian -- nothing to project beyond "
                "the momentum sectors (use the abelian rep lane)")
    return ([list(a) for a in sorted(Aset)], residues)


