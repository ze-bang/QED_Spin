"""Group structure of a symmetry found by discovery: the order of an abelian group from
its generators, and a readable description (abelian invariant factors, the residues,
dihedral / direct-product recognition).
"""
from __future__ import annotations

from itertools import product as _iproduct
from typing import Sequence



# ---------------------------------------------------------------------------
# permutation helpers (perm[i] = image of site i)
# ---------------------------------------------------------------------------
def _compose(a, b):
    """(a o b)[i] = a[b[i]] -- apply b first."""
    return tuple(a[b[i]] for i in range(len(a)))


def _inverse(p):
    inv = [0] * len(p)
    for i, j in enumerate(p):
        inv[j] = i
    return tuple(inv)


def abelian_order(generators, orders) -> int:
    """TRUE |A|: the number of DISTINCT permutations the generators span.

    Not ``prod(orders)``. The C++ minimal-generator decomposition returns a
    generating set that is minimal in COUNT, not relation-free -- on a 4x4
    torus it hands back three order-4 generators spanning a group of order
    16, where prod(orders) claims 64. Reporting the product overstates the
    group by the redundancy among the generators, which reads downstream as
    "my blocks are 4x bigger than the symmetry says they should be" when in
    fact dim/|A| is exactly right. Enumeration dedups by construction and
    stays cheap: |A| is small for physical clusters.
    """
    if not generators:
        return 1
    return len(_enumerate_abelian(generators, orders))


def _enumerate_abelian(generators, orders):
    """Every element of the abelian group as {perm-tuple: exponent
    vector}. Deduped by perm, so ``len()`` is the TRUE |A| (which may be
    smaller than prod(orders) -- see :func:`abelian_order`)."""
    n = len(generators[0])
    gens = [tuple(g) for g in generators]
    # powers[i][a] = g_i^a
    powers = []
    for g, o in zip(gens, orders):
        ps = [tuple(range(n))]
        for _ in range(o - 1):
            ps.append(_compose(g, ps[-1]))
        powers.append(ps)
    elems = {}
    for exps in _iproduct(*[range(o) for o in orders]):
        e = tuple(range(n))
        for i, a in enumerate(exps):
            e = _compose(powers[i][a], e)
        elems.setdefault(e, exps)
    return elems


# ---------------------------------------------------------------------------
# group description
# ---------------------------------------------------------------------------
def _perm_order(p):
    p = tuple(p)
    n = len(p)
    e = tuple(range(n))
    q, o = p, 1
    while q != e:
        q = _compose(p, q)
        o += 1
    return o


def _decompose_in(elems, perm):
    return elems.get(tuple(perm))


def describe_group(
    generators: Sequence[Sequence[int]],
    orders: Sequence[int],
    star_perms: Sequence[Sequence[int]] = (),
    name: str = "spatial group",
) -> str:
    """Precise, human-readable structure of the detected group.

    Reports the abelian projector subgroup A (invariant-factor shape,
    generator permutations with orders), and -- when a non-abelian
    residue is present -- each coset representative with its order and
    its conjugation relations on the generators (``p g p^-1 = ...``),
    plus recognition of the common cases (dihedral extension, direct
    product with A).
    """
    lines = []
    gens = [tuple(g) for g in generators]
    # TRUE order, not prod(orders): the generators need not be independent.
    a_order = abelian_order(gens, orders)
    shape = " x ".join(f"Z{o}" for o in orders) if orders else "trivial"
    prod = 1
    for o in orders:
        prod *= o
    if prod != a_order:
        # Say so loudly: a reader who takes prod(orders) as |A| will expect
        # blocks prod/|A| times smaller than the engine can deliver, and
        # conclude the engine is leaving reduction on the table.
        shape += (f"  [generators are NOT independent: <a0..a{len(orders)-1}> "
                  f"spans {a_order}, not prod(orders)={prod}]")
    lines.append(f"{name}: abelian projector subgroup A = {shape} "
                 f"(|A| = {a_order})")
    for i, (g, o) in enumerate(zip(gens, orders)):
        lines.append(f"  a{i} (order {o}): {list(g)}")

    live = []
    if star_perms:
        elems = _enumerate_abelian(generators, orders) if gens else {}
        for p in star_perms:
            p = tuple(p)
            if _decompose_in(elems, p) is not None:
                continue            # inside A: not a residue element
            live.append(p)
        if live:
            lines.append(f"  non-abelian residue: |G| = "
                         f"{a_order * 2 if len(live) == a_order else '>= ' + str(a_order + len(live))}"
                         f" total automorphisms retained "
                         f"({len(live)} coset representatives)")
        all_invert = True           # every residue p inverts every a_i
        shown = 0
        for p in live:
            pinv = _inverse(p)
            rels = []
            for i, g in enumerate(gens):
                c = _compose(pinv, _compose(g, p))
                exps = _decompose_in(elems, c)
                if exps is None:
                    rels.append(f"p a{i} p^-1 not in A")
                    all_invert = False
                    continue
                rel = " ".join(f"a{j}^{a}" for j, a in enumerate(exps) if a)
                rels.append(f"p a{i} p^-1 = {rel or 'e'}")
                inverted = tuple((orders[j] - 1) % orders[j] if j == i else 0
                                 for j in range(len(gens)))
                if exps != inverted:
                    all_invert = False
            if shown < 4:
                lines.append(f"  p (order {_perm_order(p)}): {list(p)}")
                for r in rels:
                    lines.append(f"      {r}")
                shown += 1
        if live and len(gens) == 1 and all_invert:
            lines.append(f"  recognised: G = D{orders[0]} "
                         f"(dihedral; p inverts the Z{orders[0]} generator)")
        elif live:
            trivial_action = all(
                _decompose_in(elems,
                              _compose(_inverse(p), _compose(g, p)))
                == tuple(1 if j == i else 0 for j in range(len(gens)))
                for p in live for i, g in enumerate(gens))
            if trivial_action:
                lines.append("  recognised: residue commutes with A "
                             "(direct-product extension)")
    if not live:
        lines.append("  no non-abelian residue: G = A (abelian)")
    return "\n".join(lines)
