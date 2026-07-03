"""Stage 7a (SymmetryEngine v2): point-group star reduction + group
structure analysis.

The sector projector is abelian-only, but the automorphism pipeline
usually finds a LARGER, non-abelian group G (e.g. the ring's dihedral
D_N, a 2D torus's translations x mirrors). This module exploits the
non-abelian residue without any new projection machinery:

  For p in G outside the abelian subgroup A, conjugation permutes the
  A-irreps: chi_k -> chi_k^p with chi_k^p(t) = chi_k(p^-1 t p). The
  unitary U_p maps the k sector onto the k^p sector, so the two blocks
  are ISOSPECTRAL -- solve one representative per orbit ("star") and
  copy the spectrum to the partners. This composes with time-reversal
  pairing and the spin-flip (k, +/-) synthetic sectors inside the C++
  plan layer (``ed::symmetry::sector_orbit_canonical``).

``sector_star_maps`` computes the k -> k^p image tables that the C++
side consumes; ``describe_group`` reports the group structure
precisely (abelian invariant factors, residue relations, dihedral /
direct-product recognition).
"""
from __future__ import annotations

import cmath
from itertools import product as _iproduct
from typing import Any, Optional, Sequence

__all__ = ["sector_star_maps", "star_maps_from_info", "describe_group"]


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


def _enumerate_abelian(generators, orders):
    """Every element of the abelian group as {perm-tuple: exponent
    vector}. |A| = prod(orders) stays small for physical clusters."""
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
# star maps
# ---------------------------------------------------------------------------
def sector_star_maps(
    generators: Sequence[Sequence[int]],
    orders: Sequence[int],
    sectors: Sequence[dict],
    star_perms: Sequence[Sequence[int]],
) -> list[list[int]]:
    """k -> image irrep index under each residue permutation.

    For each ``p`` that normalises the abelian subgroup A, returns one
    map ``m`` with ``m[k]`` the index of the A-irrep obtained by
    conjugating chi_k with p (``-1`` where unresolved; a ``p`` that
    does not normalise A is dropped entirely). Either conjugation
    direction yields a valid isospectral bijection, so the convention
    is not physically significant.
    """
    if not generators or not star_perms:
        return []
    elems = _enumerate_abelian(generators, orders)
    gens = [tuple(g) for g in generators]

    # Irrep lookup: quantum numbers m_i <-> generator phases.
    def gen_phases(qn):
        return tuple(cmath.exp(2j * cmath.pi * m / o)
                     for m, o in zip(qn, orders))

    phase_index = {}
    for k, sec in enumerate(sectors):
        key = tuple(round(z.real, 9) + 1j * round(z.imag, 9)
                    for z in gen_phases(sec["quantum_numbers"]))
        phase_index[key] = k

    maps: list[list[int]] = []
    for p in star_perms:
        p = tuple(p)
        pinv = _inverse(p)
        conj_exps = []
        ok = True
        for g in gens:
            c = _compose(pinv, _compose(g, p))
            if c not in elems:
                ok = False          # p does not normalise A
                break
            conj_exps.append(elems[c])
        if not ok:
            continue
        m_out = []
        for sec in sectors:
            qn = sec["quantum_numbers"]
            key = []
            for exps in conj_exps:
                z = 1.0 + 0.0j
                for mj, aj, oj in zip(qn, exps, orders):
                    z *= cmath.exp(2j * cmath.pi * mj * aj / oj)
                key.append(round(z.real, 9) + 1j * round(z.imag, 9))
            m_out.append(phase_index.get(tuple(key), -1))
        maps.append(m_out)
    return maps


def star_maps_from_info(info: dict, star_perms) -> list[list[int]]:
    """Convenience wrapper over the ``group_from_generators`` /
    ``_normalize_symmetry_info`` dict shape."""
    if not info or not star_perms:
        return []
    return sector_star_maps(info.get("generators", []),
                            info.get("generator_orders", []),
                            info.get("sectors", []),
                            star_perms)


# ---------------------------------------------------------------------------
# group structure report
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
    a_order = 1
    for o in orders:
        a_order *= o
    shape = " x ".join(f"Z{o}" for o in orders) if orders else "trivial"
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
