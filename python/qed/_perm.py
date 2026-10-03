"""Site permutations: the one convention of the package.

A permutation p of the n sites is a tuple read as the engine reads it: site i of the image
carries the spin site p[i] carried ((U_p s)_i = s_{p[i]}), so an operator on site k moves to
p^-1(k). Composition is ``compose(a, b)[i] = a[b[i]]`` (b first), as in ``ed::sym`` and
:mod:`qed.symmetry`.
"""

from __future__ import annotations

import math
from typing import Iterable, Optional, Sequence

Perm = tuple[int, ...]

CLOSURE_CAP = 4096  # a group the package enumerates stays at most this large


def as_perm(p: Iterable[int]) -> Perm:
    return tuple(int(x) for x in p)


def identity(n: int) -> Perm:
    return tuple(range(n))


def compose(a: Sequence[int], b: Sequence[int]) -> Perm:
    """(a o b)[i] = a[b[i]]."""
    return tuple(a[b[i]] for i in range(len(b)))


def inverse(p: Sequence[int]) -> Perm:
    q = [0] * len(p)
    for i, x in enumerate(p):
        q[x] = i
    return tuple(q)


def order(p: Sequence[int]) -> int:
    """The least m > 0 with p^m the identity: the lcm of the cycle lengths."""
    seen = [False] * len(p)
    m = 1
    for s in range(len(p)):
        if seen[s]:
            continue
        length, t = 0, s
        while not seen[t]:
            seen[t] = True
            t = p[t]
            length += 1
        m = m * length // math.gcd(m, length)
    return m


def is_permutation(p: Iterable[int], n: int) -> bool:
    return sorted(int(x) for x in p) == list(range(n))


def close_group(gens, cap: int = CLOSURE_CAP) -> Optional[list[Perm]]:
    """The group the permutations generate, sorted (the identity first); None without generators
    or when the group has more than ``cap`` elements."""
    gens = [tuple(g) for g in gens]
    if not gens:
        return None
    ident = identity(len(gens[0]))
    elems = {ident}
    frontier = [ident]
    while frontier:
        nxt = []
        for e in frontier:
            for g in gens:
                c = compose(e, g)
                if c not in elems:
                    if len(elems) >= cap:
                        return None
                    elems.add(c)
                    nxt.append(c)
        frontier = nxt
    return sorted(elems)
