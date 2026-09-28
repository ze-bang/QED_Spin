"""Builders for :class:`qed._core.MaskedOperator` observables.

A MaskedOperator is any sum of products of S+, S-, S^z, S^x, S^y and the projectors
|up><up|, |dn><dn| on any sites, stored exactly (engine convention: bit set = spin
down, so S+ acts on a set bit). They feed
``qed._core.little_group_block_observables``, which needs no symmetry of the
operator: a single bond, plaquette or string is projected onto the component that
connects the two symmetry sectors.

Everything here is a literal product; nothing is normalised behind your back:

* ``pm(i, j)``        = S+_i S-_j + S-_i S+_j
* ``current(i, j)``   = i (S+_i S-_j - S-_i S+_j)
* ``zz(i, j)``        = S^z_i S^z_j
* ``sz(sites)``       = sum_i S^z_i
* ``z_string(sites)`` = prod_i (2 S^z_i)
* ``ring(sites)``     = S+ S- S+ S- ... around ``sites`` + h.c. (even length)
* ``translate_sum(op, perms)`` = sum_g U_g op U_g^dagger
"""
from __future__ import annotations

from typing import Iterable, Sequence

from . import _core

MaskedOperator = _core.MaskedOperator

__all__ = ["MaskedOperator", "product", "identity", "pm", "current", "zz", "sz",
           "z_string", "ring", "translate_sum"]


def product(n: int, ops: str, sites: Sequence[int], coeff: complex = 1.0) -> MaskedOperator:
    """coeff * prod_k O_k(site_k), the last factor acting first; ops over '+-zxyudI'."""
    return MaskedOperator.product(n, ops, [int(s) for s in sites], complex(coeff))


def identity(n: int) -> MaskedOperator:
    return product(n, "I", [0])


def pm(n: int, i: int, j: int) -> MaskedOperator:
    return product(n, "+-", [i, j]) + product(n, "-+", [i, j])


def current(n: int, i: int, j: int) -> MaskedOperator:
    return product(n, "+-", [i, j], 1j) + product(n, "-+", [i, j], -1j)


def zz(n: int, i: int, j: int) -> MaskedOperator:
    return product(n, "zz", [i, j])


def sz(n: int, sites: Iterable[int]) -> MaskedOperator:
    out = MaskedOperator(n)
    for i in sites:
        out = out + product(n, "z", [i])
    return out


def z_string(n: int, sites: Sequence[int]) -> MaskedOperator:
    sites = list(sites)
    return product(n, "z" * len(sites), sites, 2.0 ** len(sites))


def ring(n: int, sites: Sequence[int]) -> MaskedOperator:
    sites = list(sites)
    if len(sites) % 2:
        raise ValueError("ring: an even number of sites is needed to conserve S^z")
    r = product(n, "+-" * (len(sites) // 2), sites)
    return r + r.dagger()


def translate_sum(op: MaskedOperator, perms: Iterable[Sequence[int]]) -> MaskedOperator:
    """sum_g U_g op U_g^dagger over the given permutations (new bit i = old bit perm[i])."""
    out = MaskedOperator(op.n_sites)
    for p in perms:
        out = out + op.image([int(x) for x in p], 0)
    return out
