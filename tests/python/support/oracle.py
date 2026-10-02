"""The one dense oracle of the test suites: term lists and the numpy matrices they define.

A term is (coeff, ((op, site), ...)) with op in "+-zudxyI": a product of single-site operators
(rightmost acting first); x and y are expanded into S+ / S-, u and d are the projectors on spin
up and down, I is the identity. One bit per site, a set bit is Sz = +1/2 (n_up counts set bits),
S+ acts on a clear bit and sets it -- the library's conventions. The matrices are built from the
term lists alone, never through the library's matvec, so they are an independent reference for
the grid, golden and regress suites; terms_of() reads an operator that exists only as a library
Operator through its canonical terms (Operator.terms()).
"""
from __future__ import annotations

import math

import numpy as np

Term = tuple


def _expand(coeff, ops):
    """Expand Cartesian x/y factors into S+/S- products."""
    out = [(complex(coeff), ())]
    for op, site in ops:
        if op == "x":
            parts = [(0.5, "+"), (0.5, "-")]
        elif op == "y":
            parts = [(-0.5j, "+"), (0.5j, "-")]
        else:
            parts = [(1.0, op)]
        out = [(c * pc, prev + ((po, site),)) for c, prev in out for pc, po in parts]
    return out


def dot(i, j, J=1.0, jz=None):
    """J S_i.S_j (or XXZ with jz) as expanded terms."""
    jz = J if jz is None else jz
    return [(0.5 * J, (("+", i), ("-", j))), (0.5 * J, (("-", i), ("+", j))),
            (jz, (("z", i), ("z", j)))]


def ring(a, b, c, d, K):
    """K (P + P^dagger), P the cyclic exchange of the spins on a -> b -> c -> d -> a, as its
    16 + 16 matrix elements: |P s><s| = prod_i |(Ps)_i><s_i| with |up><up| = u, |dn><dn| = d,
    |up><dn| = S+, |dn><up| = S-."""
    sites = (a, b, c, d)
    one = {(1, 1): "u", (0, 0): "d", (1, 0): "+", (0, 1): "-"}   # (new, old), 1 = up
    terms = []
    for s in range(16):
        old = [(s >> i) & 1 for i in range(4)]
        new = [old[(i - 1) % 4] for i in range(4)]   # site i receives the spin of site i - 1
        terms.append((K, tuple((one[(n, o)], x) for n, o, x in zip(new, old, sites))))
        terms.append((K, tuple((one[(o, n)], x) for n, o, x in zip(new, old, sites))))
    return terms


def triple(i, j, k, chi):
    """chi S_i.(S_j x S_k), SU(2) invariant and time-reversal odd."""
    terms = []
    for a, b, c, s in (("x", "y", "z", 1), ("y", "z", "x", 1), ("z", "x", "y", 1),
                       ("x", "z", "y", -1), ("y", "x", "z", -1), ("z", "y", "x", -1)):
        terms += _expand(s * chi, ((a, i), (b, j), (c, k)))
    return terms


def _apply(ops, s):
    """Apply a product of single-site ops (rightmost first) to basis state s."""
    amp = 1.0
    for op, site in reversed(ops):
        if op == "I":
            continue
        bit = (s >> site) & 1
        if op == "z":
            amp *= 0.5 if bit == 1 else -0.5
        elif op in "ud":
            if bit != (op == "u"):
                return 0.0, s
        elif op == "+":
            if bit == 1:
                return 0.0, s
            s ^= 1 << site
        else:
            if bit == 0:
                return 0.0, s
            s ^= 1 << site
    return amp, s


def dense(terms, N):
    dim = 1 << N
    M = np.zeros((dim, dim), dtype=complex)
    for c, ops in terms:
        for s in range(dim):
            a, t = _apply(ops, s)
            if a:
                M[t, s] += c * a
    return M


def sparse(terms, N):
    """The same matrix as :func:`dense`, stored sparse (an observable has few terms)."""
    import scipy.sparse as sp

    rows, cols, vals = [], [], []
    for c, ops in terms:
        for s in range(1 << N):
            a, t = _apply(ops, s)
            if a:
                rows.append(t)
                cols.append(s)
                vals.append(c * a)
    return sp.csr_matrix((np.asarray(vals, complex), (rows, cols)), shape=(1 << N, 1 << N))


def fourier(N, coords, shape, q, op):
    """(1/sqrt N) sum_j exp(-i Q.r_j) S^op_j as terms."""
    out = []
    for j in range(N):
        ph = sum(2 * math.pi * qa * ra / La for qa, ra, La in zip(q, coords[j], shape))
        out.append((complex(math.cos(-ph), math.sin(-ph)) / math.sqrt(N), ((op, j),)))
    return out



def records(terms):
    """Terms written as (ops, sites, coeff) records -- ("+", "-"), (i, j), c -- in this vocabulary."""
    return [(complex(c), tuple(zip(ops, sites))) for ops, sites, c in terms]


def terms_of(op):
    """A library Operator's canonical terms (Operator.terms(): (coeff, ops, sites)) in this vocabulary."""
    out = []
    for c, ops, sites in op.terms():
        out += _expand(c, tuple(zip(ops, sites)))
    return out
