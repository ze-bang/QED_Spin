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
    """(1/sqrt N) sum_j exp(-i Q.r_j) S^op_j as terms (q = (): the uniform sum)."""
    out = []
    for j in range(N):
        ph = sum(2 * math.pi * qa * ra / La for qa, ra, La in zip(q, coords[j], shape))
        out.append((complex(math.cos(-ph), math.sin(-ph)) / math.sqrt(N), ((op, j),)))
    return out



def adjoint(terms):
    """The terms of O^dagger: (A B C)^dagger = C^dagger B^dagger A^dagger, S+ <-> S-."""
    dag = {"+": "-", "-": "+"}
    return [(complex(c).conjugate(), tuple((dag.get(op, op), s) for op, s in reversed(ops)))
            for c, ops in terms]


def scale(terms):
    """s_H: the sum of |coefficient| over the terms (the scale the library's tolerances use)."""
    return float(sum(abs(c) for c, _ in terms))


# ---------------------------------------------------------------------------
# Site permutations acting on basis states (for symmetry-resolved references)
# ---------------------------------------------------------------------------

def compose(a, b):
    """(a o b)[i] = a[b[i]] -- b first, as in qed.symmetry.compose."""
    return tuple(a[x] for x in b)


def close_group(gens, N):
    """The group generated by the permutations ``gens`` (tuples), identity first."""
    ident = tuple(range(N))
    seen = {ident: None}
    out, frontier = [ident], [ident]
    gens = [tuple(g) for g in gens]
    while frontier:
        nxt = []
        for x in frontier:
            for g in gens:
                y = compose(g, x)
                if y not in seen:
                    seen[y] = None
                    out.append(y)
                    nxt.append(y)
        frontier = nxt
    return out


def state_map(perm, N):
    """img[s] = U_g|s>: bit i of U_g|s> is bit perm[i] of s (qed's convention: qed.symmetry.momentum_of
    and applyPermutation). A vector transforms as (U_g v)[img] = v."""
    s = np.arange(1 << N, dtype=np.int64)
    out = np.zeros_like(s)
    for i, p in enumerate(perm):
        out |= ((s >> int(p)) & 1) << i
    return out


def apply_perm(img, v):
    """U_g v for the state map ``img`` of g (rows of a 2D v)."""
    w = np.empty_like(v)
    w[img] = v
    return w


def joint_eigenspace(N, elems, chis, tol=1e-10):
    """Orthonormal basis (columns, full 2^N basis) of {v : U_a v = chi_a v for every a}, for an
    abelian group listed whole (``elems``) with a one-dimensional character ``chis`` on it:
    the columns P|s> of the projector P = (1/|A|) sum_a conj(chi_a) U_a over orbit
    representatives s, normalised (different orbits are orthogonal, one orbit gives one
    vector up to a factor)."""
    dim = 1 << N
    imgs = [state_map(a, N) for a in elems]
    orbit_min = np.min(np.stack(imgs), axis=0)
    reps = np.flatnonzero(orbit_min == np.arange(dim))
    cols = np.zeros((dim, len(reps)), complex)
    r = np.arange(len(reps))
    for img, chi in zip(imgs, chis):
        np.add.at(cols, (img[reps], r), np.conj(chi) / len(elems))
    norms = np.linalg.norm(cols, axis=0)
    keep = norms > tol
    return cols[:, keep] / norms[keep]


def records(terms):
    """Terms written as (ops, sites, coeff) records -- ("+", "-"), (i, j), c -- in this vocabulary."""
    return [(complex(c), tuple(zip(ops, sites))) for ops, sites, c in terms]


def terms_of(op):
    """A library Operator's canonical terms (Operator.terms(): (coeff, ops, sites)) in this vocabulary."""
    out = []
    for c, ops, sites in op.terms():
        out += _expand(c, tuple(zip(ops, sites)))
    return out
