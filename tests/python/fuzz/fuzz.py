#!/usr/bin/env python3
"""Randomized differential fuzzing of the qed public API against a dense numpy oracle.

    python tests/python/fuzz/fuzz.py --seed 1 --cases 150 --device cpu --out DIR --budget-seconds 540 --strict

Each case draws a small spin-1/2 model (N <= 12) from a family whose symmetry content is
known and then VERIFIED numerically (U(1), Sz parity, spin flip, complex conjugation,
SU(2), every candidate site permutation), a qed.Symmetry request valid for it (or, for a
small fraction, a deliberately invalid one), and one task. The task runs through the
public verbs; the reference is built independently with numpy/scipy from the same term
list in the full 2^N basis (no library matvec). One JSON line per case goes to
DIR/fuzz_<device>_<seed>.jsonl; a summary is printed at the end.

Cases run in a spawned worker process, so a segfault or a hang in the C++ engine costs one
case, not the run: the worker has a soft signal.alarm and the parent a hard kill.
--inprocess disables the isolation (debug). --replay SEED-INDEX reruns one case in-process
and prints its record. --budget-seconds stops drawing new cases once spent.

Statuses: pass | wrong | crash | refused | timeout | invalid_ok | invalid_bad, plus
harness_error (the reference or comparison code itself failed -- a harness problem, not a
library finding) and skip (not applicable after the library resolved its groups).
A non-pass record that matches an entry of known.json (an accepted failure tied to an OPEN
ledger id of tests/python/regress/manifest.json) carries that id in known_id. Under --strict the
exit code is 1 when any non-pass record is unexplained (or a harness_error occurred) and 2
when known.json names a ledger id that is not open.

Only numpy, scipy and qed are used. Conventions follow the library (0.6): one bit per site,
bit i of a state is site i, a SET bit is spin UP (S^z = +1/2), S+ sets a clear bit and S-
clears a set one; Symmetry(sz=n) selects the sector with n up spins (S^z = n - N/2); a spin-S
tower is solved at its S^z = +S member (n_up = N/2 + S); permutation p acts on states as
bit i of U|s> = bit p[i] of |s>, and a momentum theta means T|psi> = exp(-2 pi i theta)|psi>.
"""
from __future__ import annotations

import argparse
import cmath
import copy
import json
import math
import multiprocessing as mp
import os
import re
import signal
import sys
import tempfile
import time
import traceback
from fractions import Fraction

import numpy as np

_trapz = getattr(np, "trapezoid", None) or np.trapz

STATUSES = ("pass", "wrong", "crash", "refused", "timeout", "invalid_ok", "invalid_bad",
            "harness_error", "skip")
OK_STATUSES = ("pass", "invalid_ok", "skip")
# A device refusal of a block kind that has no device kernel yet (classify): documented, a pass.
DOCUMENTED_NO_KERNEL = re.compile(r"is a sector of an irrep of dimension > 1, whose device kernel")
HERE = os.path.dirname(os.path.abspath(__file__))

# =============================================================================
# Terms: (coeff, ((op, site), ...)), op in "+-zud" (u/d: |up><up|, |dn><dn|); x/y are expanded
# =============================================================================


def _expand(coeff, ops):
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
    jz = J if jz is None else jz
    return [(0.5 * J, (("+", i), ("-", j))), (0.5 * J, (("-", i), ("+", j))),
            (complex(jz), (("z", i), ("z", j)))]


def xyz(i, j, jx, jy, jz):
    return (_expand(jx, (("x", i), ("x", j))) + _expand(jy, (("y", i), ("y", j)))
            + [(complex(jz), (("z", i), ("z", j)))])


def hop(i, j, t, phi, jz):
    e = cmath.exp(1j * phi)
    return [(0.5 * t * e, (("+", i), ("-", j))), (0.5 * t * e.conjugate(), (("-", i), ("+", j))),
            (complex(jz), (("z", i), ("z", j)))]


def dm_z(i, j, D):
    """D z.(S_i x S_j) = D (Sx_i Sy_j - Sy_i Sx_j): complex, U(1), breaks flip and SU(2)."""
    return _expand(D, (("x", i), ("y", j))) + _expand(-D, (("y", i), ("x", j)))


def triple(i, j, k, chi):
    """chi S_i.(S_j x S_k): SU(2) invariant, odd under complex conjugation."""
    terms = []
    for a, b, c, s in (("x", "y", "z", 1), ("y", "z", "x", 1), ("z", "x", "y", 1),
                       ("x", "z", "y", -1), ("y", "x", "z", -1), ("z", "y", "x", -1)):
        terms += _expand(s * chi, ((a, i), (b, j), (c, k)))
    return terms


def ring4(p, K):
    """K (P + P^-1) for the cyclic exchange P of the four spins p = (a, b, c, d): site p[k]
    takes the spin of site p[k-1]. Written as |P s><s| = prod_k |(P s)_k><s_k| (four-site
    terms; SU(2) invariant, real)."""
    one = {(1, 1): "u", (0, 0): "d", (1, 0): "+", (0, 1): "-"}      # (new, old), 1 = up
    out = []
    for s in range(16):
        old = [(s >> k) & 1 for k in range(4)]
        new = [old[(k - 1) % 4] for k in range(4)]
        out.append((complex(K), tuple((one[(n, o)], p[k]) for k, (n, o) in enumerate(zip(new, old)))))
        out.append((complex(K), tuple((one[(o, n)], p[k]) for k, (n, o) in enumerate(zip(new, old)))))
    return out


def field(i, hx=0.0, hy=0.0, hz=0.0):
    t = []
    if hx:
        t += _expand(hx, (("x", i),))
    if hy:
        t += _expand(hy, (("y", i),))
    if hz:
        t.append((complex(hz), (("z", i),)))
    return t


def merge(terms):
    """Merge records of the same operator (sites are distinct inside a term, so the factors
    commute and are sorted by site)."""
    acc = {}
    for c, ops in terms:
        key = tuple(sorted(ops, key=lambda o: o[1]))
        acc[key] = acc.get(key, 0j) + complex(c)
    return [(c, ops) for ops, c in acc.items() if abs(c) > 1e-13]


# =============================================================================
# Sparse full-space operators (independent of the library)
# =============================================================================


def popcounts(N):
    s = np.arange(1 << N, dtype=np.int64)
    p = np.zeros_like(s)
    for i in range(N):
        p += (s >> i) & 1
    return p


def sparse_op(terms, N):
    """Sum of the terms in the 2^N basis; a set bit is spin up. The rightmost factor acts
    first, so same-site products are exact."""
    import scipy.sparse as sp
    dim = 1 << N
    states = np.arange(dim, dtype=np.int64)
    rows, cols, vals = [], [], []
    for c, ops in terms:
        if abs(c) < 1e-15:
            continue
        s = states.copy()
        amp = np.full(dim, complex(c))
        ok = np.ones(dim, bool)
        for op, site in reversed(ops):
            bit = (s >> site) & 1
            if op == "z":
                amp = amp * np.where(bit == 1, 0.5, -0.5)
            elif op == "+":
                ok &= bit == 0
                s = s ^ (1 << site)
            elif op == "-":
                ok &= bit == 1
                s = s ^ (1 << site)
            elif op == "u":
                ok &= bit == 1
            elif op == "d":
                ok &= bit == 0
            else:
                raise ValueError(op)
        rows.append(s[ok])
        cols.append(states[ok])
        vals.append(amp[ok])
    if rows:
        r, c_, v = np.concatenate(rows), np.concatenate(cols), np.concatenate(vals)
    else:
        r = c_ = np.zeros(0, np.int64)
        v = np.zeros(0, complex)
    return sp.coo_matrix((v, (r, c_)), shape=(dim, dim)).tocsr()


def perm_states(p, N):
    """State image under the site permutation p: bit i of U|s> = bit p[i] of |s>."""
    s = np.arange(1 << N, dtype=np.int64)
    out = np.zeros_like(s)
    for i in range(N):
        out |= ((s >> int(p[i])) & 1) << i
    return out


def perm_order(p):
    seen, order = set(), 1
    for s in range(len(p)):
        if s in seen:
            continue
        n, t = 0, s
        while t not in seen:
            seen.add(t)
            t = p[t]
            n += 1
        order = order * n // math.gcd(order, n)
    return order


def compose(a, b):
    return tuple(a[b[i]] for i in range(len(a)))


def close_group(gens, N, cap=20000):
    ident = tuple(range(N))
    G, frontier = {ident}, [ident]
    gens = [tuple(int(x) for x in g) for g in gens]
    while frontier:
        nxt = []
        for g in frontier:
            for h in gens:
                x = compose(g, h)
                if x not in G:
                    G.add(x)
                    nxt.append(x)
                    if len(G) > cap:
                        raise ValueError("group closure exceeds the cap")
        frontier = nxt
    return G


def point_residues(trans, point, N):
    """One representative per coset of the translation group among the point group the verified
    point permutations generate (identity cosets dropped): an explicit qed.Symmetries split."""
    A = close_group(trans, N) if trans else {tuple(range(N))}
    covered, out = set(A), []
    for p in sorted(close_group(point, N)) if point else []:
        if p in covered:
            continue
        out.append(list(p))
        covered |= {compose(p, a) for a in A}
    return out


def _spmax(A):
    A = A.tocoo()
    return float(np.max(np.abs(A.data))) if A.nnz else 0.0


def _commutes_states(H, q):
    return _spmax(H[q][:, q] - H) < 1e-10


# =============================================================================
# Model families
# =============================================================================

FAMILIES = {"ring": 3.0, "ladder": 2.0, "tri": 2.0, "square": 1.5, "kagome": 1.0, "sawtooth": 1.0,
            "obc_chain": 1.0, "obc_ladder": 1.0, "tri_patch": 1.0, "wheel": 0.7}
TASKS = {"eigs": 20, "vectors": 11, "expect": 10, "spectrum": 12, "th_exact": 9, "th_ftlm": 7,
         "th_oftlm": 4, "th_mtpq": 4, "th_obs": 6, "dyn0": 7, "dynT": 4, "irrep_partition": 3, "invalid": 9}
DMAX = (None, None, 64, 64, 1, 8, 0, 512)          # eigs dense crossover choices (None: automatic)


def wchoice(rng, items):
    items = list(items.items()) if isinstance(items, dict) else list(items)
    w = np.array([float(x[1]) for x in items])
    return items[int(rng.choice(len(items), p=w / w.sum()))][0]


def gen_params(rng, fams):
    fam = wchoice(rng, {f: w for f, w in FAMILIES.items() if f in fams})
    uni = bool(rng.random() < 0.25)          # uniform couplings: degenerate spectra

    def c(lo, hi, nice=1.0):
        return float(nice) if uni else float(rng.uniform(lo, hi))

    def delta():
        return 1.0 if rng.random() < 0.7 else c(0.3, 1.8, 0.5)

    p = {"uniform": uni}
    if fam == "ring":
        var = wchoice(rng, {"heis": 2, "j1j2": 2, "mg": 1, "xxz": 1.5, "xyz": 1, "flux": 1, "dm": 1.5})
        p.update(N=int(rng.integers(5, 13)), var=var, J1=c(0.5, 1.5))
        if var in ("heis", "xyz"):
            p["J2"] = 0.0
        elif var == "mg":
            p["J2"] = 0.5 * p["J1"]
        else:
            p["J2"] = 0.0 if rng.random() < 0.3 else c(-0.3, 0.8, 0.5)
        p["Delta"] = c(0.2, 2.0, 0.5) if var in ("xxz", "flux") else 1.0
        if var == "xyz":
            p.update(jx=c(0.4, 1.4, 1.0), jy=c(0.4, 1.4, 0.5), jz=c(0.4, 1.4, 0.75))
        if var == "flux":
            p["phi"] = c(0.2, 2.9, math.pi / 3)
        if var == "dm":
            p["D"] = c(0.1, 0.8, 0.3)
    elif fam == "ladder":
        p.update(L=int(rng.integers(3, 7)), Jl=c(0.5, 1.5), Jr=c(0.3, 1.5, 0.5),
                 Jd=0.0 if rng.random() < 0.6 else c(0.1, 0.6, 0.25), Delta=delta(),
                 K4=0.0 if rng.random() < 0.75 else c(-0.4, 0.4, 0.2))
    elif fam == "obc_ladder":
        p.update(L=int(rng.integers(2, 7)), Jl=c(0.5, 1.5), Jr=c(0.3, 1.5, 0.5), Delta=delta())
    elif fam == "tri":
        Lx, Ly = [(3, 3), (4, 3), (3, 4), (2, 3), (3, 2), (2, 4), (4, 2)][int(rng.integers(7))]
        p.update(Lx=Lx, Ly=Ly, J=c(0.5, 1.5), Delta=delta(),
                 chi=0.0 if rng.random() < 0.6 else c(0.1, 0.6, 0.25))
    elif fam == "square":
        Lx, Ly = [(3, 3), (4, 3), (3, 4), (2, 4), (4, 2), (2, 3), (3, 2)][int(rng.integers(7))]
        p.update(Lx=Lx, Ly=Ly, J1=c(0.5, 1.5), J2=0.0 if rng.random() < 0.5 else c(0.1, 0.8, 0.5),
                 Delta=delta(), K4=0.0 if rng.random() < 0.7 else c(-0.4, 0.4, 0.2))
    elif fam == "kagome":
        J = c(0.5, 1.5)
        p.update(J=J, Jdown=J if rng.random() < 0.6 else c(0.3, 1.5, 0.5), Delta=delta())
    elif fam == "sawtooth":
        p.update(L=int(rng.integers(3, 7)), J1=c(0.5, 1.5), J2=c(0.5, 1.5, 1.0), Delta=delta())
    elif fam == "obc_chain":
        N = int(rng.integers(5, 13))
        nb = N - 1
        half = [c(0.5, 1.5) for _ in range((nb + 1) // 2)]
        p.update(N=N, J=half + half[: nb // 2][::-1], J2=0.0 if rng.random() < 0.5 else c(0.1, 0.6, 0.5),
                 Delta=delta())
    elif fam == "tri_patch":
        p.update(L=int(rng.choice([2, 3])), J=c(0.5, 1.5), chi=0.0 if rng.random() < 0.6 else c(0.1, 0.6, 0.25))
    elif fam == "wheel":
        p.update(M=int(rng.integers(5, 10)), J=c(0.5, 1.5), Js=c(0.2, 1.5, 0.5))
    r = rng.random()
    if r < 0.6:
        p["field"] = None
    else:
        kind = wchoice(rng, {"z": 4, "x": 2, "y": 1, "xz": 1})
        p["field"] = {"z": {"hz": c(0.1, 0.7, 0.5)}, "x": {"hx": c(0.1, 0.7, 0.5)},
                      "y": {"hy": c(0.1, 0.7, 0.5)},
                      "xz": {"hx": c(0.1, 0.5, 0.3), "hz": c(0.1, 0.5, 0.4)}}[kind]
    # 15%: the library gets the raw records (duplicates, cancelling x/y parts, factors in a
    # shuffled order) instead of the merged terms; it must read the same operator either way.
    p["unmerged"] = int(rng.integers(1, 1 << 30)) if rng.random() < 0.15 else 0
    return fam, p


def build_model(fam, p):
    """{N, terms (merged), terms_lib (what the library gets), trans (generator perms), orders,
    coords, point (candidate perms)}."""
    t, trans, orders, coords, point = [], [], [], None, []
    D = p.get("Delta", 1.0)
    if fam == "ring":
        N, var = p["N"], p["var"]
        for i in range(N):
            j = (i + 1) % N
            if var == "xyz":
                t += xyz(i, j, p["jx"], p["jy"], p["jz"])
            elif var == "flux":
                t += hop(i, j, p["J1"], p["phi"], p["J1"] * D)
            else:
                t += dot(i, j, p["J1"], p["J1"] * D)
            if var == "dm":
                t += dm_z(i, j, p["D"])
            if p["J2"]:
                t += dot(i, (i + 2) % N, p["J2"], p["J2"] * D)
        trans, orders, coords = [[(i + 1) % N for i in range(N)]], [N], [[i] for i in range(N)]
        point = [[(-i) % N for i in range(N)]]
    elif fam in ("ladder", "obc_ladder"):
        L = p["L"]
        N = 2 * L
        pbc = fam == "ladder"

        def s(x, l):
            return (x % L) + L * l
        for x in range(L):
            for l in (0, 1):
                if pbc or x + 1 < L:
                    t += dot(s(x, l), s(x + 1, l), p["Jl"], p["Jl"] * D)
            t += dot(s(x, 0), s(x, 1), p["Jr"], p["Jr"] * D)
            if p.get("Jd") and (pbc or x + 1 < L):
                t += dot(s(x, 0), s(x + 1, 1), p["Jd"], p["Jd"] * D)
                t += dot(s(x, 1), s(x + 1, 0), p["Jd"], p["Jd"] * D)
            if p.get("K4") and pbc:
                t += ring4((s(x, 0), s(x + 1, 0), s(x + 1, 1), s(x, 1)), p["K4"])
        sites = [(x, l) for l in (0, 1) for x in range(L)]          # index x + L*l
        swap = [s(x, 1 - l) for x, l in sites]
        if pbc:
            trans, orders = [[s(x + 1, l) for x, l in sites]], [L]
            coords = [[x] for x, l in sites]
            point = [swap, [s(-x, l) for x, l in sites]]
        else:
            point = [swap, [s(L - 1 - x, l) for x, l in sites]]
    elif fam in ("tri", "square"):
        Lx, Ly = p["Lx"], p["Ly"]
        N = Lx * Ly

        def idx(x, y):
            return (x % Lx) + Lx * (y % Ly)
        xy = [(x, y) for y in range(Ly) for x in range(Lx)]
        for x, y in xy:
            if fam == "tri":
                for dx, dy in ((1, 0), (0, 1), (-1, 1)):
                    t += dot(idx(x, y), idx(x + dx, y + dy), p["J"], p["J"] * D)
                if p["chi"]:
                    t += triple(idx(x, y), idx(x + 1, y), idx(x, y + 1), p["chi"])
                    t += triple(idx(x + 1, y), idx(x + 1, y + 1), idx(x, y + 1), p["chi"])
            else:
                for dx, dy in ((1, 0), (0, 1)):
                    t += dot(idx(x, y), idx(x + dx, y + dy), p["J1"], p["J1"] * D)
                if p["J2"]:
                    for dx, dy in ((1, 1), (1, -1)):
                        t += dot(idx(x, y), idx(x + dx, y + dy), p["J2"], p["J2"] * D)
                if p.get("K4"):
                    t += ring4((idx(x, y), idx(x + 1, y), idx(x + 1, y + 1), idx(x, y + 1)), p["K4"])
        trans = [[idx(x + 1, y) for x, y in xy], [idx(x, y + 1) for x, y in xy]]
        orders, coords = [Lx, Ly], [[x, y] for x, y in xy]
        if fam == "tri":
            point = [[idx(-x, -y) for x, y in xy]]
        else:
            point = [[idx(-x, y) for x, y in xy], [idx(x, -y) for x, y in xy]]
        if Lx == Ly:
            point.append([idx(y, x) for x, y in xy])
    elif fam == "kagome":
        N = 12

        def site(x, y, s):
            return 3 * ((x % 2) + 2 * (y % 2)) + s
        for x in range(2):
            for y in range(2):
                A, B, C = site(x, y, 0), site(x, y, 1), site(x, y, 2)
                for a, b in ((A, B), (A, C), (B, C)):
                    t += dot(a, b, p["J"], p["J"] * D)
                for a, b in ((A, site(x - 1, y, 1)), (A, site(x, y - 1, 2)), (B, site(x + 1, y - 1, 2))):
                    t += dot(a, b, p["Jdown"], p["Jdown"] * D)
        dec = [((i // 3) % 2, (i // 3) // 2, i % 3) for i in range(N)]
        trans = [[site(x + 1, y, s) for x, y, s in dec], [site(x, y + 1, s) for x, y, s in dec]]
        orders, coords = [2, 2], [[x, y] for x, y, s in dec]
    elif fam == "sawtooth":
        L = p["L"]
        N = 2 * L

        def b(x):
            return 2 * (x % L)

        def a(x):
            return 2 * (x % L) + 1
        for x in range(L):
            t += dot(b(x), b(x + 1), p["J1"], p["J1"] * D)
            t += dot(a(x), b(x), p["J2"], p["J2"] * D)
            t += dot(a(x), b(x + 1), p["J2"], p["J2"] * D)
        trans = [[(b(i // 2 + 1) if i % 2 == 0 else a(i // 2 + 1)) for i in range(N)]]
        orders, coords = [L], [[i // 2] for i in range(N)]
        point = [[(b(-(i // 2)) if i % 2 == 0 else a(-(i // 2) - 1)) for i in range(N)]]
    elif fam == "obc_chain":
        N = p["N"]
        for i in range(N - 1):
            t += dot(i, i + 1, p["J"][i], p["J"][i] * D)
        if p["J2"]:
            for i in range(N - 2):
                t += dot(i, i + 2, p["J2"], p["J2"] * D)
        point = [[N - 1 - i for i in range(N)]]
    elif fam == "tri_patch":
        L = p["L"]
        sites = [(i, j) for j in range(L + 1) for i in range(L + 1 - j)]
        ix = {s: n for n, s in enumerate(sites)}
        N = len(sites)
        for (i, j) in sites:
            for di, dj in ((1, 0), (0, 1), (-1, 1)):
                if (i + di, j + dj) in ix:
                    t += dot(ix[(i, j)], ix[(i + di, j + dj)], p["J"], p["J"])
            if p["chi"] and (i + 1, j) in ix and (i, j + 1) in ix:
                t += triple(ix[(i, j)], ix[(i + 1, j)], ix[(i, j + 1)], p["chi"])
        point = [[ix[(L - i - j, i)] for (i, j) in sites], [ix[(j, i)] for (i, j) in sites]]
    elif fam == "wheel":
        M = p["M"]
        N = M + 1
        for i in range(M):
            t += dot(i, (i + 1) % M, p["J"], p["J"])
            t += dot(i, M, p["Js"], p["Js"])
        point = [[(i + 1) % M for i in range(M)] + [M], [(-i) % M for i in range(M)] + [M]]
    else:
        raise ValueError(fam)
    if p.get("field"):
        for i in range(N):
            t += field(i, **p["field"])
    merged = merge(t)
    if p.get("unmerged"):
        r = np.random.default_rng(int(p["unmerged"]))
        raw = []
        for c, ops in t:
            ops = list(ops)
            if len(ops) > 1:
                ops = [ops[int(k)] for k in r.permutation(len(ops))]
            raw.append((c, tuple(ops)))
        lib = raw
    else:
        lib = merged
    return {"N": N, "terms": merged, "terms_lib": lib, "trans": trans, "orders": orders, "coords": coords,
            "point": point}


def verify_content(m):
    """What H really conserves, checked on the sparse matrix."""
    N = m["N"]
    H = sparse_op(m["terms"], N)
    dim = 1 << N
    herm = _spmax(H - H.conj().T) < 1e-10
    coo = H.tocoo()
    keep = np.abs(coo.data) > 1e-12
    r, c, v = coo.row[keep], coo.col[keep], coo.data[keep]
    pop = popcounts(N)
    u1 = bool(np.all(pop[r] == pop[c]))
    parity = bool(np.all((pop[r] - pop[c]) % 2 == 0))
    real = bool(np.all(np.abs(v.imag) < 1e-12))
    # time reversal Theta = prod_i (i sigma^y_i) K: <s'^m|Theta H Theta^-1|s^m> = sign(s') sign(s) conj(H_s's),
    # sign(s) = (-1)^(down spins of s), m = all bits
    import scipy.sparse as sp
    sgn = np.where((N - pop) % 2 == 0, 1.0, -1.0)
    Ht = sp.coo_matrix((sgn[r] * sgn[c] * np.conj(v), (r ^ (dim - 1), c ^ (dim - 1))), shape=(dim, dim)).tocsr()
    theta = _spmax(Ht - H) < 1e-10
    flip = _commutes_states(H, np.arange(dim, dtype=np.int64) ^ (dim - 1))
    su2 = su2_field = False
    if u1:
        Sp = sparse_op([(1.0, (("+", i),)) for i in range(N)], N)
        C = H @ Sp - Sp @ H
        su2 = _spmax(C) < 1e-9
        # SU(2) up to a uniform field h S^z_tot: [H, S^+_tot] = h S^+_tot
        h = complex(C.multiply(Sp.conj()).sum() / Sp.multiply(Sp.conj()).sum())
        su2_field = abs(h.imag) < 1e-12 and _spmax(C - h.real * Sp) < 1e-9
    trans = bool(m["trans"]) and all(_commutes_states(H, perm_states(T, N)) for T in m["trans"])
    point = [list(map(int, P)) for P in m["point"] if _commutes_states(H, perm_states(P, N))]
    return {"N": N, "hermitian": bool(herm), "u1": u1, "parity": parity, "real": real, "flip": bool(flip),
            "su2": bool(su2), "su2_field": bool(su2_field), "theta": bool(theta), "trans": trans, "point": point,
            "s_H": float(sum(abs(c) for c, _ in m["terms"]))}


# =============================================================================
# Case generation (parent; numpy/scipy only)
# =============================================================================

def gen_ops(rng, N, model, content, n, su2_only=False, probe=False, noninv=0.0):
    """Operator specs. su2_only: SU(2)-invariant kinds, except that with probability noninv
    one op is the non-invariant zz (total_spin probes of K3-model-scale-06 /
    K1-sym-composition-05)."""
    has_T = content["trans"]
    if su2_only:
        kinds = {"bond": 3, "chir": 1}
    elif probe:
        kinds = {"sz": 1, "sx": 1, "bond": 1, "chir": 0.5, "samesite": 0.5}
        if has_T:
            kinds.update(szq=3, spq=2)
    else:
        kinds = {"bond": 3, "zz": 2, "sx": 1, "splus": 1, "current": 1, "nonherm": 0.5, "chir": 0.7,
                 "samesite": 0.5}
        if has_T:
            kinds["szq"] = 1
    out = []
    for _ in range(n):
        k = wchoice(rng, kinds)
        i, j, l = (int(x) for x in rng.choice(N, size=3, replace=False))
        spec = {"kind": k, "i": i, "j": j, "k": l}
        if k in ("szq", "spq"):
            spec["q"] = [int(rng.integers(o)) for o in model["orders"]]
        out.append(spec)
    if su2_only and noninv and rng.random() < noninv:
        out[int(rng.integers(len(out)))]["kind"] = "zz"
    return out


def op_terms(spec, model):
    k, i, j, l = spec["kind"], spec.get("i", 0), spec.get("j", 1), spec.get("k", 2)
    N = model["N"]
    if k == "bond":
        return dot(i, j)
    if k == "zz":
        return [(1.0 + 0j, (("z", i), ("z", j)))]
    if k == "sx":
        return _expand(1.0, (("x", i),))
    if k == "sz":
        return [(1.0 + 0j, (("z", i),))]
    if k == "splus":
        return [(1.0 + 0j, (("+", i),))]
    if k == "current":
        return [(0.5j, (("+", i), ("-", j))), (-0.5j, (("-", i), ("+", j)))]
    if k == "nonherm":
        return [(1.0 + 0j, (("+", i), ("-", j)))]
    if k == "samesite":          # S+_i S^z_i (S^z acting first) = -S+_i / 2, plus a zz bond
        return [(1.0 + 0j, (("+", i), ("z", i))), (0.5 + 0j, (("z", j), ("z", l)))]
    if k == "chir":
        return merge(triple(i, j, l, 1.0))
    if k in ("szq", "spq"):
        op = "z" if k == "szq" else "+"
        out = []
        for s in range(N):
            ph = sum(2 * math.pi * qa * ra / La for qa, ra, La in zip(spec["q"], model["coords"][s], model["orders"]))
            out.append((cmath.exp(-1j * ph) / math.sqrt(N), ((op, s),)))
        return out
    raise ValueError(k)


def gen_request(rng, fam, model, content, kind):
    N = model["N"]
    has_T, pts = content["trans"], content["point"]
    sp = [(None, 2), ("auto", 3)]
    if has_T:
        sp += [("split_T", 2), ("list_T", 1)]
    if has_T and pts:
        sp += [("list_TP", 1.5), ("split_TP", 1)]
    if pts:
        sp += [("list_P", 1)]
    spatial = wchoice(rng, sp)
    pg = bool(rng.random() < 0.8)
    S = None
    if content["su2"] and rng.random() < 0.3:
        Ss = [N % 2 / 2 + j for j in range(N // 2 + 1) if N % 2 / 2 + j <= N / 2]
        S = float(Ss[min(int(rng.geometric(0.5)) - 1, len(Ss) - 1)]) if rng.random() < 0.9 else float(Ss[-1])
    if S is not None:
        sz = int(round(N / 2 + S)) if rng.random() < 0.2 else "auto"     # the tower's Sz = +S member
    elif content["u1"]:
        r = rng.random()
        if r < 0.35:
            sz = "auto"
        elif r < 0.7:
            sz = int(rng.choice([0, 1, N // 2, (N + 1) // 2, N // 2, N - 1, N, int(rng.integers(0, N + 1))]))
        elif r < 0.8:
            sz = "off"
        else:
            sz = str(rng.choice(["even", "odd"]))
    elif content["parity"]:
        sz = wchoice(rng, {"auto": 5, "even": 1.75, "odd": 1.75, "off": 1.5})
    else:
        sz = wchoice(rng, {"auto": 7, "off": 3})
    flip = wchoice(rng, {"auto": 6, "off": 2.5, "require": 1.5} if content["flip"] else {"auto": 7, "off": 3})
    tr = wchoice(rng, {"auto": 6, "off": 2.5, "require": 1.5} if content["real"] else {"auto": 7, "off": 3})
    select = None
    if kind == "irrep_partition":
        spatial = "list_TP" if (has_T and pts and rng.random() < 0.5) else "auto"
        pg = True
    elif kind in ("eigs", "vectors", "spectrum", "th_exact", "th_ftlm", "expect", "th_obs") and rng.random() < 0.35:
        if spatial in ("split_T", "list_T"):
            alts = []
            for _ in range(1 if rng.random() < 0.8 else 2):
                alts.append([[int(rng.integers(o)), int(o)] for o in model["orders"]])
            select = {"kind": "momentum", "alts": alts}
        elif spatial in ("list_TP", "split_TP") and pg and fam in ("ring", "ladder", "sawtooth"):
            res = [R for R in point_residues(model["trans"], pts, N) if perm_order(R) == 2]
            if res:
                spatial = "split_TP"
                L = model["orders"][0]
                th = [0, L] if (L % 2 == 1 or rng.random() < 0.5) else [L // 2, L]
                select = {"kind": "irrep", "R": res[int(rng.integers(len(res)))], "theta": th,
                          "chi": int(rng.choice([1, -1]))}
    return {"spatial": spatial, "point_group": pg, "sz": sz, "spin_flip": flip, "time_reversal": tr,
            "total_spin": S, "select": select}


# Invalid requests after which the worker is replaced (they used to be undefined behaviour).
INVALID_RESTART = {"sz_out_of_range", "obs_wrong_size", "none_in_ops", "load_damaged", "nonfinite_H",
                   "thermal_obs_wrong_size", "dyn_obs_wrong_size"}
# Invalid requests whose documented refusal is a builtin class rather than a qed.errors one.
INVALID_BUILTIN_OK = {"nonfinite_H": (RuntimeError,)}


def gen_invalid(rng, model, content):
    N = model["N"]
    kinds = {"half_S": 1, "non_perm": 1, "bad_device": 0.5, "bad_method": 0.5, "dyn_T_nonpositive": 0.6,
             "k_zero": 0.6, "sz_out_of_range": 0.5, "sz_negative": 0.4, "sz_bool": 0.3, "eta_nonpositive": 0.4,
             "obs_wrong_size": 0.3, "none_in_ops": 0.2, "non_hermitian_H": 0.5, "negative_window": 0.4,
             "bad_perm": 1.5, "T_empty": 0.4, "T_nonfinite": 0.4, "omega_nonfinite": 0.4,
             "thermal_krylov_zero": 0.3, "thermal_samples_zero": 0.3, "mtpq_krylov": 0.3, "ftlm_steps": 0.3,
             "dyn_krylov_zero": 0.3, "dyn_samples_zero": 0.3, "neg_degeneracy_tol": 0.3,
             "dense_max_dim_negative": 0.3, "bad_spatial_string": 0.3, "bad_toggle": 0.3,
             "thermal_obs_wrong_size": 0.3, "dyn_obs_wrong_size": 0.3, "load_damaged": 0.4,
             "load_format1": 0.3, "nonfinite_H": 0.3}
    if not content["flip"]:
        kinds["flip_require"] = 2
    if not content["real"] and not content["theta"]:   # time reversal is K or Theta
        kinds["tr_require"] = 2
    if not content["su2_field"]:            # total_spin also takes SU(2) in a uniform field
        kinds["su2_on_non_su2"] = 2
    if content["su2"]:
        kinds["bad_S"] = 1.5
        kinds["sz_su2_disagree"] = 1
    if not content["u1"]:
        kinds["sz_int_non_u1"] = 2
    if not content["parity"]:
        kinds["parity_non_parity"] = 2
    if content["trans"]:
        kinds["mom_nonexistent"] = 1
        kinds["irrep_bad_R"] = 1
        if content["point"]:
            kinds["mom_not_in_group"] = 1
    kind = wchoice(rng, kinds)
    t = {"kind": "invalid", "invalid": kind}
    if kind == "bad_perm":
        H = sparse_op(model["terms"], N)
        for _ in range(40):
            i, j = (int(x) for x in rng.choice(N, size=2, replace=False))
            p = list(range(N))
            p[i], p[j] = j, i
            if not _commutes_states(H, perm_states(p, N)):
                t["perm"] = p
                break
        else:
            t["invalid"] = "non_perm"
    if kind == "bad_S":
        t["S"] = float(N / 2 + 1) if rng.random() < 0.5 else (0.5 if N % 2 == 0 else 0.0)
    if kind == "mom_not_in_group":
        # a point permutation outside the translation group (a mirror of a 2-wide torus is the identity)
        A = close_group(model["trans"], N)
        outside = [p for p in content["point"] if tuple(p) not in A]
        if outside:
            t["perm"] = outside[0]
        else:
            t["invalid"] = "non_perm"
    return t


def gen_task(rng, fam, model, content, req, kind):
    N = model["N"]
    t = {"kind": kind}
    su2_only = req["total_spin"] is not None
    if kind == "eigs":
        k_over = rng.random() < 0.06
        t.update(k=(1 << N) + 3 if k_over else int(rng.choice([1, 1, 2, 3, 4, 5, 6, 8, 12])),
                 window=0.0 if (k_over or rng.random() < 0.7) else float(rng.uniform(0.05, 1.0)),
                 prune=bool(rng.random() < 0.6), dmax=DMAX[int(rng.integers(len(DMAX)))])
    elif kind == "vectors":
        t.update(k=int(rng.integers(1, 7)), prune=bool(rng.random() < 0.6), dmax=DMAX[int(rng.integers(len(DMAX)))],
                 save=bool(rng.random() < 0.5), sz_basis=bool(content["u1"] and rng.random() < 0.5),
                 ops=gen_ops(rng, N, model, content, 1, su2_only=su2_only))
    elif kind == "expect":
        t.update(k=int(rng.integers(1, 7)), prune=bool(rng.random() < 0.6), dmax=DMAX[int(rng.integers(len(DMAX)))],
                 ops=gen_ops(rng, N, model, content, int(rng.integers(1, 4)), su2_only=su2_only, noninv=0.2),
                 me_op=gen_ops(rng, N, model, content, 1)[0])
    elif kind == "th_exact":
        t["T"] = sorted({round(float(x), 4) for x in rng.uniform(0.1, 5.0, size=8)})
    elif kind in ("th_ftlm", "th_oftlm", "th_mtpq"):
        t.update(seed=int(rng.integers(1, 10000)), exact_small=bool(rng.random() < 0.5),
                 krylov=int(rng.choice([40, 100])))
        if kind == "th_oftlm":
            t["exact_states"] = int(rng.choice([1, 4, 8, 32]))
        t["T"] = [round(float(x), 4) for x in (np.linspace(1.0, 4.0, 6) if kind == "th_mtpq" else np.linspace(0.4, 4.0, 8))]
    elif kind == "th_obs":
        t.update(method=str(rng.choice(["exact", "ftlm"])), seed=int(rng.integers(1, 10000)),
                 ops=gen_ops(rng, N, model, content, int(rng.integers(1, 4)), su2_only=su2_only, noninv=0.2))
        t["T"] = [round(float(x), 4) for x in (np.linspace(0.2, 4.0, 8) if t["method"] == "exact" else np.linspace(0.4, 4.0, 8))]
    elif kind in ("dyn0", "dynT"):
        t.update(probe=gen_ops(rng, N, model, content, 1, probe=True)[0], eta=float(rng.choice([0.05, 0.1, 0.2])),
                 seed=int(rng.integers(1, 10000)))
        if kind == "dynT":
            t.update(T=float(rng.choice([0.5, 1.0, 2.0])), samples=60, krylov=80)
        else:
            t.update(krylov=200)
    return t


def generate_case(seed, index, fams=None, tasks=None):
    fams = fams or list(FAMILIES)
    tasks = tasks or list(TASKS)
    rng = np.random.default_rng([int(seed), int(index), 0x5EED])
    for _ in range(100):
        fam, params = gen_params(rng, fams)
        model = build_model(fam, params)
        content = verify_content(model)
        N = model["N"]
        if not content["hermitian"]:
            continue
        if not content["u1"] and not content["parity"] and N > 10:
            continue
        break
    kind = wchoice(rng, {k: w for k, w in TASKS.items() if k in tasks})
    if kind == "th_mtpq" and N < 8:
        kind = "th_exact"
    if kind == "dynT" and N > 10:
        kind = "dyn0"
    restart = False
    if kind == "invalid":
        req = gen_request(rng, fam, model, content, "eigs")
        req["select"] = None
        task = gen_invalid(rng, model, content)
        restart = task["invalid"] in INVALID_RESTART
    else:
        req = gen_request(rng, fam, model, content, kind)
        task = gen_task(rng, fam, model, content, req, kind)
    env = {}           # per-case environment overrides (none drawn at present)
    return {"case_id": f"{seed}-{index}", "seed": int(seed), "index": int(index),
            "model": {"family": fam, "params": params, "N": int(N)}, "content": content,
            "request": req, "task": task, "env": env, "restart_after": restart}


# =============================================================================
# Dense oracle (worker)
# =============================================================================

class HarnessError(Exception):
    pass


class CaseTimeout(BaseException):
    pass


class QedRaised(Exception):
    def __init__(self, exc):
        super().__init__(f"{type(exc).__name__}: {exc}")
        self.exc = exc


def qcall(fn, *a, **kw):
    """Run a library call; any exception it raises is the library's (QedRaised)."""
    try:
        return fn(*a, **kw)
    except CaseTimeout:
        raise
    except Exception as e:  # noqa: BLE001
        raise QedRaised(e) from e


class Oracle:
    def __init__(self, model, content):
        self.N = N = model["N"]
        self.dim = 1 << N
        self.H = sparse_op(model["terms"], N)
        self.pop = popcounts(N)                   # the number of up spins of each state
        self.u1, self.parity = content["u1"], content["parity"]
        self.s_H = float(content.get("s_H") or sum(abs(c) for c, _ in model["terms"]))
        self._S2 = None
        self._perm = {}
        self._cache = {}

    @property
    def S2(self):
        if self._S2 is None:
            t = []
            for i in range(self.N):
                for j in range(self.N):
                    t += dot(i, j)
            self._S2 = sparse_op(t, self.N)
        return self._S2

    def natural_blocks(self):
        pop, N = self.pop, self.N
        if self.u1:
            return [np.flatnonzero(pop == a) for a in range(N + 1)]
        if self.parity:
            return [np.flatnonzero(pop % 2 == 0), np.flatnonzero(pop % 2 == 1)]
        return [np.arange(self.dim)]

    def blocks(self, sz, S):
        pop, N = self.pop, self.N
        if S is not None:
            if not self.u1:
                raise HarnessError("total spin on a non-U(1) model")
            two = int(round(2 * S))
            return [np.flatnonzero(pop == a) for a in range(N + 1)
                    if abs(N - 2 * a) <= two and (two - abs(N - 2 * a)) % 2 == 0]
        if isinstance(sz, int):
            return [np.flatnonzero(pop == sz)]
        if sz in ("even", "odd"):
            par = 0 if sz == "even" else 1
            if self.u1:
                return [np.flatnonzero(pop == a) for a in range(N + 1) if a % 2 == par]
            return [np.flatnonzero(pop % 2 == par)]
        return self.natural_blocks()

    def bperm(self, idx, p):
        key = tuple(int(x) for x in p)
        if key not in self._perm:
            self._perm[key] = perm_states(key, self.N)
        q = self._perm[key][idx]
        pos = np.full(self.dim, -1, dtype=np.int64)
        pos[idx] = np.arange(len(idx))
        j = pos[q]
        if np.any(j < 0):
            raise HarnessError("a permutation does not preserve a conserved block")
        return j

    def proj_cyclic(self, idx, p, lam):
        """Projector onto U_p = lam inside the block (lam must be an order-th root of 1)."""
        pi = self.bperm(idx, p)
        m = perm_order(list(p))
        d = len(idx)
        if abs(lam ** m - 1) > 1e-9:
            return np.zeros((d, d), complex)
        P = np.zeros((d, d), complex)
        cur, cols = np.arange(d), np.arange(d)
        for n in range(m):
            P[cur, cols] += lam ** (-n)
            cur = pi[cur]
        return P / m

    def levels(self, R):
        """Eigenpairs of H inside the restriction R: [(E, V, idx, sz)] with V in the block basis.
        R keys: sz (int n_up | 'even' | 'odd' | None), S, trans, mom (list of theta-lists), irrep (perm, chi)."""
        key = json.dumps(R, sort_keys=True, default=str)
        if key in self._cache:
            return self._cache[key]
        out = []
        for idx in self.blocks(R.get("sz"), R.get("S")):
            if len(idx) == 0:
                continue
            Hb = self.H[idx][:, idx].toarray()
            d = len(idx)
            Ps = []
            if R.get("S") is not None:
                S = R["S"]
                w, U = np.linalg.eigh(self.S2[idx][:, idx].toarray())
                Q = U[:, np.abs(w - S * (S + 1)) < 1e-6]
                Ps.append(Q @ Q.conj().T)
            if R.get("mom") is not None:
                Pm = np.zeros((d, d), complex)
                seen = set()
                for alt in R["mom"]:
                    kt = tuple(round(x % 1.0, 9) % 1.0 for x in alt)
                    if kt in seen:
                        continue
                    seen.add(kt)
                    P = np.eye(d, dtype=complex)
                    for T, th in zip(R["trans"], alt):
                        P = P @ self.proj_cyclic(idx, T, cmath.exp(-2j * math.pi * th))
                    Pm += P
                Ps.append(Pm)
            if R.get("irrep") is not None:
                Rp, chi = R["irrep"]
                Ps.append(self.proj_cyclic(idx, Rp, complex(chi)))
            if Ps:
                P = Ps[0]
                for X in Ps[1:]:
                    P = P @ X
                if len(Ps) > 1 and d <= 1024 and (np.max(np.abs(P @ P - P), initial=0.0) > 1e-8
                                                  or np.max(np.abs(P - P.conj().T), initial=0.0) > 1e-8):
                    raise HarnessError("restriction projectors do not commute")
                P = 0.5 * (P + P.conj().T)
                w, Q = np.linalg.eigh(P)
                Q = Q[:, w > 0.5]
                if Q.shape[1] == 0:
                    continue
                E, Vq = np.linalg.eigh(Q.conj().T @ Hb @ Q)
                V = Q @ Vq
            else:
                E, V = np.linalg.eigh(Hb)
            sz = (float(self.pop[idx[0]]) - self.N / 2) if self.u1 else None     # Sz = n_up - N/2
            out.append((E, V, idx, sz))
        self._cache[key] = out
        return out

    def energies(self, R):
        L = self.levels(R)
        return np.sort(np.concatenate([e for e, _, _, _ in L])) if L else np.zeros(0)

    def full(self, V, idx):
        X = np.zeros((self.dim, V.shape[1]), complex)
        X[idx] = V
        return X

    def diag(self, R, O):
        """[(E, <v|O|v>)] over the restriction's eigenvectors (arrays)."""
        Es, ds = [], []
        for E, V, idx, _ in self.levels(R):
            Ob = O[idx][:, idx].toarray()
            Es.append(E)
            ds.append(np.einsum("im,ij,jm->m", V.conj(), Ob, V))
        if not Es:
            return np.zeros(0), np.zeros(0, complex)
        E, d = np.concatenate(Es), np.concatenate(ds)
        o = np.argsort(E, kind="stable")
        return E[o], d[o]

    def thermo(self, R, T):
        L = self.levels(R)
        E = np.concatenate([e for e, _, _, _ in L])
        sz = None if not self.u1 else np.concatenate([np.full(len(e), s) for e, _, _, s in L])
        e0 = E.min()
        out = {k: [] for k in ("E", "C", "S", "F", "lnZ", "M", "chi")}
        for t in T:
            b = 1.0 / t
            w = np.exp(-b * (E - e0))
            Z = w.sum()
            e = (w * E).sum() / Z
            e2 = (w * E * E).sum() / Z
            lnZ = math.log(Z) - b * e0
            out["E"].append(e)
            out["C"].append(b * b * (e2 - e * e))
            out["S"].append(lnZ + b * e)
            out["F"].append(-lnZ / b)
            out["lnZ"].append(lnZ)
            if sz is not None:
                m = (w * sz).sum() / Z
                m2 = (w * sz * sz).sum() / Z
                out["M"].append(m)
                out["chi"].append(b * (m2 - m * m) / self.N)
        return {k: (np.array(v) if v else None) for k, v in out.items()}

    def thermal_expect(self, R, O, T):
        E, d = self.diag(R, O)
        e0 = E.min()
        return np.array([complex((np.exp(-(E - e0) / t) * d).sum() / np.exp(-(E - e0) / t).sum()) for t in T])

    def clusters(self, R, O, tol=1e-7):
        E, d = self.diag(R, O)
        out, a = [], 0
        for b in range(1, len(E) + 1):
            if b == len(E) or E[b] - E[a] > tol:
                out.append((float(E[a]), b - a, complex(d[a:b].sum())))
                a = b
        return out

    def dynamics(self, R, O, omega, eta, T=None, gm_top=None):
        """Lorentzian-broadened Lehmann sum and its exact moments (m0, m1). The ground manifold is
        every level within 1e-8 s_H of E0, as the library's default degeneracy_tol."""
        init = self.levels(R)
        targets = self.levels({"sz": None, "S": None})
        omega = np.asarray(omega, float)
        xs, ws = [], []
        if T is None:
            E0 = min(float(e.min()) for e, _, _, _ in init)
            tol = 1e-8 * max(self.s_H, 1e-300)
            cols = []
            for E, V, idx, sz in init:
                if gm_top is not None and (sz is None or abs(sz - gm_top) > 1e-9):
                    continue
                sel = E <= E0 + tol
                if np.any(sel):
                    cols.append(self.full(V[:, sel], idx))
            if not cols:
                raise HarnessError("empty ground manifold")
            X = np.hstack(cols)
            Y = O @ X
            g = X.shape[1]
            for Et, Vt, it, _ in targets:
                W = np.abs(Vt.conj().T @ Y[it]) ** 2
                ws.append(W.sum(axis=1) / g)
                xs.append(Et - E0)
        else:
            b = 1.0 / T
            E0 = min(float(e.min()) for e, _, _, _ in init)
            Z = sum(float(np.exp(-b * (e - E0)).sum()) for e, _, _, _ in init)
            for E, V, idx, _ in init:
                p = np.exp(-b * (E - E0)) / Z
                keep = p > 1e-13
                if not np.any(keep):
                    continue
                Y = O @ self.full(V[:, keep], idx)
                Ek, pk = E[keep], p[keep]
                for Et, Vt, it, _ in targets:
                    W = (np.abs(Vt.conj().T @ Y[it]) ** 2) * pk[None, :]
                    x = Et[:, None] - Ek[None, :]
                    m = W > 1e-15
                    ws.append(W[m])
                    xs.append(x[m])
        x = np.concatenate(xs) if xs else np.zeros(0)
        w = np.concatenate(ws) if ws else np.zeros(0)
        m = w > 1e-15
        x, w = x[m], w[m]
        S = np.zeros(len(omega))
        for a in range(0, len(x), 4000):
            xa, wa = x[a:a + 4000], w[a:a + 4000]
            S += (wa[None, :] * (eta / math.pi) / ((omega[:, None] - xa[None, :]) ** 2 + eta ** 2)).sum(axis=1)
        return S, float(w.sum()), float((w * x).sum()), E0


# =============================================================================
# Worker-side case execution
# =============================================================================

def mk(status, metric=None, message="", known_id=None, **extra):
    if metric is not None and not (isinstance(metric, (int, float)) and math.isfinite(metric)):
        message = f"{message} (metric {metric})".strip()
        metric = None
    return {"status": status, "metric": None if metric is None else float(metric), "message": str(message)[:1500],
            "known_id": known_id, "extra": extra}


def classify(exc, qed):
    """(status, message, extra) for an exception a library call raised on a VALID request.
    A qed.errors class names a deliberate refusal (ConvergenceError: a failure); a builtin
    ValueError / TypeError / NotImplementedError is a refusal outside the qed.errors contract;
    anything else is a crash."""
    msg = f"{type(exc).__name__}: {exc}"
    errs = getattr(qed, "errors", None)
    is_q = errs is not None and isinstance(exc, errs.QEDError)
    extra = {"error_class": type(exc).__name__, "qed_error": bool(is_q)}
    if is_q:
        if isinstance(exc, errs.ConvergenceError):
            return "crash", msg, extra
        # device='gpu' is strict: a block without a device kernel -- a sector of an irrep of
        # dimension > 1 whose reduced CSR does not fit the budgets (P7.5 uploads it when it does) --
        # is refused, naming it. That documented refusal passes.
        if isinstance(exc, errs.DeviceUnsupported) and DOCUMENTED_NO_KERNEL.search(str(exc)):
            return "pass", "documented refusal: " + msg, extra
        return "refused", msg, extra
    if isinstance(exc, (ValueError, TypeError, NotImplementedError)):
        return "refused", msg, extra
    return "crash", msg, extra


def _diag_codes(r):
    return sorted({str(c) for c, _ in (getattr(r, "diagnostics", None) or [])})


def result_extra(ctx, r):
    """Device engagement, placement and diagnostics codes of a result."""
    e = {"diagnostics": _diag_codes(r)}
    pl = getattr(r, "placement", None)
    if pl:
        e["placement"] = {str(k): int(v) for k, v in dict(pl).items() if v}
    if ctx.device == "gpu":
        nb = int(getattr(r, "device_blocks", 0) or 0)
        e.update(device_blocks=nb, device_engaged=bool(nb > 0 or (pl and (pl.get("device_krylov", 0)
                                                                          or pl.get("device_dense", 0)))))
    return e


class Ctx:
    def __init__(self, case, qed, tmpdir):
        self.case, self.qed, self.tmpdir = case, qed, tmpdir
        m = case["model"]
        self.model = build_model(m["family"], m["params"])
        self.content = case["content"]
        if not self.content["trans"]:
            self.model["trans"] = []
        self.N = self.model["N"]
        self.req, self.task = case["request"], case["task"]
        self.device = case["device"]
        self.H = qcall(self.to_op, self.model["terms_lib"], self.N)
        self._orc = None
        self.irrep_R = None
        self.notes = []

    def variant(self, key, value):
        """The same case with one request field changed (shares H and the oracle)."""
        v = copy.copy(self)
        v.req = dict(self.req)
        v.req[key] = value
        v.notes = []
        v.irrep_R = None
        v._orc = self.orc
        return v

    @property
    def orc(self):
        if self._orc is None:
            self._orc = Oracle(self.model, self.content)
        return self._orc

    def to_op(self, terms, N):
        """qed.Operator of the term list: one- to three-body S+/S-/Sz records through the typed
        setters, anything else (four-site terms, u/d projectors) through Operator.product."""
        qed = self.qed
        O = qed.Operator(int(N))
        code = {"+": qed.OP_SPLUS, "-": qed.OP_SMINUS, "z": qed.OP_SZ}
        long_ = None
        for c, ops in terms:
            if abs(c) < 1e-15:
                continue
            if len(ops) <= 3 and all(op in code for op, _ in ops):
                args = [x for op, s in ops for x in (code[op], int(s))]
                {1: O.add_one_body, 2: O.add_two_body, 3: O.add_three_body}[len(ops)](*args, complex(c))
            else:
                P = qed.Operator.product(int(N), "".join(op for op, _ in ops), [int(s) for _, s in ops], complex(c))
                long_ = P if long_ is None else long_ + P
        return O if long_ is None else O + long_

    def spatial(self, kind):
        md = self.model
        if kind is None or kind == "auto":
            return kind
        if kind == "split_T":
            return self.qed.Symmetries(abelian=[list(t) for t in md["trans"]], residues=[])
        if kind == "split_TP":
            return self.qed.Symmetries(abelian=[list(t) for t in md["trans"]],
                                       residues=point_residues(md["trans"], self.content["point"], self.N))
        if kind == "list_T":
            return [list(t) for t in md["trans"]]
        if kind == "list_TP":
            return [list(t) for t in md["trans"]] + [list(p) for p in self.content["point"]]
        if kind == "list_P":
            return [list(p) for p in self.content["point"]]
        raise HarnessError(f"spatial kind {kind}")

    def sym(self, with_select=True):
        q, r = self.qed, self.req
        s = q.Symmetry(spatial=self.spatial(r["spatial"]), sz=r["sz"], spin_flip=r["spin_flip"],
                       time_reversal=r["time_reversal"], point_group=bool(r["point_group"]),
                       total_spin=r["total_spin"])
        sel = r.get("select") if with_select else None
        if not sel:
            return s
        if sel["kind"] == "momentum":
            alts = [{tuple(self.model["trans"][j]): Fraction(m_, L) for j, (m_, L) in enumerate(a)} for a in sel["alts"]]
            return qcall(s.select, momentum=alts if len(alts) > 1 else alts[0])
        if sel["kind"] == "irrep":
            A, res = qcall(s.groups, self.H)
            T = tuple(self.model["trans"][0])
            R = tuple(int(x) for x in sel["R"])
            if T not in {tuple(a) for a in A}:
                raise HarnessError("the explicit split's abelian part lacks the translation")
            if R not in {tuple(x) for x in res}:
                raise SkipCase("the residue is not among the point-group representatives (point_group off?)")
            m_, L = sel["theta"]
            self.irrep_R = {"trans": [list(T)], "mom": [[m_ / L]], "irrep": (list(R), sel["chi"])}
            self.notes.append(f"R={list(R)} chi={sel['chi']} theta={m_}/{L}")
            return qcall(s.select, momentum={T: Fraction(m_, L)}, irrep_character={R: sel["chi"]})
        raise HarnessError(sel["kind"])

    def base_R(self):
        r = self.req
        sz = r["sz"]
        R = {"sz": sz if (isinstance(sz, int) or sz in ("even", "odd")) else None, "S": r["total_spin"],
             "trans": None, "mom": None, "irrep": None}
        sel = r.get("select")
        if sel and sel["kind"] == "momentum":
            R["trans"] = self.model["trans"]
            R["mom"] = [[m_ / L for m_, L in a] for a in sel["alts"]]
        if sel and sel["kind"] == "irrep":
            if self.irrep_R is None:
                raise HarnessError("irrep restriction requested before sym()")
            R.update(self.irrep_R)
        return R

    def candidates(self):
        """[(name, R, kind)]: the documented restriction first, then documented conventions
        (kind 'convention': a match passes) and diagnostic readings (kind 'diagnostic': a match
        is still wrong, but the message names the reading)."""
        R = self.base_R()
        out = [("main", R, "main")]
        r = self.req
        if R.get("mom") is not None and R.get("irrep") is None and self.content["real"] and r["time_reversal"] != "off":
            R2 = dict(R)
            R2["mom"] = R["mom"] + [[(-x) % 1.0 for x in a] for a in R["mom"]]
            out.append(("k_and_minus_k", R2, "convention"))
        if r["sz"] in ("even", "odd") and self.content["u1"] and r["total_spin"] is None:
            R3 = dict(R)
            R3["sz"] = None
            out.append(("parity_ignored", R3, "diagnostic"))       # the fixed C01-pyapi-05 reading
        return out

    def eigs_kw(self):
        """On the GPU the blocks the automatic crossover (or 64 / 512) would solve densely on the
        host go to the device Krylov lane instead (dense_max_dim=0), as the audit grid did."""
        t = self.task
        dmax = t.get("dmax", None)
        if self.device == "gpu" and (dmax is None or dmax >= 64):
            dmax = 0
        return {"prune": bool(t.get("prune", True)), "dense_max_dim": dmax}


class SkipCase(Exception):
    def __init__(self, msg, known_id=None):
        super().__init__(msg)
        self.known_id = known_id


def best_match(ctx, check):
    """check(R) -> (ok, err, msg). Main restriction first; a convention that matches passes; a
    diagnostic reading that matches is wrong with its name in the message."""
    first = None
    for name, R, kind in ctx.candidates():
        ok, err, msg = check(R)
        if first is None:
            first = (ok, err, msg)
        if ok:
            if kind == "diagnostic":
                return mk("wrong", err, f"matches the '{name}' reading, not the request: {msg}", reading=name)
            note = "" if kind == "main" else f"[convention {name}] "
            return mk("pass", err, note + msg)
    return mk("wrong", first[1], first[2])


def cmp_multiset(got, ref, tol):
    got, ref = np.sort(np.asarray(got, float)), np.sort(np.asarray(ref, float))
    if len(got) != len(ref):
        return False, math.inf, f"{len(got)} values vs {len(ref)} in the reference"
    if len(ref) == 0:
        return True, 0.0, "empty"
    err = float(np.max(np.abs(got - ref)))
    return err <= tol * max(1.0, float(np.max(np.abs(ref)))), err, f"max|dE| {err:.2e}"


def contained(got, ref, tol):
    j, worst = 0, 0.0
    for g in np.sort(got):
        while j < len(ref) and ref[j] < g - tol:
            j += 1
        if j >= len(ref) or abs(ref[j] - g) > tol:
            return False, float(g)
        worst = max(worst, abs(ref[j] - g))
        j += 1
    return True, worst


def cmp_lowest(got, ref, k, window, tol):
    got = np.sort(np.asarray(got, float))
    if window <= 0:
        want = min(k, len(ref))
        if len(got) != want:
            return False, math.inf, f"returned {len(got)} energies, expected {want} (reference dim {len(ref)})"
        if want == 0:
            return True, 0.0, "empty"
        err = float(np.max(np.abs(got - ref[:want])))
        return err <= tol * max(1.0, abs(float(ref[0]))), err, f"max|dE| {err:.2e}"
    want = min(k, len(ref))
    if len(got) < want:
        return False, math.inf, f"window: only {len(got)} energies for k={k}"
    err = float(np.max(np.abs(got[:want] - ref[:want]))) if want else 0.0
    ok, w = contained(got, ref, 1e-6 * max(1.0, abs(float(ref[0]))))
    if not ok:
        return False, math.inf, f"window: {w} is not an eigenvalue in the restriction"
    top = ref[want - 1] + window + 1e-6
    if np.any(got > top):
        return False, math.inf, f"window: value {got.max()} above E_k + window = {top}"
    return err <= tol * max(1.0, abs(float(ref[0]))), err, f"window: {len(got)} values, max|dE| {err:.2e}"


def momentum_labels_ok(ctx, r):
    sel = ctx.req.get("select")
    if not sel or sel["kind"] != "momentum":
        return True, ""
    allowed = set()
    for a in sel["alts"]:
        th = tuple(Fraction(m_, L) for m_, L in a)
        allowed.add(th)
        allowed.add(tuple((-x) % 1 for x in th))
    bad = []
    for i, _L in enumerate(r.levels):
        lab = tuple(qcall(r.momentum, i, ctx.model["trans"]))
        if lab not in allowed:
            bad.append((i, [str(x) for x in lab]))
    return (not bad), (f"levels labelled outside the selection: {bad[:3]}" if bad else "")


def multiplet_deficit(r, upto):
    """Levels among the first that cover `upto` states whose full-basis multiplet() returns
    fewer vectors than the level's multiplicity: [(level, got, multiplicity)]."""
    out, n = [], 0
    for i, L in enumerate(r.levels):
        if n >= upto:
            break
        n += int(L.multiplicity)
        if L.vector < 0:
            continue
        try:
            got = len(r._raw.multiplet(r._spec, i, -1, 0))
        except Exception:  # noqa: BLE001
            continue
        if got < int(L.multiplicity):
            out.append([i, got, int(L.multiplicity)])
    return out


def empty_case(ctx, call):
    """The restriction is empty: the library must raise EmptySelection."""
    try:
        r = call()
    except QedRaised as e:
        if isinstance(e.exc, ctx.qed.errors.EmptySelection):
            return mk("pass", 0.0, f"empty restriction refused: {e}")
        st, msg, ex = classify(e.exc, ctx.qed)
        return mk(st, None, f"empty restriction: {msg} (expected EmptySelection)", **ex)
    n = len(getattr(r, "energies", []))
    return mk("wrong", None, f"empty restriction returned {n} energies, no error")


# ---- tasks -------------------------------------------------------------------

def t_eigs(ctx):
    t = ctx.task
    k, window = int(t["k"]), float(t.get("window", 0.0))
    kw = ctx.eigs_kw()
    kw.update(window=window)
    sym = ctx.sym()
    ref0 = ctx.orc.energies(ctx.base_R())
    if len(ref0) == 0:
        return empty_case(ctx, lambda: qcall(ctx.qed.eigs, ctx.H, k, sym=sym, device=ctx.device, **kw))
    r = qcall(ctx.qed.eigs, ctx.H, k, sym=sym, device=ctx.device, **kw)
    got = np.asarray(r.energies, float)
    extra = dict(levels=len(r.levels), complete=bool(r.complete), pruned_blocks=int(r.pruned_blocks),
                 **result_extra(ctx, r))
    out = best_match(ctx, lambda R: cmp_lowest(got, ctx.orc.energies(R), k, window, 1e-7))
    if not r.complete:
        out = mk("wrong", out["metric"], f"complete=False without allow_partial; {out['message']}")
    ok, msg = momentum_labels_ok(ctx, r)
    if not ok and out["status"] == "pass":
        out = mk("wrong", out["metric"], msg)
    if k > len(ref0) and out["status"] == "pass":
        out["message"] = f"k={k} > restricted dim {len(ref0)}: returned all; " + out["message"]
    out["extra"].update(extra)
    return out


def t_spectrum(ctx):
    sym = ctx.sym()
    ref0 = ctx.orc.energies(ctx.base_R())
    if len(ref0) == 0:
        return empty_case(ctx, lambda: qcall(ctx.qed.spectrum, ctx.H, sym=sym, device=ctx.device))
    r = qcall(ctx.qed.spectrum, ctx.H, sym=sym, device=ctx.device)
    got = np.asarray(r.energies, float)
    out = best_match(ctx, lambda R: cmp_multiset(got, ctx.orc.energies(R), 1e-8))
    mult = sum(int(L.multiplicity) for L in r.levels)
    if mult != len(got) and out["status"] == "pass":
        out = mk("wrong", out["metric"], f"sum of level multiplicities {mult} != {len(got)} energies")
    ok, msg = momentum_labels_ok(ctx, r)
    if not ok and out["status"] == "pass":
        out = mk("wrong", out["metric"], msg)
    out["extra"].update(levels=len(r.levels), **result_extra(ctx, r))
    return out


def _in_restriction(orc, R, v):
    w = 0.0
    for _E, V, idx, _ in orc.levels(R):
        w += float(np.linalg.norm(V.conj().T @ v[idx]) ** 2)
    return w


def _sz_vectors_ok(ctx, r, a, expect_some):
    """vectors(basis='sz', n_up=a): each an eigenvector of H in the sector of a up spins."""
    orc = ctx.orc
    probs, worst = [], 0.0
    try:
        vsz = qcall(r.vectors, basis="sz", n_up=a)
    except QedRaised as e:
        return [f"vectors(basis='sz', n_up={a}) raised {e}"], worst
    ia = np.flatnonzero(orc.pop == a)
    if expect_some and len(vsz) == 0:
        probs.append(f"vectors(basis='sz', n_up={a}) returned [] although levels[0] lives there")
    for v in vsz:
        v = np.asarray(v, complex)
        if len(v) != len(ia):
            probs.append(f"sz-basis vector length {len(v)} != C(N,{a}) = {len(ia)}")
            break
        x = np.zeros(orc.dim, complex)
        x[ia] = v / np.linalg.norm(v)
        hx = orc.H @ x
        e = float(np.vdot(x, hx).real)
        rr = float(np.linalg.norm(hx - e * x))
        worst = max(worst, rr)
        if rr > 1e-6:
            probs.append(f"sz-basis (n_up={a}) vector residual {rr:.1e}")
            break
    return probs, worst


def t_vectors(ctx):
    t = ctx.task
    k = int(t["k"])
    sym = ctx.sym()
    orc = ctx.orc
    if len(orc.energies(ctx.base_R())) == 0:
        return empty_case(ctx, lambda: qcall(ctx.qed.eigs, ctx.H, k, sym=sym, vectors=True, device=ctx.device,
                                             **ctx.eigs_kw()))
    r = qcall(ctx.qed.eigs, ctx.H, k, sym=sym, vectors=True, device=ctx.device, **ctx.eigs_kw())
    vs = qcall(r.vectors)
    extra = dict(levels=len(r.levels), time_reversal=r.time_reversal,
                 tr_folded=any(bool(getattr(L, "tr_folded", False)) for L in r.levels), **result_extra(ctx, r))
    problems = []

    def check(R):
        ref = orc.energies(R)
        want = min(k, len(ref))
        if len(vs) != want:
            return False, math.inf, f"{len(vs)} vectors for k={k} (restricted dim {len(ref)})"
        if want == 0:
            return True, 0.0, ""
        V = np.array(vs, complex).T
        if V.shape[0] != orc.dim:
            return False, math.inf, f"vector length {V.shape[0]} != 2^N"
        G = V.conj().T @ V
        orth = float(np.max(np.abs(G - np.eye(want))))
        HV = orc.H @ V
        ray = np.real(np.einsum("im,im->m", V.conj(), HV)) / np.real(np.diag(G))
        res = float(max(np.linalg.norm(HV[:, m] - ray[m] * V[:, m]) for m in range(want)))
        low = float(np.max(np.abs(np.sort(ray) - ref[:want])))
        inside = min(_in_restriction(orc, R, V[:, m] / np.linalg.norm(V[:, m])) for m in range(want))
        err = max(orth, res, low, 1.0 - inside)
        return err < 1e-6, err, f"orth {orth:.1e} residual {res:.1e} lowest {low:.1e} weight-outside {1 - inside:.1e}"

    out = best_match(ctx, check)
    msgs = [out["message"]]
    # Sz-basis vectors: the solved sector of levels[0], and its mirror (flip-folded level) or the
    # tower's Sz = -S member (total_spin), which vectors() must build too.
    if t.get("sz_basis") and orc.u1 and r.levels:
        a = int(r.levels[0].n_up)
        if a >= 0:
            p, _ = _sz_vectors_ok(ctx, r, a, True)
            problems += p
            L0 = r.levels[0]
            if (ctx.req["total_spin"] is not None or int(getattr(L0, "mirror", 1)) == 2) and ctx.N - a != a:
                p, _ = _sz_vectors_ok(ctx, r, ctx.N - a, True)
                problems += p
    # save / load
    if t.get("save"):
        path = os.path.join(ctx.tmpdir, f"{ctx.case['case_id']}_{ctx.device}_{os.getpid()}.npz")
        try:
            qcall(r.save, path)
            r2 = qcall(ctx.qed.load_eigs, path)
            if not np.allclose(np.asarray(r2.energies), np.asarray(r.energies), atol=1e-12):
                problems.append("load_eigs energies differ")
            vs2 = qcall(r2.vectors)
            if len(vs2) != len(vs):
                problems.append(f"load_eigs gave {len(vs2)} vectors vs {len(vs)}")
            elif vs:
                A = np.array(vs, complex).T
                B = np.array(vs2, complex).T
                sv = np.linalg.svd(A.conj().T @ B, compute_uv=False)
                if np.max(np.abs(sv - 1)) > 1e-8:
                    problems.append(f"reloaded vectors span a different subspace (sv {np.round(sv, 6).tolist()})")
            ops = [ctx.to_op(op_terms(s, ctx.model), ctx.N) for s in t.get("ops", [])]
            if ops:
                e1, e2 = qcall(r.expect, ops), qcall(r2.expect, ops)
                if not np.allclose(e1, e2, atol=1e-10):
                    problems.append(f"reloaded expect differs by {np.max(np.abs(e1 - e2)):.1e}")
        except QedRaised as e:
            problems.append(f"save/load raised {e}")
        finally:
            try:
                os.remove(path)
            except OSError:
                pass
    if problems:
        out = mk("wrong", out["metric"], "; ".join(msgs + problems))
    if out["status"] != "pass":
        extra["mult_deficit"] = multiplet_deficit(r, k)
    out["extra"].update(extra)
    return out


def t_expect(ctx):
    t = ctx.task
    k = int(t["k"])
    sym = ctx.sym()
    orc = ctx.orc
    specs = t["ops"]
    ops = [ctx.to_op(op_terms(s, ctx.model), ctx.N) for s in specs]
    Os = [sparse_op(op_terms(s, ctx.model), ctx.N) for s in specs]
    if len(orc.energies(ctx.base_R())) == 0:
        return empty_case(ctx, lambda: qcall(ctx.qed.expect, ctx.H, ops, k, sym=sym, device=ctx.device,
                                             **ctx.eigs_kw()))
    r = qcall(ctx.qed.expect, ctx.H, ops, k, sym=sym, device=ctx.device, **ctx.eigs_kw())
    rows = [(float(e), int(m_), np.asarray(v)) for e, m_, v in zip(r.energies, r.multiplicities, r.values)]
    extra = dict(levels=len(rows), time_reversal=r.eigs.time_reversal,
                 **result_extra(ctx, r.eigs))

    def check(R):
        worst, checked, conj_ok, msgs = 0.0, 0, True, []
        for oi, O in enumerate(Os):
            for E, dim, tr in orc.clusters(R, O):
                mine = [(m_, v[oi]) for e, m_, v in rows if abs(e - E) < 1e-6]
                if sum(m_ for m_, _ in mine) != dim:
                    continue
                s = sum(m_ * v for m_, v in mine)
                d = abs(s - tr)
                worst = max(worst, d)
                if abs(np.conj(s) - tr) > 1e-7 * max(1, dim):
                    conj_ok = False
                checked += 1
        # the lowest k energies (with multiplicity) must be right whatever the partner choice
        okE, errE, msgE = cmp_lowest(np.asarray(r.eigs.energies, float), orc.energies(R), k, 0.0, 1e-7)
        if not okE:
            return False, errE, f"energies: {msgE}"
        if checked == 0:     # k cut every degenerate cluster across blocks: nothing partner-free to test
            return True, 0.0, "energies ok; no complete degenerate cluster (unchecked values)"
        ok = worst < 1e-7 * max(1, max(d for _, d, _ in orc.clusters(R, Os[0])))
        msgs.append(f"{checked} clusters, max|sum mult*<O> - Tr(P_E O)| {worst:.1e}")
        if not ok and conj_ok:
            msgs.append("CONJ (the values match the complex conjugate)")
        return ok, worst, "; ".join(msgs)

    out = best_match(ctx, check)
    # matrix elements between the returned partners (not under a total-spin restriction)
    if ctx.req["total_spin"] is None and out["status"] == "pass":
        spec = t["me_op"]
        Ome = ctx.to_op(op_terms(spec, ctx.model), ctx.N)
        Od = sparse_op(op_terms(spec, ctx.model), ctx.N)
        ev = r.eigs
        n = min(3, len(ev.levels))
        try:
            full = [np.asarray(qcall(ev._raw.multiplet, ev._spec, i, -1, 1)[0], complex) for i in range(n)]
            me = 0.0
            for i in range(n):
                for j in range(n):
                    got = complex(qcall(ev.matrix_element, Ome, i, j))
                    me = max(me, abs(got - np.vdot(full[i], Od @ full[j])))
            if me > 1e-7:
                out = mk("wrong", me, f"matrix_element off by {me:.1e} ({spec['kind']})")
            else:
                out["message"] += f"; matrix elements {me:.1e}"
        except QedRaised as e:
            st, msg, ex = classify(e.exc, ctx.qed)
            out = mk(st, None, f"matrix_element: {msg}", **ex)
    if out["status"] != "pass":
        extra["mult_deficit"] = multiplet_deficit(r.eigs, k)
    out["extra"].update(extra)
    return out


def _th_errors(got, ref, N, keys=("E", "C")):
    field_of = {"S": "entropy"}
    return {q: float(np.max(np.abs(np.asarray(getattr(got, field_of.get(q, q))) - ref[q]))) / N for q in keys}


def t_th_exact(ctx):
    T = ctx.task["T"]
    sym = ctx.sym()
    if len(ctx.orc.energies(ctx.base_R())) == 0:
        return empty_case(ctx, lambda: qcall(ctx.qed.thermal, ctx.H, T, method="exact", sym=sym, device=ctx.device))
    r = qcall(ctx.qed.thermal, ctx.H, T, method="exact", sym=sym, device=ctx.device)
    extra = dict(blocks=int(r.blocks), **result_extra(ctx, r))

    def check(R):
        ref = ctx.orc.thermo(R, T)
        errs = []
        for q, f in (("E", "E"), ("S", "entropy"), ("F", "F"), ("lnZ", "lnZ")):
            d = float(np.max(np.abs(np.asarray(getattr(r, f)) - ref[q])) / max(1.0, float(np.max(np.abs(ref[q])))))
            errs.append((q, d, 1e-8))
        dC = float(np.max(np.abs(np.asarray(r.C) - ref["C"])) / max(1.0, float(np.max(np.abs(ref["C"])))))
        errs.append(("C", dC, 1e-7))
        bad = [f"{q} {d:.1e}" for q, d, tol in errs if not d <= tol]
        worst = max(d for _, d, _ in errs)
        return (not bad), worst, ("bad: " + ", ".join(bad)) if bad else f"max rel {worst:.1e}"

    out = best_match(ctx, check)
    _check_M(ctx, r, out, T)
    out["extra"].update(extra)
    return out


def _check_M(ctx, r, out, T):
    """M / chi against the reference. Documented: present when H conserves Sz and the symmetry
    decomposes by it (None under sz='off')."""
    if out["status"] != "pass":
        return
    R = ctx.base_R()
    ref = ctx.orc.thermo(R, T)
    want = ref["M"] is not None and ctx.req["sz"] != "off"
    if not want:
        if r.M is not None and ref["M"] is None:
            out.update(mk("wrong", None, "M present for an H that does not conserve Sz"))
        return
    if r.M is None:
        out.update(mk("wrong", None, "M absent although H conserves Sz and sz is not 'off'"))
        return
    dM = float(np.max(np.abs(np.asarray(r.M) - ref["M"])))
    dchi = float(np.max(np.abs(np.asarray(r.chi) - ref["chi"])))
    tol = 1e-7 if ctx.task["kind"] == "th_exact" else 0.05 * ctx.N
    if dM > tol or dchi > max(tol, 1e-7):
        flip = float(np.max(np.abs(np.asarray(r.M) + ref["M"])))
        msg = f"M off by {dM:.2e} (sign-flipped diff {flip:.1e}), chi off by {dchi:.2e}"
        out.update(mk("wrong", max(dM, dchi), msg))


SAMPLED_TOL = {"ftlm": {"E": 0.01, "C": 0.02}, "mtpq": {"E": 0.02, "C": 0.04, "S": 0.03, "F": 0.03}}


def _sampled(ctx, method, extra_kw, tol, R_samples):
    """Sampled thermodynamics vs the dense reference (CPU), or the GPU path vs the CPU path at
    the same seed (GPU). tol: {quantity: per-site tolerance}."""
    t = ctx.task
    T = t["T"]
    sym = ctx.sym()
    N = ctx.N
    # GPU: always sample (a dense block solve would make the device-vs-host comparison vacuous)
    dmd = 0 if (ctx.device == "gpu" or not t.get("exact_small", True)) else None

    def run(dev, samples):
        return qcall(ctx.qed.thermal, ctx.H, T, method=method, sym=sym, samples=samples,
                     krylov=None if method == "mtpq" else t["krylov"], seed=t["seed"], device=dev,
                     dense_max_dim=dmd, **extra_kw)

    if len(ctx.orc.energies(ctx.base_R())) == 0:
        return empty_case(ctx, lambda: run(ctx.device, 4)), None
    if ctx.device == "gpu":   # OFTLM too: it has a device lane since P7.5
        g = run("gpu", 4)
        c = run("cpu", 4)
        d = max(float(np.max(np.abs(np.asarray(getattr(g, q)) - np.asarray(getattr(c, q))))) / N
                for q in ("E", "C", "entropy", "F", "lnZ"))
        st = "pass" if d < 1e-8 else "wrong"
        return mk(st, d, f"gpu vs cpu at seed {t['seed']}, 4 samples: {d:.1e}", **result_extra(ctx, g)), None
    r1 = run("cpu", R_samples)
    extra = dict(blocks=int(r1.blocks), **result_extra(ctx, r1))
    keys = tuple(tol)

    def check(R):
        ref = ctx.orc.thermo(R, T)
        e = _th_errors(r1, ref, N, keys)
        if e["E"] < 1e-8 and e["C"] < 1e-7:
            return True, max(e.values()), "exact (small-block dense solve)"
        ok = all(e[q] < tol[q] for q in keys)
        return ok, max(e.values()), f"R={R_samples}: " + ", ".join(f"d{q}/N {e[q]:.2e}" for q in keys)

    out = best_match(ctx, check)
    if out["status"] == "wrong" and "reading" not in out["extra"]:
        r4 = run("cpu", 4 * R_samples)
        ref = ctx.orc.thermo(ctx.base_R(), T)
        e1 = out["metric"] if out["metric"] is not None else math.inf
        e = _th_errors(r4, ref, N, keys)
        e4 = max(e.values())
        if all(e[q] < tol[q] for q in keys) or e4 < 0.65 * e1:
            out = mk("pass", e4, f"noise: R={R_samples} err {e1:.2e} -> R={4 * R_samples} err {e4:.2e}")
        else:
            out["message"] += f"; R={4 * R_samples}: {e4:.2e} (does not shrink like noise)"
    out["extra"].update(extra)
    return out, r1


def t_th_ftlm(ctx):
    out, r = _sampled(ctx, "ftlm", {}, SAMPLED_TOL["ftlm"], 50)
    if r is not None:
        _check_M(ctx, r, out, ctx.task["T"])
    return out


def t_th_oftlm(ctx):
    out, _ = _sampled(ctx, "ftlm", {"exact_states": int(ctx.task["exact_states"])}, SAMPLED_TOL["ftlm"], 40)
    return out


def t_th_mtpq(ctx):
    out, _ = _sampled(ctx, "mtpq", {}, SAMPLED_TOL["mtpq"], 16)
    return out


def t_th_obs(ctx):
    t = ctx.task
    T = t["T"]
    sym = ctx.sym()
    specs = t["ops"]
    ops = [ctx.to_op(op_terms(s, ctx.model), ctx.N) for s in specs]
    Os = [sparse_op(op_terms(s, ctx.model), ctx.N) for s in specs]
    dmd = 0 if t["method"] == "ftlm" else None          # FTLM always samples here

    def run(dev, samples):
        return qcall(ctx.qed.thermal, ctx.H, T, method=t["method"], sym=sym, samples=samples, krylov=60,
                     seed=t["seed"], device=dev, observables=ops, dense_max_dim=dmd)

    if len(ctx.orc.energies(ctx.base_R())) == 0:
        return empty_case(ctx, lambda: run(ctx.device, 4))
    if t["method"] == "ftlm" and ctx.device == "gpu":
        g, c = run("gpu", 4), run("cpu", 4)
        d = float(np.max(np.abs(np.asarray(g.O) - np.asarray(c.O))))
        return mk("pass" if d < 1e-8 else "wrong", d, f"gpu vs cpu <O>(T), 4 samples: {d:.1e}", **result_extra(ctx, g))
    samples = 1 if t["method"] == "exact" else 50
    r = run(ctx.device, samples)
    got = np.asarray(r.O, complex)
    tol = 1e-8 if t["method"] == "exact" else 0.02

    def check(R):
        ref = np.array([ctx.orc.thermal_expect(R, O, T) for O in Os])
        err = float(np.max(np.abs(got - ref)))
        cerr = float(np.max(np.abs(np.conj(got) - ref)))
        return err < tol, err, f"max|dO| {err:.2e} (conjugate {cerr:.1e})"

    out = best_match(ctx, check)
    if out["status"] == "wrong" and t["method"] == "ftlm" and "reading" not in out["extra"]:
        r4 = run("cpu", 200)
        ref = np.array([ctx.orc.thermal_expect(ctx.base_R(), O, T) for O in Os])
        e4 = float(np.max(np.abs(np.asarray(r4.O) - ref)))
        e1 = out["metric"] or math.inf
        if e4 < tol or e4 < 0.65 * e1:
            out = mk("pass", e4, f"noise: R=50 {e1:.2e} -> R=200 {e4:.2e}")
    out["extra"].update(blocks=int(r.blocks), **result_extra(ctx, r))
    return out


def _omega(ctx, eta):
    E = ctx.orc.energies({"sz": None, "S": None})
    span = float(E[-1] - E[0])
    return np.linspace(-max(1.0, 0.5 * span), span + 1.0, 401)


def _rel_l1(a, b, om):
    den = float(_trapz(np.abs(b), om))
    if den < 1e-12:
        return float(np.max(np.abs(a)))
    return float(_trapz(np.abs(a - b), om)) / den


def t_dyn(ctx):
    t = ctx.task
    T = t.get("T") if t["kind"] == "dynT" else None
    sym = ctx.sym(with_select=False)
    spec = t["probe"]
    O = ctx.to_op(op_terms(spec, ctx.model), ctx.N)
    Od = sparse_op(op_terms(spec, ctx.model), ctx.N)
    eta = float(t["eta"])
    om = _omega(ctx, eta)

    def run(dev, samples, krylov):
        return qcall(ctx.qed.dynamics, ctx.H, O, om, eta=eta, T=None if T is None else [T], sym=sym,
                     krylov=krylov, samples=samples, seed=t["seed"], device=dev)

    su2 = ctx.req["total_spin"] is not None
    # GPU T > 0: the device lane against the host lane at the same seed (one random stream for both;
    # the host lane is checked against the reference on the CPU shards). Under total_spin this also
    # keeps the case short: the device lane costs ~4.5 s per sample on 27 small blocks (host S^2 work
    # per block and sample, P2-gpu-09 / K2-task-backend-07).
    if T is not None and ctx.device == "gpu":
        g, c = run("gpu", 4, 40), run("cpu", 4, 40)
        d = _rel_l1(np.asarray(g.S[0]), np.asarray(c.S[0]), om)
        msg = f"gpu vs cpu, 4 samples: rel L1 {d:.1e}"
        tol = 1e-6
        if d >= tol:
            # The unreorthogonalised Lanczos amplifies roundoff once Ritz values converge (ghosts),
            # worst when krylov nears the block dimension: the host lane's own response to a 1e-14
            # relative change of H measures it (seen 5.6e-4 at krylov 40 on a 64-state block, 1e-14
            # at 20 and 80). The lanes must agree to 10x that.
            Hs = ctx.H * (1.0 + 1e-14)
            cs = qcall(ctx.qed.dynamics, Hs, O, om, eta=eta, T=[T], sym=sym, krylov=40, samples=4,
                       seed=t["seed"], device="cpu")
            sens = _rel_l1(np.asarray(cs.S[0]), np.asarray(c.S[0]), om)
            tol = max(tol, 10 * sens)
            msg += f"; host sensitivity to a 1e-14 change of H {sens:.1e}, tolerance {tol:.1e}"
        return mk("pass" if d < tol else "wrong", d, msg, **result_extra(ctx, g))
    samples = int(t.get("samples", 40))
    krylov = int(t["krylov"])
    r = run(ctx.device, samples, krylov)
    S = np.real(np.asarray(r.S[0]))
    extra = dict(e0=float(r.e0), ground_manifold=int(r.ground_manifold), **result_extra(ctx, r))
    R0 = ctx.base_R()
    R0 = dict(R0, trans=None, mom=None, irrep=None)
    variants = [("main", R0, None, "main")]
    if su2 and T is None:
        variants.append(("gm_Sz=S_member_only", R0, ctx.req["total_spin"], "convention"))
    if ctx.req["sz"] in ("even", "odd") and ctx.content["u1"] and not su2:
        R3 = dict(R0)
        R3["sz"] = None
        variants.append(("parity_ignored", R3, None, "diagnostic"))
    tol = 0.02 if T is None else 0.2
    first = None
    for name, R, gm_top, kind in variants:
        ref, m0, m1, E0 = ctx.orc.dynamics(R, Od, om, eta, T=T, gm_top=gm_top)
        err = _rel_l1(S, ref, om)
        mq0, mr0 = float(_trapz(S, om)), float(_trapz(ref, om))
        mq1, mr1 = float(_trapz(S * om, om)), float(_trapz(ref * om, om))
        msg = (f"rel L1 {err:.3e}; m0 qed {mq0:.5f} ref {mr0:.5f} (exact {m0:.5f}); "
               f"m1 qed {mq1:.5f} ref {mr1:.5f} (exact {m1:.5f})")
        if T is None:
            msg += f"; e0 qed {r.e0:.10f} ref {E0:.10f}"
            if abs(r.e0 - E0) > 1e-7 * max(1, abs(E0)):
                err = max(err, abs(r.e0 - E0))
        if first is None:
            first = (err, msg, E0)
        if err < tol:
            if kind == "diagnostic":
                out = mk("wrong", err, f"matches '{name}' reading: {msg}", reading=name)
            else:
                out = mk("pass", err, ("" if kind == "main" else f"[convention {name}] ") + msg)
            break
    else:
        err, msg, E0 = first
        out = mk("wrong", err, msg)
        if T is not None:
            r4 = run("cpu", 4 * samples, krylov)
            ref, _, _, _ = ctx.orc.dynamics(R0, Od, om, eta, T=T)
            e4 = _rel_l1(np.asarray(r4.S[0]), ref, om)
            if e4 < tol or e4 < 0.65 * err:
                out = mk("pass", e4, f"noise: {samples} samples {err:.3f} -> {4 * samples} samples {e4:.3f}")
    out["extra"].update(extra)
    return out


def t_irrep_partition(ctx):
    sym = ctx.sym(with_select=False)
    ident = tuple(range(ctx.N))
    got, per, labels = [], {}, []
    for d in (1, 2, 3, 4, 6, 8, 12):
        try:
            r = qcall(ctx.qed.spectrum, ctx.H, sym=qcall(sym.select, irrep_character={ident: d}), device=ctx.device)
        except QedRaised as e:
            if isinstance(e.exc, ctx.qed.errors.EmptySelection):
                per[d] = 0
                continue
            raise
        per[d] = len(r.energies)
        got += list(np.asarray(r.energies, float))
        bad = [int(L.irrep_dim) for L in r.levels if int(getattr(L, "irrep_dim", d)) not in (d, 0)]
        if bad:
            labels.append((d, sorted(set(bad))))
    out = best_match(ctx, lambda R: cmp_multiset(got, ctx.orc.energies(R), 1e-8))
    out["message"] += f"; per dimension {per}"
    if labels and out["status"] == "pass":
        out = mk("wrong", out["metric"], f"levels with irrep_dim != selected dimension: {labels}; per {per}")
    return out


def _damaged_file(ctx, H, Sym):
    q = ctx.qed
    path = os.path.join(ctx.tmpdir, f"dmg_{ctx.case['case_id']}_{os.getpid()}.npz")
    r = q.eigs(H, 2, sym=Sym(spatial=None), vectors=True, device=ctx.device)
    r.save(path)
    with np.load(path) as f:
        d = {k: f[k] for k in f.files}
    big = max((k for k in d if d[k].ndim >= 1 and k not in ("energies",)), key=lambda k: d[k].size)
    d[big] = d[big].ravel()[: max(1, d[big].size // 2)]
    np.savez(path, **d)
    try:
        return q.load_eigs(path)
    finally:
        try:
            os.remove(path)
        except OSError:
            pass


def _format1_file(ctx, H, Sym):
    q = ctx.qed
    path = os.path.join(ctx.tmpdir, f"f1_{ctx.case['case_id']}_{os.getpid()}.npz")
    r = q.eigs(H, 1, sym=Sym(spatial=None), vectors=True, device=ctx.device)
    r.save(path)
    with np.load(path) as f:
        d = {k: f[k] for k in f.files}
    d["format_version"] = np.int64(1)
    np.savez(path, **d)
    try:
        return q.load_eigs(path)
    finally:
        try:
            os.remove(path)
        except OSError:
            pass


def t_invalid(ctx):
    q, t, N = ctx.qed, ctx.task, ctx.N
    kind = t["invalid"]
    H = ctx.H
    Sym = q.Symmetry
    nan = float("nan")

    def eigs(sym, **kw):
        return q.eigs(H, kw.pop("k", 2), sym=sym, device=ctx.device, **kw)

    def sz0():
        return ctx.to_op([(1.0, (("z", 0),))], N)

    trans = ctx.model["trans"]
    calls = {
        "flip_require": lambda: eigs(Sym(spatial=None, spin_flip="require")),
        "tr_require": lambda: eigs(Sym(spatial=None, time_reversal="require")),
        "su2_on_non_su2": lambda: eigs(Sym(spatial=None, total_spin=N % 2 / 2)),
        "bad_S": lambda: eigs(Sym(spatial=None, total_spin=t.get("S", N / 2 + 1))),
        "half_S": lambda: eigs(Sym(spatial=None, total_spin=0.3)),
        "sz_int_non_u1": lambda: eigs(Sym(spatial=None, sz=N // 2)),
        "parity_non_parity": lambda: eigs(Sym(spatial=None, sz="even")),
        "bad_perm": lambda: eigs(Sym(spatial=[t["perm"]], point_group=False)),
        "non_perm": lambda: eigs(Sym(spatial=[[0] * N])),
        "sz_out_of_range": lambda: eigs(Sym(spatial=None, sz=N + 1)),
        "sz_negative": lambda: eigs(Sym(spatial=None, sz=-1)),
        "sz_bool": lambda: eigs(Sym(spatial=None, sz=True)),
        "mom_not_in_group": lambda: eigs(Sym(spatial=ctx.spatial("split_T")).select(
            momentum={tuple(t["perm"]): 0}), k=1),
        "mom_nonexistent": lambda: eigs(Sym(spatial=ctx.spatial("split_T")).select(
            momentum={tuple(trans[0]): Fraction(3, 10) if ctx.model["orders"][0] % 10 else Fraction(1, 7)}), k=1),
        "irrep_bad_R": lambda: eigs(Sym(spatial=ctx.spatial("split_T")).select(irrep_character={tuple(trans[0]): 1})),
        "bad_device": lambda: q.eigs(H, 1, device="tpu"),
        "bad_method": lambda: q.thermal(H, [1.0], method="kpm"),
        "dyn_T_nonpositive": lambda: q.dynamics(H, sz0(), [0.0, 1.0], T=[0.0], device=ctx.device),
        "k_zero": lambda: eigs(Sym(spatial=None), k=0),
        "eta_nonpositive": lambda: q.dynamics(H, sz0(), np.linspace(0, 3, 31), eta=0.0, device=ctx.device),
        "obs_wrong_size": lambda: q.expect(H, [ctx.to_op([(1.0, (("z", N + 1),))], N + 2)], 1, sym=Sym(spatial=None),
                                           device=ctx.device),
        "none_in_ops": lambda: q.expect(H, [None], 1, sym=Sym(spatial=None), device=ctx.device),
        "non_hermitian_H": lambda: q.eigs(ctx.to_op(list(ctx.model["terms"]) + [(0.3, (("+", 0),))], N), 2,
                                          sym=Sym(spatial=None), device=ctx.device),
        "negative_window": lambda: eigs(Sym(spatial=None), window=-1.0),
        # a spin-S tower is solved at n_up = N/2 + S; this n_up has |Sz| > S
        "sz_su2_disagree": lambda: eigs(Sym(spatial=None, sz=N // 2 + 1 + N % 2, total_spin=N % 2 / 2)),
        "T_empty": lambda: q.thermal(H, [], method="exact", device=ctx.device),
        "T_nonfinite": lambda: q.thermal(H, [1.0, nan], method="exact", device=ctx.device),
        "omega_nonfinite": lambda: q.dynamics(H, sz0(), [0.0, nan, 1.0], device=ctx.device),
        "thermal_krylov_zero": lambda: q.thermal(H, [1.0], method="ftlm", krylov=0, device=ctx.device),
        "thermal_samples_zero": lambda: q.thermal(H, [1.0], method="ftlm", samples=0, device=ctx.device),
        "mtpq_krylov": lambda: q.thermal(H, [1.0], method="mtpq", krylov=10, device=ctx.device),
        "ftlm_steps": lambda: q.thermal(H, [1.0], method="ftlm", steps=10, device=ctx.device),
        "dyn_krylov_zero": lambda: q.dynamics(H, sz0(), [0.0, 1.0], krylov=0, device=ctx.device),
        "dyn_samples_zero": lambda: q.dynamics(H, sz0(), [0.0, 1.0], T=[1.0], samples=0, device=ctx.device),
        "neg_degeneracy_tol": lambda: q.dynamics(H, sz0(), [0.0, 1.0], degeneracy_tol=-1.0, device=ctx.device),
        "dense_max_dim_negative": lambda: eigs(Sym(spatial=None), dense_max_dim=-1),
        "bad_spatial_string": lambda: eigs(Sym(spatial="everything")),
        "bad_toggle": lambda: eigs(Sym(spatial=None, spin_flip="maybe")),
        "thermal_obs_wrong_size": lambda: q.thermal(H, [1.0], method="exact", device=ctx.device,
                                                    observables=[ctx.to_op([(1.0, (("z", N),))], N + 1)]),
        "dyn_obs_wrong_size": lambda: q.dynamics(H, ctx.to_op([(1.0, (("z", N),))], N + 1), [0.0, 1.0],
                                                 device=ctx.device),
        "load_damaged": lambda: _damaged_file(ctx, H, Sym),
        "load_format1": lambda: _format1_file(ctx, H, Sym),
        "nonfinite_H": lambda: q.eigs(ctx.to_op(list(ctx.model["terms"]) + [(nan, (("z", 0), ("z", 1)))], N), 2,
                                      sym=Sym(spatial=None), device=ctx.device),
    }
    try:
        res = calls[kind]()
    except CaseTimeout:
        raise
    except Exception as e:  # noqa: BLE001
        cls = type(e).__name__
        msg = f"{cls}: {e}"
        is_q = isinstance(e, q.errors.QEDError)
        if not str(e).strip():
            return mk("invalid_bad", None, f"{kind}: raised without a message: {msg}", error_class=cls, qed_error=is_q)
        if not is_q and not isinstance(e, INVALID_BUILTIN_OK.get(kind, ())):
            return mk("invalid_bad", None, f"{kind}: refused with builtin {cls}, not a qed.errors class: {e}",
                      error_class=cls, qed_error=False)
        return mk("invalid_ok", None, f"{kind}: {msg}", error_class=cls, qed_error=is_q)
    desc = type(res).__name__
    for a in ("energies", "S", "values"):
        v = getattr(res, a, None)
        if v is not None:
            arr = np.asarray(v)
            desc += f" {a}.shape={arr.shape} finite={bool(np.all(np.isfinite(arr)))}"
            if a == "energies":
                desc += f" first={np.round(arr[:3], 6).tolist()}"
    if kind == "nonfinite_H" and getattr(res, "energies", None) is not None:
        # what was solved: the lowest energies of H with the NaN term left out
        ref = np.sort(np.concatenate([np.linalg.eigvalsh(ctx.orc.H[b][:, b].toarray())
                                      for b in ctx.orc.natural_blocks()]))[: len(res.energies)]
        same = bool(np.allclose(np.sort(np.asarray(res.energies, float)), ref, atol=1e-8))
        desc += f"; H without the NaN term: {np.round(ref[:3], 6).tolist()} ({'equal' if same else 'different'})"
    return mk("invalid_bad", None, f"{kind}: returned {desc} without error")


DISPATCH = {"eigs": t_eigs, "spectrum": t_spectrum, "vectors": t_vectors, "expect": t_expect,
            "th_exact": t_th_exact, "th_ftlm": t_th_ftlm, "th_oftlm": t_th_oftlm, "th_mtpq": t_th_mtpq,
            "th_obs": t_th_obs, "dyn0": t_dyn, "dynT": t_dyn, "irrep_partition": t_irrep_partition,
            "invalid": t_invalid}

# One symmetry toggle at a time, rerun on a failing case: which ones make it pass (a signature).
VARIANTS = (("time_reversal", "off"), ("point_group", False), ("spin_flip", "off"))


def _diagnose(ctx, kind):
    passes = []
    for key, val in VARIANTS:
        if ctx.req.get(key) == val:
            continue
        if key == "point_group" and ctx.req.get("spatial") is None:
            continue
        try:
            o2 = DISPATCH[kind](ctx.variant(key, val))
        except CaseTimeout:
            raise
        except Exception:  # noqa: BLE001
            continue
        if o2["status"] == "pass":
            passes.append(f"{key}={val}")
    return passes


def _alarm(signum, frame):
    raise CaseTimeout()


def run_case(case, qed, tmpdir, soft_timeout=None, diagnose=True):
    t0 = time.time()
    rec = {k: case[k] for k in ("case_id", "seed", "index", "device", "model", "content", "request", "task", "env")}
    old = {}
    out = None
    ctx = None
    if soft_timeout:
        signal.signal(signal.SIGALRM, _alarm)
        signal.alarm(max(1, int(soft_timeout)))
    try:
        for k, v in case["env"].items():
            old[k] = os.environ.get(k)
            os.environ[k] = str(v)
        ctx = Ctx(case, qed, tmpdir)
        kind = case["task"]["kind"]
        try:
            out = DISPATCH[kind](ctx)
        except QedRaised as e:
            st, msg, ex = classify(e.exc, qed)
            if case["device"] == "gpu" and isinstance(e.exc, qed.errors.DeviceUnavailable):
                msg += " [device unavailable]"
            out = mk(st, None, msg, **ex)
        if ctx.notes:
            out["message"] = f"{out['message']} | {'; '.join(ctx.notes)}"
        if diagnose and kind != "invalid" and out["status"] in ("wrong", "refused", "crash"):
            out["extra"]["passes_with"] = _diagnose(ctx, kind)
    except CaseTimeout:
        out = mk("timeout", None, f"soft alarm after {soft_timeout} s")
    except SkipCase as e:
        out = mk("skip", None, str(e), e.known_id)
    except QedRaised as e:          # building H itself
        st, msg, ex = classify(e.exc, qed)
        out = mk(st, None, f"building H: {msg}", **ex)
    except HarnessError as e:
        out = mk("harness_error", None, f"HarnessError: {e}")
    except Exception as e:  # noqa: BLE001
        out = mk("harness_error", None, f"{type(e).__name__}: {e} | {traceback.format_exc(limit=6)}")
    finally:
        if soft_timeout:
            signal.alarm(0)
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    rec.update(out)
    rec["seconds"] = round(time.time() - t0, 3)
    return rec


# =============================================================================
# Accepted known failures (known.json)
# =============================================================================

def _field(rec, path):
    cur = rec
    for part in path.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return None
    return cur


def _pred(value, spec):
    """A predicate on one record field: a list (value in it), {"re": pattern} (regex search on
    the string form), {"contains": x} (x in a list value), {"absent": true}, {"lt"/"gt": x},
    or a scalar (equality)."""
    if isinstance(spec, list):
        return value in spec
    if isinstance(spec, dict):
        ok = True
        if "re" in spec:
            ok &= value is not None and re.search(spec["re"], str(value)) is not None
        if "contains" in spec:
            ok &= isinstance(value, (list, tuple)) and spec["contains"] in value
        if "nonempty" in spec:
            ok &= bool(value) == bool(spec["nonempty"])
        if "absent" in spec:
            ok &= (value is None) == bool(spec["absent"])
        if "lt" in spec:
            ok &= isinstance(value, (int, float)) and value < spec["lt"]
        if "gt" in spec:
            ok &= isinstance(value, (int, float)) and value > spec["gt"]
        return ok
    return value == spec


def load_known(path, manifest):
    """(entries, problems): the accepted failures, and why some are not acceptable (a ledger id
    that is not open in the regression manifest)."""
    if not path or not os.path.exists(path):
        return [], []
    with open(path) as f:
        data = json.load(f)
    entries = list(data.get("accepted", []))
    problems = []
    open_ids = None
    if manifest and os.path.exists(manifest):
        with open(manifest) as f:
            open_ids = {e["id"] for e in json.load(f) if e.get("status") == "open"}
    for e in entries:
        if not e.get("ledger"):
            problems.append(f"entry without a ledger id: {e.get('name', '?')}")
        elif open_ids is not None and e["ledger"] not in open_ids:
            problems.append(f"{e.get('name', '?')}: ledger id {e['ledger']} is not open in {manifest}")
        if not e.get("match") and not e.get("cases"):
            problems.append(f"{e.get('name', '?')}: neither 'match' nor 'cases'")
    return entries, problems


def match_known(rec, entries):
    """The first accepted entry that explains a non-pass record: (ledger id, entry name) or None."""
    for e in entries:
        for c in e.get("cases", []):
            if [rec.get("seed"), rec.get("index"), rec.get("device")] == list(c):
                return e["ledger"], e.get("name", "")
        m = e.get("match")
        if m and all(_pred(_field(rec, k), v) for k, v in m.items()):
            return e["ledger"], e.get("name", "")
    return None


# =============================================================================
# Process isolation
# =============================================================================

def _worker_main(conn, logpath, tmpdir, diagnose):
    try:
        fd = os.open(logpath, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o640)
        os.dup2(fd, 1)
        os.dup2(fd, 2)
    except OSError:
        pass
    try:
        import qed
        ndev = None
        f = getattr(qed._core, "cuda_device_count", None)
        if f is not None:
            try:
                ndev = int(f())
            except Exception:  # noqa: BLE001
                ndev = None
        conn.send({"ready": True, "qed_file": qed.__file__, "version": getattr(qed, "__version__", "?"),
                   "cuda_build": bool(qed.has_cuda_build()), "cuda_devices": ndev,
                   "env": dict(qed.env_snapshot()) if hasattr(qed, "env_snapshot") else {}})
    except Exception as e:  # noqa: BLE001
        conn.send({"ready": False, "error": f"{type(e).__name__}: {e}"})
        return
    while True:
        try:
            msg = conn.recv()
        except EOFError:
            return
        if msg is None:
            return
        case, soft = msg
        print(f"=== case {case['case_id']} {case['task']['kind']}", flush=True)
        conn.send(run_case(case, qed, tmpdir, soft_timeout=soft, diagnose=diagnose))


class Worker:
    def __init__(self, logpath, tmpdir, diagnose=True, startup=300):
        ctx = mp.get_context("spawn")
        self.conn, child = ctx.Pipe()
        self.p = ctx.Process(target=_worker_main, args=(child, logpath, tmpdir, diagnose), daemon=True)
        self.p.start()
        child.close()
        if not self.conn.poll(startup):
            self.kill()
            raise RuntimeError("worker did not start in time")
        self.info = self.conn.recv()
        if not self.info.get("ready"):
            self.kill()
            raise RuntimeError(f"worker failed to import qed: {self.info.get('error')}")

    def run(self, case, timeout):
        try:
            self.conn.send((case, max(5, int(timeout) - 15)))
        except (BrokenPipeError, OSError) as e:
            return None, f"worker pipe broken: {e}"
        if self.conn.poll(timeout):
            try:
                return self.conn.recv(), None
            except (EOFError, OSError):
                self.p.join(10)
                return None, f"worker died (exit code {self.p.exitcode})"
        self.kill()
        return None, "timeout"

    def kill(self):
        try:
            self.p.kill()
            self.p.join(10)
        except Exception:  # noqa: BLE001
            pass

    def close(self):
        try:
            self.conn.send(None)
            self.p.join(10)
        except Exception:  # noqa: BLE001
            self.kill()


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (complex, np.complexfloating)):
        return [float(o.real), float(o.imag)]
    if isinstance(o, Fraction):
        return [o.numerator, o.denominator]
    return str(o)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--cases", type=int, default=150)
    ap.add_argument("--device", choices=("cpu", "gpu"), default="cpu")
    ap.add_argument("--out", default=".")
    ap.add_argument("--budget-seconds", type=float, default=3000.0,
                    help="stop drawing new cases once this much wall time is spent")
    ap.add_argument("--case-timeout", type=float, default=240.0, help="hard per-case limit (worker killed)")
    ap.add_argument("--families", default="", help="comma list restricting the model families")
    ap.add_argument("--tasks", default="", help="comma list restricting the tasks")
    ap.add_argument("--start", type=int, default=0, help="first case index")
    ap.add_argument("--known", default=os.path.join(HERE, "known.json"),
                    help="accepted known failures ('' for none)")
    ap.add_argument("--manifest", default=os.path.join(HERE, "..", "regress", "manifest.json"),
                    help="regression manifest: known.json may only name ledger ids open there")
    ap.add_argument("--strict", action="store_true",
                    help="exit 1 on any unexplained non-pass record or harness_error, 2 on a stale known.json")
    ap.add_argument("--no-diagnose", action="store_true", help="skip the toggle reruns of failing cases")
    ap.add_argument("--inprocess", action="store_true", help="no worker process (debug; a crash kills the run)")
    ap.add_argument("--replay", default="", help="SEED-INDEX: rerun one case in-process and print the record")
    a = ap.parse_args(argv)
    fams = [f for f in a.families.split(",") if f] or None
    tasks = [t for t in a.tasks.split(",") if t] or None
    os.makedirs(a.out, exist_ok=True)
    tmpdir = tempfile.mkdtemp(prefix="qedfuzz_", dir=a.out)
    known, kproblems = load_known(a.known, a.manifest)
    for p in kproblems:
        print(f"fuzz: known.json: {p}", file=sys.stderr)

    if a.replay:
        s, i = (int(x) for x in a.replay.split("-"))
        case = generate_case(s, i, fams, tasks)
        case["device"] = a.device
        import qed
        rec = run_case(case, qed, tmpdir, soft_timeout=int(a.case_timeout), diagnose=not a.no_diagnose)
        if rec["status"] not in OK_STATUSES:
            m = match_known(rec, known)
            rec["known_id"], rec["known_entry"] = (m[0], m[1]) if m else (None, None)
        print(json.dumps(rec, default=_json_default, indent=1))
        return 0

    path = os.path.join(a.out, f"fuzz_{a.device}_{a.seed}.jsonl")
    logpath = os.path.join(a.out, f"worker_{a.device}_{a.seed}.log")
    t0 = time.time()
    counts = {s: 0 for s in STATUSES}
    known_n, not_engaged, done = 0, 0, 0
    unexplained, known_hits = [], {}
    worker, qed_mod = None, None
    meta = {}
    if a.inprocess:
        import qed as qed_mod
        meta = {"qed_file": qed_mod.__file__, "version": getattr(qed_mod, "__version__", "?")}
    with open(path, "w") as fh:
        for i in range(a.start, a.start + a.cases):
            left = a.budget_seconds - (time.time() - t0)
            if left <= 5:
                break
            tc = time.time()
            try:
                case = generate_case(a.seed, i, fams, tasks)
            except Exception as e:  # noqa: BLE001
                rec = {"case_id": f"{a.seed}-{i}", "seed": a.seed, "index": i, "device": a.device,
                       "status": "harness_error", "metric": None, "known_id": None, "extra": {},
                       "message": f"generation failed: {type(e).__name__}: {e} | {traceback.format_exc(limit=4)}"}
                case = None
            if case is not None:
                case["device"] = a.device
                timeout = max(20.0, min(a.case_timeout, left + 30))
                if a.inprocess:
                    rec = run_case(case, qed_mod, tmpdir, soft_timeout=int(timeout), diagnose=not a.no_diagnose)
                else:
                    if worker is None:
                        try:
                            worker = Worker(logpath, tmpdir, diagnose=not a.no_diagnose)
                            meta = worker.info
                        except Exception as e:  # noqa: BLE001
                            print(f"fuzz: cannot start a worker ({e}); stopping", file=sys.stderr)
                            counts["harness_error"] += 1
                            unexplained.append(f"{a.seed}-{i} harness_error: no worker ({e})")
                            break
                        if a.device == "gpu" and not meta.get("cuda_devices"):
                            print("fuzz: --device gpu but the worker sees no CUDA device; stopping", file=sys.stderr)
                            counts["harness_error"] += 1
                            unexplained.append(f"{a.seed}-{i} harness_error: no CUDA device")
                            break
                    rec, err = worker.run(case, timeout)
                    if rec is None:
                        rec = {k: case[k] for k in ("case_id", "seed", "index", "device", "model", "content",
                                                    "request", "task", "env")}
                        st = "timeout" if err == "timeout" else "crash"
                        rec.update(status=st, metric=None, known_id=None, extra={},
                                   message=f"{err} (hard limit {timeout:.0f} s)" if st == "timeout" else err)
                        worker = None
                    elif case.get("restart_after"):
                        worker.close()
                        worker = None
            rec["wall_seconds"] = round(time.time() - tc, 3)
            st = rec.get("status", "harness_error")
            rec["known_id"], rec["known_entry"] = None, None
            if st not in OK_STATUSES:
                m = match_known(rec, known) if st != "harness_error" else None
                if m:
                    rec["known_id"], rec["known_entry"] = m
                    known_n += 1
                    known_hits[m[1] or m[0]] = known_hits.get(m[1] or m[0], 0) + 1
                else:
                    unexplained.append(f"{rec['case_id']} {st}: {str(rec.get('message', ''))[:160]}")
            fh.write(json.dumps(rec, default=_json_default) + "\n")
            fh.flush()
            counts[st] = counts.get(st, 0) + 1
            not_engaged += rec.get("extra", {}).get("device_engaged") is False
            done += 1
    if worker is not None:
        worker.close()
    seconds = round(time.time() - t0, 1)
    with open(os.path.join(a.out, f"fuzz_{a.device}_{a.seed}.meta.json"), "w") as fm:
        json.dump({"argv": sys.argv, "seed": a.seed, "device": a.device, "cases_run": done,
                   "cases_requested": a.cases, "budget_exhausted": done < a.cases,
                   "seconds": seconds, "library": meta, "counts": counts, "known_hits": known_hits,
                   "unexplained": unexplained, "known_problems": kproblems},
                  fm, default=_json_default, indent=1)
    try:
        os.rmdir(tmpdir)
    except OSError:
        pass
    summ = " ".join(f"{k}={v}" for k, v in counts.items() if v)
    print(f"fuzz seed={a.seed} device={a.device} cases={done}/{a.cases} {summ} known={known_n} "
          f"unexplained={len(unexplained)} gpu_not_engaged={not_engaged} seconds={seconds:.0f} out={path}")
    for u in unexplained:
        print(f"  UNEXPLAINED {u}")
    for name, n in sorted(known_hits.items()):
        print(f"  known {name}: {n}")
    if a.strict:
        if kproblems:
            return 2
        if unexplained:
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
