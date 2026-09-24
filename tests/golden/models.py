"""Model zoo for the golden-master harness.

Everything here is a term list in the ``reference.py`` vocabulary
(``(ops, sites, coeff)`` with ops in {"+", "-", "z"}), so one description feeds the
library and the independent dense reference alike.

Couplings are dyadic rationals wherever a case is compared at 1e-10: the text
transport of the symmetry lanes keeps 9 significant digits, and a dyadic coupling
survives it exactly.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from reference import (Model, chain_bonds, chiral_terms,
                       heisenberg_terms, triangular_torus)


# -----------------------------------------------------------------------------
# Tilted triangular torus with its translation group and point group
# -----------------------------------------------------------------------------
@dataclass
class TriangularTorus:
    """Triangular lattice, a1=(1,0), a2=(1/2, sqrt3/2), wrapped on the superlattice
    spanned by T1=(a,b), T2=(c,d) (integer coordinates in the (a1,a2) basis)."""
    T1: tuple
    T2: tuple
    sites: list = field(default_factory=list)      # canonical (n1, n2) per index
    index: dict = field(default_factory=dict)

    def __post_init__(self):
        (a, b), (c, d) = self.T1, self.T2
        self.det = a * d - b * c
        if self.det == 0:
            raise ValueError("degenerate superlattice")
        span = abs(a) + abs(b) + abs(c) + abs(d) + 1
        seen = {}
        for n2 in range(-span, span + 1):
            for n1 in range(-span, span + 1):
                key = self._key(n1, n2)
                if key not in seen:
                    seen[key] = (n1, n2)
        if len(seen) != abs(self.det):
            raise RuntimeError(f"found {len(seen)} cells, expected {abs(self.det)}")
        for i, key in enumerate(sorted(seen)):
            self.index[key] = i
            self.sites.append(seen[key])
        self.N = len(self.sites)

    def _key(self, n1, n2):
        (a, b), (c, d) = self.T1, self.T2
        det = self.det
        x = (n1 * d - n2 * c)
        y = (-n1 * b + n2 * a)
        if det < 0:
            x, y, det = -x, -y, -det
        return (x % det, y % det)

    def site(self, n1, n2):
        return self.index[self._key(n1, n2)]

    def _bonds(self, offsets):
        """One bond per (site, forward offset). NOT de-duplicated by site pair: on a
        small torus two offsets can wrap onto the same pair (the 12-site cluster's
        second shell does), and the torus Hamiltonian carries that bond twice."""
        out = []
        for i, (n1, n2) in enumerate(self.sites):
            for (d1, d2) in offsets:
                j = self.site(n1 + d1, n2 + d2)
                if i != j:
                    out.append((i, j))
        return out

    def nn(self):
        return self._bonds([(1, 0), (0, 1), (-1, 1)])

    def nnn(self):
        return self._bonds([(1, 1), (-1, 2), (2, -1)])

    def _perm(self, fn):
        p = [self.site(*fn(n1, n2)) for (n1, n2) in self.sites]
        if sorted(p) != list(range(self.N)):
            return None
        return p

    def translation(self, t1, t2):
        return self._perm(lambda n1, n2: (n1 + t1, n2 + t2))

    def translation_group(self):
        """Closed translation group; element 1 and the first element that is not a
        power of it are usable as generators. Order: by (t2, t1) of the canonical cell."""
        perms, seen = [], set()
        for (t1, t2) in self.sites:
            p = self.translation(t1, t2)
            if tuple(p) not in seen:
                seen.add(tuple(p))
                perms.append(p)
        ident = list(range(self.N))
        perms.sort(key=lambda p: (p != ident, p))
        return perms

    def point_group(self):
        """Every C6v element about site (0,0) that is a bijection of the torus AND maps
        the NN and NNN shells onto themselves. Identity excluded."""
        c6 = lambda n1, n2: (-n2, n1 + n2)          # noqa: E731
        mir = lambda n1, n2: (n2, n1)               # noqa: E731
        shells = [set(map(frozenset, self.nn())), set(map(frozenset, self.nnn()))]
        ops = []
        for m in (0, 1):
            for r in range(6):
                if m == 0 and r == 0:
                    continue

                def fn(n1, n2, r=r, m=m):
                    if m:
                        n1, n2 = mir(n1, n2)
                    for _ in range(r):
                        n1, n2 = c6(n1, n2)
                    return n1, n2
                p = self._perm(fn)
                if p is None:
                    continue
                if all({frozenset((p[i], p[j])) for (i, j) in map(tuple, s)} == s for s in shells):
                    ops.append(p)
        return ops


def tri_j1j2(T1, T2, J2, name):
    tt = TriangularTorus(T1, T2)
    terms = heisenberg_terms(tt.nn(), 1.0) + heisenberg_terms(tt.nnn(), J2)
    m = Model(name, tt.N, terms, u1=True, real=True, su2=True)
    return m, tt


def square_j1j2(L, J2, name):
    def idx(x, y):
        return (x % L) + L * (y % L)
    nn, nnn = set(), set()
    for y in range(L):
        for x in range(L):
            i = idx(x, y)
            for j in (idx(x + 1, y), idx(x, y + 1)):
                nn.add((min(i, j), max(i, j)))
            for j in (idx(x + 1, y + 1), idx(x - 1, y + 1)):
                nnn.add((min(i, j), max(i, j)))
    terms = heisenberg_terms(sorted(nn), 1.0) + heisenberg_terms(sorted(nnn), J2)
    return Model(name, L * L, terms, u1=True, real=True, su2=True, lattice=(L, L))


def tri_chiral_3x3():
    """J1 + J_chi on the 3x3 triangular torus: complex, time-reversal broken, so the
    spectra at k and -k differ and a k -> -k relabel of sectors is visible."""
    nn, _, tri = triangular_torus(3, 3)
    terms = heisenberg_terms(nn, 1.0) + chiral_terms(tri, 0.5)
    return Model("tri_chiral_3x3", 9, terms, u1=True, real=False, su2=True, lattice=(3, 3))


# Open clusters in the shape the NLCE engine feeds to qed.full_spectrum: no
# translations, a small point group, all 2^N eigenvalues required.
def _open(name, bonds, n, Jz=1.0):
    return Model(name, n, heisenberg_terms(bonds, 1.0, Jz), u1=True, real=True, su2=(Jz == 1.0))


def nlce_clusters():
    out = []
    out.append(_open("nlce_triangle3", [(0, 1), (1, 2), (2, 0)], 3))
    out.append(_open("nlce_rhombus4", [(0, 1), (1, 2), (2, 3), (3, 0), (1, 3)], 4))
    out.append(_open("nlce_bowtie5", [(0, 1), (1, 2), (2, 0), (2, 3), (3, 4), (4, 2)], 5))
    hexb = [(0, i) for i in range(1, 7)] + [(i, i % 6 + 1) for i in range(1, 7)]
    out.append(_open("nlce_hexagon7", hexb, 7))
    out.append(_open("nlce_hexagon7_xxz", hexb, 7, Jz=0.75))
    tri6 = [(0, 1), (1, 2), (0, 3), (1, 3), (1, 4), (2, 4), (3, 4), (3, 5), (4, 5)]
    out.append(_open("nlce_triangle6", tri6, 6))
    ladder = [(i, i + 1) for i in range(0, 9)] + [(i, i + 2) for i in range(0, 8)]
    out.append(_open("nlce_strip10", ladder, 10))
    return out


def heis_chain(N):
    return Model(f"heis_chain{N}", N, heisenberg_terms(chain_bonds(N)), u1=True, real=True, su2=True)
