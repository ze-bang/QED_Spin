"""General (tilted) supercell kagome clusters for the BFG model.

A cluster is fixed by an integer 2x2 matrix L whose ROWS are the supercell vectors
in the Bravais basis:  T_i = L[i,0] a1 + L[i,1] a2,   a1=(1,0), a2=(1/2, sqrt3/2).
N_cell = |det L|, N = 3 N_cell.  L = diag(d1, d2) reproduces
edlib.helper_kagome_bfg.generate_kagome_cluster(d1, d2, use_pbc=True) site for site.

Bonds are generated ONLY from the offset tables of generate_kagome_cluster (never by
distance): at distance 1 there are two inequivalent pairs per site, and the BFG
hexagon term uses only the hexagon diagonal (NN3_BONDS), not the straight line
through a bowtie corner.

Momenta are written in the reciprocal basis:  q = x b1 + y b2,  q.a1 = 2 pi x,
q.a2 = 2 pi y, with b1, b2 at 120 degrees, so |q|^2 ~ x^2 + y^2 - x y.
High-symmetry points:  M1=(1/2,0)  M2=(0,1/2)  M3=(1/2,1/2)  K=(2/3,1/3)  K'=(1/3,2/3).

Clean cluster: no bond is double counted by the periodic wrap.  Bond displacements
have length <= 1 (a=1), so this holds iff the shortest supercell vector obeys
|T|^2 = x^2 + x y + y^2 >= 7 (Loeschian numbers jump from 4 to 7).  The builder
also asserts it directly (no duplicate bond, no self loop, coordination 4/4/2).
"""
from __future__ import annotations

import itertools
from fractions import Fraction

import numpy as np

A1 = np.array([1.0, 0.0])
A2 = np.array([0.5, np.sqrt(3) / 2])
SUB = np.array([[0.0, 0.0], [0.5, 0.0], [0.25, np.sqrt(3) / 4]])
HEX_CENTRE = np.array([0.75, np.sqrt(3) / 4])        # a C6v centre of the kagome net

# (source_sublattice, di, dj, target_sublattice) -- identical to helper_kagome_bfg
NN_BONDS = [(0, 0, 0, 1), (0, 0, 0, 2), (1, 0, 0, 2),
            (1, +1, 0, 0), (2, 0, +1, 0), (1, +1, -1, 2)]
NN2_BONDS = [(0, +1, -1, 2), (0, -1, +1, 1), (1, +1, 0, 2),
             (1, 0, +1, 0), (2, +1, 0, 0), (2, 0, +1, 1)]
NN3_BONDS = [(0, +1, -1, 0), (1, 0, +1, 1), (2, +1, 0, 2)]   # hexagon diagonals only

HIGH_SYMMETRY = {"G": (Fraction(0), Fraction(0)),
                 "M1": (Fraction(1, 2), Fraction(0)), "M2": (Fraction(0), Fraction(1, 2)),
                 "M3": (Fraction(1, 2), Fraction(1, 2)),
                 "K": (Fraction(2, 3), Fraction(1, 3)), "K'": (Fraction(1, 3), Fraction(2, 3))}


def bravais_norm2(x, y):
    """|x a1 + y a2|^2."""
    return x * x + x * y + y * y


def reciprocal_norm2(x, y):
    """|x b1 + y b2|^2 / |b|^2."""
    return x * x + y * y - x * y


class KagomeSupercell:
    def __init__(self, L):
        self.L = np.array(L, dtype=int).reshape(2, 2)
        det = int(round(np.linalg.det(self.L)))
        if det == 0:
            raise ValueError("singular supercell")
        self.n_cells = abs(det)
        self.N = 3 * self.n_cells
        self._Linv = np.linalg.inv(self.L.astype(float))
        self.T = np.array([self.L[i, 0] * A1 + self.L[i, 1] * A2 for i in range(2)])
        self.min_T2 = min(bravais_norm2(*(m1 * self.L[0] + m2 * self.L[1]))
                          for m1 in range(-4, 5) for m2 in range(-4, 5) if (m1, m2) != (0, 0))
        self.cells = self._enumerate_cells()
        self.cell_index = {c: n for n, c in enumerate(self.cells)}

    # ---- cells -----------------------------------------------------------------
    def reduce(self, r):
        """Canonical representative of integer cell vector r modulo the supercell."""
        s = np.asarray(r, float) @ self._Linv
        s -= np.floor(s + 1e-9)
        rr = np.rint(s @ self.L).astype(int)
        return (int(rr[0]), int(rr[1]))

    def _enumerate_cells(self):
        R = 2 * int(np.abs(self.L).sum()) + 2
        cells = sorted({self.reduce((i, j)) for i in range(-R, R + 1) for j in range(-R, R + 1)})
        if len(cells) != self.n_cells:
            raise RuntimeError(f"found {len(cells)} cells, expected {self.n_cells}")
        return cells

    def site(self, cell, sub):
        return 3 * self.cell_index[self.reduce(cell)] + sub

    def position(self, i):
        c, s = divmod(i, 3)
        r = self.cells[c]
        return r[0] * A1 + r[1] * A2 + SUB[s]

    # ---- bonds -----------------------------------------------------------------
    def bonds(self, table):
        """List of (i, j, dvec) with dvec = r_j - r_i the TRUE (unwrapped) bond vector."""
        out = []
        for r in self.cells:
            for (s, di, dj, t) in table:
                i = self.site(r, s)
                j = self.site((r[0] + di, r[1] + dj), t)
                d = di * A1 + dj * A2 + SUB[t] - SUB[s]
                out.append((i, j, d))
        return out

    def check_clean(self):
        problems = []
        for name, table, coord in (("NN", NN_BONDS, 4), ("2NN", NN2_BONDS, 4), ("3NN", NN3_BONDS, 2)):
            b = self.bonds(table)
            keys = [tuple(sorted((i, j))) for (i, j, _) in b]
            if any(i == j for (i, j, _) in b):
                problems.append(f"{name}: self loop")
            if len(set(keys)) != len(keys):
                problems.append(f"{name}: {len(keys) - len(set(keys))} duplicate bonds")
            deg = np.zeros(self.N, int)
            for (i, j) in set(keys):
                deg[i] += 1; deg[j] += 1
            if not np.all(deg == coord):
                problems.append(f"{name}: coordination {sorted(set(deg.tolist()))} != {coord}")
        return problems

    @property
    def is_clean(self):
        return self.min_T2 >= 7 and not self.check_clean()

    def bond_lists(self):
        """(nn, nn2, nn3) as sorted (i, j) pairs; raises if the cluster is not clean."""
        p = self.check_clean()
        if p:
            raise ValueError(f"cluster L={self.L.tolist()} is not clean: {p}")
        return tuple(sorted({tuple(sorted((i, j))) for (i, j, _) in self.bonds(t)})
                     for t in (NN_BONDS, NN2_BONDS, NN3_BONDS))

    # ---- symmetry ----------------------------------------------------------------
    def translation_perms(self):
        """One permutation per cell vector t (in self.cells order): site i -> site(i + t)."""
        perms = []
        for t in self.cells:
            p = np.empty(self.N, dtype=np.int64)
            for c, r in enumerate(self.cells):
                for s in range(3):
                    p[3 * c + s] = self.site((r[0] + t[0], r[1] + t[1]), s)
            perms.append(p.tolist())
        return perms

    def _locate(self, x):
        """Site index at real position x (mod supercell), or None."""
        B = np.array([A1, A2]).T
        for s in range(3):
            v = np.linalg.solve(B, x - SUB[s])
            if np.all(np.abs(v - np.rint(v)) < 1e-6):
                return self.site((int(np.rint(v[0])), int(np.rint(v[1]))), s)
        return None

    def point_group_perms(self):
        """C6v operations about a hexagon centre that are symmetries of THIS cluster
        (map sites bijectively and preserve all three bond shells).
        Returns list of (name, perm)."""
        shells = [set(b) for b in self.bond_lists()]
        ops = []
        for k in range(6):
            for mirror in (False, True):
                th = np.pi * k / 3
                R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
                if mirror:
                    R = R @ np.array([[1.0, 0.0], [0.0, -1.0]])
                p = []
                for i in range(self.N):
                    y = R @ (self.position(i) - HEX_CENTRE) + HEX_CENTRE
                    j = self._locate(y)
                    if j is None:
                        break
                    p.append(j)
                if len(p) != self.N or len(set(p)) != self.N:
                    continue
                if all({tuple(sorted((p[i], p[j]))) for (i, j) in sh} == sh for sh in shells):
                    ops.append((f"{'m' if mirror else 'C'}{6 if k == 0 else 6 // np.gcd(6, k)}_{k}", p))
        return ops

    # ---- momentum ----------------------------------------------------------------
    def momentum_grid(self):
        """All allowed momenta (x, y) as Fractions in [0,1): L @ (x,y) in Z^2."""
        det = int(round(np.linalg.det(self.L)))
        adj = np.array([[self.L[1, 1], -self.L[0, 1]], [-self.L[1, 0], self.L[0, 0]]])
        pts = set()
        for n1 in range(abs(det)):
            for n2 in range(abs(det)):
                v = adj @ np.array([n1, n2])
                q = (Fraction(int(v[0]), det) % 1, Fraction(int(v[1]), det) % 1)
                pts.add(q)
        pts = sorted(pts)
        if len(pts) != self.n_cells:
            raise RuntimeError(f"{len(pts)} momenta for {self.n_cells} cells")
        return pts

    @staticmethod
    def label_momentum(q):
        """'G','M1','M2','M3','K',"K'" or 'generic', exact rational comparison mod 1."""
        x, y = Fraction(q[0]) % 1, Fraction(q[1]) % 1
        for name, (hx, hy) in HIGH_SYMMETRY.items():
            if (x - hx) % 1 == 0 and (y - hy) % 1 == 0:
                return name
        return "generic"

    def momentum_content(self):
        labels = [self.label_momentum(q) for q in self.momentum_grid()]
        return {k: labels.count(k) for k in ("G", "M1", "M2", "M3", "K", "K'", "generic")}

    def bloch_phase(self, q, t):
        """exp(2 pi i q.t) for momentum q=(x,y) and cell vector t=(i,j)."""
        return np.exp(2j * np.pi * (float(q[0]) * t[0] + float(q[1]) * t[1]))

    # ---- flux ----------------------------------------------------------------------
    def supercell_fraction(self, d):
        """Real bond vector d expressed in the supercell basis: d = c1 T1 + c2 T2."""
        return np.linalg.solve(self.T.T, d)


def build_bfg_operator_supercell(L, Jpm, Jzz=1.0, flux=(0.0, 0.0), Jzz_2nn=None, Jzz_3nn=None):
    """BFG operator on supercell L.  Optional uniform Peierls phase
    th_ij = flux1 c1 + flux2 c2 on -Jpm e^{i th} S+_i S-_j  (same gauge as bfg_flux.py).
    Returns (operator, cluster)."""
    import qed
    cl = KagomeSupercell(L)
    nn, nn2, nn3 = cl.bond_lists()
    J2 = Jzz if Jzz_2nn is None else Jzz_2nn
    J3 = Jzz if Jzz_3nn is None else Jzz_3nn
    op = qed.Operator(cl.N, 0.5)
    seen = set()
    for (i, j, d) in cl.bonds(NN_BONDS):
        key = tuple(sorted((i, j)))
        if key in seen:
            continue
        seen.add(key)
        c = cl.supercell_fraction(d)
        th = flux[0] * c[0] + flux[1] * c[1]
        ph = complex(np.cos(th), np.sin(th))
        if flux == (0.0, 0.0) or th == 0.0:
            op.add_two_body(qed.OP_SPLUS, i, qed.OP_SMINUS, j, -Jpm)
            op.add_two_body(qed.OP_SMINUS, i, qed.OP_SPLUS, j, -Jpm)
        else:
            op.add_two_body(qed.OP_SPLUS, i, qed.OP_SMINUS, j, -Jpm * ph)
            op.add_two_body(qed.OP_SMINUS, i, qed.OP_SPLUS, j, -Jpm * np.conj(ph))
        op.add_two_body(qed.OP_SZ, i, qed.OP_SZ, j, Jzz)
    for (i, j) in nn2:
        op.add_two_body(qed.OP_SZ, i, qed.OP_SZ, j, J2)
    for (i, j) in nn3:
        op.add_two_body(qed.OP_SZ, i, qed.OP_SZ, j, J3)
    return op, cl


def enumerate_supercells(n_cells):
    """All distinct index-n_cells sublattices, one Hermite normal form each:
    rows (a, b), (0, d) with a d = n_cells, 0 <= b < d."""
    out = []
    for a in range(1, n_cells + 1):
        if n_cells % a:
            continue
        d = n_cells // a
        for b in range(d):
            out.append([[a, b], [0, d]])
    return out
