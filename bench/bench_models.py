"""Benchmark models that the grid zoo (python/tests/grid/models.py) does not hold: the
nearest-neighbour Heisenberg ring and 6x6 triangular torus of the XDiag twins, the 36-site
kagome torus (Heisenberg and the BFG model), the triangle-based NLCE clusters, and the
symmetry selections the cases use (a block named by momentum and irrep characters)."""
from __future__ import annotations

import itertools

import qed


# ---- the XDiag-twin models --------------------------------------------------------------
def heisenberg_ring(N):
    """(H, generators): J = 1 nearest-neighbour ring, the translation as the one generator."""
    H = qed.input.HamiltonianBuilder(N).heisenberg([(i, (i + 1) % N) for i in range(N)], 1.0).to_operator()
    return H, [list(qed.symmetry.translation(N, 1))]


def tri36():
    """(H, lattice, spatial): the 6x6 triangular torus, J1 Heisenberg; ``spatial`` names the
    space group p6m as the 36 translations (the momenta) times the site-centred point group
    (the residues), the split a plain permutation list of the same group also gets."""
    lat = qed.lattice.TriangularSupercell("36")
    bonds = [(i, j) for (i, j, _) in lat.bonds()]
    H = qed.input.HamiltonianBuilder(lat.N).heisenberg(bonds, 1.0).to_operator()
    spatial = qed.GeneratorSet(name="tri36_p6m", description="translations x C6v",
                               generators=[list(p) for p in lat.momentum_generators()], orders=[6, 6],
                               group_size=36, star_perms=[list(p) for _, p in lat.point_group()])
    return H, lat, spatial


# ---- kagome tori (general supercell; rows of L in the Bravais basis a1, a2) ---------------
# (source sublattice, di, dj, target sublattice); third neighbours are the hexagon diagonals
# only (the BFG hexagon term), never the straight line through a bowtie corner.
KAGOME_NN = [(0, 0, 0, 1), (0, 0, 0, 2), (1, 0, 0, 2), (1, 1, 0, 0), (2, 0, 1, 0), (1, 1, -1, 2)]
KAGOME_NN2 = [(0, 1, -1, 2), (0, -1, 1, 1), (1, 1, 0, 2), (1, 0, 1, 0), (2, 1, 0, 0), (2, 0, 1, 1)]
KAGOME_NN3 = [(0, 1, -1, 0), (1, 0, 1, 1), (2, 1, 0, 2)]
KAGOME_36 = ((2, 2), (-2, 4))          # the BFG campaign's 36d torus: full C6v, 12 cells


class KagomeTorus:
    def __init__(self, L=KAGOME_36):
        (a, b), (c, d) = L
        self.L, self.det = L, abs(a * d - b * c)
        span = abs(a) + abs(b) + abs(c) + abs(d) + 1
        cells = sorted({self._reduce(i, j) for i in range(-span, span + 1) for j in range(-span, span + 1)})
        assert len(cells) == self.det, "supercell enumeration failed"
        self.index = {cell: n for n, cell in enumerate(cells)}
        self.cells = cells
        self.N = 3 * self.det

    def _reduce(self, i, j):
        """The canonical cell of (i, j) modulo the superlattice (exact integer arithmetic)."""
        (a, b), (c, d) = self.L
        det = a * d - b * c
        # fractional coordinates s = (i, j) L^-1, times det, reduced mod det
        x, y = (i * d - j * c) % abs(det), (-i * b + j * a) % abs(det)
        if det < 0:
            x, y = (-x) % -det, (-y) % -det
        return x, y

    def site(self, i, j, s):
        return 3 * self.index[self._reduce(i, j)] + s

    def bonds(self, table):
        """Sorted distinct (i, j) pairs of one bond table; raises when the torus double counts."""
        out = set()
        for (ci, cj) in self._cells_int():
            for (s, di, dj, t) in table:
                i, j = self.site(ci, cj, s), self.site(ci + di, cj + dj, t)
                if i == j:
                    raise ValueError("torus too small: a bond wraps onto its own site")
                out.add((min(i, j), max(i, j)))
        if len(out) != len(table) * self.det:
            raise ValueError("torus too small: the periodic wrap double counts a bond")
        return sorted(out)

    def _cells_int(self):
        """One integer vector per cell (any representative)."""
        span = sum(abs(x) for row in self.L for x in row) + 1
        rep = {}
        for i in range(-span, span + 1):
            for j in range(-span, span + 1):
                rep.setdefault(self._reduce(i, j), (i, j))
        return [rep[c] for c in self.cells]


def kagome36_heisenberg():
    k = KagomeTorus()
    return qed.input.HamiltonianBuilder(k.N).heisenberg(k.bonds(KAGOME_NN), 1.0).to_operator(), k


def bfg36(jpm, jzz=1.0):
    """The BFG model -Jpm sum_<ij> (S+S- + h.c.) + Jzz sum_hexagon pairs SzSz (NN, 2NN and the
    hexagon diagonals), on the 36d torus; as in the BFG campaign (Jzz = 1, Jpm < 0)."""
    k = KagomeTorus()
    b = qed.input.HamiltonianBuilder(k.N)
    b.xxz(k.bonds(KAGOME_NN), -2.0 * jpm, jzz)
    b.ising(k.bonds(KAGOME_NN2), jzz)
    b.ising(k.bonds(KAGOME_NN3), jzz)
    return b.to_operator(), k


# ---- block selections -------------------------------------------------------------------
def select_block(sym, H, momentum=None, characters=None):
    """``sym`` restricted to one momentum (``{generator index or perm: theta}``; None: Gamma)
    and the little-group irrep with ``characters(residue) -> chi`` (None: every residue 1,
    i.e. A1; return None to leave a residue unconstrained). Momenta refer to the abelian
    group :meth:`Symmetry.groups` returns, so the result names exactly one star."""
    A, R = sym.groups(H)
    ident = list(range(int(H.num_sites)))
    mom = ({tuple(a): 0 for a in A if a != ident} if momentum is None
           else {tuple(T): th for T, th in momentum.items()})
    chars = {}
    for r in R:
        c = 1.0 if characters is None else characters(r)
        if c is not None:
            chars[tuple(r)] = c
    if characters is not None and characters(ident) is not None:
        chars[tuple(ident)] = characters(ident)
    return sym.select(momentum=mom, irrep_character=chars or None)


def residue_namer(A, labelled):
    """residue -> point-group label, for residues that are a translation times one of the
    ``labelled`` [(label, perm)] (translations form a normal subgroup, so cosets match)."""
    coset = {}
    for label, p in labelled:
        for a in A:
            coset[tuple(a[i] for i in p)] = label
    ident = tuple(range(len(A[0])))
    return lambda r: "E" if tuple(r) == ident else coset.get(tuple(r))


# C6v characters at Gamma (labels of TriangularSupercell.point_group) and C3v at K.
C6V_E1 = {"E": 2, "C6^1": 1, "C6^5": 1, "C6^2": -1, "C6^4": -1, "C6^3": -2}
C3V_E = {"E": 2, "C6^2": -1, "C6^4": -1}


def char_table(name_of, table):
    """characters(residue) for select_block: table[label], None for labels it omits."""
    return lambda r: table.get(name_of(r))


# ---- triangle-based NLCE clusters -------------------------------------------------------
def _up(x, y):
    return ((x, y), (x + 1, y), (x, y + 1))


_UP_NBRS = ((1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1))   # up triangles sharing a corner


def _canonical(cells):
    """Translation- and C3v-canonical form of a set of up triangles (by their corner cell)."""
    def c3(n1, n2):
        return (-n1 - n2, n1)
    forms = []
    for mirror in (False, True):
        for rot in range(3):
            img = []
            for (x, y) in cells:
                pts = []
                for (n1, n2) in _up(x, y):
                    if mirror:
                        n1, n2 = n2, n1
                    for _ in range(rot):
                        n1, n2 = c3(n1, n2)
                    pts.append((n1, n2))
                img.append((min(p[0] for p in pts), min(p[1] for p in pts)))
            mx, my = min(c[0] for c in img), min(c[1] for c in img)
            forms.append(tuple(sorted((c[0] - mx, c[1] - my) for c in img)))
    return min(forms)


def nlce_triangle_clusters(max_order=8, max_sites=16):
    """Every connected cluster of corner-sharing up triangles with at most ``max_order``
    triangles and ``max_sites`` sites, one per isomorphism class of its bond graph (pynauty
    certificate; C3v + translation classes without pynauty). Returns [(n_sites, bonds)]."""
    try:
        import pynauty
    except ImportError:
        pynauty = None
    level = {_canonical([(0, 0)])}
    seen_graphs, out = set(), []
    for order in range(1, max_order + 1):
        for cells in sorted(level):
            sites = sorted({p for c in cells for p in _up(*c)})
            if len(sites) > max_sites:
                continue
            idx = {p: i for i, p in enumerate(sites)}
            bonds = sorted({tuple(sorted((idx[a], idx[b])))
                            for c in cells for a, b in itertools.combinations(_up(*c), 2)})
            if pynauty is not None:
                adj = {i: [] for i in range(len(sites))}
                for i, j in bonds:
                    adj[i].append(j)
                    adj[j].append(i)
                key = (len(sites), pynauty.certificate(pynauty.Graph(len(sites), adjacency_dict=adj)))
            else:
                key = cells
            if key in seen_graphs:
                continue
            seen_graphs.add(key)
            out.append((len(sites), bonds))
        if order < max_order:
            level = {_canonical(list(cells) + [(x + dx, y + dy)])
                     for cells in level for (x, y) in cells for (dx, dy) in _UP_NBRS
                     if (x + dx, y + dy) not in cells}
    return out


def cluster_operator(n_sites, bonds):
    return qed.input.HamiltonianBuilder(n_sites).heisenberg(bonds, 1.0).to_operator()


def random_xxz_chain(N=18, seed=1234):
    """The P3-krylov-01 model: open chain, random nn + nnn XXZ couplings (no degeneracies)."""
    import numpy as np
    rng = np.random.default_rng(seed)
    bonds = [(i, i + 1, rng.uniform(0.5, 1.5), rng.uniform(0.5, 1.5)) for i in range(N - 1)]
    bonds += [(i, i + 2, rng.uniform(0.2, 0.6), rng.uniform(0.2, 0.6)) for i in range(N - 2)]
    b = qed.input.HamiltonianBuilder(N)
    for (i, j, jxy, jz) in bonds:
        b.xxz([(i, j)], jxy, jz)
    return b.to_operator(), bonds


def sector_matrix(N, n_up, bonds):
    """Independent scipy sparse matrix of the random XXZ chain in one Sz sector (set bit = up)."""
    import numpy as np
    import scipy.sparse as sp
    allst = np.arange(1 << N, dtype=np.int64)
    pc = np.zeros_like(allst)
    for i in range(N):
        pc += (allst >> i) & 1
    states = allst[pc == n_up]
    D = states.size
    diag = np.zeros(D)
    rows, cols, vals = [], [], []
    for (i, j, jxy, jz) in bonds:
        bi, bj = (states >> i) & 1, (states >> j) & 1
        diag += jz * (bi - 0.5) * (bj - 0.5)
        m = bi != bj
        src = np.nonzero(m)[0]
        dst = np.searchsorted(states, states[m] ^ ((1 << i) | (1 << j)))
        rows.append(dst)
        cols.append(src)
        vals.append(np.full(src.size, 0.5 * jxy))
    Hs = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(D, D))
    return Hs + sp.diags(diag)

