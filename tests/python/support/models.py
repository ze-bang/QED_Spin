"""The model zoo of the test suites: every model the grid, golden, regress and bench suites run,
as a term list (support.oracle's vocabulary) with the symmetries it physically carries.

The same term list builds the library operator (Model.operator) and, independently, the dense
reference (support.oracle.dense). Grid models carry their translations, shape and coordinates;
golden models (make_audit_models, nlce_clusters, tri_chiral_3x3, tri_j1j2, square_j1j2) carry the
flags that pick their symmetry options. Couplings of golden models are dyadic rationals wherever
a case is compared at 1e-10.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from support.oracle import _expand, dot, records, ring, triple
from support.triangular import NN_OFFSETS, NNN_OFFSETS, TriangularTorus


@dataclass
class Model:
    name: str
    N: int
    terms: list
    translations: list = field(default_factory=list)   # generator permutations, perm[i] = image of i
    shape: tuple = ()                                   # translation orders, one per generator
    coords: list = field(default_factory=list)          # integer coordinates per site, one per generator
    u1: bool = True
    su2: bool = True
    real: bool = True
    flip: bool = True                                   # [H, prod sigma^x] = 0
    parity: bool = True                                 # (-1)^{n_up} conserved (true when u1 is)
    lattice: tuple = None                               # (Lx, Ly) of a 2D golden model
    notes: str = ""
    extra: dict = field(default_factory=dict)

    def operator(self):
        """Terms on up to three sites as records, longer ones (and projectors) through the algebra."""
        import qed
        H = qed.Operator(self.N)
        code = {"+": qed.OP_SPLUS, "-": qed.OP_SMINUS, "z": qed.OP_SZ}
        long = []
        for c, ops in self.terms:
            if abs(c) < 1e-15:
                continue
            if len(ops) > 3 or any(op not in code for op, _ in ops):
                long.append((c, ops))
                continue
            args = [x for op, s in ops for x in (code[op], s)]
            if len(ops) == 1:
                H.add_one_body(*args, c)
            elif len(ops) == 2:
                H.add_two_body(*args, c)
            else:
                H.add_three_body(*args, c)
        for c, ops in long:
            H = H + qed.Operator.product(self.N, "".join(op for op, _ in ops), [s for _, s in ops], c)
        return H

    def generator_set(self):
        """The translations as an explicit split (abelian part only)."""
        import qed
        return qed.Symmetries(abelian=[list(t) for t in self.translations])


# ---------------------------------------------------------------------------
# The grid's models
# ---------------------------------------------------------------------------

def chain(N=12, J2=0.35):
    terms = []
    for i in range(N):
        terms += dot(i, (i + 1) % N)
        terms += dot(i, (i + 2) % N, J=J2)
    T = [(i + 1) % N for i in range(N)]
    return Model(f"chain{N}", N, terms, [T], (N,), [(i,) for i in range(N)],
                 notes="J1-J2 ring: U(1), SU(2), flip, D_N, real")


def _tri_sites(Lx, Ly):
    idx = lambda x, y: (x % Lx) + Lx * (y % Ly)  # noqa: E731
    return idx, [(x, y) for y in range(Ly) for x in range(Lx)]


def triangular(L=3, chi=0.0, Ly=None, h=0.0):
    Lx, Ly = L, (L if Ly is None else Ly)
    idx, xy = _tri_sites(Lx, Ly)
    terms = []
    for x, y in xy:
        for dx, dy in ((1, 0), (0, 1), (-1, 1)):
            terms += dot(idx(x, y), idx(x + dx, y + dy))
        if chi:
            terms += triple(idx(x, y), idx(x + 1, y), idx(x, y + 1), chi)
            terms += triple(idx(x + 1, y), idx(x + 1, y + 1), idx(x, y + 1), chi)
        if h:
            terms.append((h, (("z", idx(x, y)),)))
    T1 = [idx(x + 1, y) for x, y in xy]
    T2 = [idx(x, y + 1) for x, y in xy]
    name = f"tri{Lx * Ly}" + ("chi" if chi else "") + ("h" if h else "")
    return Model(name, Lx * Ly, terms, [T1, T2], (Lx, Ly), xy, real=not chi, su2=not h,
                 notes="triangular torus; odd N gives a degenerate ground state"
                       + ("; scalar chirality: complex, TR-odd, three-body" if chi else "")
                       + ("; uniform field h S^z: no spin flip, so a sector label off N/2 (N odd) checks "
                          "which bit value is spin up" if h else ""))


def xyz_chain(N=12, jx=1.0, jy=0.6, jz=0.8):
    terms = []
    for i in range(N):
        j = (i + 1) % N
        terms += _expand(jx, (("x", i), ("x", j)))
        terms += _expand(jy, (("y", i), ("y", j)))
        terms += [(jz, (("z", i), ("z", j)))]
    T = [(i + 1) % N for i in range(N)]
    return Model(f"xyz{N}", N, terms, [T], (N,), [(i,) for i in range(N)],
                 u1=False, su2=False, notes="XYZ ring: Sz parity only, flip, translations")


def square_ring(Lx=4, Ly=3, K=0.3):
    """Heisenberg square torus with the four-site ring exchange K (P + P^dagger) on every plaquette."""
    idx = lambda x, y: (x % Lx) + Lx * (y % Ly)  # noqa: E731
    xy = [(x, y) for y in range(Ly) for x in range(Lx)]
    terms = []
    for x, y in xy:
        terms += dot(idx(x, y), idx(x + 1, y)) + dot(idx(x, y), idx(x, y + 1))
        terms += ring(idx(x, y), idx(x + 1, y), idx(x + 1, y + 1), idx(x, y + 1), K)
    T1 = [idx(x + 1, y) for x, y in xy]
    T2 = [idx(x, y + 1) for x, y in xy]
    return Model(f"sq{Lx * Ly}ring", Lx * Ly, terms, [T1, T2], (Lx, Ly), xy,
                 notes="square torus + four-site ring exchange: U(1), SU(2), flip, real")


def kagome_bq(L=2, K=0.2):
    """Heisenberg kagome torus (L x L cells) with K (S_u.S_u')(S_d.S_d') on every bowtie: u, u' the
    other two sites of a site's up triangle, d, d' of its down triangle (four sites, commuting
    factors, so Hermitian)."""
    idx = lambda x, y, s: 3 * ((x % L) + L * (y % L)) + s  # noqa: E731
    cells = [(x, y) for y in range(L) for x in range(L)]
    up = [(idx(x, y, 0), idx(x, y, 1), idx(x, y, 2)) for x, y in cells]
    down = [(idx(x, y, 1), idx(x + 1, y, 0), idx(x + 1, y - 1, 2)) for x, y in cells]
    terms = []
    for tri in up + down:
        for i in range(3):
            terms += dot(tri[i], tri[(i + 1) % 3])
    for c in range(3 * L * L):
        tu = next(t for t in up if c in t)
        td = next(t for t in down if c in t)
        a, b = (s for s in tu if s != c)
        e, f = (s for s in td if s != c)
        for c1, o1 in dot(a, b):
            for c2, o2 in dot(e, f):
                terms.append((K * c1 * c2, o1 + o2))
    T1 = [idx(x + 1, y, s) for x, y in cells for s in range(3)]
    T2 = [idx(x, y + 1, s) for x, y in cells for s in range(3)]
    coords = [(x, y) for x, y in cells for _ in range(3)]
    return Model(f"kagome{3 * L * L}bq", 3 * L * L, terms, [T1, T2], (L, L), coords,
                 notes="kagome torus + four-site bowtie biquadratic: U(1), SU(2), flip, real")


MODELS = {m.name: m for m in (chain(), triangular(3), triangular(3, chi=0.25), triangular(3, h=0.3),
                              xyz_chain(), square_ring(), kagome_bq())}



# ---------------------------------------------------------------------------
# The golden harness's models
# ---------------------------------------------------------------------------

def heisenberg_terms(bonds, J=1.0, Jz=None):
    Jz = J if Jz is None else Jz
    return records([t for (i, j) in bonds
                    for t in ((("+", "-"), (i, j), 0.5 * J), (("-", "+"), (i, j), 0.5 * J), (("z", "z"), (i, j), Jz))])


def chain_bonds(N):
    return [(i, (i + 1) % N) for i in range(N)]


def triangular_torus(Lx, Ly):
    """Sites (x,y) -> x + Lx*y; a1=(1,0), a2=(1/2, sqrt3/2). Returns nn bonds, nnn bonds,
    and ccw triangles (up and down) as index triples."""
    def idx(x, y):
        return (x % Lx) + Lx * (y % Ly)
    nn, nnn, tri = [], [], []
    for y in range(Ly):
        for x in range(Lx):
            i = idx(x, y)
            nn += [(i, idx(x + 1, y)), (i, idx(x, y + 1)), (i, idx(x - 1, y + 1))]
            nnn += [(i, idx(x + 1, y + 1)), (i, idx(x - 2, y + 1)), (i, idx(x + 1, y - 2))]
            tri.append((i, idx(x + 1, y), idx(x, y + 1)))            # up triangle, ccw
            tri.append((idx(x + 1, y), idx(x + 1, y + 1), idx(x, y + 1)))  # down triangle, ccw

    def dedup(bs):
        seen, out = set(), []
        for (i, j) in bs:
            key = (min(i, j), max(i, j))
            if key not in seen and i != j:
                seen.add(key)
                out.append((i, j))
        return out
    return dedup(nn), dedup(nnn), tri


def chiral_terms(triangles, Jchi):
    """Jchi * S_i . (S_j x S_k) for each ccw triangle, expanded in S+/S-/Sz products:
    S_i.(S_j x S_k) = (i/2) sum_cyclic [ Sz_a (S+_b S-_c - S-_b S+_c) ]."""
    t = []
    for (i, j, k) in triangles:
        for (a, b, c) in ((i, j, k), (j, k, i), (k, i, j)):
            t.append((("z", "+", "-"), (a, b, c), 0.5j * Jchi))
            t.append((("z", "-", "+"), (a, b, c), -0.5j * Jchi))
    return records(t)


def rng_terms_real(N, seed, density=1.0):
    r = np.random.default_rng(seed)
    t = []
    for i in range(N):
        for j in range(i + 1, N):
            if r.random() > density:
                continue
            J = float(r.normal()); Jz = float(r.normal())
            t += [(("+", "-"), (i, j), 0.5 * J), (("-", "+"), (i, j), 0.5 * J), (("z", "z"), (i, j), Jz)]
    return records(t)


def rng_terms_complex(N, seed):
    r = np.random.default_rng(seed)
    t = []
    for i in range(N):
        for j in range(i + 1, N):
            c = complex(r.normal(), r.normal()) * 0.5
            t += [(("+", "-"), (i, j), c), (("-", "+"), (i, j), np.conj(c)), (("z", "z"), (i, j), float(r.normal()))]
    return records(t)


def lattice_pairs(L):
    return [(int(a), int(b)) for (a, b) in L.nn_pairs()]


def make_audit_models():
    """The small models every verb is checked on."""
    import qed
    ms = []
    ms.append(Model("dimer", 2, heisenberg_terms([(0, 1)]), u1=True, real=True, su2=True))
    ms.append(Model("triangle3", 3, heisenberg_terms([(0, 1), (1, 2), (2, 0)]), u1=True, real=True, su2=True))
    ms.append(Model("chain4", 4, heisenberg_terms(chain_bonds(4)), u1=True, real=True, su2=True))
    ms.append(Model("chain5_odd", 5, heisenberg_terms(chain_bonds(5)), u1=True, real=True, su2=True))
    ms.append(Model("open_chain10", 10, heisenberg_terms([(i, i + 1) for i in range(9)]), u1=True, real=True, su2=True,
                    notes="open boundaries: reflection only"))
    ms.append(Model("j1j2_chain12", 12, heisenberg_terms(chain_bonds(12)) + heisenberg_terms([(i, (i + 2) % 12) for i in range(12)], 0.5),
                    u1=True, real=True, su2=True, notes="frustrated, degeneracies"))
    ms.append(Model("xy_chain10", 10, heisenberg_terms(chain_bonds(10), J=1.0, Jz=0.0), u1=True, real=True, su2=False))
    stag = heisenberg_terms(chain_bonds(10)) + records([(("z",), (i,), 0.3 * (-1) ** i) for i in range(10)])
    ms.append(Model("staggered_field10", 10, stag, u1=True, real=True, su2=False, flip=False,
                    notes="period-2 translation"))
    tfim = [(("z", "z"), (i, (i + 1) % 10), -1.0) for i in range(10)]
    tfim += [t for i in range(10) for t in ((("+",), (i,), -0.35), (("-",), (i,), -0.35))]  # -h Sx, h = 0.7
    ms.append(Model("tfim10", 10, records(tfim), u1=False, real=True, su2=False, parity=False,
                    notes="no U(1), no Sz parity; flip-symmetric Z2"))
    ms.append(Model("random_real8", 8, rng_terms_real(8, 1), u1=True, real=True, su2=False,
                    notes="all-to-all, no spatial symmetry"))
    ms.append(Model("random_complex8", 8, rng_terms_complex(8, 2), u1=True, real=False, su2=False,
                    notes="complex hopping + DM, no TR"))
    sq = qed.input.lattice.square(4, 3, True)
    ms.append(Model("square4x3", 12, heisenberg_terms(lattice_pairs(sq)), u1=True, real=True, su2=True, lattice=(4, 3)))
    kg = qed.input.lattice.kagome(2, 2, True)
    ms.append(Model("kagome2x2", 12, heisenberg_terms(lattice_pairs(kg)), u1=True, real=True, su2=True))
    nn, nnn, tri = triangular_torus(4, 3)
    ms.append(Model("tri_chiral4x3", 12, heisenberg_terms(nn) + heisenberg_terms(nnn, 0.2) + chiral_terms(tri, 0.5),
                    u1=True, real=False, su2=True, lattice=(4, 3)))
    return ms


def tri_j1j2(T1, T2, J2, name):
    tt = TriangularTorus((T1, T2))
    nn = [(i, j) for i, j, _ in tt.bonds(NN_OFFSETS)]
    nnn = [(i, j) for i, j, _ in tt.bonds(NNN_OFFSETS)]
    m = Model(name, tt.N, heisenberg_terms(nn, 1.0) + heisenberg_terms(nnn, J2), u1=True, real=True, su2=True)
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


# Open clusters in the shape the NLCE engine feeds to qed.spectrum: no
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
