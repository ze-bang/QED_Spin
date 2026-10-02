"""Grid models and their dense references.

Every model is a term list. The same list builds the library operator and,
independently, a dense numpy matrix, so the reference never goes through the
library's matvec. Conventions match the library: one bit per site, a set bit is
Sz = +1/2 (n_up counts set bits), S+ acts on a clear bit and sets it.
"""
from __future__ import annotations

import functools
import math
from dataclasses import dataclass, field

import numpy as np

# A term is (coeff, ((op, site), ...)) with op in "+-zud" (xy are expanded; u and d are the
# projectors on spin up and down).
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


@dataclass
class Model:
    name: str
    N: int
    terms: list
    translations: list            # generator permutations, perm[i] = image of i
    shape: tuple                  # translation orders, one per generator
    coords: list                  # integer coordinates per site, one per generator
    u1: bool = True
    su2: bool = True
    real: bool = True
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


def _apply(ops, s):
    """Apply a product of single-site ops (rightmost first) to basis state s."""
    amp = 1.0
    for op, site in reversed(ops):
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


# ---------------------------------------------------------------------------
# The models
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
# Dense oracle
# ---------------------------------------------------------------------------

@functools.lru_cache(maxsize=None)
def oracle(name):
    m = MODELS[name]
    H = dense(m.terms, m.N)
    assert np.allclose(H, H.conj().T, atol=1e-12), f"{name}: dense H not Hermitian"
    E, V = np.linalg.eigh(H)
    return Oracle(m, H, E, V)


class Oracle:
    def __init__(self, m, H, E, V):
        self.m, self.H, self.E, self.V = m, H, E, V
        dim = 1 << m.N
        self.pop = np.array([bin(s).count("1") for s in range(dim)])

    # -- spectra --------------------------------------------------------
    def block(self, mask):
        idx = np.flatnonzero(mask)
        return np.linalg.eigvalsh(self.H[np.ix_(idx, idx)])

    def spectrum(self, sel=None):
        if sel is None:
            return self.E
        kind, val = sel
        if kind == "n_up":
            return self.block(self.pop == val)
        if kind == "parity":
            return self.block(self.pop % 2 == val)
        if kind == "S":
            return self.spin_block(val)
        raise ValueError(sel)

    def spin_block(self, S):
        w, U = self.s2_eig()
        P = U[:, np.abs(w - S * (S + 1)) < 1e-6]
        return np.linalg.eigvalsh(P.conj().T @ self.H @ P)

    @functools.lru_cache(maxsize=None)
    def s2_eig(self):
        return np.linalg.eigh(self.s2())

    @functools.lru_cache(maxsize=None)
    def s2(self):
        N = self.m.N
        t = []
        for i in range(N):
            for j in range(N):
                t += dot(i, j)
        return dense(t, N)

    # -- thermodynamics --------------------------------------------------
    @staticmethod
    def thermo(E, T):
        T = np.asarray(T, float)
        e0 = E.min()
        out = {"E": [], "C": [], "S": [], "F": []}
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
        return {k: np.array(v) for k, v in out.items()}

    # -- dynamics --------------------------------------------------------
    def lehmann(self, obs_terms, omega, eta, T=None, deg_tol=1e-8, init=None):
        """S(omega) = sum_m p_m sum_n |<n|O|m>|^2 L(omega - E_n + E_m), with p_m the ground
        manifold (T=0) or Boltzmann weights over the initial states; ``init`` (a basis mask)
        restricts the initial states to the eigenstates of H inside that block, or is the
        (E, W) eigenbasis of the initial states itself (a spin tower, from ``eigbasis``)."""
        O = dense(obs_terms, self.m.N)
        E, V = self.E, self.V
        if isinstance(init, tuple):
            Eb, Vi = init
        elif init is not None:
            idx = np.flatnonzero(init)
            Eb, Vb = np.linalg.eigh(self.H[np.ix_(idx, idx)])
            Vi = np.zeros((len(E), len(Eb)), complex)
            Vi[idx] = Vb
        else:
            Eb, Vi = E, V
        W = np.abs(V.conj().T @ O @ Vi) ** 2      # |<n|O|m>|^2, m over initial states
        om = np.asarray(omega)[:, None]
        if T is None or T == 0:
            g = np.flatnonzero(Eb - Eb[0] < deg_tol)
            dE = E[None, :] - Eb[0]
            w = W[:, g].sum(axis=1)[None, :] / len(g)
            return (w * eta / math.pi / ((om - dE) ** 2 + eta ** 2)).sum(axis=1)
        b = 1.0 / T
        p = np.exp(-b * (Eb - Eb[0]))
        p /= p.sum()
        S = np.zeros(len(omega))
        for mi in np.flatnonzero(p > 1e-14):
            dE = E - Eb[mi]
            S += p[mi] * (W[:, mi][None, :] * eta / math.pi
                          / ((om - dE[None, :]) ** 2 + eta ** 2)).sum(axis=1)
        return S

    def rayleigh(self, vec):
        """(Rayleigh energy, residual ||Hv - Ev||) of a normalised vector."""
        v = np.asarray(vec, complex)
        v = v / np.linalg.norm(v)
        hv = self.H @ v
        e = float(np.vdot(v, hv).real)
        return e, float(np.linalg.norm(hv - e * v))

    def mask(self, sel):
        if sel is None:
            return None
        kind, val = sel
        if kind == "n_up":
            return self.pop == val
        if kind == "parity":
            return self.pop % 2 == val
        raise ValueError(sel)

    # -- expectation values ------------------------------------------------
    @functools.lru_cache(maxsize=None)
    def eigbasis(self, sel):
        """(E, W): eigenpairs of H inside the selection, W in the full basis."""
        if sel is None:
            return self.E, self.V
        if sel[0] == "S":
            w, U = self.s2_eig()
            P = U[:, np.abs(w - sel[1] * (sel[1] + 1)) < 1e-6]
            E, V = np.linalg.eigh(P.conj().T @ self.H @ P)
            return E, P @ V
        idx = np.flatnonzero(self.mask(sel))
        E, V = np.linalg.eigh(self.H[np.ix_(idx, idx)])
        W = np.zeros((len(self.E), len(E)), complex)
        W[idx] = V
        return E, W

    def thermal_expect(self, sel, obs_list, T):
        """<O>(T) = Tr(e^{-H/T} O) / Z over the selection, one row per operator."""
        E, W = self.eigbasis(sel)
        out = []
        for terms in obs_list:
            d = np.sum(W.conj() * (sparse(terms, self.m.N) @ W), axis=0)
            row = []
            for t in np.asarray(T, float):
                w = np.exp(-(E - E.min()) / t)
                row.append(complex((w * d).sum() / w.sum()))
            out.append(row)
        return np.array(out)

    def cluster_traces(self, sel, obs_terms, tol=1e-8):
        """[(E, dim, Tr(P_E O))] over the degenerate clusters of H inside the selection --
        what any partner choice must reproduce as sum over the cluster of multiplicity x <O>."""
        E, W = self.eigbasis(sel)
        diag = np.sum(W.conj() * (sparse(obs_terms, self.m.N) @ W), axis=0)
        out, a = [], 0
        for b in range(1, len(E) + 1):
            if b == len(E) or E[b] - E[a] > tol:
                out.append((float(E[a]), b - a, complex(diag[a:b].sum())))
                a = b
        return out
