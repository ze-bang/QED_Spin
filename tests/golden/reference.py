"""Model vocabulary and the independent dense reference for the golden harness.

Moved verbatim from benchmarks/audit_workflows.py (Model, term builders, Reference) and
benchmarks/audit_correctness.py (make_models and its helpers) so that the gate does not
depend on the benchmarks tree. A model is a list of terms (ops, sites, coeff) with ops in
{"+", "-", "z"}; the same list feeds the QED builder and the numpy reference.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

import qed
from qed.input import HamiltonianBuilder, Op


@dataclass
class Model:
    name: str
    N: int
    terms: list                       # (ops tuple, sites tuple, complex coeff)
    u1: bool                          # conserves Sz
    real: bool                        # real Hamiltonian (time-reversal symmetric)
    su2: bool = False
    flip: bool = True                 # [H, prod sigma^x] = 0
    parity: bool = True               # (-1)^{n_down} conserved (true when u1 is true)
    lattice: Optional[tuple] = None   # (Lx, Ly) for 2D models, site = x + Lx*y
    notes: str = ""

    def builder(self) -> HamiltonianBuilder:
        b = HamiltonianBuilder(self.N)
        OPS = {"+": Op.Sp, "-": Op.Sm, "z": Op.Sz}
        for ops, sites, c in self.terms:
            c = complex(c)
            if len(ops) == 1:
                b.add_one_body(OPS[ops[0]], sites[0], c)
            elif len(ops) == 2:
                b.add_two_body(OPS[ops[0]], sites[0], OPS[ops[1]], sites[1], c)
            else:
                b.add_three_body(OPS[ops[0]], sites[0], OPS[ops[1]], sites[1], OPS[ops[2]], sites[2], c)
        return b

    def operator(self):
        return self.builder().to_operator()


def heisenberg_terms(bonds, J=1.0, Jz=None):
    Jz = J if Jz is None else Jz
    t = []
    for (i, j) in bonds:
        t.append((("+", "-"), (i, j), 0.5 * J))
        t.append((("-", "+"), (i, j), 0.5 * J))
        t.append((("z", "z"), (i, j), Jz))
    return t


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
    # de-duplicate bonds as unordered pairs
    def dedup(bs):
        seen, out = set(), []
        for (i, j) in bs:
            key = (min(i, j), max(i, j))
            if key not in seen and i != j:
                seen.add(key); out.append((i, j))
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
    return t


def rng_terms_real(N, seed, density=1.0):
    r = np.random.default_rng(seed)
    t = []
    for i in range(N):
        for j in range(i + 1, N):
            if r.random() > density:
                continue
            J = float(r.normal()); Jz = float(r.normal())
            t += [(("+", "-"), (i, j), 0.5 * J), (("-", "+"), (i, j), 0.5 * J), (("z", "z"), (i, j), Jz)]
    return t


def rng_terms_complex(N, seed):
    r = np.random.default_rng(seed)
    t = []
    for i in range(N):
        for j in range(i + 1, N):
            c = complex(r.normal(), r.normal()) * 0.5
            t += [(("+", "-"), (i, j), c), (("-", "+"), (i, j), np.conj(c)), (("z", "z"), (i, j), float(r.normal()))]
    return t


def lattice_pairs(L):
    return [(int(a), int(b)) for (a, b) in L.nn_pairs()]


def make_audit_models():
    """The model set of the correctness audit (formerly audit_correctness.make_models)."""
    ms = []
    ms.append(Model("dimer", 2, heisenberg_terms([(0, 1)]), u1=True, real=True, su2=True))
    ms.append(Model("triangle3", 3, heisenberg_terms([(0, 1), (1, 2), (2, 0)]), u1=True, real=True, su2=True))
    ms.append(Model("chain4", 4, heisenberg_terms(chain_bonds(4)), u1=True, real=True, su2=True))
    ms.append(Model("chain5_odd", 5, heisenberg_terms(chain_bonds(5)), u1=True, real=True, su2=True))
    ms.append(Model("open_chain10", 10, heisenberg_terms([(i, i + 1) for i in range(9)]), u1=True, real=True, su2=True,
                    notes="open boundaries: reflection only"))
    ms.append(Model("j1j2_chain12", 12, heisenberg_terms(chain_bonds(12)) + heisenberg_terms([(i, (i + 2) % 12) for i in range(12)], 0.5),
                    u1=True, real=True, su2=True, notes="frustrated, degeneracies"))
    ms.append(Model("xy_chain10", 10, heisenberg_terms(chain_bonds(10), J=1.0, Jz=0.0), u1=True, real=True))
    stag = heisenberg_terms(chain_bonds(10)) + [(("z",), (i,), 0.3 * (-1) ** i) for i in range(10)]
    ms.append(Model("staggered_field10", 10, stag, u1=True, real=True, flip=False, notes="period-2 translation"))
    tfim = [(("z", "z"), (i, (i + 1) % 10), -1.0) for i in range(10)]
    tfim += [t for i in range(10) for t in ((("+",), (i,), -0.35), (("-",), (i,), -0.35))]  # -h Sx, h = 0.7
    ms.append(Model("tfim10", 10, tfim, u1=False, real=True, parity=False, notes="no U(1), no Sz parity; flip-symmetric Z2"))
    ms.append(Model("random_real8", 8, rng_terms_real(8, 1), u1=True, real=True, notes="all-to-all, no spatial symmetry"))
    ms.append(Model("random_complex8", 8, rng_terms_complex(8, 2), u1=True, real=False, notes="complex hopping + DM, no TR"))
    sq = qed.input.lattice.square(4, 3, True)
    ms.append(Model("square4x3", 12, heisenberg_terms(lattice_pairs(sq)), u1=True, real=True, su2=True, lattice=(4, 3)))
    kg = qed.input.lattice.kagome(2, 2, True)
    ms.append(Model("kagome2x2", 12, heisenberg_terms(lattice_pairs(kg)), u1=True, real=True, su2=True))
    nn, nnn, tri = triangular_torus(4, 3)
    ms.append(Model("tri_chiral4x3", 12, heisenberg_terms(nn) + heisenberg_terms(nnn, 0.2) + chiral_terms(tri, 0.5),
                    u1=True, real=False, su2=True, lattice=(4, 3)))
    return ms


# =============================================================================
# Independent numpy reference
# =============================================================================
class Reference:
    """Dense ED of a term list. Basis: computational bit strings, bit i = site i in the
    QED convention (set bit = DOWN spin, S^- raises the set-bit count). We only need
    the spectrum and Sz-resolved spectra, which are convention independent, plus the
    same matrix-element convention as QED for the spectral checks."""

    def __init__(self, model: Model):
        self.model = model
        N = model.N
        D = 1 << N
        self.N, self.D = N, D
        states = np.arange(D, dtype=np.int64)
        self.nset = np.array([bin(s).count("1") for s in states])
        H = np.zeros((D, D), dtype=complex)
        for ops, sites, c in model.terms:
            self._add_term(H, ops, sites, complex(c))
        self.H = H
        assert np.allclose(H, H.conj().T, atol=1e-12), "reference H not Hermitian"
        self.evals, self.evecs = np.linalg.eigh(H)

    def _add_term(self, H, ops, sites, c):
        # Apply the product right-to-left on each basis state; QED's convention: set bit =
        # down spin, S^+ clears a set bit (down -> up), S^- sets it, Sz = +1/2 for clear bit.
        D = self.D
        for s in range(D):
            amp = c; t = s; ok = True
            for op, site in zip(reversed(ops), reversed(sites)):
                bit = (t >> site) & 1
                if op == "z":
                    amp *= (-0.5 if bit else 0.5)
                elif op == "+":
                    if bit: t ^= (1 << site)
                    else: ok = False; break
                else:
                    if not bit: t ^= (1 << site)
                    else: ok = False; break
            if ok and amp != 0:
                H[t, s] += amp

    # ---- spectra ----
    def sector_evals(self, n_set):
        mask = self.nset == n_set
        Hs = self.H[np.ix_(mask, mask)]
        return np.linalg.eigvalsh(Hs)

    def lowest(self, k, n_set=None):
        ev = self.evals if n_set is None else self.sector_evals(n_set)
        return np.sort(ev)[:k]

    # ---- thermodynamics (full Hilbert space unless n_set given) ----
    def thermo(self, temps, n_set=None):
        ev = self.evals if n_set is None else self.sector_evals(n_set)
        E, C, S = [], [], []
        for T in temps:
            beta = 1.0 / T
            w = np.exp(-beta * (ev - ev.min()))
            Z = w.sum(); e = (ev * w).sum() / Z; e2 = (ev * ev * w).sum() / Z
            E.append(e); C.append(beta * beta * (e2 - e * e))
            S.append(math.log(Z) + beta * (e - ev.min()))
        return np.array(E), np.array(C), np.array(S)

    # ---- dynamical structure factor, Lehmann form ----
    def operator_matrix(self, terms):
        O = np.zeros((self.D, self.D), dtype=complex)
        for ops, sites, c in terms:
            self._add_term(O, ops, sites, complex(c))
        return O

    def gs_spectral(self, O, omega, eta, n_set=None):
        """S(w) = -1/pi Im <0| O^dag (w + E0 + i eta - H)^-1 O |0> with |0> the global GS
        (or the GS of the Sz sector n_set)."""
        if n_set is None:
            ev, U = self.evals, self.evecs
            E0 = ev[0]; psi0 = U[:, 0]
        else:
            mask = self.nset == n_set
            Hs = self.H[np.ix_(mask, mask)]
            evs, Us = np.linalg.eigh(Hs)
            E0 = evs[0]; psi0 = np.zeros(self.D, dtype=complex); psi0[mask] = Us[:, 0]
            ev, U = self.evals, self.evecs
        phi = O @ psi0
        amps = np.abs(U.conj().T @ phi) ** 2           # |<n|O|0>|^2
        S = np.zeros(len(omega))
        for n in range(self.D):
            if amps[n] < 1e-14: continue
            S += amps[n] * (eta / math.pi) / ((omega + E0 - ev[n]) ** 2 + eta * eta)
        return S

    def thermal_spectral(self, O, omega, eta, T, n_set=None):
        """S(w, T) = 1/Z sum_mn e^{-beta E_m} |<n|O|m>|^2 L_eta(w - (E_n - E_m)).
        With n_set the trace runs over the Sz block only (QED's sz= convention: the
        thermal average is the block-restricted canonical ensemble)."""
        if n_set is not None:
            mask = self.nset == n_set
            Hs = self.H[np.ix_(mask, mask)]
            ev, U = np.linalg.eigh(Hs)
            O = O[np.ix_(mask, mask)]
        else:
            ev, U = self.evals, self.evecs
        beta = 1.0 / T
        w = np.exp(-beta * (ev - ev[0])); Z = w.sum()
        M = np.abs(U.conj().T @ O @ U) ** 2
        S = np.zeros(len(omega))
        keep = np.where(w / Z > 1e-12)[0]
        for m in keep:
            col = M[:, m]; nz = np.where(col > 1e-14)[0]
            if nz.size == 0: continue
            dE = ev[nz] - ev[m]
            S += (w[m] / Z) * (col[nz][:, None] * (eta / math.pi) / ((omega[None, :] - dE[:, None]) ** 2 + eta * eta)).sum(0)
        return S
