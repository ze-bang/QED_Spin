# AUDIT-ID: C01-pyapi-03
# DEVICE: cpu
# SECONDS: 60
"""Claim: for spatial=<raw permutation list> with point_group=True, split_nonabelian
(python/qed/_groups.py:77,143) seeds the abelian part with the highest-order element, ties
broken lexicographically, so a rotation about site 0 beats the translations. The chosen A is
neither the translation group nor normal; the engine then drops every residue that does not
normalise A (lg_engine.cpp:146), momenta along lattice translations cannot be selected, and
multiplet() (lg_sectors.cpp:469) still applies the dropped residues, so EigResult.vectors()
is not orthonormal.

(a) 3x3 triangular torus, spatial=[T1, T2, C6]: report |A|, whether T1 is in A, whether A is
    normal, how many residues normalise A; select(momentum={T1: 0}) must work.
(b) K4 Heisenberg, spatial=[(0123), (01)], sz=2: vectors(basis='sz') of all 6 states must be
    orthonormal eigenvectors (checked against a dense numpy H)."""
import signal

import numpy as np

import qed
from support.triangular import TriangularTorus

signal.alarm(240)
problems = []


def inv(p):
    q = [0] * len(p)
    for i, x in enumerate(p):
        q[x] = i
    return tuple(q)


def comp(p, q):
    return tuple(p[q[i]] for i in range(len(p)))


# ---------------- (a) triangular 3x3 with translations + C6 ----------------
lat = TriangularTorus("9")
N = lat.N
b = qed.input.HamiltonianBuilder(N)
b.heisenberg([(i, j) for (i, j, _) in lat.bonds()], J=1.0)
Ht = b.to_operator()
T1, T2 = lat.momentum_generators()
C6 = dict(lat.point_group())["C6^1"]
sym = qed.Symmetry(spatial=[list(T1), list(T2), list(C6)], sz=4, spin_flip="off", time_reversal="off")
A, res = sym.groups(Ht)
A = [tuple(a) for a in A]
Aset = set(A)
res = [tuple(r) for r in res]
is_normal = all(comp(comp(inv(r), a), r) in Aset for r in res for a in A)
n_keep = sum(all(comp(comp(inv(r), a), r) in Aset for a in A) for r in res)
t_in = tuple(T1) in Aset
print(f"tri9 [T1,T2,C6]: |A|={len(A)} (translations: 9) T1 in A={t_in} A normal={is_normal} "
      f"residues={len(res)} normalising A={n_keep}")
if not t_in or not is_normal:
    problems.append(f"tri9 |A|={len(A)} T1inA={t_in} normal={is_normal} kept residues {n_keep}/{len(res)}")
try:
    e = qed.eigs(Ht, 1, sym=sym.select(momentum={tuple(T1): 0, tuple(T2): 0}))
    print(f"select(momentum=Gamma) ok: E0={e.energies[0]:.10f}")
except Exception as ex:
    print(f"select(momentum=Gamma) raised {type(ex).__name__}: {str(ex)[:100]}")
    problems.append("momentum select refused")

# ---------------- (b) K4 vectors() orthonormality ----------------
N = 4
NUP = 2
bonds = [(i, j) for i in range(N) for j in range(i + 1, N)]
b = qed.input.HamiltonianBuilder(N)
b.heisenberg(bonds, J=1.0)
Hk = b.to_operator()
states = [s for s in range(1 << N) if bin(s).count("1") == NUP]
sidx = {s: i for i, s in enumerate(states)}
D = len(states)
Hd = np.zeros((D, D))
for a, s in enumerate(states):
    for i, j in bonds:
        si, sj = (s >> i) & 1, (s >> j) & 1
        Hd[a, a] += 0.25 if si == sj else -0.25
        if si != sj:
            Hd[sidx[s ^ ((1 << i) | (1 << j))], a] += 0.5
evals = np.linalg.eigvalsh(Hd)
symk = qed.Symmetry(spatial=[[1, 2, 3, 0], [1, 0, 2, 3]], sz=NUP, spin_flip="off", time_reversal="off")
try:
    ek = qed.eigs(Hk, D, sym=symk, vectors=True)
    dE = float(np.max(np.abs(np.sort(ek.energies) - evals)))
    V = np.array([np.asarray(v, complex) for v in ek.vectors(basis="sz", n_up=NUP)])
    G = V.conj() @ V.T
    gram_err = float(np.max(np.abs(G - np.eye(len(V)))))
    res_err = max(float(np.linalg.norm(Hd @ v - np.vdot(v, Hd @ v).real * v)) for v in V)
    print(f"K4: max|dE|={dE:.2e} n_vectors={len(V)} max|Gram-I|={gram_err:.3e} "
          f"max eigen-residual={res_err:.2e} levels={[(round(L.energy, 6), int(L.multiplicity)) for L in ek.levels]}")
    if gram_err > 1e-8:
        problems.append(f"K4 vectors() not orthonormal, max|Gram-I|={gram_err:.3e}")
except Exception as ex:
    problems.append(f"K4 eigs/vectors raised {type(ex).__name__}: {str(ex)[:100]}")

if problems:
    print("REPRO: CONFIRMED " + "; ".join(problems))
else:
    print("REPRO: NOT_REPRODUCED A contains the translations, is normal, and vectors() is orthonormal")
