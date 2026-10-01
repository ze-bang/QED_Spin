# AUDIT-ID: F-C-2
# DEVICE: cpu
# SECONDS: 120
"""Claim (same root cause as C01-pyapi-03, reached through the DEFAULT spatial='auto'): on the
12-site 2x2 kagome torus in a z field, find_symmetries' maximum commuting clique
(python/qed/discovery.py ~733-745) is used as the abelian part. When it is not normal in the
automorphism group, the engine drops every residue that does not normalise A
(lg_engine.cpp:140-146) but multiplet() (src/solvers/little_group/lg_sectors.cpp:466-474) still
applies all of spec.residues, so the expanded multiplet of one level leaks into the eigenspace
owned by another level and EigResult.vectors() is not orthonormal. Fuzzer case 114-64 saw
max|Gram - I| = 0.8165 = sqrt(2/3) (the same value as the K4 repro of C01-pyapi-03) with
residuals ~1e-14. Reference: dense numpy H in the library basis (set bit = spin down)."""
import signal
import sys

import numpy as np


def _timeout(signum, frame):
    print("REPRO: INCONCLUSIVE timed out")
    sys.stdout.flush()
    raise SystemExit(0)


signal.signal(signal.SIGALRM, _timeout)
signal.alarm(280)

try:
    import qed
except Exception as e:  # noqa: BLE001
    print(f"REPRO: INCONCLUSIVE import qed failed: {e}")
    raise SystemExit(0)

N = 12
J = 1.2384511278038135
hz = 0.4772449525192085
K = 4


def site(x, y, s):
    return 3 * ((x % 2) + 2 * (y % 2)) + s


bonds = []
for x in range(2):
    for y in range(2):
        A, B, C = site(x, y, 0), site(x, y, 1), site(x, y, 2)
        bonds += [(A, B), (A, C), (B, C)]
        bonds += [(A, site(x - 1, y, 1)), (A, site(x, y - 1, 2)), (B, site(x + 1, y - 1, 2))]

H = qed.Operator(N)
for i, j in bonds:
    H.add_two_body(qed.OP_SPLUS, i, qed.OP_SMINUS, j, complex(0.5 * J))
    H.add_two_body(qed.OP_SMINUS, i, qed.OP_SPLUS, j, complex(0.5 * J))
    H.add_two_body(qed.OP_SZ, i, qed.OP_SZ, j, complex(J))
for i in range(N):
    H.add_one_body(qed.OP_SZ, i, complex(hz))

# dense reference (real; bit i = site i, set bit = spin down, S+ clears a set bit)
dim = 1 << N
s = np.arange(dim)
bits = [(s >> i) & 1 for i in range(N)]
Hd = np.zeros((dim, dim))
for i, j in bonds:
    bi, bj = bits[i], bits[j]
    Hd[s, s] += J * np.where(bi == bj, 0.25, -0.25)
    m = bi != bj
    Hd[s[m] ^ ((1 << i) | (1 << j)), s[m]] += 0.5 * J
for i in range(N):
    Hd[s, s] += hz * np.where(bits[i] == 0, 0.5, -0.5)
ref = np.linalg.eigvalsh(Hd)


def inv(p):
    q = [0] * len(p)
    for a, b in enumerate(p):
        q[b] = a
    return tuple(q)


def comp(p, q):
    return tuple(p[q[i]] for i in range(len(p)))


try:
    sym = qed.Symmetry()                       # spatial='auto', point_group=True: the default
    A, res = sym.groups(H)
    A = [tuple(a) for a in A]
    Aset = set(A)
    res = [tuple(r) for r in res]
    normal = all(comp(comp(inv(r), a), r) in Aset for r in res for a in A)
    n_norm = sum(all(comp(comp(inv(r), a), r) in Aset for a in A) for r in res)
    print(f"auto: |A|={len(A)} residues={len(res)} A normal={normal} residues normalising A={n_norm}")
    if len(A) <= 1:
        print("REPRO: INCONCLUSIVE spatial='auto' found no spatial symmetry (pynauty missing?)")
        raise SystemExit(0)
    r = qed.eigs(H, K, sym=sym, vectors=True, prune=False)
    vs = r.vectors()
    print("levels (E, mult):", [(round(L.energy, 8), int(L.multiplicity)) for L in r.levels])
except SystemExit:
    raise
except Exception as e:  # noqa: BLE001
    print(f"REPRO: INCONCLUSIVE library raised {type(e).__name__}: {str(e)[:200]}")
    raise SystemExit(0)

dE = float(np.max(np.abs(np.sort(np.asarray(r.energies)[:K]) - ref[:K])))
if len(vs) != K:
    print(f"REPRO: CONFIRMED {len(vs)} vectors for k={K} (|A|={len(A)}, A normal={normal})")
    raise SystemExit(0)
V = np.array([np.asarray(v, complex) for v in vs]).T
G = V.conj().T @ V
orth = float(np.max(np.abs(G - np.eye(K))))
HV = Hd @ V
ray = np.real(np.einsum("im,im->m", V.conj(), HV)) / np.real(np.diag(G))
resid = max(float(np.linalg.norm(HV[:, m] - ray[m] * V[:, m])) for m in range(K))
low = float(np.max(np.abs(np.sort(ray) - ref[:K])))
# rank of the returned set: a dependent set cannot span the k lowest eigenvectors
sv = np.linalg.svd(V, compute_uv=False)
print(f"energies max|dE|={dE:.1e}; vectors: max|Gram-I|={orth:.4e} residual={resid:.1e} "
      f"lowest={low:.1e} singular values={np.round(sv, 6).tolist()}")
if orth > 1e-8 or low > 1e-8:
    print(f"REPRO: CONFIRMED vectors() not the k lowest orthonormal eigenvectors: max|Gram-I|={orth:.4e}, "
          f"Rayleigh-vs-exact {low:.1e} (|A|={len(A)}, A normal={normal}, {n_norm}/{len(res)} residues normalise A)")
else:
    print(f"REPRO: NOT_REPRODUCED vectors orthonormal (max|Gram-I|={orth:.1e}); A normal={normal}")
