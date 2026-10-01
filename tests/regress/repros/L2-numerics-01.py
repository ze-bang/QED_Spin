# AUDIT-ID: L2-numerics-01
# DEVICE: cpu
# SECONDS: 60
"""Claim: qed.thermal forms C = beta^2 (<E^2> - <E>^2) from UNSHIFTED moments
(lg_sectors_thermal.cpp exact_block + the sector aggregate), so its absolute error is
~ beta^2 * eps * E^2. Adding a constant c to H (physically identical model) raises that floor
by ~((E0+c)/E0)^2, and in the low-T tail the reported C loses all accuracy (or turns negative),
while the same T with c = 0 is accurate.

Model: 12-site Heisenberg ring, Symmetry(spatial=None) (Sz blocks, all on the exact dense path).
The constant c is added as sum_i (4c/N) Sz_i Sz_i (= c * identity for spin-1/2); this is checked
against qed.eigs before use. Reference: independent dense numpy ED of the c = 0 model with a
numerically stable variance sum_n p_n (E_n - <E>)^2 (C is invariant under H -> H + c)."""
import signal
import numpy as np
import scipy.sparse as sps
import qed

signal.alarm(280)
N = 12
bonds = [(i, (i + 1) % N) for i in range(N)]
Ts = np.array([0.1, 0.05, 0.04, 0.03, 0.025, 0.02])
offsets = [0.0, 100.0, 1000.0, 10000.0]

sp_ = sps.csr_matrix(np.array([[0.0, 1.0], [0.0, 0.0]])); sm_ = sp_.T.tocsr()
sz_ = sps.csr_matrix(np.diag([0.5, -0.5]))
def at(o, i):
    return sps.kron(sps.kron(sps.identity(2 ** i), o), sps.identity(2 ** (N - i - 1)), format="csr")
SP = [at(sp_, i) for i in range(N)]; SM = [at(sm_, i) for i in range(N)]; SZ = [at(sz_, i) for i in range(N)]
Hs = sum(0.5 * (SP[i] @ SM[j] + SM[i] @ SP[j]) + SZ[i] @ SZ[j] for i, j in bonds)
ev = np.sort(np.linalg.eigvalsh(Hs.toarray()))

def c_ref(T):
    b = 1.0 / T
    x = ev - ev[0]
    w = np.exp(-b * x); p = w / w.sum()
    m = (p * x).sum()
    return b * b * (p * (x - m) ** 2).sum()

Cref = np.array([c_ref(T) for T in Ts])

def build(c):
    H = qed.Operator(N, 0.5)
    for i, j in bonds:
        H.add_two_body(qed.OP_SPLUS, i, qed.OP_SMINUS, j, 0.5)
        H.add_two_body(qed.OP_SMINUS, i, qed.OP_SPLUS, j, 0.5)
        H.add_two_body(qed.OP_SZ, i, qed.OP_SZ, j, 1.0)
    if c != 0.0:
        for i in range(N):
            H.add_two_body(qed.OP_SZ, i, qed.OP_SZ, i, 4.0 * c / N)
    return H

sym = qed.Symmetry(spatial=None)
rel = {}
try:
    for c in offsets:
        H = build(c)
        e0 = float(qed.eigs(H, 1, sym=sym).energies[0])
        if abs(e0 - (ev[0] + c)) > 1e-9 * max(1.0, abs(c)):
            print(f"REPRO: INCONCLUSIVE constant term not represented as c*I: E0={e0} expected {ev[0] + c}")
            raise SystemExit(0)
        r = qed.thermal(H, list(Ts), method="exact", sym=sym)
        C = np.asarray(r.C)
        rel[c] = np.abs(C - Cref) / Cref
        print(f"c={c:>8g}: " + "  ".join(f"T={T:.3f} C={Ci:+.3e} ref={Cr:.3e} rel={ri:.1e}"
                                      for T, Ci, Cr, ri in zip(Ts, C, Cref, rel[c])))
except SystemExit:
    raise
except Exception as ex:
    print(f"REPRO: INCONCLUSIVE raised {type(ex).__name__}: {str(ex)[:200]}")
    raise SystemExit(0)

hits = []
for k, T in enumerate(Ts):
    if Cref[k] < 1e-12 or rel[0.0][k] > 1e-3:
        continue
    for c in offsets[1:]:
        if rel[c][k] > 0.5:
            hits.append((T, c, rel[0.0][k], rel[c][k]))
            break
if hits:
    T, c, r0, rc = hits[0]
    print(f"REPRO: CONFIRMED at T={T}: rel err of C {r0:.1e} (c=0) vs {rc:.1e} (c={c:g}); "
          f"{len(hits)} temperature(s) where an added constant destroys C")
else:
    print("REPRO: NOT_REPRODUCED C stays accurate under H + c at every tested T: "
          + "; ".join(f"c={c:g} max rel {np.max(rel[c]):.1e}" for c in offsets))
