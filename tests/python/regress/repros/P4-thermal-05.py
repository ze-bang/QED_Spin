# AUDIT-ID: P4-thermal-05
# DEVICE: cpu
# SECONDS: 150
"""Claim: qed.thermal(method='exact', observables=[...]) diagonalises every block on the host with
Eigen::SelfAdjointEigenSolver (single-threaded tridiagonalisation + QR, all eigenvectors) and forms
U^H A U densely, so it is several times slower than threaded LAPACK eigh + a diagonal-only
contraction on the same blocks. Test: 12-site Heisenberg ring, Sz sectors only (largest block 924),
observable Sz_0 Sz_1. Time the library with and without the observable and an independent numpy
reference (complex zheevd with vectors per Sz sector + diag(U^H O U)); also check <O>(T) and E(T)
against that reference to 1e-8."""

import time
import numpy as np
import qed

N = 12
T = [0.3, 0.7, 1.5, 3.0]

H = qed.Operator(N)
for i in range(N):
    j = (i + 1) % N
    H.add_two_body(qed.OP_SZ, i, qed.OP_SZ, j, 1.0)
    H.add_two_body(qed.OP_SPLUS, i, qed.OP_SMINUS, j, 0.5)
    H.add_two_body(qed.OP_SMINUS, i, qed.OP_SPLUS, j, 0.5)
O = qed.Operator(N)
O.add_two_body(qed.OP_SZ, 0, qed.OP_SZ, 1, 1.0)


def sector(nup):
    states = [s for s in range(1 << N) if bin(s).count("1") == nup]
    idx = {s: k for k, s in enumerate(states)}
    D = len(states)
    M = np.zeros((D, D), complex)
    od = np.zeros(D)
    for k, s in enumerate(states):
        od[k] = (0.5 - (s & 1)) * (0.5 - ((s >> 1) & 1))
        for i in range(N):
            j = (i + 1) % N
            bi, bj = (s >> i) & 1, (s >> j) & 1
            M[k, k] += 0.25 if bi == bj else -0.25
            if bi != bj:
                M[idx[s ^ ((1 << i) | (1 << j))], k] += 0.5
    return M, od


mats = [sector(n) for n in range(N + 1)]
t0 = time.perf_counter()
evs, qs = [], []
for M, od in mats:
    e, U = np.linalg.eigh(M)
    evs.append(e)
    qs.append(np.real(np.einsum("ia,i,ia->a", U.conj(), od, U)))
tn = time.perf_counter() - t0
ev = np.concatenate(evs)
q = np.concatenate(qs)
e0 = ev.min()
Eref, Oref = [], []
for t in T:
    w = np.exp(-(ev - e0) / t)
    z = w.sum()
    Eref.append((w * ev).sum() / z)
    Oref.append((w * q).sum() / z)
Eref, Oref = np.array(Eref), np.array(Oref)

sym = qed.Symmetry(spatial=None, spin_flip="off", time_reversal="off")
t0 = time.perf_counter()
r0 = qed.thermal(H, T, method="exact", sym=sym)
tl0 = time.perf_counter() - t0
t0 = time.perf_counter()
r1 = qed.thermal(H, T, method="exact", sym=sym, observables=[O])
tl1 = time.perf_counter() - t0
dE = float(np.max(np.abs(r1.E - Eref)))
dO = float(np.max(np.abs(np.asarray(r1.O)[0] - Oref)))
print(f"numpy zheevd+diag (all sectors): {tn:.2f} s; library exact no-obs {tl0:.2f} s; with obs {tl1:.2f} s")
print(f"library vs dense: max|dE| {dE:.1e}, max|dO| {dO:.1e}")
ratio = tl1 / max(tn, 1e-9)
if dE > 1e-8 or dO > 1e-8:
    print(f"REPRO: INCONCLUSIVE library disagrees with dense reference (dE {dE:.1e}, dO {dO:.1e})")
elif ratio > 3.0:
    print(
        f"REPRO: CONFIRMED exact <O>(T) {tl1:.2f}s = {ratio:.1f}x numpy LAPACK reference {tn:.2f}s "
        f"(no-obs exact {tl0:.2f}s); values correct (dO {dO:.1e})"
    )
else:
    print(f"REPRO: NOT_REPRODUCED exact <O>(T) {tl1:.2f}s vs numpy {tn:.2f}s (ratio {ratio:.1f})")
