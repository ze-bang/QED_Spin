# AUDIT-ID: P4-thermal-12
# DEVICE: cpu
# SECONDS: 60
"""Claim: the exact small-block fallback of the sampled thermal methods (block dim <= 512) is
switched off whenever observables are requested or a seed transform (total_spin tower) is present,
so those tiny blocks are sampled with FTLM noise although an exact solve would be cheap and exact.
Test: 10-site Heisenberg ring (every block <= 252). (a) Sz sectors: FTLM E(T) without observables
(fallback -> exact) vs with one observable (sampled). (b) total_spin=0: FTLM vs an independent dense
reference over the S=0 levels. Reference: dense numpy ED (Kronecker spin operators)."""

import numpy as np
import qed

N = 10
T = [0.3, 0.6, 1.0, 2.0]
sp = np.array([[0, 1], [0, 0]], complex)
sm = sp.T.copy()
sz = np.diag([0.5, -0.5]).astype(complex)


def site(op, i):
    m = np.array([[1.0 + 0j]])
    for j in range(N):
        m = np.kron(m, op if j == i else np.eye(2))
    return m


SP = [site(sp, i) for i in range(N)]
SM = [site(sm, i) for i in range(N)]
SZ = [site(sz, i) for i in range(N)]
Hd = sum(0.5 * (SP[i] @ SM[(i + 1) % N] + SM[i] @ SP[(i + 1) % N]) + SZ[i] @ SZ[(i + 1) % N] for i in range(N))
ev_all = np.linalg.eigvalsh(Hd)
Stot2 = sum(SM) @ sum(SP) + sum(SZ) @ sum(SZ) + sum(SZ)
w, V = np.linalg.eigh(Stot2)
P = V[:, np.abs(w) < 1e-8]
ev_s0 = np.linalg.eigvalsh(P.conj().T @ Hd @ P)


def canon(ev):
    e0 = ev.min()
    return np.array([np.sum(ev * np.exp(-(ev - e0) / t)) / np.sum(np.exp(-(ev - e0) / t)) for t in T])


E_all, E_s0 = canon(ev_all), canon(ev_s0)

H = qed.Operator(N)
for i in range(N):
    j = (i + 1) % N
    H.add_two_body(qed.OP_SZ, i, qed.OP_SZ, j, 1.0)
    H.add_two_body(qed.OP_SPLUS, i, qed.OP_SMINUS, j, 0.5)
    H.add_two_body(qed.OP_SMINUS, i, qed.OP_SPLUS, j, 0.5)
O = qed.Operator(N)
O.add_two_body(qed.OP_SZ, 0, qed.OP_SZ, 1, 1.0)

sym = qed.Symmetry(spatial=None, spin_flip="off", time_reversal="off")
r_plain = qed.thermal(H, T, method="ftlm", sym=sym, samples=20, seed=3)
r_obs = qed.thermal(H, T, method="ftlm", sym=sym, samples=20, seed=3, observables=[O])
e_plain = float(np.max(np.abs(r_plain.E - E_all)))
e_obs = float(np.max(np.abs(r_obs.E - E_all)))
print(f"(a) Sz sectors, FTLM: max|dE| no observable {e_plain:.2e}, with observable {e_obs:.2e}")

e_su2 = e_su2_exact = float("nan")
try:
    s2 = qed.Symmetry(spatial=None, total_spin=0)
    r_su2 = qed.thermal(H, T, method="ftlm", sym=s2, samples=20, seed=3)
    e_su2 = float(np.max(np.abs(r_su2.E - E_s0)))
    r_su2x = qed.thermal(H, T, method="exact", sym=s2)
    e_su2_exact = float(np.max(np.abs(r_su2x.E - E_s0)))
except Exception as ex:
    print(f"(b) total_spin raised {type(ex).__name__}: {str(ex)[:150]}")
print(f"(b) total_spin=0: FTLM max|dE| {e_su2:.2e} (method='exact' {e_su2_exact:.2e})")

hit_a = e_plain < 1e-8 and e_obs > 1e-5
hit_b = np.isfinite(e_su2) and e_su2 > 1e-5 and e_su2_exact < 1e-8
if hit_a or hit_b:
    print(
        f"REPRO: CONFIRMED blocks <= 252 sampled noisily: FTLM dE no-obs {e_plain:.1e} vs with-obs {e_obs:.1e}; "
        f"total_spin FTLM dE {e_su2:.1e} vs exact {e_su2_exact:.1e}"
    )
else:
    print(f"REPRO: NOT_REPRODUCED dE no-obs {e_plain:.1e}, with-obs {e_obs:.1e}, su2 {e_su2:.1e}")
