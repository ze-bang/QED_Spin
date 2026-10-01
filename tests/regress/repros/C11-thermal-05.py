# AUDIT-ID: C11-thermal-05
# DEVICE: cpu
# SECONDS: 60
"""Claim: when an mTPQ trajectory stops short of the coldest requested T (explicit krylov= step
count), E and var are clamped to the last iterate but C = beta_target^2 * var_last, so C grows like
1/T^2 at low T and is returned as an ordinary result (only a C++ stderr line).
Setup: 12-site Heisenberg ring, Symmetry.none() (4096 states), krylov=200 -> beta reached ~3.9.
Reference: independent dense numpy ED."""
import numpy as np
import scipy.sparse as sps
import qed

N = 12
bonds = [(i, (i + 1) % N) for i in range(N)]
Ts = np.linspace(0.05, 2.0, 40)
sp_ = sps.csr_matrix(np.array([[0.0, 1.0], [0.0, 0.0]])); sm_ = sp_.T.tocsr()
sz_ = sps.csr_matrix(np.diag([0.5, -0.5]))
def at(o, i):
    return sps.kron(sps.kron(sps.identity(2 ** i), o), sps.identity(2 ** (N - i - 1)), format="csr")
SP = [at(sp_, i) for i in range(N)]; SM = [at(sm_, i) for i in range(N)]; SZ = [at(sz_, i) for i in range(N)]
Hs = sum(0.5 * (SP[i] @ SM[j] + SM[i] @ SP[j]) + SZ[i] @ SZ[j] for i, j in bonds)
ev = np.linalg.eigvalsh(Hs.toarray())
Cex = []
for T in Ts:
    b = 1 / T; w = np.exp(-b * (ev - ev[0])); z = w.sum(); e = (w * ev).sum() / z
    Cex.append(b * b * ((w * ev * ev).sum() / z - e * e))
Cex = np.array(Cex)

H = qed.Operator(N, 0.5)
for i, j in bonds:
    H.add_two_body(qed.OP_SPLUS, i, qed.OP_SMINUS, j, 0.5)
    H.add_two_body(qed.OP_SMINUS, i, qed.OP_SPLUS, j, 0.5)
    H.add_two_body(qed.OP_SZ, i, qed.OP_SZ, j, 1.0)
try:
    r = qed.thermal(H, list(Ts), method="mtpq", krylov=200, samples=50, seed=9, sym=qed.Symmetry.none())
except Exception as ex:
    print(f"REPRO: NOT_REPRODUCED thermal refused: {type(ex).__name__}: {str(ex)[:200]}")
    raise SystemExit(0)
C = np.asarray(r.C)
print("T     C_mtpq     C_exact")
for i in (0, 1, 2, 4, 6, 10, 20, 39):
    print(f"{Ts[i]:.3f} {C[i]:10.4f} {Cex[i]:10.4f}")
if C[0] > 1.0 and C[0] > 10 * Cex[0] and C[0] > 2.0 * C[2]:
    print(f"REPRO: CONFIRMED C(T=0.05)={C[0]:.3f} vs exact {Cex[0]:.4f}; C(0.05)/C({Ts[2]:.3f})={C[0]/C[2]:.2f}, no exception")
else:
    print(f"REPRO: NOT_REPRODUCED C(T=0.05)={C[0]:.4f} exact {Cex[0]:.4f}")
