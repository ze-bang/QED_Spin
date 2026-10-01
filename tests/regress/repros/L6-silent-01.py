# AUDIT-ID: L6-silent-01
# DEVICE: cpu
# SECONDS: 90
"""Claim: mTPQ ln Z / S / F per block come from a trapezoid integral of E over the CALLER's
temperature grid only (plus beta=0), not the dense trajectory; ln Z is also the block weight
when sectors are combined. (a) A single-T call returns ln Z, S, F wrong by several units.
(b) Under Sz symmetry, the sampled blocks get distorted weights against the exactly solved small
blocks, so even E depends on the grid: one-point grid vs a dense grid with the same coldest T
(identical trajectories, same seed) give different E(T).
Setup: 12-site Heisenberg ring. Reference: independent dense numpy ED."""
import numpy as np
import scipy.sparse as sps
import qed

N = 12
bonds = [(i, (i + 1) % N) for i in range(N)]
T0 = 0.3
sp_ = sps.csr_matrix(np.array([[0.0, 1.0], [0.0, 0.0]])); sm_ = sp_.T.tocsr()
sz_ = sps.csr_matrix(np.diag([0.5, -0.5]))
def at(o, i):
    return sps.kron(sps.kron(sps.identity(2 ** i), o), sps.identity(2 ** (N - i - 1)), format="csr")
SP = [at(sp_, i) for i in range(N)]; SM = [at(sm_, i) for i in range(N)]; SZ = [at(sz_, i) for i in range(N)]
Hs = sum(0.5 * (SP[i] @ SM[j] + SM[i] @ SP[j]) + SZ[i] @ SZ[j] for i, j in bonds)
ev = np.linalg.eigvalsh(Hs.toarray())
b = 1 / T0; w = np.exp(-b * (ev - ev[0])); z = w.sum()
lnZ_ex = np.log(z) - b * ev[0]; E_ex = (w * ev).sum() / z; S_ex = lnZ_ex + b * E_ex

H = qed.Operator(N)
for i, j in bonds:
    H.add_two_body(qed.OP_SPLUS, i, qed.OP_SMINUS, j, 0.5)
    H.add_two_body(qed.OP_SMINUS, i, qed.OP_SPLUS, j, 0.5)
    H.add_two_body(qed.OP_SZ, i, qed.OP_SZ, j, 1.0)

dense = list(np.linspace(T0, 10.0, 400))
try:
    A = qed.thermal(H, [T0], method="mtpq", samples=60, seed=5, sym=qed.Symmetry.none())
    B = qed.thermal(H, dense, method="mtpq", samples=60, seed=5, sym=qed.Symmetry.none())
    symz = qed.Symmetry(spatial=None, spin_flip="off", time_reversal="off")
    A2 = qed.thermal(H, [T0], method="mtpq", samples=60, seed=5, sym=symz)
    B2 = qed.thermal(H, dense, method="mtpq", samples=60, seed=5, sym=symz)
except Exception as ex:
    print(f"REPRO: INCONCLUSIVE thermal raised {type(ex).__name__}: {str(ex)[:200]}")
    raise SystemExit(0)

print(f"exact T={T0}: lnZ={lnZ_ex:.4f} S={S_ex:.4f} E={E_ex:.5f}")
print(f"(a) one-point grid: lnZ={A.lnZ[0]:.4f} S={A.S[0]:.4f} E={A.E[0]:.5f}")
print(f"(a) dense grid    : lnZ={B.lnZ[0]:.4f} S={B.S[0]:.4f} E={B.E[0]:.5f}")
print(f"(b) Sz blocks={A2.blocks}: one-point E={A2.E[0]:.5f} lnZ={A2.lnZ[0]:.4f}; dense E={B2.E[0]:.5f} lnZ={B2.lnZ[0]:.4f}")
a_bad = abs(A.lnZ[0] - lnZ_ex) > 1.0 and abs(B.lnZ[0] - lnZ_ex) < 0.5 * abs(A.lnZ[0] - lnZ_ex)
b_bad = abs(A2.E[0] - B2.E[0]) > 0.02 and abs(A2.E[0] - E_ex) > abs(B2.E[0] - E_ex)
if a_bad or b_bad:
    print(f"REPRO: CONFIRMED dlnZ(one-point)={A.lnZ[0]-lnZ_ex:.3f} vs dlnZ(dense)={B.lnZ[0]-lnZ_ex:.3f}; "
          f"Sz-resolved E one-point {A2.E[0]:.4f} vs dense {B2.E[0]:.4f} vs exact {E_ex:.4f}")
else:
    print(f"REPRO: NOT_REPRODUCED dlnZ(one-point)={A.lnZ[0]-lnZ_ex:.3f} dlnZ(dense)={B.lnZ[0]-lnZ_ex:.3f}; "
          f"E one-point {A2.E[0]:.4f} dense {B2.E[0]:.4f} exact {E_ex:.4f}")
