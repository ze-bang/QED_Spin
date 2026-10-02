# AUDIT-ID: C07-su2-06
# DEVICE: cpu
# SECONDS: 20
"""Claim: total_spin is refused for a Heisenberg model in a uniform field although [H, S^2] = 0
(lg_sectors.cpp:162 requires full SU(2)). Test: N=8 Heisenberg ring + 0.3 sum_i Sz_i;
eigs(total_spin=1) raises; dense ED confirms [H, S_tot^2] = 0."""
import numpy as np
import qed

N = 8
h = 0.3
sp = np.array([[0, 1], [0, 0]], complex); sm = sp.T.copy(); sz = np.diag([0.5, -0.5]).astype(complex)
def at(o, i):
    m = np.array([[1.0 + 0j]])
    for k in range(N):
        m = np.kron(m, o if k == i else np.eye(2))
    return m
SP = [at(sp, i) for i in range(N)]; SM = [at(sm, i) for i in range(N)]; SZ = [at(sz, i) for i in range(N)]
H = qed.Operator(N)
D = np.zeros((2 ** N, 2 ** N), complex)
for i in range(N):
    j = (i + 1) % N
    H.add_two_body(qed.OP_SPLUS, i, qed.OP_SMINUS, j, 0.5)
    H.add_two_body(qed.OP_SMINUS, i, qed.OP_SPLUS, j, 0.5)
    H.add_two_body(qed.OP_SZ, i, qed.OP_SZ, j, 1.0)
    H.add_one_body(qed.OP_SZ, i, h)
    D += 0.5 * (SP[i] @ SM[j] + SM[i] @ SP[j]) + SZ[i] @ SZ[j] + h * SZ[i]
Sp, Sm, Sz = sum(SP), sum(SM), sum(SZ)
S2 = 0.5 * (Sp @ Sm + Sm @ Sp) + Sz @ Sz
comm = float(np.max(np.abs(D @ S2 - S2 @ D)))
try:
    r = qed.eigs(H, 4, sym=qed.Symmetry(spatial=None, total_spin=1))
    print(f"REPRO: NOT_REPRODUCED total_spin=1 accepted, energies={np.asarray(r.energies)}; ||[H,S^2]||={comm:.1e}")
except Exception as ex:
    print(f"REPRO: CONFIRMED total_spin=1 refused ({type(ex).__name__}: {str(ex)[:120]}) although ||[H,S^2]||={comm:.1e}")
