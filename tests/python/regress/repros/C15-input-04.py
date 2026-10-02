# AUDIT-ID: C15-input-04
# DEVICE: cpu
# SECONDS: 30
"""Claim: HamiltonianBuilder.dm with D along z emits cancelling S+S+ / S-S- records, so a Hamiltonian that
conserves Sz (checked with a dense numpy reference) is classified Parity by sz_content and
qed.eigs(..., sym=Symmetry(spatial=None, sz=n_up)) raises 'does not conserve Sz'. Sanity: the same model
written with only the net S+S- / S-S+ records is classified U1."""
import signal

import numpy as np

import qed

signal.alarm(200)
N, J, Dz = 8, 1.0, 0.2
bonds = [(i, (i + 1) % N) for i in range(N)]
Op = qed.input.Op

b = qed.input.HamiltonianBuilder(N)
b.heisenberg(bonds, J).dm(bonds, [(0.0, 0.0, Dz)] * N)
H = b.to_operator()

b2 = qed.input.HamiltonianBuilder(N)
b2.heisenberg(bonds, J)
for i, j in bonds:   # net Dz (Sx_i Sy_j - Sy_i Sx_j) = (i Dz/2)(S+_i S-_j - S-_i S+_j)
    b2.add_two_body(Op.Sp, i, Op.Sm, j, 0.5j * Dz)
    b2.add_two_body(Op.Sm, i, Op.Sp, j, -0.5j * Dz)
H2 = b2.to_operator()

sx = np.array([[0, 0.5], [0.5, 0]], complex)
sy = np.array([[0, -0.5j], [0.5j, 0]], complex)
sz = np.array([[0.5, 0], [0, -0.5]], complex)


def site(op, i):
    out = np.array([[1.0 + 0j]])
    for k in range(N):
        out = np.kron(out, op if k == i else np.eye(2))
    return out


S = [[site(o, i) for o in (sx, sy, sz)] for i in range(N)]
Hd = sum(J * sum(S[i][a] @ S[j][a] for a in range(3)) + Dz * (S[i][0] @ S[j][1] - S[i][1] @ S[j][0])
         for i, j in bonds)
Sz = sum(S[i][2] for i in range(N))
comm = np.max(np.abs(Hd @ Sz - Sz @ Hd))
Ed = np.linalg.eigvalsh(Hd)
Eq = np.sort(np.asarray(qed.spectrum(H, sym=qed.Symmetry.none()).energies))
same = np.max(np.abs(Eq - Ed))

c1 = qed._core.sectors.sz_content(H)
c2 = qed._core.sectors.sz_content(H2)
err = None
try:
    qed.eigs(H, 1, sym=qed.Symmetry(spatial=None, sz=N // 2))
except Exception as e:  # noqa: BLE001
    err = f"{type(e).__name__}: {e}"
print(f"dense ||[H,Sz]||max={comm:.1e}, max|E_qed-E_dense|={same:.1e}, sz_content(dm)={c1}, "
      f"sz_content(net)={c2}, eigs(sz=4) -> {err}")
if same > 1e-8:
    print(f"REPRO: INCONCLUSIVE dm spectrum differs from dense reference ({same:.2e})")
elif comm < 1e-12 and str(c1).endswith("Parity") and str(c2).endswith("U1") and err is not None:
    print(f"REPRO: CONFIRMED Sz-conserving DM model classified {c1}; eigs(sz={N//2}) raised {err}")
else:
    print(f"REPRO: NOT_REPRODUCED sz_content={c1} net={c2} err={err}")
