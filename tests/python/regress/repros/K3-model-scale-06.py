# AUDIT-ID: K3-model-scale-06
# DEVICE: cpu
# SECONDS: 30
"""Claim: under total_spin, qed.expect refuses every non-SU(2)-invariant O (lg_sectors_expect.cpp:33),
although the documented quantity (average over the level's symmetry multiplet) is well defined: for
Sz_0 Sz_1 in an S=0 ground state it equals <S_0.S_1>/3. Dense reference on an 8-site ring."""

import signal
import numpy as np
import qed

signal.alarm(120)
N = 8
sx = np.array([[0, 0.5], [0.5, 0]], complex)
sy = np.array([[0, -0.5j], [0.5j, 0]], complex)
sz = np.array([[0.5, 0], [0, -0.5]], complex)


def site(o, i):
    out = np.ones((1, 1), complex)
    for j in range(N):
        out = np.kron(out, o if j == i else np.eye(2))
    return out


Hd = sum(site(s, i) @ site(s, (i + 1) % N) for i in range(N) for s in (sx, sy, sz))
E, V = np.linalg.eigh(Hd)
g = V[:, 0]
ss = float(np.real(np.vdot(g, sum(site(s, 0) @ site(s, 1) for s in (sx, sy, sz)) @ g)))

H = qed.Operator(N)
for i in range(N):
    j = (i + 1) % N
    H.add_two_body(qed.OP_SZ, i, qed.OP_SZ, j, 1.0)
    H.add_two_body(qed.OP_SPLUS, i, qed.OP_SMINUS, j, 0.5)
    H.add_two_body(qed.OP_SMINUS, i, qed.OP_SPLUS, j, 0.5)
Ozz = qed.Operator(N)
Ozz.add_two_body(qed.OP_SZ, 0, qed.OP_SZ, 1, 1.0)
try:
    v = qed.expect(H, [Ozz], 1, sym=qed.Symmetry(spatial=None, total_spin=0)).values[0, 0]
    res = f"returned {v.real:.10f}"
except Exception as e:
    res = f"{type(e).__name__}: {str(e)[:90]}"
try:
    v2 = float(qed.expect(H, [Ozz], 1, sym=qed.Symmetry(spatial=None)).values[0, 0].real)
except Exception:
    v2 = float("nan")
info = (
    f"E0 gap {E[1]-E[0]:.3f}; total_spin=0 expect(Sz0Sz1) -> {res}; without total_spin {v2:.10f}; "
    f"dense <S0.S1>/3 = {ss/3:.10f}"
)
if not res.startswith("returned") and abs(v2 - ss / 3) < 1e-8:
    print("REPRO: CONFIRMED " + info)
elif res.startswith("returned"):
    print("REPRO: NOT_REPRODUCED " + info)
else:
    print("REPRO: INCONCLUSIVE " + info)
