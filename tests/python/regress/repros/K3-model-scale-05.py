# AUDIT-ID: K3-model-scale-05
# DEVICE: cpu
# SECONDS: 30
"""Claim: EigResult.matrix_element raises for any O with three-body terms
(lg_sectors_expect.cpp:72-73), although its docstring says O is arbitrary, and the value is
well defined (computed here from the full-basis vectors and O.apply)."""

import signal
import numpy as np
import qed

signal.alarm(120)
N = 8
H = qed.Operator(N)
for i in range(N):
    j = (i + 1) % N
    H.add_two_body(qed.OP_SZ, i, qed.OP_SZ, j, 1.0)
    H.add_two_body(qed.OP_SPLUS, i, qed.OP_SMINUS, j, 0.5)
    H.add_two_body(qed.OP_SMINUS, i, qed.OP_SPLUS, j, 0.5)
# chirality-like three-body probe on sites 0,1,2: S+_0 S-_1 Sz_2 - S-_0 S+_1 Sz_2 (i times)
O = qed.Operator(N)
O.add_three_body(qed.OP_SPLUS, 0, qed.OP_SMINUS, 1, qed.OP_SZ, 2, 1j)
O.add_three_body(qed.OP_SMINUS, 0, qed.OP_SPLUS, 1, qed.OP_SZ, 2, -1j)
O.add_one_body(qed.OP_SZ, 3, 0.1)
r = qed.eigs(H, 2, vectors=True, sym=qed.Symmetry(spatial=None, sz=N // 2, spin_flip="off", time_reversal="off"))
vs = r.vectors(basis="full")
ref = complex(np.vdot(vs[0], np.asarray(O.apply(np.asarray(vs[0], complex)))))
try:
    val = r.matrix_element(O, 0, 0)
    res = f"returned {val}"
except Exception as e:
    res = f"{type(e).__name__}: {str(e)[:90]}"
info = f"matrix_element(three-body O,0,0) -> {res}; full-basis reference <v0|O|v0> = {ref:.6f}"
if not res.startswith("returned"):
    print("REPRO: CONFIRMED " + info)
else:
    print("REPRO: NOT_REPRODUCED " + info)
