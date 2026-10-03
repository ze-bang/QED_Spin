# AUDIT-ID: C12-dynamics-06
# DEVICE: cpu
# SECONDS: 40
"""Claim: no verb checks that O acts on H's sites. An O built on more sites is accepted; in dynamics
and matrix_element a term on a site >= N acts on a phantom bit (Sz there contributes a constant +1/2),
so results are silently wrong; in expect/thermal the site indexes past the N-entry permutation (UB).
Test: 8-site Heisenberg ring, O = qed.Operator(12) with Sz on all 12 sites. The physical part
sum_{i<8} Sz_i annihilates the singlet ground state, so a correct library raises (or gives 0); the
claim predicts weight 4 at omega=0 and <GS|O|GS> = 2. expect is run in a subprocess (possible UB)."""

import subprocess
import sys
import textwrap
import numpy as np
import qed

N, NO = 8, 12
eta = 0.05
H = qed.Operator(N)
for i in range(N):
    j = (i + 1) % N
    H.add_two_body(qed.OP_SZ, i, qed.OP_SZ, j, 1.0)
    H.add_two_body(qed.OP_SPLUS, i, qed.OP_SMINUS, j, 0.5)
    H.add_two_body(qed.OP_SMINUS, i, qed.OP_SPLUS, j, 0.5)
O = qed.Operator(NO)
for i in range(NO):
    O.add_one_body(qed.OP_SZ, i, 1.0)
t = [(i + 1) % N for i in range(N)]
sym = qed.Symmetry(spatial=[t], point_group=False, spin_flip="off", time_reversal="off")
notes = []
try:
    r = qed.dynamics(H, O, np.array([0.0, 1.0]), eta=eta, sym=sym)
    wgt = float(np.asarray(r.S[0])[0] * np.pi * eta)
    notes.append(f"dynamics returned, weight at omega=0 = {wgt:.6f} (correct: refusal / 0)")
    dyn_bad = abs(wgt) > 1e-6
except Exception as e:
    notes.append(f"dynamics raised {type(e).__name__}: {str(e)[:100]}")
    dyn_bad = False
try:
    ev = qed.eigs(H, 1, sym=sym, vectors=True)
    me = ev.matrix_element(O, 0, 0)
    notes.append(f"matrix_element returned {me.real:.6f} (correct: refusal / 0)")
    me_bad = abs(me) > 1e-6
except Exception as e:
    notes.append(f"matrix_element raised {type(e).__name__}: {str(e)[:100]}")
    me_bad = False
code = textwrap.dedent(f"""
    import qed
    N, NO = {N}, {NO}
    H = qed.Operator(N)
    for i in range(N):
        j = (i + 1) % N
        H.add_two_body(qed.OP_SZ, i, qed.OP_SZ, j, 1.0)
        H.add_two_body(qed.OP_SPLUS, i, qed.OP_SMINUS, j, 0.5)
        H.add_two_body(qed.OP_SMINUS, i, qed.OP_SPLUS, j, 0.5)
    O = qed.Operator(NO)
    O.add_one_body(qed.OP_SZ, NO - 1, 1.0)
    t = [(i + 1) % N for i in range(N)]
    sym = qed.Symmetry(spatial=[t], point_group=False, spin_flip="off", time_reversal="off")
    try:
        print("expect:", qed.eigs(H, 1, sym=sym, vectors=True).expect([O]).tolist())
    except Exception as e:
        print("expect raised", type(e).__name__, str(e)[:100])
""")
try:
    p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
    notes.append(f"expect subprocess rc={p.returncode} out={p.stdout.strip()[-120:]!r}")
except subprocess.TimeoutExpired:
    notes.append("expect subprocess timed out")
print("\n".join(notes))
if dyn_bad or me_bad:
    print("REPRO: CONFIRMED oversized O accepted silently: " + " | ".join(notes))
else:
    print("REPRO: NOT_REPRODUCED " + " | ".join(notes))
