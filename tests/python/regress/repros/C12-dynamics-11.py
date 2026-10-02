# AUDIT-ID: C12-dynamics-11
# DEVICE: cpu
# SECONDS: 40
"""Claim: qed.dynamics silently ignores Symmetry.select (unfolded() clears only_k0, only_irrep,
only_momentum, only_irrep_chars) and the spin_flip='require' check (it zeroes spin_flip), while
eigs honours both. Test: 8-site Heisenberg ring; dynamics with select(momentum={T: 1/2}) equals the
unselected result although eigs with the same selection returns only k=pi levels; with a field,
spin_flip='require' raises in eigs but not in dynamics."""
import cmath
from fractions import Fraction
import numpy as np
import qed

N = 8
t = [(i + 1) % N for i in range(N)]


def heis(h=0.0):
    H = qed.Operator(N)
    for i in range(N):
        j = (i + 1) % N
        H.add_two_body(qed.OP_SZ, i, qed.OP_SZ, j, 1.0)
        H.add_two_body(qed.OP_SPLUS, i, qed.OP_SMINUS, j, 0.5)
        H.add_two_body(qed.OP_SMINUS, i, qed.OP_SPLUS, j, 0.5)
        if h:
            H.add_one_body(qed.OP_SZ, i, h)
    return H


O = qed.Operator(N)
for j in range(N):
    O.add_one_body(qed.OP_SZ, j, cmath.exp(-1j * np.pi * j) / np.sqrt(N))
omega = np.linspace(0.0, 4.0, 201)
H = heis()
base = qed.Symmetry(spatial=[t], point_group=False, spin_flip="off", time_reversal="off")
sel = base.select(momentum={tuple(t): Fraction(1, 2)})
e_all = qed.eigs(H, 4, sym=base).energies
e_sel = qed.eigs(H, 4, sym=sel).energies
S_all = np.asarray(qed.dynamics(H, O, omega, sym=base).S[0])
S_sel = np.asarray(qed.dynamics(H, O, omega, sym=sel).S[0])
same = bool(np.allclose(S_all, S_sel, rtol=1e-12, atol=1e-14))
print(f"eigs unselected {np.round(e_all, 6).tolist()} / k=pi selected {np.round(e_sel, 6).tolist()}")
print(f"dynamics selected == unselected: {same}")
Hf = heis(0.1)
req = qed.Symmetry(spatial=[t], point_group=False, spin_flip="require")
try:
    qed.eigs(Hf, 1, sym=req)
    eig_req = "eigs did not raise"
except Exception as e:
    eig_req = f"eigs raised {type(e).__name__}"
try:
    qed.dynamics(Hf, O, omega, sym=req)
    dyn_req = "dynamics did not raise"
except Exception as e:
    dyn_req = f"dynamics raised {type(e).__name__}"
print(eig_req, "|", dyn_req)
selected_differs = len(e_all) != len(e_sel) or not np.allclose(e_all, e_sel)
if (same and selected_differs) or (eig_req.startswith("eigs raised") and dyn_req.endswith("did not raise")):
    print(f"REPRO: CONFIRMED select ignored by dynamics: {same and selected_differs}; require: {eig_req} / {dyn_req}")
else:
    print(f"REPRO: NOT_REPRODUCED same={same} eigs-selection-differs={selected_differs}; {eig_req} / {dyn_req}")
