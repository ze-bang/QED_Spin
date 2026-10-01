# AUDIT-ID: C08-operator-terms-03
# DEVICE: cpu
# SECONDS: 120
"""Claim: op types outside 0..2 are never validated below the Python Operator bindings. qed.dssf
passes spin_combinations straight through static_cast<uint8_t>, so spin_combinations=[(3,3)] in
the ladder basis builds terms with op_type 3, which the gate never annihilates: the operator is
(S+ + S-) (always flips). n_up_shifts maps op 3 to shift 0, so qed.dynamics treats it as
Sz-conserving. Test: build obs with (3,3), check op types, compare Operator.apply with a dense
sum of bit flips, and compare qed.dynamics against the same operator built legitimately."""
import os
import tempfile
import signal

import numpy as np

import qed

signal.alarm(280)
out = os.environ.get("QED_REGRESS_TMP") or tempfile.mkdtemp(prefix="qed_regress_")
os.makedirs(out, exist_ok=True)
N = 8
pos = os.path.join(out, "C08-operator-terms-03_positions.dat")
with open(pos, "w") as f:
    for i in range(N):
        f.write(f"{float(i)} 0.0 0.0\n")

s = qed.dssf.OperatorSpec()
s.operator_type = "sum"
s.basis = "ladder"
s.components = [3]
s.momentum_points = [[np.pi, 0.0, 0.0]]
s.num_sites = N
s.spin_length = 0.5
s.positions_file = pos
try:
    p = qed.dssf.build_observables(s)
except Exception as e:
    print(f"REPRO: NOT_REPRODUCED op index 3 rejected: {type(e).__name__}: {e}")
    raise SystemExit(0)
Ob = p.operators[0]
terms = Ob.iter_one_body_terms()
ops = sorted({int(t[0]) for t in terms})
print(f"name={p.names[0]!r} op_types={ops} nterms={len(terms)}")

# dense: sum_i phi_i X_i with X the bit flip of unit amplitude
phi = {int(site): complex(c) for op, site, c in terms}
rng = np.random.default_rng(1)
v = rng.normal(size=2 ** N) + 1j * rng.normal(size=2 ** N)
w_ref = np.zeros_like(v)
idx = np.arange(2 ** N)
for i in range(N):
    w_ref += phi[i] * v[idx ^ (1 << i)]
w = np.asarray(Ob.apply(v))
err_flip = float(np.max(np.abs(w - w_ref)))
print(f"apply vs sum_i phi_i (S+_i + S-_i): max|diff|={err_flip:.3e}  |w|={np.linalg.norm(w):.3e}")

# legit operator with the same phases: S+ + S-
Og = qed.Operator(N, 0.5)
for i in range(N):
    Og.add_one_body(0, i, phi[i])
    Og.add_one_body(1, i, phi[i])

b = qed.input.HamiltonianBuilder(N)
b.heisenberg([(i, (i + 1) % N) for i in range(N)], J=1.0)
H = b.to_operator()
omega = np.linspace(-1.0, 4.0, 101)
try:
    Sg = np.asarray(qed.dynamics(H, Og, omega, eta=0.1).S).ravel()
except Exception as e:
    Sg = None
    print(f"dynamics(legit) raised {type(e).__name__}: {e}")
try:
    Sb = np.asarray(qed.dynamics(H, Ob, omega, eta=0.1).S).ravel()
    dyn = "ok"
except Exception as e:
    Sb = None
    dyn = f"raised {type(e).__name__}: {str(e)[:100]}"
print(f"dynamics(op 3): {dyn}")
dyn_diff = None
if Sg is not None and Sb is not None:
    dyn_diff = float(np.max(np.abs(Sg - Sb)))
    print(f"max S legit={np.max(np.abs(Sg)):.4e}  max S(op3)={np.max(np.abs(Sb)):.4e}  max|diff|={dyn_diff:.3e}")

accepted = 3 in ops
flip = err_flip < 1e-10 and np.linalg.norm(w) > 1e-6
if accepted and flip:
    extra = f"; dynamics(op3) vs legit max|dS|={dyn_diff:.3e}" if dyn_diff is not None else f"; dynamics(op3) {dyn}"
    print(f"REPRO: CONFIRMED (3,3) accepted, op_type 3 acts as S+ + S- (apply err {err_flip:.1e}){extra}")
elif accepted:
    print(f"REPRO: INCONCLUSIVE op_type 3 accepted but apply differs from S+ + S- by {err_flip:.3e}")
else:
    print(f"REPRO: NOT_REPRODUCED op types {ops}")
