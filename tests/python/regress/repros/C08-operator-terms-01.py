# AUDIT-ID: C08-operator-terms-01
# DEVICE: cpu
# SECONDS: 30
"""Claim: qed.dssf.build_observable_pairs never checks the length of each momentum point for
operator_type 'sum'/'sublattice'/'experimental': compute_phase_factors and the name builder
read Q[0], Q[1], Q[2] unconditionally. A 2-component Q is accepted (out-of-bounds read of Q[2]);
an empty Q dereferences an empty vector and crashes the interpreter.
Test: (a) Q=[pi, 0] with operator_type='sum' must raise; (b) Q=[] in a child process must
raise a Python exception rather than kill the process."""

import os
import tempfile
import subprocess
import sys

import qed

out = os.environ.get("QED_REGRESS_TMP") or tempfile.mkdtemp(prefix="qed_regress_")
os.makedirs(out, exist_ok=True)
pos = os.path.join(out, "C08-operator-terms-01_positions.dat")
with open(pos, "w") as f:
    for i in range(4):
        f.write(f"{float(i)} 0.0 0.0\n")

findings = []
for typ in ("sum", "sublattice", "experimental"):
    s = qed.dssf.OperatorSpec()
    s.operator_type = typ
    s.basis = "ladder"
    s.components = [2]
    s.momentum_points = [[3.141592653589793, 0.0]]
    s.num_sites = 4
    s.unit_cell_size = 2
    s.positions_file = pos
    try:
        p = qed.dssf.build_observables(s)
        findings.append(f"{typ}:2-vector Q accepted (names[0]={p.names[0]!r})")
    except Exception as e:
        print(f"{typ}: 2-vector Q rejected: {type(e).__name__}: {e}")

CHILD = (
    r'''
import qed
s = qed.dssf.OperatorSpec()
s.operator_type = "sum"; s.basis = "ladder"; s.components = [2]
s.momentum_points = [[]]; s.num_sites = 4; s.positions_file = %r
try:
    qed.dssf.build_observables(s)
    print("CHILD accepted")
except Exception as e:
    print("CHILD raised", type(e).__name__, e)
'''
    % pos
)
try:
    r = subprocess.run([sys.executable, "-c", CHILD], capture_output=True, text=True, timeout=120)
    tail = (r.stdout.strip().splitlines() or [""])[-1]
    print(f"empty Q child: rc={r.returncode} out={tail!r}")
    if r.returncode != 0:
        findings.append(f"empty Q killed the interpreter rc={r.returncode}")
    elif "accepted" in tail:
        findings.append("empty Q accepted")
except subprocess.TimeoutExpired:
    findings.append("empty Q child hung")

if findings:
    print("REPRO: CONFIRMED " + "; ".join(findings))
else:
    print("REPRO: NOT_REPRODUCED short and empty Q are rejected")
