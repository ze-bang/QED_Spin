# AUDIT-ID: C12-dynamics-09
# DEVICE: cpu
# SECONDS: 20
"""Claim: qed.dssf phase factors rest on unchecked geometry. read_positions_file silently gives
(0,0,0) to every site missing from a short positions file (no count check), and the phase is
exp(i Q.R) in absolute units although operator_spec.h documents Q 'in units of 2pi/a'.
Test: a 2-line positions file with num_sites=4 must raise; instead sites 2 and 3 get phase 1."""
import cmath
import os
import tempfile
import qed

outdir = os.environ.get("QED_REGRESS_TMP") or tempfile.mkdtemp(prefix="qed_regress_")
os.makedirs(outdir, exist_ok=True)
pos = os.path.join(outdir, "C12-dynamics-09_positions_short.dat")
with open(pos, "w") as f:
    f.write("0.0 0.0 0.0\n1.0 0.0 0.0\n")      # only 2 of 4 sites

s = qed.dssf.OperatorSpec()
s.operator_type = "sum"
s.basis = "ladder"
s.components = [2]
s.momentum_points = [[0.5, 0.0, 0.0]]          # 'half a reciprocal unit' per the header doc
s.num_sites = 4
s.positions_file = pos
try:
    p = qed.dssf.build_observables(s)
except Exception as e:
    print(f"REPRO: NOT_REPRODUCED short positions file rejected: {type(e).__name__}: {e}")
    raise SystemExit(0)
c = {int(site): complex(coef) for op, site, coef in p.operators[0].iter_one_body_terms()}
print("coefficients:", c)
padded = all(abs(c.get(i, 0) - 0.5) < 1e-12 for i in (2, 3))
abs_units = abs(c[1] - 0.5 * cmath.exp(0.5j)) < 1e-12
if padded:
    print(f"REPRO: CONFIRMED no error for 2 positions / 4 sites; sites 2,3 coeff={c[2]:.3f},{c[3]:.3f} (phase 1); "
          f"site1 phase uses Q.R in absolute units: {abs_units}")
else:
    print(f"REPRO: NOT_REPRODUCED sites 2,3 coeff={c.get(2)},{c.get(3)}")
