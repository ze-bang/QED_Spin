# AUDIT-ID: C08-operator-terms-05
# DEVICE: cpu
# SECONDS: 20
"""Claim: read_positions_file accepts a line only if three doubles parse, so a two-column (2D)
positions file is skipped line by line without error; every site sits at the origin and the
phase factors do not depend on Q. Test: 4-site 2-column file, Q=(0,0,0) and (pi,0,0)."""
import os
import tempfile

import numpy as np

import qed

out = os.environ.get("QED_REGRESS_TMP") or tempfile.mkdtemp(prefix="qed_regress_")
os.makedirs(out, exist_ok=True)
pos = os.path.join(out, "C08-operator-terms-05_positions_2col.dat")
with open(pos, "w") as f:
    for i in range(4):
        f.write(f"{float(i)} 0.0\n")

s = qed.dssf.OperatorSpec()
s.operator_type = "sum"
s.basis = "ladder"
s.spin_combinations = [(2, 2)]
s.momentum_points = [[0.0, 0.0, 0.0], [np.pi, 0.0, 0.0]]
s.num_sites = 4
s.spin_length = 0.5
s.positions_file = pos
s.single_obs_only = True
try:
    p = qed.dssf.build_observable_pairs(s)
except Exception as e:
    print(f"REPRO: NOT_REPRODUCED 2-column file rejected: {type(e).__name__}: {e}")
    raise SystemExit(0)
c0 = {int(st): complex(c) for op, st, c in p.obs_1[0].iter_one_body_terms()}
c1 = {int(st): complex(c) for op, st, c in p.obs_1[1].iter_one_body_terms()}
print("Q=0 :", c0)
print("Q=pi:", c1)
diff = max(abs(c0[i] - c1[i]) for i in range(4))
expect_pi = [0.5 * np.exp(1j * np.pi * i) for i in range(4)]
err_pi = max(abs(c1[i] - expect_pi[i]) for i in range(4))
if diff < 1e-12:
    print(f"REPRO: CONFIRMED 2-column positions accepted silently; S(Q=0) and S(Q=pi) coefficients identical "
          f"(max diff {diff:.1e}, expected Q=pi phases off by {err_pi:.2f})")
else:
    print(f"REPRO: NOT_REPRODUCED phases differ by {diff:.3e}")
