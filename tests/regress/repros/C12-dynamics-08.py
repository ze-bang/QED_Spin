# AUDIT-ID: C12-dynamics-08
# DEVICE: cpu
# SECONDS: 20
"""Claim: qed.dssf.build_observable_pairs with operator_type='sublattice' and basis='xyz'
ignores the Cartesian basis: index 0 builds a pure S+ operator (add_sublattice is ladder-only)
while the emitted name says 'Sx'. With unit_cell_size=1 the sublattice operator must equal the
'sum' operator, which does honour basis='xyz' (S+/2 + S-/2 per site)."""
import os
import tempfile
import numpy as np
import qed

outdir = os.environ.get("QED_REGRESS_TMP") or tempfile.mkdtemp(prefix="qed_regress_")
os.makedirs(outdir, exist_ok=True)
pos = os.path.join(outdir, "C12-dynamics-08_positions.dat")
N = 4
with open(pos, "w") as f:
    for i in range(N):
        f.write(f"{float(i):.6e} 0.0 0.0\n")


def build(kind):
    s = qed.dssf.OperatorSpec()
    s.operator_type = kind
    s.basis = "xyz"
    s.spin_combinations = [(0, 0)]
    s.momentum_points = [[0.7, 0.0, 0.0]]
    s.unit_cell_size = 1
    s.num_sites = N
    s.spin_length = 0.5
    s.positions_file = pos
    s.single_obs_only = True
    p = qed.dssf.build_observable_pairs(s)
    terms = {}
    for op, site, c in p.obs_1[0].iter_one_body_terms():
        terms[(int(op), int(site))] = terms.get((int(op), int(site)), 0) + complex(c)
    return p.names[0], terms


try:
    n_sub, t_sub = build("sublattice")
    n_sum, t_sum = build("sum")
except Exception as e:
    print(f"REPRO: INCONCLUSIVE builder raised {type(e).__name__}: {e}")
    raise SystemExit(0)
print("sublattice:", n_sub, sorted(t_sub.items()))
print("sum       :", n_sum, sorted(t_sum.items()))
keys = set(t_sub) | set(t_sum)
diff = max(abs(t_sub.get(k, 0) - t_sum.get(k, 0)) for k in keys)
ops_sub = sorted({k[0] for k in t_sub})
if diff > 1e-12 and n_sub.startswith("Sx"):
    print(f"REPRO: CONFIRMED sublattice name={n_sub!r} but op types {ops_sub} (sum xyz has both S+ and S-); max coeff diff={diff:.3f}")
else:
    print(f"REPRO: NOT_REPRODUCED sublattice matches sum (diff={diff:.2e}, name={n_sub!r}, ops={ops_sub})")
