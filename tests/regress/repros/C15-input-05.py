# AUDIT-ID: C15-input-05
# DEVICE: cpu
# SECONDS: 10
"""Claim: lattice.from_cluster_file mis-parses its input silently: (a) a lone count line in the positions
block becomes an extra (phantom) site; (b) an edge block headed 'BONDS' is not recognised, so every edge
line becomes a site and the lattice has no bonds; (c) the documented 'id x y z' positions form is shifted
by one column (z lost)."""
import os
import tempfile

import qed

out = os.path.join(os.environ.get("QED_REGRESS_TMP") or tempfile.mkdtemp(prefix="qed_regress_"), "C15-input-05")
os.makedirs(out, exist_ok=True)
sq = ["0 0 0", "1 0 0", "1 1 0", "0 1 0"]
edges = ["0 1", "1 2", "2 3", "3 0"]
files = {
    "a_count_line": ["positions", "4"] + sq + ["edges"] + edges,
    "b_BONDS_header": ["positions"] + sq + ["BONDS"] + edges,
    "c_id_x_y_z": ["positions"] + [f"{k} {s}" for k, s in
                                   enumerate(["0.0 0.1 0.2", "1.0 0.1 0.2", "1.0 1.1 0.2", "0.0 1.1 0.2"])]
                  + ["edges"] + edges,
}
res = {}
for name, lines in files.items():
    p = os.path.join(out, name + ".txt")
    with open(p, "w") as f:
        f.write("\n".join(lines) + "\n")
    try:
        lat = qed.input.lattice.from_cluster_file(p)
        res[name] = dict(n=lat.num_sites, nb=len(lat.nn_bonds), p0=[round(x, 3) for x in lat.positions[0]])
    except Exception as e:  # noqa: BLE001
        res[name] = dict(err=f"{type(e).__name__}: {e}")
print(res)
bad = []
if res["a_count_line"].get("n") == 5:
    bad.append("count line -> num_sites 5 (expected 4)")
if res["b_BONDS_header"].get("n") == 8 and res["b_BONDS_header"].get("nb") == 0:
    bad.append("BONDS header -> num_sites 8, 0 bonds")
if res["c_id_x_y_z"].get("p0") == [0.0, 0.0, 0.1]:
    bad.append("'id x y z' -> position (0,0,0.1) instead of (0,0.1,0.2)")
if bad:
    print("REPRO: CONFIRMED " + "; ".join(bad))
else:
    print(f"REPRO: NOT_REPRODUCED {res}")
