"""Sweep E0 vs Jpm (our convention) on the collaborator's exact bond set.

Our convention:  H_XY = -Jpm * sum_<ij> (S+_i S-_j + S-_i S+_j)
Bonds: 48 NN (XY) + 6 hexagons x 15 Ising pairs, Jz=1.
At Jpm = 0.025 this equals their Jperp=0.05 -> DMRG E = -4.67977542686.
"""
import sys, json
sys.path.insert(0, "/home/zhouzb79/links/projects/def-ybkim/zhouzb79/QED/python")
import qed

TRIANGLES = [
    (0,1,2), (9,10,11), (18,19,20), (3,4,5), (12,13,14),
    (21,22,23), (6,7,8), (15,16,17), (24,25,26),
]
nn_bonds = []
for a,b,c in TRIANGLES:
    nn_bonds += [(a,b),(a,c),(b,c)]
nn_bonds += [(1,9),(10,18),(4,12),(13,21),(7,15),(16,24)]
nn_bonds += [(2,3),(11,12),(20,21),(5,6),(14,15),(23,24)]
nn_bonds += [(4,11),(13,20),(7,14),(16,23)]
nn_bonds += [(0,8),(9,17),(18,26)]
nn_bonds += [(1,17),(10,26)]
assert len(nn_bonds) == 48

HEX_SITES = [
    [1,2,3,4,11,9], [10,11,12,13,20,18], [4,5,6,7,14,12],
    [13,14,15,16,23,21], [7,8,0,1,17,15], [16,17,9,10,26,24],
]
hex_pairs = []
for s in HEX_SITES:
    for i in range(6):
        for j in range(i+1,6):
            hex_pairs.append((s[i], s[j]))
assert len(hex_pairs) == 90

N, n_up, Jz = 27, 13, 1.0

def E0(Jpm: float) -> float:
    op = qed.Operator(N, 0.5)
    for (i,j) in nn_bonds:                       # H_XY = -Jpm (S+S- + S-S+)
        op.add_two_body(qed.OP_SPLUS,  i, qed.OP_SMINUS, j, -Jpm)
        op.add_two_body(qed.OP_SMINUS, i, qed.OP_SPLUS,  j, -Jpm)
    for (i,j) in hex_pairs:                       # Ising Jz Sz Sz
        op.add_two_body(qed.OP_SZ, i, qed.OP_SZ, j, Jz)
    res = qed.solve(op, num_eigenvalues=2, sz=n_up, solver="LANCZOS",
                    device="cpu", compute_eigenvectors=False, plan=False,
                    force=True, max_iterations=400, tolerance=1e-12, verbose=False)
    return min(res.eigenvalues)

Jpm_vals = [-0.10, -0.05, -0.025, 0.0, 0.0125, 0.025, 0.0375, 0.05, 0.075, 0.10]
out = []
for Jpm in Jpm_vals:
    e = E0(Jpm)
    out.append((Jpm, e))
    print(f"Jpm={Jpm:+.4f}  E0={e:.11f}", flush=True)

with open("/scratch/zhouzb79/bfg_gs_results/collab_jpm_sweep.json", "w") as f:
    json.dump(out, f)
print("done")
