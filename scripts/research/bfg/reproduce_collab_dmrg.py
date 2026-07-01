"""Reproduce collaborator's DMRG BFG Hamiltonian exactly and diagonalize.

Their construction (from the log):
  - XY (Jperp) on 48 NN "two-site" bonds (N2U triangles + N2D bonds + y-PBC wraps)
  - Ising (Jz) ONLY via 6 complete hexagons, each contributing all 15 site-pairs
  - on-site hex terms disabled (constants for spin-1/2)
  - Jz=1.0, Jperp=0.05, cylinder OBC x PBC, L=27, Sz=+/-1/2

Compare to:
  collaborator DMRG converged E = -4.67977542686
"""
import sys
from pathlib import Path
sys.path.insert(0, "/home/zhouzb79/links/projects/def-ybkim/zhouzb79/QED/python")
import qed

# ── 48 NN bonds (XY / Jperp) ────────────────────────────────────────────────
TRIANGLES = [
    (0,1,2), (9,10,11), (18,19,20), (3,4,5), (12,13,14),
    (21,22,23), (6,7,8), (15,16,17), (24,25,26),
]
nn_bonds = []
for a,b,c in TRIANGLES:
    nn_bonds += [(a,b),(a,c),(b,c)]
nn_bonds += [(1,9),(10,18),(4,12),(13,21),(7,15),(16,24)]      # N2D bond 4
nn_bonds += [(2,3),(11,12),(20,21),(5,6),(14,15),(23,24)]      # N2D bond 5
nn_bonds += [(4,11),(13,20),(7,14),(16,23)]                    # N2D bond 6
nn_bonds += [(0,8),(9,17),(18,26)]                             # PBC bond 5 wrap
nn_bonds += [(1,17),(10,26)]                                   # PBC bond 6 wrap
assert len(nn_bonds) == 48, len(nn_bonds)

# ── 6 hexagons (Ising / Jz), each → all 15 pairs ────────────────────────────
HEX_SITES = [
    [1,2,3,4,11,9],     # hex1 (0,0)
    [10,11,12,13,20,18],# hex2 (0,1)
    [4,5,6,7,14,12],    # hex3 (1,0)
    [13,14,15,16,23,21],# hex4 (1,1)
    [7,8,0,1,17,15],    # hex5 (2,0)
    [16,17,9,10,26,24], # hex6 (2,1)
]
hex_pairs = []
for sites in HEX_SITES:
    for i in range(6):
        for j in range(i+1,6):
            hex_pairs.append((sites[i], sites[j]))
assert len(hex_pairs) == 90, len(hex_pairs)

N = 27
n_up = 13            # Sz = -1/2
Jz = 1.0
Jperp = 0.05

def build(coeff_xy: float):
    op = qed.Operator(N, 0.5)
    # XY: coeff_xy * (S+S- + S-S+)
    for (i,j) in nn_bonds:
        op.add_two_body(qed.OP_SPLUS,  i, qed.OP_SMINUS, j, coeff_xy)
        op.add_two_body(qed.OP_SMINUS, i, qed.OP_SPLUS,  j, coeff_xy)
    # Ising: Jz * SzSz on hexagon pairs
    for (i,j) in hex_pairs:
        op.add_two_body(qed.OP_SZ, i, qed.OP_SZ, j, Jz)
    return op

print(f"NN(XY) bonds   : {len(nn_bonds)}")
print(f"hex(Ising)pairs: {len(hex_pairs)}  (6 hex x 15)")
print(f"Target (DMRG)  : E = -4.67977542686\n")

# Test XY conventions:  c*(S+S- + S-S+)
#   +/-Jperp        : Jperp multiplies (S+S-+S-S+)
#   +/-Jperp/2      : Jperp multiplies (SxSx+SySy) = 1/2(S+S-+S-S+)
trials = [
    (-Jperp,     "-Jperp (S+S-+h.c.)"),
    (+Jperp,     "+Jperp (S+S-+h.c.)"),
    (-Jperp/2,   "-Jperp (SxSx+SySy)"),
    (+Jperp/2,   "+Jperp (SxSx+SySy)"),
]
for coeff, label in trials:
    op = build(coeff)
    res = qed.solve(op, num_eigenvalues=4, sz=n_up,
                    solver="LANCZOS", device="cpu",
                    compute_eigenvectors=False, plan=False, force=True,
                    max_iterations=400, tolerance=1e-12, verbose=False)
    e = sorted(res.eigenvalues)
    match = "  <-- MATCH" if abs(e[0] - (-4.67977542686)) < 1e-6 else ""
    print(f"[{label:22s}]  E0 = {e[0]:.11f}   (E1={e[1]:.8f}){match}")
