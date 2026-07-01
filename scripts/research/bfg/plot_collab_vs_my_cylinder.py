"""Compare collaborator's canonical-BFG cylinder geometry against my cylinder.

Left  panel: collaborator's lattice (their site numbering, coords read off the
             ASCII traversal diagram). 48 NN bonds carry the XY (Jperp) term;
             6 complete hexagons carry the Ising (Jz) term as 15 pairs each.
             One hexagon is exploded into its 6 NN + 6 2NN + 3 3NN Ising pairs.

Right panel: my CYL cluster (generate_kagome_cluster, pbc_dim1=1, pbc_dim2=0).
             NN bonds (XY) solid; 2NN/3NN Ising bonds faint. The 6 complete
             hexagons are shaded. Boundary 2NN/3NN bonds that my "full" run
             includes but the hexagon-only Ising does NOT are flagged.
"""
import sys, itertools
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import Polygon
import matplotlib.patches as mpatches

sys.path.insert(0, "/home/zhouzb79/links/projects/def-ybkim/zhouzb79/QED/python")
from edlib.helper_kagome_bfg import generate_kagome_cluster, create_nn_lists

YS = 5.196  # vertical scale so triangles look equilateral (6*sqrt(3)/2)

# ── Collaborator coordinates (read off the ASCII diagram) ───────────────────
# (char_x, level)  level 0 = top row
COLLAB_XY_CHAR = {
    0:(0,0), 1:(6,0), 9:(12,0), 10:(18,0), 18:(24,0), 19:(30,0),
    2:(3,1), 11:(15,1), 20:(27,1),
    3:(6,2), 4:(12,2), 12:(18,2), 13:(24,2), 21:(30,2), 22:(36,2),
    5:(9,3), 14:(21,3), 23:(33,3),
    6:(12,4), 7:(18,4), 15:(24,4), 16:(30,4), 24:(36,4), 25:(42,4),
    8:(15,5), 17:(27,5), 26:(39,5),
}
collab_pos = {s: (cx, -lvl * YS) for s, (cx, lvl) in COLLAB_XY_CHAR.items()}

# 48 NN bonds (XY / Jperp)
TRIANGLES = [(0,1,2),(9,10,11),(18,19,20),(3,4,5),(12,13,14),
             (21,22,23),(6,7,8),(15,16,17),(24,25,26)]
collab_nn = []
for a,b,c in TRIANGLES:
    collab_nn += [(a,b),(a,c),(b,c)]
collab_nn += [(1,9),(10,18),(4,12),(13,21),(7,15),(16,24)]   # N2D 4
collab_nn += [(2,3),(11,12),(20,21),(5,6),(14,15),(23,24)]   # N2D 5
collab_nn += [(4,11),(13,20),(7,14),(16,23)]                 # N2D 6
collab_pbc = [(0,8),(9,17),(18,26),(1,17),(10,26)]           # PBC (bcy) wraps
collab_nn += collab_pbc
assert len(collab_nn) == 48

# 6 hexagons, ordered rings
COLLAB_HEX = [
    [1,2,3,4,11,9], [10,11,12,13,20,18], [4,5,6,7,14,12],
    [13,14,15,16,23,21], [7,8,0,1,17,15], [16,17,9,10,26,24],
]

def ring_dist(ring, a, b):
    ia, ib = ring.index(a), ring.index(b)
    d = abs(ia - ib)
    return min(d, len(ring) - d)

# ── My CYL cluster ──────────────────────────────────────────────────────────
verts, bnn, b2nn, b3nn, nmap, v2cell = generate_kagome_cluster(
    3, 3, use_pbc=False, pbc_dim1=True, pbc_dim2=False)
_, _, _, mypos, _ = create_nn_lists(bnn, b2nn, b3nn, nmap, verts, v2cell)

# reference hexagons (from full PBC) and which are complete on the cylinder
_, bnn_ref, _, b3nn_ref, _, _ = generate_kagome_cluster(3, 3, use_pbc=True)
adj_ref = {i:set() for i in range(27)}
for a,b in bnn_ref: adj_ref[a].add(b); adj_ref[b].add(a)
my_hexes = {}
for a,d in b3nn_ref:
    paths=[]
    for x in adj_ref[a]:
        for y in adj_ref[x]:
            if y!=a and y in adj_ref[d] and y!=d and x!=d:
                paths.append((x,y))
    if len(paths)==2:
        p1,p2=paths
        my_hexes[frozenset([a,p1[0],p1[1],d,p2[0],p2[1]])]=True
my_hexes=list(my_hexes.keys())
my_bondset=set(frozenset(e) for e in bnn+b2nn+b3nn)
def complete(h):
    s=list(h)
    return all(frozenset((s[i],s[j])) in my_bondset for i in range(6) for j in range(i+1,6))
my_complete=[h for h in my_hexes if complete(h)]
complete_pairs=set()
for h in my_complete:
    s=list(h)
    for i in range(6):
        for j in range(i+1,6):
            complete_pairs.add(frozenset((s[i],s[j])))

# ════════════════════════════════════════════════════════════════════════════
fig, (axL, axR) = plt.subplots(1, 2, figsize=(17, 7.5))

# ── LEFT: collaborator ──────────────────────────────────────────────────────
axL.set_title("Collaborator — canonical BFG cylinder\n"
              "XY (Jperp) on 48 NN bonds  ·  Ising (Jz) on 6 hexagons (15 pairs each)",
              fontsize=10)

# shade hexagons
for ring in COLLAB_HEX:
    poly = np.array([collab_pos[s] for s in ring])
    axL.add_patch(Polygon(poly, closed=True, facecolor="#FFE0B2",
                          edgecolor="#FB8C00", lw=1.5, alpha=0.6, zorder=1))

# NN bonds
segs = [[collab_pos[a], collab_pos[b]] for a,b in collab_nn if (a,b) not in collab_pbc]
axL.add_collection(LineCollection(segs, colors="#212121", lw=2.0, zorder=3))
# PBC wrap bonds dashed
segs_pbc = [[collab_pos[a], collab_pos[b]] for a,b in collab_pbc]
axL.add_collection(LineCollection(segs_pbc, colors="#7B1FA2", lw=1.6,
                                  linestyles="dashed", zorder=3))

# explode one hexagon (Hex1) into its 15 Ising pairs by ring distance
ring = COLLAB_HEX[0]
cd = {1:("#2E7D32","NN  (6)"), 2:("#1565C0","2NN (6)"), 3:("#C62828","3NN (3)")}
for a,b in itertools.combinations(ring,2):
    d = ring_dist(ring,a,b)
    col,_ = cd[d]
    p,q = collab_pos[a], collab_pos[b]
    axL.plot([p[0],q[0]],[p[1],q[1]], color=col, lw=1.4,
             ls="-" if d==1 else (":" if d==3 else "--"), zorder=2, alpha=0.9)

# sites
for s,(x,y) in collab_pos.items():
    axL.plot(x,y,"o",ms=15,color="white",mec="k",mew=1.2,zorder=4)
    axL.annotate(f"{s:02d}",(x,y),ha="center",va="center",fontsize=7,zorder=5)

axL.set_aspect("equal"); axL.axis("off")

# ── RIGHT: my cylinder ──────────────────────────────────────────────────────
axR.set_title("My CYL cluster  (pbc_dim1=1, pbc_dim2=0)\n"
              "NN solid · 2NN/3NN Ising faint · boundary 2NN/3NN (red) absent "
              "from hexagon-only Ising", fontsize=10)

def is_wrap(a,b):
    pa,pb = np.array(mypos[a]), np.array(mypos[b])
    return np.linalg.norm(pa-pb) > 1.8   # wrapped bonds are drawn long

# shade complete hexagons
for h in my_complete:
    pts=np.array([mypos[s] for s in h])
    c=pts.mean(0)
    ang=np.arctan2(pts[:,1]-c[1], pts[:,0]-c[0])
    poly=pts[np.argsort(ang)]
    axR.add_patch(Polygon(poly, closed=True, facecolor="#FFE0B2",
                          edgecolor="#FB8C00", lw=1.5, alpha=0.6, zorder=1))

# 2NN / 3NN Ising bonds: faint if inside complete hexagon, red if boundary-only
for bonds,lab in [(b2nn,"2NN"),(b3nn,"3NN")]:
    for a,b in bonds:
        if is_wrap(a,b): continue
        inside = frozenset((a,b)) in complete_pairs
        col = "#BDBDBD" if inside else "#E53935"
        lw  = 1.0 if inside else 2.0
        axR.plot([mypos[a][0],mypos[b][0]],[mypos[a][1],mypos[b][1]],
                 color=col, lw=lw, ls=":", zorder=2)

# NN bonds (XY) solid
for a,b in bnn:
    if is_wrap(a,b): continue
    axR.plot([mypos[a][0],mypos[b][0]],[mypos[a][1],mypos[b][1]],
             color="#212121", lw=2.0, zorder=3)
# wrapped NN dashed purple
for a,b in bnn:
    if is_wrap(a,b):
        axR.plot([mypos[a][0],mypos[b][0]],[mypos[a][1],mypos[b][1]],
                 color="#7B1FA2", lw=1.4, ls="dashed", zorder=3, alpha=0.5)

for s,(x,y) in mypos.items():
    axR.plot(x,y,"o",ms=15,color="white",mec="k",mew=1.2,zorder=4)
    axR.annotate(f"{s:02d}",(x,y),ha="center",va="center",fontsize=7,zorder=5)

axR.set_aspect("equal"); axR.axis("off")

# ── legends ─────────────────────────────────────────────────────────────────
L_handles=[
    mpatches.Patch(facecolor="#FFE0B2",edgecolor="#FB8C00",label="hexagon (Ising Jz)"),
    plt.Line2D([0],[0],color="#212121",lw=2,label="NN bond (XY Jperp)"),
    plt.Line2D([0],[0],color="#7B1FA2",lw=1.6,ls="dashed",label="PBC wrap (bcy)"),
    plt.Line2D([0],[0],color="#2E7D32",lw=1.4,label="hex Ising: NN (6)"),
    plt.Line2D([0],[0],color="#1565C0",lw=1.4,ls="--",label="hex Ising: 2NN (6)"),
    plt.Line2D([0],[0],color="#C62828",lw=1.4,ls=":",label="hex Ising: 3NN (3)"),
]
axL.legend(handles=L_handles, loc="lower center", ncol=2, fontsize=8,
           bbox_to_anchor=(0.5,-0.13))

R_handles=[
    mpatches.Patch(facecolor="#FFE0B2",edgecolor="#FB8C00",label="complete hexagon"),
    plt.Line2D([0],[0],color="#212121",lw=2,label="NN bond (XY)"),
    plt.Line2D([0],[0],color="#BDBDBD",lw=1,ls=":",label="2NN/3NN inside hexagon"),
    plt.Line2D([0],[0],color="#E53935",lw=2,ls=":",label="2NN/3NN boundary (full-only)"),
    plt.Line2D([0],[0],color="#7B1FA2",lw=1.4,ls="dashed",label="PBC wrap"),
]
axR.legend(handles=R_handles, loc="lower center", ncol=2, fontsize=8,
           bbox_to_anchor=(0.5,-0.13))

fig.suptitle("BFG 3×3 kagome cylinder — collaborator vs my geometry "
             "(same lattice, 6 hexagons; differ in numbering, XY normalization, "
             "and boundary Ising bonds)", fontsize=11, y=0.99)
fig.tight_layout(rect=[0,0.02,1,0.96])
out="/scratch/zhouzb79/bfg_gs_results/collab_vs_my_cylinder.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print("Saved:", out)
print(f"Collaborator: {len(collab_nn)} NN(XY), 6 hexagons x15 = 90 Ising pairs")
print(f"My CYL: {len(bnn)} NN, {len(b2nn)} 2NN, {len(b3nn)} 3NN; "
      f"{len(my_complete)} complete hexagons")
