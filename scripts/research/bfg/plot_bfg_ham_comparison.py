"""Three-panel BFG Hamiltonian comparison for the 3x3 kagome cylinder.

Panel A — Uniform convention:  XY+Ising on ALL NN  +  Ising on ALL 2NN,3NN
Panel B — Chex convention:     XY+Ising on complete-hex NN  +  Ising on hex 2NN,3NN
Panel C — DMRG convention:     XY on ALL NN  +  Ising on complete-hex pairs only

Bond counts printed in each panel title.
"""
import sys, itertools
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import Polygon, FancyArrowPatch
import matplotlib.patches as mpatches

sys.path.insert(0, "/home/zhouzb79/links/projects/def-ybkim/zhouzb79/QED/python")
from edlib.helper_kagome_bfg import generate_kagome_cluster, create_nn_lists

# ── Build CYL cluster ────────────────────────────────────────────────────────
verts, bnn, b2nn, b3nn, nmap, v2c = generate_kagome_cluster(
    3, 3, use_pbc=False, pbc_dim1=True, pbc_dim2=False)
_, _, _, pos, _ = create_nn_lists(bnn, b2nn, b3nn, nmap, verts, v2c)

bnn_set  = set(frozenset(e) for e in bnn)
b2nn_set = set(frozenset(e) for e in b2nn)
b3nn_set = set(frozenset(e) for e in b3nn)

# ── Reference hexagons (from PBC) ─────────────────────────────────────────
_, bnn_ref, _, b3nn_ref, _, _ = generate_kagome_cluster(3, 3, use_pbc=True)
adj_ref = {i: set() for i in range(27)}
for a, b in bnn_ref:
    adj_ref[a].add(b); adj_ref[b].add(a)
all_hexes_d = {}
for a, d in b3nn_ref:
    paths = []
    for x in adj_ref[a]:
        for y in adj_ref[x]:
            if y != a and y in adj_ref[d] and y != d and x != d:
                paths.append((x, y))
    if len(paths) == 2:
        p1, p2 = paths
        all_hexes_d[frozenset([a, p1[0], p1[1], d, p2[0], p2[1]])] = True
all_hexes = list(all_hexes_d.keys())

full_bond_set = set(frozenset(e) for e in bnn + b2nn + b3nn)
def is_complete(h):
    s = list(h)
    return all(frozenset((s[i], s[j])) in full_bond_set
               for i in range(6) for j in range(i+1, 6))

complete = [h for h in all_hexes if is_complete(h)]
broken   = [h for h in all_hexes if not is_complete(h)]

# bond sets per hexagon type
chex_nn, chex_2nn, chex_3nn = set(), set(), set()
for h in complete:
    for i, j in itertools.combinations(list(h), 2):
        fs = frozenset((i, j))
        if fs in bnn_set: chex_nn.add(fs)
        elif fs in b2nn_set: chex_2nn.add(fs)
        elif fs in b3nn_set: chex_3nn.add(fs)

bdy_nn  = bnn_set  - chex_nn
bdy_2nn = b2nn_set - chex_2nn
bdy_3nn = b3nn_set - chex_3nn

# wrap detection
def is_wrap(a, b):
    pa, pb = np.array(pos[a]), np.array(pos[b])
    return np.linalg.norm(pa - pb) > 1.8

# ── Colour palette ───────────────────────────────────────────────────────────
C_XY   = "#1565C0"   # blue  — XY bond
C_IZ   = "#2E7D32"   # green — Ising bond (inside complete hex)
C_XIIZ = "#7B1FA2"   # purple — bond that carries BOTH XY and Ising
C_BDY  = "#E53935"   # red   — boundary bond (included but shouldn't be / excluded but shouldn't)
C_WRAP = "#888888"   # grey  — PBC wrap (dashed)
C_HEX  = "#FFE0B2"   # orange fill — complete hexagon
C_BHX  = "#E3F2FD"   # light blue — broken hexagon
C_BHX_EDGE = "#90CAF9"

def draw_bond(ax, a, b, color, lw=1.8, ls="-", zorder=3, alpha=1.0):
    if is_wrap(a, b):
        return  # skip wrapped bonds in geometry (too long)
    p, q = pos[a], pos[b]
    ax.plot([p[0], q[0]], [p[1], q[1]], color=color, lw=lw, ls=ls,
            zorder=zorder, alpha=alpha, solid_capstyle="round")

def draw_wrap(ax, a, b, color=C_WRAP, lw=1.4, ls="--"):
    if not is_wrap(a, b): return
    p, q = pos[a], pos[b]
    ax.plot([p[0], q[0]], [p[1], q[1]], color=color, lw=lw, ls=ls,
            zorder=2, alpha=0.5)

def draw_sites(ax):
    for s, (x, y) in pos.items():
        ax.plot(x, y, "o", ms=13, color="white", mec="#333", mew=1.2, zorder=6)
        ax.annotate(f"{s}", (x, y), ha="center", va="center",
                    fontsize=6.5, zorder=7, color="#222")

def shade_hexes(ax):
    for h in complete:
        pts = np.array([pos[s] for s in h])
        c = pts.mean(0)
        ang = np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0])
        poly = pts[np.argsort(ang)]
        ax.add_patch(Polygon(poly, closed=True, facecolor=C_HEX,
                             edgecolor="#FB8C00", lw=1.5, alpha=0.55, zorder=1))
    for h in broken:
        pts = np.array([pos[s] for s in h])
        c = pts.mean(0)
        ang = np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0])
        poly = pts[np.argsort(ang)]
        ax.add_patch(Polygon(poly, closed=True, facecolor=C_BHX,
                             edgecolor=C_BHX_EDGE, lw=1.0, alpha=0.4, zorder=1,
                             linestyle="--"))

def set_lim(ax):
    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    ax.set_xlim(min(xs)-0.4, max(xs)+0.4)
    ax.set_ylim(min(ys)-0.5, max(ys)+0.5)
    ax.set_aspect("equal")
    ax.axis("off")

# ════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(19, 7.5))

# ── Panel A: my "full" ──────────────────────────────────────────────────────
ax = axes[0]
shade_hexes(ax)

# PBC wraps
for a, b in bnn: draw_wrap(ax, a, b)

# 2NN Ising: green inside complete-hex, red on boundary
for a, b in b2nn:
    fs = frozenset((a,b))
    col = C_IZ if fs in chex_2nn else C_BDY
    draw_bond(ax, a, b, col, lw=1.4, ls=":", zorder=2, alpha=0.85)
# 3NN Ising
for a, b in b3nn:
    fs = frozenset((a,b))
    col = C_IZ if fs in chex_3nn else C_BDY
    draw_bond(ax, a, b, col, lw=1.4, ls=":", zorder=2, alpha=0.85)
# NN bonds all carry BOTH XY and Ising (purple)
for a, b in bnn:
    draw_bond(ax, a, b, C_XIIZ, lw=2.2, zorder=4)

draw_sites(ax)
set_lim(ax)
ax.set_title(
    "A)  Uniform convention\n"
    rf"NN:  $J_\pm$(XY) + $J_z$(Iz)  ×{len(bnn)}" "\n"
    rf"2NN: $J_z$(Iz)  ×{len(b2nn)}   |  3NN: $J_z$(Iz)  ×{len(b3nn)}" "\n"
    rf"Total Ising bonds: {len(bnn)+len(b2nn)+len(b3nn)}",
    fontsize=10, loc="left")

# ── Panel B: my "chex" ──────────────────────────────────────────────────────
ax = axes[1]
shade_hexes(ax)

# PBC wraps
for a, b in bnn: draw_wrap(ax, a, b)

# chex 2NN/3NN Ising (green)
for a, b in b2nn:
    if frozenset((a,b)) in chex_2nn:
        draw_bond(ax, a, b, C_IZ, lw=1.4, ls=":", zorder=2, alpha=0.85)
for a, b in b3nn:
    if frozenset((a,b)) in chex_3nn:
        draw_bond(ax, a, b, C_IZ, lw=1.4, ls=":", zorder=2, alpha=0.85)

# chex NN: XY+Ising (purple); boundary NN: grey, no XY, no Ising
for a, b in bnn:
    fs = frozenset((a,b))
    if fs in chex_nn:
        draw_bond(ax, a, b, C_XIIZ, lw=2.2, zorder=4)
    else:
        draw_bond(ax, a, b, "#BDBDBD", lw=1.4, zorder=3, alpha=0.5, ls="--")

draw_sites(ax)
set_lim(ax)
ax.set_title(
    "B)  Chex convention\n"
    rf"Hex-NN:  $J_\pm$(XY) + $J_z$(Iz)  ×{len(chex_nn)}" "\n"
    rf"Hex-2NN: $J_z$(Iz)  ×{len(chex_2nn)}   |  Hex-3NN: $J_z$(Iz)  ×{len(chex_3nn)}" "\n"
    rf"{len(bdy_nn)} boundary NN bonds XY-only (grey dashed)",
    fontsize=10, loc="left")

# ── Panel C: correct BFG ────────────────────────────────────────────────────
ax = axes[2]
shade_hexes(ax)

# PBC wraps
for a, b in bnn: draw_wrap(ax, a, b)

# Complete-hex 2NN / 3NN Ising (green dotted)
for a, b in b2nn:
    if frozenset((a,b)) in chex_2nn:
        draw_bond(ax, a, b, C_IZ, lw=1.4, ls=":", zorder=2, alpha=0.85)
for a, b in b3nn:
    if frozenset((a,b)) in chex_3nn:
        draw_bond(ax, a, b, C_IZ, lw=1.4, ls=":", zorder=2, alpha=0.85)

# All NN bonds carry XY (blue); complete-hex NN also carry Ising
for a, b in bnn:
    fs = frozenset((a,b))
    if fs in chex_nn:
        draw_bond(ax, a, b, C_XIIZ, lw=2.2, zorder=4)   # XY + Ising
    else:
        draw_bond(ax, a, b, C_XY,   lw=2.2, zorder=4)   # XY only

draw_sites(ax)
set_lim(ax)
ax.set_title(
    r"C)  DMRG convention  ($H = -J_\pm\sum_{NN}XY + \frac{J_z}{2}\sum_{hex}(S^z_{hex})^2$)" "\n"
    rf"NN: $J_\pm$(XY)  ×{len(bnn)}   (hex-NN also +$J_z$ Ising ×{len(chex_nn)})" "\n"
    rf"2NN: $J_z$(Iz)  ×{len(chex_2nn)}   |  3NN: $J_z$(Iz)  ×{len(chex_3nn)}" "\n"
    rf"Total Ising bonds: {len(chex_nn)+len(chex_2nn)+len(chex_3nn)} (complete hexagons only)",
    fontsize=10, loc="left")

# ── Shared legend ────────────────────────────────────────────────────────────
legend_handles = [
    mpatches.Patch(fc=C_HEX, ec="#FB8C00", lw=1.5, label="complete hexagon (6)"),
    mpatches.Patch(fc=C_BHX, ec=C_BHX_EDGE, lw=1, ls="--", label="broken hexagon (3)"),
    plt.Line2D([0],[0], color=C_XIIZ, lw=2.2, label=r"NN: XY + Ising"),
    plt.Line2D([0],[0], color=C_XY,   lw=2.2, label=r"NN: XY only (bdy, DMRG convention)"),
    plt.Line2D([0],[0], color=C_IZ,   lw=1.4, ls=":", label=r"2NN / 3NN Ising (inside hex)"),
    plt.Line2D([0],[0], color=C_BDY,  lw=1.4, ls=":", label=r"2NN / 3NN Ising (bdy, uniform only)"),
    plt.Line2D([0],[0], color="#BDBDBD", lw=1.4, ls="--", label=r"NN: XY-only (chex convention)"),
    plt.Line2D([0],[0], color=C_WRAP, lw=1.4, ls="--", label=r"PBC wrap (skipped in geometry)"),
]
fig.legend(handles=legend_handles, loc="lower center", ncol=4, fontsize=9,
           framealpha=0.9, bbox_to_anchor=(0.5, 0.0))

fig.suptitle(
    r"BFG kagome 3×3 cylinder — three Hamiltonian conventions  ($J_z = 1$, $N = 27$, CYL)",
    fontsize=12, y=1.01)
fig.tight_layout(rect=[0, 0.08, 1, 1.0])
out = "/scratch/zhouzb79/bfg_gs_results/bfg_hamiltonian_comparison.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print("Saved:", out)
print()
print("Bond count summary for CYL (3×3):")
print(f"  Full cluster: NN={len(bnn)}, 2NN={len(b2nn)}, 3NN={len(b3nn)}")
print(f"  Complete hexagons: {len(complete)}/9")
print(f"  Chex (in complete hex): NN={len(chex_nn)}, 2NN={len(chex_2nn)}, 3NN={len(chex_3nn)}")
print(f"  Boundary-only: NN={len(bdy_nn)}, 2NN={len(bdy_2nn)}, 3NN={len(bdy_3nn)}")
print()
print("  A) Uniform:  NN=48 (XY+Iz), 2NN=42 (Iz), 3NN=21 (Iz)")
print("  B) Chex:     NN=36 (XY+Iz), 2NN=36 (Iz), 3NN=18 (Iz), 12 bdy NN XY-only")
print("  C) DMRG:     NN=48 (XY) + 36 (also Iz), 2NN=36 (Iz), 3NN=18 (Iz)")
