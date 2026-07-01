#!/usr/bin/env python3
"""BFG kagome cluster geometry: 3x3 OBC and 4x3 PBC with NN/2NN/3NN bonds.

Produces:
  bfg_geometry_bonds.pdf/png   -- side-by-side 3x3 OBC  and  4x3 PBC

Usage:
    python plot_bfg_geometry_bonds.py [output_dir]
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection, PatchCollection
from matplotlib.patches import Polygon as MPoly, FancyArrow
import matplotlib.gridspec as gridspec
import numpy as np

OUT_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/scratch/zhouzb79/bfg_gs_results")

# ---------------------------------------------------------------------------
# Lattice vectors
# ---------------------------------------------------------------------------
a1 = np.array([1.0, 0.0])
a2 = np.array([0.5, np.sqrt(3) / 2])
# Sub-lattice offsets within one unit cell
SUB_OFF = np.array([[0.0, 0.0],
                    [0.5, 0.0],
                    [0.25, np.sqrt(3) / 4]])
SUBL_COLORS = ["#e41a1c", "#377eb8", "#4daf4a"]   # red, blue, green

# Bond style per shell index (0=NN, 1=2NN, 2=3NN)
BOND_STYLE = [
    dict(color="0.20", lw=2.0,  ls="-",   alpha=0.90, label="NN  ($d=0.5$)"),
    dict(color="#e78ac3", lw=1.2,  ls="--",  alpha=0.80, label="2NN ($d=\\sqrt{3}/2$)"),
    dict(color="#a65628", lw=0.9,  ls=":",   alpha=0.70, label="3NN ($d=1.0$)"),
]

# ---------------------------------------------------------------------------
# Cluster builders
# ---------------------------------------------------------------------------
def kagome_obc(L1: int, L2: int):
    """Build OBC kagome sites for L1 x L2 unit cells."""
    sites = []
    for n2 in range(L2):
        for n1 in range(L1):
            uc = n1 * a1 + n2 * a2
            for s in range(3):
                sites.append((uc + SUB_OFF[s], s))
    pos  = np.array([p for p, _ in sites])
    subl = np.array([s for _, s in sites], dtype=int)
    return pos, subl


def kagome_pbc(L1: int, L2: int):
    """Build PBC kagome sites for L1 x L2 unit cells.
    Returns pos, subl, supercell_matrix (2x2, rows = lattice vectors).
    """
    pos, subl = kagome_obc(L1, L2)
    SC = np.array([L1 * a1, L2 * a2])   # supercell matrix (rows)
    return pos, subl, SC


def pbc_dist_matrix(pos, SC):
    """Minimum-image distance matrix for a non-orthogonal 2D supercell SC.

    Convention: SC has *rows* as lattice vectors, so a displacement D (row vec)
    expressed in fractional coords satisfies  D = f @ SC  →  f = D @ inv(SC).
    """
    SCinv = np.linalg.inv(SC)                        # (2,2)
    D     = pos[:, None, :] - pos[None, :, :]        # (N,N,2) raw displacements
    Df    = np.tensordot(D, SCinv, axes=([2], [0]))  # = D @ SCinv, shape (N,N,2)
    Df   -= np.round(Df)                             # fold to [-0.5, 0.5)
    Dmi   = np.tensordot(Df, SC, axes=([2], [0]))   # = Df @ SC, shape (N,N,2)
    return np.sqrt((Dmi**2).sum(-1)), Df, Dmi


def bond_shells(dist, tol=1e-3):
    """Return (unique_distances sorted, shell_index per pair)."""
    flat = dist[dist > tol].ravel()
    unique = np.sort(np.unique(np.round(flat, 4)))
    return unique


def get_bonds(dist, shells, n_shells=3, tol=1e-3):
    """Return list of (i, j, shell_idx) for first n_shells."""
    bonds = []
    for s_idx, sh in enumerate(shells[:n_shells]):
        pairs = list(zip(*np.where((np.abs(dist - sh) < tol) & (np.arange(len(dist))[:, None] < np.arange(len(dist))[None, :]))))
        for i, j in pairs:
            bonds.append((i, j, s_idx))
    return bonds


# ---------------------------------------------------------------------------
# Triangle detection
# ---------------------------------------------------------------------------
def find_triangles(pos, dist, nn_dist, tol=1e-3, raw_dist=None):
    """Find NN triangles.  raw_dist (optional): if provided, also require raw
    (non-PBC) distances to be NN so that wrapping triangles are excluded."""
    N = len(pos)
    tris = []
    for i in range(N):
        for j in range(i + 1, N):
            if abs(dist[i, j] - nn_dist) > tol: continue
            if raw_dist is not None and abs(raw_dist[i, j] - nn_dist) > tol: continue
            for k in range(j + 1, N):
                if abs(dist[i, k] - nn_dist) < tol and abs(dist[j, k] - nn_dist) < tol:
                    if raw_dist is not None:
                        if (abs(raw_dist[i, k] - nn_dist) > tol or
                                abs(raw_dist[j, k] - nn_dist) > tol):
                            continue
                    ys = np.sort([pos[i, 1], pos[j, 1], pos[k, 1]])
                    is_up = (ys[2] - ys[1]) > (ys[1] - ys[0])
                    tris.append((i, j, k, is_up))
    return tris


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------
def draw_triangles(ax, pos, triangles):
    up_patches, dn_patches = [], []
    for i, j, k, is_up in triangles:
        patch = MPoly(pos[[i, j, k]], closed=True)
        (up_patches if is_up else dn_patches).append(patch)
    if up_patches:
        ax.add_collection(PatchCollection(up_patches, facecolor="#ff7f00",
                                           alpha=0.16, edgecolor="none", zorder=1))
    if dn_patches:
        ax.add_collection(PatchCollection(dn_patches, facecolor="#984ea3",
                                           alpha=0.20, edgecolor="none", zorder=1))


def draw_bonds(ax, pos, bonds, ghost_offset=None):
    """Draw bonds; ghost_offset allows PBC ghost bond drawing."""
    segs_by_shell = [[], [], []]
    for i, j, s in bonds:
        segs_by_shell[s].append([pos[i], pos[j]])
        if ghost_offset is not None:
            # also draw bond mirrored if it wraps (handled separately below)
            pass
    for s, segs in enumerate(segs_by_shell):
        if not segs:
            continue
        st = BOND_STYLE[s]
        lc = LineCollection(segs, colors=st["color"], linewidths=st["lw"],
                            linestyles=st["ls"], alpha=st["alpha"], zorder=2)
        ax.add_collection(lc)


def draw_sites(ax, pos, subl, site_labels=True, ms=260):
    for s in range(3):
        mask = subl == s
        ax.scatter(pos[mask, 0], pos[mask, 1], c=SUBL_COLORS[s], s=ms,
                   edgecolors="k", linewidths=0.7, zorder=4, label=f"Sub-{s}")
    if site_labels:
        for i in range(len(pos)):
            ax.text(pos[i, 0], pos[i, 1], str(i), ha="center", va="center",
                    fontsize=5.5 if len(pos) > 27 else 6.5,
                    color="white", fontweight="bold", zorder=5)


def draw_lattice_arrows(ax, origin, scale=0.6):
    for vec, lbl, col in [(a1, r"$\mathbf{a}_1$", "darkred"),
                           (a2, r"$\mathbf{a}_2$", "darkblue")]:
        ax.annotate("", xy=origin + vec * scale, xytext=origin,
                    arrowprops=dict(arrowstyle="-|>", color=col,
                                    lw=1.8, mutation_scale=14), zorder=8)
        ax.text(*(origin + vec * 0.72), lbl, color=col, fontsize=9,
                ha="center", va="center", zorder=8)


def draw_supercell_box(ax, SC, color="gray"):
    """Draw the parallelogram supercell."""
    O = np.array([0.0, 0.0])
    corners = np.array([O, SC[0], SC[0] + SC[1], SC[1], O])
    ax.plot(corners[:, 0], corners[:, 1], color=color, lw=1.2, ls="--",
            alpha=0.6, zorder=6)


# ---------------------------------------------------------------------------
# PBC ghost-bond drawing
# ---------------------------------------------------------------------------
def draw_pbc_bonds_with_ghosts(ax, pos, SC, dist_SC, Df, Dmi,
                                shells, n_shells=3, tol=1e-3):
    """Draw bonds including PBC wrap-around ghosts.

    For wrap-around bonds (where minimum-image displacement differs from raw):
      - Draw a line from site i to the ghost position of site j (outside the cell)
      - Draw a line from ghost i (mirrored) to site j
    This makes the PBC connectivity visually obvious.
    """
    N = len(pos)
    for s_idx, sh in enumerate(shells[:n_shells]):
        st = BOND_STYLE[s_idx]
        segs_normal = []
        segs_ghost  = []

        for i in range(N):
            for j in range(i + 1, N):
                if abs(dist_SC[i, j] - sh) > tol:
                    continue
                raw_dr  = pos[j] - pos[i]
                mi_dr   = Dmi[i, j]          # minimum-image displacement
                # If they differ significantly, bond wraps around PBC
                wraps = np.linalg.norm(raw_dr - mi_dr) > tol

                if not wraps:
                    segs_normal.append([pos[i], pos[j]])
                else:
                    # Ghost positions: project i → outside toward j, and j → outside toward i
                    ghost_j = pos[i] + mi_dr      # ghost j relative to i
                    ghost_i = pos[j] - mi_dr      # ghost i relative to j
                    segs_ghost.append([pos[i], ghost_j])
                    segs_ghost.append([ghost_i, pos[j]])

        for segs, lw_factor, alpha_factor in [
            (segs_normal, 1.0, 1.0),
            (segs_ghost,  0.8, 0.55),
        ]:
            if segs:
                lc = LineCollection(segs,
                                    colors=st["color"],
                                    linewidths=st["lw"] * lw_factor,
                                    linestyles=st["ls"],
                                    alpha=st["alpha"] * alpha_factor,
                                    zorder=2)
                ax.add_collection(lc)

        # Ghost site markers at wrap endpoints
        for i in range(N):
            for j in range(i + 1, N):
                if abs(dist_SC[i, j] - sh) > tol or s_idx > 0:
                    continue   # only show ghost sites for NN
                raw_dr = pos[j] - pos[i]
                mi_dr  = Dmi[i, j]
                if np.linalg.norm(raw_dr - mi_dr) > tol:
                    ghost_j = pos[i] + mi_dr
                    ghost_i = pos[j] - mi_dr
                    # small hollow ghost markers
                    for gp, gs in [(ghost_j, subl4[j]), (ghost_i, subl4[i])]:
                        ax.scatter(*gp, c=SUBL_COLORS[gs], s=80,
                                   edgecolors="k", linewidths=0.5,
                                   alpha=0.35, zorder=3, marker="o")


# ==========================================================================
# Build clusters
# ==========================================================================
# 3x3 OBC
pos3, subl3 = kagome_obc(3, 3)
dist3       = np.sqrt(((pos3[:, None, :] - pos3[None, :, :])**2).sum(-1))
shells3     = bond_shells(dist3)
bonds3      = get_bonds(dist3, shells3, n_shells=3)
tris3       = find_triangles(pos3, dist3, shells3[0])

# 4x3 PBC
pos4, subl4, SC4      = kagome_pbc(4, 3)
dist4, Df4, Dmi4      = pbc_dist_matrix(pos4, SC4)
shells4               = bond_shells(dist4)
# raw (non-PBC) Euclidean distances — used to exclude wrap-around triangles
dist4_raw             = np.sqrt(((pos4[:, None, :] - pos4[None, :, :])**2).sum(-1))
tris4                 = find_triangles(pos4, dist4, shells4[0], raw_dist=dist4_raw)

print(f"3x3 OBC: N={len(pos3)}, shells={np.round(shells3[:5], 4)}")
for s_idx, sh in enumerate(shells3[:3]):
    n = (np.abs(dist3 - sh) < 1e-3).sum() // 2
    print(f"  shell {s_idx+1} d={sh:.4f}: {n} bonds")
print(f"  triangles: {len(tris3)} ({sum(t[3] for t in tris3)} up, {sum(not t[3] for t in tris3)} down)")

print(f"\n4x3 PBC: N={len(pos4)}, shells={np.round(shells4[:5], 4)}")
for s_idx, sh in enumerate(shells4[:3]):
    n = (np.abs(dist4 - sh) < 1e-3).sum() // 2
    print(f"  shell {s_idx+1} d={sh:.4f}: {n} bonds")
print(f"  triangles: {len(tris4)} ({sum(t[3] for t in tris4)} up, {sum(not t[3] for t in tris4)} down)")

# ==========================================================================
# Figure: side-by-side 3x3 OBC  |  4x3 PBC
# ==========================================================================
fig, axes = plt.subplots(1, 2, figsize=(16, 7.5))
fig.suptitle("BFG kagome cluster geometry  —  NN / 2NN / 3NN bonds",
             fontsize=14, y=0.99)
fig.subplots_adjust(left=0.03, right=0.97, top=0.94, bottom=0.08,
                    wspace=0.06)

# ---- LEFT: 3x3 OBC ----
ax = axes[0]
draw_triangles(ax, pos3, tris3)
draw_bonds(ax, pos3, bonds3)
draw_sites(ax, pos3, subl3)
draw_lattice_arrows(ax, origin=np.array([-0.15, -0.65]))

# Unit cell box
uc_verts = np.array([[0, 0], a1, a1 + a2, a2])
ax.add_patch(MPoly(uc_verts, closed=True, fill=False,
                   edgecolor="gray", ls="--", lw=1.0, zorder=6))
ax.text((a1 + a2)[0] / 2, (a1 + a2)[1] / 2 + 0.03,
        "unit cell", ha="center", va="bottom",
        fontsize=7, color="gray", style="italic", zorder=7)

# Bond count annotations
n3 = [(np.abs(dist3 - shells3[s]) < 1e-3).sum() // 2 for s in range(3)]
info = "\n".join([f"  shell {s+1}: {n3[s]} bonds  (d={shells3[s]:.3f})"
                  for s in range(3)])
info = f"N = {len(pos3)} sites  (OBC)\n" + info
info += f"\n  {sum(t[3] for t in tris3)} up-tri, {sum(not t[3] for t in tris3)} down-tri"
ax.text(0.01, 0.01, info, transform=ax.transAxes, fontsize=8,
        va="bottom", bbox=dict(fc="white", ec="gray", alpha=0.85, pad=4))

ax.set_xlim(pos3[:, 0].min() - 0.6,  pos3[:, 0].max() + 0.6)
ax.set_ylim(pos3[:, 1].min() - 0.85, pos3[:, 1].max() + 0.5)
ax.set_aspect("equal")
ax.axis("off")
ax.set_title(r"3$\times$3 OBC  ($N=27$)", fontsize=12, pad=4)

# ---- RIGHT: 4x3 PBC ----
ax = axes[1]
draw_triangles(ax, pos4, tris4)

# Draw bonds with PBC ghost visualization
draw_pbc_bonds_with_ghosts(ax, pos4, SC4, dist4, Df4, Dmi4,
                            shells4, n_shells=3)

draw_sites(ax, pos4, subl4, ms=200)
draw_lattice_arrows(ax, origin=np.array([-0.2, -0.80]))

# Supercell box
draw_supercell_box(ax, SC4, color="steelblue")
# Label corners
O = np.array([0.0, 0.0])
for v, lbl in [(O, ""), (SC4[0], r"$L_1\mathbf{a}_1$"),
               (SC4[1], r"$L_2\mathbf{a}_2$"),
               (SC4[0] + SC4[1], "")]:
    if lbl:
        ax.text(v[0] + 0.07, v[1] + 0.07, lbl, fontsize=8,
                color="steelblue", zorder=7)

# PBC annotation
ax.text(SC4[0][0] / 2, SC4[1][1] + 0.35,
        "↔ periodic (PBC)", ha="center", fontsize=8.5,
        color="steelblue", style="italic", zorder=7)

n4 = [(np.abs(dist4 - shells4[s]) < 1e-3).sum() // 2 for s in range(3)]
info = "\n".join([f"  shell {s+1}: {n4[s]} bonds  (d={shells4[s]:.3f})"
                  for s in range(3)])
info = f"N = {len(pos4)} sites  (PBC)\n" + info
info += f"\n  {sum(t[3] for t in tris4)} up-tri, {sum(not t[3] for t in tris4)} down-tri"
info += "\n  ghost bonds = wrap-around (faded)"
ax.text(0.01, 0.01, info, transform=ax.transAxes, fontsize=8,
        va="bottom", bbox=dict(fc="white", ec="gray", alpha=0.85, pad=4))

ax.set_xlim(pos4[:, 0].min() - 0.7,  pos4[:, 0].max() + 0.9)
ax.set_ylim(pos4[:, 1].min() - 1.0,  pos4[:, 1].max() + 0.6)
ax.set_aspect("equal")
ax.axis("off")
ax.set_title(r"4$\times$3 PBC  ($N=36$)", fontsize=12, pad=4)

# ---- Shared legend ----
sub_handles = [mpatches.Patch(fc=SUBL_COLORS[s], ec="k", label=f"Sub-{s}")
               for s in range(3)]
bond_handles = [mpatches.Patch(fc=st["color"], alpha=st["alpha"],
                               label=st["label"])
                for st in BOND_STYLE]
tri_handles  = [mpatches.Patch(fc="#ff7f00", alpha=0.5, ec="none",
                               label=r"Up $\triangle$"),
                mpatches.Patch(fc="#984ea3", alpha=0.5, ec="none",
                               label=r"Down $\triangledown$")]
ghost_handle = [mpatches.Patch(fc="0.7", alpha=0.55, ec="none",
                               label="Ghost bond (PBC wrap)")]

fig.legend(handles=sub_handles + bond_handles + tri_handles + ghost_handle,
           loc="lower center", ncol=9, fontsize=8.5,
           framealpha=0.9, bbox_to_anchor=(0.5, 0.00))

# ==========================================================================
# Save
# ==========================================================================
OUT_DIR.mkdir(parents=True, exist_ok=True)
for ext in ["pdf", "png"]:
    out = OUT_DIR / f"bfg_geometry_bonds.{ext}"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")

plt.close("all")
print("Done.")
