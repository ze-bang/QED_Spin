#!/usr/bin/env python3
"""BFG kagome overview: geometry, E0/N vs Jpm, and component SSF in the BZ.

Produces:
  bfg_geometry.pdf/png             -- kagome cluster with sublattices & triangles
  bfg_energy_ssf_overview.pdf/png  -- E0/N vs Jpm  +  S+S-(q)  +  SzSz(q)

Usage:
    python plot_bfg_gs_overview.py [results_dir] [output_dir]
"""
from __future__ import annotations

import sys
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.colors as mcolors
from matplotlib.patches import FancyArrowPatch, Polygon as MPoly
from matplotlib.collections import LineCollection, PatchCollection
import matplotlib.gridspec as gridspec
import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
RESULTS_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/scratch/zhouzb79/bfg_gs_results")
OUT_DIR     = Path(sys.argv[2]) if len(sys.argv) > 2 else RESULTS_DIR

CASES = [
    dict(Jpm=-0.20, Jzz=1.0, label=r"$J_{\pm}=-0.20$"),
    dict(Jpm=-0.05, Jzz=1.0, label=r"$J_{\pm}=-0.05$"),
    dict(Jpm=+0.05, Jzz=1.0, label=r"$J_{\pm}=+0.05$"),
    dict(Jpm=+0.20, Jzz=1.0, label=r"$J_{\pm}=+0.20$"),
]
COLORS = ["#d62728", "#ff7f0e", "#1f77b4", "#2ca02c"]

# ---------------------------------------------------------------------------
# Cluster geometry
# ---------------------------------------------------------------------------
_pos_dat  = (RESULTS_DIR / "kagome_bfg_3x3_obc_Jpm+0.0500_Jzz1.0000"
             / "_cluster" / "positions.dat")
_raw      = np.loadtxt(_pos_dat)
POS       = _raw[:, 3:5]
SUBL      = _raw[:, 2].astype(int)
N         = len(POS)
DIST      = np.sqrt(((POS[:, None, :] - POS[None, :, :])**2).sum(-1))

# NN bond list
NN_BONDS = [(i, j) for i in range(N) for j in range(i + 1, N) if DIST[i, j] < 0.51]

# Triangle list with up/down classification
# UP  = isolated apex is the highest site
# DOWN = isolated apex is the lowest site
TRIANGLES = []
for i in range(N):
    for j in range(i + 1, N):
        if DIST[i, j] >= 0.51: continue
        for k in range(j + 1, N):
            if DIST[i, k] >= 0.51 or DIST[j, k] >= 0.51: continue
            ys = np.sort([POS[i, 1], POS[j, 1], POS[k, 1]])
            is_up = (ys[2] - ys[1]) > (ys[1] - ys[0])
            TRIANGLES.append((i, j, k, is_up))

# BZ geometry
a1 = np.array([1.0, 0.0]);  a2 = np.array([0.5, np.sqrt(3) / 2])
det = a1[0] * a2[1] - a1[1] * a2[0]
b1  = 2 * np.pi / det * np.array([ a2[1], -a2[0]])
b2  = 2 * np.pi / det * np.array([-a1[1],  a1[0]])
K2  = (2 * b1 + b2) / 3
def _rot60(v):
    c, s = 0.5, np.sqrt(3) / 2
    return np.array([c * v[0] - s * v[1], s * v[0] + c * v[1]])
BZ_CORNERS = np.array([_rot60(K2) if i == 0 else
                        _rot60(K2) for i in range(6)])
BZ_CORNERS = np.array([K2] + [_rot60(K2)] * 5)
for i in range(1, 6):
    BZ_CORNERS[i] = _rot60(BZ_CORNERS[i - 1])

SUBL_COLORS = ["#e41a1c", "#377eb8", "#4daf4a"]  # sub-0,1,2

# ---------------------------------------------------------------------------
# SSF component computation (from stored correlations)
# ---------------------------------------------------------------------------
def ssf_components(kpts, szsz, smsp, pos):
    N_ = len(pos)
    dr = pos[:, None, :] - pos[None, :, :]
    spm = (smsp.T + smsp) / 2    # symmetrised <S+_i S-_j>
    szz = np.empty(len(kpts)); sxy = np.empty(len(kpts))
    for ki, q in enumerate(kpts):
        ph = np.exp(1j * (dr[:, :, 0] * q[0] + dr[:, :, 1] * q[1]))
        szz[ki] = np.real(np.sum(ph * szsz)) / N_
        sxy[ki] = np.real(np.sum(ph * spm))  / N_
    return szz, sxy


# Load data
records = []
for c in CASES:
    jpm = f"{c['Jpm']:+.4f}"; jzz = f"{c['Jzz']:.4f}"
    base = RESULTS_DIR / f"kagome_bfg_3x3_obc_Jpm{jpm}_Jzz{jzz}"
    obs  = base / "observables.h5"
    evf  = base / "eigenvalues.h5"
    if not obs.exists(): continue
    with h5py.File(obs) as f:
        kpts = f["k_points"][:]
        szsz = f["szsz"][:].real
        smsp = f["smsp"][:].real
    with h5py.File(evf) as f:
        evals = np.sort(f["eigenvalues"][:])
        Ns    = int(f.attrs["N_sites"])
    szz_q, sxy_q = ssf_components(kpts, szsz, smsp, POS)
    records.append(dict(
        Jpm=c["Jpm"], label=c["label"],
        kpts=kpts, szz_q=szz_q, sxy_q=sxy_q,
        E0_per_site=evals[0] / Ns,
        evals=evals, N_sites=Ns,
    ))

# BZ axis extents
all_k  = np.vstack([d["kpts"] for d in records])
margin = 0.9
xlim   = (min(all_k[:, 0].min(), BZ_CORNERS[:, 0].min()) - margin,
          max(all_k[:, 0].max(), BZ_CORNERS[:, 0].max()) + margin)
ylim   = (min(all_k[:, 1].min(), BZ_CORNERS[:, 1].min()) - margin,
          max(all_k[:, 1].max(), BZ_CORNERS[:, 1].max()) + margin)

def draw_bz(ax):
    poly = plt.Polygon(BZ_CORNERS, closed=True, fill=False,
                       edgecolor="k", linewidth=1.2, alpha=0.7, zorder=5)
    ax.add_patch(poly)
    ax.plot(0, 0, "+", ms=8, color="k", mew=1.8, zorder=10)
    ax.annotate(r"$\Gamma$", [0, 0], xytext=[-0.55, 0.30],
                textcoords="data", fontsize=8, ha="center",
                fontweight="bold", zorder=12)
    for i, kc in enumerate(BZ_CORNERS):
        ax.plot(*kc, "s", ms=4, color="dimgray", zorder=8, mfc="none", mew=1.0)
        if i < 2:
            out = kc / np.linalg.norm(kc) * 0.65
            lbl = "K" if i == 0 else "K$'$"
            ax.annotate(lbl, kc, xytext=kc + out, textcoords="data",
                        fontsize=7.5, ha="center", va="center",
                        fontweight="bold", color="dimgray", zorder=12,
                        bbox=dict(fc="white", ec="none", alpha=0.7, pad=0.4))

# ==========================================================================
# Figure A: Kagome cluster geometry
# ==========================================================================
fig_a = plt.figure(figsize=(13, 6.5))
fig_a.suptitle(r"BFG kagome 3$\times$3 OBC cluster  —  $N=27$ sites,  43 NN bonds,  "
               r"9 up-$\triangledown$ + 4 down-$\triangle$ triangles",
               fontsize=13, y=0.98)

gs_a = gridspec.GridSpec(1, 2, figure=fig_a, width_ratios=[2.3, 1],
                         left=0.04, right=0.98, top=0.93, bottom=0.05,
                         wspace=0.06)
ax_geom = fig_a.add_subplot(gs_a[0])
ax_info = fig_a.add_subplot(gs_a[1])

# -- Triangle patches
tri_patches_up, tri_patches_dn = [], []
for i, j, k, is_up in TRIANGLES:
    verts = POS[[i, j, k]]
    patch = MPoly(verts, closed=True)
    (tri_patches_up if is_up else tri_patches_dn).append(patch)

if tri_patches_up:
    ax_geom.add_collection(PatchCollection(tri_patches_up, facecolor="#ff7f00",
                                            alpha=0.18, edgecolor="none", zorder=1))
if tri_patches_dn:
    ax_geom.add_collection(PatchCollection(tri_patches_dn, facecolor="#984ea3",
                                            alpha=0.22, edgecolor="none", zorder=1))

# -- NN bonds
segs = [[POS[i], POS[j]] for i, j in NN_BONDS]
lc = LineCollection(segs, colors="0.35", linewidths=1.8, zorder=2, alpha=0.8)
ax_geom.add_collection(lc)

# -- Sites
for s in [0, 1, 2]:
    mask = SUBL == s
    ax_geom.scatter(POS[mask, 0], POS[mask, 1],
                    c=SUBL_COLORS[s], s=320, zorder=4,
                    edgecolors="k", linewidths=0.8,
                    label=f"Sub-lattice {s}")

# -- Site labels
for i in range(N):
    ax_geom.text(POS[i, 0], POS[i, 1], str(i),
                 ha="center", va="center", fontsize=6.5,
                 color="white", fontweight="bold", zorder=5)

# -- Lattice vectors
origin = np.array([-0.1, -0.55])
for vec, lbl, col in [(a1, r"$\mathbf{a}_1$", "darkred"),
                       (a2, r"$\mathbf{a}_2$", "darkblue")]:
    ax_geom.annotate("", xy=origin + vec * 0.7, xytext=origin,
                     arrowprops=dict(arrowstyle="-|>", color=col,
                                     lw=1.8, mutation_scale=14),
                     zorder=8)
    ax_geom.text(*(origin + vec * 0.82), lbl, color=col,
                 fontsize=9, ha="center", va="center", zorder=8)

# -- Unit cell outline (dashed parallelogram at origin)
uc_verts = np.array([np.array([0, 0]), a1, a1 + a2, a2]) - np.array([0.0, 0.0])
uc_patch = MPoly(uc_verts, closed=True, fill=False,
                 edgecolor="gray", linestyle="--", linewidth=1.0, zorder=6)
ax_geom.add_patch(uc_patch)
ax_geom.text((a1 + a2)[0] / 2, (a1 + a2)[1] / 2 + 0.04,
             "unit cell", ha="center", va="bottom",
             fontsize=7.5, color="gray", style="italic", zorder=7)

ax_geom.set_xlim(POS[:, 0].min() - 0.55, POS[:, 0].max() + 0.55)
ax_geom.set_ylim(POS[:, 1].min() - 0.7,  POS[:, 1].max() + 0.45)
ax_geom.set_aspect("equal")
ax_geom.axis("off")

# -- Legend
from matplotlib.patches import Patch as Pt, FancyArrow
handles = [
    Pt(fc=SUBL_COLORS[0], ec="k", label="Sub-0"),
    Pt(fc=SUBL_COLORS[1], ec="k", label="Sub-1"),
    Pt(fc=SUBL_COLORS[2], ec="k", label="Sub-2"),
    Pt(fc="#ff7f00", alpha=0.5, ec="none", label=r"Up $\triangle$  (9)"),
    Pt(fc="#984ea3", alpha=0.5, ec="none", label=r"Down $\triangledown$  (4)"),
]
ax_geom.legend(handles=handles, loc="upper left", fontsize=8.5,
               framealpha=0.9, ncol=2)

# -- BZ inset
ax_bz = ax_geom.inset_axes([0.74, 0.60, 0.26, 0.38])
bz_poly = plt.Polygon(BZ_CORNERS, closed=True, fill=True,
                      facecolor="#e0f0ff", edgecolor="k", linewidth=1.2, zorder=2)
ax_bz.add_patch(bz_poly)
# All 9 k-points from first case
kpts = records[0]["kpts"]
ax_bz.scatter(kpts[:, 0], kpts[:, 1], s=28, c="k", zorder=5)
for kc in BZ_CORNERS:
    ax_bz.plot(*kc, "s", ms=4, color="steelblue", zorder=6, mfc="none", mew=1.0)
ax_bz.plot(0, 0, "+", ms=6, color="k", mew=1.5, zorder=7)
ax_bz.set_aspect("equal")
ax_bz.set_xlim(BZ_CORNERS[:, 0].min() - 0.5, BZ_CORNERS[:, 0].max() + 0.5)
ax_bz.set_ylim(BZ_CORNERS[:, 1].min() - 0.5, BZ_CORNERS[:, 1].max() + 0.5)
ax_bz.set_title("BZ  +  $k$-pts", fontsize=7, pad=2)
ax_bz.set_xticks([]); ax_bz.set_yticks([])
for sp in ax_bz.spines.values(): sp.set_linewidth(0.5)

# -- Right panel: Hamiltonian info
ax_info.axis("off")
hamtext = (
    r"BFG kagome Hamiltonian" + "\n\n"
    r"$H = \sum_{\langle ij\rangle}$"  + "\n"  +
    r"$\quad[-J_{\pm}(S^+_iS^-_j+S^-_iS^+_j) + J_{zz} S^z_i S^z_j]$"
    "\n\n"
    r"$S = 1/2$ spins on kagome NN bonds"
    "\n\n"
    r"$J_{\pm} > 0$: ferromagnetic XY  (flips favour FM)"
    "\n"
    r"$J_{\pm} < 0$: antiferromagnetic XY"
    "\n"
    r"$J_{zz} = 1.0$: Ising repulsion"
    "\n\n"
    r"Sector: $S^z_{\rm tot} = -1/2$  ($n_\uparrow=13$)"
    "\n"
    r"Hilbert space: $\binom{27}{13} \approx 2\times10^7$"
    "\n\n"
    r"Cluster: 3$\times$3 OBC  |  $N=27$ sites" + "\n"
    r"Solver: Krylov–Schur (CPU)"
    "\n\n"
    r"Studied parameter points:" + "\n"
    r"$J_{\pm} \in \{-0.20,\ -0.05,\ +0.05,\ +0.20\}$"
)
ax_info.text(0.05, 0.95, hamtext, transform=ax_info.transAxes,
             fontsize=9, va="top", ha="left",
             linespacing=1.55,
             bbox=dict(fc="#f7f7f7", ec="gray", boxstyle="round,pad=0.6"))

# ==========================================================================
# Figure B: E0/N vs Jpm  +  S+S-(q)  +  SzSz(q)
# ==========================================================================
fig_b = plt.figure(figsize=(13, 13.5))
fig_b.suptitle(
    r"BFG kagome 3$\times$3 OBC — $E_0/N$, low-energy spectrum, and $S(\mathbf{q})$",
    fontsize=13, y=0.99)

gs_b = gridspec.GridSpec(4, 4, figure=fig_b,
                          left=0.07, right=0.90, top=0.95, bottom=0.05,
                          wspace=0.30, hspace=0.50,
                          height_ratios=[0.8, 1.0, 1.4, 1.4])

# -- Row 0: E0/N vs Jpm (span all 4 columns)
ax_e0 = fig_b.add_subplot(gs_b[0, :])
Jpm_arr = np.array([d["Jpm"] for d in records])
E0_arr  = np.array([d["E0_per_site"] for d in records])

ax_e0.plot(Jpm_arr, E0_arr, "k--", lw=1.2, zorder=2)
_e0_mean = E0_arr.mean()
for i, (j, e) in enumerate(zip(Jpm_arr, E0_arr)):
    ax_e0.plot(j, e, "o", ms=10, color=COLORS[i], zorder=5,
               mec="k", mew=0.8)
    # Labels above the marker for low-energy points, below for high-energy
    # points (near ±0.05) to avoid colliding with the panel title.
    above_mean = e > _e0_mean
    yoff = -0.012 if above_mean else +0.012
    va   = "top"  if above_mean else "bottom"
    ax_e0.text(j, e + yoff, f"{e:.4f}", ha="center", va=va,
               fontsize=8, color=COLORS[i])

ax_e0.set_xlabel(r"$J_{\pm}$", fontsize=11)
ax_e0.set_ylabel(r"$E_0 / N$", fontsize=11)
ax_e0.set_title(r"Ground-state energy per site", fontsize=11)
ax_e0.xaxis.set_major_formatter(ticker.FormatStrFormatter("%+.2f"))
ax_e0.grid(True, alpha=0.3)
ax_e0.set_xlim(Jpm_arr.min() - 0.05, Jpm_arr.max() + 0.05)

# -- Row 1: Low-energy spectrum / tower of states (span all 4 cols)
LEVEL_COLORS_TOW = ["k", "#e31a1c", "#ff7f00", "#33a02c", "#1f78b4", "#6a3d9a"]
ax_tow = fig_b.add_subplot(gs_b[1, :])
for i, r in enumerate(records):
    evals_i = r["evals"]
    e0_i    = evals_i[0]
    ns_i    = r["N_sites"]
    hw = 0.28
    for n, e in enumerate(evals_i):
        delta = (e - e0_i) / ns_i
        col   = LEVEL_COLORS_TOW[n % len(LEVEL_COLORS_TOW)]
        ax_tow.hlines(delta, i - hw, i + hw,
                      colors=col, linewidths=2.5 if n == 0 else 1.8, zorder=5)
        ax_tow.text(i + hw + 0.03, delta, f"{delta:.4f}",
                    va="center", ha="left", fontsize=7.5, color=col)
    if len(evals_i) >= 2:
        d1 = (evals_i[1] - e0_i) / ns_i
        ax_tow.annotate("", xy=(i + hw + 0.30, d1),
                        xytext=(i + hw + 0.30, 0.0),
                        arrowprops=dict(arrowstyle="<->", color="gray", lw=1.0))
        ax_tow.text(i + hw + 0.36, d1 / 2, r"$\Delta_1$",
                    va="center", ha="left", fontsize=7.5, color="gray")
ax_tow.set_xticks(range(len(records)))
ax_tow.set_xticklabels([r["label"] for r in records], fontsize=9)
ax_tow.set_xlim(-0.6, len(records) - 0.10)
y_top_tow = max((r["evals"][-1] - r["evals"][0]) / r["N_sites"] for r in records)
ax_tow.set_ylim(-0.003, y_top_tow * 1.40)
ax_tow.set_ylabel(r"$(E_n - E_0)\,/\,N$", fontsize=10)
ax_tow.set_title("Low-energy spectrum (tower of states)", fontsize=10)
ax_tow.axhline(0, color="k", lw=0.7, ls="--", alpha=0.4)
ax_tow.grid(True, axis="y", alpha=0.2)

# Per-row colour scales
sxy_all = np.concatenate([d["sxy_q"] for d in records])
szz_all = np.concatenate([d["szz_q"] for d in records])
row_cfg = [
    (2, "sxy_q", "Reds",  r"$S^{xy}(\mathbf{q}) = S^{xx}+S^{yy}$  (transverse / XY)"),
    (3, "szz_q", "Blues", r"$S^{zz}(\mathbf{q})$  (longitudinal / Ising)"),
]

cbar_sms = {}
for row, key, cmap, row_title in row_cfg:
    vals_all = np.concatenate([d[key] for d in records])
    vmin, vmax = vals_all.min(), vals_all.max()

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=mcolors.Normalize(vmin, vmax))
    sm.set_array([])
    cbar_sms[key] = (sm, vmax)

    for col, d in enumerate(records):
        ax = fig_b.add_subplot(gs_b[row, col])
        kpts = d["kpts"]
        vals = d[key]

        draw_bz(ax)

        sc = ax.scatter(kpts[:, 0], kpts[:, 1], c=vals, cmap=cmap,
                        vmin=vmin, vmax=vmax,
                        s=230, zorder=7, edgecolors="k", linewidths=0.4)

        for k, v in zip(kpts, vals):
            ax.text(k[0], k[1] - 0.38, f"{v:.3f}", ha="center", va="top",
                    fontsize=6.0, color="k", zorder=9)

        ax.set_aspect("equal")
        ax.set_xlim(*xlim); ax.set_ylim(*ylim)
        ax.set_xlabel(r"$k_x$", fontsize=8)
        ax.set_ylabel(r"$k_y$", fontsize=8)

        if row == 2:
            ax.set_title(d["label"], fontsize=10)
        if col == 0:
            ax.set_ylabel(row_title + r"  $k_y$", fontsize=7.5)

        # Mean fraction
        frac = vals.mean() / (d["sxy_q"] + d["szz_q"]).mean()
        ax.text(0.03, 0.97, f"$\\bar{{f}}={frac:.2f}$",
                transform=ax.transAxes, fontsize=7.5, va="top",
                bbox=dict(fc="white", ec="none", alpha=0.8, pad=1.5))

# Colorbars — placed explicitly to the right of right=0.90 boundary
# Row positions with height_ratios [0.8,1.0,1.4,1.4], hspace=0.50, top=0.95, bot=0.05:
#   row 2 (S^xy): y_bot≈0.331, height≈0.199  →  colorbar centred ≈0.431
#   row 3 (S^zz): y_bot≈0.050, height≈0.199  →  colorbar centred ≈0.149
_cbar_params = [
    ("sxy_q", 0.345, 0.17),   # (key, y0, height)
    ("szz_q", 0.063, 0.17),
]
for key, y0, ht in _cbar_params:
    sm, _ = cbar_sms[key]
    cax = fig_b.add_axes([0.905, y0, 0.018, ht])
    fig_b.colorbar(sm, cax=cax)

# ==========================================================================
# Save
# ==========================================================================
OUT_DIR.mkdir(parents=True, exist_ok=True)
for fig, name in [(fig_a, "bfg_geometry"),
                  (fig_b, "bfg_energy_ssf_overview")]:
    for ext in ["pdf", "png"]:
        out = OUT_DIR / f"{name}.{ext}"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"Saved: {out}")

plt.close("all")
print("Done.")
