#!/usr/bin/env python3
"""Plot BFG 3×3 OBC ground-state observables.

Produces:
  bfg_3x3_obc_analysis.pdf  — S(q) in the Brillouin zone + summary panels
  bfg_3x3_obc_analysis.png  — same, rasterised

Usage:
    python plot_bfg_gs_3x3_obc.py [results_dir] [output_dir]

Defaults:
    results_dir = /scratch/zhouzb79/bfg_gs_results
    output_dir  = results_dir
"""
from __future__ import annotations

import sys
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import matplotlib.gridspec as gridspec
import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
RESULTS_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/scratch/zhouzb79/bfg_gs_results")
OUT_DIR     = Path(sys.argv[2]) if len(sys.argv) > 2 else RESULTS_DIR

CASES = [
    dict(Jpm=+0.05, Jzz=1.0, label=r"$J_{\pm}=+0.05$",
         path=RESULTS_DIR / "kagome_bfg_3x3_obc_Jpm+0.0500_Jzz1.0000" / "observables.h5"),
    dict(Jpm=+0.20, Jzz=1.0, label=r"$J_{\pm}=+0.20$",
         path=RESULTS_DIR / "kagome_bfg_3x3_obc_Jpm+0.2000_Jzz1.0000" / "observables.h5"),
    dict(Jpm=-0.05, Jzz=1.0, label=r"$J_{\pm}=-0.05$",
         path=RESULTS_DIR / "kagome_bfg_3x3_obc_Jpm-0.0500_Jzz1.0000" / "observables.h5"),
    dict(Jpm=-0.20, Jzz=1.0, label=r"$J_{\pm}=-0.20$",
         path=RESULTS_DIR / "kagome_bfg_3x3_obc_Jpm-0.2000_Jzz1.0000" / "observables.h5"),
]

# ---------------------------------------------------------------------------
# Kagome / triangular BZ geometry
# ---------------------------------------------------------------------------
# Real-space primitive vectors (unit spacing)
a1 = np.array([1.0, 0.0])
a2 = np.array([0.5, np.sqrt(3) / 2])

# Reciprocal vectors
det = a1[0] * a2[1] - a1[1] * a2[0]   # = sqrt(3)/2
b1  = 2 * np.pi / det * np.array([ a2[1], -a2[0]])  # (2π, -2π/√3)
b2  = 2 * np.pi / det * np.array([-a1[1],  a1[0]])  # (0,   4π/√3)

# BZ high-symmetry points
GAMMA = np.array([0.0, 0.0])
K1    = (b1 + b2) / 3                 # ≈ (2π/3,  2π/√3)  ← k-index 5
K2    = (2 * b1 + b2) / 3             # ≈ (4π/3,  0)       ← k-index 7
M1    = b1 / 2                        # ≈ (π, -π/√3)
M2    = b2 / 2                        # ≈ (0,  2π/√3)

# All 6 BZ corners (vertices of the hexagonal first BZ)
def _rot60(v):
    c, s = 0.5, np.sqrt(3) / 2
    return np.array([c * v[0] - s * v[1],  s * v[0] + c * v[1]])

BZ_CORNERS = [K2]                      # start at (4π/3, 0)
for _ in range(5):
    BZ_CORNERS.append(_rot60(BZ_CORNERS[-1]))
BZ_CORNERS = np.array(BZ_CORNERS)      # shape (6, 2)

# All 6 M points (edge mid-points)
BZ_M = np.array([(BZ_CORNERS[i] + BZ_CORNERS[(i + 1) % 6]) / 2 for i in range(6)])

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
def load_case(c: dict) -> dict | None:
    p = c["path"]
    if not p.exists():
        print(f"  [warn] missing: {p}")
        return None
    cluster_dir = p.parent / "_cluster"
    with h5py.File(p, "r") as f:
        return {
            "Jpm":       c["Jpm"],
            "Jzz":       c["Jzz"],
            "label":     c["label"],
            "kpts":      f["k_points"][:],          # (9, 2)
            "ssf":       f["ssf"][:].real,           # (9,)
            "xy_bonds":  f["xy_bonds"][:],           # (n_bonds,)
            "sz_bonds":  f["szsz_bonds"][:],         # (n_bonds,)
            "chiral":    f["chiral_vals"][:],        # (n_tri,)
            "bowtie":    f["bowtie_resonances"][:].real if "bowtie_resonances" in f else np.array([]),
            "sz_local":  f["sz_local"][:],           # (N,)
            "szsz":      f["szsz"][:].real,           # (N,N)
            "smsp":      f["smsp"][:].real,           # (N,N)  <S-_i S+_j>
            "gap":       float(f.attrs["gap"]),
            "gap_per_site": float(f.attrs["gap_per_site"]),
            "gsd":       int(f.attrs["gsd"]),
            "ssf_peak":  float(f.attrs["ssf_peak"]),
            "ssf_mean":  float(f.attrs["ssf_mean"]),
            "xy_std":    float(f.attrs["xy_bond_std"]),
            "bowtie_mean": float(f.attrs["bowtie_res_mean"]),
            "chiral_mean": float(f.attrs["chiral_mean"]),
            "positions": np.loadtxt(cluster_dir / "positions.dat")[:, 3:5],  # (N, 2)
        }

data = [d for c in CASES if (d := load_case(c)) is not None]
if not data:
    print("No data found.")
    sys.exit(1)

print(f"Loaded {len(data)} cases.")

# ---------------------------------------------------------------------------
# Compute S^{zz}(q) and S^{xy}(q) = S^{xx}+S^{yy} for each case
# ---------------------------------------------------------------------------
def _ssf_components(kpts, szsz, smsp, pos):
    """Return (szz_q, sxy_q) arrays of shape (n_kpts,).

    smsp[i,j] = <S-_i S+_j>  →  <S+_i S-_j> = smsp[j,i]  (real g.s.)
    S_xy(q)  = (1/N) sum_ij exp(iq.(ri-rj)) * Re(<S+_i S-_j>)
             = (1/N) sum_ij exp(iq.(ri-rj)) * (smsp[j,i] + smsp[i,j]) / 2
    S_zz(q)  = (1/N) sum_ij exp(iq.(ri-rj)) * szsz[i,j]
    """
    N  = len(pos)
    dr = pos[:, None, :] - pos[None, :, :]        # (N, N, 2)
    spm = (smsp.T + smsp) / 2                      # symmetrised <S+_i S-_j>
    szz_q = np.empty(len(kpts))
    sxy_q = np.empty(len(kpts))
    for ki, q in enumerate(kpts):
        phase     = np.exp(1j * (dr[:, :, 0] * q[0] + dr[:, :, 1] * q[1]))
        szz_q[ki] = np.real(np.sum(phase * szsz)) / N
        sxy_q[ki] = np.real(np.sum(phase * spm))  / N
    return szz_q, sxy_q

for d in data:
    d["szz_q"], d["sxy_q"] = _ssf_components(
        d["kpts"], d["szsz"], d["smsp"], d["positions"]
    )

# ---------------------------------------------------------------------------
# Helper: draw BZ hexagon with high-symmetry labels
# ---------------------------------------------------------------------------
def draw_bz(ax, lw=1.2, color="k", alpha=0.6):
    """Draw the hexagonal first BZ boundary and label Γ, K, M."""
    poly = plt.Polygon(BZ_CORNERS, closed=True, fill=False,
                       edgecolor=color, linewidth=lw, alpha=alpha, zorder=5)
    ax.add_patch(poly)

    # Γ — crosshair at origin, label offset upward so it clears the data circle
    ax.plot(0, 0, "+", ms=9, color="k", mew=1.8, zorder=10)
    ax.annotate(r"$\Gamma$", GAMMA, xytext=GAMMA + np.array([-0.55, 0.30]),
                textcoords="data", fontsize=9, ha="center", color="k",
                fontweight="bold", zorder=12)

    # K corners — small hollow square marker, label pushed radially outward
    for i, kc in enumerate(BZ_CORNERS):
        ax.plot(*kc, "s", ms=5, color="dimgray", zorder=8,
                mfc="none", mew=1.2)
        lbl = "K" if (i % 2 == 0) else "K$'$"
        # Offset label outward from Γ by ~30 % of |K|
        outward = kc / np.linalg.norm(kc) * 0.65
        ax.annotate(lbl, kc, xytext=kc + outward,
                    textcoords="data", fontsize=8, ha="center", va="center",
                    color="dimgray", fontweight="bold", zorder=12,
                    bbox=dict(fc="white", ec="none", alpha=0.7, pad=0.5))

    # M edge-midpoints — diamond marker, label outside BZ edge
    for i, m in enumerate(BZ_M):
        ax.plot(*m, "D", ms=4, color="steelblue", zorder=8, mfc="none", mew=1.0)
        if i < 2:   # only label two to avoid clutter
            outward = m / np.linalg.norm(m) * 0.60
            ax.annotate("M", m, xytext=m + outward,
                        textcoords="data", fontsize=8, ha="center", va="center",
                        color="steelblue", fontweight="bold", zorder=12,
                        bbox=dict(fc="white", ec="none", alpha=0.7, pad=0.5))

# ---------------------------------------------------------------------------
# Figure 1: S(q) in the BZ  (2×2 panels)
# ---------------------------------------------------------------------------
CMAP = "RdYlBu_r"

fig1, axes = plt.subplots(2, 2, figsize=(10, 8.5))
fig1.subplots_adjust(left=0.07, right=0.85, top=0.93, bottom=0.07,
                     wspace=0.32, hspace=0.35)
fig1.suptitle(r"Spin structure factor $S(\mathbf{q})$ — BFG kagome 3×3 OBC  ($J_{zz}=1$)",
              fontsize=13)

# Shared colour scale across all panels
ssf_all = np.concatenate([d["ssf"] for d in data])
vmin, vmax = ssf_all.min(), ssf_all.max()

# Axis limits: encompass all k-points AND BZ corners + margin
all_kpts = np.vstack([d["kpts"] for d in data])
margin   = 0.9
xlim = (min(all_kpts[:, 0].min(), BZ_CORNERS[:, 0].min()) - margin,
        max(all_kpts[:, 0].max(), BZ_CORNERS[:, 0].max()) + margin)
ylim = (min(all_kpts[:, 1].min(), BZ_CORNERS[:, 1].min()) - margin,
        max(all_kpts[:, 1].max(), BZ_CORNERS[:, 1].max()) + margin)

sc = None
for ax, d in zip(axes.flat, data):
    kpts = d["kpts"]
    ssf  = d["ssf"]

    draw_bz(ax)

    sc = ax.scatter(kpts[:, 0], kpts[:, 1], c=ssf, cmap=CMAP,
                    vmin=vmin, vmax=vmax,
                    s=280, zorder=7, edgecolors="k", linewidths=0.5)

    # Annotate each k-point with its S(q) value, placed below the circle
    for k, s in zip(kpts, ssf):
        ax.text(k[0], k[1] - 0.38, f"{s:.3f}", ha="center", va="top",
                fontsize=6.5, color="k", zorder=9)

    ax.set_title(d["label"], fontsize=11)
    ax.set_xlabel(r"$k_x$", fontsize=9)
    ax.set_ylabel(r"$k_y$", fontsize=9)
    ax.set_aspect("equal")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)

    # Stats annotation
    txt = (f"$\\Delta E/N={d['gap_per_site']:.4f}$\n"
           f"$S_{{\\rm pk}}/\\bar{{S}}={d['ssf_peak']/d['ssf_mean']:.2f}$")
    ax.text(0.03, 0.97, txt, transform=ax.transAxes,
            fontsize=7.5, va="top", ha="left",
            bbox=dict(fc="white", ec="none", alpha=0.8, pad=1.5))

# Dedicated colorbar axes — right of all panels, clear of any subplot
cbar_ax = fig1.add_axes([0.87, 0.12, 0.025, 0.75])
cbar = fig1.colorbar(sc, cax=cbar_ax)
cbar.set_label(r"$S(\mathbf{q})$", fontsize=10)

# ---------------------------------------------------------------------------
# Figure 2: Summary order parameters vs Jpm
# ---------------------------------------------------------------------------
fig2, axs = plt.subplots(2, 3, figsize=(12, 7))
fig2.suptitle(r"BFG kagome 3×3 OBC  —  order parameters vs $J_{\pm}$  ($J_{zz}=1$)",
              fontsize=13, y=0.98)

Jpm_vals = np.array([d["Jpm"] for d in data])
order    = np.argsort(Jpm_vals)
Jpm_s    = Jpm_vals[order]
colors   = ["C0", "C1", "C2", "C3"]

def stem_plot(ax, x, y, label, color="C0", marker="o"):
    ax.axhline(0, color="gray", lw=0.8, ls="--")
    ax.stem(x, y, linefmt=f"{color}-", markerfmt=f"{color}{marker}",
            basefmt=" ", label=label)
    ax.set_xlabel(r"$J_{\pm}$", fontsize=10)
    ax.tick_params(labelsize=8)

# --- Panel 0: Excitation gap ΔE/N ---
ax = axs[0, 0]
gaps = np.array([d["gap_per_site"] for d in data])[order]
ax.bar(range(len(Jpm_s)), gaps, color=[f"C{i}" for i in range(len(Jpm_s))], width=0.5)
ax.set_xticks(range(len(Jpm_s)))
ax.set_xticklabels([f"{j:+.2f}" for j in Jpm_s], fontsize=8)
ax.set_xlabel(r"$J_{\pm}$", fontsize=10)
ax.set_ylabel(r"$\Delta E / N$", fontsize=10)
ax.set_title("Excitation gap", fontsize=10)
for i, g in enumerate(gaps):
    ax.text(i, g + 0.0001, f"{g:.5f}", ha="center", va="bottom", fontsize=7)

# --- Panel 1: S(q) peak / mean ---
ax = axs[0, 1]
ratios = np.array([d["ssf_peak"] / d["ssf_mean"] for d in data])[order]
ax.bar(range(len(Jpm_s)), ratios, color=[f"C{i}" for i in range(len(Jpm_s))], width=0.5)
ax.axhline(1.0, color="k", ls="--", lw=0.8, label="uniform")
ax.set_xticks(range(len(Jpm_s)))
ax.set_xticklabels([f"{j:+.2f}" for j in Jpm_s], fontsize=8)
ax.set_xlabel(r"$J_{\pm}$", fontsize=10)
ax.set_ylabel(r"$S_{\rm pk} / \bar{S}$", fontsize=10)
ax.set_title(r"$S(\mathbf{q})$ peak / mean", fontsize=10)
for i, r in enumerate(ratios):
    ax.text(i, r + 0.01, f"{r:.2f}", ha="center", va="bottom", fontsize=8)

# --- Panel 2: Bowtie ring resonance ---
ax = axs[0, 2]
btres = np.array([d["bowtie_mean"] for d in data])[order]
bar_colors = ["C0" if v >= 0 else "C3" for v in btres]
ax.bar(range(len(Jpm_s)), btres, color=bar_colors, width=0.5)
ax.axhline(0, color="k", ls="--", lw=0.8)
ax.set_xticks(range(len(Jpm_s)))
ax.set_xticklabels([f"{j:+.2f}" for j in Jpm_s], fontsize=8)
ax.set_xlabel(r"$J_{\pm}$", fontsize=10)
ax.set_ylabel(r"$\langle P_{\rm bowtie} \rangle$", fontsize=10)
ax.set_title("Bowtie ring resonance", fontsize=10)
for i, v in enumerate(btres):
    yoff = 0.0003 if v >= 0 else -0.0006
    ax.text(i, v + yoff, f"{v:.5f}", ha="center", va="bottom", fontsize=7)

# --- Panel 3: XY bond std (bond uniformity) ---
ax = axs[1, 0]
xy_std = np.array([d["xy_std"] for d in data])[order]
ax.bar(range(len(Jpm_s)), xy_std, color=[f"C{i}" for i in range(len(Jpm_s))], width=0.5)
ax.set_xticks(range(len(Jpm_s)))
ax.set_xticklabels([f"{j:+.2f}" for j in Jpm_s], fontsize=8)
ax.set_xlabel(r"$J_{\pm}$", fontsize=10)
ax.set_ylabel(r"$\sigma(\langle S^+_iS^-_j\rangle)$", fontsize=10)
ax.set_title("XY bond inhomogeneity (σ)", fontsize=10)
for i, v in enumerate(xy_std):
    ax.text(i, v + 0.001, f"{v:.4f}", ha="center", va="bottom", fontsize=8)

# --- Panel 4: S(q) bar chart for all k-points ---
ax = axs[1, 1]
kpt_labels = [f"k{i}" for i in range(9)]
width = 0.2
for idx, d in enumerate(sorted(data, key=lambda x: x["Jpm"])):
    offset = (idx - 1.5) * width
    bars = ax.bar(np.arange(9) + offset, d["ssf"], width=width,
                  label=d["label"], alpha=0.85)
ax.axhline(1.0, color="gray", ls=":", lw=0.8)
ax.set_xticks(np.arange(9))
ax.set_xticklabels(kpt_labels, fontsize=7)
ax.set_xlabel("k-point index", fontsize=9)
ax.set_ylabel(r"$S(\mathbf{q})$", fontsize=10)
ax.set_title(r"$S(\mathbf{q})$ at each k-point", fontsize=10)
ax.legend(fontsize=7, ncol=2, loc="upper right")

# --- Panel 5: ⟨SzSz⟩ vs distance (Jpm=+0.05 and +0.20) ---
ax = axs[1, 2]
cases_to_show = sorted(data, key=lambda x: x["Jpm"])
for d in cases_to_show:
    N = d["szsz"].shape[0]
    # Only use pairs that avoid open boundaries: compare to reference site 13 (central-ish)
    ref = N // 2
    # Compute all pairwise distances using kagome site positions
    # Use N//3 unit cells ≈ 3x3 → positions computed as ints
    szsz_row = d["szsz"][ref, :]
    # Rough distance index: |i - ref|
    dist = np.abs(np.arange(N) - ref)
    # Average over each distance bucket
    max_d = dist.max()
    avg_d, avg_c = [], []
    for dd in range(max_d + 1):
        mask = dist == dd
        if mask.sum() > 0:
            avg_d.append(dd)
            avg_c.append(szsz_row[mask].mean())
    ax.plot(avg_d, avg_c, "o-", ms=4, label=d["label"], alpha=0.85)

ax.axhline(0, color="gray", ls="--", lw=0.7)
ax.set_xlabel("Site distance |i - ref|", fontsize=9)
ax.set_ylabel(r"$\langle S^z_{\rm ref} S^z_i \rangle$", fontsize=10)
ax.set_title(r"$\langle S^z S^z \rangle$ vs distance", fontsize=10)
ax.legend(fontsize=7, loc="upper right")

fig2.tight_layout(rect=[0, 0, 1, 0.96])

# ---------------------------------------------------------------------------
# Figure 3: S(q) component breakdown  —  3 rows × 4 columns
#   Row 0: S_total(q)   Row 1: S_zz(q)   Row 2: S_xy(q)
#   Column order: Jpm = -0.20, -0.05, +0.05, +0.20
# ---------------------------------------------------------------------------
data_sorted = sorted(data, key=lambda x: x["Jpm"])
n_cases     = len(data_sorted)
COMPS = [
    ("ssf",   r"$S^{\rm tot}(\mathbf{q})$",       "viridis"),
    ("szz_q", r"$S^{zz}(\mathbf{q})$",            "Blues"),
    ("sxy_q", r"$S^{xy}(\mathbf{q}) = S^{xx}+S^{yy}$", "Reds"),
]

fig3, axes3 = plt.subplots(3, n_cases,
                            figsize=(3.2 * n_cases, 9.5),
                            squeeze=False)
fig3.suptitle(r"$S(\mathbf{q})$ breakdown: total vs $S^{zz}$ vs $S^{xy}$ — BFG kagome 3×3 OBC",
              fontsize=12)
fig3.subplots_adjust(left=0.06, right=0.88, top=0.93, bottom=0.06,
                     wspace=0.28, hspace=0.40)

# Per-row colour limits (shared across all Jpm cases)
row_lims = []
for key, _, _ in COMPS:
    vals = np.concatenate([d[key] for d in data_sorted])
    row_lims.append((vals.min(), vals.max()))

cbar_axes = []   # collect one ScalarMappable per row for colorbars

for row, ((key, comp_label, cmap), (vmin, vmax)) in enumerate(
        zip(COMPS, row_lims)):

    sm = plt.cm.ScalarMappable(cmap=cmap,
                               norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    cbar_axes.append(sm)

    for col, d in enumerate(data_sorted):
        ax = axes3[row, col]
        kpts = d["kpts"]
        vals = d[key]

        draw_bz(ax)

        sc3 = ax.scatter(kpts[:, 0], kpts[:, 1], c=vals, cmap=cmap,
                         vmin=vmin, vmax=vmax,
                         s=240, zorder=7, edgecolors="k", linewidths=0.4)

        for k, v in zip(kpts, vals):
            ax.text(k[0], k[1] - 0.38, f"{v:.3f}", ha="center", va="top",
                    fontsize=6.0, color="k", zorder=9)

        ax.set_aspect("equal")
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.plot(0, 0, "+", ms=8, color="k", mew=1.5, zorder=10)
        ax.set_xlabel(r"$k_x$", fontsize=8)
        ax.set_ylabel(r"$k_y$", fontsize=8)

        # Top row: Jpm title; left col: component label
        if row == 0:
            ax.set_title(d["label"], fontsize=10)
        if col == 0:
            ax.set_ylabel(comp_label + r"  /  $k_y$", fontsize=8)

        # Fraction of total at each k-point
        if key in ("szz_q", "sxy_q"):
            frac = vals / d["ssf"]
            frac_mean = frac.mean()
            ax.text(0.03, 0.97, f"$\\bar{{f}}={frac_mean:.2f}$",
                    transform=ax.transAxes, fontsize=7.5, va="top",
                    bbox=dict(fc="white", ec="none", alpha=0.8, pad=1.5))

    # Colorbar for this row
    cbar_x = 0.895
    cbar_y  = 0.07 + (2 - row) * 0.308
    cbar_ax3 = fig3.add_axes([cbar_x, cbar_y, 0.018, 0.25])
    fig3.colorbar(sm, cax=cbar_ax3, label=comp_label)

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
OUT_DIR.mkdir(parents=True, exist_ok=True)
for fig, name in [(fig1, "bfg_3x3_obc_ssf_bz"),
                  (fig2, "bfg_3x3_obc_summary"),
                  (fig3, "bfg_3x3_obc_ssf_breakdown")]:
    for ext in ["pdf", "png"]:
        out = OUT_DIR / f"{name}.{ext}"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"Saved: {out}")

plt.close("all")
print("Done.")
