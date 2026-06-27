#!/usr/bin/env python3
"""Entanglement witnesses for BFG 3×3 OBC ground state.

Computes from the two-site reduced density matrix (ρ_{ij}, derived from
stored <SzSz> and <S-S+> correlations):

  Concurrence C(i,j)         — pairwise entanglement
  Negativity  N(i,j)         — lower bound on entanglement of formation
  Single-site tangle τ_1(i)   — 4 det ρ_i = 1 - (2<Sz_i>)²
  Residual tangle τ_r(i)     = τ_1(i) - Σ_j C(i,j)² (multipartite entanglement)
  QFI per site  F_Q/N        — multipartite entanglement depth witness

QSL fingerprints to look for
-----------------------------
* Bulk C(i,j) = 0  AND  τ_1(bulk) ~ 1  ->  τ_r ~ 1
  All bulk entanglement is genuinely multipartite (>= 3-body), not bipartite.
* F_Q/N > k  ->  entanglement depth > k  (Tóth–Hyllus 2012)
* Near-1 C on OBC boundary dimers -> frustrated singlet formation

Usage:
    python plot_bfg_gs_entanglement.py [results_dir] [output_dir]
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
from matplotlib.collections import LineCollection
from matplotlib.patches import Patch
import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
RESULTS_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/scratch/zhouzb79/bfg_gs_results")
OUT_DIR     = Path(sys.argv[2]) if len(sys.argv) > 2 else RESULTS_DIR

CASES = [
    dict(Jpm=-0.20, Jzz=1.0),
    dict(Jpm=-0.05, Jzz=1.0),
    dict(Jpm=+0.05, Jzz=1.0),
    dict(Jpm=+0.20, Jzz=1.0),
]
LABELS = [r"$J_{\pm}=-0.20$", r"$J_{\pm}=-0.05$",
          r"$J_{\pm}=+0.05$", r"$J_{\pm}=+0.20$"]
COLORS = ["#d62728", "#ff7f0e", "#1f77b4", "#2ca02c"]

# ---------------------------------------------------------------------------
# Cluster geometry (same for all cases)
# ---------------------------------------------------------------------------
_pos_path = (RESULTS_DIR
             / "kagome_bfg_3x3_obc_Jpm+0.0500_Jzz1.0000"
             / "_cluster" / "positions.dat")
_pos_raw  = np.loadtxt(_pos_path)
POS  = _pos_raw[:, 3:5]              # (27, 2)  real-space x,y
SUBL = _pos_raw[:, 2].astype(int)    # sublattice index

N    = len(POS)
DR   = POS[:, None, :] - POS[None, :, :]    # (N,N,2)
DIST = np.sqrt((DR**2).sum(-1))             # (N,N)

# Coordination number and bulk/boundary classification
NN_COUNT = (DIST < 0.51).sum(axis=1) - 1   # subtract self
BULK_MASK = NN_COUNT == 4                   # fully-coordinated sites

# First few distance shells
_shells_raw = np.sort(np.unique(np.round(DIST[DIST > 1e-6], 4)))
SHELLS      = _shells_raw[:10]

# ---------------------------------------------------------------------------
# Load & compute witnesses
# ---------------------------------------------------------------------------
def load_case(c: dict) -> dict | None:
    jpm = f"{c['Jpm']:+.4f}"; jzz = f"{c['Jzz']:.4f}"
    obs = RESULTS_DIR / f"kagome_bfg_3x3_obc_Jpm{jpm}_Jzz{jzz}" / "observables.h5"
    if not obs.exists():
        print(f"  [warn] missing: {obs}"); return None
    with h5py.File(obs) as f:
        return dict(
            Jpm      = c["Jpm"],
            szsz     = f["szsz"][:].real,
            smsp     = f["smsp"][:].real,   # <S-_iS+_j>
            sz_local = f["sz_local"][:].real,
        )


def compute_witnesses(d: dict) -> dict:
    sz   = d["sz_local"]
    szsz = d["szsz"]
    spm  = d["smsp"].T            # spm[i,j] = <S+_iS-_j>

    sz_i = sz[:, None]; sz_j = sz[None, :]
    u = np.maximum(0.25 + (sz_i + sz_j) / 2 + szsz, 0.0)   # P(↑↑)
    x = np.maximum(0.25 - (sz_i + sz_j) / 2 + szsz, 0.0)   # P(↓↓)

    # Concurrence  C = 2 max(0, |z| - √(ux))
    z    = spm
    conc = np.maximum(0.0, 2 * (np.abs(z) - np.sqrt(u * x)))
    np.fill_diagonal(conc, 0.0)

    # Negativity  N = max(0, -λ_min(ρ^{T_A}))
    lam_min = (u + x) / 2 - np.sqrt(((u - x) / 2)**2 + z**2)
    neg     = np.maximum(0.0, -lam_min)
    np.fill_diagonal(neg, 0.0)

    # Single-site tangle  τ_1(i) = 4 det ρ_i = 1 - (2<Sz_i>)²
    tau1 = 1.0 - (2 * sz)**2

    # Residual tangle  τ_r(i) = τ_1(i) - Σ_j C²(i,j)
    # (monogamy: τ_r >= 0 for any qubit state; large τ_r = multipartite)
    tau_r = tau1 - (conc**2).sum(axis=1)

    # One-site von Neumann entropy  S_1(i) / ln 2  ∈ [0,1]
    p = np.clip(0.5 + sz, 1e-15, 1 - 1e-15)
    s1 = -p * np.log(p) - (1 - p) * np.log(1 - p)     # nats

    # QFI per site for the XY observable Σ_i Sˣ_i
    # F_Q / N = 2 × S^{xy}(q=0)
    # S^{xy}(q=0) = (1/N) Σ_{ij} (<S+_iS-_j> + <S-_iS+_j>)/2
    spm_sym  = (spm + spm.T) / 2
    sxy0     = spm_sym.sum() / N
    qfi_per_n = 2 * sxy0

    # Mean concurrence by distance shell (all pairs)
    dist_flat = DIST.ravel(); conc_flat = conc.ravel()
    neg_flat  = neg.ravel()
    mean_conc_shell = np.array([
        conc_flat[np.abs(dist_flat - sh) < 1e-3].mean()
        for sh in SHELLS
    ])
    mean_neg_shell = np.array([
        neg_flat[np.abs(dist_flat - sh) < 1e-3].mean()
        for sh in SHELLS
    ])

    return dict(
        conc=conc, neg=neg, s1=s1,
        tau1=tau1, tau_r=tau_r,
        qfi_per_n=qfi_per_n,
        mean_conc_shell=mean_conc_shell,
        mean_neg_shell=mean_neg_shell,
    )


records = []
for c in CASES:
    d = load_case(c)
    if d is None: continue
    d.update(compute_witnesses(d))
    records.append(d)

if not records:
    print("No data."); sys.exit(1)

Jpm_vals = np.array([d["Jpm"] for d in records])

# ============================================================
# Figure A: Spatial maps  (2 rows × 4 cols)
#   Row 0: Concurrence bonds   (OBC boundary flagged)
#   Row 1: Single-site tangle τ_1 and residual tangle τ_r
# ============================================================
CMAP_SITE = "RdYlGn"   # green = high entropy (QSL), red = low (ordered)
CMAP_BOND = "plasma"

fig_a, axs_a = plt.subplots(2, 4, figsize=(14, 6.5))
fig_a.suptitle(r"Pairwise entanglement — BFG kagome 3×3 OBC  "
               r"(sites: $\tau_1/{\rm ln2}$, bonds: $C(i,j)$)",
               fontsize=12)
fig_a.subplots_adjust(left=0.05, right=0.86, top=0.91, bottom=0.04,
                      wspace=0.18, hspace=0.25)

conc_max = max(d["conc"].max() for d in records)

for col, (d, lbl) in enumerate(zip(records, LABELS)):
    # Row 0: concurrence bonds
    ax = axs_a[0, col]
    tau1_norm = d["tau1"]   # already in [0,1]

    # Sites: coloured by τ_1 (boundary sites with hollow marker)
    ax.scatter(POS[BULK_MASK, 0],  POS[BULK_MASK, 1],
               c=tau1_norm[BULK_MASK],
               cmap=CMAP_SITE, vmin=0, vmax=1,
               s=80, zorder=4, edgecolors="k", linewidths=0.5)
    ax.scatter(POS[~BULK_MASK, 0], POS[~BULK_MASK, 1],
               c=tau1_norm[~BULK_MASK],
               cmap=CMAP_SITE, vmin=0, vmax=1,
               s=80, zorder=4, edgecolors="gray", linewidths=0.5,
               marker="^")   # triangle for boundary

    # Bonds coloured by concurrence; separate bulk and boundary bonds
    segs_bulk, vals_bulk = [], []
    segs_bnd,  vals_bnd  = [], []
    for i in range(N):
        for j in range(i + 1, N):
            v = d["conc"][i, j]
            if v < 1e-6:
                continue
            is_bnd = (not BULK_MASK[i]) or (not BULK_MASK[j])
            if is_bnd:
                segs_bnd.append([POS[i], POS[j]]); vals_bnd.append(v)
            else:
                segs_bulk.append([POS[i], POS[j]]); vals_bulk.append(v)

    norm = mcolors.Normalize(vmin=0, vmax=conc_max)
    for segs, vals, ls in [(segs_bnd, vals_bnd, "--"),
                            (segs_bulk, vals_bulk, "-")]:
        if segs:
            lc = LineCollection(segs, array=np.array(vals),
                                cmap=CMAP_BOND, norm=norm,
                                linewidths=1.5 + 4 * np.array(vals) / conc_max,
                                linestyle=ls, alpha=0.85, zorder=3)
            ax.add_collection(lc)

    # Annotate: bulk C = 0
    ax.text(0.98, 0.02, "bulk $C=0$", transform=ax.transAxes,
            fontsize=7, ha="right", va="bottom",
            color="navy", style="italic",
            bbox=dict(fc="white", ec="none", alpha=0.8))

    ax.set_aspect("equal")
    ax.set_xlim(POS[:, 0].min() - 0.4, POS[:, 0].max() + 0.4)
    ax.set_ylim(POS[:, 1].min() - 0.4, POS[:, 1].max() + 0.4)
    ax.axis("off")
    ax.set_title(lbl, fontsize=10)

    # Row 1: residual tangle bar (per site, sorted by x+y position)
    ax2 = axs_a[1, col]
    order = np.argsort(POS[:, 0] + 2 * POS[:, 1])
    tau_r = d["tau_r"][order]
    x_ax  = np.arange(N)
    c_bar = ["steelblue" if BULK_MASK[order[i]] else "salmon" for i in range(N)]
    ax2.bar(x_ax, tau_r, color=c_bar, width=0.85, zorder=3)
    ax2.axhline(0, color="k", lw=0.6)
    ax2.axhline(1, color="g", lw=0.7, ls="--", alpha=0.5)
    ax2.set_xlim(-0.5, N - 0.5)
    ax2.set_ylim(-0.05, 1.08)
    ax2.set_xticks([])
    ax2.set_ylabel(r"$\tau_r(i)$", fontsize=8)
    ax2.tick_params(labelsize=7)
    ax2.grid(True, axis="y", alpha=0.3)
    if col == 0:
        from matplotlib.patches import Patch as P
        ax2.legend(handles=[P(fc="steelblue", label="bulk (C=0->τ_r=τ_1)"),
                             P(fc="salmon",    label="boundary")],
                   fontsize=6.5, loc="lower right")

# Colorbars
sm_site = plt.cm.ScalarMappable(cmap=CMAP_SITE,
                                 norm=mcolors.Normalize(0, 1))
sm_site.set_array([])
sm_bond = plt.cm.ScalarMappable(cmap=CMAP_BOND,
                                 norm=mcolors.Normalize(0, conc_max))
sm_bond.set_array([])
for sm, y0, lbl_ in [(sm_site, 0.52, r"$\tau_1(i)$"),
                      (sm_bond, 0.06, r"$C(i,j)$")]:
    cax = fig_a.add_axes([0.875, y0, 0.016, 0.35])
    fig_a.colorbar(sm, cax=cax, label=lbl_)

# ============================================================
# Figure B: Entanglement witness summary  (2 rows × 3 cols)
# ============================================================
fig_b, axs_b = plt.subplots(2, 3, figsize=(13, 7.5))
fig_b.suptitle(r"Entanglement witnesses — BFG kagome 3×3 OBC  "
               r"(QSL fingerprints)", fontsize=13)
fig_b.subplots_adjust(left=0.08, right=0.97, top=0.92, bottom=0.09,
                      wspace=0.38, hspace=0.42)

# --- (0,0): Boundary dimer concurrence vs Jpm ---
ax = axs_b[0, 0]
# Pair (0,1): bottom-left corner dimer, distance 0.5
# Pair (6,7): another edge dimer
c01 = np.array([d["conc"][0, 1] for d in records])
c67 = np.array([d["conc"][6, 7] for d in records])
neg01 = np.array([d["neg"][0, 1] for d in records])
neg67 = np.array([d["neg"][6, 7] for d in records])

x = np.arange(len(records))
w = 0.22
ax.bar(x - w,     c01,   width=w, color=COLORS, zorder=3)
ax.bar(x,         c67,   width=w, color=COLORS, alpha=0.6, zorder=3)
ax.bar(x + w,     neg01, width=w, color=COLORS, hatch="//", alpha=0.7, zorder=3)
ax.axhline(1, color="k", ls="--", lw=0.8, label="max singlet")
ax.set_xticks(x)
ax.set_xticklabels([f"{j:+.2f}" for j in Jpm_vals], fontsize=8)
ax.set_xlabel(r"$J_{\pm}$", fontsize=10)
ax.set_ylabel("Value", fontsize=10)
ax.set_title("OBC boundary pair entanglement", fontsize=10)
handles = [Patch(fc="gray",          label=r"$C(0,1)$  (corner dimer)"),
           Patch(fc="gray", alpha=0.6, label=r"$C(6,7)$  (edge dimer)"),
           Patch(fc="gray", alpha=0.7, hatch="//", label=r"$\mathcal{N}(0,1)$")]
ax.legend(handles=handles, fontsize=7, loc="upper right")
ax.grid(True, axis="y", alpha=0.3)
ax.set_ylim(0, 1.15)

# --- (0,1): QFI per site (entanglement depth) ---
ax = axs_b[0, 1]
qfi_vals = np.array([d["qfi_per_n"] for d in records])
ax.bar(x, qfi_vals, color=COLORS, width=0.55, zorder=3)

# Hyllus/Tóth 2012: F_Q/N > k -> entanglement depth > k
for k, ls, alpha in [(1, "-",  0.9),
                      (2, "--", 0.7),
                      (3, ":",  0.6),
                      (4, "-.", 0.5)]:
    ax.axhline(k, color="k", ls=ls, lw=1.0, alpha=alpha,
               label=f"depth > {k}  (F_Q/N > {k})")

ax.set_xticks(x)
ax.set_xticklabels([f"{j:+.2f}" for j in Jpm_vals], fontsize=8)
ax.set_xlabel(r"$J_{\pm}$", fontsize=10)
ax.set_ylabel(r"$F_Q[\rho,\,\Sigma S^x] \,/\, N$", fontsize=10)
ax.set_title("Multipartite entanglement depth (QFI)", fontsize=10)
ax.legend(fontsize=7.5, loc="upper left")
ax.grid(True, axis="y", alpha=0.3)
for i, v in enumerate(qfi_vals):
    ax.text(i, v + 0.05, f"{v:.2f}", ha="center", va="bottom", fontsize=8)

# --- (0,2): Residual tangle τ_r: bulk vs boundary mean ---
ax = axs_b[0, 2]
tau_r_bulk = np.array([d["tau_r"][BULK_MASK].mean() for d in records])
tau_r_bnd  = np.array([d["tau_r"][~BULK_MASK].mean() for d in records])
tau1_bulk  = np.array([d["tau1"][BULK_MASK].mean() for d in records])

w = 0.28
ax.bar(x - w/2, tau_r_bulk, width=w, color=COLORS, zorder=3,
       label="bulk (C=0)")
ax.bar(x + w/2, tau_r_bnd,  width=w, color=COLORS, alpha=0.5, zorder=3,
       label="boundary")
ax.plot(x, tau1_bulk, "k^--", ms=6, lw=1.2, label=r"$\tau_1$ bulk (upper bound)")
ax.axhline(1, color="gray", ls=":", lw=0.8)
ax.set_xticks(x)
ax.set_xticklabels([f"{j:+.2f}" for j in Jpm_vals], fontsize=8)
ax.set_xlabel(r"$J_{\pm}$", fontsize=10)
ax.set_ylabel(r"Mean $\tau_r(i)$", fontsize=10)
ax.set_title(r"Residual tangle $\tau_r = \tau_1 - \Sigma_j C^2$"
             "\n(multipartite entanglement per site)", fontsize=9)
ax.legend(fontsize=7.5)
ax.grid(True, axis="y", alpha=0.3)

# --- (1,0): Mean concurrence by distance shell ---
ax = axs_b[1, 0]
for d, lbl, col in zip(records, LABELS, COLORS):
    vals = d["mean_conc_shell"]
    ax.plot(SHELLS, vals, "o-", ms=5, lw=1.4, color=col, label=lbl)
ax.axvline(SHELLS[0], color="gray", ls=":", lw=0.8, label="NN")
ax.axvline(SHELLS[1], color="gray", ls="--", lw=0.8, label="2NN")
ax.set_xlabel("Bond length $r$", fontsize=10)
ax.set_ylabel(r"$\langle C(i,j) \rangle_r$", fontsize=10)
ax.set_title(r"Mean concurrence vs distance", fontsize=10)
ax.legend(fontsize=7, ncol=2)
ax.grid(True, alpha=0.3)
ax.text(0.5, 0.55, "Bulk pairs:\n$C = 0$ exactly",
        transform=ax.transAxes, fontsize=8, ha="center",
        color="navy", style="italic",
        bbox=dict(fc="lightyellow", ec="goldenrod", alpha=0.85))

# --- (1,1): Single-site entropy by site class ---
ax = axs_b[1, 1]
for d, lbl, col in zip(records, LABELS, COLORS):
    # Bulk mean and std
    s1_bulk = d["s1"][BULK_MASK]
    s1_bnd  = d["s1"][~BULK_MASK]
    ax.errorbar([d["Jpm"]], [s1_bulk.mean()],
                yerr=[[s1_bulk.mean() - s1_bulk.min()],
                      [s1_bulk.max() - s1_bulk.mean()]],
                fmt="o", ms=8, color=col, capsize=4, zorder=5)
    ax.plot(d["Jpm"], s1_bnd.mean(), "^", ms=8, color=col,
            mfc="none", mew=1.5, zorder=5)

ax.axhline(np.log(2), color="k", ls="--", lw=0.9,
           label=r"max: $\ln 2$")
ax.set_xlabel(r"$J_{\pm}$", fontsize=10)
ax.set_ylabel(r"$S_1(i)$  [nats]", fontsize=10)
ax.set_title("One-site entropy (bulk ●, boundary △)", fontsize=10)
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)
ax.set_ylim(0, np.log(2) * 1.15)

# --- (1,2): QSL summary table ---
ax = axs_b[1, 2]
ax.axis("off")
rows = [["$J_{\\pm}$", "$C_{01}$", "$C_{\\rm bulk}$",
         "$F_Q/N$", "Depth", r"$\bar{\tau}_r$"]]
for d, qfi, tau_r_b in zip(records, qfi_vals, tau_r_bulk):
    depth = int(np.floor(qfi))
    rows.append([f"{d['Jpm']:+.2f}",
                 f"{d['conc'][0,1]:.3f}",
                 "0 (exact)",
                 f"{qfi:.3f}",
                 f">{depth}",
                 f"{tau_r_b:.3f}"])

tbl = ax.table(cellText=rows[1:], colLabels=rows[0],
               cellLoc="center", loc="center",
               bbox=[0, 0.1, 1, 0.85])
tbl.auto_set_font_size(False)
tbl.set_fontsize(8.5)
for (r, c), cell in tbl.get_celld().items():
    if r == 0:
        cell.set_facecolor("#d0d0d0")
        cell.set_text_props(fontweight="bold")
    elif c == 2:   # bulk C = 0 column: highlight
        cell.set_facecolor("#e8f4e8")
ax.set_title("Summary: entanglement witnesses", fontsize=10)

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
OUT_DIR.mkdir(parents=True, exist_ok=True)
for fig, name in [(fig_a, "bfg_3x3_obc_ent_maps"),
                  (fig_b, "bfg_3x3_obc_ent_witnesses")]:
    for ext in ["pdf", "png"]:
        out = OUT_DIR / f"{name}.{ext}"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"Saved: {out}")

plt.close("all")
print("Done.")
