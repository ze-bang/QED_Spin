#!/usr/bin/env python3
"""Plot the low-energy spectrum (tower of states) from BFG 3×3 OBC ED.

Produces:
  bfg_3x3_obc_spectrum.pdf / .png

Layout:
  Left  : Ground-state energy per site E₀/N vs Jpm
  Centre: Tower-of-states diagram — converged levels (E_n − E₀)/N per Jpm case
  Right : First few excitation energies (E_n − E₀)/N vs Jpm (line plot)

Usage:
    python plot_bfg_gs_spectrum.py [results_dir] [output_dir]
"""
from __future__ import annotations

import sys
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
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

# ---------------------------------------------------------------------------
# Load eigenvalues
# ---------------------------------------------------------------------------
records = []
for c in CASES:
    jpm_str = f"{c['Jpm']:+.4f}"
    jzz_str = f"{c['Jzz']:.4f}"
    path = (RESULTS_DIR
            / f"kagome_bfg_3x3_obc_Jpm{jpm_str}_Jzz{jzz_str}"
            / "eigenvalues.h5")
    if not path.exists():
        print(f"  [warn] missing: {path}")
        continue
    with h5py.File(path) as f:
        evals = np.sort(f["eigenvalues"][:])
        N     = int(f.attrs["N_sites"])
    records.append(dict(Jpm=c["Jpm"], Jzz=c["Jzz"], evals=evals, N=N))

if not records:
    print("No data found.")
    sys.exit(1)

Jpm_vals   = np.array([r["Jpm"] for r in records])
N          = records[0]["N"]    # same for all (27)
max_levels = max(len(r["evals"]) for r in records)

# Excitation energies per site
E0_arr = np.array([r["evals"][0] for r in records])
exc    = []  # list of arrays (E_n - E_0)/N, padded with NaN
for r in records:
    delta = (r["evals"] - r["evals"][0]) / r["N"]
    padded = np.full(max_levels, np.nan)
    padded[:len(delta)] = delta
    exc.append(padded)
exc = np.array(exc)   # shape (n_cases, max_levels)

# ---------------------------------------------------------------------------
# Colours (consistent with BZ plot)
# ---------------------------------------------------------------------------
CASE_COLORS = ["#d62728", "#ff7f0e", "#1f77b4", "#2ca02c"]   # red→orange→blue→green (Jpm order)
LEVEL_COLORS = ["k", "#e31a1c", "#ff7f00", "#33a02c", "#1f78b4", "#6a3d9a"]

# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
fig = plt.figure(figsize=(13, 5.5))
fig.suptitle(r"Low-energy spectrum — BFG kagome 3×3 OBC  ($N=27$, $S^z_{\rm tot}=-1/2$)",
             fontsize=13)

gs = fig.add_gridspec(1, 3, left=0.07, right=0.97, top=0.88, bottom=0.12,
                      wspace=0.38, width_ratios=[1, 1.4, 1])

ax_e0  = fig.add_subplot(gs[0])   # E_0/N vs Jpm
ax_tow = fig.add_subplot(gs[1])   # tower of states
ax_gap = fig.add_subplot(gs[2])   # excitation energies vs Jpm

# -------------------------------------------------------------------
# Panel A: Ground-state energy per site
# -------------------------------------------------------------------
ax_e0.plot(Jpm_vals, E0_arr / N, "o-", color="k", ms=7, lw=1.5)
for i, (j, e) in enumerate(zip(Jpm_vals, E0_arr / N)):
    ax_e0.plot(j, e, "o", ms=8, color=CASE_COLORS[i], zorder=5)
ax_e0.set_xlabel(r"$J_{\pm}$", fontsize=11)
ax_e0.set_ylabel(r"$E_0 / N$", fontsize=11)
ax_e0.set_title("Ground-state energy / site", fontsize=10)
ax_e0.xaxis.set_major_formatter(ticker.FormatStrFormatter("%+.2f"))
ax_e0.grid(True, alpha=0.3)

# -------------------------------------------------------------------
# Panel B: Tower of states
# -------------------------------------------------------------------
x_ticks = []
x_labels = []
GAP_TOL = 1e-4   # levels closer than this (per site) are treated as degenerate

for i, r in enumerate(records):
    jpm    = r["Jpm"]
    evals  = r["evals"]
    e0     = evals[0]
    Nsite  = r["N"]
    x_cent = i                   # column x-position for this case

    # Draw horizontal bars for each level
    bar_half = 0.28
    for n, e in enumerate(evals):
        delta = (e - e0) / Nsite
        color = LEVEL_COLORS[n % len(LEVEL_COLORS)]
        lw    = 2.5 if n == 0 else 1.8
        ls    = "-"
        ax_tow.hlines(delta, x_cent - bar_half, x_cent + bar_half,
                      colors=color, linewidths=lw, linestyles=ls, zorder=5)
        # Label value to right
        ax_tow.text(x_cent + bar_half + 0.04, delta,
                    f"{delta:.4f}", va="center", ha="left",
                    fontsize=7, color=color, zorder=6)
        # Level index to left
        ax_tow.text(x_cent - bar_half - 0.04, delta,
                    f"$n={n}$", va="center", ha="right",
                    fontsize=7, color=color, zorder=6)

    # Draw bracket for gap Δ₁
    if len(evals) >= 2:
        delta1 = (evals[1] - e0) / Nsite
        ax_tow.annotate("", xy=(x_cent + bar_half + 0.22, delta1),
                        xytext=(x_cent + bar_half + 0.22, 0.0),
                        arrowprops=dict(arrowstyle="<->", color="gray", lw=1.0))
        ax_tow.text(x_cent + bar_half + 0.28, delta1 / 2,
                    r"$\Delta_1$", va="center", ha="left",
                    fontsize=7.5, color="gray")

    x_ticks.append(x_cent)
    x_labels.append(f"$J_{{\\pm}}={jpm:+.2f}$")

ax_tow.set_xticks(x_ticks)
ax_tow.set_xticklabels(x_labels, fontsize=9)
ax_tow.set_xlim(-0.7, len(records) - 0.3)
# y-axis: from a bit below 0 to max gap + margin
y_max = np.nanmax(exc[:, 1:]) * 1.25 if max_levels > 1 else 0.05
ax_tow.set_ylim(-0.002, max(y_max, 0.02))
ax_tow.set_ylabel(r"$(E_n - E_0)\,/\,N$", fontsize=11)
ax_tow.set_title("Tower of states", fontsize=10)
ax_tow.axhline(0, color="k", lw=0.7, ls="--", alpha=0.4)
ax_tow.grid(True, axis="y", alpha=0.2)

# -------------------------------------------------------------------
# Panel C: Excitation energies vs Jpm
# -------------------------------------------------------------------
for n in range(1, max_levels):
    vals = exc[:, n]
    mask = ~np.isnan(vals)
    if mask.sum() < 1:
        continue
    ax_gap.plot(Jpm_vals[mask], vals[mask],
                "o-", ms=6, lw=1.4,
                color=LEVEL_COLORS[n % len(LEVEL_COLORS)],
                label=f"$n={n}$", zorder=5)
    # Mark missing points
    for i, (j, v) in enumerate(zip(Jpm_vals, vals)):
        if not np.isnan(v):
            ax_gap.plot(j, v, "o", ms=8, color=CASE_COLORS[i], zorder=6,
                        mec=LEVEL_COLORS[n % len(LEVEL_COLORS)], mew=1.2)

ax_gap.set_xlabel(r"$J_{\pm}$", fontsize=11)
ax_gap.set_ylabel(r"$(E_n - E_0)\,/\,N$", fontsize=11)
ax_gap.set_title("Excitation energies", fontsize=10)
ax_gap.xaxis.set_major_formatter(ticker.FormatStrFormatter("%+.2f"))
ax_gap.legend(fontsize=9, loc="upper left")
ax_gap.grid(True, alpha=0.3)
ax_gap.set_ylim(bottom=0)

# Annotate gap values
for i, r in enumerate(records):
    if len(r["evals"]) >= 2:
        delta1 = (r["evals"][1] - r["evals"][0]) / r["N"]
        ax_gap.annotate(f"{delta1:.4f}",
                        (r["Jpm"], delta1),
                        xytext=(0, 6), textcoords="offset points",
                        fontsize=7, ha="center", color=LEVEL_COLORS[1])

# -------------------------------------------------------------------
# Save
# -------------------------------------------------------------------
OUT_DIR.mkdir(parents=True, exist_ok=True)
for ext in ["pdf", "png"]:
    out = OUT_DIR / f"bfg_3x3_obc_spectrum.{ext}"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")

plt.close("all")
print("Done.")
