#!/usr/bin/env python3
"""Plot E, C, S vs T for all thermal methods."""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

data = json.load(open("method_compare.json"))
T_ex = np.array(data["T_exact"])
E_ex = np.array(data["E_exact"])
C_ex = np.array(data["C_exact"])
S_ex = np.array(data["S_exact"])
N    = data["N"]

# ── style ─────────────────────────────────────────────────────────────────────
STYLES = {
    "mTPQ (5 samp)":  dict(color="#e74c3c", ls="--",  lw=1.4, alpha=0.8),
    "mTPQ (20 samp)": dict(color="#c0392b", ls="-",   lw=1.6, alpha=0.85),
    "mTPQ (80 samp)": dict(color="#922b21", ls="-",   lw=2.0, alpha=0.9),
    "cTPQ (20 samp)": dict(color="#e67e22", ls="-",   lw=1.6, alpha=0.85),
    "FTLM (m=20)":    dict(color="#27ae60", ls=":",   lw=1.4, alpha=0.8),
    "FTLM (m=80)":    dict(color="#1e8449", ls="--",  lw=1.6, alpha=0.85),
    "FTLM (m=200)":   dict(color="#145a32", ls="-",   lw=2.0, alpha=0.9),
    "LTLM (m=80)":    dict(color="#8e44ad", ls="-.",  lw=1.6, alpha=0.85),
    "KPM_DOS":        dict(color="#2980b9", ls="-",   lw=1.6, alpha=0.85),
}

methods = data["methods"]

# ════════════════════════════════════════════════════════════════════════════
# Figure 1 – Main comparison: E, C, S rows × 3 column groups
#   col 0: full T range
#   col 1: low-T zoom  T=[0.2, 3]
#   col 2: |error| log scale
# ════════════════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(17, 13), constrained_layout=True)
fig.suptitle(f"Thermal methods comparison — Heisenberg N={N}  (D=2^N={2**N})",
             fontsize=14, fontweight="bold")

gs_outer = gridspec.GridSpec(3, 3, figure=fig,
                              hspace=0.38, wspace=0.35)

OBS = [
    ("E", "Energy  E(T)",        E_ex, "energy",       (-4.6, 0)),
    ("C", "Specific heat  Cv(T)",C_ex, "specific_heat",(0, None)),
    ("S", "Entropy  S(T)",       S_ex, "entropy",      (0, None)),
]

for row, (key, ylabel, Y_exact, jkey, ylim) in enumerate(OBS):
    ax_full  = fig.add_subplot(gs_outer[row, 0])
    ax_zoom  = fig.add_subplot(gs_outer[row, 1])
    ax_err   = fig.add_subplot(gs_outer[row, 2])

    # exact reference
    for ax in (ax_full, ax_zoom):
        ax.plot(T_ex, Y_exact, "k-", lw=3, alpha=0.7, label="Exact", zorder=10)

    for m in methods:
        lbl = m["label"]
        st  = STYLES.get(lbl, dict(color="gray", ls="-", lw=1.2, alpha=0.7))
        T   = np.array(m["T"])
        Y   = np.array(m[key])   # 'E', 'C', or 'S'

        ax_full.plot(T, Y, label=lbl, **st)
        ax_zoom.plot(T, Y, **st)

        # residual vs exact (interpolated)
        Y_ref = np.interp(T, T_ex, Y_exact)
        dY    = np.abs(Y - Y_ref)
        ax_err.semilogy(T, np.maximum(dY, 5e-5), label=lbl, **st)

    # formatting
    for ax in (ax_full, ax_zoom):
        ax.set_xscale("log")
        ax.grid(True, which="both", alpha=0.2)
        if ylim[0] is not None: ax.set_ylim(bottom=ylim[0])
        if ylim[1] is not None: ax.set_ylim(top=ylim[1])
    ax_full.set_xlim(T_ex[0], T_ex[-1])
    ax_zoom.set_xlim(T_ex[0], 2.0)
    ax_err.set_xscale("log")
    ax_err.set_xlim(T_ex[0], T_ex[-1])
    ax_err.set_ylim(1e-4, 20)
    ax_err.grid(True, which="both", alpha=0.2)
    ax_err.axhline(0.08, color="red", ls=":", lw=1.5, alpha=0.7)
    ax_err.text(T_ex[-1]*0.15, 0.10, "tol=0.08", color="red", fontsize=7.5)

    ax_full.set_ylabel(ylabel, fontsize=10)
    ax_zoom.set_ylabel(ylabel, fontsize=10)
    ax_err.set_ylabel(f"|Δ{key}(T)|", fontsize=10)

    if row == 0:
        ax_full.set_title("Full range (log T)", fontsize=10, fontweight="bold")
        ax_zoom.set_title("Low-T zoom  (T ≤ 2, log T)", fontsize=10, fontweight="bold")
        ax_err.set_title("Absolute error vs exact (log-log)", fontsize=10, fontweight="bold")

    for ax in (ax_full, ax_zoom, ax_err):
        ax.set_xlabel("T", fontsize=9)

# shared legend
handles, labels = ax_full.get_legend_handles_labels()
# add exact
exact_line = plt.Line2D([0],[0], color="k", lw=3, alpha=0.7, label="Exact")
all_h = [exact_line] + [
    plt.Line2D([0],[0], label=m["label"], **STYLES.get(m["label"],
               dict(color="gray",ls="-",lw=1.5,alpha=0.8)))
    for m in methods
]
fig.legend(handles=all_h, loc="lower center", ncol=5,
           fontsize=9, bbox_to_anchor=(0.5, -0.02),
           framealpha=0.95)

fig.savefig("method_compare.png", dpi=150, bbox_inches="tight")
print("Saved method_compare.png")

# ════════════════════════════════════════════════════════════════════════════
# Figure 2 – Error bars: max|error| per method per observable
# ════════════════════════════════════════════════════════════════════════════
fig2, axes2 = plt.subplots(1, 3, figsize=(14, 5), constrained_layout=True)
fig2.suptitle(f"Max absolute error per method  (Heisenberg N={N})",
              fontsize=13, fontweight="bold")

obs2 = [("E","ΔE",E_ex,"energy",0.08),
        ("C","ΔCv",C_ex,"specific_heat",0.10),
        ("S","ΔS",S_ex,"entropy",0.08)]

for ax, (key, ylabel, Y_exact, jkey, tol) in zip(axes2, obs2):
    labels2, vals = [], []
    colors2 = []
    for m in methods:
        T = np.array(m["T"])
        Y = np.array(m[key])
        Y_ref = np.interp(T, T_ex, Y_exact)
        labels2.append(m["label"])
        vals.append(np.max(np.abs(Y - Y_ref)))
        colors2.append(STYLES.get(m["label"], {}).get("color","gray"))

    x = np.arange(len(labels2))
    bars = ax.bar(x, vals, color=colors2, alpha=0.8, edgecolor="black", linewidth=0.6)
    ax.axhline(tol, color="red", ls="--", lw=2, label=f"tol={tol}")
    ax.set_xticks(x)
    ax.set_xticklabels(labels2, rotation=35, ha="right", fontsize=8.5)
    ax.set_ylabel(f"max|{ylabel}|", fontsize=11)
    ax.set_title(ylabel, fontsize=11, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_ylim(0, max(tol*2, max(vals)*1.15) if vals else tol*2)
    ax.text(len(labels2)-0.5, tol*1.06, f"tol", color="red", fontsize=8)
    # annotate bar values
    for b, v in zip(bars, vals):
        ax.text(b.get_x()+b.get_width()/2, v+0.002, f"{v:.3f}",
                ha="center", va="bottom", fontsize=7.5, fontweight="bold")

fig2.savefig("method_compare_bars.png", dpi=150, bbox_inches="tight")
print("Saved method_compare_bars.png")

plt.close("all")
print("Done.")
