"""Two-panel figure:
Left  — convention difference: J_perp (DMRG) vs J_pm (ours)
Right — E0 vs J_pm (our convention), DMRG point at J_pm=0.025
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

DMRG_E   = -4.67977542686
DMRG_JPM =  0.025          # their Jperp=0.05 = our J_pm=0.025

# All ED points (our convention, collab bond set)
DATA = [
    (-0.10,  -5.36458416520),
    (-0.05,  -4.87306414337),
    (-0.025, -4.66920939387),
    ( 0.00,  -4.50000000000),
    ( 0.0125,-4.58210746302),
    ( 0.025, -4.67977120475),
    ( 0.0375,-4.79604775217),
    ( 0.05,  -4.93816504157),
    ( 0.075, -5.32640492206),
    ( 0.10,  -5.81673011267),
]
xs = [d[0] for d in DATA]
ys = [d[1] for d in DATA]

fig = plt.figure(figsize=(14, 5.5))
gs  = gridspec.GridSpec(1, 2, width_ratios=[1.0, 1.4], wspace=0.30)
axL = fig.add_subplot(gs[0])
axR = fig.add_subplot(gs[1])

# ── LEFT: convention ─────────────────────────────────────────────────────────
axL.axis("off")
axL.set_xlim(0, 1); axL.set_ylim(0, 1)

def tx(ax, y, s, **kw):
    ax.text(0.5, y, s, ha="center", va="center", transform=ax.transAxes, **kw)

tx(axL, 0.93, "Operator identity", fontsize=11, fontweight="bold", color="#333")
tx(axL, 0.83,
   r"$S^+_i S^-_j + S^-_i S^+_j \;=\; 2\,(S^x_i S^x_j + S^y_i S^y_j)$",
   fontsize=13.5)

axL.axhline(0.755, 0.05, 0.95, color="#ccc", lw=1)

tx(axL, 0.69, "Our convention", fontsize=10.5, fontweight="bold", color="#1565C0")
tx(axL, 0.595,
   r"$H_{XY} = -J_\pm \sum_{\langle ij \rangle}(S^+_i S^-_j + S^-_i S^+_j)$",
   fontsize=12.5, color="#1565C0")

axL.axhline(0.52, 0.05, 0.95, color="#ccc", lw=1)

tx(axL, 0.455, "DMRG convention", fontsize=10.5, fontweight="bold", color="#C62828")
tx(axL, 0.355,
   r"$H_{XY} = -J_\perp \sum_{\langle ij \rangle}(S^x_i S^x_j + S^y_i S^y_j)$",
   fontsize=12.5, color="#C62828")
tx(axL, 0.265,
   r"$= -\frac{J_\perp}{2} \sum_{\langle ij \rangle}(S^+_i S^-_j + S^-_i S^+_j)$",
   fontsize=12.5, color="#C62828")

axL.axhline(0.185, 0.05, 0.95, color="#ccc", lw=1)

# mapping box
axL.add_patch(plt.Rectangle((0.1, 0.045), 0.80, 0.115,
              transform=axL.transAxes,
              facecolor="#FFF8E1", edgecolor="#FB8C00", lw=2, zorder=2))
tx(axL, 0.103,
   r"$J_\perp = 2\, J_\pm \quad\Longrightarrow\quad J_\perp = 0.05 \;\leftrightarrow\; J_\pm = 0.025$",
   fontsize=11.5, color="#E65100", zorder=3)

# ── RIGHT: E0 vs J_pm ────────────────────────────────────────────────────────
axR.plot(xs, ys, "-o", color="#1565C0", lw=2, ms=6, zorder=3, label="ED (this work)")

# DMRG point
axR.scatter([DMRG_JPM], [DMRG_E], s=200, marker="*", color="#C62828",
            zorder=5, label=f"DMRG  ($J_\\perp=0.05$)")
axR.axvline(DMRG_JPM, color="#C62828", ls="--", lw=1.2, alpha=0.55)
axR.axhline(DMRG_E,   color="#C62828", ls="--", lw=1.2, alpha=0.55)

# annotation
axR.annotate(
    rf"$J_\pm = {DMRG_JPM}$,  $E_0 = {DMRG_E:.5f}$" "\n"
    r"ED: $-4.67977120$   residual $4\times10^{-6}$",
    xy=(DMRG_JPM, DMRG_E),
    xytext=(DMRG_JPM + 0.018, DMRG_E + 0.19),
    fontsize=9.5, color="#C62828",
    arrowprops=dict(arrowstyle="->", color="#C62828", lw=1.2),
    bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#C62828", alpha=0.85))

axR.set_xlabel(r"$J_\pm$  (our convention)", fontsize=13)
axR.set_ylabel(r"$E_0$", fontsize=13)
axR.set_title(
    r"BFG $3\times3$ kagome cylinder — $E_0$ vs $J_\pm$"
    "\n"
    r"(48 NN XY + 6-hexagon Ising, $J_z = 1$, $N = 27$)",
    fontsize=11)
axR.grid(True, alpha=0.3)
axR.legend(fontsize=10.5, loc="lower left")

fig.suptitle(
    r"Benchmarking against DMRG: the only difference is a factor-of-2 convention  ($J_\perp = 2\,J_\pm$)",
    fontsize=12, y=1.01)
fig.tight_layout()

out = "/scratch/zhouzb79/bfg_gs_results/convention_and_benchmark.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print("Saved:", out)
