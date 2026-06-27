"""Illustrate the Jperp convention difference and the ED -> DMRG energy match.

Panel A: the spin-operator identity and the two XY conventions, showing
         Jperp_theirs = 2 * Jpm_mine.
Panel B: ED ground-state energy for each candidate XY convention vs the
         collaborator's DMRG result; only -Jperp(SxSx+SySy) matches.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DMRG = -4.67977542686

# ED trials (collaborator bonds: 48 NN XY + 90 hexagon Ising pairs, Jz=1, Jperp=0.05)
trials = [
    (r"$-J_\perp\,(S^+S^-{+}\,\mathrm{h.c.})$",      -4.93816504157, False),
    (r"$+J_\perp\,(S^+S^-{+}\,\mathrm{h.c.})$",      -4.87306414337, False),
    (r"$-J_\perp\,(S^xS^x{+}S^yS^y)$",               -4.67977120475, True),
]

fig = plt.figure(figsize=(15, 6.2))
gsp = fig.add_gridspec(1, 2, width_ratios=[1.05, 1.0], wspace=0.18)
axA = fig.add_subplot(gsp[0, 0])
axB = fig.add_subplot(gsp[0, 1])

# ── Panel A: convention identity ────────────────────────────────────────────
axA.axis("off")
axA.set_title("XY coupling convention", fontsize=13, pad=12)

axA.text(0.5, 0.93, "Operator identity", ha="center", fontsize=11,
         fontweight="bold", color="#333")
axA.text(0.5, 0.85,
         r"$S^+_iS^-_j + S^-_iS^+_j \;=\; 2\,(S^x_iS^x_j + S^y_iS^y_j)$",
         ha="center", fontsize=14)

axA.axhline(0.79, 0.06, 0.94, color="#ccc", lw=1)

# mine
axA.text(0.5, 0.72, "My convention", ha="center", fontsize=11,
         fontweight="bold", color="#1565C0")
axA.text(0.5, 0.63,
         r"$H_{XY} = -J_{pm}\sum_{\langle ij\rangle}(S^+_iS^-_j + S^-_iS^+_j)$",
         ha="center", fontsize=13, color="#1565C0")

axA.axhline(0.555, 0.06, 0.94, color="#ccc", lw=1)

# theirs
axA.text(0.5, 0.485, "Collaborator convention", ha="center", fontsize=11,
         fontweight="bold", color="#C62828")
axA.text(0.5, 0.37,
         r"$H_{XY} = -J_{\perp}\sum_{\langle ij\rangle}(S^x_iS^x_j + S^y_iS^y_j)$"
         "\n"
         r"$\qquad\;\, = -\frac{J_\perp}{2}\sum_{\langle ij\rangle}(S^+_iS^-_j + S^-_iS^+_j)$",
         ha="center", va="center", fontsize=12.5, color="#C62828")

axA.axhline(0.265, 0.06, 0.94, color="#ccc", lw=1)

# mapping box
axA.add_patch(plt.Rectangle((0.18, 0.115), 0.64, 0.11, transform=axA.transAxes,
              facecolor="#FFF8E1", edgecolor="#FB8C00", lw=2))
axA.text(0.5, 0.17,
         r"$J_\perp^{\,\mathrm{theirs}} \;=\; 2\,J_{pm}^{\,\mathrm{mine}}$",
         ha="center", va="center", fontsize=15, color="#E65100")
axA.text(0.5, 0.045,
         r"their $J_\perp=0.05 \;\;\Longleftrightarrow\;\;$ my $J_{pm}=0.025$",
         ha="center", fontsize=11, color="#666")

# ── Panel B: ED vs DMRG ─────────────────────────────────────────────────────
axB.set_title("ED reproduction of collaborator's Hamiltonian\n"
              r"(3$\times$3 kagome cyl, $J_z{=}1$, $J_\perp{=}0.05$, 48 NN XY + 6 hex Ising)",
              fontsize=11)

ypos = np.arange(len(trials))[::-1]
for y, (label, E, match) in zip(ypos, trials):
    color = "#2E7D32" if match else "#9E9E9E"
    axB.barh(y, E, height=0.55, color=color, alpha=0.85,
             edgecolor="black", lw=0.8, zorder=3)
    axB.text(E - 0.004, y, f"{E:.6f}", va="center", ha="right",
             fontsize=10, color="black", fontweight="bold", zorder=4)
    if match:
        axB.text(E + 0.5*(0 - E) , y, "MATCH", va="center", ha="center",
                 fontsize=11, color="white", fontweight="bold", zorder=5)

# DMRG target line
axB.axvline(DMRG, color="#C62828", lw=2, ls="--", zorder=2)
axB.text(DMRG - 0.004, len(trials)-0.35, f"DMRG = {DMRG:.6f}  ",
         color="#C62828", fontsize=10, ha="right", va="bottom", fontweight="bold")

axB.set_yticks(ypos)
axB.set_yticklabels([t[0] for t in trials], fontsize=12)
axB.set_xlabel(r"$E_0$", fontsize=13)
axB.set_xlim(-5.02, -4.63)
axB.grid(True, axis="x", alpha=0.3)
axB.set_ylim(-0.6, len(trials)-0.2)

# inset: residual of the matching one
res = abs(trials[-1][1] - DMRG)
axB.text(0.99, 0.02,
         f"match residual = {res:.1e}\n(Lanczos vs DMRG conv.)",
         transform=axB.transAxes, ha="right", va="bottom", fontsize=8.5,
         color="#555", bbox=dict(boxstyle="round", fc="white", ec="#bbb"))

fig.suptitle("The only difference in $J_\\perp$: a factor-of-2 XY normalization convention",
             fontsize=13, y=1.02)
fig.tight_layout()
out = "/scratch/zhouzb79/bfg_gs_results/jperp_convention.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print("Saved:", out)
