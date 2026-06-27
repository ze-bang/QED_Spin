"""Plot E0 vs Jpm (our convention) on the collaborator's bond set.
Mark that Jpm=0.025 reproduces the DMRG energy.
"""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DMRG = -4.67977542686
DMRG_JPM = 0.025   # our convention: Jpm = Jperp/2 = 0.05/2

# Load main sweep
with open("/scratch/zhouzb79/bfg_gs_results/collab_jpm_sweep.json") as f:
    data = json.load(f)

# Merge any extra single-point jobs (bfg_jpm_m010_*.out etc.)
import glob, re
for path in glob.glob("/scratch/zhouzb79/bfg_jpm_m010_*.out"):
    for line in open(path):
        m = re.match(r"Jpm=([+-]\d+\.\d+)\s+E0=([+-]\d+\.\d+)", line)
        if m:
            jpm, e0 = float(m.group(1)), float(m.group(2))
            if not any(abs(d[0] - jpm) < 1e-8 for d in data):
                data.append((jpm, e0))
                print(f"Added extra point: Jpm={jpm:+.4f}  E0={e0:.8f}")

data.sort(key=lambda t: t[0])
xs = [d[0] for d in data]
ys = [d[1] for d in data]

fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(xs, ys, "-o", color="#1565C0", lw=2, ms=6, zorder=3, label="ED")

# DMRG point
ax.scatter([DMRG_JPM], [DMRG], s=160, marker="*", color="#C62828",
           zorder=5, label="DMRG")
ax.axvline(DMRG_JPM, color="#C62828", ls="--", lw=1, alpha=0.6, zorder=1)
ax.axhline(DMRG, color="#C62828", ls="--", lw=1, alpha=0.6, zorder=1)
ax.annotate(rf"$J_\pm = {DMRG_JPM}$" "\n" rf"$E_0 = {DMRG:.6f}$",
            xy=(DMRG_JPM, DMRG), xytext=(DMRG_JPM + 0.012, DMRG + 0.06),
            fontsize=11, color="#C62828",
            arrowprops=dict(arrowstyle="->", color="#C62828"))

ax.set_xlabel(r"$J_\pm$  (our convention)", fontsize=13)
ax.set_ylabel(r"$E_0$", fontsize=13)
ax.set_title(r"BFG kagome $3\times3$ cylinder: $E_0$ vs $J_\pm$"
             "\n" r"(48 NN XY + 6-hexagon Ising, $J_z=1$)", fontsize=12)
ax.grid(True, alpha=0.3)
ax.legend(fontsize=11, loc="upper right")
fig.tight_layout()
out = "/scratch/zhouzb79/bfg_gs_results/collab_e0_vs_jpm.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print("Saved:", out)
