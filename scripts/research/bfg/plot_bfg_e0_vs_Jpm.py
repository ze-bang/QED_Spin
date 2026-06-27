"""plot_bfg_e0_vs_Jpm.py

Plot ground-state energy per site  E0 / N  as a function of Jpm
for the 3×3 kagome BFG cluster with three boundary conditions:
  OBC   — open on all edges
  cyl1  — PBC along a1, OBC along a2
  PBC   — torus (full periodic)

Shows two datasets side by side:
  solid  lines — full Hamiltonian (all bonds)
  dashed lines — complete-hexagons-only Hamiltonian (_chex suffix)

Reads from:
  <results_dir>/kagome_bfg_3x3_{bc}_Jpm{jpm}_Jzz1.0000/ed_results.h5
  <results_dir>/kagome_bfg_3x3_{bc}_Jpm{jpm}_Jzz1.0000_chex/ed_results.h5
  dataset: eigendata/eigenvalues  (sorted; first entry = E0)

Usage:
  python plot_bfg_e0_vs_Jpm.py [--results-dir DIR] [--out FILE]
"""

import argparse
import re
import sys
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------
BC_COLORS = {
    "obc":  "#2196F3",   # blue
    "cyl1": "#FF9800",   # orange
    "pbc":  "#4CAF50",   # green
}
BC_LABELS = {
    "obc":  "OBC",
    "cyl1": "CYL (PBC∥a₁)",
    "pbc":  "PBC (torus)",
}
BC_MARKERS = {"obc": "o", "cyl1": "s", "pbc": "^"}
BC_ORDER = ["obc", "cyl1", "pbc"]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _read_e0(h5_path: Path, label: str) -> float | None:
    if not h5_path.exists():
        print(f"  [missing]  {label}", file=sys.stderr)
        return None
    try:
        with h5py.File(h5_path, "r") as f:
            evals = f["eigendata/eigenvalues"][()]
    except Exception as exc:
        print(f"  [error reading {h5_path}]: {exc}", file=sys.stderr)
        return None
    if len(evals) == 0:
        print(f"  [0 eigenvalues] {label}", file=sys.stderr)
        return None
    return float(np.min(evals))


def load_results(results_dir: Path) -> tuple[dict, dict]:
    """Return (data_full, data_chex) each mapping {bc: {Jpm: E0/N}}."""
    pattern = re.compile(
        r"kagome_bfg_3x3_(?P<bc>obc|cyl1|cyl2|pbc)_Jpm(?P<jpm>[+-]\d+\.\d+)_Jzz"
        r"[\d.]+(?P<chex>_chex)?$"
    )
    data_full: dict[str, dict[float, float]] = {bc: {} for bc in BC_ORDER}
    data_chex: dict[str, dict[float, float]] = {bc: {} for bc in BC_ORDER}

    N = 27  # 3×3 kagome
    for d in sorted(results_dir.iterdir()):
        if not d.is_dir():
            continue
        m = pattern.match(d.name)
        if not m:
            continue
        bc   = m.group("bc")
        jpm  = float(m.group("jpm"))
        chex = m.group("chex") is not None
        if bc not in BC_ORDER:
            continue

        e0 = _read_e0(d / "ed_results.h5", d.name)
        if e0 is None:
            continue

        target = data_chex if chex else data_full
        target[bc][jpm] = e0 / N
        tag = "chex" if chex else "full"
        print(f"  {bc:5s} [{tag}]  Jpm={jpm:+.4f}  E0/N={e0/N:.6f}")

    return data_full, data_chex


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot(data_full: dict, data_chex: dict, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.8), sharey=True)

    for ax, (data, title_suffix) in zip(
        axes,
        [(data_full, "Full Hamiltonian"),
         (data_chex, "Complete hexagons only")],
    ):
        for bc_idx, bc in enumerate(BC_ORDER):
            pts = data.get(bc, {})
            if not pts:
                continue
            jpm_arr = np.array(sorted(pts.keys()))
            e0_arr  = np.array([pts[j] for j in jpm_arr])
            ax.plot(jpm_arr, e0_arr,
                    color=BC_COLORS[bc], marker=BC_MARKERS[bc],
                    ls="-", ms=6, lw=1.8,
                    label=BC_LABELS[bc], zorder=3)
            # Stagger vertical offsets so labels don't overlap
            y_offsets = [14, 5, -12]   # obc above, cyl middle, pbc below
            y_off = y_offsets[bc_idx]
            for jpm, e0 in zip(jpm_arr, e0_arr):
                ax.annotate(f"{e0:.4f}",
                            xy=(jpm, e0),
                            xytext=(0, y_off), textcoords="offset points",
                            ha="center", va="center",
                            fontsize=6.0, color=BC_COLORS[bc],
                            rotation=0)

        ax.set_xlabel(r"$J_{\pm}$", fontsize=13)
        ax.set_title(title_suffix, fontsize=11)
        ax.axvline(0, color="grey", lw=0.8, ls="--", alpha=0.6)
        ax.xaxis.set_minor_locator(mticker.AutoMinorLocator())
        ax.yaxis.set_minor_locator(mticker.AutoMinorLocator())
        ax.legend(fontsize=10, framealpha=0.85)
        ax.grid(True, which="major", alpha=0.3)

    axes[0].set_ylabel(r"$E_0 / N$", fontsize=13)
    fig.suptitle(
        r"BFG kagome $3\times3$: $E_0/N$ vs $J_{\pm}$"
        r"  ($J_{zz}=1$, NN+2NN+3NN Ising)",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"\nSaved: {out_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results-dir", default="/scratch/zhouzb79/bfg_gs_results",
                   help="Directory containing kagome_bfg_* result subdirs")
    p.add_argument("--out", default=None,
                   help="Output PNG path (default: <results_dir>/bfg_e0_vs_Jpm.png)")
    args = p.parse_args()

    results_dir = Path(args.results_dir)
    out_path    = Path(args.out) if args.out else results_dir / "bfg_e0_vs_Jpm.png"

    print(f"Scanning {results_dir} ...")
    data_full, data_chex = load_results(results_dir)
    plot(data_full, data_chex, out_path)


if __name__ == "__main__":
    main()
