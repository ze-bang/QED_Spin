#!/usr/bin/env python3
"""Explainer figure for the BFG (Balents-Fisher-Girvin) kagome model.

Four panels:
  (a) lattice + the three Ising bond shells, one hexagon exploded into its
      15 pairs  ->  the Ising term is a sum of hexagon charges
  (b) the Hamiltonian as written, and where each piece lives in this repo
  (c) the classical ground-state manifold: Q_h = 0 on every hexagon
      (3 up / 3 down), plus what a violated hexagon costs
  (d) the bow-tie flip: the 4-spin ring exchange on the outer corners of two
      corner-sharing triangles

Geometry conventions match edlib.helper_kagome_bfg.generate_kagome_cluster:
    a1 = (1,0), a2 = (1/2, sqrt3/2); sublattice offsets (0,0), (1/2,0),
    (1/4, sqrt3/4); NN = 1/2, 2NN = sqrt3/2, 3NN = 1.

Usage:  python plot_bfg_model_explainer.py [--out DIR]
"""
from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyArrowPatch, Polygon

# ── palette ────────────────────────────────────────────────────────────────
C_NN   = "#0072B2"   # NN bonds (XY + Ising)
C_2NN  = "#E69F00"   # 2NN Ising
C_3NN  = "#CC79A7"   # 3NN Ising
C_HEX  = "#FFF2CC"   # hexagon fill
C_TRI  = "#D9EAF7"   # triangle fill
C_UP   = "#C62828"   # up spin
C_DN   = "#1565C0"   # down spin
C_BAD  = "#E53935"   # constraint-violating hexagon
C_TXT  = "#212121"

A1 = np.array([1.0, 0.0])
A2 = np.array([0.5, np.sqrt(3) / 2])
OFFS = [np.array([0.0, 0.0]), np.array([0.5, 0.0]),
        np.array([0.25, np.sqrt(3) / 4])]

# a hexagon anchored at cell (i,j), in (di, dj, sublattice) form
HEX_PATTERN = [(0, 0, 1), (0, 0, 2), (0, 1, 0), (0, 1, 1), (1, 0, 0), (1, 0, 2)]


# ── lattice patch ──────────────────────────────────────────────────────────
class Patch:
    """Open kagome patch of nx x ny unit cells, with bonds/triangles/hexagons."""

    def __init__(self, nx: int, ny: int):
        self.pos, self.cell = {}, {}
        sid = 0
        for i in range(nx):
            for j in range(ny):
                for s in range(3):
                    self.pos[sid] = i * A1 + j * A2 + OFFS[s]
                    self.cell[(i, j, s)] = sid
                    sid += 1
        self.n = sid
        self.nx, self.ny = nx, ny

        def shell(d):
            return [(a, b) for a, b in combinations(range(self.n), 2)
                    if abs(np.linalg.norm(self.pos[a] - self.pos[b]) - d) < 1e-6]

        self.nn = shell(0.5)
        self.nn2 = shell(np.sqrt(3) / 2)
        self.nn3 = shell(1.0)

        adj = {i: set() for i in range(self.n)}
        for a, b in self.nn:
            adj[a].add(b)
            adj[b].add(a)
        self.adj = adj
        self.tris = sorted({tuple(sorted((i, a, b)))
                            for i in range(self.n)
                            for a, b in combinations(sorted(adj[i]), 2)
                            if b in adj[a]})
        self.hexes = []
        for i in range(nx):
            for j in range(ny):
                key = [(i + di, j + dj, s) for di, dj, s in HEX_PATTERN]
                if all(k in self.cell for k in key):
                    self.hexes.append([self.cell[k] for k in key])
        # rings ordered by angle about the hexagon centre (for drawing)
        self.hex_rings = []
        for h in self.hexes:
            c = np.mean([self.pos[s] for s in h], axis=0)
            self.hex_rings.append(
                sorted(h, key=lambda s: np.arctan2(*(self.pos[s] - c)[::-1])))

    def xy(self, s):
        return self.pos[s]

    def hex_center(self, h):
        return np.mean([self.pos[s] for s in h], axis=0)


def draw_bonds(ax, patch, pairs, **kw):
    for a, b in pairs:
        p, q = patch.xy(a), patch.xy(b)
        ax.plot([p[0], q[0]], [p[1], q[1]], **kw)


def draw_sites(ax, patch, r=0.075, fc="white", ec="#37474F", sites=None, z=5):
    for s in (sites if sites is not None else range(patch.n)):
        ax.add_patch(Circle(patch.xy(s), r, fc=fc, ec=ec, lw=1.2, zorder=z))


def spin_arrow(ax, p, up, scale=1.0, lw=2.2):
    col = C_UP if up else C_DN
    dy = 0.17 * scale * (1 if up else -1)
    ax.annotate("", xy=(p[0], p[1] + dy / 2), xytext=(p[0], p[1] - dy / 2),
                arrowprops=dict(arrowstyle="-|>", color=col, lw=lw,
                                mutation_scale=11), zorder=7)


def q_hex(patch, h, spins):
    """Hexagon charge Q_h = sum of S^z over the six sites (units of 1/2)."""
    return sum(0.5 if spins[s] else -0.5 for s in h)


def find_flat_config(patch, seed=3):
    """Local search for a config with Q_h = 0 on every complete hexagon."""
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(400):
        spins = {s: bool(rng.integers(2)) for s in range(patch.n)}
        for _ in range(4000):
            cost = sum(q_hex(patch, h, spins) ** 2 for h in patch.hexes)
            if cost == 0:
                return spins
            s = int(rng.integers(patch.n))
            spins[s] = not spins[s]
            new = sum(q_hex(patch, h, spins) ** 2 for h in patch.hexes)
            if new > cost and rng.random() > 0.25:
                spins[s] = not spins[s]
        best = spins
    raise RuntimeError("no Q_h = 0 configuration found — do not draw `best`, "
                       "the panel would be labelled with charges it does not have")


# ── panel (a): bonds and the hexagon charge ────────────────────────────────
def panel_lattice(ax):
    p = Patch(4, 3)
    for t in p.tris:
        ax.add_patch(Polygon([p.xy(s) for s in t], fc=C_TRI, ec="none", zorder=1))
    draw_bonds(ax, p, p.nn, color=C_NN, lw=2.2, zorder=2, solid_capstyle="round")

    # highlight one hexagon and explode it into its 15 Ising pairs
    hi = len(p.hexes) // 2
    h, ring = p.hexes[hi], p.hex_rings[hi]
    ax.add_patch(Polygon([p.xy(s) for s in ring], fc=C_HEX, ec="#F9A825",
                         lw=2.0, zorder=1.5))
    inside = set(h)
    for a, b in p.nn2:
        if a in inside and b in inside:
            draw_bonds(ax, p, [(a, b)], color=C_2NN, lw=1.8, ls="--", zorder=3)
    for a, b in p.nn3:
        if a in inside and b in inside:
            draw_bonds(ax, p, [(a, b)], color=C_3NN, lw=1.8, ls=":", zorder=3)

    draw_sites(ax, p)
    draw_sites(ax, p, r=0.085, fc="#FFE082", sites=h, z=6)
    c = p.hex_center(h)
    ax.annotate(r"one hexagon $h$: its $\binom{6}{2}=15$ pairs are exactly"
                "\n" r"6 NN $+$ 6 2NN $+$ 3 3NN, each counted once, so"
                "\n" r"$J_z\!\!\sum_{i<j\in h}\!S^z_iS^z_j=\frac{J_z}{2}Q_h^2"
                r"-\frac{3J_z}{8}$,   $Q_h=\sum_{i\in h}S^z_i$",
                xy=(c[0], c[1] - 0.36), xytext=(0.5, 0.01),
                textcoords="axes fraction",
                fontsize=10.5, color=C_TXT, ha="center", va="bottom",
                arrowprops=dict(arrowstyle="->", color="#F9A825", lw=1.6),
                bbox=dict(fc="white", ec="#F9A825", lw=1.2, boxstyle="round,pad=0.4"))

    ax.legend(handles=[
        Line2D([], [], color=C_NN, lw=2.6, label=r"NN: XY $J_\pm$ + Ising $J_z$"),
        Line2D([], [], color=C_2NN, lw=1.8, ls="--", label=r"2NN: Ising $J_z$ only"),
        Line2D([], [], color=C_3NN, lw=1.8, ls=":", label=r"3NN: Ising $J_z$ only"),
    ], loc="upper left", fontsize=9.5, framealpha=0.95,
        bbox_to_anchor=(-0.02, 1.02))
    ax.set_title("(a)  kagome bonds — the Ising shells build hexagon charges",
                 fontsize=12.5, loc="left", fontweight="bold", pad=10)
    finish(ax, p, pad=0.35, extra_bottom=0.8)


# ── panel (b): Hamiltonian + implementation ────────────────────────────────
def panel_hamiltonian(ax):
    ax.axis("off")
    ax.set_title("(b)  the model, and where it lives in the code",
                 fontsize=12.5, loc="left", fontweight="bold")

    ax.text(0.5, 0.955,
            r"$H=-J_\pm\!\!\sum_{\langle ij\rangle_{\rm NN}}\!\!"
            r"\left(S^+_iS^-_j+S^-_iS^+_j\right)"
            r"\;+\;J_z\!\!\sum_{(ij)\in\{{\rm NN},2{\rm NN},3{\rm NN}\}}\!\!"
            r"S^z_iS^z_j$",
            ha="center", va="top", fontsize=13.5, transform=ax.transAxes)
    ax.text(0.5, 0.845,
            r"$=\;-J_\pm\!\!\sum_{\langle ij\rangle}\!\left(S^+_iS^-_j+{\rm h.c.}\right)"
            r"\;+\;\frac{J_z}{2}\sum_{h\,\in\,{\rm hexagons}}Q_h^2"
            r"\;+\;{\rm const},\qquad Q_h=\!\!\sum_{i\in h}\!S^z_i$",
            ha="center", va="top", fontsize=13.5, transform=ax.transAxes,
            bbox=dict(fc="#FFFDE7", ec="#F9A825", lw=1.3,
                      boxstyle="round,pad=0.45"))
    ax.text(0.5, 0.735,
            r"easy-axis limit $J_\pm \ll J_z$  ·  BFG spin-liquid regime at "
            r"$J_\pm>0$, $J_z>0$",
            ha="center", va="top", fontsize=10.5, color="#555", style="italic",
            transform=ax.transAxes)

    rows = [
        ("bond shells",
         "helper_kagome_bfg.generate_kagome_cluster\n"
         "NN_BONDS / NN2_BONDS / NN3_BONDS  →  edges, edges_2nn, edges_3nn"),
        ("operator, in memory",
         "run_bfg_ground_state.build_bfg_operator\n"
         "op.add_two_body(OP_SPLUS, i, OP_SMINUS, j, -Jpm)   # + h.c.\n"
         "op.add_two_body(OP_SZ,    i, OP_SZ,     j,  Jzz)   # NN/2NN/3NN"),
        ("operator, on disk",
         "helper_kagome_bfg.prepare_hamiltonian_parameters → InterAll.dat\n"
         "rows:  2 i 2 j  Jzz 0  |  0 i 1 j -Jpm 0  |  1 i 0 j -Jpm 0\n"
         "op codes:  0 = S+,  1 = S-,  2 = Sz"),
        ("solve",
         "U(1): fixed n_up = N/2 sector; translations Z_dim1 x Z_dim2\n"
         "one diagonalisation per momentum sector\n"
         "GSD = 4 on the torus is the Z2 topological-order check"),
        ("observables",
         "compute_bfg_order_parameters.find_bowties\n"
         "                          .compute_bowtie_resonance  →  P_bt\n"
         "plus spin structure factor, bond nematicity, chirality"),
    ]
    y = 0.665
    for label, body in rows:
        n = body.count("\n") + 1
        hgt = 0.0335 * n + 0.030
        ax.add_patch(plt.Rectangle((0.03, y - hgt), 0.94, hgt,
                                   transform=ax.transAxes, fc="#F5F7FA",
                                   ec="#B0BEC5", lw=1.0, zorder=1,
                                   clip_on=False))
        ax.text(0.05, y - 0.022, label, transform=ax.transAxes, fontsize=9.5,
                fontweight="bold", color="#37474F", va="top")
        ax.text(0.285, y - 0.020, body, transform=ax.transAxes, fontsize=8.6,
                family="monospace", color=C_TXT, va="top", linespacing=1.45)
        y -= hgt + 0.022


# ── panel (c): the ground-state manifold ───────────────────────────────────
def panel_ground_state(ax):
    p = Patch(4, 3)
    spins = find_flat_config(p)
    draw_bonds(ax, p, p.nn, color="#90A4AE", lw=1.8, zorder=2)
    for h, ring in zip(p.hexes, p.hex_rings):
        ax.add_patch(Polygon([p.xy(s) for s in ring], fc=C_HEX, ec="#F9A825",
                             lw=1.3, zorder=1))
        c = p.hex_center(h)
        ax.text(c[0], c[1], r"$Q_h\!=\!0$", ha="center", va="center",
                fontsize=8.5, color="#F57F17", fontweight="bold", zorder=8)
    draw_sites(ax, p, r=0.062, fc="white")
    for s in range(p.n):
        spin_arrow(ax, p.xy(s), spins[s], scale=0.95, lw=2.0)

    # flipping any single spin charges the two hexagons that share it
    site = sorted(set(p.hexes[0]) & set(p.hexes[1]))[0]
    ax.add_patch(Circle(p.xy(site), 0.17, fc="none", ec=C_BAD, lw=2.0,
                        ls="--", zorder=9))
    for h in (p.hexes[0], p.hexes[1]):
        c = p.hex_center(h)
        ax.add_patch(Circle(c, 0.40, fc="none", ec=C_BAD, lw=1.8, ls=":", zorder=9))
    ax.annotate("flip this one spin and the two hexagons sharing it go to\n"
                r"$Q_h=\pm1$: a pair of gapped $Z_2$ charges (spinons), cost $\sim J_z$",
                xy=p.xy(site), xytext=(0.5, -0.02), textcoords="axes fraction",
                fontsize=9.5, ha="center", va="bottom", color=C_BAD,
                arrowprops=dict(arrowstyle="->", color=C_BAD, lw=1.5),
                bbox=dict(fc="white", ec=C_BAD, lw=1.0, boxstyle="round,pad=0.35"))

    ax.set_title("(c)  ground states at $J_\\pm\\!=\\!0$:  $Q_h=0$ (3$\\uparrow$/3$\\downarrow$) on "
                 "every hexagon\n" + " " * 6 +
                 "— extensively degenerate; the constraint is a $Z_2$ Gauss law",
                 fontsize=12.0, loc="left", fontweight="bold", pad=10)
    finish(ax, p, pad=0.4, extra_bottom=1.0)


# ── panel (d): the bow-tie flip ────────────────────────────────────────────
def pick_bowtie(p: Patch):
    """Return (s0, ring, hexes4) for a bow-tie whose four hexagons are complete.

    ring = the four outer corners in ring order; consecutive corners share a
    hexagon, and those four hexagons are the ones the flip must not charge.
    """
    site_tri = {s: [t for t in p.tris if s in t] for s in range(p.n)}
    site_hex = {s: [i for i, h in enumerate(p.hexes) if s in h] for s in range(p.n)}
    centre_pt = np.mean([p.xy(s) for s in range(p.n)], axis=0)
    for s0 in sorted(range(p.n), key=lambda s: np.linalg.norm(p.xy(s) - centre_pt)):
        if len(site_tri[s0]) != 2:
            continue
        o1 = [v for v in site_tri[s0][0] if v != s0]
        o2 = [v for v in site_tri[s0][1] if v != s0]
        # chain the four corners into a ring via shared hexagons
        ring = [o1[0], o1[1]]
        nxt = [v for v in o2 if set(site_hex[v]) & set(site_hex[ring[1]])]
        if not nxt:
            continue
        ring.append(nxt[0])
        ring.append(o2[0] if o2[1] == nxt[0] else o2[1])
        pairs = [(ring[i], ring[(i + 1) % 4]) for i in range(4)]
        shared = [set(site_hex[a]) & set(site_hex[b]) for a, b in pairs]
        if any(len(s) != 1 for s in shared):
            continue
        return s0, ring, [next(iter(s)) for s in shared]
    raise RuntimeError("no fully-surrounded bow-tie in this patch")


def panel_bowtie(ax):
    p = Patch(5, 4)
    s0, ring, hexes4 = pick_bowtie(p)

    draw_bonds(ax, p, p.nn, color="#CFD8DC", lw=1.6, zorder=1)
    for hi in hexes4:
        h_ring = p.hex_rings[hi]
        ax.add_patch(Polygon([p.xy(s) for s in h_ring], fc=C_HEX, ec="#F9A825",
                             lw=1.4, zorder=2))
        c = p.hex_center(p.hexes[hi])
        ax.text(c[0], c[1], r"$\Delta Q_h\!=\!0$", ha="center", va="center",
                fontsize=8.5, color="#F57F17", fontweight="bold", zorder=8)
    for t in p.tris:
        if s0 in t:
            ax.add_patch(Polygon([p.xy(s) for s in t], fc=C_TRI, ec=C_NN,
                                 lw=2.6, zorder=3))

    # cyclic flip arrows around the ring; alternating +/- along the ring
    for i in range(4):
        a, b = p.xy(ring[i]), p.xy(ring[(i + 1) % 4])
        ax.add_patch(FancyArrowPatch(a + 0.25 * (b - a), a + 0.75 * (b - a),
                                     arrowstyle="-|>", mutation_scale=13,
                                     lw=2.0, color="#2E7D32", ls="--", zorder=6,
                                     connectionstyle="arc3,rad=0.3"))
    draw_sites(ax, p, r=0.07, fc="white")
    draw_sites(ax, p, r=0.10, fc="white", ec="#2E7D32", sites=ring, z=7)
    draw_sites(ax, p, r=0.10, fc="#ECEFF1", ec="#455A64", sites=[s0], z=7)
    for k, s in enumerate(ring):
        spin_arrow(ax, p.xy(s), k % 2 == 0, scale=1.15, lw=2.4)
        off = 0.24 * (p.xy(s) - p.xy(s0)) / np.linalg.norm(p.xy(s) - p.xy(s0))
        ax.text(*(p.xy(s) + off * 1.5), f"{k + 1}", ha="center", va="center",
                fontsize=11, color="#2E7D32", fontweight="bold", zorder=9)
    ax.annotate(r"shared corner $s_0$: untouched",
                xy=p.xy(s0), xytext=(p.xy(s0)[0] + 0.05, p.xy(s0)[1] - 1.15),
                ha="center", va="top", fontsize=9.5, color="#455A64", zorder=9,
                arrowprops=dict(arrowstyle="->", color="#455A64", lw=1.3),
                bbox=dict(fc="white", ec="#B0BEC5", lw=1.0,
                          boxstyle="round,pad=0.3"))

    ax.text(0.5, 0.235,
            r"$P_{\rm bt}=\left\langle S^+_1S^-_2S^+_3S^-_4+{\rm h.c.}\right\rangle$"
            r"$\;:\;|\!\uparrow\downarrow\uparrow\downarrow\rangle"
            r"\leftrightarrow|\!\downarrow\uparrow\downarrow\uparrow\rangle$",
            transform=ax.transAxes, ha="center", va="top", fontsize=12.5,
            bbox=dict(fc="#E8F5E9", ec="#2E7D32", lw=1.3, boxstyle="round,pad=0.4"))
    ax.text(0.5, 0.135,
            "Two triangles sharing ONE corner. The flip acts on the four outer corners\n"
            "and leaves the shared corner alone; the ring alternates a triangle edge (NN)\n"
            "and a hop across the corner (2NN). Consecutive corners sit in a common\n"
            r"hexagon, so each of the four hexagons gets one $S^+$ and one $S^-$ and stays"
            "\n"
            r"at $Q_h=0$ — this is the leading move that keeps the state inside the"
            "\n"
            r"ground-state manifold, generated at order $J_\pm^2/J_z$.",
            transform=ax.transAxes, ha="center", va="top", fontsize=9.6,
            color=C_TXT, linespacing=1.5)

    ax.set_title("(d)  the bow-tie flip — the move that survives the constraint",
                 fontsize=12.5, loc="left", fontweight="bold", pad=10)
    c0 = p.xy(s0)
    ax.set_xlim(c0[0] - 1.6, c0[0] + 1.6)
    ax.set_ylim(c0[1] - 2.9, c0[1] + 1.2)
    ax.set_aspect("equal")
    ax.axis("off")


def finish(ax, patch, pad=0.3, extra_bottom=0.0):
    xs = [patch.xy(s)[0] for s in range(patch.n)]
    ys = [patch.xy(s)[1] for s in range(patch.n)]
    ax.set_xlim(min(xs) - pad, max(xs) + pad)
    ax.set_ylim(min(ys) - pad - extra_bottom, max(ys) + pad)
    ax.set_aspect("equal")
    ax.axis("off")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    default_out = Path(__file__).resolve().parent / "bfg_visualization"
    ap.add_argument("--out", default=str(default_out),
                    help="output directory (default: ./bfg_visualization "
                         "next to this script)")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(15.0, 12.0))
    fig.suptitle("The BFG (Balents–Fisher–Girvin) easy-axis kagome model",
                 fontsize=16, fontweight="bold", y=0.985)
    panel_lattice(axes[0, 0])
    panel_hamiltonian(axes[0, 1])
    panel_ground_state(axes[1, 0])
    panel_bowtie(axes[1, 1])
    fig.tight_layout(rect=(0, 0.0, 1, 0.965))

    for ext in ("png", "pdf"):
        path = out / f"bfg_model_explainer.{ext}"
        fig.savefig(path, dpi=200, bbox_inches="tight")
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
