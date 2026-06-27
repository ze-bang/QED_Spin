"""
Visualize the "complete hexagons only" Hamiltonian for OBC, CYL, PBC.

For each boundary condition:
  - Identify which hexagons have ALL 15 internal bond-pairs present
  - Color bonds: green = kept (in complete hexagon), red = removed (broken hexagon)
  - Print bond counts

Usage:
    python viz_and_run_complete_hex.py [--save /path/to/output.png]
"""

import sys, os, itertools, argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection

sys.path.insert(0, '/home/zhouzb79/links/projects/def-ybkim/zhouzb79/QED/python')
from edlib.helper_kagome_bfg import generate_kagome_cluster, create_nn_lists

# ─── helpers ───────────────────────────────────────────────────────────────

def get_bonds(dim1, dim2, pbc, pbc_dim1, pbc_dim2):
    verts, bnn, b2nn, b3nn, nmap, v2cell = generate_kagome_cluster(
        dim1, dim2, use_pbc=pbc, pbc_dim1=pbc_dim1, pbc_dim2=pbc_dim2)
    _, _, _, pos, _ = create_nn_lists(bnn, b2nn, b3nn, nmap, verts, v2cell)
    # pos is a dict {vertex_id: (x, y)}
    # bonds as frozensets for easy lookup
    bond_set = set(frozenset((a, b)) for (a, b) in bnn + b2nn + b3nn)
    return verts, bnn, b2nn, b3nn, pos, bond_set


def find_pbc_hexagons(bnn_pbc, N=27):
    """Return the 9 hexagons of the 3×3 PBC kagome as frozensets of 6 site IDs."""
    adj = {i: set() for i in range(N)}
    for (a, b) in bnn_pbc:
        adj[a].add(b); adj[b].add(a)

    hexagons = {}
    # generate 3NN bonds from the PBC cluster to use as "seeds"
    verts_pbc, bnn_p, b2nn_p, b3nn_p, nmap_p, v2cell_p = \
        generate_kagome_cluster(3, 3, use_pbc=True)

    for (a, d) in b3nn_p:
        paths = []
        for b in adj[a]:
            for c in adj[b]:
                if c != a and c in adj[d] and c != d and b != d:
                    paths.append((b, c))
        if len(paths) == 2:
            p1, p2 = paths
            hexa = frozenset([a, p1[0], p1[1], d, p2[0], p2[1]])
            hexagons[hexa] = True

    assert len(hexagons) == 9, f"Expected 9 hexagons, found {len(hexagons)}"
    return list(hexagons.keys())


def hexagon_is_complete(hexa, bond_set):
    """Return True if all C(6,2)=15 site-pairs inside the hexagon are in bond_set."""
    sites = list(hexa)
    for i in range(len(sites)):
        for j in range(i+1, len(sites)):
            if frozenset((sites[i], sites[j])) not in bond_set:
                return False
    return True


def classify_bonds(bonds_all, hexagons, bond_set):
    """
    Classify each bond as 'keep' (in at least one complete hexagon) or 'remove'.
    Returns: keep_bonds, remove_bonds (each a list of (a,b) tuples)
    """
    complete_hexa_bonds = set()
    for hexa in hexagons:
        if hexagon_is_complete(hexa, bond_set):
            sites = list(hexa)
            for i in range(len(sites)):
                for j in range(i+1, len(sites)):
                    complete_hexa_bonds.add(frozenset((sites[i], sites[j])))

    keep, remove = [], []
    for (a, b) in bonds_all:
        fb = frozenset((a, b))
        if fb in complete_hexa_bonds:
            keep.append((a, b))
        else:
            remove.append((a, b))
    return keep, remove


# ─── main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save', default='/scratch/zhouzb79/bfg_gs_results/complete_hex_geometry.png')
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.save), exist_ok=True)

    # Generate PBC cluster once to get the 9 hexagon definitions
    verts_pbc, bnn_pbc, b2nn_pbc, b3nn_pbc, pos_pbc, bset_pbc = get_bonds(3, 3, True, True, True)
    hexagons = find_pbc_hexagons(bnn_pbc, N=27)
    print(f"Reference hexagons (from PBC): {len(hexagons)}")

    # Boundary conditions to study
    bcs = [
        ('OBC', dict(pbc=False, pbc_dim1=False, pbc_dim2=False)),
        ('CYL', dict(pbc=False, pbc_dim1=True,  pbc_dim2=False)),
        ('PBC', dict(pbc=True,  pbc_dim1=True,  pbc_dim2=True)),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("BFG 3×3 Kagome — Complete Hexagons Only\n"
                 "Green=kept (complete hexagon), Red=removed (incomplete hexagon)",
                 fontsize=13)

    summary = {}

    for ax, (bc_label, bc_kwargs) in zip(axes, bcs):
        verts, bnn, b2nn, b3nn, pos, bond_set = get_bonds(3, 3, **bc_kwargs)

        bonds_all = bnn + b2nn + b3nn

        # Check which of the 9 PBC hexagons are complete in this BC
        complete = [h for h in hexagons if hexagon_is_complete(h, bond_set)]
        incomplete = [h for h in hexagons if not hexagon_is_complete(h, bond_set)]

        print(f"\n{bc_label}: {len(complete)} complete / {len(incomplete)} incomplete hexagons")

        keep_bonds, remove_bonds = classify_bonds(bonds_all, hexagons, bond_set)
        print(f"  Total bonds: {len(bonds_all)}  Keep: {len(keep_bonds)}  Remove: {len(remove_bonds)}")
        print(f"  NN keep: {sum(1 for b in keep_bonds if frozenset(b) in set(frozenset(x) for x in bnn))}")
        print(f"  2NN keep: {sum(1 for b in keep_bonds if frozenset(b) in set(frozenset(x) for x in b2nn))}")
        print(f"  3NN keep: {sum(1 for b in keep_bonds if frozenset(b) in set(frozenset(x) for x in b3nn))}")

        summary[bc_label] = dict(complete=len(complete), total_bonds=len(bonds_all),
                                 keep=len(keep_bonds), remove=len(remove_bonds))

        # ─── Plot ───
        ax.set_aspect('equal')
        ax.set_title(f"{bc_label}: {len(complete)}/9 complete hexagons\n"
                     f"Keep {len(keep_bonds)}/{len(bonds_all)} bonds", fontsize=11)

        # Draw removed bonds (red, thin dashed)
        if remove_bonds:
            segs_rm = [[ pos[a], pos[b] ] for (a, b) in remove_bonds]
            lc_rm = LineCollection(segs_rm, colors='#d62728', linewidths=1.2,
                                   linestyles='dashed', alpha=0.55, zorder=1)
            ax.add_collection(lc_rm)

        # Draw kept bonds (green, thick solid)
        if keep_bonds:
            segs_kp = [[ pos[a], pos[b] ] for (a, b) in keep_bonds]
            # Color by bond type
            bnn_set  = set(frozenset(x) for x in bnn)
            b2nn_set = set(frozenset(x) for x in b2nn)
            b3nn_set = set(frozenset(x) for x in b3nn)
            colors_k = []
            for (a, b) in keep_bonds:
                fb = frozenset((a,b))
                if fb in bnn_set:
                    colors_k.append('#2ca02c')      # green: NN
                elif fb in b2nn_set:
                    colors_k.append('#1f77b4')      # blue: 2NN
                else:
                    colors_k.append('#ff7f0e')      # orange: 3NN
            lc_kp = LineCollection(segs_kp, colors=colors_k, linewidths=2.0, zorder=2)
            ax.add_collection(lc_kp)

        # Draw all sites
        xs = [pos[v][0] for v in range(len(pos))]
        ys = [pos[v][1] for v in range(len(pos))]
        ax.scatter(xs, ys, s=60, c='k', zorder=5)
        for v in range(len(pos)):
            ax.annotate(str(v), (pos[v][0], pos[v][1]),
                        textcoords='offset points', xytext=(4, 4), fontsize=7)

        # Shade complete hexagon centers
        for hexa in complete:
            cx = np.mean([pos[s][0] for s in hexa])
            cy = np.mean([pos[s][1] for s in hexa])
            ax.plot(cx, cy, 'g*', markersize=10, zorder=4)

        for hexa in incomplete:
            cx = np.mean([pos[s][0] for s in hexa])
            cy = np.mean([pos[s][1] for s in hexa])
            ax.plot(cx, cy, 'rx', markersize=10, markeredgewidth=2, zorder=4)

        ax.autoscale()
        ax.margins(0.1)
        ax.axis('off')

    # Legend
    legend_handles = [
        mpatches.Patch(color='#2ca02c', label='NN bond (kept)'),
        mpatches.Patch(color='#1f77b4', label='2NN bond (kept)'),
        mpatches.Patch(color='#ff7f0e', label='3NN bond (kept)'),
        mpatches.Patch(color='#d62728', label='Any bond (removed — broken hexagon)'),
        plt.Line2D([0],[0], marker='*', color='g', linewidth=0, markersize=10,
                   label='Complete hexagon center'),
        plt.Line2D([0],[0], marker='x', color='r', linewidth=0, markersize=10,
                   markeredgewidth=2, label='Incomplete hexagon center'),
    ]
    fig.legend(handles=legend_handles, loc='lower center', ncol=3, fontsize=9,
               bbox_to_anchor=(0.5, -0.04))

    plt.tight_layout(rect=[0,0.06,1,1])
    plt.savefig(args.save, dpi=150, bbox_inches='tight')
    print(f"\nGeometry plot saved → {args.save}")

    # ─── Print summary ───
    print("\n=== Summary ===")
    for bc_label, d in summary.items():
        print(f"  {bc_label}: {d['complete']}/9 complete hexagons, "
              f"{d['keep']}/{d['total_bonds']} bonds kept")


if __name__ == '__main__':
    main()
