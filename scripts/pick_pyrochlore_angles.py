#!/usr/bin/env python3
"""
Interactive angle picker for pyrochlore geometry plots.

Rotate the 3D views with your mouse. The current elevation and azimuth
are displayed live in the window title.  When you're happy with the angles:

  • Press  ENTER / RETURN  to save & quit
  • Press  ESC              to quit without saving

Saved angles are written to  build/figures/pyrochlore_angles.json
and printed to the terminal so you can paste them into the main script.
"""

import numpy as np
import matplotlib
matplotlib.use('TkAgg')          # need an interactive backend
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import os, sys, json
from importlib.util import spec_from_file_location, module_from_spec

# ── Import helpers ──────────────────────────────────────────────────────
def _import_helper(name, path):
    spec = spec_from_file_location(name, path)
    module = module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

_script_dir = os.path.dirname(os.path.abspath(__file__))
_edlib_dir  = os.path.join(_script_dir, '..', 'python', 'edlib')

helper_pyrochlore = _import_helper(
    'helper_pyrochlore',
    os.path.join(_edlib_dir, 'helper_pyrochlore.py')
)
helper_pyrochlore_super = _import_helper(
    'helper_pyrochlore_super',
    os.path.join(_edlib_dir, 'helper_pyrochlore_super.py')
)

generate_pyrochlore_cluster       = helper_pyrochlore.generate_pyrochlore_cluster
get_sublattice_index              = helper_pyrochlore.get_sublattice_index
generate_pyrochlore_super_cluster = helper_pyrochlore_super.generate_pyrochlore_super_cluster

# ── Drawing helpers (same as main script) ───────────────────────────────
TETRA_FACE_COLOR  = (0.72, 0.90, 1.0)
TETRA_EDGE_COLOR  = (0.3, 0.3, 0.3)
SITE_COLOR        = '#1a1a1a'
BOUNDARY_COLOR    = '#2ECC71'
INTER_BOND_COLOR  = '#888888'


def set_equal_aspect_3d(ax, points):
    pts = np.asarray(points)
    mins = pts.min(axis=0); maxs = pts.max(axis=0)
    c = (mins + maxs) / 2.0
    r = (maxs - mins).max() / 2.0 * 1.1
    ax.set_xlim(c[0]-r, c[0]+r)
    ax.set_ylim(c[1]-r, c[1]+r)
    ax.set_zlim(c[2]-r, c[2]+r)


def find_tetrahedra_from_edges(vertices, edges):
    adj = {v: set() for v in vertices}
    for a, b in edges:
        adj[a].add(b); adj[b].add(a)
    tets = []
    for a in adj:
        for b in [x for x in adj[a] if x > a]:
            cab = adj[a] & adj[b]
            for c in [x for x in cab if x > b]:
                cabc = cab & adj[c]
                for d in [x for x in cabc if x > c]:
                    tets.append((a, b, c, d))
    return tets


def find_collinear_chains(vertices, min_length=4):
    from itertools import combinations
    sites = sorted(vertices.keys())
    seen = set(); chains = []
    for v1, v2 in combinations(sites, 2):
        p1 = np.array(vertices[v1]); p2 = np.array(vertices[v2])
        d = p2 - p1; n = np.linalg.norm(d)
        if n < 0.01: continue
        dh = d / n
        col = [v1, v2]
        for v3 in sites:
            if v3 in (v1, v2): continue
            if np.linalg.norm(np.cross(np.array(vertices[v3]) - p1, dh)) < 0.01:
                col.append(v3)
        if len(col) >= min_length:
            key = tuple(sorted(col))
            if key not in seen:
                seen.add(key)
                projs = sorted([(v, np.dot(np.array(vertices[v]) - p1, dh)) for v in col],
                               key=lambda x: x[1])
                chains.append([p[0] for p in projs])
    return chains, set(v for ch in chains for v in ch)


def draw_geometry(ax, vertices, edges, marker_size=50):
    """Draw the full geometry on axis *ax*."""
    tets = find_tetrahedra_from_edges(vertices, edges)
    intra = set()
    for t in tets:
        for i in range(4):
            for j in range(i+1, 4):
                intra.add(tuple(sorted([t[i], t[j]])))
    inter = set(tuple(sorted(e)) for e in edges) - intra

    chains, _ = find_collinear_chains(vertices)
    chain_edges = set()
    for ch in chains:
        for k in range(len(ch)-1):
            chain_edges.add(tuple(sorted([ch[k], ch[k+1]])))

    # Tetrahedra
    for t in tets:
        coords = np.array([vertices[v] for v in t], dtype=float)
        faces = [[coords[i] for i in f] for f in [(0,1,2),(0,1,3),(0,2,3),(1,2,3)]]
        ax.add_collection3d(Poly3DCollection(
            faces, alpha=0.25, facecolor=TETRA_FACE_COLOR,
            edgecolor=TETRA_EDGE_COLOR, linewidth=0.7))

    # Inter-tet bonds (non-chain)
    for v1, v2 in inter:
        ek = tuple(sorted([v1, v2]))
        if ek in chain_edges: continue
        p1, p2 = np.array(vertices[v1]), np.array(vertices[v2])
        ax.plot([p1[0],p2[0]], [p1[1],p2[1]], [p1[2],p2[2]],
                color=INTER_BOND_COLOR, alpha=0.7, lw=1.2, ls='--', zorder=1)

    # Chain edges (green)
    drawn = set(); first = True
    for ch in chains:
        for k in range(len(ch)-1):
            ek = tuple(sorted([ch[k], ch[k+1]]))
            if ek not in drawn:
                drawn.add(ek)
                p1, p2 = np.array(vertices[ch[k]]), np.array(vertices[ch[k+1]])
                ax.plot([p1[0],p2[0]], [p1[1],p2[1]], [p1[2],p2[2]],
                        color=BOUNDARY_COLOR, alpha=0.9, lw=2.2, zorder=4,
                        label='Artificial 4-site loops' if first else None)
                first = False

    # Sites
    all_pos = np.array([vertices[v] for v in sorted(vertices)])
    ax.scatter(all_pos[:,0], all_pos[:,1], all_pos[:,2],
               s=marker_size, c=SITE_COLOR, marker='o',
               alpha=1.0, edgecolors='white', linewidth=0.5,
               depthshade=True, zorder=5)

    set_equal_aspect_3d(ax, list(vertices.values()))
    ax.set_axis_off()


# ── Main interactive viewer ─────────────────────────────────────────────
def main():
    print("Generating geometries …")
    v_std, e_std, _, _ = generate_pyrochlore_cluster(2, 2, 2, use_pbc=False)
    v_sup, e_sup, _, _, _ = generate_pyrochlore_super_cluster(1, 1, 1, use_pbc=False)

    # Starting angles (current values in the main script)
    angles = {
        'super':    {'elev': 30, 'azim': 45},
        'standard': {'elev': 25, 'azim': 225},
    }

    fig = plt.figure(figsize=(16, 7))
    fig.canvas.manager.set_window_title(
        "Rotate views  |  ENTER = save & quit  |  ESC = quit")

    ax1 = fig.add_subplot(1, 3, 1, projection='3d')
    draw_geometry(ax1, v_sup, e_sup, marker_size=55)
    ax1.view_init(elev=angles['super']['elev'], azim=angles['super']['azim'])
    ax1.set_title(r'(a) Super cluster  $1\times1\times1$', fontweight='bold')

    ax2 = fig.add_subplot(1, 3, (2, 3), projection='3d')
    draw_geometry(ax2, v_std, e_std, marker_size=45)
    ax2.view_init(elev=angles['standard']['elev'], azim=angles['standard']['azim'])
    ax2.set_title(r'(b) Standard  $2\times2\times2$', fontweight='bold')

    plt.tight_layout(pad=0.5, w_pad=1.0)

    # ── Live angle readout via a timer ──────────────────────────────────
    def update_title(event=None):
        e1, a1 = ax1.elev, ax1.azim
        e2, a2 = ax2.elev, ax2.azim
        angles['super']    = {'elev': round(e1, 1), 'azim': round(a1, 1)}
        angles['standard'] = {'elev': round(e2, 1), 'azim': round(a2, 1)}
        fig.canvas.manager.set_window_title(
            f"(a) elev={e1:.1f}  azim={a1:.1f}   │   "
            f"(b) elev={e2:.1f}  azim={a2:.1f}   │   "
            f"ENTER=save  ESC=quit")

    timer = fig.canvas.new_timer(interval=200)   # refresh every 200 ms
    timer.add_callback(update_title)
    timer.start()

    saved = [False]

    def on_key(event):
        if event.key in ('enter', 'return'):
            saved[0] = True
            plt.close(fig)
        elif event.key == 'escape':
            plt.close(fig)

    fig.canvas.mpl_connect('key_press_event', on_key)

    print("\n╔══════════════════════════════════════════════════════════╗")
    print("║  Rotate the plots with your mouse.                     ║")
    print("║  Current angles are shown in the window title bar.     ║")
    print("║                                                        ║")
    print("║  Press ENTER  when you're happy → saves angles & quits ║")
    print("║  Press ESC    to quit without saving                   ║")
    print("╚══════════════════════════════════════════════════════════╝\n")

    plt.show()

    # ── Report ──────────────────────────────────────────────────────────
    print()
    print(f"  (a) Super cluster :  elev = {angles['super']['elev']},  azim = {angles['super']['azim']}")
    print(f"  (b) Standard 2×2×2:  elev = {angles['standard']['elev']},  azim = {angles['standard']['azim']}")

    if saved[0]:
        out_dir = os.path.join(_script_dir, '..', 'build', 'figures')
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, 'pyrochlore_angles.json')
        with open(out_path, 'w') as f:
            json.dump(angles, f, indent=2)
        print(f"\n  ✓ Angles saved to {out_path}")
        print(f"\n  Paste into plot_pyrochlore_geometries.py:")
        print(f"    Super:    elev={angles['super']['elev']}, azim={angles['super']['azim']}")
        print(f"    Standard: elev={angles['standard']['elev']}, azim={angles['standard']['azim']}")
    else:
        print("\n  (not saved)")


if __name__ == '__main__':
    main()
