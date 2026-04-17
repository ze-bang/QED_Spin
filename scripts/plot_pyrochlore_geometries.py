#!/usr/bin/env python3
"""
Publication-ready visualization of pyrochlore lattice geometries.

Generates side-by-side comparison of:
1. 2×2×2 standard pyrochlore lattice (helper_pyrochlore.py)
2. 1×1×1 pyrochlore super lattice (helper_pyrochlore_super.py)

Output formats: PNG, PDF, EPS for publication use.
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.ticker import NullFormatter
import os
import sys
import json
from importlib.util import spec_from_file_location, module_from_spec

# Direct imports to avoid pynauty dependency in __init__.py
def _import_helper(name, path):
    """Import a module directly from file path."""
    spec = spec_from_file_location(name, path)
    module = module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

_script_dir = os.path.dirname(os.path.abspath(__file__))
_edlib_dir = os.path.join(_script_dir, '..', 'python', 'edlib')

helper_pyrochlore = _import_helper(
    'helper_pyrochlore',
    os.path.join(_edlib_dir, 'helper_pyrochlore.py')
)
helper_pyrochlore_super = _import_helper(
    'helper_pyrochlore_super', 
    os.path.join(_edlib_dir, 'helper_pyrochlore_super.py')
)

generate_pyrochlore_cluster = helper_pyrochlore.generate_pyrochlore_cluster
get_sublattice_index = helper_pyrochlore.get_sublattice_index
generate_pyrochlore_super_cluster = helper_pyrochlore_super.generate_pyrochlore_super_cluster

# ============================================================================
# Publication-quality matplotlib configuration
# ============================================================================
def setup_publication_style():
    """Configure matplotlib for publication-quality figures."""
    mpl.rcParams.update({
        # Font settings
        'font.family': 'serif',
        'font.serif': ['Computer Modern Roman', 'Times New Roman', 'DejaVu Serif'],
        'font.size': 10,
        'mathtext.fontset': 'cm',
        
        # Axes settings
        'axes.labelsize': 11,
        'axes.titlesize': 12,
        'axes.titleweight': 'bold',
        'axes.linewidth': 1.0,
        
        # Tick settings
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'xtick.major.width': 0.8,
        'ytick.major.width': 0.8,
        'xtick.major.size': 4,
        'ytick.major.size': 4,
        
        # Legend settings
        'legend.fontsize': 9,
        'legend.framealpha': 0.95,
        'legend.edgecolor': 'black',
        'legend.fancybox': False,
        
        # Figure settings
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.02,
        
        # Line settings
        'lines.linewidth': 1.0,
        'lines.markersize': 6,
    })

# ============================================================================
# Color palettes
# ============================================================================
# Tetrahedra colours
TETRA_FACE_COLOR = (0.45, 0.65, 0.85)  # Medium steel-blue (darker for depth cue)
TETRA_EDGE_COLOR = (0.25, 0.25, 0.25)  # Dark gray edges on tetrahedra

# Site colors
SITE_COLOR = '#1a1a1a'            # Dark gray/black for bulk sites
BOUNDARY_SITE_COLOR = '#2ECC71'   # Green for shared/boundary sites

# Bond colors
INTER_BOND_COLOR = '#888888'      # Gray for inter-tetrahedron bonds

# ============================================================================
# Helper functions
# ============================================================================
def set_equal_aspect_3d(ax, points):
    """Set equal aspect ratio for 3D axes."""
    pts = np.asarray(points)
    mins = pts.min(axis=0)
    maxs = pts.max(axis=0)
    centers = (mins + maxs) / 2.0
    max_range = (maxs - mins).max() / 2.0 * 1.1  # 10% padding
    
    ax.set_xlim(centers[0] - max_range, centers[0] + max_range)
    ax.set_ylim(centers[1] - max_range, centers[1] + max_range)
    ax.set_zlim(centers[2] - max_range, centers[2] + max_range)


def find_tetrahedra_from_edges(vertices, edges):
    """Find all tetrahedra (4-cliques) in the nearest-neighbor graph."""
    # Build adjacency list
    adj = {v: set() for v in vertices}
    for a, b in edges:
        adj[a].add(b)
        adj[b].add(a)
    
    # Find 4-cliques (tetrahedra)
    tetra_list = []
    for a in adj:
        for b in [x for x in adj[a] if x > a]:
            common_ab = adj[a].intersection(adj[b])
            for c in [x for x in common_ab if x > b]:
                common_abc = common_ab.intersection(adj[c])
                for d in [x for x in common_abc if x > c]:
                    tetra_list.append((a, b, c, d))
    
    return tetra_list


def plot_tetrahedron(ax, coords, alpha=0.30, edge_color=TETRA_EDGE_COLOR, edge_width=0.8):
    """
    Plot a single tetrahedron with dark-ish faces and thin dark edges.
    """
    faces = [
        [coords[0], coords[1], coords[2]],
        [coords[0], coords[1], coords[3]],
        [coords[0], coords[2], coords[3]],
        [coords[1], coords[2], coords[3]],
    ]
    poly = Poly3DCollection(faces, alpha=alpha, facecolor=TETRA_FACE_COLOR,
                            edgecolor=edge_color, linewidth=edge_width)
    ax.add_collection3d(poly)


def find_collinear_chains(vertices, min_length=4):
    """
    Find all collinear chains of sites with at least min_length sites.
    
    Returns:
        chains: list of lists, each containing vertex IDs sorted along the line
        chain_sites: set of all vertex IDs that belong to any chain
    """
    from itertools import combinations
    
    all_sites = sorted(vertices.keys())
    lines_found = set()
    chains = []
    
    for v1, v2 in combinations(all_sites, 2):
        p1 = np.array(vertices[v1])
        p2 = np.array(vertices[v2])
        d = p2 - p1
        norm = np.linalg.norm(d)
        if norm < 0.01:
            continue
        d_hat = d / norm
        
        collinear = [v1, v2]
        for v3 in all_sites:
            if v3 in (v1, v2):
                continue
            p3 = np.array(vertices[v3])
            cross = np.linalg.norm(np.cross(p3 - p1, d_hat))
            if cross < 0.01:
                collinear.append(v3)
        
        if len(collinear) >= min_length:
            key = tuple(sorted(collinear))
            if key not in lines_found:
                lines_found.add(key)
                # Sort by projection along line direction
                projs = sorted(
                    [(v, np.dot(np.array(vertices[v]) - p1, d_hat)) for v in collinear],
                    key=lambda x: x[1]
                )
                chains.append([p[0] for p in projs])
    
    chain_sites = set()
    for chain in chains:
        chain_sites.update(chain)
    
    return chains, chain_sites


def plot_pyrochlore_geometry(ax, vertices, edges, sublattice_indices=None,
                             title=None, show_labels=False, show_legend=False,
                             marker_size=50, elev=22, azim=135):
    """
    Create a publication-quality 3D plot of a pyrochlore lattice.
    
    - Light blue tetrahedra faces with thin dark edges
    - Inter-tetrahedron bonds drawn as dashed gray lines
    - Collinear chains of 4 sites (boundary edges) highlighted green
    - All other sites in dark gray
    - No axis lines, labels, grid, or panes
    """
    # ── Topology analysis ──
    adj = {v: set() for v in vertices}
    for a, b in edges:
        adj[a].add(b)
        adj[b].add(a)
    
    tetra_list = find_tetrahedra_from_edges(vertices, edges)
    
    # Separate intra-tetrahedron edges from inter-tetrahedron edges
    intra_edges = set()
    for tet in tetra_list:
        for i in range(4):
            for j in range(i + 1, 4):
                intra_edges.add(tuple(sorted([tet[i], tet[j]])))
    inter_edges = set(tuple(sorted(e)) for e in edges) - intra_edges
    
    # Find collinear chains of 4 sites (boundary edges of the lattice)
    chains, chain_sites = find_collinear_chains(vertices, min_length=4)
    
    # Build set of edges along chains (consecutive pairs in each chain)
    chain_edge_set = set()
    for chain in chains:
        for k in range(len(chain) - 1):
            chain_edge_set.add(tuple(sorted([chain[k], chain[k + 1]])))
    
    # ── Draw tetrahedra ──
    for tetra in tetra_list:
        coords = np.array([vertices[v] for v in tetra], dtype=float)
        plot_tetrahedron(ax, coords, alpha=0.40, edge_color=TETRA_EDGE_COLOR, edge_width=0.7)
    
    # ── Draw inter-tetrahedron bonds (non-chain ones) ──
    for v1, v2 in inter_edges:
        edge_key = tuple(sorted([v1, v2]))
        if edge_key in chain_edge_set:
            continue  # will be drawn as green chain edge
        p1 = np.array(vertices[v1])
        p2 = np.array(vertices[v2])
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]],
                color=INTER_BOND_COLOR, alpha=0.7, linewidth=1.2,
                linestyle='--', zorder=1)
    
    # ── Draw sites ──
    # All sites in uniform dark color
    all_pos = np.array([vertices[v] for v in sorted(vertices)])
    ax.scatter(all_pos[:, 0], all_pos[:, 1], all_pos[:, 2],
               s=marker_size, c=SITE_COLOR, marker='o',
               alpha=1.0, edgecolors='white', linewidth=0.5,
               depthshade=True, zorder=5)
    
    # ── Draw chain edges in green (the straight-line 4-site loops) ──
    chain_edge_drawn = set()
    first_chain = True
    for chain in chains:
        for k in range(len(chain) - 1):
            edge_key = tuple(sorted([chain[k], chain[k + 1]]))
            if edge_key not in chain_edge_drawn:
                chain_edge_drawn.add(edge_key)
                p1 = np.array(vertices[chain[k]])
                p2 = np.array(vertices[chain[k + 1]])
                ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]],
                        color=BOUNDARY_SITE_COLOR, alpha=0.9, linewidth=2.2,
                        linestyle='-', zorder=4,
                        label='Artificial 4-site loops' if first_chain else None)
                first_chain = False
    
    # Add vertex labels if requested
    if show_labels:
        for v_id, pos in vertices.items():
            ax.text(pos[0], pos[1], pos[2], str(v_id), fontsize=6,
                    ha='center', va='center', zorder=10)
    
    # ── Aspect & view ──
    set_equal_aspect_3d(ax, list(vertices.values()))
    ax.view_init(elev=elev, azim=azim)
    
    # ── Strip all axes, grid, panes ──
    ax.set_axis_off()
    
    # Legend
    if show_legend and chains:
        leg = ax.legend(loc='upper left', frameon=True, fancybox=False,
                        shadow=False, framealpha=0.95, edgecolor='black',
                        borderpad=0.4, handlelength=1.2, handletextpad=0.4,
                        labelspacing=0.3, markerscale=0.8)
        leg.get_frame().set_linewidth(0.5)


def create_comparison_figure(output_dir=None, show=True):
    """
    Create a side-by-side comparison of the two pyrochlore geometries.
    
    Left panel: 2×2×2 standard pyrochlore
    Right panel: 1×1×1 pyrochlore super cluster
    """
    setup_publication_style()
    
    # Generate geometries
    print("Generating 2×2×2 standard pyrochlore lattice...")
    vertices_std, edges_std, _, node_mapping_std = generate_pyrochlore_cluster(2, 2, 2, use_pbc=False)
    sublattice_std = {v: get_sublattice_index(v) for v in vertices_std}
    
    print("Generating 1×1×1 pyrochlore super cluster...")
    vertices_super, edges_super, _, node_mapping_super, vertex_to_cell = generate_pyrochlore_super_cluster(1, 1, 1, use_pbc=False)
    sublattice_super = {v: vertex_to_cell[v][4] for v in vertices_super}  # site_idx from (i,j,k,tet,site)
    
    # Print statistics
    print(f"\nStandard pyrochlore 2×2×2:")
    print(f"  Sites: {len(vertices_std)}")
    print(f"  Bonds: {len(edges_std)}")
    print(f"  Tetrahedra: {len(find_tetrahedra_from_edges(vertices_std, edges_std))}")
    
    print(f"\nPyrochlore super 1×1×1:")
    print(f"  Sites: {len(vertices_super)}")
    print(f"  Bonds: {len(edges_super)}")
    print(f"  Tetrahedra: {len(find_tetrahedra_from_edges(vertices_super, edges_super))}")
    
    # Load saved angles from JSON if available, else use defaults
    angles_file = os.path.join(os.path.dirname(__file__), '..', 'build', 'figures', 'pyrochlore_angles.json')
    super_elev, super_azim = 30, 45
    std_elev, std_azim = 25, 225
    if os.path.exists(angles_file):
        with open(angles_file) as f:
            saved = json.load(f)
        super_elev = saved.get('super', {}).get('elev', super_elev)
        super_azim = saved.get('super', {}).get('azim', super_azim)
        std_elev   = saved.get('standard', {}).get('elev', std_elev)
        std_azim   = saved.get('standard', {}).get('azim', std_azim)
        print(f"\nUsing saved angles from {angles_file}")
        print(f"  Super:    elev={super_elev}, azim={super_azim}")
        print(f"  Standard: elev={std_elev},  azim={std_azim}")
    
    # Create figure with asymmetric subplots: (a) super cluster smaller, (b) 2×2×2 larger
    fig = plt.figure(figsize=(16, 7))
    
    # (a) Left panel: super pyrochlore (smaller)
    ax1 = fig.add_subplot(1, 3, 1, projection='3d')
    plot_pyrochlore_geometry(
        ax1, vertices_super, edges_super, sublattice_super,
        show_legend=True, marker_size=55, elev=super_elev, azim=super_azim
    )
    
    # (b) Right panel: standard pyrochlore (larger, spans 2 columns)
    ax2 = fig.add_subplot(1, 3, (2, 3), projection='3d')
    plot_pyrochlore_geometry(
        ax2, vertices_std, edges_std, sublattice_std,
        show_legend=False, marker_size=45, elev=std_elev, azim=std_azim
    )
    
    # Tighten margins
    fig.subplots_adjust(left=-0.05, right=1.05, bottom=-0.05, top=0.92, wspace=-0.20)
    
    # Place titles at the same height using fig.text (in figure coordinates)
    title_y = 0.94
    # Centre of each axes in figure x-coordinates
    bb1 = ax1.get_position()
    bb2 = ax2.get_position()
    fig.text((bb1.x0 + bb1.x1) / 2, title_y,
             r'(a) Pyrochlore super cluster ($1\times1\times1$)',
             ha='center', va='top', fontsize=12, fontweight='bold')
    fig.text((bb2.x0 + bb2.x1) / 2, title_y,
             r'(b) Pyrochlore lattice ($2\times2\times2$ unit cells)',
             ha='center', va='top', fontsize=12, fontweight='bold')
    
    # Save figures
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(__file__), '..', 'build', 'figures')
    
    os.makedirs(output_dir, exist_ok=True)
    
    base_name = os.path.join(output_dir, 'pyrochlore_geometry_comparison')
    
    # Save in multiple formats
    plt.savefig(f'{base_name}.png', dpi=300, bbox_inches='tight', pad_inches=0.0)
    plt.savefig(f'{base_name}.pdf', bbox_inches='tight', pad_inches=0.0)
    plt.savefig(f'{base_name}.svg', bbox_inches='tight', pad_inches=0.0)
    plt.savefig(f'{base_name}.eps', bbox_inches='tight', pad_inches=0.0)
    
    print(f"\nFigures saved to:")
    print(f"  {base_name}.png")
    print(f"  {base_name}.pdf")
    print(f"  {base_name}.svg  (← best for Inkscape)")
    print(f"  {base_name}.eps")
    
    if show:
        plt.show()
    else:
        plt.close(fig)
    
    return fig


def create_individual_figures(output_dir=None, show=True):
    """Create individual figures for each geometry (single-column width)."""
    setup_publication_style()
    
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(__file__), '..', 'build', 'figures')
    
    os.makedirs(output_dir, exist_ok=True)
    
    # ========================
    # Standard Pyrochlore 2×2×2
    # ========================
    print("Generating 2×2×2 standard pyrochlore lattice...")
    vertices_std, edges_std, _, _ = generate_pyrochlore_cluster(2, 2, 2, use_pbc=False)
    sublattice_std = {v: get_sublattice_index(v) for v in vertices_std}
    
    fig1 = plt.figure(figsize=(6.5, 6))
    ax1 = fig1.add_subplot(111, projection='3d')
    plot_pyrochlore_geometry(
        ax1, vertices_std, edges_std, sublattice_std,
        title=r'Pyrochlore lattice ($2\times2\times2$)',
        show_legend=True, marker_size=40, elev=35, azim=45
    )
    plt.tight_layout(pad=0.5)
    
    base1 = os.path.join(output_dir, 'pyrochlore_standard_2x2x2')
    plt.savefig(f'{base1}.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{base1}.pdf', bbox_inches='tight')
    print(f"Saved: {base1}.png, {base1}.pdf")
    plt.close(fig1)
    
    # ========================
    # Super Pyrochlore 1×1×1
    # ========================
    print("Generating 1×1×1 pyrochlore super cluster...")
    vertices_super, edges_super, _, _, vertex_to_cell = generate_pyrochlore_super_cluster(1, 1, 1, use_pbc=False)
    sublattice_super = {v: vertex_to_cell[v][4] for v in vertices_super}
    
    fig2 = plt.figure(figsize=(6.5, 6))
    ax2 = fig2.add_subplot(111, projection='3d')
    plot_pyrochlore_geometry(
        ax2, vertices_super, edges_super, sublattice_super,
        title=r'Pyrochlore super cluster ($1\times1\times1$)',
        show_legend=True, marker_size=50, elev=35, azim=45
    )
    plt.tight_layout(pad=0.5)
    
    base2 = os.path.join(output_dir, 'pyrochlore_super_1x1x1')
    plt.savefig(f'{base2}.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{base2}.pdf', bbox_inches='tight')
    print(f"Saved: {base2}.png, {base2}.pdf")
    plt.close(fig2)


def create_annotated_figure(output_dir=None, show=True):
    """
    Create a detailed annotated figure showing the pyrochlore structure
    with lattice vectors and tetrahedron orientations labeled.
    """
    setup_publication_style()
    
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(__file__), '..', 'build', 'figures')
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate 1×1×1 super cluster for detailed view
    vertices, edges, _, _, vertex_to_cell = generate_pyrochlore_super_cluster(1, 1, 1, use_pbc=False)
    sublattice = {v: vertex_to_cell[v][4] for v in vertices}
    
    fig = plt.figure(figsize=(8, 7.5))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the geometry
    plot_pyrochlore_geometry(
        ax, vertices, edges, sublattice,
        title=r'Pyrochlore lattice structure',
        show_legend=True, marker_size=60, elev=35, azim=45
    )
    
    # Add lattice vector arrows
    origin = np.array([0, 0, 0])
    lattice_vecs = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ])
    labels = [r'$\mathbf{a}_1$', r'$\mathbf{a}_2$', r'$\mathbf{a}_3$']
    arrow_colors = ['#D32F2F', '#388E3C', '#1976D2']
    
    for vec, label, color in zip(lattice_vecs, labels, arrow_colors):
        ax.quiver(origin[0], origin[1], origin[2],
                  vec[0], vec[1], vec[2],
                  arrow_length_ratio=0.12, color=color,
                  linewidth=2.0, alpha=0.9)
        end_pos = origin + vec * 1.08
        ax.text(end_pos[0], end_pos[1], end_pos[2], label,
                fontsize=11, fontweight='bold', color=color,
                ha='center', va='center')
    
    # Add tetrahedron orientation labels
    # Find center of an up-tetrahedron and down-tetrahedron
    tetra_list = find_tetrahedra_from_edges(vertices, edges)
    
    up_tetra = None
    down_tetra = None
    for tetra in tetra_list:
        coords = np.array([vertices[v] for v in tetra])
        v0, v1, v2, v3 = coords
        volume = np.dot(v0 - v3, np.cross(v1 - v3, v2 - v3)) / 6.0
        if volume > 0 and up_tetra is None:
            up_tetra = coords.mean(axis=0)
        elif volume < 0 and down_tetra is None:
            down_tetra = coords.mean(axis=0)
        if up_tetra is not None and down_tetra is not None:
            break
    
    plt.tight_layout(pad=0.8)
    
    base = os.path.join(output_dir, 'pyrochlore_annotated')
    plt.savefig(f'{base}.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{base}.pdf', bbox_inches='tight')
    print(f"Saved: {base}.png, {base}.pdf")
    
    if show:
        plt.show()
    else:
        plt.close(fig)


# ============================================================================
# Main execution
# ============================================================================
if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Generate publication-ready pyrochlore lattice visualizations',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python plot_pyrochlore_geometries.py                    # Side-by-side comparison
  python plot_pyrochlore_geometries.py --individual      # Individual figures
  python plot_pyrochlore_geometries.py --annotated       # Annotated figure
  python plot_pyrochlore_geometries.py --all             # All figures
  python plot_pyrochlore_geometries.py -o ./figures      # Custom output directory
        """
    )
    
    parser.add_argument('-o', '--output', type=str, default=None,
                        help='Output directory for figures')
    parser.add_argument('--individual', action='store_true',
                        help='Create individual figures for each geometry')
    parser.add_argument('--annotated', action='store_true',
                        help='Create annotated figure with lattice vectors')
    parser.add_argument('--all', action='store_true',
                        help='Create all figure types')
    parser.add_argument('--no-show', action='store_true',
                        help='Do not display figures (just save)')
    
    args = parser.parse_args()
    
    # Prevent display if requested
    if args.no_show:
        mpl.use('Agg')
    
    show = not args.no_show
    
    if args.all:
        create_comparison_figure(args.output, show=show)
        create_individual_figures(args.output, show=show)
        create_annotated_figure(args.output, show=show)
    elif args.individual:
        create_individual_figures(args.output, show=show)
    elif args.annotated:
        create_annotated_figure(args.output, show=show)
    else:
        create_comparison_figure(args.output, show=show)
