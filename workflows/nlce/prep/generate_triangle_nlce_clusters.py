#!/usr/bin/env python3
"""
Generate topologically distinct clusters for triangle-based NLCE on the
triangular lattice.

This script implements the triangle-based NLCE where:
- Order 0: Single site (seed)
- Order n: Clusters of n up-pointing triangles connected through shared vertices
- Clusters are identified by PHYSICAL SITE GRAPH ISOMORPHISM
- Two triangle clusters are equivalent iff their induced site+bond graphs are isomorphic

Key insight: Every bond in the triangular lattice belongs to exactly ONE 
up-pointing triangle (and one down-pointing). This ensures proper NLCE counting.

The isomorphism criterion is the physical lattice structure induced by the
triangles, NOT the meta-graph topology. This correctly captures that clusters
with the same meta-graph topology but different site counts are distinct.

Reference cluster counts (triangle-based NLCE):
  Order 0: 1 cluster  (single site)           ΣL = 1
  Order 1: 1 cluster  (single triangle)       ΣL = 1/3
  Order 2: 1 cluster  (two triangles)         ΣL = 1
  Order 3: 3 clusters                         ΣL = 11/3
  Order 4: 5 clusters                         ΣL = 44/3
  Order 5: 12 clusters                        ΣL = 62
  Order 6: 35 clusters                        ΣL = 814/3
  Order 7: 98 clusters                        ΣL = 3652/3
  Order 8: 299 clusters                       ΣL = 5563
"""

import argparse
import numpy as np
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import itertools
import os
from collections import defaultdict
from fractions import Fraction


def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Generate triangle-based NLCE clusters on triangular lattice.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python generate_triangle_nlce_clusters.py --max_order 4
  python generate_triangle_nlce_clusters.py --max_order 6 --visualize --output_dir ./tri_clusters
        """
    )
    parser.add_argument('--max_order', type=int, required=True,
                        help='Maximum order (number of triangles) to generate')
    parser.add_argument('--visualize', action='store_true', default=True,
                        help='Visualize each cluster and save as PNG (default: True)')
    parser.add_argument('--no_visualize', action='store_true',
                        help='Disable visualization')
    parser.add_argument('--lattice_size', type=int, default=0,
                        help='Size of finite lattice (default: max_order + 3)')
    parser.add_argument('--output_dir', type=str, default='.',
                        help='Output directory for cluster information')
    return parser.parse_args()


def create_triangular_lattice_with_triangles(L):
    """
    Create a triangular lattice and identify all up-pointing triangles.
    
    Args:
        L: Number of unit cells in each direction (with PBC)
    
    Returns:
        site_lattice: NetworkX graph of sites
        site_pos: Dictionary mapping site IDs to 2D positions
        triangles: List of up-pointing triangles, each as (v0, v1, v2)
        triangle_pos: Dictionary mapping triangle ID to centroid position
    """
    # Triangular lattice vectors
    a1 = np.array([1.0, 0.0])
    a2 = np.array([0.5, np.sqrt(3)/2])
    
    site_lattice = nx.Graph()
    site_pos = {}
    site_mapping = {}  # (i, j) -> site_id
    
    # Generate sites
    site_id = 0
    for i, j in itertools.product(range(L), repeat=2):
        position = i * a1 + j * a2
        site_pos[site_id] = position
        site_mapping[(i, j)] = site_id
        site_lattice.add_node(site_id, pos=position, ij=(i, j))
        site_id += 1
    
    # Add edges (nearest neighbors)
    for i, j in itertools.product(range(L), repeat=2):
        site = site_mapping[(i, j)]
        neighbors = [
            site_mapping[((i + 1) % L, j)],
            site_mapping[(i, (j + 1) % L)],
            site_mapping[((i + 1) % L, (j - 1 + L) % L)],
        ]
        for neighbor in neighbors:
            site_lattice.add_edge(site, neighbor)
    
    # Identify up-pointing triangles
    # An up-triangle at (i,j) has vertices: (i,j), (i+1,j), (i,j+1)
    triangles = []
    triangle_pos = {}
    
    for i, j in itertools.product(range(L), repeat=2):
        v0 = site_mapping[(i, j)]
        v1 = site_mapping[((i + 1) % L, j)]
        v2 = site_mapping[(i, (j + 1) % L)]
        
        tri_id = len(triangles)
        triangles.append((v0, v1, v2))
        
        # Centroid position
        p0 = site_pos[v0]
        p1 = site_pos[v1]
        p2 = site_pos[v2]
        # Handle PBC wrapping for position calculation
        centroid = (p0 + a1/3 + a2/3)
        triangle_pos[tri_id] = centroid
    
    return site_lattice, site_pos, triangles, triangle_pos, site_mapping


def build_triangle_meta_graph(triangles, L):
    """
    Build the meta-graph where nodes are up-triangles and edges connect
    triangles that share a vertex.
    
    NOTE: We only use this meta-graph for enumeration and connectivity checking.
    The isomorphism criterion is based on the physical site graph, NOT this meta-graph.
    
    Args:
        triangles: List of triangles as (v0, v1, v2)
        L: Lattice size
    
    Returns:
        meta_graph: NetworkX Graph connecting adjacent triangles
        vertex_to_triangles: Dict mapping vertex -> list of (tri_id, vertex_type)
    """
    # Build vertex -> triangle mapping
    vertex_to_triangles = defaultdict(list)
    for tri_id, (v0, v1, v2) in enumerate(triangles):
        vertex_to_triangles[v0].append((tri_id, 0))
        vertex_to_triangles[v1].append((tri_id, 1))
        vertex_to_triangles[v2].append((tri_id, 2))
    
    # Build meta-graph (uncolored - we use site graph isomorphism for equivalence)
    meta_graph = nx.Graph()
    for tri_id in range(len(triangles)):
        meta_graph.add_node(tri_id)
    
    # Store shared vertex info for position computation
    edge_info = {}
    
    # Add edges between triangles sharing vertices
    for vertex, tri_list in vertex_to_triangles.items():
        if len(tri_list) > 1:
            for i, (tri1, vtype1) in enumerate(tri_list):
                for tri2, vtype2 in tri_list[i+1:]:
                    if not meta_graph.has_edge(tri1, tri2):
                        meta_graph.add_edge(tri1, tri2)
                        # Store vertex types for position computation
                        if tri1 < tri2:
                            edge_info[(tri1, tri2)] = (vtype1, vtype2)
                        else:
                            edge_info[(tri2, tri1)] = (vtype2, vtype1)
    
    # Store info as edge attributes
    for (u, v), (c1, c2) in edge_info.items():
        meta_graph[u][v]['vtype'] = (c1, c2)
    
    return meta_graph, vertex_to_triangles


def get_site_graph(triangles, tri_nodes):
    """
    Get the physical site graph induced by a set of triangles.
    
    This is the KEY function for isomorphism - two triangle clusters are equivalent
    iff their site graphs are isomorphic.
    
    Args:
        triangles: List of all triangles
        tri_nodes: Set/frozenset of triangle IDs in the cluster
    
    Returns:
        G: NetworkX graph of physical sites and bonds
    """
    sites = set()
    edges = set()
    for tri_id in tri_nodes:
        v0, v1, v2 = triangles[tri_id]
        sites.update([v0, v1, v2])
        edges.add((min(v0, v1), max(v0, v1)))
        edges.add((min(v1, v2), max(v1, v2)))
        edges.add((min(v0, v2), max(v0, v2)))
    
    G = nx.Graph()
    G.add_nodes_from(sites)
    G.add_edges_from(edges)
    return G


def are_physically_isomorphic(triangles, nodes1, nodes2):
    """
    Check if two triangle clusters have isomorphic site graphs.
    
    This is the CORRECT isomorphism criterion for triangle-based NLCE.
    """
    G1 = get_site_graph(triangles, nodes1)
    G2 = get_site_graph(triangles, nodes2)
    return nx.is_isomorphic(G1, G2)


def graph_hash(site_graph):
    """
    Compute a hash for a site graph that's invariant under node relabeling.
    Used for bucketing before full isomorphism check.
    """
    n = site_graph.number_of_nodes()
    m = site_graph.number_of_edges()
    
    if n == 0:
        return "empty"
    
    # Degree sequence
    degrees = tuple(sorted([d for _, d in site_graph.degree()]))
    
    return (n, m, degrees)


def generate_triangle_clusters(meta_graph, triangles, max_order):
    """
    Generate all topologically distinct triangle clusters up to max_order.
    
    Uses PHYSICAL SITE GRAPH ISOMORPHISM to determine cluster equivalence.
    This is the correct criterion - two triangle clusters are equivalent iff
    their induced site+bond graphs are isomorphic.
    
    Args:
        meta_graph: The meta-graph of triangle connectivity (for enumeration)
        triangles: List of triangles (for site graph construction)
        max_order: Maximum number of triangles in a cluster
    
    Returns:
        distinct_clusters: List of frozensets of triangle IDs
        multiplicities: List of multiplicities L(c)
        orders: List of order (number of triangles) for each cluster
    """
    distinct_clusters = []
    multiplicities = []
    orders = []
    
    N_triangles = meta_graph.number_of_nodes()
    nodes_sorted = sorted(meta_graph.nodes())
    
    print(f"  Meta-graph has {N_triangles} triangles, {meta_graph.number_of_edges()} edges")
    
    for order in range(1, max_order + 1):
        print(f"Generating clusters of order {order} (triangles)...")
        
        # Hash buckets for clustering by isomorphism
        # Key = (n_sites, n_bonds, degree_sequence) of physical site graph
        buckets = defaultdict(list)
        
        for anchor in nodes_sorted:
            # Anchored expansion to avoid duplicates
            start = frozenset([anchor])
            frontier = set(n for n in meta_graph.neighbors(anchor) if n >= anchor)
            visited = set()
            
            stack = [(start, frontier)]
            while stack:
                current, fr = stack.pop()
                
                if current in visited:
                    continue
                visited.add(current)
                
                if len(current) == order:
                    # Get physical site graph for hashing and isomorphism
                    site_graph = get_site_graph(triangles, current)
                    sig = graph_hash(site_graph)
                    
                    # Check isomorphism within bucket using PHYSICAL SITE GRAPH
                    placed = False
                    for idx, (rep_nodes, cnt) in enumerate(buckets[sig]):
                        if are_physically_isomorphic(triangles, current, rep_nodes):
                            buckets[sig][idx] = (rep_nodes, cnt + 1)
                            placed = True
                            break
                    
                    if not placed:
                        buckets[sig].append((current, 1))
                    continue
                
                # Expand frontier
                for nxt in list(fr):
                    new_set = current | {nxt}
                    new_frontier = (fr | set(meta_graph.neighbors(nxt))) - new_set
                    new_frontier = {x for x in new_frontier if x >= anchor}
                    stack.append((new_set, new_frontier))
        
        # Collect representatives
        order_clusters = []
        order_mults = []
        
        for sig_groups in buckets.values():
            for rep_nodes, raw_count in sig_groups:
                # Multiplicity L(c) = raw_count / (3 * N_triangles)
                # Factor of 3 because each site belongs to 3 up-triangles
                L = Fraction(raw_count, 3 * N_triangles)
                order_clusters.append(rep_nodes)
                order_mults.append(L)
        
        # Sort by multiplicity (descending), then by number of sites
        def cluster_sort_key(idx):
            nodes = order_clusters[idx]
            mult = order_mults[idx]
            site_graph = get_site_graph(triangles, nodes)
            return (-float(mult), -site_graph.number_of_nodes())
        
        sorted_indices = sorted(range(len(order_clusters)), key=cluster_sort_key)
        order_clusters = [order_clusters[i] for i in sorted_indices]
        order_mults = [order_mults[i] for i in sorted_indices]
        
        distinct_clusters.extend(order_clusters)
        multiplicities.extend(order_mults)
        orders.extend([order] * len(order_clusters))
        
        # Print summary
        print(f"  Found {len(order_clusters)} distinct clusters of order {order}")
        sum_L = sum(order_mults)
        print(f"  Sum of multiplicities: ΣL = {sum_L} = {float(sum_L):.6f}")
        for idx, (cluster, mult) in enumerate(zip(order_clusters, order_mults)):
            site_graph = get_site_graph(triangles, cluster)
            print(f"    Cluster {idx+1}: {site_graph.number_of_nodes()} sites, "
                  f"{site_graph.number_of_edges()} bonds, L = {mult}")
    
    return distinct_clusters, multiplicities, orders


def get_cluster_sites(triangles, cluster_nodes):
    """
    Get the set of lattice sites covered by a cluster of triangles.
    """
    sites = set()
    for tri_id in cluster_nodes:
        v0, v1, v2 = triangles[tri_id]
        sites.update([v0, v1, v2])
    return sites


def get_cluster_bonds(triangles, cluster_nodes):
    """
    Get the set of unique bonds in a cluster of triangles.
    Returns as sorted list of tuples for consistency.
    """
    bonds = set()
    for tri_id in cluster_nodes:
        v0, v1, v2 = triangles[tri_id]
        bonds.add((min(v0, v1), max(v0, v1)))
        bonds.add((min(v1, v2), max(v1, v2)))
        bonds.add((min(v0, v2), max(v0, v2)))
    return list(bonds)


def identify_subclusters(distinct_clusters, multiplicities, orders, triangles, meta_graph):
    """
    Identify subclusters for each cluster and compute embedding counts.
    Uses physical site graph isomorphism for matching.
    """
    clusters_by_order = defaultdict(list)
    for i, cluster in enumerate(distinct_clusters):
        clusters_by_order[orders[i]].append((i, cluster))
    
    subclusters_info = {}
    
    for i, cluster_nodes in enumerate(distinct_clusters):
        cluster_order = orders[i]
        subclusters_info[i] = []
        
        if cluster_order <= 1:
            continue
        
        for sub_order in range(1, cluster_order):
            candidates = clusters_by_order[sub_order]
            subcluster_counts = defaultdict(int)
            
            # Enumerate all connected sub_order subsets of triangles
            for subset in itertools.combinations(cluster_nodes, sub_order):
                subset = frozenset(subset)
                subset_subgraph = meta_graph.subgraph(subset)
                
                if not nx.is_connected(subset_subgraph):
                    continue
                
                # Find matching candidate using PHYSICAL SITE GRAPH ISOMORPHISM
                for cand_idx, cand_nodes in candidates:
                    if are_physically_isomorphic(triangles, subset, cand_nodes):
                        subcluster_counts[cand_idx] += 1
                        break
            
            for sub_idx, count in subcluster_counts.items():
                subclusters_info[i].append((sub_idx, count))
        
        subclusters_info[i].sort(key=lambda x: orders[x[0]])
    
    return subclusters_info


def save_cluster_info(triangles, site_pos, cluster_nodes, 
                      cluster_id, order, multiplicity, meta_graph, output_dir):
    """
    Save cluster information to a file compatible with the ED pipeline.
    """
    sites = sorted(get_cluster_sites(triangles, cluster_nodes))
    bonds = get_cluster_bonds(triangles, cluster_nodes)
    
    # Compute unwrapped positions
    tri_pos, site_positions = compute_triangle_positions(triangles, site_pos, cluster_nodes, meta_graph)
    
    # Center positions
    all_pos = np.array([site_positions[s] for s in sites])
    center = np.mean(all_pos, axis=0)
    
    # Create mapping from original site IDs to local indices
    site_to_idx = {s: i for i, s in enumerate(sites)}
    
    filename = f"{output_dir}/cluster_{cluster_id}_order_{order}.dat"
    
    with open(filename, 'w') as f:
        f.write(f"# Triangle-based NLCE Cluster\n")
        f.write(f"# Cluster ID: {cluster_id}\n")
        f.write(f"# Order (number of triangles): {order}\n")
        f.write(f"# Multiplicity: {multiplicity} = {float(multiplicity):.6f}\n")
        f.write(f"# Number of vertices: {len(sites)}\n")
        f.write(f"# Number of edges: {len(bonds)}\n")
        f.write(f"# Number of triangles: {len(cluster_nodes)}\n")
        f.write(f"# Lattice type: triangular (triangle-based)\n\n")
        
        f.write("# Vertices (index, x, y, z):\n")
        for i, site in enumerate(sites):
            pos = site_positions[site] - center
            f.write(f"{i}, {pos[0]:.6f}, {pos[1]:.6f}, 0.000000\n")
        
        f.write("\n# Edges (vertex1, vertex2):\n")
        for v1, v2 in bonds:
            f.write(f"{site_to_idx[v1]}, {site_to_idx[v2]}\n")
        
        # Build adjacency matrix
        n = len(sites)
        adj = np.zeros((n, n), dtype=int)
        for v1, v2 in bonds:
            i1, i2 = site_to_idx[v1], site_to_idx[v2]
            adj[i1, i2] = 1
            adj[i2, i1] = 1
        
        f.write("\n# Adjacency Matrix:\n")
        for row in adj:
            f.write(' '.join(map(str, row)) + '\n')
        
        f.write("\n# Triangles (vertex indices in local numbering):\n")
        for tri_id in sorted(cluster_nodes):
            v0, v1, v2 = triangles[tri_id]
            f.write(f"{site_to_idx[v0]}, {site_to_idx[v1]}, {site_to_idx[v2]}\n")


def save_subclusters_info(subclusters_info, distinct_clusters, multiplicities, orders, 
                          triangles, output_dir):
    """
    Save subcluster information for NLCE weight calculation.
    """
    with open(f"{output_dir}/subclusters_info.txt", 'w') as f:
        f.write("# Subclusters information for triangle-based NLCE\n")
        f.write("# Format: Cluster_ID, Order, Multiplicity, Subclusters[(ID, Count), ...]\n\n")
        
        for i, cluster in enumerate(distinct_clusters):
            cluster_id = i + 1
            order = orders[i]
            mult = multiplicities[i]
            
            if order > 0:
                site_graph = get_site_graph(triangles, cluster)
                n_sites = site_graph.number_of_nodes()
                n_bonds = site_graph.number_of_edges()
            else:
                n_sites = 1
                n_bonds = 0
            
            subclusters = subclusters_info.get(i, [])
            sub_str = ", ".join([f"({j+1}, {cnt})" for j, cnt in subclusters])
            
            f.write(f"Cluster {cluster_id} (Order {order}, Sites={n_sites}, Bonds={n_bonds}, L={mult}):\n")
            if subclusters:
                f.write(f"  Subclusters: {sub_str}\n")
            else:
                f.write("  No subclusters\n")
            f.write("\n")


def compute_triangle_positions(triangles, site_pos, cluster_nodes, meta_graph):
    """
    Compute visualization positions for triangles in a cluster,
    properly unwrapping PBC.
    
    Returns:
        tri_pos: Dict mapping tri_id -> centroid position
        site_positions: Dict mapping site_id -> 2D position
    """
    if len(cluster_nodes) == 0:
        return {}, {}
    
    # Triangular lattice vectors
    a1 = np.array([1.0, 0.0])
    a2 = np.array([0.5, np.sqrt(3)/2])
    
    cluster_nodes_list = list(cluster_nodes)
    subgraph = meta_graph.subgraph(cluster_nodes)
    
    # BFS to place triangles consistently
    tri_pos = {}
    site_positions = {}
    visited = set()
    
    # Start from first triangle at origin
    start_tri = cluster_nodes_list[0]
    tri_pos[start_tri] = np.array([0.0, 0.0])
    
    # Place sites of first triangle
    v0, v1, v2 = triangles[start_tri]
    site_positions[v0] = np.array([0.0, 0.0])
    site_positions[v1] = a1.copy()
    site_positions[v2] = a2.copy()
    
    visited.add(start_tri)
    queue = [start_tri]
    
    while queue:
        current_tri = queue.pop(0)
        current_sites = set(triangles[current_tri])
        
        for neighbor_tri in subgraph.neighbors(current_tri):
            if neighbor_tri in visited:
                continue
            
            neighbor_sites = set(triangles[neighbor_tri])
            shared_sites = current_sites & neighbor_sites
            
            if not shared_sites:
                continue
            
            # Find a shared site that's already positioned
            shared_site = None
            for s in shared_sites:
                if s in site_positions:
                    shared_site = s
                    break
            
            if shared_site is None:
                continue
            
            shared_pos = site_positions[shared_site]
            
            # Determine neighbor triangle's vertex types
            v0_n, v1_n, v2_n = triangles[neighbor_tri]
            
            # Find which vertex type the shared site is in the neighbor triangle
            if shared_site == v0_n:
                # Shared at bottom-left of neighbor
                site_positions[v0_n] = shared_pos
                site_positions[v1_n] = shared_pos + a1
                site_positions[v2_n] = shared_pos + a2
            elif shared_site == v1_n:
                # Shared at bottom-right of neighbor
                site_positions[v1_n] = shared_pos
                site_positions[v0_n] = shared_pos - a1
                site_positions[v2_n] = shared_pos - a1 + a2
            else:  # shared_site == v2_n
                # Shared at top of neighbor
                site_positions[v2_n] = shared_pos
                site_positions[v0_n] = shared_pos - a2
                site_positions[v1_n] = shared_pos - a2 + a1
            
            tri_pos[neighbor_tri] = (site_positions[v0_n] + site_positions[v1_n] + 
                                      site_positions[v2_n]) / 3
            
            visited.add(neighbor_tri)
            queue.append(neighbor_tri)
    
    return tri_pos, site_positions


def visualize_cluster(triangles, site_pos, meta_graph, cluster_nodes,
                      cluster_id, order, multiplicity, subclusters_info,
                      all_clusters, all_multiplicities, all_orders, output_dir):
    """
    Visualize a triangle cluster with its subclusters and NLCE formula.
    """
    # Compute positions
    tri_pos, site_positions = compute_triangle_positions(triangles, site_pos, cluster_nodes, meta_graph)
    
    if not site_positions:
        return None
    
    # Center the cluster
    all_pos = np.array(list(site_positions.values()))
    center = np.mean(all_pos, axis=0)
    site_positions = {k: v - center for k, v in site_positions.items()}
    tri_pos = {k: v - center for k, v in tri_pos.items()}
    
    # Get subclusters for this cluster
    cluster_subclusters = subclusters_info.get(cluster_id - 1, [])
    
    # Determine layout
    n_subclusters = len(cluster_subclusters)
    n_cols = min(4, n_subclusters + 1) if n_subclusters > 0 else 1
    n_rows = (n_subclusters + 1 + n_cols - 1) // n_cols if n_subclusters > 0 else 1
    
    fig_width = 4 * n_cols
    fig_height = 4 * n_rows + 1.5
    fig = plt.figure(figsize=(fig_width, fig_height))
    
    # Main cluster plot
    ax = fig.add_subplot(n_rows, n_cols, 1)
    
    # Color palette for triangles
    colors = plt.cm.Set3(np.linspace(0, 1, max(12, len(cluster_nodes))))
    
    # Draw triangles
    for idx, tri_id in enumerate(sorted(cluster_nodes)):
        v0, v1, v2 = triangles[tri_id]
        pts = np.array([site_positions[v0], site_positions[v1], site_positions[v2]])
        triangle = plt.Polygon(pts, alpha=0.4, facecolor=colors[idx % len(colors)],
                              edgecolor='blue', linewidth=2)
        ax.add_patch(triangle)
    
    # Draw sites
    sites = get_cluster_sites(triangles, cluster_nodes)
    site_graph = get_site_graph(triangles, cluster_nodes)
    
    for site in sites:
        pos = site_positions[site]
        ax.scatter(pos[0], pos[1], c='red', s=100, zorder=5, 
                  edgecolors='darkred', linewidths=1.5)
    
    # Draw bonds
    for u, v in site_graph.edges():
        if u in site_positions and v in site_positions:
            p1, p2 = site_positions[u], site_positions[v]
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 'b-', lw=2, alpha=0.7)
    
    n_sites = site_graph.number_of_nodes()
    n_bonds = site_graph.number_of_edges()
    ax.set_title(f'C{cluster_id} ({order} tri, {n_sites} sites, {n_bonds} bonds)\nL={multiplicity}', 
                fontsize=11, fontweight='bold')
    ax.set_aspect('equal')
    ax.axis('off')
    
    # Plot subclusters
    for plot_idx, (sub_idx, count) in enumerate(cluster_subclusters):
        ax_sub = fig.add_subplot(n_rows, n_cols, plot_idx + 2)
        
        sub_nodes = all_clusters[sub_idx]
        sub_mult = all_multiplicities[sub_idx]
        sub_order = all_orders[sub_idx]
        
        sub_tri_pos, sub_site_pos = compute_triangle_positions(triangles, site_pos, sub_nodes, meta_graph)
        
        if sub_site_pos:
            all_pos = np.array(list(sub_site_pos.values()))
            center = np.mean(all_pos, axis=0)
            sub_site_pos = {k: v - center for k, v in sub_site_pos.items()}
            
            for idx, tri_id in enumerate(sorted(sub_nodes)):
                v0, v1, v2 = triangles[tri_id]
                pts = np.array([sub_site_pos[v0], sub_site_pos[v1], sub_site_pos[v2]])
                triangle = plt.Polygon(pts, alpha=0.4, facecolor='lightgreen',
                                      edgecolor='green', linewidth=2)
                ax_sub.add_patch(triangle)
            
            sub_sites = get_cluster_sites(triangles, sub_nodes)
            for site in sub_sites:
                pos = sub_site_pos[site]
                ax_sub.scatter(pos[0], pos[1], c='green', s=80, zorder=5)
        
        sub_site_graph = get_site_graph(triangles, sub_nodes)
        ax_sub.set_title(f'−{count}×C{sub_idx+1} ({sub_order} tri, '
                        f'{sub_site_graph.number_of_nodes()} sites)\nL={sub_mult}',
                        fontsize=10, color='green')
        ax_sub.set_aspect('equal')
        ax_sub.axis('off')
    
    # NLCE formula
    if cluster_subclusters:
        terms = [f"{cnt}·W(C{idx+1})" for idx, cnt in cluster_subclusters]
        formula = f"W(C{cluster_id}) = P(C{cluster_id}) − " + " − ".join(terms)
    else:
        formula = f"W(C{cluster_id}) = P(C{cluster_id})"
    
    fig.text(0.5, 0.02, f"NLCE: {formula}", ha='center', fontsize=10,
             style='italic', bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    plt.tight_layout(rect=[0, 0.06, 1, 1])
    
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.join(output_dir, f'cluster_{cluster_id}_order_{order}.png')
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"    Saved visualization: {filename}")
    return filename


def main():
    args = parse_arguments()
    
    # Handle visualization flag
    do_visualize = args.visualize and not args.no_visualize
    
    L = args.lattice_size if args.lattice_size > 0 else args.max_order + 3
    
    print("="*80)
    print("Triangle-Based NLCE Cluster Generator")
    print("(Using Physical Site Graph Isomorphism)")
    print("="*80)
    print(f"Parameters:")
    print(f"  Maximum order: {args.max_order} triangles")
    print(f"  Lattice size: {L}x{L}")
    print(f"  Output directory: {args.output_dir}")
    print(f"  Visualization: {'enabled' if do_visualize else 'disabled'}")
    print("="*80)
    
    # Create lattice and identify triangles
    print("\nCreating triangular lattice...")
    site_lattice, site_pos, triangles, tri_pos, site_mapping = create_triangular_lattice_with_triangles(L)
    print(f"  {site_lattice.number_of_nodes()} sites, {len(triangles)} up-triangles")
    
    # Build meta-graph
    print("\nBuilding triangle meta-graph...")
    meta_graph, vertex_to_triangles = build_triangle_meta_graph(triangles, L)
    print(f"  Meta-graph: {meta_graph.number_of_nodes()} nodes, {meta_graph.number_of_edges()} edges")
    
    # Handle order 0 (single site) separately
    print("\nAdding order 0 cluster (single site)...")
    order0_cluster = frozenset()  # Empty triangle set
    order0_mult = Fraction(1, 1)
    
    # Generate triangle clusters
    print("\nGenerating triangle clusters...")
    distinct_clusters, multiplicities, orders = generate_triangle_clusters(
        meta_graph, triangles, args.max_order)
    
    # Prepend order 0
    distinct_clusters = [order0_cluster] + distinct_clusters
    multiplicities = [order0_mult] + multiplicities
    orders = [0] + orders
    
    print(f"\nTotal: {len(distinct_clusters)} distinct clusters (including order 0)")
    
    # Identify subclusters
    print("\nIdentifying subclusters...")
    subclusters_info = identify_subclusters(
        distinct_clusters, multiplicities, orders, triangles, meta_graph)
    
    # Create output directory
    output_info_dir = os.path.join(args.output_dir, f'cluster_info_order_{args.max_order}')
    os.makedirs(output_info_dir, exist_ok=True)
    
    # Save cluster info
    print(f"\nSaving cluster information to {output_info_dir}...")
    for i, (cluster_nodes, mult, order) in enumerate(zip(distinct_clusters, multiplicities, orders)):
        if order == 0:
            # Special case for single site
            with open(f"{output_info_dir}/cluster_{i+1}_order_0.dat", 'w') as f:
                f.write("# Triangle-based NLCE Cluster\n")
                f.write(f"# Cluster ID: {i+1}\n")
                f.write("# Order: 0 (single site seed)\n")
                f.write(f"# Multiplicity: 1\n")
                f.write("# Number of vertices: 1\n")
                f.write("# Number of edges: 0\n")
                f.write("# Number of triangles: 0\n")
                f.write("# Lattice type: triangular (triangle-based)\n\n")
                f.write("# Vertices (index, x, y, z):\n")
                f.write("0, 0.000000, 0.000000, 0.000000\n")
                f.write("\n# Edges (vertex1, vertex2):\n")
                f.write("\n# Adjacency Matrix:\n")
                f.write("0\n")
        else:
            save_cluster_info(triangles, site_pos, cluster_nodes, i+1, order, 
                            mult, meta_graph, output_info_dir)
    
    save_subclusters_info(subclusters_info, distinct_clusters, multiplicities, 
                          orders, triangles, output_info_dir)
    
    # Visualize
    if do_visualize:
        print("\nVisualizing clusters...")
        viz_dir = os.path.join(args.output_dir, f'cluster_viz_order_{args.max_order}')
        os.makedirs(viz_dir, exist_ok=True)
        
        for i, (cluster_nodes, mult, order) in enumerate(zip(distinct_clusters, multiplicities, orders)):
            if order == 0:
                continue  # Skip visualization for single site
            visualize_cluster(triangles, site_pos, meta_graph, cluster_nodes,
                            i+1, order, mult, subclusters_info,
                            distinct_clusters, multiplicities, orders, viz_dir)
        print(f"  Saved visualizations to {viz_dir}")
    
    # Summary
    print("\n" + "="*80)
    print("Summary of Triangle-Based NLCE Clusters:")
    print("="*80)
    print(f"{'Order':<8} {'#Clusters':<12} {'ΣL(c)':<20}")
    print("-"*80)
    
    clusters_by_order = defaultdict(list)
    for i, (order, mult) in enumerate(zip(orders, multiplicities)):
        clusters_by_order[order].append((i+1, mult))
    
    for order in sorted(clusters_by_order.keys()):
        clusters = clusters_by_order[order]
        sum_L = sum(m for _, m in clusters)
        print(f"{order:<8} {len(clusters):<12} {str(sum_L):<20} = {float(sum_L):.6f}")
    
    print("="*80)
    print("\nReference values from paper:")
    print("  Order 0: 1 cluster,  ΣL = 1")
    print("  Order 1: 1 cluster,  ΣL = 1/3")
    print("  Order 2: 1 cluster,  ΣL = 1")
    print("  Order 3: 3 clusters, ΣL = 11/3")
    print("  Order 4: 5 clusters, ΣL = 44/3")
    print("  Order 5: 12 clusters, ΣL = 62")
    print("  Order 6: 35 clusters, ΣL = 814/3")
    print("="*80)
    print("Done!")


if __name__ == "__main__":
    main()
