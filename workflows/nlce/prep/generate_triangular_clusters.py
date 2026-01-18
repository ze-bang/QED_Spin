#!/usr/bin/env python3
"""
Generate topologically distinct clusters on a triangular lattice for
numerical linked cluster expansion (NLCE) calculations.

This script implements the standard site-based NLCE where:
- Clusters are connected subgraphs of the triangular lattice
- Order = number of sites in the cluster
- Multiplicity L(c) = |Emb(c→L)| / (N_site * |Aut(c)|)

The triangular lattice has coordination z=6 (each site has 6 nearest neighbors).

Reference cluster counts (site-based NLCE on triangular lattice):
  Order 1: 1 cluster (single site)
  Order 2: 1 cluster (nearest-neighbor bond)
  Order 3: 2 clusters (triangle, bent chain)
  Order 4: 5 clusters
  Order 5: 13 clusters
  ...
"""

import argparse
import numpy as np
import networkx as nx
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for saving plots
import matplotlib.pyplot as plt
import itertools
import sys
import os
from collections import defaultdict


def extract_cluster_info(lattice, pos, cluster_nodes):
    """
    Extract detailed information about a cluster.
    
    Args:
        lattice: NetworkX graph of triangular lattice (spin sites)
        pos: Dictionary mapping site IDs to 2D positions
        cluster_nodes: List of site IDs in this cluster
    
    Returns:
        Dictionary containing vertices, positions, edges,
        adjacency matrix, and node mapping
    """
    # Create subgraph for this cluster
    subgraph = lattice.subgraph(cluster_nodes)
    
    # Get vertex positions
    vertex_positions = {v: pos[v] for v in cluster_nodes}
    
    # Get edges in the cluster
    edges = list(subgraph.edges())
    
    # Create adjacency matrix
    nodes = sorted(list(cluster_nodes))
    node_to_idx = {node: i for i, node in enumerate(nodes)}
    adj_matrix = np.zeros((len(nodes), len(nodes)), dtype=int)
    
    for u, v in edges:
        adj_matrix[node_to_idx[u], node_to_idx[v]] = 1
        adj_matrix[node_to_idx[v], node_to_idx[u]] = 1
    
    return {
        'vertices': list(cluster_nodes),
        'vertex_positions': vertex_positions,
        'edges': edges,
        'adjacency_matrix': adj_matrix,
        'node_mapping': node_to_idx
    }


def save_cluster_info(cluster_info, cluster_id, order, multiplicity, output_dir='.'):
    """
    Save detailed information about a cluster to a file.
    """
    filename = f"{output_dir}/cluster_{cluster_id}_order_{order}.dat"
    
    with open(filename, 'w') as f:
        f.write(f"# Cluster ID: {cluster_id}\n")
        f.write(f"# Order (number of sites): {order}\n")
        f.write(f"# Multiplicity: {multiplicity}\n")
        f.write(f"# Number of vertices: {len(cluster_info['vertices'])}\n")
        f.write(f"# Number of edges: {len(cluster_info['edges'])}\n")
        f.write(f"# Lattice type: triangular\n\n")
        
        f.write("# Vertices (index, x, y, z):\n")
        for v in cluster_info['vertices']:
            pos = cluster_info['vertex_positions'][v]
            # Store as 2D coordinates (z=0 for triangular lattice)
            f.write(f"{v}, {pos[0]:.6f}, {pos[1]:.6f}, 0.000000\n")
        
        f.write("\n# Edges (vertex1, vertex2):\n")
        for u, v in cluster_info['edges']:
            f.write(f"{u}, {v}\n")
        
        f.write("\n# Adjacency Matrix:\n")
        for row in cluster_info['adjacency_matrix']:
            f.write(' '.join(map(str, row)) + '\n')
        
        f.write("\n# Node Mapping (original_id: matrix_index):\n")
        for node, idx in cluster_info['node_mapping'].items():
            f.write(f"{node}: {idx}\n")


def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Generate topologically distinct clusters on a triangular lattice for NLCE.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python generate_triangular_clusters.py --max_order 4
  python generate_triangular_clusters.py --max_order 3 --visualize --output_dir ./clusters
        """
    )
    parser.add_argument('--max_order', type=int, required=True, 
                        help='Maximum order of clusters to generate (number of sites)')
    parser.add_argument('--visualize', action='store_true', 
                        help='Visualize each cluster and save as PNG')
    parser.add_argument('--lattice_size', type=int, default=0, 
                        help='Size of finite lattice (default: max_order + 2)')
    parser.add_argument('--output_dir', type=str, default='.', 
                        help='Output directory for cluster information')
    parser.add_argument('--no_pbc', action='store_true',
                        help='Disable periodic boundary conditions (not recommended)')
    return parser.parse_args()


def create_triangular_lattice(L, periodic=True):
    """
    Create a triangular lattice of size L×L unit cells with periodic 
    boundary conditions.
    
    The triangular lattice has coordination z=6. Each unit cell has 1 site.
    
    Args:
        L: Number of unit cells in each direction
        periodic: Use periodic boundary conditions (default: True)
    
    Returns:
        G: NetworkX graph representing the lattice (nodes = spin sites)
        pos: Dictionary mapping node IDs to 2D positions
    """
    # Triangular lattice vectors
    a1 = np.array([1.0, 0.0])
    a2 = np.array([0.5, np.sqrt(3)/2])
    
    G = nx.Graph()
    pos = {}
    
    # Generate lattice sites
    site_id = 0
    site_mapping = {}  # (i, j) -> site_id
    
    for i, j in itertools.product(range(L), repeat=2):
        position = i * a1 + j * a2
        pos[site_id] = position
        site_mapping[(i, j)] = site_id
        G.add_node(site_id, pos=position)
        site_id += 1
    
    # Add edges (nearest neighbors) - 6 neighbors per site on triangular lattice
    for i, j in itertools.product(range(L), repeat=2):
        site = site_mapping[(i, j)]
        
        if periodic:
            # 6 nearest neighbors in triangular lattice
            neighbors = [
                site_mapping[((i + 1) % L, j)],           # +a1
                site_mapping[((i - 1 + L) % L, j)],       # -a1
                site_mapping[(i, (j + 1) % L)],           # +a2
                site_mapping[(i, (j - 1 + L) % L)],       # -a2
                site_mapping[((i + 1) % L, (j - 1 + L) % L)],  # +a1-a2
                site_mapping[((i - 1 + L) % L, (j + 1) % L)],  # -a1+a2
            ]
            for neighbor in neighbors:
                if site < neighbor:  # Avoid adding edges twice
                    G.add_edge(site, neighbor)
        else:
            # Open boundary conditions
            if i + 1 < L:
                G.add_edge(site, site_mapping[(i + 1, j)])
            if j + 1 < L:
                G.add_edge(site, site_mapping[(i, j + 1)])
            if i + 1 < L and j > 0:
                G.add_edge(site, site_mapping[(i + 1, j - 1)])
    
    return G, pos


def compute_automorphism_count(G, nodes):
    """
    Compute |Aut(c)|, the number of automorphisms of the cluster graph.
    """
    H = G.subgraph(nodes).copy()
    mapping = {n: i for i, n in enumerate(sorted(H.nodes()))}
    H = nx.relabel_nodes(H, mapping, copy=True)
    
    GM = nx.isomorphism.GraphMatcher(H, H)
    return sum(1 for _ in GM.isomorphisms_iter())


def _wl_hash_subgraph(G, nodes):
    """
    Compute an isomorphism-invariant hash for the induced subgraph.
    """
    H = G.subgraph(nodes).copy()
    mapping = {n: i for i, n in enumerate(sorted(H.nodes()))}
    H = nx.relabel_nodes(H, mapping, copy=True)
    
    try:
        from networkx.algorithms.graph_hashing import weisfeiler_lehman_graph_hash
        return weisfeiler_lehman_graph_hash(H)
    except ImportError:
        pass
    
    try:
        from networkx.algorithms.isomorphism import weisfeiler_lehman_graph_hash
        return weisfeiler_lehman_graph_hash(H)
    except ImportError:
        pass
    
    # Fallback: degree sequence signature
    degs = sorted([d for _, d in H.degree()])
    return f"{len(H)}|{H.number_of_edges()}|{tuple(degs)}"


def generate_clusters(lattice, max_order):
    """
    Generate all topologically distinct site-based clusters up to max_order.
    
    Uses anchored expansion to enumerate all connected subgraphs,
    then groups by isomorphism class.
    
    Args:
        lattice: NetworkX graph of the triangular lattice
        max_order: Maximum cluster size (number of sites)
    
    Returns:
        distinct_clusters: List of cluster representatives (each is list of node IDs)
        multiplicities: List of multiplicities L(c) per site
        all_mult_details: List of dicts with formula terms for each cluster
    """
    distinct_clusters = []
    multiplicities = []
    all_mult_details = []
    N = lattice.number_of_nodes()
    nodes_sorted = sorted(lattice.nodes())
    
    # Check coordination
    degrees = [lattice.degree(n) for n in nodes_sorted]
    if min(degrees) == max(degrees) == 6:
        print(f"  Lattice has {N} sites (all with z=6, PBC working correctly)")
    else:
        print(f"  Lattice has {N} sites (coordination varies: {min(degrees)}-{max(degrees)})")
    
    for order in range(1, max_order + 1):
        print(f"Generating clusters of order {order}...")
        
        if order == 1:
            # Single site: L = 1 (every site is equivalent)
            first_site = nodes_sorted[0]
            distinct_clusters.append([first_site])
            multiplicities.append(1.0)
            all_mult_details.append({
                'raw_count': N,
                'N_sites': N,
                'Aut': 1,
                'L': 1.0
            })
            print(f"  Found 1 distinct cluster of order 1")
            print(f"  Multiplicity formula: L = |Emb(c→L)| / N_sites = raw_count / {N}")
            print(f"    Topology 1: L = {N} / {N} = 1.0000")
            continue
        
        # Hash buckets for clustering by isomorphism
        buckets = defaultdict(list)
        
        for anchor in nodes_sorted:
            # Anchored expansion: only allow nodes >= anchor
            start = frozenset([anchor])
            frontier = set(n for n in lattice.neighbors(anchor) if n >= anchor)
            visited = set()
            
            stack = [(start, frontier)]
            while stack:
                current, fr = stack.pop()
                
                if current in visited:
                    continue
                visited.add(current)
                
                if len(current) == order:
                    sig = _wl_hash_subgraph(lattice, current)
                    
                    # Check isomorphism within same hash bucket
                    placed = False
                    for idx, (rep_nodes, cnt) in enumerate(buckets[sig]):
                        if nx.is_isomorphic(
                            lattice.subgraph(current),
                            lattice.subgraph(rep_nodes)
                        ):
                            buckets[sig][idx] = (rep_nodes, cnt + 1)
                            placed = True
                            break
                    
                    if not placed:
                        buckets[sig].append((current, 1))
                    continue
                
                # Expand frontier
                for nxt in list(fr):
                    new_set = current | {nxt}
                    new_frontier = (fr | set(lattice.neighbors(nxt))) - new_set
                    new_frontier = {x for x in new_frontier if x >= anchor}
                    stack.append((new_set, new_frontier))
        
        # Collect representatives and compute multiplicities
        reps = []
        mults = []
        order_mult_details = []
        
        for sig_groups in buckets.values():
            for rep_nodes, raw_count in sig_groups:
                # L(c) = |Emb(c→L)| / N_sites
                # raw_count = |Emb(c→L)| (number of labeled embeddings)
                L = raw_count / N
                
                reps.append(sorted(rep_nodes))
                mults.append(L)
                order_mult_details.append({
                    'raw_count': raw_count,
                    'N_sites': N,
                    'L': L
                })
        
        # Sort clusters deterministically
        def cluster_sort_key(idx):
            rep = reps[idx]
            mult = mults[idx]
            subgraph = lattice.subgraph(rep)
            node_map = {n: i for i, n in enumerate(sorted(rep))}
            edges = sorted((node_map[u], node_map[v]) if node_map[u] < node_map[v] 
                          else (node_map[v], node_map[u]) for u, v in subgraph.edges())
            return (-mult, -len(edges), tuple(edges))
        
        sorted_indices = sorted(range(len(reps)), key=cluster_sort_key)
        reps = [reps[i] for i in sorted_indices]
        mults = [mults[i] for i in sorted_indices]
        order_mult_details = [order_mult_details[i] for i in sorted_indices]
        
        distinct_clusters.extend(reps)
        multiplicities.extend(mults)
        all_mult_details.extend(order_mult_details)
        
        # Print verbose multiplicity calculations
        print(f"  Found {len(reps)} distinct clusters of order {order}")
        print(f"  Multiplicity formula: L = |Emb(c→L)| / N_sites = raw_count / {N}")
        for idx, details in enumerate(order_mult_details):
            print(f"    Topology {idx+1}: L = {details['raw_count']} / {details['N_sites']} = {details['L']:.4f}")
    
    return distinct_clusters, multiplicities, all_mult_details


def count_embeddings(source_graph, target_graph):
    """
    Count the number of injective graph homomorphisms (embeddings).
    """
    if source_graph.number_of_nodes() > target_graph.number_of_nodes():
        return 0
    
    GM = nx.isomorphism.GraphMatcher(target_graph, source_graph)
    count = 0
    for _ in GM.subgraph_isomorphisms_iter():
        count += 1
    
    return count


def compute_subcluster_multiplicities(cluster_nodes, subcluster_nodes, lattice, verbose=False):
    """
    Compute Y_cs = |Emb(s→c)| / |Aut(s)|
    """
    cluster_subgraph = lattice.subgraph(cluster_nodes).copy()
    subcluster_subgraph = lattice.subgraph(subcluster_nodes).copy()
    
    labeled_embeddings = count_embeddings(subcluster_subgraph, cluster_subgraph)
    aut_s = compute_automorphism_count(lattice, subcluster_nodes)
    Y_cs = labeled_embeddings // aut_s
    
    if verbose:
        return Y_cs, {
            'Emb_s_to_c': labeled_embeddings,
            'Aut_s': aut_s,
            'Y_cs': Y_cs
        }
    return Y_cs


def identify_subclusters(distinct_clusters, lattice):
    """
    Identify all topological subclusters for each distinct cluster 
    and their multiplicities Y_cs.
    """
    clusters_by_order = defaultdict(list)
    for i, cluster in enumerate(distinct_clusters):
        clusters_by_order[len(cluster)].append((i, cluster))
    
    subclusters_info = {}
    
    for i, cluster in enumerate(distinct_clusters):
        cluster_order = len(cluster)
        subclusters_info[i] = []
        
        if cluster_order == 1:
            continue
        
        for order in range(1, cluster_order):
            candidates = clusters_by_order[order]
            subcluster_counts = defaultdict(int)
            
            for subcluster_set in itertools.combinations(cluster, order):
                subgraph = lattice.subgraph(subcluster_set)
                
                if not nx.is_connected(subgraph):
                    continue
                
                for cand_idx, cand_cluster in candidates:
                    cand_subgraph = lattice.subgraph(cand_cluster)
                    if nx.is_isomorphic(subgraph, cand_subgraph):
                        subcluster_counts[cand_idx] += 1
                        break
            
            for subcluster_idx, count in subcluster_counts.items():
                subclusters_info[i].append((subcluster_idx, count))
        
        subclusters_info[i].sort(key=lambda x: len(distinct_clusters[x[0]]))
    
    return subclusters_info


def save_subclusters_info(subclusters_info, distinct_clusters, multiplicities, output_dir):
    """
    Save information about subclusters of each distinct cluster to a file.
    """
    with open(f"{output_dir}/subclusters_info.txt", 'w') as f:
        f.write("# Subclusters information for each topologically distinct cluster\n")
        f.write("# Triangular lattice NLCE (site-based)\n")
        f.write("# Format: Cluster_ID, Order, Multiplicity, Subclusters[(ID, Multiplicity), ...]\n\n")
        
        for i, cluster in enumerate(distinct_clusters):
            cluster_id = i + 1
            order = len(cluster)
            multiplicity = multiplicities[i]
            
            subclusters = subclusters_info.get(i, [])
            subcluster_str = ", ".join([f"({j+1}, {count})" for j, count in subclusters])
            
            f.write(f"Cluster {cluster_id} (Order {order}):\n")
            if subclusters:
                f.write(f"  Subclusters: {subcluster_str}\n")
            else:
                f.write("  No subclusters (order 1 cluster)\n")
            f.write("\n")


def compute_cluster_positions(lattice, cluster_nodes):
    """
    Compute visualization positions for a cluster by unwrapping PBC.
    Uses BFS to place each site relative to its neighbors using triangular lattice vectors.
    """
    if len(cluster_nodes) == 0:
        return {}
    
    # Triangular lattice vectors (6 directions)
    a1 = np.array([1.0, 0.0])
    a2 = np.array([0.5, np.sqrt(3)/2])
    
    # All 6 neighbor directions on triangular lattice
    directions = [
        a1,        # +a1
        -a1,       # -a1
        a2,        # +a2
        -a2,       # -a2
        a1 - a2,   # +a1-a2
        -a1 + a2,  # -a1+a2
    ]
    
    subgraph = lattice.subgraph(cluster_nodes)
    cluster_nodes_list = list(cluster_nodes)
    
    # BFS to assign positions
    pos = {}
    visited = set()
    
    # Start from first node at origin
    start = cluster_nodes_list[0]
    pos[start] = np.array([0.0, 0.0])
    visited.add(start)
    queue = [start]
    
    while queue:
        current = queue.pop(0)
        current_pos = pos[current]
        
        for neighbor in subgraph.neighbors(current):
            if neighbor not in visited:
                # Find the direction that places neighbor at unit distance
                # On triangular lattice, all neighbors are at distance 1
                best_dir = directions[0]
                pos[neighbor] = current_pos + best_dir
                
                # Check if this neighbor has other visited neighbors
                # and adjust position to be consistent
                visited_neighbors = [n for n in subgraph.neighbors(neighbor) if n in visited and n != current]
                if visited_neighbors:
                    # Try all directions and pick one consistent with other neighbors
                    for d in directions:
                        test_pos = current_pos + d
                        consistent = True
                        for vn in visited_neighbors:
                            dist = np.linalg.norm(test_pos - pos[vn])
                            if not (0.9 < dist < 1.1):  # Should be ~1 for neighbors
                                consistent = False
                                break
                        if consistent:
                            pos[neighbor] = test_pos
                            break
                
                visited.add(neighbor)
                queue.append(neighbor)
    
    # Convert to regular dict with tuples
    return {v: (p[0], p[1]) for v, p in pos.items()}


def visualize_cluster(lattice, pos, cluster_nodes, cluster_index, multiplicity, 
                      subclusters_info, all_clusters, all_multiplicities, output_dir='.'):
    """
    Visualize a cluster in 2D using proper triangular lattice geometry.
    Shows the main cluster and its subclusters with NLCE subtraction info.
    
    Args:
        lattice: The full lattice graph
        pos: Position dictionary for lattice nodes
        cluster_nodes: Set of nodes in this cluster
        cluster_index: 1-based index of this cluster
        multiplicity: L(c) for this cluster
        subclusters_info: Dict mapping cluster index to list of (subcluster_idx, count)
        all_clusters: List of all distinct clusters
        all_multiplicities: List of all multiplicities
        output_dir: Directory to save visualization
    """
    # Get subclusters for this cluster (0-based index)
    cluster_subclusters = subclusters_info.get(cluster_index - 1, [])
    
    # Calculate number of subplots needed
    n_subclusters = len(cluster_subclusters)
    n_cols = min(4, n_subclusters + 1) if n_subclusters > 0 else 1
    n_rows = (n_subclusters + 1 + n_cols - 1) // n_cols if n_subclusters > 0 else 1
    
    # Create figure with subplots
    fig_width = 4 * n_cols
    fig_height = 4 * n_rows + 1.5  # Extra space for NLCE formula
    fig = plt.figure(figsize=(fig_width, fig_height))
    
    # Create subgraph for main cluster
    subgraph = lattice.subgraph(cluster_nodes)
    
    # Compute proper positions (unwrapped from PBC)
    cluster_pos = compute_cluster_positions(lattice, cluster_nodes)
    
    # Center the cluster
    if cluster_pos:
        xs_all = [p[0] for p in cluster_pos.values()]
        ys_all = [p[1] for p in cluster_pos.values()]
        cx, cy = np.mean(xs_all), np.mean(ys_all)
        cluster_pos = {v: (cluster_pos[v][0] - cx, cluster_pos[v][1] - cy) for v in cluster_pos}
    
    # Main cluster plot
    ax_main = fig.add_subplot(n_rows, n_cols, 1)
    
    # Draw edges
    for u, v in subgraph.edges():
        ax_main.plot([cluster_pos[u][0], cluster_pos[v][0]],
                [cluster_pos[u][1], cluster_pos[v][1]], 'b-', lw=2, alpha=0.7)
    
    # Draw vertices
    xs = [cluster_pos[v][0] for v in subgraph.nodes()]
    ys = [cluster_pos[v][1] for v in subgraph.nodes()]
    ax_main.scatter(xs, ys, c='red', s=200, zorder=5, edgecolors='darkred', linewidths=2)
    
    # Label vertices with local indices (0, 1, 2, ...)
    for i, v in enumerate(sorted(subgraph.nodes())):
        ax_main.annotate(str(i), (cluster_pos[v][0], cluster_pos[v][1]), fontsize=10, ha='center', va='center',
                   color='white', fontweight='bold')
    
    ax_main.set_title(f'C{cluster_index} (n={len(cluster_nodes)})\nL={multiplicity:.4f}',
                 fontsize=11, fontweight='bold')
    ax_main.set_aspect('equal')
    ax_main.axis('off')
    
    # Plot each subcluster
    for plot_idx, (sub_idx, count) in enumerate(cluster_subclusters):
        ax_sub = fig.add_subplot(n_rows, n_cols, plot_idx + 2)
        
        sub_cluster = all_clusters[sub_idx]
        sub_mult = all_multiplicities[sub_idx]
        sub_order = len(sub_cluster)
        
        # Compute positions for subcluster
        sub_pos = compute_cluster_positions(lattice, sub_cluster)
        
        # Center
        if sub_pos:
            xs_all = [p[0] for p in sub_pos.values()]
            ys_all = [p[1] for p in sub_pos.values()]
            cx, cy = np.mean(xs_all), np.mean(ys_all)
            sub_pos = {v: (sub_pos[v][0] - cx, sub_pos[v][1] - cy) for v in sub_pos}
        
        sub_subgraph = lattice.subgraph(sub_cluster)
        
        # Draw edges
        for u, v in sub_subgraph.edges():
            ax_sub.plot([sub_pos[u][0], sub_pos[v][0]],
                    [sub_pos[u][1], sub_pos[v][1]], 'g-', lw=2, alpha=0.7)
        
        # Draw vertices
        xs = [sub_pos[v][0] for v in sub_subgraph.nodes()]
        ys = [sub_pos[v][1] for v in sub_subgraph.nodes()]
        ax_sub.scatter(xs, ys, c='green', s=150, zorder=5, edgecolors='darkgreen', linewidths=2)
        
        # Label vertices
        for i, v in enumerate(sorted(sub_subgraph.nodes())):
            ax_sub.annotate(str(i), (sub_pos[v][0], sub_pos[v][1]), fontsize=9, ha='center', va='center',
                       color='white', fontweight='bold')
        
        ax_sub.set_title(f'−{count}×C{sub_idx+1} (n={sub_order})\nL={sub_mult:.4f}',
                     fontsize=10, color='green')
        ax_sub.set_aspect('equal')
        ax_sub.axis('off')
    
    # Add NLCE formula at the bottom
    if cluster_subclusters:
        subcluster_terms = [f"{count}·W(C{sub_idx+1})" for sub_idx, count in cluster_subclusters]
        formula = f"W(C{cluster_index}) = P(C{cluster_index}) − " + " − ".join(subcluster_terms)
    else:
        formula = f"W(C{cluster_index}) = P(C{cluster_index})  [no subclusters]"
    
    fig.text(0.5, 0.02, f"NLCE Weight: {formula}", ha='center', fontsize=11, 
             style='italic', bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    plt.tight_layout(rect=[0, 0.06, 1, 1])
    
    # Save figure
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.join(output_dir, f'cluster_{cluster_index}_order_{len(cluster_nodes)}.png')
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    
    return filename


def main():
    args = parse_arguments()
    
    # Determine lattice size
    L = args.lattice_size if args.lattice_size > 0 else args.max_order + 2
    periodic = not args.no_pbc
    
    print("="*80)
    print("Triangular Lattice NLCE Cluster Generator (Site-Based)")
    print("="*80)
    print(f"Parameters:")
    print(f"  Maximum order: {args.max_order}")
    print(f"  Lattice size: {L}x{L}")
    print(f"  Periodic BC: {periodic}")
    print(f"  Output directory: {args.output_dir}")
    print("="*80)
    
    # Create triangular lattice
    print("\nCreating triangular lattice...")
    G, pos = create_triangular_lattice(L, periodic=periodic)
    print(f"  Created lattice with {G.number_of_nodes()} sites and {G.number_of_edges()} edges")
    
    # Generate clusters
    print("\nGenerating topologically distinct clusters...")
    distinct_clusters, multiplicities, mult_details = generate_clusters(G, args.max_order)
    print(f"\nTotal: {len(distinct_clusters)} distinct clusters")
    
    # Identify subclusters
    print("\nIdentifying subclusters and computing multiplicities...")
    subclusters_info = identify_subclusters(distinct_clusters, G)
    
    # Create output directory
    output_info_dir = os.path.join(args.output_dir, f'cluster_info_order_{args.max_order}')
    os.makedirs(output_info_dir, exist_ok=True)
    
    # Save cluster information
    print(f"\nSaving cluster information to {output_info_dir}...")
    for i, (cluster, mult, details) in enumerate(zip(distinct_clusters, multiplicities, mult_details)):
        cluster_info = extract_cluster_info(G, pos, cluster)
        save_cluster_info(cluster_info, i+1, len(cluster), mult, output_info_dir)
    
    # Save subclusters information
    save_subclusters_info(subclusters_info, distinct_clusters, multiplicities, output_info_dir)
    
    # Visualize clusters if requested
    if args.visualize:
        print("\nVisualizing clusters...")
        viz_dir = os.path.join(args.output_dir, f'cluster_visualizations_order_{args.max_order}')
        os.makedirs(viz_dir, exist_ok=True)
        
        for i, (cluster, mult) in enumerate(zip(distinct_clusters, multiplicities)):
            visualize_cluster(G, pos, cluster, i+1, mult, 
                            subclusters_info, distinct_clusters, multiplicities, viz_dir)
        print(f"  Visualizations saved to {viz_dir}")
    
    # Summary
    print("\n" + "="*80)
    print("Summary of Distinct Clusters:")
    print("="*80)
    print(f"{'Order':<8} {'Count':<10} {'Clusters'}")
    print("-"*80)
    
    clusters_by_order = defaultdict(list)
    for i, cluster in enumerate(distinct_clusters):
        clusters_by_order[len(cluster)].append((i+1, multiplicities[i]))
    
    for order in sorted(clusters_by_order.keys()):
        clusters = clusters_by_order[order]
        cluster_strs = [f"C{c[0]}(L={c[1]:.4f})" for c in clusters]
        print(f"{order:<8} {len(clusters):<10} {', '.join(cluster_strs)}")
    
    print("="*80)
    print("Done!")


if __name__ == "__main__":
    main()
