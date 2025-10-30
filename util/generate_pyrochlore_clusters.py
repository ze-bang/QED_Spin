import argparse
import numpy as np
import networkx as nx
from mpl_toolkits.mplot3d import Axes3D
import itertools
import sys
from collections import defaultdict
import os
import collections
import pickle
import hashlib

#!/usr/bin/env python3
"""
Generate topologically distinct clusters on a pyrochlore lattice for
numerical linked cluster expansion (NLCE) calculations.
"""

import matplotlib.pyplot as plt

def get_cache_key(max_order, lattice_size):
    """Generate a unique cache key based on computation parameters."""
    key_str = f"pyrochlore_clusters_order_{max_order}_lattice_{lattice_size}"
    return hashlib.md5(key_str.encode()).hexdigest()

def get_cache_path(cache_dir, max_order, lattice_size):
    """Get the path to the cache file."""
    cache_key = get_cache_key(max_order, lattice_size)
    return os.path.join(cache_dir, f"clusters_cache_{cache_key}.pkl")

def save_cache(cache_path, data):
    """Save computation results to cache file."""
    cache_dir = os.path.dirname(cache_path)
    os.makedirs(cache_dir, exist_ok=True)
    
    with open(cache_path, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    print(f"Cache saved to {cache_path}")

def load_cache(cache_path):
    """Load computation results from cache file."""
    if not os.path.exists(cache_path):
        return None
    
    try:
        with open(cache_path, 'rb') as f:
            data = pickle.load(f)
        print(f"Cache loaded from {cache_path}")
        return data
    except (pickle.UnpicklingError, EOFError, OSError) as e:
        print(f"Warning: Failed to load cache from {cache_path}: {e}")
        return None

def verify_cache_validity(cache_data, max_order):
    """Verify that cached data is complete and valid for the requested max_order."""
    if cache_data is None:
        return False
    
    required_keys = ['distinct_clusters', 'multiplicities', 'lattice', 'pos', 
                     'tetrahedra', 'tet_graph', 'max_order', 'lattice_size']
    
    # Check all required keys exist
    if not all(key in cache_data for key in required_keys):
        print("Cache invalid: missing required keys")
        return False
    
    # Check that cache covers the requested max_order
    if cache_data['max_order'] < max_order:
        print(f"Cache invalid: cached max_order {cache_data['max_order']} < requested {max_order}")
        return False
    
    return True

def extract_cluster_info(lattice, pos, tetrahedra, cluster):
    """
    Extract detailed information about a cluster.
    """
    # Get all vertices in the cluster
    vertices = set()
    for tet_idx in cluster:
        vertices.update(tetrahedra[tet_idx])
    
    # Create subgraph for this cluster
    subgraph = lattice.subgraph(vertices)
    
    # Get vertex positions
    vertex_positions = {v: pos[v] for v in vertices}
    
    # Get edges in the cluster
    edges = list(subgraph.edges())
    
    # Get the tetrahedra that make up the cluster
    cluster_tetrahedra = [tetrahedra[tet_idx] for tet_idx in cluster]
    
    # Create adjacency matrix
    nodes = sorted(list(vertices))
    node_to_idx = {node: i for i, node in enumerate(nodes)}
    adj_matrix = np.zeros((len(nodes), len(nodes)), dtype=int)
    
    for u, v in edges:
        adj_matrix[node_to_idx[u], node_to_idx[v]] = 1
        adj_matrix[node_to_idx[v], node_to_idx[u]] = 1
    
    return {
        'vertices': list(vertices),
        'vertex_positions': vertex_positions,
        'edges': edges,
        'tetrahedra': cluster_tetrahedra,
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
        f.write(f"# Order (number of tetrahedra): {order}\n")
        f.write(f"# Multiplicity: {multiplicity}\n")
        f.write(f"# Number of vertices: {len(cluster_info['vertices'])}\n")
        f.write(f"# Number of edges: {len(cluster_info['edges'])}\n\n")
        
        f.write("# Vertices (index, x, y, z):\n")
        for v in cluster_info['vertices']:
            pos = cluster_info['vertex_positions'][v]
            f.write(f"{v}, {pos[0]:.6f}, {pos[1]:.6f}, {pos[2]:.6f}\n")
        
        f.write("\n# Edges (vertex1, vertex2):\n")
        for u, v in cluster_info['edges']:
            f.write(f"{u}, {v}\n")
        
        f.write("\n# Tetrahedra (vertex1, vertex2, vertex3, vertex4):\n")
        for tet in cluster_info['tetrahedra']:
            f.write(f"{', '.join(map(str, tet))}\n")
        
        f.write("\n# Adjacency Matrix:\n")
        for row in cluster_info['adjacency_matrix']:
            f.write(' '.join(map(str, row)) + '\n')
        
        f.write("\n# Node Mapping (original_id: matrix_index):\n")
        for node, idx in cluster_info['node_mapping'].items():
            f.write(f"{node}: {idx}\n")

def parse_arguments():
    parser = argparse.ArgumentParser(description='Generate topologically distinct clusters on a pyrochlore lattice.')
    parser.add_argument('--max_order', type=int, required=True, help='Maximum order of clusters to generate')
    parser.add_argument('--visualize', action='store_true', help='Visualize each cluster')
    parser.add_argument('--lattice_size', type=int, default=0, help='Size of finite lattice (default: 2*max_order)')
    parser.add_argument('--output_dir', type=str, default='.', help='Output directory for cluster information')
    parser.add_argument('--cache_dir', type=str, default='./cache', help='Directory for cache files')
    parser.add_argument('--no_cache', action='store_true', help='Disable cache (recompute everything)')
    parser.add_argument('--force_recompute', action='store_true', help='Force recomputation even if cache exists')
    return parser.parse_args()

def create_pyrochlore_lattice(L):
    """
    Create a pyrochlore lattice of size L×L×L unit cells.
    
    Returns:
    - G: NetworkX graph representing the lattice
    - pos: Dictionary mapping node IDs to 3D positions
    - tetrahedra: List of tetrahedra, each represented as a list of 4 node IDs
    """
    # FCC lattice vectors
    a1 = np.array([0, 0.5, 0.5])
    a2 = np.array([0.5, 0, 0.5])
    a3 = np.array([0.5, 0.5, 0])
    
    # Positions within unit cell
    basis_pos = np.array(
        [[0.125,0.125,0.125],
         [0.125,-0.125,-0.125],
         [-0.125,0.125,-0.125],
         [-0.125,-0.125,0.125]])


    G = nx.Graph()
    pos = {}
    tetrahedra = []
    
    # Generate lattice sites
    site_id = 0
    site_mapping = {}
    
    for i, j, k in itertools.product(range(L), repeat=3):
        cell_origin = i*a1 + j*a2 + k*a3
        
        for b, basis in enumerate(basis_pos):
            position = cell_origin + basis
            pos[site_id] = position
            site_mapping[(i, j, k, b)] = site_id
            G.add_node(site_id, pos=position)
            site_id += 1
    
    # Generate tetrahedra (two types per unit cell)
    for i, j, k in itertools.product(range(L), repeat=3):
        # First tetrahedron in the unit cell
        if all(0 <= x < L for x in [i, j, k]):
            tet1 = [
                site_mapping.get((i, j, k, 0)),
                site_mapping.get((i, j, k, 1)),
                site_mapping.get((i, j, k, 2)),
                site_mapping.get((i, j, k, 3))
            ]
            if None not in tet1:
                tetrahedra.append(tet1)
                # Add edges within tetrahedron
                for v1, v2 in itertools.combinations(tet1, 2):
                    G.add_edge(v1, v2)
        
        # Second tetrahedron (spans across unit cells)
        if all(0 <= x < L for x in [i, j, k, i+1, j+1, k+1]):
            tet2 = [
                site_mapping.get((i, j, k, 0)),
                site_mapping.get((i+1, j, k, 1)),
                site_mapping.get((i, j+1, k, 2)),
                site_mapping.get((i, j, k+1, 3))
            ]
            if None not in tet2:
                tetrahedra.append(tet2)
                # Add edges within tetrahedron
                for v1, v2 in itertools.combinations(tet2, 2):
                    G.add_edge(v1, v2)
    
    return G, pos, tetrahedra

def build_tetrahedron_graph(tetrahedra):
    """Build a graph where nodes are tetrahedra and edges represent shared vertices.
    Optimized: O(sum_v deg(v)^2) via vertex->tetra incidence; on pyrochlore, deg≈2.
    """
    tet_graph = nx.Graph()
    tet_graph.add_nodes_from(range(len(tetrahedra)))

    incident = defaultdict(list)  # vertex_id -> [tet_indices]
    for t_idx, tet in enumerate(tetrahedra):
        for v in tet:
            incident[v].append(t_idx)

    # Connect tetrahedra that share a vertex
    for tets in incident.values():
        if len(tets) > 1:
            for i in range(len(tets) - 1):
                for j in range(i + 1, len(tets)):
                    tet_graph.add_edge(tets[i], tets[j])

    return tet_graph

def _wl_hash_subgraph(G, nodes):
    """Compute an isomorphism-invariant hash for the induced subgraph on nodes."""
    H = G.subgraph(nodes).copy()
    # Relabel to a compact range so hash is label-invariant
    mapping = {n: i for i, n in enumerate(sorted(H.nodes()))}
    H = nx.relabel_nodes(H, mapping, copy=True)
    try:
        # NetworkX >= 2.6
        from networkx.algorithms.graph_hashing import weisfeiler_lehman_graph_hash
        return weisfeiler_lehman_graph_hash(H)
    except Exception:
        try:
            # Some older versions had it under networkx.algorithms.isomorphism
            from networkx.algorithms.isomorphism import weisfeiler_lehman_graph_hash
            return weisfeiler_lehman_graph_hash(H)
        except Exception:
            # Last-resort cheap signature (slower/more collisions)
            degs = sorted([d for _, d in H.degree()])
            return f"{len(H)}|{H.number_of_edges()}|{tuple(degs)}"

def generate_clusters(tet_graph, max_order):
    """
    Generate all topologically distinct clusters up to max_order and their multiplicities.
    Uses anchored expansion + WL hashing for fast dedup, exact iso only within same-hash buckets.
    """
    distinct_clusters = []
    multiplicities = []
    N = tet_graph.number_of_nodes()
    nodes_sorted = sorted(tet_graph.nodes())

    for order in range(1, max_order + 1):
        print(f"Generating clusters of order {order}...")

        # Order-1: one representative, multiplicity 1 per site
        if order == 1:
            first_tet = nodes_sorted[0]
            distinct_clusters.append([first_tet])
            multiplicities.append(0.125)  # per-site normalization
            continue

        # Hash-bucket: signature -> list of groups; each group is (rep_nodes_set, embedding_count)
        buckets = defaultdict(list)

        for anchor in nodes_sorted:
            # Anchored expansion: only allow nodes >= anchor to avoid cross-anchor duplicates
            start = frozenset([anchor])
            frontier = set(n for n in tet_graph.neighbors(anchor) if n >= anchor)
            visited = set()  # per-anchor visited subclusters to avoid re-derivations

            # DFS stack for connected expansions
            stack = [(start, frontier)]
            while stack:
                current, fr = stack.pop()

                if current in visited:
                    continue
                visited.add(current)

                if len(current) == order:
                    sig = _wl_hash_subgraph(tet_graph, current)

                    # Refine within the same hash by exact isomorphism to avoid rare WL collisions
                    placed = False
                    for idx, (rep_nodes, cnt) in enumerate(buckets[sig]):
                        if nx.is_isomorphic(
                            tet_graph.subgraph(current),
                            tet_graph.subgraph(rep_nodes),
                        ):
                            buckets[sig][idx] = (rep_nodes, cnt + 1)
                            placed = True
                            break
                    if not placed:
                        buckets[sig].append((current, 1))
                    continue

                    # (no len(current) > order branch needed)

                # Expand by one tetrahedron from the frontier
                # Keep anchor constraint: only nodes >= anchor
                for nxt in list(fr):
                    new_set = current | {nxt}
                    new_frontier = (fr | set(tet_graph.neighbors(nxt))) - new_set
                    # Enforce anchor-filter on frontier
                    new_frontier = {x for x in new_frontier if x >= anchor}
                    stack.append((new_set, new_frontier))

        # Collect representatives and compute multiplicities (per site)
        reps = []
        mults = []
        for sig_groups in buckets.values():
            for rep_nodes, count in sig_groups:
                reps.append(sorted(rep_nodes))
                mults.append(count / (8*N))  # embeddings per tetrahedron (site)

        distinct_clusters.extend(reps)
        multiplicities.extend(mults)
        print(f"Found {len(reps)} distinct clusters of order {order}")

    return distinct_clusters, multiplicities


def visualize_cluster(lattice, pos, tetrahedra, cluster, cluster_index):
    """Visualize a single cluster in 3D."""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Get all vertices in the cluster
    vertices = set()
    for tet_idx in cluster:
        vertices.update(tetrahedra[tet_idx])
    
    # Create subgraph for this cluster
    subgraph = lattice.subgraph(vertices)
    
    # Draw vertices
    xs = [pos[v][0] for v in subgraph.nodes()]
    ys = [pos[v][1] for v in subgraph.nodes()]
    zs = [pos[v][2] for v in subgraph.nodes()]
    ax.scatter(xs, ys, zs, c='r', s=100, label='Vertices')
    
    # Draw edges
    for u, v in subgraph.edges():
        ax.plot([pos[u][0], pos[v][0]],
                [pos[u][1], pos[v][1]],
                [pos[u][2], pos[v][2]], 'k-', lw=1)
    
    # Draw tetrahedra
    for tet_idx in cluster:
        tet = tetrahedra[tet_idx]
        # Draw faces of tetrahedron
        for face in itertools.combinations(tet, 3):
            triangle = np.array([pos[v] for v in face])
            ax.plot_trisurf(triangle[:, 0], triangle[:, 1], triangle[:, 2],
                          color='b', alpha=0.2)
    
    ax.set_title(f'Cluster {cluster_index} - {len(cluster)} tetrahedra')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_box_aspect([1, 1, 1])
    
    plt.tight_layout()
    plt.savefig(f'cluster_{cluster_index}_order_{len(cluster)}.png')
    plt.close()

def identify_subclusters(distinct_clusters, tet_graph):
    """
    Identify all topological subclusters for each distinct cluster and their multiplicities.
    
    Args:
    - distinct_clusters: List of topologically distinct clusters
    - tet_graph: NetworkX graph where nodes are tetrahedra
    
    Returns:
    - subclusters_info: Dictionary mapping cluster indices to their subclusters info
    """
    # Group distinct clusters by order
    clusters_by_order = defaultdict(list)
    for i, cluster in enumerate(distinct_clusters):
        clusters_by_order[len(cluster)].append((i, cluster))
    
    # Initialize results
    subclusters_info = {}
    
    # Process each distinct cluster
    for i, cluster in enumerate(distinct_clusters):
        cluster_order = len(cluster)
        subclusters_info[i] = []
        
        # Skip if cluster_order is 1 (no subclusters)
        if cluster_order == 1:
            continue
        
        # Find subclusters for each lower order
        for order in range(1, cluster_order):
            # Get candidate distinct clusters of this order
            candidates = clusters_by_order[order]
            
            # Track subcluster counts for this order
            subcluster_counts = defaultdict(int)
            
            # Generate all subclusters of the current order
            for subcluster_set in itertools.combinations(cluster, order):
                subcluster = frozenset(subcluster_set)
                subgraph = tet_graph.subgraph(subcluster)
                
                # Find matching distinct cluster
                for cand_idx, cand_cluster in candidates:
                    cand_subgraph = tet_graph.subgraph(cand_cluster)
                    if nx.is_isomorphic(subgraph, cand_subgraph):
                        subcluster_counts[cand_idx] += 1
                        break
            
            # Add to results
            for subcluster_idx, count in subcluster_counts.items():
                subclusters_info[i].append((subcluster_idx, count))
        
        # Sort by order
        subclusters_info[i].sort(key=lambda x: len(distinct_clusters[x[0]]))
    
    return subclusters_info

def save_subclusters_info(subclusters_info, distinct_clusters, multiplicities, output_dir):
    """
    Save information about subclusters of each distinct cluster to a file.
    
    Args:
    - subclusters_info: Dictionary mapping cluster indices to their subclusters info
    - distinct_clusters: List of topologically distinct clusters
    - multiplicities: List of multiplicities for each cluster
    - output_dir: Directory to save the output file
    """
    with open(f"{output_dir}/subclusters_info.txt", 'w') as f:
        f.write("# Subclusters information for each topologically distinct cluster\n")
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


def main():
    args = parse_arguments()
    max_order = args.max_order
    
    # Set lattice size
    L = args.lattice_size if args.lattice_size > 0 else max(10, 6*max_order)
    
    # Check cache unless disabled or forced recompute
    cache_data = None
    cache_path = get_cache_path(args.cache_dir, max_order, L)
    
    if not args.no_cache and not args.force_recompute:
        print(f"Checking for cached results...")
        cache_data = load_cache(cache_path)
        
        if verify_cache_validity(cache_data, max_order):
            print("Using cached results!")
            lattice = cache_data['lattice']
            pos = cache_data['pos']
            tetrahedra = cache_data['tetrahedra']
            tet_graph = cache_data['tet_graph']
            distinct_clusters = cache_data['distinct_clusters']
            multiplicities = cache_data['multiplicities']
        else:
            cache_data = None  # Invalid cache
    
    # Compute if no valid cache
    if cache_data is None:
        print(f"Generating pyrochlore lattice of size {L}×{L}×{L}...")
        lattice, pos, tetrahedra = create_pyrochlore_lattice(L)
        print(f"Generated lattice with {lattice.number_of_nodes()} sites and {len(tetrahedra)} tetrahedra")
        
        print("Building tetrahedron adjacency graph...")
        tet_graph = build_tetrahedron_graph(tetrahedra)
        
        print(f"Generating clusters up to order {max_order}...")
        distinct_clusters, multiplicities = generate_clusters(tet_graph, max_order)
        
        # Save to cache unless disabled
        if not args.no_cache:
            cache_data = {
                'lattice': lattice,
                'pos': pos,
                'tetrahedra': tetrahedra,
                'tet_graph': tet_graph,
                'distinct_clusters': distinct_clusters,
                'multiplicities': multiplicities,
                'max_order': max_order,
                'lattice_size': L
            }
            save_cache(cache_path, cache_data)
    
    # Organize clusters by order
    clusters_by_order = defaultdict(list)
    for i, cluster in enumerate(distinct_clusters):
        order = len(cluster)
        clusters_by_order[order].append((i, cluster, multiplicities[i]))
    
    # Print results
    print("\nCluster statistics:")
    for order in sorted(clusters_by_order.keys()):
        print(f"  Order {order}: {len(clusters_by_order[order])} distinct clusters")
    
    # Create output directory for cluster info
    output_dir = args.output_dir + f"/cluster_info_order_{max_order}"
    os.makedirs(output_dir, exist_ok=True)


    save_subclusters_info(identify_subclusters(distinct_clusters, tet_graph), distinct_clusters, multiplicities, output_dir)
    print(f"Subclusters information saved to {output_dir}/subclusters_info.txt")
    
    # Extract and save detailed information for each cluster
    print("\nExtracting and saving detailed cluster information...")
    for i, (cluster, multiplicity) in enumerate(zip(distinct_clusters, multiplicities)):
        cluster_id = i + 1
        order = len(cluster)
        print(f"  Processing cluster {cluster_id} (order {order})...")
        
        # Extract detailed information
        cluster_info = extract_cluster_info(lattice, pos, tetrahedra, cluster)
        
        # Save to file
        save_cluster_info(cluster_info, cluster_id, order, multiplicity, output_dir)

        print(f"  Saved to {output_dir}/cluster_{cluster_id}_order_{order}.dat")

    
    print(f"Detailed cluster information saved to {output_dir}/ directory")
    
    # Visualize clusters if requested
    if args.visualize:
        print("\nVisualizing clusters...")
        for i, cluster in enumerate(distinct_clusters):
            visualize_cluster(lattice, pos, tetrahedra, cluster, i+1)
        print(f"Visualization images saved as cluster_*.png")

if __name__ == "__main__":
    main()