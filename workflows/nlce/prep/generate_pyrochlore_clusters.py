#!/usr/bin/env python3
"""
Generate topologically distinct clusters on a pyrochlore lattice for
numerical linked cluster expansion (NLCE) calculations.

This script implements the tetrahedron-basis NLCE where:
- Clusters are connected subgraphs on the diamond lattice
  (vertices = tetrahedra, edges = shared pyrochlore spins)
- Multiplicity L(c) = |Emb(c→L)| / (N_site * |Aut(c)|) per pyrochlore site
- Subcluster multiplicity Y_cs = |Emb(s→c)| / |Aut(s)|

The diamond lattice has coordination z=4 (each tetrahedron shares a spin
with 4 neighbors). Pyrochlore normalization: L_pyro = L_tet / 2
since N_site = 2 * N_tet (2 pyrochlore sites per tetrahedron).

Reference multiplicities (per pyrochlore site, L_pyro = L_tet/2):
  Order 1: L = 0.5 (single tetrahedron)
  Order 2: L = z/4 = 1 (chain of 2)
  Order 3: L = z(z-1)/4 = 3 (chain of 3)
  Order 4: L = z(z-1)²/4 = 9 (chain of 4), L = C(z,3)/2 = 2 (3-star)
"""

import argparse
import numpy as np
import networkx as nx
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import itertools
import sys
import os
from collections import defaultdict


def extract_cluster_info(lattice, pos, tetrahedra, cluster):
    """
    Extract detailed information about a cluster.
    
    Args:
        lattice: NetworkX graph of pyrochlore lattice (spin sites)
        pos: Dictionary mapping site IDs to 3D positions
        tetrahedra: List of tetrahedra (each is list of 4 site IDs)
        cluster: List of tetrahedron indices in this cluster
    
    Returns:
        Dictionary containing vertices, positions, edges, tetrahedra, 
        adjacency matrix, and node mapping
    """
    # Get all vertices (pyrochlore spin sites) in the cluster
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
    parser = argparse.ArgumentParser(
        description='Generate topologically distinct clusters on a pyrochlore lattice for NLCE.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python generate_pyrochlore_clusters.py --max_order 4
  python generate_pyrochlore_clusters.py --max_order 3 --visualize --output_dir ./clusters
        """
    )
    parser.add_argument('--max_order', type=int, required=True, 
                        help='Maximum order of clusters to generate (number of tetrahedra)')
    parser.add_argument('--visualize', action='store_true', 
                        help='Visualize each cluster and save as PNG')
    parser.add_argument('--lattice_size', type=int, default=0, 
                        help='Size of finite lattice (default: max_order + 2)')
    parser.add_argument('--output_dir', type=str, default='.', 
                        help='Output directory for cluster information')
    parser.add_argument('--no_pbc', action='store_true',
                        help='Disable periodic boundary conditions (not recommended)')
    return parser.parse_args()


def create_pyrochlore_lattice(L, periodic=True):
    """
    Create a pyrochlore lattice of size L×L×L unit cells with periodic 
    boundary conditions.
    
    The pyrochlore lattice consists of corner-sharing tetrahedra.
    Each unit cell contains 4 pyrochlore sites and 2 tetrahedra
    (one "up" and one "down" type).
    
    With PBC, there are exactly 2*L³ tetrahedra and 4*L³ spin sites,
    and every tetrahedron has coordination z=4 on the diamond lattice.
    
    Args:
        L: Number of unit cells in each direction
        periodic: Use periodic boundary conditions (default: True)
    
    Returns:
        G: NetworkX graph representing the lattice (nodes = spin sites)
        pos: Dictionary mapping node IDs to 3D positions
        tetrahedra: List of tetrahedra, each as a list of 4 node IDs
    """
    # FCC lattice vectors (conventional cubic cell)
    a1 = np.array([0, 0.5, 0.5])
    a2 = np.array([0.5, 0, 0.5])
    a3 = np.array([0.5, 0.5, 0])
    
    # Pyrochlore basis positions within unit cell (on FCC sites)
    # These form the vertices of a tetrahedron centered at origin
    basis_pos = np.array([
        [ 0.125,  0.125,  0.125],
        [ 0.125, -0.125, -0.125],
        [-0.125,  0.125, -0.125],
        [-0.125, -0.125,  0.125]
    ])

    G = nx.Graph()
    pos = {}
    tetrahedra = []
    
    # Generate lattice sites
    site_id = 0
    site_mapping = {}  # (i, j, k, basis) -> site_id
    
    for i, j, k in itertools.product(range(L), repeat=3):
        cell_origin = i * a1 + j * a2 + k * a3
        
        for b, basis in enumerate(basis_pos):
            position = cell_origin + basis
            pos[site_id] = position
            site_mapping[(i, j, k, b)] = site_id
            G.add_node(site_id, pos=position)
            site_id += 1
    
    # Generate tetrahedra (two types per unit cell)
    for i, j, k in itertools.product(range(L), repeat=3):
        # First tetrahedron ("up" type) - all 4 basis sites in same unit cell
        tet1 = [
            site_mapping[(i, j, k, 0)],
            site_mapping[(i, j, k, 1)],
            site_mapping[(i, j, k, 2)],
            site_mapping[(i, j, k, 3)]
        ]
        tetrahedra.append(tet1)
        # Add edges within tetrahedron (complete graph K_4)
        for v1, v2 in itertools.combinations(tet1, 2):
            G.add_edge(v1, v2)
        
        # Second tetrahedron ("down" type) - spans across neighboring unit cells
        # With PBC, use modular arithmetic
        if periodic:
            tet2 = [
                site_mapping[(i, j, k, 0)],
                site_mapping[((i + 1) % L, j, k, 1)],
                site_mapping[(i, (j + 1) % L, k, 2)],
                site_mapping[(i, j, (k + 1) % L, 3)]
            ]
            tetrahedra.append(tet2)
            for v1, v2 in itertools.combinations(tet2, 2):
                G.add_edge(v1, v2)
        else:
            # Open boundary: only include if all neighbors exist
            if i + 1 < L and j + 1 < L and k + 1 < L:
                tet2 = [
                    site_mapping[(i, j, k, 0)],
                    site_mapping[(i + 1, j, k, 1)],
                    site_mapping[(i, j + 1, k, 2)],
                    site_mapping[(i, j, k + 1, 3)]
                ]
                tetrahedra.append(tet2)
                for v1, v2 in itertools.combinations(tet2, 2):
                    G.add_edge(v1, v2)
    
    return G, pos, tetrahedra


def build_tetrahedron_graph(tetrahedra):
    """
    Build the diamond lattice graph where nodes are tetrahedra and edges 
    represent shared pyrochlore spins.
    
    This is the key structure for NLCE: clusters are connected subgraphs
    of this tetrahedron (diamond) graph.
    
    Args:
        tetrahedra: List of tetrahedra (each is list of 4 site IDs)
    
    Returns:
        NetworkX graph where nodes = tetrahedra, edges = shared vertices
    """
    tet_graph = nx.Graph()
    tet_graph.add_nodes_from(range(len(tetrahedra)))

    # Build vertex -> tetrahedra incidence map
    incident = defaultdict(list)
    for t_idx, tet in enumerate(tetrahedra):
        for v in tet:
            incident[v].append(t_idx)

    # Connect tetrahedra that share a vertex (pyrochlore spin)
    for tets in incident.values():
        if len(tets) > 1:
            for i in range(len(tets)):
                for j in range(i + 1, len(tets)):
                    tet_graph.add_edge(tets[i], tets[j])

    return tet_graph


def compute_automorphism_count(G, nodes):
    """
    Compute |Aut(c)|, the number of automorphisms of the cluster graph.
    
    An automorphism is a permutation of nodes that preserves adjacency.
    This is needed for the multiplicity formula:
    L(c) = |labeled embeddings| / (N_tet * |Aut(c)|)
    
    Args:
        G: The full tetrahedron graph
        nodes: Set/list of nodes defining the cluster
    
    Returns:
        Number of automorphisms of the induced subgraph
    """
    H = G.subgraph(nodes).copy()
    # Relabel to compact range for consistent comparison
    mapping = {n: i for i, n in enumerate(sorted(H.nodes()))}
    H = nx.relabel_nodes(H, mapping, copy=True)
    
    # Use VF2 to find all automorphisms
    GM = nx.isomorphism.GraphMatcher(H, H)
    return sum(1 for _ in GM.isomorphisms_iter())


def _wl_hash_subgraph(G, nodes):
    """
    Compute an isomorphism-invariant hash for the induced subgraph.
    Uses Weisfeiler-Lehman graph hashing for fast comparison.
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


def generate_clusters(tet_graph, max_order):
    """
    Generate all topologically distinct clusters up to max_order.
    
    Uses anchored expansion to enumerate all connected subgraphs,
    then groups by isomorphism class. Multiplicities are computed as:
    
    L_tet(c) = |Emb(c→L)| / N_tet  (unlabeled embeddings per tetrahedron)
    
    With PBC, all tetrahedra have z=4 coordination, so we can use all nodes.
    The anchored expansion ensures each unlabeled cluster is counted once.
    
    For order n tree topologies on diamond lattice (z=4):
      - Order 1: L = 1
      - Order 2: L = z/2 = 2  
      - Order 3: L = z(z-1)/2 = 6
      - Order 4 chain: L = z(z-1)²/2 = 18
      - Order 4 star: L = C(z,3) = 4
    
    Args:
        tet_graph: Diamond lattice graph (tetrahedra as nodes)
        max_order: Maximum cluster size (number of tetrahedra)
    
    Returns:
        distinct_clusters: List of cluster representatives (each is list of node IDs)
        multiplicities: List of multiplicities L_tet(c) per tetrahedron
    """
    distinct_clusters = []
    multiplicities = []
    N = tet_graph.number_of_nodes()
    nodes_sorted = sorted(tet_graph.nodes())
    
    # With PBC, all nodes should have coordination 4
    # For open boundaries, we still use all nodes but results may have boundary effects
    degrees = [tet_graph.degree(n) for n in nodes_sorted]
    if min(degrees) == max(degrees) == 4:
        print(f"  Lattice has {N} tetrahedra (all with z=4, PBC working correctly)")
    else:
        print(f"  Lattice has {N} tetrahedra (coordination varies: {min(degrees)}-{max(degrees)})")
    
    for order in range(1, max_order + 1):
        print(f"Generating clusters of order {order}...")
        
        if order == 1:
            # Single tetrahedron: L_tet = 1, L_pyro = 0.5
            first_tet = nodes_sorted[0]
            distinct_clusters.append([first_tet])
            multiplicities.append(0.5)  # L_pyro = L_tet / 2 = 1/2
            continue
        
        # Hash buckets: signature -> list of (rep_nodes_frozenset, raw_embedding_count)
        # With PBC, all nodes are equivalent, so we can anchor at any node
        buckets = defaultdict(list)
        
        for anchor in nodes_sorted:
            # Anchored expansion: only allow nodes >= anchor
            # This ensures each unlabeled embedding is counted once
            start = frozenset([anchor])
            frontier = set(n for n in tet_graph.neighbors(anchor) if n >= anchor)
            visited = set()
            
            stack = [(start, frontier)]
            while stack:
                current, fr = stack.pop()
                
                if current in visited:
                    continue
                visited.add(current)
                
                if len(current) == order:
                    # Found a cluster of target size
                    sig = _wl_hash_subgraph(tet_graph, current)
                    
                    # Check isomorphism within same hash bucket
                    placed = False
                    for idx, (rep_nodes, cnt) in enumerate(buckets[sig]):
                        if nx.is_isomorphic(
                            tet_graph.subgraph(current),
                            tet_graph.subgraph(rep_nodes)
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
                    new_frontier = (fr | set(tet_graph.neighbors(nxt))) - new_set
                    new_frontier = {x for x in new_frontier if x >= anchor}
                    stack.append((new_set, new_frontier))
        
        # Collect representatives and compute multiplicities
        reps = []
        mults = []
        
        for sig_groups in buckets.values():
            for rep_nodes, raw_count in sig_groups:
                # raw_count = number of anchored embeddings found
                # Due to anchoring, each unlabeled embedding is counted once
                # L_tet(c) = raw_count / N (per tetrahedron)
                # L_pyro(c) = L_tet / 2 (per pyrochlore site, since N_site = 2 * N_tet)
                
                reps.append(sorted(rep_nodes))
                mults.append(raw_count / N / 2)  # Divide by 2 for pyrochlore normalization
        
        distinct_clusters.extend(reps)
        multiplicities.extend(mults)
        print(f"  Found {len(reps)} distinct clusters of order {order}")
    
    return distinct_clusters, multiplicities


def count_embeddings(source_graph, target_graph):
    """
    Count the number of injective graph homomorphisms (embeddings) 
    from source_graph into target_graph.
    
    An embedding f: V_s → V_t satisfies:
    (i,j) ∈ E_s ⟹ (f(i), f(j)) ∈ E_t
    
    Args:
        source_graph: The cluster to embed
        target_graph: The graph to embed into
    
    Returns:
        Number of embeddings (labeled)
    """
    if source_graph.number_of_nodes() > target_graph.number_of_nodes():
        return 0
    
    # Use subgraph isomorphism to find all embeddings
    GM = nx.isomorphism.GraphMatcher(target_graph, source_graph)
    
    # Each subgraph isomorphism corresponds to an embedding
    count = 0
    for _ in GM.subgraph_isomorphisms_iter():
        count += 1
    
    return count


def compute_subcluster_multiplicities(cluster_nodes, subcluster_nodes, tet_graph):
    """
    Compute Y_cs = |Emb(s→c)| / |Aut(s)|
    
    This is the number of ways to embed subcluster s into cluster c,
    divided by automorphisms of s (to count unlabeled embeddings).
    
    Args:
        cluster_nodes: Nodes of the larger cluster c
        subcluster_nodes: Nodes of the subcluster s  
        tet_graph: The full tetrahedron graph
    
    Returns:
        Y_cs: subcluster multiplicity
    """
    cluster_subgraph = tet_graph.subgraph(cluster_nodes).copy()
    subcluster_subgraph = tet_graph.subgraph(subcluster_nodes).copy()
    
    # Count labeled embeddings
    labeled_embeddings = count_embeddings(subcluster_subgraph, cluster_subgraph)
    
    # Compute |Aut(s)|
    aut_s = compute_automorphism_count(tet_graph, subcluster_nodes)
    
    # Y_cs = |Emb(s→c)| / |Aut(s)|
    return labeled_embeddings // aut_s


def identify_subclusters(distinct_clusters, tet_graph):
    """
    Identify all topological subclusters for each distinct cluster 
    and their multiplicities Y_cs.
    
    For each cluster c, we find all connected subclusters s ⊂ c,
    match them to distinct cluster types, and compute:
    Y_cs = |Emb(s→c)| / |Aut(s)|
    
    Args:
        distinct_clusters: List of topologically distinct cluster representatives
        tet_graph: NetworkX graph where nodes are tetrahedra
    
    Returns:
        Dictionary mapping cluster indices to list of (subcluster_idx, Y_cs)
    """
    # Group distinct clusters by order for efficient lookup
    clusters_by_order = defaultdict(list)
    for i, cluster in enumerate(distinct_clusters):
        clusters_by_order[len(cluster)].append((i, cluster))
    
    subclusters_info = {}
    
    for i, cluster in enumerate(distinct_clusters):
        cluster_order = len(cluster)
        subclusters_info[i] = []
        
        if cluster_order == 1:
            continue
        
        # For each possible subcluster order
        for order in range(1, cluster_order):
            candidates = clusters_by_order[order]
            subcluster_counts = defaultdict(int)
            
            # Enumerate all connected subsets of size 'order'
            for subcluster_set in itertools.combinations(cluster, order):
                subgraph = tet_graph.subgraph(subcluster_set)
                
                # Check if connected
                if not nx.is_connected(subgraph):
                    continue
                
                # Find which distinct cluster type this matches
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
    
    Format includes Y_cs values for the inclusion-exclusion weight formula:
    W_P(c) = P(c) - Σ_s Y_cs * W_P(s)
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


def verify_multiplicities(distinct_clusters, multiplicities, tet_graph):
    """
    Verify computed multiplicities against known analytical values.
    
    For diamond lattice (z=4), pyrochlore normalization (L_pyro = L_tet/2):
      Order 1: L_pyro = 0.5
      Order 2: L_pyro = z/4 = 1
      Order 3: L_pyro = z(z-1)/4 = 3
      Order 4 chain: L_pyro = z(z-1)²/4 = 9
      Order 4 star: L_pyro = C(z,3)/2 = 2
    """
    z = 4  # Diamond lattice coordination
    
    # Expected multiplicities per pyrochlore site (L_tet / 2)
    expected = {
        1: [0.5],  # Single tetrahedron
        2: [1.0],  # Chain of 2 (z/4)
        3: [3.0],  # Chain of 3 (z(z-1)/4)
    }
    
    # For order 4, we have two topologies
    # Need to distinguish chain from star based on graph structure
    
    clusters_by_order = defaultdict(list)
    for i, cluster in enumerate(distinct_clusters):
        clusters_by_order[len(cluster)].append((i, cluster, multiplicities[i]))
    
    print("\n" + "="*60)
    print("MULTIPLICITY VERIFICATION (per pyrochlore site)")
    print("="*60)
    
    for order in sorted(clusters_by_order.keys()):
        print(f"\nOrder {order}:")
        for i, cluster, mult in clusters_by_order[order]:
            subgraph = tet_graph.subgraph(cluster)
            
            # Determine topology type
            max_degree = max(d for _, d in subgraph.degree())
            if order == 4:
                if max_degree == 3:
                    topology = "3-star (K_{1,3})"
                    expected_mult = 2.0  # C(4,3)/2
                else:
                    topology = "4-chain"
                    expected_mult = 9.0  # z(z-1)²/4
            elif order == 3:
                topology = "3-chain"
                expected_mult = 3.0
            elif order == 2:
                topology = "2-chain"
                expected_mult = 1.0
            elif order == 1:
                topology = "single"
                expected_mult = 0.5
            else:
                topology = "unknown"
                expected_mult = None
            
            status = ""
            if expected_mult is not None:
                if abs(mult - expected_mult) < 0.01:
                    status = "✓ CORRECT"
                else:
                    status = f"✗ EXPECTED {expected_mult}"
            
            print(f"  Cluster {i+1}: {topology}, L_pyro = {mult:.4f} {status}")


def main():
    args = parse_arguments()
    max_order = args.max_order
    
    # Set lattice size - with PBC, we only need enough to fit the largest cluster
    # Diamond lattice shortest loop is 6, so L = max_order + 2 is sufficient
    L = args.lattice_size if args.lattice_size > 0 else max(3, max_order + 2)
    use_pbc = not args.no_pbc
    
    pbc_str = "with PBC" if use_pbc else "open boundary"
    print(f"Generating pyrochlore lattice of size {L}×{L}×{L} ({pbc_str})...")
    lattice, pos, tetrahedra = create_pyrochlore_lattice(L, periodic=use_pbc)
    print(f"Generated lattice with {lattice.number_of_nodes()} sites and {len(tetrahedra)} tetrahedra")
    
    print("\nBuilding tetrahedron adjacency graph (diamond lattice)...")
    tet_graph = build_tetrahedron_graph(tetrahedra)
    print(f"Diamond graph: {tet_graph.number_of_nodes()} nodes, {tet_graph.number_of_edges()} edges")
    
    # Check coordination number
    degrees = [d for _, d in tet_graph.degree()]
    avg_deg = np.mean(degrees) if degrees else 0
    min_deg = min(degrees) if degrees else 0
    max_deg = max(degrees) if degrees else 0
    print(f"Coordination: min={min_deg}, max={max_deg}, avg={avg_deg:.2f} (expected: 4 with PBC)")
    
    print(f"\nGenerating clusters up to order {max_order}...")
    distinct_clusters, multiplicities = generate_clusters(tet_graph, max_order)
    
    # Verify multiplicities against known values
    verify_multiplicities(distinct_clusters, multiplicities, tet_graph)
    
    # Organize clusters by order
    clusters_by_order = defaultdict(list)
    for i, cluster in enumerate(distinct_clusters):
        order = len(cluster)
        clusters_by_order[order].append((i, cluster, multiplicities[i]))
    
    # Print results
    print("\n" + "="*60)
    print("CLUSTER STATISTICS (per pyrochlore site)")
    print("="*60)
    for order in sorted(clusters_by_order.keys()):
        print(f"  Order {order}: {len(clusters_by_order[order])} distinct cluster(s)")
        for i, cluster, mult in clusters_by_order[order]:
            print(f"    Cluster {i+1}: L_pyro = {mult:.4f}")
    
    # Create output directory
    output_dir = args.output_dir + f"/cluster_info_order_{max_order}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Identify and save subclusters information
    print("\nIdentifying subclusters for inclusion-exclusion...")
    subclusters_info = identify_subclusters(distinct_clusters, tet_graph)
    save_subclusters_info(subclusters_info, distinct_clusters, multiplicities, output_dir)
    print(f"Subclusters information saved to {output_dir}/subclusters_info.txt")
    
    # Print Y_cs values
    print("\n" + "="*60)
    print("SUBCLUSTER MULTIPLICITIES (Y_cs)")
    print("="*60)
    for i, cluster in enumerate(distinct_clusters):
        if len(cluster) == 1:
            continue
        print(f"\nCluster {i+1} (order {len(cluster)}):")
        subclusters = subclusters_info.get(i, [])
        for sub_idx, count in subclusters:
            sub_order = len(distinct_clusters[sub_idx])
            print(f"  Y_{{c,s{sub_idx+1}}} = {count} (subcluster of order {sub_order})")
    
    # Extract and save detailed information for each cluster
    print("\nExtracting and saving detailed cluster information...")
    for i, (cluster, multiplicity) in enumerate(zip(distinct_clusters, multiplicities)):
        cluster_id = i + 1
        order = len(cluster)
        print(f"  Processing cluster {cluster_id} (order {order})...")
        
        # Extract detailed information (transform to pyrochlore representation)
        cluster_info = extract_cluster_info(lattice, pos, tetrahedra, cluster)
        
        # Save to file
        save_cluster_info(cluster_info, cluster_id, order, multiplicity, output_dir)
        print(f"  Saved to {output_dir}/cluster_{cluster_id}_order_{order}.dat")
    
    print(f"\nDetailed cluster information saved to {output_dir}/ directory")
    
    # Visualize clusters if requested
    if args.visualize:
        print("\nVisualizing clusters...")
        for i, cluster in enumerate(distinct_clusters):
            visualize_cluster(lattice, pos, tetrahedra, cluster, i + 1)
        print(f"Visualization images saved as cluster_*.png")
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Total distinct clusters: {len(distinct_clusters)}")
    for order in sorted(clusters_by_order.keys()):
        count = len(clusters_by_order[order])
        print(f"  Order {order}: {count} topology(ies)")


if __name__ == "__main__":
    main()
