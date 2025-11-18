#!/usr/bin/env python3
"""
Generate topologically distinct clusters on a pyrochlore lattice for
numerical linked cluster expansion (NLCE) calculations.

This implementation follows the correct NLCE multiplicity (lattice constant) calculation:
    L(c) = lim_{M→∞} H_M(c) / (|Aut(c)| × N_subunit(M))

where:
- H_M(c) = number of injective graph embeddings of cluster c into a torus of size M
- |Aut(c)| = size of the graph automorphism group of c
- N_subunit(M) = number of subunits (tetrahedra or sites) in the torus

References:
- Tang, Khatami, Rigol (NLCE review)
- Singh, Advances in Physics (High-order convergent expansions)
"""

import argparse
import numpy as np
import networkx as nx
from networkx.algorithms import isomorphism
from mpl_toolkits.mplot3d import Axes3D
import itertools
import sys
from collections import defaultdict, deque
import os
import collections

import matplotlib.pyplot as plt


def parse_arguments():
    parser = argparse.ArgumentParser(description='Generate topologically distinct clusters on a pyrochlore lattice.')
    parser.add_argument('--max_order', type=int, required=True, help='Maximum order of clusters to generate')
    parser.add_argument('--visualize', action='store_true', help='Visualize each cluster')
    parser.add_argument('--lattice_size', type=int, default=0, help='Size of finite lattice (default: 2*max_order)')
    parser.add_argument('--output_dir', type=str, default='.', help='Output directory for cluster information')
    parser.add_argument('--subunit', type=str, choices=['tetrahedron', 'site'], default='site', help='Expansion subunit type')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output')
    return parser.parse_args()


def create_pyrochlore_lattice(L, periodic=True):
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
        
        i_next = (i + 1) % L if periodic else i + 1
        j_next = (j + 1) % L if periodic else j + 1
        k_next = (k + 1) % L if periodic else k + 1

        # Second tetrahedron (spans across unit cells)
        if all(0 <= x < L for x in [i, j, k, i_next, j_next, k_next]):
            tet2 = [
                site_mapping.get((i, j, k, 0)),
                site_mapping.get((i_next, j, k, 1)),
                site_mapping.get((i, j_next, k, 2)),
                site_mapping.get((i, j, k_next, 3))
            ]
            if None not in tet2:
                tetrahedra.append(tet2)
                # Add edges within tetrahedron
                for v1, v2 in itertools.combinations(tet2, 2):
                    G.add_edge(v1, v2)
    return G, pos, tetrahedra


def build_tetrahedron_graph(tetrahedra):
    """
    Build a graph where nodes are tetrahedra and edges represent shared vertices.
    
    Args:
        tetrahedra: List of tetrahedra (each a tuple of 4 vertex IDs)
    
    Returns:
        tet_graph: NetworkX graph with tetrahedra as nodes
    """
    tet_graph = nx.Graph()
    tet_graph.add_nodes_from(range(len(tetrahedra)))

    # Build incidence map: vertex -> list of tetrahedron indices
    incident = defaultdict(list)
    for t_idx, tet in enumerate(tetrahedra):
        for v in tet:
            incident[v].append(t_idx)

    # Connect tetrahedra that share at least one vertex
    for tets in incident.values():
        if len(tets) > 1:
            for i in range(len(tets) - 1):
                for j in range(i + 1, len(tets)):
                    tet_graph.add_edge(tets[i], tets[j])

    return tet_graph


def compute_automorphism_group_size(G):
    """
    Compute the size of the automorphism group of graph G.
    
    Uses NetworkX's VF2 algorithm to find all automorphisms.
    
    Args:
        G: NetworkX graph
        
    Returns:
        Size of the automorphism group |Aut(G)|
    """
    # Relabel nodes to 0..n-1 for canonical form
    mapping = {node: i for i, node in enumerate(sorted(G.nodes()))}
    G_relabeled = nx.relabel_nodes(G, mapping, copy=True)
    
    # Use graph matcher to find all automorphisms
    GM = isomorphism.GraphMatcher(G_relabeled, G_relabeled)
    
    # Count all isomorphisms (automorphisms)
    count = 0
    for _ in GM.isomorphisms_iter():
        count += 1
    
    return count


def count_injective_embeddings(cluster_graph, lattice_graph, max_embeddings=None, verbose=False):
    """
    Count the number of injective graph homomorphisms (subgraph isomorphisms)
    from cluster_graph into lattice_graph.
    
    This is the H_M(c) in the NLCE multiplicity formula.
    
    Args:
        cluster_graph: NetworkX graph representing the cluster topology
        lattice_graph: NetworkX graph representing the finite lattice (torus)
        max_embeddings: Optional limit on the number of embeddings to count
        verbose: Print progress information
        
    Returns:
        Number of injective embeddings
    """
    # Use VF2 algorithm for subgraph isomorphism
    GM = isomorphism.GraphMatcher(lattice_graph, cluster_graph)
    
    count = 0
    for mapping in GM.subgraph_isomorphisms_iter():
        count += 1
        if max_embeddings is not None and count >= max_embeddings:
            break
    
    if verbose:
        print(f"    Found {count} injective embeddings")
    
    return count


def compute_cluster_multiplicity(cluster_nodes, tet_graph, lattice_graph, tetrahedra, 
                                  subunit='tetrahedron', verbose=False):
    """
    Compute the correct NLCE multiplicity (lattice constant) for a cluster.
    
    Formula: L(c) = H_M(c) / (|Aut(c)| × N_subunit)
    
    where:
    - H_M(c) = number of injective embeddings in the torus
    - |Aut(c)| = automorphism group size
    - N_subunit = number of subunits (tetrahedra or sites) in the torus
    
    Args:
        cluster_nodes: List/set of tetrahedron indices forming the cluster
        tet_graph: Tetrahedron adjacency graph (full lattice)
        lattice_graph: Site-level graph (full lattice)
        tetrahedra: List of all tetrahedra
        subunit: 'tetrahedron' or 'site' for normalization
        verbose: Print detailed calculation steps
        
    Returns:
        multiplicity: The lattice constant L(c)
    """
    if verbose:
        print(f"  Computing multiplicity for cluster with {len(cluster_nodes)} tetrahedra...")
    
    # Step 1: Create the cluster's tetrahedron-level graph
    cluster_tet_graph = tet_graph.subgraph(cluster_nodes).copy()
    
    # Relabel to canonical form (0, 1, 2, ...)
    mapping = {node: i for i, node in enumerate(sorted(cluster_nodes))}
    cluster_tet_graph = nx.relabel_nodes(cluster_tet_graph, mapping)
    
    # Step 2: Compute automorphism group size
    aut_size = compute_automorphism_group_size(cluster_tet_graph)
    if verbose:
        print(f"    Automorphism group size: {aut_size}")
    
    # Step 3: Count injective embeddings in the full tetrahedron graph
    num_embeddings = count_injective_embeddings(
        cluster_tet_graph, 
        tet_graph, 
        verbose=verbose
    )
    
    # Step 4: Determine number of subunits
    if subunit == 'tetrahedron':
        N_subunit = tet_graph.number_of_nodes()
    elif subunit == 'site':
        N_subunit = lattice_graph.number_of_nodes()
    else:
        raise ValueError(f"Unknown subunit type: {subunit}")
    
    # Step 5: Compute multiplicity
    multiplicity = num_embeddings / (aut_size * N_subunit)
    
    if verbose:
        print(f"    H_M(c) = {num_embeddings}")
        print(f"    |Aut(c)| = {aut_size}")
        print(f"    N_subunit = {N_subunit}")
        print(f"    L(c) = {num_embeddings}/{aut_size}/{N_subunit} = {multiplicity}")
    
    return multiplicity


def verify_multiplicity_sanity(order, multiplicity, subunit='tetrahedron'):
    """
    Verify that computed multiplicities satisfy known constraints.
    
    Returns True if the multiplicity seems reasonable.
    """
    # Order 1 cluster should have multiplicity = 1 (one tetrahedron per tetrahedron)
    if order == 1:
        if subunit == 'tetrahedron':
            return abs(multiplicity - 1.0) < 1e-6
        elif subunit == 'site':
            # One tetrahedron has 4 sites, so L = 1/4 per site
            return abs(multiplicity - 0.25) < 1e-6
    
    # Multiplicity should be positive
    if multiplicity <= 0:
        return False
    
    # For site-based expansion, multiplicity can be fractional
    # For tetrahedron-based, it's typically of order 1
    return True


def _wl_hash_subgraph(G, nodes):
    """
    Compute an isomorphism-invariant hash for the induced subgraph on nodes.
    Uses Weisfeiler-Lehman graph hashing for fast deduplication.
    """
    H = G.subgraph(nodes).copy()
    # Relabel to canonical form
    mapping = {n: i for i, n in enumerate(sorted(H.nodes()))}
    H = nx.relabel_nodes(H, mapping, copy=True)
    
    try:
        from networkx.algorithms.graph_hashing import weisfeiler_lehman_graph_hash
        return weisfeiler_lehman_graph_hash(H)
    except (ImportError, AttributeError):
        # Fallback: use a simple structural signature
        degs = sorted([d for _, d in H.degree()])
        return f"{len(H)}|{H.number_of_edges()}|{tuple(degs)}"


def find_central_node(G, pos_dict=None):
    """
    Find a central node in the graph to use as anchor for cluster generation.
    
    Uses geometric center if positions are available, otherwise uses
    graph-theoretic center (node minimizing eccentricity).
    
    Args:
        G: NetworkX graph
        pos_dict: Optional dictionary of node positions {node_id: np.array([x, y, z])}
        
    Returns:
        Central node ID
    """
    if pos_dict is not None and len(pos_dict) > 0:
        # Compute geometric center
        nodes = list(G.nodes())
        positions = np.array([pos_dict[n] for n in nodes])
        center = np.mean(positions, axis=0)
        
        # Find node closest to geometric center
        distances = [np.linalg.norm(pos_dict[n] - center) for n in nodes]
        central_idx = np.argmin(distances)
        return nodes[central_idx]
    else:
        # Use graph-theoretic center (minimize eccentricity)
        # For large graphs, use approximation via BFS from random node
        nodes = list(G.nodes())
        if len(nodes) == 0:
            return None
        
        # Pick node with maximum degree as heuristic for central node
        degrees = dict(G.degree())
        central_node = max(nodes, key=lambda n: degrees[n])
        return central_node


def generate_clusters(tet_graph, lattice_graph, tetrahedra, max_order, 
                      subunit='tetrahedron', verbose=False, pos=None):
    """
    Generate all topologically distinct clusters up to max_order 
    with correct NLCE multiplicities.
    
    Uses:
    - Anchored expansion to avoid duplicate generation
    - Central node as primary anchor for symmetric cluster representatives
    - WL hashing for fast topology deduplication
    - Exact isomorphism checks within hash buckets
    - Correct multiplicity calculation via automorphism groups
    
    Args:
        tet_graph: Tetrahedron adjacency graph
        lattice_graph: Site-level lattice graph
        tetrahedra: List of all tetrahedra
        max_order: Maximum cluster size (number of tetrahedra)
        subunit: 'tetrahedron' or 'site' for normalization
        verbose: Print detailed progress
        pos: Optional dictionary of site positions (for finding geometric center)
        
    Returns:
        distinct_clusters: List of topologically distinct clusters
        multiplicities: List of correct NLCE multiplicities L(c)
    """
    distinct_clusters = []
    multiplicities = []
    N = tet_graph.number_of_nodes()
    nodes_sorted = sorted(tet_graph.nodes())
    
    # Compute tetrahedron centers if site positions are available
    tet_positions = None
    if pos is not None:
        tet_positions = {}
        for tet_idx, tet in enumerate(tetrahedra):
            # Compute center of mass of tetrahedron
            tet_pos = np.array([pos[v] for v in tet])
            tet_positions[tet_idx] = np.mean(tet_pos, axis=0)

    for order in range(1, max_order + 1):
        print(f"\nGenerating clusters of order {order}...")
        
        # Order 1: single tetrahedron
        if order == 1:
            first_tet = nodes_sorted[0]
            distinct_clusters.append([first_tet])
            
            # Compute multiplicity correctly
            mult = compute_cluster_multiplicity(
                [first_tet], tet_graph, lattice_graph, tetrahedra,
                subunit=subunit, verbose=verbose
            )
            multiplicities.append(mult)
            
            # Sanity check
            if not verify_multiplicity_sanity(order, mult, subunit):
                print(f"WARNING: Order {order} multiplicity {mult} failed sanity check!")
            
            print(f"  Found 1 distinct cluster with L(c) = {mult:.6f}")
            continue
        
        # For higher orders: generate via anchored expansion
        # Hash buckets: signature -> [(representative_nodes, embedding_count)]
        buckets = defaultdict(list)
        
        # Find central node to use as primary anchor for representative clusters
        # This generates more symmetric clusters
        central_node = find_central_node(tet_graph, pos_dict=tet_positions)
        
        if verbose:
            print(f"  Using central node {central_node} as primary anchor")
        
        # Use central node as anchor, but still iterate through all nodes
        # to ensure we find all topologically distinct clusters
        # Prioritize central node to get nice representative clusters first
        anchor_order = [central_node] + [n for n in nodes_sorted if n != central_node]
        
        for anchor in anchor_order:
            # Only expand to nodes >= anchor (prevents duplicate generation)
            start = frozenset([anchor])
            frontier = set(n for n in tet_graph.neighbors(anchor) if n >= anchor)
            visited = set()
            
            # DFS to grow connected clusters
            stack = [(start, frontier)]
            while stack:
                current, fr = stack.pop()
                
                if current in visited:
                    continue
                visited.add(current)
                
                if len(current) == order:
                    # Compute WL hash for fast deduplication
                    sig = _wl_hash_subgraph(tet_graph, current)
                    
                    # Check exact isomorphism within same hash bucket
                    placed = False
                    for idx, (rep_nodes, cnt) in enumerate(buckets[sig]):
                        if nx.is_isomorphic(
                            tet_graph.subgraph(current),
                            tet_graph.subgraph(rep_nodes)
                        ):
                            # Same topology - increment count
                            buckets[sig][idx] = (rep_nodes, cnt + 1)
                            placed = True
                            break
                    
                    if not placed:
                        # New topology
                        buckets[sig].append((current, 1))
                    continue
                
                if len(current) > order:
                    continue
                
                # Expand by adding one neighbor from frontier
                for nxt in list(fr):
                    new_set = current | {nxt}
                    new_frontier = (fr | set(tet_graph.neighbors(nxt))) - new_set
                    # Maintain anchor constraint
                    new_frontier = {x for x in new_frontier if x >= anchor}
                    stack.append((new_set, new_frontier))
        
        # Compute correct multiplicities for each representative
        print(f"  Computing multiplicities for {sum(len(g) for g in buckets.values())} topologies...")
        
        for sig_groups in buckets.values():
            for rep_nodes, _ in sig_groups:
                # Compute correct multiplicity using automorphism groups
                mult = compute_cluster_multiplicity(
                    rep_nodes, tet_graph, lattice_graph, tetrahedra,
                    subunit=subunit, verbose=verbose
                )
                
                # Sanity check
                if not verify_multiplicity_sanity(order, mult, subunit):
                    print(f"WARNING: Cluster with {order} tetrahedra has suspicious multiplicity {mult}")
                
                distinct_clusters.append(sorted(rep_nodes))
                multiplicities.append(mult)
        
        num_topologies = len([m for m in multiplicities if len(distinct_clusters[multiplicities.index(m)]) == order])
        print(f"  Found {num_topologies} distinct topologies for order {order}")
        
        # Print multiplicity statistics
        order_mults = [m for i, m in enumerate(multiplicities) if len(distinct_clusters[i]) == order]
        if order_mults:
            print(f"    Multiplicity range: [{min(order_mults):.6f}, {max(order_mults):.6f}]")
            print(f"    Mean multiplicity: {np.mean(order_mults):.6f}")

    return distinct_clusters, multiplicities


def extract_cluster_info(lattice, pos, tetrahedra, cluster):
    """
    Extract detailed information about a cluster.
    
    Args:
        lattice: Site-level NetworkX graph
        pos: Dictionary of site positions
        tetrahedra: List of all tetrahedra
        cluster: List of tetrahedron indices
        
    Returns:
        Dictionary with cluster information
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
        f.write(f"# Multiplicity (lattice constant): {multiplicity:.12f}\n")
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


def identify_subclusters(distinct_clusters, tet_graph):
    """
    Identify all topological subclusters for each distinct cluster and their multiplicities.
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
    """
    with open(f"{output_dir}/subclusters_info.txt", 'w') as f:
        f.write("# Subclusters information for each topologically distinct cluster\n")
        f.write("# Format: Cluster_ID, Order, Multiplicity, Subclusters[(ID, Count), ...]\n\n")
        
        for i, cluster in enumerate(distinct_clusters):
            cluster_id = i + 1
            order = len(cluster)
            multiplicity = multiplicities[i]
            
            subclusters = subclusters_info.get(i, [])
            subcluster_str = ", ".join([f"({j+1}, {count})" for j, count in subclusters])
            
            f.write(f"Cluster {cluster_id} (Order {order}, L(c) = {multiplicity:.12f}):\n")
            if subclusters:
                f.write(f"  Subclusters: {subcluster_str}\n")
            else:
                f.write("  No subclusters (order 1 cluster)\n")
            f.write("\n")


def visualize_cluster(lattice, pos, tetrahedra, cluster, cluster_index, output_dir):
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
    
    # Draw tetrahedra (as transparent faces)
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
    plt.savefig(f'{output_dir}/cluster_{cluster_index}_order_{len(cluster)}.png')
    plt.close()


def main():
    args = parse_arguments()
    max_order = args.max_order
    subunit = args.subunit
    verbose = args.verbose
    
    # Determine lattice size
    # Rule of thumb: torus should be at least 2x the linear size of the largest cluster
    # For pyrochlore, estimate cluster diameter ~ sqrt(order)
    if args.lattice_size > 0:
        L = args.lattice_size
    else:
        # Conservative estimate: L ≥ 2 * (estimated_diameter + 2)
        estimated_diameter = int(np.ceil(np.sqrt(max_order))) + 1
        L = max(6, 2 * estimated_diameter)
    
    print("="*70)
    print("NLCE Cluster Generation for Pyrochlore Lattice")
    print("="*70)
    print(f"Maximum order: {max_order}")
    print(f"Lattice size: {L}×{L}×{L}")
    print(f"Subunit for normalization: {subunit}")
    print(f"Output directory: {args.output_dir}")
    print("="*70)
    
    print(f"\nGenerating pyrochlore lattice (torus) of size {L}×{L}×{L}...")
    lattice, pos, tetrahedra = create_pyrochlore_lattice(L)
    print(f"  Sites: {lattice.number_of_nodes()}")
    print(f"  Edges: {lattice.number_of_edges()}")
    print(f"  Tetrahedra: {len(tetrahedra)}")
    
    print("\nBuilding tetrahedron adjacency graph...")
    tet_graph = build_tetrahedron_graph(tetrahedra)
    print(f"  Tetrahedron graph nodes: {tet_graph.number_of_nodes()}")
    print(f"  Tetrahedron graph edges: {tet_graph.number_of_edges()}")
    
    print(f"\nGenerating topologically distinct clusters up to order {max_order}...")
    print("(This may take some time for large orders)")
    distinct_clusters, multiplicities = generate_clusters(
        tet_graph, lattice, tetrahedra, max_order,
        subunit=subunit, verbose=verbose, pos=pos
    )
    
    # Organize clusters by order
    clusters_by_order = defaultdict(list)
    for i, cluster in enumerate(distinct_clusters):
        order = len(cluster)
        clusters_by_order[order].append((i, cluster, multiplicities[i]))
    
    # Print summary
    print("\n" + "="*70)
    print("CLUSTER STATISTICS")
    print("="*70)
    for order in sorted(clusters_by_order.keys()):
        order_clusters = clusters_by_order[order]
        print(f"Order {order}: {len(order_clusters)} distinct topologies")
        
        # Show multiplicity range
        mults = [m for _, _, m in order_clusters]
        print(f"  Multiplicity range: [{min(mults):.6f}, {max(mults):.6f}]")
    
    # Create output directory
    output_dir = args.output_dir + f"/cluster_info_order_{max_order}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Identify and save subclusters
    print("\nIdentifying subclusters for inclusion-exclusion...")
    subclusters_info = identify_subclusters(distinct_clusters, tet_graph)
    save_subclusters_info(subclusters_info, distinct_clusters, multiplicities, output_dir)
    print(f"  Saved to {output_dir}/subclusters_info.txt")
    
    # Extract and save detailed information for each cluster
    print("\nSaving cluster data files...")
    for i, (cluster, multiplicity) in enumerate(zip(distinct_clusters, multiplicities)):
        cluster_id = i + 1
        order = len(cluster)
        
        if verbose:
            print(f"  Cluster {cluster_id} (order {order}, L(c) = {multiplicity:.6f})")
        
        # Extract detailed information
        cluster_info = extract_cluster_info(lattice, pos, tetrahedra, cluster)
        
        # Save to file
        save_cluster_info(cluster_info, cluster_id, order, multiplicity, output_dir)
        
        # Visualize if requested
        if args.visualize:
            visualize_cluster(lattice, pos, tetrahedra, cluster, cluster_id, output_dir)
    
    print(f"\nAll data saved to {output_dir}/")
    
    if args.visualize:
        print(f"Visualizations saved as {output_dir}/cluster_*.png")
    
    print("\n" + "="*70)
    print("DONE")
    print("="*70)


if __name__ == "__main__":
    main()
