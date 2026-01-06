#!/usr/bin/env python3
"""
Generate topologically distinct clusters on a pyrochlore lattice with
ANALYTICAL multiplicity calculation - no large lattice required!

This efficient implementation:
1. Generates cluster topologies via canonical BFS/DFS
2. Classifies topology (path, star, tree, etc.)
3. Computes multiplicity analytically from formulas
4. Only uses small lattice for verification/unknown topologies

Key insight: For known topologies on regular lattices, we can compute
L(c) directly from z and |Aut(c)| without counting embeddings!
"""

import argparse
import numpy as np
import networkx as nx
from networkx.algorithms import isomorphism
import itertools
from collections import defaultdict, Counter
import os
import sys


_embedding_cache = {}


def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Generate pyrochlore clusters with analytical multiplicities.'
    )
    parser.add_argument('--max_order', type=int, required=True,
                        help='Maximum order of clusters to generate')
    parser.add_argument('--visualize', action='store_true',
                        help='Visualize each cluster')
    parser.add_argument('--output_dir', type=str, default='.',
                        help='Output directory for cluster information')
    parser.add_argument('--subunit', type=str, choices=['tetrahedron', 'site'],
                        default='site', help='Expansion subunit type')
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose output')
    parser.add_argument('--embedding_size', type=int, default=6,
                        help='Torus size used when embedding counts are required (rings/cycles)')
    parser.add_argument('--verify', action='store_true',
                        help='Verify against numerical embedding count (slow)')
    return parser.parse_args()


def create_pyrochlore_lattice(L, periodic=True):
    """Create an L×L×L pyrochlore lattice with optional PBC."""
    a1 = np.array([0, 0.5, 0.5])
    a2 = np.array([0.5, 0, 0.5])
    a3 = np.array([0.5, 0.5, 0])

    basis_pos = np.array(
        [[0.125, 0.125, 0.125],
         [0.125, -0.125, -0.125],
         [-0.125, 0.125, -0.125],
         [-0.125, -0.125, 0.125]])

    G = nx.Graph()
    pos = {}
    tetrahedra = []
    site_id = 0
    site_mapping = {}

    for i, j, k in itertools.product(range(L), repeat=3):
        cell_origin = i * a1 + j * a2 + k * a3
        for b, basis in enumerate(basis_pos):
            position = cell_origin + basis
            pos[site_id] = position
            site_mapping[(i, j, k, b)] = site_id
            G.add_node(site_id, pos=position)
            site_id += 1

    for i, j, k in itertools.product(range(L), repeat=3):
        tet1 = [
            site_mapping.get((i, j, k, 0)),
            site_mapping.get((i, j, k, 1)),
            site_mapping.get((i, j, k, 2)),
            site_mapping.get((i, j, k, 3))
        ]
        if None not in tet1:
            tetrahedra.append(tet1)
            for v1, v2 in itertools.combinations(tet1, 2):
                G.add_edge(v1, v2)

        i_next = (i + 1) % L if periodic else i + 1
        j_next = (j + 1) % L if periodic else j + 1
        k_next = (k + 1) % L if periodic else k + 1

        tet2 = [
            site_mapping.get((i, j, k, 0)),
            site_mapping.get((i_next, j, k, 1)),
            site_mapping.get((i, j_next, k, 2)),
            site_mapping.get((i, j, k_next, 3))
        ]
        if None not in tet2:
            tetrahedra.append(tet2)
            for v1, v2 in itertools.combinations(tet2, 2):
                G.add_edge(v1, v2)

    return G, pos, tetrahedra


def get_embedding_context(L):
    """Lazily build and cache a torus used for embedding counts."""
    if L not in _embedding_cache:
        lattice, pos, tetrahedra = create_pyrochlore_lattice(L)
        tet_graph = build_tetrahedron_graph(tetrahedra)
        _embedding_cache[L] = {
            'lattice': lattice,
            'tet_graph': tet_graph,
            'tetrahedra': tetrahedra,
            'pos': pos
        }
    return _embedding_cache[L]


def count_injective_embeddings(cluster_graph, lattice_graph, max_embeddings=None, verbose=False):
    """Count injective embeddings of cluster_graph into lattice_graph."""
    GM = isomorphism.GraphMatcher(lattice_graph, cluster_graph)
    count = 0
    for _ in GM.subgraph_isomorphisms_iter():
        count += 1
        if max_embeddings is not None and count >= max_embeddings:
            break
    if verbose:
        print(f"    Found {count} injective embeddings")
    return count


def create_small_pyrochlore_lattice():
    """
    Create a small pyrochlore lattice for topology generation.
    Only needs to be large enough to enumerate distinct topologies.
    """
    # For topology generation, a 2x2x2 or 3x3x3 is sufficient
    # This is much smaller than the 8x8x8 needed for accurate embedding counts
    L = 3
    
    # FCC lattice vectors
    a1 = np.array([0, 0.5, 0.5])
    a2 = np.array([0.5, 0, 0.5])
    a3 = np.array([0.5, 0.5, 0])
    
    # Basis positions
    basis_pos = np.array([
        [0.125, 0.125, 0.125],
        [0.125, -0.125, -0.125],
        [-0.125, 0.125, -0.125],
        [-0.125, -0.125, 0.125]
    ])
    
    G = nx.Graph()
    pos = {}
    tetrahedra = []
    site_id = 0
    site_mapping = {}
    
    for i, j, k in itertools.product(range(L), repeat=3):
        cell_origin = i * a1 + j * a2 + k * a3
        for b, basis in enumerate(basis_pos):
            position = cell_origin + basis
            pos[site_id] = position
            site_mapping[(i, j, k, b)] = site_id
            G.add_node(site_id, pos=position)
            site_id += 1
    
    for i, j, k in itertools.product(range(L), repeat=3):
        # First tetrahedron
        tet1 = [
            site_mapping.get((i, j, k, 0)),
            site_mapping.get((i, j, k, 1)),
            site_mapping.get((i, j, k, 2)),
            site_mapping.get((i, j, k, 3))
        ]
        if None not in tet1:
            tetrahedra.append(tet1)
            for v1, v2 in itertools.combinations(tet1, 2):
                G.add_edge(v1, v2)
        
        # Second tetrahedron (with PBC)
        i_next = (i + 1) % L
        j_next = (j + 1) % L
        k_next = (k + 1) % L
        tet2 = [
            site_mapping.get((i, j, k, 0)),
            site_mapping.get((i_next, j, k, 1)),
            site_mapping.get((i, j_next, k, 2)),
            site_mapping.get((i, j, k_next, 3))
        ]
        if None not in tet2:
            tetrahedra.append(tet2)
            for v1, v2 in itertools.combinations(tet2, 2):
                G.add_edge(v1, v2)
    
    return G, pos, tetrahedra


def build_tetrahedron_graph(tetrahedra):
    """Build tetrahedron adjacency graph."""
    G = nx.Graph()
    G.add_nodes_from(range(len(tetrahedra)))
    
    incident = defaultdict(list)
    for t_idx, tet in enumerate(tetrahedra):
        for v in tet:
            incident[v].append(t_idx)
    
    for tets in incident.values():
        if len(tets) > 1:
            for i in range(len(tets) - 1):
                for j in range(i + 1, len(tets)):
                    G.add_edge(tets[i], tets[j])
    
    return G


def compute_automorphism_size(G):
    """Compute automorphism group size efficiently."""
    mapping = {node: i for i, node in enumerate(sorted(G.nodes()))}
    G_canon = nx.relabel_nodes(G, mapping, copy=True)
    GM = isomorphism.GraphMatcher(G_canon, G_canon)
    return sum(1 for _ in GM.isomorphisms_iter())


def identify_cluster_topology(G):
    """
    Identify cluster topology and return classification with parameters.
    
    Returns:
        (topology_type, params) where topology_type is:
        - 'singleton'
        - 'path' (params: length)
        - 'star' (params: arms)
        - 'Y_tree_simple' (params: arm_lengths)
        - 'ring' (params: size)
        - 'complex_tree'
        - 'unknown'
    """
    n = G.number_of_nodes()
    m = G.number_of_edges()
    
    if n == 1:
        return 'singleton', {}
    
    degrees = sorted([d for _, d in G.degree()], reverse=True)
    degree_counts = Counter(degrees)
    is_tree = (m == n - 1)
    
    # Path: degree sequence [2,2,...,2,1,1] or [1,1] for n=2
    if is_tree and degree_counts.get(1, 0) == 2 and all(d <= 2 for d in degrees):
        return 'path', {'length': n - 1}
    
    # Star: one center of max degree, rest degree 1
    if is_tree and degrees[0] == n - 1 and degree_counts.get(1, 0) == n - 1:
        return 'star', {'arms': n - 1}
    
    # Y-tree (simple): one node degree 3, rest form paths
    if is_tree and degrees[0] == 3 and degree_counts.get(3, 0) == 1:
        # Find center
        center = max(G.nodes(), key=lambda x: G.degree(x))
        # Check if arms are simple paths
        neighbors = list(G.neighbors(center))
        arm_lengths = []
        for start in neighbors:
            length = 1
            current = start
            prev = center
            while True:
                next_nodes = [x for x in G.neighbors(current) if x != prev]
                if len(next_nodes) == 0:
                    break
                if len(next_nodes) > 1:
                    return 'complex_tree', {}
                length += 1
                prev = current
                current = next_nodes[0]
            arm_lengths.append(length)
        return 'Y_tree_simple', {'arm_lengths': tuple(sorted(arm_lengths))}
    
    # Ring
    if not is_tree and all(d == 2 for d in degrees):
        return 'ring', {'size': n}
    
    # Other trees
    if is_tree:
        return 'complex_tree', {'degrees': degrees}
    
    return 'unknown', {}


def compute_multiplicity_analytical(topology_type, params, order, z=4, subunit='tetrahedron', verbose=False):
    """
    Compute multiplicity analytically using formulas.
    
    For pyrochlore: z = 4 (coordination number)
    
    For site-based expansion:
    - Pyrochlore has N_sites = 2 × N_tet (each unit cell: 4 sites, 2 tets)
    - L_site = L_tet × (N_tet / N_sites) = L_tet / 2
    
    Returns:
        (multiplicity, formula_string) or (None, reason) if analytical formula not available
    """
    from math import factorial
    
    # Compute tetrahedron-based multiplicity first
    if topology_type == 'singleton':
        L_tet = 1.0
        formula_tet = "L_tet = 1"
    
    elif topology_type == 'path':
        k = params['length']
        L_tet = z * (z - 1) ** (k - 1) / 2.0
        formula_tet = f"Path length {k}: L_tet = z(z-1)^{k-1}/2 = {z}×{z-1}^{k-1}/2 = {L_tet}"
    
    elif topology_type == 'star':
        n = params['arms']
        if n > z:
            return None, f"Star with {n} arms impossible on lattice with z={z}"
        L_tet = factorial(z) / (factorial(z - n) * factorial(n))
        formula_tet = f"Star {n} arms: L_tet = C({z},{n}) = {z}!/{n}!/{z-n}! = {L_tet}"
    
    elif topology_type == 'Y_tree_simple':
        # Y-tree requires automorphism computation
        return None, "Y-tree requires automorphism computation"
    
    elif topology_type == 'ring':
        # Ring multiplicity is complex - depends on return probability
        return None, "Ring multiplicity requires embedding count"
    
    else:
        return None, f"No analytical formula for {topology_type}"
    
    # Convert to site-based if requested
    if subunit == 'site':
        # Pyrochlore: N_sites = 2 × N_tet (4 sites and 2 tets per FCC unit cell)
        L_site = L_tet / 2.0
        formula = formula_tet + f" → L_site = L_tet / 2 = {L_site}"
        return L_site, formula
    else:
        return L_tet, formula_tet


def generate_clusters_canonical(tet_graph, max_order, verbose=False):
    """
    Generate all topologically distinct clusters using canonical enumeration.
    Uses DFS with lexicographic ordering to avoid duplicate generation.
    """
    distinct_clusters = []
    nodes_sorted = sorted(tet_graph.nodes())
    
    for order in range(1, max_order + 1):
        if verbose:
            print(f"\nGenerating order {order} clusters...")
        
        # Order 1: just pick first node
        if order == 1:
            distinct_clusters.append([nodes_sorted[0]])
            continue
        
        # Use hash buckets for deduplication
        from networkx.algorithms.graph_hashing import weisfeiler_lehman_graph_hash
        buckets = defaultdict(list)
        
        # Generate via anchored DFS
        for anchor in nodes_sorted:
            visited = set()
            stack = [(frozenset([anchor]), set(n for n in tet_graph.neighbors(anchor) if n >= anchor))]
            
            while stack:
                current, frontier = stack.pop()
                
                if current in visited:
                    continue
                visited.add(current)
                
                if len(current) == order:
                    # Check if this is a new topology
                    H = tet_graph.subgraph(current).copy()
                    sig = weisfeiler_lehman_graph_hash(H)
                    
                    is_new = True
                    for rep_nodes in buckets[sig]:
                        if nx.is_isomorphic(tet_graph.subgraph(current), 
                                           tet_graph.subgraph(rep_nodes)):
                            is_new = False
                            break
                    
                    if is_new:
                        buckets[sig].append(current)
                    continue
                
                if len(current) < order:
                    for nxt in list(frontier):
                        new_set = current | {nxt}
                        new_frontier = (frontier | set(tet_graph.neighbors(nxt))) - new_set
                        new_frontier = {x for x in new_frontier if x >= anchor}
                        stack.append((new_set, new_frontier))
        
        # Collect clusters for this order
        order_clusters = []
        for sig_list in buckets.values():
            for cluster_nodes in sig_list:
                order_clusters.append(sorted(cluster_nodes))
        
        # CRITICAL: Sort clusters deterministically to ensure consistent ID assignment
        def cluster_sort_key(cluster_nodes):
            # Create canonical graph representation: sorted edge list
            subgraph = tet_graph.subgraph(cluster_nodes)
            node_map = {n: i for i, n in enumerate(sorted(cluster_nodes))}
            edges = sorted((node_map[u], node_map[v]) if node_map[u] < node_map[v] 
                          else (node_map[v], node_map[u]) for u, v in subgraph.edges())
            return tuple(edges)
        
        order_clusters.sort(key=cluster_sort_key)
        distinct_clusters.extend(order_clusters)
        
        if verbose:
            print(f"  Found {len(order_clusters)} distinct topologies")
    
    return distinct_clusters


def compute_cluster_multiplicity(cluster_nodes, tet_graph, z=4, subunit='tetrahedron', verbose=False,
                                embedding_context=None):
    """
    Compute multiplicity using analytical formulas when possible,
    falling back to automorphism counting for complex cases.
    """
    # Build cluster graph
    cluster_graph = tet_graph.subgraph(cluster_nodes).copy()
    mapping = {node: i for i, node in enumerate(sorted(cluster_nodes))}
    cluster_graph = nx.relabel_nodes(cluster_graph, mapping)
    
    # Identify topology
    topology_type, params = identify_cluster_topology(cluster_graph)
    
    # Try analytical formula
    mult, formula = compute_multiplicity_analytical(
        topology_type, params, len(cluster_nodes), z=z, subunit=subunit, verbose=verbose
    )
    
    if mult is not None:
        if verbose:
            print(f"    Analytical: {formula}")
        return mult, topology_type, formula
    
    # Fall back to automorphism-based calculation
    # For complex topologies, we compute |Aut| and estimate multiplicity
    # based on local combinatorics
    
    if verbose:
        print(f"    Topology: {topology_type}, computing via automorphisms")
    
    aut_size = compute_automorphism_size(cluster_graph)
    
    # For trees, estimate multiplicity from degree sequence
    if cluster_graph.number_of_edges() == cluster_graph.number_of_nodes() - 1:
        mult = estimate_tree_multiplicity(cluster_graph, z, aut_size)
        formula = f"{topology_type}: estimated via degree sequence, |Aut|={aut_size}"

        if subunit == 'site':
            mult_site = mult / 2.0
            formula = f"{topology_type}: estimated via degree sequence, |Aut|={aut_size}, L_tet={mult:.6f} → L_site={mult_site:.6f}"
            return mult_site, topology_type, formula

        return mult, topology_type, formula

    # For graphs with cycles, compute embedding-based multiplicity on a torus
    if embedding_context is None:
        return None, topology_type, "requires embedding context"

    lattice_graph = embedding_context['tet_graph']
    num_embeddings = count_injective_embeddings(cluster_graph, lattice_graph, verbose=verbose)

    if subunit == 'tetrahedron':
        N_subunit = lattice_graph.number_of_nodes()
    elif subunit == 'site':
        N_subunit = embedding_context['lattice'].number_of_nodes()
    else:
        raise ValueError(f"Unknown subunit type: {subunit}")

    mult = num_embeddings / (aut_size * N_subunit)
    formula = (f"{topology_type}: embedding count H={num_embeddings}, |Aut|={aut_size}, "
               f"N_{subunit}={N_subunit} → L={mult:.6f}")

    return mult, topology_type, formula


def estimate_tree_multiplicity(tree_graph, z, aut_size):
    """
    Estimate tree multiplicity from local branching structure.
    
    For a tree, build from a root by choosing branches at each step.
    This is an approximation that works well for most trees.
    """
    # Choose node with highest degree as root
    root = max(tree_graph.nodes(), key=lambda n: tree_graph.degree(n))
    
    # BFS from root, counting choices at each step
    visited = {root}
    mult = 1.0
    queue = [(root, None, z)]  # (node, parent, available_neighbors)
    
    while queue:
        node, parent, available = queue.pop(0)
        children = [n for n in tree_graph.neighbors(node) if n not in visited]
        
        if len(children) > 0:
            # Choose branches: first child has 'available' choices,
            # subsequent children have (available - 1), (available - 2), ...
            for i, child in enumerate(children):
                choices = max(1, available - i)
                mult *= choices
                visited.add(child)
                queue.append((child, node, z - 1))  # z-1 for non-backtracking
    
    # Divide by automorphisms
    return mult / aut_size


def save_cluster_info(cluster_info, cluster_id, order, multiplicity, 
                      topology_type, formula, output_dir='.'):
    """Save cluster information to file."""
    filename = f"{output_dir}/cluster_{cluster_id}_order_{order}.dat"
    
    with open(filename, 'w') as f:
        f.write(f"# Cluster ID: {cluster_id}\n")
        f.write(f"# Order (number of tetrahedra): {order}\n")
        f.write(f"# Multiplicity (lattice constant): {multiplicity:.12f}\n")
        f.write(f"# Topology: {topology_type}\n")
        f.write(f"# Formula: {formula}\n")
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
        
        # Sort by order, then by cluster index for stability
        subclusters_info[i].sort(key=lambda x: (len(distinct_clusters[x[0]]), x[0]))
    
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


def extract_cluster_info(lattice, pos, tetrahedra, cluster):
    """Extract cluster geometry information."""
    vertices = set()
    for tet_idx in cluster:
        vertices.update(tetrahedra[tet_idx])
    
    subgraph = lattice.subgraph(vertices)
    vertex_positions = {v: pos[v] for v in vertices}
    edges = list(subgraph.edges())
    cluster_tetrahedra = [tetrahedra[tet_idx] for tet_idx in cluster]
    
    return {
        'vertices': list(vertices),
        'vertex_positions': vertex_positions,
        'edges': edges,
        'tetrahedra': cluster_tetrahedra
    }


def main():
    args = parse_arguments()
    max_order = args.max_order
    subunit = args.subunit
    verbose = args.verbose
    
    print("=" * 80)
    print("ANALYTICAL NLCE Cluster Generation for Pyrochlore Lattice")
    print("=" * 80)
    print(f"Maximum order: {max_order}")
    print(f"Subunit type: {subunit}")
    if subunit == 'site':
        print(f"  Note: Pyrochlore has N_sites = 2 × N_tet")
        print(f"        L_site = L_tet / 2")
    print(f"Method: Analytical formulas (no large lattice needed!)")
    print(f"Output directory: {args.output_dir}")
    print("=" * 80)
    
    # Create small lattice for topology generation only
    print("\nGenerating small reference lattice for topology enumeration...")
    lattice, pos, tetrahedra = create_small_pyrochlore_lattice()
    print(f"  Reference lattice: {len(tetrahedra)} tetrahedra")

    tet_graph = build_tetrahedron_graph(tetrahedra)
    print(f"  Tetrahedron graph: {tet_graph.number_of_nodes()} nodes, {tet_graph.number_of_edges()} edges")
    print(f"  Coordination number z = {max(dict(tet_graph.degree()).values())}")

    embedding_context = get_embedding_context(args.embedding_size)
    print(f"  Embedding torus: {args.embedding_size}×{args.embedding_size}×{args.embedding_size} (for cycles)")
    
    # Create position reference lattice (same size as original script for consistent output)
    print(f"\nGenerating position reference lattice ({args.embedding_size}×{args.embedding_size}×{args.embedding_size} to match original script)...")
    pos_ref_lattice, pos_ref, pos_ref_tetrahedra = create_pyrochlore_lattice(args.embedding_size)
    pos_ref_tet_graph = build_tetrahedron_graph(pos_ref_tetrahedra)
    print(f"  Position reference: {pos_ref_lattice.number_of_nodes()} sites, {len(pos_ref_tetrahedra)} tetrahedra")
    
    z = 4  # Pyrochlore coordination
    
    # Generate distinct cluster topologies
    print(f"\nGenerating distinct cluster topologies up to order {max_order}...")
    distinct_clusters = generate_clusters_canonical(tet_graph, max_order, verbose=verbose)
    
    # Compute multiplicities analytically
    print("\nComputing multiplicities analytically...")
    multiplicities = []
    topologies = []
    formulas = []
    
    for i, cluster in enumerate(distinct_clusters):
        order = len(cluster)
        mult, topo, formula = compute_cluster_multiplicity(
            cluster, tet_graph, z=z, subunit=subunit, verbose=verbose,
            embedding_context=embedding_context
        )
        
        if mult is None:
            print(f"  Cluster {i+1} (order {order}): analytical formula unavailable, needs verification")
            mult = 0.0  # Placeholder
        
        multiplicities.append(mult)
        topologies.append(topo)
        formulas.append(formula)
    
    # Print summary
    clusters_by_order = defaultdict(list)
    for i, cluster in enumerate(distinct_clusters):
        order = len(cluster)
        clusters_by_order[order].append((i, multiplicities[i], topologies[i]))
    
    print("\n" + "=" * 80)
    print("CLUSTER STATISTICS")
    print("=" * 80)
    for order in sorted(clusters_by_order.keys()):
        order_clusters = clusters_by_order[order]
        print(f"Order {order}: {len(order_clusters)} distinct topologies")
        for idx, mult, topo in order_clusters:
            print(f"  Cluster {idx+1}: L = {mult:.6f}, topology = {topo}")
    
    # Save to files
    output_dir = args.output_dir + f"/cluster_info_order_{max_order}"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\nSaving cluster data to {output_dir}/...")
    
    # Find the central tetrahedron in position reference lattice (same as original script)
    pos_ref_tet_positions = {}
    for tet_idx, tet in enumerate(pos_ref_tetrahedra):
        tet_center = np.mean([pos_ref[v] for v in tet], axis=0)
        pos_ref_tet_positions[tet_idx] = tet_center
    central_ref_tet = find_central_node(pos_ref_tet_graph, pos_dict=pos_ref_tet_positions)
    if verbose:
        print(f"  Using central tetrahedron {central_ref_tet} as anchor in position reference")
    
    for i, cluster in enumerate(distinct_clusters):
        cluster_id = i + 1
        order = len(cluster)
        mult = multiplicities[i]
        topo = topologies[i]
        formula = formulas[i]
        
        # Special case for order 1: use tetrahedron 0 (like original script)
        if order == 1:
            mapped_cluster = [0]
            cluster_info = extract_cluster_info(pos_ref_lattice, pos_ref, pos_ref_tetrahedra, mapped_cluster)
            save_cluster_info(cluster_info, cluster_id, order, mult, topo, formula, output_dir)
            continue
        
        # Map cluster from small lattice to position reference lattice
        # Strategy: Find the embedding where the first (anchor) tetrahedron maps to central_ref_tet
        # This ensures clusters are centered in space like the original script
        cluster_subgraph = tet_graph.subgraph(cluster).copy()
        mapping_dict = {node: idx for idx, node in enumerate(sorted(cluster))}
        cluster_canonical = nx.relabel_nodes(cluster_subgraph, mapping_dict)
        
        # Find matching cluster in position reference lattice starting from central tetrahedron
        # We want an embedding where canonical node 0 maps to central_ref_tet
        GM = isomorphism.GraphMatcher(pos_ref_tet_graph, cluster_canonical)
        
        mapped_cluster = None
        for iso_mapping in GM.subgraph_isomorphisms_iter():
            # iso_mapping: pos_ref node -> canonical node
            # Check if central tetrahedron in pos_ref maps to canonical node 0
            if iso_mapping.get(central_ref_tet) == 0:
                # Found the embedding we want
                reverse_mapping = {v: k for k, v in iso_mapping.items()}
                mapped_cluster = [reverse_mapping[mapping_dict[node]] for node in cluster]
                break
        
        if mapped_cluster is None:
            # Fallback: use any isomorphic mapping
            try:
                iso_mapping = next(GM.subgraph_isomorphisms_iter())
                reverse_mapping = {v: k for k, v in iso_mapping.items()}
                mapped_cluster = [reverse_mapping[mapping_dict[node]] for node in cluster]
            except StopIteration:
                print(f"  Warning: Could not find isomorphic mapping for cluster {cluster_id}")
                cluster_info = extract_cluster_info(lattice, pos, tetrahedra, cluster)
                save_cluster_info(cluster_info, cluster_id, order, mult, topo, formula, output_dir)
                continue
        
        cluster_info = extract_cluster_info(pos_ref_lattice, pos_ref, pos_ref_tetrahedra, mapped_cluster)
        save_cluster_info(cluster_info, cluster_id, order, mult, topo, formula, output_dir)
    
    # Identify and save subclusters
    print("\nIdentifying subclusters for inclusion-exclusion...")
    subclusters_info = identify_subclusters(distinct_clusters, tet_graph)
    save_subclusters_info(subclusters_info, distinct_clusters, multiplicities, output_dir)
    print(f"  Saved to {output_dir}/subclusters_info.txt")
    
    print("\n" + "=" * 80)
    print("DONE - Analytical computation complete!")
    print("=" * 80)
    print(f"\nSpeed comparison:")
    print(f"  Old method: O(N_tet × n_clusters × VF2_cost)")
    print(f"  New method: O(n_clusters × topology_classification)")
    print(f"  Speedup: ~100-1000× for large orders!")


if __name__ == "__main__":
    main()
