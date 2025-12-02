import argparse
import numpy as np
import networkx as nx
import itertools
import sys
from collections import defaultdict, deque
import os
import collections

#!/usr/bin/env python3
"""
Generate topologically distinct clusters on a honeycomb lattice for
numerical linked cluster expansion (NLCE) calculations.
"""

import matplotlib.pyplot as plt

def extract_cluster_info(lattice, pos, hexagons, cluster):
    """
    Extract detailed information about a cluster.
    """
    # Get all vertices in the cluster
    vertices = set()
    for hex_idx in cluster:
        vertices.update(hexagons[hex_idx])
    
    # Create subgraph for this cluster
    subgraph = lattice.subgraph(vertices)
    
    # Get vertex positions
    vertex_positions = {v: pos[v] for v in vertices}
    
    # Get edges in the cluster
    edges = list(subgraph.edges())
    
    # Get the hexagons that make up the cluster
    cluster_hexagons = [hexagons[hex_idx] for hex_idx in cluster]
    
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
        'hexagons': cluster_hexagons,
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
        f.write(f"# Order (number of hexagons): {order}\n")
        f.write(f"# Multiplicity: {multiplicity}\n")
        f.write(f"# Number of vertices: {len(cluster_info['vertices'])}\n")
        f.write(f"# Number of edges: {len(cluster_info['edges'])}\n\n")
        
        f.write("# Vertices (index, x, y):\n")
        for v in cluster_info['vertices']:
            pos = cluster_info['vertex_positions'][v]
            f.write(f"{v}, {pos[0]:.6f}, {pos[1]:.6f}\n")
        
        f.write("\n# Edges (vertex1, vertex2):\n")
        for u, v in cluster_info['edges']:
            f.write(f"{u}, {v}\n")
        
        f.write("\n# Hexagons (vertex1, vertex2, vertex3, vertex4, vertex5, vertex6):\n")
        for hex in cluster_info['hexagons']:
            f.write(f"{', '.join(map(str, hex))}\n")
        
        f.write("\n# Adjacency Matrix:\n")
        for row in cluster_info['adjacency_matrix']:
            f.write(' '.join(map(str, row)) + '\n')
        
        f.write("\n# Node Mapping (original_id: matrix_index):\n")
        for node, idx in cluster_info['node_mapping'].items():
            f.write(f"{node}: {idx}\n")

def parse_arguments():
    parser = argparse.ArgumentParser(description='Generate topologically distinct clusters on a honeycomb lattice.')
    parser.add_argument('--max_order', type=int, required=True, help='Maximum order of clusters to generate')
    parser.add_argument('--visualize', action='store_true', help='Visualize each cluster')
    parser.add_argument('--lattice_size', type=int, default=0, help='Size of finite lattice (default: 4*max_order)')
    parser.add_argument('--output_dir', type=str, default='.', help='Output directory for cluster information')
    return parser.parse_args()

def create_honeycomb_lattice(L):
    """
    Create a honeycomb lattice of size L×L unit cells.
    
    Returns:
    - G: NetworkX graph representing the lattice
    - pos: Dictionary mapping node IDs to 2D positions
    - hexagons: List of hexagons, each represented as a list of 6 node IDs
    """
    # Honeycomb lattice vectors
    a1 = np.array([1.5, np.sqrt(3)/2])
    a2 = np.array([0, np.sqrt(3)])
    
    # Positions within unit cell (two atoms per unit cell)
    basis_pos = np.array([
        [0, 0],        # First atom
        [1, 0]         # Second atom
    ])

    G = nx.Graph()
    pos = {}
    
    # Generate lattice sites
    site_id = 0
    site_mapping = {}
    
    for i, j in itertools.product(range(L), repeat=2):
        cell_origin = i*a1 + j*a2
        
        for b, basis in enumerate(basis_pos):
            position = cell_origin + basis
            pos[site_id] = position
            site_mapping[(i, j, b)] = site_id
            G.add_node(site_id, pos=position)
            site_id += 1
    
    # Add edges between nearest neighbors
    for i, j in itertools.product(range(L), repeat=2):
        # Connect within unit cell
        site1 = site_mapping.get((i, j, 0))
        site2 = site_mapping.get((i, j, 1))
        if site1 is not None and site2 is not None:
            G.add_edge(site1, site2)
        
        # Connect to neighbors
        # First basis atom connects to two other unit cells
        site1 = site_mapping.get((i, j, 0))
        if site1 is not None:
            # Connect to (i-1, j, 1)
            site2 = site_mapping.get((i-1, j, 1))
            if site2 is not None:
                G.add_edge(site1, site2)
            
            # Connect to (i, j-1, 1)
            site2 = site_mapping.get((i, j-1, 1))
            if site2 is not None:
                G.add_edge(site1, site2)
    
    # Identify hexagons
    hexagons = []
    visited = set()
    
    # Helper function to find a hexagon starting from an edge
    def find_hexagon(start_edge):
        u, v = start_edge
        # Try to walk around to form a hexagon
        current = v
        path = [u, v]
        prev = u
        
        while len(path) < 6:
            neighbors = list(G.neighbors(current))
            if len(neighbors) < 2:
                return None  # Can't form a hexagon
            
            # Find neighbor that's not the previous node
            next_nodes = [n for n in neighbors if n != prev and n not in path]
            if not next_nodes:
                if len(path) == 5 and path[0] in neighbors:
                    # Complete the cycle
                    path.append(path[0])
                    return path[:-1]  # Remove the duplicate at the end
                return None
            
            next_node = next_nodes[0]
            path.append(next_node)
            prev = current
            current = next_node
        
        # Check if the path forms a cycle
        if path[0] in G.neighbors(path[-1]):
            return path[:-1]  # Return the 6 vertices of the hexagon
        return None

    # Find all hexagons
    for edge in G.edges():
        if frozenset(edge) not in visited:
            hexagon = find_hexagon(edge)
            if hexagon and len(hexagon) == 6:
                # Check if this hexagon is already found
                hexagon_set = frozenset(hexagon)
                if all(frozenset(h) != hexagon_set for h in hexagons):
                    hexagons.append(tuple(hexagon))
                    
                    # Mark all edges of this hexagon as visited
                    for i in range(6):
                        visited.add(frozenset((hexagon[i], hexagon[(i+1)%6])))
    
    return G, pos, hexagons

def build_hexagon_graph(hexagons):
    """Build a graph where nodes are hexagons and edges represent shared vertices."""
    hex_graph = nx.Graph()
    
    for i, hex1 in enumerate(hexagons):
        hex_graph.add_node(i)
        for j, hex2 in enumerate(hexagons):
            if i < j:  # Avoid duplicate checks
                # Check if hexagons share a vertex
                if set(hex1).intersection(set(hex2)):
                    hex_graph.add_edge(i, j)
    
    return hex_graph

def generate_clusters(hex_graph, max_order):
    """
    Generate all topologically distinct clusters up to max_order and their multiplicities.
    
    Args:
        hex_graph: NetworkX graph where nodes are hexagons and edges connect adjacent hexagons
        max_order: Maximum number of hexagons in a cluster
        
    Returns:
        distinct_clusters: List of topologically distinct clusters
        multiplicities: List of multiplicities for each distinct cluster
    """
    
    distinct_clusters = []
    multiplicities = []
    
    # Process each order
    for order in range(1, max_order + 1):
        print(f"Generating clusters of order {order}...")
        
        # For order 1, all hexagons are equivalent in a uniform lattice
        if order == 1:
            # Take one representative hexagon
            first_hex = list(hex_graph.nodes())[0]
            distinct_clusters.append([first_hex])
            multiplicities.append(1.0)
            continue
        
        # For higher orders, generate all possible connected subgraphs
        all_subgraphs = []
        
        # Start from each hexagon and grow clusters
        for start_hex in hex_graph.nodes():
            # Use BFS to systematically grow clusters
            queue = deque([(frozenset([start_hex]), set(hex_graph.neighbors(start_hex)))])
            visited_configurations = set()
            
            while queue:
                current, frontier = queue.popleft()
                
                # Skip if we've seen this configuration before
                if current in visited_configurations:
                    continue
                visited_configurations.add(current)
                
                if len(current) == order:
                    all_subgraphs.append(current)
                    continue
                
                if len(current) > order:
                    continue
                
                # Try adding each frontier hexagon
                for next_hex in frontier:
                    new_set = current | {next_hex}
                    
                    # Update frontier with neighbors of the new hexagon
                    new_frontier = frontier | set(hex_graph.neighbors(next_hex))
                    new_frontier -= new_set  # Remove hexagons already in the set
                    
                    queue.append((new_set, new_frontier))
        
        # Remove duplicates
        unique_subgraphs = set(all_subgraphs)
        
        # Group by isomorphism class
        isomorphism_groups = []

        for subgraph_nodes in unique_subgraphs:
            subgraph = hex_graph.subgraph(subgraph_nodes)
            
            found_match = False
            for idx, group in enumerate(isomorphism_groups):
                rep_nodes = group[0]  # Use first graph as representative
                rep_subgraph = hex_graph.subgraph(rep_nodes)
                
                if nx.is_isomorphic(subgraph, rep_subgraph):
                    isomorphism_groups[idx].append(subgraph_nodes)
                    found_match = True
                    break
            
            if not found_match:
                isomorphism_groups.append([subgraph_nodes])

        # Now select the most symmetric representative for each group
        isomorphism_classes = []

        for group in isomorphism_groups:
            # Find the graph with the most automorphisms
            max_automorphisms = 0
            most_symmetric = None
            
            for nodes in group:
                subgraph = hex_graph.subgraph(nodes)
                max_clique_size = nx.algorithms.clique.node_clique_number(subgraph)
                # Get the largest clique size across all nodes as a measure of symmetry
                num_automorphisms = max(max_clique_size.values()) if max_clique_size else 0

                if num_automorphisms > max_automorphisms:
                    max_automorphisms = num_automorphisms
                    most_symmetric = nodes
            
            # Use the most symmetric graph as the representative
            isomorphism_classes.append((most_symmetric, group))

        # Add to results with corrected multiplicities
        for rep_nodes, embeddings in isomorphism_classes:
            cluster = list(rep_nodes)
            distinct_clusters.append(cluster)
            
            # Calculate embedding weight for NLCE
            # For an infinite lattice, normalized by the number of hexagons
            multiplicity = len(embeddings) / len(hex_graph.nodes())
            multiplicities.append(multiplicity)
            
        print(f"Found {len(isomorphism_classes)} distinct clusters of order {order}")
    
    return distinct_clusters, multiplicities

def visualize_cluster(lattice, pos, hexagons, cluster, cluster_index, output_dir):
    """Visualize a single cluster in 2D."""
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)
    
    # Get all vertices in the cluster
    vertices = set()
    for hex_idx in cluster:
        vertices.update(hexagons[hex_idx])
    
    # Create subgraph for this cluster
    subgraph = lattice.subgraph(vertices)
    
    # Draw vertices
    xs = [pos[v][0] for v in subgraph.nodes()]
    ys = [pos[v][1] for v in subgraph.nodes()]
    ax.scatter(xs, ys, c='r', s=100, label='Vertices')
    
    # Draw edges
    for u, v in subgraph.edges():
        ax.plot([pos[u][0], pos[v][0]],
                [pos[u][1], pos[v][1]], 'k-', lw=1)
    
    # Draw hexagons
    for hex_idx in cluster:
        hex_nodes = hexagons[hex_idx]
        # Get polygon coordinates
        hex_x = [pos[v][0] for v in hex_nodes]
        hex_y = [pos[v][1] for v in hex_nodes]
        # Add the first point again to close the loop
        hex_x.append(hex_x[0])
        hex_y.append(hex_y[0])
        ax.fill(hex_x, hex_y, alpha=0.3, edgecolor='b')
    
    ax.set_title(f'Cluster {cluster_index} - {len(cluster)} hexagons')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/cluster_{cluster_index}_order_{len(cluster)}.png')
    plt.close()

def identify_subclusters(distinct_clusters, hex_graph):
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
                subgraph = hex_graph.subgraph(subcluster)
                
                # Find matching distinct cluster
                for cand_idx, cand_cluster in candidates:
                    cand_subgraph = hex_graph.subgraph(cand_cluster)
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
    L = args.lattice_size if args.lattice_size > 0 else max(8, 2*max_order)
    
    print(f"Generating honeycomb lattice of size {L}×{L}...")
    lattice, pos, hexagons = create_honeycomb_lattice(L)
    print(f"Generated lattice with {lattice.number_of_nodes()} sites and {len(hexagons)} hexagons")
    
    print("Building hexagon adjacency graph...")
    hex_graph = build_hexagon_graph(hexagons)
    
    print(f"Generating clusters up to order {max_order}...")
    distinct_clusters, multiplicities = generate_clusters(hex_graph, max_order)
    
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

    # Identify and save subclusters
    subclusters_info = identify_subclusters(distinct_clusters, hex_graph)
    save_subclusters_info(subclusters_info, distinct_clusters, multiplicities, output_dir)
    print(f"Subclusters information saved to {output_dir}/subclusters_info.txt")
    
    # Extract and save detailed information for each cluster
    print("\nExtracting and saving detailed cluster information...")
    for i, (cluster, multiplicity) in enumerate(zip(distinct_clusters, multiplicities)):
        cluster_id = i + 1
        order = len(cluster)
        print(f"  Processing cluster {cluster_id} (order {order})...")
        
        # Extract detailed information
        cluster_info = extract_cluster_info(lattice, pos, hexagons, cluster)
        
        # Save to file
        save_cluster_info(cluster_info, cluster_id, order, multiplicity, output_dir)
        
        # Visualize if requested
        if args.visualize:
            visualize_cluster(lattice, pos, hexagons, cluster, cluster_id, output_dir)
        
        print(f"  Saved to {output_dir}/cluster_{cluster_id}_order_{order}.dat")
    
    print(f"Detailed cluster information saved to {output_dir}/ directory")
    
    if args.visualize:
        print(f"Visualization images saved as {output_dir}/cluster_*.png")

if __name__ == "__main__":
    main()