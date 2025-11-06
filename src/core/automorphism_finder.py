# filepath: /home/pc_linux/exact_diagonalization_cpp/src/core/automorphism_finder.py
import numpy as np
from pynauty import Graph, autgrp
from collections import defaultdict, deque
import argparse
import os
import subprocess
import json
import networkx as nx


def read_trans_file(filename):
    """Read vertex weights from Trans.dat file."""
    vertex_weights = {}
    try:
        with open(filename, 'r') as f:
            lines = f.readlines()
            # Skip header lines
            for line in lines:
                line = line.strip()
                if line and not line.startswith('=') and not line.startswith('num'):
                    parts = line.split()
                    if len(parts) >= 4:
                        vertex_type = int(parts[0])
                        vertex_id = int(parts[1])
                        weight_real = float(parts[2])
                        weight_imag = float(parts[3])
                        vertex_weights[vertex_id] = (vertex_type, weight_real, weight_imag)
    except FileNotFoundError:
        raise FileNotFoundError(f"Trans.dat file not found: {filename}")
    except Exception as e:
        raise RuntimeError(f"Error reading Trans.dat file: {e}")
    return vertex_weights

def read_interall_file(filename):
    """Read edge information from InterAll.dat file."""
    edges = []
    try:
        with open(filename, 'r') as f:
            lines = f.readlines()
            # Skip header lines
            for line in lines:
                line = line.strip()
                if line and not line.startswith('=') and not line.startswith('num'):
                    parts = line.split()
                    if len(parts) >= 6:
                        type1 = int(parts[0])
                        vertex1 = int(parts[1])
                        type2 = int(parts[2])
                        vertex2 = int(parts[3])
                        weight_real = float(parts[4])
                        weight_imag = float(parts[5])
                        edges.append({
                            'vertex1': vertex1,
                            'vertex2': vertex2,
                            'type1': type1,
                            'type2': type2,
                            'weight': (weight_real, weight_imag)
                        })
    except FileNotFoundError:
        raise FileNotFoundError(f"InterAll.dat file not found: {filename}")
    except Exception as e:
        raise RuntimeError(f"Error reading InterAll.dat file: {e}")
    return edges

def _round_tuple(t, decimals=8):
    """Round all floats in a tuple to given decimals."""
    return tuple(round(x, decimals) for x in t)

def _edge_label(edge, decimals=8):
    """Create a stable, undirected edge label from edge record."""
    # Ensure the edge label is independent of direction
    t1, t2 = edge['type1'], edge['type2']
    w = _round_tuple(edge['weight'], decimals)
    tmin, tmax = (t1, t2) if t1 <= t2 else (t2, t1)
    return (tmin, tmax, w)

def compute_vertex_colors(vertex_weights, edges, decimals=8, wl_iterations=10):
    """Compute vertex colors using Weisfeiler-Lehman refinement with edge/vertex labels.
    
    Args:
        vertex_weights: Dictionary of vertex_id -> (type, weight_real, weight_imag)
        edges: List of edge dictionaries
        decimals: Number of decimal places for rounding (default: 8)
        wl_iterations: Maximum WL refinement iterations (default: 10, increased from 5)
    
    Returns:
        Dictionary mapping vertex_id -> color (integer)
    """
    # Collect all vertices that appear in either Trans.dat or edges
    vertex_ids = set(vertex_weights.keys())
    for e in edges:
        vertex_ids.add(e['vertex1'])
        vertex_ids.add(e['vertex2'])
    vertex_ids = sorted(vertex_ids)

    # Build adjacency with edge labels (preserve multiplicity)
    adj_list = {v: [] for v in vertex_ids}
    for edge in edges:
        v1, v2 = edge['vertex1'], edge['vertex2']
        elabel = _edge_label(edge, decimals=decimals)
        if v1 != v2:
            adj_list[v1].append((v2, elabel))
            adj_list[v2].append((v1, elabel))

    # Initial vertex labels (rounded to avoid floating noise)
    init_label = {}
    for v in vertex_ids:
        vtype, wre, wim = vertex_weights.get(v, (0, 0.0, 0.0))
        init_label[v] = ('v', int(vtype), round(wre, decimals), round(wim, decimals))

    # WL refinement
    # colors[v] is an integer color id; token[v] is a structural token used to assign ids deterministically
    token = {v: init_label[v] for v in vertex_ids}
    colors = {}
    for it in range(wl_iterations):
        # Assign integer colors deterministically by sorting unique tokens
        unique_tokens = sorted(set(token.values()))
        token_to_id = {tok: i for i, tok in enumerate(unique_tokens)}
        colors_new = {v: token_to_id[token[v]] for v in vertex_ids}

        # Build next iteration tokens
        next_token = {}
        for v in vertex_ids:
            # multiset of neighbor (edge_label, neighbor_color)
            neigh = [(lbl, colors_new.get(u, -1)) for (u, lbl) in adj_list[v]]
            neigh.sort()
            next_token[v] = (colors_new[v], tuple(neigh))

        # Check stabilization
        if all(token[v] == next_token[v] for v in vertex_ids):
            colors = colors_new
            break

        token = next_token
        colors = colors_new

    # Final deterministic color compaction
    unique_final = sorted(set(colors.values()))
    final_map = {c: i for i, c in enumerate(unique_final)}
    vertex_colors = {v: final_map[colors[v]] for v in vertex_ids}
    return vertex_colors

def construct_colored_graph(vertex_weights, edges):
    """Construct a colored undirected graph for pynauty with robust vertex indexing.
    
    Returns:
        tuple: (graph, vertex_colors, idx_to_vid, vid_to_idx)
            - graph: pynauty Graph object
            - vertex_colors: dict mapping original vertex_id -> color
            - idx_to_vid: list mapping nauty index -> original vertex_id
            - vid_to_idx: dict mapping original vertex_id -> nauty index
    """
    # Compute WL-refined vertex colors
    vertex_colors = compute_vertex_colors(vertex_weights, edges)

    # Build stable vertex index mapping 0..n-1
    all_vertices = sorted(vertex_colors.keys())
    vid_to_idx = {v: i for i, v in enumerate(all_vertices)}
    idx_to_vid = all_vertices  # This is a list: idx_to_vid[i] gives original vertex_id
    n = len(all_vertices)

    # Build adjacency dictionary in terms of contiguous indices
    adjacency_dict = {i: [] for i in range(n)}
    for edge in edges:
        v1, v2 = edge['vertex1'], edge['vertex2']
        if v1 == v2:
            continue
        if v1 not in vid_to_idx or v2 not in vid_to_idx:
            continue
        i, j = vid_to_idx[v1], vid_to_idx[v2]
        # Avoid parallel edges at nauty level; WL already encoded multiplicity/labels into vertex colors
        if j not in adjacency_dict[i]:
            adjacency_dict[i].append(j)
        if i not in adjacency_dict[j]:
            adjacency_dict[j].append(i)

    # Create vertex coloring as list of sets of indices
    color_to_vertices = defaultdict(list)
    for v, c in vertex_colors.items():
        color_to_vertices[c].append(vid_to_idx[v])

    coloring = [set(sorted(ids)) for _, ids in sorted(color_to_vertices.items(), key=lambda kv: kv[0])]

    # Create pynauty graph (ensure keys are 0..n-1)
    g = Graph(n, directed=False, adjacency_dict=adjacency_dict, vertex_coloring=coloring)
    return g, vertex_colors, idx_to_vid, vid_to_idx


class AutomorphismCliqueAnalyzer:
    """Class for analyzing cliques of compatible automorphisms with caching"""
    
    def __init__(self):
        """Initialize with cache for commutation graph"""
        self._cached_graph = None
        self._cached_automorphisms_hash = None
    
    def do_permutations_commute(self, perm1, perm2):
        """Check if two permutations commute - optimized version"""
        # Quick checks first
        if perm1 is perm2:
            return True
        if len(perm1) != len(perm2):
            return False
        
        # Early exit optimization: check if they commute by testing composition
        # Only check positions that differ from identity or are affected by either permutation
        n = len(perm1)
        
        # Find positions affected by each permutation
        affected1 = {i for i in range(n) if perm1[i] != i}
        affected2 = {i for i in range(n) if perm2[i] != i}
        
        # If they affect disjoint sets of positions, they commute
        if affected1.isdisjoint(affected2):
            return True
        
        # Check commutation only at affected positions
        check_positions = affected1 | affected2
        for i in check_positions:
            if perm1[perm2[i]] != perm2[perm1[i]]:
                return False
                
        return True
    
    def build_commutation_graph(self, automorphisms):
        """Build a graph where nodes are automorphisms and edges connect commuting pairs.
        
        Uses caching to avoid rebuilding the graph multiple times for the same automorphisms.
        """
        # Create a hash of the automorphisms for cache validation
        auto_hash = hash(tuple(tuple(perm) for perm in automorphisms))
        
        # Return cached graph if available
        if self._cached_graph is not None and self._cached_automorphisms_hash == auto_hash:
            return self._cached_graph
        
        n_autos = len(automorphisms)
        print(f"Building commutation graph for {n_autos} automorphisms...")
        G = nx.Graph()
        
        # Add nodes for each automorphism
        for i in range(n_autos):
            G.add_node(i)
        
        # Add edges between commuting automorphisms
        edge_count = 0
        total_pairs = (n_autos * (n_autos - 1)) // 2
        
        # Show progress for large graphs
        show_progress = total_pairs > 10000
        progress_step = total_pairs // 20 if show_progress else total_pairs + 1
        
        pairs_checked = 0
        for i in range(n_autos):
            for j in range(i+1, n_autos):
                if self.do_permutations_commute(automorphisms[i], automorphisms[j]):
                    G.add_edge(i, j)
                    edge_count += 1
                
                pairs_checked += 1
                if show_progress and pairs_checked % progress_step == 0:
                    pct = 100 * pairs_checked / total_pairs
                    print(f"  Progress: {pct:.0f}% ({pairs_checked}/{total_pairs} pairs checked, {edge_count} commuting pairs found)")
        
        print(f"  Completed: Found {edge_count} commuting pairs out of {total_pairs} total pairs")
        
        # Cache the graph
        self._cached_graph = G
        self._cached_automorphisms_hash = auto_hash
        
        return G
    
    def find_maximum_clique(self, automorphisms):
        """Find the maximum clique of commuting automorphisms using NetworkX"""
        # Build the commutation graph (cached)
        comm_graph = self.build_commutation_graph(automorphisms)
        
        print(f"Finding maximum clique...")
        # Use NetworkX to find the maximum clique
        max_clique_indices = list(nx.find_cliques(comm_graph))
        
        # Get the maximum clique by size
        if max_clique_indices:
            max_clique = max(max_clique_indices, key=len)
            print(f"  Maximum clique size: {len(max_clique)}")
            return max_clique
        else:
            return []
    
    def generate_automorphism_graph(self, automorphisms, filename, finder):
        """Generate a DOT file visualizing the automorphism commutation graph"""
        # Use cached graph
        comm_graph = self.build_commutation_graph(automorphisms)
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        # Generate DOT file
        dot_file = f"{filename}/automorphism_graph.dot"
        print(f"Generating DOT file...")
        with open(dot_file, 'w') as f:
            f.write("graph AutomorphismGraph {\n")
            f.write("  node [shape=box];\n")
            
            # Add nodes
            for i, auto in enumerate(automorphisms):
                cycles = finder.permutation_to_cycle_notation(auto)
                label = ' '.join(str(c) for c in cycles) if cycles else 'id'
                f.write(f'  {i} [label="{label}"];\n')
            
            # Add edges
            for i, j in comm_graph.edges():
                f.write(f"  {i} -- {j};\n")
                
            f.write("}\n")
        
        print(f"Generated automorphism graph: {dot_file}")
        print(f"Visualize with: dot -Tpng {dot_file} -o {filename}/automorphism_graph.png")
        
        # Try to generate the image directly
        self.save_graph_image(f"{filename}/automorphism_graph.png", dot_file)
        
    def visualize_clique(self, automorphisms, clique, output_dir, finder):
        """Generate a visualization highlighting a specific clique in the graph
        
        Args:
            automorphisms: List of all automorphisms
            clique: List of indices into automorphisms representing the clique
            output_dir: Directory path where output files should be saved
            finder: AutomorphismFinder instance for cycle notation conversion
        """
        # Use cached graph
        comm_graph = self.build_commutation_graph(automorphisms)
        
        # Ensure directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate DOT file in the output directory
        dot_file = os.path.join(output_dir, "max_clique_visualization.dot")
        print(f"Generating clique visualization...")
        with open(dot_file, 'w') as f:
            f.write("graph AutomorphismClique {\n")
            f.write("  node [shape=box];\n")
            
            # Create set for fast lookup
            clique_set = set(clique)
            
            # Add nodes
            for i, auto in enumerate(automorphisms):
                cycles = finder.permutation_to_cycle_notation(auto)
                label = ' '.join(str(c) for c in cycles) if cycles else 'id'
                
                if i in clique_set:
                    # Highlight clique members
                    f.write(f'  {i} [label="{label}", style=filled, fillcolor=lightblue];\n')
                else:
                    f.write(f'  {i} [label="{label}"];\n')
            
            # Add edges
            for i, j in comm_graph.edges():
                if i in clique_set and j in clique_set:
                    # Highlight edges within the clique
                    f.write(f"  {i} -- {j} [color=blue, penwidth=2.0];\n")
                else:
                    f.write(f"  {i} -- {j};\n")
                
            f.write("}\n")
        
        print(f"Generated clique visualization: {dot_file}")
        png_file = os.path.join(output_dir, "max_clique_visualization.png")
        print(f"Visualize with: dot -Tpng {dot_file} -o {png_file}")
        
        # Try to generate the image directly
        self.save_graph_image(png_file, dot_file)
        
    def save_graph_image(self, output_file, dot_file):
        """Generate PNG image from DOT file using Graphviz"""
        try:
            subprocess.run(['dot', '-Tpng', dot_file, '-o', output_file], check=True)
            print(f"Generated graph image: {output_file}")
        except subprocess.CalledProcessError:
            print(f"Failed to generate image. Make sure Graphviz is installed.")
        except FileNotFoundError:
            print(f"Graphviz command 'dot' not found. Please install Graphviz.")

class AutomorphismFinder:
    """Class for finding and analyzing automorphisms"""
    
    def generate_all_automorphisms(self, generators, n):
        """Generate all automorphisms from the given generators using BFS.
        
        Args:
            generators: List of generator permutations from nauty
            n: Number of vertices
            
        Returns:
            List of all automorphisms (permutations)
        """
        if not generators:
            return [list(range(n))]
        
        identity = list(range(n))
        
        # Start with identity and generators
        automorphisms = [identity]
        automorphisms.extend([list(gen) for gen in generators])
        
        # Generate all possible combinations (closure under composition)
        # Use deque for efficient O(1) popleft instead of O(n) pop(0)
        queue = deque(automorphisms)
        seen = {tuple(perm) for perm in automorphisms}
        
        while queue:
            perm1 = queue.popleft()
            for gen in generators:
                # Compose perm1 with generator and generator with perm1 for faster closure
                new_perm1 = [perm1[gen[i]] for i in range(n)]  # perm1 ∘ gen
                if tuple(new_perm1) not in seen:
                    seen.add(tuple(new_perm1))
                    automorphisms.append(new_perm1)
                    queue.append(new_perm1)
                
                new_perm2 = [gen[perm1[i]] for i in range(n)]  # gen ∘ perm1
                if tuple(new_perm2) not in seen:
                    seen.add(tuple(new_perm2))
                    automorphisms.append(new_perm2)
                    queue.append(new_perm2)
        
        return automorphisms
    
    def permutation_to_cycle_notation(self, perm):
        """Convert permutation to cycle notation.
        
        Args:
            perm: List representing a permutation
            
        Returns:
            List of tuples representing non-trivial cycles, or empty list if identity
        """
        n = len(perm)
        visited = [False] * n
        cycles = []
        
        for i in range(n):
            if not visited[i] and perm[i] != i:
                cycle = [i]
                j = perm[i]
                visited[i] = True
                
                while j != i:
                    cycle.append(j)
                    visited[j] = True
                    j = perm[j]
                
                if len(cycle) > 1:
                    cycles.append(tuple(cycle))
        
        return cycles  # Return empty list for identity instead of [(0,)]
    
    def save_automorphisms_json(self, automorphisms, filename):
        """Save automorphisms to JSON file"""
        try:
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            
            with open(filename, 'w') as f:
                json.dump(automorphisms, f, indent=2)
            
            print(f"Saved {len(automorphisms)} automorphisms to {filename}")
        except Exception as e:
            raise RuntimeError(f"Error saving automorphisms to JSON: {e}")


class MaximalAbelianSubgroupFinder:
    """Class for finding minimal generators of abelian subgroups"""
    
    def permutation_order(self, perm):
        """Calculate the order of a permutation"""
        n = len(perm)
        visited = [False] * n
        lcm = 1
        
        for i in range(n):
            if not visited[i]:
                cycle_length = 0
                j = i
                
                while not visited[j]:
                    visited[j] = True
                    j = perm[j]
                    cycle_length += 1
                
                if cycle_length > 1:
                    lcm = self._lcm(lcm, cycle_length)
        
        return lcm
    
    def _gcd(self, a, b):
        """Calculate greatest common divisor"""
        while b:
            a, b = b, a % b
        return a
    
    def _lcm(self, a, b):
        """Calculate least common multiple"""
        return abs(a * b) // self._gcd(a, b)
    
    def compose_permutations(self, perm1, perm2):
        """Compose two permutations: result[i] = perm1[perm2[i]]"""
        return [perm1[perm2[i]] for i in range(len(perm1))]
    
    def is_generated_by(self, element, generators, max_iterations=100000):
        """Check if element can be generated by the given generators using BFS.
        
        Args:
            element: Permutation to check
            generators: List of generator permutations
            max_iterations: Maximum number of elements to generate before giving up
            
        Returns:
            bool: True if element can be generated from generators
        """
        if not generators:
            return element == list(range(len(element)))
        
        n = len(element)
        identity = list(range(n))
        
        # Start with identity
        generated = {tuple(identity)}
        queue = deque([identity])
        
        iterations = 0
        while queue and iterations < max_iterations:
            current = queue.popleft()
            iterations += 1
            
            for gen in generators:
                # Compose with generator
                new_perm = self.compose_permutations(current, gen)
                
                if tuple(new_perm) == tuple(element):
                    return True
                
                if tuple(new_perm) not in generated:
                    generated.add(tuple(new_perm))
                    queue.append(new_perm)
        
        # If we hit the iteration limit, assume not generated
        return False
    
    def find_minimal_generators(self, permutations):
        """Find a minimal set of generators for the given abelian group.
        
        Optimized for abelian groups where we know the structure better.
        """
        if not permutations:
            return []
        
        n = len(permutations[0])
        identity = list(range(n))
        
        # Remove identity if present
        non_identity_perms = [p for p in permutations if p != identity]
        
        if not non_identity_perms:
            return []
        
        print(f"Finding minimal generators for abelian subgroup of size {len(permutations)}...")
        
        # Sort by order (smallest first) to prefer simpler generators
        sorted_perms = sorted(non_identity_perms, key=lambda p: self.permutation_order(p))
        
        generators = []
        
        # For abelian groups, we can be more efficient
        # Track the size of the generated subgroup instead of all elements
        generated_count = 1  # Start with identity
        
        for idx, perm in enumerate(sorted_perms):
            # Check if this permutation is already generated by current generators
            if not self.is_generated_by(perm, generators):
                generators.append(perm)
                # For abelian groups, the new subgroup size is old_size * order(new_generator)
                order = self.permutation_order(perm)
                generated_count *= order
                print(f"  Added generator {len(generators)}: order {order}, subgroup size now {generated_count}")
                
                # Early exit if we've generated the entire group
                if generated_count >= len(permutations):
                    print(f"  Generated full group, stopping search")
                    break
        
        # Compute orders of generators
        generator_info = []
        for gen in generators:
            order = self.permutation_order(gen)
            generator_info.append({
                'permutation': gen,
                'order': order,
                'cycles': AutomorphismFinder().permutation_to_cycle_notation(gen)
            })
        
        print(f"Found {len(generator_info)} generators")
        return generator_info


def main():
    parser = argparse.ArgumentParser(description='Hamiltonian Automorphism Finder')
    parser.add_argument('--data_dir', type=str, default='.', help='Directory containing data files')
    parser.add_argument('--generate-viz', action='store_true', help='Generate DOT visualizations (default: skip)')
    parser.add_argument('--viz-limit', type=int, default=200, help='Maximum number of automorphisms to visualize when --generate-viz is set (default: 200)')
    args = parser.parse_args()

    inter_all_file = os.path.join(args.data_dir, "InterAll.dat")
    trans_file = os.path.join(args.data_dir, "Trans.dat")

    # Read data files
    vertex_weights = read_trans_file(trans_file)
    edges = read_interall_file(inter_all_file)

    # Construct colored graph with vertex mappings
    graph, vertex_colors, idx_to_vid, vid_to_idx = construct_colored_graph(vertex_weights, edges)
    
    # Find automorphism group
    aut_group = autgrp(graph)
    
    # Print results
    print("Number of vertices:", len(vertex_colors))
    print("Number of edges:", len(edges))
    print("\nVertex colors:")
    for vertex in sorted(vertex_colors.keys()):
        print(f"  Vertex {vertex}: Color {vertex_colors[vertex]}")
    
    print("\nAutomorphism group information:")
    print(f"  Number of generators: {len(aut_group[0])}")
    print(f"  Order of automorphism group: {aut_group[1]}")
    print(f"  Orbit partition: {aut_group[2]}")
    
    output_dir = os.path.join(args.data_dir, "automorphism_results")
    os.makedirs(output_dir, exist_ok=True)
    
    finder = AutomorphismFinder()
    
    # Generate all automorphisms from generators (these use nauty indices 0..n-1)
    n_vertices = len(vertex_colors)
    all_automorphisms_nauty = finder.generate_all_automorphisms(aut_group[0], n_vertices)
    
    # Convert automorphisms from nauty indices to original vertex IDs
    all_automorphisms = []
    for perm_nauty in all_automorphisms_nauty:
        # perm_nauty[i] = j means nauty index i maps to nauty index j
        # We want: original_vid[i] maps to original_vid[j]
        # So: perm_original[vid] = idx_to_vid[perm_nauty[vid_to_idx[vid]]]
        perm_original = [idx_to_vid[perm_nauty[vid_to_idx[vid]]] for vid in idx_to_vid]
        all_automorphisms.append(perm_original)
    
    # Save all automorphisms to JSON (now with correct vertex IDs)
    automorphisms_file = os.path.join(output_dir, "automorphisms.json")
    finder.save_automorphisms_json(all_automorphisms, automorphisms_file)
    
    # Save vertex ID mapping for reference
    mapping_file = os.path.join(output_dir, "vertex_mapping.json")
    with open(mapping_file, 'w') as f:
        json.dump({
            "idx_to_vid": idx_to_vid,
            "vid_to_idx": vid_to_idx,
            "note": "Nauty uses contiguous indices 0..n-1, mapped to original vertex IDs"
        }, f, indent=2)
    print(f"Saved vertex mapping to {mapping_file}")

    # Initialize the analyzer
    analyzer = AutomorphismCliqueAnalyzer()

    # Find maximum clique of commuting automorphisms
    max_clique_indices = analyzer.find_maximum_clique(all_automorphisms)
    
    # Save maximum clique to JSON
    max_clique_file = os.path.join(output_dir, "max_clique.json")
    max_clique = [all_automorphisms[i] for i in max_clique_indices]
    with open(max_clique_file, 'w') as f:
        json.dump(max_clique, f, indent=2)

    print(f"\nSaved maximum clique of {len(max_clique_indices)} commuting automorphisms to {max_clique_file}")

    # Generate visualization of the automorphism graph (optional, disabled by default)
    if args.generate_viz:
        if len(all_automorphisms) > args.viz_limit:
            print(f"\nSkipping visualization: {len(all_automorphisms)} automorphisms exceeds limit of {args.viz_limit}")
            print(f"  Use --viz-limit {len(all_automorphisms)} to force visualization")
        else:
            print(f"\nGenerating visualizations...")
            analyzer.generate_automorphism_graph(all_automorphisms, output_dir, finder)
            analyzer.visualize_clique(all_automorphisms, max_clique_indices, output_dir, finder)
    else:
        print(f"\nSkipping visualizations (use --generate-viz to enable)")

    # Find minimal generators
    print("\nFinding maximal abelian subgroup generators:")
    subgroup_finder = MaximalAbelianSubgroupFinder()
    generators = subgroup_finder.find_minimal_generators(max_clique)
    for gen in generators:
        print(f"  Generator: {gen}")
    
    # Save generators to JSON
    generators_file = os.path.join(output_dir, "minimal_generators.json")
    with open(generators_file, 'w') as f:
        json.dump(generators, f, indent=2)
    print(f"Saved minimal generators to {generators_file}")

    # Generate sector metadata for symmetry sectors
    sector_metadata = generate_sector_metadata(generators)
    sector_metadata_file = os.path.join(output_dir, "sector_metadata.json")
    with open(sector_metadata_file, 'w') as f:
        json.dump(sector_metadata, f, indent=2)
    print(f"Saved sector metadata to {sector_metadata_file}")
    print(f"Number of symmetry sectors: {len(sector_metadata['sectors'])}")


def generate_sector_metadata(generators):
    """
    Generate metadata for all symmetry sectors (irreducible representations).
    
    For abelian groups, each sector is characterized by quantum numbers corresponding
    to the eigenvalues of the symmetry operators (generators).
    
    Args:
        generators: List of generator info dictionaries with 'permutation' and 'order'
        
    Returns:
        Dictionary with sector metadata including quantum numbers and phase factors
    """
    if not generators:
        # No generators -> only trivial sector
        return {
            "num_generators": 0,
            "generator_orders": [],
            "sectors": [
                {
                    "sector_id": 0,
                    "quantum_numbers": [],
                    "phase_factors": []
                }
            ]
        }
    
    # Extract generator orders
    generator_orders = [gen['order'] for gen in generators]
    num_generators = len(generator_orders)
    
    print(f"\nGenerating sector metadata for abelian group:")
    print(f"  Generators: {num_generators}")
    print(f"  Orders: {generator_orders}")
    
    # For abelian groups, quantum numbers range from 0 to order-1 for each generator
    # Each sector corresponds to a unique combination of quantum numbers
    sectors = []
    sector_id = 0
    
    # Generate all possible quantum number combinations
    def generate_quantum_numbers(idx, current_qn):
        nonlocal sector_id
        
        if idx == num_generators:
            # Compute phase factors for each generator for this sector
            # phase_k = exp(2πi * quantum_number_k / order_k)
            # The C++ code will compose these based on power representation
            phase_factors = []
            for j in range(num_generators):
                phase_angle = 2.0 * np.pi * current_qn[j] / generator_orders[j]
                # Store as complex number: real and imaginary parts
                phase_factors.append({
                    "real": np.cos(phase_angle),
                    "imag": np.sin(phase_angle)
                })
            
            sectors.append({
                "sector_id": sector_id,
                "quantum_numbers": list(current_qn),
                "phase_factors": phase_factors
            })
            sector_id += 1
            return
        
        # Try all quantum numbers for current generator
        for qn in range(generator_orders[idx]):
            generate_quantum_numbers(idx + 1, current_qn + [qn])
    
    generate_quantum_numbers(0, [])
    
    print(f"  Total sectors: {len(sectors)}")
    
    return {
        "num_generators": num_generators,
        "generator_orders": generator_orders,
        "sectors": sectors
    }


if __name__ == "__main__":
    main()