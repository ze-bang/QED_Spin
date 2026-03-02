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
    """Construct a colored undirected graph for pynauty with edge-type subdivision.
    
    Uses the subdivision trick for edge-colored graphs: for each pair (i,j) of 
    interacting sites, an auxiliary vertex is inserted whose color encodes the 
    bond type (the full set of coupling terms on that bond). This ensures that 
    nauty's automorphisms can only map bonds of the same type to each other.
    
    Returns:
        tuple: (graph, vertex_colors, idx_to_vid, vid_to_idx)
            - graph: pynauty Graph object
            - vertex_colors: dict mapping original vertex_id -> color
            - idx_to_vid: list mapping nauty index -> original vertex_id
            - vid_to_idx: dict mapping original vertex_id -> nauty index
    """
    # Compute WL-refined vertex colors for original vertices
    vertex_colors = compute_vertex_colors(vertex_weights, edges)

    # Build stable vertex index mapping for original vertices (0..n-1)
    all_vertices = sorted(vertex_colors.keys())
    vid_to_idx = {v: i for i, v in enumerate(all_vertices)}
    idx_to_vid = list(all_vertices)
    n_original = len(all_vertices)

    # --- Compute bond signatures ---
    # Group edges by ordered pair (min_vertex, max_vertex)
    from collections import defaultdict as _dd
    bond_terms = _dd(list)
    for edge in edges:
        v1, v2 = edge['vertex1'], edge['vertex2']
        if v1 == v2:
            continue
        if v1 not in vid_to_idx or v2 not in vid_to_idx:
            continue
        # Store canonical direction info: which vertex is 'left' vs 'right'
        lo, hi = min(v1, v2), max(v1, v2)
        # Encode direction relative to canonical order
        if v1 == lo:
            term = (edge['type1'], edge['type2'], 
                    _round_tuple(edge['weight']))
        else:
            term = (edge['type2'], edge['type1'], 
                    _round_tuple(edge['weight']))
        bond_terms[(lo, hi)].append(term)
    
    # Create canonical bond signature for each pair
    bond_pairs = sorted(bond_terms.keys())
    bond_signatures = {}
    for pair in bond_pairs:
        sig = tuple(sorted(bond_terms[pair]))
        bond_signatures[pair] = sig
    
    # Assign colors to unique bond signatures
    unique_sigs = sorted(set(bond_signatures.values()))
    sig_to_color = {sig: i for i, sig in enumerate(unique_sigs)}
    
    n_bonds = len(bond_pairs)
    n_total = n_original + n_bonds  # original vertices + auxiliary vertices
    
    print(f"  Edge-colored graph: {n_original} vertices + {n_bonds} auxiliary bond vertices")
    print(f"  Unique bond types: {len(unique_sigs)}")
    
    # Build adjacency for expanded graph
    adjacency_dict = {i: [] for i in range(n_total)}
    
    for bond_idx, (lo, hi) in enumerate(bond_pairs):
        aux_idx = n_original + bond_idx  # index of auxiliary vertex
        i, j = vid_to_idx[lo], vid_to_idx[hi]
        # Connect both endpoints to the auxiliary vertex
        adjacency_dict[i].append(aux_idx)
        adjacency_dict[aux_idx].append(i)
        adjacency_dict[j].append(aux_idx)
        adjacency_dict[aux_idx].append(j)
    
    # Build vertex coloring
    # Original vertices: color from WL
    # Auxiliary vertices: color based on bond signature (offset by max vertex color + 1)
    max_vertex_color = max(vertex_colors.values()) + 1 if vertex_colors else 0
    
    color_to_vertices = _dd(list)
    for v, c in vertex_colors.items():
        color_to_vertices[c].append(vid_to_idx[v])
    
    for bond_idx, pair in enumerate(bond_pairs):
        aux_idx = n_original + bond_idx
        bond_color = max_vertex_color + sig_to_color[bond_signatures[pair]]
        color_to_vertices[bond_color].append(aux_idx)
    
    coloring = [set(sorted(ids)) for _, ids in 
                sorted(color_to_vertices.items(), key=lambda kv: kv[0])]
    
    # Create pynauty graph
    g = Graph(n_total, directed=False, adjacency_dict=adjacency_dict, 
              vertex_coloring=coloring)
    return g, vertex_colors, idx_to_vid, vid_to_idx


def filter_hamiltonian_automorphisms(automorphisms, edges):
    """Filter automorphisms to keep only those that preserve the Hamiltonian.
    
    An automorphism σ is a valid Hamiltonian symmetry iff for every interaction 
    term (type1, site1, type2, site2, weight), the mapped term
    (type1, σ(site1), type2, σ(site2), weight) also exists in the Hamiltonian.
    
    Since operators on different sites commute, O_a(i) O_b(j) = O_b(j) O_a(i),
    so the reversed ordering (type2, σ(site2), type1, σ(site1), weight) is also 
    accepted as a match.
    
    Args:
        automorphisms: List of permutations (each is a list of site indices)
        edges: List of edge dictionaries from read_interall_file
    
    Returns:
        List of valid automorphisms
    """
    # Build lookup of all Hamiltonian terms as a set
    ham_terms = set()
    for edge in edges:
        key = (edge['type1'], edge['vertex1'], edge['type2'], edge['vertex2'],
               _round_tuple(edge['weight']))
        ham_terms.add(key)
    
    valid = []
    for sigma in automorphisms:
        is_valid = True
        for edge in edges:
            w = _round_tuple(edge['weight'])
            sv1 = sigma[edge['vertex1']]
            sv2 = sigma[edge['vertex2']]
            mapped_key = (edge['type1'], sv1, edge['type2'], sv2, w)
            # Also check reversed site order (operators on different sites commute)
            reversed_key = (edge['type2'], sv2, edge['type1'], sv1, w)
            if mapped_key not in ham_terms and reversed_key not in ham_terms:
                is_valid = False
                break
        if is_valid:
            valid.append(sigma)
    
    return valid


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
    
    # Generate all automorphisms from generators
    # The graph has n_original + n_auxiliary vertices (subdivision trick)
    n_original = len(vertex_colors)
    n_total = graph.number_of_vertices
    print(f"\nGenerating all automorphisms (expanded graph has {n_total} vertices)...")
    all_automorphisms_nauty = finder.generate_all_automorphisms(aut_group[0], n_total)
    
    # Extract original-vertex permutations from expanded graph automorphisms
    # Each expanded automorphism maps original vertices to original vertices
    # (auxiliary vertices can only map to auxiliary vertices of the same color)
    seen = set()
    all_automorphisms = []
    for perm_nauty in all_automorphisms_nauty:
        # Extract just the original vertex portion (first n_original indices)
        perm_original = [idx_to_vid[perm_nauty[vid_to_idx[vid]]] for vid in idx_to_vid]
        key = tuple(perm_original)
        if key not in seen:
            seen.add(key)
            all_automorphisms.append(perm_original)
    
    print(f"Found {len(all_automorphisms)} unique graph automorphisms (projected to original vertices)")
    
    # Post-filter: verify each automorphism actually commutes with the Hamiltonian
    # This is a safety check — the subdivision trick should already give correct results
    all_automorphisms_pre_filter = len(all_automorphisms)
    all_automorphisms = filter_hamiltonian_automorphisms(all_automorphisms, edges)
    
    if len(all_automorphisms) < all_automorphisms_pre_filter:
        n_removed = all_automorphisms_pre_filter - len(all_automorphisms)
        print(f"WARNING: Removed {n_removed} automorphisms that don't commute with Hamiltonian")
        print(f"  This may indicate edge-type encoding issues in the graph construction")
    print(f"Valid Hamiltonian automorphisms: {len(all_automorphisms)}")
    
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
    sector_metadata = generate_sector_metadata(generators, max_clique)
    sector_metadata_file = os.path.join(output_dir, "sector_metadata.json")
    with open(sector_metadata_file, 'w') as f:
        json.dump(sector_metadata, f, indent=2)
    print(f"Saved sector metadata to {sector_metadata_file}")
    print(f"Number of symmetry sectors: {len(sector_metadata['sectors'])}")


def _find_relation_subgroup(generators, group_elements):
    """
    Find the relation subgroup K of the generator presentation.
    
    The generators g_0, ..., g_{k-1} with orders o_0, ..., o_{k-1} define a 
    surjective homomorphism phi: Z_{o_0} x ... x Z_{o_{k-1}} -> G via
    phi(a_0,...,a_{k-1}) = g_0^{a_0} * ... * g_{k-1}^{a_{k-1}}.
    
    K = ker(phi) is the relation subgroup. A quantum number tuple q is a valid
    irrep label iff the character chi_q is trivial on K, i.e.,
    sum_k q_k * r_k / o_k is an integer for every relation r in K.
    
    Returns:
        List of tuples representing elements of K (excluding identity if trivial)
    """
    gen_perms = [g['permutation'] for g in generators]
    gen_orders = [g['order'] for g in generators]
    num_gen = len(generators)
    n_sites = len(gen_perms[0])
    identity = list(range(n_sites))
    
    def compose_perm(a, b):
        return [a[b[i]] for i in range(n_sites)]
    
    def power_perm(perm, power):
        result = list(range(n_sites))
        for _ in range(power):
            result = compose_perm(perm, result)
        return result
    
    # Enumerate all tuples in Z_{o_0} x ... x Z_{o_{k-1}} and find those
    # that map to the identity permutation
    import itertools
    K = []
    ranges = [range(o) for o in gen_orders]
    for r in itertools.product(*ranges):
        if all(ri == 0 for ri in r):
            continue  # skip identity, always in K
        # Compute g_0^{r_0} * g_1^{r_1} * ... * g_{k-1}^{r_{k-1}}
        result = list(identity)
        for k in range(num_gen):
            gk_pow = power_perm(gen_perms[k], r[k])
            result = compose_perm(result, gk_pow)
        if result == identity:
            K.append(r)
    
    return K


def generate_sector_metadata(generators, group_elements=None):
    """
    Generate metadata for all symmetry sectors (irreducible representations).
    
    For abelian groups, each sector is characterized by quantum numbers corresponding
    to the eigenvalues of the symmetry operators (generators).
    
    When generators are not algebraically independent (i.e., there exist non-trivial
    relations among them), the naive product of orders overcounts the number of irreps.
    We detect such relations and filter to keep only valid irrep labels.
    
    Args:
        generators: List of generator info dictionaries with 'permutation' and 'order'
        group_elements: List of group element permutations (max_clique). If provided,
                        enables relation detection and sector filtering.
        
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
    product_of_orders = 1
    for o in generator_orders:
        product_of_orders *= o
    
    print(f"\nGenerating sector metadata for abelian group:")
    print(f"  Generators: {num_generators}")
    print(f"  Orders: {generator_orders}")
    print(f"  Product of orders: {product_of_orders}")
    
    # Detect relations among generators if group elements are provided
    relation_subgroup = []
    if group_elements is not None:
        group_size = len(group_elements)
        print(f"  Group size |G|: {group_size}")
        if product_of_orders != group_size:
            print(f"  WARNING: Product of orders ({product_of_orders}) != |G| ({group_size})")
            print(f"  Generators have non-trivial relations. Finding relation subgroup K...")
            relation_subgroup = _find_relation_subgroup(generators, group_elements)
            print(f"  |K| = {len(relation_subgroup) + 1} (including identity)")
            for r in relation_subgroup:
                parts = []
                for k in range(num_generators):
                    if r[k] != 0:
                        if r[k] == 1:
                            parts.append(f"g{k}")
                        else:
                            parts.append(f"g{k}^{r[k]}")
                print(f"    Relation: {' * '.join(parts)} = e  {list(r)}")
        else:
            print(f"  Generators are independent (product of orders = |G|)")
    
    # Generate all possible quantum number combinations
    all_sectors = []
    
    def generate_quantum_numbers(idx, current_qn):
        if idx == num_generators:
            all_sectors.append(list(current_qn))
            return
        for qn in range(generator_orders[idx]):
            generate_quantum_numbers(idx + 1, current_qn + [qn])
    
    generate_quantum_numbers(0, [])
    
    # Filter to valid sectors if there are relations
    if relation_subgroup:
        valid_qns = []
        for qn in all_sectors:
            is_valid = True
            for r in relation_subgroup:
                # Check: sum_k q_k * r_k / o_k must be an integer
                s = sum(qn[k] * r[k] / generator_orders[k] for k in range(num_generators))
                if abs(s - round(s)) > 1e-10:
                    is_valid = False
                    break
            if is_valid:
                valid_qns.append(qn)
        
        n_removed = len(all_sectors) - len(valid_qns)
        print(f"  Filtered {n_removed} invalid sectors (phantom irreps)")
        print(f"  Valid sectors: {len(valid_qns)}")
        all_sectors = valid_qns
    
    # Build sector metadata with phase factors
    sectors = []
    for sector_id, qn in enumerate(all_sectors):
        phase_factors = []
        for j in range(num_generators):
            phase_angle = 2.0 * np.pi * qn[j] / generator_orders[j]
            phase_factors.append({
                "real": float(np.cos(phase_angle)),
                "imag": float(np.sin(phase_angle))
            })
        
        sectors.append({
            "sector_id": sector_id,
            "quantum_numbers": qn,
            "phase_factors": phase_factors
        })
    
    print(f"  Total sectors: {len(sectors)}")
    
    return {
        "num_generators": num_generators,
        "generator_orders": generator_orders,
        "sectors": sectors
    }


if __name__ == "__main__":
    main()