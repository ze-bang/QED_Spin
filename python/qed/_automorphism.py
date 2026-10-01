"""Graph automorphisms of a spin Hamiltonian (pynauty), behind ``qed.find_symmetries``.

The terms become a coloured graph: vertex colours from the one-body terms, a subdivision
vertex per interacting pair whose colour encodes the bond's couplings, and one per triple of
sites carrying three-body terms. Its automorphisms that leave H invariant are the site
symmetries; ``qed._groups`` splits them for the sector engine.
"""
import itertools
from collections import defaultdict, deque

import numpy as np
from pynauty import Graph, autgrp

from . import _log


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
    # Collect all vertices that appear in the one-body terms or the edges
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

def _triple_signature(triple, terms):
    """A label of the three-body terms on a site triple that does not depend on how the triple
    is ordered: the least, over its six orderings, of the terms written by position."""
    best = None
    for order in itertools.permutations(triple):
        pos = {s: k for k, s in enumerate(order)}
        sig = tuple(sorted(tuple(sorted((pos[s], o) for s, o in zip(sites, ops))) + (w,)
                           for sites, ops, w in terms))
        if best is None or sig < best:
            best = sig
    return best


def construct_colored_graph(vertex_weights, edges, triples=()):
    """Construct a colored undirected graph for pynauty with edge-type subdivision.

    Uses the subdivision trick for edge-colored graphs: for each pair (i,j) of
    interacting sites, an auxiliary vertex is inserted whose color encodes the
    bond type (the full set of coupling terms on that bond). This ensures that
    nauty's automorphisms can only map bonds of the same type to each other.
    ``triples`` holds the three-body terms as ``((s1, s2, s3), (op1, op2, op3), coeff)``:
    each triple of distinct sites gets an auxiliary vertex joined to its three sites and
    coloured by :func:`_triple_signature`. That colour ignores the orientation of the
    triple (a scalar chirality and its reverse look alike), so the group found may still
    hold elements that reverse it; the exact term check that follows removes them.

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

    # Three-body terms, one auxiliary vertex per triple of distinct sites; a term that repeats
    # a site constrains nothing here, and the exact check after the search still sees it.
    triple_terms = _dd(list)
    for sites, ops, coeff in triples:
        if len(set(sites)) == 3 and all(s in vid_to_idx for s in sites):
            c = complex(coeff)
            triple_terms[tuple(sorted(sites))].append((sites, ops, _round_tuple((c.real, c.imag))))
    triple_list = sorted(triple_terms)
    triple_signatures = {t: _triple_signature(t, triple_terms[t]) for t in triple_list}
    unique_triple_sigs = sorted(set(triple_signatures.values()))
    triple_to_color = {sig: len(unique_sigs) + i for i, sig in enumerate(unique_triple_sigs)}

    n_bonds = len(bond_pairs)
    n_total = n_original + n_bonds + len(triple_list)  # original vertices + auxiliary vertices

    _log.log(_log.DEBUG, "edge-coloured graph: %d vertices + %d auxiliary bond vertices (%d types) "
             "+ %d triple vertices (%d types)", n_original, n_bonds, len(unique_sigs),
             len(triple_list), len(unique_triple_sigs))

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
    for t_idx, t in enumerate(triple_list):
        aux_idx = n_original + n_bonds + t_idx
        for s in t:
            adjacency_dict[vid_to_idx[s]].append(aux_idx)
            adjacency_dict[aux_idx].append(vid_to_idx[s])
    
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
    for t_idx, t in enumerate(triple_list):
        color_to_vertices[max_vertex_color + triple_to_color[triple_signatures[t]]].append(
            n_original + n_bonds + t_idx)

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
    
def filter_translation_automorphisms(automorphisms, sites, lattice_vectors, cluster_dims, tol=1e-6):
    """Filter automorphisms to keep only pure translations.
    
    A permutation sigma is a pure translation if there exists a displacement T
    such that for every site i:
      position[sigma(i)] ≡ position[i] + T   (mod supercell)
    AND sigma maps each site to the same sublattice.
    
    Uses supercell vectors (N_k * a_k) for the modular arithmetic so that
    distinct translations within the cluster are distinguishable.
    
    Args:
        automorphisms: List of permutations
        sites: dict from read_positions_file
        lattice_vectors: list of np.array from read_lattice_vectors (primitive)
        cluster_dims: list of int [N1, N2, ...] from read_lattice_vectors
        tol: tolerance for floating point comparison
    
    Returns:
        List of translation automorphisms (always includes identity)
    """
    if cluster_dims is None:
        raise ValueError("cluster_dims is required: could not parse '# Unit cells:' line from lattice_parameters.dat")
    
    site_ids = sorted(sites.keys())
    n = len(site_ids)
    lat_dim = len(lattice_vectors[0])  # Dimension of lattice vectors
    num_lat_vecs = len(lattice_vectors)
    
    if lat_dim != num_lat_vecs:
        raise ValueError(f"Non-square lattice matrix ({lat_dim}D vectors, {num_lat_vecs} vectors): "
                         "translation filtering requires a square lattice basis")
    
    # Build SUPERCELL lattice matrix: S_k = N_k * a_k
    # This ensures translations by different primitive vectors are distinguishable
    supercell_vectors = []
    for k in range(num_lat_vecs):
        Nk = cluster_dims[k] if k < len(cluster_dims) else 1
        supercell_vectors.append(Nk * lattice_vectors[k])
    
    S = np.column_stack(supercell_vectors)  # lat_dim x num_lat_vecs
    S_inv = np.linalg.inv(S)
    
    identity = list(range(max(site_ids) + 1))
    translations = []
    
    for sigma in automorphisms:
        # Check if it's identity first (always a translation)
        if sigma == identity:
            translations.append(sigma)
            continue
        
        is_translation = True
        ref_displacement_frac = None
        
        for sid in site_ids:
            mapped_sid = sigma[sid]
            
            # Must map to same sublattice
            if sites[sid]['sublattice'] != sites[mapped_sid]['sublattice']:
                is_translation = False
                break
            
            # Compute displacement in fractional coordinates of SUPERCELL
            displacement = sites[mapped_sid]['position'] - sites[sid]['position']
            # Project to lattice subspace (handles 2D lattice with 3D coordinates)
            displacement_lat = displacement[:lat_dim]
            displacement_frac = S_inv @ displacement_lat
            
            # Reduce modulo supercell (mod 1 in supercell fractional coords)
            displacement_frac_mod = displacement_frac - np.round(displacement_frac)
            
            if ref_displacement_frac is None:
                ref_displacement_frac = displacement_frac_mod
            else:
                # All sites must have the same displacement (mod supercell)
                diff = displacement_frac_mod - ref_displacement_frac
                diff = diff - np.round(diff)
                if np.max(np.abs(diff)) > tol:
                    is_translation = False
                    break
        
        if is_translation:
            translations.append(sigma)
    
    return translations

