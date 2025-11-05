import numpy as np
import sys
import os
import re

try:
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib as mpl
    from matplotlib.ticker import NullFormatter
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

def generate_pyrochlore_super_cluster(dim1, dim2, dim3, use_pbc=False):
    """
    Generate a pyrochlore lattice cluster with tetrahedra stacked in a tetrahedral fashion
    
    Tetrahedron centers are at:
    - (0, 0, 0)
    - (0, 1/2, 1/2)
    - (1/2, 0, 1/2)
    - (1/2, 1/2, 0)
    
    Each tetrahedron has 4 sites at offsets:
    - (0, 0, 0)
    - (0, 0.25, 0.25)
    - (0.25, 0, 0.25)
    - (0.25, 0.25, 0)
    
    Args:
        dim1, dim2, dim3: Dimensions of the lattice (number of unit cells)
        use_pbc: Whether to use periodic boundary conditions
        
    Returns:
        vertices: Dictionary of {vertex_id: (x, y, z)}
        edges: List of (vertex1, vertex2) tuples
        tetrahedra: List of (v1, v2, v3, v4) tuples
        node_mapping: Dictionary mapping original IDs to matrix indices
        vertex_to_cell: Map vertex_id to (i, j, k, tet_idx, site_idx)
    """
    # Tetrahedron centers in the unit cell
    tetrahedron_centers = np.array([
        [0, 0, 0],
        [0, 0.5, 0.5],
        [0.5, 0, 0.5],
        [0.5, 0.5, 0]
    ])
    
    # Site offsets within each tetrahedron
    site_offsets = np.array([
        [0, 0, 0],
        [0, 0.25, 0.25],
        [0.25, 0, 0.25],
        [0.25, 0.25, 0]
    ])

    # Unit cell lattice vectors
    lattice_vectors = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ])
    
    # Generate vertices
    vertices = {}
    vertex_id = 0
    vertex_to_cell = {}  # Map vertex_id to (i, j, k, tet_idx, site_idx)
    cell_to_vertex = {}  # Map (i, j, k, tet_idx, site_idx) to vertex_id
    
    for i in range(dim1):
        for j in range(dim2):
            for k in range(dim3):
                # Unit cell position
                unit_cell_pos = i * lattice_vectors[0] + j * lattice_vectors[1] + k * lattice_vectors[2]
                
                # For each tetrahedron in the unit cell
                for tet_idx in range(4):
                    tet_center = unit_cell_pos + tetrahedron_centers[tet_idx]
                    
                    # For each site in the tetrahedron
                    for site_idx in range(4):
                        position = tet_center + site_offsets[site_idx]
                        vertices[vertex_id] = tuple(position)
                        vertex_to_cell[vertex_id] = (i, j, k, tet_idx, site_idx)
                        cell_to_vertex[(i, j, k, tet_idx, site_idx)] = vertex_id
                        vertex_id += 1
    
    # Helper function to get vertex id with PBC
    def get_vertex_with_pbc(i, j, k, tet_idx, site_idx):
        if use_pbc:
            # Apply periodic boundary conditions
            i = i % dim1
            j = j % dim2
            k = k % dim3
        else:
            # Check if out of bounds for open boundary conditions
            if i < 0 or i >= dim1 or j < 0 or j >= dim2 or k < 0 or k >= dim3:
                return None
        
        key = (i, j, k, tet_idx, site_idx)
        return cell_to_vertex.get(key, None)
    
    # Generate edges based on nearest neighbors
    edges = set()  # Use set to avoid duplicates
    
    for v1_id, v1_info in vertex_to_cell.items():
        i1, j1, k1, tet1, site1 = v1_info
        pos1 = np.array(vertices[v1_id])
        
        # Check neighbors within same unit cell and adjacent cells
        for di in [-1, 0, 1]:
            for dj in [-1, 0, 1]:
                for dk in [-1, 0, 1]:
                    # Check all tetrahedra
                    for tet2 in range(4):
                        # Check all sites in the tetrahedron
                        for site2 in range(4):
                            v2_id = get_vertex_with_pbc(i1+di, j1+dj, k1+dk, tet2, site2)
                            
                            if v2_id is None or v2_id == v1_id:
                                continue
                            
                            # Calculate actual distance (considering PBC if needed)
                            pos2 = np.array(vertices[v2_id])
                            
                            if use_pbc:
                                # Calculate minimum image distance
                                delta = pos2 - pos1
                                # Apply minimum image convention
                                for dim_idx in range(3):
                                    box_size = [dim1, dim2, dim3][dim_idx]
                                    if delta[dim_idx] > box_size/2:
                                        delta[dim_idx] -= box_size
                                    elif delta[dim_idx] < -box_size/2:
                                        delta[dim_idx] += box_size
                                dist = np.linalg.norm(delta)
                            else:
                                dist = np.linalg.norm(pos2 - pos1)
                            
                            # Check if they are nearest neighbors
                            # Threshold for nearest neighbors in pyrochlore
                            if dist < 0.4 and dist > 0.01:  # Avoid self-connections
                                edge = tuple(sorted([v1_id, v2_id]))
                                edges.add(edge)
    
    # Convert edges set to list
    edges = list(edges)
    
    # Generate tetrahedra list
    tetrahedra = []
    for i in range(dim1):
        for j in range(dim2):
            for k in range(dim3):
                for tet_idx in range(4):
                    # Get the 4 vertices of this tetrahedron
                    tet_vertices = []
                    for site_idx in range(4):
                        v_id = cell_to_vertex[(i, j, k, tet_idx, site_idx)]
                        tet_vertices.append(v_id)
                    tetrahedra.append(tuple(tet_vertices))
    
    # Create node mapping
    node_mapping = {i: i for i in range(len(vertices))}
    
    return vertices, edges, tetrahedra, node_mapping, vertex_to_cell

def get_sublattice_index(vertex_id, vertex_to_cell):
    """
    Return the sublattice index based on which site in the tetrahedron
    """
    if vertex_id in vertex_to_cell:
        _, _, _, _, site_idx = vertex_to_cell[vertex_id]
        return site_idx
    return vertex_id % 4

def create_nn_lists(edges, node_mapping, vertices, vertex_to_cell):
    """
    Create nearest neighbor lists from the edge information
    
    Args:
        edges: List of (vertex1, vertex2) tuples
        node_mapping: Dictionary mapping original IDs to matrix indices
        vertices: Dictionary of {vertex_id: (x, y, z)}
        vertex_to_cell: Map vertex_id to (i, j, k, tet_idx, site_idx)
        
    Returns:
        nn_list: Dictionary mapping each site to its nearest neighbors
        positions: Dictionary mapping each site to its position
        sublattice_indices: Dictionary mapping each site to its sublattice index
    """
    nn_list = {}
    positions = {}
    sublattice_indices = {}
    
    # Initialize empty lists for all vertices
    for vertex_id in vertices:
        nn_list[vertex_id] = []
        positions[vertex_id] = vertices[vertex_id]
        sublattice_indices[vertex_id] = get_sublattice_index(vertex_id, vertex_to_cell)
    
    # Fill nearest neighbor lists based on edges
    for v1, v2 in edges:
        nn_list[v1].append(v2)
        nn_list[v2].append(v1)
    
    return nn_list, positions, sublattice_indices

def write_cluster_nn_list(output_dir, cluster_name, nn_list, positions, sublattice_indices, node_mapping):
    """
    Write nearest neighbor list, positions, and sublattice indices to a file
    """
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    
    # Write nearest neighbor list
    with open(f"{output_dir}/{cluster_name}_nn_list.dat", 'w') as f:
        f.write("# Nearest neighbor list for cluster: " + cluster_name + "\n")
        f.write("# Format: site_id, number_of_neighbors, [neighbor_ids]\n\n")
        
        for site_id in sorted(nn_list.keys()):
            neighbors = nn_list[site_id]
            matrix_index = node_mapping.get(site_id, site_id)
            
            f.write(f"{site_id} {len(neighbors)}")
            for neighbor in neighbors:
                f.write(f" {neighbor}")
            f.write("\n")
    
    # Write site positions and sublattice indices
    with open(f"{output_dir}/positions.dat", 'w') as f:
        f.write("# Site information for cluster: " + cluster_name + "\n")
        f.write("# site_id, matrix_index, sublattice_index, x, y, z\n\n")
        
        for site_id in sorted(positions.keys()):
            pos = positions[site_id]
            sub_idx = sublattice_indices[site_id]
            matrix_index = node_mapping.get(site_id, site_id)
            
            f.write(f"{site_id} {matrix_index} {sub_idx} {pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f}\n")

    # Write lattice parameters
    with open(f"{output_dir}/{cluster_name}_lattice_parameters.dat", 'w') as f:
        f.write("# Pyrochlore super lattice parameters\n\n")
        
        # Tetrahedron centers
        tetrahedron_centers = np.array([
            [0, 0, 0],
            [0, 0.5, 0.5],
            [0.5, 0, 0.5],
            [0.5, 0.5, 0]
        ])
        
        # Site offsets
        site_offsets = np.array([
            [0, 0, 0],
            [0, 0.25, 0.25],
            [0.25, 0, 0.25],
            [0.25, 0.25, 0]
        ])

        
        # Write tetrahedron centers
        f.write("# Tetrahedron centers in unit cell\n")
        f.write("# tet_index, x, y, z\n")
        for i, center in enumerate(tetrahedron_centers):
            f.write(f"{i} {center[0]:.6f} {center[1]:.6f} {center[2]:.6f}\n")
        
        f.write("\n")
        
        # Write site offsets
        f.write("# Site offsets within each tetrahedron\n")
        f.write("# site_index, x, y, z\n")
        for i, offset in enumerate(site_offsets):
            f.write(f"{i} {offset[0]:.6f} {offset[1]:.6f} {offset[2]:.6f}\n")

def prepare_hamiltonian_parameters(output_dir, non_kramer, nn_list, positions, sublattice_indices, 
                                  node_mapping, Jxx, Jyy, Jzz, h, theta, field_dir):
    """
    Prepare Hamiltonian parameters for exact diagonalization
    """
    # Prepare Hamiltonian parameters
    Jpm = -(Jxx+Jyy)/4
    Jpmpm = (Jxx-Jyy)/4
    
    # Define local z-axes for pyrochlore lattice
    z_local = np.array([
        np.array([1, 1, 1]) / np.sqrt(3),
        np.array([1, -1, -1]) / np.sqrt(3),
        np.array([-1, 1, -1]) / np.sqrt(3),
        np.array([-1, -1, 1]) / np.sqrt(3)
    ])
    
    # Normalize field direction
    field_dir = np.array(field_dir)
    field_dir = field_dir / np.linalg.norm(field_dir)
    
    interALL = []
    transfer = []

    gamma = np.exp(1j*2*np.pi/3)
    # Define non-kramer pyrochlore factor
    non_kramer_factor = np.array([[0, 1, gamma, gamma**2],
                                  [1, 0, gamma**2, gamma],
                                  [gamma, gamma**2, 0, 1],
                                  [gamma**2, gamma, 1, 0]])
    
    # Generate Heisenberg interactions
    for site_id in sorted(nn_list.keys()):
        # Site index (for the Hamiltonian)
        i = site_id
        
        # Zeeman term
        sub_idx = sublattice_indices[site_id]
        local_field = h * np.dot(field_dir, z_local[sub_idx])
        local_field_x = local_field * np.sin(theta)
        local_field_z = local_field * np.cos(theta)
        transfer.append([0, node_mapping[i], -local_field_x/2, 0])
        transfer.append([1, node_mapping[i], -local_field_x/2, 0])
        transfer.append([2, node_mapping[i], -local_field_z, 0])  # Sz term
        # Exchange interactions
        for neighbor_id in nn_list[site_id]:
            if site_id < neighbor_id:  # Only add each bond once
                j = neighbor_id
                
                # Heisenberg interactions
                interALL.append([2, node_mapping[i], 2, node_mapping[j], Jzz, 0])  # Sz-Sz
                interALL.append([0, node_mapping[i], 1, node_mapping[j], -Jpm, 0])   # S+-S-
                interALL.append([1, node_mapping[i], 0, node_mapping[j], -Jpm, 0])   # S--S+
                if non_kramer:
                    Jpmpm_ = Jpmpm * non_kramer_factor[sublattice_indices[i], sublattice_indices[j]]
                else:
                    Jpmpm_ = Jpmpm
                interALL.append([1, node_mapping[i], 1, node_mapping[j], np.real(Jpmpm_), np.imag(Jpmpm_)])  # S--S-
                interALL.append([0, node_mapping[i], 0, node_mapping[j], np.real(Jpmpm_), -np.imag(Jpmpm_)])  # S+-S+
    
    # Convert to arrays
    interALL = np.array(interALL) if interALL else np.empty((0, 6))
    transfer = np.array(transfer) if transfer else np.empty((0, 4))
    
    # Write interaction and transfer files
    write_interALL(output_dir, interALL, f"InterAll.dat")
    write_transfer(output_dir, transfer, f"Trans.dat")
    
    # Write field strength
    fstrength = np.array([[h]])
    np.savetxt(f"{output_dir}/field_strength.dat", fstrength)
    
    # Write correlation functions
    max_site = len(nn_list)
    opname = ['S+', 'S-', 'Sz']
    
    for i in range(3):
        write_one_body_correlations(output_dir, i, max_site, f"one_body_correlations{opname[i]}.dat")
        for j in range(3):
            write_two_body_correlations(output_dir, i, j, max_site, f"two_body_correlations{opname[i]}{opname[j]}.dat")

def write_interALL(output_dir, interALL, file_name):
    """Write interaction parameters to a file"""
    num_param = len(interALL)
    with open(f"{output_dir}/{file_name}", 'w') as f:
        f.write("===================\n")
        f.write(f"num {num_param:8d}\n")
        f.write("===================\n")
        f.write("===================\n")
        f.write("===================\n")
        
        for i in range(num_param):
            f.write(f" {int(interALL[i,0]):8d} " \
                   f" {int(interALL[i,1]):8d}   " \
                   f" {int(interALL[i,2]):8d}   " \
                   f" {int(interALL[i,3]):8d}   " \
                   f" {interALL[i,4]:8f}   " \
                   f" {interALL[i,5]:8f}   " \
                   f"\n")

def write_transfer(output_dir, transfer, file_name):
    """Write transfer (field) parameters to a file"""
    num_param = len(transfer)
    with open(f"{output_dir}/{file_name}", 'w') as f:
        f.write("===================\n")
        f.write(f"num {num_param:8d}\n")
        f.write("===================\n")
        f.write("===================\n")
        f.write("===================\n")
        
        for i in range(num_param):
            f.write(f" {int(transfer[i,0]):8d} " \
                   f" {int(transfer[i,1]):8d}   " \
                   f" {transfer[i,2]:8f}   " \
                   f" {transfer[i,3]:8f}" \
                   f"\n")

def write_one_body_correlations(output_dir, Op, N, file_name):
    """Write one-body correlation parameters to a file"""
    with open(f"{output_dir}/{file_name}", 'w') as f:
        f.write("===================\n")
        f.write(f"loc {N:8d}\n")
        f.write("===================\n")
        f.write("===================\n")
        f.write("===================\n")
        
        for i in range(N):
            f.write(f" {Op:8d} " \
                   f" {i:8d}   " \
                   f" {1:8f}   " \
                   f" {0:8f}" \
                   f"\n")

def write_two_body_correlations(output_dir, Op1, Op2, N, file_name):
    """Write two-body correlation parameters to a file"""
    num_green_two = N * N
    with open(f"{output_dir}/{file_name}", 'w') as f:
        f.write("===================\n")
        f.write(f"loc {num_green_two:8d}\n")
        f.write("===================\n")
        f.write("===================\n")
        f.write("===================\n")
        
        for i in range(N):
            for j in range(N):
                f.write(f" {Op1:8d} " \
                       f" {i:8d}   " \
                       f" {Op2:8d}   " \
                       f" {j:8d}   " \
                       f" {1:8f}   " \
                       f" {0:8f}   " \
                       f"\n")

def find_counter_term_chains(vertices, nn_list, vertex_to_cell, dim1, dim2, dim3, use_pbc):
    """
    Find chains (or loops) of 4 connected sites that preserve the spin ice manifold.
    
    In pyrochlore lattice, each site belongs to TWO tetrahedra (one up-pointing 
    and one down-pointing). The ice rule must be satisfied for both.
    
    The criterion is that for each tetrahedron touched by the chain, it must be 
    visited an even number of times. This ensures that flipping spins along the 
    chain maintains the "2-in, 2-out" ice rule for all tetrahedra.
    
    Returns:
        List of chains, where each chain is a list of 4 vertex IDs
    """
    # Find all tetrahedra as 4-cliques (complete subgraphs with 4 vertices) in the NN graph
    # This correctly identifies tetrahedra based on actual connectivity, including PBC effects
    tetrahedra = []
    sites = sorted(nn_list.keys())
    for i, a in enumerate(sites):
        for b in [x for x in nn_list[a] if x > a]:
            common_ab = set(nn_list[a]).intersection(set(nn_list[b]))
            for c in [x for x in common_ab if x > b]:
                common_abc = common_ab.intersection(set(nn_list[c]))
                for d in [x for x in common_abc if x > c]:
                    tetrahedra.append(tuple(sorted([a, b, c, d])))
    
    # Build mapping: vertex to ALL tetrahedra it belongs to
    # In pyrochlore, each site belongs to 2 tetrahedra (one up, one down)
    vertex_to_tets = {}
    for tet in tetrahedra:
        for v in tet:
            if v not in vertex_to_tets:
                vertex_to_tets[v] = []
            vertex_to_tets[v].append(tet)
    
    def check_ice_rule_preserved(chain):
        """
        Check if a chain of 4 sites preserves the ice rule.
        Each tetrahedron touched must be visited an even number of times.
        Since each site belongs to 2 tetrahedra, we need to check all tetrahedra
        that are touched by any site in the chain.
        """
        tet_visit_count = {}
        for vertex_id in chain:
            # Each vertex belongs to 2 tetrahedra
            for tet_key in vertex_to_tets.get(vertex_id, []):
                tet_visit_count[tet_key] = tet_visit_count.get(tet_key, 0) + 1
        
        # Check if all counts are even
        for count in tet_visit_count.values():
            if count % 2 != 0:
                return False
        return True
    
    chains = []
    visited_chains = set()
    
    # Try all possible starting vertices
    for start_vertex in sorted(vertices.keys()):
        # Try to build chains starting from this vertex
        # Use DFS to explore all possible 4-site paths
        def dfs_find_chains(current_path):
            if len(current_path) == 4:
                # Check if this chain preserves ice rule
                if check_ice_rule_preserved(current_path):
                    # Normalize chain representation to avoid duplicates
                    # (sort to get canonical form)
                    chain_normalized = tuple(sorted(current_path))
                    if chain_normalized not in visited_chains:
                        visited_chains.add(chain_normalized)
                        chains.append(list(current_path))
                return
            
            current_vertex = current_path[-1]
            for neighbor in nn_list[current_vertex]:
                # Allow revisiting vertices for loops, but limit to reasonable paths
                if neighbor not in current_path or (len(current_path) == 3 and neighbor == current_path[0]):
                    # If we're at the 3rd site and neighbor is the start, we have a 4-site loop
                    if len(current_path) == 3 and neighbor == current_path[0]:
                        # Create a closed loop
                        loop_path = current_path + [neighbor]
                        if check_ice_rule_preserved(current_path):  # Check the 4 unique vertices
                            chain_normalized = tuple(sorted(current_path))
                            if chain_normalized not in visited_chains:
                                visited_chains.add(chain_normalized)
                                chains.append(list(current_path))
                    elif neighbor not in current_path:
                        dfs_find_chains(current_path + [neighbor])
        
        dfs_find_chains([start_vertex])
    
    # Verbose output: verify each chain
    print("\n" + "="*80)
    print(f"VERIFICATION: Found {len(chains)} chains that preserve ice rule")
    print("="*80)
    for i, chain in enumerate(chains):
        print(f"\nChain {i+1}: {chain}")
        
        # Build detailed tracking of which sites contribute to which tetrahedra
        tet_contributions = {}  # Maps tet -> list of (site, [tets_for_site])
        for vertex_id in chain:
            site_tets = vertex_to_tets.get(vertex_id, [])
            for tet_key in site_tets:
                if tet_key not in tet_contributions:
                    tet_contributions[tet_key] = []
                tet_contributions[tet_key].append((vertex_id, site_tets))
        
        # Show each site and which tetrahedra it belongs to
        print(f"  Site-to-Tetrahedra membership in this chain:")
        for vertex_id in chain:
            site_tets = vertex_to_tets.get(vertex_id, [])
            print(f"    Site {vertex_id} belongs to {len(site_tets)} tetrahedra: {site_tets}")
        
        # Find all tetrahedra touched by this chain
        tet_visit_count = {}
        for vertex_id in chain:
            for tet_key in vertex_to_tets.get(vertex_id, []):
                tet_visit_count[tet_key] = tet_visit_count.get(tet_key, 0) + 1
        
        print(f"\n  Tetrahedra touched and visit breakdown:")
        all_even = True
        for tet_key, count in sorted(tet_visit_count.items()):
            even_str = "✓ EVEN" if count % 2 == 0 else "✗ ODD"
            print(f"    Tetrahedron {tet_key}:")
            print(f"      Visited {count} times {even_str}")
            print(f"      Contributing sites:", end=" ")
            contributing_sites = [site for site, _ in tet_contributions[tet_key]]
            print(f"{contributing_sites}")
            print(f"      Detail: ", end="")
            for site in contributing_sites:
                print(f"site {site} ∈ {tet_key}", end="; ")
            print()
            if count % 2 != 0:
                all_even = False
        
        if all_even:
            print(f"  ✓ All tetrahedra visited even number of times - ICE RULE PRESERVED")
        else:
            print(f"  ✗ ERROR: Some tetrahedra visited odd number of times!")
    
    print("\n" + "="*80)
    print(f"Verification complete: {len(chains)} valid chains")
    print("="*80 + "\n")
    
    return chains

def write_counter_term(output_dir, chains, Jxx, Jyy, Jzz, counterterm_coeff=1.0, file_name="CounterTerm.dat"):
    """
    Write counter term chains to a file in InterAll.dat format.
    Each chain contributes terms for each connected edge.
    
    For each edge in a chain, we add two lines:
    - One with operators 0 1 0 1 (S+ S- S+ S-)
    - One with operators 1 0 1 0 (S- S+ S- S+)
    
    Coefficient is counterterm_coeff * 4*(Jpm^2)/Jzz where Jpm = -(Jxx+Jyy)/4
    
    Args:
        counterterm_coeff: Multiplier for the counter term strength (default 1.0)
    """
    # Calculate coefficient
    Jpm = -(Jxx + Jyy) / 4
    coeff = counterterm_coeff * 4 * (Jpm**2) / Jzz
    
    # Actually, re-reading the request: "two lines per connected edge"
    # Let me reinterpret: for each edge in the chain, write two lines
    counter_terms = []
    
    for chain in chains:
        # First line: 0 site_i 1 site_j 0 site_i 1 site_j (S+ S- S+ S-)
        counter_terms.append([0, chain[0], 1, chain[1], 0, chain[2], 1, chain[3], coeff, 0])

        # Second line: 1 site_i 0 site_j 1 site_i 0 site_j (S- S+ S- S+)
        counter_terms.append([1, chain[0], 0, chain[1], 1, chain[2], 0, chain[3], coeff, 0])

    num_terms = len(counter_terms)
    
    with open(f"{output_dir}/{file_name}", 'w') as f:
        f.write("===================\n")
        f.write(f"num {num_terms:8d}\n")
        f.write("===================\n")
        f.write("===================\n")
        f.write("===================\n")
        
        for term in counter_terms:
            f.write(f" {int(term[0]):8d} " \
                   f" {int(term[1]):8d}   " \
                   f" {int(term[2]):8d}   " \
                   f" {int(term[3]):8d}   " \
                   f" {int(term[4]):8d}   " \
                   f" {int(term[5]):8d}   " \
                   f" {int(term[6]):8d}   " \
                   f" {int(term[7]):8d}   " \
                   f" {term[8]:8f}   " \
                   f" {term[9]:8f}   " \
                   f"\n")

def plot_counter_term_chains(vertices, edges, chains, output_dir, cluster_name, sublattice_indices=None):
    """
    Plot the cluster with counter term chains highlighted
    """
    try:
        import matplotlib.pyplot as plt
        from matplotlib.ticker import NullFormatter

        # Publication-quality params
        mpl.rcParams['font.family'] = 'serif'
        mpl.rcParams['font.serif'] = ['Computer Modern Roman', 'Times New Roman']
        mpl.rcParams['font.size'] = 10
        mpl.rcParams['axes.labelsize'] = 12
        mpl.rcParams['axes.titlesize'] = 12
        mpl.rcParams['xtick.labelsize'] = 10
        mpl.rcParams['ytick.labelsize'] = 10
        mpl.rcParams['legend.fontsize'] = 9
        mpl.rcParams['figure.dpi'] = 100
        mpl.rcParams['savefig.dpi'] = 300
        mpl.rcParams['axes.linewidth'] = 1.0
        mpl.rcParams['xtick.major.width'] = 0.8
        mpl.rcParams['ytick.major.width'] = 0.8

        # Helper: equal aspect ratio
        def set_equal_aspect_3d(ax, pts):
            pts = np.asarray(pts)
            mins = pts.min(axis=0)
            maxs = pts.max(axis=0)
            centers = (mins + maxs) / 2.0
            max_range = ((maxs - mins).max()) / 2.0
            ax.set_xlim(centers[0] - max_range, centers[0] + max_range)
            ax.set_ylim(centers[1] - max_range, centers[1] + max_range)
            ax.set_zlim(centers[2] - max_range, centers[2] + max_range)

        # Create figure with two subplots
        fig = plt.figure(figsize=(14, 6.5))
        
        # Muted, colorblind-friendly sublattice palette
        muted_colors = ['#0072B2', '#009E73', '#E69F00', '#CC79A7']
        
        # Create a set of all edges in chains for easy lookup
        chain_edges = set()
        for chain in chains:
            for i in range(len(chain) - 1):
                edge = tuple(sorted([chain[i], chain[i+1]]))
                chain_edges.add(edge)
        
        # Get all vertices in chains
        chain_vertices = set()
        for chain in chains:
            chain_vertices.update(chain)
        
        if sublattice_indices is None:
            sublattice_indices = {}
        
        # --- Left plot: Regular lattice structure ---
        ax1 = fig.add_subplot(121, projection='3d')
        
        # Plot all edges faintly
        for v1, v2 in edges:
            p1 = np.array(vertices[v1])
            p2 = np.array(vertices[v2])
            ax1.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]],
                    color='gray', alpha=0.3, linewidth=0.5, zorder=1)
        
        # Plot vertices by sublattice
        for sub_idx in range(4):
            sub_ids = [v for v in vertices if sublattice_indices.get(v, v % 4) == sub_idx]
            if not sub_ids:
                continue
            sub_positions = np.array([vertices[v] for v in sub_ids])
            ax1.scatter(sub_positions[:, 0], sub_positions[:, 1], sub_positions[:, 2],
                        s=50, c=muted_colors[sub_idx], marker='o', alpha=0.8,
                        edgecolors='black', linewidth=0.5,
                        label=f'Sublattice {sub_idx}', depthshade=True, zorder=2)
        
        ax1.set_title('Pyrochlore Lattice Structure', fontsize=12, pad=10)
        set_equal_aspect_3d(ax1, list(vertices.values()))
        ax1.view_init(elev=24, azim=135)
        ax1.grid(True, alpha=0.3, linewidth=0.5)
        ax1.xaxis.set_major_formatter(NullFormatter())
        ax1.yaxis.set_major_formatter(NullFormatter())
        ax1.zaxis.set_major_formatter(NullFormatter())
        
        leg1 = ax1.legend(loc='upper left', frameon=True, fancybox=False, shadow=False, 
                         framealpha=0.9, edgecolor='black', borderpad=0.4, 
                         columnspacing=0.8, handlelength=1.5, handletextpad=0.4)
        leg1.get_frame().set_linewidth(0.5)
        
        # --- Right plot: Counter term chains highlighted ---
        ax2 = fig.add_subplot(122, projection='3d')
        
        # Plot all edges very faintly
        for v1, v2 in edges:
            edge = tuple(sorted([v1, v2]))
            p1 = np.array(vertices[v1])
            p2 = np.array(vertices[v2])
            if edge in chain_edges:
                # Counter term edges - thick and colored
                ax2.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]],
                        color='red', alpha=0.8, linewidth=2.5, zorder=3)
            else:
                # Regular edges - very faint
                ax2.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]],
                        color='gray', alpha=0.15, linewidth=0.3, zorder=1)
        
        # Plot vertices
        for sub_idx in range(4):
            sub_ids = [v for v in vertices if sublattice_indices.get(v, v % 4) == sub_idx]
            if not sub_ids:
                continue
            
            # Separate chain vertices from others
            chain_sub_ids = [v for v in sub_ids if v in chain_vertices]
            other_sub_ids = [v for v in sub_ids if v not in chain_vertices]
            
            # Other vertices - small and faint
            if other_sub_ids:
                other_positions = np.array([vertices[v] for v in other_sub_ids])
                ax2.scatter(other_positions[:, 0], other_positions[:, 1], other_positions[:, 2],
                           s=20, c=muted_colors[sub_idx], marker='o', alpha=0.3,
                           edgecolors='black', linewidth=0.3, depthshade=True, zorder=2)
            
            # Chain vertices - larger and prominent
            if chain_sub_ids:
                chain_positions = np.array([vertices[v] for v in chain_sub_ids])
                ax2.scatter(chain_positions[:, 0], chain_positions[:, 1], chain_positions[:, 2],
                           s=60, c=muted_colors[sub_idx], marker='o', alpha=0.9,
                           edgecolors='red', linewidth=1.0,
                           label=f'Sublattice {sub_idx}' if sub_idx == 0 else None,
                           depthshade=True, zorder=4)
        
        # Add labels to chain vertices
        for v in chain_vertices:
            pos = vertices[v]
            ax2.text(pos[0], pos[1], pos[2], f'{v}', fontsize=7, 
                    ha='center', va='bottom', color='black', weight='bold', zorder=5)
        
        ax2.set_title(f'Counter Term Chains ({len(chains)} chains)', fontsize=12, pad=10)
        set_equal_aspect_3d(ax2, list(vertices.values()))
        ax2.view_init(elev=24, azim=135)
        ax2.grid(True, alpha=0.3, linewidth=0.5)
        ax2.xaxis.set_major_formatter(NullFormatter())
        ax2.yaxis.set_major_formatter(NullFormatter())
        ax2.zaxis.set_major_formatter(NullFormatter())
        
        # Add custom legend for counter term chains
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color='red', linewidth=2.5, label='Counter term chains'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', 
                   markeredgecolor='red', markersize=8, label='Chain vertices', linewidth=0)
        ]
        leg2 = ax2.legend(handles=legend_elements, loc='upper left', frameon=True,
                         fancybox=False, shadow=False, framealpha=0.9, 
                         edgecolor='black', borderpad=0.4, columnspacing=0.8,
                         handlelength=1.5, handletextpad=0.4)
        leg2.get_frame().set_linewidth(0.5)
        
        plt.tight_layout(pad=1.0)
        
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)
        
        plt.savefig(f"{output_dir}/{cluster_name}_counter_terms.png",
                    dpi=300, bbox_inches='tight', pad_inches=0.05)
        plt.savefig(f"{output_dir}/{cluster_name}_counter_terms.pdf",
                    bbox_inches='tight', pad_inches=0.05)
        
        print(f"Counter term visualization saved to: {output_dir}/{cluster_name}_counter_terms.png")
        plt.close(fig)
        return True

    except ImportError:
        print("Warning: matplotlib not installed, skipping counter term plot")
        return False


def plot_cluster(vertices, edges, output_dir, cluster_name, sublattice_indices=None):
    """
    Plot the cluster showing sites, bonds, and tetrahedra with muted sublattice colors
    and two-tone tetrahedron shading. Adjust POV for clearer 3D perception.
    """
    try:
        import matplotlib.pyplot as plt

        # Publication-quality params
        mpl.rcParams['font.family'] = 'serif'
        mpl.rcParams['font.serif'] = ['Computer Modern Roman', 'Times New Roman']
        mpl.rcParams['font.size'] = 10
        mpl.rcParams['axes.labelsize'] = 12
        mpl.rcParams['axes.titlesize'] = 12
        mpl.rcParams['xtick.labelsize'] = 10
        mpl.rcParams['ytick.labelsize'] = 10
        mpl.rcParams['legend.fontsize'] = 10
        mpl.rcParams['figure.dpi'] = 100
        mpl.rcParams['savefig.dpi'] = 300
        mpl.rcParams['axes.linewidth'] = 1.0
        mpl.rcParams['xtick.major.width'] = 0.8
        mpl.rcParams['ytick.major.width'] = 0.8

        # Helper: equal aspect ratio
        def set_equal_aspect_3d(ax, pts):
            pts = np.asarray(pts)
            mins = pts.min(axis=0)
            maxs = pts.max(axis=0)
            centers = (mins + maxs) / 2.0
            max_range = ((maxs - mins).max()) / 2.0
            ax.set_xlim(centers[0] - max_range, centers[0] + max_range)
            ax.set_ylim(centers[1] - max_range, centers[1] + max_range)
            ax.set_zlim(centers[2] - max_range, centers[2] + max_range)

        # Helper: plot one tetrahedron with requested color scheme
        def _plot_tetra(ax, coords, down=1, a=0.35, ap=1):
            # Two-tone colors per user's spec
            c = (59/256, 137/256, 255/256) if down == 1 else (42/256, 232/256, 137/256)
            tri_idx = [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]]
            ax.plot_trisurf(coords[:, 0], coords[:, 1], coords[:, 2],
                            triangles=tri_idx,
                            edgecolor=[[0.1, 0.1, 0.1]],
                            linewidth=0.6, alpha=a, shade=True, color=c, zorder=0)

        # Build adjacency
        adj = {v: set() for v in vertices}
        for a, b in edges:
            adj[a].add(b)
            adj[b].add(a)

        # Detect tetrahedra as 4-cliques in NN graph
        tetra_indices = []
        for a in adj:
            for b in [x for x in adj[a] if x > a]:
                common_ab = adj[a].intersection(adj[b])
                for c in [x for x in common_ab if x > b]:
                    common_abc = common_ab.intersection(adj[c])
                    for d in [x for x in common_abc if x > c]:
                        tetra_indices.append((a, b, c, d))

        # Precompute positions and defaults
        v_ids = list(vertices.keys())
        pos_arr = np.array([vertices[v] for v in v_ids])

        # Figure
        fig = plt.figure(figsize=(7.2, 7.2/1.35))
        ax = fig.add_subplot(111, projection='3d')

        # Plot tetrahedra first
        for (a, b, c, d) in tetra_indices:
            coords = np.array([vertices[a], vertices[b], vertices[c], vertices[d]], dtype=float)
            # Orientation via signed volume to choose color
            V = np.linalg.det(np.vstack([coords[0]-coords[3], coords[1]-coords[3], coords[2]-coords[3]])) / 6.0
            down = 1 if V < 0 else -1
            _plot_tetra(ax, coords, down=down, a=0.35, ap=1)

        # Muted, colorblind-friendly sublattice palette
        muted_colors = ['#0072B2', '#009E73', '#E69F00', '#CC79A7']

        # Plot vertices by sublattice
        if sublattice_indices is None:
            sublattice_indices = {}

        for sub_idx in range(4):
            sub_ids = [v for v in vertices if sublattice_indices.get(v, v % 4) == sub_idx]
            if not sub_ids:
                continue
            sub_positions = np.array([vertices[v] for v in sub_ids])
            ax.scatter(sub_positions[:, 0], sub_positions[:, 1], sub_positions[:, 2],
                        s=45, c=muted_colors[sub_idx], marker='o', alpha=0.9,
                        edgecolors='black', linewidth=0.4,
                        label=f'Sublattice {sub_idx}', depthshade=True, zorder=2)

        # Plot edges faintly
        for v1, v2 in edges:
            p1 = np.array(vertices[v1])
            p2 = np.array(vertices[v2])
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]],
                    color='k', alpha=0.5, linewidth=0.5, zorder=1)

        # Aspect and limits
        set_equal_aspect_3d(ax, list(vertices.values()))

        # POV: isometric-like view for better 3D readability
        ax.view_init(elev=24, azim=135)

        # Show grid but hide axis numbers
        ax.grid(True, alpha=0.35, linewidth=0.6)
        ax.xaxis.set_major_formatter(NullFormatter())
        ax.yaxis.set_major_formatter(NullFormatter())
        ax.zaxis.set_major_formatter(NullFormatter())

        # Legend
        leg = ax.legend(loc='upper left', frameon=True,
                        fancybox=False, shadow=False, framealpha=0.9,
                        edgecolor='black', borderpad=0.5, columnspacing=1.0,
                        handlelength=1.5, handletextpad=0.5)
        leg.get_frame().set_linewidth(0.5)

        # Layout and save
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)

        plt.tight_layout(pad=0.6)

        plt.savefig(f"{output_dir}/{cluster_name}_plot.png",
                    dpi=300, bbox_inches='tight', pad_inches=0.02)
        plt.savefig(f"{output_dir}/{cluster_name}_plot.pdf",
                    bbox_inches='tight', pad_inches=0.02)
        plt.savefig(f"{output_dir}/{cluster_name}_plot.eps",
                    bbox_inches='tight', pad_inches=0.02)

        plt.close(fig)
        return True

    except ImportError:
        print("Warning: matplotlib not installed, skipping cluster plot")
        return False

def main():
    """Main function to process command line arguments and run the program"""
    if len(sys.argv) < 13:
        print("Usage: python helper_pyrochlore_super.py Jxx Jyy Jzz h fieldx fieldy fieldz output_dir dim1 dim2 dim3 pbc [non_kramer] [theta] [counterterm_coeff]")
        sys.exit(1)
    
    # Parse command line arguments
    Jxx = float(sys.argv[1])
    Jyy = float(sys.argv[2])
    Jzz = float(sys.argv[3])
    h = float(sys.argv[4])
    field_dir = [float(sys.argv[5]), float(sys.argv[6]), float(sys.argv[7])]
    output_dir = sys.argv[8]
    dim1 = int(sys.argv[9])
    dim2 = int(sys.argv[10])
    dim3 = int(sys.argv[11])
    use_pbc = bool(int(sys.argv[12]))
    non_kramer = bool(int(sys.argv[13])) if len(sys.argv) > 13 else False
    theta = float(sys.argv[14]) if len(sys.argv) > 14 else 0.0  # Default theta=0.0 if not provided
    theta = theta * np.pi
    counterterm_coeff = float(sys.argv[15]) if len(sys.argv) > 15 else 1.0  # Default counterterm_coeff=1.0 if not provided
    # Ensure output directory exists
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    
    # Create cluster name
    pbc_str = "pbc" if use_pbc else "obc"
    non_kramer_str = "non_kramer" if non_kramer else "kramer"
    cluster_name = f"pyrochlore_super_{dim1}x{dim2}x{dim3}_{pbc_str}_{non_kramer_str}"

    # Generate cluster
    vertices, edges, tetrahedra, node_mapping, vertex_to_cell = generate_pyrochlore_super_cluster(dim1, dim2, dim3, use_pbc)
    
    # Create nearest neighbor lists
    nn_list, positions, sublattice_indices = create_nn_lists(edges, node_mapping, vertices, vertex_to_cell)
    
    # Write nearest neighbor list and site info
    write_cluster_nn_list(output_dir, cluster_name, nn_list, positions, sublattice_indices, node_mapping)
    
    # Prepare Hamiltonian parameters
    prepare_hamiltonian_parameters(output_dir, non_kramer, nn_list, positions, sublattice_indices, 
                                  node_mapping, Jxx, Jyy, Jzz, h, theta, field_dir)

    # Find and write counter term chains
    chains = find_counter_term_chains(vertices, nn_list, vertex_to_cell, dim1, dim2, dim3, use_pbc)
    write_counter_term(output_dir, chains, Jxx, Jyy, Jzz, counterterm_coeff)

    # Plot cluster
    plot_cluster(vertices, edges, output_dir, cluster_name, sublattice_indices)
    
    # Plot counter term chains
    plot_counter_term_chains(vertices, edges, chains, output_dir, cluster_name, sublattice_indices)
    
    print(f"Generated pyrochlore super lattice cluster with dimensions {dim1}x{dim2}x{dim3}")
    print(f"Number of sites: {len(vertices)}")
    print(f"Number of bonds: {len(edges)}")
    print(f"Number of tetrahedra: {len(tetrahedra)}")
    print(f"Sites per unit cell: 16 (4 tetrahedra × 4 sites)")
    print(f"Counter term coefficient: {counterterm_coeff}")
    print(f"Output written to: {output_dir}")

if __name__ == "__main__":
    main()