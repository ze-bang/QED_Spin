"""
Kagome lattice generator for the √3 × √3 super unit cell

The √3 × √3 supercell of the kagome lattice contains 9 sites per supercell.
This is the natural unit cell for √3 × √3 magnetic orderings.

Supercell lattice vectors (relative to primitive kagome):
- A1 = a1 + 2*a2 = (2, √3)
- A2 = -2*a1 + a2 = (-1.5, √3/2)  ->  rotated: A2 = a1 - a2 = (0.5, -√3/2) transformed
Actually using the standard convention:
- A1 = a1 - a2 
- A2 = a1 + 2*a2
which gives |A1| = |A2| = √3 (in units where |a1| = 1)

The 9 sites come from 3 primitive kagome cells that tile the supercell.
"""

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


def generate_kagome_sqrt3_cluster(dim1, dim2, use_pbc=False):
    """
    Generate a kagome lattice cluster using the √3 × √3 super unit cell
    
    The √3 × √3 supercell contains 9 sites arranged as follows:
    - 3 primitive unit cells fit within each supercell
    - Each primitive cell has 3 sites → 9 sites total per supercell
    
    Supercell lattice vectors (in units where primitive a1 = (1, 0)):
    - A1 = a1 - a2 = (0.5, -√3/2)   →  rotated to A1 = (√3/2, -1/2) * √3 = (1.5, -√3/2)
    Actually, the standard √3×√3 supercell uses:
    - A1 = 2*a1 - a2 = (1.5, -√3/2)
    - A2 = a1 + a2 = (1.5, √3/2)
    
    This gives |A1| = |A2| = √3 and angle between them = 60°
    
    Args:
        dim1, dim2: Dimensions of the lattice (number of √3×√3 supercells)
        use_pbc: Whether to use periodic boundary conditions
        
    Returns:
        vertices: Dictionary of {vertex_id: (x, y)}
        edges: List of (vertex1, vertex2) tuples for nearest neighbors
        edges_2nn: List of (vertex1, vertex2) tuples for second nearest neighbors
        edges_3nn: List of (vertex1, vertex2) tuples for third nearest neighbors
        node_mapping: Dictionary mapping original IDs to matrix indices
        vertex_to_cell: Map vertex_id to (i, j, site_idx)
    """
    # Primitive kagome lattice vectors
    a1 = np.array([1.0, 0.0])
    a2 = np.array([0.5, np.sqrt(3)/2])
    
    # √3 × √3 supercell lattice vectors
    # Standard choice that gives a 60° rhombus with edge length √3
    A1 = 2*a1 - a2  # = (1.5, -√3/2)
    A2 = a1 + a2    # = (1.5, √3/2)
    
    # The 9 sites within the supercell come from 3 primitive unit cells
    # Primitive cell origins within supercell: (0,0), (1,0), (0,1) in primitive coords
    # Each primitive cell has 3 sites at offsets: (0,0), (0.5,0), (0.25, √3/4)
    
    # Site positions within primitive unit cell
    primitive_site_offsets = np.array([
        [0.0, 0.0],                    # Site 0
        [0.5, 0.0],                    # Site 1
        [0.25, np.sqrt(3)/4]           # Site 2
    ])
    
    # Primitive cell origins within supercell (in primitive a1, a2 coordinates)
    # The √3×√3 supercell spans: A1 = 2*a1 - a2, A2 = a1 + a2
    # Sites at primitive (m, n) are in this supercell if:
    #   0 ≤ 2m - n < 3  and  0 ≤ m + n < 3 (for one choice)
    # We enumerate all 9 sites explicitly:
    
    supercell_primitive_cells = [
        (0, 0),  # Origin
        (1, 0),  # +a1
        (0, 1),  # +a2
    ]
    
    # Generate the 9 site offsets within the supercell
    supercell_site_offsets = []
    supercell_site_sublattice = []  # Track which kagome sublattice each site belongs to
    
    for prim_i, prim_j in supercell_primitive_cells:
        prim_origin = prim_i * a1 + prim_j * a2
        for site_idx in range(3):
            pos = prim_origin + primitive_site_offsets[site_idx]
            supercell_site_offsets.append(pos)
            supercell_site_sublattice.append(site_idx)  # Original kagome sublattice
    
    supercell_site_offsets = np.array(supercell_site_offsets)
    # Now we have 9 sites per supercell
    
    # Generate vertices
    vertices = {}
    vertex_id = 0
    vertex_to_cell = {}  # Map vertex_id to (i, j, site_idx)
    cell_to_vertex = {}  # Map (i, j, site_idx) to vertex_id
    
    for i in range(dim1):
        for j in range(dim2):
            # Supercell position
            supercell_pos = i * A1 + j * A2
            
            # For each of the 9 sites in the supercell
            for site_idx in range(9):
                position = supercell_pos + supercell_site_offsets[site_idx]
                vertices[vertex_id] = tuple(position)
                vertex_to_cell[vertex_id] = (i, j, site_idx)
                cell_to_vertex[(i, j, site_idx)] = vertex_id
                vertex_id += 1
    
    # Store sublattice mapping (which of the 3 kagome sublattices each of the 9 sites belongs to)
    site_to_kagome_sublattice = supercell_site_sublattice
    
    # Helper function to get vertex id with PBC
    def get_vertex_with_pbc(i, j, site_idx):
        if use_pbc:
            i, j = i % dim1, j % dim2
        elif i < 0 or i >= dim1 or j < 0 or j >= dim2:
            return None
        return cell_to_vertex.get((i, j, site_idx), None)
    
    def add_bond(edge_list, v1, v2):
        """Add a bond to edge list if v2 is valid and not a self-loop"""
        if v2 is not None and v1 != v2:
            edge_list.append(tuple(sorted([v1, v2])))
    
    # ==========================================================================
    # Bond connectivity for √3×√3 kagome supercell
    # 
    # The 9 sites in the supercell are indexed 0-8:
    # Sites 0,1,2 from primitive cell (0,0)
    # Sites 3,4,5 from primitive cell (1,0) 
    # Sites 6,7,8 from primitive cell (0,1)
    #
    # We need to enumerate all NN, 2NN, 3NN bonds
    # For 3NN, we only include the BFG-relevant bonds (same kagome sublattice,
    # along specific chain directions through hexagon centers)
    # ==========================================================================
    
    # NN distance = 0.5 in primitive units
    # 2NN distance = √3/2 ≈ 0.866
    # 3NN distance = 1.0
    
    NN_DIST = 0.5
    NN2_DIST = np.sqrt(3)/2
    NN3_DIST = 1.0
    TOL = 0.01
    
    # Define the specific 3NN directions for BFG model (in Cartesian coordinates)
    # For kagome lattice:
    # - Sublattice 0 (site_idx 0, 3, 6): chain along a1 - a2 = (0.5, -√3/2)
    # - Sublattice 1 (site_idx 1, 4, 7): chain along a2 = (0.5, √3/2)
    # - Sublattice 2 (site_idx 2, 5, 8): chain along a1 = (1.0, 0)
    NN3_DIRECTIONS = {
        0: np.array([0.5, -np.sqrt(3)/2]),   # a1 - a2
        1: np.array([0.5, np.sqrt(3)/2]),    # a2
        2: np.array([1.0, 0.0]),             # a1
    }
    
    # Precompute all bonds by distance within and between supercells
    # For internal bonds and bonds to neighboring supercells
    
    # Build bond tables by computing distances
    NN_BONDS = []   # (src_site, di, dj, tgt_site) where di, dj are supercell offsets
    NN2_BONDS = []
    NN3_BONDS = []
    
    # Check bonds within supercell and to neighboring supercells
    for src_site in range(9):
        src_pos = supercell_site_offsets[src_site]
        src_kagome_sub = site_to_kagome_sublattice[src_site]
        
        for di in [-1, 0, 1]:
            for dj in [-1, 0, 1]:
                cell_offset = di * A1 + dj * A2
                
                for tgt_site in range(9):
                    # Skip self for same cell
                    if di == 0 and dj == 0 and src_site >= tgt_site:
                        continue
                    # For different cells, only count in positive direction to avoid double counting
                    if di < 0 or (di == 0 and dj < 0):
                        continue
                    if di == 0 and dj == 0 and src_site > tgt_site:
                        continue
                        
                    tgt_pos = cell_offset + supercell_site_offsets[tgt_site]
                    disp = tgt_pos - src_pos
                    dist = np.linalg.norm(disp)
                    
                    if abs(dist - NN_DIST) < TOL:
                        NN_BONDS.append((src_site, di, dj, tgt_site))
                    elif abs(dist - NN2_DIST) < TOL:
                        NN2_BONDS.append((src_site, di, dj, tgt_site))
                    elif abs(dist - NN3_DIST) < TOL:
                        # For 3NN, check if it's along the BFG chain direction
                        # Must be same kagome sublattice and along the correct direction
                        tgt_kagome_sub = site_to_kagome_sublattice[tgt_site]
                        if src_kagome_sub == tgt_kagome_sub:
                            expected_dir = NN3_DIRECTIONS[src_kagome_sub]
                            # Check if displacement is parallel to expected direction
                            # (either +dir or -dir)
                            unit_disp = disp / dist
                            unit_expected = expected_dir / np.linalg.norm(expected_dir)
                            if abs(abs(np.dot(unit_disp, unit_expected)) - 1.0) < TOL:
                                NN3_BONDS.append((src_site, di, dj, tgt_site))
    
    # ==========================================================================
    # Generate edges using the bond tables
    # Use sets to avoid duplicate bonds (can happen with small clusters + PBC)
    # ==========================================================================
    
    edges_set = set()
    edges_2nn_set = set()
    edges_3nn_set = set()
    
    for i in range(dim1):
        for j in range(dim2):
            # Add NN bonds
            for src_site, di, dj, tgt_site in NN_BONDS:
                v_src = cell_to_vertex[(i, j, src_site)]
                v_tgt = get_vertex_with_pbc(i + di, j + dj, tgt_site)
                if v_tgt is not None and v_src != v_tgt:
                    edges_set.add(tuple(sorted([v_src, v_tgt])))
            
            # Add 2NN bonds
            for src_site, di, dj, tgt_site in NN2_BONDS:
                v_src = cell_to_vertex[(i, j, src_site)]
                v_tgt = get_vertex_with_pbc(i + di, j + dj, tgt_site)
                if v_tgt is not None and v_src != v_tgt:
                    edges_2nn_set.add(tuple(sorted([v_src, v_tgt])))
            
            # Add 3NN bonds
            for src_site, di, dj, tgt_site in NN3_BONDS:
                v_src = cell_to_vertex[(i, j, src_site)]
                v_tgt = get_vertex_with_pbc(i + di, j + dj, tgt_site)
                if v_tgt is not None and v_src != v_tgt:
                    edges_3nn_set.add(tuple(sorted([v_src, v_tgt])))
    
    # Convert sets to lists
    edges = list(edges_set)
    edges_2nn = list(edges_2nn_set)
    edges_3nn = list(edges_3nn_set)
    
    # Create node mapping
    node_mapping = {i: i for i in range(len(vertices))}
    
    # Store additional info for later use
    vertex_to_cell['_supercell_site_offsets'] = supercell_site_offsets
    vertex_to_cell['_site_to_kagome_sublattice'] = site_to_kagome_sublattice
    vertex_to_cell['_A1'] = A1
    vertex_to_cell['_A2'] = A2
    
    return vertices, edges, edges_2nn, edges_3nn, node_mapping, vertex_to_cell


def get_sublattice_index(vertex_id, vertex_to_cell):
    """
    Return the sublattice index (0-8 for √3×√3 supercell)
    For compatibility, also maps to kagome sublattice (0, 1, or 2)
    """
    if vertex_id in vertex_to_cell:
        _, _, site_idx = vertex_to_cell[vertex_id]
        return site_idx
    return vertex_id % 9


def get_kagome_sublattice_index(vertex_id, vertex_to_cell):
    """
    Return the original kagome sublattice index (0, 1, or 2)
    """
    if '_site_to_kagome_sublattice' in vertex_to_cell:
        site_to_kagome = vertex_to_cell['_site_to_kagome_sublattice']
        if vertex_id in vertex_to_cell:
            _, _, site_idx = vertex_to_cell[vertex_id]
            return site_to_kagome[site_idx]
    # Fallback
    return vertex_id % 3


def create_nn_lists(edges, edges_2nn, edges_3nn, node_mapping, vertices, vertex_to_cell):
    """
    Create nearest neighbor lists from the edge information
    
    Args:
        edges: List of nearest neighbor edges
        edges_2nn: List of second nearest neighbor edges
        edges_3nn: List of third nearest neighbor edges
        node_mapping: Dictionary mapping original IDs to matrix indices
        vertices: Dictionary of {vertex_id: (x, y)}
        vertex_to_cell: Map vertex_id to (i, j, site_idx)
        
    Returns:
        nn_list: Dictionary mapping each site to its nearest neighbors
        nn_list_2nn: Dictionary mapping each site to its second nearest neighbors
        nn_list_3nn: Dictionary mapping each site to its third nearest neighbors
        positions: Dictionary mapping each site to its position
        sublattice_indices: Dictionary mapping each site to its sublattice index
    """
    nn_list = {}
    nn_list_2nn = {}
    nn_list_3nn = {}
    positions = {}
    sublattice_indices = {}
    
    # Initialize empty lists for all vertices
    for vertex_id in vertices:
        nn_list[vertex_id] = []
        nn_list_2nn[vertex_id] = []
        nn_list_3nn[vertex_id] = []
        positions[vertex_id] = vertices[vertex_id]
        sublattice_indices[vertex_id] = get_sublattice_index(vertex_id, vertex_to_cell)
    
    # Fill nearest neighbor lists based on edges
    for v1, v2 in edges:
        nn_list[v1].append(v2)
        nn_list[v2].append(v1)
    
    # Fill second nearest neighbor lists
    for v1, v2 in edges_2nn:
        nn_list_2nn[v1].append(v2)
        nn_list_2nn[v2].append(v1)
    
    # Fill third nearest neighbor lists
    for v1, v2 in edges_3nn:
        nn_list_3nn[v1].append(v2)
        nn_list_3nn[v2].append(v1)
    
    return nn_list, nn_list_2nn, nn_list_3nn, positions, sublattice_indices


def write_cluster_nn_list(output_dir, cluster_name, nn_list, nn_list_2nn, nn_list_3nn, 
                          positions, sublattice_indices, node_mapping, vertex_to_cell=None):
    """
    Write nearest neighbor lists, positions, and sublattice indices to files
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
    
    # Write second nearest neighbor list
    with open(f"{output_dir}/{cluster_name}_2nn_list.dat", 'w') as f:
        f.write("# Second nearest neighbor list for cluster: " + cluster_name + "\n")
        f.write("# Format: site_id, number_of_neighbors, [neighbor_ids]\n\n")
        
        for site_id in sorted(nn_list_2nn.keys()):
            neighbors = nn_list_2nn[site_id]
            matrix_index = node_mapping.get(site_id, site_id)
            
            f.write(f"{site_id} {len(neighbors)}")
            for neighbor in neighbors:
                f.write(f" {neighbor}")
            f.write("\n")
    
    # Write third nearest neighbor list
    with open(f"{output_dir}/{cluster_name}_3nn_list.dat", 'w') as f:
        f.write("# Third nearest neighbor list for cluster: " + cluster_name + "\n")
        f.write("# Format: site_id, number_of_neighbors, [neighbor_ids]\n\n")
        
        for site_id in sorted(nn_list_3nn.keys()):
            neighbors = nn_list_3nn[site_id]
            matrix_index = node_mapping.get(site_id, site_id)
            
            f.write(f"{site_id} {len(neighbors)}")
            for neighbor in neighbors:
                f.write(f" {neighbor}")
            f.write("\n")
    
    # Write site positions and sublattice indices
    with open(f"{output_dir}/positions.dat", 'w') as f:
        f.write("# Site information for cluster: " + cluster_name + "\n")
        f.write("# site_id, matrix_index, sublattice_index (0-8), kagome_sublattice (0-2), x, y\n\n")
        
        for site_id in sorted(positions.keys()):
            pos = positions[site_id]
            sub_idx = sublattice_indices[site_id]
            kagome_sub = get_kagome_sublattice_index(site_id, vertex_to_cell)
            matrix_index = node_mapping.get(site_id, site_id)
            
            f.write(f"{site_id} {matrix_index} {sub_idx} {kagome_sub} {pos[0]:.6f} {pos[1]:.6f} 0.000000\n")

    # Write lattice parameters
    with open(f"{output_dir}/{cluster_name}_lattice_parameters.dat", 'w') as f:
        f.write("# Kagome lattice parameters (√3 × √3 supercell)\n")
        f.write("# Generated for BFG (Balents-Fisher-Girvin) model\n\n")
        
        # Primitive kagome lattice vectors
        a1 = np.array([1.0, 0.0])
        a2 = np.array([0.5, np.sqrt(3)/2])
        
        # Supercell vectors
        A1 = 2*a1 - a2
        A2 = a1 + a2
        
        # Write lattice type
        f.write("# Lattice type: Kagome √3×√3 supercell (2D)\n")
        f.write("# Sites per supercell: 9\n")
        f.write("# Coordination number (NN): 4 per site\n\n")
        
        # Write dimensions
        f.write("# Cluster dimensions\n")
        if vertex_to_cell:
            # Filter out metadata keys
            valid_entries = [(k, v) for k, v in vertex_to_cell.items() 
                           if isinstance(k, int)]
            if valid_entries:
                max_i = max(v[0] for k, v in valid_entries)
                max_j = max(v[1] for k, v in valid_entries)
                f.write(f"# √3×√3 supercells: {max_i + 1} x {max_j + 1}\n")
                f.write(f"# Total sites: {len(positions)}\n\n")
        
        # Write primitive lattice vectors
        f.write("# Primitive kagome lattice vectors\n")
        f.write("# vector_index, x, y\n")
        f.write(f"0 {a1[0]:.6f} {a1[1]:.6f}\n")
        f.write(f"1 {a2[0]:.6f} {a2[1]:.6f}\n\n")
        
        # Write supercell lattice vectors
        f.write("# √3×√3 supercell lattice vectors\n")
        f.write("# A1 = 2*a1 - a2, A2 = a1 + a2\n")
        f.write("# vector_index, x, y\n")
        f.write(f"0 {A1[0]:.6f} {A1[1]:.6f}\n")
        f.write(f"1 {A2[0]:.6f} {A2[1]:.6f}\n")
        
        f.write("\n")
        
        # Write site offsets within supercell
        f.write("# Site offsets within √3×√3 supercell\n")
        f.write("# site_index, kagome_sublattice, x, y\n")
        if '_supercell_site_offsets' in vertex_to_cell:
            offsets = vertex_to_cell['_supercell_site_offsets']
            kagome_sub = vertex_to_cell.get('_site_to_kagome_sublattice', list(range(9)))
            for idx, offset in enumerate(offsets):
                f.write(f"{idx} {kagome_sub[idx]} {offset[0]:.6f} {offset[1]:.6f}\n")
        
        f.write("\n")
        
        # Calculate and write characteristic distances
        f.write("# Characteristic distances\n")
        nn_dist = 0.5  # Distance between NN sites
        f.write(f"# NN distance: {nn_dist:.6f}\n")
        f.write(f"# 2NN distance: {nn_dist * np.sqrt(3):.6f}\n")
        f.write(f"# 3NN distance: {nn_dist * 2:.6f}\n\n")
        
        # Write reciprocal lattice vectors for supercell
        f.write("# Reciprocal lattice vectors (for √3×√3 supercell)\n")
        det = A1[0]*A2[1] - A1[1]*A2[0]
        B1 = 2*np.pi*np.array([A2[1], -A2[0]]) / det
        B2 = 2*np.pi*np.array([-A1[1], A1[0]]) / det
        f.write("# vector_index, kx, ky\n")
        f.write(f"0 {B1[0]:.6f} {B1[1]:.6f}\n")
        f.write(f"1 {B2[0]:.6f} {B2[1]:.6f}\n")
        
        f.write("\n")
        
        # Write allowed k-points for this finite cluster
        f.write("# Allowed momentum points (k-points) for finite cluster\n")
        if vertex_to_cell:
            valid_entries = [(k, v) for k, v in vertex_to_cell.items() 
                           if isinstance(k, int)]
            if valid_entries:
                max_i = max(v[0] for k, v in valid_entries)
                max_j = max(v[1] for k, v in valid_entries)
                dim1_actual = max_i + 1
                dim2_actual = max_j + 1
                
                f.write(f"# Grid dimensions: {dim1_actual} x {dim2_actual}\n")
                f.write(f"# k-point mesh: For periodic BC, k = (n1/N1)*B1 + (n2/N2)*B2\n")
                f.write("# Format: k_index, n1, n2, kx, ky\n")
                
                # Generate k-points
                k_index = 0
                for n1 in range(dim1_actual):
                    for n2 in range(dim2_actual):
                        kx = (n1 / dim1_actual) * B1[0] + (n2 / dim2_actual) * B2[0]
                        ky = (n1 / dim1_actual) * B1[1] + (n2 / dim2_actual) * B2[1]
                        f.write(f"{k_index} {n1} {n2} {kx:.6f} {ky:.6f}\n")
                        k_index += 1
                
                f.write(f"\n# Total number of k-points: {k_index}\n")
                
                # Add high-symmetry points for reference
                f.write("\n# High-symmetry points in the reduced Brillouin zone\n")
                f.write("# Gamma: (0, 0)\n")
                f.write(f"# K: ({B1[0]/3 + B2[0]/3:.6f}, {B1[1]/3 + B2[1]/3:.6f})\n")
                f.write(f"# M: ({B1[0]/2:.6f}, {B1[1]/2:.6f})\n")


def prepare_hamiltonian_parameters(output_dir, nn_list, nn_list_2nn, nn_list_3nn, 
                                  positions, sublattice_indices, node_mapping, 
                                  Jpm, Jzz, Jzz_2nn, Jzz_3nn, h, theta, field_dir):
    """
    Prepare Hamiltonian parameters for exact diagonalization
    
    BFG Model:
    H = sum_{<ij>_NN} [Jpm(S_i^+ S_j^- + S_i^- S_j^+) + Jzz S_i^z S_j^z]
        + sum_{<ij>_2NN} Jzz_2nn S_i^z S_j^z
        + sum_{<ij>_3NN} Jzz_3nn S_i^z S_j^z
        - h * (sin(theta) S^x + cos(theta) S^z)
    
    Args:
        Jpm: XY exchange for nearest neighbors (Jpm = -(Jxx+Jyy)/4)
        Jzz: Ising exchange for nearest neighbors
        Jzz_2nn: Ising exchange for second nearest neighbors
        Jzz_3nn: Ising exchange for third nearest neighbors
        h: Magnetic field strength
        theta: Field angle (0 = along z, pi/2 = along x)
        field_dir: Field direction vector [x, y, z] (for 2D kagome, typically [1, 0, 0] or similar)
    """
    # Normalize field direction (though for 2D kagome we mainly use theta)
    field_dir = np.array(field_dir)
    if np.linalg.norm(field_dir) > 1e-10:
        field_dir = field_dir / np.linalg.norm(field_dir)
    
    interALL = []
    transfer = []
    
    # Generate exchange interactions
    for site_id in sorted(nn_list.keys()):
        # Site index (for the Hamiltonian)
        i = site_id
        
        # Zeeman term
        transfer.append([2, node_mapping[i], -h, 0])  # Sz term
        
        # Nearest neighbor interactions
        for neighbor_id in nn_list[site_id]:
            if site_id < neighbor_id:  # Only add each bond once
                j = neighbor_id
                
                # BFG interactions: Jpm(S+S- + S-S+) + Jzz Sz Sz
                interALL.append([2, node_mapping[i], 2, node_mapping[j], Jzz, 0])     # Sz-Sz
                interALL.append([0, node_mapping[i], 1, node_mapping[j], -Jpm, 0])    # S+-S-
                interALL.append([1, node_mapping[i], 0, node_mapping[j], -Jpm, 0])    # S--S+
        
        # Second nearest neighbor interactions (only Jzz)
        for neighbor_id in nn_list_2nn[site_id]:
            if site_id < neighbor_id:  # Only add each bond once
                j = neighbor_id
                interALL.append([2, node_mapping[i], 2, node_mapping[j], Jzz_2nn, 0])  # Sz-Sz
        
        # Third nearest neighbor interactions (only Jzz)
        for neighbor_id in nn_list_3nn[site_id]:
            if site_id < neighbor_id:  # Only add each bond once
                j = neighbor_id
                interALL.append([2, node_mapping[i], 2, node_mapping[j], Jzz_3nn, 0])  # Sz-Sz
        
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


def plot_connectivity_by_site(vertices, edges, edges_2nn, edges_3nn, output_dir, cluster_name, 
                               vertex_to_cell, dim1, dim2, use_pbc=True):
    """
    Plot NN, 2NN, 3NN connectivity for selected sites to visually verify connections.
    
    For the √3×√3 supercell, we show connectivity for sites 0, 3, 6 (one from each
    primitive cell within the supercell) as representative examples.
    """
    try:
        import matplotlib.pyplot as plt
        from collections import defaultdict
        
        # Build neighbor lists
        nn_list = defaultdict(list)
        nn2_list = defaultdict(list)
        nn3_list = defaultdict(list)
        
        for v1, v2 in edges:
            nn_list[v1].append(v2)
            nn_list[v2].append(v1)
        for v1, v2 in edges_2nn:
            nn2_list[v1].append(v2)
            nn2_list[v2].append(v1)
        for v1, v2 in edges_3nn:
            nn3_list[v1].append(v2)
            nn3_list[v2].append(v1)
        
        # Get kagome sublattice for coloring
        kagome_sublattices = {}
        for v in vertices:
            kagome_sublattices[v] = get_kagome_sublattice_index(v, vertex_to_cell)
        
        sublattice_colors = ['#D55E00', '#009E73', '#56B4E9']
        bond_colors = {'NN': '#0072B2', '2NN': '#E69F00', '3NN': '#CC79A7'}
        
        # For PBC minimum image
        A1 = vertex_to_cell.get('_A1', np.array([1.5, -np.sqrt(3)/2]))
        A2 = vertex_to_cell.get('_A2', np.array([1.5, np.sqrt(3)/2]))
        L1, L2 = dim1 * A1, dim2 * A2
        
        def min_image_pos(pos_ref, pos_neighbor):
            """Get minimum image position of neighbor relative to reference"""
            if not use_pbc:
                return pos_neighbor
            best_pos = pos_neighbor
            best_dist = np.linalg.norm(pos_neighbor - pos_ref)
            for n1 in [-1, 0, 1]:
                for n2 in [-1, 0, 1]:
                    shifted = pos_neighbor + n1 * L1 + n2 * L2
                    dist = np.linalg.norm(shifted - pos_ref)
                    if dist < best_dist:
                        best_dist = dist
                        best_pos = shifted
            return best_pos
        
        # Plot connectivity for sites 0, 1, 2 (one per kagome sublattice)
        representative_sites = [0, 1, 2] if len(vertices) >= 3 else [0]
        
        fig, axes = plt.subplots(len(representative_sites), 3, figsize=(15, 5*len(representative_sites)))
        if len(representative_sites) == 1:
            axes = axes.reshape(1, -1)
        fig.suptitle(f'Kagome √3×√3 Lattice Connectivity ({dim1}x{dim2} {"PBC" if use_pbc else "OBC"})', 
                     fontsize=16, y=1.02)
        
        for row, center_site in enumerate(representative_sites):
            if center_site >= len(vertices):
                continue
            center_pos = np.array(vertices[center_site])
            center_sub = kagome_sublattices[center_site]
            
            for col, (neighbor_type, neighbor_list, bond_label) in enumerate([
                ('NN', nn_list, 'NN (d=0.5)'),
                ('2NN', nn2_list, '2NN (d=0.866)'),
                ('3NN', nn3_list, '3NN (d=1.0)')
            ]):
                ax = axes[row, col]
                
                # Plot all sites
                for v, pos in vertices.items():
                    pos = np.array(pos)
                    sub = kagome_sublattices[v]
                    is_neighbor = v in neighbor_list[center_site]
                    is_center = v == center_site
                    alpha = 1.0 if is_center or is_neighbor else 0.2
                    size = 150 if is_center else (100 if is_neighbor else 60)
                    ax.scatter(pos[0], pos[1], s=size, c=sublattice_colors[sub], 
                              alpha=alpha, edgecolors='black', linewidth=0.5, zorder=2)
                    if is_center or is_neighbor:
                        ax.annotate(str(v), pos, fontsize=8, ha='center', va='center', zorder=5)
                
                # Plot bonds to neighbors with minimum image
                neighbors = neighbor_list[center_site]
                for n in neighbors:
                    n_pos = min_image_pos(center_pos, np.array(vertices[n]))
                    ax.plot([center_pos[0], n_pos[0]], [center_pos[1], n_pos[1]],
                           color=bond_colors[neighbor_type], linewidth=2.5, alpha=0.8, zorder=1)
                    ax.scatter(n_pos[0], n_pos[1], s=200, c=sublattice_colors[kagome_sublattices[n]], 
                              marker='*', edgecolors='black', linewidth=1, zorder=4)
                
                # Highlight center site with red border
                ax.scatter(center_pos[0], center_pos[1], s=250, c=sublattice_colors[center_sub],
                          marker='o', edgecolors='red', linewidth=3, zorder=3)
                
                ax.set_title(f'Site {center_site} (Kagome sub {center_sub}): {bond_label} ({len(neighbors)} nbrs)', 
                            fontsize=10)
                ax.set_aspect('equal')
                ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/{cluster_name}_connectivity_by_site.png', dpi=150, bbox_inches='tight')
        print(f'Saved: {output_dir}/{cluster_name}_connectivity_by_site.png')
        plt.close(fig)
        
        # Plot 2: Overview of all bonds
        fig2, axes2 = plt.subplots(1, 3, figsize=(18, 6))
        fig2.suptitle(f'All Sites: NN, 2NN, 3NN Connectivity ({dim1}x{dim2} {"PBC" if use_pbc else "OBC"})', 
                      fontsize=14)
        
        for col, (neighbor_type, neighbor_list, edges_list, bond_label) in enumerate([
            ('NN', nn_list, edges, 'NN (d=0.5, coord=4)'),
            ('2NN', nn2_list, edges_2nn, '2NN (d=0.866, coord=4)'),
            ('3NN', nn3_list, edges_3nn, '3NN (d=1.0, coord=6)')
        ]):
            ax = axes2[col]
            # Count bond multiplicities to visualize with varying thickness
            from collections import Counter
            bond_counts = Counter(edges_list)
            for (v1, v2), count in bond_counts.items():
                p1, p2 = np.array(vertices[v1]), np.array(vertices[v2])
                # Make duplicate bonds thicker to show multiplicity
                linewidth = 1.5 + (count - 1) * 1.0
                alpha = min(0.6 + (count - 1) * 0.15, 0.9)
                ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 
                       color=bond_colors[neighbor_type], linewidth=linewidth, alpha=alpha)
                # Add label for duplicate bonds
                if count > 1:
                    mid = (p1 + p2) / 2
                    ax.annotate(f'×{count}', mid, fontsize=9, ha='center', va='center',
                               bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                               zorder=10)
            for v, pos in vertices.items():
                pos = np.array(pos)
                ax.scatter(pos[0], pos[1], s=100, c=sublattice_colors[kagome_sublattices[v]], 
                          edgecolors='black', linewidth=0.8)
                ax.annotate(str(v), pos, fontsize=7, ha='center', va='center', 
                           bbox=dict(boxstyle='round,pad=0.1', facecolor='white', alpha=0.7))
            counts = [len(neighbor_list[v]) for v in vertices]
            unique_bonds = len(bond_counts)
            total_bonds = len(edges_list)
            ax.set_title(f'{bond_label} ({min(counts)}-{max(counts)} per site)\n{unique_bonds} unique, {total_bonds} total', 
                        fontsize=11)
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/{cluster_name}_all_bonds_overview.png', dpi=150, bbox_inches='tight')
        print(f'Saved: {output_dir}/{cluster_name}_all_bonds_overview.png')
        plt.close(fig2)
        
        return True
        
    except ImportError:
        print("Warning: matplotlib not installed, skipping connectivity plots")
        return False


def plot_cluster(vertices, edges, edges_2nn, edges_3nn, output_dir, cluster_name, 
                 sublattice_indices=None, vertex_to_cell=None):
    """
    Plot the kagome √3×√3 lattice cluster showing sites and bonds
    """
    try:
        import matplotlib.pyplot as plt

        # Publication-quality params
        mpl.rcParams['font.family'] = 'sans-serif'
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

        # Create figure
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Get kagome sublattice for each vertex for coloring
        kagome_sublattices = {}
        for v in vertices:
            kagome_sublattices[v] = get_kagome_sublattice_index(v, vertex_to_cell) if vertex_to_cell else v % 3
        
        # Plot third nearest neighbor bonds (magenta/purple)
        for v1, v2 in edges_3nn:
            p1 = np.array(vertices[v1])
            p2 = np.array(vertices[v2])
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]],
                    color='#CC79A7', alpha=0.5, linewidth=1.2, zorder=1, linestyle=':')
        
        # Plot second nearest neighbor bonds (orange)
        for v1, v2 in edges_2nn:
            p1 = np.array(vertices[v1])
            p2 = np.array(vertices[v2])
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]],
                    color='#E69F00', alpha=0.6, linewidth=1.5, zorder=2, linestyle='--')
        
        # Plot nearest neighbor edges (dark blue - thickest)
        for v1, v2 in edges:
            p1 = np.array(vertices[v1])
            p2 = np.array(vertices[v2])
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]],
                    color='#0072B2', alpha=0.8, linewidth=2.5, zorder=3)
        
        # Plot vertices by kagome sublattice with distinct colors
        sublattice_colors = ['#D55E00', '#009E73', '#56B4E9']  # Red-orange, Green, Light blue
        for sub_idx in range(3):
            sub_ids = [v for v in vertices if kagome_sublattices.get(v, v % 3) == sub_idx]
            if not sub_ids:
                continue
            sub_positions = np.array([vertices[v] for v in sub_ids])
            ax.scatter(sub_positions[:, 0], sub_positions[:, 1],
                        s=120, c=sublattice_colors[sub_idx], marker='o', alpha=0.95,
                        edgecolors='black', linewidth=1.2,
                        label=f'Kagome sublattice {sub_idx}', zorder=4)
        
        # Add site labels
        for v, pos in vertices.items():
            ax.annotate(str(v), pos, fontsize=7, ha='center', va='center', zorder=5)
        
        ax.set_title('Kagome √3×√3 Supercell Lattice (BFG Model)', fontsize=14, pad=10)
        ax.set_xlabel('x', fontsize=12)
        ax.set_ylabel('y', fontsize=12)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3, linewidth=0.5)
        
        # Custom legend with distinct colors
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor=sublattice_colors[0], 
                   markersize=10, markeredgecolor='black', label='Kagome sublattice 0', linewidth=0),
            Line2D([0], [0], marker='o', color='w', markerfacecolor=sublattice_colors[1], 
                   markersize=10, markeredgecolor='black', label='Kagome sublattice 1', linewidth=0),
            Line2D([0], [0], marker='o', color='w', markerfacecolor=sublattice_colors[2], 
                   markersize=10, markeredgecolor='black', label='Kagome sublattice 2', linewidth=0),
            Line2D([0], [0], color='#0072B2', linewidth=2.5, label='NN bonds (Jpm + Jzz)', alpha=0.8),
            Line2D([0], [0], color='#E69F00', linewidth=1.5, linestyle='--', label='2NN bonds (Jzz_2nn)', alpha=0.6),
            Line2D([0], [0], color='#CC79A7', linewidth=1.2, linestyle=':', label='3NN bonds (Jzz_3nn)', alpha=0.5)
        ]
        
        leg = ax.legend(handles=legend_elements, loc='upper right', frameon=True,
                       fancybox=False, shadow=False, framealpha=0.9, 
                       edgecolor='black', borderpad=0.5, columnspacing=1.0,
                       handlelength=2.0, handletextpad=0.5)
        leg.get_frame().set_linewidth(0.5)
        
        plt.tight_layout(pad=0.5)
        
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)
        
        plt.savefig(f"{output_dir}/{cluster_name}_plot.png",
                    dpi=300, bbox_inches='tight', pad_inches=0.05)
        plt.savefig(f"{output_dir}/{cluster_name}_plot.pdf",
                    bbox_inches='tight', pad_inches=0.05)
        
        print(f"Cluster visualization saved to: {output_dir}/{cluster_name}_plot.png")
        plt.close(fig)
        return True

    except ImportError:
        print("Warning: matplotlib not installed, skipping cluster plot")
        return False


def main():
    """Main function to process command line arguments and run the program"""
    if len(sys.argv) < 10:
        print("Usage: python helper_kagome_bfg_sqrt3.py Jpm Jzz h fieldx fieldy fieldz output_dir dim1 dim2 pbc [Jzz_2nn] [Jzz_3nn] [theta]")
        print("\nBFG Model on Kagome Lattice (√3 × √3 supercell):")
        print("  H = sum_{<ij>_NN} [Jpm(S_i^+ S_j^- + S_i^- S_j^+) + Jzz S_i^z S_j^z]")
        print("      + sum_{<ij>_2NN} Jzz_2nn S_i^z S_j^z")
        print("      + sum_{<ij>_3NN} Jzz_3nn S_i^z S_j^z")
        print("      - h * (sin(theta) S^x + cos(theta) S^z)")
        print("\nThis version uses the √3 × √3 supercell with 9 sites per supercell.")
        print("The supercell is natural for √3 × √3 magnetic orderings.")
        print("\nParameters:")
        print("  Jpm: XY exchange for nearest neighbors")
        print("  Jzz: Ising exchange for nearest neighbors")
        print("  h: Magnetic field strength")
        print("  fieldx, fieldy, fieldz: Field direction (normalized automatically)")
        print("  output_dir: Directory for output files")
        print("  dim1, dim2: Lattice dimensions (number of √3×√3 supercells)")
        print("  pbc: Use periodic boundary conditions (0 or 1)")
        print("  Jzz_2nn: Ising exchange for second nearest neighbors (optional, default = Jzz)")
        print("  Jzz_3nn: Ising exchange for third nearest neighbors (optional, default = Jzz)")
        print("  theta: Field angle in units of pi (optional, default 0.0)")
        sys.exit(1)
    
    # Parse command line arguments
    Jpm = float(sys.argv[1])
    Jzz = float(sys.argv[2])
    h = float(sys.argv[3])
    field_dir = [float(sys.argv[4]), float(sys.argv[5]), float(sys.argv[6])]
    output_dir = sys.argv[7]
    dim1 = int(sys.argv[8])
    dim2 = int(sys.argv[9])
    use_pbc = bool(int(sys.argv[10]))
    Jzz_2nn = float(sys.argv[11]) if len(sys.argv) > 11 else Jzz  # Default to Jzz
    Jzz_3nn = float(sys.argv[12]) if len(sys.argv) > 12 else Jzz  # Default to Jzz
    theta = float(sys.argv[13]) if len(sys.argv) > 13 else 0.0  # Default theta=0.0 if not provided
    theta = theta * np.pi
    
    # Ensure output directory exists
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    
    # Create cluster name
    pbc_str = "pbc" if use_pbc else "obc"
    cluster_name = f"kagome_bfg_sqrt3_{dim1}x{dim2}_{pbc_str}"

    # Generate cluster
    vertices, edges, edges_2nn, edges_3nn, node_mapping, vertex_to_cell = generate_kagome_sqrt3_cluster(dim1, dim2, use_pbc)
    
    # Create nearest neighbor lists
    nn_list, nn_list_2nn, nn_list_3nn, positions, sublattice_indices = create_nn_lists(
        edges, edges_2nn, edges_3nn, node_mapping, vertices, vertex_to_cell)
    
    # Write nearest neighbor lists and site info
    write_cluster_nn_list(output_dir, cluster_name, nn_list, nn_list_2nn, nn_list_3nn, 
                          positions, sublattice_indices, node_mapping, vertex_to_cell)
    
    # Prepare Hamiltonian parameters
    prepare_hamiltonian_parameters(output_dir, nn_list, nn_list_2nn, nn_list_3nn, 
                                  positions, sublattice_indices, node_mapping, 
                                  Jpm, Jzz, Jzz_2nn, Jzz_3nn, h, theta, field_dir)

    # Plot cluster
    plot_cluster(vertices, edges, edges_2nn, edges_3nn, output_dir, cluster_name, 
                 sublattice_indices, vertex_to_cell)
    
    # Plot connectivity by site for visual verification
    plot_connectivity_by_site(vertices, edges, edges_2nn, edges_3nn, output_dir, cluster_name,
                              vertex_to_cell, dim1, dim2, use_pbc)
    
    print(f"\nGenerated kagome √3×√3 supercell lattice cluster with dimensions {dim1}x{dim2}")
    print(f"Number of sites: {len(vertices)}")
    print(f"Number of NN bonds: {len(edges)}")
    print(f"Number of 2NN bonds: {len(edges_2nn)}")
    print(f"Number of 3NN bonds: {len(edges_3nn)}")
    print(f"Sites per √3×√3 supercell: 9")
    print(f"\nBFG Model Parameters:")
    print(f"  Jpm (NN XY exchange): {Jpm}")
    print(f"  Jzz (NN Ising exchange): {Jzz}")
    print(f"  Jzz_2nn (2NN Ising exchange): {Jzz_2nn}")
    print(f"  Jzz_3nn (3NN Ising exchange): {Jzz_3nn}")
    print(f"  h (field strength): {h}")
    print(f"  theta (field angle): {theta/np.pi}π")
    print(f"\nOutput written to: {output_dir}")


if __name__ == "__main__":
    main()
