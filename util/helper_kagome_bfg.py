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

def generate_kagome_cluster(dim1, dim2, use_pbc=False):
    """
    Generate a kagome lattice cluster
    
    The kagome lattice has 3 sites per unit cell arranged in a corner-sharing triangle pattern.
    Unit cell basis vectors:
    - a1 = (1, 0)
    - a2 = (0.5, sqrt(3)/2)
    
    Site positions within unit cell:
    - Site 0: (0, 0)
    - Site 1: (0.5, 0)
    - Site 2: (0.25, sqrt(3)/4)
    
    Args:
        dim1, dim2: Dimensions of the lattice (number of unit cells)
        use_pbc: Whether to use periodic boundary conditions
        
    Returns:
        vertices: Dictionary of {vertex_id: (x, y)}
        edges: List of (vertex1, vertex2) tuples for nearest neighbors
        edges_2nn: List of (vertex1, vertex2) tuples for second nearest neighbors
        edges_3nn: List of (vertex1, vertex2) tuples for third nearest neighbors
        node_mapping: Dictionary mapping original IDs to matrix indices
        vertex_to_cell: Map vertex_id to (i, j, site_idx)
    """
    # Unit cell lattice vectors
    a1 = np.array([1.0, 0.0])
    a2 = np.array([0.5, np.sqrt(3)/2])
    
    # Site positions within unit cell (relative to unit cell origin)
    site_offsets = np.array([
        [0.0, 0.0],                    # Site 0
        [0.5, 0.0],                    # Site 1
        [0.25, np.sqrt(3)/4]           # Site 2
    ])
    
    # Generate vertices
    vertices = {}
    vertex_id = 0
    vertex_to_cell = {}  # Map vertex_id to (i, j, site_idx)
    cell_to_vertex = {}  # Map (i, j, site_idx) to vertex_id
    
    for i in range(dim1):
        for j in range(dim2):
            # Unit cell position
            unit_cell_pos = i * a1 + j * a2
            
            # For each site in the unit cell
            for site_idx in range(3):
                position = unit_cell_pos + site_offsets[site_idx]
                vertices[vertex_id] = tuple(position)
                vertex_to_cell[vertex_id] = (i, j, site_idx)
                cell_to_vertex[(i, j, site_idx)] = vertex_id
                vertex_id += 1
    
    # Helper function to get vertex id with PBC
    def get_vertex_with_pbc(i, j, site_idx):
        if use_pbc:
            # Apply periodic boundary conditions
            i = i % dim1
            j = j % dim2
        else:
            # Check if out of bounds for open boundary conditions
            if i < 0 or i >= dim1 or j < 0 or j >= dim2:
                return None
        
        key = (i, j, site_idx)
        return cell_to_vertex.get(key, None)
    
    # Generate nearest neighbor edges
    edges = set()  # Use set to avoid duplicates
    
    # Nearest neighbor connectivity for kagome lattice
    # Each unit cell has internal bonds and bonds to neighboring cells
    # The kagome lattice has a triangular motif where each site has 4 NN
    for i in range(dim1):
        for j in range(dim2):
            v0 = cell_to_vertex[(i, j, 0)]
            v1 = cell_to_vertex[(i, j, 1)]
            v2 = cell_to_vertex[(i, j, 2)]
            
            # Internal bonds within unit cell (each unit cell forms a triangle)
            edges.add(tuple(sorted([v0, v1])))
            edges.add(tuple(sorted([v0, v2])))
            edges.add(tuple(sorted([v1, v2])))
            
            v1_im1j = get_vertex_with_pbc(i-1, j, 1)
            if v1_im1j is not None:
                edges.add(tuple(sorted([v0, v1_im1j])))
            
            v1_ijm1 = get_vertex_with_pbc(i, j-1, 2)
            if v1_ijm1 is not None:
                edges.add(tuple(sorted([v0, v1_ijm1])))
            
            v0_ijm1 = get_vertex_with_pbc(i+1, j, 0)
            if v0_ijm1 is not None:
                edges.add(tuple(sorted([v1, v0_ijm1])))

            v0_im1j = get_vertex_with_pbc(i+1, j-1, 2)
            if v0_im1j is not None:
                edges.add(tuple(sorted([v1, v0_im1j])))

            v2_ijm1 = get_vertex_with_pbc(i, j+1, 0)
            if v2_ijm1 is not None:
                edges.add(tuple(sorted([v2, v2_ijm1])))

            v2_im1j = get_vertex_with_pbc(i-1, j+1, 1)
            if v2_im1j is not None:
                edges.add(tuple(sorted([v2, v2_im1j])))

    # Convert edges set to list
    edges = list(edges)
    
    # Generate second nearest neighbor edges (explicit connectivity)
    # 2NN are hexagon edges: sites separated by one intermediate site
    edges_2nn = set()
    
    for i in range(dim1):
        for j in range(dim2):
            v0 = cell_to_vertex[(i, j, 0)]
            v1 = cell_to_vertex[(i, j, 1)]
            v2 = cell_to_vertex[(i, j, 2)]
            
            # Site 0 2NN connections
            # Connect to site 1 of same cell through site 2
            # Already connected via NN, so these are the "next" ones
            v1_im1j = get_vertex_with_pbc(i-1, j, 2)
            if v1_im1j is not None:
                edges_2nn.add(tuple(sorted([v0, v1_im1j])))
            
            v0_ijm1 = get_vertex_with_pbc(i, j-1, 1)
            if v0_ijm1 is not None:
                edges_2nn.add(tuple(sorted([v0, v0_ijm1])))
            
            v1_im1jm1 = get_vertex_with_pbc(i-1, j+1, 1)
            if v1_im1jm1 is not None:
                edges_2nn.add(tuple(sorted([v0, v1_im1jm1])))
            
            v0_ip1j = get_vertex_with_pbc(i+1, j-1, 2)
            if v0_ip1j is not None:
                edges_2nn.add(tuple(sorted([v0, v0_ip1j])))
            
            v1_ijm1 = get_vertex_with_pbc(i, j-1, 2)
            if v1_ijm1 is not None:
                edges_2nn.add(tuple(sorted([v1, v1_ijm1])))
            
            v0_ip1jm1 = get_vertex_with_pbc(i+1, j-1, 0)
            if v0_ip1jm1 is not None:
                edges_2nn.add(tuple(sorted([v1, v0_ip1jm1])))
            
            v1_ijm11 = get_vertex_with_pbc(i, j+1, 0)
            if v1_ijm11 is not None:
                edges_2nn.add(tuple(sorted([v1, v1_ijm11])))
            
            v0_ip1jm11 = get_vertex_with_pbc(i+1, j, 2)
            if v0_ip1jm11 is not None:
                edges_2nn.add(tuple(sorted([v1, v0_ip1jm11])))

            # Site 2 2NN connections
            v2_im1j = get_vertex_with_pbc(i-1, j, 1)
            if v2_im1j is not None:
                edges_2nn.add(tuple(sorted([v2, v2_im1j])))
            
            v2_ijp1 = get_vertex_with_pbc(i, j+1, 1)
            if v2_ijp1 is not None:
                edges_2nn.add(tuple(sorted([v2, v2_ijp1])))
            
            v2_im1jp1 = get_vertex_with_pbc(i-1, j+1, 0)
            if v2_im1jp1 is not None:
                edges_2nn.add(tuple(sorted([v2, v2_im1jp1])))

            v2_im1jp11 = get_vertex_with_pbc(i+1, j, 0)
            if v2_im1jp11 is not None:
                edges_2nn.add(tuple(sorted([v2, v2_im1jp11])))

    # Convert to list
    edges_2nn = list(edges_2nn)
    
    # Generate third nearest neighbor edges (explicit connectivity)
    # 3NN connect sites across the hexagons (longer diagonal connections)
    edges_3nn = set()
    
    for i in range(dim1):
        for j in range(dim2):
            v0 = cell_to_vertex[(i, j, 0)]
            v1 = cell_to_vertex[(i, j, 1)]
            v2 = cell_to_vertex[(i, j, 2)]
            
            # Site 0 3NN connections
            v0_ip1jm1 = get_vertex_with_pbc(i+1, j-1, 0)
            if v0_ip1jm1 is not None:
                edges_3nn.add(tuple(sorted([v0, v0_ip1jm1])))
            
            v0_im1jp1 = get_vertex_with_pbc(i-1, j+1, 0)
            if v0_im1jp1 is not None:
                edges_3nn.add(tuple(sorted([v0, v0_im1jp1])))
            
            # Site 1 3NN connections
            v1_ip1jm1 = get_vertex_with_pbc(i, j+1, 1)
            if v1_ip1jm1 is not None:
                edges_3nn.add(tuple(sorted([v1, v1_ip1jm1])))
            
            v1_im1jp1 = get_vertex_with_pbc(i, j-1, 1)
            if v1_im1jp1 is not None:
                edges_3nn.add(tuple(sorted([v1, v1_im1jp1])))
            
            
            # Site 2 3NN connections
            v2_ip1jm1 = get_vertex_with_pbc(i+1, j, 2)
            if v2_ip1jm1 is not None:
                edges_3nn.add(tuple(sorted([v2, v2_ip1jm1])))
            
            v2_im1jp1 = get_vertex_with_pbc(i-1, j, 2)
            if v2_im1jp1 is not None:
                edges_3nn.add(tuple(sorted([v2, v2_im1jp1])))
            
    
    # Convert to list
    edges_3nn = list(edges_3nn)
    
    # Create node mapping
    node_mapping = {i: i for i in range(len(vertices))}
    
    return vertices, edges, edges_2nn, edges_3nn, node_mapping, vertex_to_cell

def get_sublattice_index(vertex_id, vertex_to_cell):
    """
    Return the sublattice index (0, 1, or 2 for kagome)
    """
    if vertex_id in vertex_to_cell:
        _, _, site_idx = vertex_to_cell[vertex_id]
        return site_idx
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
                          positions, sublattice_indices, node_mapping):
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
        f.write("# site_id, matrix_index, sublattice_index, x, y\n\n")
        
        for site_id in sorted(positions.keys()):
            pos = positions[site_id]
            sub_idx = sublattice_indices[site_id]
            matrix_index = node_mapping.get(site_id, site_id)
            
            f.write(f"{site_id} {matrix_index} {sub_idx} {pos[0]:.6f} {pos[1]:.6f}\n")

    # Write lattice parameters
    with open(f"{output_dir}/{cluster_name}_lattice_parameters.dat", 'w') as f:
        f.write("# Kagome lattice parameters\n\n")
        
        # Unit cell vectors
        a1 = np.array([1.0, 0.0])
        a2 = np.array([0.5, np.sqrt(3)/2])
        
        # Site offsets
        site_offsets = np.array([
            [0.0, 0.0],
            [0.5, 0.0],
            [0.25, np.sqrt(3)/4]
        ])
        
        # Write unit cell vectors
        f.write("# Unit cell lattice vectors\n")
        f.write("# vector_index, x, y\n")
        f.write(f"0 {a1[0]:.6f} {a1[1]:.6f}\n")
        f.write(f"1 {a2[0]:.6f} {a2[1]:.6f}\n")
        
        f.write("\n")
        
        # Write site offsets
        f.write("# Site offsets within each unit cell\n")
        f.write("# site_index, x, y\n")
        for i, offset in enumerate(site_offsets):
            f.write(f"{i} {offset[0]:.6f} {offset[1]:.6f}\n")

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
        local_field_x = h * np.sin(theta)
        local_field_z = h * np.cos(theta)
        transfer.append([0, node_mapping[i], -local_field_x/2, 0])
        transfer.append([1, node_mapping[i], -local_field_x/2, 0])
        transfer.append([2, node_mapping[i], -local_field_z, 0])  # Sz term
        
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
                # interALL.append([0, node_mapping[i], 1, node_mapping[j], -Jpm, 0])    # S+-S-
                # interALL.append([1, node_mapping[i], 0, node_mapping[j], -Jpm, 0])    # S--S+
        
        # Third nearest neighbor interactions (only Jzz)
        for neighbor_id in nn_list_3nn[site_id]:
            if site_id < neighbor_id:  # Only add each bond once
                j = neighbor_id
                interALL.append([2, node_mapping[i], 2, node_mapping[j], Jzz_3nn, 0])  # Sz-Sz
                # interALL.append([0, node_mapping[i], 1, node_mapping[j], -Jpm, 0])    # S+-S-
                # interALL.append([1, node_mapping[i], 0, node_mapping[j], -Jpm, 0])    # S--S+
        
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

def plot_cluster(vertices, edges, edges_2nn, edges_3nn, output_dir, cluster_name, sublattice_indices=None):
    """
    Plot the kagome lattice cluster showing sites and bonds
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

        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Muted, colorblind-friendly sublattice palette
        muted_colors = ['#0072B2', '#009E73', '#E69F00']
        
        if sublattice_indices is None:
            sublattice_indices = {}
        
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
        
        # Plot vertices by sublattice with distinct colors
        sublattice_colors = ['#D55E00', '#009E73', '#56B4E9']  # Red-orange, Green, Light blue
        for sub_idx in range(3):
            sub_ids = [v for v in vertices if sublattice_indices.get(v, v % 3) == sub_idx]
            if not sub_ids:
                continue
            sub_positions = np.array([vertices[v] for v in sub_ids])
            ax.scatter(sub_positions[:, 0], sub_positions[:, 1],
                        s=120, c=sublattice_colors[sub_idx], marker='o', alpha=0.95,
                        edgecolors='black', linewidth=1.2,
                        label=f'Sublattice {sub_idx}', zorder=4)
        
        ax.set_title('Kagome Lattice (BFG Model)', fontsize=14, pad=10)
        ax.set_xlabel('x', fontsize=12)
        ax.set_ylabel('y', fontsize=12)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3, linewidth=0.5)
        
        # Custom legend with distinct colors
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor=sublattice_colors[0], 
                   markersize=10, markeredgecolor='black', label='Sublattice 0', linewidth=0),
            Line2D([0], [0], marker='o', color='w', markerfacecolor=sublattice_colors[1], 
                   markersize=10, markeredgecolor='black', label='Sublattice 1', linewidth=0),
            Line2D([0], [0], marker='o', color='w', markerfacecolor=sublattice_colors[2], 
                   markersize=10, markeredgecolor='black', label='Sublattice 2', linewidth=0),
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
        print("Usage: python helper_kagome_bfg.py Jpm Jzz h fieldx fieldy fieldz output_dir dim1 dim2 pbc [Jzz_2nn] [Jzz_3nn] [theta]")
        print("\nBFG Model on Kagome Lattice:")
        print("  H = sum_{<ij>_NN} [Jpm(S_i^+ S_j^- + S_i^- S_j^+) + Jzz S_i^z S_j^z]")
        print("      + sum_{<ij>_2NN} Jzz_2nn S_i^z S_j^z")
        print("      + sum_{<ij>_3NN} Jzz_3nn S_i^z S_j^z")
        print("      - h * (sin(theta) S^x + cos(theta) S^z)")
        print("\nParameters:")
        print("  Jpm: XY exchange for nearest neighbors")
        print("  Jzz: Ising exchange for nearest neighbors")
        print("  h: Magnetic field strength")
        print("  fieldx, fieldy, fieldz: Field direction (normalized automatically)")
        print("  output_dir: Directory for output files")
        print("  dim1, dim2: Lattice dimensions (unit cells)")
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
    cluster_name = f"kagome_bfg_{dim1}x{dim2}_{pbc_str}"

    # Generate cluster
    vertices, edges, edges_2nn, edges_3nn, node_mapping, vertex_to_cell = generate_kagome_cluster(dim1, dim2, use_pbc)
    
    # Create nearest neighbor lists
    nn_list, nn_list_2nn, nn_list_3nn, positions, sublattice_indices = create_nn_lists(
        edges, edges_2nn, edges_3nn, node_mapping, vertices, vertex_to_cell)
    
    # Write nearest neighbor lists and site info
    write_cluster_nn_list(output_dir, cluster_name, nn_list, nn_list_2nn, nn_list_3nn, 
                          positions, sublattice_indices, node_mapping)
    
    # Prepare Hamiltonian parameters
    prepare_hamiltonian_parameters(output_dir, nn_list, nn_list_2nn, nn_list_3nn, 
                                  positions, sublattice_indices, node_mapping, 
                                  Jpm, Jzz, Jzz_2nn, Jzz_3nn, h, theta, field_dir)

    # Plot cluster
    plot_cluster(vertices, edges, edges_2nn, edges_3nn, output_dir, cluster_name, sublattice_indices)
    
    print(f"\nGenerated kagome lattice cluster with dimensions {dim1}x{dim2}")
    print(f"Number of sites: {len(vertices)}")
    print(f"Number of NN bonds: {len(edges)}")
    print(f"Number of 2NN bonds: {len(edges_2nn)}")
    print(f"Number of 3NN bonds: {len(edges_3nn)}")
    print(f"Sites per unit cell: 3")
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
