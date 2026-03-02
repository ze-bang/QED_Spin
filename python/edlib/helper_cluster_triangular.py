import numpy as np
import sys
import os
import re

"""
Helper functions for preparing Hamiltonian parameters for triangular lattice NLCE clusters.

Supports:
- J1-J2 Heisenberg model (nearest and next-nearest neighbor)
- XXZ model with anisotropy
- Magnetic field along arbitrary direction
- Kitaev-Heisenberg model on triangular lattice
- Anisotropic exchange model (YbMgGaO4-type) with bond-dependent phases:
  H = Σ_{⟨ij⟩} [J_zz S_i^z S_j^z + J_± (S_i^+ S_j^- + S_i^- S_j^+)
               + J_±± (γ_ij S_i^+ S_j^+ + γ_ij* S_i^- S_j^-)
               - i J_z±/2 ((γ_ij* S_i^+ - γ_ij S_i^-) S_j^z + S_i^z (γ_ij* S_j^+ - γ_ij S_j^-)]
  where γ_ij = 1, e^{i2π/3}, e^{-i2π/3} for bonds along a1, a2, a3 directions.
"""

def read_cluster_file(filepath):
    """
    Reads a cluster file and extracts vertices, edges, triangles, and node mapping.
    Compatible with triangular lattice cluster files.
    
    Args:
        filepath: Path to the cluster file
        
    Returns:
        vertices: Dictionary of {vertex_id: (x, y, z)} - z=0 for 2D
        edges: List of (vertex1, vertex2) tuples
        triangles: List of (v1, v2, v3) tuples
        node_mapping: Dictionary mapping original IDs to matrix indices
    """
    vertices = {}
    edges = []
    triangles = []
    node_mapping = {}
    
    section = None
    
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            
            # Skip empty lines and comments that don't indicate sections
            if not line or (line.startswith('#') and 'Vertices' not in line and 
                            'Edges' not in line and 'Triangles' not in line and 
                            'Node Mapping' not in line):
                continue
                
            if 'Vertices' in line:
                section = 'vertices'
                continue
            elif 'Edges' in line:
                section = 'edges'
                continue
            elif 'Triangles' in line:
                section = 'triangles'
                continue
            elif 'Node Mapping' in line:
                section = 'node_mapping'
                continue
            elif line.startswith('#'):
                continue  # Skip other comments
            
            if section == 'vertices':
                parts = line.split(',')
                if len(parts) >= 3:
                    vertex_id = int(parts[0].strip())
                    x = float(parts[1].strip())
                    y = float(parts[2].strip())
                    z = float(parts[3].strip()) if len(parts) >= 4 else 0.0
                    vertices[vertex_id] = (x, y, z)
            
            elif section == 'edges':
                parts = line.split(',')
                if len(parts) >= 2:
                    vertex1 = int(parts[0].strip())
                    vertex2 = int(parts[1].strip())
                    edges.append((vertex1, vertex2))
            
            elif section == 'triangles':
                parts = line.split(',')
                if len(parts) >= 3:
                    v1 = int(parts[0].strip())
                    v2 = int(parts[1].strip())
                    v3 = int(parts[2].strip())
                    triangles.append((v1, v2, v3))
            
            elif section == 'node_mapping':
                match = re.search(r'(\d+):\s*(\d+)', line)
                if match:
                    original_id = int(match.group(1))
                    matrix_index = int(match.group(2))
                    node_mapping[original_id] = matrix_index
    
    return vertices, edges, triangles, node_mapping


def get_bond_type(v1_pos, v2_pos):
    """
    Determine the bond type for triangular lattice.
    Bond types based on direction:
    - Type 0: along a1 direction (horizontal)
    - Type 1: along a2 direction (60 degrees)
    - Type 2: along a3 direction (120 degrees)
    
    This is useful for anisotropic models like Kitaev-Heisenberg.
    
    Args:
        v1_pos: Position (x, y, z) of first vertex
        v2_pos: Position (x, y, z) of second vertex
        
    Returns:
        Bond type (0, 1, or 2)
    """
    # Triangular lattice vectors
    a1 = np.array([1.0, 0.0])
    a2 = np.array([0.5, np.sqrt(3)/2])
    a3 = np.array([-0.5, np.sqrt(3)/2])
    
    # Compute bond vector (ignore z component)
    delta = np.array([v2_pos[0] - v1_pos[0], v2_pos[1] - v1_pos[1]])
    delta_norm = delta / (np.linalg.norm(delta) + 1e-10)
    
    # Check alignment with each direction (considering PBC wrapping)
    # Use dot product to determine closest direction
    dots = [
        abs(np.dot(delta_norm, a1 / np.linalg.norm(a1))),
        abs(np.dot(delta_norm, a2 / np.linalg.norm(a2))),
        abs(np.dot(delta_norm, a3 / np.linalg.norm(a3)))
    ]
    
    return np.argmax(dots)


def get_bond_phase(v1_pos, v2_pos):
    """
    Determine the bond-dependent phase factor γ_ij for the anisotropic exchange model.
    
    Phase factors follow the convention:
    - γ = 1 (φ=0) for bonds along δ₁ (a1 direction, horizontal)
    - γ = e^{-i2π/3} (φ=-2π/3) for bonds along δ₂ (a2 direction, 60°)  
    - γ = e^{i2π/3} (φ=2π/3) for bonds along δ₃ (a3 direction, 120°)
    
    This matches the convention: φ̃_α = {0, -2π/3, 2π/3}
    
    Args:
        v1_pos: Position (x, y, z) of first vertex
        v2_pos: Position (x, y, z) of second vertex
        
    Returns:
        gamma: Complex phase factor e^{iφ}
        phi: Phase angle in radians
    """
    bond_type = get_bond_type(v1_pos, v2_pos)
    
    # Phase angles for each bond direction
    # δ₁ (a1, horizontal): φ = 0
    # δ₂ (a2, 60°):        φ = -2π/3
    # δ₃ (a3, 120°):       φ = +2π/3
    phases = [0.0, -2.0 * np.pi / 3.0, 2.0 * np.pi / 3.0]
    
    phi = phases[bond_type]
    gamma = np.exp(1j * phi)
    
    return gamma, phi


def create_nn_lists(edges, node_mapping, vertices):
    """
    Create nearest neighbor lists from the edge information.
    
    Args:
        edges: List of (vertex1, vertex2) tuples
        node_mapping: Dictionary mapping original IDs to matrix indices
        vertices: Dictionary of {vertex_id: (x, y, z)}
        
    Returns:
        nn_list: Dictionary mapping each site to its nearest neighbors
        positions: Dictionary mapping each site to its position
    """
    nn_list = {}
    positions = {}
    
    # Initialize empty lists for all vertices
    for vertex_id in vertices:
        nn_list[vertex_id] = []
        positions[vertex_id] = vertices[vertex_id]
    
    # Fill nearest neighbor lists based on edges
    for v1, v2 in edges:
        nn_list[v1].append(v2)
        nn_list[v2].append(v1)
    
    return nn_list, positions


def compute_nnn_pairs(vertices, positions, nn_list):
    """
    Compute next-nearest neighbor pairs for J2 interactions on triangular lattice.
    NNN pairs are sites connected through two NN bonds.
    
    Args:
        vertices: Dictionary of {vertex_id: (x, y, z)}
        positions: Dictionary mapping each site to its position
        nn_list: Dictionary mapping each site to its nearest neighbors
        
    Returns:
        nnn_pairs: Set of (site1, site2) tuples (site1 < site2)
    """
    nnn_pairs = set()
    
    for site in vertices:
        # NNN are neighbors of neighbors (excluding self and direct neighbors)
        direct_neighbors = set(nn_list[site])
        
        for nn in direct_neighbors:
            for nnn in nn_list.get(nn, []):
                if nnn != site and nnn not in direct_neighbors:
                    pair = (min(site, nnn), max(site, nnn))
                    nnn_pairs.add(pair)
    
    return nnn_pairs


def write_cluster_nn_list(output_dir, cluster_name, nn_list, positions, node_mapping):
    """
    Write nearest neighbor list and positions to files.
    
    Args:
        output_dir: Directory to write output files
        cluster_name: Name of the cluster
        nn_list: Dictionary mapping each site to its nearest neighbors
        positions: Dictionary mapping each site to its position
        node_mapping: Dictionary mapping original IDs to matrix indices
    """
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    
    # Write nearest neighbor list
    with open(f"{output_dir}/{cluster_name}_nn_list.dat", 'w') as f:
        f.write("# Nearest neighbor list for triangular lattice cluster: " + cluster_name + "\n")
        f.write("# Format: site_id, number_of_neighbors, [neighbor_ids]\n\n")
        
        for site_id in sorted(nn_list.keys()):
            neighbors = nn_list[site_id]
            
            f.write(f"{site_id} {len(neighbors)}")
            for neighbor in sorted(neighbors):
                f.write(f" {neighbor}")
            f.write("\n")
    
    # Write site positions
    # For triangular lattice, all sites are equivalent (no sublattice)
    with open(f"{output_dir}/{cluster_name}_site_info.dat", 'w') as f:
        f.write("# Site information for triangular lattice cluster: " + cluster_name + "\n")
        f.write("# site_id, matrix_index, sublattice_index, x, y, z\n")
        f.write("# (sublattice_index is always 0 for triangular lattice)\n\n")
        
        for site_id in sorted(positions.keys()):
            pos = positions[site_id]
            matrix_index = node_mapping.get(site_id, site_id)
            sub_idx = 0  # Triangular lattice has no sublattice structure
            
            f.write(f"{site_id} {matrix_index} {sub_idx} {pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f}\n")


def prepare_hamiltonian_parameters(cluster_filepath, output_dir, J1, J2=0.0, Jz_ratio=1.0, 
                                    h=0.0, field_dir=(0, 0, 1), model='xxz_j1j2',
                                    Jzz=None, Jpm=None, Jpmpm=None, Jzpm=None,
                                    Gamma=None, Gamma_prime=None,
                                    g_ab=2.0, g_c=2.0):
    """
    Prepare Hamiltonian parameters for exact diagonalization of triangular lattice.
    
    Supports several models:
    - 'xxz_j1j2': XXZ J1-J2 model (Jz=J1, Jxy=J1*Jz_ratio; isotropic when Jz_ratio=1)
    - 'kitaev': JKΓΓ' model on triangular lattice with bond-dependent interactions
      H = Σ_{⟨ij⟩} [J S_i · S_j + K S^γ_i S^γ_j 
                   + Γ (S^α_i S^β_j + S^β_i S^α_j)
                   + Γ' (S^γ_i S^α_j + S^α_i S^γ_j + S^γ_i S^β_j + S^β_i S^γ_j)]
      where γ ∈ {x, y, z} labels the bond type and (α, β) are the other two.
      J = J1 (Heisenberg), K = J2 (Kitaev), Γ = Gamma, Γ' = Gamma_prime
    - 'anisotropic': Bond-dependent anisotropic exchange (YbMgGaO4-type):
      H = Σ_{⟨ij⟩} [J_zz S_i^z S_j^z + J_± (S_i^+ S_j^- + S_i^- S_j^+)
                   + J_±± (γ_ij S_i^+ S_j^+ + γ_ij* S_i^- S_j^-)
                   - i J_z±/2 ((γ_ij* S_i^+ - γ_ij S_i^-) S_j^z + h.c.)]
      where γ_ij = 1, e^{i2π/3}, e^{-i2π/3} for bonds along a1, a2, a3.
    
    Args:
        cluster_filepath: Path to the cluster file
        output_dir: Directory to write output files
        J1: Nearest-neighbor exchange coupling (or J_Heisenberg for kitaev model)
        J2: Next-nearest-neighbor exchange coupling (or J_Kitaev for kitaev model)
        Jz_ratio: Jxy/Jz coupling ratio for XXZ model (Jxy = J1 * Jz_ratio, Jz = J1)
        h: Magnetic field strength (in Tesla when using physical units)
        field_dir: Field direction (3-vector), default is (0, 0, 1) for out-of-plane
        model: Model type ('xxz_j1j2', 'kitaev', 'anisotropic')
        Jzz: S^z S^z coupling for anisotropic model
        Jpm: S^+ S^- + S^- S^+ coupling for anisotropic model (J_±)
        Jpmpm: S^+ S^+ + S^- S^- coupling for anisotropic model (J_±±)
        Jzpm: S^z (S^+ - S^-) coupling for anisotropic model (J_z±)
        Gamma: Γ off-diagonal symmetric exchange for kitaev model (default: 0)
        Gamma_prime: Γ' off-diagonal exchange for kitaev model (default: 0)
        g_ab: In-plane g-factor for anisotropic Zeeman (default: 2.0)
        g_c: Out-of-plane (c-axis) g-factor for anisotropic Zeeman (default: 2.0)
    """
    # Extract cluster name from filepath
    cluster_name = os.path.basename(cluster_filepath).split('.')[0]
    
    # Read cluster info
    vertices, edges, triangles, node_mapping = read_cluster_file(cluster_filepath)
    
    # If node_mapping is empty, create contiguous 0-based mapping
    if not node_mapping:
        sorted_vertices = sorted(vertices.keys())
        node_mapping = {v: i for i, v in enumerate(sorted_vertices)}
    
    # Create nearest neighbor lists
    nn_list, positions = create_nn_lists(edges, node_mapping, vertices)
    
    # Write NN list and site info
    write_cluster_nn_list(output_dir, cluster_name, nn_list, positions, node_mapping)
    
    # Normalize field direction
    field_dir = np.array(field_dir, dtype=float)
    if np.linalg.norm(field_dir) > 0:
        field_dir = field_dir / np.linalg.norm(field_dir)
    
    interALL = []
    transfer = []
    
    # Process nearest-neighbor interactions
    for site_id in sorted(nn_list.keys()):
        i = site_id
        
        # Zeeman term: H_Z = -μ_B Σ_i [g_ab (B_x S^x + B_y S^y) + g_c B_z S^z]
        # h already absorbs μ_B if in appropriate units;
        # g_ab, g_c provide the anisotropic g-tensor weighting
        hz = h * field_dir[2] * g_c   # out-of-plane component
        hx = h * field_dir[0] * g_ab  # in-plane x-component
        hy = h * field_dir[1] * g_ab  # in-plane y-component
        
        # Sz term: -h_z * Sz
        if abs(hz) > 1e-10:
            transfer.append([2, node_mapping[i], -hz, 0])
        
        # Sx term: -h_x * (S+ + S-)/2
        if abs(hx) > 1e-10:
            transfer.append([0, node_mapping[i], -0.5 * hx, 0])  # S+
            transfer.append([1, node_mapping[i], -0.5 * hx, 0])  # S-
        
        # Sy term: -h_y * (S+ - S-)/(2i) = h_y * i * (S+ - S-)/2
        if abs(hy) > 1e-10:
            transfer.append([0, node_mapping[i], 0, 0.5 * hy])   # S+ with imaginary coupling
            transfer.append([1, node_mapping[i], 0, -0.5 * hy])  # S- with imaginary coupling
        
        # Nearest-neighbor exchange interactions
        for neighbor_id in nn_list[site_id]:
            if site_id < neighbor_id:
                j = neighbor_id
                
                if model == 'xxz_j1j2':
                    # XXZ J1-J2 model: H = Jxy*(Sx_i Sx_j + Sy_i Sy_j) + Jz*Sz_i Sz_j
                    # where Jz = J1, Jxy = J1 * Jz_ratio  (Jz_ratio=1 → isotropic Heisenberg)
                    Jxy = J1 * Jz_ratio
                    interALL.append([2, node_mapping[i], 2, node_mapping[j], J1, 0])       # Sz-Sz = J1
                    interALL.append([0, node_mapping[i], 1, node_mapping[j], 0.5*Jxy, 0]) # S+-S- = Jxy/2
                    interALL.append([1, node_mapping[i], 0, node_mapping[j], 0.5*Jxy, 0]) # S--S+ = Jxy/2
                    
                elif model == 'kitaev':
                    # ============================================================
                    # JKΓΓ' model on triangular lattice
                    # ============================================================
                    # H = Σ_{⟨ij⟩} [J S_i · S_j + K S^γ_i S^γ_j
                    #              + Γ (S^α_i S^β_j + S^β_i S^α_j)
                    #              + Γ'(S^γ_i S^α_j + S^α_i S^γ_j
                    #                 + S^γ_i S^β_j + S^β_i S^γ_j)]
                    #
                    # where γ ∈ {x,y,z} labels the bond type, (α,β) the other two.
                    #
                    # ---- Spin operators in S+/S- basis ----
                    #   Sx = (S+ + S-)/2
                    #   Sy = (S+ - S-)/(2i) = -i(S+ - S-)/2
                    #   Sz = Sz
                    #
                    # ---- Bilinear products ----
                    #   SxSx = (1/4)(S+S+ + S+S- + S-S+ + S-S-)
                    #   SySy = (-1/4)(S+S+ - S+S- - S-S+ + S-S-)
                    #        = (1/4)(-S+S+ + S+S- + S-S+ - S-S-)
                    #   SxSy + SySx = (-i/2)(S+S+ - S-S-)
                    #   SxSz + SzSx = (1/2)(S+Sz + S-Sz + SzS+ + SzS-)
                    #   SySz + SzSy = (-i/2)(S+Sz - S-Sz + SzS+ - SzS-)
                    #
                    # ---- Collecting into [op_i, site_i, op_j, site_j, Re, Im] ----
                    # where op: 0=S+, 1=S-, 2=Sz
                    #
                    # For each bond type we write the full exchange matrix
                    #
                    #        ⎛ J_xx  J_xy  J_xz ⎞
                    #   J =  ⎜ J_yx  J_yy  J_yz ⎟
                    #        ⎝ J_zx  J_zy  J_zz ⎠
                    #
                    # and convert H_bond = Σ_{μν} J_{μν} S^μ_i S^ν_j to S+/S- form.
                    # The diagonal part (SxSx, SySy, SzSz) contributes:
                    #   S+S+: (J_xx - J_yy)/4
                    #   S+S-: (J_xx + J_yy)/4
                    #   S-S+: (J_xx + J_yy)/4
                    #   S-S-: (J_xx - J_yy)/4
                    #   SzSz: J_zz
                    #
                    # Off-diagonal symmetric (J_xy, J_xz, J_yz) contributes:
                    #   From J_xy(SxSy + SySx) = (-i J_xy/2)(S+S+ - S-S-)
                    #     → S+S+: -i J_xy/2     S-S-: +i J_xy/2
                    #
                    #   From J_xz(SxSz + SzSx) = (J_xz/2)(S+Sz + S-Sz + SzS+ + SzS-)
                    #     → S+Sz, S-Sz, SzS+, SzS-: all +J_xz/2 (real)
                    #
                    #   From J_yz(SySz + SzSy) = (-i J_yz/2)(S+Sz - S-Sz + SzS+ - SzS-)
                    #     → S+Sz: -i J_yz/2   S-Sz: +i J_yz/2
                    #       SzS+: -i J_yz/2   SzS-: +i J_yz/2
                    #
                    # Combining S+Sz = (J_xz - i J_yz)/2
                    #           S-Sz = (J_xz + i J_yz)/2
                    #           SzS+ = (J_xz - i J_yz)/2
                    #           SzS- = (J_xz + i J_yz)/2
                    #
                    # Full result per bond:
                    #   S+S+: (J_xx - J_yy)/4 - i J_xy/2
                    #   S+S-: (J_xx + J_yy)/4
                    #   S-S+: (J_xx + J_yy)/4
                    #   S-S-: (J_xx - J_yy)/4 + i J_xy/2
                    #   SzSz: J_zz
                    #   S+Sz: (J_xz - i J_yz)/2
                    #   S-Sz: (J_xz + i J_yz)/2
                    #   SzS+: (J_xz - i J_yz)/2
                    #   SzS-: (J_xz + i J_yz)/2
                    # ============================================================
                    
                    J_H = J1   # Heisenberg = J
                    J_K = J2   # Kitaev = K
                    _G  = Gamma if Gamma is not None else 0.0        # Γ
                    _Gp = Gamma_prime if Gamma_prime is not None else 0.0  # Γ'
                    
                    bond_type = get_bond_type(positions[i], positions[j])
                    mi = node_mapping[i]
                    mj = node_mapping[j]
                    
                    if bond_type == 0:
                        # x-bond (γ=x, α=y, β=z)
                        # J_xx = J+K,  J_yy = J,    J_zz = J
                        # J_yz = Γ,    J_xy = Γ',   J_xz = Γ'
                        #
                        # S+S+: ((J+K) - J)/4 - iΓ'/2 = K/4 - iΓ'/2
                        # S+S-: ((J+K) + J)/4 = (2J+K)/4
                        # S-S+: (2J+K)/4
                        # S-S-: K/4 + iΓ'/2
                        # SzSz: J
                        # S+Sz: (Γ' - iΓ)/2
                        # S-Sz: (Γ' + iΓ)/2
                        # SzS+: (Γ' - iΓ)/2
                        # SzS-: (Γ' + iΓ)/2
                        interALL.append([0, mi, 0, mj, J_K/4.0, -_Gp/2.0])               # S+S+
                        interALL.append([0, mi, 1, mj, (2*J_H + J_K)/4.0, 0])            # S+S-
                        interALL.append([1, mi, 0, mj, (2*J_H + J_K)/4.0, 0])            # S-S+
                        interALL.append([1, mi, 1, mj, J_K/4.0, _Gp/2.0])                # S-S-
                        interALL.append([2, mi, 2, mj, J_H, 0])                           # SzSz
                        interALL.append([0, mi, 2, mj, _Gp/2.0, -_G/2.0])                # S+Sz
                        interALL.append([1, mi, 2, mj, _Gp/2.0, _G/2.0])                 # S-Sz
                        interALL.append([2, mi, 0, mj, _Gp/2.0, -_G/2.0])                # SzS+
                        interALL.append([2, mi, 1, mj, _Gp/2.0, _G/2.0])                 # SzS-
                    elif bond_type == 1:
                        # y-bond (γ=y, α=x, β=z)
                        # J_xx = J,    J_yy = J+K,  J_zz = J
                        # J_xz = Γ,    J_xy = Γ',   J_yz = Γ'
                        #
                        # S+S+: (J - (J+K))/4 - iΓ'/2 = -K/4 - iΓ'/2
                        # S+S-: (J + (J+K))/4 = (2J+K)/4
                        # S-S+: (2J+K)/4
                        # S-S-: -K/4 + iΓ'/2
                        # SzSz: J
                        # S+Sz: (Γ - iΓ')/2
                        # S-Sz: (Γ + iΓ')/2
                        # SzS+: (Γ - iΓ')/2
                        # SzS-: (Γ + iΓ')/2
                        interALL.append([0, mi, 0, mj, -J_K/4.0, -_Gp/2.0])              # S+S+
                        interALL.append([0, mi, 1, mj, (2*J_H + J_K)/4.0, 0])            # S+S-
                        interALL.append([1, mi, 0, mj, (2*J_H + J_K)/4.0, 0])            # S-S+
                        interALL.append([1, mi, 1, mj, -J_K/4.0, _Gp/2.0])               # S-S-
                        interALL.append([2, mi, 2, mj, J_H, 0])                           # SzSz
                        interALL.append([0, mi, 2, mj, _G/2.0, -_Gp/2.0])                # S+Sz
                        interALL.append([1, mi, 2, mj, _G/2.0, _Gp/2.0])                 # S-Sz
                        interALL.append([2, mi, 0, mj, _G/2.0, -_Gp/2.0])                # SzS+
                        interALL.append([2, mi, 1, mj, _G/2.0, _Gp/2.0])                 # SzS-
                    else:
                        # z-bond (γ=z, α=x, β=y)
                        # J_xx = J,    J_yy = J,    J_zz = J+K
                        # J_xy = Γ,    J_xz = Γ',   J_yz = Γ'
                        #
                        # S+S+: (J - J)/4 - iΓ/2 = -iΓ/2
                        # S+S-: (J + J)/4 = J/2
                        # S-S+: J/2
                        # S-S-: +iΓ/2
                        # SzSz: J+K
                        # S+Sz: (Γ' - iΓ')/2
                        # S-Sz: (Γ' + iΓ')/2
                        # SzS+: (Γ' - iΓ')/2
                        # SzS-: (Γ' + iΓ')/2
                        interALL.append([0, mi, 0, mj, 0, -_G/2.0])                      # S+S+
                        interALL.append([0, mi, 1, mj, J_H/2.0, 0])                      # S+S-
                        interALL.append([1, mi, 0, mj, J_H/2.0, 0])                      # S-S+
                        interALL.append([1, mi, 1, mj, 0, _G/2.0])                       # S-S-
                        interALL.append([2, mi, 2, mj, J_H + J_K, 0])                    # SzSz
                        interALL.append([0, mi, 2, mj, _Gp/2.0, -_Gp/2.0])              # S+Sz
                        interALL.append([1, mi, 2, mj, _Gp/2.0, _Gp/2.0])               # S-Sz
                        interALL.append([2, mi, 0, mj, _Gp/2.0, -_Gp/2.0])              # SzS+
                        interALL.append([2, mi, 1, mj, _Gp/2.0, _Gp/2.0])               # SzS-
                
                elif model == 'anisotropic':
                    # Anisotropic exchange model (YbMgGaO4-type) with bond-dependent phases
                    # H = Σ_{⟨ij⟩} [J_zz S_i^z S_j^z 
                    #              + J_± (S_i^+ S_j^- + S_i^- S_j^+)
                    #              + J_±± (γ_ij S_i^+ S_j^+ + γ_ij* S_i^- S_j^-)
                    #              - i J_z±/2 ((γ_ij* S_i^+ - γ_ij S_i^-) S_j^z + h.c.)]
                    #
                    # where γ_ij = e^{iφ} with φ = 0, 2π/3, -2π/3 for a1, a2, a3 bonds
                    
                    # Use provided parameters or default to J1-based values
                    _Jzz = Jzz if Jzz is not None else J1
                    _Jpm = Jpm if Jpm is not None else 0.5 * J1
                    _Jpmpm = Jpmpm if Jpmpm is not None else 0.0
                    _Jzpm = Jzpm if Jzpm is not None else 0.0
                    
                    # Get bond phase factor
                    gamma, phi = get_bond_phase(positions[i], positions[j])
                    cos_phi = np.cos(phi)
                    sin_phi = np.sin(phi)
                    
                    # Term 1: J_zz S_i^z S_j^z
                    interALL.append([2, node_mapping[i], 2, node_mapping[j], _Jzz, 0])
                    
                    # Term 2: J_± (S_i^+ S_j^- + S_i^- S_j^+)
                    interALL.append([0, node_mapping[i], 1, node_mapping[j], _Jpm, 0])  # S+_i S-_j
                    interALL.append([1, node_mapping[i], 0, node_mapping[j], _Jpm, 0])  # S-_i S+_j
                    
                    # Term 3: J_±± (γ_ij S_i^+ S_j^+ + γ_ij* S_i^- S_j^-)
                    # γ = e^{iφ} = cos(φ) + i sin(φ)
                    # γ* = e^{-iφ} = cos(φ) - i sin(φ)
                    # γ S+S+ has real part cos(φ) and imaginary part sin(φ)
                    # γ* S-S- has real part cos(φ) and imaginary part -sin(φ)
                    interALL.append([0, node_mapping[i], 0, node_mapping[j], _Jpmpm * cos_phi, _Jpmpm * sin_phi])  # γ S+S+
                    interALL.append([1, node_mapping[i], 1, node_mapping[j], _Jpmpm * cos_phi, -_Jpmpm * sin_phi]) # γ* S-S-
                    
                    # Term 4: -i J_z±/2 [(γ* S_i^+ - γ S_i^-) S_j^z + S_i^z (γ* S_j^+ - γ S_j^-)]
                    # 
                    # Expand (γ* S_i^+ - γ S_i^-) S_j^z:
                    #   = (cos(φ) - i sin(φ)) S_i^+ S_j^z - (cos(φ) + i sin(φ)) S_i^- S_j^z
                    # 
                    # Multiply by -i/2:
                    #   -i/2 * [(cos(φ) - i sin(φ)) S+ Sz - (cos(φ) + i sin(φ)) S- Sz]
                    #   = -i/2 * [cos(φ) S+ Sz - i sin(φ) S+ Sz - cos(φ) S- Sz - i sin(φ) S- Sz]
                    #   = [-i cos(φ)/2 - sin(φ)/2] S+ Sz + [i cos(φ)/2 - sin(φ)/2] S- Sz
                    # 
                    # For S+ Sz term: real = -sin(φ)/2, imag = -cos(φ)/2
                    # For S- Sz term: real = -sin(φ)/2, imag = +cos(φ)/2
                    
                    _Jzpm_half = _Jzpm / 2.0
                    
                    # (γ* S_i^+ - γ S_i^-) S_j^z contribution
                    interALL.append([0, node_mapping[i], 2, node_mapping[j], -_Jzpm_half * sin_phi, -_Jzpm_half * cos_phi])  # S+_i Sz_j
                    interALL.append([1, node_mapping[i], 2, node_mapping[j], -_Jzpm_half * sin_phi, _Jzpm_half * cos_phi])   # S-_i Sz_j
                    
                    # S_i^z (γ* S_j^+ - γ S_j^-) contribution (same form by symmetry)
                    interALL.append([2, node_mapping[i], 0, node_mapping[j], -_Jzpm_half * sin_phi, -_Jzpm_half * cos_phi])  # Sz_i S+_j
                    interALL.append([2, node_mapping[i], 1, node_mapping[j], -_Jzpm_half * sin_phi, _Jzpm_half * cos_phi])   # Sz_i S-_j
    
    # Process next-nearest-neighbor interactions (J2)
    if abs(J2) > 1e-10 and model != 'kitaev':  # For Kitaev, J2 is repurposed
        nnn_pairs = compute_nnn_pairs(vertices, positions, nn_list)
        
        for i, j in nnn_pairs:
            if i in node_mapping and j in node_mapping:
                # Isotropic J2 Heisenberg coupling (J2 is always isotropic)
                interALL.append([2, node_mapping[i], 2, node_mapping[j], J2, 0])
                interALL.append([0, node_mapping[i], 1, node_mapping[j], 0.5*J2, 0])
                interALL.append([1, node_mapping[i], 0, node_mapping[j], 0.5*J2, 0])
    
    # Convert to arrays
    interALL = np.array(interALL) if interALL else np.zeros((0, 6))
    transfer = np.array(transfer) if transfer else np.zeros((0, 4))
    
    # Write interaction and transfer files
    write_interALL(output_dir, interALL, "InterAll.dat")
    write_transfer(output_dir, transfer, "Trans.dat")
    
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
    
    return nn_list, positions


def write_interALL(output_dir, interALL, file_name):
    """Write interaction parameters to a file (skipping zero-coupling terms)."""
    _s = lambda v: 0.0 if abs(v) < 1e-15 else float(v)
    nonzero = [i for i in range(len(interALL))
               if abs(interALL[i,4]) > 1e-15 or abs(interALL[i,5]) > 1e-15]
    num_param = len(nonzero)
    with open(f"{output_dir}/{file_name}", 'w') as f:
        f.write("===================\n")
        f.write(f"num {num_param:8d}\n")
        f.write("===================\n")
        f.write("===================\n")
        f.write("===================\n")
        
        for i in nonzero:
            f.write(f" {int(interALL[i,0]):8d} "
                   f" {int(interALL[i,1]):8d}   "
                   f" {int(interALL[i,2]):8d}   "
                   f" {int(interALL[i,3]):8d}   "
                   f" {_s(interALL[i,4]):8f}   "
                   f" {_s(interALL[i,5]):8f}   "
                   f"\n")


def write_transfer(output_dir, transfer, file_name):
    """Write transfer (field) parameters to a file (skipping zero-coupling terms)."""
    _s = lambda v: 0.0 if abs(v) < 1e-15 else float(v)
    nonzero = [i for i in range(len(transfer))
               if abs(transfer[i,2]) > 1e-15 or abs(transfer[i,3]) > 1e-15]
    num_param = len(nonzero)
    with open(f"{output_dir}/{file_name}", 'w') as f:
        f.write("===================\n")
        f.write(f"num {num_param:8d}\n")
        f.write("===================\n")
        f.write("===================\n")
        f.write("===================\n")
        
        for i in nonzero:
            f.write(f" {int(transfer[i,0]):8d} "
                   f" {int(transfer[i,1]):8d}   "
                   f" {_s(transfer[i,2]):8f}   "
                   f" {_s(transfer[i,3]):8f}"
                   f"\n")


def write_one_body_correlations(output_dir, Op, N, file_name):
    """Write one-body correlation parameters to a file."""
    with open(f"{output_dir}/{file_name}", 'w') as f:
        f.write("===================\n")
        f.write(f"loc {N:8d}\n")
        f.write("===================\n")
        f.write("===================\n")
        f.write("===================\n")
        
        for i in range(N):
            f.write(f" {Op:8d} "
                   f" {i:8d}   "
                   f" {1:8f}   "
                   f" {0:8f}"
                   f"\n")


def write_two_body_correlations(output_dir, Op1, Op2, N, file_name):
    """Write two-body correlation parameters to a file."""
    num_green_two = N * N
    with open(f"{output_dir}/{file_name}", 'w') as f:
        f.write("===================\n")
        f.write(f"loc {num_green_two:8d}\n")
        f.write("===================\n")
        f.write("===================\n")
        f.write("===================\n")
        
        for i in range(N):
            for j in range(N):
                f.write(f" {Op1:8d} "
                       f" {i:8d}   "
                       f" {Op2:8d}   "
                       f" {j:8d}   "
                       f" {1:8f}   "
                       f" {0:8f}   "
                       f"\n")


if __name__ == "__main__":
    # Command-line interface
    import argparse
    
    parser = argparse.ArgumentParser(description='Prepare Hamiltonian parameters for triangular lattice ED')
    parser.add_argument('--J1', type=float, default=1.0, help='Nearest-neighbor exchange')
    parser.add_argument('--J2', type=float, default=0.0, help='NNN exchange or Kitaev coupling')
    parser.add_argument('--h', type=float, default=0.0, help='Magnetic field strength')
    parser.add_argument('--field_dir', type=float, nargs=3, default=[0, 0, 1], help='Field direction (x, y, z)')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory')
    parser.add_argument('--cluster_file', type=str, required=True, help='Path to cluster file')
    parser.add_argument('--model', type=str, default='xxz_j1j2', 
                       choices=['xxz_j1j2', 'kitaev', 'anisotropic'],
                       help='Model type')
    parser.add_argument('--Jz_ratio', type=float, default=1.0, help='Jxy/Jz ratio for XXZ model (Jxy = Jz_ratio * J1, Jz = J1)')
    
    # Anisotropic model parameters
    parser.add_argument('--Jzz', type=float, default=None, help='J_zz for anisotropic model')
    parser.add_argument('--Jpm', type=float, default=None, help='J_± for anisotropic model')
    parser.add_argument('--Jpmpm', type=float, default=None, help='J_±± for anisotropic model')
    parser.add_argument('--Jzpm', type=float, default=None, help='J_z± for anisotropic model')
    
    # JKΓΓ' (Kitaev) model parameters
    parser.add_argument('--Gamma', type=float, default=None, help='Γ off-diagonal symmetric exchange for kitaev model')
    parser.add_argument('--Gamma_prime', type=float, default=None, help="Γ' off-diagonal exchange for kitaev model")
    
    # Anisotropic g-tensor
    parser.add_argument('--g_ab', type=float, default=2.0,
                       help='In-plane g-factor for Zeeman term (default: 2.0)')
    parser.add_argument('--g_c', type=float, default=2.0,
                       help='Out-of-plane (c-axis) g-factor for Zeeman term (default: 2.0)')
    
    args = parser.parse_args()
    
    field_dir = tuple(args.field_dir)
    
    prepare_hamiltonian_parameters(
        args.cluster_file, args.output_dir, args.J1, args.J2, 
        Jz_ratio=args.Jz_ratio, h=args.h, field_dir=field_dir, model=args.model,
        Jzz=args.Jzz, Jpm=args.Jpm, Jpmpm=args.Jpmpm, Jzpm=args.Jzpm,
        Gamma=args.Gamma, Gamma_prime=args.Gamma_prime,
        g_ab=args.g_ab, g_c=args.g_c
    )
