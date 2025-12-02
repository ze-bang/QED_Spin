import numpy as np
import sys
import os
import argparse

# filepath: /home/pc_linux/exact_diagonalization_cpp/util/helper_honeycomb_c3.py
import matplotlib.pyplot as plt
from helper_honeycomb import *  # Import all functions from the honeycomb helper

def parse_config_file(config_file):
    """Parse configuration file and return parameter dictionary"""
    params = {}
    if not os.path.exists(config_file):
        print(f"Warning: Config file {config_file} not found")
        return params
        
    with open(config_file, 'r') as f:
        for line in f:
            line = line.strip()
            # Skip comments and empty lines
            if line.startswith('#') or not line:
                continue
            
            # Parse parameter = value format
            if '=' in line:
                key, value = line.split('=', 1)
                key = key.strip()
                value = value.strip()
                
                # Convert values to appropriate types
                if key in ['num_trials', 'pbc']:
                    params[key] = int(value)
                elif key in ['h', 'J1xy', 'J1z', 'D', 'E', 'F', 'G', 'J3xy', 'J3z']:
                    params[key] = float(value)
                elif key == 'field_dir':
                    # Parse comma-separated values for field direction
                    coords = [float(x.strip()) for x in value.split(',')]
                    if len(coords) == 3:
                        params['hx'], params['hy'], params['hz'] = coords
                elif key == 'size':
                    params[key] = int(value)
                else:
                    params[key] = value
    
    return params

def parse_c3_arguments():
    """Parse command line arguments for C3 symmetric lattice generation"""
    parser = argparse.ArgumentParser(description='Generate C3 symmetric honeycomb lattice')
    parser.add_argument('--config', type=str, help='Configuration file path')
    parser.add_argument('--size', type=int, default=4, help='Size parameter for the C3 symmetric lattice')
    parser.add_argument('--pbc', type=int, default=0, help='Use periodic boundary conditions (1) or open boundary conditions (0)')
    parser.add_argument('--J1xy', type=float, default=-6.54, help='J1 coupling constant')
    parser.add_argument('--J1z', type=float, default=0.36, help='Delta1 coupling constant')
    parser.add_argument('--J3xy', type=float, default=0.15, help='Jpmpm coupling constant')
    parser.add_argument('--J3z', type=float, default=-3.76, help='Jzpm coupling constant')
    parser.add_argument('--D', type=float, default=-0.21, help='D coupling constant')
    parser.add_argument('--E', type=float, default=1.7, help='E coupling constant')
    parser.add_argument('--F', type=float, default=1.7, help='F coupling constant')
    parser.add_argument('--G', type=float, default=1.7, help='G coupling constant')
    parser.add_argument('--h', type=float, default=0.0, help='Field strength')
    parser.add_argument('--hx', type=float, default=1.0, help='Field strength in x-direction')
    parser.add_argument('--hy', type=float, default=0.0, help='Field strength in y-direction')
    parser.add_argument('--hz', type=float, default=0.0, help='Field strength in z-direction')
    parser.add_argument('--outdir', type=str, default='output_c3', help='Output directory')
    
    return parser.parse_args()

def generate_c3_sites(size):
    """
    Generate sites for a C3 symmetric honeycomb lattice by applying 120° rotations.
    Only sites that are mapped to valid lattice sites under all rotations are kept.
    
    Args:
        size: Size parameter determining the extent of the lattice
        
    Returns:
        sites: List of (i, j, u) coordinates for each site in the C3 symmetric lattice
        center_pos: Position of the center site in Cartesian coordinates
    """
    # Create a large enough rhombus lattice
    dim1 = size
    dim2 = size
    
    # Calculate the center of the lattice (in unit cell coordinates)
    center_i = size//2
    center_j = size//2
    center_u = 0  # Use sublattice A as center
    
    # Calculate the center position in Cartesian coordinates
    # center_pos =  center_i * basis[0] + center_j * basis[1] + site_basis[center_u]
    center_pos = center_i * basis[0] + center_j * basis[1] + (site_basis[center_u] + site_basis[1 - center_u]) / 2 + np.array([0.5, 0.0])
    
    # Generate all possible sites in the large lattice
    all_sites = []
    for i in range(dim1):
        for j in range(dim2):
            for u in range(2):
                all_sites.append((i, j, u))
    
    # Create a mapping from position to (i,j,u) coordinates
    pos_to_iju = {}
    for i, j, u in all_sites:
        pos = tuple((i * basis[0] + j * basis[1] + site_basis[u]).round(6))
        pos_to_iju[pos] = (i, j, u)
    
    # Define rotation matrices for 120° and 240° rotations
    rotation_60 = np.array([[0.5, -np.sqrt(3)/2], [np.sqrt(3)/2, 0.5]])
    rotation_120 = rotation_60 @ rotation_60  # 120° rotation
    rotation_180 = np.array([[-1, 0], [0, -1]])  # 180° rotation
    rotation_240 = rotation_120 @ rotation_120  # 240° rotation
    rotation_300 = rotation_60.T  # 300° rotation (inverse of 60°)


    # Keep only sites that are C3 symmetric
    c3_sites = []
    for i, j, u in all_sites:
        site_pos = i * basis[0] + j * basis[1] + site_basis[u]
        rel_pos = site_pos - center_pos

        # Apply 60° rotation
        rot_pos_60 = np.dot(rotation_60, rel_pos) + center_pos
        rot_pos_60_tuple = tuple(rot_pos_60.round(6))

        # Apply 120° rotation
        rot_pos_120 = np.dot(rotation_120, rel_pos) + center_pos
        rot_pos_120_tuple = tuple(rot_pos_120.round(6))

        rot_pos_180 = np.dot(rotation_180, rel_pos) + center_pos
        rot_pos_180_tuple = tuple(rot_pos_180.round(6))

        # Apply 240° rotation
        rot_pos_240 = np.dot(rotation_240, rel_pos) + center_pos
        rot_pos_240_tuple = tuple(rot_pos_240.round(6))

        rot_pos_300 = np.dot(rotation_300, rel_pos) + center_pos
        rot_pos_300_tuple = tuple(rot_pos_300.round(6))

        # Check if both rotated positions map to valid lattice sites
        if (rot_pos_60_tuple in pos_to_iju and  
            rot_pos_120_tuple in pos_to_iju and
            rot_pos_180_tuple in pos_to_iju and
            rot_pos_240_tuple in pos_to_iju and
            rot_pos_300_tuple in pos_to_iju):
            c3_sites.append((i, j, u))

        # if (rot_pos_120_tuple in pos_to_iju and
        #     rot_pos_240_tuple in pos_to_iju):
        #     c3_sites.append((i, j, u))

    return c3_sites, center_pos

def reindex_sites(c3_sites):
    """
    Reindex the sites to have consecutive indices
    
    Args:
        c3_sites: List of (i, j, u) coordinates
        
    Returns:
        site_map: Dictionary mapping (i, j, u) to new index
        inverse_map: Dictionary mapping new index to (i, j, u)
    """
    site_map = {}
    inverse_map = {}
    
    for idx, (i, j, u) in enumerate(c3_sites):
        site_map[(i, j, u)] = idx
        inverse_map[idx] = (i, j, u)
    
    return site_map, inverse_map

def generate_c3_interactions(c3_sites, site_map, use_pbc, J_values, h, field_dir):
    """
    Generate interaction terms for the C3 symmetric lattice
    
    Args:
        c3_sites: List of (i, j, u) coordinates
        site_map: Mapping from (i, j, u) to new indices
        use_pbc: Whether to use periodic boundary conditions
        J_values: List of coupling constants
        
    Returns:
        interALL: Array of interaction terms
        transfer: Array of transfer terms
    """
    interALL = []
    transfer = []
    
    # Size of the lattice for PBC (not used with C3 symmetric lattice)
    dim1 = max([i for i, _, _ in c3_sites]) + 1
    dim2 = max([j for _, j, _ in c3_sites]) + 1
    
    # For each site in our C3 lattice
    for i, j, u in c3_sites:
        site_idx = site_map[(i, j, u)]
        
        # Get neighbor coordinates for NN, NNN, and NNNN
        if u == 0:  # Sublattice A
            nn_coords = [
                (i, j, 1),          # Z-bond
                (i, j-1, 1),        # X-bond
                (i+1, j-1, 1)       # Y-bond
            ]
            nnn_coords = [
                (i+1, j, u), (i, j+1, u), (i+1, j-1, u),
                (i-1, j, u), (i, j-1, u), (i-1, j+1, u)
            ]
            nnnn_coords = [
                (i+1, j, 1), (i-1, j, 1), (i+1, j-2, 1)
            ]
        else:  # Sublattice B (u == 1)
            nn_coords = [
                (i, j, 0),          # Z-bond
                (i, j+1, 0),        # X-bond
                (i-1, j+1, 0)       # Y-bond
            ]
            nnn_coords = [
                (i+1, j, u), (i, j+1, u), (i+1, j-1, u),
                (i-1, j, u), (i, j-1, u), (i-1, j+1, u)
            ]
            nnnn_coords = [
                (i-1, j, 0), (i+1, j, 0), (i-1, j+2, 0)
            ]
        
        # Add nearest neighbor interactions
        for idx, (ni, nj, nu) in enumerate(nn_coords):
            if (ni, nj, nu) in site_map:
                neighbor_idx = site_map[(ni, nj, nu)]
                bond_type = bond_types[idx]
                
                if site_idx < neighbor_idx:  # Only add each bond once
                    term = KitaevNN(J_values, site_idx, neighbor_idx, bond_type, len(c3_sites))
                    if term.size > 0:
                        interALL.append(term)
        
        # Add next-nearest neighbor interactions
        for ni, nj, nu in nnn_coords:
            if (ni, nj, nu) in site_map:
                neighbor_idx = site_map[(ni, nj, nu)]
                if site_idx < neighbor_idx:
                    term = J2NNN(J_values, site_idx, neighbor_idx, "nnn", len(c3_sites))
                    if term.size > 0:
                        interALL.append(term)
        
        # Add next-next-nearest neighbor interactions
        for ni, nj, nu in nnnn_coords:
            if (ni, nj, nu) in site_map:
                neighbor_idx = site_map[(ni, nj, nu)]
                if site_idx < neighbor_idx:
                    term = J3NNNN(J_values, site_idx, neighbor_idx, "nnnn", len(c3_sites))
                    if term.size > 0:
                        interALL.append(term)
        
        # Add Zeeman terms
        zeeman_term = Zeeman(h, *field_dir)  # Use field_dir for the magnetic field
        for term in zeeman_term:
            transfer.append(np.array([[term[0], site_idx, term[2], term[3]]]))
    
    interALL = np.vstack([arr for arr in interALL if arr.size > 0]) if interALL else np.array([])
    transfer = np.vstack([arr for arr in transfer if arr.size > 0]) if transfer else np.array([])
    
    return interALL, transfer

def plot_c3_lattice(c3_sites, site_map, inverse_map, center_pos, output_dir):
    """
    Plot the C3 symmetric lattice
    
    Args:
        c3_sites: List of (i, j, u) coordinates
        site_map: Mapping from (i, j, u) to new indices
        inverse_map: Mapping from new indices to (i, j, u)
        center_pos: Position of the center in Cartesian coordinates
        output_dir: Directory to save the plot
    """
    plt.figure(figsize=(10, 8))
    
    # Plot sites
    positions = []
    sublattice_indices = []
    
    for i, j, u in c3_sites:
        unit_cell_pos = i * basis[0] + j * basis[1]
        position = unit_cell_pos + site_basis[u]
        positions.append(position)
        sublattice_indices.append(u)
    
    # Convert to numpy arrays
    positions = np.array(positions)
    sublattice_indices = np.array(sublattice_indices)
    
    # Plot by sublattice
    colors = ['r', 'b']
    for u in range(2):
        mask = sublattice_indices == u
        plt.scatter(
            positions[mask, 0],
            positions[mask, 1],
            s=80,
            c=colors[u],
            marker='o',
            label=f'Sublattice {u}'
        )
    
    # Plot connections for nearest neighbors
    bond_colors = {'x': 'red', 'y': 'green', 'z': 'blue'}
    
    for idx, (i, j, u) in enumerate(c3_sites):
        site_idx = site_map[(i, j, u)]
        site_pos = i * basis[0] + j * basis[1] + site_basis[u]
        
        # Define neighbors based on sublattice
        if u == 0:  # Sublattice A
            neighbors = [
                ((i, j, 1), 'z'),
                ((i, j-1, 1), 'x'),
                ((i+1, j-1, 1), 'y')
            ]
        else:  # Sublattice B
            neighbors = [
                ((i, j, 0), 'z'),
                ((i, j+1, 0), 'x'),
                ((i-1, j+1, 0), 'y')
            ]
        
        # Draw bonds
        for neighbor_coords, bond_type in neighbors:
            if neighbor_coords in site_map:
                neighbor_idx = site_map[neighbor_coords]
                neighbor_i, neighbor_j, neighbor_u = neighbor_coords
                neighbor_pos = neighbor_i * basis[0] + neighbor_j * basis[1] + site_basis[neighbor_u]
                
                if site_idx < neighbor_idx:  # Only draw each bond once
                    plt.plot(
                        [site_pos[0], neighbor_pos[0]],
                        [site_pos[1], neighbor_pos[1]],
                        color=bond_colors[bond_type],
                        alpha=0.7,
                        linewidth=2,
                        label=f'{bond_type}-bond' if (idx == 0) else ""
                    )
    
    # Mark the center
    plt.plot(center_pos[0], center_pos[1], 'ko', markersize=12, label='Center')
    
    # Add the three C3 axes
    axis_length = max(np.max(positions[:, 0]) - np.min(positions[:, 0]),
                       np.max(positions[:, 1]) - np.min(positions[:, 1])) / 1.5
    
    angles = [0, 2*np.pi/3, 4*np.pi/3]
    for angle in angles:
        x = axis_length * np.cos(angle)
        y = axis_length * np.sin(angle)
        plt.arrow(center_pos[0], center_pos[1], x, y, head_width=0.2,
                  head_length=0.3, fc='k', ec='k', alpha=0.6)
    
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(f'C3 Symmetric Honeycomb Lattice ({len(c3_sites)} sites)')
    
    # Add a legend with unique entries
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    
    # Equal aspect ratio
    plt.axis('equal')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Save the figure
    plt.savefig(f"{output_dir}c3_honeycomb_lattice.png", dpi=300, bbox_inches='tight')
    plt.close()

def write_c3_site_positions(c3_sites, site_map, output_dir):
    """Write C3 symmetric site positions to a file"""
    with open(output_dir+"site_positions.dat", 'wt') as f:
        f.write("# index, x, y\n")
        
        for i, j, u in c3_sites:
            # Get the site index
            site_idx = site_map[(i, j, u)]
            
            # Calculate position in Cartesian coordinates
            unit_cell_pos = i * basis[0] + j * basis[1]
            position = unit_cell_pos + site_basis[u]
            
            # Write to file
            f.write(f"{site_idx} {position[0]:.6f} {position[1]:.6f}\n")

def main_c3():
    """Main function to generate C3 symmetric honeycomb lattice"""
    # Parse arguments
    args = parse_c3_arguments()
    
    # Parse config file if provided
    config_params = {}
    if args.config:
        config_params = parse_config_file(args.config)
    
    # Use config file parameters as defaults, override with command line arguments
    def get_param(name, default_value):
        if hasattr(args, name) and getattr(args, name) != default_value:
            return getattr(args, name)
        return config_params.get(name, default_value)
    
    # Extract parameters with config file support
    size = get_param('size', 4)
    use_pbc = get_param('pbc', 0) == 1
    
    # Coupling parameters
    J1xy = get_param('J1xy', -6.54)
    J1z = get_param('J1z', 0.36)
    J3xy = get_param('J3xy', 0.15)
    J3z = get_param('J3z', -3.76)
    D = get_param('D', -0.21)
    E = get_param('E', 1.7)
    F = get_param('F', 1.7)
    G = get_param('G', 1.7)

    J_values = [J1xy, J1z, J3xy, J3z, D, E, F, G]
    
    # Field parameters
    h = get_param('h', 0.0)
    hx = get_param('hx', 1.0)
    hy = get_param('hy', 0.0)
    hz = get_param('hz', 0.0)

    field_dir = [hx, hy, hz]
    
    # Output directory - handle config file 'dir' parameter
    output_dir = config_params.get('dir', args.outdir) + "/"
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    
    # Generate C3 symmetric sites
    c3_sites, center_pos = generate_c3_sites(size)
    
    # Reindex sites
    site_map, inverse_map = reindex_sites(c3_sites)
    
    # Generate interactions for the C3 lattice
    interALL, transfer = generate_c3_interactions(c3_sites, site_map, use_pbc, J_values, h, field_dir)
    
    # Write output files
    write_interALL(interALL, output_dir, "InterAll.dat")
    write_transfer(transfer, output_dir, "Trans.dat")
    write_c3_site_positions(c3_sites, site_map, output_dir)
    write_lattice_parameters(output_dir)
    
    # Plot the C3 symmetric lattice
    plot_c3_lattice(c3_sites, site_map, inverse_map, center_pos, output_dir)
    
    # Write one-body and two-body correlation functions
    one_body_correlations("one_body_correlations.dat", len(c3_sites), output_dir)
    two_body_correlations("two_body_correlations.dat", len(c3_sites), output_dir)
    
    # Write spin operators for different k-points
    spin_operators(0, [0, 0], "observables_S+_Gamma.dat", len(c3_sites), 1, 1, output_dir)
    spin_operators(1, [0, 0], "observables_S-_Gamma.dat", len(c3_sites), 1, 1, output_dir)
    spin_operators(2, [0, 0], "observables_Sz_Gamma.dat", len(c3_sites), 1, 1, output_dir)
    
    print(f"Generated C3 symmetric honeycomb lattice with {len(c3_sites)} sites")
    print(f"Output saved to {output_dir}")
    if args.config:
        print(f"Used config file: {args.config}")

if __name__ == "__main__":
    main_c3()