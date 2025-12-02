import numpy as np
import sys
import os
import argparse

# filepath: /home/pc_linux/exact_diagonalization_cpp/util/helper_honeycomb_c3.py
import matplotlib.pyplot as plt
from helper_honeycomb import *  # Import all functions from the honeycomb helper

def parse_c3_arguments():
    """Parse command line arguments for C3 symmetric lattice generation"""
    parser = argparse.ArgumentParser(description='Generate C3 symmetric honeycomb lattice')
    parser.add_argument('--size', type=int, default=4, help='Size parameter for the C3 symmetric lattice')
    parser.add_argument('--pbc', type=int, default=0, help='Use periodic boundary conditions (1) or open boundary conditions (0)')
    parser.add_argument('--J1', type=float, default=-6.54, help='J1 coupling constant')
    parser.add_argument('--delta1', type=float, default=0.36, help='Delta1 coupling constant')
    parser.add_argument('--Jpmpm', type=float, default=0.15, help='Jpmpm coupling constant')
    parser.add_argument('--Jzpm', type=float, default=-3.76, help='Jzpm coupling constant')
    parser.add_argument('--J2', type=float, default=-0.21, help='J2 coupling constant')
    parser.add_argument('--J3', type=float, default=1.7, help='J3 coupling constant')
    parser.add_argument('--delta3', type=float, default=0.03, help='Delta3 coupling constant')
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

def generate_c3_interactions(c3_sites, site_map, use_pbc, J_values):
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
        
        # Add next-next-nearest neighbor interactions
        for ni, nj, nu in nnnn_coords:
            if (ni, nj, nu) in site_map:
                neighbor_idx = site_map[(ni, nj, nu)]
                if site_idx < neighbor_idx:
                    term = J3NNNN(J_values, site_idx, neighbor_idx, "nnnn", len(c3_sites))
                    if term.size > 0:
                        interALL.append(term)
        
        # Add Zeeman terms
        h = J_values[-1]  # Assuming h is the last element in J_values
        zeeman_term = Zeeman(h, 1.0, 0.0, 0.0)  # Default to x-direction field
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

def read_param_file(filepath):
    """Reads parameters from a file."""
    params = {}
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#') or line.startswith('//'):
                continue
            key, value = line.split('=', 1)
            key = key.strip()
            value = value.strip()
            
            # Handle different value types
            if ',' in value:
                params[key] = [float(v.strip()) for v in value.split(',')]
            else:
                try:
                    params[key] = float(value)
                except ValueError:
                    params[key] = value
    return params

def main_c3():
    """Main function to generate C3 symmetric honeycomb lattice"""
    # Add a new argument for the parameter file
    parser = argparse.ArgumentParser(description='Generate C3 symmetric honeycomb lattice', add_help=False)
    parser.add_argument('--param_file', type=str, default=None, help='Path to parameter file')
    
    # Parse known and unknown args to avoid conflicts if param file is used
    args, unknown = parser.parse_known_args()

    # Parse the rest of the arguments
    base_parser = argparse.ArgumentParser()
    base_parser.add_argument('--size', type=int, default=4, help='Size parameter for the C3 symmetric lattice')
    base_parser.add_argument('--pbc', type=int, default=0, help='Use periodic boundary conditions (1) or open boundary conditions (0)')
    base_parser.add_argument('--J1', type=float, default=-6.54, help='J1 coupling constant')
    base_parser.add_argument('--delta1', type=float, default=0.36, help='Delta1 coupling constant')
    base_parser.add_argument('--Jpmpm', type=float, default=0.15, help='Jpmpm coupling constant')
    base_parser.add_argument('--Jzpm', type=float, default=-3.76, help='Jzpm coupling constant')
    base_parser.add_argument('--J2', type=float, default=-0.21, help='J2 coupling constant')
    base_parser.add_argument('--J3', type=float, default=1.7, help='J3 coupling constant')
    base_parser.add_argument('--delta3', type=float, default=0.03, help='Delta3 coupling constant')
    base_parser.add_argument('--h', type=float, default=0.0, help='Field strength')
    base_parser.add_argument('--hx', type=float, default=1.0, help='Field strength in x-direction')
    base_parser.add_argument('--hy', type=float, default=0.0, help='Field strength in y-direction')
    base_parser.add_argument('--hz', type=float, default=0.0, help='Field strength in z-direction')
    base_parser.add_argument('--outdir', type=str, default='output_c3', help='Output directory')
    
    # Parse the remaining arguments
    args = base_parser.parse_args(unknown, namespace=args)

    # If a parameter file is provided, read it and override the arguments
    if args.param_file:
        file_params = read_param_file(args.param_file)
        
        # Map file parameters to argument names
        param_map = {
            'h': 'h',
            'dir': 'outdir',
            'J1xy': 'J1',
            'J1z': 'delta1', # Assuming J1z maps to delta1 for Jz = J1 + delta1
            'J3xy': 'J3',
            'J3z': 'delta3', # Assuming J3z maps to delta3 for J3z = J3 + delta3
        }
        
        for key, value in file_params.items():
            if key in param_map:
                setattr(args, param_map[key], value)
        
        if 'field_dir' in file_params:
            args.hx, args.hy, args.hz = file_params['field_dir']
        
        # The provided file does not specify Jpmpm, Jzpm, J2.
        # They will keep their default or command-line values.

    # Extract parameters
    size = args.size
    use_pbc = args.pbc == 1
    J1, delta1, Jpmpm, Jzpm, J2, J3, delta3 = args.J1, args.delta1, args.Jpmpm, args.Jzpm, args.J2, args.J3, args.delta3
    J_values = [J1, delta1, Jpmpm, Jzpm, J2, J3, delta3]
    
    # Field parameters
    h = args.h
    hx, hy, hz = args.hx, args.hy, args.hz
    
    # Output directory
    output_dir = args.outdir + "/"
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    
    # Generate C3 symmetric sites
    c3_sites, center_pos = generate_c3_sites(size)
    
    # Reindex sites
    site_map, inverse_map = reindex_sites(c3_sites)
    
    # Generate interactions for the C3 lattice
    interALL, transfer = generate_c3_interactions(c3_sites, site_map, use_pbc, J_values + [h])
    
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



if __name__ == "__main__":
    main_c3()