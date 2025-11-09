import numpy as np
import sys
import os
import re

def read_cluster_file(filepath):
    """
    Reads a cluster file and extracts vertices, edges, tetrahedra, and node mapping
    
    Args:
        filepath: Path to the cluster file
        
    Returns:
        vertices: Dictionary of {vertex_id: (x, y, z)}
        edges: List of (vertex1, vertex2) tuples
        tetrahedra: List of (v1, v2, v3, v4) tuples
        node_mapping: Dictionary mapping original IDs to matrix indices
    """
    vertices = {}
    edges = []
    tetrahedra = []
    node_mapping = {}
    
    section = None
    
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            
            # Skip empty lines and comments that don't indicate sections
            if not line or (line.startswith('#') and 'Vertices' not in line and 
                            'Edges' not in line and 'Tetrahedra' not in line and 
                            'Node Mapping' not in line):
                continue
                
            if 'Vertices' in line:
                section = 'vertices'
                continue
            elif 'Edges' in line:
                section = 'edges'
                continue
            elif 'Tetrahedra' in line:
                section = 'tetrahedra'
                continue
            elif 'Node Mapping' in line:
                section = 'node_mapping'
                continue
            
            if section == 'vertices':
                parts = line.split(',')
                if len(parts) >= 4:
                    vertex_id = int(parts[0].strip())
                    x = float(parts[1].strip())
                    y = float(parts[2].strip())
                    z = float(parts[3].strip())
                    vertices[vertex_id] = (x, y, z)
            
            elif section == 'edges':
                parts = line.split(',')
                if len(parts) >= 2:
                    vertex1 = int(parts[0].strip())
                    vertex2 = int(parts[1].strip())
                    edges.append((vertex1, vertex2))
            
            elif section == 'tetrahedra':
                parts = line.split(',')
                if len(parts) >= 4:
                    v1 = int(parts[0].strip())
                    v2 = int(parts[1].strip())
                    v3 = int(parts[2].strip())
                    v4 = int(parts[3].strip())
                    tetrahedra.append((v1, v2, v3, v4))
            
            elif section == 'node_mapping':
                match = re.search(r'(\d+):\s*(\d+)', line)
                if match:
                    original_id = int(match.group(1))
                    matrix_index = int(match.group(2))
                    node_mapping[original_id] = matrix_index
    
    return vertices, edges, tetrahedra, node_mapping

def get_sublattice_index(vertex_id):
    """Return the sublattice index (vertex_id mod 4)"""
    return vertex_id % 4

def create_nn_lists(edges, node_mapping, vertices):
    """
    Create nearest neighbor lists from the edge information
    
    Args:
        edges: List of (vertex1, vertex2) tuples
        node_mapping: Dictionary mapping original IDs to matrix indices
        vertices: Dictionary of {vertex_id: (x, y, z)}
        
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
        sublattice_indices[vertex_id] = get_sublattice_index(vertex_id)
    
    # Fill nearest neighbor lists based on edges
    for v1, v2 in edges:
        nn_list[v1].append(v2)
        nn_list[v2].append(v1)
    
    return nn_list, positions, sublattice_indices

def write_cluster_nn_list(output_dir, cluster_name, nn_list, positions, sublattice_indices, node_mapping):
    """
    Write nearest neighbor list, positions, and sublattice indices to a file
    
    Args:
        output_dir: Directory to write output files
        cluster_name: Name of the cluster
        nn_list: Dictionary mapping each site to its nearest neighbors
        positions: Dictionary mapping each site to its position
        sublattice_indices: Dictionary mapping each site to its sublattice index
        node_mapping: Dictionary mapping original IDs to matrix indices
    """
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    
    # Write nearest neighbor list
    with open(f"{output_dir}/{cluster_name}_nn_list.dat", 'w') as f:
        f.write("# Nearest neighbor list for cluster: " + cluster_name + "\n")
        f.write("# Format: site_id, number_of_neighbors, [neighbor_ids]\n\n")
        
        for site_id in sorted(nn_list.keys()):
            neighbors = nn_list[site_id]
            matrix_index = node_mapping.get(site_id, site_id)  # Use matrix index if available
            
            f.write(f"{site_id} {len(neighbors)}")
            for neighbor in sorted(neighbors):
                f.write(f" {neighbor}")
            f.write("\n")
    
    # Write site positions and sublattice indices
    with open(f"{output_dir}/{cluster_name}_site_info.dat", 'w') as f:
        f.write("# Site information for cluster: " + cluster_name + "\n")
        f.write("# site_id, matrix_index, sublattice_index, x, y, z\n\n")
        
        for site_id in sorted(positions.keys()):
            pos = positions[site_id]
            sub_idx = sublattice_indices[site_id]
            matrix_index = node_mapping.get(site_id, site_id)  # Use matrix index if available
            
            f.write(f"{site_id} {matrix_index} {sub_idx} {pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f}\n")

def prepare_hamiltonian_parameters(cluster_filepath, output_dir, Jxx, Jyy, Jzz, h, field_dir, random_field_width=0):
    """
    Prepare Hamiltonian parameters for exact diagonalization
    
    Args:
        cluster_filepath: Path to the cluster file
        output_dir: Directory to write output files
        Jxx, Jyy, Jzz: Exchange couplings
        h: Field strength
        field_dir: Field direction (3-vector)
    """
    # Extract cluster name from filepath
    cluster_name = os.path.basename(cluster_filepath).split('.')[0]
    
    # Read cluster info
    vertices, edges, tetrahedra, node_mapping = read_cluster_file(cluster_filepath)
    
    # Create nearest neighbor lists
    nn_list, positions, sublattice_indices = create_nn_lists(edges, node_mapping, vertices)
    
    # Write NN list and site info
    write_cluster_nn_list(output_dir, cluster_name, nn_list, positions, sublattice_indices, node_mapping)
    
    # Prepare Hamiltonian parameters

    Jpm = -(Jxx+Jyy)/4
    Jpmpm = (Jxx-Jyy)/4

    # Jpm = Jxx
    # Jpmpm = Jyy
    
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
    
    # Calculate field projection along local z-axes
    B = np.array([h * np.dot(field_dir, z_local[sublattice_indices[site_id]]) for site_id in sorted(sublattice_indices.keys())])
    
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
        local_field = h * 5.5 * np.dot(field_dir, z_local[site_id % 4])
        transfer.append([2, node_mapping[i], -local_field, 0])

        # Add random transverse field from a Lorentzian distribution
        # The transverse field is coupled to Sx = 0.5 * (S+ + S-)
        # We need a parameter for the width of the distribution, let's assume a default value for now.
        # For a real application, this should be an input parameter.
        transverse_field_width = float(random_field_width) # Example width (gamma) of the Lorentzian distribution

        if transverse_field_width > 0:
            # Sample from a folded Cauchy (Lorentzian) distribution
            # np.random.standard_cauchy() samples from a Cauchy distribution with location=0, scale=1
            # We scale it by the width and take the absolute value for a strictly positive field.
            random_h_x = np.abs(transverse_field_width * np.random.standard_cauchy())
        else:
            random_h_x = 0.0

        # Sx component is 0.5 * (S+ + S-), so we add terms for S+ (op 0) and S- (op 1)
        # The coupling is -h_x * Sx = -0.5*h_x*S+ - 0.5*h_x*S-
        transfer.append([0, node_mapping[i], -0.5 * random_h_x, 0]) # S+ term
        transfer.append([1, node_mapping[i], -0.5 * random_h_x, 0]) # S- term
        
        # Exchange interactions
        for neighbor_id in nn_list[site_id]:
            if site_id < neighbor_id:  # Only add each bond once
                j = neighbor_id
                
                # Heisenberg interactions
                interALL.append([2, node_mapping[i], 2, node_mapping[j], Jzz, 0])  # Sz-Sz
                interALL.append([0, node_mapping[i], 1, node_mapping[j], -Jpm, 0])   # S+-S-
                interALL.append([1, node_mapping[i], 0, node_mapping[j], -Jpm, 0])   # S--S+
                Jpmpm_ = Jpmpm * non_kramer_factor[i % 4, j % 4]
                interALL.append([0, node_mapping[i], 0, node_mapping[j], np.real(Jpmpm_), -np.imag(Jpmpm_)])  # S--S-
                interALL.append([1, node_mapping[i], 1, node_mapping[j], np.real(Jpmpm_), np.imag(Jpmpm_)])  # S+-S+
    
    # Convert to arrays
    interALL = np.array(interALL)
    transfer = np.array(transfer)
    
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
    
    return nn_list, positions, sublattice_indices

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

def plot_cluster(vertices, edges, output_dir, cluster_name, sublattice_indices=None):
    """
    Plot the cluster showing sites and their nearest neighbor connections
    
    Args:
        vertices: Dictionary of {vertex_id: (x, y, z)}
        edges: List of (vertex1, vertex2) tuples
        output_dir: Directory to save the plot
        cluster_name: Name of the cluster
        sublattice_indices: Dictionary mapping each site to its sublattice index
    """
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        
        # Create 3D plot
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Colors for each sublattice
        colors = ['r', 'g', 'b', 'purple']
        
        # Plot vertices
        for vertex_id, pos in vertices.items():
            if sublattice_indices is None:
                sub_idx = get_sublattice_index(vertex_id)
            else:
                sub_idx = sublattice_indices[vertex_id]
            
            ax.scatter(pos[0], pos[1], pos[2], s=80, c=colors[sub_idx], marker='o')
            
            # Add vertex label
            ax.text(pos[0], pos[1], pos[2], str(vertex_id), fontsize=9)
        
        # Plot edges
        for v1, v2 in edges:
            pos1 = vertices[v1]
            pos2 = vertices[v2]
            
            ax.plot([pos1[0], pos2[0]], [pos1[1], pos2[1]], [pos1[2], pos2[2]], 'k-', alpha=0.5)
        
        # Set labels and title
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'Cluster: {cluster_name}')
        
        # Equal aspect ratio
        ax.set_box_aspect([1, 1, 1])
        
        # Save the figure
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)
        plt.savefig(f"{output_dir}/{cluster_name}_plot.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        return True
    except ImportError:
        print("Warning: matplotlib not installed, skipping cluster plot")
        return False

def main():
    """Main function to process command line arguments and run the program"""
    if len(sys.argv) < 10:
        print("Usage: python helper_cluster.py Jxx Jyy Jzz h fieldx fieldy fieldz output_dir cluster_file")
        sys.exit(1)
    
    # Parse command line arguments
    Jxx = float(sys.argv[1])
    Jyy = float(sys.argv[2])
    Jzz = float(sys.argv[3])
    h = float(sys.argv[4])
    field_dir = [float(sys.argv[5]), float(sys.argv[6]), float(sys.argv[7])]
    output_dir = sys.argv[8]
    cluster_filepath = sys.argv[9]
    transverse_field_width = sys.argv[10] if len(sys.argv) > 10 else 0  # Default width for random field
    
    # Ensure output directory exists
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    
    # Extract cluster name
    cluster_name = os.path.basename(cluster_filepath).split('.')[0]
    
    # Process cluster
    vertices, edges, tetrahedra, node_mapping = read_cluster_file(cluster_filepath)
    nn_list, positions, sublattice_indices = create_nn_lists(edges, node_mapping, vertices)
    
    # Prepare Hamiltonian
    prepare_hamiltonian_parameters(cluster_filepath, output_dir, Jxx, Jyy, Jzz, h, field_dir, transverse_field_width)

    # Plot cluster
    plot_cluster(vertices, edges, output_dir, cluster_name, sublattice_indices)
    
    print(f"Processed cluster {cluster_name}")
    print(f"Number of sites: {len(vertices)}")
    print(f"Number of bonds: {len(edges)}")
    print(f"Output written to: {output_dir}")

if __name__ == "__main__":
    main()