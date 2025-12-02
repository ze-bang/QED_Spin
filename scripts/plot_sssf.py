import numpy as np
from matplotlib import cm
import os
from opt_einsum import contract
#!/usr/bin/env python3
import matplotlib.pyplot as plt
def honeycomb_reciprocal_basis():
    """
    Calculate reciprocal lattice vectors for a honeycomb lattice.
    
    Real space basis vectors:
    a1 = (1, 0, 0)
    a2 = (1/2, sqrt(3)/2, 0)
    
    Returns:
        numpy.ndarray: Reciprocal lattice vectors b1 and b2
    """
    # a1 = np.array([0, 1, 0])
    # a2 = np.array([np.sqrt(3)/2, 1/2, 0])
    # a3 = np.array([0, 0, 1])  # Third basis vector (perpendicular to plane)
    
    a1 = np.array([1, 0, 0])
    a2 = np.array([1/2, np.sqrt(3)/2, 0])
    a3 = np.array([0, 0, 1])  # Third basis vector (perpendicular to plane)
    


    # Calculate reciprocal lattice vectors using the formula:
    # b_i = 2π * (a_j × a_k) / (a_i · (a_j × a_k))
    # where i,j,k are cyclic
    
    # Calculate cross products
    a2_cross_a3 = np.cross(a2, a3)
    a3_cross_a1 = np.cross(a3, a1)
    a1_cross_a2 = np.cross(a1, a2)
    
    # Calculate dot products for normalization
    vol = np.dot(a1, np.cross(a2, a3))
    
    # Calculate reciprocal lattice vectors
    b1 = 2 * np.pi * a2_cross_a3 / vol
    b2 = 2 * np.pi * a3_cross_a1 / vol
    b3 = 2 * np.pi * a1_cross_a2 / vol
    
    # Return the in-plane reciprocal lattice vectors
    return np.array([b1, b2, b3])

KBasis = honeycomb_reciprocal_basis()

def read_site_positions(filename):
    """Read site positions from file"""
    positions = {}
    with open(filename, 'r') as f:
        for line in f:
            if line.strip().startswith('#') or not line.strip():
                continue
            parts = line.strip().split()
            if len(parts) >= 3:
                site_idx = int(parts[0])
                x = float(parts[1])
                y = float(parts[2])
                positions[site_idx] = (x, y)
    return positions

def read_spin_correlations(filename):
    """Read spin correlations from file"""
    correlations = []
    with open(filename, 'r') as f:
        for line in f:
            if line.strip().startswith('#') or not line.strip():
                continue
            parts = line.strip().split()
            if len(parts) >= 6:
                i = int(parts[0])
                j = int(parts[1])
                sz_sz_real = float(parts[2])
                sz_sz_imag = float(parts[3])
                sp_sm_real = float(parts[4])
                sp_sm_imag = float(parts[5])
                if i != j:  # Only consider non-diagonal elements
                    correlations.append((i, j, sz_sz_real, sz_sz_imag, sp_sm_real, sp_sm_imag))
    return correlations

def compute_structure_factor(q_values, positions, correlations):
    """Compute static spin structure factor for given q values"""
    num_sites = len(positions)
    s_q_sz_sz = np.zeros((len(q_values), len(q_values)), dtype=complex)
    s_q_sp_sm = np.zeros((len(q_values), len(q_values)), dtype=complex)
    
    for qi, qx in enumerate(q_values):
        for qj, qy in enumerate(q_values):
            q_vec = qx * KBasis[0] + qy * KBasis[1]
            q_vec = q_vec[:2]  # Only consider in-plane components
            for i, j, sz_sz_real, sz_sz_imag, sp_sm_real, sp_sm_imag in correlations:
                r_i = np.array(positions[i])
                r_j = np.array(positions[j])
                phase = np.exp(1j * np.dot(q_vec, r_i - r_j))
                s_q_sz_sz[qi, qj] += (sz_sz_real + 1j * sz_sz_imag) * phase
                s_q_sp_sm[qi, qj] += (sp_sm_real + 1j * sp_sm_imag) * phase
    
    # Normalize
    s_q_sz_sz /= num_sites
    s_q_sp_sm /= num_sites
    

    return s_q_sz_sz, s_q_sp_sm


def SSSF(dirname):
    # Paths to input files
    site_positions_file = os.path.join(dirname, "site_positions.dat")
    spin_correlations_file = os.path.join(dirname, "output/spin_expectations/spin_correlations_T0.dat")

    # Read data
    positions = read_site_positions(site_positions_file)
    correlations = read_spin_correlations(spin_correlations_file)
    
    # Define q grid (in Brillouin zone)
    q_min, q_max = 0, 1
    num_q_points = 100
    q_values = np.linspace(q_min, q_max, num_q_points)
    

    # Compute structure factors
    s_q_sz_sz, s_q_sp_sm = compute_structure_factor(q_values, positions, correlations)
    
    # Plot results
    plt.figure(figsize=(12, 5))
    
    # SzSz structure factor
    plt.subplot(121)
    plt.title('SzSz Structure Factor')
    im1 = plt.imshow(np.real(s_q_sz_sz), extent=[q_min, q_max, q_min, q_max], 
                     origin='lower', cmap=cm.viridis)
    plt.colorbar(im1, label='S(q) SzSz')
    plt.xlabel('b1')
    plt.ylabel('b2')
    
    # S+S- structure factor
    plt.subplot(122)
    plt.title('S+S- Structure Factor')
    im2 = plt.imshow(np.real(s_q_sp_sm), extent=[q_min, q_max, q_min, q_max], 
                     origin='lower', cmap=cm.viridis)
    plt.colorbar(im2, label='S(q) S+S-')
    plt.xlabel('b1')
    plt.ylabel('b2')
    
    plt.tight_layout()
    
    # Save the plot
    output_dir = dirname + "/output/plots"
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'static_spin_structure_factor.png'), dpi=300)
    plt.close()


def scan_directory(root_dir): 
    # Check if the directory exists
    if not os.path.isdir(root_dir):
        print(f"Error: {root_dir} is not a valid directory")
        return
        
    # Find all subdirectories
    subdirs = [os.path.join(root_dir, d) for d in os.listdir(root_dir) 
              if os.path.isdir(os.path.join(root_dir, d))]
        
    if not subdirs:
        print(f"No subdirectories found in {root_dir}")
        return
    
    Magnetization = []
    Field = []

    # Apply SSSF to each subdirectory
    for subdir in subdirs:
        print(f"Processing {subdir}...")
        try:
            SSSF(subdir)
            print(f"Successfully processed {subdir}")
        except Exception as e:
            print(f"Error processing {subdir}: {e}")
        # Extract field value from directory name
        field_value = os.path.basename(subdir).split('_')[-1]
        field_value = field_value.replace('h=', '')


        spin_expectation = os.path.join(subdir, "output/spin_expectations/spin_expectations_T0.dat")
        S = np.loadtxt(spin_expectation, comments='#')[:, 1:]
        S_mag = np.mean(S, axis=0)
        Magnetization.append(np.linalg.norm(S_mag))
        Field.append(float(field_value))

    #Sort the results by field value
    sorted_indices = np.argsort(Field)
    Magnetization = np.array(Magnetization)[sorted_indices]
    Field = np.array(Field)[sorted_indices]

    return Magnetization, Field

def main():
    # Get directory path from user
    root_dir = input("Enter the root directory path: ")
    
    root_dir_A = root_dir + "_A"
    root_dir_B = root_dir + "_B"

    mag_A, field_A = scan_directory(root_dir_A)
    mag_B, field_B = scan_directory(root_dir_B)


    plt.figure(figsize=(10, 6))
    plt.plot(field_A, mag_A, 'o-', color='red', markersize=5, label='field // A')
    plt.plot(field_B, mag_B, 'o-', color='blue', markersize=5, label='field // B')
    plt.xlabel('Field (K)')
    plt.ylabel('Magnetization')
    plt.legend()
    plt.grid()
    plt.savefig('Magnetization_vs_Field.png', dpi=300)
    plt.show()

if __name__ == "__main__":
    main()