import numpy as np
from numba import jit, prange
import sys
from pathlib import Path

import scipy.sparse as sp

def read_eigenvector(filename):
    """Read eigenvector from file format: dimension, then index real imag lines"""
    data = []
    indices = []
    
    with open(filename, 'r') as f:
        dim = int(f.readline().strip())
        
        for line in f:
            parts = line.strip().split()
            if len(parts) == 3:
                idx = int(parts[0])
                real_part = float(parts[1])
                imag_part = float(parts[2])
                indices.append(idx)
                data.append(complex(real_part, imag_part))
    
    # Create full eigenvector
    eigenvector = np.zeros(dim, dtype=np.complex128)
    eigenvector[indices] = data
    
    # Normalize
    norm = np.sqrt(np.sum(np.abs(eigenvector)**2))
    if norm > 0:
        eigenvector /= norm
    
    return eigenvector, dim

def read_site_positions(filename):
    """Read site positions from file"""
    positions = []
    with open(filename, 'r') as f:
        for line in f:
            if line.startswith('#'):
                continue
            parts = line.strip().split()
            if len(parts) == 3:
                idx = int(parts[0])
                x = float(parts[1])
                y = float(parts[2])
                positions.append((idx, x, y))
    
    # Sort by index and extract x, y arrays
    positions.sort(key=lambda p: p[0])
    x_coords = np.array([p[1] for p in positions])
    y_coords = np.array([p[2] for p in positions])
    
    return x_coords, y_coords

@jit(nopython=True)
def get_bit(state, site):
    """Get bit value (0 or 1) at site position"""
    return (state >> site) & 1

@jit(nopython=True, parallel=True)
def compute_sz_expectation(eigenvector, n_sites):
    """Compute <Sz_i> for each site using basis states"""
    dim = len(eigenvector)
    sz_exp = np.zeros(n_sites)
    
    for site in prange(n_sites):
        exp_val = 0.0
        for state in range(dim):
            if np.abs(eigenvector[state]) > 1e-15:  # Skip negligible components
                # Sz|state> = (+1/2 for spin up, -1/2 for spin down)
                spin = 0.5 if get_bit(state, site) == 1 else -0.5
                exp_val += spin * np.abs(eigenvector[state])**2
        sz_exp[site] = exp_val.real
    
    return sz_exp

def create_sx_operator_sparse(site, n_sites):
    """Create sparse Sx operator for a single site"""
    dim = 2**n_sites
    rows = []
    cols = []
    data = []
    
    for state in range(dim):
        # Sx flips the spin at site
        flipped_state = state ^ (1 << site)
        rows.append(flipped_state)
        cols.append(state)
        data.append(0.5)  # Sx = 1/2 * (S+ + S-)
    
    return sp.csr_matrix((data, (rows, cols)), shape=(dim, dim), dtype=np.float64)

def create_sy_operator_sparse(site, n_sites):
    """Create sparse Sy operator for a single site"""
    dim = 2**n_sites
    rows = []
    cols = []
    data = []
    
    for state in range(dim):
        bit = get_bit(state, site)
        flipped_state = state ^ (1 << site)
        rows.append(flipped_state)
        cols.append(state)
        # Sy = -i/2 * (S+ - S-), sign depends on original spin
        data.append(0.5j if bit == 0 else -0.5j)
    
    return sp.csr_matrix((data, (rows, cols)), shape=(dim, dim), dtype=np.complex128)

def compute_sx_sy_expectations(eigenvector, n_sites, verbose=True):
    """Compute <Sx_i> and <Sy_i> using sparse matrices"""
    dim = len(eigenvector)
    sx_exp = np.zeros(n_sites)
    sy_exp = np.zeros(n_sites)
    
    for site in range(n_sites):
        if verbose:
            print(f"Computing Sx, Sy for site {site}/{n_sites}...", end='\r')
        
        # Create sparse operators
        sx_op = create_sx_operator_sparse(site, n_sites)
        sy_op = create_sy_operator_sparse(site, n_sites)
        
        # Compute expectations: <psi|O|psi>
        sx_exp[site] = np.real(np.conj(eigenvector) @ (sx_op @ eigenvector))
        sy_exp[site] = np.real(np.conj(eigenvector) @ (sy_op @ eigenvector))
    
    if verbose:
        print()
    
    return sx_exp, sy_exp

def find_most_probable_states(eigenvector, n_top=5):
    """Find the most probable basis state(s) in the superposition"""
    probabilities = np.abs(eigenvector)**2
    top_indices = np.argsort(probabilities)[-n_top:][::-1]
    
    return [(idx, probabilities[idx]) for idx in top_indices if probabilities[idx] > 1e-10]

def get_collapsed_spin_configuration(state, n_sites):
    """Get the spin configuration for a specific basis state after collapse"""
    sx = np.zeros(n_sites)  # Sx = 0 for definite Sz states
    sy = np.zeros(n_sites)  # Sy = 0 for definite Sz states
    sz = np.zeros(n_sites)
    
    for site in range(n_sites):
        # In the Sz basis, collapsed state has definite Sz values
        sz[site] = 0.5 if get_bit(state, site) == 1 else -0.5
    
    return sx, sy, sz

def state_to_string(state, n_sites):
    """Convert basis state to string representation"""
    config = []
    for site in range(n_sites):
        config.append('↑' if get_bit(state, site) == 1 else '↓')
    return ''.join(config)

def compute_spin_configuration(eigenvector_file, n_probable_states=5, verbose=True):
    """Main function to compute spin configuration"""
    
    # Read eigenvector
    if verbose:
        print(f"Reading eigenvector from {eigenvector_file}...")
    eigenvector, dim = read_eigenvector(eigenvector_file)
    
    # Determine number of sites
    n_sites = int(np.log2(dim))
    if 2**n_sites != dim:
        raise ValueError(f"Dimension {dim} is not a power of 2")
    
    if verbose:
        print(f"System size: {n_sites} sites, Hilbert space dimension: {dim}")
        print(f"Number of non-zero components: {np.sum(np.abs(eigenvector) > 1e-15)}")
    
    # Find most probable states
    print("\n" + "="*60)
    print(f"Top {n_probable_states} most probable basis states in superposition:")
    print("="*60)
    top_states = find_most_probable_states(eigenvector, n_top=n_probable_states)

    

    all_collapsed_configs = []
    for i, (state_idx, prob) in enumerate(top_states):
        config_str = state_to_string(state_idx, n_sites)
        print(f"{i+1}. State |{state_idx:d}⟩ = |{config_str}⟩")
        print(f"   Probability: {prob:.4f} ({prob*100:.2f}%)")
        print(f"   Amplitude: {eigenvector[state_idx]:.4f}")
        
        # Get collapsed configuration for this state
        sx_c, sy_c, sz_c = get_collapsed_spin_configuration(state_idx, n_sites)
        all_collapsed_configs.append((state_idx, prob, sx_c, sy_c, sz_c))
    
    # Compute quantum expectation values for comparison
    if verbose:
        print("\nComputing quantum expectation values <Sz>...")
    sz_exp = compute_sz_expectation(eigenvector, n_sites)
    
    if verbose:
        print("Computing quantum expectation values <Sx> and <Sy>...")
    sx_exp, sy_exp = compute_sx_sy_expectations(eigenvector, n_sites, verbose)
    
    return all_collapsed_configs, sx_exp, sy_exp, sz_exp, n_sites

def select_equal_probability_top_states(eigenvector, atol=1e-6, rtol=1e-3):
    """Select all basis states tied for the maximum probability within tolerance.

    Returns:
        indices (np.ndarray): indices of tied top-probability states
        max_p (float): the maximum probability value
    """
    probabilities = np.abs(eigenvector) ** 2
    if probabilities.size == 0:
        return np.array([], dtype=np.int64), 0.0
    max_p = probabilities.max()
    if max_p <= 0:
        return np.array([], dtype=np.int64), 0.0
    mask = np.isclose(probabilities, max_p, rtol=rtol, atol=atol)
    indices = np.flatnonzero(mask)
    return indices, float(max_p)

def build_equal_superposition_state(indices, dim):
    """Build a normalized equal-superposition state over given basis state indices."""
    psi = np.zeros(dim, dtype=np.complex128)
    if len(indices) == 0: 
        return psi
    amp = 1.0 / np.sqrt(len(indices))
    psi[indices] = amp
    return psi

def plot_expectations_per_site(sx, sy, sz, title="Expectation values", outfile="spin_expectations_equal_superposition.png"):
    """Plot <Sx>, <Sy>, <Sz> vs site and save to file."""
    try:
        import matplotlib.pyplot as plt
        sites = np.arange(len(sz))
        plt.figure(figsize=(12, 3.2))
        plt.plot(sites, sx, 'r.-', label='<Sx>')
        plt.plot(sites, sy, 'g.-', label='<Sy>')
        plt.plot(sites, sz, 'b.-', label='<Sz>', linewidth=2)
        plt.axhline(0.0, color='k', alpha=0.3)
        plt.grid(True, alpha=0.3)
        plt.xlabel('Site')
        plt.ylabel('Expectation value')
        plt.title(title)
        plt.legend(loc='best')
        plt.tight_layout()
        plt.savefig(outfile, dpi=150, bbox_inches='tight')
        plt.show()
        print(f"Plot saved as {outfile}")
    except ImportError:
        print("Matplotlib not available, skipping plot")

def save_equal_superposition_expectations(sx, sy, sz, filename="spin_config_equal_superposition.dat"):
    """Save per-site expectations for the equal superposition state."""
    with open(filename, 'w') as f:
        f.write("# Site    <Sx>           <Sy>           <Sz>\n")
        for i in range(len(sz)):
            f.write(f"{i:5d} {sx[i]:12.8f} {sy[i]:12.8f} {sz[i]:12.8f}\n")
    print(f"Equal-superposition expectations saved to {filename}")

def save_spin_configuration(all_collapsed_configs, sx_exp, sy_exp, sz_exp, 
                          output_file="spin_config.dat"):
    """Save both collapsed and expectation value configurations"""
    n_sites = len(sx_exp)
    with open(output_file, 'w') as f:
        f.write(f"# Top {len(all_collapsed_configs)} most probable collapsed states\n")
        for i, (state_idx, prob, _, _, sz_collapsed) in enumerate(all_collapsed_configs):
            f.write(f"# State {i+1}: |{state_idx}⟩ (probability = {prob:.4f})\n")
        f.write(f"# Site   Sz(state1)   <Sx>         <Sy>         <Sz>\n")
        
        # Write the most probable state's Sz values along with expectation values
        _, _, _, _, sz_collapsed = all_collapsed_configs[0]
        for i in range(n_sites):
            f.write(f"{i:5d} {sz_collapsed[i]:12.8f} {sx_exp[i]:12.8f} {sy_exp[i]:12.8f} {sz_exp[i]:12.8f}\n")
    print(f"Spin configuration saved to {output_file}")

def plot_multiple_spin_configurations(all_collapsed_configs, sx_exp, sy_exp, sz_exp):
    """Plot multiple probable collapsed states and quantum expectation values"""
    try:
        import matplotlib.pyplot as plt
        
        n_states_to_plot = min(4, len(all_collapsed_configs))  # Plot up to 4 states
        n_sites = len(sx_exp)
        sites = np.arange(n_sites)
        
        fig, axes = plt.subplots(n_states_to_plot + 1, 1, figsize=(12, 3*(n_states_to_plot+1)))
        if n_states_to_plot == 1:
            axes = [axes]
        
        # Plot each probable collapsed state
        for idx in range(n_states_to_plot):
            state_idx, prob, _, _, sz_collapsed = all_collapsed_configs[idx]
            ax = axes[idx]
            
            ax.stem(sites, sz_collapsed, linefmt='b-', markerfmt='bo', basefmt=' ')
            ax.set_ylabel('Sz', fontsize=10)
            ax.set_ylim(-0.6, 0.6)
            ax.grid(True, alpha=0.3)
            ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
            
            # Add spin arrows on top
            for i, sz in enumerate(sz_collapsed):
                if sz > 0:
                    ax.annotate('↑', xy=(i, sz), xytext=(i, sz+0.05), 
                              ha='center', fontsize=10, color='red')
                else:
                    ax.annotate('↓', xy=(i, sz), xytext=(i, sz-0.05), 
                              ha='center', fontsize=10, color='blue')
            
            ax.set_title(f'State #{idx+1}: |{state_idx}⟩ (P = {prob:.3f})', fontsize=11)
        
        # Plot quantum expectation values
        ax_exp = axes[-1]
        ax_exp.plot(sites, sx_exp, 'r.-', label='<Sx>', alpha=0.7)
        ax_exp.plot(sites, sy_exp, 'g.-', label='<Sy>', alpha=0.7)
        ax_exp.plot(sites, sz_exp, 'b.-', label='<Sz>', linewidth=2)
        ax_exp.set_ylabel('Expectation value', fontsize=10)
        ax_exp.set_xlabel('Site', fontsize=10)
        ax_exp.grid(True, alpha=0.3)
        ax_exp.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        ax_exp.legend(loc='best')
        ax_exp.set_title('Quantum Expectation Values (Full Superposition)', fontsize=11)
        
        plt.suptitle(f'Top {n_states_to_plot} Most Probable States', fontsize=14, y=1.01)
        plt.tight_layout()
        plt.savefig('spin_configuration_multiple.png', dpi=150, bbox_inches='tight')
        plt.show()
        print("Plot saved as spin_configuration_multiple.png")
        
    except ImportError:
        print("Matplotlib not available, skipping plot")

def plot_multiple_real_space_configurations(all_collapsed_configs, sx_exp, sy_exp, sz_exp,
                                           positions_file="site_positions.dat"):
    """Plot multiple probable states in real space"""
    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Circle
        import matplotlib.cm as cm
        
        # Read site positions
        try:
            x_coords, y_coords = read_site_positions(positions_file)
        except FileNotFoundError:
            print(f"Warning: {positions_file} not found, skipping real space plot")
            return
        
        n_sites = len(sx_exp)
        if len(x_coords) != n_sites:
            print(f"Warning: Position file has {len(x_coords)} sites but we have {n_sites} spins")
            return
        
        n_states_to_plot = min(4, len(all_collapsed_configs))
        
        # Create figure with subplots
        fig = plt.figure(figsize=(16, 4*((n_states_to_plot+1)//2 + 1)))
        
        # Plot each probable state
        for plot_idx in range(n_states_to_plot):
            state_idx, prob, _, _, sz_collapsed = all_collapsed_configs[plot_idx]
            ax = fig.add_subplot((n_states_to_plot+1)//2 + 1, 2, plot_idx+1)
            
            # Plot lattice connections
            for i in range(n_sites):
                for j in range(i+1, n_sites):
                    dist = np.sqrt((x_coords[i]-x_coords[j])**2 + (y_coords[i]-y_coords[j])**2)
                    if dist < 0.65:
                        ax.plot([x_coords[i], x_coords[j]], [y_coords[i], y_coords[j]], 
                               'gray', alpha=0.3, linewidth=1, zorder=1)
            
            # Plot spins
            for i in range(n_sites):
                color = 'red' if sz_collapsed[i] > 0 else 'blue'
                marker = '↑' if sz_collapsed[i] > 0 else '↓'
                
                circle = Circle((x_coords[i], y_coords[i]), 0.15, 
                              facecolor='white', edgecolor='black', linewidth=1.5, zorder=2)
                ax.add_patch(circle)
                
                ax.text(x_coords[i], y_coords[i], marker, 
                       fontsize=16, ha='center', va='center', 
                       color=color, weight='bold', zorder=3)
            
            ax.set_aspect('equal')
            ax.set_xlabel('x', fontsize=10)
            ax.set_ylabel('y', fontsize=10)
            ax.set_title(f'State #{plot_idx+1}: |{state_idx}⟩ (P={prob:.3f})', fontsize=11)
            ax.grid(True, alpha=0.2)
        
        # Plot quantum expectation values in the last subplot
        ax_exp = fig.add_subplot((n_states_to_plot+1)//2 + 1, 2, n_states_to_plot+1)
        
        # Plot lattice connections
        for i in range(n_sites):
            for j in range(i+1, n_sites):
                dist = np.sqrt((x_coords[i]-x_coords[j])**2 + (y_coords[i]-y_coords[j])**2)
                if dist < 0.65:
                    ax_exp.plot([x_coords[i], x_coords[j]], [y_coords[i], y_coords[j]], 
                               'gray', alpha=0.3, linewidth=1, zorder=1)
        
        # Calculate spin magnitudes
        spin_magnitudes = np.sqrt(sx_exp**2 + sy_exp**2 + sz_exp**2)
        
        # Normalize for coloring
        norm = plt.Normalize(vmin=-0.5, vmax=0.5)
        cmap = cm.RdBu_r
        
        for i in range(n_sites):
            color = cmap(norm(sz_exp[i]))
            circle_size = 0.1 + 0.2 * (spin_magnitudes[i] / 0.5)
            circle = Circle((x_coords[i], y_coords[i]), circle_size,
                          facecolor=color, edgecolor='black', linewidth=1.5, 
                          alpha=0.8, zorder=2)
            ax_exp.add_patch(circle)
            
            if spin_magnitudes[i] > 0.05:
                arrow_scale = 0.4
                ax_exp.arrow(x_coords[i], y_coords[i],
                            sx_exp[i]*arrow_scale, sz_exp[i]*arrow_scale,
                            head_width=0.05, head_length=0.03,
                            fc='black', ec='black', alpha=0.7, zorder=3)
        
        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax_exp, label='<Sz>')
        
        ax_exp.set_aspect('equal')
        ax_exp.set_xlabel('x', fontsize=10)
        ax_exp.set_ylabel('y', fontsize=10)
        ax_exp.set_title('Quantum Expectation Values', fontsize=11)
        ax_exp.grid(True, alpha=0.2)
        
        plt.suptitle(f'Top {n_states_to_plot} Most Probable States in Real Space', fontsize=14, y=1.01)
        plt.tight_layout()
        plt.savefig('spin_configuration_real_space_multiple.png', dpi=150, bbox_inches='tight')
        plt.show()
        print("Real space plot saved as spin_configuration_real_space_multiple.png")
        
    except ImportError:
        print("Matplotlib not available, skipping real space plot")

if __name__ == "__main__":
    # Parse command line arguments
    if len(sys.argv) > 1:
        eigenvector_file = sys.argv[1]
    else:
        eigenvector_file = "eigenvector_block0_0.dat"

    if len(sys.argv) > 2:
        positions_file = sys.argv[2]
    else:
        positions_file = "site_positions.dat"
    
    if len(sys.argv) > 3:
        n_probable_states = int(sys.argv[3])
    else:
        n_probable_states = 5  # Default to plotting top 5 states
    
    # Check if file exists
    if not Path(eigenvector_file).exists():
        print(f"Error: File {eigenvector_file} not found")
        sys.exit(1)
    
    # Compute spin configuration
    all_collapsed_configs, sx_exp, sy_exp, sz_exp, n_sites = compute_spin_configuration(
        eigenvector_file, n_probable_states=n_probable_states)
    
    # Print summary
    print("\n" + "="*60)
    print("Summary:")
    print("="*60)
    
    total_prob = sum(prob for _, prob, _, _, _ in all_collapsed_configs)
    print(f"Total probability of top {len(all_collapsed_configs)} states: {total_prob:.4f} ({total_prob*100:.2f}%)")
    
    for i, (state_idx, prob, _, _, sz_collapsed) in enumerate(all_collapsed_configs[:3]):  # Show top 3
        print(f"\nState #{i+1}: |{state_idx}⟩ (P={prob:.3f}):")
        print(f"  Total Sz: {np.sum(sz_collapsed):.6f}")
        print(f"  Configuration: {state_to_string(state_idx, n_sites)}")
    
    print(f"\nQuantum Expectation Values (Full Superposition):")
    print(f"  Total <Sx>: {np.sum(sx_exp):.6f}")
    print(f"  Total <Sy>: {np.sum(sy_exp):.6f}")
    print(f"  Total <Sz>: {np.sum(sz_exp):.6f}")
    print(f"  Average |<S>|: {np.mean(np.sqrt(sx_exp**2 + sy_exp**2 + sz_exp**2)):.6f}")
    
    # Save results
    save_spin_configuration(all_collapsed_configs, sx_exp, sy_exp, sz_exp)
    
    # Plot multiple states
    plot_multiple_spin_configurations(all_collapsed_configs, sx_exp, sy_exp, sz_exp)
    
    # Plot real space configurations
    plot_multiple_real_space_configurations(all_collapsed_configs, sx_exp, sy_exp, sz_exp, positions_file)

    # Construct an equal superposition of the most-probable tied states and plot expectations
    eigenvector_full, dim = read_eigenvector(eigenvector_file)
    top_equal_indices, max_p = select_equal_probability_top_states(eigenvector_full)
    if len(top_equal_indices) > 0:
        print(f"\nEqual-superposition over {len(top_equal_indices)} tied top-probability states (p_max = {max_p:.6g}).")
        print(f"Indices: {', '.join(map(str, top_equal_indices.tolist()))}")
        psi_equal = build_equal_superposition_state(top_equal_indices, dim)
        # Expectations for the equal superposition
        sz_eq = compute_sz_expectation(psi_equal, n_sites)
        sx_eq, sy_eq = compute_sx_sy_expectations(psi_equal, n_sites, verbose=False)
        # Save and plot
        save_equal_superposition_expectations(sx_eq, sy_eq, sz_eq)
        plot_expectations_per_site(sx_eq, sy_eq, sz_eq,
                                   title=f"Expectation values (Equal superposition of {len(top_equal_indices)} top states)",
                                   outfile="spin_expectations_equal_superposition.png")
        print(f"Totals (equal superposition): <Sx>={np.sum(sx_eq):.6f}, <Sy>={np.sum(sy_eq):.6f}, <Sz>={np.sum(sz_eq):.6f}")
    else:
        print("\nNo non-zero amplitudes found to construct an equal superposition.")
