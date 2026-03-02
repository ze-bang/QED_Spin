#!/usr/bin/env python3
"""
Helper functions for preparing Hamiltonian parameters for triangle-based NLCE clusters.

For triangle-based NLCE on the triangular lattice:
- Order 0: single site (1 site, 0 bonds)
- Order n (n>=1): n up-pointing triangles connected by shared vertices

This module converts cluster JSON data to Hamiltonian input format for ED.
"""

import numpy as np
import json
import os
import sys

# Lattice vectors for triangular lattice
A1 = np.array([1.0, 0.0])
A2 = np.array([0.5, np.sqrt(3)/2])


def load_cluster_json(filepath):
    """Load cluster data from JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def get_bond_direction_vector(direction_idx):
    """
    Get unit vector for bond direction.
    
    Direction 0: along a1 = (1, 0)
    Direction 1: along a2 = (0.5, sqrt(3)/2)  
    Direction 2: along a2 - a1 = (-0.5, sqrt(3)/2)
    """
    if direction_idx == 0:
        return np.array([1.0, 0.0])
    elif direction_idx == 1:
        return np.array([0.5, np.sqrt(3)/2])
    elif direction_idx == 2:
        return np.array([-0.5, np.sqrt(3)/2])
    else:
        raise ValueError(f"Unknown direction: {direction_idx}")


def get_anisotropic_phase(direction_idx):
    """
    Get the phase γ for anisotropic exchange based on bond direction.
    
    For the anisotropic exchange Hamiltonian (YbMgGaO4-type):
    γ_ij = e^{i φ̃_α} where:
      φ̃_0 = 0        (bonds along a1)
      φ̃_1 = -2π/3   (bonds along a2)  
      φ̃_2 = +2π/3   (bonds along a2 - a1)
    
    Returns complex phase e^{iφ}.
    """
    if direction_idx == 0:
        return 1.0 + 0.0j  # e^{i*0}
    elif direction_idx == 1:
        return np.exp(-2j * np.pi / 3)  # e^{-i*2π/3}
    elif direction_idx == 2:
        return np.exp(2j * np.pi / 3)   # e^{i*2π/3}
    else:
        raise ValueError(f"Unknown direction: {direction_idx}")


def prepare_xxz_j1j2(cluster_data, J1=1.0, J2=0.0, Jz_ratio=1.0, h=0.0, h_direction=(0, 0, 1),
                     g_ab=2.0, g_c=2.0):
    """
    Prepare XXZ J1-J2 model parameters (unified model).
    
    H = sum_{<ij>} [Jxy*(Sx_i Sx_j + Sy_i Sy_j) + J1*Sz_i Sz_j]
      + J2 * sum_{<<ij>>} S_i · S_j
      - μ_B sum_i [g_ab (B_x S^x + B_y S^y) + g_c B_z S^z]
    
    where Jxy = J1 * Jz_ratio.  Jz_ratio=1 gives isotropic Heisenberg.
    J2 is always isotropic Heisenberg.
    
    This subsumes both the pure Heisenberg (Jz_ratio=1) and XXZ cases.
    
    Args:
        cluster_data: Cluster JSON data
        J1: Nearest-neighbor exchange (default: 1.0)
        J2: Next-nearest neighbor exchange, isotropic (default: 0.0)
        Jz_ratio: Jxy/Jz ratio (default: 1.0 = isotropic Heisenberg)
                  Convention: Jz = J1, Jxy = J1 * Jz_ratio
        h: Magnetic field strength
        h_direction: Field direction (normalized internally)
        g_ab: In-plane g-factor (default: 2.0)
        g_c: Out-of-plane g-factor (default: 2.0)
        
    Returns:
        Dictionary with Hamiltonian parameters for ED
    """
    n_sites = cluster_data['n_sites']
    bonds = cluster_data['bonds']
    
    Jxy = J1 * Jz_ratio
    
    # NN bonds: XXZ with Jz=J1, Jxy=J1*Jz_ratio
    interactions = []
    for i, j in bonds:
        interactions.append({
            'site1': i,
            'site2': j,
            'Jxx': Jxy,
            'Jyy': Jxy,
            'Jzz': J1
        })
    
    # TODO: NNN (J2) bonds need second-neighbor bond list from cluster data.
    # When available, add isotropic J2 interactions here.
    
    # Normalize field direction
    h_dir = np.array(h_direction, dtype=float)
    h_dir = h_dir / np.linalg.norm(h_dir) if np.linalg.norm(h_dir) > 0 else np.array([0, 0, 1])
    
    # Build Zeeman single-site terms
    zeeman_terms = _build_zeeman_terms(n_sites, h, h_dir, g_ab, g_c)
    
    return {
        'n_sites': n_sites,
        'interactions': interactions,
        'zeeman_terms': zeeman_terms,
        'h': h,
        'h_direction': h_dir.tolist(),
        'model': 'xxz_j1j2',
        'J1': J1,
        'J2': J2,
        'Jz_ratio': Jz_ratio
    }


def prepare_anisotropic_exchange(cluster_data, Jzz=1.0, Jpm=0.0, Jpmpm=0.0, Jzpm=0.0, 
                                  h=0.0, h_direction=(0, 0, 1),
                                  g_ab=2.0, g_c=2.0):
    """
    Prepare anisotropic exchange model (YbMgGaO4-type) parameters.
    
    H = Σ_{⟨ij⟩_α} [J_zz S_i^z S_j^z 
                    + J_± (S_i^+ S_j^- + S_i^- S_j^+)
                    + J_±± (γ_α S_i^+ S_j^+ + γ_α* S_i^- S_j^-)
                    - i J_z±/2 ((γ_α* S_i^+ - γ_α S_i^-) S_j^z 
                               + S_i^z (γ_α* S_j^+ - γ_α S_j^-))]
      - μ_B Σ_i [g_ab (B_x S^x + B_y S^y) + g_c B_z S^z]
    
    where γ_α = e^{iφ̃_α} with phases:
      φ̃_0 = 0       for bonds along a1 direction
      φ̃_1 = -2π/3  for bonds along a2 direction  
      φ̃_2 = +2π/3  for bonds along a2-a1 direction
    
    This can be rewritten in terms of Jxx, Jyy, Jzz, Jxy, Jxz, Jyz:
    
    S^+ S^- + S^- S^+ = 2(S^x S^x + S^y S^y)
    S^+ S^+ + S^- S^- = 2(S^x S^x - S^y S^y)  [for real γ]
    
    For complex γ = e^{iφ}:
    γ S^+ S^+ + γ* S^- S^- = 2 cos(φ)(S^x S^x - S^y S^y) - 2 sin(φ)(S^x S^y + S^y S^x)
    
    Args:
        cluster_data: Cluster JSON data
        Jzz: S^z S^z coupling
        Jpm: S^+ S^- + h.c. coupling (= J_±)
        Jpmpm: S^+ S^+ + h.c. coupling with phase (= J_±±)
        Jzpm: Mixed S^z S^± coupling (= J_z±)
        h: Magnetic field strength
        h_direction: Field direction
        g_ab: In-plane g-factor (default: 2.0)
        g_c: Out-of-plane g-factor (default: 2.0)
        
    Returns:
        Dictionary with Hamiltonian parameters for ED
    """
    n_sites = cluster_data['n_sites']
    bonds = cluster_data['bonds']
    bond_directions = cluster_data.get('bond_directions', [0] * len(bonds))
    
    interactions = []
    for bond_idx, (i, j) in enumerate(bonds):
        direction = bond_directions[bond_idx]
        gamma = get_anisotropic_phase(direction)
        
        # Convert to Cartesian coupling matrix
        # J_± term: S^+ S^- + S^- S^+ = 2(Sx Sx + Sy Sy)
        # So Jxx = Jyy = 2 * Jpm
        Jxx = 2 * Jpm
        Jyy = 2 * Jpm
        
        # J_±± term with phase γ:
        # γ S^+ S^+ + γ* S^- S^- = 2 cos(φ)(Sx Sx - Sy Sy) - 2 sin(φ)(Sx Sy + Sy Sx)
        phi = np.angle(gamma)
        Jxx += 2 * Jpmpm * np.cos(phi)
        Jyy -= 2 * Jpmpm * np.cos(phi)
        Jxy = -2 * Jpmpm * np.sin(phi)  # This is Jxy + Jyx = 2*Jxy for symmetric
        
        # J_z± term: 
        # -i/2 * Jzpm * [(γ* S^+ - γ S^-) Sz + Sz (γ* S^+ - γ S^-)]
        # = -i/2 * Jzpm * [γ*(S^x + iS^y) - γ(S^x - iS^y)] * Sz * 2
        # = -i/2 * Jzpm * [(γ* - γ)S^x + i(γ* + γ)S^y] * Sz * 2
        # γ* - γ = -2i sin(φ), γ* + γ = 2 cos(φ)
        # = -i/2 * Jzpm * [-2i sin(φ) Sx + 2i cos(φ) Sy] Sz * 2
        # = Jzpm * [sin(φ) Sx Sz - cos(φ) Sy Sz] * 2
        Jxz = 2 * Jzpm * np.sin(phi)
        Jyz = -2 * Jzpm * np.cos(phi)
        
        interactions.append({
            'site1': i,
            'site2': j,
            'Jxx': Jxx,
            'Jyy': Jyy,
            'Jzz': Jzz,
            'Jxy': Jxy,
            'Jxz': Jxz,
            'Jyz': Jyz,
            'direction': direction,
            'gamma_re': gamma.real,
            'gamma_im': gamma.imag
        })
    
    # Normalize field direction
    h_dir = np.array(h_direction, dtype=float)
    h_dir = h_dir / np.linalg.norm(h_dir) if np.linalg.norm(h_dir) > 0 else np.array([0, 0, 1])
    
    # Build Zeeman single-site terms
    zeeman_terms = _build_zeeman_terms(n_sites, h, h_dir, g_ab, g_c)
    
    return {
        'n_sites': n_sites,
        'interactions': interactions,
        'zeeman_terms': zeeman_terms,
        'h': h,
        'h_direction': h_dir.tolist(),
        'model': 'anisotropic_exchange',
        'Jzz': Jzz,
        'Jpm': Jpm,
        'Jpmpm': Jpmpm,
        'Jzpm': Jzpm
    }


def _build_zeeman_terms(n_sites, h, h_dir, g_ab, g_c):
    """
    Build Zeeman single-site terms for the anisotropic g-tensor.
    
    H_Z = -μ_B Σ_i [g_ab (B_x S_i^x + B_y S_i^y) + g_c B_z S_i^z]
    
    Returns list of dicts with keys: site, Sx_coeff, Sy_coeff, Sz_coeff
    (all real; sign convention: H_Z = Σ_i [hx Sx + hy Sy + hz Sz])
    """
    if abs(h) < 1e-15:
        return []
    
    # Effective field components with g-tensor
    hx = -h * h_dir[0] * g_ab   # -μ_B g_ab B_x
    hy = -h * h_dir[1] * g_ab   # -μ_B g_ab B_y
    hz = -h * h_dir[2] * g_c    # -μ_B g_c  B_z
    
    terms = []
    for site in range(n_sites):
        terms.append({
            'site': site,
            'hx': hx,
            'hy': hy,
            'hz': hz
        })
    return terms


def write_ed_config(ham_params, output_path, cluster_data, 
                    method='FULL', compute_thermo=True,
                    temp_min=0.01, temp_max=10.0, temp_bins=100):
    """
    Write ED configuration file for a cluster.
    
    Args:
        ham_params: Hamiltonian parameters from prepare_* functions
        output_path: Path to write config file
        cluster_data: Original cluster JSON data
        method: Diagonalization method (FULL, LANCZOS, etc.)
        compute_thermo: Whether to compute thermodynamic properties
        temp_min, temp_max, temp_bins: Temperature range for thermodynamics
    """
    n_sites = ham_params['n_sites']
    
    # Build the config file content
    lines = [
        f"n_sites = {n_sites}",
        f"method = {method}",
        "",
        "# Thermodynamic settings"
    ]
    
    if compute_thermo:
        lines.extend([
            "thermo = true",
            f"temp_min = {temp_min}",
            f"temp_max = {temp_max}",
            f"temp_bins = {temp_bins}",
        ])
    else:
        lines.append("thermo = false")
    
    lines.append("")
    lines.append("# Interactions")
    
    for idx, inter in enumerate(ham_params['interactions']):
        site1 = inter['site1']
        site2 = inter['site2']
        
        # Write coupling matrix elements
        _s = lambda v: 0.0 if abs(v) < 1e-15 else float(v)
        Jxx = _s(inter.get('Jxx', 0.0))
        Jyy = _s(inter.get('Jyy', 0.0))
        Jzz = _s(inter.get('Jzz', 0.0))
        Jxy = _s(inter.get('Jxy', 0.0))
        Jxz = _s(inter.get('Jxz', 0.0))
        Jyz = _s(inter.get('Jyz', 0.0))
        
        # Skip interactions where all couplings are zero
        if Jxx == 0.0 and Jyy == 0.0 and Jzz == 0.0 and Jxy == 0.0 and Jxz == 0.0 and Jyz == 0.0:
            continue
        
        lines.append(f"interaction{idx} = {site1}, {site2}, {Jxx}, {Jyy}, {Jzz}, {Jxy}, {Jxz}, {Jyz}")
    
    # Magnetic field
    if ham_params.get('h', 0.0) != 0.0:
        h = ham_params['h']
        h_dir = ham_params.get('h_direction', [0, 0, 1])
        lines.append("")
        lines.append("# Magnetic field")
        lines.append(f"h = {h}")
        lines.append(f"h_direction = {h_dir[0]}, {h_dir[1]}, {h_dir[2]}")
    
    # Zeeman single-site terms (anisotropic g-tensor)
    zeeman_terms = ham_params.get('zeeman_terms', [])
    if zeeman_terms:
        lines.append("")
        lines.append("# Zeeman terms (anisotropic g-tensor)")
        lines.append(f"# H_Z = sum_i [hx*Sx + hy*Sy + hz*Sz]")
        for zt in zeeman_terms:
            site = zt['site']
            _s = lambda v: 0.0 if abs(v) < 1e-15 else float(v)
            hx = _s(zt['hx'])
            hy = _s(zt['hy'])
            hz = _s(zt['hz'])
            # Skip zeeman terms where all components are zero
            if hx == 0.0 and hy == 0.0 and hz == 0.0:
                continue
            lines.append(f"zeeman{site} = {site}, {hx}, {hy}, {hz}")
    
    # Write the file
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))
    
    return output_path


def prepare_cluster_for_ed(cluster_json_path, output_dir, model='xxz_j1j2', **model_params):
    """
    Prepare a cluster for ED calculation.
    
    Args:
        cluster_json_path: Path to cluster JSON file
        output_dir: Directory to write ED input files
        model: Model type ('xxz_j1j2', 'anisotropic')
        **model_params: Model-specific parameters
        
    Returns:
        Path to ED config file
    """
    cluster_data = load_cluster_json(cluster_json_path)
    
    _g_ab = model_params.get('g_ab', 2.0)
    _g_c = model_params.get('g_c', 2.0)
    
    if model == 'xxz_j1j2':
        ham_params = prepare_xxz_j1j2(cluster_data, 
                                       J1=model_params.get('J1', 1.0),
                                       J2=model_params.get('J2', 0.0),
                                       Jz_ratio=model_params.get('Jz_ratio', 1.0),
                                       h=model_params.get('h', 0.0),
                                       h_direction=model_params.get('h_direction', (0, 0, 1)),
                                       g_ab=_g_ab, g_c=_g_c)
    elif model == 'anisotropic':
        ham_params = prepare_anisotropic_exchange(cluster_data,
                                                   Jzz=model_params.get('Jzz', 1.0),
                                                   Jpm=model_params.get('Jpm', 0.0),
                                                   Jpmpm=model_params.get('Jpmpm', 0.0),
                                                   Jzpm=model_params.get('Jzpm', 0.0),
                                                   h=model_params.get('h', 0.0),
                                                   h_direction=model_params.get('h_direction', (0, 0, 1)),
                                                   g_ab=_g_ab, g_c=_g_c)
    else:
        raise ValueError(f"Unknown model: {model}")
    
    config_path = os.path.join(output_dir, 'ed_config.cfg')
    write_ed_config(ham_params, config_path, cluster_data,
                    method=model_params.get('method', 'FULL'),
                    compute_thermo=model_params.get('compute_thermo', True),
                    temp_min=model_params.get('temp_min', 0.01),
                    temp_max=model_params.get('temp_max', 10.0),
                    temp_bins=model_params.get('temp_bins', 100))
    
    return config_path


def get_lattice_constant_triangular(order, cluster_data):
    """
    Compute the lattice constant (multiplicity per site) for triangle-based NLCE.
    
    For triangle-based expansion:
    - Order 0 (single site): L = 1 (one per site)
    - Order n (n triangles): L = 2 / |Aut(c)|
      (factor of 2 because there are 2 triangles per site in the infinite lattice)
    
    Args:
        order: NLCE order (number of triangles)
        cluster_data: Cluster JSON data containing weight and other info
        
    Returns:
        Lattice constant (multiplicity per site)
    """
    if order == 0:
        return 1.0
    
    # For triangle clusters, there are 2 up-triangles per site in infinite lattice
    # The lattice constant is: 2 / |Aut(c)| for a cluster with automorphism group Aut(c)
    # We store the weight, but need automorphism count
    # For now, return 2/n_triangles as approximation (refined later with actual automorphisms)
    
    n_triangles = cluster_data.get('n_triangles', order)
    
    # More accurate: each triangle appears 1/(symmetry factor) times per site
    # For the infinite triangular lattice, the multiplicity depends on the cluster shape
    # This is a placeholder - should be computed from symmetry analysis
    return 2.0 / n_triangles if n_triangles > 0 else 1.0


def visualize_bond_types(cluster_data, output_path='triangle_bond_types.png', annotate_bonds=True):
    """
    Visualize bond direction types for a triangle-based cluster.

    Args:
        cluster_data: Cluster JSON data with sites, bonds, and optional bond_directions
        output_path: Path to save the figure
        annotate_bonds: Whether to label bonds with index and direction

    Returns:
        Saved output path, or None if matplotlib is unavailable
    """
    try:
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
    except ImportError:
        print("Warning: matplotlib not installed, skipping bond type visualization")
        return None

    bonds = cluster_data.get('bonds', [])
    if not bonds:
        raise ValueError("cluster_data has no bonds to visualize")

    n_sites = cluster_data.get('n_sites', 0)
    bond_directions = cluster_data.get('bond_directions', [0] * len(bonds))
    if len(bond_directions) < len(bonds):
        bond_directions = list(bond_directions) + [0] * (len(bonds) - len(bond_directions))

    if 'positions' in cluster_data and len(cluster_data['positions']) == n_sites:
        positions = np.array(cluster_data['positions'], dtype=float)
    elif 'sites' in cluster_data and len(cluster_data['sites']) == n_sites:
        lattice_sites = np.array(cluster_data['sites'], dtype=float)
        positions = np.outer(lattice_sites[:, 0], A1) + np.outer(lattice_sites[:, 1], A2)
    else:
        raise ValueError("cluster_data requires either 'positions' or 'sites' with length n_sites")

    direction_colors = {0: 'tab:blue', 1: 'tab:orange', 2: 'tab:green'}
    direction_labels = {
        0: 'dir 0: a1, phi=0',
        1: 'dir 1: a2, phi=-2pi/3',
        2: 'dir 2: a2-a1, phi=+2pi/3'
    }

    fig, ax = plt.subplots(figsize=(7.5, 6.5))

    for bond_idx, (i, j) in enumerate(bonds):
        d = int(bond_directions[bond_idx])
        color = direction_colors.get(d, 'tab:red')
        p1 = positions[i]
        p2 = positions[j]
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color=color, linewidth=2.5, alpha=0.9, zorder=1)

        if annotate_bonds:
            midpoint = 0.5 * (p1 + p2)
            ax.text(midpoint[0], midpoint[1], f"b{bond_idx}:d{d}", fontsize=8,
                    ha='center', va='center',
                    bbox={'boxstyle': 'round,pad=0.2', 'facecolor': 'white', 'alpha': 0.8, 'edgecolor': 'none'},
                    zorder=3)

    ax.scatter(positions[:, 0], positions[:, 1], s=100, c='black', zorder=2)
    for site_idx, pos in enumerate(positions):
        ax.text(pos[0], pos[1], f"{site_idx}", fontsize=9, color='white',
                ha='center', va='center', zorder=4)

    handles = [
        Line2D([0], [0], color=direction_colors[0], linewidth=2.5, label=direction_labels[0]),
        Line2D([0], [0], color=direction_colors[1], linewidth=2.5, label=direction_labels[1]),
        Line2D([0], [0], color=direction_colors[2], linewidth=2.5, label=direction_labels[2]),
    ]

    ax.legend(handles=handles, loc='best', frameon=True)
    ax.set_title('Triangle-based cluster bond types')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.25)
    plt.tight_layout()

    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    return output_path


if __name__ == '__main__':
    # Test with a sample cluster
    test_cluster = {
        'order': 1,
        'n_triangles': 1,
        'n_sites': 3,
        'n_bonds': 3,
        'sites': [[0, 0], [1, 0], [0, 1]],
        'positions': [[0.0, 0.0], [1.0, 0.0], [0.5, 0.866025]],
        'bonds': [[0, 1], [1, 2], [0, 2]],
        'bond_directions': [0, 2, 1],
        'weight': 1
    }
    
    print("Testing XXZ J1-J2 model (isotropic, Jz_ratio=1):")
    ham = prepare_xxz_j1j2(test_cluster, J1=1.0)
    for inter in ham['interactions']:
        print(f"  Bond {inter['site1']}-{inter['site2']}: "
              f"Jxx={inter['Jxx']}, Jyy={inter['Jyy']}, Jzz={inter['Jzz']}")
    
    print("\nTesting XXZ J1-J2 model (anisotropic, Jz_ratio=0.5):")
    ham = prepare_xxz_j1j2(test_cluster, J1=1.0, Jz_ratio=0.5)
    for inter in ham['interactions']:
        print(f"  Bond {inter['site1']}-{inter['site2']}: "
              f"Jxx={inter['Jxx']}, Jyy={inter['Jyy']}, Jzz={inter['Jzz']}")
    
    print("\nTesting anisotropic exchange model:")
    ham = prepare_anisotropic_exchange(test_cluster, Jzz=1.0, Jpm=0.5, Jpmpm=0.1, Jzpm=0.05)
    for inter in ham['interactions']:
        print(f"  Bond {inter['site1']}-{inter['site2']} (dir={inter['direction']}): "
              f"γ={inter['gamma_re']:.3f}+{inter['gamma_im']:.3f}i")
        print(f"    Jxx={inter['Jxx']:.4f}, Jyy={inter['Jyy']:.4f}, Jzz={inter['Jzz']:.4f}")
        print(f"    Jxy={inter['Jxy']:.4f}, Jxz={inter['Jxz']:.4f}, Jyz={inter['Jyz']:.4f}")

    plot_path = visualize_bond_types(test_cluster, output_path='triangle_bond_types_test.png')
    if plot_path is not None:
        print(f"\nBond-type plot saved to: {plot_path}")
