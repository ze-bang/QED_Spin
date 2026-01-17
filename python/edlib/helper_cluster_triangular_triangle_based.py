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


def prepare_heisenberg_j1j2(cluster_data, J1=1.0, J2=0.0):
    """
    Prepare J1-J2 Heisenberg model parameters.
    
    H = J1 * sum_{<ij>} S_i · S_j + J2 * sum_{<<ij>>} S_i · S_j
    
    Args:
        cluster_data: Cluster JSON data
        J1: Nearest-neighbor exchange (default: 1.0)
        J2: Next-nearest neighbor exchange (default: 0.0)
        
    Returns:
        Dictionary with Hamiltonian parameters for ED
    """
    n_sites = cluster_data['n_sites']
    bonds = cluster_data['bonds']
    
    # For J1-J2, we only have nearest-neighbor bonds in the cluster
    # (J2 would require second-neighbor bonds, which need to be computed separately)
    
    interactions = []
    for i, j in bonds:
        interactions.append({
            'site1': i,
            'site2': j,
            'Jxx': J1,
            'Jyy': J1,
            'Jzz': J1
        })
    
    return {
        'n_sites': n_sites,
        'interactions': interactions,
        'model': 'heisenberg_j1j2',
        'J1': J1,
        'J2': J2
    }


def prepare_xxz_model(cluster_data, Jxy=1.0, Jz=1.0, h=0.0, h_direction=(0, 0, 1)):
    """
    Prepare XXZ model parameters.
    
    H = sum_{<ij>} [Jxy (S_i^x S_j^x + S_i^y S_j^y) + Jz S_i^z S_j^z] - h sum_i S_i · n
    
    Args:
        cluster_data: Cluster JSON data
        Jxy: XY coupling strength
        Jz: Z coupling strength
        h: Magnetic field strength
        h_direction: Field direction (normalized internally)
        
    Returns:
        Dictionary with Hamiltonian parameters for ED
    """
    n_sites = cluster_data['n_sites']
    bonds = cluster_data['bonds']
    
    interactions = []
    for i, j in bonds:
        interactions.append({
            'site1': i,
            'site2': j,
            'Jxx': Jxy,
            'Jyy': Jxy,
            'Jzz': Jz
        })
    
    # Normalize field direction
    h_dir = np.array(h_direction, dtype=float)
    h_dir = h_dir / np.linalg.norm(h_dir) if np.linalg.norm(h_dir) > 0 else np.array([0, 0, 1])
    
    return {
        'n_sites': n_sites,
        'interactions': interactions,
        'h': h,
        'h_direction': h_dir.tolist(),
        'model': 'xxz',
        'Jxy': Jxy,
        'Jz': Jz
    }


def prepare_anisotropic_exchange(cluster_data, Jzz=1.0, Jpm=0.0, Jpmpm=0.0, Jzpm=0.0, 
                                  h=0.0, h_direction=(0, 0, 1)):
    """
    Prepare anisotropic exchange model (YbMgGaO4-type) parameters.
    
    H = Σ_{⟨ij⟩_α} [J_zz S_i^z S_j^z 
                    + J_± (S_i^+ S_j^- + S_i^- S_j^+)
                    + J_±± (γ_α S_i^+ S_j^+ + γ_α* S_i^- S_j^-)
                    - i J_z±/2 ((γ_α* S_i^+ - γ_α S_i^-) S_j^z 
                               + S_i^z (γ_α* S_j^+ - γ_α S_j^-))]
    
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
    
    return {
        'n_sites': n_sites,
        'interactions': interactions,
        'h': h,
        'h_direction': h_dir.tolist(),
        'model': 'anisotropic_exchange',
        'Jzz': Jzz,
        'Jpm': Jpm,
        'Jpmpm': Jpmpm,
        'Jzpm': Jzpm
    }


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
        Jxx = inter.get('Jxx', 0.0)
        Jyy = inter.get('Jyy', 0.0)
        Jzz = inter.get('Jzz', 0.0)
        Jxy = inter.get('Jxy', 0.0)
        Jxz = inter.get('Jxz', 0.0)
        Jyz = inter.get('Jyz', 0.0)
        
        lines.append(f"interaction{idx} = {site1}, {site2}, {Jxx}, {Jyy}, {Jzz}, {Jxy}, {Jxz}, {Jyz}")
    
    # Magnetic field
    if ham_params.get('h', 0.0) != 0.0:
        h = ham_params['h']
        h_dir = ham_params.get('h_direction', [0, 0, 1])
        lines.append("")
        lines.append("# Magnetic field")
        lines.append(f"h = {h}")
        lines.append(f"h_direction = {h_dir[0]}, {h_dir[1]}, {h_dir[2]}")
    
    # Write the file
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))
    
    return output_path


def prepare_cluster_for_ed(cluster_json_path, output_dir, model='heisenberg', **model_params):
    """
    Prepare a cluster for ED calculation.
    
    Args:
        cluster_json_path: Path to cluster JSON file
        output_dir: Directory to write ED input files
        model: Model type ('heisenberg', 'xxz', 'anisotropic')
        **model_params: Model-specific parameters
        
    Returns:
        Path to ED config file
    """
    cluster_data = load_cluster_json(cluster_json_path)
    
    if model == 'heisenberg' or model == 'heisenberg_j1j2':
        ham_params = prepare_heisenberg_j1j2(cluster_data, 
                                              J1=model_params.get('J1', 1.0),
                                              J2=model_params.get('J2', 0.0))
    elif model == 'xxz':
        ham_params = prepare_xxz_model(cluster_data,
                                        Jxy=model_params.get('Jxy', 1.0),
                                        Jz=model_params.get('Jz', 1.0),
                                        h=model_params.get('h', 0.0),
                                        h_direction=model_params.get('h_direction', (0, 0, 1)))
    elif model == 'anisotropic':
        ham_params = prepare_anisotropic_exchange(cluster_data,
                                                   Jzz=model_params.get('Jzz', 1.0),
                                                   Jpm=model_params.get('Jpm', 0.0),
                                                   Jpmpm=model_params.get('Jpmpm', 0.0),
                                                   Jzpm=model_params.get('Jzpm', 0.0),
                                                   h=model_params.get('h', 0.0),
                                                   h_direction=model_params.get('h_direction', (0, 0, 1)))
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
    
    print("Testing Heisenberg model:")
    ham = prepare_heisenberg_j1j2(test_cluster, J1=1.0)
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
