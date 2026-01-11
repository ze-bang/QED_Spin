#!/usr/bin/env python3
"""
BFG Kagome Order Parameter Calculator

Post-processing script for computing order parameters from exact diagonalization
wavefunctions on the kagome lattice for the Balents-Fisher-Girvin (BFG) model.

Order parameters computed:
A) Translation symmetry breaking (solids / density waves)
   1. Off-diagonal structure factor S(q) = <S^+S^-> - Bragg peaks indicate crystalline order

B) Rotational symmetry breaking (nematic / stripe orientation)
   2. Nematic order from bond-orientation anisotropy - C6 → C2 breaking
   3. Stripe structure factor - preferred direction detection

C) Resonating bond/plaquette order
   4. Bond/dimer structure factor - VBS order detection
   5. Plaquette/bow-tie resonance order - BFG-native 4-spin ring flip order

Usage:
    python compute_bfg_order_parameters.py <wavefunction_file> <cluster_dir> [options]
    
Examples:
    python compute_bfg_order_parameters.py results.h5 ./cluster_data --eigenvector-index 0
    python compute_bfg_order_parameters.py results.h5 ./cluster_data --all-eigenvalues 10

Author: Auto-generated for BFG kagome model
"""

import numpy as np
import os
import sys
import argparse
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import h5py

# Try to import numba for JIT compilation
try:
    from numba import jit, prange
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    # Create dummy decorators
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    prange = range

# Try to import matplotlib for plotting
try:
    import matplotlib.pyplot as plt
    from matplotlib import cm
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


# =============================================================================
# Bitwise Operator Helper Functions
# =============================================================================

@jit(nopython=True)
def get_bit(state: int, site: int) -> int:
    """Get bit value (0 or 1) at site position"""
    return (state >> site) & 1


@jit(nopython=True)
def flip_bit(state: int, site: int) -> int:
    """Flip bit at site position"""
    return state ^ (1 << site)


def apply_sz(state: int, site: int) -> Tuple[int, complex]:
    """
    Apply S^z operator at site.
    S^z|↑⟩ = +1/2|↑⟩, S^z|↓⟩ = -1/2|↓⟩
    
    Convention: bit=1 → spin up (+1/2), bit=0 → spin down (-1/2)
    """
    bit = get_bit(state, site)
    eigenvalue = 0.5 if bit == 1 else -0.5
    return state, eigenvalue


def apply_sp(state: int, site: int) -> Tuple[Optional[int], complex]:
    """
    Apply S^+ operator at site.
    S^+|↓⟩ = |↑⟩, S^+|↑⟩ = 0
    
    Returns (new_state, coefficient) or (None, 0) if annihilated.
    """
    bit = get_bit(state, site)
    if bit == 0:  # spin down → can raise
        return flip_bit(state, site), 1.0
    else:  # spin up → annihilated
        return None, 0.0


def apply_sm(state: int, site: int) -> Tuple[Optional[int], complex]:
    """
    Apply S^- operator at site.
    S^-|↑⟩ = |↓⟩, S^-|↓⟩ = 0
    
    Returns (new_state, coefficient) or (None, 0) if annihilated.
    """
    bit = get_bit(state, site)
    if bit == 1:  # spin up → can lower
        return flip_bit(state, site), 1.0
    else:  # spin down → annihilated
        return None, 0.0


# =============================================================================
# Lattice Data Loading
# =============================================================================

def load_kagome_cluster(cluster_dir: str, kpoints_file: str = None) -> Dict:
    """
    Load kagome cluster data from helper_kagome_bfg.py output files.
    
    Args:
        cluster_dir: Directory containing cluster data files
        kpoints_file: Optional path to lattice parameters file with k-points
    
    Returns dict with:
        - n_sites: number of sites
        - positions: dict {site_id: (x, y)}
        - sublattice: dict {site_id: sublattice_index}
        - nn_list: dict {site_id: [neighbor_ids]} for NN
        - nn2_list: dict {site_id: [neighbor_ids]} for 2NN
        - nn3_list: dict {site_id: [neighbor_ids]} for 3NN
        - edges_nn: list of (i, j) tuples for NN bonds
        - edges_2nn: list of (i, j) tuples for 2NN bonds
        - a1, a2: lattice vectors
        - b1, b2: reciprocal lattice vectors
        - k_points: allowed momentum points (if available)
    """
    cluster = {}
    
    # Load positions file
    pos_file = os.path.join(cluster_dir, "positions.dat")
    if os.path.exists(pos_file):
        positions = {}
        sublattice = {}
        with open(pos_file, 'r') as f:
            for line in f:
                if line.startswith('#') or not line.strip():
                    continue
                parts = line.strip().split()
                if len(parts) >= 5:
                    site_id = int(parts[0])
                    # matrix_index = int(parts[1])  # usually same as site_id
                    sub_idx = int(parts[2])
                    x = float(parts[3])
                    y = float(parts[4])
                    positions[site_id] = np.array([x, y])
                    sublattice[site_id] = sub_idx
        cluster['positions'] = positions
        cluster['sublattice'] = sublattice
        cluster['n_sites'] = len(positions)
    else:
        raise FileNotFoundError(f"Positions file not found: {pos_file}")
    
    # Find NN list file
    nn_files = [f for f in os.listdir(cluster_dir) if f.endswith('_nn_list.dat')]
    if nn_files:
        cluster['nn_list'] = load_neighbor_list(os.path.join(cluster_dir, nn_files[0]))
    
    # Find 2NN list file
    nn2_files = [f for f in os.listdir(cluster_dir) if f.endswith('_2nn_list.dat')]
    if nn2_files:
        cluster['nn2_list'] = load_neighbor_list(os.path.join(cluster_dir, nn2_files[0]))
    
    # Find 3NN list file
    nn3_files = [f for f in os.listdir(cluster_dir) if f.endswith('_3nn_list.dat')]
    if nn3_files:
        cluster['nn3_list'] = load_neighbor_list(os.path.join(cluster_dir, nn3_files[0]))
    
    # Build edge lists from NN lists (avoid double counting)
    if 'nn_list' in cluster:
        cluster['edges_nn'] = build_edge_list(cluster['nn_list'])
    if 'nn2_list' in cluster:
        cluster['edges_2nn'] = build_edge_list(cluster['nn2_list'])
    if 'nn3_list' in cluster:
        cluster['edges_3nn'] = build_edge_list(cluster['nn3_list'])
    
    # Kagome lattice vectors
    cluster['a1'] = np.array([1.0, 0.0])
    cluster['a2'] = np.array([0.5, np.sqrt(3)/2])
    
    # Reciprocal lattice vectors: b_i = 2π(a_j × z) / (a1 × a2) · z
    det = cluster['a1'][0] * cluster['a2'][1] - cluster['a1'][1] * cluster['a2'][0]
    cluster['b1'] = 2 * np.pi * np.array([cluster['a2'][1], -cluster['a2'][0]]) / det
    cluster['b2'] = 2 * np.pi * np.array([-cluster['a1'][1], cluster['a1'][0]]) / det
    
    # Try to load allowed k-points from lattice parameters file
    cluster['k_points'] = load_allowed_kpoints(cluster_dir, kpoints_file)
    
    return cluster


def load_allowed_kpoints(cluster_dir: str, kpoints_file: str = None) -> Optional[np.ndarray]:
    """
    Load allowed momentum points from the lattice parameters file.
    
    For a finite cluster, only discrete k-points are allowed by the
    boundary conditions. This function reads them from the 
    *_lattice_parameters.dat file.
    
    Search order:
    1. Explicit kpoints_file if provided
    2. cluster_dir
    3. Parent directory of cluster_dir
    
    Returns:
        Array of shape (n_k, 2) with (kx, ky) for each allowed k-point,
        or None if file not found.
    """
    param_file = None
    
    # Option 1: Explicit file path
    if kpoints_file and os.path.exists(kpoints_file):
        param_file = kpoints_file
    else:
        # Option 2: Look in cluster_dir
        if os.path.isdir(cluster_dir):
            param_files = [f for f in os.listdir(cluster_dir) 
                           if f.endswith('_lattice_parameters.dat')]
            if param_files:
                param_file = os.path.join(cluster_dir, param_files[0])
        
        # Option 3: Look in parent directory
        if param_file is None:
            parent_dir = os.path.dirname(os.path.abspath(cluster_dir))
            if os.path.isdir(parent_dir):
                param_files = [f for f in os.listdir(parent_dir) 
                               if f.endswith('_lattice_parameters.dat')]
                if param_files:
                    param_file = os.path.join(parent_dir, param_files[0])
    
    if param_file is None:
        return None
    
    k_points = []
    in_kpoint_section = False
    
    with open(param_file, 'r') as f:
        for line in f:
            line = line.strip()
            
            # Detect start of k-point list (after "Format: k_index, n1, n2, kx, ky")
            if 'k_index' in line and 'kx' in line and 'ky' in line:
                in_kpoint_section = True
                continue
            
            # End of k-point section (blank line or new section)
            if in_kpoint_section and (line.startswith('#') or not line):
                if line.startswith('# Total number'):
                    continue  # Still in section
                if not line:
                    break  # End of section
                continue
            
            if in_kpoint_section:
                parts = line.split()
                if len(parts) >= 5:
                    try:
                        kx = float(parts[3])
                        ky = float(parts[4])
                        k_points.append([kx, ky])
                    except (ValueError, IndexError):
                        pass
    
    if k_points:
        k_points = np.array(k_points)
        print(f"  Loaded {len(k_points)} allowed k-points from {param_files[0]}")
        return k_points
    else:
        print("  Warning: Could not parse k-points from lattice parameters file")
        return None


def load_neighbor_list(filename: str) -> Dict[int, List[int]]:
    """Load neighbor list from file"""
    nn_list = defaultdict(list)
    with open(filename, 'r') as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            parts = line.strip().split()
            if len(parts) >= 2:
                site_id = int(parts[0])
                n_neighbors = int(parts[1])
                neighbors = [int(parts[i+2]) for i in range(n_neighbors)]
                nn_list[site_id] = neighbors
    return dict(nn_list)


def build_edge_list(nn_list: Dict[int, List[int]]) -> List[Tuple[int, int]]:
    """Build edge list from neighbor list, avoiding duplicates"""
    edges = set()
    for site, neighbors in nn_list.items():
        for neighbor in neighbors:
            edge = tuple(sorted([site, neighbor]))
            edges.add(edge)
    return list(edges)


# =============================================================================
# Wavefunction Loading
# =============================================================================

def load_wavefunction(filepath: str, eigenvector_index: int = 0) -> np.ndarray:
    """
    Load wavefunction from HDF5 file.
    
    Args:
        filepath: Path to HDF5 file
        eigenvector_index: Index of eigenvector to load (default: 0 = ground state)
    
    Returns:
        Complex 1D array representing the wavefunction in the computational basis
    """
    with h5py.File(filepath, 'r') as f:
        dataset_name = f'/eigendata/eigenvector_{eigenvector_index}'
        if dataset_name not in f:
            raise KeyError(f"Eigenvector {eigenvector_index} not found in file")
        data = f[dataset_name][:]
        
        # Handle compound dtype (real, imag) from C++ code
        if data.dtype.names and 'real' in data.dtype.names:
            return data['real'] + 1j * data['imag']
        else:
            return data


def load_eigenvalue(filepath: str, index: int = 0) -> float:
    """Load eigenvalue from HDF5 file"""
    with h5py.File(filepath, 'r') as f:
        eigenvalues = f['/eigendata/eigenvalues'][:]
        return eigenvalues[index]


# =============================================================================
# Order Parameter Computations
# =============================================================================

# -----------------------------------------------------------------------------
# A) OFF-DIAGONAL STRUCTURE FACTOR S(q) = <S^+S^-> - Translation Symmetry Breaking
# -----------------------------------------------------------------------------

def compute_sz_expectation(psi: np.ndarray, n_sites: int) -> np.ndarray:
    """
    Compute <S^z_i> for each site.
    
    Args:
        psi: Wavefunction in computational basis
        n_sites: Number of sites
    
    Returns:
        Array of <S^z_i> values
    """
    dim = len(psi)
    sz_exp = np.zeros(n_sites)
    
    for state in range(dim):
        prob = np.abs(psi[state])**2
        if prob < 1e-15:
            continue
        for site in range(n_sites):
            sz = 0.5 if get_bit(state, site) == 1 else -0.5
            sz_exp[site] += sz * prob
    
    return sz_exp


def compute_szsz_correlation(psi: np.ndarray, n_sites: int) -> np.ndarray:
    """
    Compute <S^z_i S^z_j> correlation matrix.
    
    Args:
        psi: Wavefunction in computational basis
        n_sites: Number of sites
    
    Returns:
        n_sites × n_sites array of <S^z_i S^z_j> values
    """
    dim = len(psi)
    szsz = np.zeros((n_sites, n_sites))
    
    for state in range(dim):
        prob = np.abs(psi[state])**2
        if prob < 1e-15:
            continue
        for i in range(n_sites):
            sz_i = 0.5 if get_bit(state, i) == 1 else -0.5
            for j in range(n_sites):
                sz_j = 0.5 if get_bit(state, j) == 1 else -0.5
                szsz[i, j] += sz_i * sz_j * prob
    
    return szsz


def compute_spsm_correlation(psi: np.ndarray, n_sites: int) -> np.ndarray:
    """
    Compute <S^+_i S^-_j> correlation matrix.
    
    Note: <S^+> = <S^-> = 0 in any state that conserves S^z_total,
    so we don't need to compute the connected correlation.
    
    Args:
        psi: Wavefunction in computational basis
        n_sites: Number of sites
    
    Returns:
        n_sites × n_sites array of <S^+_i S^-_j> values (complex)
    """
    dim = len(psi)
    spsm = np.zeros((n_sites, n_sites), dtype=complex)
    
    for state in range(dim):
        coeff = psi[state]
        if np.abs(coeff) < 1e-15:
            continue
        
        for i in range(n_sites):
            for j in range(n_sites):
                # S^+_i S^-_j: raise spin at i, lower spin at j
                # For i != j: flip i up and j down
                # For i == j: S^+_i S^-_i = (1/2 + S^z_i) only for spin down at i
                
                if i == j:
                    # S^+_i S^-_i = |↑⟩⟨↓|·|↓⟩⟨↑| = |↑⟩⟨↑| on spin-down state
                    # This equals (1/2 - S^z) for spin up, 0 for spin down
                    # Wait, S^+ S^- = (1/2 + S^z) 
                    # S^+ |↓⟩ = |↑⟩, S^- |↑⟩ = |↓⟩
                    # S^+ S^- |↑⟩ = S^+ |↓⟩ = |↑⟩ → eigenvalue 1
                    # S^+ S^- |↓⟩ = 0
                    # So S^+ S^- = (1/2 + S^z) = n_up
                    bit_i = get_bit(state, i)
                    if bit_i == 1:  # spin up
                        spsm[i, i] += np.abs(coeff)**2
                else:
                    # S^+_i S^-_j with i != j
                    # Need spin down at i (to raise) and spin up at j (to lower)
                    bit_i = get_bit(state, i)
                    bit_j = get_bit(state, j)
                    
                    if bit_i == 0 and bit_j == 1:  # i down, j up
                        # S^+_i |...0_i...⟩ = |...1_i...⟩
                        # S^-_j |...1_j...⟩ = |...0_j...⟩
                        new_state = set_bit(state, i, 1)
                        new_state = set_bit(new_state, j, 0)
                        # Matrix element: <new_state| S^+_i S^-_j |state> = 1
                        spsm[i, j] += np.conj(psi[new_state]) * coeff
    
    return spsm


def compute_diagonal_structure_factor(
    psi: np.ndarray,
    cluster: Dict,
    q_grid: np.ndarray
) -> np.ndarray:
    """
    Compute diagonal spin structure factor:
    S^{zz}(q) = (1/N) Σ_{j,k} (<S^z_j S^z_k> - <S^z_j><S^z_k>) exp(iq·(r_j - r_k))
    
    Args:
        psi: Wavefunction
        cluster: Cluster data dict
        q_grid: Array of q vectors, shape (n_q, 2)
    
    Returns:
        Array of S^{zz}(q) values, shape (n_q,)
    """
    n_sites = cluster['n_sites']
    positions = cluster['positions']
    
    # Compute correlations
    sz_exp = compute_sz_expectation(psi, n_sites)
    szsz = compute_szsz_correlation(psi, n_sites)
    
    # Connected correlation: <SzSz>_c = <SzSz> - <Sz><Sz>
    szsz_connected = szsz - np.outer(sz_exp, sz_exp)
    
    # Structure factor for each q
    s_q = np.zeros(len(q_grid), dtype=complex)
    
    for iq, q in enumerate(q_grid):
        for i in range(n_sites):
            r_i = positions[i]
            for j in range(n_sites):
                r_j = positions[j]
                phase = np.exp(1j * np.dot(q, r_i - r_j))
                s_q[iq] += szsz_connected[i, j] * phase
        s_q[iq] /= n_sites
    
    return s_q


def compute_translation_order_parameter(
    psi: np.ndarray,
    cluster: Dict,
    n_q_points: int = 50
) -> Dict:
    """
    Compute translation symmetry breaking order parameter.
    
    Uses S^+ S^- correlations (off-diagonal structure factor):
        S(q) = (1/N) Σ_{i,j} <S^+_i S^-_j> exp(iq·(r_i - r_j))
    
    Note: We use the raw <S^+_i S^-_j> without subtracting <S^+><S^->
    since <S^+> = <S^-> = 0 in any S^z-conserving state.
    
    Uses the discrete allowed k-points for the finite cluster if available,
    otherwise falls back to a dense q-grid for visualization.
    
    m_trans = sqrt(S(Q_max) / N)
    
    Returns dict with:
        - s_q_2d: structure factor on 2D grid (for visualization)
        - s_q_discrete: structure factor at allowed k-points
        - k_points: allowed k-point coordinates
        - m_trans: order parameter value
        - q_max: wavevector with maximum S(q)
    """
    n_sites = cluster['n_sites']
    b1, b2 = cluster['b1'], cluster['b2']
    positions = cluster['positions']
    
    # Pre-compute S^+ S^- correlations
    # Note: <S^+> = <S^-> = 0 so no need for connected correlation
    print(f"    Computing S^+ S^- correlation matrix ({n_sites}x{n_sites})...")
    spsm = compute_spsm_correlation(psi, n_sites)
    
    # Also compute Sz for reference
    sz_exp = compute_sz_expectation(psi, n_sites)
    
    # =========================================================================
    # Compute at discrete allowed k-points (for order parameter)
    # =========================================================================
    k_points = cluster.get('k_points', None)
    
    if k_points is not None:
        n_k = len(k_points)
        s_q_discrete = np.zeros(n_k, dtype=complex)
        
        for ik, q in enumerate(k_points):
            for i in range(n_sites):
                r_i = positions[i]
                for j in range(n_sites):
                    r_j = positions[j]
                    phase = np.exp(1j * np.dot(q, r_i - r_j))
                    s_q_discrete[ik] += spsm[i, j] * phase
            s_q_discrete[ik] /= n_sites
        
        # Find maximum over ALL k-points (including q=0)
        s_q_abs_discrete = np.abs(s_q_discrete)
        max_idx = np.argmax(s_q_abs_discrete)
        s_q_max = s_q_abs_discrete[max_idx]
        q_max = k_points[max_idx]
        q_max_idx = max_idx
        
        # Order parameter
        m_trans = np.sqrt(np.abs(s_q_max) / n_sites)
        
        print(f"    S(q) [S^+S^-] computed at {n_k} allowed k-points")
        print(f"    Maximum at q = ({q_max[0]:.4f}, {q_max[1]:.4f}), S(q) = {s_q_max:.6f}")
    else:
        s_q_discrete = None
        q_max_idx = None
    
    # =========================================================================
    # Also compute on dense grid (for visualization)
    # =========================================================================
    q1_vals = np.linspace(-1, 1, n_q_points)
    q2_vals = np.linspace(-1, 1, n_q_points)
    s_q_2d = np.zeros((n_q_points, n_q_points), dtype=complex)
    
    for i1, q1 in enumerate(q1_vals):
        for i2, q2 in enumerate(q2_vals):
            q = q1 * b1 + q2 * b2
            for i in range(n_sites):
                r_i = positions[i]
                for j in range(n_sites):
                    r_j = positions[j]
                    phase = np.exp(1j * np.dot(q, r_i - r_j))
                    s_q_2d[i1, i2] += spsm[i, j] * phase
            s_q_2d[i1, i2] /= n_sites
    
    # If no discrete k-points, find max from dense grid
    if k_points is None:
        s_q_abs = np.abs(s_q_2d)
        max_idx = np.unravel_index(np.argmax(s_q_abs), s_q_abs.shape)
        s_q_max = s_q_abs[max_idx]
        q_max = q1_vals[max_idx[0]] * b1 + q2_vals[max_idx[1]] * b2
        m_trans = np.sqrt(np.abs(s_q_max) / n_sites)
        q_max_idx = None
    
    return {
        's_q_2d': s_q_2d,
        'q1_vals': q1_vals,
        'q2_vals': q2_vals,
        's_q_discrete': s_q_discrete,
        'k_points': k_points,
        'm_trans': m_trans,
        'q_max': q_max,
        'q_max_idx': q_max_idx,
        's_q_max': s_q_max,
        'sz_exp': sz_exp,
        'spsm': spsm  # Store the correlation matrix
    }


# -----------------------------------------------------------------------------
# B) NEMATIC ORDER - Rotational Symmetry Breaking
# -----------------------------------------------------------------------------

def compute_xy_bond_expectation(
    psi: np.ndarray,
    n_sites: int,
    site_i: int,
    site_j: int
) -> complex:
    """
    Compute XY bond expectation: <S^+_i S^-_j + S^-_i S^+_j>
    
    This is the "flip-flop" term that resonates between |↑↓⟩ and |↓↑⟩
    """
    dim = len(psi)
    expectation = 0.0 + 0.0j
    
    for state in range(dim):
        coeff = psi[state]
        if np.abs(coeff) < 1e-15:
            continue
        
        # S^+_i S^-_j term
        new_state_1, c1 = apply_sp(state, site_i)
        if new_state_1 is not None:
            new_state_1, c2 = apply_sm(new_state_1, site_j)
            if new_state_1 is not None:
                expectation += np.conj(psi[new_state_1]) * coeff * c1 * c2
        
        # S^-_i S^+_j term
        new_state_2, c3 = apply_sm(state, site_i)
        if new_state_2 is not None:
            new_state_2, c4 = apply_sp(new_state_2, site_j)
            if new_state_2 is not None:
                expectation += np.conj(psi[new_state_2]) * coeff * c3 * c4
    
    return expectation


def get_bond_orientation(cluster: Dict, site_i: int, site_j: int) -> int:
    """
    Determine bond orientation (0, 1, or 2) for kagome lattice.
    
    Bond orientations on kagome:
    - α=0: bonds along a1 direction (horizontal)
    - α=1: bonds at +60° 
    - α=2: bonds at -60° (or +120°)
    
    Returns orientation index 0, 1, or 2
    """
    positions = cluster['positions']
    r_i = positions[site_i]
    r_j = positions[site_j]
    
    # Bond vector
    dr = r_j - r_i
    
    # Handle periodic boundary conditions by taking minimum image
    # (Assume bonds are short-range, so unwrapped dr is close to 0.5)
    
    # Compute angle from horizontal
    angle = np.arctan2(dr[1], dr[0])
    angle_deg = np.degrees(angle) % 180  # Map to [0, 180)
    
    # Kagome bond angles: 0°, 60°, 120° (mod 180)
    if angle_deg < 30 or angle_deg >= 150:
        return 0  # ~0° (horizontal)
    elif 30 <= angle_deg < 90:
        return 1  # ~60°
    else:
        return 2  # ~120°


def compute_nematic_order(psi: np.ndarray, cluster: Dict) -> Dict:
    """
    Compute nematic order from bond-orientation anisotropy.
    
    ψ_nem = Σ_{α=0,1,2} ω^α * O̅_α
    where O̅_α = (1/N_α) Σ_{⟨ij⟩∈α} <S^+_i S^-_j + h.c.>
    and ω = exp(2πi/3)
    
    m_nem = |ψ_nem| detects C_6 → C_2 breaking
    """
    n_sites = cluster['n_sites']
    edges = cluster.get('edges_nn', [])
    
    # Classify bonds by orientation
    bonds_by_orientation = {0: [], 1: [], 2: []}
    for (i, j) in edges:
        alpha = get_bond_orientation(cluster, i, j)
        bonds_by_orientation[alpha].append((i, j))
    
    # Compute average bond observable for each orientation
    omega = np.exp(2j * np.pi / 3)
    O_bar = np.zeros(3, dtype=complex)
    
    for alpha in range(3):
        bonds = bonds_by_orientation[alpha]
        if len(bonds) == 0:
            continue
        total = 0.0 + 0.0j
        for (i, j) in bonds:
            total += compute_xy_bond_expectation(psi, n_sites, i, j)
        O_bar[alpha] = total / len(bonds)
    
    # Nematic order parameter
    psi_nem = sum(omega**alpha * O_bar[alpha] for alpha in range(3))
    m_nem = np.abs(psi_nem)
    
    # Also compute bond anisotropy measure
    O_bar_abs = np.abs(O_bar)
    if np.sum(O_bar_abs) > 1e-10:
        anisotropy = np.max(O_bar_abs) - np.min(O_bar_abs)
    else:
        anisotropy = 0.0
    
    return {
        'O_bar': O_bar,
        'psi_nem': psi_nem,
        'm_nem': m_nem,
        'anisotropy': anisotropy,
        'bonds_by_orientation': {k: len(v) for k, v in bonds_by_orientation.items()}
    }


def compute_stripe_structure_factor(psi: np.ndarray, cluster: Dict) -> Dict:
    """
    Compute stripe/nematic structure factor for loop model.
    
    In the BFG loop mapping, we define n_{i,α} = 1 if a "loop segment"
    at site i points in direction α.
    
    For the spin model, we approximate this using the XY bond energy
    on bonds of each orientation connected to site i.
    
    S_stripe = (1/N) Σ_{i,j} Σ_{α,β} <n_{i,α} n_{j,β}> exp(2πi(β-α)/3)
    
    m_stripe ≠ 0 implies rotation symmetry breaking
    """
    n_sites = cluster['n_sites']
    nn_list = cluster.get('nn_list', {})
    
    # Compute "loop occupation" n_{i,α} as average XY bond strength on α-bonds from site i
    n_i_alpha = np.zeros((n_sites, 3), dtype=complex)
    
    for i in range(n_sites):
        neighbors = nn_list.get(i, [])
        for j in neighbors:
            alpha = get_bond_orientation(cluster, i, j)
            bond_exp = compute_xy_bond_expectation(psi, n_sites, i, j)
            n_i_alpha[i, alpha] += np.abs(bond_exp)  # Use magnitude as "occupation"
    
    # Normalize by number of bonds of each type per site
    for i in range(n_sites):
        for alpha in range(3):
            if n_i_alpha[i, alpha] > 0:
                n_i_alpha[i, alpha] /= 2  # Approximate normalization
    
    # Compute stripe structure factor
    omega = np.exp(2j * np.pi / 3)
    S_stripe = 0.0 + 0.0j
    
    for i in range(n_sites):
        for j in range(n_sites):
            for alpha in range(3):
                for beta in range(3):
                    phase = omega**(beta - alpha)
                    S_stripe += n_i_alpha[i, alpha] * np.conj(n_i_alpha[j, beta]) * phase
    
    S_stripe /= n_sites
    m_stripe = np.sqrt(np.abs(S_stripe) / n_sites)
    
    return {
        'n_i_alpha': n_i_alpha,
        'S_stripe': S_stripe,
        'm_stripe': m_stripe
    }


# -----------------------------------------------------------------------------
# C) BOND/DIMER STRUCTURE FACTOR - VBS Order
# -----------------------------------------------------------------------------

def compute_all_xy_bonds(psi: np.ndarray, cluster: Dict) -> Dict[Tuple[int, int], complex]:
    """Compute all XY bond expectations"""
    n_sites = cluster['n_sites']
    edges = cluster.get('edges_nn', [])
    
    bond_exp = {}
    for (i, j) in edges:
        bond_exp[(i, j)] = compute_xy_bond_expectation(psi, n_sites, i, j)
    
    return bond_exp


def plot_bond_expectation_map(
    psi: np.ndarray, 
    cluster: Dict, 
    output_path: str = None,
    title: str = None,
    show_site_labels: bool = True,
    colormap: str = 'RdBu_r'
) -> Dict:
    """
    Plot bond expectation values in real space on the kagome lattice.
    
    Visualizes <S^+_i S^-_j + S^-_i S^+_j> for each NN bond.
    Bond colors show the expectation value magnitude/sign.
    Bond widths show the magnitude.
    
    Best used with OBC clusters for clear visualization.
    
    Args:
        psi: Wavefunction
        cluster: Cluster dictionary with positions, edges
        output_path: Where to save the figure (optional)
        title: Plot title (optional)
        show_site_labels: Whether to show site index labels
        colormap: Matplotlib colormap for bond coloring
        
    Returns:
        Dictionary with bond expectation data
    """
    if not HAS_MATPLOTLIB:
        print("WARNING: matplotlib not available, skipping bond map")
        return {}
    
    from matplotlib.collections import LineCollection
    from matplotlib.colors import Normalize
    import matplotlib.cm as cm
    
    positions = cluster['positions']
    edges = cluster.get('edges_nn', [])
    n_sites = cluster['n_sites']
    
    # Compute all bond expectations
    bond_exp = compute_all_xy_bonds(psi, cluster)
    
    # Get real parts for coloring (XY bonds are real for real wavefunctions)
    bond_values = {b: np.real(bond_exp[b]) for b in bond_exp}
    bond_magnitudes = {b: np.abs(bond_exp[b]) for b in bond_exp}
    
    # Statistics
    values_list = list(bond_values.values())
    mag_list = list(bond_magnitudes.values())
    vmin, vmax = min(values_list), max(values_list)
    
    # Symmetric color scale around 0
    v_abs_max = max(abs(vmin), abs(vmax))
    if v_abs_max < 1e-10:
        v_abs_max = 1.0
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # =========================================================================
    # Left panel: Bond values with color (sign-sensitive)
    # =========================================================================
    ax = axes[0]
    
    # Draw bonds as colored lines
    segments = []
    colors = []
    for (i, j) in edges:
        ri, rj = positions[i], positions[j]
        segments.append([(ri[0], ri[1]), (rj[0], rj[1])])
        colors.append(bond_values[(i, j)])
    
    # Normalize colors symmetrically around 0
    norm = Normalize(vmin=-v_abs_max, vmax=v_abs_max)
    lc = LineCollection(segments, cmap=colormap, norm=norm, linewidths=3)
    lc.set_array(np.array(colors))
    ax.add_collection(lc)
    
    # Add colorbar
    cbar = plt.colorbar(lc, ax=ax)
    cbar.set_label(r'$\langle S^+_i S^-_j + S^-_i S^+_j \rangle$', fontsize=11)
    
    # Draw sites
    x_coords = [positions[i][0] for i in range(n_sites)]
    y_coords = [positions[i][1] for i in range(n_sites)]
    ax.scatter(x_coords, y_coords, c='black', s=80, zorder=5)
    
    # Site labels
    if show_site_labels and n_sites <= 60:
        for i in range(n_sites):
            ax.annotate(str(i), positions[i], fontsize=7, ha='center', va='center',
                       color='white', fontweight='bold')
    
    ax.set_aspect('equal')
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('y', fontsize=12)
    ax.set_title('XY Bond Expectations (signed)', fontsize=13)
    
    # Adjust limits with padding
    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)
    padding = 0.5
    ax.set_xlim(x_min - padding, x_max + padding)
    ax.set_ylim(y_min - padding, y_max + padding)
    
    # =========================================================================
    # Right panel: Bond magnitudes with width encoding
    # =========================================================================
    ax = axes[1]
    
    # Normalize magnitudes for line widths
    max_mag = max(mag_list) if max(mag_list) > 1e-10 else 1.0
    min_width, max_width = 1.0, 8.0
    
    # Draw bonds with width proportional to magnitude
    for (i, j) in edges:
        ri, rj = positions[i], positions[j]
        mag = bond_magnitudes[(i, j)]
        width = min_width + (max_width - min_width) * (mag / max_mag)
        color = cm.viridis(mag / max_mag)
        ax.plot([ri[0], rj[0]], [ri[1], rj[1]], color=color, linewidth=width, solid_capstyle='round')
    
    # Draw sites
    ax.scatter(x_coords, y_coords, c='black', s=80, zorder=5)
    
    if show_site_labels and n_sites <= 60:
        for i in range(n_sites):
            ax.annotate(str(i), positions[i], fontsize=7, ha='center', va='center',
                       color='white', fontweight='bold')
    
    # Add magnitude colorbar
    sm = cm.ScalarMappable(cmap='viridis', norm=Normalize(vmin=0, vmax=max_mag))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label(r'$|\langle S^+_i S^-_j + S^-_i S^+_j \rangle|$', fontsize=11)
    
    ax.set_aspect('equal')
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('y', fontsize=12)
    ax.set_title('XY Bond Magnitudes (width ∝ |bond|)', fontsize=13)
    ax.set_xlim(x_min - padding, x_max + padding)
    ax.set_ylim(y_min - padding, y_max + padding)
    
    # Overall title
    if title:
        fig.suptitle(title, fontsize=14, y=1.02)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.savefig(output_path.replace('.png', '.pdf'), bbox_inches='tight')
        print(f"  Saved bond map to {output_path}")
    
    plt.close()
    
    # Summary statistics
    print(f"  Bond expectation statistics:")
    print(f"    Range: [{min(values_list):.6f}, {max(values_list):.6f}]")
    print(f"    Mean:  {np.mean(values_list):.6f}")
    print(f"    Std:   {np.std(values_list):.6f}")
    
    return {
        'bond_exp': bond_exp,
        'bond_values': bond_values,
        'bond_magnitudes': bond_magnitudes,
        'mean': np.mean(values_list),
        'std': np.std(values_list)
    }


def plot_bond_orientation_map(
    psi: np.ndarray,
    cluster: Dict,
    output_path: str = None,
    title: str = None
) -> Dict:
    """
    Plot bond expectations colored by bond orientation (α = 0, 1, 2).
    
    This shows the stripe/nematic order visually - if bonds of one 
    orientation are systematically stronger, it indicates C3 breaking.
    
    Args:
        psi: Wavefunction
        cluster: Cluster dictionary
        output_path: Where to save the figure
        title: Plot title
        
    Returns:
        Dictionary with orientation-resolved bond data
    """
    if not HAS_MATPLOTLIB:
        return {}
    
    positions = cluster['positions']
    edges = cluster.get('edges_nn', [])
    n_sites = cluster['n_sites']
    
    # Compute all bond expectations
    bond_exp = compute_all_xy_bonds(psi, cluster)
    
    # Group by orientation
    orientation_colors = ['#e41a1c', '#377eb8', '#4daf4a']  # Red, Blue, Green for α=0,1,2
    orientation_names = ['α=0 (→)', 'α=1 (↗)', 'α=2 (↖)']
    
    bond_by_orientation = {0: [], 1: [], 2: []}
    
    for (i, j) in edges:
        alpha = get_bond_orientation(cluster, i, j)
        mag = np.abs(bond_exp[(i, j)])
        bond_by_orientation[alpha].append(((i, j), mag))
    
    # Compute mean magnitude per orientation
    mean_by_orientation = {}
    for alpha in range(3):
        if bond_by_orientation[alpha]:
            mean_by_orientation[alpha] = np.mean([m for _, m in bond_by_orientation[alpha]])
        else:
            mean_by_orientation[alpha] = 0.0
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # =========================================================================
    # Left panel: Bonds colored by orientation
    # =========================================================================
    ax = axes[0]
    
    max_mag = max(np.abs(bond_exp[b]) for b in bond_exp)
    if max_mag < 1e-10:
        max_mag = 1.0
    
    for alpha in range(3):
        for (i, j), mag in bond_by_orientation[alpha]:
            ri, rj = positions[i], positions[j]
            width = 1.0 + 5.0 * (mag / max_mag)
            ax.plot([ri[0], rj[0]], [ri[1], rj[1]], 
                   color=orientation_colors[alpha], linewidth=width, 
                   solid_capstyle='round', alpha=0.8)
    
    # Draw sites
    x_coords = [positions[i][0] for i in range(n_sites)]
    y_coords = [positions[i][1] for i in range(n_sites)]
    ax.scatter(x_coords, y_coords, c='black', s=60, zorder=5)
    
    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], color=orientation_colors[alpha], linewidth=4, 
                              label=f'{orientation_names[alpha]}: ⟨|D|⟩={mean_by_orientation[alpha]:.4f}')
                      for alpha in range(3)]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=10)
    
    ax.set_aspect('equal')
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('y', fontsize=12)
    ax.set_title('Bonds by Orientation (width ∝ |D_bond|)', fontsize=13)
    
    # Adjust limits
    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)
    padding = 0.5
    ax.set_xlim(x_min - padding, x_max + padding)
    ax.set_ylim(y_min - padding, y_max + padding)
    
    # =========================================================================
    # Right panel: Bar chart of mean bond strength by orientation
    # =========================================================================
    ax = axes[1]
    
    orientations = [0, 1, 2]
    means = [mean_by_orientation[alpha] for alpha in orientations]
    
    bars = ax.bar(orientations, means, color=orientation_colors, edgecolor='black', linewidth=1.5)
    ax.set_xticks(orientations)
    ax.set_xticklabels(orientation_names, fontsize=11)
    ax.set_ylabel(r'Mean $|\langle S^+_i S^-_j + h.c. \rangle|$', fontsize=12)
    ax.set_title('Bond Strength by Orientation', fontsize=13)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add values on bars
    for bar, val in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
               f'{val:.4f}', ha='center', va='bottom', fontsize=10)
    
    # Indicate anisotropy
    if max(means) > 0:
        anisotropy = (max(means) - min(means)) / np.mean(means)
        ax.text(0.95, 0.95, f'Anisotropy: {anisotropy:.3f}', 
               transform=ax.transAxes, ha='right', va='top', fontsize=11,
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    if title:
        fig.suptitle(title, fontsize=14, y=1.02)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.savefig(output_path.replace('.png', '.pdf'), bbox_inches='tight')
        print(f"  Saved orientation map to {output_path}")
    
    plt.close()
    
    return {
        'bond_by_orientation': bond_by_orientation,
        'mean_by_orientation': mean_by_orientation,
        'anisotropy': (max(means) - min(means)) / np.mean(means) if np.mean(means) > 0 else 0
    }


def compute_bond_structure_factor(psi: np.ndarray, cluster: Dict, n_q_points: int = 50) -> Dict:
    """
    Compute bond/dimer structure factor.
    
    D_{ij} = S^+_i S^-_j + S^-_i S^+_j  (XY bond operator)
    
    S_D(q) = (1/N_b) Σ_{b,b'} exp(iq·(r_b - r_{b'})) <δD_b δD_{b'}>
    where δD_b = D_b - <D>
    
    Uses discrete allowed k-points for the order parameter calculation.
    """
    edges = cluster.get('edges_nn', [])
    positions = cluster['positions']
    b1, b2 = cluster['b1'], cluster['b2']
    n_bonds = len(edges)
    
    if n_bonds == 0:
        return {'error': 'No bonds found'}
    
    # Compute bond expectations
    bond_exp = compute_all_xy_bonds(psi, cluster)
    
    # Bond center positions
    bond_centers = {}
    for (i, j) in edges:
        bond_centers[(i, j)] = (positions[i] + positions[j]) / 2
    
    # Mean bond value
    D_mean = np.mean([np.real(bond_exp[b]) for b in edges])
    
    # Connected bond correlations δD
    delta_D = {b: bond_exp[b] - D_mean for b in edges}
    
    # =========================================================================
    # Compute at discrete allowed k-points (for order parameter)
    # =========================================================================
    k_points = cluster.get('k_points', None)
    
    if k_points is not None:
        n_k = len(k_points)
        s_d_discrete = np.zeros(n_k, dtype=complex)
        
        for ik, q in enumerate(k_points):
            for b in edges:
                r_b = bond_centers[b]
                for bp in edges:
                    r_bp = bond_centers[bp]
                    phase = np.exp(1j * np.dot(q, r_b - r_bp))
                    s_d_discrete[ik] += delta_D[b] * np.conj(delta_D[bp]) * phase
            s_d_discrete[ik] /= n_bonds
        
        # Find maximum over ALL k-points (including q=0)
        s_d_abs_discrete = np.abs(s_d_discrete)
        max_idx = np.argmax(s_d_abs_discrete)
        s_d_max = s_d_abs_discrete[max_idx]
        q_max = k_points[max_idx]
        q_max_idx = max_idx
        
        # VBS order parameter
        m_vbs = np.sqrt(np.abs(s_d_max) / n_bonds)
        
        print(f"    S_D(q) computed at {n_k} allowed k-points")
        print(f"    Maximum at q = ({q_max[0]:.4f}, {q_max[1]:.4f}), S_D(q) = {s_d_max:.6f}")
    else:
        s_d_discrete = None
        q_max_idx = None
    
    # =========================================================================
    # Also compute on dense grid (for visualization)
    # =========================================================================
    q1_vals = np.linspace(-1, 1, n_q_points)
    q2_vals = np.linspace(-1, 1, n_q_points)
    s_d_2d = np.zeros((n_q_points, n_q_points), dtype=complex)
    
    for i1, q1 in enumerate(q1_vals):
        for i2, q2 in enumerate(q2_vals):
            q = q1 * b1 + q2 * b2
            for b in edges:
                r_b = bond_centers[b]
                for bp in edges:
                    r_bp = bond_centers[bp]
                    phase = np.exp(1j * np.dot(q, r_b - r_bp))
                    s_d_2d[i1, i2] += delta_D[b] * np.conj(delta_D[bp]) * phase
            s_d_2d[i1, i2] /= n_bonds
    
    # If no discrete k-points, find max from dense grid
    if k_points is None:
        s_d_abs = np.abs(s_d_2d)
        max_idx = np.unravel_index(np.argmax(s_d_abs), s_d_abs.shape)
        s_d_max = s_d_abs[max_idx]
        q_max = q1_vals[max_idx[0]] * b1 + q2_vals[max_idx[1]] * b2
        m_vbs = np.sqrt(np.abs(s_d_max) / n_bonds)
        q_max_idx = None
    
    return {
        's_d_2d': s_d_2d,
        'q1_vals': q1_vals,
        'q2_vals': q2_vals,
        's_d_discrete': s_d_discrete,
        'k_points': k_points,
        'm_vbs': m_vbs,
        'q_max': q_max,
        'q_max_idx': q_max_idx,
        's_d_max': s_d_max,
        'D_mean': D_mean,
        'bond_exp': bond_exp
    }


# -----------------------------------------------------------------------------
# D) PLAQUETTE/BOW-TIE RESONANCE ORDER - BFG Native
# -----------------------------------------------------------------------------

def find_triangles(cluster: Dict) -> List[Tuple[int, int, int]]:
    """
    Find all NN triangles in the kagome lattice.
    
    A triangle is 3 sites where all pairs are NN connected.
    
    Returns list of (s1, s2, s3) tuples with s1 < s2 < s3.
    """
    edges_nn = cluster.get('edges_nn', [])
    n_sites = cluster['n_sites']
    
    # Build NN adjacency
    nn_adj = defaultdict(set)
    for (i, j) in edges_nn:
        nn_adj[i].add(j)
        nn_adj[j].add(i)
    
    triangles = []
    
    # For each site, look for triangles where it's the smallest index
    for s1 in range(n_sites):
        neighbors = nn_adj[s1]
        neighbor_list = [n for n in neighbors if n > s1]
        
        # Check all pairs of neighbors > s1
        for i, s2 in enumerate(neighbor_list):
            for s3 in neighbor_list[i+1:]:
                # Check if s2-s3 are also NN
                if s3 in nn_adj[s2]:
                    triangles.append((s1, s2, s3))
    
    return triangles


def find_bowties(cluster: Dict) -> List[Tuple[int, int, int, int, int, np.ndarray]]:
    """
    Find all bow-tie plaquettes in the kagome lattice.
    
    A bow-tie consists of TWO NN triangles sharing exactly ONE vertex (corner).
    This forms a 5-site structure::
    
              s2
             /  \\
           s1----s0----s3     (s0 = shared vertex)
                  \\    /
                    s4
    
    The bow-tie has 5 sites total:
    - 1 shared center vertex (s0)
    - 2 sites from triangle 1 (s1, s2)
    - 2 sites from triangle 2 (s3, s4)
    
    Returns list of (s0, s1, s2, s3, s4, center_position) for each bow-tie,
    where s0 is the shared vertex, (s0, s1, s2) is triangle 1, (s0, s3, s4) is triangle 2.
    """
    positions = cluster['positions']
    
    # Find all triangles
    triangles = find_triangles(cluster)
    
    # Build index: which triangles contain each vertex
    vertex_to_triangles = defaultdict(list)
    for idx, (a, b, c) in enumerate(triangles):
        vertex_to_triangles[a].append(idx)
        vertex_to_triangles[b].append(idx)
        vertex_to_triangles[c].append(idx)
    
    bowties = []
    found_set = set()  # To avoid duplicates
    
    # For each vertex, find all pairs of triangles sharing ONLY that vertex
    for shared_vertex, tri_indices in vertex_to_triangles.items():
        if len(tri_indices) < 2:
            continue
        
        # Check all pairs of triangles at this vertex
        for i in range(len(tri_indices)):
            for j in range(i + 1, len(tri_indices)):
                t1 = triangles[tri_indices[i]]
                t2 = triangles[tri_indices[j]]
                
                # Get the other two vertices of each triangle
                other1 = [v for v in t1 if v != shared_vertex]
                other2 = [v for v in t2 if v != shared_vertex]
                
                # Check that triangles share ONLY the shared_vertex (no edge sharing)
                if set(other1) & set(other2):
                    continue  # They share more than one vertex
                
                # This is a valid bowtie
                s0 = shared_vertex
                s1, s2 = other1[0], other1[1]
                s3, s4 = other2[0], other2[1]
                
                # Create canonical key to avoid duplicates
                # Sort each triangle's other vertices, then sort triangles
                pair1 = tuple(sorted([s1, s2]))
                pair2 = tuple(sorted([s3, s4]))
                if pair1 > pair2:
                    pair1, pair2 = pair2, pair1
                    s1, s2, s3, s4 = s3, s4, s1, s2
                
                key = (s0, pair1, pair2)
                if key in found_set:
                    continue
                found_set.add(key)
                
                # Center position (weighted towards shared vertex)
                all_sites = [s0, s1, s2, s3, s4]
                center = np.mean([positions[s] for s in all_sites], axis=0)
                
                bowties.append((s0, s1, s2, s3, s4, center))
    
    return bowties


def compute_bowtie_resonance(
    psi: np.ndarray,
    n_sites: int,
    s1: int, s2: int, s3: int, s4: int
) -> complex:
    """
    Compute bow-tie ring flip expectation:
    P_r = <S^+_1 S^-_2 S^+_3 S^-_4 + h.c.>
    
    This is the BFG ring-flip operator that resonates between
    |↑↓↑↓⟩ and |↓↑↓↑⟩ configurations around the bow-tie.
    """
    dim = len(psi)
    expectation = 0.0 + 0.0j
    
    for state in range(dim):
        coeff = psi[state]
        if np.abs(coeff) < 1e-15:
            continue
        
        # S^+_1 S^-_2 S^+_3 S^-_4 term
        new_state, factor = state, 1.0
        
        # Apply S^+_1
        result = apply_sp(new_state, s1)
        if result[0] is None:
            pass
        else:
            new_state, c = result
            factor *= c
            
            # Apply S^-_2
            result = apply_sm(new_state, s2)
            if result[0] is None:
                pass
            else:
                new_state, c = result
                factor *= c
                
                # Apply S^+_3
                result = apply_sp(new_state, s3)
                if result[0] is None:
                    pass
                else:
                    new_state, c = result
                    factor *= c
                    
                    # Apply S^-_4
                    result = apply_sm(new_state, s4)
                    if result[0] is not None:
                        new_state, c = result
                        factor *= c
                        expectation += np.conj(psi[new_state]) * coeff * factor
        
        # Hermitian conjugate: S^-_1 S^+_2 S^-_3 S^+_4
        new_state, factor = state, 1.0
        
        # Apply S^-_1
        result = apply_sm(new_state, s1)
        if result[0] is None:
            pass
        else:
            new_state, c = result
            factor *= c
            
            # Apply S^+_2
            result = apply_sp(new_state, s2)
            if result[0] is None:
                pass
            else:
                new_state, c = result
                factor *= c
                
                # Apply S^-_3
                result = apply_sm(new_state, s3)
                if result[0] is None:
                    pass
                else:
                    new_state, c = result
                    factor *= c
                    
                    # Apply S^+_4
                    result = apply_sp(new_state, s4)
                    if result[0] is not None:
                        new_state, c = result
                        factor *= c
                        expectation += np.conj(psi[new_state]) * coeff * factor
    
    return expectation


def compute_triangle_chiral(
    psi: np.ndarray,
    n_sites: int,
    s1: int, s2: int, s3: int
) -> complex:
    """
    Compute triangle chiral/resonance expectation:
    χ = <S^+_1 S^-_2 S^+_3 + S^-_1 S^+_2 S^-_3>
    
    This measures the 3-site ring exchange around a triangle.
    """
    dim = len(psi)
    expectation = 0.0 + 0.0j
    
    for state in range(dim):
        coeff = psi[state]
        if np.abs(coeff) < 1e-15:
            continue
        
        # S^+_1 S^-_2 S^+_3 term
        new_state, factor = state, 1.0
        
        result = apply_sp(new_state, s1)
        if result[0] is not None:
            new_state, c = result
            factor *= c
            
            result = apply_sm(new_state, s2)
            if result[0] is not None:
                new_state, c = result
                factor *= c
                
                result = apply_sp(new_state, s3)
                if result[0] is not None:
                    new_state, c = result
                    factor *= c
                    expectation += np.conj(psi[new_state]) * coeff * factor
        
        # Hermitian conjugate: S^-_1 S^+_2 S^-_3
        new_state, factor = state, 1.0
        
        result = apply_sm(new_state, s1)
        if result[0] is not None:
            new_state, c = result
            factor *= c
            
            result = apply_sp(new_state, s2)
            if result[0] is not None:
                new_state, c = result
                factor *= c
                
                result = apply_sm(new_state, s3)
                if result[0] is not None:
                    new_state, c = result
                    factor *= c
                    expectation += np.conj(psi[new_state]) * coeff * factor
    
    return expectation


def compute_plaquette_order(psi: np.ndarray, cluster: Dict, n_q_points: int = 30) -> Dict:
    """
    Compute plaquette/bow-tie resonance order.
    
    For a 5-site bowtie (2 triangles sharing a vertex at s0)::
    
              s2
             /  \\
           s1----s0----s3
                  \\    /
                    s4
    
    The ring-flip operator acts on the 4 OUTER corners (s1, s2, s3, s4),
    EXCLUDING the shared center vertex (s0):
    
        P_bt = <S^+_{s1} S^-_{s2} S^+_{s3} S^-_{s4} + h.c.>
    
    This flips spins around the "bowtie ring": s1 → s2 → s3 → s4 → s1
    
    S_P(q) = (1/N_bt) Σ_{bt,bt'} exp(iq·(R_bt - R_{bt'})) <δP_bt δP_{bt'}>
    where δP_bt = P_bt - <P>
    
    Uses discrete allowed k-points for the order parameter calculation.
    """
    n_sites = cluster['n_sites']
    b1, b2 = cluster['b1'], cluster['b2']
    
    # Find all bow-ties (5-site: s0, s1, s2, s3, s4, center)
    bowties = find_bowties(cluster)
    n_plaquettes = len(bowties)
    
    if n_plaquettes == 0:
        return {
            'error': 'No bow-tie plaquettes found',
            'n_plaquettes': 0
        }
    
    print(f"  Found {n_plaquettes} bow-tie plaquettes (5-site, 2 triangles sharing corner)")
    print(f"  Ring-flip on 4 outer corners: S^+_s1 S^-_s2 S^+_s3 S^-_s4 + h.c.")
    
    # Compute resonance expectation for each bow-tie
    # 4-spin ring flip on outer corners (s1, s2, s3, s4), excluding center s0
    P_r = {}
    centers = {}
    
    for idx, bowtie in enumerate(bowties):
        s0, s1, s2, s3, s4, center = bowtie
        key = (s0, s1, s2, s3, s4)
        
        # 4-spin ring flip on outer corners: S^+_s1 S^-_s2 S^+_s3 S^-_s4 + h.c.
        P_r[key] = compute_bowtie_resonance(psi, n_sites, s1, s2, s3, s4)
        centers[key] = center
        
        if (idx + 1) % 10 == 0:
            print(f"    Computed {idx + 1}/{n_plaquettes} plaquettes", end='\r')
    print()
    
    # Mean plaquette value
    P_mean = np.mean([np.real(P_r[k]) for k in P_r])
    
    # Connected plaquette correlations
    delta_P = {k: P_r[k] - P_mean for k in P_r}
    plaquette_keys = list(P_r.keys())
    
    # =========================================================================
    # Compute at discrete allowed k-points (for order parameter)
    # =========================================================================
    k_points = cluster.get('k_points', None)
    
    if k_points is not None:
        n_k = len(k_points)
        s_p_discrete = np.zeros(n_k, dtype=complex)
        
        for ik, q in enumerate(k_points):
            for r in plaquette_keys:
                R_r = centers[r]
                for rp in plaquette_keys:
                    R_rp = centers[rp]
                    phase = np.exp(1j * np.dot(q, R_r - R_rp))
                    s_p_discrete[ik] += delta_P[r] * np.conj(delta_P[rp]) * phase
            s_p_discrete[ik] /= n_plaquettes
        
        # Find maximum over ALL k-points (including q=0)
        s_p_abs_discrete = np.abs(s_p_discrete)
        max_idx = np.argmax(s_p_abs_discrete)
        s_p_max = s_p_abs_discrete[max_idx]
        q_max = k_points[max_idx]
        q_max_idx = max_idx
        
        # Plaquette order parameter
        m_plaquette = np.sqrt(np.abs(s_p_max) / n_plaquettes)
        
        # q=0 value for resonance strength
        q0_idx = np.argmin(np.linalg.norm(k_points, axis=1))
        s_p_q0 = s_p_abs_discrete[q0_idx]
        
        print(f"    S_P(q) computed at {n_k} allowed k-points")
        print(f"    Maximum at q = ({q_max[0]:.4f}, {q_max[1]:.4f}), S_P(q) = {s_p_max:.6f}")
    else:
        s_p_discrete = None
        q_max_idx = None
        s_p_q0 = None
    
    # =========================================================================
    # Also compute on dense grid (for visualization)
    # =========================================================================
    q1_vals = np.linspace(-1, 1, n_q_points)
    q2_vals = np.linspace(-1, 1, n_q_points)
    s_p_2d = np.zeros((n_q_points, n_q_points), dtype=complex)
    
    for i1, q1 in enumerate(q1_vals):
        for i2, q2 in enumerate(q2_vals):
            q = q1 * b1 + q2 * b2
            for r in plaquette_keys:
                R_r = centers[r]
                for rp in plaquette_keys:
                    R_rp = centers[rp]
                    phase = np.exp(1j * np.dot(q, R_r - R_rp))
                    s_p_2d[i1, i2] += delta_P[r] * np.conj(delta_P[rp]) * phase
            s_p_2d[i1, i2] /= n_plaquettes
    
    # If no discrete k-points, find max from dense grid
    if k_points is None:
        s_p_abs = np.abs(s_p_2d)
        max_idx = np.unravel_index(np.argmax(s_p_abs), s_p_abs.shape)
        s_p_max = s_p_abs[max_idx]
        q_max = q1_vals[max_idx[0]] * b1 + q2_vals[max_idx[1]] * b2
        m_plaquette = np.sqrt(np.abs(s_p_max) / n_plaquettes)
        center_idx = n_q_points // 2
        s_p_q0 = np.abs(s_p_2d[center_idx, center_idx])
        q_max_idx = None
    
    # Resonance strength (liquid-like)
    resonance_strength = np.mean([np.abs(P_r[k]) for k in P_r])
    
    return {
        's_p_2d': s_p_2d,
        'q1_vals': q1_vals,
        'q2_vals': q2_vals,
        's_p_discrete': s_p_discrete,
        'k_points': k_points,
        'm_plaquette': m_plaquette,
        'q_max': q_max,
        'q_max_idx': q_max_idx,
        's_p_max': s_p_max,
        's_p_q0': s_p_q0,
        'P_mean': P_mean,
        'resonance_strength': resonance_strength,
        'P_r': P_r,
        'n_plaquettes': n_plaquettes
    }


# =============================================================================
# Geometry Visualization Functions
# =============================================================================

def plot_lattice_geometry(cluster: Dict, output_dir: str):
    """
    Visualize kagome lattice geometry to verify setup is correct.
    
    Creates plots showing:
    1. All sites colored by sublattice
    2. NN, 2NN, 3NN bonds in different colors
    3. Unit cell and lattice vectors
    """
    if not HAS_MATPLOTLIB:
        print("Matplotlib not available, skipping geometry plots")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    positions = cluster['positions']
    sublattice = cluster.get('sublattice', {})
    edges_nn = cluster.get('edges_nn', [])
    edges_2nn = cluster.get('edges_2nn', [])
    edges_3nn = cluster.get('edges_3nn', [])
    
    # Sublattice colors
    sublattice_colors = ['#D55E00', '#009E73', '#56B4E9']  # Orange, Green, Blue
    
    # Extract coordinates
    x_coords = np.array([positions[i][0] for i in sorted(positions.keys())])
    y_coords = np.array([positions[i][1] for i in sorted(positions.keys())])
    site_colors = [sublattice_colors[sublattice.get(i, i % 3)] for i in sorted(positions.keys())]
    
    # =========================================================================
    # Plot 1: Full lattice with all bond types
    # =========================================================================
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Plot bonds (back to front: 3NN, 2NN, NN)
    for (i, j) in edges_3nn:
        r_i, r_j = positions[i], positions[j]
        ax.plot([r_i[0], r_j[0]], [r_i[1], r_j[1]], 
                'g-', alpha=0.3, lw=1, zorder=1, label='3NN' if (i,j) == edges_3nn[0] else '')
    
    for (i, j) in edges_2nn:
        r_i, r_j = positions[i], positions[j]
        ax.plot([r_i[0], r_j[0]], [r_i[1], r_j[1]], 
                'orange', alpha=0.5, lw=1.5, zorder=2, label='2NN' if (i,j) == edges_2nn[0] else '')
    
    for (i, j) in edges_nn:
        r_i, r_j = positions[i], positions[j]
        ax.plot([r_i[0], r_j[0]], [r_i[1], r_j[1]], 
                'b-', alpha=0.7, lw=2, zorder=3, label='NN' if (i,j) == edges_nn[0] else '')
    
    # Plot sites
    for i in sorted(positions.keys()):
        r = positions[i]
        color = sublattice_colors[sublattice.get(i, i % 3)]
        ax.scatter(r[0], r[1], c=color, s=150, zorder=5, edgecolors='black', linewidth=1)
        ax.annotate(str(i), (r[0], r[1]), fontsize=8, ha='center', va='center', zorder=6)
    
    # Add lattice vectors
    a1, a2 = cluster['a1'], cluster['a2']
    origin = np.array([np.min(x_coords) - 0.5, np.min(y_coords) - 0.5])
    ax.arrow(origin[0], origin[1], a1[0], a1[1], head_width=0.1, head_length=0.05, 
             fc='red', ec='red', zorder=10)
    ax.arrow(origin[0], origin[1], a2[0], a2[1], head_width=0.1, head_length=0.05, 
             fc='purple', ec='purple', zorder=10)
    ax.text(origin[0] + a1[0]/2, origin[1] + a1[1]/2 - 0.2, 'a₁', color='red', fontsize=12)
    ax.text(origin[0] + a2[0]/2 - 0.2, origin[1] + a2[1]/2, 'a₂', color='purple', fontsize=12)
    
    ax.set_aspect('equal')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(f'Kagome Lattice Geometry\n{cluster["n_sites"]} sites, {len(edges_nn)} NN bonds, {len(edges_2nn)} 2NN bonds')
    
    # Legend
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    legend_elements = [
        Line2D([0], [0], color='b', lw=2, label=f'NN ({len(edges_nn)})'),
        Line2D([0], [0], color='orange', lw=1.5, label=f'2NN ({len(edges_2nn)})'),
        Line2D([0], [0], color='g', lw=1, label=f'3NN ({len(edges_3nn)})'),
        Patch(facecolor=sublattice_colors[0], edgecolor='k', label='Sublattice 0'),
        Patch(facecolor=sublattice_colors[1], edgecolor='k', label='Sublattice 1'),
        Patch(facecolor=sublattice_colors[2], edgecolor='k', label='Sublattice 2'),
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    plt.savefig(os.path.join(output_dir, 'lattice_geometry.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # =========================================================================
    # Plot 2: Bond orientations (for nematic order verification)
    # =========================================================================
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Bond orientation colors
    orient_colors = ['#E69F00', '#56B4E9', '#CC79A7']  # Yellow, Cyan, Pink
    
    for (i, j) in edges_nn:
        r_i, r_j = positions[i], positions[j]
        dr = r_j - r_i
        angle = np.degrees(np.arctan2(dr[1], dr[0])) % 180
        if angle < 30 or angle >= 150:
            color = orient_colors[0]  # ~0°
        elif 30 <= angle < 90:
            color = orient_colors[1]  # ~60°
        else:
            color = orient_colors[2]  # ~120°
        ax.plot([r_i[0], r_j[0]], [r_i[1], r_j[1]], color=color, lw=3, alpha=0.8)
    
    # Plot sites
    ax.scatter(x_coords, y_coords, c='gray', s=80, zorder=5, edgecolors='black')
    
    ax.set_aspect('equal')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('NN Bond Orientations (for Nematic Order)\nα=0 (0°), α=1 (60°), α=2 (120°)')
    
    legend_elements = [
        Line2D([0], [0], color=orient_colors[0], lw=3, label='α=0 (~0°)'),
        Line2D([0], [0], color=orient_colors[1], lw=3, label='α=1 (~60°)'),
        Line2D([0], [0], color=orient_colors[2], lw=3, label='α=2 (~120°)'),
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    plt.savefig(os.path.join(output_dir, 'bond_orientations.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Geometry plots saved to {output_dir}")


def plot_bowties(cluster: Dict, bowties: List, output_dir: str, P_r: Dict = None):
    """
    Visualize bow-tie plaquettes to verify geometry.
    
    Args:
        cluster: Cluster data
        bowties: List of (s0, s1, s2, s3, s4, center) tuples - 5-site bowties
        output_dir: Output directory
        P_r: Optional dict of plaquette expectation values for coloring
    """
    if not HAS_MATPLOTLIB:
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    positions = cluster['positions']
    edges_nn = cluster.get('edges_nn', [])
    edges_2nn = cluster.get('edges_2nn', [])
    
    fig, ax = plt.subplots(figsize=(14, 12))
    
    # Plot all NN bonds (light gray)
    for (i, j) in edges_nn:
        r_i, r_j = positions[i], positions[j]
        ax.plot([r_i[0], r_j[0]], [r_i[1], r_j[1]], 'lightgray', lw=1, zorder=1)
    
    # Plot all 2NN bonds (very light)
    for (i, j) in edges_2nn:
        r_i, r_j = positions[i], positions[j]
        ax.plot([r_i[0], r_j[0]], [r_i[1], r_j[1]], color='#f0f0f0', lw=0.5, zorder=0)
    
    # Color scale for plaquette values
    if P_r is not None:
        P_vals = np.array([np.real(P_r[k]) for k in P_r])
        vmin, vmax = np.min(P_vals), np.max(P_vals)
        norm = plt.Normalize(vmin=vmin, vmax=vmax)
        cmap = plt.cm.RdBu_r
    
    # Plot each bow-tie (5-site: 2 triangles sharing a corner)
    for idx, bowtie in enumerate(bowties):
        s0, s1, s2, s3, s4, center = bowtie
        
        # Get color from P_r if available
        if P_r is not None:
            key = (s0, s1, s2, s3, s4)
            if key in P_r:
                color = cmap(norm(np.real(P_r[key])))
                alpha = 0.6
            else:
                color = 'cyan'
                alpha = 0.3
        else:
            color = plt.cm.tab20(idx % 20)
            alpha = 0.4
        
        # Draw the two triangles
        from matplotlib.patches import Polygon
        
        # Triangle 1: (s0, s1, s2)
        tri1_coords = [positions[s0], positions[s1], positions[s2]]
        polygon1 = Polygon(tri1_coords, closed=True, facecolor=color, 
                          edgecolor='blue', alpha=alpha, lw=1.5, zorder=2)
        ax.add_patch(polygon1)
        
        # Triangle 2: (s0, s3, s4)
        tri2_coords = [positions[s0], positions[s3], positions[s4]]
        polygon2 = Polygon(tri2_coords, closed=True, facecolor=color, 
                          edgecolor='blue', alpha=alpha, lw=1.5, zorder=2)
        ax.add_patch(polygon2)
        
        # Mark shared vertex
        ax.scatter(positions[s0][0], positions[s0][1], c='red', s=40, marker='o', zorder=7)
    
    # Plot sites on top
    x_coords = np.array([positions[i][0] for i in sorted(positions.keys())])
    y_coords = np.array([positions[i][1] for i in sorted(positions.keys())])
    ax.scatter(x_coords, y_coords, c='white', s=100, zorder=5, edgecolors='black', linewidth=1.5)
    
    # Add site labels
    for i in sorted(positions.keys()):
        r = positions[i]
        ax.annotate(str(i), (r[0], r[1]), fontsize=7, ha='center', va='center', zorder=6)
    
    ax.set_aspect('equal')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    
    title = f'Bow-tie Plaquettes ({len(bowties)} total)\n(2 triangles sharing 1 corner)'
    if P_r is not None:
        title += '\nColored by triangle chirality correlation'
        # Add colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, shrink=0.6)
        cbar.set_label('P_bt (bowtie order)')
    
    ax.set_title(title)
    
    plt.savefig(os.path.join(output_dir, 'bowtie_plaquettes.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # =========================================================================
    # BLOW-UP FIGURE: Detailed individual bowtie visualization
    # =========================================================================
    plot_bowties_blowup(cluster, bowties, output_dir, P_r)


def plot_bowties_blowup(cluster: Dict, bowties: List, output_dir: str, P_r: Dict = None):
    """
    Create a blow-up visualization showing individual bowties in detail.
    
    Shows:
    1. A schematic of the bowtie structure (2 triangles sharing a corner)
    2. Statistics about the bowties
    3. A grid of individual bowties with their connectivity
    """
    if not HAS_MATPLOTLIB:
        return
    
    from matplotlib.patches import Polygon, FancyArrowPatch, Circle
    from matplotlib.lines import Line2D
    
    positions = cluster['positions']
    edges_nn = cluster.get('edges_nn', [])
    edges_2nn = cluster.get('edges_2nn', [])
    n_sites = cluster['n_sites']
    
    # Build NN set for fast lookup
    nn_set = set((min(i, j), max(i, j)) for (i, j) in edges_nn)
    
    def is_nn(a, b):
        return (min(a, b), max(a, b)) in nn_set
    
    # Limit number of bowties to show (for large clusters)
    max_show = min(24, len(bowties))
    n_cols = min(6, max_show)
    n_rows = (max_show + n_cols - 1) // n_cols
    
    # Create figure with multiple panels
    fig = plt.figure(figsize=(20, 4 * n_rows + 4))
    
    # Add top panel for schematic
    gs = fig.add_gridspec(n_rows + 1, n_cols, height_ratios=[1.5] + [1]*n_rows,
                          hspace=0.3, wspace=0.25)
    
    # =========================================================================
    # Top panel: Schematic of 5-site bowtie structure
    # =========================================================================
    ax_schematic = fig.add_subplot(gs[0, :3])
    
    # Draw schematic 5-site bowtie (2 triangles sharing center vertex)
    schematic_pos = {
        0: np.array([1.0, 0.0]),    # Shared center vertex (s0)
        1: np.array([0.0, 0.5]),    # Triangle 1 vertex (s1)
        2: np.array([0.0, -0.5]),   # Triangle 1 vertex (s2)
        3: np.array([2.0, 0.5]),    # Triangle 2 vertex (s3)
        4: np.array([2.0, -0.5]),   # Triangle 2 vertex (s4)
    }
    
    # Draw triangle 1: (s0, s1, s2) - all NN bonds
    tri1_sites = [0, 1, 2]
    for i in range(3):
        si, sj = tri1_sites[i], tri1_sites[(i+1) % 3]
        ax_schematic.plot([schematic_pos[si][0], schematic_pos[sj][0]], 
                          [schematic_pos[si][1], schematic_pos[sj][1]], 
                          'b-', lw=4)
    
    # Draw triangle 2: (s0, s3, s4) - all NN bonds
    tri2_sites = [0, 3, 4]
    for i in range(3):
        si, sj = tri2_sites[i], tri2_sites[(i+1) % 3]
        ax_schematic.plot([schematic_pos[si][0], schematic_pos[sj][0]], 
                          [schematic_pos[si][1], schematic_pos[sj][1]], 
                          'b-', lw=4)
    
    # Fill triangles
    from matplotlib.patches import Polygon
    tri1_coords = [schematic_pos[s] for s in tri1_sites]
    tri2_coords = [schematic_pos[s] for s in tri2_sites]
    polygon1 = Polygon(tri1_coords, closed=True, facecolor='lightblue', alpha=0.5, zorder=1)
    polygon2 = Polygon(tri2_coords, closed=True, facecolor='lightgreen', alpha=0.5, zorder=1)
    ax_schematic.add_patch(polygon1)
    ax_schematic.add_patch(polygon2)
    
    # Draw sites
    labels = ['s0\n(shared)', 's1', 's2', 's3', 's4']
    colors = ['red', 'white', 'white', 'white', 'white']
    for idx, pos in schematic_pos.items():
        circle = Circle(pos, 0.12, facecolor=colors[idx], edgecolor='black', lw=2, zorder=5)
        ax_schematic.add_patch(circle)
        offset = np.array([0, 0.25]) if idx == 0 else np.array([0, 0])
        ax_schematic.annotate(labels[idx], pos + offset, fontsize=10, ha='center', va='center', 
                             fontweight='bold', zorder=6)
    
    # Labels
    ax_schematic.text(0.3, 0.0, 'Triangle 1\n(s0,s1,s2)', fontsize=9, ha='center', color='blue')
    ax_schematic.text(1.7, 0.0, 'Triangle 2\n(s0,s3,s4)', fontsize=9, ha='center', color='green')
    
    ax_schematic.set_xlim(-0.5, 2.5)
    ax_schematic.set_ylim(-1, 1)
    ax_schematic.set_aspect('equal')
    ax_schematic.set_title('Bowtie Structure: 2 NN-Triangles Sharing 1 Corner\n'
                           'Ring-flip on 4 OUTER corners: $S^+_{s1} S^-_{s2} S^+_{s3} S^-_{s4} + h.c.$\n'
                           '(center vertex s0 is EXCLUDED)', 
                           fontsize=11, fontweight='bold')
    ax_schematic.axis('off')
    
    # Legend
    legend_elements = [
        Line2D([0], [0], color='blue', lw=3, label='NN bond'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, 
               label='Shared vertex (excluded)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='white', 
               markeredgecolor='black', markersize=10, label='Outer corners (ring-flip)'),
    ]
    ax_schematic.legend(handles=legend_elements, loc='upper right')
    
    # =========================================================================
    # Statistics panel
    # =========================================================================
    ax_stats = fig.add_subplot(gs[0, 3:])
    ax_stats.axis('off')
    
    # Count bowties by site participation
    site_count = defaultdict(int)
    shared_count = defaultdict(int)
    for bowtie in bowties:
        s0, s1, s2, s3, s4, center = bowtie
        shared_count[s0] += 1
        for s in [s0, s1, s2, s3, s4]:
            site_count[s] += 1
    
    # Count triangles
    triangles = find_triangles(cluster)
    
    stats_text = [
        f"Total bowties: {len(bowties)}",
        f"Total NN triangles: {len(triangles)}",
        f"Sites in cluster: {n_sites}",
        f"NN bonds: {len(edges_nn)}",
        f"",
        f"Bowties per site (any role):",
        f"  Min: {min(site_count.values()) if site_count else 0}",
        f"  Max: {max(site_count.values()) if site_count else 0}",
        f"  Avg: {np.mean(list(site_count.values())):.1f}" if site_count else "",
        f"",
        f"Bowties per shared vertex:",
        f"  Min: {min(shared_count.values()) if shared_count else 0}",
        f"  Max: {max(shared_count.values()) if shared_count else 0}",
    ]
    
    ax_stats.text(0.1, 0.95, '\n'.join(stats_text), transform=ax_stats.transAxes,
                  fontsize=11, verticalalignment='top', fontfamily='monospace',
                  bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax_stats.set_title('Statistics', fontsize=12, fontweight='bold')
    
    # =========================================================================
    # Individual bowtie panels
    # =========================================================================
    for idx in range(max_show):
        row = idx // n_cols + 1  # +1 because row 0 is schematic
        col = idx % n_cols
        ax = fig.add_subplot(gs[row, col])
        
        s0, s1, s2, s3, s4, center = bowties[idx]
        sites = [s0, s1, s2, s3, s4]
        
        # Get positions and center them
        bowtie_pos = {s: positions[s].copy() for s in sites}
        centroid = np.mean([bowtie_pos[s] for s in sites], axis=0)
        for s in sites:
            bowtie_pos[s] = bowtie_pos[s] - centroid
        
        # Draw triangle 1: (s0, s1, s2)
        tri1 = [s0, s1, s2]
        for i in range(3):
            si, sj = tri1[i], tri1[(i+1) % 3]
            pi, pj = bowtie_pos[si], bowtie_pos[sj]
            ax.plot([pi[0], pj[0]], [pi[1], pj[1]], 'b-', lw=3, zorder=1)
        
        # Draw triangle 2: (s0, s3, s4)
        tri2 = [s0, s3, s4]
        for i in range(3):
            si, sj = tri2[i], tri2[(i+1) % 3]
            pi, pj = bowtie_pos[si], bowtie_pos[sj]
            ax.plot([pi[0], pj[0]], [pi[1], pj[1]], 'b-', lw=3, zorder=1)
        
        # Fill triangles
        tri1_coords = [bowtie_pos[s] for s in tri1]
        tri2_coords = [bowtie_pos[s] for s in tri2]
        polygon1 = Polygon(tri1_coords, closed=True, facecolor='lightblue', alpha=0.4, zorder=0)
        polygon2 = Polygon(tri2_coords, closed=True, facecolor='lightgreen', alpha=0.4, zorder=0)
        ax.add_patch(polygon1)
        ax.add_patch(polygon2)
        
        # Draw sites
        for s in sites:
            p = bowtie_pos[s]
            color = 'red' if s == s0 else 'white'
            ax.scatter(p[0], p[1], c=color, s=400, zorder=5, 
                      edgecolors='black', linewidth=2)
            ax.annotate(str(s), p, fontsize=9, ha='center', va='center', 
                       fontweight='bold', zorder=6)
        
        ax.set_aspect('equal')
        # Auto-scale based on positions
        all_x = [bowtie_pos[s][0] for s in sites]
        all_y = [bowtie_pos[s][1] for s in sites]
        margin = 0.3
        ax.set_xlim(min(all_x) - margin, max(all_x) + margin)
        ax.set_ylim(min(all_y) - margin, max(all_y) + margin)
        ax.axis('off')
        ax.set_title(f'#{idx}: center={s0}', fontsize=9)
    
    plt.suptitle(f'Bowtie Plaquettes Blow-up View\n'
                 f'Showing {max_show} of {len(bowties)} total bowties (5-site, 2 triangles sharing corner)', 
                 fontsize=14, fontweight='bold', y=1.02)
    
    plt.savefig(os.path.join(output_dir, 'bowtie_plaquettes_blowup.png'), 
                dpi=150, bbox_inches='tight')
    plt.close()


# =============================================================================
# Spatially-Resolved Order Parameter Visualization
# =============================================================================

def plot_local_sz(cluster: Dict, sz_exp: np.ndarray, output_dir: str):
    """
    Plot local ⟨S^z_i⟩ expectation values on the lattice.
    """
    if not HAS_MATPLOTLIB:
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    positions = cluster['positions']
    n_sites = cluster['n_sites']
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Extract coordinates
    x_coords = np.array([positions[i][0] for i in range(n_sites)])
    y_coords = np.array([positions[i][1] for i in range(n_sites)])
    
    # Color by Sz value
    vmax = max(abs(np.min(sz_exp)), abs(np.max(sz_exp)))
    if vmax < 1e-10:
        vmax = 0.5
    
    scatter = ax.scatter(x_coords, y_coords, c=sz_exp, s=300, cmap='RdBu_r', 
                         vmin=-vmax, vmax=vmax, edgecolors='black', linewidth=1, zorder=5)
    
    # Add site labels
    for i in range(n_sites):
        ax.annotate(f'{sz_exp[i]:.2f}', (x_coords[i], y_coords[i]), 
                   fontsize=6, ha='center', va='center', zorder=6,
                   color='white' if abs(sz_exp[i]) > vmax*0.5 else 'black')
    
    ax.set_aspect('equal')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(f'Local ⟨S$^z_i$⟩ Expectation Values\nMean = {np.mean(sz_exp):.4f}, Std = {np.std(sz_exp):.4f}')
    
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.7)
    cbar.set_label('⟨S$^z$⟩')
    
    plt.savefig(os.path.join(output_dir, 'local_Sz.png'), dpi=150, bbox_inches='tight')
    plt.close()


def plot_local_bonds(cluster: Dict, bond_exp: Dict, output_dir: str):
    """
    Plot local XY bond expectation values ⟨S⁺_i S⁻_j + h.c.⟩ on bonds.
    """
    if not HAS_MATPLOTLIB:
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    positions = cluster['positions']
    edges = cluster.get('edges_nn', [])
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Get bond values
    bond_vals = np.array([np.real(bond_exp[b]) for b in edges])
    vmax = max(abs(np.min(bond_vals)), abs(np.max(bond_vals)))
    if vmax < 1e-10:
        vmax = 1.0
    
    norm = plt.Normalize(vmin=-vmax, vmax=vmax)
    cmap = plt.cm.RdBu_r
    
    # Plot bonds colored by value
    for idx, (i, j) in enumerate(edges):
        r_i, r_j = positions[i], positions[j]
        color = cmap(norm(bond_vals[idx]))
        ax.plot([r_i[0], r_j[0]], [r_i[1], r_j[1]], color=color, lw=4, zorder=2, solid_capstyle='round')
    
    # Plot sites
    x_coords = np.array([positions[i][0] for i in sorted(positions.keys())])
    y_coords = np.array([positions[i][1] for i in sorted(positions.keys())])
    ax.scatter(x_coords, y_coords, c='white', s=80, zorder=5, edgecolors='black', linewidth=1)
    
    ax.set_aspect('equal')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(f'Local XY Bond Values ⟨S⁺_i S⁻_j + h.c.⟩\nMean = {np.mean(bond_vals):.4f}, Std = {np.std(bond_vals):.4f}')
    
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.7)
    cbar.set_label('XY bond value')
    
    plt.savefig(os.path.join(output_dir, 'local_XY_bonds.png'), dpi=150, bbox_inches='tight')
    plt.close()


def plot_bond_orientation_histogram(cluster: Dict, bond_exp: Dict, output_dir: str):
    """
    Plot histogram of bond values grouped by orientation.
    """
    if not HAS_MATPLOTLIB:
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    positions = cluster['positions']
    edges = cluster.get('edges_nn', [])
    
    # Classify bonds by orientation
    bonds_by_orient = {0: [], 1: [], 2: []}
    orient_colors = ['#E69F00', '#56B4E9', '#CC79A7']
    
    for (i, j) in edges:
        r_i, r_j = positions[i], positions[j]
        dr = r_j - r_i
        angle = np.degrees(np.arctan2(dr[1], dr[0])) % 180
        if angle < 30 or angle >= 150:
            alpha = 0
        elif 30 <= angle < 90:
            alpha = 1
        else:
            alpha = 2
        bonds_by_orient[alpha].append(np.real(bond_exp[(i, j)]))
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for alpha in range(3):
        ax = axes[alpha]
        vals = bonds_by_orient[alpha]
        if len(vals) > 0:
            ax.hist(vals, bins=20, color=orient_colors[alpha], edgecolor='black', alpha=0.7)
            ax.axvline(np.mean(vals), color='red', linestyle='--', lw=2, 
                      label=f'Mean = {np.mean(vals):.4f}')
            ax.set_xlabel('XY bond value')
            ax.set_ylabel('Count')
            ax.set_title(f'Orientation α={alpha}\n({len(vals)} bonds)')
            ax.legend()
    
    plt.suptitle('XY Bond Distribution by Orientation', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'bond_orientation_histogram.png'), dpi=150, bbox_inches='tight')
    plt.close()


def plot_szsz_correlation_map(cluster: Dict, szsz: np.ndarray, sz_exp: np.ndarray, output_dir: str):
    """
    Plot spatial correlation ⟨S^z_0 S^z_j⟩_c as function of distance from reference site.
    """
    if not HAS_MATPLOTLIB:
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    positions = cluster['positions']
    n_sites = cluster['n_sites']
    
    # Connected correlation
    szsz_connected = szsz - np.outer(sz_exp, sz_exp)
    
    # Reference site = 0
    ref_site = 0
    r_ref = positions[ref_site]
    
    distances = []
    correlations = []
    
    for j in range(n_sites):
        r_j = positions[j]
        dist = np.linalg.norm(r_j - r_ref)
        distances.append(dist)
        correlations.append(szsz_connected[ref_site, j])
    
    distances = np.array(distances)
    correlations = np.array(correlations)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left: Correlation vs distance
    ax = axes[0]
    ax.scatter(distances, correlations, c='blue', alpha=0.7, s=50)
    ax.axhline(0, color='gray', linestyle='--', lw=1)
    ax.set_xlabel('Distance from site 0')
    ax.set_ylabel('⟨S$^z_0$ S$^z_j$⟩$_c$')
    ax.set_title('Connected Sz-Sz Correlation vs Distance')
    ax.grid(True, alpha=0.3)
    
    # Right: Spatial map
    ax = axes[1]
    
    vmax = max(abs(np.min(correlations)), abs(np.max(correlations)))
    if vmax < 1e-10:
        vmax = 0.1
    
    x_coords = np.array([positions[i][0] for i in range(n_sites)])
    y_coords = np.array([positions[i][1] for i in range(n_sites)])
    
    scatter = ax.scatter(x_coords, y_coords, c=correlations, s=200, cmap='RdBu_r',
                         vmin=-vmax, vmax=vmax, edgecolors='black', linewidth=1)
    
    # Mark reference site
    ax.scatter(r_ref[0], r_ref[1], c='yellow', s=400, marker='*', edgecolors='black', 
               linewidth=2, zorder=10, label='Ref site')
    
    ax.set_aspect('equal')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Spatial Map of ⟨S$^z_0$ S$^z_j$⟩$_c$')
    ax.legend()
    
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.7)
    cbar.set_label('Correlation')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'szsz_correlation_map.png'), dpi=150, bbox_inches='tight')
    plt.close()


def plot_all_spatial_order_parameters(results: Dict, cluster: Dict, output_dir: str):
    """
    Generate all spatially-resolved order parameter plots.
    """
    if not HAS_MATPLOTLIB:
        print("Matplotlib not available, skipping spatial plots")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Local Sz
    if 'translation' in results and 'sz_exp' in results['translation']:
        print("  Plotting local ⟨S^z⟩...")
        plot_local_sz(cluster, results['translation']['sz_exp'], output_dir)
        
        # SzSz correlation map
        if 'szsz' in results['translation']:
            print("  Plotting Sz-Sz correlation map...")
            plot_szsz_correlation_map(cluster, results['translation']['szsz'],
                                      results['translation']['sz_exp'], output_dir)
    
    # 2. Local bond values
    if 'bond' in results and 'bond_exp' in results['bond']:
        print("  Plotting local XY bond values...")
        plot_local_bonds(cluster, results['bond']['bond_exp'], output_dir)
        plot_bond_orientation_histogram(cluster, results['bond']['bond_exp'], output_dir)
    
    # 3. Bow-tie plaquettes with resonance values
    if 'plaquette' in results and 'P_r' in results['plaquette']:
        print("  Plotting bow-tie plaquettes with resonance values...")
        bowties = find_bowties(cluster)
        plot_bowties(cluster, bowties, output_dir, results['plaquette']['P_r'])
    
    # Generate the comprehensive state visualization
    print("  Generating comprehensive state visualization...")
    plot_state_visualization(results, cluster, output_dir)
    
    print(f"Spatial plots saved to {output_dir}")


def plot_state_visualization(results: Dict, cluster: Dict, output_dir: str):
    """
    Generate a comprehensive visualization of the quantum state showing
    all order parameters in a single, intuitive figure.
    
    This creates a multi-panel figure showing:
    1. Local magnetization ⟨S^z_i⟩ on sites
    2. Bond order (XY correlations) on NN bonds
    3. Bowtie resonance values
    4. Structure factors in k-space
    5. Order parameter summary with phase interpretation
    """
    if not HAS_MATPLOTLIB:
        return
    
    from matplotlib.patches import Patch, Circle
    from matplotlib.lines import Line2D
    from matplotlib.gridspec import GridSpec
    import matplotlib.colors as mcolors
    
    os.makedirs(output_dir, exist_ok=True)
    
    positions = cluster['positions']
    n_sites = cluster['n_sites']
    edges_nn = cluster.get('edges_nn', [])
    
    # Extract coordinates
    x_coords = np.array([positions[i][0] for i in range(n_sites)])
    y_coords = np.array([positions[i][1] for i in range(n_sites)])
    
    # =========================================================================
    # Create the main figure with GridSpec layout
    # =========================================================================
    fig = plt.figure(figsize=(20, 16))
    gs = GridSpec(3, 4, figure=fig, hspace=0.3, wspace=0.3,
                  height_ratios=[1.2, 1, 1])
    
    # =========================================================================
    # Panel 1: Local ⟨S^z_i⟩ and bond order combined (top left, spans 2 cols)
    # =========================================================================
    ax1 = fig.add_subplot(gs[0, :2])
    
    # Get data
    if 'translation' in results and 'sz_exp' in results['translation']:
        sz_exp = results['translation']['sz_exp']
    else:
        sz_exp = np.zeros(n_sites)
    
    if 'bond' in results and 'bond_exp' in results['bond']:
        bond_exp = results['bond']['bond_exp']
    else:
        bond_exp = {}
    
    # Plot bonds first (background)
    if bond_exp:
        bond_vals = np.array([np.real(bond_exp.get(b, 0)) for b in edges_nn])
        bond_vmax = max(abs(np.min(bond_vals)), abs(np.max(bond_vals)), 0.1)
        bond_norm = plt.Normalize(vmin=-bond_vmax, vmax=bond_vmax)
        bond_cmap = plt.cm.PRGn
        
        for idx, (i, j) in enumerate(edges_nn):
            r_i, r_j = positions[i], positions[j]
            color = bond_cmap(bond_norm(bond_vals[idx]))
            lw = 2 + 4 * abs(bond_vals[idx]) / bond_vmax  # Thicker = stronger
            ax1.plot([r_i[0], r_j[0]], [r_i[1], r_j[1]], 
                     color=color, lw=lw, zorder=2, solid_capstyle='round')
    
    # Plot sites colored by Sz
    sz_vmax = max(abs(np.min(sz_exp)), abs(np.max(sz_exp)), 0.1)
    scatter = ax1.scatter(x_coords, y_coords, c=sz_exp, s=350, cmap='RdBu_r',
                          vmin=-sz_vmax, vmax=sz_vmax, edgecolors='black', 
                          linewidth=1.5, zorder=5)
    
    # Add site labels
    for i in range(n_sites):
        val = sz_exp[i]
        text_color = 'white' if abs(val) > sz_vmax * 0.4 else 'black'
        ax1.annotate(f'{val:.2f}', (x_coords[i], y_coords[i]),
                     fontsize=6, ha='center', va='center', zorder=6, color=text_color)
    
    ax1.set_aspect('equal')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_title('Real-Space State Visualization\n'
                  f'Sites: ⟨S$^z$⟩ (color), Bonds: XY order (width & color)',
                  fontsize=12, fontweight='bold')
    
    cbar1 = plt.colorbar(scatter, ax=ax1, shrink=0.6, pad=0.02)
    cbar1.set_label('⟨S$^z_i$⟩', fontsize=10)
    
    # =========================================================================
    # Panel 2: Bowtie resonance map (top right, spans 2 cols)
    # =========================================================================
    ax2 = fig.add_subplot(gs[0, 2:])
    
    if 'plaquette' in results and 'P_r' in results['plaquette']:
        P_r = results['plaquette']['P_r']
        bowties = find_bowties(cluster)
        
        # Plot lattice in gray
        for (i, j) in edges_nn:
            r_i, r_j = positions[i], positions[j]
            ax2.plot([r_i[0], r_j[0]], [r_i[1], r_j[1]], 
                     'gray', lw=1, alpha=0.3, zorder=1)
        
        # Get resonance values
        P_vals = []
        centers = []
        for bt in bowties:
            s0, s1, s2, s3, s4, center = bt
            key = (s0, s1, s2, s3, s4)
            if key in P_r:
                P_vals.append(np.real(P_r[key]))
                centers.append(center)
        
        if P_vals:
            P_vals = np.array(P_vals)
            centers = np.array(centers)
            
            P_vmax = max(abs(np.min(P_vals)), abs(np.max(P_vals)), 0.1)
            
            scatter2 = ax2.scatter(centers[:, 0], centers[:, 1], c=P_vals, s=400,
                                   cmap='coolwarm', vmin=-P_vmax, vmax=P_vmax,
                                   edgecolors='black', linewidth=1, marker='h', zorder=5)
            
            cbar2 = plt.colorbar(scatter2, ax=ax2, shrink=0.6, pad=0.02)
            cbar2.set_label('P$_{bt}$ resonance', fontsize=10)
            
            # Add mean resonance info
            ax2.text(0.02, 0.98, f'⟨|P|⟩ = {np.mean(np.abs(P_vals)):.4f}\nσ(P) = {np.std(P_vals):.4f}',
                     transform=ax2.transAxes, fontsize=10, va='top',
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    else:
        ax2.text(0.5, 0.5, 'Plaquette data not available', transform=ax2.transAxes,
                 ha='center', va='center', fontsize=14)
    
    ax2.scatter(x_coords, y_coords, c='lightgray', s=30, zorder=4, edgecolors='gray')
    ax2.set_aspect('equal')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_title('Bowtie Resonance Map\n'
                  f'P$_{{bt}}$ = ⟨S$^+_{{s1}}$ S$^-_{{s2}}$ S$^+_{{s3}}$ S$^-_{{s4}}$ + h.c.⟩',
                  fontsize=12, fontweight='bold')
    
    # =========================================================================
    # Panel 3: S^zz(q) structure factor (middle left)
    # =========================================================================
    ax3 = fig.add_subplot(gs[1, 0])
    
    if 'translation' in results and 's_q_2d' in results['translation']:
        data = results['translation']
        extent = [data['q1_vals'][0], data['q1_vals'][-1],
                  data['q2_vals'][0], data['q2_vals'][-1]]
        im3 = ax3.imshow(np.abs(data['s_q_2d']).T, origin='lower', extent=extent,
                         cmap='viridis', aspect='equal')
        ax3.set_xlabel('q₁ (b₁)')
        ax3.set_ylabel('q₂ (b₂)')
        ax3.set_title(f'S$^{{zz}}$(q)\nm$_{{trans}}$ = {data["m_trans"]:.4f}', fontsize=10)
        plt.colorbar(im3, ax=ax3, shrink=0.8)
    
    # =========================================================================
    # Panel 4: S_D(q) bond structure factor (middle center-left)
    # =========================================================================
    ax4 = fig.add_subplot(gs[1, 1])
    
    if 'bond' in results and 's_d_2d' in results['bond']:
        data = results['bond']
        extent = [data['q1_vals'][0], data['q1_vals'][-1],
                  data['q2_vals'][0], data['q2_vals'][-1]]
        im4 = ax4.imshow(np.abs(data['s_d_2d']).T, origin='lower', extent=extent,
                         cmap='plasma', aspect='equal')
        ax4.set_xlabel('q₁ (b₁)')
        ax4.set_ylabel('q₂ (b₂)')
        ax4.set_title(f'S$_D$(q) - VBS\nm$_{{vbs}}$ = {data["m_vbs"]:.4f}', fontsize=10)
        plt.colorbar(im4, ax=ax4, shrink=0.8)
    
    # =========================================================================
    # Panel 5: S_P(q) plaquette structure factor (middle center-right)
    # =========================================================================
    ax5 = fig.add_subplot(gs[1, 2])
    
    if 'plaquette' in results and 's_p_2d' in results['plaquette']:
        data = results['plaquette']
        extent = [data['q1_vals'][0], data['q1_vals'][-1],
                  data['q2_vals'][0], data['q2_vals'][-1]]
        im5 = ax5.imshow(np.abs(data['s_p_2d']).T, origin='lower', extent=extent,
                         cmap='inferno', aspect='equal')
        ax5.set_xlabel('q₁ (b₁)')
        ax5.set_ylabel('q₂ (b₂)')
        ax5.set_title(f'S$_P$(q) - Plaquette\nm$_{{plaq}}$ = {data["m_plaquette"]:.4f}', fontsize=10)
        plt.colorbar(im5, ax=ax5, shrink=0.8)
    
    # =========================================================================
    # Panel 6: Order parameter summary bar chart (middle right)
    # =========================================================================
    ax6 = fig.add_subplot(gs[1, 3])
    
    # Collect order parameters
    op_names = []
    op_values = []
    op_colors = []
    
    if 'translation' in results:
        op_names.append('m_trans')
        op_values.append(results['translation']['m_trans'])
        op_colors.append('#2ecc71')  # Green
    
    if 'nematic' in results:
        op_names.append('m_nem')
        op_values.append(results['nematic']['m_nem'])
        op_colors.append('#9b59b6')  # Purple
    
    if 'stripe' in results:
        op_names.append('m_stripe')
        op_values.append(results['stripe']['m_stripe'])
        op_colors.append('#e74c3c')  # Red
    
    if 'bond' in results and 'm_vbs' in results['bond']:
        op_names.append('m_vbs')
        op_values.append(results['bond']['m_vbs'])
        op_colors.append('#3498db')  # Blue
    
    if 'plaquette' in results and 'm_plaquette' in results['plaquette']:
        op_names.append('m_plaq')
        op_values.append(results['plaquette']['m_plaquette'])
        op_colors.append('#f39c12')  # Orange
        
        op_names.append('⟨|P|⟩')
        op_values.append(results['plaquette']['resonance_strength'])
        op_colors.append('#e67e22')  # Dark orange
    
    if op_values:
        bars = ax6.barh(op_names, op_values, color=op_colors, edgecolor='black')
        ax6.set_xlabel('Order Parameter Value')
        ax6.set_title('Order Parameter Summary', fontsize=10, fontweight='bold')
        ax6.axvline(0.1, color='red', linestyle='--', alpha=0.5, label='Threshold (0.1)')
        ax6.set_xlim(0, max(op_values) * 1.2 + 0.05)
        
        # Add value labels
        for bar, val in zip(bars, op_values):
            ax6.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                     f'{val:.4f}', va='center', fontsize=9)
    
    # =========================================================================
    # Panel 7: Phase interpretation (bottom, spans all cols)
    # =========================================================================
    ax7 = fig.add_subplot(gs[2, :])
    ax7.axis('off')
    
    # Determine dominant phase
    phase_scores = {}
    
    if 'translation' in results:
        phase_scores['MAGNETIC ORDER\n(AFM/FM/Spiral)'] = results['translation']['m_trans']
    if 'nematic' in results:
        phase_scores['NEMATIC\n(Bond-oriented liquid)'] = results['nematic']['m_nem']
    if 'bond' in results and 'm_vbs' in results['bond']:
        phase_scores['VBS\n(Valence bond solid)'] = results['bond']['m_vbs']
    if 'plaquette' in results and 'm_plaquette' in results['plaquette']:
        phase_scores['PLAQUETTE CRYSTAL\n(Ordered bowties)'] = results['plaquette']['m_plaquette']
        
        # Check for spin liquid signature
        m_plaq = results['plaquette']['m_plaquette']
        res_strength = results['plaquette']['resonance_strength']
        if res_strength > 0.05 and m_plaq < 0.1:
            phase_scores['SPIN LIQUID\n(Resonating phase)'] = res_strength
    
    # Sort by score
    sorted_phases = sorted(phase_scores.items(), key=lambda x: x[1], reverse=True)
    
    # Create interpretation text
    interpretation = "═" * 90 + "\n"
    interpretation += "                              PHASE INTERPRETATION\n"
    interpretation += "═" * 90 + "\n\n"
    
    if sorted_phases:
        top_phase, top_score = sorted_phases[0]
        
        if top_score > 0.1:
            interpretation += f"  DOMINANT PHASE:  {top_phase.replace(chr(10), ' ')}  (score = {top_score:.4f})\n\n"
        else:
            interpretation += f"  STATE:  QUANTUM DISORDERED / PARAMAGNETIC  (all order parameters < 0.1)\n\n"
        
        interpretation += "  Order Parameter Ranking:\n"
        for i, (phase, score) in enumerate(sorted_phases):
            status = "●" if score > 0.1 else "○"
            interpretation += f"    {i+1}. {status} {phase.replace(chr(10), ' ')}: {score:.4f}\n"
    
    interpretation += "\n" + "─" * 90 + "\n"
    interpretation += "  Legend:  ● = significant (> 0.1)    ○ = weak (< 0.1)\n"
    interpretation += "─" * 90
    
    ax7.text(0.02, 0.95, interpretation, transform=ax7.transAxes,
             fontsize=11, fontfamily='monospace', va='top',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9, edgecolor='gray'))
    
    # =========================================================================
    # Save figure
    # =========================================================================
    plt.savefig(os.path.join(output_dir, 'state_visualization.png'), 
                dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  State visualization saved to {os.path.join(output_dir, 'state_visualization.png')}")


# =============================================================================
# Structure Factor Visualization Functions
# =============================================================================

def plot_observable_definitions(output_dir: str):
    """
    Create a reference figure explaining all computed observables.
    
    This provides a visual summary of what each order parameter measures.
    """
    if not HAS_MATPLOTLIB:
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    fig = plt.figure(figsize=(16, 20))
    
    # Title
    fig.suptitle('BFG Kagome Order Parameters: Observable Definitions', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    # Create text content
    content = """
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[A] OFF-DIAGONAL STRUCTURE FACTOR S(q) = ⟨S^+S^-⟩ — Translation Symmetry Breaking
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    Definition:
        S(q) = (1/N) Σ_{i,j} exp[iq·(r_i - r_j)] ⟨S^+_i S^-_j⟩
        
        Note: ⟨S^+⟩ = ⟨S^-⟩ = 0 in S^z-conserving states, so no connected
        correlation is needed.

    Order Parameter:
        m_trans = √(max_q |S(q)| / N)

    Physical Meaning:
        • Measures CRYSTALLINE / DENSITY WAVE order
        • Bragg peaks at q ≠ 0 indicate long-range magnetic order
        • Peak positions reveal the ordering wavevector
        • m_trans > 0.1 → significant translation symmetry breaking

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[B] NEMATIC ORDER — Rotational Symmetry Breaking (C₆ → C₂)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    Definition:
        O_α = Σ_{⟨ij⟩∈α} ⟨S^+_i S^-_j + S^-_i S^+_j⟩   (α = 0°, 60°, 120°)
        
        ψ_nem = O_0 + ω O_1 + ω² O_2    where ω = exp(2πi/3)

    Order Parameter:
        m_nem = |ψ_nem| / Σ_α |O_α|

    Physical Meaning:
        • Measures BOND ORIENTATION ANISOTROPY
        • Kagome has 3 bond directions at 0°, 60°, 120°
        • ψ_nem ≠ 0 means bonds prefer certain directions
        • Detects C₆ → C₂ rotational symmetry breaking
        • m_nem > 0.1 → significant nematic order

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[C] BOND/DIMER STRUCTURE FACTOR S_D(q) — VBS Order
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    Definition:
        D_{ij} = S^+_i S^-_j + S^-_i S^+_j    (XY bond energy)
        
        S_D(q) = (1/N_b) Σ_{b,b'} exp[iq·(R_b - R_{b'})] ⟨δD_b δD_{b'}⟩
        
        where δD_b = D_b - ⟨D⟩   (connected correlation)

    Order Parameter:
        m_vbs = √(max_{q≠0} |S_D(q)| / N_b)

    Physical Meaning:
        • Measures VALENCE BOND SOLID (VBS) order
        • Bragg peaks indicate dimerization / bond crystallization
        • Detects patterns like columnar VBS, plaquette VBS, etc.
        • m_vbs > 0.1 → significant dimer order

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[D] BOWTIE / PLAQUETTE ORDER — BFG Native Observable
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    Bowtie Structure (5-site, 2 NN triangles sharing 1 corner):
    
              s2
             /  \\
           s1----s0----s3   ← s0 is shared vertex (EXCLUDED from ring-flip)
                  \\    /
                    s4

    Definition:
        The 4-spin ring-flip acts on the 4 OUTER corners (s1, s2, s3, s4),
        EXCLUDING the shared center vertex (s0):
        
        P_bt = ⟨S^+_{s1} S^-_{s2} S^+_{s3} S^-_{s4} + h.c.⟩
        
        This resonates between |↑↓↑↓⟩ and |↓↑↓↑⟩ on the 4 outer sites.
        
        S_P(q) = (1/N_bt) Σ_{bt,bt'} exp[iq·(R_bt - R_{bt'})] ⟨δP_bt δP_{bt'}⟩

    Order Parameters:
        m_plaq = √(max_{q≠0} |S_P(q)| / N_bt)    (plaquette crystal)
        ⟨|P|⟩  = mean resonance strength         (liquid indicator)

    Physical Meaning:
        • Bowtie = 2 NN triangles sharing exactly 1 vertex
        • χ_tri measures 3-site ring exchange (chiral fluctuations)
        • P_bt correlates the two triangles in a bowtie
        • m_plaq > 0.1 → plaquette crystal order
        • ⟨|P|⟩ large + m_plaq small → RESONATING liquid phase

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PHASE IDENTIFICATION GUIDE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    ┌─────────────────────────────────────────────────────────────────────┐
    │  m_trans large  →  MAGNETIC ORDER (AFM, FM, spiral, etc.)          │
    │  m_nem large    →  NEMATIC PHASE (bond-oriented liquid)            │
    │  m_vbs large    →  VALENCE BOND SOLID (dimerized state)            │
    │  m_plaq large   →  PLAQUETTE CRYSTAL (ordered bowties)             │
    │  ⟨|P|⟩ large, m_plaq small  →  SPIN LIQUID (resonating phase)      │
    │  All small      →  QUANTUM DISORDERED / PARAMAGNETIC               │
    └─────────────────────────────────────────────────────────────────────┘
    """
    
    ax = fig.add_subplot(111)
    ax.axis('off')
    ax.text(0.02, 0.98, content, transform=ax.transAxes,
            fontsize=10, fontfamily='monospace',
            verticalalignment='top', horizontalalignment='left')
    
    plt.savefig(os.path.join(output_dir, 'observable_definitions.png'), 
                dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"Observable definitions saved to {output_dir}/observable_definitions.png")


def plot_structure_factors(results: Dict, output_dir: str):
    """Plot all computed structure factors"""
    if not HAS_MATPLOTLIB:
        print("Matplotlib not available, skipping plots")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Off-diagonal structure factor S(q) = <S^+S^->
    if 'translation' in results and 's_q_2d' in results['translation']:
        data = results['translation']
        fig, ax = plt.subplots(figsize=(8, 6))
        extent = [data['q1_vals'][0], data['q1_vals'][-1], 
                  data['q2_vals'][0], data['q2_vals'][-1]]
        im = ax.imshow(np.abs(data['s_q_2d']).T, origin='lower', extent=extent,
                       cmap='viridis', aspect='equal')
        ax.set_xlabel('q₁ (units of b₁)')
        ax.set_ylabel('q₂ (units of b₂)')
        ax.set_title(f'S(q) [S$^+$S$^-$] - Translation Order\nm_trans = {data["m_trans"]:.4f}')
        plt.colorbar(im, ax=ax, label='|S(q)|')
        plt.savefig(os.path.join(output_dir, 'SpmSm_structure_factor.png'), dpi=150, bbox_inches='tight')
        plt.close()
    
    # 2. Bond/dimer structure factor
    if 'bond' in results and 's_d_2d' in results['bond']:
        data = results['bond']
        fig, ax = plt.subplots(figsize=(8, 6))
        extent = [data['q1_vals'][0], data['q1_vals'][-1], 
                  data['q2_vals'][0], data['q2_vals'][-1]]
        im = ax.imshow(np.abs(data['s_d_2d']).T, origin='lower', extent=extent,
                       cmap='plasma', aspect='equal')
        ax.set_xlabel('q₁ (units of b₁)')
        ax.set_ylabel('q₂ (units of b₂)')
        ax.set_title(f'S_D(q) - Bond/VBS Order\nm_vbs = {data["m_vbs"]:.4f}')
        plt.colorbar(im, ax=ax, label='|S_D(q)|')
        plt.savefig(os.path.join(output_dir, 'bond_structure_factor.png'), dpi=150, bbox_inches='tight')
        plt.close()
    
    # 3. Plaquette structure factor
    if 'plaquette' in results and 's_p_2d' in results['plaquette']:
        data = results['plaquette']
        fig, ax = plt.subplots(figsize=(8, 6))
        extent = [data['q1_vals'][0], data['q1_vals'][-1], 
                  data['q2_vals'][0], data['q2_vals'][-1]]
        im = ax.imshow(np.abs(data['s_p_2d']).T, origin='lower', extent=extent,
                       cmap='inferno', aspect='equal')
        ax.set_xlabel('q₁ (units of b₁)')
        ax.set_ylabel('q₂ (units of b₂)')
        ax.set_title(f'S_P(q) - Plaquette/Bow-tie Order\nm_plaq = {data["m_plaquette"]:.4f}')
        plt.colorbar(im, ax=ax, label='|S_P(q)|')
        plt.savefig(os.path.join(output_dir, 'plaquette_structure_factor.png'), dpi=150, bbox_inches='tight')
        plt.close()
    
    # 4. Summary bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    
    order_params = []
    labels = []
    
    if 'translation' in results:
        order_params.append(results['translation']['m_trans'])
        labels.append('m_trans\n(density wave)')
    
    if 'nematic' in results:
        order_params.append(results['nematic']['m_nem'])
        labels.append('m_nem\n(nematic)')
    
    if 'stripe' in results:
        order_params.append(results['stripe']['m_stripe'])
        labels.append('m_stripe\n(stripe)')
    
    if 'bond' in results and 'm_vbs' in results['bond']:
        order_params.append(results['bond']['m_vbs'])
        labels.append('m_vbs\n(VBS)')
    
    if 'plaquette' in results and 'm_plaquette' in results['plaquette']:
        order_params.append(results['plaquette']['m_plaquette'])
        labels.append('m_plaq\n(plaquette)')
        order_params.append(results['plaquette']['resonance_strength'])
        labels.append('⟨|P|⟩\n(resonance)')
    
    if order_params:
        colors = plt.cm.Set2(np.linspace(0, 1, len(order_params)))
        bars = ax.bar(range(len(order_params)), order_params, color=colors)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels)
        ax.set_ylabel('Order Parameter Value')
        ax.set_title('BFG Kagome Order Parameters Summary')
        
        # Add value labels on bars
        for bar, val in zip(bars, order_params):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{val:.4f}', ha='center', va='bottom', fontsize=9)
        
        plt.savefig(os.path.join(output_dir, 'order_parameters_summary.png'), 
                    dpi=150, bbox_inches='tight')
        plt.close()
    
    print(f"Plots saved to {output_dir}")


def save_results(results: Dict, output_file: str):
    """Save results to HDF5 file"""
    with h5py.File(output_file, 'w') as f:
        f.attrs['description'] = 'BFG Kagome Order Parameters'
        
        for category, data in results.items():
            if isinstance(data, dict):
                grp = f.create_group(category)
                for key, value in data.items():
                    if isinstance(value, np.ndarray):
                        grp.create_dataset(key, data=value)
                    elif isinstance(value, (int, float, complex)):
                        grp.attrs[key] = value
                    elif isinstance(value, dict):
                        # Skip nested dicts for simplicity
                        pass
    
    print(f"Results saved to {output_file}")


def print_summary(results: Dict):
    """Print summary of computed order parameters"""
    print("\n" + "="*70)
    print("BFG KAGOME ORDER PARAMETER SUMMARY")
    print("="*70)
    
    if 'translation' in results:
        data = results['translation']
        print("\n[A] TRANSLATION SYMMETRY BREAKING (Diagonal Structure Factor)")
        print(f"    m_trans = {data['m_trans']:.6f}")
        print(f"    S^{{zz}}(Q_max) = {data['s_q_max']:.6f}")
        q_max = data['q_max']
        print(f"    Q_max = ({q_max[0]:.4f}, {q_max[1]:.4f})")
        if data.get('q_max_idx') is not None:
            print(f"    (k-point index = {data['q_max_idx']})")
        if data['m_trans'] > 0.1:
            print("    → SIGNIFICANT: Possible crystalline/density wave order")
        else:
            print("    → Weak or no translation symmetry breaking")
    
    if 'nematic' in results:
        data = results['nematic']
        print("\n[B] ROTATIONAL SYMMETRY BREAKING (Nematic Order)")
        print(f"    m_nem = {data['m_nem']:.6f}")
        print(f"    Bond anisotropy = {data['anisotropy']:.6f}")
        print(f"    ψ_nem = {data['psi_nem']:.6f}")
        print(f"    Bonds per orientation: {data['bonds_by_orientation']}")
        if data['m_nem'] > 0.1:
            print("    → SIGNIFICANT: C₆ → C₂ symmetry breaking")
        else:
            print("    → Rotational symmetry preserved")
    
    if 'stripe' in results:
        data = results['stripe']
        print("\n[B'] STRIPE STRUCTURE FACTOR")
        print(f"    m_stripe = {data['m_stripe']:.6f}")
        print(f"    S_stripe = {data['S_stripe']:.6f}")
    
    if 'bond' in results:
        data = results['bond']
        if 'error' not in data:
            print("\n[C] BOND/DIMER ORDER (VBS)")
            print(f"    m_vbs = {data['m_vbs']:.6f}")
            print(f"    S_D(Q_max) = {data['s_d_max']:.6f}")
            q_max = data['q_max']
            print(f"    Q_max = ({q_max[0]:.4f}, {q_max[1]:.4f})")
            if data.get('q_max_idx') is not None:
                print(f"    (k-point index = {data['q_max_idx']})")
            print(f"    <D> (mean XY bond) = {data['D_mean']:.6f}")
            if data['m_vbs'] > 0.1:
                print("    → SIGNIFICANT: Valence bond solid order")
            else:
                print("    → No VBS order detected")
    
    if 'plaquette' in results:
        data = results['plaquette']
        if 'error' not in data:
            print("\n[D] PLAQUETTE/BOW-TIE RESONANCE ORDER (BFG Native)")
            print(f"    m_plaquette = {data['m_plaquette']:.6f}")
            print(f"    S_P(Q_max) = {data['s_p_max']:.6f}")
            q_max = data['q_max']
            print(f"    Q_max = ({q_max[0]:.4f}, {q_max[1]:.4f})")
            if data.get('q_max_idx') is not None:
                print(f"    (k-point index = {data['q_max_idx']})")
            if data.get('s_p_q0') is not None:
                print(f"    S_P(q=0) = {data['s_p_q0']:.6f}")
            print(f"    <|P|> (resonance strength) = {data['resonance_strength']:.6f}")
            print(f"    Number of bow-ties = {data['n_plaquettes']}")
            
            if data['m_plaquette'] > 0.1:
                print("    → SIGNIFICANT: Plaquette/bow-tie crystal order")
            elif data['resonance_strength'] > 0.1 and data['m_plaquette'] < 0.1:
                print("    → LIQUID-LIKE: Strong resonance without ordering")
            else:
                print("    → Weak plaquette correlations")
        else:
            print(f"\n[D] PLAQUETTE ORDER: {data['error']}")
    
    print("\n" + "="*70)


# =============================================================================
# Jpm Scan Functions (with multiprocessing support)
# =============================================================================

def extract_jpm_value(dirname: str) -> Optional[float]:
    """Extract Jpm value from directory name like 'Jpm=0.1' or 'Jpm=-0.25'"""
    import re
    match = re.match(r'Jpm=(-?[\d.]+)', dirname)
    if match:
        return float(match.group(1))
    return None


def process_single_jpm(jpm_val: float, jpm_dir: str, args) -> Dict:
    """
    Process a single Jpm directory and return results.
    
    Returns:
        Dictionary with results for this Jpm value
    """
    result = {
        'jpm_val': jpm_val,
        'success': False,
        'energy': np.nan,
        'k_points': None,  # Store k-points array
        'translation': {
            'm_translation': np.nan, 
            's_q_max': np.nan,  # Max of S(q) = <S^+S^->
            'q_max_idx': None,
            's_q_discrete': None  # Full S(q) at all k-points
        },
        'nematic': {'m_nematic': np.nan, 'C6_breaking': np.nan},
        'stripe': {'m_stripe': np.nan, 's_stripe_max': np.nan},
        'bond': {
            'm_vbs': np.nan, 
            's_d_max': np.nan, 
            'D_mean': np.nan, 
            'q_max_idx': None,
            's_d_discrete': None  # Full S_D(q) at all k-points
        },
        'plaquette': {
            'm_plaquette': np.nan, 
            's_p_max': np.nan, 
            's_p_q0': np.nan, 
            'resonance_strength': np.nan, 
            'q_max_idx': None,
            's_p_discrete': None  # Full S_P(q) at all k-points
        },
    }
    
    # Find wavefunction file
    wf_file = os.path.join(jpm_dir, 'output', 'ed_results.h5')
    if not os.path.exists(wf_file):
        print(f"  WARNING: Wavefunction file not found: {wf_file}")
        return result
    
    try:
        # Load cluster data with optional kpoints_file from args
        kpoints_file = getattr(args, 'kpoints_file', None)
        cluster = load_kagome_cluster(jpm_dir, kpoints_file=kpoints_file)
        
        # Store k-points
        result['k_points'] = cluster.get('k_points', None)
        
        # Load wavefunction
        psi = load_wavefunction(wf_file, args.eigenvector_index)
        
        # Load energy
        try:
            E = load_eigenvalue(wf_file, args.eigenvector_index)
        except:
            E = np.nan
        
        result['energy'] = E
        
        # Compute order parameters
        trans = compute_translation_order_parameter(psi, cluster, n_q_points=args.n_q_points)
        if 'error' not in trans:
            result['translation'] = {
                'm_translation': trans.get('m_trans', np.nan),
                's_q_max': trans.get('s_q_max', np.nan),  # Max of S(q) = <S^+S^->
                'q_max_idx': trans.get('q_max_idx', None),
                's_q_discrete': trans.get('s_q_discrete', None)  # Full S(q) array
            }
        
        nem = compute_nematic_order(psi, cluster)
        if 'error' not in nem:
            result['nematic'] = {
                'm_nematic': nem.get('m_nem', np.nan),
                'C6_breaking': nem.get('anisotropy', np.nan)
            }
        
        stripe = compute_stripe_structure_factor(psi, cluster)
        if 'error' not in stripe:
            result['stripe'] = {
                'm_stripe': stripe.get('m_stripe', np.nan),
                's_stripe_max': np.abs(stripe.get('S_stripe', np.nan))  # Use S_stripe magnitude
            }
        
        bond = compute_bond_structure_factor(psi, cluster, n_q_points=args.n_q_points)
        if 'error' not in bond:
            result['bond'] = {
                'm_vbs': bond.get('m_vbs', np.nan),
                's_d_max': bond.get('s_d_max', np.nan),
                'D_mean': bond.get('D_mean', np.nan),
                'q_max_idx': bond.get('q_max_idx', None),
                's_d_discrete': bond.get('s_d_discrete', None)  # Full S_D(q) array
            }
        
        if not args.skip_plaquette:
            plaq = compute_plaquette_order(psi, cluster, n_q_points=min(30, args.n_q_points))
            if 'error' not in plaq:
                result['plaquette'] = {
                    'm_plaquette': plaq.get('m_plaquette', np.nan),
                    's_p_max': plaq.get('s_p_max', np.nan),
                    's_p_q0': plaq.get('s_p_q0', np.nan),
                    'resonance_strength': plaq.get('resonance_strength', np.nan),
                    'q_max_idx': plaq.get('q_max_idx', None),
                    's_p_discrete': plaq.get('s_p_discrete', None)  # Full S_P(q) array
                }
        
        # Generate bond map plots if requested
        if getattr(args, 'plot_bond_map', False):
            output_dir = os.path.join(jpm_dir, 'output')
            os.makedirs(output_dir, exist_ok=True)
            
            title = f"Jpm = {jpm_val:.4f}, E = {E:.4f}" if not np.isnan(E) else f"Jpm = {jpm_val:.4f}"
            plot_bond_expectation_map(
                psi, cluster,
                output_path=os.path.join(output_dir, 'bond_expectation_map.png'),
                title=title,
                show_site_labels=(cluster['n_sites'] <= 48)
            )
            plot_bond_orientation_map(
                psi, cluster,
                output_path=os.path.join(output_dir, 'bond_orientation_map.png'),
                title=title
            )
        
        result['success'] = True
        
    except Exception as e:
        print(f"  ERROR processing Jpm = {jpm_val}: {e}")
        import traceback
        traceback.print_exc()
    
    return result


def _worker_wrapper(task):
    """Wrapper for multiprocessing Pool - unpacks arguments"""
    jpm_val, jpm_dir, args_dict = task
    
    # Convert args_dict back to namespace-like object
    class Args:
        pass
    args = Args()
    for k, v in args_dict.items():
        setattr(args, k, v)
    
    return process_single_jpm(jpm_val, jpm_dir, args)


def scan_all_jpm_directories(scan_dir: str, args) -> Dict:
    """
    Scan all Jpm=* subdirectories in the given directory and compute order parameters.
    Uses Python multiprocessing for parallelization.
    
    Args:
        scan_dir: Path to directory containing Jpm=* subdirectories
        args: Parsed arguments from argparse
    
    Returns:
        Dictionary with aggregated results for all Jpm values
    """
    import glob
    import sys
    from multiprocessing import Pool, cpu_count
    
    # Find all Jpm=* subdirectories
    jpm_dirs = glob.glob(os.path.join(scan_dir, 'Jpm=*'))
    
    if not jpm_dirs:
        print(f"ERROR: No Jpm=* subdirectories found in {scan_dir}")
        return {}
    
    # Extract Jpm values and sort
    jpm_data = []
    for jpm_dir in jpm_dirs:
        dirname = os.path.basename(jpm_dir)
        jpm_val = extract_jpm_value(dirname)
        if jpm_val is not None:
            jpm_data.append((jpm_val, jpm_dir))
    
    jpm_data.sort(key=lambda x: x[0])
    n_total = len(jpm_data)
    
    # Determine number of workers
    n_workers = getattr(args, 'n_workers', None)
    if n_workers is None or n_workers <= 0:
        n_workers = min(cpu_count(), n_total, 8)  # Default to min of cpus, tasks, or 8
    
    # Try to find kpoints file in scan_dir if not explicitly provided
    kpoints_file = getattr(args, 'kpoints_file', None)
    if kpoints_file is None:
        # Look for lattice parameters file in scan_dir
        param_files = [f for f in os.listdir(scan_dir) 
                       if f.endswith('_lattice_parameters.dat')]
        if param_files:
            kpoints_file = os.path.join(scan_dir, param_files[0])
            print(f"Found k-points file in scan directory: {param_files[0]}")
    
    print(f"Found {n_total} Jpm directories in {scan_dir}")
    print(f"Jpm range: {jpm_data[0][0]} to {jpm_data[-1][0]}")
    if kpoints_file:
        print(f"Using k-points from: {kpoints_file}")
    print(f"Using {n_workers} parallel workers")
    print("="*70)
    sys.stdout.flush()
    
    # Convert args to dict for pickling
    args_dict = {
        'eigenvector_index': args.eigenvector_index,
        'n_q_points': args.n_q_points,
        'skip_plaquette': args.skip_plaquette,
        'kpoints_file': kpoints_file,
        'plot_bond_map': getattr(args, 'plot_bond_map', False),
    }
    
    # Prepare tasks
    tasks = [(jpm_val, jpm_dir, args_dict) for jpm_val, jpm_dir in jpm_data]
    
    # Run in parallel
    if n_workers > 1:
        with Pool(processes=n_workers) as pool:
            results_list = []
            for idx, result in enumerate(pool.imap(_worker_wrapper, tasks)):
                jpm_val = result['jpm_val']
                if result['success']:
                    print(f"[{idx+1}/{n_total}] Jpm={jpm_val}: E={result['energy']:.6f}, "
                          f"m_trans={result['translation']['m_translation']:.4f}, "
                          f"m_nem={result['nematic']['m_nematic']:.4f}", flush=True)
                else:
                    print(f"[{idx+1}/{n_total}] Jpm={jpm_val}: FAILED", flush=True)
                results_list.append(result)
    else:
        # Serial execution
        results_list = []
        for idx, (jpm_val, jpm_dir) in enumerate(jpm_data):
            print(f"[{idx+1}/{n_total}] Processing Jpm = {jpm_val}", flush=True)
            result = process_single_jpm(jpm_val, jpm_dir, args)
            if result['success']:
                print(f"  Jpm={jpm_val}: E={result['energy']:.6f}, "
                      f"m_trans={result['translation']['m_translation']:.4f}, "
                      f"m_nem={result['nematic']['m_nematic']:.4f}", flush=True)
            else:
                print(f"  Jpm={jpm_val}: FAILED", flush=True)
            results_list.append(result)
    
    # Sort by Jpm value
    results_list.sort(key=lambda x: x['jpm_val'])
    
    # Build final results dictionary
    all_results = {
        'jpm_values': [],
        'energies': [],
        'k_points': None,  # Will store shared k-points array
        'translation': {
            'm_translation': [], 
            's_q_max': [],  # Max of S(q) = <S^+S^->
            'q_max_idx': [],
            's_q_all': []  # Full S(q) at all k-points for each Jpm
        },
        'nematic': {'m_nematic': [], 'C6_breaking': []},
        'stripe': {'m_stripe': [], 's_stripe_max': []},
        'bond': {
            'm_vbs': [], 
            's_d_max': [], 
            'D_mean': [],
            'q_max_idx': [],
            's_d_all': []  # Full S_D(q) at all k-points for each Jpm
        },
        'plaquette': {
            'm_plaquette': [], 
            's_p_max': [], 
            's_p_q0': [], 
            'resonance_strength': [],
            'q_max_idx': [],
            's_p_all': []  # Full S_P(q) at all k-points for each Jpm
        },
        'metadata': {
            'scan_dir': scan_dir,
            'n_jpm_points': len(results_list)
        }
    }
    
    # Get k-points from first successful result
    for r in results_list:
        if r.get('k_points') is not None:
            all_results['k_points'] = r['k_points']
            break
    
    for r in results_list:
        all_results['jpm_values'].append(r['jpm_val'])
        all_results['energies'].append(r['energy'])
        
        # Translation order
        all_results['translation']['m_translation'].append(r['translation'].get('m_translation', np.nan))
        all_results['translation']['s_q_max'].append(r['translation'].get('s_q_max', np.nan))
        all_results['translation']['q_max_idx'].append(r['translation'].get('q_max_idx', None))
        all_results['translation']['s_q_all'].append(r['translation'].get('s_q_discrete', None))
        
        # Nematic order
        all_results['nematic']['m_nematic'].append(r['nematic'].get('m_nematic', np.nan))
        all_results['nematic']['C6_breaking'].append(r['nematic'].get('C6_breaking', np.nan))
        
        # Stripe order
        all_results['stripe']['m_stripe'].append(r['stripe'].get('m_stripe', np.nan))
        all_results['stripe']['s_stripe_max'].append(r['stripe'].get('s_stripe_max', np.nan))
        
        # Bond order
        all_results['bond']['m_vbs'].append(r['bond'].get('m_vbs', np.nan))
        all_results['bond']['s_d_max'].append(r['bond'].get('s_d_max', np.nan))
        all_results['bond']['D_mean'].append(r['bond'].get('D_mean', np.nan))
        all_results['bond']['q_max_idx'].append(r['bond'].get('q_max_idx', None))
        all_results['bond']['s_d_all'].append(r['bond'].get('s_d_discrete', None))
        
        # Plaquette order
        all_results['plaquette']['m_plaquette'].append(r['plaquette'].get('m_plaquette', np.nan))
        all_results['plaquette']['s_p_max'].append(r['plaquette'].get('s_p_max', np.nan))
        all_results['plaquette']['s_p_q0'].append(r['plaquette'].get('s_p_q0', np.nan))
        all_results['plaquette']['resonance_strength'].append(r['plaquette'].get('resonance_strength', np.nan))
        all_results['plaquette']['q_max_idx'].append(r['plaquette'].get('q_max_idx', None))
        all_results['plaquette']['s_p_all'].append(r['plaquette'].get('s_p_discrete', None))
    
    # Convert lists to numpy arrays
    all_results['jpm_values'] = np.array(all_results['jpm_values'])
    all_results['energies'] = np.array(all_results['energies'])
    
    # Convert scalar lists to arrays
    for key in ['translation', 'nematic', 'stripe', 'bond', 'plaquette']:
        for subkey in all_results[key]:
            if subkey in ['q_max_idx', 's_q_all', 's_d_all', 's_p_all']:
                # These stay as lists (may contain None or arrays of different shapes)
                pass
            else:
                all_results[key][subkey] = np.array(all_results[key][subkey])
    
    # Convert S(q) arrays to 2D matrices if all are present
    # Shape will be (n_jpm, n_kpoints)
    for key, sq_name in [('translation', 's_q_all'), ('bond', 's_d_all'), ('plaquette', 's_p_all')]:
        sq_list = all_results[key][sq_name]
        if all(sq is not None for sq in sq_list):
            try:
                all_results[key][sq_name] = np.array(sq_list)  # (n_jpm, n_kpoints)
            except ValueError:
                pass  # Arrays have different shapes, keep as list
    
    return all_results


def save_scan_results(results: Dict, output_dir: str):
    """Save scan results to HDF5 file"""
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, 'jpm_scan_results.h5')
    
    with h5py.File(filepath, 'w') as f:
        f.create_dataset('jpm_values', data=results['jpm_values'])
        f.create_dataset('energies', data=results['energies'])
        
        # Save k-points if available
        if results.get('k_points') is not None:
            f.create_dataset('k_points', data=results['k_points'])
        
        for key in ['translation', 'nematic', 'stripe', 'bond', 'plaquette']:
            grp = f.create_group(key)
            for subkey, data in results[key].items():
                # Skip data that can't be saved (lists with None, etc.)
                if data is None:
                    continue
                if isinstance(data, list):
                    # Check if it's a list of None values or mixed
                    if all(x is None for x in data):
                        continue
                    # Try to convert to array, skip if it fails
                    try:
                        data = np.array(data)
                    except (ValueError, TypeError):
                        continue
                
                if isinstance(data, np.ndarray):
                    grp.create_dataset(subkey, data=data)
        
        # Metadata
        meta = f.create_group('metadata')
        for k, v in results['metadata'].items():
            if isinstance(v, str):
                meta.attrs[k] = v
            else:
                meta.attrs[k] = v
    
    print(f"Saved scan results to {filepath}")


def plot_jpm_scan_results(results: Dict, output_dir: str):
    """Generate plots of order parameters vs Jpm"""
    if not HAS_MATPLOTLIB:
        print("WARNING: matplotlib not available, skipping plots")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    jpm = results['jpm_values']
    
    # Figure 1: All order parameters
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Energy
    ax = axes[0, 0]
    ax.plot(jpm, results['energies'], 'ko-', markersize=4)
    ax.set_xlabel('Jpm')
    ax.set_ylabel('Energy')
    ax.set_title('Ground State Energy')
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    ax.grid(True, alpha=0.3)
    
    # Translation order
    ax = axes[0, 1]
    ax.plot(jpm, results['translation']['m_translation'], 'bo-', markersize=4, label='m_translation')
    ax.set_xlabel('Jpm')
    ax.set_ylabel('Order Parameter')
    ax.set_title('Translation Order (Solid/CDW)')
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Nematic order
    ax = axes[0, 2]
    ax.plot(jpm, results['nematic']['m_nematic'], 'go-', markersize=4, label='m_nematic')
    ax.set_xlabel('Jpm')
    ax.set_ylabel('Order Parameter')
    ax.set_title('Nematic Order (C6 Breaking)')
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Stripe order
    ax = axes[1, 0]
    ax.plot(jpm, results['stripe']['m_stripe'], 'ro-', markersize=4, label='m_stripe')
    ax.set_xlabel('Jpm')
    ax.set_ylabel('Order Parameter')
    ax.set_title('Stripe Order')
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # VBS order
    ax = axes[1, 1]
    ax.plot(jpm, results['bond']['m_vbs'], 'mo-', markersize=4, label='m_vbs')
    ax.set_xlabel('Jpm')
    ax.set_ylabel('Order Parameter')
    ax.set_title('VBS Order (Dimer)')
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Plaquette order
    ax = axes[1, 2]
    ax.plot(jpm, results['plaquette']['m_plaquette'], 'co-', markersize=4, label='m_plaquette')
    ax.plot(jpm, results['plaquette']['resonance_strength'], 'c^--', markersize=4, alpha=0.6, label='resonance')
    ax.set_xlabel('Jpm')
    ax.set_ylabel('Order Parameter')
    ax.set_title('Plaquette/Bow-tie Order')
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'order_parameters_vs_jpm.png'), dpi=150)
    plt.savefig(os.path.join(output_dir, 'order_parameters_vs_jpm.pdf'))
    plt.close()
    
    # Figure 2: Combined order parameters comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(jpm, results['translation']['m_translation'], 'b-o', markersize=5, label='Translation (CDW)')
    ax.plot(jpm, results['nematic']['m_nematic'], 'g-s', markersize=5, label='Nematic')
    ax.plot(jpm, results['stripe']['m_stripe'], 'r-^', markersize=5, label='Stripe')
    ax.plot(jpm, results['bond']['m_vbs'], 'm-d', markersize=5, label='VBS')
    ax.plot(jpm, results['plaquette']['m_plaquette'], 'c-v', markersize=5, label='Plaquette')
    
    ax.set_xlabel('Jpm', fontsize=12)
    ax.set_ylabel('Order Parameter', fontsize=12)
    ax.set_title('BFG Kagome: Order Parameters vs Jpm', fontsize=14)
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'order_parameters_combined.png'), dpi=150)
    plt.savefig(os.path.join(output_dir, 'order_parameters_combined.pdf'))
    plt.close()
    
    # Figure 3: q_max_idx vs Jpm (phase transition indicator)
    plot_qmax_vs_jpm(results, output_dir)
    
    # Figure 4: Phase diagram with color-coded regions
    plot_phase_diagram(results, output_dir)
    
    # Figure 5: S(q) heatmap at all k-points vs Jpm
    plot_sq_heatmap(results, output_dir)
    
    print(f"Saved plots to {output_dir}")


def plot_qmax_vs_jpm(results: Dict, output_dir: str):
    """
    Plot the index of the k-point with maximum S(q) vs Jpm.
    
    This serves as a phase transition indicator - jumps in q_max_idx 
    indicate potential phase boundaries.
    
    Args:
        results: Dictionary from scan_all_jpm_directories
        output_dir: Directory to save plots
    """
    if not HAS_MATPLOTLIB:
        return
    
    jpm = results['jpm_values']
    k_points = results.get('k_points', None)
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
    
    # Translation S(q)
    ax = axes[0]
    q_max_trans = results['translation'].get('q_max_idx', [])
    if q_max_trans and any(x is not None for x in q_max_trans):
        q_max_trans_arr = [x if x is not None else -1 for x in q_max_trans]
        ax.scatter(jpm, q_max_trans_arr, c=q_max_trans_arr, cmap='viridis', s=50, marker='o')
        ax.plot(jpm, q_max_trans_arr, 'k-', alpha=0.3, linewidth=0.5)
        ax.set_ylabel('q_max index', fontsize=11)
        ax.set_title('Translation Order S(q) [S^+S^-] - Maximum q-point', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
        
        # Add k-point labels if available
        if k_points is not None:
            unique_idx = sorted(set(x for x in q_max_trans_arr if x >= 0))
            for idx in unique_idx:
                if idx < len(k_points):
                    kx, ky = k_points[idx]
                    ax.axhline(y=idx, color='gray', linestyle=':', alpha=0.2)
                    ax.text(jpm.max() * 1.02, idx, f'({kx:.2f}, {ky:.2f})', 
                           fontsize=8, va='center')
    else:
        ax.text(0.5, 0.5, 'No data', transform=ax.transAxes, ha='center')
    
    # Bond S_D(q)
    ax = axes[1]
    q_max_bond = results['bond'].get('q_max_idx', [])
    if q_max_bond and any(x is not None for x in q_max_bond):
        q_max_bond_arr = [x if x is not None else -1 for x in q_max_bond]
        ax.scatter(jpm, q_max_bond_arr, c=q_max_bond_arr, cmap='plasma', s=50, marker='s')
        ax.plot(jpm, q_max_bond_arr, 'k-', alpha=0.3, linewidth=0.5)
        ax.set_ylabel('q_max index', fontsize=11)
        ax.set_title('Bond Order S_D(q) - Maximum q-point', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
        
        if k_points is not None:
            unique_idx = sorted(set(x for x in q_max_bond_arr if x >= 0))
            for idx in unique_idx:
                if idx < len(k_points):
                    kx, ky = k_points[idx]
                    ax.axhline(y=idx, color='gray', linestyle=':', alpha=0.2)
                    ax.text(jpm.max() * 1.02, idx, f'({kx:.2f}, {ky:.2f})', 
                           fontsize=8, va='center')
    else:
        ax.text(0.5, 0.5, 'No data', transform=ax.transAxes, ha='center')
    
    # Plaquette S_P(q)
    ax = axes[2]
    q_max_plaq = results['plaquette'].get('q_max_idx', [])
    if q_max_plaq and any(x is not None for x in q_max_plaq):
        q_max_plaq_arr = [x if x is not None else -1 for x in q_max_plaq]
        ax.scatter(jpm, q_max_plaq_arr, c=q_max_plaq_arr, cmap='cividis', s=50, marker='^')
        ax.plot(jpm, q_max_plaq_arr, 'k-', alpha=0.3, linewidth=0.5)
        ax.set_ylabel('q_max index', fontsize=11)
        ax.set_title('Plaquette Order S_P(q) - Maximum q-point', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
        
        if k_points is not None:
            unique_idx = sorted(set(x for x in q_max_plaq_arr if x >= 0))
            for idx in unique_idx:
                if idx < len(k_points):
                    kx, ky = k_points[idx]
                    ax.axhline(y=idx, color='gray', linestyle=':', alpha=0.2)
                    ax.text(jpm.max() * 1.02, idx, f'({kx:.2f}, {ky:.2f})', 
                           fontsize=8, va='center')
    else:
        ax.text(0.5, 0.5, 'No data', transform=ax.transAxes, ha='center')
    
    axes[2].set_xlabel('Jpm', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'qmax_vs_jpm.png'), dpi=150)
    plt.savefig(os.path.join(output_dir, 'qmax_vs_jpm.pdf'))
    plt.close()


def plot_phase_diagram(results: Dict, output_dir: str):
    """
    Plot color-coded phase regions based on dominant order parameter.
    
    Each Jpm value is colored based on which order parameter is largest.
    
    Args:
        results: Dictionary from scan_all_jpm_directories
        output_dir: Directory to save plots
    """
    if not HAS_MATPLOTLIB:
        return
    
    jpm = results['jpm_values']
    
    # Collect all order parameter values
    order_params = {
        'Translation': results['translation']['m_translation'],
        'Nematic': results['nematic']['m_nematic'],
        'Stripe': results['stripe']['m_stripe'],
        'VBS': results['bond']['m_vbs'],
        'Plaquette': results['plaquette']['m_plaquette'],
    }
    
    # Define colors for each phase
    phase_colors = {
        'Translation': 'blue',
        'Nematic': 'green',
        'Stripe': 'red',
        'VBS': 'magenta',
        'Plaquette': 'cyan',
        'None': 'gray'
    }
    
    # Determine dominant phase at each Jpm
    dominant_phase = []
    for i in range(len(jpm)):
        max_val = -np.inf
        max_phase = 'None'
        for name, vals in order_params.items():
            if np.isnan(vals[i]):
                continue
            if vals[i] > max_val:
                max_val = vals[i]
                max_phase = name
        dominant_phase.append(max_phase)
    
    # Create figure with two subplots
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [3, 1]})
    
    # Top plot: All order parameters with phase region shading
    ax = axes[0]
    
    # Plot each order parameter
    for name, vals in order_params.items():
        ax.plot(jpm, vals, '-o', markersize=4, label=name, color=phase_colors[name])
    
    # Add vertical lines at phase boundaries
    for i in range(1, len(dominant_phase)):
        if dominant_phase[i] != dominant_phase[i-1]:
            # Phase transition between Jpm[i-1] and Jpm[i]
            jpm_boundary = (jpm[i-1] + jpm[i]) / 2
            ax.axvline(x=jpm_boundary, color='black', linestyle='-', linewidth=2, alpha=0.7)
    
    ax.set_xlabel('Jpm', fontsize=12)
    ax.set_ylabel('Order Parameter', fontsize=12)
    ax.set_title('BFG Kagome Phase Diagram: Order Parameters vs Jpm', fontsize=14)
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=10)
    
    # Bottom plot: Phase bar showing dominant phase
    ax = axes[1]
    
    # Create colored bars for each Jpm region
    if len(jpm) > 1:
        for i in range(len(jpm)):
            phase = dominant_phase[i]
            color = phase_colors[phase]
            
            # Calculate bar width
            if i == 0:
                left = jpm[0] - (jpm[1] - jpm[0]) / 2
                right = (jpm[0] + jpm[1]) / 2
            elif i == len(jpm) - 1:
                left = (jpm[i-1] + jpm[i]) / 2
                right = jpm[i] + (jpm[i] - jpm[i-1]) / 2
            else:
                left = (jpm[i-1] + jpm[i]) / 2
                right = (jpm[i] + jpm[i+1]) / 2
            
            ax.axvspan(left, right, color=color, alpha=0.7)
    
    ax.set_xlim(jpm.min() - 0.05 * (jpm.max() - jpm.min()), 
                jpm.max() + 0.05 * (jpm.max() - jpm.min()))
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_xlabel('Jpm', fontsize=12)
    ax.set_ylabel('Dominant\nPhase', fontsize=10)
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    
    # Add legend for phase colors
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=color, label=name, alpha=0.7) 
                       for name, color in phase_colors.items() if name != 'None']
    ax.legend(handles=legend_elements, loc='upper center', ncol=5, fontsize=9,
              bbox_to_anchor=(0.5, -0.15))
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'phase_diagram.png'), dpi=150, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'phase_diagram.pdf'), bbox_inches='tight')
    plt.close()


def plot_sq_heatmap(results: Dict, output_dir: str):
    """
    Plot heatmap of S(q) at all k-points vs Jpm.
    
    Shows how spectral weight at each k-point evolves with Jpm.
    
    Args:
        results: Dictionary from scan_all_jpm_directories
        output_dir: Directory to save plots
    """
    if not HAS_MATPLOTLIB:
        return
    
    jpm = results['jpm_values']
    k_points = results.get('k_points', None)
    
    # Check if we have S(q) data for all k-points
    s_q_trans = results['translation'].get('s_q_all', None)
    s_q_bond = results['bond'].get('s_d_all', None)
    s_q_plaq = results['plaquette'].get('s_p_all', None)
    
    # Convert complex arrays to real (take absolute value)
    if isinstance(s_q_trans, np.ndarray) and np.iscomplexobj(s_q_trans):
        s_q_trans = np.abs(s_q_trans)
    if isinstance(s_q_bond, np.ndarray) and np.iscomplexobj(s_q_bond):
        s_q_bond = np.abs(s_q_bond)
    if isinstance(s_q_plaq, np.ndarray) and np.iscomplexobj(s_q_plaq):
        s_q_plaq = np.abs(s_q_plaq)
    
    # Count how many heatmaps we can make
    n_plots = sum([
        isinstance(s_q_trans, np.ndarray),
        isinstance(s_q_bond, np.ndarray),
        isinstance(s_q_plaq, np.ndarray)
    ])
    
    if n_plots == 0:
        print("  No S(q) data available for heatmap plots")
        return
    
    # Determine k-point labels
    if k_points is not None:
        k_labels = [f'({k[0]:.2f}, {k[1]:.2f})' for k in k_points]
        k_indices = np.arange(len(k_points))
    else:
        # Use generic labels
        n_k = s_q_trans.shape[1] if isinstance(s_q_trans, np.ndarray) else \
              s_q_bond.shape[1] if isinstance(s_q_bond, np.ndarray) else \
              s_q_plaq.shape[1]
        k_labels = [f'q_{i}' for i in range(n_k)]
        k_indices = np.arange(n_k)
    
    fig, axes = plt.subplots(n_plots, 1, figsize=(14, 5 * n_plots))
    if n_plots == 1:
        axes = [axes]
    
    ax_idx = 0
    
    # Translation S(q) heatmap
    if isinstance(s_q_trans, np.ndarray):
        ax = axes[ax_idx]
        # s_q_trans shape: (n_jpm, n_kpoints)
        # We want Jpm on x-axis, k-point index on y-axis
        im = ax.pcolormesh(jpm, k_indices, np.real(s_q_trans.T), shading='nearest', cmap='hot')
        plt.colorbar(im, ax=ax, label='|S(q)|')
        ax.set_xlabel('Jpm', fontsize=12)
        ax.set_ylabel('k-point index', fontsize=12)
        ax.set_title('Translation Order: |S(q)| [S^+S^-] at all k-points', fontsize=13)
        ax.axvline(x=0, color='white', linestyle='--', alpha=0.7, linewidth=1)
        
        # Mark the maximum at each Jpm
        q_max_idx = results['translation'].get('q_max_idx', [])
        if q_max_idx and any(x is not None for x in q_max_idx):
            for i, (j, qidx) in enumerate(zip(jpm, q_max_idx)):
                if qidx is not None:
                    ax.plot(j, qidx, 'w*', markersize=8, markeredgecolor='black', markeredgewidth=0.5)
        
        # Add k-point labels on right side
        if len(k_labels) <= 20:
            ax.set_yticks(k_indices)
            ax.set_yticklabels(k_labels, fontsize=8)
        
        ax_idx += 1
    
    # Bond S_D(q) heatmap
    if isinstance(s_q_bond, np.ndarray):
        ax = axes[ax_idx]
        im = ax.pcolormesh(jpm, k_indices, np.real(s_q_bond.T), shading='nearest', cmap='viridis')
        plt.colorbar(im, ax=ax, label='|S_D(q)|')
        ax.set_xlabel('Jpm', fontsize=12)
        ax.set_ylabel('k-point index', fontsize=12)
        ax.set_title('Bond Order: |S_D(q)| at all k-points', fontsize=13)
        ax.axvline(x=0, color='white', linestyle='--', alpha=0.7, linewidth=1)
        
        q_max_idx = results['bond'].get('q_max_idx', [])
        if q_max_idx and any(x is not None for x in q_max_idx):
            for i, (j, qidx) in enumerate(zip(jpm, q_max_idx)):
                if qidx is not None:
                    ax.plot(j, qidx, 'w*', markersize=8, markeredgecolor='black', markeredgewidth=0.5)
        
        if len(k_labels) <= 20:
            ax.set_yticks(k_indices)
            ax.set_yticklabels(k_labels, fontsize=8)
        
        ax_idx += 1
    
    # Plaquette S_P(q) heatmap
    if isinstance(s_q_plaq, np.ndarray):
        ax = axes[ax_idx]
        im = ax.pcolormesh(jpm, k_indices, np.real(s_q_plaq.T), shading='nearest', cmap='plasma')
        plt.colorbar(im, ax=ax, label='|S_P(q)|')
        ax.set_xlabel('Jpm', fontsize=12)
        ax.set_ylabel('k-point index', fontsize=12)
        ax.set_title('Plaquette Order: |S_P(q)| at all k-points', fontsize=13)
        ax.axvline(x=0, color='white', linestyle='--', alpha=0.7, linewidth=1)
        
        q_max_idx = results['plaquette'].get('q_max_idx', [])
        if q_max_idx and any(x is not None for x in q_max_idx):
            for i, (j, qidx) in enumerate(zip(jpm, q_max_idx)):
                if qidx is not None:
                    ax.plot(j, qidx, 'w*', markersize=8, markeredgecolor='black', markeredgewidth=0.5)
        
        if len(k_labels) <= 20:
            ax.set_yticks(k_indices)
            ax.set_yticklabels(k_labels, fontsize=8)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'sq_heatmap.png'), dpi=150)
    plt.savefig(os.path.join(output_dir, 'sq_heatmap.pdf'))
    plt.close()


def load_and_merge_individual_results(output_base: str) -> Dict:
    """
    Load individual Jpm results from subdirectories and merge them.
    Used for --plot-only mode after parallel array jobs complete.
    
    Args:
        output_base: Directory containing Jpm=* subdirectories with individual results
    
    Returns:
        Merged results dictionary
    """
    import glob
    import re
    
    # Find all Jpm subdirectories with results
    jpm_dirs = glob.glob(os.path.join(output_base, 'Jpm=*'))
    jpm_data = []
    
    for jpm_dir in jpm_dirs:
        h5_file = os.path.join(jpm_dir, 'order_parameters.h5')
        if os.path.exists(h5_file):
            dirname = os.path.basename(jpm_dir)
            jpm_val = extract_jpm_value(dirname)
            if jpm_val is not None:
                jpm_data.append((jpm_val, h5_file))
    
    jpm_data.sort(key=lambda x: x[0])
    print(f"Found {len(jpm_data)} completed Jpm calculations")
    
    if len(jpm_data) == 0:
        print("ERROR: No results found to merge")
        return {}
    
    # Initialize storage
    all_results = {
        'jpm_values': [],
        'energies': [],
        'translation': {'m_translation': [], 's_q_max': []},
        'nematic': {'m_nematic': [], 'C6_breaking': []},
        'stripe': {'m_stripe': [], 's_stripe_max': []},
        'bond': {'m_vbs': [], 's_d_max': [], 'D_mean': []},
        'plaquette': {'m_plaquette': [], 's_p_max': [], 's_p_q0': [], 'resonance_strength': []},
        'metadata': {
            'output_base': output_base,
            'n_jpm_points': len(jpm_data)
        }
    }
    
    # Read each result file
    for jpm_val, h5_file in jpm_data:
        print(f"  Reading Jpm={jpm_val}")
        try:
            with h5py.File(h5_file, 'r') as f:
                all_results['jpm_values'].append(jpm_val)
                
                # Energy from metadata
                if 'metadata' in f and 'energy' in f['metadata'].attrs:
                    all_results['energies'].append(f['metadata'].attrs['energy'])
                else:
                    all_results['energies'].append(np.nan)
                
                # Translation
                if 'translation' in f:
                    grp = f['translation']
                    all_results['translation']['m_translation'].append(grp.attrs.get('m_translation', np.nan))
                    # Try new key first, fall back to old key for backwards compatibility
                    s_q_max = grp.attrs.get('s_q_max', grp.attrs.get('s_zz_max', np.nan))
                    all_results['translation']['s_q_max'].append(s_q_max)
                else:
                    for k in all_results['translation']:
                        all_results['translation'][k].append(np.nan)
                
                # Nematic
                if 'nematic' in f:
                    grp = f['nematic']
                    all_results['nematic']['m_nematic'].append(grp.attrs.get('m_nematic', np.nan))
                    all_results['nematic']['C6_breaking'].append(grp.attrs.get('C6_breaking', np.nan))
                else:
                    for k in all_results['nematic']:
                        all_results['nematic'][k].append(np.nan)
                
                # Stripe
                if 'stripe' in f:
                    grp = f['stripe']
                    all_results['stripe']['m_stripe'].append(grp.attrs.get('m_stripe', np.nan))
                    all_results['stripe']['s_stripe_max'].append(grp.attrs.get('s_stripe_max', np.nan))
                else:
                    for k in all_results['stripe']:
                        all_results['stripe'][k].append(np.nan)
                
                # Bond/VBS
                if 'bond' in f:
                    grp = f['bond']
                    all_results['bond']['m_vbs'].append(grp.attrs.get('m_vbs', np.nan))
                    all_results['bond']['s_d_max'].append(grp.attrs.get('s_d_max', np.nan))
                    all_results['bond']['D_mean'].append(grp.attrs.get('D_mean', np.nan))
                else:
                    for k in all_results['bond']:
                        all_results['bond'][k].append(np.nan)
                
                # Plaquette
                if 'plaquette' in f:
                    grp = f['plaquette']
                    all_results['plaquette']['m_plaquette'].append(grp.attrs.get('m_plaquette', np.nan))
                    all_results['plaquette']['s_p_max'].append(grp.attrs.get('s_p_max', np.nan))
                    all_results['plaquette']['s_p_q0'].append(grp.attrs.get('s_p_q0', np.nan))
                    all_results['plaquette']['resonance_strength'].append(grp.attrs.get('resonance_strength', np.nan))
                else:
                    for k in all_results['plaquette']:
                        all_results['plaquette'][k].append(np.nan)
        except Exception as e:
            print(f"    ERROR reading {h5_file}: {e}")
            continue
    
    # Convert to arrays
    all_results['jpm_values'] = np.array(all_results['jpm_values'])
    all_results['energies'] = np.array(all_results['energies'])
    for key in ['translation', 'nematic', 'stripe', 'bond', 'plaquette']:
        for subkey in all_results[key]:
            all_results[key][subkey] = np.array(all_results[key][subkey])
    
    return all_results


# =============================================================================
# Main Function
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Compute BFG Kagome order parameters from wavefunction',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('wavefunction_file', type=str, nargs='?', default=None,
                        help='Path to HDF5 file containing eigenvectors')
    parser.add_argument('cluster_dir', type=str, nargs='?', default=None,
                        help='Directory containing cluster data (positions.dat, NN lists, etc.)')
    parser.add_argument('--scan-dir', type=str, default=None,
                        help='Directory containing Jpm=* subdirectories to scan (e.g., BFG_scan_comprehensive_pbc_3x3_cpu_energy)')
    parser.add_argument('--plot-only', type=str, default=None,
                        help='Load individual results from Jpm subdirectories and generate plots (use after parallel jobs complete)')
    parser.add_argument('--eigenvector-index', '-i', type=int, default=0,
                        help='Index of eigenvector to analyze (default: 0 = ground state)')
    parser.add_argument('--output-dir', '-o', type=str, default='./bfg_order_params',
                        help='Output directory for plots and data')
    parser.add_argument('--n-q-points', '-n', type=int, default=50,
                        help='Number of q-points for structure factor grids')
    parser.add_argument('--skip-plaquette', action='store_true',
                        help='Skip plaquette calculation (slow for large systems)')
    parser.add_argument('--n-workers', '-w', type=int, default=4,
                        help='Number of parallel workers for --scan-dir mode (default: 4)')
    parser.add_argument('--no-plots', action='store_true',
                        help='Skip generating plots')
    parser.add_argument('--geometry-only', action='store_true',
                        help='Only plot lattice geometry (no wavefunction analysis)')
    parser.add_argument('--kpoints-file', type=str, default=None,
                        help='Path to lattice parameters file containing allowed k-points '
                             '(auto-detected from cluster_dir or scan-dir if not specified)')
    parser.add_argument('--plot-bond-map', action='store_true',
                        help='Generate real-space bond expectation map (best for OBC clusters)')
    
    args = parser.parse_args()
    
    # Check if plot-only mode is requested
    if args.plot_only is not None:
        print("="*70)
        print("BFG ORDER PARAMETER PLOT-ONLY MODE")
        print("="*70)
        print(f"Loading results from: {args.plot_only}")
        
        # Load and merge individual results
        results = load_and_merge_individual_results(args.plot_only)
        
        if len(results.get('jpm_values', [])) == 0:
            print("ERROR: No results found")
            return {}
        
        # Save merged results
        save_scan_results(results, args.plot_only)
        
        # Generate plots
        plot_jpm_scan_results(results, args.plot_only)
        
        print("\n" + "="*70)
        print("PLOT GENERATION COMPLETE")
        print("="*70)
        print(f"Processed {len(results['jpm_values'])} Jpm values")
        print(f"Jpm range: {results['jpm_values'].min():.4f} to {results['jpm_values'].max():.4f}")
        print(f"Results saved to: {args.plot_only}")
        
        return results
    
    # Check if scan mode is requested
    if args.scan_dir is not None:
        print("="*70)
        print("BFG ORDER PARAMETER SCAN MODE")
        print("="*70)
        print(f"Scanning directory: {args.scan_dir}")
        
        # Run the scan
        results = scan_all_jpm_directories(args.scan_dir, args)
        
        if len(results.get('jpm_values', [])) == 0:
            print("ERROR: No results obtained from scan")
            return {}
        
        # Save results
        save_scan_results(results, args.output_dir)
        
        # Plot results
        if not args.no_plots:
            plot_jpm_scan_results(results, args.output_dir)
        
        print("\n" + "="*70)
        print("SCAN COMPLETE")
        print("="*70)
        print(f"Processed {len(results['jpm_values'])} Jpm values")
        print(f"Jpm range: {results['jpm_values'].min():.4f} to {results['jpm_values'].max():.4f}")
        print(f"Results saved to: {args.output_dir}")
        
        return results
    
    # Original single-file mode
    if args.wavefunction_file is None or args.cluster_dir is None:
        parser.error("Either provide wavefunction_file and cluster_dir, or use --scan-dir")
    
    # Load cluster data
    print(f"Loading cluster data from {args.cluster_dir}...")
    cluster = load_kagome_cluster(args.cluster_dir, kpoints_file=args.kpoints_file)
    print(f"  Loaded {cluster['n_sites']} sites")
    print(f"  NN bonds: {len(cluster.get('edges_nn', []))}")
    print(f"  2NN bonds: {len(cluster.get('edges_2nn', []))}")
    print(f"  3NN bonds: {len(cluster.get('edges_3nn', []))}")
    if cluster.get('k_points') is not None:
        print(f"  Allowed k-points: {len(cluster['k_points'])}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Always plot geometry first to verify setup
    if not args.no_plots:
        print("\nPlotting lattice geometry...")
        plot_lattice_geometry(cluster, args.output_dir)
        
        # Also show bow-tie identification without resonance values
        bowties = find_bowties(cluster)
        print(f"  Found {len(bowties)} bow-tie plaquettes")
        if len(bowties) > 0:
            plot_bowties(cluster, bowties, args.output_dir, P_r=None)
        
        # Also generate observable definitions
        plot_observable_definitions(args.output_dir)
    
    # If geometry-only mode, stop here
    if args.geometry_only:
        print(f"\nGeometry plots saved to {args.output_dir}")
        return {'cluster': cluster}
    
    # Load wavefunction
    print(f"\nLoading eigenvector {args.eigenvector_index} from {args.wavefunction_file}...")
    psi = load_wavefunction(args.wavefunction_file, args.eigenvector_index)
    print(f"  Wavefunction dimension: {len(psi)}")
    print(f"  Non-zero components: {np.sum(np.abs(psi) > 1e-15)}")
    
    # Try to load eigenvalue
    try:
        E = load_eigenvalue(args.wavefunction_file, args.eigenvector_index)
        print(f"  Energy: {E:.6f}")
    except:
        E = None
    
    n_sites = cluster['n_sites']
    
    # Compute all order parameters
    results = {'metadata': {
        'n_sites': n_sites,
        'eigenvector_index': args.eigenvector_index,
        'energy': E
    }}
    
    print("\n" + "-"*50)
    print("Computing order parameters...")
    print("-"*50)
    
    # A) Translation order
    print("\n[1/5] Computing off-diagonal structure factor S(q) = <S^+S^->...")
    results['translation'] = compute_translation_order_parameter(
        psi, cluster, n_q_points=args.n_q_points)
    
    # B) Nematic order
    print("[2/5] Computing nematic order...")
    results['nematic'] = compute_nematic_order(psi, cluster)
    
    # B') Stripe structure factor
    print("[3/5] Computing stripe structure factor...")
    results['stripe'] = compute_stripe_structure_factor(psi, cluster)
    
    # C) Bond/dimer structure factor
    print("[4/5] Computing bond/dimer structure factor...")
    results['bond'] = compute_bond_structure_factor(
        psi, cluster, n_q_points=args.n_q_points)
    
    # D) Plaquette/bow-tie order
    if not args.skip_plaquette:
        print("[5/5] Computing plaquette/bow-tie order...")
        results['plaquette'] = compute_plaquette_order(
            psi, cluster, n_q_points=min(30, args.n_q_points))
    else:
        print("[5/5] Skipping plaquette calculation (--skip-plaquette)")
        results['plaquette'] = {'error': 'Skipped by user'}
    
    # Print summary
    print_summary(results)
    
    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    save_results(results, os.path.join(args.output_dir, 'order_parameters.h5'))
    
    # Generate plots
    if not args.no_plots:
        print("\nGenerating observable definitions reference...")
        plot_observable_definitions(args.output_dir)
        
        print("\nGenerating structure factor plots...")
        plot_structure_factors(results, args.output_dir)
        
        print("\nGenerating spatially-resolved plots...")
        plot_all_spatial_order_parameters(results, cluster, args.output_dir)
        
        # Real-space bond map (best for OBC)
        if args.plot_bond_map:
            print("\nGenerating real-space bond expectation map...")
            title = f"Bond Expectations (N={n_sites}, E={E:.4f})" if E is not None else f"Bond Expectations (N={n_sites})"
            plot_bond_expectation_map(
                psi, cluster, 
                output_path=os.path.join(args.output_dir, 'bond_expectation_map.png'),
                title=title
            )
            plot_bond_orientation_map(
                psi, cluster,
                output_path=os.path.join(args.output_dir, 'bond_orientation_map.png'),
                title=title
            )
    
    print(f"\nDone! Results saved to {args.output_dir}")
    
    return results


if __name__ == "__main__":
    main()
