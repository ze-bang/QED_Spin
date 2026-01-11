#!/usr/bin/env python3
"""
BFG Kagome Order Parameter Calculator

Post-processing script for computing order parameters from exact diagonalization
wavefunctions on the kagome lattice for the Balents-Fisher-Girvin (BFG) model.

Order parameters computed:
A) Translation symmetry breaking (solids / density waves)
   1. Diagonal structure factor S^{zz}(q) - Bragg peaks indicate crystalline order

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

def load_kagome_cluster(cluster_dir: str) -> Dict:
    """
    Load kagome cluster data from helper_kagome_bfg.py output files.
    
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
    
    return cluster


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
# A) DIAGONAL STRUCTURE FACTOR S^{zz}(q) - Translation Symmetry Breaking
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
    
    m_trans = sqrt(S^{zz}(Q_max) / N)
    
    Returns dict with:
        - s_q: structure factor on 2D grid
        - q_grid: q-point coordinates
        - m_trans: order parameter value
        - q_max: wavevector with maximum S(q)
    """
    n_sites = cluster['n_sites']
    b1, b2 = cluster['b1'], cluster['b2']
    
    # Create q-grid in reciprocal space
    q1_vals = np.linspace(-1, 1, n_q_points)
    q2_vals = np.linspace(-1, 1, n_q_points)
    
    # 2D grid
    s_q_2d = np.zeros((n_q_points, n_q_points), dtype=complex)
    
    # Pre-compute correlations
    sz_exp = compute_sz_expectation(psi, n_sites)
    szsz = compute_szsz_correlation(psi, n_sites)
    szsz_connected = szsz - np.outer(sz_exp, sz_exp)
    positions = cluster['positions']
    
    for i1, q1 in enumerate(q1_vals):
        for i2, q2 in enumerate(q2_vals):
            q = q1 * b1 + q2 * b2
            for i in range(n_sites):
                r_i = positions[i]
                for j in range(n_sites):
                    r_j = positions[j]
                    phase = np.exp(1j * np.dot(q, r_i - r_j))
                    s_q_2d[i1, i2] += szsz_connected[i, j] * phase
            s_q_2d[i1, i2] /= n_sites
    
    # Find maximum (excluding q=0)
    s_q_abs = np.abs(s_q_2d)
    # Mask out q=0 region
    center = n_q_points // 2
    s_q_masked = s_q_abs.copy()
    s_q_masked[center-2:center+3, center-2:center+3] = 0
    
    max_idx = np.unravel_index(np.argmax(s_q_masked), s_q_abs.shape)
    s_q_max = s_q_abs[max_idx]
    q_max = q1_vals[max_idx[0]] * b1 + q2_vals[max_idx[1]] * b2
    
    # Order parameter
    m_trans = np.sqrt(np.abs(s_q_max) / n_sites)
    
    return {
        's_q_2d': s_q_2d,
        'q1_vals': q1_vals,
        'q2_vals': q2_vals,
        'm_trans': m_trans,
        'q_max': q_max,
        's_q_max': s_q_max,
        'sz_exp': sz_exp,
        'szsz': szsz
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


def compute_bond_structure_factor(psi: np.ndarray, cluster: Dict, n_q_points: int = 50) -> Dict:
    """
    Compute bond/dimer structure factor.
    
    D_{ij} = S^+_i S^-_j + S^-_i S^+_j  (XY bond operator)
    
    S_D(q) = (1/N_b) Σ_{b,b'} exp(iq·(r_b - r_{b'})) <δD_b δD_{b'}>
    where δD_b = D_b - <D>
    
    Bragg peaks at q ≠ 0 indicate translation-breaking VBS
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
    
    # Structure factor on q-grid
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
    
    # Find maximum (excluding q=0)
    s_d_abs = np.abs(s_d_2d)
    center = n_q_points // 2
    s_d_masked = s_d_abs.copy()
    s_d_masked[center-2:center+3, center-2:center+3] = 0
    
    max_idx = np.unravel_index(np.argmax(s_d_masked), s_d_abs.shape)
    s_d_max = s_d_abs[max_idx]
    q_max = q1_vals[max_idx[0]] * b1 + q2_vals[max_idx[1]] * b2
    
    # VBS order parameter
    m_vbs = np.sqrt(np.abs(s_d_max) / n_bonds)
    
    return {
        's_d_2d': s_d_2d,
        'q1_vals': q1_vals,
        'q2_vals': q2_vals,
        'm_vbs': m_vbs,
        'q_max': q_max,
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
    This forms a 5-site structure:
    
           s2                 s4
          /  \\               /  \\
        s1----s0----s3     or similar
              (shared vertex)
    
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
    
    For a 5-site bowtie (2 triangles sharing a vertex), we compute triangle-based
    order parameters on each constituent triangle.
    
    P_tri = <S^+_a S^-_b S^+_c + h.c.>  (3-site chiral term)
    
    S_P(q) = (1/N_bt) Σ_{bt,bt'} exp(iq·(R_bt - R_{bt'})) <δP_bt δP_{bt'}>
    where δP_bt = P_bt - <P>
    
    Bragg peaks in S_P(q) indicate plaquette/bond crystal.
    Large isotropic S_P(q=0) indicates liquid-like resonance.
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
    
    # Compute resonance expectation for each bow-tie
    # For 5-site bowtie, compute sum of chiral terms on both triangles
    P_r = {}
    centers = {}
    
    for idx, bowtie in enumerate(bowties):
        s0, s1, s2, s3, s4, center = bowtie
        key = (s0, s1, s2, s3, s4)
        
        # Triangle 1: (s0, s1, s2) - compute 3-site resonance
        # Triangle 2: (s0, s3, s4) - compute 3-site resonance
        # Bowtie order = correlation between the two triangles
        P_tri1 = compute_triangle_chiral(psi, n_sites, s0, s1, s2)
        P_tri2 = compute_triangle_chiral(psi, n_sites, s0, s3, s4)
        
        # Bowtie order: product of chiral terms (resonance correlation)
        P_r[key] = P_tri1 * np.conj(P_tri2) + np.conj(P_tri1) * P_tri2
        centers[key] = center
        
        if (idx + 1) % 10 == 0:
            print(f"    Computed {idx + 1}/{n_plaquettes} plaquettes", end='\r')
    print()
    
    # Mean plaquette value
    P_mean = np.mean([np.real(P_r[k]) for k in P_r])
    
    # Connected plaquette correlations
    delta_P = {k: P_r[k] - P_mean for k in P_r}
    
    # Structure factor on q-grid
    q1_vals = np.linspace(-1, 1, n_q_points)
    q2_vals = np.linspace(-1, 1, n_q_points)
    s_p_2d = np.zeros((n_q_points, n_q_points), dtype=complex)
    
    plaquette_keys = list(P_r.keys())
    
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
    
    # Order parameters
    center_idx = n_q_points // 2
    s_p_q0 = np.abs(s_p_2d[center_idx, center_idx])  # q=0 value (resonance strength)
    
    # Find maximum away from q=0
    s_p_abs = np.abs(s_p_2d)
    s_p_masked = s_p_abs.copy()
    s_p_masked[center_idx-2:center_idx+3, center_idx-2:center_idx+3] = 0
    
    max_idx = np.unravel_index(np.argmax(s_p_masked), s_p_abs.shape)
    s_p_max = s_p_abs[max_idx]
    q_max = q1_vals[max_idx[0]] * b1 + q2_vals[max_idx[1]] * b2
    
    # Plaquette crystal order parameter
    m_plaquette = np.sqrt(np.abs(s_p_max) / n_plaquettes)
    
    # Resonance strength (liquid-like)
    resonance_strength = np.mean([np.abs(P_r[k]) for k in P_r])
    
    return {
        's_p_2d': s_p_2d,
        'q1_vals': q1_vals,
        'q2_vals': q2_vals,
        'm_plaquette': m_plaquette,
        'q_max': q_max,
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
                           'Order: $\\langle\\chi_1 \\chi_2^* + \\chi_1^* \\chi_2\\rangle$ where '
                           '$\\chi = S^+_a S^-_b S^+_c + h.c.$', 
                           fontsize=11, fontweight='bold')
    ax_schematic.axis('off')
    
    # Legend
    legend_elements = [
        Line2D([0], [0], color='blue', lw=3, label='NN bond'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, 
               label='Shared vertex'),
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
    
    print(f"Spatial plots saved to {output_dir}")


# =============================================================================
# Structure Factor Visualization Functions
# =============================================================================

def plot_structure_factors(results: Dict, output_dir: str):
    """Plot all computed structure factors"""
    if not HAS_MATPLOTLIB:
        print("Matplotlib not available, skipping plots")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Diagonal structure factor S^{zz}(q)
    if 'translation' in results and 's_q_2d' in results['translation']:
        data = results['translation']
        fig, ax = plt.subplots(figsize=(8, 6))
        extent = [data['q1_vals'][0], data['q1_vals'][-1], 
                  data['q2_vals'][0], data['q2_vals'][-1]]
        im = ax.imshow(np.abs(data['s_q_2d']).T, origin='lower', extent=extent,
                       cmap='viridis', aspect='equal')
        ax.set_xlabel('q₁ (units of b₁)')
        ax.set_ylabel('q₂ (units of b₂)')
        ax.set_title(f'S$^{{zz}}$(q) - Translation Order\nm_trans = {data["m_trans"]:.4f}')
        plt.colorbar(im, ax=ax, label='|S$^{zz}$(q)|')
        plt.savefig(os.path.join(output_dir, 'Szz_structure_factor.png'), dpi=150, bbox_inches='tight')
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
        print(f"    Q_max = ({data['q_max'][0]:.3f}, {data['q_max'][1]:.3f})")
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
# Main Function
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Compute BFG Kagome order parameters from wavefunction',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('wavefunction_file', type=str,
                        help='Path to HDF5 file containing eigenvectors')
    parser.add_argument('cluster_dir', type=str,
                        help='Directory containing cluster data (positions.dat, NN lists, etc.)')
    parser.add_argument('--eigenvector-index', '-i', type=int, default=0,
                        help='Index of eigenvector to analyze (default: 0 = ground state)')
    parser.add_argument('--output-dir', '-o', type=str, default='./bfg_order_params',
                        help='Output directory for plots and data')
    parser.add_argument('--n-q-points', '-n', type=int, default=50,
                        help='Number of q-points for structure factor grids')
    parser.add_argument('--skip-plaquette', action='store_true',
                        help='Skip plaquette calculation (slow for large systems)')
    parser.add_argument('--no-plots', action='store_true',
                        help='Skip generating plots')
    parser.add_argument('--geometry-only', action='store_true',
                        help='Only plot lattice geometry (no wavefunction analysis)')
    
    args = parser.parse_args()
    
    # Load cluster data
    print(f"Loading cluster data from {args.cluster_dir}...")
    cluster = load_kagome_cluster(args.cluster_dir)
    print(f"  Loaded {cluster['n_sites']} sites")
    print(f"  NN bonds: {len(cluster.get('edges_nn', []))}")
    print(f"  2NN bonds: {len(cluster.get('edges_2nn', []))}")
    print(f"  3NN bonds: {len(cluster.get('edges_3nn', []))}")
    
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
    print("\n[1/5] Computing diagonal structure factor S^{zz}(q)...")
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
        print("\nGenerating structure factor plots...")
        plot_structure_factors(results, args.output_dir)
        
        print("\nGenerating spatially-resolved plots...")
        plot_all_spatial_order_parameters(results, cluster, args.output_dir)
    
    print(f"\nDone! Results saved to {args.output_dir}")
    
    return results


if __name__ == "__main__":
    main()
