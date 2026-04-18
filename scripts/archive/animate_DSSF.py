#!/usr/bin/env python3
"""
Script to read spectral_beta_inf.dat files across all h=# directories
and create animated/stacked plots for each species as a function of h.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import os
import glob
from pathlib import Path
import re
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

# Configuration
BASE_DIR = "/scratch/zhouzb79/DSSF_PCD_mag_field_sweep_pi_flux"
OUTPUT_DIR = os.path.join(BASE_DIR, "spectral_animations")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Create organized subdirectories
SUBDIRS = {
    'individual': os.path.join(OUTPUT_DIR, "1_individual_species"),
    'do_channels': os.path.join(OUTPUT_DIR, "2_DO_channels"),
    'combined': os.path.join(OUTPUT_DIR, "3_combined_SmSp_SzSz"),
    'sublattice': os.path.join(OUTPUT_DIR, "4_sublattice_correlations"),
    'transverse': os.path.join(OUTPUT_DIR, "5_global_transverse"),
    'transverse_experimental': os.path.join(OUTPUT_DIR, "5a_global_transverse_experimental"),
    'magnetization': os.path.join(OUTPUT_DIR, "6_magnetization"),
    'summary': os.path.join(OUTPUT_DIR, "0_summary")
}

# Create all subdirectories
for subdir in SUBDIRS.values():
    os.makedirs(subdir, exist_ok=True)

# Conversion factors
# For h values (magnetic field)
H_CONVERSION_FACTOR = 0.063 / (2.5 * 0.0578)
# For omega values (energy/frequency): Jzz = 0.063 meV in real material
ENERGY_CONVERSION_FACTOR = 0.063  # converts from Jzz units to meV

# Frequency range limits (in Jzz units)
FREQ_MIN = -0.5
FREQ_MAX = 4.0

# Sublattice grouping configuration
# Define which sublattice indices belong to set A and set B
SUBLATTICE_A = [0]  # Modify as needed
SUBLATTICE_B = [1, 2, 3]  # Modify as needed

# Pyrochlore local frame definitions (for global magnetization calculation)
# For pyrochlore lattice, each of the 4 sublattices has a local coordinate frame
# z-axis (local quantization axis)
z_local = np.array([
    [1, 1, 1] / np.sqrt(3),      # Sublattice 0
    [1, -1, -1] / np.sqrt(3),    # Sublattice 1
    [-1, 1, -1] / np.sqrt(3),    # Sublattice 2
    [-1, -1, 1] / np.sqrt(3)     # Sublattice 3
])

# x-axis (local frame)
x_local = np.array([
    [-2, 1, 1] / np.sqrt(6),     # Sublattice 0
    [-2, -1, -1] / np.sqrt(6),   # Sublattice 1
    [2, 1, -1] / np.sqrt(6),     # Sublattice 2
    [2, -1, 1] / np.sqrt(6)      # Sublattice 3
])

# y-axis (local frame)
y_local = np.array([
    [0, -1, 1] / np.sqrt(2),     # Sublattice 0
    [0, 1, -1] / np.sqrt(2),     # Sublattice 1
    [0, -1, -1] / np.sqrt(2),    # Sublattice 2
    [0, 1, 1] / np.sqrt(2)       # Sublattice 3
])

# Stack into a single array: localframe[component, sublattice, xyz]
localframe = np.array([x_local, y_local, z_local])

# Magnetic field direction (normalized)
# Specify the direction of the applied magnetic field in global coordinates
# Common choices:
# [0, 0, 1] for field along z-axis
# [1, 1, 1]/sqrt(3) for field along [111] direction
# [1, 1, 0]/sqrt(2) for field along [110] direction
# [1, -1, 0]/sqrt(2) for field along [1-10] direction
FIELD_DIRECTION = np.array([1, 1, 1])  # Default: [111] direction
FIELD_DIRECTION = FIELD_DIRECTION / np.linalg.norm(FIELD_DIRECTION)  # Normalize

# Pyrochlore lattice local z-axes (sublattice quantization axes)
# These are the <111> directions for the four sublattices
PYROCHLORE_Z_AXES = np.array([
    [1, 1, 1],      # sublattice 0
    [-1, -1, 1],    # sublattice 1
    [-1, 1, -1],    # sublattice 2
    [1, -1, -1]     # sublattice 3
]) / np.sqrt(3)  # Normalize

# Q-vector for transverse operator (default: (0,0,0), can be modified)
Q_VECTOR = np.array([0, 0, 0])

# Factor to apply to SmSp when calculating total
SMSP_FACTOR = 0.5

# Experimental angle (will be calculated from magnetization data at highest field)
# This angle rotates the local frame such that s_local_rotated_x = 0
EXPERIMENTAL_ANGLE = None  # Will be set after computing magnetization

def calculate_experimental_angle(h_values, h_dirs):
    """
    Calculate the experimental angle by requiring that s_local_rotated_x is uniformly zero.
    
    This function computes the angle θ such that:
    s_local_rotated_x = cos(θ) * s_local_x - sin(θ) * s_local_z = 0
    
    Therefore: tan(θ) = s_local_x / s_local_z
    Or: θ = arctan(s_local_x / s_local_z)
    
    We compute this at the largest field value only.
    
    Returns:
    - theta: experimental angle in radians
    """
    print("\n" + "="*80)
    print("COMPUTING EXPERIMENTAL ANGLE FROM MAGNETIZATION")
    print("="*80)
    
    # Use only the largest field value
    h_max = max(h_values)
    h_dir = h_dirs[h_max]
    
    file_path = os.path.join(h_dir, "structure_factor_results", "beta_inf", "spin_configuration.txt")
    
    if not os.path.exists(file_path):
        print(f"  ⚠ Spin configuration not found for h={h_max}! Using default angle.")
        return 0.31416104734  # Fallback to previous value
    
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()[1:]  # Skip header
        
        sp_values = []
        sm_values = []
        sz_values = []
        
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 4:
                # Parse S+, S-, Sz
                sp_str = parts[1].strip('()').split(',')
                sp_complex = complex(float(sp_str[0]), float(sp_str[1]))
                sp_values.append(sp_complex)
                
                sm_str = parts[2].strip('()').split(',')
                sm_complex = complex(float(sm_str[0]), float(sm_str[1]))
                sm_values.append(sm_complex)
                
                sz_str = parts[3].strip('()').split(',')
                sz_real = float(sz_str[0])
                sz_values.append(sz_real)
        
        sp_values = np.array(sp_values)
        sm_values = np.array(sm_values)
        sz_values = np.array(sz_values)
        
        # Calculate Sx from S+ and S-
        sx_values = (sp_values + sm_values) / 2.0
        
        # Total magnetization (local frame)
        mag_x = np.sum(sx_values.real)
        mag_z = np.sum(sz_values)
        
        # Calculate angle
        if np.abs(mag_z) < 1e-10:  # Avoid division by zero
            print(f"  ⚠ Mz too small at largest field! Using default angle.")
            return 0.31416104734
        
        theta = np.arctan2(mag_x, mag_z)
        h_converted = h_max * H_CONVERSION_FACTOR
        
        print(f"  Using largest field: h={h_converted:.4f} T (h={h_max:.4f} in code units)")
        print(f"  Mx = {mag_x:.6f}")
        print(f"  Mz = {mag_z:.6f}")
        print(f"  θ = arctan2(Mx, Mz) = {np.degrees(theta):.4f}°")
        print(f"  θ (radians) = {theta:.10f}")
        print(f"\n  Verification: cos(θ)*Mx - sin(θ)*Mz = {np.cos(theta)*mag_x - np.sin(theta)*mag_z:.10f}")
        print("="*80)
        
        return theta
    
    except Exception as e:
        print(f"  ✗ Error processing magnetization data: {e}")
        import traceback
        traceback.print_exc()
        return 0.31416104734  # Fallback

def extract_h_value(h_dir):
    """Extract numerical h value from directory name like 'h=0.1'"""
    match = re.search(r'h=([0-9.]+)', h_dir)
    if match:
        return float(match.group(1))
    return None

def find_all_h_directories():
    """Find all h=# directories and sort them by h value"""
    h_dirs = glob.glob(os.path.join(BASE_DIR, "h=*"))
    h_dirs = [d for d in h_dirs if os.path.isdir(d)]
    
    # Extract h values and sort
    h_data = []
    for d in h_dirs:
        h_val = extract_h_value(d)
        if h_val is not None:
            h_data.append((h_val, d))
    
    h_data.sort(key=lambda x: x[0])
    return h_data

def find_all_species(h_dir):
    """Find all species directories in a given h directory"""
    processed_data_dir = os.path.join(h_dir, "structure_factor_results", "processed_data")
    if not os.path.exists(processed_data_dir):
        return []
    
    species_dirs = glob.glob(os.path.join(processed_data_dir, "*"))
    species_dirs = [d for d in species_dirs if os.path.isdir(d)]
    species_names = [os.path.basename(d) for d in species_dirs]
    
    return sorted(species_names)

def read_spectral_data(h_dir, species):
    """Read spectral_beta_inf.dat file for a given h and species"""
    file_path = os.path.join(h_dir, "structure_factor_results", "processed_data", 
                              species, "spectral_beta_inf.dat")
    
    if not os.path.exists(file_path):
        return None, None
    
    try:
        data = np.loadtxt(file_path)
        freq = data[:, 0]
        spectral = data[:, 1]
        return freq, spectral
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None, None

def parse_sublattice_indices(species_name):
    """Extract sublattice indices from species name like 'SmSp_q_Qx0_Qy0_Qz0_sub2_sub3'"""
    match = re.search(r'_sub(\d+)_sub(\d+)', species_name)
    if match:
        return int(match.group(1)), int(match.group(2))
    return None, None

def get_base_species_name(species_name):
    """Remove sublattice suffix from species name"""
    # Remove _subX_subY pattern
    return re.sub(r'_sub\d+_sub\d+$', '', species_name)

def find_sublattice_species(h_dir):
    """Find all species with sublattice information"""
    all_species = find_all_species(h_dir)
    sublattice_species = [sp for sp in all_species if '_sub' in sp]
    return sublattice_species

def calculate_transverse_weight(sub1, sub2, Q=None):
    """
    Calculate the transverse operator weight for a sublattice pair.
    
    Transverse operator: (z_mu · z_nu - (z_mu · Q̂)(z_nu · Q̂))
    
    Parameters:
    - sub1, sub2: sublattice indices (0-3 for pyrochlore)
    - Q: Q-vector (defaults to global Q_VECTOR)
    
    Returns:
    - weight: transverse operator weight
    """
    if Q is None:
        Q = Q_VECTOR
    
    # Get sublattice z-axes
    z_mu = PYROCHLORE_Z_AXES[sub1]
    z_nu = PYROCHLORE_Z_AXES[sub2]
    
    # Calculate z_mu · z_nu
    dot_product = np.dot(z_mu, z_nu)
    
    # If Q is zero vector, second term vanishes
    Q_norm = np.linalg.norm(Q)
    if Q_norm < 1e-10:
        return dot_product
    
    # Normalize Q
    Q_hat = Q / Q_norm
    
    # Calculate (z_mu · Q̂)(z_nu · Q̂)
    z_mu_dot_Q = np.dot(z_mu, Q_hat)
    z_nu_dot_Q = np.dot(z_nu, Q_hat)
    
    # Return transverse weight
    return dot_product - z_mu_dot_Q * z_nu_dot_Q

def create_sublattice_correlation_data(h_values, h_dirs, base_species_pattern):
    """
    Create AA, BB, and AB correlation data by summing over sublattice pairs.
    
    Parameters:
    - h_values: list of h values
    - h_dirs: dict mapping h -> directory path
    - base_species_pattern: base pattern like "SmSp_q_Qx0_Qy0_Qz0" or "SzSz_q_Qx0_Qy0_Qz6.28319"
    
    Returns:
    - Three dicts (freq_data, spectral_data) for AA, BB, AB correlations
    """
    freq_data_aa = {}
    spectral_data_aa = {}
    freq_data_bb = {}
    spectral_data_bb = {}
    freq_data_ab = {}
    spectral_data_ab = {}
    
    for h in h_values:
        h_dir = h_dirs[h]
        
        # Find all sublattice species matching the base pattern
        all_species = find_all_species(h_dir)
        matching_species = [sp for sp in all_species if sp.startswith(base_species_pattern) and '_sub' in sp]
        
        # Initialize accumulators
        aa_spectral = None
        bb_spectral = None
        ab_spectral = None
        freq_ref = None
        
        for species in matching_species:
            sub1, sub2 = parse_sublattice_indices(species)
            if sub1 is None or sub2 is None:
                continue
            
            freq, spectral = read_spectral_data(h_dir, species)
            if freq is None or spectral is None:
                continue
            
            # Set reference frequency grid
            if freq_ref is None:
                freq_ref = freq
                aa_spectral = np.zeros_like(spectral)
                bb_spectral = np.zeros_like(spectral)
                ab_spectral = np.zeros_like(spectral)
            
            # Interpolate to reference grid if needed
            if len(freq) != len(freq_ref) or not np.allclose(freq, freq_ref):
                spectral = np.interp(freq_ref, freq, spectral)
            
            # Categorize correlation
            if sub1 in SUBLATTICE_A and sub2 in SUBLATTICE_A:
                # AA correlation
                aa_spectral += spectral
            elif sub1 in SUBLATTICE_B and sub2 in SUBLATTICE_B:
                # BB correlation
                bb_spectral += spectral
            elif (sub1 in SUBLATTICE_A and sub2 in SUBLATTICE_B) or \
                 (sub1 in SUBLATTICE_B and sub2 in SUBLATTICE_A):
                # AB correlation (cross-correlation)
                ab_spectral += spectral
        
        # Store results if we found any data
        if freq_ref is not None:
            freq_data_aa[h] = freq_ref
            spectral_data_aa[h] = aa_spectral
            freq_data_bb[h] = freq_ref
            spectral_data_bb[h] = bb_spectral
            freq_data_ab[h] = freq_ref
            spectral_data_ab[h] = ab_spectral
    
    return (freq_data_aa, spectral_data_aa), \
           (freq_data_bb, spectral_data_bb), \
           (freq_data_ab, spectral_data_ab)

def create_global_transverse_sublattice_data(h_values, h_dirs, base_species_pattern):
    """
    Create AA, BB, and AB correlation data with transverse operator weighting.
    
    Applies the transverse operator: (z_mu · z_nu - (z_mu · Q̂)(z_nu · Q̂))
    to weight each sublattice pair correlation.
    
    Parameters:
    - h_values: list of h values
    - h_dirs: dict mapping h -> directory path
    - base_species_pattern: base pattern like "SmSp_q_Qx0_Qy0_Qz0" or "SzSz_q_Qx0_Qy0_Qz6.28319"
    
    Returns:
    - Three dicts (freq_data, spectral_data) for transverse-weighted AA, BB, AB correlations
    """
    freq_data_aa = {}
    spectral_data_aa = {}
    freq_data_bb = {}
    spectral_data_bb = {}
    freq_data_ab = {}
    spectral_data_ab = {}
    
    for h in h_values:
        h_dir = h_dirs[h]
        
        # Find all sublattice species matching the base pattern
        all_species = find_all_species(h_dir)
        matching_species = [sp for sp in all_species if sp.startswith(base_species_pattern) and '_sub' in sp]
        
        # Initialize accumulators
        aa_spectral = None
        bb_spectral = None
        ab_spectral = None
        freq_ref = None
        
        for species in matching_species:
            sub1, sub2 = parse_sublattice_indices(species)
            if sub1 is None or sub2 is None:
                continue
            
            freq, spectral = read_spectral_data(h_dir, species)
            if freq is None or spectral is None:
                continue
            
            # Calculate transverse weight for this sublattice pair
            weight = calculate_transverse_weight(sub1, sub2)
            
            # Set reference frequency grid
            if freq_ref is None:
                freq_ref = freq
                aa_spectral = np.zeros_like(spectral)
                bb_spectral = np.zeros_like(spectral)
                ab_spectral = np.zeros_like(spectral)
            
            # Interpolate to reference grid if needed
            if len(freq) != len(freq_ref) or not np.allclose(freq, freq_ref):
                spectral = np.interp(freq_ref, freq, spectral)
            
            # Apply transverse weight and categorize correlation
            weighted_spectral = spectral * weight
            
            if sub1 in SUBLATTICE_A and sub2 in SUBLATTICE_A:
                # AA correlation
                aa_spectral += weighted_spectral
            elif sub1 in SUBLATTICE_B and sub2 in SUBLATTICE_B:
                # BB correlation
                bb_spectral += weighted_spectral
            elif (sub1 in SUBLATTICE_A and sub2 in SUBLATTICE_B) or \
                 (sub1 in SUBLATTICE_B and sub2 in SUBLATTICE_A):
                # AB correlation (cross-correlation)
                ab_spectral += weighted_spectral
        
        # Store results if we found any data
        if freq_ref is not None:
            freq_data_aa[h] = freq_ref
            spectral_data_aa[h] = aa_spectral
            freq_data_bb[h] = freq_ref
            spectral_data_bb[h] = bb_spectral
            freq_data_ab[h] = freq_ref
            spectral_data_ab[h] = ab_spectral
    
    return (freq_data_aa, spectral_data_aa), \
           (freq_data_bb, spectral_data_bb), \
           (freq_data_ab, spectral_data_ab)


def calculate_experimental_transverse_weight(sub1, sub2, theta, Q=None):
    """
    Calculate the experimental transverse operator weight for a sublattice pair.
    
    This combines the transverse operator with the experimental angle rotation:
    
    Transverse operator: (z_mu · z_nu - (z_mu · Q̂)(z_nu · Q̂))
    Experimental rotation: cos²(θ) SzSz + sin²(θ) SxSx + sin(θ)cos(θ)(SxSz + SzSx)
    
    The combined weight is computed as:
    W_exp_trans(μ,ν) = W_trans(μ,ν) for the SzSz component
    
    For experimental channel, we need to apply the rotation to the transverse weight.
    The total contribution is:
    S_exp_trans = Σ_μν [cos²θ * w_μν * SzSz_μν + sin²θ * w_μν * SxSx_μν + sinθcosθ * w_μν * (SxSz + SzSx)_μν]
    
    where w_μν is the transverse weight.
    
    Parameters:
    - sub1, sub2: sublattice indices (0-3 for pyrochlore)
    - theta: experimental angle in radians
    - Q: Q-vector (defaults to global Q_VECTOR)
    
    Returns:
    - weight_dict: dictionary with weights for 'SzSz', 'SxSx', 'SxSz' components
    """
    if Q is None:
        Q = Q_VECTOR
    
    # Get sublattice z-axes
    z_mu = PYROCHLORE_Z_AXES[sub1]
    z_nu = PYROCHLORE_Z_AXES[sub2]
    
    # Calculate z_mu · z_nu (base transverse weight)
    dot_product = np.dot(z_mu, z_nu)
    
    # If Q is zero vector, second term vanishes
    Q_norm = np.linalg.norm(Q)
    if Q_norm < 1e-10:
        trans_weight = dot_product
    else:
        # Normalize Q
        Q_hat = Q / Q_norm
        
        # Calculate (z_mu · Q̂)(z_nu · Q̂)
        z_mu_dot_Q = np.dot(z_mu, Q_hat)
        z_nu_dot_Q = np.dot(z_nu, Q_hat)
        
        # Transverse weight
        trans_weight = dot_product - z_mu_dot_Q * z_nu_dot_Q
    
    # Apply experimental rotation coefficients
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    cos2_theta = cos_theta**2
    sin2_theta = sin_theta**2
    sin_cos_theta = sin_theta * cos_theta
    
    return {
        'SzSz': trans_weight * cos2_theta,
        'SxSx': trans_weight * sin2_theta,
        'SxSz': trans_weight * 2 * sin_cos_theta
    }


def create_global_transverse_experimental_sublattice_data(h_values, h_dirs, base_species_pattern, theta):
    """
    Create AA, BB, and AB correlation data with transverse operator weighting
    AND experimental channel rotation.
    
    This combines:
    1. Transverse operator: (z_mu · z_nu - (z_mu · Q̂)(z_nu · Q̂))
    2. Experimental rotation: cos²(θ) SzSz + sin²(θ) SxSx + sin(θ)cos(θ)(SxSz + SzSx)
    
    Parameters:
    - h_values: list of h values
    - h_dirs: dict mapping h -> directory path
    - base_species_pattern: base pattern like "SmSp_q_Qx0_Qy0_Qz0" or "SzSz_q_Qx0_Qy0_Qz6.28319"
    - theta: experimental angle in radians
    
    Returns:
    - Three dicts (freq_data, spectral_data) for transverse-experimental-weighted AA, BB, AB correlations
    """
    freq_data_aa = {}
    spectral_data_aa = {}
    freq_data_bb = {}
    spectral_data_bb = {}
    freq_data_ab = {}
    spectral_data_ab = {}
    
    for h in h_values:
        h_dir = h_dirs[h]
        
        # Find all sublattice species matching the base pattern
        all_species = find_all_species(h_dir)
        matching_species = [sp for sp in all_species if sp.startswith(base_species_pattern) and '_sub' in sp]
        
        # Initialize accumulators
        aa_spectral = None
        bb_spectral = None
        ab_spectral = None
        freq_ref = None
        
        for species in matching_species:
            sub1, sub2 = parse_sublattice_indices(species)
            if sub1 is None or sub2 is None:
                continue
            
            freq, spectral = read_spectral_data(h_dir, species)
            if freq is None or spectral is None:
                continue
            
            # Calculate experimental transverse weights for this sublattice pair
            weights = calculate_experimental_transverse_weight(sub1, sub2, theta)
            
            # For the base species pattern (SzSz or SmSp), apply the SzSz weight
            # The full experimental channel would require SxSx and SxSz data too,
            # but here we approximate using only the SzSz component with the transverse weight
            # multiplied by cos²(θ), which is the dominant term at small angles
            weight = weights['SzSz'] + weights['SxSx'] + weights['SxSz']
            
            # Alternative: use only transverse weight (like before) but with note about experimental
            # For now, we use the sum which approximates the full experimental channel
            # when the different Cartesian components have similar spectral weights
            
            # Set reference frequency grid
            if freq_ref is None:
                freq_ref = freq
                aa_spectral = np.zeros_like(spectral)
                bb_spectral = np.zeros_like(spectral)
                ab_spectral = np.zeros_like(spectral)
            
            # Interpolate to reference grid if needed
            if len(freq) != len(freq_ref) or not np.allclose(freq, freq_ref):
                spectral = np.interp(freq_ref, freq, spectral)
            
            # Apply experimental transverse weight and categorize correlation
            weighted_spectral = spectral * weight
            
            if sub1 in SUBLATTICE_A and sub2 in SUBLATTICE_A:
                # AA correlation
                aa_spectral += weighted_spectral
            elif sub1 in SUBLATTICE_B and sub2 in SUBLATTICE_B:
                # BB correlation
                bb_spectral += weighted_spectral
            elif (sub1 in SUBLATTICE_A and sub2 in SUBLATTICE_B) or \
                 (sub1 in SUBLATTICE_B and sub2 in SUBLATTICE_A):
                # AB correlation (cross-correlation)
                ab_spectral += weighted_spectral
        
        # Store results if we found any data
        if freq_ref is not None:
            freq_data_aa[h] = freq_ref
            spectral_data_aa[h] = aa_spectral
            freq_data_bb[h] = freq_ref
            spectral_data_bb[h] = bb_spectral
            freq_data_ab[h] = freq_ref
            spectral_data_ab[h] = ab_spectral
    
    return (freq_data_aa, spectral_data_aa), \
           (freq_data_bb, spectral_data_bb), \
           (freq_data_ab, spectral_data_ab)


def read_spin_configuration(h_dir):
    """Read spin configuration file and calculate magnetization including Sx, Sy, Sz components
    Also calculates global magnetization in the pyrochlore lattice global coordinate frame"""
    file_path = os.path.join(h_dir, "structure_factor_results", "beta_inf", "spin_configuration.txt")
    
    if not os.path.exists(file_path):
        return None
    
    try:
        # Extract S+, S-, Sz values from format (real,imag)
        with open(file_path, 'r') as f:
            lines = f.readlines()[1:]  # Skip header
        
        sp_values = []  # S+ values
        sm_values = []  # S- values
        sz_values = []  # Sz values
        site_indices = []  # Site indices to determine sublattice
        
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 4:
                # Site index (column 0)
                site_idx = int(parts[0])
                site_indices.append(site_idx)
                
                # Parse S+ (column 1), S- (column 2), Sz (column 3)
                # All are in format (real,imag)
                
                # S+ (parts[1])
                sp_str = parts[1].strip('()').split(',')
                sp_complex = complex(float(sp_str[0]), float(sp_str[1]))
                sp_values.append(sp_complex)
                
                # S- (parts[2])
                sm_str = parts[2].strip('()').split(',')
                sm_complex = complex(float(sm_str[0]), float(sm_str[1]))
                sm_values.append(sm_complex)
                
                # Sz (parts[3])
                sz_str = parts[3].strip('()').split(',')
                sz_real = float(sz_str[0])
                sz_values.append(sz_real)
        
        # Convert to numpy arrays
        sp_values = np.array(sp_values)
        sm_values = np.array(sm_values)
        sz_values = np.array(sz_values)
        site_indices = np.array(site_indices)
        
        # Calculate Sx and Sy from S+ and S-
        # Sx = (S+ + S-) / 2
        # Sy = (S+ - S-) / (2i) = -i(S+ - S-) / 2
        sx_values = (sp_values + sm_values) / 2.0
        sy_values = -1j * (sp_values - sm_values) / 2.0
        
        # Calculate total magnetization for each component (local frame)
        mag_x = np.sum(sx_values.real)  # Take real part of Sx
        mag_y = np.sum(sy_values.real)  # Take real part of Sy
        mag_z = np.sum(sz_values)
        
        # Calculate total magnitude of magnetization (local frame)
        mag_total = np.sqrt(mag_x**2 + mag_y**2 + mag_z**2)
        
        # Calculate per-site magnetizations (local frame)
        n_sites = len(sz_values)
        mag_x_per_site = mag_x / n_sites
        mag_y_per_site = mag_y / n_sites
        mag_z_per_site = mag_z / n_sites
        mag_total_per_site = mag_total / n_sites
        
        # Calculate GLOBAL magnetization (pyrochlore lattice global frame)
        # Assume sites cycle through sublattices 0,1,2,3,0,1,2,3,...
        n_sites_per_sublattice = n_sites // 4
        
        # Initialize global magnetization vector
        mag_global = np.zeros(3)
        
        for i in range(n_sites):
            sublattice = i % 4  # Determine which sublattice (0, 1, 2, or 3)
            
            # Local spin components (real parts)
            s_local = np.array([
                sx_values[i].real,
                sy_values[i].real,
                sz_values[i]
            ])
            
            # Transform to global coordinates using local frame basis
            # s_global = sum_alpha s_local[alpha] * localframe[alpha, sublattice, :]
            s_global = (s_local[0] * localframe[0, sublattice, :] +
                       s_local[1] * localframe[1, sublattice, :] +
                       s_local[2] * localframe[2, sublattice, :])
            
            mag_global += s_global
        
        # Calculate global magnetization magnitude and components
        mag_global_x = mag_global[0]
        mag_global_y = mag_global[1]
        mag_global_z = mag_global[2]
        mag_global_total = np.linalg.norm(mag_global)
        
        # Per-site global magnetization
        mag_global_x_per_site = mag_global_x / n_sites
        mag_global_y_per_site = mag_global_y / n_sites
        mag_global_z_per_site = mag_global_z / n_sites
        mag_global_total_per_site = mag_global_total / n_sites
        
        # Calculate magnetization component along the field direction
        # M_parallel = M_global · field_direction
        mag_along_field = np.dot(mag_global, FIELD_DIRECTION)
        mag_along_field_per_site = mag_along_field / n_sites
        
        return {
            # Local frame magnetizations
            'total_magnetization': mag_total,
            'magnetization_per_site': mag_total_per_site,
            'mag_x': mag_x,
            'mag_y': mag_y,
            'mag_z': mag_z,
            'mag_x_per_site': mag_x_per_site,
            'mag_y_per_site': mag_y_per_site,
            'mag_z_per_site': mag_z_per_site,
            # Global frame magnetizations
            'mag_global_x': mag_global_x,
            'mag_global_y': mag_global_y,
            'mag_global_z': mag_global_z,
            'mag_global_total': mag_global_total,
            'mag_global_x_per_site': mag_global_x_per_site,
            'mag_global_y_per_site': mag_global_y_per_site,
            'mag_global_z_per_site': mag_global_z_per_site,
            'mag_global_total_per_site': mag_global_total_per_site,
            # Magnetization along field direction
            'mag_along_field': mag_along_field,
            'mag_along_field_per_site': mag_along_field_per_site,
            # Raw data
            'num_sites': n_sites,
            'sx_values': sx_values,
            'sy_values': sy_values,
            'sz_values': sz_values
        }
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        import traceback
        traceback.print_exc()
        return None

def create_do_channel(h_values, h_dirs, base_species):
    """
    Create DO (Difference Orbital) channel by summing SF and NSF channels.
    Returns freq_data and spectral_data dictionaries for the DO channel.
    """
    freq_data_do = {}
    spectral_data_do = {}
    
    sf_species = base_species + "_SF"
    nsf_species = base_species + "_NSF"
    
    for h in h_values:
        h_dir = h_dirs[h]
        
        # Read SF channel
        freq_sf, spectral_sf = read_spectral_data(h_dir, sf_species)
        # Read NSF channel
        freq_nsf, spectral_nsf = read_spectral_data(h_dir, nsf_species)
        
        if freq_sf is not None and spectral_sf is not None and \
           freq_nsf is not None and spectral_nsf is not None:
            
            # Check if frequencies match
            if len(freq_sf) == len(freq_nsf) and np.allclose(freq_sf, freq_nsf):
                # Direct sum if frequencies match
                freq_data_do[h] = freq_sf
                spectral_data_do[h] = spectral_sf + spectral_nsf
            else:
                # Interpolate NSF to SF grid if they don't match
                spectral_nsf_interp = np.interp(freq_sf, freq_nsf, spectral_nsf)
                freq_data_do[h] = freq_sf
                spectral_data_do[h] = spectral_sf + spectral_nsf_interp
    
    return freq_data_do, spectral_data_do

def create_stacked_plot(h_values, freq_data, spectral_data, species, output_file):
    """Create a 2D stacked plot showing spectral function vs frequency for different h values"""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Use a colormap
    colors = cm.viridis(np.linspace(0, 1, len(h_values)))
    
    # Plot each h value with an offset for visibility
    offset = 0
    offset_step = np.abs(np.mean([np.max(spec) - np.min(spec) for spec in spectral_data.values()])) * 1.2
    
    for i, h in enumerate(h_values):
        if h in spectral_data and h in freq_data:
            # Filter data to frequency range
            freq = freq_data[h]
            spec = spectral_data[h]
            mask = (freq >= FREQ_MIN) & (freq <= FREQ_MAX)
            # Convert frequency to meV
            freq_meV = freq[mask] * ENERGY_CONVERSION_FACTOR
            h_converted = h * H_CONVERSION_FACTOR
            ax.plot(freq_meV, spec[mask] + offset, 
                   label=f'h={h_converted:.2f}', color=colors[i], linewidth=1.5, alpha=0.8)
            offset += offset_step
    
    ax.set_xlabel('Energy (meV)', fontsize=12)
    ax.set_ylabel('Spectral Function (offset for clarity)', fontsize=12)
    ax.set_title(f'Spectral Function vs Energy - {species}', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=8, ncol=2)
    ax.set_xlim([FREQ_MIN * ENERGY_CONVERSION_FACTOR, FREQ_MAX * ENERGY_CONVERSION_FACTOR])
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved stacked plot: {output_file}")
    plt.close()

def create_3d_surface_plot(h_values, freq_data, spectral_data, species, output_file):
    """Create a 3D surface plot showing spectral function vs frequency and h"""
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Get a common frequency grid (use the first one as reference)
    if not h_values or h_values[0] not in freq_data:
        print(f"No data available for 3D plot of {species}")
        return
    
    # Filter frequency range to [-3, 6] (in Jzz units)
    freq_ref_full = freq_data[h_values[0]]
    mask = (freq_ref_full >= FREQ_MIN) & (freq_ref_full <= FREQ_MAX)
    freq_ref = freq_ref_full[mask]
    # Convert to meV
    freq_ref_meV = freq_ref * ENERGY_CONVERSION_FACTOR
    
    # Convert h values for plotting
    h_values_converted = [h * H_CONVERSION_FACTOR for h in h_values]
    
    # Create meshgrid
    H, FREQ = np.meshgrid(h_values_converted, freq_ref_meV)
    SPECTRAL = np.zeros_like(H)
    
    for i, h in enumerate(h_values):
        if h in spectral_data:
            # Filter data to frequency range
            freq_h = freq_data[h]
            spec_h = spectral_data[h]
            mask_h = (freq_h >= FREQ_MIN) & (freq_h <= FREQ_MAX)
            freq_h_filtered = freq_h[mask_h]
            spec_h_filtered = spec_h[mask_h]
            
            # Interpolate to common grid
            SPECTRAL[:, i] = np.interp(freq_ref, freq_h_filtered, spec_h_filtered)
    
    # Create surface plot
    surf = ax.plot_surface(H, FREQ, SPECTRAL, cmap=cm.viridis, 
                           linewidth=0, antialiased=True, alpha=0.8)
    
    ax.set_xlabel('Magnetic Field (h) [T]', fontsize=12)
    ax.set_ylabel('Energy (meV)', fontsize=12)
    ax.set_zlabel('Spectral Function', fontsize=12)
    ax.set_title(f'Spectral Function Surface - {species}', fontsize=14, fontweight='bold')
    
    # Add colorbar
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved 3D surface plot: {output_file}")
    plt.close()

def create_heatmap_plot(h_values, freq_data, spectral_data, species, output_file):
    """Create a 2D heatmap showing spectral function vs frequency and h"""
    # Get a common frequency grid
    if not h_values or h_values[0] not in freq_data:
        print(f"No data available for heatmap of {species}")
        return
    
    # Filter frequency range to [-3, 6] (in Jzz units)
    freq_ref_full = freq_data[h_values[0]]
    mask = (freq_ref_full >= FREQ_MIN) & (freq_ref_full <= FREQ_MAX)
    freq_ref = freq_ref_full[mask]
    
    # Create data matrix
    spectral_matrix = np.zeros((len(freq_ref), len(h_values)))
    
    for i, h in enumerate(h_values):
        if h in spectral_data:
            # Filter data to frequency range
            freq_h = freq_data[h]
            spec_h = spectral_data[h]
            mask_h = (freq_h >= FREQ_MIN) & (freq_h <= FREQ_MAX)
            freq_h_filtered = freq_h[mask_h]
            spec_h_filtered = spec_h[mask_h]
            
            # Interpolate to common grid
            spectral_matrix[:, i] = np.interp(freq_ref, freq_h_filtered, spec_h_filtered)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Convert h values and frequency for plotting
    h_values_converted = np.array(h_values) * H_CONVERSION_FACTOR
    freq_meV = freq_ref * ENERGY_CONVERSION_FACTOR
    
    # Create meshgrid for pcolormesh (edges needed)
    # Add boundary points for proper bin edges
    h_edges = np.zeros(len(h_values_converted) + 1)
    h_edges[0] = h_values_converted[0] - (h_values_converted[1] - h_values_converted[0]) / 2
    h_edges[-1] = h_values_converted[-1] + (h_values_converted[-1] - h_values_converted[-2]) / 2
    for i in range(1, len(h_values_converted)):
        h_edges[i] = (h_values_converted[i-1] + h_values_converted[i]) / 2
    
    freq_edges = np.zeros(len(freq_meV) + 1)
    freq_edges[0] = freq_meV[0] - (freq_meV[1] - freq_meV[0]) / 2
    freq_edges[-1] = freq_meV[-1] + (freq_meV[-1] - freq_meV[-2]) / 2
    for i in range(1, len(freq_meV)):
        freq_edges[i] = (freq_meV[i-1] + freq_meV[i]) / 2
    
    # Create heatmap with proper non-uniform spacing
    im = ax.pcolormesh(h_edges, freq_edges, spectral_matrix, 
                       cmap='viridis', shading='flat')
    
    ax.set_xlabel('Magnetic Field (h) [T]', fontsize=12)
    ax.set_ylabel('Energy (meV)', fontsize=12)
    ax.set_title(f'Spectral Function Heatmap - {species}', fontsize=14, fontweight='bold')
    
    # Add colorbar
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('Spectral Function', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved heatmap: {output_file}")
    plt.close()

def create_animation(h_values, freq_data, spectral_data, species, output_file):
    """Create an animation showing spectral function evolving with h"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Get y-axis limits
    all_spectral = [spectral_data[h] for h in h_values if h in spectral_data]
    if not all_spectral:
        print(f"No data available for animation of {species}")
        return
    
    ymin = min([np.min(s) for s in all_spectral])
    ymax = max([np.max(s) for s in all_spectral])
    y_range = ymax - ymin
    ymin -= 0.1 * y_range
    ymax += 0.1 * y_range
    
    # Set x-axis limits to [-3, 6] in Jzz units, converted to meV
    line, = ax.plot([], [], 'b-', linewidth=2)
    ax.set_xlim(FREQ_MIN * ENERGY_CONVERSION_FACTOR, FREQ_MAX * ENERGY_CONVERSION_FACTOR)
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel('Energy (meV)', fontsize=12)
    ax.set_ylabel('Spectral Function', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    title = ax.set_title('', fontsize=14, fontweight='bold')
    
    def init():
        line.set_data([], [])
        return line, title
    
    def animate(frame):
        h = h_values[frame]
        if h in freq_data and h in spectral_data:
            # Filter data to frequency range [-3, 6] (in Jzz units)
            freq = freq_data[h]
            spec = spectral_data[h]
            mask = (freq >= FREQ_MIN) & (freq <= FREQ_MAX)
            # Convert frequency to meV
            freq_meV = freq[mask] * ENERGY_CONVERSION_FACTOR
            line.set_data(freq_meV, spec[mask])
            h_converted = h * H_CONVERSION_FACTOR
            title.set_text(f'Spectral Function - {species} - h={h_converted:.3f} T')
        return line, title
    
    anim = FuncAnimation(fig, animate, init_func=init, 
                        frames=len(h_values), interval=200, blit=True)
    
    # Save animation
    writer = PillowWriter(fps=0.5)
    anim.save(output_file, writer=writer)
    print(f"Saved animation: {output_file}")
    plt.close()

def create_combined_component_plot_direct(h_values, all_freq_data, all_spectral_data, smsp_species, szsz_species, display_name, output_file):
    """
    Create a plot showing SmSp, SzSz, and their sum (Total) for given species.
    
    Parameters:
    - h_values: list of h values
    - all_freq_data: dict mapping species -> h -> freq
    - all_spectral_data: dict mapping species -> h -> spectral
    - smsp_species: full SmSp species name (e.g., "SmSp_q_Qx0_Qy0_Qz0_SF")
    - szsz_species: full SzSz species name (e.g., "SzSz_q_Qx0_Qy0_Qz0_SF")
    - display_name: name to display in plot title
    - output_file: path to save the plot
    """
    # Check if both components exist
    if smsp_species not in all_freq_data or szsz_species not in all_freq_data:
        print(f"  ⚠ Cannot create combined plot: missing {smsp_species} or {szsz_species}")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    # Select specific h values: 0T, closest to 0.3T, closest to 1T, closest to 2.5T
    target_fields_tesla = [0.0, 0.3, 1.0, 2.5]
    h_values_tesla = np.array(h_values) * H_CONVERSION_FACTOR
    
    selected_h_indices = []
    for target_field in target_fields_tesla:
        # Find the closest h value to the target field
        closest_idx = np.argmin(np.abs(h_values_tesla - target_field))
        selected_h_indices.append(closest_idx)
    
    for plot_idx, h_idx in enumerate(selected_h_indices):
        h = h_values[h_idx]
        ax = axes[plot_idx]
        
        # Get SmSp data
        if h in all_freq_data[smsp_species] and h in all_spectral_data[smsp_species]:
            freq_spm = all_freq_data[smsp_species][h]
            spec_spm = all_spectral_data[smsp_species][h]/2  # Apply factor to SmSp
            mask_spm = (freq_spm >= FREQ_MIN) & (freq_spm <= FREQ_MAX)
            freq_spm_meV = freq_spm[mask_spm] * ENERGY_CONVERSION_FACTOR
            spec_spm_filtered = spec_spm[mask_spm]
        else:
            freq_spm_meV = None
            spec_spm_filtered = None
        
        # Get SzSz data
        if h in all_freq_data[szsz_species] and h in all_spectral_data[szsz_species]:
            freq_szz = all_freq_data[szsz_species][h]
            spec_szz = all_spectral_data[szsz_species][h]
            mask_szz = (freq_szz >= FREQ_MIN) & (freq_szz <= FREQ_MAX)
            freq_szz_meV = freq_szz[mask_szz] * ENERGY_CONVERSION_FACTOR
            spec_szz_filtered = spec_szz[mask_szz]
        else:
            freq_szz_meV = None
            spec_szz_filtered = None
        
        # Plot individual components
        if freq_spm_meV is not None:
            ax.plot(freq_spm_meV, spec_spm_filtered, 'b-', label='SmSp', linewidth=2, alpha=0.7)
        if freq_szz_meV is not None:
            ax.plot(freq_szz_meV, spec_szz_filtered, 'r-', label='SzSz', linewidth=2, alpha=0.7)
        
        # Calculate and plot sum if both exist
        if freq_spm_meV is not None and freq_szz_meV is not None:
            # Interpolate to common grid if needed
            if len(freq_spm_meV) == len(freq_szz_meV) and np.allclose(freq_spm_meV, freq_szz_meV):
                spec_total = spec_spm_filtered + spec_szz_filtered
                freq_total = freq_spm_meV
            else:
                # Use Spm grid as reference
                spec_szz_interp = np.interp(freq_spm[mask_spm], freq_szz[mask_szz], spec_szz_filtered)
                spec_total = spec_spm_filtered + spec_szz_interp
                freq_total = freq_spm_meV
            
            ax.plot(freq_total, spec_total, 'g-', label='Total (SmSp+SzSz)', linewidth=2.5, alpha=0.9)
        
        h_converted = h * H_CONVERSION_FACTOR
        ax.set_xlabel('Energy (meV)', fontsize=11)
        ax.set_ylabel('Spectral Function', fontsize=11)
        ax.set_title(f'{display_name} - h={h_converted:.3f} T', fontsize=12, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([FREQ_MIN * ENERGY_CONVERSION_FACTOR, FREQ_MAX * ENERGY_CONVERSION_FACTOR])
    
    plt.suptitle(f'Spectral Components Comparison - {display_name}', 
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved combined component plot: {output_file}")
    plt.close()

def create_combined_heatmap_direct(h_values, all_freq_data, all_spectral_data, smsp_species, szsz_species, display_name, output_file):
    """
    Create a 3-panel heatmap showing SmSp, SzSz, and Total side by side.
    
    Parameters:
    - h_values: list of h values
    - all_freq_data: dict mapping species -> h -> freq
    - all_spectral_data: dict mapping species -> h -> spectral
    - smsp_species: full SmSp species name
    - szsz_species: full SzSz species name
    - display_name: name to display in plot title
    - output_file: path to save the plot
    """
    # Check if both components exist
    if smsp_species not in all_freq_data or szsz_species not in all_freq_data:
        print(f"  ⚠ Cannot create combined heatmap: missing {smsp_species} or {szsz_species}")
        return
    
    # Get a common frequency grid
    if not h_values or h_values[0] not in all_freq_data[smsp_species]:
        print(f"  ⚠ No data available for combined heatmap of {display_name}")
        return
    
    freq_ref_full = all_freq_data[smsp_species][h_values[0]]
    mask = (freq_ref_full >= FREQ_MIN) & (freq_ref_full <= FREQ_MAX)
    freq_ref = freq_ref_full[mask]
    
    # Create data matrices for SmSp, SzSz, and Total
    spm_matrix = np.zeros((len(freq_ref), len(h_values)))
    szz_matrix = np.zeros((len(freq_ref), len(h_values)))
    total_matrix = np.zeros((len(freq_ref), len(h_values)))
    
    for i, h in enumerate(h_values):
        # Process SmSp
        if h in all_spectral_data[smsp_species]:
            freq_h = all_freq_data[smsp_species][h]
            spec_h = all_spectral_data[smsp_species][h]
            mask_h = (freq_h >= FREQ_MIN) & (freq_h <= FREQ_MAX)
            spm_matrix[:, i] = np.interp(freq_ref, freq_h[mask_h], spec_h[mask_h])/2
        
        # Process SzSz
        if h in all_spectral_data[szsz_species]:
            freq_h = all_freq_data[szsz_species][h]
            spec_h = all_spectral_data[szsz_species][h]
            mask_h = (freq_h >= FREQ_MIN) & (freq_h <= FREQ_MAX)
            szz_matrix[:, i] = np.interp(freq_ref, freq_h[mask_h], spec_h[mask_h])
        
        # Calculate total
        total_matrix[:, i] = spm_matrix[:, i] + szz_matrix[:, i]
    
    # Create 3-panel figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Convert h values and frequency for plotting
    h_values_converted = np.array(h_values) * H_CONVERSION_FACTOR
    freq_meV = freq_ref * ENERGY_CONVERSION_FACTOR
    
    # Create meshgrid edges for pcolormesh (non-uniform spacing)
    h_edges = np.zeros(len(h_values_converted) + 1)
    h_edges[0] = h_values_converted[0] - (h_values_converted[1] - h_values_converted[0]) / 2
    h_edges[-1] = h_values_converted[-1] + (h_values_converted[-1] - h_values_converted[-2]) / 2
    for i in range(1, len(h_values_converted)):
        h_edges[i] = (h_values_converted[i-1] + h_values_converted[i]) / 2
    
    freq_edges = np.zeros(len(freq_meV) + 1)
    freq_edges[0] = freq_meV[0] - (freq_meV[1] - freq_meV[0]) / 2
    freq_edges[-1] = freq_meV[-1] + (freq_meV[-1] - freq_meV[-2]) / 2
    for i in range(1, len(freq_meV)):
        freq_edges[i] = (freq_meV[i-1] + freq_meV[i]) / 2
    
    # Determine common color scale for consistency
    vmin = min(spm_matrix.min(), szz_matrix.min(), total_matrix.min())
    vmax = max(spm_matrix.max(), szz_matrix.max(), total_matrix.max())
    
    # Plot SmSp
    im1 = axes[0].pcolormesh(h_edges, freq_edges, spm_matrix, 
                             cmap='viridis', shading='flat', vmin=vmin, vmax=vmax)
    axes[0].set_xlabel('Magnetic Field (h) [T]', fontsize=11)
    axes[0].set_ylabel('Energy (meV)', fontsize=11)
    axes[0].set_title('SmSp', fontsize=12, fontweight='bold')
    fig.colorbar(im1, ax=axes[0])
    
    # Plot SzSz
    im2 = axes[1].pcolormesh(h_edges, freq_edges, szz_matrix, 
                             cmap='viridis', shading='flat', vmin=vmin, vmax=vmax)
    axes[1].set_xlabel('Magnetic Field (h) [T]', fontsize=11)
    axes[1].set_ylabel('Energy (meV)', fontsize=11)
    axes[1].set_title('SzSz', fontsize=12, fontweight='bold')
    fig.colorbar(im2, ax=axes[1])
    
    # Plot Total
    im3 = axes[2].pcolormesh(h_edges, freq_edges, total_matrix, 
                             cmap='viridis', shading='flat', vmin=vmin, vmax=vmax)
    axes[2].set_xlabel('Magnetic Field (h) [T]', fontsize=11)
    axes[2].set_ylabel('Energy (meV)', fontsize=11)
    axes[2].set_title('Total (SmSp+SzSz)', fontsize=12, fontweight='bold')
    fig.colorbar(im3, ax=axes[2])
    
    plt.suptitle(f'Spectral Components Heatmap - {display_name}', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved combined heatmap: {output_file}")
    plt.close()

def create_combined_animation_direct(h_values, all_freq_data, all_spectral_data, smsp_species, szsz_species, display_name, output_file):
    """
    Create an animation showing SmSp, SzSz, and Total evolving with h.
    
    Parameters:
    - h_values: list of h values
    - all_freq_data: dict mapping species -> h -> freq
    - all_spectral_data: dict mapping species -> h -> spectral
    - smsp_species: full SmSp species name
    - szsz_species: full SzSz species name
    - display_name: name to display in plot title
    - output_file: path to save the animation
    """
    # Check if both components exist
    if smsp_species not in all_freq_data or szsz_species not in all_freq_data:
        print(f"  ⚠ Cannot create combined animation: missing {smsp_species} or {szsz_species}")
        return
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Get y-axis limits by checking all data
    all_smsp = [all_spectral_data[smsp_species][h] for h in h_values if h in all_spectral_data[smsp_species]]
    all_szsz = [all_spectral_data[szsz_species][h] for h in h_values if h in all_spectral_data[szsz_species]]
    
    if not all_smsp or not all_szsz:
        print(f"  ⚠ No data available for animation of {display_name}")
        return
    
    # Calculate y-limits considering both components and their sum
    ymin = min(min([np.min(s) for s in all_smsp]), min([np.min(s) for s in all_szsz]))
    ymax_smsp = max([np.max(s) for s in all_smsp])/2
    ymax_szsz = max([np.max(s) for s in all_szsz])
    ymax_total = ymax_smsp + ymax_szsz  # Approximate maximum of sum
    ymax = ymax_total
    
    y_range = ymax - ymin
    ymin -= 0.1 * y_range
    ymax += 0.1 * y_range
    
    # Create line objects
    line_smsp, = ax.plot([], [], 'b-', linewidth=2, alpha=0.7, label='SmSp')
    line_szsz, = ax.plot([], [], 'r-', linewidth=2, alpha=0.7, label='SzSz')
    line_total, = ax.plot([], [], 'g-', linewidth=2.5, alpha=0.9, label='Total (SmSp+SzSz)')
    
    ax.set_xlim(FREQ_MIN * ENERGY_CONVERSION_FACTOR, FREQ_MAX * ENERGY_CONVERSION_FACTOR)
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel('Energy (meV)', fontsize=12)
    ax.set_ylabel('Spectral Function', fontsize=12)
    ax.legend(loc='best', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    title = ax.set_title('', fontsize=14, fontweight='bold')
    
    def init():
        line_smsp.set_data([], [])
        line_szsz.set_data([], [])
        line_total.set_data([], [])
        return line_smsp, line_szsz, line_total, title
    
    def animate(frame):
        h = h_values[frame]
        h_converted = h * H_CONVERSION_FACTOR
        
        # Get SmSp data
        if h in all_freq_data[smsp_species] and h in all_spectral_data[smsp_species]:
            freq_smsp = all_freq_data[smsp_species][h]
            spec_smsp = all_spectral_data[smsp_species][h]/2
            mask_smsp = (freq_smsp >= FREQ_MIN) & (freq_smsp <= FREQ_MAX)
            freq_smsp_meV = freq_smsp[mask_smsp] * ENERGY_CONVERSION_FACTOR
            spec_smsp_filtered = spec_smsp[mask_smsp]
            line_smsp.set_data(freq_smsp_meV, spec_smsp_filtered)
        else:
            line_smsp.set_data([], [])
            freq_smsp_meV = None
            spec_smsp_filtered = None
        
        # Get SzSz data
        if h in all_freq_data[szsz_species] and h in all_spectral_data[szsz_species]:
            freq_szsz = all_freq_data[szsz_species][h]
            spec_szsz = all_spectral_data[szsz_species][h]
            mask_szsz = (freq_szsz >= FREQ_MIN) & (freq_szsz <= FREQ_MAX)
            freq_szsz_meV = freq_szsz[mask_szsz] * ENERGY_CONVERSION_FACTOR
            spec_szsz_filtered = spec_szsz[mask_szsz]
            line_szsz.set_data(freq_szsz_meV, spec_szsz_filtered)
        else:
            line_szsz.set_data([], [])
            freq_szsz_meV = None
            spec_szsz_filtered = None
        
        # Calculate and plot total
        if freq_smsp_meV is not None and freq_szsz_meV is not None:
            # Interpolate to common grid if needed
            if len(freq_smsp_meV) == len(freq_szsz_meV) and np.allclose(freq_smsp_meV, freq_szsz_meV):
                freq_total = freq_smsp_meV
                spec_total = spec_smsp_filtered + spec_szsz_filtered
            else:
                # Use the first frequency grid as reference
                freq_total = freq_smsp_meV
                spec_smsp_interp = spec_smsp_filtered
                spec_szsz_interp = np.interp(freq_smsp_meV, freq_szsz_meV, spec_szsz_filtered)
                spec_total = spec_smsp_interp + spec_szsz_interp
            
            line_total.set_data(freq_total, spec_total)
        else:
            line_total.set_data([], [])
        
        title.set_text(f'Spectral Components - {display_name} - h={h_converted:.3f} T')
        return line_smsp, line_szsz, line_total, title
    
    anim = FuncAnimation(fig, animate, init_func=init, 
                        frames=len(h_values), interval=200, blit=True)
    
    # Save animation
    writer = PillowWriter(fps=0.5)
    anim.save(output_file, writer=writer)
    print(f"  ✓ Saved combined animation: {output_file}")
    plt.close()

def create_sublattice_comparison_plot(h_values, freq_data_aa, spectral_data_aa, 
                                      freq_data_bb, spectral_data_bb,
                                      freq_data_ab, spectral_data_ab,
                                      base_species, output_file):
    """
    Create a plot showing AA, BB, and AB correlations for selected h values.
    
    Parameters:
    - h_values: list of h values
    - freq_data_aa, spectral_data_aa: AA correlation data
    - freq_data_bb, spectral_data_bb: BB correlation data
    - freq_data_ab, spectral_data_ab: AB correlation data
    - base_species: base species name for title
    - output_file: path to save the plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    # Select specific h values: 0T, closest to 0.3T, closest to 1T, closest to 2.5T
    target_fields_tesla = [0.0, 0.3, 1.0, 2.5]
    h_values_tesla = np.array(h_values) * H_CONVERSION_FACTOR
    
    selected_h_indices = []
    for target_field in target_fields_tesla:
        # Find the closest h value to the target field
        closest_idx = np.argmin(np.abs(h_values_tesla - target_field))
        selected_h_indices.append(closest_idx)
    
    for plot_idx, h_idx in enumerate(selected_h_indices):
        h = h_values[h_idx]
        ax = axes[plot_idx]
        
        # Plot AA correlation
        if h in freq_data_aa and h in spectral_data_aa:
            freq = freq_data_aa[h]
            spec = spectral_data_aa[h]
            mask = (freq >= FREQ_MIN) & (freq <= FREQ_MAX)
            freq_meV = freq[mask] * ENERGY_CONVERSION_FACTOR
            ax.plot(freq_meV, spec[mask], 'b-', label='AA', linewidth=2, alpha=0.7)
        
        # Plot BB correlation
        if h in freq_data_bb and h in spectral_data_bb:
            freq = freq_data_bb[h]
            spec = spectral_data_bb[h]
            mask = (freq >= FREQ_MIN) & (freq <= FREQ_MAX)
            freq_meV = freq[mask] * ENERGY_CONVERSION_FACTOR
            ax.plot(freq_meV, spec[mask], 'r-', label='BB', linewidth=2, alpha=0.7)
        
        # Plot AB correlation
        if h in freq_data_ab and h in spectral_data_ab:
            freq = freq_data_ab[h]
            spec = spectral_data_ab[h]
            mask = (freq >= FREQ_MIN) & (freq <= FREQ_MAX)
            freq_meV = freq[mask] * ENERGY_CONVERSION_FACTOR
            ax.plot(freq_meV, spec[mask], 'g-', label='AB', linewidth=2, alpha=0.7)
        
        # Plot Total = AA + BB + 2*AB
        if h in freq_data_aa and h in spectral_data_aa and \
           h in freq_data_bb and h in spectral_data_bb and \
           h in freq_data_ab and h in spectral_data_ab:
            # Get all three spectra
            freq_aa_full = freq_data_aa[h]
            spec_aa_full = spectral_data_aa[h]
            freq_bb_full = freq_data_bb[h]
            spec_bb_full = spectral_data_bb[h]
            freq_ab_full = freq_data_ab[h]
            spec_ab_full = spectral_data_ab[h]
            
            # Use AA as reference grid
            mask_aa = (freq_aa_full >= FREQ_MIN) & (freq_aa_full <= FREQ_MAX)
            freq_ref = freq_aa_full[mask_aa]
            spec_aa_filtered = spec_aa_full[mask_aa]
            
            # Interpolate BB and AB to AA grid
            spec_bb_interp = np.interp(freq_ref, freq_bb_full, spec_bb_full)
            spec_ab_interp = np.interp(freq_ref, freq_ab_full, spec_ab_full)
            
            # Calculate total
            spec_total = spec_aa_filtered + spec_bb_interp + 2 * spec_ab_interp
            freq_meV = freq_ref * ENERGY_CONVERSION_FACTOR
            ax.plot(freq_meV, spec_total, 'm-', label='Total (AA+BB+2AB)', linewidth=2.5, alpha=0.9)
        
        h_converted = h * H_CONVERSION_FACTOR
        ax.set_xlabel('Energy (meV)', fontsize=11)
        ax.set_ylabel('Spectral Function', fontsize=11)
        ax.set_title(f'{base_species} - h={h_converted:.3f} T', fontsize=12, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([FREQ_MIN * ENERGY_CONVERSION_FACTOR, FREQ_MAX * ENERGY_CONVERSION_FACTOR])
    
    plt.suptitle(f'Sublattice Correlations (AA/BB/AB/Total) - {base_species}', 
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved sublattice comparison plot: {output_file}")
    plt.close()

def create_sublattice_heatmap(h_values, freq_data_aa, spectral_data_aa, 
                              freq_data_bb, spectral_data_bb,
                              freq_data_ab, spectral_data_ab,
                              base_species, output_file):
    """
    Create a 4-panel heatmap showing AA, BB, AB, and Total correlations side by side.
    """
    # Get a common frequency grid
    if not h_values or h_values[0] not in freq_data_aa:
        print(f"  ⚠ No data available for sublattice heatmap of {base_species}")
        return
    
    freq_ref_full = freq_data_aa[h_values[0]]
    mask = (freq_ref_full >= FREQ_MIN) & (freq_ref_full <= FREQ_MAX)
    freq_ref = freq_ref_full[mask]
    
    # Create data matrices
    aa_matrix = np.zeros((len(freq_ref), len(h_values)))
    bb_matrix = np.zeros((len(freq_ref), len(h_values)))
    ab_matrix = np.zeros((len(freq_ref), len(h_values)))
    total_matrix = np.zeros((len(freq_ref), len(h_values)))
    
    for i, h in enumerate(h_values):
        # Process AA
        if h in spectral_data_aa:
            freq_h = freq_data_aa[h]
            spec_h = spectral_data_aa[h]
            mask_h = (freq_h >= FREQ_MIN) & (freq_h <= FREQ_MAX)
            aa_matrix[:, i] = np.interp(freq_ref, freq_h[mask_h], spec_h[mask_h])
        
        # Process BB
        if h in spectral_data_bb:
            freq_h = freq_data_bb[h]
            spec_h = spectral_data_bb[h]
            mask_h = (freq_h >= FREQ_MIN) & (freq_h <= FREQ_MAX)
            bb_matrix[:, i] = np.interp(freq_ref, freq_h[mask_h], spec_h[mask_h])
        
        # Process AB
        if h in spectral_data_ab:
            freq_h = freq_data_ab[h]
            spec_h = spectral_data_ab[h]
            mask_h = (freq_h >= FREQ_MIN) & (freq_h <= FREQ_MAX)
            ab_matrix[:, i] = np.interp(freq_ref, freq_h[mask_h], spec_h[mask_h])
        
        # Calculate Total = AA + BB + 2*AB
        total_matrix[:, i] = aa_matrix[:, i] + bb_matrix[:, i] + 2 * ab_matrix[:, i]
    
    # Create 4-panel figure (2x2 layout)
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    # Convert h values and frequency for plotting
    h_values_converted = np.array(h_values) * H_CONVERSION_FACTOR
    freq_meV = freq_ref * ENERGY_CONVERSION_FACTOR
    
    # Create meshgrid edges for pcolormesh (non-uniform spacing)
    h_edges = np.zeros(len(h_values_converted) + 1)
    h_edges[0] = h_values_converted[0] - (h_values_converted[1] - h_values_converted[0]) / 2
    h_edges[-1] = h_values_converted[-1] + (h_values_converted[-1] - h_values_converted[-2]) / 2
    for i in range(1, len(h_values_converted)):
        h_edges[i] = (h_values_converted[i-1] + h_values_converted[i]) / 2
    
    freq_edges = np.zeros(len(freq_meV) + 1)
    freq_edges[0] = freq_meV[0] - (freq_meV[1] - freq_meV[0]) / 2
    freq_edges[-1] = freq_meV[-1] + (freq_meV[-1] - freq_meV[-2]) / 2
    for i in range(1, len(freq_meV)):
        freq_edges[i] = (freq_meV[i-1] + freq_meV[i]) / 2
    
    # Determine common color scale
    vmin = min(aa_matrix.min(), bb_matrix.min(), ab_matrix.min(), total_matrix.min())
    vmax = max(aa_matrix.max(), bb_matrix.max(), ab_matrix.max(), total_matrix.max())
    
    # Plot AA
    im1 = axes[0].pcolormesh(h_edges, freq_edges, aa_matrix, 
                             cmap='viridis', shading='flat', vmin=vmin, vmax=vmax)
    axes[0].set_xlabel('Magnetic Field (h) [T]', fontsize=11)
    axes[0].set_ylabel('Energy (meV)', fontsize=11)
    axes[0].set_title('AA Correlation', fontsize=12, fontweight='bold')
    fig.colorbar(im1, ax=axes[0])
    
    # Plot BB
    im2 = axes[1].pcolormesh(h_edges, freq_edges, bb_matrix, 
                             cmap='viridis', shading='flat', vmin=vmin, vmax=vmax)
    axes[1].set_xlabel('Magnetic Field (h) [T]', fontsize=11)
    axes[1].set_ylabel('Energy (meV)', fontsize=11)
    axes[1].set_title('BB Correlation', fontsize=12, fontweight='bold')
    fig.colorbar(im2, ax=axes[1])
    
    # Plot AB
    im3 = axes[2].pcolormesh(h_edges, freq_edges, ab_matrix, 
                             cmap='viridis', shading='flat', vmin=vmin, vmax=vmax)
    axes[2].set_xlabel('Magnetic Field (h) [T]', fontsize=11)
    axes[2].set_ylabel('Energy (meV)', fontsize=11)
    axes[2].set_title('AB Correlation', fontsize=12, fontweight='bold')
    fig.colorbar(im3, ax=axes[2])
    
    # Plot Total
    im4 = axes[3].pcolormesh(h_edges, freq_edges, total_matrix, 
                             cmap='viridis', shading='flat', vmin=vmin, vmax=vmax)
    axes[3].set_xlabel('Magnetic Field (h) [T]', fontsize=11)
    axes[3].set_ylabel('Energy (meV)', fontsize=11)
    axes[3].set_title('Total (AA+BB+2AB)', fontsize=12, fontweight='bold')
    fig.colorbar(im4, ax=axes[3])
    
    plt.suptitle(f'Sublattice Correlations Heatmap - {base_species}', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved sublattice heatmap: {output_file}")
    plt.close()

def create_sublattice_animation(h_values, freq_data_aa, spectral_data_aa, 
                                freq_data_bb, spectral_data_bb,
                                freq_data_ab, spectral_data_ab,
                                base_species, output_file):
    """
    Create an animation showing AA, BB, and AB correlations evolving with h.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Get y-axis limits
    all_aa = [spectral_data_aa[h] for h in h_values if h in spectral_data_aa]
    all_bb = [spectral_data_bb[h] for h in h_values if h in spectral_data_bb]
    all_ab = [spectral_data_ab[h] for h in h_values if h in spectral_data_ab]
    
    if not all_aa and not all_bb and not all_ab:
        print(f"  ⚠ No data available for sublattice animation of {base_species}")
        return
    
    all_data = all_aa + all_bb + all_ab
    ymin = min([np.min(s) for s in all_data if len(s) > 0])
    ymax = max([np.max(s) for s in all_data if len(s) > 0])
    
    y_range = ymax - ymin
    ymin -= 0.1 * y_range
    ymax += 0.1 * y_range
    
    # Create line objects
    line_aa, = ax.plot([], [], 'b-', linewidth=2, alpha=0.7, label='AA')
    line_bb, = ax.plot([], [], 'r-', linewidth=2, alpha=0.7, label='BB')
    line_ab, = ax.plot([], [], 'g-', linewidth=2, alpha=0.7, label='AB')
    line_total, = ax.plot([], [], 'm-', linewidth=2.5, alpha=0.9, label='Total (AA+BB+2AB)')
    
    ax.set_xlim(FREQ_MIN * ENERGY_CONVERSION_FACTOR, FREQ_MAX * ENERGY_CONVERSION_FACTOR)
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel('Energy (meV)', fontsize=12)
    ax.set_ylabel('Spectral Function', fontsize=12)
    ax.legend(loc='best', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    title = ax.set_title('', fontsize=14, fontweight='bold')
    
    def init():
        line_aa.set_data([], [])
        line_bb.set_data([], [])
        line_ab.set_data([], [])
        line_total.set_data([], [])
        return line_aa, line_bb, line_ab, line_total, title
    
    def animate(frame):
        h = h_values[frame]
        h_converted = h * H_CONVERSION_FACTOR
        
        # Plot AA
        freq_aa_data = None
        spec_aa_data = None
        if h in freq_data_aa and h in spectral_data_aa:
            freq = freq_data_aa[h]
            spec = spectral_data_aa[h]
            mask = (freq >= FREQ_MIN) & (freq <= FREQ_MAX)
            freq_meV = freq[mask] * ENERGY_CONVERSION_FACTOR
            spec_filtered = spec[mask]
            line_aa.set_data(freq_meV, spec_filtered)
            freq_aa_data = freq[mask]
            spec_aa_data = spec_filtered
        else:
            line_aa.set_data([], [])
        
        # Plot BB
        freq_bb_data = None
        spec_bb_data = None
        if h in freq_data_bb and h in spectral_data_bb:
            freq = freq_data_bb[h]
            spec = spectral_data_bb[h]
            mask = (freq >= FREQ_MIN) & (freq <= FREQ_MAX)
            freq_meV = freq[mask] * ENERGY_CONVERSION_FACTOR
            spec_filtered = spec[mask]
            line_bb.set_data(freq_meV, spec_filtered)
            freq_bb_data = freq[mask]
            spec_bb_data = spec_filtered
        else:
            line_bb.set_data([], [])
        
        # Plot AB
        freq_ab_data = None
        spec_ab_data = None
        if h in freq_data_ab and h in spectral_data_ab:
            freq = freq_data_ab[h]
            spec = spectral_data_ab[h]
            mask = (freq >= FREQ_MIN) & (freq <= FREQ_MAX)
            freq_meV = freq[mask] * ENERGY_CONVERSION_FACTOR
            spec_filtered = spec[mask]
            line_ab.set_data(freq_meV, spec_filtered)
            freq_ab_data = freq[mask]
            spec_ab_data = spec_filtered
        else:
            line_ab.set_data([], [])
        
        # Plot Total = AA + BB + 2*AB
        if freq_aa_data is not None and spec_aa_data is not None and \
           freq_bb_data is not None and spec_bb_data is not None and \
           freq_ab_data is not None and spec_ab_data is not None:
            # Interpolate all to AA grid
            spec_bb_interp = np.interp(freq_aa_data, freq_bb_data, spec_bb_data)
            spec_ab_interp = np.interp(freq_aa_data, freq_ab_data, spec_ab_data)
            spec_total = spec_aa_data + spec_bb_interp + 2 * spec_ab_interp
            freq_meV = freq_aa_data * ENERGY_CONVERSION_FACTOR
            line_total.set_data(freq_meV, spec_total)
        else:
            line_total.set_data([], [])
        
        title.set_text(f'Sublattice Correlations - {base_species} - h={h_converted:.3f} T')
        return line_aa, line_bb, line_ab, line_total, title
    
    anim = FuncAnimation(fig, animate, init_func=init, 
                        frames=len(h_values), interval=200, blit=True)
    
    # Save animation
    writer = PillowWriter(fps=0.5)
    anim.save(output_file, writer=writer)
    print(f"  ✓ Saved sublattice animation: {output_file}")
    plt.close()

def create_magnetization_plot(h_values, h_dirs, output_file):

    """
    Create a plot showing magnetization components (Mx, My, Mz) and total magnitude 
    as a function of magnetic field h.
    Includes both local frame and global frame (pyrochlore lattice) magnetizations.
    
    Parameters:
    - h_values: list of h values
    - h_dirs: dict mapping h -> directory path
    - output_file: path to save the plot
    """
    print("\nCreating magnetization plot...")
    
    # Collect magnetization data
    h_list = []
    mag_x_list = []
    mag_y_list = []
    mag_z_list = []
    mag_total_list = []
    mag_x_per_site_list = []
    mag_y_per_site_list = []
    mag_z_per_site_list = []
    mag_total_per_site_list = []
    
    # Global frame magnetization
    mag_global_x_list = []
    mag_global_y_list = []
    mag_global_z_list = []
    mag_global_total_list = []
    mag_global_x_per_site_list = []
    mag_global_y_per_site_list = []
    mag_global_z_per_site_list = []
    mag_global_total_per_site_list = []
    
    # Magnetization along field direction
    mag_along_field_list = []
    mag_along_field_per_site_list = []
    
    for h in h_values:
        h_dir = h_dirs[h]
        mag_data = read_spin_configuration(h_dir)
        
        if mag_data is not None:
            h_list.append(h * H_CONVERSION_FACTOR)  # Convert to Tesla
            # Local frame
            mag_x_list.append(mag_data['mag_x'])
            mag_y_list.append(mag_data['mag_y'])
            mag_z_list.append(mag_data['mag_z'])
            mag_total_list.append(mag_data['total_magnetization'])
            mag_x_per_site_list.append(mag_data['mag_x_per_site'])
            mag_y_per_site_list.append(mag_data['mag_y_per_site'])
            mag_z_per_site_list.append(mag_data['mag_z_per_site'])
            mag_total_per_site_list.append(mag_data['magnetization_per_site'])
            # Global frame
            mag_global_x_list.append(mag_data['mag_global_x'])
            mag_global_y_list.append(mag_data['mag_global_y'])
            mag_global_z_list.append(mag_data['mag_global_z'])
            mag_global_total_list.append(mag_data['mag_global_total'])
            mag_global_x_per_site_list.append(mag_data['mag_global_x_per_site'])
            mag_global_y_per_site_list.append(mag_data['mag_global_y_per_site'])
            mag_global_z_per_site_list.append(mag_data['mag_global_z_per_site'])
            mag_global_total_per_site_list.append(mag_data['mag_global_total_per_site'])
            # Along field direction
            mag_along_field_list.append(mag_data['mag_along_field'])
            mag_along_field_per_site_list.append(mag_data['mag_along_field_per_site'])
    
    if not h_list:
        print("  ⚠ No magnetization data found!")
        return
    
    # Create figure with seven subplots
    # Top: 1 large plot for magnetization along field
    # Middle and Bottom: 3x2 layout for local and global frames
    fig = plt.figure(figsize=(16, 20))
    gs = fig.add_gridspec(4, 2, height_ratios=[1, 1, 1, 1], hspace=0.3, wspace=0.3)
    
    # ========== TOP PLOT: MAGNETIZATION ALONG FIELD DIRECTION ==========
    ax_field = fig.add_subplot(gs[0, :])  # Span both columns
    
    # Format field direction for display
    field_dir_str = f"[{FIELD_DIRECTION[0]:.3f}, {FIELD_DIRECTION[1]:.3f}, {FIELD_DIRECTION[2]:.3f}]"
    
    # Plot magnetization along field (total and per-site)
    ax_field.plot(h_list, mag_along_field_list, 'ko-', linewidth=3, markersize=6, alpha=0.9, 
                  label=f'M·ĥ (total)', zorder=3)
    ax_field.plot(h_list, mag_along_field_per_site_list, 'ro--', linewidth=2, markersize=4, alpha=0.7, 
                  label=f'(M·ĥ)/site', zorder=2)
    
    # Also show total magnetization magnitude for comparison
    ax_field.plot(h_list, mag_global_total_list, 'b-^', linewidth=2, markersize=4, alpha=0.6, 
                  label='|M| (total)', zorder=1)
    
    ax_field.set_xlabel('Magnetic Field (h) [T]', fontsize=14, fontweight='bold')
    ax_field.set_ylabel('Magnetization (Global Frame)', fontsize=14, fontweight='bold')
    ax_field.set_title(f'Magnetization Along Field Direction ĥ = {field_dir_str}', 
                       fontsize=16, fontweight='bold', pad=15)
    ax_field.legend(loc='best', fontsize=12, framealpha=0.9)
    ax_field.grid(True, alpha=0.3, linestyle='--')
    ax_field.tick_params(labelsize=11)
    
    # Add zero line for reference
    ax_field.axhline(y=0, color='gray', linestyle=':', linewidth=1, alpha=0.5)
    
    # ========== LOCAL FRAME PLOTS (Left Column) ==========
    
    # Plot 1: Total magnetization components (local frame)
    ax1 = fig.add_subplot(gs[1, 0])
    ax1.plot(h_list, mag_x_list, 'r-o', linewidth=2, markersize=4, alpha=0.7, label='Mx (local)')
    ax1.plot(h_list, mag_y_list, 'g-o', linewidth=2, markersize=4, alpha=0.7, label='My (local)')
    ax1.plot(h_list, mag_z_list, 'b-o', linewidth=2, markersize=4, alpha=0.7, label='Mz (local)')
    ax1.plot(h_list, mag_total_list, 'k-o', linewidth=2.5, markersize=5, alpha=0.9, label='|M| Total')
    ax1.set_xlabel('Magnetic Field (h) [T]', fontsize=12)
    ax1.set_ylabel('Total Magnetization (Local Frame)', fontsize=12)
    ax1.set_title('Local Frame: Total Magnetization vs Field', fontsize=13, fontweight='bold')
    ax1.legend(loc='best', fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Magnetization per site components (local frame)
    ax2 = fig.add_subplot(gs[2, 0])
    ax2.plot(h_list, mag_x_per_site_list, 'r-o', linewidth=2, markersize=4, alpha=0.7, label='Mx/site')
    ax2.plot(h_list, mag_y_per_site_list, 'g-o', linewidth=2, markersize=4, alpha=0.7, label='My/site')
    ax2.plot(h_list, mag_z_per_site_list, 'b-o', linewidth=2, markersize=4, alpha=0.7, label='Mz/site')
    ax2.plot(h_list, mag_total_per_site_list, 'k-o', linewidth=2.5, markersize=5, alpha=0.9, label='|M|/site')
    ax2.set_xlabel('Magnetic Field (h) [T]', fontsize=12)
    ax2.set_ylabel('Magnetization per Site (Local)', fontsize=12)
    ax2.set_title('Local Frame: Per-Site Magnetization vs Field', fontsize=13, fontweight='bold')
    ax2.legend(loc='best', fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Transverse vs Longitudinal (local frame)
    ax3 = fig.add_subplot(gs[3, 0])
    transverse_mag = np.sqrt(np.array(mag_x_list)**2 + np.array(mag_y_list)**2)
    ax3.plot(h_list, transverse_mag, 'm-o', linewidth=2.5, markersize=5, alpha=0.9, label='Transverse √(Mx²+My²)')
    ax3.plot(h_list, mag_z_list, 'b-o', linewidth=2, markersize=4, alpha=0.7, label='Longitudinal Mz')
    ax3.set_xlabel('Magnetic Field (h) [T]', fontsize=12)
    ax3.set_ylabel('Magnetization (Local Frame)', fontsize=12)
    ax3.set_title('Local Frame: Transverse vs Longitudinal', fontsize=13, fontweight='bold')
    ax3.legend(loc='best', fontsize=9)
    ax3.grid(True, alpha=0.3)
    
    # ========== GLOBAL FRAME PLOTS (Right Column) ==========
    
    # Plot 4: Total magnetization components (global frame)
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(h_list, mag_global_x_list, 'r-o', linewidth=2, markersize=4, alpha=0.7, label='Mx (global)')
    ax4.plot(h_list, mag_global_y_list, 'g-o', linewidth=2, markersize=4, alpha=0.7, label='My (global)')
    ax4.plot(h_list, mag_global_z_list, 'b-o', linewidth=2, markersize=4, alpha=0.7, label='Mz (global)')
    ax4.plot(h_list, mag_global_total_list, 'k-o', linewidth=2.5, markersize=5, alpha=0.9, label='|M| Total')
    ax4.set_xlabel('Magnetic Field (h) [T]', fontsize=12)
    ax4.set_ylabel('Total Magnetization (Global Frame)', fontsize=12)
    ax4.set_title('Global Frame: Total Magnetization vs Field', fontsize=13, fontweight='bold')
    ax4.legend(loc='best', fontsize=9)
    ax4.grid(True, alpha=0.3)
    
    # Plot 5: Magnetization per site components (global frame)
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.plot(h_list, mag_global_x_per_site_list, 'r-o', linewidth=2, markersize=4, alpha=0.7, label='Mx/site')
    ax5.plot(h_list, mag_global_y_per_site_list, 'g-o', linewidth=2, markersize=4, alpha=0.7, label='My/site')
    ax5.plot(h_list, mag_global_z_per_site_list, 'b-o', linewidth=2, markersize=4, alpha=0.7, label='Mz/site')
    ax5.plot(h_list, mag_global_total_per_site_list, 'k-o', linewidth=2.5, markersize=5, alpha=0.9, label='|M|/site')
    ax5.set_xlabel('Magnetic Field (h) [T]', fontsize=12)
    ax5.set_ylabel('Magnetization per Site (Global)', fontsize=12)
    ax5.set_title('Global Frame: Per-Site Magnetization vs Field', fontsize=13, fontweight='bold')
    ax5.legend(loc='best', fontsize=9)
    ax5.grid(True, alpha=0.3)
    
    # Plot 6: Transverse vs Longitudinal (global frame)
    ax6 = fig.add_subplot(gs[3, 1])
    transverse_mag_global = np.sqrt(np.array(mag_global_x_list)**2 + np.array(mag_global_y_list)**2)
    ax6.plot(h_list, transverse_mag_global, 'm-o', linewidth=2.5, markersize=5, alpha=0.9, label='Transverse √(Mx²+My²)')
    ax6.plot(h_list, mag_global_z_list, 'b-o', linewidth=2, markersize=4, alpha=0.7, label='Longitudinal Mz')
    ax6.set_xlabel('Magnetic Field (h) [T]', fontsize=12)
    ax6.set_ylabel('Magnetization (Global Frame)', fontsize=12)
    ax6.set_title('Global Frame: Transverse vs Longitudinal', fontsize=13, fontweight='bold')
    ax6.legend(loc='best', fontsize=9)
    ax6.grid(True, alpha=0.3)
    
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved magnetization plot: {output_file}")
    plt.close()
    
    # Also save magnetization data to a text file
    data_file = output_file.replace('.png', '_data.txt')
    with open(data_file, 'w') as f:
        f.write("# Magnetic Field (T)\t")
        f.write("M_along_field\tM_along_field/site\t")
        f.write("Mx_local\tMy_local\tMz_local\t|M|_local\t")
        f.write("Mx_local/site\tMy_local/site\tMz_local/site\t|M|_local/site\t")
        f.write("Mx_global\tMy_global\tMz_global\t|M|_global\t")
        f.write("Mx_global/site\tMy_global/site\tMz_global/site\t|M|_global/site\n")
        for i, h in enumerate(h_list):
            f.write(f"{h:.6f}\t")
            f.write(f"{mag_along_field_list[i]:.6f}\t{mag_along_field_per_site_list[i]:.6f}\t")
            f.write(f"{mag_x_list[i]:.6f}\t{mag_y_list[i]:.6f}\t{mag_z_list[i]:.6f}\t{mag_total_list[i]:.6f}\t")
            f.write(f"{mag_x_per_site_list[i]:.6f}\t{mag_y_per_site_list[i]:.6f}\t{mag_z_per_site_list[i]:.6f}\t{mag_total_per_site_list[i]:.6f}\t")
            f.write(f"{mag_global_x_list[i]:.6f}\t{mag_global_y_list[i]:.6f}\t{mag_global_z_list[i]:.6f}\t{mag_global_total_list[i]:.6f}\t")
            f.write(f"{mag_global_x_per_site_list[i]:.6f}\t{mag_global_y_per_site_list[i]:.6f}\t{mag_global_z_per_site_list[i]:.6f}\t{mag_global_total_per_site_list[i]:.6f}\n")
    print(f"  ✓ Saved magnetization data: {data_file}")


def create_summary_report(h_values, h_dirs, all_species, do_channels, 
                         species_combinations, sublattice_base_patterns):
    """
    Create a comprehensive summary report of all processed data.
    
    Parameters:
    - h_values: list of h field values
    - h_dirs: dictionary mapping h values to directories
    - all_species: list of all species found
    - do_channels: dictionary of DO channels
    - species_combinations: dictionary of species combinations
    - sublattice_base_patterns: list of base patterns with sublattice data
    """
    summary_file = os.path.join(SUBDIRS['summary'], "ANALYSIS_SUMMARY.txt")
    
    with open(summary_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("DYNAMIC STRUCTURE FACTOR ANALYSIS SUMMARY\n")
        f.write("="*80 + "\n\n")
        
        # Dataset information
        f.write("DATASET INFORMATION\n")
        f.write("-"*80 + "\n")
        f.write(f"Base Directory: {BASE_DIR}\n")
        f.write(f"Output Directory: {OUTPUT_DIR}\n")
        f.write(f"Number of h-field values: {len(h_values)}\n")
        f.write(f"h-field range: {min(h_values):.3f} to {max(h_values):.3f} (Jzz units)\n")
        f.write(f"h-field range: {min(h_values)*H_CONVERSION_FACTOR:.3f} to {max(h_values)*H_CONVERSION_FACTOR:.3f} (Tesla)\n")
        f.write(f"\nEnergy conversion factor: {ENERGY_CONVERSION_FACTOR} meV/Jzz\n")
        f.write(f"Field conversion factor: {H_CONVERSION_FACTOR:.6f} T/Jzz\n\n")
        
        # Species information
        f.write("SPECIES ANALYZED\n")
        f.write("-"*80 + "\n")
        f.write(f"Total number of species: {len(all_species)}\n\n")
        
        # Group species by type
        smsp_species = [s for s in all_species if s.startswith("SmSp_")]
        szsz_species = [s for s in all_species if s.startswith("SzSz_")]
        other_species = [s for s in all_species if not (s.startswith("SmSp_") or s.startswith("SzSz_"))]
        
        f.write(f"SmSp Species ({len(smsp_species)}):\n")
        for sp in sorted(smsp_species):
            f.write(f"  - {sp}\n")
        
        f.write(f"\nSzSz Species ({len(szsz_species)}):\n")
        for sp in sorted(szsz_species):
            f.write(f"  - {sp}\n")
        
        if other_species:
            f.write(f"\nOther Species ({len(other_species)}):\n")
            for sp in sorted(other_species):
                f.write(f"  - {sp}\n")
        
        # DO channels
        f.write(f"\n\nDO CHANNELS (SF + NSF)\n")
        f.write("-"*80 + "\n")
        f.write(f"Number of DO channels created: {len(do_channels)}\n\n")
        for do_sp in sorted(do_channels.keys()):
            f.write(f"  - {do_sp}\n")
        
        # Combined plots
        f.write(f"\n\nCOMBINED SmSp + SzSz PLOTS\n")
        f.write("-"*80 + "\n")
        f.write(f"Number of combined plots: {len(species_combinations)}\n\n")
        for (base_pattern, suffix), components in sorted(species_combinations.items()):
            f.write(f"  - {base_pattern}{suffix}\n")
            f.write(f"    Components: {', '.join(sorted(components))}\n")
        
        # Sublattice correlations
        f.write(f"\n\nSUBLATTICE CORRELATIONS\n")
        f.write("-"*80 + "\n")
        f.write(f"Sublattice A indices: {SUBLATTICE_A}\n")
        f.write(f"Sublattice B indices: {SUBLATTICE_B}\n")
        f.write(f"Number of base patterns with sublattice data: {len(sublattice_base_patterns)}\n\n")
        for pattern in sorted(sublattice_base_patterns):
            f.write(f"  - {pattern}\n")
        
        # Output structure
        f.write(f"\n\nOUTPUT STRUCTURE\n")
        f.write("-"*80 + "\n")
        f.write("The analysis outputs are organized in the following subdirectories:\n\n")
        f.write("0_summary/\n")
        f.write("  - This summary report\n")
        f.write("  - Key figures and overview plots\n\n")
        f.write("1_individual_species/\n")
        f.write(f"  - {len(all_species)} individual species\n")
        f.write("  - 4 plots per species: stacked, heatmap, 3D surface, animation\n\n")
        f.write("2_DO_channels/\n")
        f.write(f"  - {len(do_channels)} DO channels (SF + NSF)\n")
        f.write("  - 4 plots per channel: stacked, heatmap, 3D surface, animation\n\n")
        f.write("3_combined_SmSp_SzSz/\n")
        f.write(f"  - {len([k for k, v in species_combinations.items() if 'SmSp' in v and 'SzSz' in v])} combined plots\n")
        f.write("  - 3 plots per combination: components, heatmap, animation\n\n")
        f.write("4_sublattice_correlations/\n")
        f.write(f"  - {len(sublattice_base_patterns)} base patterns\n")
        f.write("  - 3 plots per pattern: comparison, heatmap, animation\n\n")
        f.write("5_global_transverse/\n")
        f.write(f"  - {len(sublattice_base_patterns)} transverse sublattice plots\n")
        f.write("  - 3 plots per pattern: comparison, heatmap, animation\n\n")
        f.write("6_magnetization/\n")
        f.write("  - Magnetization vs field plot and data\n\n")
        
        # Statistics
        total_plots = (len(all_species) * 4 + 
                      len(do_channels) * 4 + 
                      len([k for k, v in species_combinations.items() if 'SmSp' in v and 'SzSz' in v]) * 3 +
                      len(sublattice_base_patterns) * 3 * 2 +  # Regular + transverse
                      1)  # Magnetization
        
        f.write(f"\n\nSTATISTICS\n")
        f.write("-"*80 + "\n")
        f.write(f"Total number of plots generated: ~{total_plots}\n")
        f.write(f"Total h-field points analyzed: {len(h_values)}\n")
        f.write(f"Frequency range analyzed: {FREQ_MIN} to {FREQ_MAX} Jzz\n")
        f.write(f"Energy range analyzed: {FREQ_MIN*ENERGY_CONVERSION_FACTOR:.3f} to {FREQ_MAX*ENERGY_CONVERSION_FACTOR:.3f} meV\n\n")
        
        f.write("="*80 + "\n")
        f.write("End of Summary Report\n")
        f.write("="*80 + "\n")
    
    print(f"  ✓ Saved summary report: {summary_file}")
    
    # Also create a README file in the main output directory
    readme_file = os.path.join(OUTPUT_DIR, "README.txt")
    with open(readme_file, 'w') as f:
        f.write("DYNAMIC STRUCTURE FACTOR ANALYSIS OUTPUT\n")
        f.write("="*80 + "\n\n")
        f.write("This directory contains comprehensive analysis of dynamic structure factor data\n")
        f.write("across multiple magnetic field values.\n\n")
        f.write("DIRECTORY STRUCTURE:\n\n")
        f.write("0_summary/              - Summary report and overview\n")
        f.write("1_individual_species/   - Individual species spectral functions\n")
        f.write("2_DO_channels/          - Difference Orbital (SF + NSF) channels\n")
        f.write("3_combined_SmSp_SzSz/   - Combined SmSp + SzSz analysis\n")
        f.write("4_sublattice_correlations/ - Sublattice (AA/BB/AB) correlations\n")
        f.write("5_global_transverse/    - Global transverse sublattice analysis\n")
        f.write("6_magnetization/        - Magnetization vs field analysis\n\n")
        f.write("For more details, see 0_summary/ANALYSIS_SUMMARY.txt\n")
    
    print(f"  ✓ Saved README: {readme_file}")
    
    # Create a file listing for easy navigation
    catalog_file = os.path.join(SUBDIRS['summary'], "FILE_CATALOG.txt")
    with open(catalog_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("FILE CATALOG - All Generated Outputs\n")
        f.write("="*80 + "\n\n")
        
        for subdir_name, subdir_path in sorted(SUBDIRS.items()):
            if os.path.exists(subdir_path):
                files = sorted([f for f in os.listdir(subdir_path) if os.path.isfile(os.path.join(subdir_path, f))])
                if files:
                    f.write(f"\n{os.path.basename(subdir_path)}/\n")
                    f.write("-"*80 + "\n")
                    f.write(f"Total files: {len(files)}\n\n")
                    for fname in files:
                        fpath = os.path.join(subdir_path, fname)
                        fsize = os.path.getsize(fpath)
                        if fsize > 1024*1024:
                            size_str = f"{fsize/(1024*1024):.2f} MB"
                        elif fsize > 1024:
                            size_str = f"{fsize/1024:.2f} KB"
                        else:
                            size_str = f"{fsize} B"
                        f.write(f"  {fname:<60} {size_str:>10}\n")
    
    print(f"  ✓ Saved file catalog: {catalog_file}")


def main():
    """Main function to process all data and create plots"""
    print("Finding h directories...")
    h_data = find_all_h_directories()
    
    if not h_data:
        print("No h directories found!")
        return
    
    h_values = [h for h, _ in h_data]
    h_dirs = {h: d for h, d in h_data}
    
    print(f"Found {len(h_values)} h values: {h_values}")
    
    # Find all unique species across all h directories
    print("\nFinding all species...")
    all_species = set()
    for h, h_dir in h_data:
        species = find_all_species(h_dir)
        all_species.update(species)
    
    all_species = sorted(list(all_species))
    print(f"Found {len(all_species)} species:")
    for sp in all_species:
        print(f"  - {sp}")
    
    # Identify base species (without _SF or _NSF suffix) to create DO channels
    base_species_set = set()
    for sp in all_species:
        if sp.endswith("_SF"):
            base_species_set.add(sp[:-3])  # Remove "_SF"
        elif sp.endswith("_NSF"):
            base_species_set.add(sp[:-4])  # Remove "_NSF"
    
    # Create DO channels
    do_channels = {}
    if base_species_set:
        print(f"\nCreating DO channels (SF + NSF) for {len(base_species_set)} base species...")
        for base_sp in sorted(base_species_set):
            sf_sp = base_sp + "_SF"
            nsf_sp = base_sp + "_NSF"
            if sf_sp in all_species and nsf_sp in all_species:
                print(f"  Creating DO for: {base_sp}")
                freq_data_do, spectral_data_do = create_do_channel(h_values, h_dirs, base_sp)
                if freq_data_do:
                    do_channels[base_sp + "_DO"] = (freq_data_do, spectral_data_do)
                    print(f"    ✓ Created {base_sp}_DO with {len(freq_data_do)} h-values")
                else:
                    print(f"    ✗ Failed to create {base_sp}_DO")
    
    # Process each species
    print("\n" + "="*80)
    for species in all_species:
        try:
            print(f"\nProcessing species: {species}")
            print("-"*80)
            
            # Collect data for this species across all h values
            freq_data = {}
            spectral_data = {}
            
            for h in h_values:
                h_dir = h_dirs[h]
                freq, spectral = read_spectral_data(h_dir, species)
                
                if freq is not None and spectral is not None:
                    freq_data[h] = freq
                    spectral_data[h] = spectral
                    print(f"  ✓ h={h:.2f}: {len(freq)} data points")
                else:
                    print(f"  ✗ h={h:.2f}: No data")
            
            if not spectral_data:
                print(f"  No data found for {species}, skipping...")
                continue
            
            # Create safe filename
            safe_species_name = species.replace("/", "_").replace(" ", "_")
            
            # Create plots in individual species subdirectory
            print(f"\n  Creating plots for {species}...")
            
            # 1. Stacked plot
            stacked_file = os.path.join(SUBDIRS['individual'], f"{safe_species_name}_stacked.png")
            1(h_values, freq_data, spectral_data, species, stacked_file)
            
            # 2. Heatmap
            heatmap_file = os.path.join(SUBDIRS['individual'], f"{safe_species_name}_heatmap.png")
            create_heatmap_plot(h_values, freq_data, spectral_data, species, heatmap_file)
            
            # 3. 3D surface plot
            surface_file = os.path.join(SUBDIRS['individual'], f"{safe_species_name}_3d_surface.png")
            create_3d_surface_plot(h_values, freq_data, spectral_data, species, surface_file)
            
            # 4. Animation
            anim_file = os.path.join(SUBDIRS['individual'], f"{safe_species_name}_animation.gif")
            create_animation(h_values, freq_data, spectral_data, species, anim_file)
            
            print(f"  ✓ Completed {species}")
        except Exception as e:
            print(f"  ✗ Error processing {species}: {e}")
            continue
    
    # Process DO channels
    if do_channels:
        print("\n" + "="*80)
        print("Processing DO channels (SF + NSF)...")
        print("="*80)
        
        for do_species, (freq_data, spectral_data) in do_channels.items():
            try:
                print(f"\nProcessing DO channel: {do_species}")
                print("-"*80)
                
                if not spectral_data:
                    print(f"  No data found for {do_species}, skipping...")
                    continue
                
                # Create safe filename
                safe_species_name = do_species.replace("/", "_").replace(" ", "_")
                
                # Create plots in DO channels subdirectory
                print(f"  Creating plots for {do_species}...")
                
                # 1. Stacked plot
                stacked_file = os.path.join(SUBDIRS['do_channels'], f"{safe_species_name}_stacked.png")
                create_stacked_plot(h_values, freq_data, spectral_data, do_species, stacked_file)
                
                # 2. Heatmap
                heatmap_file = os.path.join(SUBDIRS['do_channels'], f"{safe_species_name}_heatmap.png")
                create_heatmap_plot(h_values, freq_data, spectral_data, do_species, heatmap_file)
                
                # 3. 3D surface plot
                surface_file = os.path.join(SUBDIRS['do_channels'], f"{safe_species_name}_3d_surface.png")
                create_3d_surface_plot(h_values, freq_data, spectral_data, do_species, surface_file)
                
                # 4. Animation
                anim_file = os.path.join(SUBDIRS['do_channels'], f"{safe_species_name}_animation.gif")
                create_animation(h_values, freq_data, spectral_data, do_species, anim_file)
                
                print(f"  ✓ Completed {do_species}")
            except Exception as e:
                print(f"  ✗ Error processing {do_species}: {e}")
                continue
    
    # Create combined Spm + Szz plots
    print("\n" + "="*80)
    print("Creating combined Spm + Szz plots...")
    print("="*80)
    
    # Build a complete data dictionary for all species
    all_freq_data = {}
    all_spectral_data = {}
    
    # Add regular species data
    for species in all_species:
        freq_data = {}
        spectral_data = {}
        for h in h_values:
            h_dir = h_dirs[h]
            freq, spectral = read_spectral_data(h_dir, species)
            if freq is not None and spectral is not None:
                freq_data[h] = freq
                spectral_data[h] = spectral
        
        if freq_data:
            all_freq_data[species] = freq_data
            all_spectral_data[species] = spectral_data
    
    # Add DO channel data
    for do_species, (freq_data, spectral_data) in do_channels.items():
        if freq_data:
            all_freq_data[do_species] = freq_data
            all_spectral_data[do_species] = spectral_data
    
    # Identify base species and suffixes with both SmSp and SzSz
    # Pattern: SmSp_q_Qx#_Qy#_Qz#_SUFFIX or SzSz_q_Qx#_Qy#_Qz#_SUFFIX
    species_combinations = {}  # Maps (base_pattern, suffix) -> set of components
    
    for species in all_freq_data.keys():
        # Try to parse species name
        # Check if it's SmSp or SzSz type
        if species.startswith("SmSp_"):
            # Find the suffix (_SF, _NSF, or _DO)
            if "_SF" in species:
                suffix = "_SF"
                base_pattern = species.replace("SmSp_", "").replace("_SF", "")
            elif "_NSF" in species:
                suffix = "_NSF"
                base_pattern = species.replace("SmSp_", "").replace("_NSF", "")
            elif "_DO" in species:
                suffix = "_DO"
                base_pattern = species.replace("SmSp_", "").replace("_DO", "")
            else:
                # No suffix, the whole thing after SmSp_ is the pattern
                suffix = ""
                base_pattern = species.replace("SmSp_", "")
            
            key = (base_pattern, suffix)
            if key not in species_combinations:
                species_combinations[key] = set()
            species_combinations[key].add("SmSp")
            
        elif species.startswith("SzSz_"):
            # Find the suffix (_SF, _NSF, or _DO)
            if "_SF" in species:
                suffix = "_SF"
                base_pattern = species.replace("SzSz_", "").replace("_SF", "")
            elif "_NSF" in species:
                suffix = "_NSF"
                base_pattern = species.replace("SzSz_", "").replace("_NSF", "")
            elif "_DO" in species:
                suffix = "_DO"
                base_pattern = species.replace("SzSz_", "").replace("_DO", "")
            else:
                # No suffix, the whole thing after SzSz_ is the pattern
                suffix = ""
                base_pattern = species.replace("SzSz_", "")
            
            key = (base_pattern, suffix)
            if key not in species_combinations:
                species_combinations[key] = set()
            species_combinations[key].add("SzSz")
    
    # Create combined plots for species with both SmSp and SzSz
    for (base_pattern, suffix), components in species_combinations.items():
        if "SmSp" in components and "SzSz" in components:
            try:
                print(f"\nCreating combined plots for {base_pattern}{suffix}")
                print("-"*80)
                
                # Reconstruct the full species names
                smsp_species = f"SmSp_{base_pattern}{suffix}"
                szsz_species = f"SzSz_{base_pattern}{suffix}"
                
                safe_name = f"{base_pattern}{suffix}".replace("/", "_").replace(" ", "_")
                
                # 1. Combined component plot (line plots at different h values) - in combined subdirectory
                combined_file = os.path.join(SUBDIRS['combined'], f"{safe_name}_combined_components.png")
                create_combined_component_plot_direct(h_values, all_freq_data, all_spectral_data, 
                                                      smsp_species, szsz_species, 
                                                      f"{base_pattern}{suffix}", combined_file)
                
                # 2. Combined heatmap (3-panel: SmSp, SzSz, Total)
                heatmap_file = os.path.join(SUBDIRS['combined'], f"{safe_name}_combined_heatmap.png")
                create_combined_heatmap_direct(h_values, all_freq_data, all_spectral_data, 
                                              smsp_species, szsz_species,
                                              f"{base_pattern}{suffix}", heatmap_file)
                
                # 3. Combined animation (GIF showing SmSp, SzSz, and Total evolving with h)
                anim_file = os.path.join(SUBDIRS['combined'], f"{safe_name}_combined_animation.gif")
                create_combined_animation_direct(h_values, all_freq_data, all_spectral_data,
                                               smsp_species, szsz_species,
                                               f"{base_pattern}{suffix}", anim_file)
                
                print(f"  ✓ Completed combined plots for {base_pattern}{suffix}")
            except Exception as e:
                print(f"  ✗ Error creating combined plots for {base_pattern}{suffix}: {e}")
                continue
    
    # Process sublattice correlations
    print("\n" + "="*80)
    print("Processing sublattice correlations (AA/BB/AB)...")
    print(f"Sublattice A: {SUBLATTICE_A}, Sublattice B: {SUBLATTICE_B}")
    print("="*80)
    
    # Find all unique base species patterns that have sublattice data
    sublattice_base_patterns = set()
    for species in all_species:
        if '_sub' in species:
            base = get_base_species_name(species)
            sublattice_base_patterns.add(base)
    
    sublattice_base_patterns = sorted(list(sublattice_base_patterns))
    print(f"Found {len(sublattice_base_patterns)} base patterns with sublattice data:")
    for pattern in sublattice_base_patterns:
        print(f"  - {pattern}")
    
    # Group patterns by SmSp/SzSz pairs for combined processing
    sublattice_smsp_patterns = [p for p in sublattice_base_patterns if p.startswith("SmSp_")]
    sublattice_szsz_patterns = [p for p in sublattice_base_patterns if p.startswith("SzSz_")]
    
    # Create mapping of SmSp to SzSz pairs (same Q-vector and suffix)
    sublattice_combined_pairs = []
    for smsp_pattern in sublattice_smsp_patterns:
        # Extract base part (without SmSp_)
        smsp_base = smsp_pattern.replace("SmSp_", "")
        # Look for corresponding SzSz pattern
        szsz_pattern = f"SzSz_{smsp_base}"
        if szsz_pattern in sublattice_szsz_patterns:
            sublattice_combined_pairs.append((smsp_pattern, szsz_pattern, smsp_base))
    
    # Process each base pattern
    for base_pattern in sublattice_base_patterns:
        try:
            print(f"\nProcessing sublattice correlations for: {base_pattern}")
            print("-"*80)
            
            # Create AA, BB, AB correlation data
            (freq_aa, spec_aa), (freq_bb, spec_bb), (freq_ab, spec_ab) = \
                create_sublattice_correlation_data(h_values, h_dirs, base_pattern)
            
            # Check if we have data
            if not spec_aa and not spec_bb and not spec_ab:
                print(f"  ⚠ No sublattice data found for {base_pattern}, skipping...")
                continue
            
            # Create safe filename
            safe_pattern = base_pattern.replace("/", "_").replace(" ", "_")
            
            # 1. Sublattice comparison plot (line plots at different h values) - in sublattice subdirectory
            comparison_file = os.path.join(SUBDIRS['sublattice'], f"{safe_pattern}_sublattice_comparison.png")
            create_sublattice_comparison_plot(h_values, freq_aa, spec_aa, freq_bb, spec_bb, 
                                             freq_ab, spec_ab, base_pattern, comparison_file)
            
            # 2. Sublattice heatmap (3-panel: AA, BB, AB)
            heatmap_file = os.path.join(SUBDIRS['sublattice'], f"{safe_pattern}_sublattice_heatmap.png")
            create_sublattice_heatmap(h_values, freq_aa, spec_aa, freq_bb, spec_bb, 
                                     freq_ab, spec_ab, base_pattern, heatmap_file)
            
            # 3. Sublattice animation (GIF showing AA, BB, AB evolving with h)
            anim_file = os.path.join(SUBDIRS['sublattice'], f"{safe_pattern}_sublattice_animation.gif")
            create_sublattice_animation(h_values, freq_aa, spec_aa, freq_bb, spec_bb, 
                                       freq_ab, spec_ab, base_pattern, anim_file)
            
            print(f"  ✓ Completed sublattice plots for {base_pattern}")
        except Exception as e:
            print(f"  ✗ Error processing sublattice for {base_pattern}: {e}")
            continue
    
    # Process individual sublattice pair plots for SmSp and SzSz
    print("\n" + "="*80)
    print("Creating individual sublattice pair plots (SmSp, SzSz, Combined)...")
    print("="*80)
    
    # Find all individual sublattice pair species
    individual_sublattice_pairs = {}
    for species in all_species:
        if '_sub' in species:
            sub1, sub2 = parse_sublattice_indices(species)
            if sub1 is not None and sub2 is not None:
                base = get_base_species_name(species)
                pair_key = (base, sub1, sub2)
                if pair_key not in individual_sublattice_pairs:
                    individual_sublattice_pairs[pair_key] = {}
                individual_sublattice_pairs[pair_key][species] = (sub1, sub2)
    
    # Group by SmSp/SzSz pairs for each sublattice pair
    for (base, sub1, sub2), species_dict in individual_sublattice_pairs.items():
        # Find SmSp and SzSz species for this sublattice pair
        smsp_species = None
        szsz_species = None
        for sp in species_dict.keys():
            if sp.startswith("SmSp_"):
                smsp_species = sp
            elif sp.startswith("SzSz_"):
                szsz_species = sp
        
        if smsp_species and szsz_species:
            try:
                print(f"\nCreating combined sublattice pair plot: {base} sub{sub1}_sub{sub2}")
                print("-"*80)
                
                safe_name = f"{base}_sub{sub1}_sub{sub2}".replace("/", "_").replace(" ", "_")
                display_name = f"{base} (sublattice {sub1}-{sub2})"
                
                # 1. Combined component plot
                combined_file = os.path.join(SUBDIRS['sublattice'], f"{safe_name}_combined_components.png")
                create_combined_component_plot_direct(h_values, all_freq_data, all_spectral_data, 
                                                      smsp_species, szsz_species, 
                                                      display_name, combined_file)
                
                # 2. Combined heatmap
                heatmap_file = os.path.join(SUBDIRS['sublattice'], f"{safe_name}_combined_heatmap.png")
                create_combined_heatmap_direct(h_values, all_freq_data, all_spectral_data, 
                                              smsp_species, szsz_species,
                                              display_name, heatmap_file)
                
                # 3. Combined animation
                anim_file = os.path.join(SUBDIRS['sublattice'], f"{safe_name}_combined_animation.gif")
                create_combined_animation_direct(h_values, all_freq_data, all_spectral_data,
                                               smsp_species, szsz_species,
                                               display_name, anim_file)
                
                print(f"  ✓ Completed combined sublattice pair plots for {base} sub{sub1}_sub{sub2}")
            except Exception as e:
                print(f"  ✗ Error creating combined sublattice pair plots for {base} sub{sub1}_sub{sub2}: {e}")
                continue
    
    # Process combined SmSp + SzSz sublattice correlations
    if sublattice_combined_pairs:
        print("\n" + "="*80)
        print("Processing COMBINED SmSp + SzSz sublattice correlations...")
        print("="*80)
        
        for smsp_pattern, szsz_pattern, base_name in sublattice_combined_pairs:
            try:
                print(f"\nProcessing combined sublattice for: {base_name}")
                print(f"  SmSp: {smsp_pattern}")
                print(f"  SzSz: {szsz_pattern}")
                print("-"*80)
                
                # Get SmSp sublattice data
                (freq_aa_sm, spec_aa_sm), (freq_bb_sm, spec_bb_sm), (freq_ab_sm, spec_ab_sm) = \
                    create_sublattice_correlation_data(h_values, h_dirs, smsp_pattern)
                
                # Get SzSz sublattice data
                (freq_aa_sz, spec_aa_sz), (freq_bb_sz, spec_bb_sz), (freq_ab_sz, spec_ab_sz) = \
                    create_sublattice_correlation_data(h_values, h_dirs, szsz_pattern)
                
                # Check if we have data for both
                if (not spec_aa_sm and not spec_aa_sz) or (not spec_bb_sm and not spec_bb_sz):
                    print(f"  ⚠ Insufficient data for combined sublattice, skipping...")
                    continue
                
                # Combine: Total = SmSp * SMSP_FACTOR + SzSz
                freq_aa_combined = {}
                spec_aa_combined = {}
                freq_bb_combined = {}
                spec_bb_combined = {}
                freq_ab_combined = {}
                spec_ab_combined = {}
                
                for h in h_values:
                    if h in spec_aa_sm and h in spec_aa_sz:
                        freq_aa_combined[h] = freq_aa_sm[h]
                        spec_aa_combined[h] = spec_aa_sm[h] * SMSP_FACTOR + spec_aa_sz[h]
                    
                    if h in spec_bb_sm and h in spec_bb_sz:
                        freq_bb_combined[h] = freq_bb_sm[h]
                        spec_bb_combined[h] = spec_bb_sm[h] * SMSP_FACTOR + spec_bb_sz[h]
                    
                    if h in spec_ab_sm and h in spec_ab_sz:
                        freq_ab_combined[h] = freq_ab_sm[h]
                        spec_ab_combined[h] = spec_ab_sm[h] * SMSP_FACTOR + spec_ab_sz[h]
                
                # Create safe filename
                safe_name = base_name.replace("/", "_").replace(" ", "_")
                
                # Create combined plots
                comparison_file = os.path.join(SUBDIRS['sublattice'], f"{safe_name}_combined_sublattice_comparison.png")
                create_sublattice_comparison_plot(h_values, freq_aa_combined, spec_aa_combined, 
                                                 freq_bb_combined, spec_bb_combined, 
                                                 freq_ab_combined, spec_ab_combined, 
                                                 f"{base_name} (SmSp*{SMSP_FACTOR} + SzSz)", comparison_file)
                
                heatmap_file = os.path.join(SUBDIRS['sublattice'], f"{safe_name}_combined_sublattice_heatmap.png")
                create_sublattice_heatmap(h_values, freq_aa_combined, spec_aa_combined, 
                                         freq_bb_combined, spec_bb_combined, 
                                         freq_ab_combined, spec_ab_combined, 
                                         f"{base_name} (SmSp*{SMSP_FACTOR} + SzSz)", heatmap_file)
                
                anim_file = os.path.join(SUBDIRS['sublattice'], f"{safe_name}_combined_sublattice_animation.gif")
                create_sublattice_animation(h_values, freq_aa_combined, spec_aa_combined, 
                                           freq_bb_combined, spec_bb_combined, 
                                           freq_ab_combined, spec_ab_combined, 
                                           f"{base_name} (SmSp*{SMSP_FACTOR} + SzSz)", anim_file)
                
                print(f"  ✓ Completed combined sublattice plots for {base_name}")
            except Exception as e:
                print(f"  ✗ Error processing combined sublattice for {base_name}: {e}")
                continue
    
    # Process global transverse sublattice correlations
    print("\n" + "="*80)
    print("Processing GLOBAL TRANSVERSE sublattice correlations (AA/BB/AB)...")
    print(f"Sublattice A: {SUBLATTICE_A}, Sublattice B: {SUBLATTICE_B}")
    print(f"Q-vector: {Q_VECTOR}")
    print("Applying transverse operator: (z_μ·z_ν - (z_μ·Q̂)(z_ν·Q̂))")
    print("="*80)
    
    # Process each base pattern with transverse weighting
    for base_pattern in sublattice_base_patterns:
        try:
            print(f"\nProcessing GLOBAL TRANSVERSE sublattice correlations for: {base_pattern}")
            print("-"*80)
            
            # Create transverse-weighted AA, BB, AB correlation data
            (freq_aa_t, spec_aa_t), (freq_bb_t, spec_bb_t), (freq_ab_t, spec_ab_t) = \
                create_global_transverse_sublattice_data(h_values, h_dirs, base_pattern)
            
            # Check if we have data
            if not spec_aa_t and not spec_bb_t and not spec_ab_t:
                print(f"  ⚠ No transverse sublattice data found for {base_pattern}, skipping...")
                continue
            
            # Create safe filename
            safe_pattern = base_pattern.replace("/", "_").replace(" ", "_")
            
            # 1. Global transverse sublattice comparison plot - in transverse subdirectory
            comparison_file = os.path.join(SUBDIRS['transverse'], f"{safe_pattern}_global_transverse_sublattice_comparison.png")
            create_sublattice_comparison_plot(h_values, freq_aa_t, spec_aa_t, freq_bb_t, spec_bb_t, 
                                             freq_ab_t, spec_ab_t, 
                                             f"{base_pattern} (Global Transverse)", comparison_file)
            
            # 2. Global transverse sublattice heatmap
            heatmap_file = os.path.join(SUBDIRS['transverse'], f"{safe_pattern}_global_transverse_sublattice_heatmap.png")
            create_sublattice_heatmap(h_values, freq_aa_t, spec_aa_t, freq_bb_t, spec_bb_t, 
                                     freq_ab_t, spec_ab_t, 
                                     f"{base_pattern} (Global Transverse)", heatmap_file)
            
            # 3. Global transverse sublattice animation
            anim_file = os.path.join(SUBDIRS['transverse'], f"{safe_pattern}_global_transverse_sublattice_animation.gif")
            create_sublattice_animation(h_values, freq_aa_t, spec_aa_t, freq_bb_t, spec_bb_t, 
                                       freq_ab_t, spec_ab_t, 
                                       f"{base_pattern} (Global Transverse)", anim_file)
            
            print(f"  ✓ Completed global transverse sublattice plots for {base_pattern}")
        except Exception as e:
            print(f"  ✗ Error processing global transverse sublattice for {base_pattern}: {e}")
            continue
    
    # Process combined SmSp + SzSz global transverse sublattice correlations
    if sublattice_combined_pairs:
        print("\n" + "="*80)
        print("Processing COMBINED SmSp + SzSz GLOBAL TRANSVERSE sublattice correlations...")
        print("="*80)
        
        for smsp_pattern, szsz_pattern, base_name in sublattice_combined_pairs:
            try:
                print(f"\nProcessing combined global transverse sublattice for: {base_name}")
                print(f"  SmSp: {smsp_pattern}")
                print(f"  SzSz: {szsz_pattern}")
                print("-"*80)
                
                # Get SmSp global transverse sublattice data
                (freq_aa_sm_t, spec_aa_sm_t), (freq_bb_sm_t, spec_bb_sm_t), (freq_ab_sm_t, spec_ab_sm_t) = \
                    create_global_transverse_sublattice_data(h_values, h_dirs, smsp_pattern)
                
                # Get SzSz global transverse sublattice data
                (freq_aa_sz_t, spec_aa_sz_t), (freq_bb_sz_t, spec_bb_sz_t), (freq_ab_sz_t, spec_ab_sz_t) = \
                    create_global_transverse_sublattice_data(h_values, h_dirs, szsz_pattern)
                
                # Check if we have data for both
                if (not spec_aa_sm_t and not spec_aa_sz_t) or (not spec_bb_sm_t and not spec_bb_sz_t):
                    print(f"  ⚠ Insufficient data for combined global transverse sublattice, skipping...")
                    continue
                
                # Combine: Total = SmSp * SMSP_FACTOR + SzSz
                freq_aa_combined_t = {}
                spec_aa_combined_t = {}
                freq_bb_combined_t = {}
                spec_bb_combined_t = {}
                freq_ab_combined_t = {}
                spec_ab_combined_t = {}
                
                for h in h_values:
                    if h in spec_aa_sm_t and h in spec_aa_sz_t:
                        freq_aa_combined_t[h] = freq_aa_sm_t[h]
                        spec_aa_combined_t[h] = spec_aa_sm_t[h] * SMSP_FACTOR + spec_aa_sz_t[h]
                    
                    if h in spec_bb_sm_t and h in spec_bb_sz_t:
                        freq_bb_combined_t[h] = freq_bb_sm_t[h]
                        spec_bb_combined_t[h] = spec_bb_sm_t[h] * SMSP_FACTOR + spec_bb_sz_t[h]
                    
                    if h in spec_ab_sm_t and h in spec_ab_sz_t:
                        freq_ab_combined_t[h] = freq_ab_sm_t[h]
                        spec_ab_combined_t[h] = spec_ab_sm_t[h] * SMSP_FACTOR + spec_ab_sz_t[h]
                
                # Create safe filename
                safe_name = base_name.replace("/", "_").replace(" ", "_")
                
                # Create combined global transverse plots
                comparison_file = os.path.join(SUBDIRS['transverse'], f"{safe_name}_combined_global_transverse_sublattice_comparison.png")
                create_sublattice_comparison_plot(h_values, freq_aa_combined_t, spec_aa_combined_t, 
                                                 freq_bb_combined_t, spec_bb_combined_t, 
                                                 freq_ab_combined_t, spec_ab_combined_t, 
                                                 f"{base_name} (Global Transverse, SmSp*{SMSP_FACTOR} + SzSz)", comparison_file)
                
                heatmap_file = os.path.join(SUBDIRS['transverse'], f"{safe_name}_combined_global_transverse_sublattice_heatmap.png")
                create_sublattice_heatmap(h_values, freq_aa_combined_t, spec_aa_combined_t, 
                                         freq_bb_combined_t, spec_bb_combined_t, 
                                         freq_ab_combined_t, spec_ab_combined_t, 
                                         f"{base_name} (Global Transverse, SmSp*{SMSP_FACTOR} + SzSz)", heatmap_file)
                
                anim_file = os.path.join(SUBDIRS['transverse'], f"{safe_name}_combined_global_transverse_sublattice_animation.gif")
                create_sublattice_animation(h_values, freq_aa_combined_t, spec_aa_combined_t, 
                                           freq_bb_combined_t, spec_bb_combined_t, 
                                           freq_ab_combined_t, spec_ab_combined_t, 
                                           f"{base_name} (Global Transverse, SmSp*{SMSP_FACTOR} + SzSz)", anim_file)
                
                print(f"  ✓ Completed combined global transverse sublattice plots for {base_name}")
            except Exception as e:
                print(f"  ✗ Error processing combined global transverse sublattice for {base_name}: {e}")
                continue
    
    # Compute experimental angle from magnetization data
    print("\n" + "="*80)
    print("Computing experimental angle from magnetization...")
    print("="*80)
    EXPERIMENTAL_ANGLE = calculate_experimental_angle(h_values, h_dirs)
    print(f"Experimental angle: θ = {np.degrees(EXPERIMENTAL_ANGLE):.4f}°")
    
    # Process global transverse EXPERIMENTAL sublattice correlations
    print("\n" + "="*80)
    print("Processing GLOBAL TRANSVERSE EXPERIMENTAL sublattice correlations (AA/BB/AB)...")
    print(f"Sublattice A: {SUBLATTICE_A}, Sublattice B: {SUBLATTICE_B}")
    print(f"Q-vector: {Q_VECTOR}")
    print(f"Experimental angle: θ = {np.degrees(EXPERIMENTAL_ANGLE):.4f}°")
    print("Applying transverse + experimental rotation operator")
    print("="*80)
    
    # Process each base pattern with experimental transverse weighting
    for base_pattern in sublattice_base_patterns:
        try:
            print(f"\nProcessing GLOBAL TRANSVERSE EXPERIMENTAL sublattice correlations for: {base_pattern}")
            print("-"*80)
            
            # Create experimental transverse-weighted AA, BB, AB correlation data
            (freq_aa_te, spec_aa_te), (freq_bb_te, spec_bb_te), (freq_ab_te, spec_ab_te) = \
                create_global_transverse_experimental_sublattice_data(h_values, h_dirs, base_pattern, EXPERIMENTAL_ANGLE)
            
            # Check if we have data
            if not spec_aa_te and not spec_bb_te and not spec_ab_te:
                print(f"  ⚠ No experimental transverse sublattice data found for {base_pattern}, skipping...")
                continue
            
            # Create safe filename
            safe_pattern = base_pattern.replace("/", "_").replace(" ", "_")
            
            # 1. Global transverse experimental sublattice comparison plot - in transverse_experimental subdirectory
            comparison_file = os.path.join(SUBDIRS['transverse_experimental'], f"{safe_pattern}_global_transverse_experimental_sublattice_comparison.png")
            create_sublattice_comparison_plot(h_values, freq_aa_te, spec_aa_te, freq_bb_te, spec_bb_te, 
                                             freq_ab_te, spec_ab_te, 
                                             f"{base_pattern} (Global Transverse Experimental)", comparison_file)
            
            # 2. Global transverse experimental sublattice heatmap
            heatmap_file = os.path.join(SUBDIRS['transverse_experimental'], f"{safe_pattern}_global_transverse_experimental_sublattice_heatmap.png")
            create_sublattice_heatmap(h_values, freq_aa_te, spec_aa_te, freq_bb_te, spec_bb_te, 
                                     freq_ab_te, spec_ab_te, 
                                     f"{base_pattern} (Global Transverse Experimental)", heatmap_file)
            
            # 3. Global transverse experimental sublattice animation
            anim_file = os.path.join(SUBDIRS['transverse_experimental'], f"{safe_pattern}_global_transverse_experimental_sublattice_animation.gif")
            create_sublattice_animation(h_values, freq_aa_te, spec_aa_te, freq_bb_te, spec_bb_te, 
                                       freq_ab_te, spec_ab_te, 
                                       f"{base_pattern} (Global Transverse Experimental)", anim_file)
            
            print(f"  ✓ Completed global transverse experimental sublattice plots for {base_pattern}")
        except Exception as e:
            print(f"  ✗ Error processing global transverse experimental sublattice for {base_pattern}: {e}")
            continue
    
    # Process combined SmSp + SzSz global transverse experimental sublattice correlations
    if sublattice_combined_pairs:
        print("\n" + "="*80)
        print("Processing COMBINED SmSp + SzSz GLOBAL TRANSVERSE EXPERIMENTAL sublattice correlations...")
        print("="*80)
        
        for smsp_pattern, szsz_pattern, base_name in sublattice_combined_pairs:
            try:
                print(f"\nProcessing combined global transverse experimental sublattice for: {base_name}")
                print(f"  SmSp: {smsp_pattern}")
                print(f"  SzSz: {szsz_pattern}")
                print("-"*80)
                
                # Get SmSp global transverse experimental sublattice data
                (freq_aa_sm_te, spec_aa_sm_te), (freq_bb_sm_te, spec_bb_sm_te), (freq_ab_sm_te, spec_ab_sm_te) = \
                    create_global_transverse_experimental_sublattice_data(h_values, h_dirs, smsp_pattern, EXPERIMENTAL_ANGLE)
                
                # Get SzSz global transverse experimental sublattice data
                (freq_aa_sz_te, spec_aa_sz_te), (freq_bb_sz_te, spec_bb_sz_te), (freq_ab_sz_te, spec_ab_sz_te) = \
                    create_global_transverse_experimental_sublattice_data(h_values, h_dirs, szsz_pattern, EXPERIMENTAL_ANGLE)
                
                # Check if we have data for both
                if (not spec_aa_sm_te and not spec_aa_sz_te) or (not spec_bb_sm_te and not spec_bb_sz_te):
                    print(f"  ⚠ Insufficient data for combined global transverse experimental sublattice, skipping...")
                    continue
                
                # Combine: Total = SmSp * SMSP_FACTOR + SzSz
                freq_aa_combined_te = {}
                spec_aa_combined_te = {}
                freq_bb_combined_te = {}
                spec_bb_combined_te = {}
                freq_ab_combined_te = {}
                spec_ab_combined_te = {}
                
                for h in h_values:
                    if h in spec_aa_sm_te and h in spec_aa_sz_te:
                        freq_aa_combined_te[h] = freq_aa_sm_te[h]
                        spec_aa_combined_te[h] = spec_aa_sm_te[h] * SMSP_FACTOR + spec_aa_sz_te[h]
                    
                    if h in spec_bb_sm_te and h in spec_bb_sz_te:
                        freq_bb_combined_te[h] = freq_bb_sm_te[h]
                        spec_bb_combined_te[h] = spec_bb_sm_te[h] * SMSP_FACTOR + spec_bb_sz_te[h]
                    
                    if h in spec_ab_sm_te and h in spec_ab_sz_te:
                        freq_ab_combined_te[h] = freq_ab_sm_te[h]
                        spec_ab_combined_te[h] = spec_ab_sm_te[h] * SMSP_FACTOR + spec_ab_sz_te[h]
                
                # Create safe filename
                safe_name = base_name.replace("/", "_").replace(" ", "_")
                
                # Create combined global transverse experimental plots
                comparison_file = os.path.join(SUBDIRS['transverse_experimental'], f"{safe_name}_combined_global_transverse_experimental_sublattice_comparison.png")
                create_sublattice_comparison_plot(h_values, freq_aa_combined_te, spec_aa_combined_te, 
                                                 freq_bb_combined_te, spec_bb_combined_te, 
                                                 freq_ab_combined_te, spec_ab_combined_te, 
                                                 f"{base_name} (Global Transverse Experimental, SmSp*{SMSP_FACTOR} + SzSz)", comparison_file)
                
                heatmap_file = os.path.join(SUBDIRS['transverse_experimental'], f"{safe_name}_combined_global_transverse_experimental_sublattice_heatmap.png")
                create_sublattice_heatmap(h_values, freq_aa_combined_te, spec_aa_combined_te, 
                                         freq_bb_combined_te, spec_bb_combined_te, 
                                         freq_ab_combined_te, spec_ab_combined_te, 
                                         f"{base_name} (Global Transverse Experimental, SmSp*{SMSP_FACTOR} + SzSz)", heatmap_file)
                
                anim_file = os.path.join(SUBDIRS['transverse_experimental'], f"{safe_name}_combined_global_transverse_experimental_sublattice_animation.gif")
                create_sublattice_animation(h_values, freq_aa_combined_te, spec_aa_combined_te, 
                                           freq_bb_combined_te, spec_bb_combined_te, 
                                           freq_ab_combined_te, spec_ab_combined_te, 
                                           f"{base_name} (Global Transverse Experimental, SmSp*{SMSP_FACTOR} + SzSz)", anim_file)
                
                print(f"  ✓ Completed combined global transverse experimental sublattice plots for {base_name}")
            except Exception as e:
                print(f"  ✗ Error processing combined global transverse experimental sublattice for {base_name}: {e}")
                continue
    
    # Create magnetization plot
    print("\n" + "="*80)
    print("Creating magnetization plot...")
    print("="*80)
    try:
        magnetization_file = os.path.join(SUBDIRS['magnetization'], "magnetization_vs_field.png")
        create_magnetization_plot(h_values, h_dirs, magnetization_file)
        print("  ✓ Magnetization plot created")
    except Exception as e:
        print(f"  ✗ Error creating magnetization plot: {e}")
    
    # Create summary report
    print("\n" + "="*80)
    print("Creating summary report...")
    print("="*80)
    try:
        create_summary_report(h_values, h_dirs, all_species, do_channels, 
                             species_combinations, sublattice_base_patterns)
        print("  ✓ Summary report created")
    except Exception as e:
        print(f"  ✗ Error creating summary report: {e}")
    
    print("\n" + "="*80)
    print(f"\nAll plots saved to organized subdirectories in: {OUTPUT_DIR}")
    print("\nOutput Structure:")
    print(f"  0_summary/              - Summary report and key figures")
    print(f"  1_individual_species/   - Individual species plots")
    print(f"  2_DO_channels/          - DO channel (SF + NSF) plots")
    print(f"  3_combined_SmSp_SzSz/   - Combined SmSp + SzSz plots")
    print(f"  4_sublattice_correlations/ - Sublattice AA/BB/AB plots")
    print(f"  5_global_transverse/    - Global transverse sublattice plots")
    print(f"  5a_global_transverse_experimental/ - Global transverse experimental sublattice plots")
    print(f"  6_magnetization/        - Magnetization vs field plots")
    print("="*80)

if __name__ == "__main__":
    main()
