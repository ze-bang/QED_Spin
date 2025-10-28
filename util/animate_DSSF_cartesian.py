#!/usr/bin/env python3
"""
Script to read spectral_beta_inf.dat files across all h=# directories
and create animated/stacked plots for each species in CARTESIAN BASIS (SxSx, SySy, SzSz, SxSy, SySz, SxSz).

Transforms from the spherical basis components that are available in the dataset
(SmSp ≡ S- S+, SmSm ≡ S- S-, SmSz ≡ S- Sz, SpSz ≡ S+ Sz, SzSz) to the Cartesian
basis using:
- Sx = (S+ + S-)/2
- Sy = (S+ - S-)/(2i)
- Sz = Sz

Hermiticity relates the missing spherical correlators via complex conjugation, so
only the five measured channels above are required. The Cartesian components are
then reconstructed through
- SxSx = 0.5 * (Re[SmSp] + Re[SmSm])
- SySy = 0.5 * (Re[SmSp] - Re[SmSm])
- SzSz = Re[SzSz]
- SxSy = 0.5 * (Im[SmSp] - Im[SmSm])
- SxSz = 0.5 * Re[SmSz + SpSz]
- SzSx = 0.5 * Re[(SmSz + SpSz)*]
- SySz = 0.5 * Re[i (SmSz - SpSz)]
and similarly for other combinations. The real parts are taken before plotting so
that the visualised spectra remain real-valued.
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
BASE_DIR = "/scratch/zhouzb79/DSSF_PCD_mag_field_sweep_CZO_pi_4"
OUTPUT_DIR = os.path.join(BASE_DIR, "spectral_animations_cartesian")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Create organized subdirectories
SUBDIRS = {
    'individual': os.path.join(OUTPUT_DIR, "1_individual_cartesian_components"),
    'combined': os.path.join(OUTPUT_DIR, "2_combined_diagonal"),
    'off_diagonal': os.path.join(OUTPUT_DIR, "3_off_diagonal_components"),
    'experimental': os.path.join(OUTPUT_DIR, "4_experimental_channels"),
    'sublattice': os.path.join(OUTPUT_DIR, "5_sublattice_cartesian"),
    'transverse': os.path.join(OUTPUT_DIR, "6_global_transverse_cartesian"),
    'magnetization': os.path.join(OUTPUT_DIR, "7_magnetization"),
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
FIELD_DIRECTION = np.array([1, 1, 1])  # Default: [111] direction
FIELD_DIRECTION = FIELD_DIRECTION / np.linalg.norm(FIELD_DIRECTION)  # Normalize

# Pyrochlore lattice local z-axes (sublattice quantization axes)
PYROCHLORE_Z_AXES = np.array([
    [1, 1, 1],      # sublattice 0
    [-1, -1, 1],    # sublattice 1
    [-1, 1, -1],    # sublattice 2
    [1, -1, -1]     # sublattice 3
]) / np.sqrt(3)  # Normalize

# Q-vector for transverse operator (default: (0,0,0), can be modified)
Q_VECTOR = np.array([0, 0, 0])

# Experimental channel rotation angle (in radians)
# Defines the mixing between SzSz, SxSx, SxSz, and SzSx components
# Channel = cos²(θ) SzSz + sin²(θ) SxSx + sin(θ)cos(θ)(SxSz + SzSx)
# This will be computed from magnetization data
EXPERIMENTAL_ANGLE = None  # Will be set after computing magnetization

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


def extract_q_pattern(species_name):
    """Extract Q-vector pattern from species name (e.g., 'q_Qx0_Qy0_Qz0')"""
    # Match Q-vector pattern: q_Qx<number>_Qy<number>_Qz<number>
    # Number can include digits, dots, minus signs, but not trailing underscores
    match = re.search(r'(q_Qx[0-9.-]+_Qy[0-9.-]+_Qz[0-9.-]+)', species_name)
    if match:
        # Remove any trailing underscores that might have been captured
        pattern = match.group(1).rstrip('_')
        return pattern
    return None


def extract_suffix(species_name):
    """Extract suffix (_SF, _NSF, _DO) from species name"""
    if species_name.endswith('_SF'):
        return '_SF'
    elif species_name.endswith('_NSF'):
        return '_NSF'
    elif species_name.endswith('_DO'):
        return '_DO'
    return ''


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
        
        # Calculate Sx and Sy from S+ and S-
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
        print(f"  Using default angle.")
        return 0.31416104734  # Fallback to previous value


def construct_cartesian_component(h_values, h_dirs, q_pattern, suffix, cart_component):
    """
    Construct a Cartesian component (SxSx, SySy, SzSz, SxSy, SySz, SxSz) from spherical components.
    
    Transformations:
    - SxSx = (SmSp + SmSm + SpSp + SpSm)/4
    - SySy = -(SmSp - SmSm - SpSp + SpSm)/4
    - SzSz = SzSz
    - SxSy = -i(SmSp + SmSm - SpSp - SpSm)/4
    - SySz = -i(SpSz - SmSz)/2
    - SxSz = (SpSz + SmSz)/2
    
    Note: SpSm = (SmSp)*, SmSm = (SpSp)*, so we assume hermiticity for spectral functions.
    For real spectral functions, we take real parts.
    
    Parameters:
    - h_values: list of h field values
    - h_dirs: dict mapping h -> directory
    - q_pattern: Q-vector pattern string
    - suffix: suffix (_SF, _NSF, _DO, etc.)
    - cart_component: which Cartesian component ('SxSx', 'SySy', 'SzSz', 'SxSy', 'SySz', 'SxSz')
    
    Returns:
    - freq_data: dict mapping h -> frequency array
    - spectral_data: dict mapping h -> spectral array
    """
    
    freq_data = {}
    spectral_data = {}
    
    # Map of which spherical components are needed for each Cartesian component
    component_map = {
        'SxSx': ['SmSp', 'SmSm'],
        'SySy': ['SmSp', 'SmSm'],
        'SzSz': ['SzSz'],
        'SxSy': ['SmSp', 'SmSm'],
        'SySz': ['SmSz', 'SpSz'],
        'SxSz': ['SmSz', 'SpSz'],
        'SzSx': ['SmSz', 'SpSz']
    }
    
    required_components = component_map[cart_component]
    
    for h in h_values:
        h_dir = h_dirs[h]
        
        # Read required spherical components
        spherical_data = {}
        for sph_comp in required_components:
            species_name = f"{sph_comp}_{q_pattern}{suffix}"
            freq, spec = read_spectral_data(h_dir, species_name)
            if freq is not None and spec is not None:
                spherical_data[sph_comp] = (freq, spec)
        
        # Check if we have all required components
        if len(spherical_data) != len(required_components):
            continue  # Missing components, skip this h value
        
        # Use first component as reference frequency grid
        ref_comp = required_components[0]
        freq_ref = spherical_data[ref_comp][0]
        
        # Interpolate all to reference grid
        interpolated = {}
        for sph_comp in required_components:
            freq_comp, spec_comp = spherical_data[sph_comp]
            if len(freq_comp) == len(freq_ref) and np.allclose(freq_comp, freq_ref):
                interpolated[sph_comp] = spec_comp
            else:
                interpolated[sph_comp] = np.interp(freq_ref, freq_comp, spec_comp)
        
        # Pre-fetch commonly used spherical pieces
        smsp = interpolated.get('SmSp')
        smsm = interpolated.get('SmSm')
        smsz = interpolated.get('SmSz')
        spsz = interpolated.get('SpSz')
        szsz = interpolated.get('SzSz')
        
        # Construct Cartesian component from real/imag combinations
        if cart_component == 'SxSx':
            spec_cart = 0.5 * (np.real(smsp) + np.real(smsm))
        
        elif cart_component == 'SySy':
            spec_cart = 0.5 * (np.real(smsp) - np.real(smsm))
        
        elif cart_component == 'SzSz':
            spec_cart = np.real(szsz)
        
        elif cart_component == 'SxSy':
            spec_cart = 0.5 * (np.imag(smsp) - np.imag(smsm))
        
        elif cart_component == 'SySz':
            spec_cart = 0.5 * np.real(1j * (smsz - spsz))
        
        elif cart_component == 'SxSz':
            spec_cart = 0.5 * np.real(smsz + spsz)
        
        elif cart_component == 'SzSx':
            spec_cart = 0.5 * np.real(np.conjugate(smsz + spsz))
        
        freq_data[h] = freq_ref
        spectral_data[h] = spec_cart
    
    return freq_data, spectral_data


def construct_cartesian_sublattice_component(h_values, h_dirs, q_pattern, suffix, 
                                             sub1, sub2, cart_component):
    """
    Construct a Cartesian component for a specific sublattice pair.
    
    Parameters:
    - h_values: list of h field values
    - h_dirs: dict mapping h -> directory
    - q_pattern: Q-vector pattern string
    - suffix: suffix (_SF, _NSF, _DO, etc.)
    - sub1, sub2: sublattice indices
    - cart_component: which Cartesian component ('SxSx', 'SySy', 'SzSz', 'SxSy', 'SySz', 'SxSz')
    
    Returns:
    - freq_data: dict mapping h -> frequency array
    - spectral_data: dict mapping h -> spectral array
    """
    
    freq_data = {}
    spectral_data = {}
    
    # Map of which spherical components are needed
    component_map = {
        'SxSx': ['SmSp', 'SmSm'],
        'SySy': ['SmSp', 'SmSm'],
        'SzSz': ['SzSz'],
        'SxSy': ['SmSp', 'SmSm'],
        'SySz': ['SmSz', 'SpSz'],
        'SxSz': ['SmSz', 'SpSz'],
        'SzSx': ['SmSz', 'SpSz']
    }
    
    required_components = component_map[cart_component]
    
    for h in h_values:
        h_dir = h_dirs[h]
        
        # Read required spherical components for this sublattice pair
        spherical_data = {}
        for sph_comp in required_components:
            species_name = f"{sph_comp}_{q_pattern}_sub{sub1}_sub{sub2}{suffix}"
            freq, spec = read_spectral_data(h_dir, species_name)
            if freq is not None and spec is not None:
                spherical_data[sph_comp] = (freq, spec)
        
        # Check if we have all required components
        if len(spherical_data) != len(required_components):
            continue
        
        # Use first component as reference frequency grid
        ref_comp = required_components[0]
        freq_ref = spherical_data[ref_comp][0]
        
        # Interpolate all to reference grid
        interpolated = {}
        for sph_comp in required_components:
            freq_comp, spec_comp = spherical_data[sph_comp]
            if len(freq_comp) == len(freq_ref) and np.allclose(freq_comp, freq_ref):
                interpolated[sph_comp] = spec_comp
            else:
                interpolated[sph_comp] = np.interp(freq_ref, freq_comp, spec_comp)
        
        # Construct Cartesian component (same logic as the global case)
        smsp = interpolated.get('SmSp')
        smsm = interpolated.get('SmSm')
        smsz = interpolated.get('SmSz')
        spsz = interpolated.get('SpSz')
        szsz = interpolated.get('SzSz')

        if cart_component == 'SxSx':
            spec_cart = 0.5 * (np.real(smsp) + np.real(smsm))
        elif cart_component == 'SySy':
            spec_cart = 0.5 * (np.real(smsp) - np.real(smsm))
        elif cart_component == 'SzSz':
            spec_cart = np.real(szsz)
        elif cart_component == 'SxSy':
            spec_cart = 0.5 * (np.imag(smsp) - np.imag(smsm))
        elif cart_component == 'SySz':
            spec_cart = 0.5 * np.real(1j * (smsz - spsz))
        elif cart_component == 'SxSz':
            spec_cart = 0.5 * np.real(smsz + spsz)
        
        freq_data[h] = freq_ref
        spectral_data[h] = spec_cart
    
    return freq_data, spectral_data


def create_cartesian_sublattice_correlation_data(h_values, h_dirs, q_pattern, suffix, cart_component):
    """
    Create AA, BB, and AB correlation data for a Cartesian component by summing over sublattice pairs.
    
    Returns:
    - Three tuples (freq_data, spectral_data) for AA, BB, AB correlations
    """
    freq_data_aa = {}
    spectral_data_aa = {}
    freq_data_bb = {}
    spectral_data_bb = {}
    freq_data_ab = {}
    spectral_data_ab = {}
    
    # Get all possible sublattice pairs (assuming 4 sublattices: 0, 1, 2, 3)
    all_sublattices = set(SUBLATTICE_A + SUBLATTICE_B)
    
    for h in h_values:
        aa_spectral = None
        bb_spectral = None
        ab_spectral = None
        freq_ref = None
        
        for sub1 in all_sublattices:
            for sub2 in all_sublattices:
                freq_dict, spec_dict = construct_cartesian_sublattice_component(
                    [h], h_dirs, q_pattern, suffix, sub1, sub2, cart_component
                )
                
                if h not in spec_dict:
                    continue
                
                freq = freq_dict[h]
                spec = spec_dict[h]
                
                # Set reference frequency grid
                if freq_ref is None:
                    freq_ref = freq
                    aa_spectral = np.zeros_like(spec)
                    bb_spectral = np.zeros_like(spec)
                    ab_spectral = np.zeros_like(spec)
                
                # Interpolate if needed
                if len(freq) != len(freq_ref) or not np.allclose(freq, freq_ref):
                    spec = np.interp(freq_ref, freq, spec)
                
                # Categorize correlation
                if sub1 in SUBLATTICE_A and sub2 in SUBLATTICE_A:
                    aa_spectral += spec
                elif sub1 in SUBLATTICE_B and sub2 in SUBLATTICE_B:
                    bb_spectral += spec
                elif (sub1 in SUBLATTICE_A and sub2 in SUBLATTICE_B) or \
                     (sub1 in SUBLATTICE_B and sub2 in SUBLATTICE_A):
                    ab_spectral += spec
        
        # Store results
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


def calculate_transverse_weight(sub1, sub2, Q=None):
    """Calculate the transverse operator weight for a sublattice pair."""
    if Q is None:
        Q = Q_VECTOR
    
    z_mu = PYROCHLORE_Z_AXES[sub1]
    z_nu = PYROCHLORE_Z_AXES[sub2]
    
    dot_product = np.dot(z_mu, z_nu)
    
    Q_norm = np.linalg.norm(Q)
    if Q_norm < 1e-10:
        return dot_product
    
    Q_hat = Q / Q_norm
    z_mu_dot_Q = np.dot(z_mu, Q_hat)
    z_nu_dot_Q = np.dot(z_nu, Q_hat)
    
    return dot_product - z_mu_dot_Q * z_nu_dot_Q


def construct_experimental_channel(h_values, h_dirs, q_pattern, suffix, theta):
    """
    Construct an experimental channel with angle-dependent mixing:
    S_exp(θ) = cos²(θ) SzSz + sin²(θ) SxSx + sin(θ)cos(θ)(SxSz + SzSx)
    
    Note: For same-site correlations, SxSz ≠ SzSx in general, so we need both components.
    
    Parameters:
    - h_values: list of h field values
    - h_dirs: dict mapping h -> directory
    - q_pattern: Q-vector pattern string
    - suffix: suffix (_SF, _NSF, _DO, etc.)
    - theta: rotation angle in radians (single value)
    
    Returns:
    - freq_data: dict mapping h -> frequency array
    - spectral_data: dict mapping h -> spectral array
    """
    # Calculate mixing coefficients
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    cos2_theta = cos_theta**2
    sin2_theta = sin_theta**2
    sin_cos_theta = sin_theta * cos_theta
    
    # Construct the required Cartesian components
    # Need: SzSz, SxSx, SxSz
    # Note: For Q=0 (same-site), SzSx = SxSz for real expectation values
    freq_szz, spec_szz = construct_cartesian_component(h_values, h_dirs, q_pattern, suffix, 'SzSz')
    freq_sxx, spec_sxx = construct_cartesian_component(h_values, h_dirs, q_pattern, suffix, 'SxSx')
    freq_sxz, spec_sxz = construct_cartesian_component(h_values, h_dirs, q_pattern, suffix, 'SxSz')
    
    freq_data = {}
    spectral_data = {}
    
    # Construct experimental channel for each h value
    for h in h_values:
        if h not in spec_szz or h not in spec_sxx or h not in spec_sxz:
            continue
        
        # Use SzSz as reference frequency grid
        freq_ref = freq_szz[h]
        
        # Get spectral data
        szz = spec_szz[h]
        sxx = spec_sxx[h]
        sxz = spec_sxz[h]
        
        # Interpolate if needed
        freq_sxx_h = freq_sxx[h]
        if len(freq_sxx_h) != len(freq_ref) or not np.allclose(freq_sxx_h, freq_ref):
            sxx = np.interp(freq_ref, freq_sxx_h, sxx)
        
        freq_sxz_h = freq_sxz[h]
        if len(freq_sxz_h) != len(freq_ref) or not np.allclose(freq_sxz_h, freq_ref):
            sxz = np.interp(freq_ref, freq_sxz_h, sxz)
        
        # Construct experimental channel: cos²(θ) SzSz + sin²(θ) SxSx + sin(θ)cos(θ)(SxSz + SzSx)
        spec_exp = cos2_theta * szz + sin2_theta * sxx + 2 * sin_cos_theta * sxz
        
        freq_data[h] = freq_ref
        spectral_data[h] = spec_exp
    
    return freq_data, spectral_data


def combine_sf_nsf_to_do(freq_sf, spec_sf, freq_nsf, spec_nsf, h_values):
    """
    Combine SF and NSF channels to create DO channel (SF + NSF).
    
    Parameters:
    - freq_sf: dict mapping h -> frequency array for SF
    - spec_sf: dict mapping h -> spectral array for SF
    - freq_nsf: dict mapping h -> frequency array for NSF
    - spec_nsf: dict mapping h -> spectral array for NSF
    - h_values: list of h field values
    
    Returns:
    - freq_data: dict mapping h -> frequency array
    - spectral_data: dict mapping h -> spectral array (SF + NSF)
    """
    freq_data = {}
    spectral_data = {}
    
    for h in h_values:
        if h not in spec_sf or h not in spec_nsf:
            continue
        
        # Use SF as reference frequency grid
        freq_ref = freq_sf[h]
        sf = spec_sf[h]
        
        # Interpolate NSF to match SF frequency grid if needed
        freq_nsf_h = freq_nsf[h]
        nsf = spec_nsf[h]
        
        if len(freq_nsf_h) != len(freq_ref) or not np.allclose(freq_nsf_h, freq_ref):
            nsf = np.interp(freq_ref, freq_nsf_h, nsf)
        
        # DO = SF + NSF
        spec_do = sf + nsf
        
        freq_data[h] = freq_ref
        spectral_data[h] = spec_do
    
    return freq_data, spectral_data


def read_spin_configuration(h_dir):
    """Read spin configuration file and calculate magnetization"""
    file_path = os.path.join(h_dir, "structure_factor_results", "beta_inf", "spin_configuration.txt")
    
    if not os.path.exists(file_path):
        return None
    
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()[1:]  # Skip header
        
        sp_values = []
        sm_values = []
        sz_values = []
        site_indices = []
        
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 4:
                site_idx = int(parts[0])
                site_indices.append(site_idx)
                
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
        site_indices = np.array(site_indices)
        
        # Calculate Sx and Sy from S+ and S-
        sx_values = (sp_values + sm_values) / 2.0
        sy_values = -1j * (sp_values - sm_values) / 2.0
        
        # Total magnetization (local frame)
        mag_x = np.sum(sx_values.real)
        mag_y = np.sum(sy_values.real)
        mag_z = np.sum(sz_values)
        mag_total = np.sqrt(mag_x**2 + mag_y**2 + mag_z**2)
        
        n_sites = len(sz_values)
        mag_x_per_site = mag_x / n_sites
        mag_y_per_site = mag_y / n_sites
        mag_z_per_site = mag_z / n_sites
        mag_total_per_site = mag_total / n_sites
        
        # Rotated local magnetization (applying theta rotation in local frame)
        cos_theta = np.cos(EXPERIMENTAL_ANGLE)
        sin_theta = np.sin(EXPERIMENTAL_ANGLE)
        
        mag_x_rotated_local = cos_theta * mag_x - sin_theta * mag_z
        mag_y_rotated_local = mag_y
        mag_z_rotated_local = sin_theta * mag_x + cos_theta * mag_z
        mag_total_rotated_local = np.sqrt(mag_x_rotated_local**2 + mag_y_rotated_local**2 + mag_z_rotated_local**2)
        
        mag_x_rotated_local_per_site = mag_x_rotated_local / n_sites
        mag_y_rotated_local_per_site = mag_y_rotated_local / n_sites
        mag_z_rotated_local_per_site = mag_z_rotated_local / n_sites
        mag_total_rotated_local_per_site = mag_total_rotated_local / n_sites
        
        # Global magnetization
        n_sites_per_sublattice = n_sites // 4
        mag_global = np.zeros(3)
        
        for i in range(n_sites):
            sublattice = i % 4
            s_local = np.array([
                sx_values[i].real,
                sy_values[i].real,
                sz_values[i]
            ])

            s_local_rotated_x = np.cos(EXPERIMENTAL_ANGLE) * s_local[0] - np.sin(EXPERIMENTAL_ANGLE) * s_local[2]
            s_local_rotated_y = s_local[1]
            s_local_rotated_z = np.sin(EXPERIMENTAL_ANGLE) * s_local[0] + np.cos(EXPERIMENTAL_ANGLE) * s_local[2]

            s_local_rotated = np.array([s_local_rotated_x, s_local_rotated_y, s_local_rotated_z])

            s_global = (s_local_rotated[0] * localframe[0, sublattice, :] +
                       s_local_rotated[1] * localframe[1, sublattice, :] +
                       s_local_rotated[2] * localframe[2, sublattice, :])
            mag_global += s_global
        
        mag_global_x = mag_global[0]
        mag_global_y = mag_global[1]
        mag_global_z = mag_global[2]
        mag_global_total = np.linalg.norm(mag_global)
        
        mag_global_x_per_site = mag_global_x / n_sites
        mag_global_y_per_site = mag_global_y / n_sites
        mag_global_z_per_site = mag_global_z / n_sites
        mag_global_total_per_site = mag_global_total / n_sites
        
        mag_along_field = np.dot(mag_global, FIELD_DIRECTION)
        mag_along_field_per_site = mag_along_field / n_sites
        
        return {
            'total_magnetization': mag_total,
            'magnetization_per_site': mag_total_per_site,
            'mag_x': mag_x,
            'mag_y': mag_y,
            'mag_z': mag_z,
            'mag_x_per_site': mag_x_per_site,
            'mag_y_per_site': mag_y_per_site,
            'mag_z_per_site': mag_z_per_site,
            'mag_x_rotated_local': mag_x_rotated_local,
            'mag_y_rotated_local': mag_y_rotated_local,
            'mag_z_rotated_local': mag_z_rotated_local,
            'mag_total_rotated_local': mag_total_rotated_local,
            'mag_x_rotated_local_per_site': mag_x_rotated_local_per_site,
            'mag_y_rotated_local_per_site': mag_y_rotated_local_per_site,
            'mag_z_rotated_local_per_site': mag_z_rotated_local_per_site,
            'mag_total_rotated_local_per_site': mag_total_rotated_local_per_site,
            'mag_global_x': mag_global_x,
            'mag_global_y': mag_global_y,
            'mag_global_z': mag_global_z,
            'mag_global_total': mag_global_total,
            'mag_global_x_per_site': mag_global_x_per_site,
            'mag_global_y_per_site': mag_global_y_per_site,
            'mag_global_z_per_site': mag_global_z_per_site,
            'mag_global_total_per_site': mag_global_total_per_site,
            'mag_along_field': mag_along_field,
            'mag_along_field_per_site': mag_along_field_per_site,
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


# Plotting functions (similar to original but adapted for Cartesian components)

def create_stacked_plot(h_values, freq_data, spectral_data, component_name, output_file):
    """Create a 2D stacked plot showing spectral function vs frequency for different h values"""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    colors = cm.viridis(np.linspace(0, 1, len(h_values)))
    
    offset = 0
    offset_step = np.abs(np.mean([np.max(spec) - np.min(spec) for spec in spectral_data.values()])) * 1.2
    
    for i, h in enumerate(h_values):
        if h in spectral_data and h in freq_data:
            freq = freq_data[h]
            spec = spectral_data[h]
            mask = (freq >= FREQ_MIN) & (freq <= FREQ_MAX)
            freq_meV = freq[mask] * ENERGY_CONVERSION_FACTOR
            h_converted = h * H_CONVERSION_FACTOR
            ax.plot(freq_meV, spec[mask] + offset, 
                   label=f'h={h_converted:.2f}', color=colors[i], linewidth=1.5, alpha=0.8)
            offset += offset_step
    
    ax.set_xlabel('Energy (meV)', fontsize=12)
    ax.set_ylabel('Spectral Function (offset for clarity)', fontsize=12)
    ax.set_title(f'Spectral Function vs Energy - {component_name}', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=8, ncol=2)
    ax.set_xlim([FREQ_MIN * ENERGY_CONVERSION_FACTOR, FREQ_MAX * ENERGY_CONVERSION_FACTOR])
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved stacked plot: {output_file}")
    plt.close()


def create_heatmap_plot(h_values, freq_data, spectral_data, component_name, output_file):
    """Create a 2D heatmap showing spectral function vs frequency and h"""
    if not h_values or h_values[0] not in freq_data:
        print(f"No data available for heatmap of {component_name}")
        return
    
    freq_ref_full = freq_data[h_values[0]]
    mask = (freq_ref_full >= FREQ_MIN) & (freq_ref_full <= FREQ_MAX)
    freq_ref = freq_ref_full[mask]
    
    spectral_matrix = np.zeros((len(freq_ref), len(h_values)))
    
    for i, h in enumerate(h_values):
        if h in spectral_data:
            freq_h = freq_data[h]
            spec_h = spectral_data[h]
            mask_h = (freq_h >= FREQ_MIN) & (freq_h <= FREQ_MAX)
            freq_h_filtered = freq_h[mask_h]
            spec_h_filtered = spec_h[mask_h]
            spectral_matrix[:, i] = np.interp(freq_ref, freq_h_filtered, spec_h_filtered)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    h_values_converted = np.array(h_values) * H_CONVERSION_FACTOR
    freq_meV = freq_ref * ENERGY_CONVERSION_FACTOR
    
    # Create edges for pcolormesh
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
    
    im = ax.pcolormesh(h_edges, freq_edges, spectral_matrix, 
                       cmap='viridis', shading='flat')
    
    ax.set_xlabel('Magnetic Field (h) [T]', fontsize=12)
    ax.set_ylabel('Energy (meV)', fontsize=12)
    ax.set_title(f'Spectral Function Heatmap - {component_name}', fontsize=14, fontweight='bold')
    
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('Spectral Function', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved heatmap: {output_file}")
    plt.close()


def create_animation(h_values, freq_data, spectral_data, component_name, output_file):
    """Create an animation showing spectral function evolving with h"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    all_spectral = [spectral_data[h] for h in h_values if h in spectral_data]
    if not all_spectral:
        print(f"No data available for animation of {component_name}")
        return
    
    ymin = min([np.min(s) for s in all_spectral])
    ymax = max([np.max(s) for s in all_spectral])
    y_range = ymax - ymin
    ymin -= 0.1 * y_range
    ymax += 0.1 * y_range
    
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
            freq = freq_data[h]
            spec = spectral_data[h]
            mask = (freq >= FREQ_MIN) & (freq <= FREQ_MAX)
            freq_meV = freq[mask] * ENERGY_CONVERSION_FACTOR
            line.set_data(freq_meV, spec[mask])
            h_converted = h * H_CONVERSION_FACTOR
            title.set_text(f'Spectral Function - {component_name} - h={h_converted:.3f} T')
        return line, title
    
    anim = FuncAnimation(fig, animate, init_func=init, 
                        frames=len(h_values), interval=200, blit=True)
    
    writer = PillowWriter(fps=0.5)
    anim.save(output_file, writer=writer)
    print(f"Saved animation: {output_file}")
    plt.close()


def create_overlay_animation(h_values, freq_data_dict, spectral_data_dict, component_names, colors, q_pattern, suffix, output_file):
    """
    Create an overlay animation showing multiple spectral functions on the same plot.
    Components are weighted by experimental channel coefficients:
    - SzSz: cos²(θ)
    - SxSx: sin²(θ)
    - SxSz: 2sin(θ)cos(θ)
    
    Parameters:
    - h_values: list of h field values
    - freq_data_dict: dict mapping component name -> freq_data dict
    - spectral_data_dict: dict mapping component name -> spectral_data dict
    - component_names: list of component names (e.g., ['SxSx', 'SzSz', 'SxSz'])
    - colors: dict mapping component name -> color
    - q_pattern: Q-vector pattern string
    - suffix: suffix string
    - output_file: path to save the animation
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Calculate mixing coefficients
    cos_theta = np.cos(EXPERIMENTAL_ANGLE)
    sin_theta = np.sin(EXPERIMENTAL_ANGLE)
    cos2_theta = cos_theta**2
    sin2_theta = sin_theta**2
    sin_cos_theta = sin_theta * cos_theta
    
    # Define coefficients for each component
    coefficients = {
        'SzSz': cos2_theta,
        'SxSx': sin2_theta,
        'SxSz': 2 * sin_cos_theta
    }
    
    # Find global y-limits across all weighted components
    all_spectral = []
    for comp_name in component_names:
        spectral_data = spectral_data_dict[comp_name]
        coeff = coefficients[comp_name]
        all_spectral.extend([coeff * spectral_data[h] for h in h_values if h in spectral_data])
    
    if not all_spectral:
        print(f"No data available for overlay animation")
        return
    
    ymin = min([np.min(s) for s in all_spectral])
    ymax = max([np.max(s) for s in all_spectral])
    y_range = ymax - ymin
    ymin -= 0.1 * y_range
    ymax += 0.1 * y_range
    
    # Create line objects for each component with coefficient in label
    lines = {}
    for comp_name in component_names:
        coeff = coefficients[comp_name]
        if comp_name == 'SzSz':
            label = f'{comp_name} (cos²θ = {coeff:.3f})'
        elif comp_name == 'SxSx':
            label = f'{comp_name} (sin²θ = {coeff:.3f})'
        elif comp_name == 'SxSz':
            label = f'{comp_name} (2sinθcosθ = {coeff:.3f})'
        else:
            label = f'{comp_name} ({coeff:.3f})'
        
        line, = ax.plot([], [], linewidth=2.5, color=colors[comp_name], label=label, alpha=0.8)
        lines[comp_name] = line
    
    # Add total intensity line
    line_total, = ax.plot([], [], linewidth=3.0, color='black', label='Total', alpha=0.9, linestyle='--')
    lines['Total'] = line_total
    
    ax.set_xlim(FREQ_MIN * ENERGY_CONVERSION_FACTOR, FREQ_MAX * ENERGY_CONVERSION_FACTOR)
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel('Energy (meV)', fontsize=12)
    ax.set_ylabel('Spectral Function', fontsize=12)
    ax.legend(loc='upper right', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    title = ax.set_title('', fontsize=14, fontweight='bold')
    
    def init():
        for line in lines.values():
            line.set_data([], [])
        return list(lines.values()) + [title]
    
    def animate(frame):
        h = h_values[frame]
        h_converted = h * H_CONVERSION_FACTOR
        
        # Track total intensity
        total_spec = None
        freq_meV_ref = None
        
        for comp_name in component_names:
            freq_data = freq_data_dict[comp_name]
            spectral_data = spectral_data_dict[comp_name]
            coeff = coefficients[comp_name]
            
            if h in freq_data and h in spectral_data:
                freq = freq_data[h]
                spec = spectral_data[h]
                mask = (freq >= FREQ_MIN) & (freq <= FREQ_MAX)
                freq_meV = freq[mask] * ENERGY_CONVERSION_FACTOR
                weighted_spec = coeff * spec[mask]
                
                # Apply coefficient weighting
                lines[comp_name].set_data(freq_meV, weighted_spec)
                
                # Accumulate total
                if total_spec is None:
                    total_spec = weighted_spec.copy()
                    freq_meV_ref = freq_meV.copy()
                else:
                    # Interpolate if needed - check lengths first
                    if len(freq_meV) != len(freq_meV_ref) or not np.allclose(freq_meV, freq_meV_ref):
                        weighted_spec_interp = np.interp(freq_meV_ref, freq_meV, weighted_spec)
                        total_spec += weighted_spec_interp
                    else:
                        total_spec += weighted_spec
            else:
                lines[comp_name].set_data([], [])
        
        # Plot total
        if total_spec is not None and freq_meV_ref is not None:
            lines['Total'].set_data(freq_meV_ref, total_spec)
        else:
            lines['Total'].set_data([], [])
        
        theta_deg = np.degrees(EXPERIMENTAL_ANGLE)
        title.set_text(f'Experimental Components {q_pattern}{suffix} - θ={theta_deg:.1f}° - h={h_converted:.3f} T')
        return list(lines.values()) + [title]
    
    anim = FuncAnimation(fig, animate, init_func=init, 
                        frames=len(h_values), interval=200, blit=True)
    
    writer = PillowWriter(fps=0.5)
    anim.save(output_file, writer=writer)
    print(f"Saved overlay animation: {output_file}")
    plt.close()


def create_sidebyside_heatmap(h_values, freq_data_dict, spectral_data_dict, component_names, q_pattern, suffix, output_file):
    """
    Create side-by-side heatmaps for multiple components.
    Components are weighted by experimental channel coefficients:
    - SzSz: cos²(θ)
    - SxSx: sin²(θ)
    - SxSz: 2sin(θ)cos(θ)
    
    Parameters:
    - h_values: list of h field values
    - freq_data_dict: dict mapping component name -> freq_data dict
    - spectral_data_dict: dict mapping component name -> spectral_data dict
    - component_names: list of component names (e.g., ['SxSx', 'SzSz', 'SxSz'])
    - q_pattern: Q-vector pattern string
    - suffix: suffix string
    - output_file: path to save the figure
    """
    n_components = len(component_names)
    fig, axes = plt.subplots(1, n_components, figsize=(8*n_components, 6))
    
    if n_components == 1:
        axes = [axes]
    
    theta_deg = np.degrees(EXPERIMENTAL_ANGLE)
    
    # Calculate mixing coefficients
    cos_theta = np.cos(EXPERIMENTAL_ANGLE)
    sin_theta = np.sin(EXPERIMENTAL_ANGLE)
    cos2_theta = cos_theta**2
    sin2_theta = sin_theta**2
    sin_cos_theta = sin_theta * cos_theta
    
    # Define coefficients for each component
    coefficients = {
        'SzSz': cos2_theta,
        'SxSx': sin2_theta,
        'SxSz': 2 * sin_cos_theta
    }
    
    for idx, comp_name in enumerate(component_names):
        ax = axes[idx]
        freq_data = freq_data_dict[comp_name]
        spectral_data = spectral_data_dict[comp_name]
        coeff = coefficients[comp_name]
        
        if not h_values or h_values[0] not in freq_data:
            print(f"No data available for heatmap of {comp_name}")
            continue
        
        freq_ref_full = freq_data[h_values[0]]
        mask = (freq_ref_full >= FREQ_MIN) & (freq_ref_full <= FREQ_MAX)
        freq_ref = freq_ref_full[mask]
        
        spectral_matrix = np.zeros((len(freq_ref), len(h_values)))
        
        for i, h in enumerate(h_values):
            if h in spectral_data:
                freq_h = freq_data[h]
                spec_h = spectral_data[h]
                mask_h = (freq_h >= FREQ_MIN) & (freq_h <= FREQ_MAX)
                freq_h_filtered = freq_h[mask_h]
                spec_h_filtered = spec_h[mask_h]
                # Apply coefficient weighting
                spectral_matrix[:, i] = coeff * np.interp(freq_ref, freq_h_filtered, spec_h_filtered)
        
        h_values_converted = np.array(h_values) * H_CONVERSION_FACTOR
        freq_meV = freq_ref * ENERGY_CONVERSION_FACTOR
        
        # Create edges for pcolormesh
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
        
        im = ax.pcolormesh(h_edges, freq_edges, spectral_matrix, 
                          cmap='viridis', shading='flat')
        
        ax.set_xlabel('Magnetic Field (h) [T]', fontsize=12)
        ax.set_ylabel('Energy (meV)', fontsize=12)
        
        # Add coefficient to title
        if comp_name == 'SzSz':
            coeff_label = f'cos²θ = {coeff:.3f}'
        elif comp_name == 'SxSx':
            coeff_label = f'sin²θ = {coeff:.3f}'
        elif comp_name == 'SxSz':
            coeff_label = f'2sinθcosθ = {coeff:.3f}'
        else:
            coeff_label = f'{coeff:.3f}'
        
        ax.set_title(f'{comp_name} ({coeff_label})', fontsize=13, fontweight='bold')
        
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label('Spectral Function', fontsize=10)
    
    plt.suptitle(f'Experimental Components {q_pattern}{suffix} - θ={theta_deg:.1f}°', 
                 fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved side-by-side heatmap: {output_file}")
    plt.close()


def create_overlay_with_scaled_sxx(h_values, freq_data_dict, spectral_data_dict, component_names, colors, q_pattern, suffix, scaling_factor, output_file):
    """
    Create an overlay animation showing multiple spectral functions with SxSx scaled by a factor.
    Components are weighted by experimental channel coefficients:
    - SzSz: cos²(θ)
    - SxSx: scaling_factor * sin²(θ)
    - SxSz: 2sin(θ)cos(θ)
    
    Parameters:
    - h_values: list of h field values
    - freq_data_dict: dict mapping component name -> freq_data dict
    - spectral_data_dict: dict mapping component name -> spectral_data dict
    - component_names: list of component names (e.g., ['SxSx', 'SzSz', 'SxSz'])
    - colors: dict mapping component name -> color
    - q_pattern: Q-vector pattern string
    - suffix: suffix string
    - scaling_factor: multiplicative factor for SxSx component
    - output_file: path to save the animation
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Calculate mixing coefficients
    cos_theta = np.cos(EXPERIMENTAL_ANGLE)
    sin_theta = np.sin(EXPERIMENTAL_ANGLE)
    cos2_theta = cos_theta**2
    sin2_theta = sin_theta**2
    sin_cos_theta = sin_theta * cos_theta
    
    # Define coefficients for each component with scaling factor applied to SxSx
    coefficients = {
        'SzSz': cos2_theta,
        'SxSx': scaling_factor * sin2_theta,
        'SxSz': 2 * sin_cos_theta
    }
    
    # Find global y-limits across all weighted components
    all_spectral = []
    for comp_name in component_names:
        spectral_data = spectral_data_dict[comp_name]
        coeff = coefficients[comp_name]
        all_spectral.extend([coeff * spectral_data[h] for h in h_values if h in spectral_data])
    
    if not all_spectral:
        print(f"No data available for scaled overlay animation")
        return
    
    ymin = min([np.min(s) for s in all_spectral])
    ymax = max([np.max(s) for s in all_spectral])
    y_range = ymax - ymin
    ymin -= 0.1 * y_range
    ymax += 0.1 * y_range
    
    # Create line objects for each component with coefficient in label
    lines = {}
    for comp_name in component_names:
        coeff = coefficients[comp_name]
        if comp_name == 'SzSz':
            label = f'{comp_name} (cos²θ = {coeff:.3f})'
        elif comp_name == 'SxSx':
            label = f'{comp_name} ({scaling_factor}×sin²θ = {coeff:.3f})'
        elif comp_name == 'SxSz':
            label = f'{comp_name} (2sinθcosθ = {coeff:.3f})'
        else:
            label = f'{comp_name} ({coeff:.3f})'
        
        line, = ax.plot([], [], linewidth=2.5, color=colors[comp_name], label=label, alpha=0.8)
        lines[comp_name] = line
    
    # Add total intensity line
    line_total, = ax.plot([], [], linewidth=3.0, color='black', label='Total', alpha=0.9, linestyle='--')
    lines['Total'] = line_total
    
    ax.set_xlim(FREQ_MIN * ENERGY_CONVERSION_FACTOR, FREQ_MAX * ENERGY_CONVERSION_FACTOR)
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel('Energy (meV)', fontsize=12)
    ax.set_ylabel('Spectral Function', fontsize=12)
    ax.legend(loc='upper right', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    title = ax.set_title('', fontsize=14, fontweight='bold')
    
    def init():
        for line in lines.values():
            line.set_data([], [])
        return list(lines.values()) + [title]
    
    def animate(frame):
        h = h_values[frame]
        h_converted = h * H_CONVERSION_FACTOR
        
        # Track total intensity
        total_spec = None
        freq_meV_ref = None
        
        for comp_name in component_names:
            freq_data = freq_data_dict[comp_name]
            spectral_data = spectral_data_dict[comp_name]
            coeff = coefficients[comp_name]
            
            if h in freq_data and h in spectral_data:
                freq = freq_data[h]
                spec = spectral_data[h]
                mask = (freq >= FREQ_MIN) & (freq <= FREQ_MAX)
                freq_meV = freq[mask] * ENERGY_CONVERSION_FACTOR
                weighted_spec = coeff * spec[mask]
                
                # Apply coefficient weighting
                lines[comp_name].set_data(freq_meV, weighted_spec)
                
                # Accumulate total
                if total_spec is None:
                    total_spec = weighted_spec.copy()
                    freq_meV_ref = freq_meV.copy()
                else:
                    # Interpolate if needed - check lengths first
                    if len(freq_meV) != len(freq_meV_ref) or not np.allclose(freq_meV, freq_meV_ref):
                        weighted_spec_interp = np.interp(freq_meV_ref, freq_meV, weighted_spec)
                        total_spec += weighted_spec_interp
                    else:
                        total_spec += weighted_spec
            else:
                lines[comp_name].set_data([], [])
        
        # Plot total
        if total_spec is not None and freq_meV_ref is not None:
            lines['Total'].set_data(freq_meV_ref, total_spec)
        else:
            lines['Total'].set_data([], [])
        
        theta_deg = np.degrees(EXPERIMENTAL_ANGLE)
        title.set_text(f'Experimental Components {q_pattern}{suffix} (SxSx×{scaling_factor}) - θ={theta_deg:.1f}° - h={h_converted:.3f} T')
        return list(lines.values()) + [title]
    
    anim = FuncAnimation(fig, animate, init_func=init, 
                        frames=len(h_values), interval=200, blit=True)
    
    writer = PillowWriter(fps=0.5)
    anim.save(output_file, writer=writer)
    print(f"Saved scaled overlay animation: {output_file}")
    plt.close()


def create_sidebyside_heatmap_scaled(h_values, freq_data_dict, spectral_data_dict, component_names, q_pattern, suffix, scaling_factor, output_file):
    """
    Create side-by-side heatmaps for multiple components with SxSx scaled.
    Components are weighted by experimental channel coefficients:
    - SzSz: cos²(θ)
    - SxSx: scaling_factor * sin²(θ)
    - SxSz: 2sin(θ)cos(θ)
    
    Parameters:
    - h_values: list of h field values
    - freq_data_dict: dict mapping component name -> freq_data dict
    - spectral_data_dict: dict mapping component name -> spectral_data dict
    - component_names: list of component names (e.g., ['SxSx', 'SzSz', 'SxSz'])
    - q_pattern: Q-vector pattern string
    - suffix: suffix string
    - scaling_factor: multiplicative factor for SxSx component
    - output_file: path to save the figure
    """
    n_components = len(component_names)
    fig, axes = plt.subplots(1, n_components, figsize=(8*n_components, 6))
    
    if n_components == 1:
        axes = [axes]
    
    theta_deg = np.degrees(EXPERIMENTAL_ANGLE)
    
    # Calculate mixing coefficients
    cos_theta = np.cos(EXPERIMENTAL_ANGLE)
    sin_theta = np.sin(EXPERIMENTAL_ANGLE)
    cos2_theta = cos_theta**2
    sin2_theta = sin_theta**2
    sin_cos_theta = sin_theta * cos_theta
    
    # Define coefficients for each component with scaling factor applied to SxSx
    coefficients = {
        'SzSz': cos2_theta,
        'SxSx': scaling_factor * sin2_theta,
        'SxSz': 2 * sin_cos_theta
    }
    
    for idx, comp_name in enumerate(component_names):
        ax = axes[idx]
        freq_data = freq_data_dict[comp_name]
        spectral_data = spectral_data_dict[comp_name]
        coeff = coefficients[comp_name]
        
        if not h_values or h_values[0] not in freq_data:
            print(f"No data available for heatmap of {comp_name}")
            continue
        
        freq_ref_full = freq_data[h_values[0]]
        mask = (freq_ref_full >= FREQ_MIN) & (freq_ref_full <= FREQ_MAX)
        freq_ref = freq_ref_full[mask]
        
        spectral_matrix = np.zeros((len(freq_ref), len(h_values)))
        
        for i, h in enumerate(h_values):
            if h in spectral_data:
                freq_h = freq_data[h]
                spec_h = spectral_data[h]
                mask_h = (freq_h >= FREQ_MIN) & (freq_h <= FREQ_MAX)
                freq_h_filtered = freq_h[mask_h]
                spec_h_filtered = spec_h[mask_h]
                # Apply coefficient weighting
                spectral_matrix[:, i] = coeff * np.interp(freq_ref, freq_h_filtered, spec_h_filtered)
        
        h_values_converted = np.array(h_values) * H_CONVERSION_FACTOR
        freq_meV = freq_ref * ENERGY_CONVERSION_FACTOR
        
        # Create edges for pcolormesh
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
        
        im = ax.pcolormesh(h_edges, freq_edges, spectral_matrix, 
                          cmap='viridis', shading='flat')
        
        ax.set_xlabel('Magnetic Field (h) [T]', fontsize=12)
        ax.set_ylabel('Energy (meV)', fontsize=12)
        
        # Add coefficient to title
        if comp_name == 'SzSz':
            coeff_label = f'cos²θ = {coeff:.3f}'
        elif comp_name == 'SxSx':
            coeff_label = f'{scaling_factor}×sin²θ = {coeff:.3f}'
        elif comp_name == 'SxSz':
            coeff_label = f'2sinθcosθ = {coeff:.3f}'
        else:
            coeff_label = f'{coeff:.3f}'
        
        ax.set_title(f'{comp_name} ({coeff_label})', fontsize=13, fontweight='bold')
        
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label('Spectral Function', fontsize=10)
    
    plt.suptitle(f'Experimental Components {q_pattern}{suffix} (SxSx×{scaling_factor}) - θ={theta_deg:.1f}°', 
                 fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved scaled side-by-side heatmap: {output_file}")
    plt.close()


def create_combined_diagonal_plot(h_values, all_freq_data, all_spectral_data, q_pattern, suffix, output_file):
    """
    Create a plot showing SxSx, SySy, SzSz and their sum (Trace) for selected h values.
    """
    cart_components = ['SxSx', 'SySy', 'SzSz']
    
    # Check if all components exist
    if not all(comp in all_freq_data for comp in cart_components):
        print(f"  ⚠ Cannot create combined diagonal plot: missing components")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    # Select specific h values
    target_fields_tesla = [0.0, 0.3, 1.0, 2.5]
    h_values_tesla = np.array(h_values) * H_CONVERSION_FACTOR
    
    selected_h_indices = []
    for target_field in target_fields_tesla:
        closest_idx = np.argmin(np.abs(h_values_tesla - target_field))
        selected_h_indices.append(closest_idx)
    
    for plot_idx, h_idx in enumerate(selected_h_indices):
        h = h_values[h_idx]
        ax = axes[plot_idx]
        
        # Get components
        freq_ref = None
        specs = {}
        for comp in cart_components:
            if h in all_freq_data[comp] and h in all_spectral_data[comp]:
                freq = all_freq_data[comp][h]
                spec = all_spectral_data[comp][h]
                mask = (freq >= FREQ_MIN) & (freq <= FREQ_MAX)
                if freq_ref is None:
                    freq_ref = freq[mask] * ENERGY_CONVERSION_FACTOR
                specs[comp] = spec[mask]
            else:
                specs[comp] = None
        
        # Plot individual components
        colors = {'SxSx': 'r', 'SySy': 'g', 'SzSz': 'b'}
        for comp in cart_components:
            if specs[comp] is not None:
                ax.plot(freq_ref, specs[comp], f'{colors[comp]}-', 
                       label=comp, linewidth=2, alpha=0.7)
        
        # Calculate and plot trace
        if all(specs[comp] is not None for comp in cart_components):
            trace = sum(specs[comp] for comp in cart_components)
            ax.plot(freq_ref, trace, 'k-', label='Trace (SxSx+SySy+SzSz)', 
                   linewidth=2.5, alpha=0.9)
        
        h_converted = h * H_CONVERSION_FACTOR
        ax.set_xlabel('Energy (meV)', fontsize=11)
        ax.set_ylabel('Spectral Function', fontsize=11)
        ax.set_title(f'{q_pattern}{suffix} - h={h_converted:.3f} T', fontsize=12, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([FREQ_MIN * ENERGY_CONVERSION_FACTOR, FREQ_MAX * ENERGY_CONVERSION_FACTOR])
    
    plt.suptitle(f'Diagonal Cartesian Components - {q_pattern}{suffix}', 
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved combined diagonal plot: {output_file}")
    plt.close()


def create_magnetization_plot(h_values, h_dirs, output_dir):
    """Create individual magnetization plots (atomized)"""
    print("\nCreating magnetization plots...")
    
    h_list = []
    mag_x_list = []
    mag_y_list = []
    mag_z_list = []
    mag_total_list = []
    mag_x_rotated_local_list = []
    mag_y_rotated_local_list = []
    mag_z_rotated_local_list = []
    mag_total_rotated_local_list = []
    mag_global_x_list = []
    mag_global_y_list = []
    mag_global_z_list = []
    mag_global_total_list = []
    mag_along_field_list = []
    
    for h in h_values:
        h_dir = h_dirs[h]
        mag_data = read_spin_configuration(h_dir)
        
        if mag_data is not None:
            h_list.append(h * H_CONVERSION_FACTOR)
            mag_x_list.append(mag_data['mag_x'])
            mag_y_list.append(mag_data['mag_y'])
            mag_z_list.append(mag_data['mag_z'])
            mag_total_list.append(mag_data['total_magnetization'])
            mag_x_rotated_local_list.append(mag_data['mag_x_rotated_local'])
            mag_y_rotated_local_list.append(mag_data['mag_y_rotated_local'])
            mag_z_rotated_local_list.append(mag_data['mag_z_rotated_local'])
            mag_total_rotated_local_list.append(mag_data['mag_total_rotated_local'])
            mag_global_x_list.append(mag_data['mag_global_x'])
            mag_global_y_list.append(mag_data['mag_global_y'])
            mag_global_z_list.append(mag_data['mag_global_z'])
            mag_global_total_list.append(mag_data['mag_global_total'])
            mag_along_field_list.append(mag_data['mag_along_field'])
    
    if not h_list:
        print("  ⚠ No magnetization data found!")
        return
    
    theta_deg = np.degrees(EXPERIMENTAL_ANGLE)
    field_dir_str = f"[{FIELD_DIRECTION[0]:.3f}, {FIELD_DIRECTION[1]:.3f}, {FIELD_DIRECTION[2]:.3f}]"
    
    # Plot 1: Magnetization along field
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(h_list, mag_along_field_list, 'ko-', linewidth=3, markersize=6, alpha=0.9)
    ax.set_xlabel('Magnetic Field (h) [T]', fontsize=14, fontweight='bold')
    ax.set_ylabel('Magnetization', fontsize=14, fontweight='bold')
    ax.set_title(f'Magnetization Along Field Direction ĥ = {field_dir_str}', 
                  fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "mag_along_field.png"), dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: mag_along_field.png")
    plt.close()
    
    # Plot 2: Local frame components (original)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(h_list, mag_x_list, 'r-o', linewidth=2, markersize=4, label='Mx (local)')
    ax.plot(h_list, mag_y_list, 'g-o', linewidth=2, markersize=4, label='My (local)')
    ax.plot(h_list, mag_z_list, 'b-o', linewidth=2, markersize=4, label='Mz (local)')
    ax.plot(h_list, mag_total_list, 'k-o', linewidth=2.5, markersize=5, label='|M| Total')
    ax.set_xlabel('Magnetic Field (h) [T]', fontsize=12)
    ax.set_ylabel('Magnetization (Local Frame)', fontsize=12)
    ax.set_title('Local Frame Magnetization (Original)', fontsize=13, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "mag_local_original.png"), dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: mag_local_original.png")
    plt.close()
    
    # Plot 3: Local frame components (rotated by theta)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(h_list, mag_x_rotated_local_list, 'r-o', linewidth=2, markersize=4, 
            label=f"Mx' (rotated, θ={theta_deg:.0f}°)")
    ax.plot(h_list, mag_y_rotated_local_list, 'g-o', linewidth=2, markersize=4, 
            label=f"My' (rotated, θ={theta_deg:.0f}°)")
    ax.plot(h_list, mag_z_rotated_local_list, 'b-o', linewidth=2, markersize=4, 
            label=f"Mz' (rotated, θ={theta_deg:.0f}°)")
    ax.plot(h_list, mag_total_rotated_local_list, 'k-o', linewidth=2.5, markersize=5, label="|M'| Total")
    ax.set_xlabel('Magnetic Field (h) [T]', fontsize=12)
    ax.set_ylabel('Magnetization (Rotated Local Frame)', fontsize=12)
    ax.set_title(f'Local Frame Magnetization (Rotated θ={theta_deg:.0f}°)', fontsize=13, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "mag_local_rotated.png"), dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: mag_local_rotated.png")
    plt.close()
    
    # Plot 4: Global frame components
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(h_list, mag_global_x_list, 'r-o', linewidth=2, markersize=4, label='Mx (global)')
    ax.plot(h_list, mag_global_y_list, 'g-o', linewidth=2, markersize=4, label='My (global)')
    ax.plot(h_list, mag_global_z_list, 'b-o', linewidth=2, markersize=4, label='Mz (global)')
    ax.plot(h_list, mag_global_total_list, 'k-o', linewidth=2.5, markersize=5, label='|M| Total')
    ax.set_xlabel('Magnetic Field (h) [T]', fontsize=12)
    ax.set_ylabel('Magnetization (Global Frame)', fontsize=12)
    ax.set_title(f'Global Frame Magnetization (Rotated θ={theta_deg:.0f}°)', fontsize=13, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "mag_global.png"), dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: mag_global.png")
    plt.close()
    
    # Plot 5: Transverse vs Longitudinal (local original)
    fig, ax = plt.subplots(figsize=(10, 6))
    transverse_mag = np.sqrt(np.array(mag_x_list)**2 + np.array(mag_y_list)**2)
    ax.plot(h_list, transverse_mag, 'm-o', linewidth=2.5, markersize=5, label='Transverse (Mx² + My²)^(1/2)')
    ax.plot(h_list, mag_z_list, 'b-o', linewidth=2, markersize=4, label='Longitudinal (Mz)')
    ax.set_xlabel('Magnetic Field (h) [T]', fontsize=12)
    ax.set_ylabel('Magnetization (Local)', fontsize=12)
    ax.set_title('Local: Transverse vs Longitudinal (Original)', fontsize=13, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "mag_local_transverse_vs_longitudinal.png"), dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: mag_local_transverse_vs_longitudinal.png")
    plt.close()
    
    # Plot 6: Transverse vs Longitudinal (local rotated)
    fig, ax = plt.subplots(figsize=(10, 6))
    transverse_mag_rotated = np.sqrt(np.array(mag_x_rotated_local_list)**2 + np.array(mag_y_rotated_local_list)**2)
    ax.plot(h_list, transverse_mag_rotated, 'm-o', linewidth=2.5, markersize=5, 
            label=f"Transverse (Mx'² + My'²)^(1/2)")
    ax.plot(h_list, mag_z_rotated_local_list, 'b-o', linewidth=2, markersize=4, 
            label=f"Longitudinal (Mz')")
    ax.set_xlabel('Magnetic Field (h) [T]', fontsize=12)
    ax.set_ylabel('Magnetization (Rotated Local)', fontsize=12)
    ax.set_title(f'Local: Transverse vs Longitudinal (Rotated θ={theta_deg:.0f}°)', 
                 fontsize=13, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "mag_local_rotated_transverse_vs_longitudinal.png"), 
                dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: mag_local_rotated_transverse_vs_longitudinal.png")
    plt.close()
    
    # Plot 7: Transverse vs Longitudinal (global)
    fig, ax = plt.subplots(figsize=(10, 6))
    transverse_mag_global = np.sqrt(np.array(mag_global_x_list)**2 + np.array(mag_global_y_list)**2)
    ax.plot(h_list, transverse_mag_global, 'm-o', linewidth=2.5, markersize=5, 
            label='Transverse (Mx² + My²)^(1/2)')
    ax.plot(h_list, mag_global_z_list, 'b-o', linewidth=2, markersize=4, label='Longitudinal (Mz)')
    ax.set_xlabel('Magnetic Field (h) [T]', fontsize=12)
    ax.set_ylabel('Magnetization (Global)', fontsize=12)
    ax.set_title(f'Global: Transverse vs Longitudinal (Rotated θ={theta_deg:.0f}°)', 
                 fontsize=13, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "mag_global_transverse_vs_longitudinal.png"), 
                dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: mag_global_transverse_vs_longitudinal.png")
    plt.close()
    
    # Create summary plot (combined overview)
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    # Plot 1: Magnetization along field
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(h_list, mag_along_field_list, 'ko-', linewidth=3, markersize=6, alpha=0.9)
    ax1.set_xlabel('Magnetic Field (h) [T]', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Magnetization', fontsize=14, fontweight='bold')
    ax1.set_title(f'Magnetization Along Field Direction ĥ = {field_dir_str}', 
                  fontsize=16, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Local frame components (original)
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(h_list, mag_x_list, 'r-o', linewidth=2, markersize=4, label='Mx')
    ax2.plot(h_list, mag_y_list, 'g-o', linewidth=2, markersize=4, label='My')
    ax2.plot(h_list, mag_z_list, 'b-o', linewidth=2, markersize=4, label='Mz')
    ax2.plot(h_list, mag_total_list, 'k-o', linewidth=2.5, markersize=5, label='|M|')
    ax2.set_xlabel('Magnetic Field (h) [T]', fontsize=12)
    ax2.set_ylabel('Magnetization', fontsize=12)
    ax2.set_title('Local Frame (Original)', fontsize=13, fontweight='bold')
    ax2.legend(loc='best', fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Local frame components (rotated)
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.plot(h_list, mag_x_rotated_local_list, 'r-o', linewidth=2, markersize=4, label="Mx'")
    ax3.plot(h_list, mag_y_rotated_local_list, 'g-o', linewidth=2, markersize=4, label="My'")
    ax3.plot(h_list, mag_z_rotated_local_list, 'b-o', linewidth=2, markersize=4, label="Mz'")
    ax3.plot(h_list, mag_total_rotated_local_list, 'k-o', linewidth=2.5, markersize=5, label="|M'|")
    ax3.set_xlabel('Magnetic Field (h) [T]', fontsize=12)
    ax3.set_ylabel('Magnetization', fontsize=12)
    ax3.set_title(f'Local Frame (Rotated θ={theta_deg:.0f}°)', fontsize=13, fontweight='bold')
    ax3.legend(loc='best', fontsize=9)
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Global frame components
    ax4 = fig.add_subplot(gs[2, 0])
    ax4.plot(h_list, mag_global_x_list, 'r-o', linewidth=2, markersize=4, label='Mx')
    ax4.plot(h_list, mag_global_y_list, 'g-o', linewidth=2, markersize=4, label='My')
    ax4.plot(h_list, mag_global_z_list, 'b-o', linewidth=2, markersize=4, label='Mz')
    ax4.plot(h_list, mag_global_total_list, 'k-o', linewidth=2.5, markersize=5, label='|M|')
    ax4.set_xlabel('Magnetic Field (h) [T]', fontsize=12)
    ax4.set_ylabel('Magnetization', fontsize=12)
    ax4.set_title(f'Global Frame (Rotated θ={theta_deg:.0f}°)', fontsize=13, fontweight='bold')
    ax4.legend(loc='best', fontsize=9)
    ax4.grid(True, alpha=0.3)
    
    # Plot 5: Comparison of longitudinal components
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.plot(h_list, mag_z_list, 'b-o', linewidth=2, markersize=4, label='Mz (local original)')
    ax5.plot(h_list, mag_z_rotated_local_list, 'c-s', linewidth=2, markersize=4, 
             label=f"Mz' (local rotated θ={theta_deg:.0f}°)")
    ax5.plot(h_list, mag_global_z_list, 'm-^', linewidth=2, markersize=4, label='Mz (global)')
    ax5.set_xlabel('Magnetic Field (h) [T]', fontsize=12)
    ax5.set_ylabel('Longitudinal Magnetization', fontsize=12)
    ax5.set_title('Comparison of Longitudinal Components', fontsize=13, fontweight='bold')
    ax5.legend(loc='best', fontsize=9)
    ax5.grid(True, alpha=0.3)
    
    plt.savefig(os.path.join(output_dir, "mag_summary.png"), dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: mag_summary.png")
    plt.close()
    
    print(f"\n  ✓ All magnetization plots created in {output_dir}")


def main():
    """Main function to process all data and create Cartesian basis plots"""
    global EXPERIMENTAL_ANGLE  # We need to modify the global variable
    
    print("="*80)
    print("CARTESIAN BASIS DYNAMIC STRUCTURE FACTOR ANALYSIS")
    print("="*80)
    
    print("\nFinding h directories...")
    h_data = find_all_h_directories()
    
    if not h_data:
        print("No h directories found!")
        return
    
    h_values = [h for h, _ in h_data]
    h_dirs = {h: d for h, d in h_data}
    
    print(f"Found {len(h_values)} h values: {h_values}")
    
    # STEP 1: Calculate experimental angle from magnetization data
    # EXPERIMENTAL_ANGLE = calculate_experimental_angle(h_values, h_dirs)
    # EXPERIMENTAL_ANGLE = 0.31416104734 # For Cartesian basis, we set angle to 0
    EXPERIMENTAL_ANGLE = 40/180*np.pi  # 40 degrees in radians
    # Find all unique species (in spherical basis)
    print("\nFinding all spherical basis species...")
    all_species = set()
    for h, h_dir in h_data:
        species = find_all_species(h_dir)
        all_species.update(species)
    
    all_species = sorted(list(all_species))
    print(f"Found {len(all_species)} spherical basis species")
    
    # Identify Q-patterns and suffixes
    q_patterns = set()
    suffixes = set()
    
    for species in all_species:
        q_pat = extract_q_pattern(species)
        if q_pat:
            q_patterns.add(q_pat)
        suffix = extract_suffix(species)
        suffixes.add(suffix)
    
    q_patterns = sorted(list(q_patterns))
    suffixes = sorted(list(suffixes))
    
    print(f"\nFound {len(q_patterns)} Q-patterns:")
    for q_pat in q_patterns:
        print(f"  - {q_pat}")
    
    print(f"\nFound {len(suffixes)} suffixes: {suffixes}")
    
    # Define Cartesian components to compute
    diagonal_components = ['SxSx', 'SySy', 'SzSz']
    off_diagonal_components = ['SxSy', 'SySz', 'SxSz']
    all_cart_components = diagonal_components + off_diagonal_components
    
    print(f"\nWill construct {len(all_cart_components)} Cartesian components:")
    print(f"  Diagonal: {diagonal_components}")
    print(f"  Off-diagonal: {off_diagonal_components}")
    
    # Process each Q-pattern and suffix combination
    print("\n" + "="*80)
    print("CONSTRUCTING CARTESIAN COMPONENTS")
    print("="*80)
    
    all_cartesian_data = {}  # Maps (q_pattern, suffix, cart_component) -> (freq_data, spectral_data)
    
    for q_pattern in q_patterns:
        for suffix in suffixes:
            print(f"\nProcessing Q={q_pattern}, suffix={suffix}")
            print("-"*80)
            
            for cart_comp in all_cart_components:
                try:
                    print(f"  Constructing {cart_comp}...")
                    freq_data, spectral_data = construct_cartesian_component(
                        h_values, h_dirs, q_pattern, suffix, cart_comp
                    )
                    
                    if spectral_data:
                        all_cartesian_data[(q_pattern, suffix, cart_comp)] = (freq_data, spectral_data)
                        print(f"    ✓ Success: {len(spectral_data)} h-values")
                    else:
                        print(f"    ✗ No data")
                except Exception as e:
                    print(f"    ✗ Error: {e}")
    
    # Create individual component plots
    print("\n" + "="*80)
    print("CREATING INDIVIDUAL CARTESIAN COMPONENT PLOTS")
    print("="*80)
    
    for (q_pattern, suffix, cart_comp), (freq_data, spectral_data) in all_cartesian_data.items():
        try:
            print(f"\nPlotting {cart_comp} for Q={q_pattern}, suffix={suffix}")
            
            safe_name = f"{cart_comp}_{q_pattern}{suffix}".replace("/", "_").replace(" ", "_")
            component_name = f"{cart_comp} {q_pattern}{suffix}"
            
            # Determine subdirectory
            if cart_comp in diagonal_components:
                subdir = SUBDIRS['individual']
            else:
                subdir = SUBDIRS['off_diagonal']
            
            # 1. Stacked plot
            stacked_file = os.path.join(subdir, f"{safe_name}_stacked.png")
            create_stacked_plot(h_values, freq_data, spectral_data, component_name, stacked_file)
            
            # 2. Heatmap
            heatmap_file = os.path.join(subdir, f"{safe_name}_heatmap.png")
            create_heatmap_plot(h_values, freq_data, spectral_data, component_name, heatmap_file)
            
            # 3. Animation
            anim_file = os.path.join(subdir, f"{safe_name}_animation.gif")
            create_animation(h_values, freq_data, spectral_data, component_name, anim_file)
            
            print(f"  ✓ Completed {cart_comp}")
        except Exception as e:
            print(f"  ✗ Error: {e}")
    
    # Create combined diagonal plots (SxSx + SySy + SzSz + Trace)
    print("\n" + "="*80)
    print("CREATING COMBINED DIAGONAL PLOTS")
    print("="*80)
    
    for q_pattern in q_patterns:
        for suffix in suffixes:
            try:
                print(f"\nCombined diagonal for Q={q_pattern}, suffix={suffix}")
                
                # Prepare data dict for this Q-pattern and suffix
                qsuffix_data_freq = {}
                qsuffix_data_spec = {}
                
                for cart_comp in diagonal_components:
                    key = (q_pattern, suffix, cart_comp)
                    if key in all_cartesian_data:
                        freq_data, spectral_data = all_cartesian_data[key]
                        qsuffix_data_freq[cart_comp] = freq_data
                        qsuffix_data_spec[cart_comp] = spectral_data
                
                if len(qsuffix_data_freq) == len(diagonal_components):
                    safe_name = f"{q_pattern}{suffix}".replace("/", "_").replace(" ", "_")
                    combined_file = os.path.join(SUBDIRS['combined'], f"{safe_name}_diagonal_combined.png")
                    create_combined_diagonal_plot(h_values, qsuffix_data_freq, qsuffix_data_spec, 
                                                  q_pattern, suffix, combined_file)
                    print(f"  ✓ Created combined diagonal plot")
                else:
                    print(f"  ⚠ Missing some diagonal components, skipping")
            except Exception as e:
                print(f"  ✗ Error: {e}")
    
    # Create experimental channels
    print("\n" + "="*80)
    print("CREATING EXPERIMENTAL CHANNEL")
    theta_deg = np.degrees(EXPERIMENTAL_ANGLE)
    print(f"Angle: θ={theta_deg:.0f}°")
    print("="*80)
    
    # Process standard suffixes (_SF, _NSF)
    standard_suffixes = [s for s in suffixes if s in ['_SF', '_NSF']]
    
    for q_pattern in q_patterns:
        for suffix in standard_suffixes:
            print(f"\nProcessing experimental channel for Q={q_pattern}, suffix={suffix}")
            print("-"*80)
            
            try:
                # Get the individual components needed for experimental channel
                print(f"  Retrieving individual components (SxSx, SzSz, SxSz)...")
                
                # Check if components are already constructed
                key_sxx = (q_pattern, suffix, 'SxSx')
                key_szz = (q_pattern, suffix, 'SzSz')
                key_sxz = (q_pattern, suffix, 'SxSz')
                
                if key_sxx not in all_cartesian_data or key_szz not in all_cartesian_data or key_sxz not in all_cartesian_data:
                    print(f"    ✗ Missing required components")
                    continue
                
                freq_sxx, spec_sxx = all_cartesian_data[key_sxx]
                freq_szz, spec_szz = all_cartesian_data[key_szz]
                freq_sxz, spec_sxz = all_cartesian_data[key_sxz]
                
                print(f"    ✓ Components retrieved")
                
                # Create individual component plots in experimental subdirectory
                print(f"  Creating individual component plots...")
                
                # Plot SxSx
                safe_name = f"SxSx_{q_pattern}{suffix}".replace("/", "_").replace(" ", "_")
                component_name = f"SxSx {q_pattern}{suffix} (Experimental)"
                
                stacked_file = os.path.join(SUBDIRS['experimental'], f"{safe_name}_stacked.png")
                create_stacked_plot(h_values, freq_sxx, spec_sxx, component_name, stacked_file)
                
                heatmap_file = os.path.join(SUBDIRS['experimental'], f"{safe_name}_heatmap.png")
                create_heatmap_plot(h_values, freq_sxx, spec_sxx, component_name, heatmap_file)
                
                anim_file = os.path.join(SUBDIRS['experimental'], f"{safe_name}_animation.gif")
                create_animation(h_values, freq_sxx, spec_sxx, component_name, anim_file)
                
                # Plot SzSz
                safe_name = f"SzSz_{q_pattern}{suffix}".replace("/", "_").replace(" ", "_")
                component_name = f"SzSz {q_pattern}{suffix} (Experimental)"
                
                stacked_file = os.path.join(SUBDIRS['experimental'], f"{safe_name}_stacked.png")
                create_stacked_plot(h_values, freq_szz, spec_szz, component_name, stacked_file)
                
                heatmap_file = os.path.join(SUBDIRS['experimental'], f"{safe_name}_heatmap.png")
                create_heatmap_plot(h_values, freq_szz, spec_szz, component_name, heatmap_file)
                
                anim_file = os.path.join(SUBDIRS['experimental'], f"{safe_name}_animation.gif")
                create_animation(h_values, freq_szz, spec_szz, component_name, anim_file)
                
                # Plot SxSz
                safe_name = f"SxSz_{q_pattern}{suffix}".replace("/", "_").replace(" ", "_")
                component_name = f"SxSz {q_pattern}{suffix} (Experimental)"
                
                stacked_file = os.path.join(SUBDIRS['experimental'], f"{safe_name}_stacked.png")
                create_stacked_plot(h_values, freq_sxz, spec_sxz, component_name, stacked_file)
                
                heatmap_file = os.path.join(SUBDIRS['experimental'], f"{safe_name}_heatmap.png")
                create_heatmap_plot(h_values, freq_sxz, spec_sxz, component_name, heatmap_file)
                
                anim_file = os.path.join(SUBDIRS['experimental'], f"{safe_name}_animation.gif")
                create_animation(h_values, freq_sxz, spec_sxz, component_name, anim_file)
                
                print(f"    ✓ Individual component plots created")
                
                # Create overlay animation and side-by-side heatmap
                print(f"  Creating comparison plots...")
                component_names = ['SxSx', 'SzSz', 'SxSz']
                colors = {'SxSx': 'red', 'SzSz': 'blue', 'SxSz': 'green'}
                freq_dict = {'SxSx': freq_sxx, 'SzSz': freq_szz, 'SxSz': freq_sxz}
                spec_dict = {'SxSx': spec_sxx, 'SzSz': spec_szz, 'SxSz': spec_sxz}
                
                # Overlay animation
                overlay_anim_file = os.path.join(SUBDIRS['experimental'], 
                                                f"overlay_{q_pattern}{suffix}_animation.gif".replace("/", "_").replace(" ", "_"))
                create_overlay_animation(h_values, freq_dict, spec_dict, component_names, 
                                       colors, q_pattern, suffix, overlay_anim_file)
                
                # Side-by-side heatmap
                sidebyside_file = os.path.join(SUBDIRS['experimental'], 
                                              f"sidebyside_{q_pattern}{suffix}_heatmap.png".replace("/", "_").replace(" ", "_"))
                create_sidebyside_heatmap(h_values, freq_dict, spec_dict, component_names, 
                                        q_pattern, suffix, sidebyside_file)
                
                # Scaled overlay animation with SxSx multiplied by scaling factor
                scaling_factor = 2.0  # Adjust this value as needed
                scaled_overlay_anim_file = os.path.join(SUBDIRS['experimental'], 
                                                f"overlay_scaled_{q_pattern}{suffix}_animation.gif".replace("/", "_").replace(" ", "_"))
                create_overlay_with_scaled_sxx(h_values, freq_dict, spec_dict, component_names, 
                                              colors, q_pattern, suffix, scaling_factor, scaled_overlay_anim_file)
                
                # Scaled side-by-side heatmap with SxSx multiplied by scaling factor
                scaled_sidebyside_file = os.path.join(SUBDIRS['experimental'], 
                                              f"sidebyside_scaled_{q_pattern}{suffix}_heatmap.png".replace("/", "_").replace(" ", "_"))
                create_sidebyside_heatmap_scaled(h_values, freq_dict, spec_dict, component_names, 
                                        q_pattern, suffix, scaling_factor, scaled_sidebyside_file)
                
                print(f"    ✓ Comparison plots created")
                
                # Now construct the combined experimental channel
                print(f"  Constructing combined experimental channel θ={theta_deg:.0f}°...")
                
                freq_data, spectral_data = construct_experimental_channel(
                    h_values, h_dirs, q_pattern, suffix, EXPERIMENTAL_ANGLE
                )
                
                if spectral_data:
                    print(f"    ✓ Success: {len(spectral_data)} h-values")
                    
                    # Create plots for combined channel
                    safe_name = f"experimental_combined_{q_pattern}{suffix}".replace("/", "_").replace(" ", "_")
                    component_name = f"Experimental Combined θ={theta_deg:.0f}° {q_pattern}{suffix}"
                    
                    # Stacked plot
                    stacked_file = os.path.join(SUBDIRS['experimental'], f"{safe_name}_stacked.png")
                    create_stacked_plot(h_values, freq_data, spectral_data, component_name, stacked_file)
                    
                    # Heatmap
                    heatmap_file = os.path.join(SUBDIRS['experimental'], f"{safe_name}_heatmap.png")
                    create_heatmap_plot(h_values, freq_data, spectral_data, component_name, heatmap_file)
                    
                    # Animation
                    anim_file = os.path.join(SUBDIRS['experimental'], f"{safe_name}_animation.gif")
                    create_animation(h_values, freq_data, spectral_data, component_name, anim_file)
                    
                    print(f"    ✓ Combined experimental channel plots created")
                else:
                    print(f"    ✗ No data for combined channel")
            except Exception as e:
                print(f"    ✗ Error: {e}")
                import traceback
                traceback.print_exc()
    
    # Process DO channel (SF + NSF) if both are available
    print("\n" + "="*80)
    print("CREATING DO CHANNEL (SF + NSF)")
    print("="*80)
    
    if '_SF' in suffixes and '_NSF' in suffixes:
        for q_pattern in q_patterns:
            print(f"\nProcessing DO channel for Q={q_pattern}")
            print("-"*80)
            
            try:
                # Check if both SF and NSF components exist
                sf_components_exist = all([
                    (q_pattern, '_SF', comp) in all_cartesian_data 
                    for comp in ['SxSx', 'SzSz', 'SxSz']
                ])
                nsf_components_exist = all([
                    (q_pattern, '_NSF', comp) in all_cartesian_data 
                    for comp in ['SxSx', 'SzSz', 'SxSz']
                ])
                
                if not sf_components_exist or not nsf_components_exist:
                    print(f"    ✗ Missing SF or NSF components")
                    continue
                
                # Retrieve SF components
                freq_sxx_sf, spec_sxx_sf = all_cartesian_data[(q_pattern, '_SF', 'SxSx')]
                freq_szz_sf, spec_szz_sf = all_cartesian_data[(q_pattern, '_SF', 'SzSz')]
                freq_sxz_sf, spec_sxz_sf = all_cartesian_data[(q_pattern, '_SF', 'SxSz')]
                
                # Retrieve NSF components
                freq_sxx_nsf, spec_sxx_nsf = all_cartesian_data[(q_pattern, '_NSF', 'SxSx')]
                freq_szz_nsf, spec_szz_nsf = all_cartesian_data[(q_pattern, '_NSF', 'SzSz')]
                freq_sxz_nsf, spec_sxz_nsf = all_cartesian_data[(q_pattern, '_NSF', 'SxSz')]
                
                # Combine to make DO
                freq_sxx_do, spec_sxx_do = combine_sf_nsf_to_do(freq_sxx_sf, spec_sxx_sf, freq_sxx_nsf, spec_sxx_nsf, h_values)
                freq_szz_do, spec_szz_do = combine_sf_nsf_to_do(freq_szz_sf, spec_szz_sf, freq_szz_nsf, spec_szz_nsf, h_values)
                freq_sxz_do, spec_sxz_do = combine_sf_nsf_to_do(freq_sxz_sf, spec_sxz_sf, freq_sxz_nsf, spec_sxz_nsf, h_values)
                
                print(f"    ✓ DO components created")
                
                # Create individual component plots
                print(f"  Creating individual DO component plots...")
                
                # Plot SxSx_DO
                safe_name = f"SxSx_{q_pattern}_DO".replace("/", "_").replace(" ", "_")
                component_name = f"SxSx {q_pattern}_DO (Experimental)"
                
                stacked_file = os.path.join(SUBDIRS['experimental'], f"{safe_name}_stacked.png")
                create_stacked_plot(h_values, freq_sxx_do, spec_sxx_do, component_name, stacked_file)
                
                heatmap_file = os.path.join(SUBDIRS['experimental'], f"{safe_name}_heatmap.png")
                create_heatmap_plot(h_values, freq_sxx_do, spec_sxx_do, component_name, heatmap_file)
                
                anim_file = os.path.join(SUBDIRS['experimental'], f"{safe_name}_animation.gif")
                create_animation(h_values, freq_sxx_do, spec_sxx_do, component_name, anim_file)
                
                # Plot SzSz_DO
                safe_name = f"SzSz_{q_pattern}_DO".replace("/", "_").replace(" ", "_")
                component_name = f"SzSz {q_pattern}_DO (Experimental)"
                
                stacked_file = os.path.join(SUBDIRS['experimental'], f"{safe_name}_stacked.png")
                create_stacked_plot(h_values, freq_szz_do, spec_szz_do, component_name, stacked_file)
                
                heatmap_file = os.path.join(SUBDIRS['experimental'], f"{safe_name}_heatmap.png")
                create_heatmap_plot(h_values, freq_szz_do, spec_szz_do, component_name, heatmap_file)
                
                anim_file = os.path.join(SUBDIRS['experimental'], f"{safe_name}_animation.gif")
                create_animation(h_values, freq_szz_do, spec_szz_do, component_name, anim_file)
                
                # Plot SxSz_DO
                safe_name = f"SxSz_{q_pattern}_DO".replace("/", "_").replace(" ", "_")
                component_name = f"SxSz {q_pattern}_DO (Experimental)"
                
                stacked_file = os.path.join(SUBDIRS['experimental'], f"{safe_name}_stacked.png")
                create_stacked_plot(h_values, freq_sxz_do, spec_sxz_do, component_name, stacked_file)
                
                heatmap_file = os.path.join(SUBDIRS['experimental'], f"{safe_name}_heatmap.png")
                create_heatmap_plot(h_values, freq_sxz_do, spec_sxz_do, component_name, heatmap_file)
                
                anim_file = os.path.join(SUBDIRS['experimental'], f"{safe_name}_animation.gif")
                create_animation(h_values, freq_sxz_do, spec_sxz_do, component_name, anim_file)
                
                print(f"    ✓ Individual DO component plots created")
                
                # Create overlay animation and side-by-side heatmap for DO
                print(f"  Creating DO comparison plots...")
                component_names = ['SxSx', 'SzSz', 'SxSz']
                colors = {'SxSx': 'red', 'SzSz': 'blue', 'SxSz': 'green'}
                freq_dict = {'SxSx': freq_sxx_do, 'SzSz': freq_szz_do, 'SxSz': freq_sxz_do}
                spec_dict = {'SxSx': spec_sxx_do, 'SzSz': spec_szz_do, 'SxSz': spec_sxz_do}
                
                # Overlay animation
                overlay_anim_file = os.path.join(SUBDIRS['experimental'], 
                                                f"overlay_{q_pattern}_DO_animation.gif".replace("/", "_").replace(" ", "_"))
                create_overlay_animation(h_values, freq_dict, spec_dict, component_names, 
                                       colors, q_pattern, '_DO', overlay_anim_file)
                
                # Side-by-side heatmap
                sidebyside_file = os.path.join(SUBDIRS['experimental'], 
                                              f"sidebyside_{q_pattern}_DO_heatmap.png".replace("/", "_").replace(" ", "_"))
                create_sidebyside_heatmap(h_values, freq_dict, spec_dict, component_names, 
                                        q_pattern, '_DO', sidebyside_file)
                
                # Scaled overlay animation with SxSx multiplied by scaling factor
                scaling_factor = 2.0  # Adjust this value as needed
                scaled_overlay_anim_file = os.path.join(SUBDIRS['experimental'], 
                                                f"overlay_scaled_{q_pattern}_DO_animation.gif".replace("/", "_").replace(" ", "_"))
                create_overlay_with_scaled_sxx(h_values, freq_dict, spec_dict, component_names, 
                                              colors, q_pattern, '_DO', scaling_factor, scaled_overlay_anim_file)
                
                # Scaled side-by-side heatmap with SxSx multiplied by scaling factor
                scaled_sidebyside_file = os.path.join(SUBDIRS['experimental'], 
                                              f"sidebyside_scaled_{q_pattern}_DO_heatmap.png".replace("/", "_").replace(" ", "_"))
                create_sidebyside_heatmap_scaled(h_values, freq_dict, spec_dict, component_names, 
                                        q_pattern, '_DO', scaling_factor, scaled_sidebyside_file)
                
                print(f"    ✓ DO comparison plots created")
                
                # Construct combined experimental DO channel from DO components
                print(f"  Constructing combined DO experimental channel θ={theta_deg:.0f}°...")
                
                # Calculate mixing coefficients
                cos_theta = np.cos(EXPERIMENTAL_ANGLE)
                sin_theta = np.sin(EXPERIMENTAL_ANGLE)
                cos2_theta = cos_theta**2
                sin2_theta = sin_theta**2
                sin_cos_theta = sin_theta * cos_theta
                
                freq_data_do = {}
                spectral_data_do = {}
                
                # Construct experimental channel for each h value
                for h in h_values:
                    if h not in spec_szz_do or h not in spec_sxx_do or h not in spec_sxz_do:
                        continue
                    
                    # Use SzSz as reference frequency grid
                    freq_ref = freq_szz_do[h]
                    
                    # Get spectral data
                    szz = spec_szz_do[h]
                    sxx = spec_sxx_do[h]
                    sxz = spec_sxz_do[h]
                    
                    # Interpolate if needed
                    freq_sxx_h = freq_sxx_do[h]
                    if len(freq_sxx_h) != len(freq_ref) or not np.allclose(freq_sxx_h, freq_ref):
                        sxx = np.interp(freq_ref, freq_sxx_h, sxx)
                    
                    freq_sxz_h = freq_sxz_do[h]
                    if len(freq_sxz_h) != len(freq_ref) or not np.allclose(freq_sxz_h, freq_ref):
                        sxz = np.interp(freq_ref, freq_sxz_h, sxz)
                    
                    # Construct experimental channel: cos²(θ) SzSz + sin²(θ) SxSx + sin(θ)cos(θ)(SxSz + SzSx)
                    spec_exp = cos2_theta * szz + sin2_theta * sxx + 2 * sin_cos_theta * sxz
                    
                    freq_data_do[h] = freq_ref
                    spectral_data_do[h] = spec_exp
                
                if spectral_data_do:
                    print(f"    ✓ Success: {len(spectral_data_do)} h-values")
                    
                    # Create plots for combined DO channel
                    safe_name = f"experimental_combined_{q_pattern}_DO".replace("/", "_").replace(" ", "_")
                    component_name = f"Experimental Combined θ={theta_deg:.0f}° {q_pattern}_DO"
                    
                    # Stacked plot
                    stacked_file = os.path.join(SUBDIRS['experimental'], f"{safe_name}_stacked.png")
                    create_stacked_plot(h_values, freq_data_do, spectral_data_do, component_name, stacked_file)
                    
                    # Heatmap
                    heatmap_file = os.path.join(SUBDIRS['experimental'], f"{safe_name}_heatmap.png")
                    create_heatmap_plot(h_values, freq_data_do, spectral_data_do, component_name, heatmap_file)
                    
                    # Animation
                    anim_file = os.path.join(SUBDIRS['experimental'], f"{safe_name}_animation.gif")
                    create_animation(h_values, freq_data_do, spectral_data_do, component_name, anim_file)
                    
                    print(f"    ✓ Combined DO experimental channel plots created")
                else:
                    print(f"    ✗ No data for combined DO channel")
                    
            except Exception as e:
                print(f"    ✗ Error: {e}")
                import traceback
                traceback.print_exc()
    else:
        print("  ⚠ Both _SF and _NSF required for DO channel - skipping")
    
    # Create magnetization plot
    print("\n" + "="*80)
    print("CREATING MAGNETIZATION PLOT")
    print("="*80)
    try:
        create_magnetization_plot(h_values, h_dirs, SUBDIRS['magnetization'])
        print("  ✓ Magnetization plots created")
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
    
    # Create summary
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print(f"\nAll plots saved to: {OUTPUT_DIR}")
    print("\nOutput Structure:")
    print(f"  0_summary/                      - Summary and overview")
    print(f"  1_individual_cartesian_components/ - Individual diagonal components (SxSx, SySy, SzSz)")
    print(f"  2_combined_diagonal/            - Combined SxSx+SySy+SzSz+Trace plots")
    print(f"  3_off_diagonal_components/      - Off-diagonal components (SxSy, SySz, SxSz)")
    print(f"  4_experimental_channels/        - Experimental channel (cos²θ SzSz + sin²θ SxSx + sinθcosθ(SxSz+SzSx))")
    print(f"  5_sublattice_cartesian/         - Sublattice correlations (Cartesian basis)")
    print(f"  6_global_transverse_cartesian/  - Global transverse analysis (Cartesian basis)")
    print(f"  7_magnetization/                - Magnetization vs field")
    print(f"\nExperimental channel angle: θ = {np.degrees(EXPERIMENTAL_ANGLE):.0f}°")
    print("="*80)


if __name__ == "__main__":
    main()
