#!/usr/bin/env python3
"""
Script to read processed spectral data from calc_QFI.py output and create
animated/stacked plots for analysis.

This script reads the processed_data directory structure created by calc_QFI.py:
- structure_factor_results/
  - processed_data/
    - {species}/  (e.g., 'SpSm_q_Qx0_Qy0_Qz0', 'SzSz_q_Qx0_Qy0_Qz0_sub0_sub1', etc.)
      - spectral_beta_{beta}.dat
      - peaks_beta_{beta}.dat

The species names come from TPQ_DSSF.cpp and include the operator_type prefix:
- sum: SpSm_q_Qx0_Qy0_Qz0, SzSz_q_Qx0_Qy0_Qz0, etc.
- sublattice: SpSm_q_Qx0_Qy0_Qz0_sub0_sub1, SzSz_q_Qx0_Qy0_Qz0_sub2_sub3, etc.
- transverse: SpSm_q_Qx0_Qy0_Qz0_SF, SzSz_q_Qx0_Qy0_Qz0_NSF, etc.
- experimental: Experimental_q_Qx0_Qy0_Qz0_theta0, etc.
- transverse_experimental: TransverseExperimental_q_Qx0_Qy0_Qz0_theta0_SF, etc.

Special Features:
1. Automatically detects and combines SF/NSF pairs into DO (double-differential) channels
   - DO = SF + NSF
   - Works for both Transverse and TransverseExperimental operator types
   - DO channels are plotted with dashed lines in comparison plots

2. Overlay visualization for TransverseExperimental
   - Automatically finds corresponding SxSx and SzSz Transverse species
   - Computes and overlays: cos²(θ)·SxSx + sin²(θ)·SzSz
   - This theoretical combination should match TransverseExperimental data
   - Overlay plots use dotted lines in comparison plots
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
BASE_DIR = "/scratch/zhouzb79/DSSF_PCD_mag_field_sweep_CZO_pi_4"  # Change to your directory
OUTPUT_DIR = os.path.join(BASE_DIR, "spectral_animations")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Create organized subdirectories
SUBDIRS = {
    'individual': os.path.join(OUTPUT_DIR, "1_individual_species"),
    'combined': os.path.join(OUTPUT_DIR, "2_combined_plots"),
    'beta_evolution': os.path.join(OUTPUT_DIR, "3_beta_evolution"),
    'heatmaps': os.path.join(OUTPUT_DIR, "4_heatmaps"),
    'h_evolution': os.path.join(OUTPUT_DIR, "5_h_field_evolution"),
    'summary': os.path.join(OUTPUT_DIR, "0_summary")
}

# Create all subdirectories
for subdir in SUBDIRS.values():
    os.makedirs(subdir, exist_ok=True)

# Conversion factors (adjust for your system)
# For magnetic field: adjust based on your system
H_CONVERSION_FACTOR = 0.063 / (2.5 * 0.0578)  # Default: no conversion
# For omega values (energy/frequency): Jzz = 0.063 meV for example material
ENERGY_CONVERSION_FACTOR = 0.063  # converts from Jzz units to meV

# Frequency range limits (in Jzz units)
FREQ_MIN = -3.0
FREQ_MAX = 6.0


def extract_h_value(h_dir):
    """Extract numerical h value from directory name like 'h=0.1'"""
    match = re.search(r'h=([0-9.eE+-]+)', os.path.basename(h_dir))
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            pass
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


def find_all_species(base_dir):
    """Find all species directories in processed_data"""
    processed_data_dir = os.path.join(base_dir, "structure_factor_results", "processed_data")
    if not os.path.exists(processed_data_dir):
        print(f"Warning: {processed_data_dir} does not exist")
        return []
    
    species_dirs = glob.glob(os.path.join(processed_data_dir, "*"))
    species_dirs = [d for d in species_dirs if os.path.isdir(d)]
    species_names = [os.path.basename(d) for d in species_dirs]
    
    return sorted(species_names)


def find_all_beta_values(base_dir, species):
    """Find all beta values for which spectral data exists for a given species"""
    species_dir = os.path.join(base_dir, "structure_factor_results", "processed_data", species)
    
    if not os.path.exists(species_dir):
        return []
    
    # Find all spectral_beta_*.dat files
    spectral_files = glob.glob(os.path.join(species_dir, "spectral_beta_*.dat"))
    
    beta_values = []
    for f in spectral_files:
        basename = os.path.basename(f)
        match = re.search(r'spectral_beta_([0-9.eE+-]+|inf)\.dat', basename)
        if match:
            beta_str = match.group(1)
            if beta_str.lower() == 'inf':
                beta_values.append((np.inf, 'inf', f))
            else:
                try:
                    beta = float(beta_str)
                    beta_values.append((beta, beta_str, f))
                except ValueError:
                    pass
    
    # Sort by beta value
    beta_values.sort(key=lambda x: x[0])
    
    return beta_values


def read_spectral_data(file_path):
    """Read spectral data file from calc_QFI.py output"""
    if not os.path.exists(file_path):
        return None, None, None
    
    try:
        data = np.loadtxt(file_path)
        freq = data[:, 0]
        spectral_real = data[:, 1]
        spectral_imag = data[:, 2] if data.shape[1] > 2 else np.zeros_like(freq)
        return freq, spectral_real, spectral_imag
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None, None, None


def read_peaks_data(species_dir, beta_str):
    """Read peak data file for a given species and beta"""
    peak_file = os.path.join(species_dir, f"peaks_beta_{beta_str}.dat")
    
    if not os.path.exists(peak_file):
        return None
    
    try:
        data = np.loadtxt(peak_file)
        if data.ndim == 1:
            data = data.reshape(1, -1)
        return data  # columns: freq, height, prominence
    except Exception as e:
        print(f"Error reading {peak_file}: {e}")
        return None


def parse_species_name(species_name):
    """
    Parse species name to extract components.
    
    Examples:
    - 'SpSm_q_Qx0_Qy0_Qz0' -> operator='SpSm', q_pattern='q_Qx0_Qy0_Qz0'
    - 'SzSz_q_Qx0_Qy0_Qz0_sub0_sub1' -> operator='SzSz', q_pattern='q_Qx0_Qy0_Qz0', sublattices=(0,1)
    - 'SpSm_q_Qx0_Qy0_Qz0_SF' -> operator='SpSm', q_pattern='q_Qx0_Qy0_Qz0', channel='SF'
    - 'Experimental_q_Qx0_Qy0_Qz0_theta0' -> operator='Experimental', q_pattern='q_Qx0_Qy0_Qz0', theta=0
    - 'TransverseExperimental_q_Qx0_Qy0_Qz0_theta0_SF' -> operator='TransverseExperimental', q_pattern='q_Qx0_Qy0_Qz0', theta=0, channel='SF'
    """
    
    # Extract Q-vector pattern
    q_match = re.search(r'(q_Qx[0-9.-]+_Qy[0-9.-]+_Qz[0-9.-]+)', species_name)
    q_pattern = q_match.group(1) if q_match else None
    
    # Extract theta for experimental operators (must do before channel extraction)
    theta_match = re.search(r'_theta([0-9.-]+)', species_name)
    theta = float(theta_match.group(1)) if theta_match else None
    
    # Extract channel suffix (_SF, _NSF, _DO)
    channel = None
    if species_name.endswith('_SF'):
        channel = 'SF'
    elif species_name.endswith('_NSF'):
        channel = 'NSF'
    elif species_name.endswith('_DO'):
        channel = 'DO'
    
    # Extract operator (everything before q_pattern)
    if q_pattern:
        operator = species_name.split(q_pattern)[0].rstrip('_')
    else:
        operator = species_name
    
    # Extract sublattices if present
    sub_match = re.search(r'_sub(\d+)_sub(\d+)', species_name)
    sublattices = (int(sub_match.group(1)), int(sub_match.group(2))) if sub_match else None
    
    return {
        'operator': operator,
        'q_pattern': q_pattern,
        'sublattices': sublattices,
        'channel': channel,
        'theta': theta
    }


def get_operator_latex_name(operator):
    """Convert operator name to LaTeX format for plotting"""
    
    # Map common operators
    operator_map = {
        'SpSm': r'$S^+S^-$',
        'SmSp': r'$S^-S^+$',
        'SzSz': r'$S^zS^z$',
        'SpSz': r'$S^+S^z$',
        'SmSz': r'$S^-S^z$',
        'SzSp': r'$S^zS^+$',
        'SzSm': r'$S^zS^-$',
        'SmSm': r'$S^-S^-$',
        'SpSp': r'$S^+S^+$',
        'SxSx': r'$S^xS^x$',
        'SySy': r'$S^yS^y$',
        'SxSy': r'$S^xS^y$',
        'SySx': r'$S^yS^x$',
        'SxSz': r'$S^xS^z$',
        'SzSx': r'$S^zS^x$',
        'SySz': r'$S^yS^z$',
        'SzSy': r'$S^zS^y$',
        'Experimental': r'Experimental',
        'TransverseExperimental': r'Transverse Experimental'
    }
    
    return operator_map.get(operator, operator)


# ============================================================================
# Data Processing Functions
# ============================================================================

def get_base_species_name(species):
    """Remove channel suffix (_SF, _NSF) from species name"""
    if species.endswith('_SF') or species.endswith('_NSF'):
        return species[:-3]
    elif species.endswith('_DO'):
        return species[:-3]
    return species


def find_sf_nsf_pairs(all_species):
    """
    Find pairs of SF/NSF species that can be combined into DO.
    This applies to both Transverse and TransverseExperimental operator types.
    
    Returns:
    - do_species: list of tuples (base_name, sf_species, nsf_species, operator_type)
      where operator_type is either 'Transverse' or 'TransverseExperimental'
    """
    # Group species by base name
    from collections import defaultdict
    species_by_base = defaultdict(list)
    
    for species in all_species:
        if species.endswith('_SF') or species.endswith('_NSF'):
            base_name = get_base_species_name(species)
            species_by_base[base_name].append(species)
    
    # Find pairs
    do_species = []
    for base_name, species_list in species_by_base.items():
        sf_species = [s for s in species_list if s.endswith('_SF')]
        nsf_species = [s for s in species_list if s.endswith('_NSF')]
        
        if sf_species and nsf_species:
            # Pair them up (should be 1:1)
            for sf in sf_species:
                for nsf in nsf_species:
                    # Determine operator type
                    if 'TransverseExperimental' in sf:
                        operator_type = 'TransverseExperimental'
                    else:
                        operator_type = 'Transverse'
                    do_species.append((base_name + '_DO', sf, nsf, operator_type))
    
    return do_species


def find_transverse_overlay_pairs(all_species):
    """
    Find TransverseExperimental species and their corresponding SxSx and SzSz Transverse species
    for overlay plotting: cos²θ·SxSx + sin²θ·SzSz should match TransverseExperimental.
    
    Returns:
    - overlay_pairs: list of tuples (transverse_exp_species, sxsx_species, szsz_species, theta)
    """
    overlay_pairs = []
    
    for species in all_species:
        if 'TransverseExperimental' in species:
            info = parse_species_name(species)
            if info['theta'] is not None and info['q_pattern'] is not None:
                # Construct the corresponding SxSx and SzSz Transverse species names
                # TransverseExperimental_q_Qx0_Qy0_Qz0_theta30_SF -> 
                #   SxSx_q_Qx0_Qy0_Qz0_SF and SzSz_q_Qx0_Qy0_Qz0_SF
                
                channel_suffix = ''
                if info['channel']:
                    channel_suffix = f"_{info['channel']}"
                
                sxsx_species = f"SxSx_{info['q_pattern']}{channel_suffix}"
                szsz_species = f"SzSz_{info['q_pattern']}{channel_suffix}"
                
                # Check if both exist
                if sxsx_species in all_species and szsz_species in all_species:
                    overlay_pairs.append((species, sxsx_species, szsz_species, info['theta']))
    
    return overlay_pairs


def compute_transverse_overlay(base_dir, sxsx_species, szsz_species, theta_deg, beta_values):
    """
    Compute the overlay: cos²(theta) * SxSx + sin²(theta) * SzSz
    
    Args:
    - base_dir: base directory
    - sxsx_species: SxSx species name
    - szsz_species: SzSz species name
    - theta_deg: theta angle in degrees
    - beta_values: list of (beta, beta_str, file_path) tuples
    
    Returns:
    - spectral_data_dict: dict mapping beta_str -> (freq, spectral_real, spectral_imag)
    """
    spectral_data_dict = {}
    
    # Convert theta to radians
    theta_rad = np.deg2rad(theta_deg)
    cos2_theta = np.cos(theta_rad) ** 2
    sin2_theta = np.sin(theta_rad) ** 2
    
    sxsx_dir = os.path.join(base_dir, "structure_factor_results", "processed_data", sxsx_species)
    szsz_dir = os.path.join(base_dir, "structure_factor_results", "processed_data", szsz_species)
    
    for beta, beta_str, _ in beta_values:
        # Read SxSx data
        sxsx_file = os.path.join(sxsx_dir, f"spectral_beta_{beta_str}.dat")
        freq_sxsx, spectral_sxsx_real, spectral_sxsx_imag = read_spectral_data(sxsx_file)
        
        # Read SzSz data
        szsz_file = os.path.join(szsz_dir, f"spectral_beta_{beta_str}.dat")
        freq_szsz, spectral_szsz_real, spectral_szsz_imag = read_spectral_data(szsz_file)
        
        if freq_sxsx is None or freq_szsz is None:
            continue
        
        # Check that frequency arrays match
        if not np.allclose(freq_sxsx, freq_szsz):
            print(f"    Warning: Frequency arrays don't match for SxSx/SzSz at β={beta_str}, skipping")
            continue
        
        # Compute overlay: cos²θ·SxSx + sin²θ·SzSz
        spectral_overlay_real = cos2_theta * spectral_sxsx_real + sin2_theta * spectral_szsz_real
        spectral_overlay_imag = cos2_theta * spectral_sxsx_imag + sin2_theta * spectral_szsz_imag
        
        spectral_data_dict[beta_str] = (freq_sxsx, spectral_overlay_real, spectral_overlay_imag)
    
    return spectral_data_dict


def combine_sf_nsf_to_do(base_dir, sf_species, nsf_species, beta_values):
    """
    Combine SF and NSF channel data to create DO (double-differential) channel.
    DO = SF + NSF
    
    Returns:
    - spectral_data_dict: dict mapping beta_str -> (freq, spectral_real, spectral_imag)
    - peak_data_dict: dict mapping beta_str -> peak_data
    """
    spectral_data_dict = {}
    peak_data_dict = {}
    
    sf_dir = os.path.join(base_dir, "structure_factor_results", "processed_data", sf_species)
    nsf_dir = os.path.join(base_dir, "structure_factor_results", "processed_data", nsf_species)
    
    for beta, beta_str, _ in beta_values:
        # Read SF data
        sf_file = os.path.join(sf_dir, f"spectral_beta_{beta_str}.dat")
        freq_sf, spectral_sf_real, spectral_sf_imag = read_spectral_data(sf_file)
        
        # Read NSF data
        nsf_file = os.path.join(nsf_dir, f"spectral_beta_{beta_str}.dat")
        freq_nsf, spectral_nsf_real, spectral_nsf_imag = read_spectral_data(nsf_file)
        
        if freq_sf is None or freq_nsf is None:
            continue
        
        # Check that frequency arrays match
        if not np.allclose(freq_sf, freq_nsf):
            print(f"    Warning: Frequency arrays don't match for SF/NSF at β={beta_str}, skipping")
            continue
        
        # Combine: DO = SF + NSF
        spectral_do_real = spectral_sf_real + spectral_nsf_real
        spectral_do_imag = spectral_sf_imag + spectral_nsf_imag
        
        spectral_data_dict[beta_str] = (freq_sf, spectral_do_real, spectral_do_imag)
        
        # Combine peaks (if both exist)
        sf_peaks = read_peaks_data(sf_dir, beta_str)
        nsf_peaks = read_peaks_data(nsf_dir, beta_str)
        
        if sf_peaks is not None and nsf_peaks is not None:
            # Merge peak lists and sort by frequency
            combined_peaks = np.vstack([sf_peaks, nsf_peaks])
            combined_peaks = combined_peaks[combined_peaks[:, 0].argsort()]
            peak_data_dict[beta_str] = combined_peaks
        elif sf_peaks is not None:
            peak_data_dict[beta_str] = sf_peaks
        elif nsf_peaks is not None:
            peak_data_dict[beta_str] = nsf_peaks
    
    return spectral_data_dict, peak_data_dict


# ============================================================================
# Plotting Functions
# ============================================================================

def create_stacked_plot(beta_values, spectral_data_dict, species, output_file):
    """Create a 2D stacked plot showing spectral function vs frequency for different beta values"""
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    colors = cm.viridis(np.linspace(0, 1, len(beta_values)))
    
    offset = 0
    max_spectral = 0
    
    for idx, (beta, beta_str, _) in enumerate(beta_values):
        freq, spectral_real, _ = spectral_data_dict[beta_str]
        
        if freq is None:
            continue
        
        # Filter frequency range
        mask = (freq >= FREQ_MIN) & (freq <= FREQ_MAX)
        freq_filtered = freq[mask]
        spectral_filtered = spectral_real[mask]
        
        # Track maximum for normalization
        if len(spectral_filtered) > 0:
            max_spectral = max(max_spectral, np.max(spectral_filtered))
        
        # Plot with offset
        beta_label = r'$\beta = \infty$' if np.isinf(beta) else f'$\\beta = {beta:.2f}$'
        ax.plot(freq_filtered, spectral_filtered + offset, 
                label=beta_label, color=colors[idx], linewidth=1.5)
        
        offset += max_spectral * 0.5 if max_spectral > 0 else 1.0
    
    ax.set_xlabel('Frequency (ω/Jzz)', fontsize=12)
    ax.set_ylabel('Spectral Function S(ω) [offset]', fontsize=12)
    
    info = parse_species_name(species)
    operator_label = get_operator_latex_name(info['operator'])
    title = f'Spectral Function: {operator_label}'
    if info['q_pattern']:
        title += f" @ {info['q_pattern']}"
    if info['sublattices']:
        title += f" (sublattices {info['sublattices'][0]},{info['sublattices'][1]})"
    if info['channel']:
        title += f" [{info['channel']}]"
    
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()
    
    print(f"  Created stacked plot: {output_file}")


def create_heatmap_plot(beta_values, spectral_data_dict, species, output_file):
    """Create a 2D heatmap showing spectral function as function of frequency and beta"""
    
    # Collect all data
    all_freq = []
    all_beta = []
    all_spectral = []
    
    for beta, beta_str, _ in beta_values:
        freq, spectral_real, _ = spectral_data_dict[beta_str]
        
        if freq is None:
            continue
        
        # Filter frequency range
        mask = (freq >= FREQ_MIN) & (freq <= FREQ_MAX)
        freq_filtered = freq[mask]
        spectral_filtered = spectral_real[mask]
        
        # For inf beta, use a large finite value for plotting
        beta_plot = 100.0 if np.isinf(beta) else beta
        
        for f, s in zip(freq_filtered, spectral_filtered):
            all_freq.append(f)
            all_beta.append(beta_plot)
            all_spectral.append(s)
    
    if len(all_freq) == 0:
        print(f"  No data to plot for {species}")
        return
    
    all_freq = np.array(all_freq)
    all_beta = np.array(all_beta)
    all_spectral = np.array(all_spectral)
    
    # Create grid for interpolation
    freq_grid = np.linspace(all_freq.min(), all_freq.max(), 200)
    beta_grid = np.linspace(all_beta.min(), all_beta.max(), 50)
    freq_mesh, beta_mesh = np.meshgrid(freq_grid, beta_grid)
    
    # Interpolate data onto grid
    from scipy.interpolate import griddata
    spectral_mesh = griddata((all_freq, all_beta), all_spectral, 
                             (freq_mesh, beta_mesh), method='linear')
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(12, 8))
    
    im = ax.pcolormesh(freq_mesh, beta_mesh, spectral_mesh, 
                       shading='auto', cmap='hot')
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Spectral Function S(ω)', fontsize=12)
    
    ax.set_xlabel('Frequency (ω/Jzz)', fontsize=12)
    ax.set_ylabel('Temperature (β)', fontsize=12)
    
    # Mark infinity line if present
    if any(np.isinf(beta) for beta, _, _ in beta_values):
        ax.axhline(y=100.0, color='cyan', linestyle='--', linewidth=2, 
                   label=r'$\beta = \infty$')
        ax.legend()
    
    info = parse_species_name(species)
    operator_label = get_operator_latex_name(info['operator'])
    title = f'Spectral Function Heatmap: {operator_label}'
    if info['q_pattern']:
        title += f" @ {info['q_pattern']}"
    if info['sublattices']:
        title += f" (sublattices {info['sublattices'][0]},{info['sublattices'][1]})"
    if info['channel']:
        title += f" [{info['channel']}]"
    
    ax.set_title(title, fontsize=14)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()
    
    print(f"  Created heatmap plot: {output_file}")


def create_animation(beta_values, spectral_data_dict, peak_data_dict, species, output_file):
    """Create an animated plot showing spectral function evolution with beta"""
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Find global limits for consistent axes
    all_spectral = []
    for beta, beta_str, _ in beta_values:
        freq, spectral_real, _ = spectral_data_dict[beta_str]
        if freq is not None:
            mask = (freq >= FREQ_MIN) & (freq <= FREQ_MAX)
            all_spectral.extend(spectral_real[mask])
    
    if len(all_spectral) == 0:
        print(f"  No data to animate for {species}")
        return
    
    y_max = max(all_spectral) * 1.1
    
    def update(frame):
        ax.clear()
        
        beta, beta_str, _ = beta_values[frame]
        freq, spectral_real, _ = spectral_data_dict[beta_str]
        
        if freq is None:
            return
        
        # Filter frequency range
        mask = (freq >= FREQ_MIN) & (freq <= FREQ_MAX)
        freq_filtered = freq[mask]
        spectral_filtered = spectral_real[mask]
        
        # Plot spectral function
        ax.plot(freq_filtered, spectral_filtered, 'b-', linewidth=2)
        
        # Plot peaks if available
        peaks = peak_data_dict.get(beta_str)
        if peaks is not None:
            peak_freqs = peaks[:, 0]
            peak_heights = peaks[:, 1]
            ax.scatter(peak_freqs, peak_heights, color='red', s=100, 
                      marker='x', zorder=5, label='Peaks')
        
        ax.set_xlabel('Frequency (ω/Jzz)', fontsize=12)
        ax.set_ylabel('Spectral Function S(ω)', fontsize=12)
        ax.set_xlim(FREQ_MIN, FREQ_MAX)
        ax.set_ylim(0, y_max)
        
        beta_label = r'$\beta = \infty$' if np.isinf(beta) else f'$\\beta = {beta:.2f}$'
        
        info = parse_species_name(species)
        operator_label = get_operator_latex_name(info['operator'])
        title = f'{operator_label}'
        if info['q_pattern']:
            title += f" @ {info['q_pattern']}"
        if info['sublattices']:
            title += f" (sub{info['sublattices'][0]},{info['sublattices'][1]})"
        if info['channel']:
            title += f" [{info['channel']}]"
        title += f" at {beta_label}"
        
        ax.set_title(title, fontsize=14)
        ax.grid(True, alpha=0.3)
        if peaks is not None:
            ax.legend()
    
    anim = FuncAnimation(fig, update, frames=len(beta_values), interval=500)
    
    writer = PillowWriter(fps=0.5)
    anim.save(output_file, writer=writer)
    plt.close()
    
    print(f"  Created animation: {output_file}")


def create_comparison_plot(base_dir, species_list, beta_str, output_file, do_species_list=None, overlay_pairs_list=None):
    """
    Create a comparison plot of multiple species at the same beta.
    
    Args:
    - base_dir: base directory
    - species_list: list of species names to plot
    - beta_str: beta value as string
    - output_file: output file path
    - do_species_list: list of (do_name, sf_species, nsf_species, operator_type) tuples
    - overlay_pairs_list: list of (transverse_exp, sxsx, szsz, theta) tuples for overlay
    """
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Count total species including DO and overlays
    total_species = len(species_list)
    if do_species_list:
        total_species += len(do_species_list)
    if overlay_pairs_list:
        total_species += len(overlay_pairs_list)
    
    colors = cm.tab10(np.linspace(0, 1, min(total_species, 10)))
    if total_species > 10:
        colors = cm.tab20(np.linspace(0, 1, min(total_species, 20)))
    
    idx = 0
    
    # Plot regular species
    for species in species_list:
        species_dir = os.path.join(base_dir, "structure_factor_results", "processed_data", species)
        file_path = os.path.join(species_dir, f"spectral_beta_{beta_str}.dat")
        
        freq, spectral_real, _ = read_spectral_data(file_path)
        
        if freq is None:
            continue
        
        # Filter frequency range
        mask = (freq >= FREQ_MIN) & (freq <= FREQ_MAX)
        freq_filtered = freq[mask]
        spectral_filtered = spectral_real[mask]
        
        info = parse_species_name(species)
        operator_label = get_operator_latex_name(info['operator'])
        label = operator_label
        if info['sublattices']:
            label += f" (sub{info['sublattices'][0]},{info['sublattices'][1]})"
        if info['channel']:
            label += f" [{info['channel']}]"
        if info['theta'] is not None:
            label += f" (θ={info['theta']}°)"
        
        ax.plot(freq_filtered, spectral_filtered, 
                label=label, color=colors[idx % len(colors)], linewidth=2)
        idx += 1
    
    # Plot DO species (combined SF+NSF)
    if do_species_list:
        for do_tuple in do_species_list:
            if len(do_tuple) == 4:
                do_name, sf_species, nsf_species, operator_type = do_tuple
            else:
                # Backward compatibility
                do_name, sf_species, nsf_species = do_tuple
                operator_type = 'Transverse'
            
            # Find beta values for this DO pair
            sf_beta_values = find_all_beta_values(base_dir, sf_species)
            nsf_beta_values = find_all_beta_values(base_dir, nsf_species)
            
            sf_beta_strs = set([bs for _, bs, _ in sf_beta_values])
            nsf_beta_strs = set([bs for _, bs, _ in nsf_beta_values])
            
            if beta_str not in sf_beta_strs or beta_str not in nsf_beta_strs:
                continue
            
            # Get the beta value
            beta_val = next((beta for beta, bs, _ in sf_beta_values if bs == beta_str), None)
            if beta_val is None:
                continue
            
            beta_values = [(beta_val, beta_str, None)]
            
            # Combine data
            spectral_data_dict, _ = combine_sf_nsf_to_do(
                base_dir, sf_species, nsf_species, beta_values)
            
            if beta_str not in spectral_data_dict:
                continue
            
            freq, spectral_real, _ = spectral_data_dict[beta_str]
            
            # Filter frequency range
            mask = (freq >= FREQ_MIN) & (freq <= FREQ_MAX)
            freq_filtered = freq[mask]
            spectral_filtered = spectral_real[mask]
            
            info = parse_species_name(do_name)
            operator_label = get_operator_latex_name(info['operator'])
            label = operator_label
            if info['sublattices']:
                label += f" (sub{info['sublattices'][0]},{info['sublattices'][1]})"
            if info['channel']:
                label += f" [{info['channel']}]"
            if info['theta'] is not None:
                label += f" (θ={info['theta']}°)"
            
            ax.plot(freq_filtered, spectral_filtered, 
                    label=label, color=colors[idx % len(colors)], linewidth=2, linestyle='--')
            idx += 1
    
    # Plot overlay: cos²θ·SxSx + sin²θ·SzSz on TransverseExperimental
    if overlay_pairs_list:
        for transverse_exp, sxsx_species, szsz_species, theta in overlay_pairs_list:
            # Find beta values for the transverse experimental species
            exp_beta_values = find_all_beta_values(base_dir, transverse_exp)
            exp_beta_strs = set([bs for _, bs, _ in exp_beta_values])
            
            if beta_str not in exp_beta_strs:
                continue
            
            # Get the beta value
            beta_val = next((beta for beta, bs, _ in exp_beta_values if bs == beta_str), None)
            if beta_val is None:
                continue
            
            beta_values = [(beta_val, beta_str, None)]
            
            # Compute overlay
            overlay_data_dict = compute_transverse_overlay(
                base_dir, sxsx_species, szsz_species, theta, beta_values)
            
            if beta_str not in overlay_data_dict:
                continue
            
            freq, spectral_real, _ = overlay_data_dict[beta_str]
            
            # Filter frequency range
            mask = (freq >= FREQ_MIN) & (freq <= FREQ_MAX)
            freq_filtered = freq[mask]
            spectral_filtered = spectral_real[mask]
            
            # Create label
            label = f"cos²({theta}°)·SxSx + sin²({theta}°)·SzSz"
            
            ax.plot(freq_filtered, spectral_filtered, 
                    label=label, color=colors[idx % len(colors)], linewidth=2, linestyle=':')
            idx += 1
    
    ax.set_xlabel('Frequency (ω/Jzz)', fontsize=12)
    ax.set_ylabel('Spectral Function S(ω)', fontsize=12)
    
    beta_label = r'$\beta = \infty$' if beta_str == 'inf' else f'$\\beta = {beta_str}$'
    ax.set_title(f'Spectral Function Comparison at {beta_label}', fontsize=14)
    ax.legend(fontsize=9, loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()
    
    print(f"  Created comparison plot: {output_file}")



def create_h_evolution_plot(h_data, species, beta_str, output_file):
    """
    Create a plot showing how spectral function evolves with magnetic field h.
    Stacked plot with offset for different h values.
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    colors = cm.viridis(np.linspace(0, 1, len(h_data)))
    
    offset = 0
    max_spectral = 0
    
    for idx, (h_val, h_dir) in enumerate(h_data):
        species_dir = os.path.join(h_dir, "structure_factor_results", "processed_data", species)
        file_path = os.path.join(species_dir, f"spectral_beta_{beta_str}.dat")
        
        freq, spectral_real, _ = read_spectral_data(file_path)
        
        if freq is None:
            continue
        
        # Filter frequency range
        mask = (freq >= FREQ_MIN) & (freq <= FREQ_MAX)
        freq_filtered = freq[mask]
        spectral_filtered = spectral_real[mask]
        
        # Track maximum for normalization
        if len(spectral_filtered) > 0:
            max_spectral = max(max_spectral, np.max(spectral_filtered))
        
        # Plot with offset
        h_label = f'h = {h_val * H_CONVERSION_FACTOR:.3f}'
        ax.plot(freq_filtered, spectral_filtered + offset, 
                label=h_label, color=colors[idx], linewidth=1.5)
        
        offset += max_spectral * 0.5 if max_spectral > 0 else 1.0
    
    ax.set_xlabel('Frequency (ω/Jzz)', fontsize=12)
    ax.set_ylabel('Spectral Function S(ω) [offset]', fontsize=12)
    
    info = parse_species_name(species)
    operator_label = get_operator_latex_name(info['operator'])
    beta_label = r'$\beta = \infty$' if beta_str == 'inf' else f'$\\beta = {beta_str}$'
    title = f'H-Field Evolution: {operator_label} at {beta_label}'
    if info['q_pattern']:
        title += f" @ {info['q_pattern']}"
    if info['sublattices']:
        title += f" (sub{info['sublattices'][0]},{info['sublattices'][1]})"
    if info['channel']:
        title += f" [{info['channel']}]"
    
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()
    
    print(f"  Created h-evolution plot: {output_file}")


def create_h_heatmap(h_data, species, beta_str, output_file):
    """
    Create a 2D heatmap showing spectral function as function of frequency (x) and h field (y).
    Similar to animate_DSSF_cartesian.py style.
    """
    if not h_data:
        print(f"  No h data available for {species}")
        return
    
    # Get reference frequency array from first h directory
    h_val_ref, h_dir_ref = h_data[0]
    species_dir_ref = os.path.join(h_dir_ref, "structure_factor_results", "processed_data", species)
    file_path_ref = os.path.join(species_dir_ref, f"spectral_beta_{beta_str}.dat")
    freq_ref_full, _, _ = read_spectral_data(file_path_ref)
    
    if freq_ref_full is None:
        print(f"  No reference data for {species}")
        return
    
    # Filter to frequency range
    mask = (freq_ref_full >= FREQ_MIN) & (freq_ref_full <= FREQ_MAX)
    freq_ref = freq_ref_full[mask]
    
    # Build spectral matrix: rows = frequency, columns = h values
    h_values = [h_val for h_val, _ in h_data]
    spectral_matrix = np.zeros((len(freq_ref), len(h_values)))
    
    for i, (h_val, h_dir) in enumerate(h_data):
        species_dir = os.path.join(h_dir, "structure_factor_results", "processed_data", species)
        file_path = os.path.join(species_dir, f"spectral_beta_{beta_str}.dat")
        
        freq_h, spectral_h, _ = read_spectral_data(file_path)
        
        if freq_h is None:
            continue
        
        # Filter and interpolate to reference frequency grid
        mask_h = (freq_h >= FREQ_MIN) & (freq_h <= FREQ_MAX)
        freq_h_filtered = freq_h[mask_h]
        spectral_h_filtered = spectral_h[mask_h]
        
        # Interpolate to match reference frequency grid
        spectral_matrix[:, i] = np.interp(freq_ref, freq_h_filtered, spectral_h_filtered)
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Convert h values
    h_values_converted = np.array(h_values) * H_CONVERSION_FACTOR
    freq_converted = freq_ref * ENERGY_CONVERSION_FACTOR
    
    # Create edges for pcolormesh
    h_edges = np.zeros(len(h_values_converted) + 1)
    if len(h_values_converted) > 1:
        h_edges[0] = h_values_converted[0] - (h_values_converted[1] - h_values_converted[0]) / 2
        h_edges[-1] = h_values_converted[-1] + (h_values_converted[-1] - h_values_converted[-2]) / 2
        for i in range(1, len(h_values_converted)):
            h_edges[i] = (h_values_converted[i-1] + h_values_converted[i]) / 2
    else:
        h_edges[0] = h_values_converted[0] - 0.5
        h_edges[1] = h_values_converted[0] + 0.5
    
    freq_edges = np.zeros(len(freq_converted) + 1)
    if len(freq_converted) > 1:
        freq_edges[0] = freq_converted[0] - (freq_converted[1] - freq_converted[0]) / 2
        freq_edges[-1] = freq_converted[-1] + (freq_converted[-1] - freq_converted[-2]) / 2
        for i in range(1, len(freq_converted)):
            freq_edges[i] = (freq_converted[i-1] + freq_converted[i]) / 2
    else:
        freq_edges[0] = freq_converted[0] - 0.5
        freq_edges[1] = freq_converted[0] + 0.5
    
    # Plot with h on y-axis, frequency on x-axis
    im = ax.pcolormesh(freq_edges, h_edges, spectral_matrix.T, 
                       cmap='viridis', shading='flat')
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Spectral Function S(ω)', fontsize=12)
    
    ax.set_xlabel('Energy (meV)' if ENERGY_CONVERSION_FACTOR != 1.0 else 'Frequency (ω/Jzz)', fontsize=12)
    ax.set_ylabel('Magnetic Field (h)', fontsize=12)
    
    info = parse_species_name(species)
    operator_label = get_operator_latex_name(info['operator'])
    beta_label = r'$\beta = \infty$' if beta_str == 'inf' else f'$\\beta = {beta_str}$'
    title = f'Spectral Function Heatmap: {operator_label} at {beta_label}'
    if info['q_pattern']:
        title += f" @ {info['q_pattern']}"
    if info['sublattices']:
        title += f" (sub{info['sublattices'][0]},{info['sublattices'][1]})"
    if info['channel']:
        title += f" [{info['channel']}]"
    
    ax.set_title(title, fontsize=14)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Created h-heatmap: {output_file}")


def create_h_animation(h_data, species, beta_str, output_file):
    """
    Create an animated plot showing spectral function evolution with h field.
    Similar to animate_DSSF_cartesian.py style - animates through h values.
    """
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Find global y-limits for consistent axes
    all_spectral = []
    freq_ref = None
    
    for h_val, h_dir in h_data:
        species_dir = os.path.join(h_dir, "structure_factor_results", "processed_data", species)
        file_path = os.path.join(species_dir, f"spectral_beta_{beta_str}.dat")
        
        freq, spectral_real, _ = read_spectral_data(file_path)
        if freq is not None:
            mask = (freq >= FREQ_MIN) & (freq <= FREQ_MAX)
            all_spectral.extend(spectral_real[mask])
            if freq_ref is None:
                freq_ref = freq[mask]
    
    if len(all_spectral) == 0:
        print(f"  No data to animate for {species}")
        return
    
    ymin = min(all_spectral)
    ymax = max(all_spectral)
    y_range = ymax - ymin
    ymin -= 0.1 * y_range
    ymax += 0.1 * y_range
    
    line, = ax.plot([], [], 'b-', linewidth=2)
    ax.set_xlim(FREQ_MIN * ENERGY_CONVERSION_FACTOR, FREQ_MAX * ENERGY_CONVERSION_FACTOR)
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel('Energy (meV)' if ENERGY_CONVERSION_FACTOR != 1.0 else 'Frequency (ω/Jzz)', fontsize=12)
    ax.set_ylabel('Spectral Function S(ω)', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    info = parse_species_name(species)
    operator_label = get_operator_latex_name(info['operator'])
    beta_label = r'$\beta = \infty$' if beta_str == 'inf' else f'$\\beta = {beta_str}$'
    
    title = ax.set_title('', fontsize=14, fontweight='bold')
    
    def init():
        line.set_data([], [])
        return line, title
    
    def animate(frame):
        h_val, h_dir = h_data[frame]
        species_dir = os.path.join(h_dir, "structure_factor_results", "processed_data", species)
        file_path = os.path.join(species_dir, f"spectral_beta_{beta_str}.dat")
        
        freq, spectral_real, _ = read_spectral_data(file_path)
        
        if freq is not None:
            mask = (freq >= FREQ_MIN) & (freq <= FREQ_MAX)
            freq_converted = freq[mask] * ENERGY_CONVERSION_FACTOR
            line.set_data(freq_converted, spectral_real[mask])
            
            h_label = f'h = {h_val * H_CONVERSION_FACTOR:.3f}'
            title_text = f'{operator_label} at {beta_label}, {h_label}'
            if info['q_pattern']:
                title_text += f" @ {info['q_pattern']}"
            if info['sublattices']:
                title_text += f" (sub{info['sublattices'][0]},{info['sublattices'][1]})"
            if info['channel']:
                title_text += f" [{info['channel']}]"
            title.set_text(title_text)
        
        return line, title
    
    anim = FuncAnimation(fig, animate, init_func=init,
                        frames=len(h_data), interval=200, blit=True)
    
    writer = PillowWriter(fps=0.5)
    anim.save(output_file, writer=writer)
    plt.close()
    
    print(f"  Created h-animation: {output_file}")


# ============================================================================
# Main Processing Function
# ============================================================================

def main():
    """Main processing function"""
    
    print("="*80)
    print("SPECTRAL FUNCTION ANIMATION AND ANALYSIS")
    print("="*80)
    print(f"Base directory: {BASE_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    print()
    
    # Check if we have h=* directories (multiple field strengths)
    h_data = find_all_h_directories()
    
    if h_data:
        print(f"Found {len(h_data)} magnetic field directories:")
        for h_val, h_dir in h_data:
            print(f"  - h = {h_val * H_CONVERSION_FACTOR:.4f} ({os.path.basename(h_dir)})")
        print()
        
        # Process multiple h directories
        process_multiple_h_directories(h_data)
    else:
        # Single directory mode (no h=* subdirectories)
        print("No h=* directories found, processing single directory")
        print()
        process_single_directory(BASE_DIR)
    
    print("\n" + "="*80)
    print("PROCESSING COMPLETE")
    print("="*80)
    print(f"Output saved to: {OUTPUT_DIR}")


def process_single_directory(base_dir):
    """Process a single directory (no h field scan)"""
    
    # Find all species
    all_species = find_all_species(base_dir)
    
    if not all_species:
        print("No species found in processed_data directory!")
        return
    
    print(f"Found {len(all_species)} species:")
    for species in all_species:
        print(f"  - {species}")
    print()
    
    # Find SF/NSF pairs that can be combined into DO
    do_species_list = find_sf_nsf_pairs(all_species)
    
    if do_species_list:
        print(f"\nFound {len(do_species_list)} SF/NSF pairs to combine into DO:")
        for do_tuple in do_species_list:
            if len(do_tuple) == 4:
                do_name, sf_name, nsf_name, operator_type = do_tuple
                print(f"  - {do_name} = {sf_name} + {nsf_name} ({operator_type})")
            else:
                do_name, sf_name, nsf_name = do_tuple
                print(f"  - {do_name} = {sf_name} + {nsf_name}")
        print()
    
    # Find TransverseExperimental species and their overlay pairs
    overlay_pairs = find_transverse_overlay_pairs(all_species)
    
    if overlay_pairs:
        print(f"\nFound {len(overlay_pairs)} TransverseExperimental species with overlay pairs:")
        for transverse_exp, sxsx, szsz, theta in overlay_pairs:
            print(f"  - {transverse_exp} ~ cos²({theta}°)·{sxsx} + sin²({theta}°)·{szsz}")
        print()
    
    # Process each species
    for species in all_species:
        print(f"\n{'='*60}")
        print(f"Processing species: {species}")
        print(f"{'='*60}")
        
        # Find all beta values for this species
        beta_values = find_all_beta_values(BASE_DIR, species)
        
        if not beta_values:
            print(f"  No spectral data found for {species}")
            continue
        
        print(f"  Found {len(beta_values)} beta values:")
        for beta, beta_str, _ in beta_values:
            beta_label = "inf" if np.isinf(beta) else f"{beta:.4g}"
            print(f"    - β = {beta_label}")
        
        # Load all spectral data
        spectral_data_dict = {}
        peak_data_dict = {}
        species_dir = os.path.join(base_dir, "structure_factor_results", "processed_data", species)
        
        for beta, beta_str, file_path in beta_values:
            freq, spectral_real, spectral_imag = read_spectral_data(file_path)
            spectral_data_dict[beta_str] = (freq, spectral_real, spectral_imag)
            
            # Also load peak data
            peaks = read_peaks_data(species_dir, beta_str)
            peak_data_dict[beta_str] = peaks
        
        # Create safe filename
        safe_species_name = species.replace('/', '_').replace('\\', '_')
        
        # Create individual plots
        print(f"  Generating plots...")
        
        # Stacked plot
        stacked_file = os.path.join(SUBDIRS['individual'], f"{safe_species_name}_stacked.png")
        create_stacked_plot(beta_values, spectral_data_dict, species, stacked_file)
        
        # Heatmap
        heatmap_file = os.path.join(SUBDIRS['heatmaps'], f"{safe_species_name}_heatmap.png")
        create_heatmap_plot(beta_values, spectral_data_dict, species, heatmap_file)
        
        # Animation
        animation_file = os.path.join(SUBDIRS['beta_evolution'], f"{safe_species_name}_animation.gif")
        create_animation(beta_values, spectral_data_dict, peak_data_dict, species, animation_file)
    
    # Process DO (combined SF+NSF) channels
    if do_species_list:
        print(f"\n{'='*60}")
        print("Processing DO (SF+NSF) combined channels")
        print(f"{'='*60}")
        
        for do_tuple in do_species_list:
            if len(do_tuple) == 4:
                do_name, sf_species, nsf_species, operator_type = do_tuple
            else:
                # Backward compatibility
                do_name, sf_species, nsf_species = do_tuple
                operator_type = 'Unknown'
            
            print(f"\nCombining {sf_species} + {nsf_species} -> {do_name} ({operator_type})")
            
            # Find common beta values between SF and NSF
            sf_beta_values = find_all_beta_values(base_dir, sf_species)
            nsf_beta_values = find_all_beta_values(base_dir, nsf_species)
            
            sf_beta_strs = set([beta_str for _, beta_str, _ in sf_beta_values])
            nsf_beta_strs = set([beta_str for _, beta_str, _ in nsf_beta_values])
            common_beta_strs = sf_beta_strs & nsf_beta_strs
            
            # Filter to common betas
            beta_values = [(beta, beta_str, file_path) for beta, beta_str, file_path in sf_beta_values 
                          if beta_str in common_beta_strs]
            
            if not beta_values:
                print(f"  No common beta values found, skipping")
                continue
            
            print(f"  Found {len(beta_values)} common beta values")
            
            # Combine SF and NSF data
            spectral_data_dict, peak_data_dict = combine_sf_nsf_to_do(
                base_dir, sf_species, nsf_species, beta_values)
            
            if not spectral_data_dict:
                print(f"  No data to combine, skipping")
                continue
            
            # Create safe filename
            safe_do_name = do_name.replace('/', '_').replace('\\', '_')
            
            # Create plots for DO channel
            print(f"  Generating DO plots...")
            
            # Stacked plot
            stacked_file = os.path.join(SUBDIRS['individual'], f"{safe_do_name}_stacked.png")
            create_stacked_plot(beta_values, spectral_data_dict, do_name, stacked_file)
            
            # Heatmap
            heatmap_file = os.path.join(SUBDIRS['heatmaps'], f"{safe_do_name}_heatmap.png")
            create_heatmap_plot(beta_values, spectral_data_dict, do_name, heatmap_file)
            
            # Animation
            animation_file = os.path.join(SUBDIRS['beta_evolution'], f"{safe_do_name}_animation.gif")
            create_animation(beta_values, spectral_data_dict, peak_data_dict, do_name, animation_file)
    
    # Process overlay pairs (cos²θ·SxSx + sin²θ·SzSz)
    if overlay_pairs:
        print(f"\n{'='*60}")
        print("Processing TransverseExperimental overlay pairs")
        print(f"{'='*60}")
        
        for transverse_exp, sxsx_species, szsz_species, theta in overlay_pairs:
            print(f"\nCreating overlay for {transverse_exp}")
            print(f"  cos²({theta}°)·{sxsx_species} + sin²({theta}°)·{szsz_species}")
            
            # Find common beta values
            exp_beta_values = find_all_beta_values(base_dir, transverse_exp)
            sxsx_beta_values = find_all_beta_values(base_dir, sxsx_species)
            szsz_beta_values = find_all_beta_values(base_dir, szsz_species)
            
            exp_beta_strs = set([bs for _, bs, _ in exp_beta_values])
            sxsx_beta_strs = set([bs for _, bs, _ in sxsx_beta_values])
            szsz_beta_strs = set([bs for _, bs, _ in szsz_beta_values])
            common_beta_strs = exp_beta_strs & sxsx_beta_strs & szsz_beta_strs
            
            # Filter to common betas
            beta_values = [(beta, beta_str, file_path) for beta, beta_str, file_path in exp_beta_values 
                          if beta_str in common_beta_strs]
            
            if not beta_values:
                print(f"  No common beta values found, skipping")
                continue
            
            print(f"  Found {len(beta_values)} common beta values")
            
            # Compute overlay
            overlay_data_dict = compute_transverse_overlay(
                base_dir, sxsx_species, szsz_species, theta, beta_values)
            
            if not overlay_data_dict:
                print(f"  No overlay data computed, skipping")
                continue
            
            # Create safe filename for overlay
            safe_overlay_name = f"Overlay_theta{theta}_{sxsx_species}_{szsz_species}".replace('/', '_').replace('\\', '_')
            overlay_name = f"cos²({theta}°)·SxSx + sin²({theta}°)·SzSz"
            
            # Create plots for overlay
            print(f"  Generating overlay plots...")
            
            # Note: We create visualizations but not individual stacked/heatmap/animation for overlay
            # They will appear in comparison plots
    
    # Create comparison plots at key beta values
    print(f"\n{'='*60}")
    print("Creating comparison plots")
    print(f"{'='*60}")
    
    # Combine regular species and DO species for comparison
    all_species_for_comparison = list(all_species)
    
    # Extract do_names from do_species_list (handling both 3-tuple and 4-tuple formats)
    do_names = []
    for do_tuple in do_species_list:
        if len(do_tuple) >= 3:
            do_names.append(do_tuple[0])  # do_name is always the first element
    
    # Find common beta values across all species (including DO)
    beta_sets = []
    for species in all_species:
        beta_values = find_all_beta_values(base_dir, species)
        beta_strs = set([beta_str for _, beta_str, _ in beta_values])
        beta_sets.append(beta_strs)
    
    # For DO species, we need to check SF channel (representative)
    for do_tuple in do_species_list:
        if len(do_tuple) >= 2:
            sf_species = do_tuple[1]  # sf_species is always the second element
            beta_values = find_all_beta_values(base_dir, sf_species)
            beta_strs = set([beta_str for _, beta_str, _ in beta_values])
            beta_sets.append(beta_strs)
    
    if beta_sets:
        common_betas = set.intersection(*beta_sets)
        print(f"  Found {len(common_betas)} common beta values: {sorted(common_betas)}")
        
        for beta_str in sorted(common_betas):
            print(f"  Creating comparison for β = {beta_str}")
            comparison_file = os.path.join(SUBDIRS['combined'], f"comparison_beta_{beta_str}.png")
            
            # Pass DO species info and overlay pairs for comparison plot
            create_comparison_plot(base_dir, all_species, beta_str, comparison_file, 
                                 do_species_list=do_species_list,
                                 overlay_pairs_list=overlay_pairs)


def process_multiple_h_directories(h_data):
    """Process multiple h=* directories (magnetic field scan)"""
    
    # First, find all species that are common across all h directories
    all_species_per_h = []
    for h_val, h_dir in h_data:
        species = find_all_species(h_dir)
        all_species_per_h.append(set(species))
    
    if not all_species_per_h:
        print("No species found in any h directory!")
        return
    
    # Find common species across all h values
    common_species = set.intersection(*all_species_per_h)
    
    if not common_species:
        print("No common species found across all h directories!")
        return
    
    print(f"Found {len(common_species)} common species across all h values:")
    for species in sorted(common_species):
        print(f"  - {species}")
    print()
    
    # For each common species, create h-evolution plots
    for species in sorted(common_species):
        print(f"\n{'='*60}")
        print(f"Processing h-evolution for: {species}")
        print(f"{'='*60}")
        
        # Find common beta values across all h directories for this species
        beta_sets = []
        for h_val, h_dir in h_data:
            beta_values = find_all_beta_values(h_dir, species)
            beta_strs = set([beta_str for _, beta_str, _ in beta_values])
            beta_sets.append(beta_strs)
        
        if not beta_sets:
            continue
        
        common_betas = set.intersection(*beta_sets)
        
        if not common_betas:
            print(f"  No common beta values across h directories, skipping")
            continue
        
        print(f"  Found {len(common_betas)} common beta values: {sorted(common_betas)}")
        
        # Create h-evolution plots for each common beta
        safe_species_name = species.replace('/', '_').replace('\\', '_')
        
        for beta_str in sorted(common_betas):
            print(f"  Creating h-evolution plots for β = {beta_str}")
            
            # Stacked plot
            h_stacked_file = os.path.join(SUBDIRS['h_evolution'], 
                                         f"{safe_species_name}_beta_{beta_str}_h_stacked.png")
            create_h_evolution_plot(h_data, species, beta_str, h_stacked_file)
            
            # Heatmap
            h_heatmap_file = os.path.join(SUBDIRS['h_evolution'], 
                                         f"{safe_species_name}_beta_{beta_str}_h_heatmap.png")
            create_h_heatmap(h_data, species, beta_str, h_heatmap_file)
            
            # Animation
            h_animation_file = os.path.join(SUBDIRS['h_evolution'], 
                                           f"{safe_species_name}_beta_{beta_str}_h_animation.gif")
            create_h_animation(h_data, species, beta_str, h_animation_file)


if __name__ == "__main__":
    main()
