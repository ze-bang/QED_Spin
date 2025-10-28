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
- Automatically detects and combines SF/NSF pairs into DO (double-differential) channels
- DO = SF + NSF
- DO channels are plotted with dashed lines in comparison plots
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
BASE_DIR = "/scratch/zhouzb79/DSSF_example_output"  # Change to your directory
OUTPUT_DIR = os.path.join(BASE_DIR, "spectral_animations")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Create organized subdirectories
SUBDIRS = {
    'individual': os.path.join(OUTPUT_DIR, "1_individual_species"),
    'combined': os.path.join(OUTPUT_DIR, "2_combined_plots"),
    'beta_evolution': os.path.join(OUTPUT_DIR, "3_beta_evolution"),
    'heatmaps': os.path.join(OUTPUT_DIR, "4_heatmaps"),
    'summary': os.path.join(OUTPUT_DIR, "0_summary")
}

# Create all subdirectories
for subdir in SUBDIRS.values():
    os.makedirs(subdir, exist_ok=True)

# Conversion factors (adjust for your system)
# For omega values (energy/frequency): Jzz = 0.063 meV for example material
ENERGY_CONVERSION_FACTOR = 0.063  # converts from Jzz units to meV

# Frequency range limits (in Jzz units)
FREQ_MIN = -3.0
FREQ_MAX = 6.0


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
    
    Returns:
    - do_species: list of tuples (base_name, sf_species, nsf_species)
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
                    do_species.append((base_name + '_DO', sf, nsf))
    
    return do_species


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
    
    writer = PillowWriter(fps=2)
    anim.save(output_file, writer=writer)
    plt.close()
    
    print(f"  Created animation: {output_file}")


def create_comparison_plot(base_dir, species_list, beta_str, output_file, do_species_list=None):
    """Create a comparison plot of multiple species at the same beta"""
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Count total species including DO
    total_species = len(species_list)
    if do_species_list:
        total_species += len(do_species_list)
    
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
        
        ax.plot(freq_filtered, spectral_filtered, 
                label=label, color=colors[idx % len(colors)], linewidth=2)
        idx += 1
    
    # Plot DO species (combined SF+NSF)
    if do_species_list:
        for do_name, sf_species, nsf_species in do_species_list:
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
            
            ax.plot(freq_filtered, spectral_filtered, 
                    label=label, color=colors[idx % len(colors)], linewidth=2, linestyle='--')
            idx += 1
    
    ax.set_xlabel('Frequency (ω/Jzz)', fontsize=12)
    ax.set_ylabel('Spectral Function S(ω)', fontsize=12)
    
    beta_label = r'$\beta = \infty$' if beta_str == 'inf' else f'$\\beta = {beta_str}$'
    ax.set_title(f'Spectral Function Comparison at {beta_label}', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()
    
    print(f"  Created comparison plot: {output_file}")


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
    
    # Find all species
    all_species = find_all_species(BASE_DIR)
    
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
        for do_name, sf_name, nsf_name in do_species_list:
            print(f"  - {do_name} = {sf_name} + {nsf_name}")
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
        species_dir = os.path.join(BASE_DIR, "structure_factor_results", "processed_data", species)
        
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
        
        for do_name, sf_species, nsf_species in do_species_list:
            print(f"\nCombining {sf_species} + {nsf_species} -> {do_name}")
            
            # Find common beta values between SF and NSF
            sf_beta_values = find_all_beta_values(BASE_DIR, sf_species)
            nsf_beta_values = find_all_beta_values(BASE_DIR, nsf_species)
            
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
                BASE_DIR, sf_species, nsf_species, beta_values)
            
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
    
    # Create comparison plots at key beta values
    print(f"\n{'='*60}")
    print("Creating comparison plots")
    print(f"{'='*60}")
    
    # Combine regular species and DO species for comparison
    all_species_for_comparison = list(all_species)
    do_names = [do_name for do_name, _, _ in do_species_list]
    
    # Find common beta values across all species (including DO)
    beta_sets = []
    for species in all_species:
        beta_values = find_all_beta_values(BASE_DIR, species)
        beta_strs = set([beta_str for _, beta_str, _ in beta_values])
        beta_sets.append(beta_strs)
    
    # For DO species, we need to check SF channel (representative)
    for do_name, sf_species, _ in do_species_list:
        beta_values = find_all_beta_values(BASE_DIR, sf_species)
        beta_strs = set([beta_str for _, beta_str, _ in beta_values])
        beta_sets.append(beta_strs)
    
    if beta_sets:
        common_betas = set.intersection(*beta_sets)
        print(f"  Found {len(common_betas)} common beta values: {sorted(common_betas)}")
        
        for beta_str in sorted(common_betas):
            print(f"  Creating comparison for β = {beta_str}")
            comparison_file = os.path.join(SUBDIRS['combined'], f"comparison_beta_{beta_str}.png")
            
            # Pass DO species info for comparison plot
            create_comparison_plot(BASE_DIR, all_species, beta_str, comparison_file, 
                                 do_species_list=do_species_list)
    
    print("\n" + "="*80)
    print("PROCESSING COMPLETE")
    print("="*80)
    print(f"Output saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
