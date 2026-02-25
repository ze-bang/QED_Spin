"""
Calculate and plot heatmaps of energy and specific heat from SS_rand*.dat files.

This script works with thermodynamic data files that contain:
# inv_temp energy variance num doublon step

Specific heat is computed as: C_v = beta^2 * variance
where variance = <E^2> - <E>^2

Expected directory structure:
- data_dir/
  - Jpm=-0.1/
    - output/
      - SS_rand0.dat
      - SS_rand1.dat  (optional)
  - Jpm=0.1/
    - output/
      - SS_rand0.dat
"""

import os
import re
import glob
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from collections import defaultdict
from scipy.interpolate import interp1d

# Use Computer Modern fonts (LaTeX style)
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['Computer Modern Roman', 'CMU Serif', 'DejaVu Serif']
mpl.rcParams['mathtext.fontset'] = 'cm'
mpl.rcParams['axes.unicode_minus'] = False
mpl.rcParams['text.usetex'] = False  # Set to True if LaTeX is available
mpl.rcParams['axes.labelsize'] = 12
mpl.rcParams['axes.titlesize'] = 14
mpl.rcParams['xtick.labelsize'] = 10
mpl.rcParams['ytick.labelsize'] = 10
mpl.rcParams['legend.fontsize'] = 10
mpl.rcParams['figure.titlesize'] = 14

try:
    import h5py
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False
    print("Warning: h5py not available, will only read .dat files")

# NumPy compatibility
if hasattr(np, 'trapezoid'):
    np_trapz = np.trapezoid
else:
    np_trapz = np.trapz


def parse_ss_file(filepath):
    """
    Parse SS_rand*.dat file and return thermodynamic data.
    Skip the first entry which is typically noisy (smallest beta).
    
    Returns: (inv_temp, energy, variance) arrays
    """
    try:
        data = np.loadtxt(filepath, comments='#')
        if data.ndim == 1:
            data = data.reshape(1, -1)
        
        # Skip the first entry (index 0) which is the smallest beta and typically noisy
        if len(data) > 1:
            data = data[1:, :]
        
        inv_temp = data[:, 0]  # beta
        energy = data[:, 1]
        variance = data[:, 2]
        
        return inv_temp, energy, variance
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None, None, None


def parse_h5_file(filepath):
    """
    Parse ed_results.h5 file and return thermodynamic data from all TPQ samples.
    
    The HDF5 file structure is:
    - tpq/samples/sample_*/thermodynamics: (N, 5) array with columns
      [inv_temp, energy, variance, num_doublon, step]
    
    Returns: list of (inv_temp, energy, variance) tuples, one per sample
    """
    if not HAS_H5PY:
        return []
    
    samples_data = []
    try:
        with h5py.File(filepath, 'r') as f:
            # Check if tpq/samples exists
            if 'tpq/samples' not in f:
                print(f"  No tpq/samples group in {filepath}")
                return []
            
            samples_group = f['tpq/samples']
            
            for sample_name in sorted(samples_group.keys()):
                sample = samples_group[sample_name]
                if 'thermodynamics' not in sample:
                    continue
                
                data = sample['thermodynamics'][:]
                if data.ndim == 1:
                    data = data.reshape(1, -1)
                
                # Skip the first entry (index 0) which is the smallest beta and typically noisy
                if len(data) > 1:
                    data = data[1:, :]
                
                inv_temp = data[:, 0]  # beta
                energy = data[:, 1]
                variance = data[:, 2]
                
                samples_data.append((inv_temp, energy, variance))
            
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return []
    
    return samples_data


def compute_specific_heat(beta, variance):
    """
    Compute specific heat from variance.
    C_v = beta^2 * variance = beta^2 * (<E^2> - <E>^2)
    """
    return beta**2 * variance


def load_thermodynamic_data(data_dir, param_pattern='Jpm', n_sites=1):
    """
    Load thermodynamic data from ed_results.h5 or SS_rand*.dat files across parameter sweep.
    
    Supports two formats:
    1. ed_results.h5 (HDF5 format, preferred) - reads from tpq/samples/sample_*/thermodynamics
    2. SS_rand*.dat (text format, fallback)
    
    Parameters:
    data_dir: Root directory containing parameter subdirectories
    param_pattern: Parameter name pattern (e.g., 'Jpm', 'h', 'J')
    n_sites: Number of sites for per-site normalization
    
    Returns:
    data_points: list of (param_value, beta, energy_per_site, specific_heat_per_site, ...)
    """
    
    print(f"\n{'='*70}")
    print(f"Loading thermodynamic data for {param_pattern} sweep")
    print(f"Data directory: {data_dir}")
    print(f"Number of sites: {n_sites} (all quantities will be per-site)")
    print(f"{'='*70}\n")
    
    # Find all subdirectories matching the pattern
    subdirs = sorted(glob.glob(os.path.join(data_dir, f'{param_pattern}=*')))
    print(f"Found {len(subdirs)} {param_pattern} subdirectories")
    
    param_regex = re.compile(f'{param_pattern}=([-]?[\\d\\.]+)')
    
    all_data = []
    
    for subdir in subdirs:
        match = param_regex.search(os.path.basename(subdir))
        if not match:
            continue
            
        param_val = float(match.group(1))
        
        # Find data files in output subdirectory
        output_dir = os.path.join(subdir, 'output')
        if not os.path.isdir(output_dir):
            # Try looking directly in the subdir
            output_dir = subdir
        
        # Collect data from all random samples
        all_beta = []
        all_energy = []
        all_cv = []
        
        # First, try to load from HDF5 file (preferred format)
        h5_file = os.path.join(output_dir, 'ed_results.h5')
        if HAS_H5PY and os.path.isfile(h5_file):
            samples_data = parse_h5_file(h5_file)
            if samples_data:
                print(f"  {param_pattern}={param_val}: Found HDF5 file with {len(samples_data)} samples")
                
                for beta, energy, variance in samples_data:
                    if beta is None:
                        continue
                    
                    # Filter out invalid data
                    valid_mask = (beta > 0) & np.isfinite(energy) & np.isfinite(variance) & (variance >= 0)
                    beta = beta[valid_mask]
                    energy = energy[valid_mask]
                    variance = variance[valid_mask]
                    
                    if len(beta) == 0:
                        continue
                    
                    cv = compute_specific_heat(beta, variance)
                    
                    # Normalize by number of sites
                    all_beta.append(beta)
                    all_energy.append(energy / n_sites)
                    all_cv.append(cv / n_sites)
        
        # Fall back to SS_rand*.dat files if no HDF5 data found
        if not all_beta:
            ss_files = glob.glob(os.path.join(output_dir, 'SS_rand*.dat'))
            
            if not ss_files:
                print(f"  No data files (ed_results.h5 or SS_rand*.dat) found in {output_dir}")
                continue
            
            print(f"  {param_pattern}={param_val}: Found {len(ss_files)} SS files")
            
            for ss_file in ss_files:
                beta, energy, variance = parse_ss_file(ss_file)
                
                if beta is None:
                    continue
                
                # Filter out invalid data (beta should be positive, variance should be positive)
                valid_mask = (beta > 0) & np.isfinite(energy) & np.isfinite(variance) & (variance >= 0)
                beta = beta[valid_mask]
                energy = energy[valid_mask]
                variance = variance[valid_mask]
                
                if len(beta) == 0:
                    continue
                
                cv = compute_specific_heat(beta, variance)
                
                # Normalize by number of sites
                all_beta.append(beta)
                all_energy.append(energy / n_sites)
                all_cv.append(cv / n_sites)
        
        if not all_beta:
            continue
        
        # Average over samples with proper equal weighting
        # Use the first sample's beta grid as reference
        beta_combined = all_beta[0]
        n_samples = len(all_beta)
        
        # Interpolate all samples onto the reference beta grid
        all_energy_interp = []
        all_cv_interp = []
        
        for i in range(n_samples):
            if len(all_beta[i]) > 1:
                # Interpolate this sample onto the reference grid
                f_energy = interp1d(all_beta[i], all_energy[i], 
                                   bounds_error=False, fill_value=np.nan)
                f_cv = interp1d(all_beta[i], all_cv[i], 
                               bounds_error=False, fill_value=np.nan)
                all_energy_interp.append(f_energy(beta_combined))
                all_cv_interp.append(f_cv(beta_combined))
            elif len(all_beta[i]) == 1:
                # Single point - can only use if beta matches
                interp_energy = np.full_like(beta_combined, np.nan)
                interp_cv = np.full_like(beta_combined, np.nan)
                idx = np.argmin(np.abs(beta_combined - all_beta[i][0]))
                if np.abs(beta_combined[idx] - all_beta[i][0]) < 0.01 * all_beta[i][0]:
                    interp_energy[idx] = all_energy[i][0]
                    interp_cv[idx] = all_cv[i][0]
                all_energy_interp.append(interp_energy)
                all_cv_interp.append(interp_cv)
        
        if not all_energy_interp:
            continue
        
        # Stack and compute mean and standard error
        energy_stack = np.array(all_energy_interp)  # Shape: (n_samples, n_beta)
        cv_stack = np.array(all_cv_interp)
        
        energy_combined = np.nanmean(energy_stack, axis=0)
        cv_combined = np.nanmean(cv_stack, axis=0)
        
        # Compute standard error of the mean (SEM = std / sqrt(n))
        # Count valid samples at each beta point
        n_valid_energy = np.sum(~np.isnan(energy_stack), axis=0)
        n_valid_cv = np.sum(~np.isnan(cv_stack), axis=0)
        
        energy_std = np.nanstd(energy_stack, axis=0, ddof=1)  # ddof=1 for sample std
        cv_std = np.nanstd(cv_stack, axis=0, ddof=1)
        
        # SEM, avoiding division by zero
        energy_sem = np.where(n_valid_energy > 1, energy_std / np.sqrt(n_valid_energy), np.nan)
        cv_sem = np.where(n_valid_cv > 1, cv_std / np.sqrt(n_valid_cv), np.nan)
        
        if n_samples > 1:
            print(f"    Averaged {n_samples} samples, energy SEM range: [{np.nanmin(energy_sem):.2e}, {np.nanmax(energy_sem):.2e}]")
        
        # Store data points (now with 6 columns: param, beta, energy, cv, energy_sem, cv_sem)
        for j in range(len(beta_combined)):
            all_data.append((param_val, beta_combined[j], energy_combined[j], cv_combined[j],
                           energy_sem[j], cv_sem[j]))
    
    print(f"\nLoaded {len(all_data)} total data points")
    return all_data


def create_thermodynamic_heatmaps(data_points, data_dir, param_pattern='Jpm', n_sites=1, Jzz=1.0):
    """
    Create heatmaps of energy and specific heat per site vs parameter and beta.
    
    Parameters:
    data_points: List of (param, beta, energy_per_site, cv_per_site, energy_sem, cv_sem)
    data_dir: Root data directory
    param_pattern: Parameter name
    n_sites: Number of sites (for labeling)
    Jzz: Value of Jzz for theoretical prediction lines
    """
    
    print(f"\n{'='*70}")
    print(f"Creating thermodynamic heatmaps")
    print(f"{'='*70}\n")
    
    plot_outdir = os.path.join(data_dir, f'thermodynamic_plots_{param_pattern}')
    os.makedirs(plot_outdir, exist_ok=True)
    print(f"Output directory: {plot_outdir}")
    
    # Convert to array
    arr = np.array(data_points, dtype=float)
    param_vals = arr[:, 0]
    beta_vals = arr[:, 1]
    energy_vals = arr[:, 2]
    cv_vals = arr[:, 3]
    energy_sem = arr[:, 4]
    cv_sem = arr[:, 5]
    
    print(f"{param_pattern} range: [{param_vals.min():.3f}, {param_vals.max():.3f}]")
    print(f"Beta range: [{beta_vals.min():.3f}, {beta_vals.max():.3f}]")
    print(f"Energy per site range: [{energy_vals.min():.6f}, {energy_vals.max():.6f}]")
    print(f"Specific heat per site range: [{cv_vals.min():.6f}, {cv_vals.max():.6f}]")
    print(f"Energy per site SEM range: [{np.nanmin(energy_sem):.2e}, {np.nanmax(energy_sem):.2e}]")
    print(f"Specific heat per site SEM range: [{np.nanmin(cv_sem):.2e}, {np.nanmax(cv_sem):.2e}]")
    
    # Save raw data
    raw_data_file = os.path.join(plot_outdir, f'thermodynamic_data_{param_pattern}.dat')
    header = f'{param_pattern} beta energy_per_site cv_per_site energy_per_site_sem cv_per_site_sem'
    np.savetxt(raw_data_file, arr, header=header)
    print(f"Saved raw data to {raw_data_file}")
    
    # Get unique parameter and beta values
    unique_params = np.unique(param_vals)
    
    # Get beta grid from reference parameter
    ref_param = unique_params[len(unique_params)//2]
    ref_mask = np.isclose(param_vals, ref_param, rtol=1e-8, atol=1e-12)
    target_beta = np.unique(beta_vals[ref_mask])
    target_beta.sort()
    
    # Skip the first beta point (smallest beta) for plotting
    if len(target_beta) > 1:
        target_beta = target_beta[1:]
    
    print(f"Reference {param_pattern}={ref_param}, target beta grid size: {len(target_beta)}")
    
    # Split into negative and positive parameter values
    param_neg = unique_params[unique_params < 0]
    param_pos = unique_params[unique_params >= 0]
    
    print(f"Negative {param_pattern} values: {len(param_neg)}, Non-negative values: {len(param_pos)}")
    
    # Create heatmaps for energy per site
    _create_heatmap(param_vals, beta_vals, energy_vals, 
                   unique_params, target_beta, 
                   'Energy per Site', 'energy_per_site', plot_outdir, param_pattern)
    
    # Create heatmaps for specific heat per site
    _create_heatmap(param_vals, beta_vals, cv_vals, 
                   unique_params, target_beta, 
                   'Specific Heat per Site ($C_v$/N)', 'cv_per_site', plot_outdir, param_pattern)
    
    # Create heatmaps for energy SEM per site (standard error)
    _create_heatmap(param_vals, beta_vals, energy_sem, 
                   unique_params, target_beta, 
                   'Energy per Site SEM', 'energy_per_site_sem', plot_outdir, param_pattern)
    
    # Create heatmaps for specific heat SEM per site (standard error)
    _create_heatmap(param_vals, beta_vals, cv_sem, 
                   unique_params, target_beta, 
                   'Specific Heat per Site SEM', 'cv_per_site_sem', plot_outdir, param_pattern)
    
    # Create heatmaps for log(specific heat)
    # Filter out non-positive values before taking log
    cv_positive_mask = cv_vals > 0
    if np.any(cv_positive_mask):
        log_cv_vals = np.log10(cv_vals[cv_positive_mask])
        _create_heatmap(param_vals[cv_positive_mask], beta_vals[cv_positive_mask], log_cv_vals, 
                       unique_params, target_beta, 
                       'Log Specific Heat (log₁₀ $C_v$)', 'specific_heat_log', plot_outdir, param_pattern)
    
    # Create line plots at fixed beta values
    _create_fixed_beta_plots(param_vals, beta_vals, energy_vals, cv_vals,
                            energy_sem, cv_sem, target_beta, plot_outdir, param_pattern, Jzz=Jzz)
    
    # Create line plots at fixed parameter values  
    _create_fixed_param_plots(param_vals, beta_vals, energy_vals, cv_vals,
                             energy_sem, cv_sem, unique_params, plot_outdir, param_pattern, Jzz=Jzz)
    
    print(f"\n{'='*70}")
    print("Heatmap generation complete!")
    print(f"{'='*70}\n")


def _interpolate_to_grid(param_vals, beta_vals, values, unique_params, target_beta):
    """Interpolate data onto regular grid."""
    
    Z = np.full((len(target_beta), len(unique_params)), np.nan)
    
    for i, p in enumerate(unique_params):
        mask = np.isclose(param_vals, p, rtol=1e-8, atol=1e-12)
        beta_p = beta_vals[mask]
        values_p = values[mask]
        
        if len(beta_p) < 2:
            continue
        
        # Sort by beta
        sort_idx = np.argsort(beta_p)
        beta_p = beta_p[sort_idx]
        values_p = values_p[sort_idx]
        
        # Remove duplicates
        unique_beta, unique_idx = np.unique(beta_p, return_index=True)
        values_unique = values_p[unique_idx]
        
        if len(unique_beta) < 2:
            continue
        
        # Interpolate
        try:
            f = interp1d(unique_beta, values_unique, 
                        bounds_error=False, fill_value=np.nan,
                        kind='linear')
            Z[:, i] = f(target_beta)
        except Exception as e:
            print(f"  Interpolation failed for param={p}: {e}")
    
    return Z


def _create_heatmap(param_vals, beta_vals, values, 
                   unique_params, target_beta, 
                   quantity_name, quantity_label, plot_outdir, param_pattern):
    """Create a heatmap for a given quantity."""
    
    print(f"\nCreating {quantity_name} heatmap...")
    
    # Interpolate to regular grid
    Z = _interpolate_to_grid(param_vals, beta_vals, values, unique_params, target_beta)
    
    # Check for valid data
    valid_count = np.count_nonzero(~np.isnan(Z))
    print(f"  Grid shape: {Z.shape}, valid values: {valid_count}")
    
    if valid_count == 0:
        print(f"  WARNING: No valid data for {quantity_name} heatmap")
        return
    
    # Save grid data
    grid_file = os.path.join(plot_outdir, f'{quantity_label}_grid_{param_pattern}.npz')
    np.savez(grid_file, param=unique_params, beta=target_beta, Z=Z)
    
    # Also save as text
    header_cols = ['beta'] + [f'{param_pattern}={v:g}' for v in unique_params]
    header = ' '.join(header_cols)
    out = np.column_stack((target_beta, Z))
    txt_file = os.path.join(plot_outdir, f'{quantity_label}_grid_{param_pattern}.dat')
    np.savetxt(txt_file, out, header=header)
    
    # Get color limits
    vmin = np.nanmin(Z)
    vmax = np.nanmax(Z)
    
    # Split into negative and positive parameters
    param_neg = unique_params[unique_params < 0]
    param_pos = unique_params[unique_params >= 0]
    
    # Create full heatmap
    P, B = np.meshgrid(unique_params, target_beta)
    
    plt.figure(figsize=(12, 8))
    pcm = plt.pcolormesh(P, B, Z, shading='auto', cmap='viridis')
    plt.yscale('log')
    plt.gca().invert_yaxis()  # Largest beta at bottom, smallest at top
    plt.xlabel(param_pattern)
    plt.ylabel('Inverse Temperature β')
    plt.title(f'{quantity_name} Heatmap')
    plt.colorbar(pcm, label=quantity_name)
    
    fname = os.path.join(plot_outdir, f'{quantity_label}_heatmap_full_{param_pattern}.png')
    plt.savefig(fname, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {fname}")
    
    # Create separate negative and positive heatmaps if both exist
    if len(param_neg) > 1:
        Z_neg = Z[:, unique_params < 0]
        PN, BN = np.meshgrid(param_neg, target_beta)
        
        plt.figure(figsize=(10, 8))
        pcm = plt.pcolormesh(PN, BN, Z_neg, shading='auto', cmap='viridis')
        plt.yscale('log')
        plt.gca().invert_yaxis()  # Largest beta at bottom, smallest at top
        plt.xlabel(param_pattern)
        plt.ylabel('Inverse Temperature β')
        plt.title(f'{quantity_name} Heatmap ({param_pattern} < 0)')
        plt.colorbar(pcm, label=quantity_name)
        
        fname = os.path.join(plot_outdir, f'{quantity_label}_heatmap_neg_{param_pattern}.png')
        plt.savefig(fname, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {fname}")
    
    if len(param_pos) > 1:
        Z_pos = Z[:, unique_params >= 0]
        PP, BP = np.meshgrid(param_pos, target_beta)
        
        plt.figure(figsize=(10, 8))
        pcm = plt.pcolormesh(PP, BP, Z_pos, shading='auto', cmap='viridis')
        plt.yscale('log')
        plt.gca().invert_yaxis()  # Largest beta at bottom, smallest at top
        plt.xlabel(param_pattern)
        plt.ylabel('Inverse Temperature β')
        plt.title(f'{quantity_name} Heatmap ({param_pattern} ≥ 0)')
        plt.colorbar(pcm, label=quantity_name)
        
        fname = os.path.join(plot_outdir, f'{quantity_label}_heatmap_pos_{param_pattern}.png')
        plt.savefig(fname, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {fname}")
    
    # Create side-by-side if both exist
    if len(param_neg) > 1 and len(param_pos) > 1:
        _create_side_by_side_heatmap(param_neg, param_pos, target_beta, Z,
                                     unique_params, quantity_name, quantity_label,
                                     plot_outdir, param_pattern)


def _create_side_by_side_heatmap(param_neg, param_pos, target_beta, Z, 
                                 unique_params, quantity_name, quantity_label,
                                 plot_outdir, param_pattern):
    """Create side-by-side heatmap for negative and positive parameters."""
    
    Z_neg = Z[:, unique_params < 0]
    Z_pos = Z[:, unique_params >= 0]
    
    # Calculate width ratio
    neg_span = float(np.ptp(param_neg)) if len(param_neg) > 0 else 0.0
    pos_span = float(np.ptp(param_pos)) if len(param_pos) > 0 else 0.0
    wr = neg_span / pos_span if pos_span > 0 else 1.0
    wr = float(np.clip(wr, 0.5, 3.0))
    
    fig, (axL, axR) = plt.subplots(
        1, 2, figsize=(14, 8), sharey=True,
        gridspec_kw={'wspace': 0.05, 'hspace': 0.0, 'width_ratios': [wr, 1.0]}
    )
    
    vmin = np.nanmin(Z)
    vmax = np.nanmax(Z)
    
    PN, BN = np.meshgrid(param_neg, target_beta)
    PP, BP = np.meshgrid(param_pos, target_beta)
    
    pcm_neg = axL.pcolormesh(PN, BN, Z_neg, shading='auto', cmap='viridis', vmin=vmin, vmax=vmax)
    pcm_pos = axR.pcolormesh(PP, BP, Z_pos, shading='auto', cmap='viridis', vmin=vmin, vmax=vmax)
    
    # Set log scale and invert y-axis for both plots
    # This puts largest beta at bottom, smallest at top
    axL.set_yscale('log')
    axR.set_yscale('log')
    axL.invert_yaxis()  # Since sharey=True, this inverts both axes
    
    axL.set_xlabel(param_pattern)
    axR.set_xlabel(param_pattern)
    axL.set_ylabel('Inverse Temperature β')
    axR.tick_params(labelleft=False)
    
    # Add colorbar
    cbar = fig.colorbar(pcm_pos, ax=[axL, axR], location='right', pad=0.02)
    cbar.set_label(quantity_name)
    
    fig.suptitle(f'{quantity_name} Heatmap ({param_pattern}<0 | {param_pattern}≥0)')
    
    fname = os.path.join(plot_outdir, f'{quantity_label}_heatmap_side_by_side_{param_pattern}.png')
    fig.savefig(fname, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {fname}")


def _param_to_latex(param_pattern):
    """Convert parameter name to LaTeX format."""
    latex_map = {
        'Jpm': r'J_{\pm}',
        'Jpmpm': r'J_{\pm\pm}',
        'Jzpm': r'J_{z\pm}',
        'J1': r'J_1',
        'J2': r'J_2',
        'J3': r'J_3',
        'h': r'h',
        'Gamma': r'\Gamma',
        'K': r'K',
    }
    return latex_map.get(param_pattern, param_pattern)


def _create_fixed_beta_plots(param_vals, beta_vals, energy_vals, cv_vals,
                            energy_sem, cv_sem, target_beta, plot_outdir, param_pattern, Jzz=1.0):
    """Create individual line plots for selected fixed beta values with error bars."""
    
    print("\nCreating fixed-beta line plots (selected beta values with full param span)...")
    
    param_latex = _param_to_latex(param_pattern)
    
    # Create output subdirectory for individual plots
    individual_plot_dir = os.path.join(plot_outdir, 'fixed_beta_plots')
    os.makedirs(individual_plot_dir, exist_ok=True)
    
    # Select a few beta values that span the range (e.g., 5-7 values)
    n_beta = len(target_beta)
    if n_beta <= 7:
        selected_beta_indices = list(range(n_beta))
    else:
        # Select ~7 evenly spaced indices including first and last
        selected_beta_indices = [0]
        step = (n_beta - 1) / 6  # 6 intervals for 7 points
        for i in range(1, 6):
            selected_beta_indices.append(int(round(i * step)))
        selected_beta_indices.append(n_beta - 1)
        selected_beta_indices = sorted(set(selected_beta_indices))
    
    selected_betas_for_plots = target_beta[selected_beta_indices]
    print(f"  Selected {len(selected_betas_for_plots)} beta values: {selected_betas_for_plots}")
    
    # Create individual plots for selected beta values only
    for beta in selected_betas_for_plots:
        # Find data points at approximately this beta (use larger tolerance for binning)
        mask = np.isclose(beta_vals, beta, rtol=0.05, atol=1e-4)
        
        if not np.any(mask):
            continue
        
        params_at_beta = param_vals[mask]
        energy_at_beta = energy_vals[mask]
        cv_at_beta = cv_vals[mask]
        energy_err = energy_sem[mask]
        cv_err = cv_sem[mask]
        
        # Sort by parameter
        sort_idx = np.argsort(params_at_beta)
        params_sorted = params_at_beta[sort_idx]
        energy_sorted = energy_at_beta[sort_idx]
        cv_sorted = cv_at_beta[sort_idx]
        energy_err_sorted = energy_err[sort_idx]
        cv_err_sorted = cv_err[sort_idx]
        
        # Create individual plot for this beta value
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Energy plot
        ax1.errorbar(params_sorted, energy_sorted, 
                    yerr=energy_err_sorted, fmt='o-', linewidth=1.5, 
                    markersize=4, capsize=3, color='C0', ecolor='C0', alpha=0.8)
        ax1.fill_between(params_sorted, 
                        energy_sorted - energy_err_sorted, 
                        energy_sorted + energy_err_sorted,
                        alpha=0.2, color='C0')
        ax1.set_xlabel(f'${param_latex}$')
        ax1.set_ylabel(r'Energy per Site $E/N$')
        ax1.set_title(f'$\\beta$ = {beta:.2f}')
        ax1.grid(True, alpha=0.3)
        
        # Specific heat plot
        ax2.errorbar(params_sorted, cv_sorted, 
                    yerr=cv_err_sorted, fmt='o-', linewidth=1.5, 
                    markersize=4, capsize=3, color='C1', ecolor='C1', alpha=0.8)
        ax2.fill_between(params_sorted, 
                        cv_sorted - cv_err_sorted, 
                        cv_sorted + cv_err_sorted,
                        alpha=0.2, color='C1')
        ax2.set_xlabel(f'${param_latex}$')
        ax2.set_ylabel(r'Specific Heat per Site $C_v/N$')
        ax2.set_title(f'$\\beta$ = {beta:.2f}')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Use beta value in filename
        beta_str = f'{beta:.4f}'.replace('.', 'p')
        fname = os.path.join(individual_plot_dir, f'thermodynamic_vs_{param_pattern}_beta={beta_str}.png')
        plt.savefig(fname, dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"  Saved {len(selected_betas_for_plots)} individual plots to {individual_plot_dir}/")
    
    # Also create a summary plot with selected beta values
    beta_indices = [0, len(target_beta)//4, len(target_beta)//2, 
                   3*len(target_beta)//4, len(target_beta)-1]
    beta_indices = list(set([i for i in beta_indices if i < len(target_beta)]))
    selected_betas = target_beta[beta_indices]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    for beta in selected_betas:
        mask = np.isclose(beta_vals, beta, rtol=0.05, atol=1e-4)
        
        if not np.any(mask):
            continue
        
        params_at_beta = param_vals[mask]
        energy_at_beta = energy_vals[mask]
        cv_at_beta = cv_vals[mask]
        energy_err = energy_sem[mask]
        cv_err = cv_sem[mask]
        
        # Sort by parameter
        sort_idx = np.argsort(params_at_beta)
        
        label = r'$\beta$=' + f'{beta:.1f}'
        ax1.errorbar(params_at_beta[sort_idx], energy_at_beta[sort_idx], 
                    yerr=energy_err[sort_idx], fmt='-o', markersize=3, 
                    label=label, capsize=2, alpha=0.8)
        ax2.errorbar(params_at_beta[sort_idx], cv_at_beta[sort_idx], 
                    yerr=cv_err[sort_idx], fmt='-o', markersize=3, 
                    label=label, capsize=2, alpha=0.8)
    
    ax1.set_xlabel(f'${param_latex}$')
    ax1.set_ylabel(r'Energy per Site $E/N$')
    ax1.set_title(f'Energy per Site vs ${param_latex}$ at fixed $\\beta$')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.set_xlabel(f'${param_latex}$')
    ax2.set_ylabel(r'Specific Heat per Site $C_v/N$')
    ax2.set_title(f'Specific Heat per Site vs ${param_latex}$ at fixed $\\beta$')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fname = os.path.join(plot_outdir, f'thermodynamic_vs_{param_pattern}_fixed_beta_summary.png')
    plt.savefig(fname, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved summary: {fname}")


def _create_fixed_param_plots(param_vals, beta_vals, energy_vals, cv_vals,
                             energy_sem, cv_sem, unique_params, plot_outdir, param_pattern, Jzz=1.0):
    """Create individual line plots for each fixed parameter value with error bars.
    
    Also plots theoretical prediction lines:
    - 12 * Jpm^3 / Jzz^2  (energy scale)
    - 4 * Jpm^2 / Jzz     (another energy scale)
    """
    
    print("\nCreating fixed-parameter line plots (one per parameter value)...")
    print(f"  Using Jzz = {Jzz} for theoretical predictions")
    
    param_latex = _param_to_latex(param_pattern)
    
    # Create output subdirectory for individual plots
    individual_plot_dir = os.path.join(plot_outdir, f'fixed_{param_pattern}_plots')
    os.makedirs(individual_plot_dir, exist_ok=True)
    
    # Create individual plots for EACH parameter value
    for param in unique_params:
        mask = np.isclose(param_vals, param, rtol=1e-8, atol=1e-12)
        
        if not np.any(mask):
            continue
        
        beta_at_param = beta_vals[mask]
        energy_at_param = energy_vals[mask]
        cv_at_param = cv_vals[mask]
        energy_err = energy_sem[mask]
        cv_err = cv_sem[mask]
        
        # Sort by beta
        sort_idx = np.argsort(beta_at_param)
        beta_sorted = beta_at_param[sort_idx]
        energy_sorted = energy_at_param[sort_idx]
        cv_sorted = cv_at_param[sort_idx]
        energy_err_sorted = energy_err[sort_idx]
        cv_err_sorted = cv_err[sort_idx]
        
        # Create individual plot for this parameter value
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Calculate theoretical beta values (vertical lines at β = 1/energy_scale)
        # Energy scales: 12*|Jpm|^3/Jzz^2 and 4*Jpm^2/Jzz
        # Corresponding beta: β = Jzz^2/(12*|Jpm|^3) and β = Jzz/(4*Jpm^2)
        beta_theory_1 = None
        beta_theory_2 = None
        if param != 0 and Jzz != 0:
            scale1 = 12 * abs(param)**3 / Jzz**2  # 12*|Jpm|^3/Jzz^2
            scale2 = 4 * param**2 / Jzz           # 4*Jpm^2/Jzz
            if scale1 > 0:
                beta_theory_1 = 1.0 / scale1  # = Jzz^2/(12*|Jpm|^3)
            if scale2 > 0:
                beta_theory_2 = 1.0 / scale2  # = Jzz/(4*Jpm^2)
        
        # Energy plot
        ax1.errorbar(beta_sorted, energy_sorted, 
                    yerr=energy_err_sorted, fmt='o-', linewidth=1.5, 
                    markersize=4, capsize=3, color='C0', ecolor='C0', alpha=0.8)
        ax1.fill_between(beta_sorted, 
                        energy_sorted - energy_err_sorted, 
                        energy_sorted + energy_err_sorted,
                        alpha=0.2, color='C0')
        
        # Add vertical lines at theoretical beta values
        if beta_theory_1 is not None:
            ax1.axvline(beta_theory_1, color='C2', linestyle='--', linewidth=1.5, alpha=0.8,
                       label=r'$\beta = J_{zz}^2/(12|J_{\pm}|^3)$')
        if beta_theory_2 is not None:
            ax1.axvline(beta_theory_2, color='C3', linestyle=':', linewidth=1.5, alpha=0.8,
                       label=r'$\beta = J_{zz}/(4J_{\pm}^2)$')
        if beta_theory_1 is not None or beta_theory_2 is not None:
            ax1.legend(loc='best', fontsize=9)
        
        ax1.set_xlabel(r'Inverse Temperature $\beta$')
        ax1.set_ylabel(r'Energy per Site $E/N$')
        ax1.set_xscale('log')
        ax1.set_title(f'${param_latex}$ = {param:.4f}')
        ax1.grid(True, alpha=0.3)
        
        # Specific heat plot
        ax2.errorbar(beta_sorted, cv_sorted, 
                    yerr=cv_err_sorted, fmt='o-', linewidth=1.5, 
                    markersize=4, capsize=3, color='C0', ecolor='C0', alpha=0.8)
        ax2.fill_between(beta_sorted, 
                        cv_sorted - cv_err_sorted, 
                        cv_sorted + cv_err_sorted,
                        alpha=0.2, color='C0')
        
        # Add same vertical lines to specific heat plot
        if beta_theory_1 is not None:
            ax2.axvline(beta_theory_1, color='C2', linestyle='--', linewidth=1.5, alpha=0.8,
                       label=r'$\beta = J_{zz}^2/(12|J_{\pm}|^3)$')
        if beta_theory_2 is not None:
            ax2.axvline(beta_theory_2, color='C3', linestyle=':', linewidth=1.5, alpha=0.8,
                       label=r'$\beta = J_{zz}/(4J_{\pm}^2)$')
        if beta_theory_1 is not None or beta_theory_2 is not None:
            ax2.legend(loc='best', fontsize=9)
        
        ax2.set_xlabel(r'Inverse Temperature $\beta$')
        ax2.set_ylabel(r'Specific Heat per Site $C_v/N$')
        ax2.set_xscale('log')
        ax2.set_title(f'${param_latex}$ = {param:.4f}')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Use parameter value in filename (handle negative values)
        param_str = f'{param:.4f}'.replace('-', 'neg')
        fname = os.path.join(individual_plot_dir, f'thermodynamic_vs_beta_{param_pattern}={param_str}.png')
        plt.savefig(fname, dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"  Saved {len(unique_params)} individual plots to {individual_plot_dir}/")
    
    # Also create a summary plot with selected parameter values
    n_params = len(unique_params)
    param_indices = [0, n_params//4, n_params//2, 3*n_params//4, n_params-1]
    param_indices = list(set([i for i in param_indices if i < n_params]))
    selected_params = unique_params[param_indices]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    for param in selected_params:
        mask = np.isclose(param_vals, param, rtol=1e-8, atol=1e-12)
        
        if not np.any(mask):
            continue
        
        beta_at_param = beta_vals[mask]
        energy_at_param = energy_vals[mask]
        cv_at_param = cv_vals[mask]
        energy_err = energy_sem[mask]
        cv_err = cv_sem[mask]
        
        # Sort by beta
        sort_idx = np.argsort(beta_at_param)
        
        label = f'${param_latex}$={param:.3f}'
        ax1.errorbar(beta_at_param[sort_idx], energy_at_param[sort_idx], 
                    yerr=energy_err[sort_idx], fmt='-', linewidth=1.5, 
                    label=label, capsize=2, alpha=0.8)
        ax2.errorbar(beta_at_param[sort_idx], cv_at_param[sort_idx], 
                    yerr=cv_err[sort_idx], fmt='-', linewidth=1.5, 
                    label=label, capsize=2, alpha=0.8)
    
    ax1.set_xlabel(r'Inverse Temperature $\beta$')
    ax1.set_ylabel(r'Energy per Site $E/N$')
    ax1.set_xscale('log')
    ax1.set_title(f'Energy per Site vs $\\beta$ at fixed ${param_latex}$')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.set_xlabel(r'Inverse Temperature $\beta$')
    ax2.set_ylabel(r'Specific Heat per Site $C_v/N$')
    ax2.set_xscale('log')
    ax2.set_title(f'Specific Heat per Site vs $\\beta$ at fixed ${param_latex}$')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fname = os.path.join(plot_outdir, f'thermodynamic_vs_beta_fixed_{param_pattern}_summary.png')
    plt.savefig(fname, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved summary: {fname}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Calculate thermodynamic heatmaps from SS_rand*.dat files')
    parser.add_argument('directory', type=str,
                       help='Directory containing parameter sweep folders')
    parser.add_argument('--param', type=str, default='Jpm',
                       help='Parameter name pattern (default: Jpm)')
    parser.add_argument('--sites', type=int, default=None,
                       help='Number of sites for per-site normalization (default: auto-detect from directory name)')
    parser.add_argument('--Jzz', type=float, default=1.0,
                       help='Value of Jzz for theoretical prediction lines (default: 1.0)')
    
    args = parser.parse_args()
    
    # Try to auto-detect number of sites from directory name if not specified
    n_sites = args.sites
    if n_sites is None:
        # Try to extract from directory name like "2x3" or "3x4"
        dir_name = os.path.basename(os.path.normpath(args.directory))
        size_match = re.search(r'(\d+)x(\d+)', dir_name)
        if size_match:
            n_sites = int(size_match.group(1)) * int(size_match.group(2))
            print(f"Auto-detected {n_sites} sites from directory name")
        else:
            n_sites = 1  # Default: no normalization
            print("Warning: Could not auto-detect number of sites, using 1 (no normalization)")
    
    # Load thermodynamic data
    data_points = load_thermodynamic_data(args.directory, args.param, n_sites=n_sites)
    
    if not data_points:
        print("ERROR: No thermodynamic data found!")
        exit(1)
    
    # Create heatmaps
    create_thermodynamic_heatmaps(data_points, args.directory, args.param, n_sites=n_sites, Jzz=args.Jzz)
    
    print("\n" + "="*70)
    print("Processing complete!")
    print("="*70)
