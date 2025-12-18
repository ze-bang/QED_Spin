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
from collections import defaultdict
from scipy.interpolate import interp1d

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


def compute_specific_heat(beta, variance):
    """
    Compute specific heat from variance.
    C_v = beta^2 * variance = beta^2 * (<E^2> - <E>^2)
    """
    return beta**2 * variance


def load_thermodynamic_data(data_dir, param_pattern='Jpm'):
    """
    Load thermodynamic data from SS_rand*.dat files across parameter sweep.
    
    Parameters:
    data_dir: Root directory containing parameter subdirectories
    param_pattern: Parameter name pattern (e.g., 'Jpm', 'h', 'J')
    
    Returns:
    data_points: list of (param_value, beta, energy, specific_heat)
    """
    
    print(f"\n{'='*70}")
    print(f"Loading thermodynamic data for {param_pattern} sweep")
    print(f"Data directory: {data_dir}")
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
        
        # Find SS_rand*.dat files in output subdirectory
        output_dir = os.path.join(subdir, 'output')
        if not os.path.isdir(output_dir):
            # Try looking directly in the subdir
            output_dir = subdir
        
        ss_files = glob.glob(os.path.join(output_dir, 'SS_rand*.dat'))
        
        if not ss_files:
            print(f"  No SS_rand*.dat files found in {output_dir}")
            continue
        
        print(f"  {param_pattern}={param_val}: Found {len(ss_files)} SS files")
        
        # Collect data from all random samples
        all_beta = []
        all_energy = []
        all_cv = []
        
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
            
            all_beta.append(beta)
            all_energy.append(energy)
            all_cv.append(cv)
        
        if not all_beta:
            continue
        
        # Average over samples if multiple files exist
        # For now, use the first file's data (they should be on the same beta grid)
        beta_combined = all_beta[0]
        energy_combined = all_energy[0]
        cv_combined = all_cv[0]
        
        # If multiple samples, average them
        if len(all_beta) > 1:
            # Interpolate all samples onto the first sample's beta grid
            for i in range(1, len(all_beta)):
                if len(all_beta[i]) > 1:
                    # Interpolate
                    f_energy = interp1d(all_beta[i], all_energy[i], 
                                       bounds_error=False, fill_value=np.nan)
                    f_cv = interp1d(all_beta[i], all_cv[i], 
                                   bounds_error=False, fill_value=np.nan)
                    energy_combined = np.nanmean([energy_combined, f_energy(beta_combined)], axis=0)
                    cv_combined = np.nanmean([cv_combined, f_cv(beta_combined)], axis=0)
        
        # Store data points
        for j in range(len(beta_combined)):
            all_data.append((param_val, beta_combined[j], energy_combined[j], cv_combined[j]))
    
    print(f"\nLoaded {len(all_data)} total data points")
    return all_data


def create_thermodynamic_heatmaps(data_points, data_dir, param_pattern='Jpm'):
    """
    Create heatmaps of energy and specific heat vs parameter and beta.
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
    
    print(f"{param_pattern} range: [{param_vals.min():.3f}, {param_vals.max():.3f}]")
    print(f"Beta range: [{beta_vals.min():.3f}, {beta_vals.max():.3f}]")
    print(f"Energy range: [{energy_vals.min():.3f}, {energy_vals.max():.3f}]")
    print(f"Specific heat range: [{cv_vals.min():.6f}, {cv_vals.max():.6f}]")
    
    # Save raw data
    raw_data_file = os.path.join(plot_outdir, f'thermodynamic_data_{param_pattern}.dat')
    header = f'{param_pattern} beta energy specific_heat'
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
    
    # Create heatmaps for energy
    _create_heatmap(param_vals, beta_vals, energy_vals, 
                   unique_params, target_beta, 
                   'Energy', 'energy', plot_outdir, param_pattern)
    
    # Create heatmaps for specific heat
    _create_heatmap(param_vals, beta_vals, cv_vals, 
                   unique_params, target_beta, 
                   'Specific Heat ($C_v$)', 'specific_heat', plot_outdir, param_pattern)
    
    # Create line plots at fixed beta values
    _create_fixed_beta_plots(param_vals, beta_vals, energy_vals, cv_vals,
                            target_beta, plot_outdir, param_pattern)
    
    # Create line plots at fixed parameter values  
    _create_fixed_param_plots(param_vals, beta_vals, energy_vals, cv_vals,
                             unique_params, plot_outdir, param_pattern)
    
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


def _create_fixed_beta_plots(param_vals, beta_vals, energy_vals, cv_vals,
                            target_beta, plot_outdir, param_pattern):
    """Create line plots at fixed beta values."""
    
    print("\nCreating fixed-beta line plots...")
    
    # Select a few representative beta values
    beta_indices = [0, len(target_beta)//4, len(target_beta)//2, 
                   3*len(target_beta)//4, len(target_beta)-1]
    beta_indices = [i for i in beta_indices if i < len(target_beta)]
    selected_betas = target_beta[beta_indices]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    for beta in selected_betas:
        # Find data points close to this beta
        beta_tol = 0.1 * beta if beta > 0 else 0.1
        mask = np.abs(beta_vals - beta) < beta_tol
        
        if not np.any(mask):
            continue
        
        params_at_beta = param_vals[mask]
        energy_at_beta = energy_vals[mask]
        cv_at_beta = cv_vals[mask]
        
        # Sort by parameter
        sort_idx = np.argsort(params_at_beta)
        
        label = f'β={beta:.1f}'
        ax1.plot(params_at_beta[sort_idx], energy_at_beta[sort_idx], '-o', 
                markersize=3, label=label)
        ax2.plot(params_at_beta[sort_idx], cv_at_beta[sort_idx], '-o', 
                markersize=3, label=label)
    
    ax1.set_xlabel(param_pattern)
    ax1.set_ylabel('Energy')
    ax1.set_title(f'Energy vs {param_pattern} at fixed β')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    ax2.set_xlabel(param_pattern)
    ax2.set_ylabel('Specific Heat $C_v$')
    ax2.set_title(f'Specific Heat vs {param_pattern} at fixed β')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fname = os.path.join(plot_outdir, f'thermodynamic_vs_{param_pattern}_fixed_beta.png')
    plt.savefig(fname, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {fname}")


def _create_fixed_param_plots(param_vals, beta_vals, energy_vals, cv_vals,
                             unique_params, plot_outdir, param_pattern):
    """Create line plots at fixed parameter values."""
    
    print("\nCreating fixed-parameter line plots...")
    
    # Select a few representative parameter values
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
        
        # Sort by beta
        sort_idx = np.argsort(beta_at_param)
        
        label = f'{param_pattern}={param:.3f}'
        ax1.plot(beta_at_param[sort_idx], energy_at_param[sort_idx], '-', 
                linewidth=1, label=label)
        ax2.plot(beta_at_param[sort_idx], cv_at_param[sort_idx], '-', 
                linewidth=1, label=label)
    
    ax1.set_xlabel('Inverse Temperature β')
    ax1.set_ylabel('Energy')
    ax1.set_xscale('log')
    ax1.set_title(f'Energy vs β at fixed {param_pattern}')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    ax2.set_xlabel('Inverse Temperature β')
    ax2.set_ylabel('Specific Heat $C_v$')
    ax2.set_xscale('log')
    ax2.set_title(f'Specific Heat vs β at fixed {param_pattern}')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fname = os.path.join(plot_outdir, f'thermodynamic_vs_beta_fixed_{param_pattern}.png')
    plt.savefig(fname, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {fname}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Calculate thermodynamic heatmaps from SS_rand*.dat files')
    parser.add_argument('directory', type=str,
                       help='Directory containing parameter sweep folders')
    parser.add_argument('--param', type=str, default='Jpm',
                       help='Parameter name pattern (default: Jpm)')
    
    args = parser.parse_args()
    
    # Load thermodynamic data
    data_points = load_thermodynamic_data(args.directory, args.param)
    
    if not data_points:
        print("ERROR: No thermodynamic data found!")
        exit(1)
    
    # Create heatmaps
    create_thermodynamic_heatmaps(data_points, args.directory, args.param)
    
    print("\n" + "="*70)
    print("Processing complete!")
    print("="*70)
