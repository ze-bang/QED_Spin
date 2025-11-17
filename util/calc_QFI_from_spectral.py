"""
Calculate Quantum Fisher Information (QFI) from pre-computed spectral function files.

This script works with frequency-domain spectral data that has already been computed,
rather than performing FFT from time-domain correlation data.

Expected file format: {species}_spectral_sample_{N}_beta_{beta}.txt
Example: SxSx_q_Qx0_Qy0_Qz0_spectral_sample_0_beta_inf.txt

File structure:
# Header lines (ignored)
frequency  spectral_function  error
...
"""

import os
import sys
import re
import glob
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from scipy.signal import find_peaks, peak_prominences

# Try to import mpi4py, but make it optional
try:
    from mpi4py import MPI
    HAS_MPI = True
except ImportError:
    HAS_MPI = False

# NumPy compatibility: use trapezoid (new) or trapz (old)
if hasattr(np, 'trapezoid'):
    np_trapz = np.trapezoid
else:
    np_trapz = np.trapz


def parse_spectral_filename(filename):
    """
    Extract species (including momentum), beta, and sample index from spectral filenames.
    
    Expected format: {species}_spectral_sample_{N}_beta_{beta}.txt
    Example: SxSx_q_Qx0_Qy0_Qz0_spectral_sample_0_beta_inf.txt
    
    Returns: (species_with_momentum, beta(float or np.inf), sample_idx) or (None, None, None)
    """
    basename = os.path.basename(filename)
    # Match pattern: anything_spectral_sample{N}_beta_{beta}.txt
    m = re.match(r'^(.+?)_spectral_sample_(\d+)_beta_([0-9.+-eE]+|inf|infty)\.txt$', 
                 basename, re.IGNORECASE)
    if not m:
        return None, None, None
    
    species_with_momentum = m.group(1)
    sample_idx = int(m.group(2))
    beta_token = m.group(3)
    
    if beta_token.lower() in ("inf", "infty"):
        beta_val = np.inf
    else:
        try:
            beta_val = float(beta_token)
        except ValueError:
            return None, None, None
    
    return species_with_momentum, beta_val, sample_idx


def load_spectral_file(filepath):
    """
    Load spectral function data from file.
    
    Returns: (omega_array, spectral_function_array)
    """
    try:
        # Load data, skipping comment lines
        data = np.loadtxt(filepath, comments='#')
        
        if data.ndim == 1:
            # Single row - shouldn't happen but handle it
            omega = np.array([data[0]])
            spectral = np.array([data[1]])
        else:
            omega = data[:, 0]
            spectral = data[:, 1]
        
        return omega, spectral
    
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None, None


def find_spectral_peaks(omega, spectral_function, min_prominence=0.1, 
                       min_height=None, omega_range=(0, 6)):
    """
    Find peak positions in spectral function data.
    
    Parameters:
    omega: Frequency array
    spectral_function: Spectral function values
    min_prominence: Minimum prominence for peak detection
    min_height: Minimum height for peak detection (if None, uses 10% of max)
    omega_range: Frequency range to search for peaks (min_freq, max_freq)
    
    Returns:
    peak_positions: List of peak frequencies
    peak_heights: List of peak heights
    peak_prominences: List of peak prominences
    """
    # Filter data to specified frequency range
    mask = (omega >= omega_range[0]) & (omega <= omega_range[1])
    omega_filtered = omega[mask]
    spec_filtered = spectral_function[mask]
    
    if len(spec_filtered) == 0:
        return [], [], []
    
    # Set minimum height if not provided
    if min_height is None:
        min_height = 0.1 * np.max(spec_filtered)
    
    # Find peaks
    peaks, properties = find_peaks(spec_filtered, height=min_height, prominence=min_prominence)
    
    if len(peaks) == 0:
        return [], [], []
    
    # Get peak positions in original frequency units
    peak_positions = omega_filtered[peaks]
    peak_heights = spec_filtered[peaks]
    
    # Calculate prominences
    prominences, _, _ = peak_prominences(spec_filtered, peaks)
    
    return peak_positions.tolist(), peak_heights.tolist(), prominences.tolist()


def calculate_qfi_from_spectral(omega, spectral_function, beta):
    """
    Calculate quantum Fisher information from spectral function.
    
    QFI = 4 * ∫ S(ω) * [coth(βω/2) - (βω/2)/sinh²(βω/2)] dω
    
    For β → ∞: QFI = 4 * ∫ S(ω) dω (only positive frequencies)
    """
    # Extract positive frequencies
    positive_mask = omega > 0
    omega_pos = omega[positive_mask]
    s_omega_pos = spectral_function[positive_mask]
    
    if len(omega_pos) == 0:
        print("Warning: No positive frequency data found!")
        return 0.0
    
    if np.isinf(beta):
        # For infinite beta, integrand is just S(ω)
        integrand = s_omega_pos
    else:
        # Finite beta case
        x = beta * omega_pos / 2.0
        # Avoid numerical issues for small x
        with np.errstate(divide='ignore', invalid='ignore'):
            coth_term = 1.0 / np.tanh(x)
            sinh_term = 1.0 / np.sinh(x)**2
            integrand = s_omega_pos * (coth_term - x * sinh_term)
        
        # Handle any NaN or inf values
        integrand = np.nan_to_num(integrand, nan=0.0, posinf=0.0, neginf=0.0)
    
    qfi = 4.0 * np_trapz(integrand, omega_pos)
    return qfi


def parse_QFI_data_from_spectral(structure_factor_dir, beta_tol=1e-2):
    """
    Parse QFI data from pre-computed spectral function files.
    
    Directory structure:
    - structure_factor_results/
      - beta_{value}/
        - {operator_type}/  (e.g., 'sum', 'transverse', 'sublattice', etc.)
          - {species}_spectral_sample{N}_beta_{beta}.txt
    
    Parameters:
    structure_factor_dir: Directory containing beta_* subdirectories
    beta_tol: Tolerance for grouping beta values
    """
    
    # Initialize data structures
    species_data = defaultdict(lambda: defaultdict(list))
    species_names = set()
    beta_bins = []
    beta_bin_values = defaultdict(list)
    
    # Step 1: Discover and organize data files
    print(f"Scanning directory: {structure_factor_dir}")
    _collect_spectral_files(structure_factor_dir, species_data, species_names, 
                           beta_bins, beta_bin_values, beta_tol)
    
    print(f"\nFound species (with momentum): {sorted(species_names)}")
    
    # Step 2: Process each species
    all_species_qfi_data = defaultdict(list)
    
    for species, beta_groups in species_data.items():
        print(f"\nProcessing species: {species}")
        _process_species_spectral(species, beta_groups, beta_bin_values, 
                                  structure_factor_dir, all_species_qfi_data)
    
    # Step 3: Generate summary plots
    _create_summary_plots(all_species_qfi_data, structure_factor_dir)
    
    print("\nProcessing complete!")
    return all_species_qfi_data


def _collect_spectral_files(structure_factor_dir, species_data, species_names, 
                            beta_bins, beta_bin_values, beta_tol):
    """Collect and organize all spectral data files by species and beta values."""
    
    beta_dirs = glob.glob(os.path.join(structure_factor_dir, 'beta_*'))
    print(f"Found {len(beta_dirs)} beta directories")
    
    for beta_dir in beta_dirs:
        beta_val = _extract_beta_from_dirname(beta_dir)
        if beta_val is None:
            continue
        
        # Scan all subdirectories (operator types) within this beta directory
        operator_subdirs = [d for d in glob.glob(os.path.join(beta_dir, '*')) 
                          if os.path.isdir(d)]
        
        if not operator_subdirs:
            print(f"Warning: No operator subdirectories in {beta_dir}")
            continue
        
        for op_subdir in operator_subdirs:
            op_name = os.path.basename(op_subdir)
            spectral_files = glob.glob(os.path.join(op_subdir, '*_spectral_sample*_beta_*.txt'))
            
            print(f"  {op_name}: found {len(spectral_files)} spectral files")
            
            for fpath in spectral_files:
                species, file_beta, sample_idx = parse_spectral_filename(fpath)
                if species is None:
                    continue
                
                # Use beta from filename (more reliable than directory name)
                beta_bin_idx = _assign_beta_bin(file_beta, beta_bins, beta_tol)
                beta_bin_values[beta_bin_idx].append(file_beta)
                
                species_data[species][beta_bin_idx].append(fpath)
                species_names.add(species)


def _extract_beta_from_dirname(beta_dir):
    """Extract beta value from directory name."""
    beta_match = re.search(r'beta_([\d\.]+|inf|infty)', 
                          os.path.basename(beta_dir), re.IGNORECASE)
    if not beta_match:
        return None
        
    beta_token = beta_match.group(1)
    if beta_token.lower() in ("inf", "infty"):
        return np.inf
    
    try:
        return float(beta_token)
    except ValueError:
        return None


def _assign_beta_bin(beta_val, bins, tol):
    """Assign beta value to tolerance-based bin."""
    # Group infinities together
    if np.isinf(beta_val):
        for i, c in enumerate(bins):
            if np.isinf(c):
                return i
        bins.append(beta_val)
        return len(bins) - 1
        
    # Check existing bins
    for i, c in enumerate(bins):
        if not np.isinf(c) and abs(beta_val - c) < tol:
            return i
            
    # Create new bin
    bins.append(beta_val)
    return len(bins) - 1


def _process_species_spectral(species, beta_groups, beta_bin_values, 
                              structure_factor_dir, all_species_qfi_data):
    """Process all spectral data for a single species across different beta values."""
    
    for beta_bin_idx, file_list in beta_groups.items():
        beta = _get_bin_beta(beta_bin_values[beta_bin_idx])
        beta_label = 'inf' if np.isinf(beta) else f'{beta:.6g}'
        
        print(f"  Beta bin {beta_bin_idx} (β≈{beta_label}): {len(file_list)} files")
        
        # Load and average spectral data from all samples
        mean_omega, mean_spectral = _load_and_average_spectral(file_list)
        
        if mean_omega is None:
            print(f"    Failed to load data")
            continue
        
        # Calculate QFI from averaged spectral function
        qfi = calculate_qfi_from_spectral(mean_omega, mean_spectral, beta)
        
        # Find peaks in the spectral function
        peak_positions, peak_heights, peak_prominences = find_spectral_peaks(
            mean_omega, mean_spectral, min_prominence=0.1, omega_range=(0, 6))
        
        # Save results
        results = {
            'omega': mean_omega,
            'spectral_function': mean_spectral,
            'qfi': qfi,
            'peak_positions': peak_positions,
            'peak_heights': peak_heights,
            'peak_prominences': peak_prominences
        }
        
        _save_species_results(species, beta, results, structure_factor_dir)
        
        # Store for summary plots
        all_species_qfi_data[species].append((beta, qfi))
        
        print(f"    QFI = {qfi:.4f}, Peaks at: {peak_positions}")


def _get_bin_beta(beta_vals):
    """Get representative beta value for a bin."""
    if len(beta_vals) == 0:
        return 0.0
    elif any(np.isinf(v) for v in beta_vals):
        return np.inf
    else:
        return np.mean(beta_vals)


def _load_and_average_spectral(file_list):
    """Load and average spectral data from multiple files."""
    
    all_omega = []
    all_spectral = []
    
    for fpath in file_list:
        omega, spectral = load_spectral_file(fpath)
        
        if omega is None or spectral is None:
            print(f"    Warning: Failed to load {fpath}")
            continue
        
        all_omega.append(omega)
        all_spectral.append(spectral)
    
    if not all_omega:
        return None, None
    
    # Check if all omega arrays are identical
    ref_omega = all_omega[0]
    all_match = all(np.allclose(omega, ref_omega) for omega in all_omega)
    
    if all_match:
        # Simple average if all omega arrays match
        mean_omega = ref_omega
        mean_spectral = np.mean(all_spectral, axis=0)
    else:
        # Need to interpolate to common grid if they don't match
        print("    Warning: Omega arrays don't match across samples, using first sample's grid")
        mean_omega = ref_omega
        # For now, just use the first sample - could implement interpolation if needed
        mean_spectral = all_spectral[0]
    
    return mean_omega, mean_spectral


def _save_species_results(species, beta, results, structure_factor_dir):
    """Save computed results for a species at given beta."""
    
    # Create output directory
    outdir = os.path.join(structure_factor_dir, 'processed_data', species)
    os.makedirs(outdir, exist_ok=True)
    
    beta_label = 'inf' if np.isinf(beta) else f'{beta:.6g}'
    
    # Save spectral data
    data_out = np.column_stack((results['omega'], results['spectral_function']))
    data_filename = os.path.join(outdir, f'spectral_averaged_beta_{beta_label}.dat')
    np.savetxt(data_filename, data_out, header='frequency spectral_function')
    
    # Save QFI value
    qfi_filename = os.path.join(outdir, f'qfi_beta_{beta_label}.txt')
    with open(qfi_filename, 'w') as f:
        f.write(f"# QFI for {species} at beta={beta_label}\n")
        f.write(f"{results['qfi']:.10e}\n")
    
    # Save peak information
    if results['peak_positions']:
        peaks_out = np.column_stack((
            results['peak_positions'],
            results['peak_heights'],
            results['peak_prominences']
        ))
        peaks_filename = os.path.join(outdir, f'peaks_beta_{beta_label}.dat')
        np.savetxt(peaks_filename, peaks_out, 
                  header='frequency height prominence')
    
    # Create spectral function plot
    _plot_spectral_function(species, beta, results, outdir)


def _plot_spectral_function(species, beta, results, outdir):
    """Plot spectral function with peaks marked."""
    
    plt.figure(figsize=(10, 6))
    
    beta_label = 'inf' if np.isinf(beta) else f'{beta:.6g}'
    
    # Plot spectral function
    plt.scatter(results['omega'], results['spectral_function'], 
                label=f'Spectral function (Beta≈{beta_label}, QFI={results["qfi"]:.4f})', 
                alpha=0.7, s=20)
    
    # Mark peaks
    if results['peak_positions']:
        plt.scatter(results['peak_positions'], results['peak_heights'],
                   color='red', marker='x', s=100, linewidths=2,
                   label=f'Peaks (N={len(results["peak_positions"])})')
    
    plt.xlabel('Frequency ω')
    plt.ylabel('Spectral Function S(ω)')
    plt.xlim(-3, 6)
    plt.title(f'Spectral Function for {species} at Beta≈{beta_label}')
    plt.grid(True)
    plt.legend()
    
    plot_filename = os.path.join(outdir, f'spectral_function_{species}_beta_{beta_label}.png')
    plt.savefig(plot_filename, dpi=300)
    plt.close()


def _create_summary_plots(all_species_qfi_data, structure_factor_dir):
    """Create summary plots for QFI vs beta for all species."""
    
    plot_outdir = os.path.join(structure_factor_dir, 'plots')
    os.makedirs(plot_outdir, exist_ok=True)
    
    for species, qfi_data in all_species_qfi_data.items():
        _plot_qfi_vs_beta(species, qfi_data, plot_outdir)


def _plot_qfi_vs_beta(species, qfi_data, plot_outdir):
    """Plot QFI vs beta for a species."""
    
    # Sort data
    qfi_data.sort(key=lambda x: (np.inf if np.isinf(x[0]) else x[0]))
    qfi_beta_array = np.array(qfi_data, dtype=float)
    
    # Save data
    data_filename = os.path.join(plot_outdir, f'qfi_vs_beta_{species}.dat')
    np.savetxt(data_filename, qfi_beta_array, header='beta qfi')
    
    # Create plot
    plt.figure(figsize=(10, 6))
    
    finite_mask = np.isfinite(qfi_beta_array[:, 0])
    if np.any(finite_mask):
        plt.plot(qfi_beta_array[finite_mask, 0], qfi_beta_array[finite_mask, 1], 
                'o-', label='Finite β')
    
    # Handle β=∞ points
    if np.any(~finite_mask):
        _add_infinity_annotation(qfi_beta_array, finite_mask)
    
    plt.xlabel('Beta (β)')
    plt.ylabel('QFI')
    plt.title(f'QFI vs. Beta for {species}')
    plt.xscale('log')
    plt.grid(True)
    plt.legend()
    
    plot_filename = os.path.join(plot_outdir, f'qfi_vs_beta_{species}.png')
    plt.savefig(plot_filename, dpi=300)
    plt.close()


def _add_infinity_annotation(qfi_beta_array, finite_mask):
    """Add β=∞ annotation to plot."""
    
    beta_inf_qfi = qfi_beta_array[~finite_mask, 1]
    
    # Position annotation
    if np.any(finite_mask):
        x_annot = qfi_beta_array[finite_mask, 0].max() * 1.5
    else:
        x_annot = 1.0
    
    plt.scatter([x_annot], [beta_inf_qfi.mean()], 
                marker='*', s=160, c='red', label='β→∞')
    plt.text(x_annot, beta_inf_qfi.mean(), 
            f' β→∞ QFI={beta_inf_qfi.mean():.4g}',
            fontsize=9, ha='left', va='bottom')


def parse_QFI_across_parameter(data_dir, param_pattern='Jpm'):
    """
    Scan subdirectories with parameter sweep and compute QFI for each.
    
    Parameters:
    data_dir: Root directory containing parameter subdirectories
    param_pattern: Parameter name pattern (e.g., 'Jpm', 'h', 'J')
    """
    
    if HAS_MPI:
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()
    else:
        rank = 0
        size = 1
        comm = None
    
    # Find all subdirectories matching the pattern
    subdirs = []
    if rank == 0:
        subdirs = sorted(glob.glob(os.path.join(data_dir, f'{param_pattern}=*')))
        print(f"Found {len(subdirs)} {param_pattern} subdirectories")
    
    # Broadcast subdirs to all processes (only if MPI is available)
    if HAS_MPI:
        subdirs = comm.bcast(subdirs, root=0)
    
    # Distribute subdirectories among processes
    local_subdirs = []
    for i, subdir in enumerate(subdirs):
        if i % size == rank:
            local_subdirs.append(subdir)
    
    # Each process handles its assigned subdirectories
    local_param_qfi_data = {}
    param_regex = re.compile(f'{param_pattern}=([-]?[\d\.]+)')
    
    for subdir in local_subdirs:
        param_match = param_regex.search(os.path.basename(subdir))
        if not param_match:
            continue
        
        param_value = float(param_match.group(1))
        print(f"[Rank {rank}] Processing {param_pattern}={param_value}")
        
        sf_dir = os.path.join(subdir, 'structure_factor_results')
        if not os.path.exists(sf_dir):
            print(f"  Warning: No structure_factor_results directory")
            continue
        
        qfi_data = parse_QFI_data_from_spectral(sf_dir)
        
        if qfi_data:
            local_param_qfi_data[param_value] = qfi_data
    
    # Gather all results at rank 0 (only if MPI is available)
    if HAS_MPI:
        gathered = comm.gather(local_param_qfi_data, root=0)
    else:
        gathered = [local_param_qfi_data]
    
    if rank == 0:
        # Merge all results
        merged_data = {}
        for local_data in gathered:
            merged_data.update(local_data)
        
        # Create summary plots
        _plot_parameter_sweep_summary(merged_data, data_dir, param_pattern)
        
        return merged_data
    else:
        return None


def _plot_parameter_sweep_summary(merged_data, data_dir, param_pattern):
    """Create summary plots for parameter sweep."""
    
    plot_outdir = os.path.join(data_dir, f'plots_{param_pattern}')
    os.makedirs(plot_outdir, exist_ok=True)
    
    # Organize data by species
    species_data = defaultdict(list)
    
    for param_value, qfi_dict in merged_data.items():
        for species, qfi_list in qfi_dict.items():
            for beta, qfi in qfi_list:
                species_data[species].append((param_value, beta, qfi))
    
    # Plot heatmaps for each species
    for species, data_points in species_data.items():
        _plot_parameter_beta_heatmap(species, data_points, plot_outdir, param_pattern)


def _plot_parameter_beta_heatmap(species, data_points, plot_outdir, param_pattern):
    """Create heatmap of QFI vs parameter and beta."""
    
    # Convert to array
    arr = np.array(data_points, dtype=float)
    param_vals = arr[:, 0]
    beta_vals = arr[:, 1]
    qfi_vals = arr[:, 2]
    
    # Create grid for heatmap
    param_unique = np.unique(param_vals)
    beta_unique = np.unique(beta_vals[np.isfinite(beta_vals)])
    
    if len(param_unique) < 2 or len(beta_unique) < 2:
        print(f"Not enough data for heatmap: {species}")
        return
    
    # Create meshgrid
    param_grid, beta_grid = np.meshgrid(param_unique, beta_unique)
    qfi_grid = np.zeros_like(param_grid)
    
    # Fill grid
    for i, beta in enumerate(beta_unique):
        for j, param in enumerate(param_unique):
            mask = (np.abs(beta_vals - beta) < 1e-6) & (np.abs(param_vals - param) < 1e-6)
            if np.any(mask):
                qfi_grid[i, j] = np.mean(qfi_vals[mask])
            else:
                qfi_grid[i, j] = np.nan
    
    # Create plot
    plt.figure(figsize=(12, 8))
    plt.pcolormesh(param_grid, beta_grid, qfi_grid, shading='auto', cmap='viridis')
    plt.colorbar(label='QFI')
    plt.xlabel(f'{param_pattern}')
    plt.ylabel('Beta (β)')
    plt.title(f'QFI Heatmap: {species}')
    plt.yscale('log')
    
    plot_filename = os.path.join(plot_outdir, f'qfi_heatmap_{species}.png')
    plt.savefig(plot_filename, dpi=300)
    plt.close()
    
    # Save data
    data_filename = os.path.join(plot_outdir, f'qfi_heatmap_{species}.dat')
    np.savetxt(data_filename, arr, header=f'{param_pattern} beta qfi')


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Calculate QFI from pre-computed spectral function files')
    parser.add_argument('directory', type=str,
                       help='Directory containing structure_factor_results')
    parser.add_argument('--beta-tol', type=float, default=1e-2,
                       help='Tolerance for grouping beta values (default: 1e-2)')
    parser.add_argument('--param-sweep', type=str, default='Jpm',
                       help='Parameter name for sweep analysis (e.g., Jpm, h)')
    
    args = parser.parse_args()
    
    if args.param_sweep:
        # Parameter sweep mode
        print(f"Running parameter sweep analysis for {args.param_sweep}")
        results = parse_QFI_across_parameter(args.directory, args.param_sweep)
    else:
        # Single directory mode
        sf_dir = os.path.join(args.directory, 'structure_factor_results') \
                if not args.directory.endswith('structure_factor_results') \
                else args.directory
        
        if not os.path.exists(sf_dir):
            print(f"Error: Directory not found: {sf_dir}")
            sys.exit(1)
        
        print(f"Processing spectral data from: {sf_dir}")
        results = parse_QFI_data_from_spectral(sf_dir, beta_tol=args.beta_tol)
    
    print("\n" + "="*70)
    print("Processing complete!")
    print("="*70)
