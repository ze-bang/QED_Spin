import os
import glob
import numpy as np
from collections import defaultdict
import re
import sys

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.interpolate import griddata
from scipy.signal import find_peaks, peak_prominences
from collections import defaultdict
from scipy.interpolate import interp1d

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

# Function to apply broadening in time domain
def apply_time_broadening(t_values, data, broadening_type='gaussian', sigma=None, gamma=None):
    """
    Apply broadening to time correlation data before FFT
    
    Parameters:
    t_values: Time array
    data: Complex correlation data
    broadening_type: 'gaussian' or 'lorentzian'
    sigma: Width parameter for Gaussian broadening
    gamma: Width parameter for Lorentzian broadening
    
    Returns:
    broadened_data: Data with broadening applied
    """
    if broadening_type == 'gaussian' and sigma is not None:
        # Apply Gaussian broadening: multiply by exp(-t²/(2σ²))
        broadening_factor = np.exp(-t_values**2 / (2 * sigma**2))
        broadened_data = data * broadening_factor
        print(f"Applied Gaussian broadening in time domain with σ = {sigma:.4f}")
        
    elif broadening_type == 'lorentzian' and gamma is not None:
        # Apply Lorentzian broadening: multiply by exp(-γ|t|)
        broadening_factor = np.exp(-gamma * np.abs(t_values))
        broadened_data = data * broadening_factor
        print(f"Applied Lorentzian broadening in time domain with γ = {gamma:.4f}")
        
    else:
        broadened_data = data
        print("No time domain broadening applied")
    
    return broadened_data, broadening_factor if 'broadening_factor' in locals() else np.ones_like(data)

def parse_filename_new(filename):
    """Extract species (including momentum), beta, and sample index from filenames.

    Supports numeric beta and the special strings 'inf' / 'infty' (case-insensitive).
    
    Handles both old and new filename patterns:
    - Old: time_corr_rand{N}_{species}_beta={beta}.dat
    - New: time_corr_rand{N}_{species}_beta={beta}.dat (same format)
    
    The species name may include momentum information like:
    - SpSm_q_Qx0_Qy0_Qz6.28319
    - SzSz_q_Qx0_Qy0_Qz0
    - Experimental_q_Qx0_Qy0_Qz0_theta0
    - etc.

    Returns: (species_with_momentum, beta(float or np.inf), sample_idx) or (None, None, None)
    """
    basename = os.path.basename(filename)
    # Updated regex to be more flexible with species names
    # Species can contain letters, numbers, underscores, dots, and minus signs
    m = re.match(r'^time_corr_(?:rand|sample)(\d+)_(.+?)_beta=([0-9.+-eE]+|inf|infty)\.dat$', basename, re.IGNORECASE)
    if not m:
        return None, None, None
    sample_idx = int(m.group(1))
    species_with_momentum = m.group(2)
    beta_token = m.group(3)
    if beta_token.lower() in ("inf", "infty"):
        beta = np.inf
    else:
        try:
            beta = float(beta_token)
        except ValueError:
            return None, None, None
    return species_with_momentum, beta, sample_idx

def find_spectral_peaks(omega, spectral_function, min_prominence=0.1, min_height=None, omega_range=(0, 6)):
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

def parse_QFI_data_new(structure_factor_dir, beta_tol=1e-2, average_after_fft=True):
    """Parse QFI data from the new directory structure and compute spectral functions.
    
    Updated to work with TPQ_DSSF.cpp output structure:
    - structure_factor_results/
      - beta_{value}/
        - {operator_type}/  (e.g., 'sum', 'transverse', 'sublattice', 'experimental', etc.)
          - time_corr_rand{N}_{species}_beta={beta}.dat
    
    The function automatically scans all subdirectories within each beta_* folder,
    making it flexible to handle any operator_type name.
    
    Parameters:
    structure_factor_dir: Directory containing beta_* subdirectories
    beta_tol: Tolerance for grouping beta values
    average_after_fft: If True, average spectra after FFT (recommended).
                       If False, average in time domain (assumes identical time arrays).
    """
    
    # Initialize data structures
    species_data = defaultdict(lambda: defaultdict(list))
    species_names = set()
    beta_bins = []
    beta_bin_values = defaultdict(list)
    
    # Step 1: Discover and organize data files
    print(f"Scanning directory: {structure_factor_dir}")
    _collect_data_files(structure_factor_dir, species_data, species_names, 
                        beta_bins, beta_bin_values, beta_tol)
    
    print(f"\nFound species (with momentum): {sorted(species_names)}")
    print(f"Averaging strategy: {'After FFT' if average_after_fft else 'Before FFT (time domain)'}")
    
    # Step 2: Process each species
    all_species_qfi_data = defaultdict(list)
    
    for species, beta_groups in species_data.items():
        print(f"\n{'='*60}")
        print(f"Processing species: {species}")
        print(f"{'='*60}")
        
        _process_species_data(species, beta_groups, beta_bin_values, 
                            structure_factor_dir, all_species_qfi_data,
                            average_after_fft=average_after_fft)
    
    # Step 3: Generate summary plots
    _create_summary_plots(all_species_qfi_data, structure_factor_dir)
    
    print("\nProcessing complete!")
    return all_species_qfi_data


def _collect_data_files(structure_factor_dir, species_data, species_names, 
                        beta_bins, beta_bin_values, beta_tol):
    """Collect and organize all data files by species and beta values.
    
    Updated to scan all subdirectories within each beta_* folder to support
    the new structure from TPQ_DSSF.cpp where operator_type determines the subdirectory name
    (e.g., 'sum', 'transverse', 'sublattice', 'experimental', etc.)
    """
    
    beta_dirs = glob.glob(os.path.join(structure_factor_dir, 'beta_*'))
    print(f"Found {len(beta_dirs)} beta directories")
    
    for beta_dir in beta_dirs:
        beta_value = _extract_beta_from_dirname(beta_dir)
        if beta_value is None:
            continue
            
        bin_idx = _assign_beta_bin(beta_value, beta_bins, beta_tol)
        beta_bin_values[bin_idx].append(beta_value)
        
        # Scan all subdirectories within this beta directory
        # This handles both old structure (taylor, global, etc.) and new structure (sum, transverse, etc.)
        if os.path.isdir(beta_dir):
            subdirs = [d for d in os.listdir(beta_dir) if os.path.isdir(os.path.join(beta_dir, d))]
            
            if not subdirs:
                print(f"Warning: No subdirectories found in {beta_dir}")
                continue
            
            print(f"  Beta={beta_value:.6g}: Found subdirectories: {subdirs}")
            
            for subdir in subdirs:
                subdir_path = os.path.join(beta_dir, subdir)
                files = glob.glob(os.path.join(subdir_path, 'time_corr_*.dat'))
                
                for file_path in files:
                    species_with_momentum, _, _ = parse_filename_new(file_path)
                    if species_with_momentum:
                        species_names.add(species_with_momentum)
                        species_data[species_with_momentum][bin_idx].append(file_path)
            
            # Count total files found for this beta
            total_files = sum(len(glob.glob(os.path.join(beta_dir, subdir, 'time_corr_*.dat'))) 
                            for subdir in subdirs)
            print(f"  Beta={beta_value:.6g}: Found {total_files} correlation files across {len(subdirs)} subdirectories")


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
        if np.isinf(c):
            continue
        if abs(beta_val - c) <= tol:
            return i
            
    # Create new bin
    bins.append(beta_val)
    return len(bins) - 1


def _process_species_data(species, beta_groups, beta_bin_values, 
                            structure_factor_dir, all_species_qfi_data, 
                            average_after_fft=True):
    """Process all data for a single species across different beta values.
    
    Parameters:
    average_after_fft: If True, average spectra after FFT (recommended when time arrays differ).
                       If False, average in time domain (original behavior).
    """
    
    for beta_bin_idx, file_list in beta_groups.items():
        # Get representative beta for this bin
        beta = _get_bin_beta(beta_bin_values.get(beta_bin_idx, []))
        print(f"\n  Beta≈{beta:.6g} (bin {beta_bin_idx}): {len(file_list)} files")
        
        if average_after_fft:
            # New approach: average after FFT
            result = _load_and_average_data(file_list, average_after_fft=True)
            if result[0] is None:
                print(f"    No valid data for beta={beta}")
                continue
            
            mean_spectrum, mean_spectrum_imag, omega, time_arrays = result
            
            # Extract positive frequencies and compensate
            omega_pos, s_omega_compensated = _extract_positive_frequencies(mean_spectrum, omega)
            
            # Calculate QFI
            qfi = _calculate_qfi(omega_pos, s_omega_compensated, beta)
            
            # Find peaks
            peak_positions, peak_heights, peak_prominences = find_spectral_peaks(
                omega_pos, s_omega_compensated, min_prominence=0.1, omega_range=(0, 6))
            
            results = {
                'omega': omega,
                'S_omega_real': mean_spectrum,
                'S_omega_imag': mean_spectrum_imag,
                'omega_pos': omega_pos,
                's_omega_compensated': s_omega_compensated,
                'qfi': qfi,
                'peak_positions': peak_positions,
                'peak_heights': peak_heights,
                'peak_prominences': peak_prominences
            }
            
        else:
            # Original approach: average in time domain
            mean_correlation, reference_time = _load_and_average_data(file_list, average_after_fft=False)
            if mean_correlation is None:
                print(f"    No valid data for beta={beta}")
                continue
            
            # Compute spectral function and QFI
            results = _compute_spectral_and_qfi(mean_correlation, reference_time, beta)
        
        # Save results
        _save_species_results(species, beta, results, structure_factor_dir)
        
        # Store QFI for summary plots
        all_species_qfi_data[species].append((beta, results['qfi']))


def _get_bin_beta(beta_vals):
    """Get representative beta value for a bin."""
    if len(beta_vals) == 0:
        return np.nan
    elif any(np.isinf(v) for v in beta_vals):
        return np.inf
    else:
        return float(np.mean(beta_vals))


def _load_and_average_data(file_list, average_after_fft=True):
    """Load and average correlation data from multiple files.
    
    Parameters:
    file_list: List of file paths to load
    average_after_fft: If True, compute FFT for each sample and average spectra.
                      If False, average time-domain data (only works if time arrays match).
    
    Returns:
    If average_after_fft=False: (mean_correlation, reference_time)
    If average_after_fft=True: (mean_spectral_function, omega, all_time_arrays)
    """
    print(f"    [DEBUG] Loading {len(file_list)} files...")
    all_data = []
    
    for idx, file_path in enumerate(file_list):
        try:
            data = np.loadtxt(file_path, comments='#')
            time = data[:, 0]
            real_part = data[:, 1]
            imag_part = data[:, 2] if data.shape[1] > 2 else np.zeros_like(real_part)
            val = real_part + 1j * imag_part
            all_data.append((time, val))
            if idx == 0:
                print(f"    [DEBUG] First file: {os.path.basename(file_path)}, time shape: {time.shape}, data shape: {data.shape}")
        except Exception as e:
            print(f"    [ERROR] reading {file_path}: {e}")
    
    if not all_data:
        print(f"    [DEBUG] No valid data loaded!")
        return None, None, None if average_after_fft else None, None
    
    print(f"    [DEBUG] Successfully loaded {len(all_data)} files")
    
    if not average_after_fft:
        # Original behavior: average in time domain
        # Warning: assumes all time arrays are identical!
        reference_time = all_data[0][0]
        all_complex_values = np.vstack([data[1] for data in all_data])
        mean_correlation = np.mean(all_complex_values, axis=0)
        print(f"    [DEBUG] Averaged in time domain, result shape: {mean_correlation.shape}")
        return mean_correlation, reference_time
    
    else:
        # New behavior: average after FFT
        # This is more robust when time arrays differ between samples
        print(f"    Computing FFT for {len(all_data)} samples individually...")
        
        # Check if time arrays are consistent
        time_arrays_match = all(np.array_equal(all_data[0][0], d[0]) for d in all_data)
        if not time_arrays_match:
            print(f"    WARNING: Time arrays differ between samples - will interpolate to common grid!")
            # Debug: show time array lengths
            time_lengths = [len(d[0]) for d in all_data]
            print(f"    [DEBUG] Time array lengths: {time_lengths}")
        
        all_spectra = []
        all_spectra_imag = []
        all_omegas = []
        
        for idx, (time, correlation) in enumerate(all_data):
            # Prepare full time range
            t_full, C_full = _prepare_time_data(correlation, time)
            
            # Subtract mean
            C_full = C_full - np.mean(C_full)
            
            # Apply broadening and compute FFT
            gamma = 0.15
            S_omega_real, S_omega_imag, omega = _compute_spectral_function(t_full, C_full, gamma)
            
            all_spectra.append(S_omega_real)
            all_spectra_imag.append(S_omega_imag)
            all_omegas.append(omega)
            
            if idx == 0:
                print(f"    [DEBUG] Sample {idx}: omega shape = {omega.shape}, spectrum shape = {S_omega_real.shape}")
        
        # Create common frequency grid
        # Use the grid from the longest time series (finest frequency resolution)
        omega_ref = max(all_omegas, key=len)
        print(f"    [DEBUG] Reference omega grid length: {len(omega_ref)}")
        
        # Interpolate all spectra onto common grid if needed
        interpolated_spectra = []
        interpolated_spectra_imag = []
        for i, (S_omega, S_omega_imag, omega) in enumerate(zip(all_spectra, all_spectra_imag, all_omegas)):
            if len(omega) != len(omega_ref):
                # Need to interpolate due to different lengths
                print(f"      Sample {i}: interpolating from {len(omega)} to {len(omega_ref)} points")
                from scipy.interpolate import interp1d
                f_interp_real = interp1d(omega, S_omega, kind='linear', 
                                   bounds_error=False, fill_value=0.0)
                f_interp_imag = interp1d(omega, S_omega_imag, kind='linear', 
                                   bounds_error=False, fill_value=0.0)
                S_interpolated = f_interp_real(omega_ref)
                S_interpolated_imag = f_interp_imag(omega_ref)
                interpolated_spectra.append(S_interpolated)
                interpolated_spectra_imag.append(S_interpolated_imag)
            elif not np.allclose(omega, omega_ref, rtol=1e-9, atol=1e-12):
                # Lengths match but values differ - still need interpolation
                print(f"      Sample {i}: interpolating due to frequency grid mismatch")
                from scipy.interpolate import interp1d
                f_interp_real = interp1d(omega, S_omega, kind='linear',
                                   bounds_error=False, fill_value=0.0)
                f_interp_imag = interp1d(omega, S_omega_imag, kind='linear',
                                   bounds_error=False, fill_value=0.0)
                S_interpolated = f_interp_real(omega_ref)
                S_interpolated_imag = f_interp_imag(omega_ref)
                interpolated_spectra.append(S_interpolated)
                interpolated_spectra_imag.append(S_interpolated_imag)
            else:
                # Grids match perfectly
                interpolated_spectra.append(S_omega)
                interpolated_spectra_imag.append(S_omega_imag)
        
        # Average spectra
        mean_spectrum = np.mean(interpolated_spectra, axis=0)
        mean_spectrum_imag = np.mean(interpolated_spectra_imag, axis=0)
        print(f"    Averaged {len(interpolated_spectra)} spectra in frequency domain")
        print(f"    [DEBUG] Mean spectrum shape: {mean_spectrum.shape}, min: {mean_spectrum.min():.6e}, max: {mean_spectrum.max():.6e}")
        
        return mean_spectrum, mean_spectrum_imag, omega_ref, [d[0] for d in all_data]


def _compute_spectral_and_qfi(mean_correlation, reference_time, beta):
    """Compute spectral function and QFI from correlation data."""
    
    # Prepare time data
    t_full, C_full = _prepare_time_data(mean_correlation, reference_time)
    
    # Subtract mean from correlation data
    C_full = C_full - np.mean(C_full)
    print(f"    Subtracted mean from correlation data (mean = {np.mean(C_full):.6e})")

    # Apply taper to reduce spectral leakage
    def apply_taper(t_data, c_data, taper_type='hann'):
        """Apply a windowing function to the time data."""
        n = len(t_data)
        
        if taper_type == 'hann':
            window = np.hanning(n)
        elif taper_type == 'hamming':
            window = np.hamming(n)
        elif taper_type == 'blackman':
            window = np.blackman(n)
        elif taper_type == 'tukey':
            # Tukey window with 10% taper on each side
            window = np.ones(n)
            taper_fraction = 0.1
            taper_len = int(n * taper_fraction)
            # Apply cosine taper to both ends
            for i in range(taper_len):
                window[i] = 0.5 * (1 - np.cos(np.pi * i / taper_len))
                window[n-1-i] = 0.5 * (1 - np.cos(np.pi * i / taper_len))
        else:
            # No taper
            window = np.ones(n)
        
        return c_data * window
    
    # Apply taper
    # C_full = apply_taper(t_full, C_full, taper_type='tukey')
    print(f"    Applied Tukey taper to time data")

    # Apply broadening and compute FFT
    gamma = 0.15
    S_omega_real, S_omega_imag, omega = _compute_spectral_function(t_full, C_full, gamma)
    
    # Extract positive frequencies and compensate
    omega_pos, s_omega_compensated = _extract_positive_frequencies(S_omega_real, omega)
    
    # Calculate QFI
    qfi = _calculate_qfi(omega_pos, s_omega_compensated, beta)
    
    # Find peaks
    peak_positions, peak_heights, peak_prominences = find_spectral_peaks(
        omega_pos, s_omega_compensated, min_prominence=0.1, omega_range=(0, 6))
    
    return {
        'omega': omega,
        'S_omega_real': S_omega_real,
        'S_omega_imag': S_omega_imag,
        'omega_pos': omega_pos,
        's_omega_compensated': s_omega_compensated,
        'qfi': qfi,
        'peak_positions': peak_positions,
        'peak_heights': peak_heights,
        'peak_prominences': peak_prominences
    }


def _prepare_time_data(mean_correlation, reference_time):
    """Prepare correlation data for FFT (data starts from t=0)."""
    # Raw data already starts from t=0 with positive times only
    # No need to construct negative times or reorder
    print(f"    Using time data as-is: [{reference_time.min():.2f}, {reference_time.max():.2f}]")
    return reference_time, mean_correlation


def _compute_spectral_function(t_full, C_full, gamma):
    """Compute spectral function via FFT with broadening."""
    
    # Apply broadening directly (no need to reorder since data starts from t=0)
    C_broadened, _ = apply_time_broadening(
        t_full, C_full, 'lorentzian', gamma=gamma)
    
    # FFT convention
    C_broadened = np.conj(C_broadened)
    
    # Compute FFT
    dt = t_full[1] - t_full[0] if len(t_full) > 1 else 1.0
    C_w = np.fft.fft(C_broadened)
    S_w = dt * np.fft.fftshift(C_w) / (2 * np.pi)
    
    # Frequency axis
    omega = np.fft.fftshift(np.fft.fftfreq(len(C_broadened), d=dt)) * 2 * np.pi
    
    return S_w.real, S_w.imag, omega


def _extract_positive_frequencies(S_omega_real, omega):
    """Extract positive frequencies and apply compensation."""
    
    # Calculate integral before truncation
    # integral_before = np_trapz(S_omega_real, omega)
    
    # Extract positive frequencies
    positive_mask = omega > 0
    omega_pos = omega[positive_mask]
    s_omega_pos = S_omega_real[positive_mask]
    
    # Calculate compensation factor
    # integral_after = np_trapz(s_omega_pos, omega_pos)
    # compensation_factor = integral_before / integral_after if integral_after != 0 else 1.0
    compensation_factor = 1
    s_omega_compensated = s_omega_pos * compensation_factor
    
    print(f"    Compensation factor: {compensation_factor:.6f}")
    
    return omega_pos, s_omega_compensated


def _calculate_qfi(omega_pos, s_omega_pos, beta):
    """Calculate quantum Fisher information."""
    
    if np.isinf(beta):
        # β→∞ limit
        integrand = s_omega_pos
    else:
        integrand = s_omega_pos * np.tanh(beta * omega_pos / 2.0) * (1 - np.exp(-beta * omega_pos))
    
    return 4 * np_trapz(integrand, omega_pos)


def _save_species_results(species, beta, results, structure_factor_dir):
    """Save computed results for a species at given beta."""
    
    # Create output directory
    outdir = os.path.join(structure_factor_dir, 'processed_data', species)
    os.makedirs(outdir, exist_ok=True)
    
    beta_label = 'inf' if np.isinf(beta) else f'{beta:.6g}'
    
    # Save spectral data
    data_out = np.column_stack((results['omega'], results['S_omega_real'], results['S_omega_imag']))
    data_filename = os.path.join(outdir, f'spectral_beta_{beta_label}.dat')
    np.savetxt(data_filename, data_out, header='freq spectral_function_real spectral_function_imag')
    
    # Save peak information
    if results['peak_positions']:
        peak_data = np.column_stack((
            results['peak_positions'], 
            results['peak_heights'], 
            results['peak_prominences']
        ))
        peak_filename = os.path.join(outdir, f'peaks_beta_{beta_label}.dat')
        np.savetxt(peak_filename, peak_data, header='freq height prominence')
    
    # Create spectral function plot
    _plot_spectral_function(species, beta, results, outdir)
    
    print(f"    QFI = {results['qfi']:.4f}, Peaks at: {results['peak_positions']}")


def _plot_spectral_function(species, beta, results, outdir):
    """Plot spectral function with peaks marked."""
    
    plt.figure(figsize=(10, 6))
    
    beta_label = 'inf' if np.isinf(beta) else f'{beta:.6g}'
    
    # Plot real and imaginary parts of spectral function
    plt.scatter(results['omega'], results['S_omega_real'], 
                label=f'Real part (Beta≈{beta_label}, QFI={results["qfi"]:.4f})', 
                alpha=0.7, s=20)
    plt.scatter(results['omega'], results['S_omega_imag'], 
                label=f'Imaginary part (Beta≈{beta_label})', 
                alpha=0.7, s=20, marker='^')
    
    # Mark peaks on the real part
    if results['peak_positions']:
        plt.scatter(results['peak_positions'], results['peak_heights'], 
                    color='red', s=80, marker='x', zorder=5,
                    label=f'Peaks: {[f"{pos:.2f}" for pos in results["peak_positions"]]}')
    
    plt.xlabel('Frequency (rad/s)')
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
        if not qfi_data:
            continue
            
        _plot_qfi_vs_beta(species, qfi_data, plot_outdir)
        _plot_qfi_derivative(species, qfi_data, plot_outdir)


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
        plt.plot(qfi_beta_array[finite_mask, 0], 
                qfi_beta_array[finite_mask, 1], 'o-', label='Finite β')
    
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
        x_annot = np.max(qfi_beta_array[finite_mask, 0])
    else:
        x_annot = 1.0
    
    plt.scatter([x_annot], [beta_inf_qfi.mean()], 
                marker='*', s=160, c='red', label='β→∞')
    plt.text(x_annot, beta_inf_qfi.mean(), 
            f' β→∞ QFI={beta_inf_qfi.mean():.4g}',
            fontsize=9, ha='left', va='bottom')


def _plot_qfi_derivative(species, qfi_data, plot_outdir):
    """Plot derivative of QFI with respect to beta."""
    
    # Sort and convert to array
    qfi_data.sort(key=lambda x: (np.inf if np.isinf(x[0]) else x[0]))
    qfi_beta_array = np.array(qfi_data, dtype=float)
    
    # Extract finite betas only
    finite_qfi = qfi_beta_array[np.isfinite(qfi_beta_array[:, 0])]
    
    if len(finite_qfi) <= 1:
        return
    
    betas = finite_qfi[:, 0]
    qfis = finite_qfi[:, 1]
    
    # Calculate derivative using central differences
    mid_betas = (betas[:-1] + betas[1:]) / 2
    delta_beta = np.diff(betas)
    delta_qfi = np.diff(qfis)
    qfi_derivative = delta_qfi / delta_beta
    
    # Save derivative data
    derivative_data = np.column_stack((mid_betas, qfi_derivative))
    data_filename = os.path.join(plot_outdir, f'qfi_derivative_vs_beta_{species}.dat')
    np.savetxt(data_filename, derivative_data, header='beta dQFI/dbeta')
    
    # Create plot
    plt.figure(figsize=(10, 6))
    plt.plot(mid_betas, qfi_derivative, 'o-')
    plt.xlabel('Beta (β)')
    plt.ylabel('dQFI/dβ')
    plt.title(f'Derivative of QFI vs. Beta for {species}')
    plt.xscale('log')
    plt.grid(True)
    
    plot_filename = os.path.join(plot_outdir, f'qfi_derivative_vs_beta_{species}.png')
    plt.savefig(plot_filename, dpi=300)
    plt.close()

def parse_QFI_across_Jpm(data_dir):
    
    if HAS_MPI:
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()
    else:
        # Serial mode
        rank = 0
        size = 1
        comm = None
    
    # Find all subdirectories matching the pattern Jpm=*
    subdirs = []
    if rank == 0:
        subdirs = glob.glob(os.path.join(data_dir, 'Jpm=*'))
        subdirs.sort()  # Ensure consistent ordering
    
    # Broadcast subdirs to all processes (only if MPI is available)
    if HAS_MPI:
        subdirs = comm.bcast(subdirs, root=0)
    
    # Distribute subdirectories among processes
    local_subdirs = []
    for i, subdir in enumerate(subdirs):
        if i % size == rank:
            local_subdirs.append(subdir)
    
    # Each process handles its assigned subdirectories
    local_jpm_qfi_data = {}
    
    for subdir in local_subdirs:
        # Extract Jpm value from the directory name
        match = re.search(r'Jpm=([-]?[\d\.]+)', os.path.basename(subdir))
        if not match:
            continue
        jpm_value = float(match.group(1))
        
        # Path to structure_factor_results directory
        structure_factor_dir = os.path.join(subdir, 'structure_factor_results')
        
        if not os.path.exists(structure_factor_dir):
            print(f"[Rank {rank}] Structure factor directory not found: {structure_factor_dir}")
            continue
        
        print(f"[Rank {rank}] Processing directory: {subdir} for Jpm={jpm_value}")
        # Run the QFI analysis for the current Jpm value
        species_qfi_data = parse_QFI_data_new(structure_factor_dir)
        local_jpm_qfi_data[jpm_value] = species_qfi_data
    
    # Gather all results at rank 0 (only if MPI is available)
    if HAS_MPI:
        all_jpm_qfi_data = comm.gather(local_jpm_qfi_data, root=0)
    else:
        all_jpm_qfi_data = [local_jpm_qfi_data]
    
    if rank == 0:
        # Merge all results
        jpm_qfi_data = {}
        for process_data in all_jpm_qfi_data:
            jpm_qfi_data.update(process_data)
        
        # Reorganize data by species for heatmap plotting
        all_qfi_data = defaultdict(list)
        all_derivative_data = defaultdict(list)
        
        for jpm, all_species_data in jpm_qfi_data.items():
            for species, qfi_beta_list in all_species_data.items():
                for beta, qfi in qfi_beta_list:
                    all_qfi_data[species].append((jpm, beta, qfi))
                
                # Calculate derivatives for this species and jpm
                if len(qfi_beta_list) > 1:
                    qfi_beta_list.sort()
                    qfi_beta_array = np.array(qfi_beta_list)
                    betas = qfi_beta_array[:, 0]
                    qfis = qfi_beta_array[:, 1]
                    
                    # Use central differences for the derivative
                    mid_betas = (betas[:-1] + betas[1:]) / 2
                    delta_beta = np.diff(betas)
                    delta_qfi = np.diff(qfis)
                    qfi_derivative = delta_qfi / delta_beta
                    
                    for mid_beta, derivative in zip(mid_betas, qfi_derivative):
                        all_derivative_data[species].append((jpm, mid_beta, derivative))

        # Create output directory
        plot_outdir = os.path.join(data_dir, 'plots')
        os.makedirs(plot_outdir, exist_ok=True)
        return jpm_qfi_data
    else:
        return None



def parse_QFI_across_hi(data_dir):
    """
    Scan subdirectories named 'h=i=*' under data_dir, run QFI parsing per folder,
    and build heatmaps across the parameter h=i.
    """
    if HAS_MPI:
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()
    else:
        # Serial mode
        rank = 0
        size = 1
        comm = None

    # Discover parameter sweep folders on rank 0
    if rank == 0:
        sweep_dirs = sorted(glob.glob(os.path.join(data_dir, 'h=*')))
    else:
        sweep_dirs = None
    
    # Broadcast only if MPI is available
    if HAS_MPI:
        sweep_dirs = comm.bcast(sweep_dirs, root=0)

    # Round-robin assignment
    my_dirs = sweep_dirs[rank::size]

    # Local compute
    local_results = {}
    param_regex = re.compile(r'h=([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)')
    for d in my_dirs:
        m = param_regex.search(os.path.basename(d))
        if not m:
            continue
        hi_val = float(m.group(1))
        sf_path = os.path.join(d, 'structure_factor_results')
        if not os.path.isdir(sf_path):
            print(f"[Rank {rank}] Missing structure_factor_results at: {sf_path}")
            continue
        print(f"[Rank {rank}] Processing {d} (h=i={hi_val})")
        local_results[hi_val] = parse_QFI_data_new(sf_path)

    # Gather and merge on root (only if MPI is available)
    if HAS_MPI:
        gathered = comm.gather(local_results, root=0)
    else:
        gathered = [local_results]
    
    if rank != 0:
        return None

    merged = {}
    for part in gathered:
        merged.update(part)

    # Reformat per-species arrays and compute derivatives
    by_species = defaultdict(list)
    by_species_deriv = defaultdict(list)
    for hi, species_map in merged.items():
        for sp, beta_qfi in species_map.items():
            for b, q in beta_qfi:
                by_species[sp].append((hi, b, q))

            if len(beta_qfi) > 1:
                bq = np.array(sorted(beta_qfi, key=lambda x: x[0]), dtype=float)
                bvals, qvals = bq[:, 0], bq[:, 1]
                mid = 0.5 * (bvals[:-1] + bvals[1:])
                dq = np.diff(qvals)
                db = np.diff(bvals)
                deriv = dq / db
                for mb, dv in zip(mid, deriv):
                    by_species_deriv[sp].append((hi, mb, dv))

    # Plotting
    out_dir = os.path.join(data_dir, 'plots_hi')
    os.makedirs(out_dir, exist_ok=True)

    # Heatmaps for QFI
    for sp, triples in by_species.items():
        if not triples:
            continue
        arr = np.array(triples, dtype=float)
        X, Y, Z = arr[:, 0], arr[:, 1], arr[:, 2]

        # Build grid in parameter (linear) and beta (log)
        x_min, x_max = np.nanmin(X), np.nanmax(X)
        y_min, y_max = np.nanmin(Y[Y > 0]), np.nanmax(Y)
        x_lin = np.linspace(x_min, x_max, 120)
        y_log = np.logspace(np.log10(y_min), np.log10(y_max), 120)
        XX, YY = np.meshgrid(x_lin, y_log)
        ZZ = griddata((X, Y), Z, (XX, YY), method='cubic')

        plt.figure(figsize=(11, 7))
        mesh = plt.pcolormesh(XX, YY, ZZ, shading='auto', cmap='viridis')
        plt.colorbar(mesh, label='QFI')
        plt.scatter(X, Y, s=14, c='k', alpha=0.6, label='samples')
        plt.yscale('log')
        plt.xlabel('h=i')
        plt.ylabel('Beta (β)')
        plt.title(f'QFI heatmap (h=i sweep): {sp}')
        plt.legend(loc='best')
        fout = os.path.join(out_dir, f'qfi_heatmap_hi_{sp}.png')
        plt.savefig(fout, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {fout}")

    # Heatmaps for dQFI/dβ
    for sp, triples in by_species_deriv.items():
        if not triples:
            continue
        arr = np.array(triples, dtype=float)
        X, Y, Z = arr[:, 0], arr[:, 1], arr[:, 2]

        x_min, x_max = np.nanmin(X), np.nanmax(X)
        y_min, y_max = np.nanmin(Y[Y > 0]), np.nanmax(Y)
        x_lin = np.linspace(x_min, x_max, 120)
        y_log = np.logspace(np.log10(y_min), np.log10(y_max), 120)
        XX, YY = np.meshgrid(x_lin, y_log)
        ZZ = griddata((X, Y), Z, (XX, YY), method='cubic')

        plt.figure(figsize=(11, 7))
        mesh = plt.pcolormesh(XX, YY, ZZ, shading='auto', cmap='viridis')
        plt.colorbar(mesh, label='dQFI/dβ')
        plt.scatter(X, Y, s=14, c='k', alpha=0.6, label='samples')
        plt.yscale('log')
        plt.xlabel('h=i')
        plt.ylabel('Beta (β)')
        plt.title(f'dQFI/dβ heatmap (h=i sweep): {sp}')
        plt.legend(loc='best')
        fout = os.path.join(out_dir, f'qfi_derivative_heatmap_hi_{sp}.png')
        plt.savefig(fout, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {fout}")

    print("h=i sweep processing complete!")
    return merged


def track_peak_evolution_across_h(data_dir, target_beta=None):
    """
    Track the evolution of spectral peak positions as a function of h parameter.
    
    Parameters:
    data_dir: Directory containing h=* subdirectories
    target_beta: Specific beta value to focus on (if None, uses the largest available beta)
    
    Returns:
    peak_evolution_data: Dictionary containing peak tracking results for each species
    """
    if HAS_MPI:
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()
    else:
        # Serial mode
        rank = 0
        size = 1
        comm = None

    # Discover parameter sweep folders on rank 0
    if rank == 0:
        sweep_dirs = sorted(glob.glob(os.path.join(data_dir, 'h=*')))
    else:
        sweep_dirs = None
    
    # Broadcast only if MPI is available
    if HAS_MPI:
        sweep_dirs = comm.bcast(sweep_dirs, root=0)

    # Round-robin assignment
    my_dirs = sweep_dirs[rank::size]

    # Local compute
    local_peak_data = {}
    param_regex = re.compile(r'h=([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)')
    
    for d in my_dirs:
        m = param_regex.search(os.path.basename(d))
        if not m:
            continue
        hi_val = float(m.group(1))
        
        # Look for processed spectral data
        sf_path = os.path.join(d, 'structure_factor_results')
        if not os.path.isdir(sf_path):
            continue
            
        print(f"[Rank {rank}] Tracking peaks for h={hi_val}")
        
        # Find all species subdirectories in processed_data
        processed_data_dir = os.path.join(sf_path, 'processed_data')
        if not os.path.isdir(processed_data_dir):
            continue
            
        species_dirs = [d for d in os.listdir(processed_data_dir) 
                       if os.path.isdir(os.path.join(processed_data_dir, d))]
        
        local_peak_data[hi_val] = {}
        
        for species in species_dirs:
            species_dir = os.path.join(processed_data_dir, species)
            
            # Find peak files
            peak_files = glob.glob(os.path.join(species_dir, 'peaks_beta_*.dat'))
            
            species_peaks = []
            for peak_file in peak_files:
                # Extract beta from filename
                beta_match = re.search(r'peaks_beta_([\d\.]+|inf)\.dat', os.path.basename(peak_file))
                if not beta_match:
                    continue
                    
                beta_str = beta_match.group(1)
                if beta_str == 'inf':
                    beta_val = np.inf
                else:
                    beta_val = float(beta_str)
                
                try:
                    # Load peak data
                    peak_data = np.loadtxt(peak_file)
                    if peak_data.size == 0:
                        continue
                    if peak_data.ndim == 1:
                        peak_data = peak_data.reshape(1, -1)
                    
                    # Store peaks for this beta
                    for row in peak_data:
                        freq, height, prominence = row
                        species_peaks.append((beta_val, freq, height, prominence))
                        
                except Exception as e:
                    print(f"[Rank {rank}] Error reading {peak_file}: {e}")
            
            if species_peaks:
                local_peak_data[hi_val][species] = species_peaks

    # Gather and merge on root (only if MPI is available)
    if HAS_MPI:
        gathered = comm.gather(local_peak_data, root=0)
    else:
        gathered = [local_peak_data]
    
    if rank != 0:
        return None

    merged_peak_data = {}
    for part in gathered:
        for hi, species_data in part.items():
            if hi not in merged_peak_data:
                merged_peak_data[hi] = {}
            merged_peak_data[hi].update(species_data)

    # Process and plot peak evolution
    out_dir = os.path.join(data_dir, 'peak_evolution')
    os.makedirs(out_dir, exist_ok=True)

    # Organize data by species
    species_peak_evolution = defaultdict(lambda: defaultdict(list))
    
    for hi, species_data in merged_peak_data.items():
        for species, peaks in species_data.items():
            for beta_val, freq, height, prominence in peaks:
                species_peak_evolution[species][beta_val].append((hi, freq, height, prominence))

    # Plot peak evolution for each species and beta
    for species, beta_data in species_peak_evolution.items():
        for beta_val, peak_list in beta_data.items():
            if len(peak_list) < 2:  # Need at least 2 points to track evolution
                continue
                
            # Sort by h value
            peak_list.sort(key=lambda x: x[0])
            peak_array = np.array(peak_list, dtype=float)
            
            h_vals = peak_array[:, 0]
            freqs = peak_array[:, 1]
            heights = peak_array[:, 2]
            prominences = peak_array[:, 3]
            
            # Plot frequency evolution
            plt.figure(figsize=(12, 8))
            
            # Subplot 1: Peak frequency vs h
            plt.subplot(2, 2, 1)
            plt.scatter(h_vals, freqs, c=heights, cmap='viridis', s=50, alpha=0.7)
            plt.colorbar(label='Peak Height')
            plt.xlabel('h parameter')
            plt.ylabel('Peak Frequency (rad/s)')
            plt.title(f'Peak Frequency Evolution: {species}, β={beta_val:.3g}')
            plt.grid(True, alpha=0.3)
            
            # Subplot 2: Peak height vs h
            plt.subplot(2, 2, 2)
            plt.plot(h_vals, heights, 'o-', alpha=0.7)
            plt.xlabel('h parameter')
            plt.ylabel('Peak Height')
            plt.title(f'Peak Height Evolution: {species}, β={beta_val:.3g}')
            plt.grid(True, alpha=0.3)
            
            # Subplot 3: Peak prominence vs h
            plt.subplot(2, 2, 3)
            plt.plot(h_vals, prominences, 's-', alpha=0.7, color='orange')
            plt.xlabel('h parameter')
            plt.ylabel('Peak Prominence')
            plt.title(f'Peak Prominence Evolution: {species}, β={beta_val:.3g}')
            plt.grid(True, alpha=0.3)
            
            # Subplot 4: 2D trajectory in frequency-h space
            plt.subplot(2, 2, 4)
            plt.plot(h_vals, freqs, 'o-', alpha=0.7)
            plt.xlabel('h parameter')
            plt.ylabel('Peak Frequency (rad/s)')
            plt.title(f'Peak Trajectory: {species}, β={beta_val:.3g}')
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            beta_label = 'inf' if np.isinf(beta_val) else f'{beta_val:.3g}'
            plot_filename = os.path.join(out_dir, f'peak_evolution_{species}_beta_{beta_label}.png')
            plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
            plt.close()
            
            # Save peak evolution data
            data_filename = os.path.join(out_dir, f'peak_evolution_{species}_beta_{beta_label}.dat')
            header = 'h_parameter peak_frequency peak_height peak_prominence'
            np.savetxt(data_filename, peak_array, header=header)
            
            print(f"Saved peak evolution for {species}, β={beta_val:.3g}: {plot_filename}")

    # Create summary plots showing all species evolution at a specific beta
    if target_beta is not None:
        plt.figure(figsize=(12, 8))
        
        for species, beta_data in species_peak_evolution.items():
            # Find closest beta to target
            available_betas = list(beta_data.keys())
            if not available_betas:
                continue
                
            closest_beta = min(available_betas, key=lambda x: abs(x - target_beta) if not np.isinf(x) else float('inf'))
            
            if abs(closest_beta - target_beta) > 0.1 * target_beta:  # Skip if too far from target
                continue
                
            peak_list = beta_data[closest_beta]
            if len(peak_list) < 2:
                continue
                
            peak_list.sort(key=lambda x: x[0])
            peak_array = np.array(peak_list, dtype=float)
            
            h_vals = peak_array[:, 0]
            freqs = peak_array[:, 1]
            
            plt.plot(h_vals, freqs, 'o-', label=f'{species} (β={closest_beta:.3g})', alpha=0.7)
        
        plt.xlabel('h parameter')
        plt.ylabel('Peak Frequency (rad/s)')
        plt.title(f'Peak Frequency Evolution Comparison (β≈{target_beta:.3g})')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        summary_filename = os.path.join(out_dir, f'peak_evolution_summary_beta_{target_beta:.3g}.png')
        plt.savefig(summary_filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved summary plot: {summary_filename}")

    print("Peak evolution tracking complete!")
    return species_peak_evolution


def plot_heatmaps_from_processed_data(data_dir):
    """Plot heatmaps and fixed-beta line plots by reading processed QFI data from subdirectories.
       Only plot rows where there is no NaN, and save all intermediate and plot data."""
    
    print(f"\n{'='*70}")
    print(f"Starting heatmap plotting from processed data")
    print(f"Data directory: {data_dir}")
    print(f"{'='*70}\n")
    
    # Step 1: Load all QFI and derivative data
    all_qfi_data, all_derivative_data = load_processed_data(data_dir)
    
    print(f"Loaded QFI data for {len(all_qfi_data)} species")
    print(f"Loaded derivative data for {len(all_derivative_data)} species")
    
    # Step 2: Create output directory
    plot_outdir = os.path.join(data_dir, 'plots')
    os.makedirs(plot_outdir, exist_ok=True)
    print(f"Output directory: {plot_outdir}")
    
    # Step 3: Save raw data points
    save_raw_data_points(all_qfi_data, all_derivative_data, plot_outdir)
    
    # Step 4: Process and plot QFI heatmaps
    print(f"\n{'='*70}")
    print("Processing QFI heatmaps...")
    print(f"{'='*70}")
    for species, data_points in all_qfi_data.items():
        if data_points:
            try:
                process_species_heatmap(
                    species, data_points, plot_outdir, 
                    data_type='qfi', ref_target=0.08
                )
            except Exception as e:
                print(f"ERROR processing QFI heatmap for {species}: {e}")
                import traceback
                traceback.print_exc()
    
    # Step 5: Process and plot derivative heatmaps
    print(f"\n{'='*70}")
    print("Processing derivative heatmaps...")
    print(f"{'='*70}")
    for species, data_points in all_derivative_data.items():
        if data_points:
            try:
                process_species_heatmap(
                    species, data_points, plot_outdir,
                    data_type='derivative', ref_target=0.09
                )
            except Exception as e:
                print(f"ERROR processing derivative heatmap for {species}: {e}")
                import traceback
                traceback.print_exc()
    
    print(f"\n{'='*70}")
    print("Heatmap plotting complete!")
    print(f"{'='*70}\n")


def load_processed_data(data_dir):
    """Load QFI and derivative data from all Jpm subdirectories."""
    print(f"[DEBUG] load_processed_data: scanning directory {data_dir}")
    all_qfi_data = defaultdict(list)
    all_derivative_data = defaultdict(list)
    
    subdirs = glob.glob(os.path.join(data_dir, 'Jpm=*'))
    print(f"[DEBUG] Found {len(subdirs)} Jpm subdirectories")
    
    for subdir in subdirs:
        jpm_value = extract_jpm_value(subdir)
        if jpm_value is None:
            print(f"[DEBUG] Could not extract Jpm value from {subdir}")
            continue
            
        plots_dir = os.path.join(subdir, 'structure_factor_results', 'plots')
        if not os.path.exists(plots_dir):
            print(f"[DEBUG] Plots directory does not exist: {plots_dir}")
            continue
            
        print(f"Reading processed data from: {plots_dir} for Jpm={jpm_value}")
        
        # Load QFI files
        load_data_files(
            plots_dir, 'qfi_vs_beta_*.dat', 
            jpm_value, all_qfi_data
        )
        
        # Load derivative files
        load_data_files(
            plots_dir, 'qfi_derivative_vs_beta_*.dat',
            jpm_value, all_derivative_data
        )
    
    print(f"[DEBUG] Loaded data for {len(all_qfi_data)} species (QFI), {len(all_derivative_data)} species (derivative)")
    return all_qfi_data, all_derivative_data


def extract_jpm_value(subdir):
    """Extract Jpm value from directory name."""
    match = re.search(r'Jpm=([-]?[\d\.]+)', os.path.basename(subdir))
    return float(match.group(1)) if match else None


def load_data_files(plots_dir, pattern, jpm_value, data_dict):
    """Load data files matching pattern and add to data dictionary."""
    files = glob.glob(os.path.join(plots_dir, pattern))
    print(f"[DEBUG] load_data_files: pattern={pattern}, found {len(files)} files")
    
    for file_path in files:
        species = extract_species_from_filename(file_path, pattern)
        if not species:
            print(f"[DEBUG] Could not extract species from {os.path.basename(file_path)}")
            continue
            
        try:
            data = np.loadtxt(file_path)
            if data.size == 0:
                print(f"[DEBUG] Empty file: {file_path}")
                continue
            if data.ndim == 1:
                data = data.reshape(1, -1)
            
            for row in data:
                beta, value = row[0], row[1]
                data_dict[species].append((jpm_value, beta, value))
            
            print(f"[DEBUG] Loaded {len(data)} rows from {os.path.basename(file_path)} for species {species}")
        except Exception as e:
            print(f"[ERROR] reading {file_path}: {e}")


def extract_species_from_filename(file_path, pattern):
    """Extract species name from filename."""
    base_pattern = pattern.replace('*', '(.+?)')
    match = re.search(base_pattern, os.path.basename(file_path))
    return match.group(1) if match else None


def save_raw_data_points(all_qfi_data, all_derivative_data, plot_outdir):
    """Save raw assembled points for QFI and derivatives."""
    for species, data_points in all_qfi_data.items():
        if data_points:
            arr = np.array(data_points, dtype=float)
            np.savetxt(
                os.path.join(plot_outdir, f'qfi_points_{species}.dat'), 
                arr, header='Jpm beta qfi'
            )
    
    for species, data_points in all_derivative_data.items():
        if data_points:
            arr = np.array(data_points, dtype=float)
            np.savetxt(
                os.path.join(plot_outdir, f'qfi_derivative_points_{species}.dat'),
                arr, header='Jpm beta dQFI_dbeta'
            )


def process_species_heatmap(species, data_points, plot_outdir, data_type, ref_target):
    """Process and create heatmaps for a single species."""
    
    print(f"\n{'='*60}")
    print(f"Processing heatmap for species: {species}, type: {data_type}")
    print(f"Number of data points: {len(data_points)}")
    
    # Convert data to array
    arr = np.array(data_points, dtype=float)
    jpm_vals, beta_vals, values = arr[:, 0], arr[:, 1], arr[:, 2]
    
    print(f"Jpm range: [{jpm_vals.min():.3f}, {jpm_vals.max():.3f}]")
    print(f"Beta range: [{beta_vals.min():.3f}, {beta_vals.max():.3f}]")
    print(f"Value range: [{values.min():.3f}, {values.max():.3f}]")
    
    # Get beta grid
    target_beta = get_beta_grid(jpm_vals, beta_vals, ref_target)
    if target_beta.size < 2:
        print(f"WARNING: Insufficient beta grid size ({target_beta.size}). Skipping species.")
        return
    
    print(f"Target beta grid size: {target_beta.size}")
    
    # Split into positive and negative Jpm
    jpm_neg, jpm_pos = split_jpm_values(jpm_vals)
    
    print(f"Negative Jpm values: {jpm_neg.size}, Positive Jpm values: {jpm_pos.size}")
    
    # Interpolate data onto regular grid
    Z_neg, Z_pos = interpolate_to_grid(
        jpm_vals, beta_vals, values, 
        jpm_neg, jpm_pos, target_beta
    )
    
    if Z_neg is not None:
        print(f"Z_neg shape: {Z_neg.shape}, NaN count: {np.isnan(Z_neg).sum()}")
    if Z_pos is not None:
        print(f"Z_pos shape: {Z_pos.shape}, NaN count: {np.isnan(Z_pos).sum()}")
    
    # Save grids
    save_grids(species, target_beta, jpm_neg, jpm_pos, 
               Z_neg, Z_pos, plot_outdir, data_type)
    
    # Filter rows with no NaN
    filtered_data = filter_nan_rows(target_beta, Z_neg, Z_pos, True)
    
    print(f"Filtered data keys: {filtered_data.keys()}")
    
    # Save filtered grids
    save_filtered_grids(species, filtered_data, jpm_neg, jpm_pos, 
                       plot_outdir, data_type)
    
    # Create plots
    try:
        create_heatmap_plots(species, filtered_data, jpm_neg, jpm_pos, 
                            plot_outdir, data_type)
        print(f"Successfully created heatmap plots for {species}")
    except Exception as e:
        print(f"ERROR creating heatmap plots for {species}: {e}")
        import traceback
        traceback.print_exc()
    
    # Create line plots at fixed beta
    try:
        create_fixed_beta_plots(species, target_beta, Z_neg, Z_pos, 
                               jpm_neg, jpm_pos, filtered_data, 
                               plot_outdir, data_type)
        print(f"Successfully created fixed beta plots for {species}")
    except Exception as e:
        print(f"ERROR creating fixed beta plots for {species}: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"{'='*60}\n")


def get_beta_grid(jpm_vals, beta_vals, ref_target):
    """Create beta grid based on reference Jpm value."""
    unique_jpm = np.unique(jpm_vals)
    if unique_jpm.size == 0:
        return np.array([])
    
    # Find Jpm closest to reference target
    ref_jpm = unique_jpm[np.argmin(np.abs(unique_jpm - ref_target))]
    ref_mask = np.isclose(jpm_vals, ref_jpm, rtol=1e-8, atol=1e-12)
    
    beta_ref = beta_vals[ref_mask]
    beta_ref = beta_ref[beta_ref > 0]
    target_beta = np.unique(beta_ref)
    target_beta.sort()
    
    # Fallback to all positive betas if reference grid insufficient
    if target_beta.size < 2:
        target_beta = np.unique(beta_vals[beta_vals > 0])
        target_beta.sort()
    
    return target_beta


def split_jpm_values(jpm_vals):
    """Split Jpm values into negative and positive arrays."""
    jpm_neg = np.unique(jpm_vals[jpm_vals < 0])
    jpm_neg.sort()
    jpm_pos = np.unique(jpm_vals[jpm_vals > 0])
    jpm_pos.sort()
    return jpm_neg, jpm_pos


def interpolate_to_grid(jpm_vals, beta_vals, values, jpm_neg, jpm_pos, target_beta):
    """Interpolate data onto regular beta grid for each Jpm value."""
    
    print(f"[DEBUG] interpolate_to_grid: target_beta size={len(target_beta)}, jpm_neg size={len(jpm_neg)}, jpm_pos size={len(jpm_pos)}")
    
    def interp_at_jpm(j):
        mask = np.isclose(jpm_vals, j, rtol=1e-8, atol=1e-12)
        b = beta_vals[mask]
        v = values[mask]
        
        if b.size == 0:
            print(f"[DEBUG] Jpm={j:.3f}: No data points found")
            return np.full_like(target_beta, np.nan, dtype=float)
        
        # Sort and average duplicates
        order = np.argsort(b)
        b, v = b[order], v[order]
        
        bu, inv = np.unique(b, return_inverse=True)
        v_mean = np.zeros_like(bu, dtype=float)
        counts = np.zeros_like(bu, dtype=int)
        np.add.at(v_mean, inv, v)
        np.add.at(counts, inv, 1)
        v_mean = v_mean / np.maximum(counts, 1)
        
        if bu.size < 2:
            print(f"[DEBUG] Jpm={j:.3f}: Insufficient unique beta values ({bu.size}), returning NaN")
            return np.full_like(target_beta, np.nan, dtype=float)
        
        f = interp1d(bu, v_mean, kind='linear', bounds_error=False, fill_value=np.nan)
        result = f(target_beta)
        nan_count = np.isnan(result).sum()
        if nan_count > 0:
            print(f"[DEBUG] Jpm={j:.3f}: Interpolation produced {nan_count} NaN values out of {len(result)}")
        return result
    
    Z_neg = np.column_stack([interp_at_jpm(j) for j in jpm_neg]) if jpm_neg.size > 0 else None
    Z_pos = np.column_stack([interp_at_jpm(j) for j in jpm_pos]) if jpm_pos.size > 0 else None
    
    if Z_neg is not None:
        print(f"[DEBUG] Z_neg created: shape={Z_neg.shape}, NaN count={np.isnan(Z_neg).sum()}")
    if Z_pos is not None:
        print(f"[DEBUG] Z_pos created: shape={Z_pos.shape}, NaN count={np.isnan(Z_pos).sum()}")
    
    return Z_neg, Z_pos


def filter_nan_rows(target_beta, Z_neg, Z_pos, naive = False):
    """Filter out rows containing NaN values."""
    result = {}

    if naive:
        result['Z_neg_f'] = Z_neg
        result['Z_pos_f'] = Z_pos
        result['beta_neg_f'] = target_beta
        result['beta_pos_f'] = target_beta
        # Create masks for naive mode too (all True if data exists)
        if Z_neg is not None and Z_neg.size > 0:
            result['mask_neg'] = np.ones(len(target_beta), dtype=bool)
        if Z_pos is not None and Z_pos.size > 0:
            result['mask_pos'] = np.ones(len(target_beta), dtype=bool)
        return result
    else:
        if Z_neg is not None and Z_neg.size > 0:
            mask_neg = np.all(np.isfinite(Z_neg), axis=1)
            if np.any(mask_neg):
                result['Z_neg_f'] = Z_neg[mask_neg, :]
                result['beta_neg_f'] = target_beta[mask_neg]
                result['mask_neg'] = mask_neg
        
        if Z_pos is not None and Z_pos.size > 0:
            mask_pos = np.all(np.isfinite(Z_pos), axis=1)
            if np.any(mask_pos):
                result['Z_pos_f'] = Z_pos[mask_pos, :]
                result['beta_pos_f'] = target_beta[mask_pos]
                result['mask_pos'] = mask_pos
        return result


def save_grids(species, target_beta, jpm_neg, jpm_pos, Z_neg, Z_pos, plot_outdir, data_type):
    """Save unfiltered grid data."""
    prefix = 'qfi' if data_type == 'qfi' else 'qfi_derivative'
    
    if Z_neg is not None:
        save_grid_data(plot_outdir, f'{prefix}_grid_neg_{species}', 
                      target_beta, jpm_neg, Z_neg)
    
    if Z_pos is not None:
        save_grid_data(plot_outdir, f'{prefix}_grid_pos_{species}',
                      target_beta, jpm_pos, Z_pos)


def save_filtered_grids(species, filtered_data, jpm_neg, jpm_pos, plot_outdir, data_type):
    """Save filtered grid data."""
    prefix = 'qfi' if data_type == 'qfi' else 'qfi_derivative'
    
    if 'Z_neg_f' in filtered_data:
        save_grid_data(plot_outdir, f'{prefix}_grid_neg_filtered_{species}',
                      filtered_data['beta_neg_f'], jpm_neg, filtered_data['Z_neg_f'])
    
    if 'Z_pos_f' in filtered_data:
        save_grid_data(plot_outdir, f'{prefix}_grid_pos_filtered_{species}',
                      filtered_data['beta_pos_f'], jpm_pos, filtered_data['Z_pos_f'])


def save_grid_data(plot_outdir, base_name, beta, jpm, Z):
    """Save grid data in both npz and text formats."""
    np.savez(os.path.join(plot_outdir, f'{base_name}.npz'),
             beta=beta, jpm=jpm, Z=Z)
    
    # Save as text
    header_cols = ['beta'] + [f'Jpm={v:g}' for v in jpm]
    header = ' '.join(header_cols)
    out = np.column_stack((beta, Z))
    np.savetxt(os.path.join(plot_outdir, f'{base_name}.dat'), 
               out, header=header)


def create_heatmap_plots(species, filtered_data, jpm_neg, jpm_pos, plot_outdir, data_type):
    """Create heatmap plots for the species."""
    
    # Get color scale limits
    vmin, vmax = get_color_limits(filtered_data)
    if vmin is None:
        return
    
    # Labels
    value_label = 'QFI' if data_type == 'qfi' else 'dQFI/dβ'
    prefix = 'qfi' if data_type == 'qfi' else 'qfi_derivative'
    
    # Plot negative Jpm heatmap
    if 'Z_neg_f' in filtered_data:
        plot_single_heatmap(
            jpm_neg, filtered_data['beta_neg_f'], filtered_data['Z_neg_f'],
            vmin, vmax, value_label, 
            f'{value_label} Heatmap (Jpm<0) for {species}',
            os.path.join(plot_outdir, f'{prefix}_heatmap_neg_{species}.png')
        )
    
    # Plot positive Jpm heatmap
    if 'Z_pos_f' in filtered_data:
        plot_single_heatmap(
            jpm_pos, filtered_data['beta_pos_f'], filtered_data['Z_pos_f'],
            vmin, vmax, value_label,
            f'{value_label} Heatmap (Jpm>0) for {species}',
            os.path.join(plot_outdir, f'{prefix}_heatmap_pos_{species}.png')
        )
    
    # Plot side-by-side view
    if 'Z_neg_f' in filtered_data and 'Z_pos_f' in filtered_data:
        plot_side_by_side_heatmap(
            species, filtered_data, jpm_neg, jpm_pos,
            vmin, vmax, value_label, plot_outdir, prefix
        )


def get_color_limits(filtered_data):
    """Get unified color scale limits."""
    z_list = []
    if 'Z_neg_f' in filtered_data:
        z_list.append(filtered_data['Z_neg_f'])
    if 'Z_pos_f' in filtered_data:
        z_list.append(filtered_data['Z_pos_f'])
    
    if not z_list:
        return None, None
    
    vmin = np.nanmin([np.nanmin(z) for z in z_list])
    vmax = np.nanmax([np.nanmax(z) for z in z_list])
    return vmin, vmax


def plot_single_heatmap(jpm, beta, Z, vmin, vmax, value_label, title, filename):
    """Create a single heatmap plot."""
    print(f"[DEBUG] plot_single_heatmap: jpm shape={jpm.shape}, beta shape={beta.shape}, Z shape={Z.shape}")
    print(f"[DEBUG] vmin={vmin:.6f}, vmax={vmax:.6f}")
    
    J, B = np.meshgrid(jpm, beta)
    print(f"[DEBUG] Meshgrid J shape={J.shape}, B shape={B.shape}")
    
    plt.figure(figsize=(12, 8))
    try:
        plt.pcolormesh(J, B, Z, shading='auto', cmap='viridis', vmin=vmin, vmax=vmax)
        plt.yscale('log')
        plt.gca().invert_yaxis()  # large beta at bottom
        plt.xlabel('Jpm')
        plt.ylabel('Beta (β)')
        plt.title(title)
        plt.colorbar(label=value_label)
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"[DEBUG] Successfully saved heatmap to {filename}")
    except Exception as e:
        print(f"[ERROR] Failed to create heatmap: {e}")
        import traceback
        traceback.print_exc()
    finally:
        plt.close()


def plot_side_by_side_heatmap(species, filtered_data, jpm_neg, jpm_pos, 
                              vmin, vmax, value_label, plot_outdir, prefix):
    """Create side-by-side heatmap plot."""
    
    beta_neg_f = filtered_data['beta_neg_f']
    beta_pos_f = filtered_data['beta_pos_f']
    Z_neg_f = filtered_data['Z_neg_f']
    Z_pos_f = filtered_data['Z_pos_f']
    
    # Create meshgrids
    JN, BN = np.meshgrid(jpm_neg, beta_neg_f)
    JP, BP = np.meshgrid(jpm_pos, beta_pos_f)
    
    # Calculate y-axis limits
    y_min = float(min(beta_neg_f.min(), beta_pos_f.min()))
    y_max = float(max(beta_neg_f.max(), beta_pos_f.max()))
    
    # Calculate width ratio
    wr = calculate_width_ratio(jpm_neg, jpm_pos)
    
    # Create figure
    fig, (axL, axR) = plt.subplots(
        1, 2, figsize=(14, 8), sharey=True,
        gridspec_kw={'wspace': 0.0, 'hspace': 0.0, 'width_ratios': [wr, 1.0]}
    )
    
    # Plot heatmaps
    axL.pcolormesh(JN, BN, Z_neg_f, shading='auto', cmap='viridis', vmin=vmin, vmax=vmax)
    axR.pcolormesh(JP, BP, Z_pos_f, shading='auto', cmap='viridis', vmin=vmin, vmax=vmax)
    
    # Format axes
    for ax in (axL, axR):
        ax.set_yscale('log')
        ax.set_ylim(y_max, y_min)  # large beta at bottom
        ax.set_xlabel('Jpm')
    
    axL.set_ylabel('Beta (β)')
    axR.tick_params(labelleft=False)
    
    # Add colorbar
    fig.subplots_adjust(wspace=0.0)
    cbar = fig.colorbar(axL.collections[0], ax=[axL, axR], location='right', pad=0.02)
    cbar.set_label(value_label)
    
    fig.suptitle(f'{value_label} Heatmap (Jpm<0 | Jpm>0) for {species}')
    fig.savefig(os.path.join(plot_outdir, f'{prefix}_heatmap_side_by_side_{species}.png'),
                dpi=300, bbox_inches='tight')
    plt.close()


def calculate_width_ratio(jpm_neg, jpm_pos):
    """Calculate width ratio for side-by-side plots."""
    neg_span = float(np.ptp(jpm_neg)) if jpm_neg.size > 0 else 0.0
    pos_span = float(np.ptp(jpm_pos)) if jpm_pos.size > 0 else 0.0
    
    if pos_span > 0:
        wr = neg_span / pos_span
    else:
        wr = 1.5
    
    return float(np.clip(wr, 1.2, 3.0))


def create_fixed_beta_plots(species, target_beta, Z_neg, Z_pos, jpm_neg, jpm_pos, 
                           filtered_data, plot_outdir, data_type):
    """Create line plots at fixed beta values."""
    
    # Find largest valid beta indices
    idx_neg = find_largest_valid_beta_index(Z_neg, filtered_data.get('mask_neg'))
    idx_pos = find_largest_valid_beta_index(Z_pos, filtered_data.get('mask_pos'))
    
    if idx_neg is None and idx_pos is None:
        return
    
    # Setup plot
    plt.figure(figsize=(10, 6))
    color = 'C0' if data_type == 'qfi' else 'C1'
    value_label = 'QFI' if data_type == 'qfi' else 'dQFI/dβ'
    prefix = 'qfi' if data_type == 'qfi' else 'qfi_derivative'
    
    # Plot negative Jpm segment
    if idx_neg is not None:
        plot_fixed_beta_segment(
            jpm_neg, Z_neg[idx_neg, :], target_beta[idx_neg],
            color, 'neg', plot_outdir, species, prefix
        )
    
    # Plot positive Jpm segment
    if idx_pos is not None:
        plot_fixed_beta_segment(
            jpm_pos, Z_pos[idx_pos, :], target_beta[idx_pos],
            color, 'pos', plot_outdir, species, prefix
        )
    
    plt.xlabel('Jpm')
    plt.ylabel(value_label)
    plt.title(f'{value_label} vs Jpm at largest β rows (no NaN) for {species}')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=9)
    
    fname = os.path.join(plot_outdir, f'{prefix}_vs_jpm_fixed_beta_{species}.png')
    plt.savefig(fname, dpi=300, bbox_inches='tight')
    plt.close()


def find_largest_valid_beta_index(Z, mask):
    """Find the largest beta index with no NaN values."""
    if Z is None or Z.size == 0 or mask is None or not np.any(mask):
        return None
    
    valid_idxs = np.where(mask)[0]
    return valid_idxs[-1] if valid_idxs.size > 0 else None


def plot_fixed_beta_segment(jpm, values, beta, color, segment, plot_outdir, species, prefix):
    """Plot a segment of fixed beta data."""
    plt.plot(jpm, values, '-', lw=1.8, color=color, label=f'β={beta:.3g} ({segment})')
    
    # Save data
    filename = os.path.join(plot_outdir, f'{prefix}_vs_jpm_fixed_beta_{segment}_{species}.dat')
    header = 'Jpm beta ' + ('QFI' if 'qfi' in prefix and 'derivative' not in prefix else 'dQFI_dbeta')
    np.savetxt(filename, np.column_stack((jpm, np.full_like(jpm, beta), values)), header=header)


if __name__ == "__main__":
    # Path to the directory containing the data files
    data_dir = sys.argv[1] if len(sys.argv) > 1 else 'data'
    across_QFI = sys.argv[2] if len(sys.argv) > 2 else 'False'
    across_QFI = across_QFI.lower() == 'true'
    if across_QFI:
        parse_QFI_across_Jpm(data_dir)
        parse_QFI_across_hi(data_dir)
        plot_heatmaps_from_processed_data(data_dir)
    else:
        parse_QFI_data_new(data_dir)
    print("All processing complete.")