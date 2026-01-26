"""
Calculate Quantum Fisher Information (QFI) from spectral function data.

This script supports both time-domain correlation data (performing FFT) and
pre-computed frequency-domain spectral data.

Supports two data sources:
1. HDF5 files (preferred): ed_results.h5 with time_correlations data
2. Text files (legacy): {species}_spectral_sample_{N}_beta_{beta}.txt

New HDF5 structure (time-domain correlations):
    /dynamical/time_correlations/<group_name>/
        times            - array of time values
        correlation_real - Re[C(t)]
        correlation_imag - Im[C(t)]
        Attributes: beta, sample_index, operator, label
    Group naming: {operator}_sample{N}_beta{value}_tpq
    Example: SzSz_q_Qx0_Qy0_Qz0_sample0_beta111.8182_tpq

Legacy HDF5 structure (frequency-domain spectral):
    /dynamical/<operator_name>/
        frequencies     - array of omega values
        spectral_real   - Re[S(q,ω)]

Text file format:
    # Header lines (ignored)
    omega  Re[S(q,ω)]  Im[S(q,ω)]  Re[error]  Im[error]
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
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d

# Try to import h5py, but make it optional
try:
    import h5py
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False
    print("Warning: h5py not available, HDF5 reading disabled")

# Try to import mpi4py, but make it optional
# try:
#     from mpi4py import MPI
#     HAS_MPI = True
# except ImportError:
#     HAS_MPI = False

from mpi4py import MPI
HAS_MPI = True
# NumPy compatibility: use trapezoid (new) or trapz (old)
if hasattr(np, 'trapezoid'):
    np_trapz = np.trapezoid
else:
    np_trapz = np.trapz

ENABLE_BETA_EXTENSION = True
EXTENDED_BETA_MAX = 1000.0  # Maximum beta for extended calculations

# ==============================================================================
# Excluded parameter values (corrupted data)
# ==============================================================================
EXCLUDED_JPM_VALUES = [-0.425, -0.35, -0.3, 0.08]  # These Jpm values appear to be corrupted
QFI_SCALE_FACTOR = 1  # Multiply all QFI values by this factor

# ==============================================================================
# Spectral Width Rescaling Configuration
# ==============================================================================
# Enable spectral width rescaling to correct for insufficient Lanczos space
# at high temperatures. When enabled, the lowest temperature spectrum is used
# as reference for the correct spectral width, and all other spectra are
# rescaled to match this width before QFI calculation.
ENABLE_SPECTRAL_RESCALING = False  # Set to False to disable rescaling

# ==============================================================================
# Spectral Intensity Scaling Configuration
# ==============================================================================
# Enable spectral intensity scaling to ensure the integral of the spectral
# function matches the static structure factor at each temperature.
# The static structure factor is read from sssf_*_expectation.txt files.
ENABLE_INTENSITY_SCALING = False  # Set to False to disable intensity scaling

# ==============================================================================
# Aggressive Rescaling Configuration
# ==============================================================================
# At high temperatures, the Lanczos-induced spectral broadening is more severe.
# This factor controls how aggressively to compress the spectrum.
# - 1.0: Standard rescaling (just match reference width)
# - >1.0: More aggressive compression (recommended: 1.5-2.0 for high-T corrections)
# The effective scaling is: scale_factor^aggressive_factor
SPECTRAL_RESCALE_AGGRESSIVE_FACTOR = 1  # Base factor (used at intermediate temps)

# Temperature-dependent rescaling: interpolates between these values based on beta
# At low T (high beta): use AGGRESSIVE_FACTOR_LOW_T
# At high T (low beta): use AGGRESSIVE_FACTOR_HIGH_T
AGGRESSIVE_FACTOR_LOW_T = 1.0   # At lowest temperature (highest beta)
AGGRESSIVE_FACTOR_HIGH_T = 1.0  # At highest temperature (lowest beta) - more aggressive

# Beta thresholds for temperature-dependent rescaling
# Below this beta (high T), use maximum aggressive factor
BETA_THRESHOLD_HIGH_T = 5.0
# Above this beta (low T), use minimum aggressive factor  
BETA_THRESHOLD_LOW_T = 50.0

# ==============================================================================
# NaN Interpolation Configuration
# ==============================================================================
# Enable interpolation of NaN values in spectral data
ENABLE_NAN_INTERPOLATION = True  # Set to False to disable NaN interpolation

# Use moment-based width (standard deviation) instead of threshold-based width
# Moment-based is more robust to noise and numerical artifacts
USE_MOMENT_BASED_WIDTH = False

# ==============================================================================
# Post-hoc Spectral Smoothing Configuration
# ==============================================================================
# Enable Gaussian smoothing of spectral function to reduce noise.
# This is useful when the original Lanczos broadening was too small,
# resulting in noisy spectral functions.
ENABLE_SPECTRAL_SMOOTHING = True  # Set to True to enable smoothing

# Gaussian smoothing width (sigma) in units of omega.
# Typical values: 0.01-0.1 depending on the frequency resolution.
# Larger values = more smoothing but may smear out real features.
SPECTRAL_SMOOTHING_SIGMA = 0.015

# ==============================================================================
# High-Beta Spectral Weight Normalization Configuration
# ==============================================================================
# At very high beta (low temperature), numerical instabilities in the Lanczos
# calculation can cause loss of spectral weight. This option enables normalization
# of high-beta spectral functions to match a reference spectral integral from
# a more reliable (lower) beta value.
ENABLE_HIGH_BETA_NORMALIZATION = False  # Set to True to enable normalization

# Beta threshold above which normalization is applied.
# Spectral functions with beta > this value will be normalized.
HIGH_BETA_NORMALIZATION_THRESHOLD = 250.0

# ==============================================================================
# High-Beta QFI Monotonicity Enforcement
# ==============================================================================
# At very high beta (low temperature), numerical instabilities can cause QFI to
# artificially decrease. This option enforces that QFI is monotonically non-decreasing
# with increasing beta (decreasing temperature) above a threshold.
ENABLE_HIGH_BETA_QFI_FLOOR = False  # Set to True to enable QFI floor enforcement

# Beta threshold above which QFI floor is enforced
# For beta > this value, QFI will be at least as large as the max QFI seen at lower beta
HIGH_BETA_QFI_FLOOR_THRESHOLD = 230.0

# ==============================================================================
# FFT and Spectral Function Utilities
# ==============================================================================

def time_to_spectral_fft(times, correlation_real, correlation_imag, 
                          omega_max=10.0, n_omega=1000):
    """
    Convert time-domain correlation function to frequency-domain spectral function.
    
    Uses FFT to compute S(ω) from C(t):
        S(ω) = ∫ C(t) e^{iωt} dt
    
    Parameters:
    times: Array of time values
    correlation_real: Re[C(t)]
    correlation_imag: Im[C(t)]
    omega_max: Maximum frequency for output
    n_omega: Number of frequency points
    
    Returns: (omega_array, spectral_function_array)
    """
    # Construct complex correlation
    correlation = correlation_real + 1j * correlation_imag
    
    # Compute FFT
    dt = times[1] - times[0] if len(times) > 1 else 1.0
    n = len(times)
    
    # Use numpy FFT
    fft_vals = np.fft.fft(correlation) * dt
    freqs = np.fft.fftfreq(n, dt) * 2 * np.pi  # Convert to angular frequency
    
    # Sort by frequency
    sort_idx = np.argsort(freqs)
    freqs = freqs[sort_idx]
    fft_vals = fft_vals[sort_idx]
    
    # Interpolate to uniform omega grid if needed
    omega = np.linspace(-omega_max, omega_max, n_omega)
    
    # Only keep frequencies within range
    mask = (freqs >= -omega_max) & (freqs <= omega_max)
    if np.sum(mask) > 2:
        from scipy.interpolate import interp1d
        interp_real = interp1d(freqs[mask], np.real(fft_vals[mask]), 
                               kind='linear', fill_value=0.0, bounds_error=False)
        spectral = interp_real(omega)
    else:
        spectral = np.real(fft_vals[:n_omega]) if len(fft_vals) >= n_omega else np.zeros(n_omega)
        omega = freqs[:n_omega] if len(freqs) >= n_omega else np.linspace(-omega_max, omega_max, n_omega)
    
    return omega, spectral


# ==============================================================================
# HDF5 Support Functions
# ==============================================================================

def load_time_correlation_from_hdf5(h5_path, group_name):
    """
    Load time-domain correlation data from new HDF5 structure.
    
    Parameters:
    h5_path: Path to ed_results.h5 file
    group_name: Name of the group in /dynamical/time_correlations/
    
    Returns: (times, correlation_real, correlation_imag, beta) or (None, None, None, None)
    """
    if not HAS_H5PY:
        return None, None, None, None
    
    try:
        with h5py.File(h5_path, 'r') as f:
            if 'dynamical' not in f:
                return None, None, None, None
            
            dyn_group = f['dynamical']
            
            # Check for time_correlations subgroup
            if 'time_correlations' not in dyn_group:
                return None, None, None, None
            
            tc_group = dyn_group['time_correlations']
            if group_name not in tc_group:
                return None, None, None, None
            
            grp = tc_group[group_name]
            times = grp['times'][:]
            corr_real = grp['correlation_real'][:]
            corr_imag = grp['correlation_imag'][:]
            
            # Get beta from attribute
            beta = grp.attrs.get('beta', np.nan)
            
            return times, corr_real, corr_imag, beta
            
    except Exception as e:
        print(f"Error loading HDF5 time correlation {h5_path}/{group_name}: {e}")
        return None, None, None, None


def load_spectral_from_hdf5(h5_path, dataset_name):
    """
    Load spectral function data from HDF5 file.
    
    Supports two formats:
    1. New: time_correlations groups (converts via FFT)
    2. Legacy: direct spectral data with frequencies/spectral_real
    
    Parameters:
    h5_path: Path to ed_results.h5 file
    dataset_name: Name of the dataset/group in /dynamical/ hierarchy
    
    Returns: (omega_array, spectral_function_array) or (None, None)
    """
    if not HAS_H5PY:
        return None, None
    
    try:
        with h5py.File(h5_path, 'r') as f:
            if 'dynamical' not in f:
                return None, None
            
            dyn_group = f['dynamical']
            
            # First, try new time_correlations format
            if 'time_correlations' in dyn_group:
                tc_group = dyn_group['time_correlations']
                if dataset_name in tc_group:
                    grp = tc_group[dataset_name]
                    times = grp['times'][:]
                    corr_real = grp['correlation_real'][:]
                    corr_imag = grp['correlation_imag'][:]
                    
                    # Convert to spectral via FFT
                    omega, spectral = time_to_spectral_fft(times, corr_real, corr_imag)
                    return omega, spectral
            
            # Fall back to legacy format: direct spectral data
            if dataset_name in dyn_group:
                ds = dyn_group[dataset_name]
                if 'frequencies' in ds and 'spectral_real' in ds:
                    omega = ds['frequencies'][:]
                    spectral = ds['spectral_real'][:]
                    return omega, spectral
            
            return None, None
            
    except Exception as e:
        print(f"Error loading HDF5 {h5_path}/{dataset_name}: {e}")
        return None, None


def list_spectral_datasets_hdf5(h5_path):
    """
    List all spectral datasets available in an HDF5 file.
    
    Supports both new time_correlations format and legacy spectral format.
    
    Parameters:
    h5_path: Path to ed_results.h5 file
    
    Returns: List of (dataset_name, species, beta, sample_idx) tuples
    """
    if not HAS_H5PY:
        return []
    
    datasets = []
    try:
        with h5py.File(h5_path, 'r') as f:
            if 'dynamical' not in f:
                return []
            
            dyn_group = f['dynamical']
            
            # First check new time_correlations format
            if 'time_correlations' in dyn_group:
                tc_group = dyn_group['time_correlations']
                for name in tc_group.keys():
                    # Parse new format: {operator}_sample{N}_beta{value}_tpq
                    species, beta, sample_idx = parse_time_correlation_name(name)
                    if species is not None:
                        datasets.append((name, species, beta, sample_idx))
            
            # Also check legacy format (direct children of dynamical)
            for name in dyn_group.keys():
                if name in ('samples', 'time_correlations'):
                    continue
                
                # Check if this is a group with spectral data
                if isinstance(dyn_group[name], h5py.Group):
                    grp = dyn_group[name]
                    if 'frequencies' in grp and 'spectral_real' in grp:
                        # Parse legacy format: {species}_spectral_sample_{N}_beta_{beta}
                        species, beta, sample_idx = parse_spectral_dataset_name(name)
                        if species is not None:
                            datasets.append((name, species, beta, sample_idx))
                    
    except Exception as e:
        print(f"Error reading HDF5 {h5_path}: {e}")
    
    return datasets


def load_spectral_from_dssf_hdf5(h5_path, operator_name, temperature_or_beta, sample_idx=None):
    """
    Load spectral function data from dssf_results.h5 file.
    
    Structure: /spectral/<operator_name>/<T_value or beta_value>/sample_<idx>/real,imag
    
    Parameters:
    h5_path: Path to dssf_results.h5 file
    operator_name: Name of operator (e.g., 'SzSz_q_Qx0_Qy0_Qz0')
    temperature_or_beta: Temperature or beta group name (e.g., 'T_0.500000' or 'beta_8.93617')
    sample_idx: If provided, load only this specific sample. If None, average all samples.
    
    Returns: (omega_array, spectral_function_array, beta_value) or (None, None, None)
    """
    if not HAS_H5PY:
        return None, None, None
    
    try:
        with h5py.File(h5_path, 'r') as f:
            # Get frequencies (shared across all operators)
            if '/spectral/frequencies' not in f:
                print(f"Warning: No frequencies dataset in {h5_path}")
                return None, None, None
            
            frequencies = f['/spectral/frequencies'][:]
            
            # Construct path to spectral data
            spectral_path = f'/spectral/{operator_name}/{temperature_or_beta}'
            if spectral_path not in f:
                print(f"Warning: Path {spectral_path} not found in {h5_path}")
                return None, None, None
            
            group = f[spectral_path]
            
            # If sample_idx is provided, load only that sample
            if sample_idx is not None:
                sample_key = f'sample_{sample_idx}'
                if sample_key not in group:
                    print(f"Warning: {sample_key} not found in {spectral_path}")
                    return None, None, None
                
                sample_group = group[sample_key]
                spectral = sample_group['real'][:]
            else:
                # Collect all samples and average
                all_spectral = []
                sample_keys = [k for k in group.keys() if k.startswith('sample_')]
                
                for sample_key in sample_keys:
                    sample_group = group[sample_key]
                    real_part = sample_group['real'][:]
                    all_spectral.append(real_part)
                
                if not all_spectral:
                    print(f"Warning: No samples found in {spectral_path}")
                    return None, None, None
                
                # Average over samples
                if ENABLE_NAN_INTERPOLATION:
                    # nanmean handles NaN by ignoring them
                    spectral = np.nanmean(all_spectral, axis=0)
                else:
                    # Regular mean propagates NaN values
                    spectral = np.mean(all_spectral, axis=0)
            
            # Extract beta value from temperature/beta group name
            beta = None
            if temperature_or_beta.startswith('T_'):
                T = float(temperature_or_beta.split('_')[1])
                beta = 1.0 / T if T > 0 else np.inf
            elif temperature_or_beta.startswith('beta_'):
                beta_str = temperature_or_beta.split('_')[1]
                beta = np.inf if beta_str.lower() in ('inf', 'infty') else float(beta_str)
            
            return frequencies, spectral, beta
            
    except Exception as e:
        print(f"Error loading from dssf_results.h5: {h5_path}, {operator_name}, {temperature_or_beta}: {e}")
        return None, None, None


def list_spectral_datasets_dssf_hdf5(h5_path):
    """
    List all spectral datasets available in a dssf_results.h5 file.
    
    Returns: List of (operator_name, temp_beta_group, beta_value, sample_idx) tuples
            Each individual sample is listed separately.
    """
    if not HAS_H5PY:
        return []
    
    datasets = []
    try:
        with h5py.File(h5_path, 'r') as f:
            if '/spectral' not in f:
                return []
            
            spectral_group = f['/spectral']
            
            for operator_name in spectral_group.keys():
                if operator_name == 'frequencies':
                    continue
                
                operator_group = spectral_group[operator_name]
                if not isinstance(operator_group, h5py.Group):
                    continue
                
                for temp_beta_group in operator_group.keys():
                    tb_group = operator_group[temp_beta_group]
                    if not isinstance(tb_group, h5py.Group):
                        continue
                    
                    # Extract beta value
                    beta = None
                    if temp_beta_group.startswith('T_'):
                        T = float(temp_beta_group.split('_')[1])
                        beta = 1.0 / T if T > 0 else np.inf
                    elif temp_beta_group.startswith('beta_'):
                        beta_str = temp_beta_group.split('_')[1]
                        beta = np.inf if beta_str.lower() in ('inf', 'infty') else float(beta_str)
                    
                    # List each sample individually
                    sample_keys = [k for k in tb_group.keys() if k.startswith('sample_')]
                    for sample_key in sample_keys:
                        # Extract sample index from 'sample_N'
                        sample_idx = int(sample_key.split('_')[1])
                        datasets.append((operator_name, temp_beta_group, beta, sample_idx))
    
    except Exception as e:
        print(f"Error listing dssf_results.h5: {h5_path}: {e}")
    
    return datasets


def parse_time_correlation_name(name):
    """
    Parse new HDF5 time_correlations group name.
    
    Format: {operator}_sample{N}_beta{value}_tpq
    Example: SzSz_q_Qx0_Qy0_Qz0_sample0_beta111.8182_tpq
    
    Returns: (species, beta, sample_idx) or (None, None, None)
    """
    # Match pattern: {operator}_sample{N}_beta{value}_tpq
    m = re.match(r'^(.+?)_sample(\d+)_beta([0-9.+-eE]+|inf|infty)_tpq$', 
                 name, re.IGNORECASE)
    if not m:
        return None, None, None
    
    species = m.group(1)
    sample_idx = int(m.group(2))
    beta_token = m.group(3)
    
    if beta_token.lower() in ("inf", "infty"):
        beta_val = np.inf
    else:
        try:
            beta_val = float(beta_token)
        except ValueError:
            return None, None, None
    
    return species, beta_val, sample_idx


def parse_spectral_dataset_name(name):
    """
    Parse legacy HDF5 dataset name to extract species, beta, and sample index.
    
    Format: {species}_spectral_sample_{N}_beta_{beta}
    (Same as text file but without .txt extension)
    
    Returns: (species, beta, sample_idx) or (None, None, None)
    """
    # Match pattern: anything_spectral_sample{N}_beta_{beta}
    m = re.match(r'^(.+?)_spectral_sample_(\d+)_beta_([0-9.+-eE]+|inf|infty)$', 
                 name, re.IGNORECASE)
    if not m:
        return None, None, None
    
    species = m.group(1)
    sample_idx = int(m.group(2))
    beta_token = m.group(3)
    
    if beta_token.lower() in ("inf", "infty"):
        beta_val = np.inf
    else:
        try:
            beta_val = float(beta_token)
        except ValueError:
            return None, None, None
    
    return species, beta_val, sample_idx


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
    Load spectral function data from file or HDF5.
    
    Supports three formats:
    1. Text file path: /path/to/file.txt
    2. HDF5 reference: HDF5:/path/to/ed_results.h5:dataset_name
    3. DSSF HDF5 reference: DSSF_HDF5:/path/to/dssf_results.h5:operator_name:temp_beta_group
    
    Returns: (omega_array, spectral_function_array)
    """
    # Check if this is a DSSF HDF5 reference
    if filepath.startswith('DSSF_HDF5:'):
        # Parse DSSF HDF5 reference: DSSF_HDF5:<h5_path>:<operator_name>:<temp_beta_group>:<sample_idx>
        parts = filepath[10:].split(':', 3)  # Remove DSSF_HDF5: prefix and split
        if len(parts) == 4:
            h5_path, operator_name, temp_beta_group, sample_idx_str = parts
            sample_idx = int(sample_idx_str)
            omega, spectral, beta = load_spectral_from_dssf_hdf5(h5_path, operator_name, temp_beta_group, sample_idx)
            return omega, spectral
        elif len(parts) == 3:
            # Legacy format without sample_idx - average all samples
            h5_path, operator_name, temp_beta_group = parts
            omega, spectral, beta = load_spectral_from_dssf_hdf5(h5_path, operator_name, temp_beta_group)
            return omega, spectral
        else:
            print(f"Invalid DSSF HDF5 reference format: {filepath}")
            return None, None
    
    # Check if this is an HDF5 reference
    if filepath.startswith('HDF5:'):
        # Parse HDF5 reference: HDF5:<h5_path>:<dataset_name>
        parts = filepath[5:].split(':', 1)  # Remove HDF5: prefix and split
        if len(parts) == 2:
            h5_path, dataset_name = parts
            return load_spectral_from_hdf5(h5_path, dataset_name)
        else:
            print(f"Invalid HDF5 reference format: {filepath}")
            return None, None
    
    # Load from text file
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
                       min_height=None, omega_range=(0, 6), find_all_peaks=False):
    """
    Find peak positions in spectral function data.
    
    Parameters:
    omega: Frequency array
    spectral_function: Spectral function values
    min_prominence: Minimum prominence for peak detection (use very small value like 1e-10 to capture all finite peaks)
    min_height: Minimum height for peak detection (if None, disabled to capture all peaks)
    omega_range: Frequency range to search for peaks (min_freq, max_freq)
    find_all_peaks: If True, find ALL local maxima (derivative=0, convexity=down) without filtering
    
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
    
    if find_all_peaks:
        # Find ALL local maxima using derivative and convexity criteria
        peaks = find_all_local_maxima(spec_filtered)
    else:
        # Use minimal filtering to capture all finite peaks
        # Set very small prominence to capture any peak with finite prominence
        effective_prominence = min_prominence if min_prominence is not None else 1e-10
        
        # Disable height filter if min_height is None (capture all peaks regardless of height)
        if min_height is None:
            peaks, properties = find_peaks(spec_filtered, prominence=effective_prominence)
        else:
            peaks, properties = find_peaks(spec_filtered, height=min_height, prominence=effective_prominence)
    
    # Check if the first point (omega=0 boundary) is a local maximum
    # scipy.find_peaks doesn't detect boundary points, so check manually
    peaks = list(peaks)
    if len(spec_filtered) >= 2 and spec_filtered[0] > spec_filtered[1]:
        # First point is higher than second - it's a boundary maximum
        if 0 not in peaks:
            peaks.insert(0, 0)
    
    peaks = np.array(peaks)
    
    if len(peaks) == 0:
        return [], [], []
    
    # Get peak positions in original frequency units
    peak_positions = omega_filtered[peaks]
    peak_heights = spec_filtered[peaks]
    
    # Calculate prominences
    prominences, _, _ = peak_prominences(spec_filtered, peaks)
    
    return peak_positions.tolist(), peak_heights.tolist(), prominences.tolist()


def find_all_local_maxima(data):
    """
    Find ALL local maxima in a 1D array using derivative and convexity criteria.
    
    A local maximum is defined as a point where:
    1. The first derivative changes from positive to negative (or is zero)
    2. The second derivative is negative (concave down)
    
    Parameters:
    data: 1D numpy array
    
    Returns:
    peaks: Array of indices where local maxima occur
    """
    if len(data) < 3:
        return np.array([], dtype=int)
    
    # Compute first derivative (forward difference)
    d1 = np.diff(data)
    
    # Compute second derivative (central difference approximation)
    d2 = np.diff(data, n=2)
    
    # Find points where first derivative changes sign from positive to negative
    # This means d1[i] >= 0 and d1[i+1] <= 0, which corresponds to a peak at index i+1
    sign_change = (d1[:-1] >= 0) & (d1[1:] <= 0)
    
    # Also check that second derivative is negative (concave down) at these points
    # d2 is computed at indices 1 to len(data)-2 relative to original array
    # sign_change is at indices 0 to len(data)-3
    # We need to align them: sign_change[i] corresponds to peak at original index i+1
    # d2[i] is the second derivative at original index i+1
    
    # Require concave down (negative second derivative) with some tolerance for numerical noise
    concave_down = d2 < 0
    
    # Combine conditions
    is_peak = sign_change & concave_down
    
    # Convert to peak indices in original array
    # sign_change[i] being True means peak at index i+1
    peak_indices = np.where(is_peak)[0] + 1
    
    # Also check boundary condition: ensure peaks are actual local maxima
    # by verifying they are higher than their immediate neighbors
    valid_peaks = []
    for idx in peak_indices:
        if idx > 0 and idx < len(data) - 1:
            if data[idx] >= data[idx-1] and data[idx] >= data[idx+1]:
                valid_peaks.append(idx)
        elif idx == 0 and len(data) > 1 and data[idx] >= data[idx+1]:
            valid_peaks.append(idx)
        elif idx == len(data) - 1 and len(data) > 1 and data[idx] >= data[idx-1]:
            valid_peaks.append(idx)
    
    return np.array(valid_peaks, dtype=int)


# ==============================================================================
# Spectral Width Rescaling Functions
# ==============================================================================

def determine_spectral_width(omega, spectral_function, threshold_fraction=0.01):
    """
    Determine the effective spectral width from a spectral function.
    
    The spectral width is defined as the frequency range where the spectral
    function exceeds a threshold (fraction of maximum value).
    
    Parameters:
    omega: Frequency array
    spectral_function: Spectral function values
    threshold_fraction: Fraction of max value to use as threshold (default: 1%)
    
    Returns:
    (omega_min, omega_max, center, width) - boundaries and characteristics of spectral support
    """
    # Focus on positive frequencies for spectral width determination
    pos_mask = omega > 0
    omega_pos = omega[pos_mask]
    spec_pos = spectral_function[pos_mask]
    
    if len(spec_pos) == 0:
        return None, None, None, None
    
    # Find the threshold
    max_val = np.max(spec_pos)
    if max_val <= 0:
        return None, None, None, None
    
    threshold = threshold_fraction * max_val
    
    # Find indices where spectral function exceeds threshold
    above_threshold = spec_pos > threshold
    
    if not np.any(above_threshold):
        return None, None, None, None
    
    # Find the first and last indices above threshold
    indices = np.where(above_threshold)[0]
    omega_min = omega_pos[indices[0]]
    omega_max = omega_pos[indices[-1]]
    
    # Calculate spectral center of mass (first moment)
    total_weight = np_trapz(spec_pos, omega_pos)
    if total_weight > 0:
        center = np_trapz(omega_pos * spec_pos, omega_pos) / total_weight
    else:
        center = (omega_min + omega_max) / 2
    
    width = omega_max - omega_min
    
    return omega_min, omega_max, center, width


def determine_spectral_moments(omega, spectral_function):
    """
    Determine spectral moments (mean and standard deviation) from a spectral function.
    
    This provides a more robust measure of spectral width than threshold-based methods.
    
    Parameters:
    omega: Frequency array
    spectral_function: Spectral function values
    
    Returns:
    (center, std_width) - spectral center (first moment) and standard deviation (sqrt of second central moment)
    """
    # Focus on positive frequencies
    pos_mask = omega > 0
    omega_pos = omega[pos_mask]
    spec_pos = spectral_function[pos_mask]
    
    if len(spec_pos) == 0:
        return None, None
    
    # Normalize spectral function to act as a probability distribution
    total_weight = np_trapz(spec_pos, omega_pos)
    if total_weight <= 0:
        return None, None
    
    # First moment (mean frequency)
    center = np_trapz(omega_pos * spec_pos, omega_pos) / total_weight
    
    # Second central moment (variance)
    variance = np_trapz((omega_pos - center)**2 * spec_pos, omega_pos) / total_weight
    std_width = np.sqrt(variance) if variance > 0 else 0.0
    
    return center, std_width


def get_temperature_dependent_aggressive_factor(beta, ref_beta=None):
    """
    Compute temperature-dependent aggressive rescaling factor.
    
    At high temperatures (low beta), use more aggressive rescaling.
    At low temperatures (high beta), use less aggressive rescaling.
    
    Parameters:
    beta: Current inverse temperature
    ref_beta: Reference beta (highest beta / lowest temperature)
    
    Returns:
    aggressive_factor: Value between AGGRESSIVE_FACTOR_LOW_T and AGGRESSIVE_FACTOR_HIGH_T
    """
    if np.isinf(beta):
        return AGGRESSIVE_FACTOR_LOW_T
    
    # Linear interpolation in log(beta) space between thresholds
    if beta <= BETA_THRESHOLD_HIGH_T:
        # Very high temperature - maximum aggression
        return AGGRESSIVE_FACTOR_HIGH_T
    elif beta >= BETA_THRESHOLD_LOW_T:
        # Very low temperature - minimum aggression
        return AGGRESSIVE_FACTOR_LOW_T
    else:
        # Interpolate in log space
        log_beta = np.log(beta)
        log_low = np.log(BETA_THRESHOLD_HIGH_T)
        log_high = np.log(BETA_THRESHOLD_LOW_T)
        
        # Linear interpolation: high_T factor at low beta, low_T factor at high beta
        t = (log_beta - log_low) / (log_high - log_low)
        factor = AGGRESSIVE_FACTOR_HIGH_T + t * (AGGRESSIVE_FACTOR_LOW_T - AGGRESSIVE_FACTOR_HIGH_T)
        
        return factor


def interpolate_nan_values(omega, spectral_function):
    """
    Interpolate NaN values in the spectral function.
    
    Uses linear interpolation to fill in NaN values.
    
    Parameters:
    omega: Frequency array
    spectral_function: Spectral function values (may contain NaN)
    
    Returns:
    spectral_function with NaN values interpolated
    """
    if not np.any(np.isnan(spectral_function)):
        return spectral_function
    
    # Find valid (non-NaN) indices
    valid_mask = ~np.isnan(spectral_function)
    
    if not np.any(valid_mask):
        # All NaN - can't interpolate
        print("    Warning: All spectral values are NaN, returning zeros")
        return np.zeros_like(spectral_function)
    
    if np.all(valid_mask):
        # No NaN values
        return spectral_function
    
    # Interpolate
    result = spectral_function.copy()
    valid_omega = omega[valid_mask]
    valid_spec = spectral_function[valid_mask]
    
    f = interp1d(valid_omega, valid_spec, kind='linear', bounds_error=False, 
                 fill_value=(valid_spec[0], valid_spec[-1]))
    
    nan_mask = np.isnan(result)
    result[nan_mask] = f(omega[nan_mask])
    
    n_interpolated = np.sum(nan_mask)
    if n_interpolated > 0:
        print(f"    Interpolated {n_interpolated} NaN values in spectral function")
    
    return result


def smooth_spectral_function(omega, spectral_function, sigma=None):
    """
    Apply Gaussian smoothing to a spectral function.
    
    This is useful for reducing noise in spectral functions when the original
    Lanczos broadening was too small.
    
    Parameters:
    omega: Frequency array
    spectral_function: Spectral function values
    sigma: Gaussian smoothing width in units of omega (if None, uses SPECTRAL_SMOOTHING_SIGMA)
    
    Returns:
    Smoothed spectral function
    """
    if sigma is None:
        sigma = SPECTRAL_SMOOTHING_SIGMA
    
    if sigma <= 0:
        return spectral_function
    
    # Convert sigma from omega units to index units
    # Assumes uniform omega grid
    if len(omega) < 2:
        return spectral_function
    
    d_omega = omega[1] - omega[0]
    if d_omega <= 0:
        return spectral_function
    
    sigma_idx = sigma / d_omega
    
    # Apply Gaussian filter
    smoothed = gaussian_filter1d(spectral_function, sigma=sigma_idx, mode='nearest')
    
    return smoothed


def rescale_spectrum_to_reference(omega, spectral_function, ref_center, ref_width,
                                   current_center=None, current_width=None,
                                   use_moments=True, aggressive_factor=1.0):
    """
    Rescale a spectral function to match a reference spectral width.
    
    This applies a linear rescaling of the frequency axis such that the
    spectrum's center and width match the reference.
    
    Parameters:
    omega: Frequency array of the spectrum to rescale
    spectral_function: Spectral function values
    ref_center: Reference spectral center (from lowest temperature)
    ref_width: Reference spectral width (from lowest temperature)
    current_center: Current spectral center (if None, computed from data)
    current_width: Current spectral width (if None, computed from data)
    use_moments: If True, use moment-based width (more robust); if False, use threshold-based
    aggressive_factor: Additional scaling factor to apply (>1 makes rescaling more aggressive)
    
    Returns:
    (rescaled_omega, rescaled_spectral) - new omega grid and interpolated spectral values
    """
    # Determine current spectral characteristics if not provided
    if current_center is None or current_width is None:
        if use_moments:
            current_center, current_width = determine_spectral_moments(omega, spectral_function)
        else:
            _, _, current_center, current_width = determine_spectral_width(omega, spectral_function)
    
    if current_center is None or current_width is None or current_width == 0:
        print("    Warning: Could not determine spectral width, skipping rescaling")
        return omega, spectral_function
    
    if ref_width is None or ref_width == 0:
        print("    Warning: Reference width is invalid, skipping rescaling")
        return omega, spectral_function
    
    # Calculate the scaling factor with aggressive correction
    # If aggressive_factor > 1, the spectrum will be compressed more aggressively
    base_scale_factor = ref_width / current_width
    scale_factor = base_scale_factor ** aggressive_factor
    
    # Apply the rescaling: shift to center, scale, shift back to reference center
    # omega_new = ref_center + (omega - current_center) * scale_factor
    # But we want to keep the original omega grid and interpolate the spectral values
    
    # Create the inverse mapping: for each point on original grid, 
    # find where it maps from in the rescaled spectrum
    # original_omega -> rescaled_omega: omega_r = ref_center + (omega - current_center) * scale_factor
    # We want spectral at original omega, so we need to sample from the rescaled function
    # S_new(omega) = S_old(omega_old) where omega = ref_center + (omega_old - current_center) * scale_factor
    # => omega_old = current_center + (omega - ref_center) / scale_factor
    
    # For each omega in the original grid, find the corresponding omega_old
    omega_old = current_center + (omega - ref_center) / scale_factor
    
    # Interpolate the original spectrum at these new positions
    f = interp1d(omega, spectral_function, kind='linear', bounds_error=False, fill_value=0.0)
    rescaled_spectral = f(omega_old)
    
    # The rescaled spectrum is on the original omega grid
    return omega, rescaled_spectral


def get_reference_spectral_integral(file_list_by_beta, beta_threshold=None):
    """
    Find the maximum spectral integral from beta values below a threshold.
    
    This is used for high-beta normalization where numerical instabilities
    can cause loss of spectral weight. The maximum integral from reliable
    (lower) beta values is used as the reference.
    
    Parameters:
    file_list_by_beta: Dictionary mapping beta_bin_idx -> list of file paths
    beta_threshold: Only consider beta values below this threshold (default: HIGH_BETA_NORMALIZATION_THRESHOLD)
    
    Returns:
    (ref_integral, ref_beta) - reference spectral integral and the beta it came from
    """
    if beta_threshold is None:
        beta_threshold = HIGH_BETA_NORMALIZATION_THRESHOLD
    
    max_integral = -np.inf
    max_integral_beta = None
    
    for beta_bin_idx, file_list in file_list_by_beta.items():
        if not file_list:
            continue
            
        # Get beta value for this bin
        first_file = file_list[0]
        beta = None
        
        if first_file.startswith('DSSF_HDF5:'):
            parts = first_file[10:].split(':')
            if len(parts) >= 3:
                beta_group = parts[2]
                beta_match = re.search(r'beta_([0-9.+-eE]+|inf)', beta_group)
                if beta_match:
                    beta_str = beta_match.group(1)
                    beta = np.inf if beta_str.lower() == 'inf' else float(beta_str)
        else:
            parsed = parse_spectral_filename(first_file) or parse_spectral_dataset_name(
                first_file.split(':')[-1] if 'HDF5:' in first_file else first_file)
            if parsed and parsed[1] is not None:
                beta = parsed[1]
        
        if beta is None:
            continue
            
        # Skip beta values above threshold (these are the ones we want to normalize)
        if not np.isinf(beta) and beta > beta_threshold:
            continue
        
        # Load and compute spectral integral
        mean_omega, mean_spectral, _ = _load_and_average_spectral(file_list)
        if mean_omega is None:
            continue
        
        # Compute integral over positive frequencies
        pos_mask = mean_omega > 0
        if not np.any(pos_mask):
            continue
        integral = np_trapz(mean_spectral[pos_mask], mean_omega[pos_mask])
        
        if integral > max_integral:
            max_integral = integral
            max_integral_beta = beta
    
    if max_integral_beta is None:
        return None, None
    
    return max_integral, max_integral_beta


def normalize_spectral_to_reference(omega, spectral_function, ref_integral, current_integral=None):
    """
    Normalize a spectral function to match a reference spectral integral.
    
    Parameters:
    omega: Frequency array
    spectral_function: Spectral function values
    ref_integral: Reference spectral integral to match
    current_integral: Current integral (if None, computed from data)
    
    Returns:
    (normalized_spectral, scale_factor) - normalized spectral function and the scale factor used
    """
    if current_integral is None:
        pos_mask = omega > 0
        if not np.any(pos_mask):
            return spectral_function, 1.0
        current_integral = np_trapz(spectral_function[pos_mask], omega[pos_mask])
    
    if current_integral <= 0 or ref_integral <= 0:
        return spectral_function, 1.0
    
    scale_factor = ref_integral / current_integral
    return spectral_function * scale_factor, scale_factor


def get_reference_spectral_moments_from_files(file_list_by_beta):
    """
    Determine the reference spectral moments from the lowest temperature (highest beta) data.
    
    Uses moment-based width (standard deviation) which is more robust than threshold-based.
    
    Parameters:
    file_list_by_beta: Dictionary mapping beta_bin_idx -> list of file paths
    
    Returns:
    (ref_center, ref_std_width, ref_beta) - reference values from the lowest temperature
    """
    # Find the highest beta (lowest temperature)
    max_beta = -np.inf
    max_beta_bin = None
    
    for beta_bin_idx, file_list in file_list_by_beta.items():
        if file_list:
            first_file = file_list[0]
            # Try to extract beta from filename
            if first_file.startswith('DSSF_HDF5:'):
                parts = first_file[10:].split(':')
                if len(parts) >= 3:
                    beta_group = parts[2]
                    beta_match = re.search(r'beta_([0-9.+-eE]+|inf)', beta_group)
                    if beta_match:
                        beta_str = beta_match.group(1)
                        beta = np.inf if beta_str.lower() == 'inf' else float(beta_str)
                        if beta > max_beta:
                            max_beta = beta
                            max_beta_bin = beta_bin_idx
            else:
                parsed = parse_spectral_filename(first_file) or parse_spectral_dataset_name(first_file.split(':')[-1] if 'HDF5:' in first_file else first_file)
                if parsed and parsed[1] is not None:
                    beta = parsed[1]
                    if beta > max_beta:
                        max_beta = beta
                        max_beta_bin = beta_bin_idx
    
    if max_beta_bin is None:
        print("    Warning: Could not find highest beta bin for reference width")
        return None, None, None
    
    # Load data from highest beta bin
    file_list = file_list_by_beta[max_beta_bin]
    mean_omega, mean_spectral, _ = _load_and_average_spectral(file_list)
    
    if mean_omega is None:
        print("    Warning: Could not load spectral data for reference width")
        return None, None, None
    
    # Determine spectral moments (more robust than threshold-based width)
    ref_center, ref_std_width = determine_spectral_moments(mean_omega, mean_spectral)
    
    if ref_center is None:
        print("    Warning: Could not determine spectral moments from reference data")
        return None, None, None
    
    print(f"    Reference spectral moments from β={max_beta:.4g}: center={ref_center:.4f}, std_width={ref_std_width:.4f}")
    
    return ref_center, ref_std_width, max_beta


def get_reference_spectral_width_from_files(file_list_by_beta):
    """
    Determine the reference spectral width from the lowest temperature (highest beta) data.
    
    Parameters:
    file_list_by_beta: Dictionary mapping beta_bin_idx -> list of file paths
    
    Returns:
    (ref_center, ref_width, ref_beta) - reference values from the lowest temperature
    """
    # Find the highest beta (lowest temperature)
    max_beta = -np.inf
    max_beta_bin = None
    
    for beta_bin_idx, file_list in file_list_by_beta.items():
        # We need to determine the actual beta value for this bin
        # This is a bit tricky since we don't have direct access to beta values here
        # We'll load the first file and parse its beta
        if file_list:
            first_file = file_list[0]
            # Try to extract beta from filename
            if first_file.startswith('DSSF_HDF5:'):
                parts = first_file[10:].split(':')
                if len(parts) >= 3:
                    beta_group = parts[2]  # e.g., 'beta_8.93617' or 'T_0.5'
                    beta_match = re.search(r'beta_([0-9.+-eE]+|inf)', beta_group)
                    if beta_match:
                        beta_str = beta_match.group(1)
                        beta = np.inf if beta_str.lower() == 'inf' else float(beta_str)
                        if beta > max_beta:
                            max_beta = beta
                            max_beta_bin = beta_bin_idx
            else:
                # Try parsing from filename
                parsed = parse_spectral_filename(first_file) or parse_spectral_dataset_name(first_file.split(':')[-1] if 'HDF5:' in first_file else first_file)
                if parsed and parsed[1] is not None:
                    beta = parsed[1]
                    if beta > max_beta:
                        max_beta = beta
                        max_beta_bin = beta_bin_idx
    
    if max_beta_bin is None:
        print("    Warning: Could not find highest beta bin for reference width")
        return None, None, None
    
    # Load data from highest beta bin
    file_list = file_list_by_beta[max_beta_bin]
    mean_omega, mean_spectral, _ = _load_and_average_spectral(file_list)
    
    if mean_omega is None:
        print("    Warning: Could not load spectral data for reference width")
        return None, None, None
    
    # Determine spectral width
    _, _, ref_center, ref_width = determine_spectral_width(mean_omega, mean_spectral)
    
    if ref_center is None:
        print("    Warning: Could not determine spectral width from reference data")
        return None, None, None
    
    print(f"    Reference spectral width from β={max_beta:.4g}: center={ref_center:.4f}, width={ref_width:.4f}")
    
    return ref_center, ref_width, max_beta


# ==============================================================================
# Static Structure Factor Loading and Intensity Scaling
# ==============================================================================

def load_static_structure_factor(sssf_dir, operator_name):
    """
    Load static structure factor data from sssf_*_expectation.txt files.
    
    File format:
    # Header lines starting with #
    # Columns: Jpm  Temperature  expectation
    
    Parameters:
    sssf_dir: Directory containing sssf_*_expectation.txt files (typically sssf_heatmaps/)
    operator_name: Operator name to match (e.g., 'SzSz_q_Qx0_Qy0_Qz0')
    
    Returns:
    Dictionary mapping (Jpm, Temperature) -> static_structure_factor value
    or None if file not found
    """
    # Map spectral operator names to SSSF operator names
    # Spectral files use SzSz, but SSSF might use different conventions
    sssf_operator = operator_name
    
    # Try different possible filename patterns
    possible_files = [
        os.path.join(sssf_dir, f'sssf_{sssf_operator}_expectation.txt'),
        os.path.join(sssf_dir, f'sssf_{sssf_operator.replace("SzSz", "SzSz")}_expectation.txt'),
    ]
    
    # Also try with SmSp/SpSm mappings (S+S- = 2*S_transverse)
    if 'SzSz' in operator_name:
        base_q = operator_name.replace('SzSz_q_', '')
        possible_files.extend([
            os.path.join(sssf_dir, f'sssf_SzSz_q_{base_q}_expectation.txt'),
        ])
    
    sssf_file = None
    for f in possible_files:
        if os.path.exists(f):
            sssf_file = f
            break
    
    if sssf_file is None:
        return None
    
    # Parse the file
    sssf_data = {}
    try:
        with open(sssf_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                parts = line.split()
                if len(parts) >= 3:
                    try:
                        jpm = float(parts[0])
                        temp = float(parts[1])
                        expectation = float(parts[2])
                        sssf_data[(jpm, temp)] = expectation
                    except ValueError:
                        continue
        
        print(f"    Loaded {len(sssf_data)} static structure factor values from {os.path.basename(sssf_file)}")
        return sssf_data
        
    except Exception as e:
        print(f"    Warning: Error loading SSSF file {sssf_file}: {e}")
        return None


def find_closest_sssf_value(sssf_data, target_jpm, target_beta, jpm_tol=1e-4, temp_tol=0.05):
    """
    Find the closest static structure factor value for given Jpm and beta.
    
    Parameters:
    sssf_data: Dictionary mapping (Jpm, Temperature) -> SSSF value
    target_jpm: Target Jpm value (can be None for single-parameter runs)
    target_beta: Target inverse temperature
    jpm_tol: Tolerance for Jpm matching
    temp_tol: Relative tolerance for temperature matching
    
    Returns:
    (sssf_value, matched_temp) or (None, None) if no match found
    """
    if sssf_data is None or len(sssf_data) == 0:
        return None, None
    
    target_temp = 1.0 / target_beta if not np.isinf(target_beta) else 0.0
    
    best_match = None
    best_dist = np.inf
    
    for (jpm, temp), value in sssf_data.items():
        # Check Jpm match (if target_jpm is provided)
        if target_jpm is not None:
            if abs(jpm - target_jpm) > jpm_tol:
                continue
        
        # Check temperature match
        if np.isinf(target_beta):
            # For T=0 (beta=inf), look for smallest temperature
            temp_dist = temp
        else:
            # Relative temperature distance
            temp_dist = abs(temp - target_temp) / max(target_temp, 1e-6)
        
        if temp_dist < best_dist:
            best_dist = temp_dist
            best_match = (value, temp)
    
    # Check if match is within tolerance
    if best_match is not None:
        if np.isinf(target_beta):
            # Accept if temperature is reasonably small
            if best_match[1] < 0.1:  # T < 0.1 is close enough to T=0
                return best_match
        elif best_dist < temp_tol:
            return best_match
    
    return None, None


def compute_spectral_integral(omega, spectral_function):
    """
    Compute the integral of the spectral function over all frequencies.
    
    This should equal the static structure factor:
    ∫ S(ω) dω = <O†O> (static structure factor)
    
    Parameters:
    omega: Frequency array
    spectral_function: Spectral function values
    
    Returns:
    Integral value
    """
    return np_trapz(spectral_function, omega)


def scale_spectral_to_sssf(omega, spectral_function, target_sssf):
    """
    Scale spectral function intensity so its integral matches the target static structure factor.
    
    Parameters:
    omega: Frequency array
    spectral_function: Spectral function values
    target_sssf: Target static structure factor value
    
    Returns:
    (scaled_spectral_function, scale_factor)
    """
    current_integral = compute_spectral_integral(omega, spectral_function)
    
    if current_integral <= 0 or not np.isfinite(current_integral):
        print(f"    Warning: Invalid spectral integral ({current_integral}), skipping intensity scaling")
        return spectral_function, 1.0
    
    if target_sssf <= 0 or not np.isfinite(target_sssf):
        print(f"    Warning: Invalid target SSSF ({target_sssf}), skipping intensity scaling")
        return spectral_function, 1.0
    
    scale_factor = target_sssf / current_integral
    scaled_spectral = spectral_function * scale_factor
    
    return scaled_spectral, scale_factor


def get_sssf_data_for_species(structure_factor_dir, species, param_value=None):
    """
    Load static structure factor data for a given species.
    
    Searches in common locations for sssf_*_expectation.txt files.
    
    Parameters:
    structure_factor_dir: Base directory for structure factor results
    species: Species/operator name (e.g., 'SzSz_q_Qx0_Qy0_Qz0')
    param_value: Parameter value (e.g., Jpm) for parameter sweep runs
    
    Returns:
    Dictionary mapping (Jpm, Temperature) -> SSSF value, or None
    """
    # Try different possible locations for SSSF files
    possible_dirs = [
        os.path.join(structure_factor_dir, '..', 'sssf_heatmaps'),
        os.path.join(structure_factor_dir, 'sssf_heatmaps'),
        os.path.join(structure_factor_dir, '..', '..', 'sssf_heatmaps'),
        structure_factor_dir,
    ]
    
    for sssf_dir in possible_dirs:
        sssf_dir = os.path.normpath(sssf_dir)
        if os.path.exists(sssf_dir):
            sssf_data = load_static_structure_factor(sssf_dir, species)
            if sssf_data is not None and len(sssf_data) > 0:
                return sssf_data
    
    return None


def calculate_qfi_from_spectral(omega, spectral_function, beta):
    """
    Calculate quantum Fisher information from spectral function.
    
    QFI = 4 * ∫ S(ω) * tanh(βω/2) * (1 - exp(-βω)) dω
    
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
        integrand = s_omega_pos * np.tanh(beta * omega_pos / 2.0) * (1 - np.exp(-beta * omega_pos))

        # Handle any NaN or inf values
        # integrand = np.nan_to_num(integrand, nan=0.0, posinf=0.0, neginf=0.0)
    
    qfi = 4.0 * np_trapz(integrand, omega_pos)
    return qfi


def parse_QFI_data_from_spectral(structure_factor_dir, beta_tol=1e-2, param_value=None):
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
    param_value: Optional parameter value (e.g., Jpm) for intensity scaling lookup
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
                                  structure_factor_dir, all_species_qfi_data,
                                  param_value=param_value)
    
    # Step 3: Generate summary plots
    _create_summary_plots(all_species_qfi_data, structure_factor_dir)
    
    print("\nProcessing complete!")
    return all_species_qfi_data


def _collect_spectral_files(structure_factor_dir, species_data, species_names, 
                            beta_bins, beta_bin_values, beta_tol):
    """Collect and organize all spectral data from HDF5 and text files.
    
    Data sources (merged, HDF5 preferred for duplicates):
    1. HDF5 files: ed_results.h5 in operator subdirectories with /dynamical/ group
    2. Text files: *_spectral_sample*_beta_*.txt
    
    When the same species/sample/beta is found in both HDF5 and text, HDF5 is used.
    """
    
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
            h5_count = 0
            txt_count = 0
            
            # Track what we've found in HDF5 to avoid duplicates
            h5_species_samples = set()  # (species, sample_idx, beta) tuples
            
            # Try ed_results.h5 first (preferred)
            h5_path = os.path.join(op_subdir, 'ed_results.h5')
            if HAS_H5PY and os.path.exists(h5_path):
                datasets = list_spectral_datasets_hdf5(h5_path)
                for dataset_name, species, file_beta, sample_idx in datasets:
                    # Create HDF5 reference path
                    ref_path = f"HDF5:{h5_path}:{dataset_name}"
                    
                    # Use beta from dataset name (more reliable than directory name)
                    beta_bin_idx = _assign_beta_bin(file_beta, beta_bins, beta_tol)
                    beta_bin_values[beta_bin_idx].append(file_beta)
                    
                    species_data[species][beta_bin_idx].append(ref_path)
                    species_names.add(species)
                    h5_species_samples.add((species, sample_idx, file_beta))
                    h5_count += 1
            
            # Also check text files for any data not in HDF5
            spectral_files = glob.glob(os.path.join(op_subdir, '*_spectral_sample*_beta_*.txt'))
            
            for fpath in spectral_files:
                species, file_beta, sample_idx = parse_spectral_filename(fpath)
                if species is None:
                    continue
                
                # Skip if already found in HDF5
                if (species, sample_idx, file_beta) in h5_species_samples:
                    continue
                
                # Use beta from filename (more reliable than directory name)
                beta_bin_idx = _assign_beta_bin(file_beta, beta_bins, beta_tol)
                beta_bin_values[beta_bin_idx].append(file_beta)
                
                species_data[species][beta_bin_idx].append(fpath)
                species_names.add(species)
                txt_count += 1
            
            # Print summary
            if h5_count > 0 and txt_count > 0:
                print(f"  {op_name}: {h5_count} from HDF5 + {txt_count} from text files")
            elif h5_count > 0:
                print(f"  {op_name}: found {h5_count} spectral datasets in HDF5")
            elif txt_count > 0:
                print(f"  {op_name}: found {txt_count} spectral text files")
    
    # Also scan for dssf_results.h5 files in the main structure_factor_results directory
    # These files contain spectral data organized by operator and temperature/beta
    dssf_h5_path = os.path.join(structure_factor_dir, 'dssf_results.h5')
    if HAS_H5PY and os.path.exists(dssf_h5_path):
        print(f"\nFound dssf_results.h5 file, processing...")
        dssf_count = 0
        datasets = list_spectral_datasets_dssf_hdf5(dssf_h5_path)
        
        for operator_name, temp_beta_group, beta_val, sample_idx in datasets:
            if beta_val is None:
                continue
            
            # Create DSSF HDF5 reference path (now includes sample_idx)
            ref_path = f"DSSF_HDF5:{dssf_h5_path}:{operator_name}:{temp_beta_group}:{sample_idx}"
            
            # Assign to beta bin
            beta_bin_idx = _assign_beta_bin(beta_val, beta_bins, beta_tol)
            beta_bin_values[beta_bin_idx].append(beta_val)
            
            # Use operator_name as species
            species_data[operator_name][beta_bin_idx].append(ref_path)
            species_names.add(operator_name)
            dssf_count += 1
        
        print(f"  Loaded {dssf_count} spectral datasets from dssf_results.h5")


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
                              structure_factor_dir, all_species_qfi_data,
                              param_value=None):
    """Process all spectral data for a single species across different beta values.
    
    Uses spectral width rescaling to correct for insufficient Lanczos space at high temperatures.
    The lowest temperature (highest beta) spectrum is used as reference for the correct spectral width.
    
    Also applies intensity scaling to match the static structure factor at each temperature.
    
    Parameters:
    species: Species/operator name
    beta_groups: Dictionary mapping beta_bin_idx -> list of file paths
    beta_bin_values: Dictionary mapping beta_bin_idx -> list of beta values
    structure_factor_dir: Directory containing spectral data
    all_species_qfi_data: Output dictionary to store QFI results
    param_value: Optional parameter value (e.g., Jpm) for SSSF lookup
    """
    
    # First, determine the reference spectral width from the lowest temperature data
    # Only if rescaling is enabled
    if ENABLE_SPECTRAL_RESCALING:
        # Use moment-based width if configured (more robust)
        if USE_MOMENT_BASED_WIDTH:
            ref_center, ref_width, ref_beta = get_reference_spectral_moments_from_files(beta_groups)
        else:
            ref_center, ref_width, ref_beta = get_reference_spectral_width_from_files(beta_groups)
        
        if ref_center is not None:
            print(f"  Using temperature-dependent spectral width rescaling (reference from β={ref_beta:.4g})")
            print(f"  Aggressive factors: {AGGRESSIVE_FACTOR_LOW_T:.2f} (low T) to {AGGRESSIVE_FACTOR_HIGH_T:.2f} (high T)")
        else:
            print(f"  Warning: No reference spectral width available, proceeding without rescaling")
    else:
        ref_center, ref_width, ref_beta = None, None, None
        print(f"  Spectral width rescaling is disabled")
    
    # Load static structure factor data for intensity scaling
    sssf_data = None
    if ENABLE_INTENSITY_SCALING:
        sssf_data = get_sssf_data_for_species(structure_factor_dir, species, param_value)
        if sssf_data is not None:
            print(f"  Using intensity scaling from static structure factor data")
        else:
            print(f"  Warning: No static structure factor data found, proceeding without intensity scaling")
    
    # Get reference spectral integral for high-beta normalization
    ref_spectral_integral = None
    ref_integral_beta = None
    if ENABLE_HIGH_BETA_NORMALIZATION:
        ref_spectral_integral, ref_integral_beta = get_reference_spectral_integral(
            beta_groups, HIGH_BETA_NORMALIZATION_THRESHOLD)
        if ref_spectral_integral is not None:
            print(f"  Using high-beta normalization (reference integral={ref_spectral_integral:.4f} from β={ref_integral_beta:.4g})")
        else:
            print(f"  Warning: Could not determine reference spectral integral for high-beta normalization")
    
    for beta_bin_idx, file_list in beta_groups.items():
        beta = _get_bin_beta(beta_bin_values[beta_bin_idx])
        beta_label = 'inf' if np.isinf(beta) else f'{beta:.6g}'
        
        print(f"  Beta bin {beta_bin_idx} (β≈{beta_label}): {len(file_list)} files")
        
        # Load and average spectral data from all samples
        mean_omega, mean_spectral, individual_data = _load_and_average_spectral(file_list)
        
        if mean_omega is None:
            print(f"    Failed to load data")
            continue
        
        # Interpolate NaN values in the spectral data (if enabled)
        if ENABLE_NAN_INTERPOLATION:
            mean_spectral = interpolate_nan_values(mean_omega, mean_spectral)
            
            # Also interpolate individual sample data
            interpolated_individual_data = []
            for omega, spectral, fpath in individual_data:
                interp_spec = interpolate_nan_values(omega, spectral)
                interpolated_individual_data.append((omega, interp_spec, fpath))
            individual_data = interpolated_individual_data
        
        # Apply Gaussian smoothing to reduce noise (if enabled)
        if ENABLE_SPECTRAL_SMOOTHING:
            mean_spectral = smooth_spectral_function(mean_omega, mean_spectral)
            
            # Also smooth individual sample data
            smoothed_individual_data = []
            for omega, spectral, fpath in individual_data:
                smooth_spec = smooth_spectral_function(omega, spectral)
                smoothed_individual_data.append((omega, smooth_spec, fpath))
            individual_data = smoothed_individual_data
            print(f"    Applied Gaussian smoothing (σ={SPECTRAL_SMOOTHING_SIGMA})")
        
        # Apply spectral width rescaling if reference is available
        # Skip rescaling for the reference beta itself (or very close to it)
        if ref_center is not None and ref_width is not None:
            if not (np.isinf(beta) and np.isinf(ref_beta)) and abs(beta - ref_beta) > 1e-3 * ref_beta:
                # Get temperature-dependent aggressive factor
                temp_aggressive_factor = get_temperature_dependent_aggressive_factor(beta, ref_beta)
                
                # Rescale the averaged spectrum with temperature-dependent aggressive factor
                _, rescaled_spectral = rescale_spectrum_to_reference(
                    mean_omega, mean_spectral, ref_center, ref_width,
                    use_moments=USE_MOMENT_BASED_WIDTH, 
                    aggressive_factor=temp_aggressive_factor)
                
                # Also rescale individual sample data
                rescaled_individual_data = []
                for omega, spectral, fpath in individual_data:
                    _, rescaled_spec = rescale_spectrum_to_reference(
                        omega, spectral, ref_center, ref_width,
                        use_moments=USE_MOMENT_BASED_WIDTH,
                        aggressive_factor=temp_aggressive_factor)
                    rescaled_individual_data.append((omega, rescaled_spec, fpath))
                
                print(f"    Applied spectral rescaling (aggressive_factor={temp_aggressive_factor:.2f})")
            else:
                rescaled_spectral = mean_spectral
                rescaled_individual_data = individual_data
        else:
            rescaled_spectral = mean_spectral
            rescaled_individual_data = individual_data
        
        # Apply intensity scaling to match static structure factor
        if sssf_data is not None and ENABLE_INTENSITY_SCALING:
            target_sssf, matched_temp = find_closest_sssf_value(sssf_data, param_value, beta)
            if target_sssf is not None:
                # Scale the averaged spectrum
                rescaled_spectral, scale_factor = scale_spectral_to_sssf(
                    mean_omega, rescaled_spectral, target_sssf)
                
                # Also scale individual sample data
                scaled_individual_data = []
                for omega, spectral, fpath in rescaled_individual_data:
                    scaled_spec, _ = scale_spectral_to_sssf(omega, spectral, target_sssf)
                    scaled_individual_data.append((omega, scaled_spec, fpath))
                rescaled_individual_data = scaled_individual_data
                
                print(f"    Applied intensity scaling (factor={scale_factor:.4f}) to match SSSF at T={matched_temp:.4g}")
            else:
                print(f"    Warning: No matching SSSF value for β={beta_label}")
        
        # Apply high-beta spectral weight normalization if enabled
        # This corrects for numerical instabilities that cause loss of spectral weight at very low temperatures
        if ENABLE_HIGH_BETA_NORMALIZATION and ref_spectral_integral is not None:
            # Only apply to beta values above the threshold
            if not np.isinf(beta) and beta > HIGH_BETA_NORMALIZATION_THRESHOLD:
                # Normalize the averaged spectrum
                rescaled_spectral, norm_factor = normalize_spectral_to_reference(
                    mean_omega, rescaled_spectral, ref_spectral_integral)
                
                # Also normalize individual sample data
                normalized_individual_data = []
                for omega, spectral, fpath in rescaled_individual_data:
                    norm_spec, _ = normalize_spectral_to_reference(
                        omega, spectral, ref_spectral_integral)
                    normalized_individual_data.append((omega, norm_spec, fpath))
                rescaled_individual_data = normalized_individual_data
                
                print(f"    Applied high-beta normalization (factor={norm_factor:.4f})")
        
        # Calculate QFI from (rescaled) averaged spectral function
        qfi = calculate_qfi_from_spectral(mean_omega, rescaled_spectral, beta) * QFI_SCALE_FACTOR
        
        # Calculate QFI for each individual sample
        per_sample_qfi = []
        for omega, spectral, fpath in rescaled_individual_data:
            sample_qfi = calculate_qfi_from_spectral(omega, spectral, beta) * QFI_SCALE_FACTOR
            per_sample_qfi.append((sample_qfi, fpath))
        
        # Find ALL peaks in the spectral function (including weak ones)
        # Using very small prominence (1e-10) to capture all finite peaks
        peak_positions, peak_heights, peak_prominences = find_spectral_peaks(
            mean_omega, mean_spectral, min_prominence=1e-10, min_height=None, omega_range=(0, 6))
        
        # Save results
        results = {
            'omega': mean_omega,
            'spectral_function': mean_spectral,
            'qfi': qfi,
            'peak_positions': peak_positions,
            'peak_heights': peak_heights,
            'peak_prominences': peak_prominences,
            'per_sample_qfi': per_sample_qfi  # Add per-sample data
        }
        
        _save_species_results(species, beta, results, structure_factor_dir)
        
        # Store for summary plots (including per-sample data)
        all_species_qfi_data[species].append((beta, qfi))
        
        # Store per-sample QFI for parameter sweep heatmaps
        for sample_qfi, fpath in per_sample_qfi:
            # Extract sample index from filepath
            sample_key = f"{species}_sample_{_extract_sample_idx(fpath)}"
            if sample_key not in all_species_qfi_data:
                all_species_qfi_data[sample_key] = []
            all_species_qfi_data[sample_key].append((beta, sample_qfi))
        
        print(f"    QFI = {qfi:.4f}, Peaks at: {peak_positions}")
    
    # Apply high-beta QFI floor enforcement if enabled
    # This ensures QFI doesn't artificially decrease at very low temperatures
    if ENABLE_HIGH_BETA_QFI_FLOOR:
        _apply_qfi_floor(species, all_species_qfi_data, HIGH_BETA_QFI_FLOOR_THRESHOLD)


def _apply_qfi_floor(species, all_species_qfi_data, beta_threshold):
    """
    Enforce QFI floor for high beta values.
    
    For beta > threshold, QFI is set to at least the maximum QFI seen at beta <= threshold.
    This corrects for numerical instabilities that cause artificial QFI decrease at low T.
    """
    if species not in all_species_qfi_data:
        return
    
    qfi_list = all_species_qfi_data[species]
    if not qfi_list:
        return
    
    # Sort by beta (ascending)
    sorted_data = sorted(qfi_list, key=lambda x: (np.inf if np.isinf(x[0]) else x[0]))
    
    # Find max QFI at or below threshold
    max_qfi_below_threshold = 0.0
    for beta, qfi in sorted_data:
        if not np.isinf(beta) and beta <= beta_threshold:
            max_qfi_below_threshold = max(max_qfi_below_threshold, qfi)
    
    if max_qfi_below_threshold <= 0:
        # No reference found, use max of first half
        n = len(sorted_data)
        for beta, qfi in sorted_data[:n//2 + 1]:
            max_qfi_below_threshold = max(max_qfi_below_threshold, qfi)
    
    # Apply floor to high-beta values
    corrected_list = []
    corrections_made = 0
    for beta, qfi in sorted_data:
        if not np.isinf(beta) and beta > beta_threshold:
            if qfi < max_qfi_below_threshold:
                corrected_list.append((beta, max_qfi_below_threshold))
                corrections_made += 1
            else:
                corrected_list.append((beta, qfi))
                # Update floor for subsequent points
                max_qfi_below_threshold = max(max_qfi_below_threshold, qfi)
        else:
            corrected_list.append((beta, qfi))
    
    # Replace the data
    all_species_qfi_data[species] = corrected_list
    
    if corrections_made > 0:
        print(f"  Applied QFI floor for {species}: {corrections_made} points corrected to floor={max_qfi_below_threshold:.4f}")


def _apply_qfi_floor_per_param(data_points, beta_threshold):
    """
    Apply QFI floor per parameter value (e.g., per Jpm).
    
    For each unique parameter value, ensures QFI is monotonically non-decreasing
    with increasing beta (decreasing temperature) ONLY for beta > threshold.
    Below threshold, QFI values are kept as-is.
    
    Parameters:
    data_points: List of (param_value, beta, qfi) tuples
    beta_threshold: Only enforce monotonicity for beta > this value
    
    Returns:
    Corrected list of (param_value, beta, qfi) tuples
    """
    if not data_points:
        return data_points
    
    # Group by parameter value
    param_groups = defaultdict(list)
    for param_val, beta, qfi in data_points:
        param_groups[param_val].append((beta, qfi))
    
    # Apply monotonicity within each parameter group
    corrected_points = []
    total_corrections = 0
    
    for param_val, beta_qfi_list in param_groups.items():
        # Sort by beta (ascending: low beta = high T first)
        sorted_data = sorted(beta_qfi_list, key=lambda x: (np.inf if np.isinf(x[0]) else x[0]))
        
        # Find the QFI at the last beta point at or below threshold
        # This becomes the starting floor for higher beta values
        running_max = 0.0
        for beta, qfi in sorted_data:
            if not np.isinf(beta) and beta <= beta_threshold:
                running_max = qfi  # Keep updating to get the last one (highest beta <= threshold)
        
        # Apply: below threshold keep as-is, above threshold enforce monotonicity
        for beta, qfi in sorted_data:
            if np.isinf(beta) or beta <= beta_threshold:
                # Below threshold: keep original value
                corrected_points.append((param_val, beta, qfi))
            else:
                # Above threshold: enforce monotonicity
                if qfi < running_max:
                    corrected_points.append((param_val, beta, running_max))
                    total_corrections += 1
                else:
                    corrected_points.append((param_val, beta, qfi))
                    running_max = qfi
    
    if total_corrections > 0:
        print(f"  Applied per-parameter QFI floor: {total_corrections} points corrected (threshold β={beta_threshold})")
    
    return corrected_points


def _get_bin_beta(beta_vals):
    """Get representative beta value for a bin."""
    if len(beta_vals) == 0:
        return 0.0
    elif any(np.isinf(v) for v in beta_vals):
        return np.inf
    else:
        return np.mean(beta_vals)


def _load_and_average_spectral(file_list):
    """Load and average spectral data from multiple files.
    
    Returns: (mean_omega, mean_spectral, list_of_individual_data)
    where list_of_individual_data = [(omega, spectral, filepath), ...]
    """
    
    all_omega = []
    all_spectral = []
    individual_data = []  # Store individual sample data
    
    for fpath in file_list:
        omega, spectral = load_spectral_file(fpath)
        
        if omega is None or spectral is None:
            print(f"    Warning: Failed to load {fpath}")
            continue
        
        all_omega.append(omega)
        all_spectral.append(spectral)
        individual_data.append((omega, spectral, fpath))
    
    if not all_omega:
        return None, None, []
    
    # Check if all omega arrays are identical
    ref_omega = all_omega[0]
    all_match = all(np.allclose(omega, ref_omega) for omega in all_omega)
    
    if all_match:
        # Simple average if all omega arrays match
        mean_omega = ref_omega
        if ENABLE_NAN_INTERPOLATION:
            # nanmean handles NaN by ignoring them
            mean_spectral = np.nanmean(all_spectral, axis=0)
        else:
            # Regular mean propagates NaN values
            mean_spectral = np.mean(all_spectral, axis=0)
    else:
        # Interpolate to common grid when omega arrays don't match
        print("    Warning: Omega arrays don't match across samples, interpolating to common grid")
        
        # Find the common omega range
        omega_min = max(omega[0] for omega in all_omega)
        omega_max = min(omega[-1] for omega in all_omega)
        
        # Use the densest grid size among all samples
        n_points = max(len(omega) for omega in all_omega)
        mean_omega = np.linspace(omega_min, omega_max, n_points)
        
        # Interpolate each spectral function to the common grid
        interpolated_spectral = []
        for omega, spectral in zip(all_omega, all_spectral):
            f = interp1d(omega, spectral, kind='linear', bounds_error=False, fill_value=0.0)
            interpolated_spectral.append(f(mean_omega))
        
        # Average the interpolated spectral functions
        if ENABLE_NAN_INTERPOLATION:
            # nanmean handles NaN by ignoring them
            mean_spectral = np.nanmean(interpolated_spectral, axis=0)
        else:
            # Regular mean propagates NaN values
            mean_spectral = np.mean(interpolated_spectral, axis=0)
    
    return mean_omega, mean_spectral, individual_data


def _extract_sample_idx(filepath):
    """Extract sample index from filepath or dataset name."""
    # Handle DSSF HDF5 references: DSSF_HDF5:path:operator:beta_group:sample_idx
    if filepath.startswith('DSSF_HDF5:'):
        parts = filepath[10:].split(':')
        if len(parts) >= 4:
            try:
                return int(parts[3])
            except (ValueError, IndexError):
                pass
        # Fallback: try regex
        m = re.search(r'sample[_]?(\d+)', filepath, re.IGNORECASE)
        if m:
            return int(m.group(1))
    # Handle regular HDF5 references
    elif filepath.startswith('HDF5:'):
        # Extract from dataset name
        m = re.search(r'sample[_]?(\d+)', filepath, re.IGNORECASE)
        if m:
            return int(m.group(1))
    else:
        # Extract from text filename
        species, beta, sample_idx = parse_spectral_filename(filepath)
        if sample_idx is not None:
            return sample_idx
    return 0  # Default if not found


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
    
    # Save per-sample QFI values if available
    if 'per_sample_qfi' in results and results['per_sample_qfi']:
        per_sample_filename = os.path.join(outdir, f'qfi_per_sample_beta_{beta_label}.txt')
        with open(per_sample_filename, 'w') as f:
            f.write(f"# Per-sample QFI for {species} at beta={beta_label}\n")
            f.write(f"# sample_index QFI filepath\n")
            for i, (qfi_val, fpath) in enumerate(results['per_sample_qfi']):
                sample_idx = _extract_sample_idx(fpath)
                f.write(f"{sample_idx} {qfi_val:.10e} {fpath}\n")
    
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
    plt.xlim(-1, 1)
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


def load_processed_qfi_data(data_dir, param_pattern='Jpm'):
    """
    Load already processed QFI data from subdirectories and create heatmaps.
    Skips the spectral function processing step.
    
    Parameters:
    data_dir: Root directory containing parameter subdirectories
    param_pattern: Parameter name pattern (e.g., 'Jpm', 'h', 'J')
    """
    
    print(f"\n{'='*70}")
    print(f"Loading processed QFI data for {param_pattern} sweep")
    print(f"Data directory: {data_dir}")
    print(f"{'='*70}\n")
    
    # Organize data by species
    species_data = defaultdict(list)
    
    # Find all subdirectories matching the pattern
    subdirs = sorted(glob.glob(os.path.join(data_dir, f'{param_pattern}=*')))
    print(f"Found {len(subdirs)} {param_pattern} subdirectories")
    
    param_regex = re.compile(rf'{param_pattern}=([-]?[\d\.]+)')
    
    for subdir in subdirs:
        param_match = param_regex.search(os.path.basename(subdir))
        if not param_match:
            continue
        
        param_value = float(param_match.group(1))
        
        
        # Skip excluded Jpm values (corrupted data)
        if param_pattern == "Jpm" and any(abs(param_value - excl) < 1e-6 for excl in EXCLUDED_JPM_VALUES):
            print(f"  Skipping {param_pattern}={param_value} (marked as corrupted)")
            continue
        # Look for processed data in plots directory
        plots_dir = os.path.join(subdir, 'structure_factor_results', 'plots')
        if not os.path.exists(plots_dir):
            print(f"  Warning: No plots directory in {os.path.basename(subdir)}")
            continue
        
        print(f"Loading data from: {os.path.basename(subdir)}")
        
        # Load QFI vs beta files
        qfi_files = glob.glob(os.path.join(plots_dir, 'qfi_vs_beta_*.dat'))
        
        # Filter out per-sample files (only load averaged data)
        qfi_files = [f for f in qfi_files if '_sample_' not in os.path.basename(f)]
        
        for file_path in qfi_files:
            species = _extract_species_from_filename(file_path, 'qfi_vs_beta_*.dat')
            if not species:
                print(f"  Warning: Could not extract species from {os.path.basename(file_path)}")
                continue
            
            try:
                data = np.loadtxt(file_path)
                if data.size == 0:
                    print(f"  Warning: Empty file: {file_path}")
                    continue
                if data.ndim == 1:
                    data = data.reshape(1, -1)
                
                for row in data:
                    beta, qfi = row[0], row[1] * QFI_SCALE_FACTOR
                    species_data[species].append((param_value, beta, qfi))
                
                print(f"  Loaded {len(data)} data points from {os.path.basename(file_path)}")
            except Exception as e:
                print(f"  ERROR reading {file_path}: {e}")
    
    print(f"\nLoaded data for {len(species_data)} species")
    
    if not species_data:
        print("ERROR: No data loaded! Check directory structure and file patterns.")
        return None
    
    # Create summary plots
    _plot_parameter_sweep_summary(species_data, data_dir, param_pattern)
    
    return species_data


def _extract_species_from_filename(file_path, pattern):
    """Extract species name from filename."""
    base_pattern = pattern.replace('*', '(.+?)')
    match = re.search(base_pattern, os.path.basename(file_path))
    return match.group(1) if match else None


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
    param_regex = re.compile(rf'{param_pattern}=([-]?[\d\.]+)')
    
    for subdir in local_subdirs:
        param_match = param_regex.search(os.path.basename(subdir))
        if not param_match:
            continue
        
        param_value = float(param_match.group(1))
        print(f"[Rank {rank}] Processing {param_pattern}={param_value}")
        
        # Skip excluded Jpm values (corrupted data)
        if param_pattern == "Jpm" and any(abs(param_value - excl) < 1e-6 for excl in EXCLUDED_JPM_VALUES):
            print(f"[Rank {rank}] Skipping {param_pattern}={param_value} (marked as corrupted)")
            continue
        
        sf_dir = os.path.join(subdir, 'structure_factor_results')
        if not os.path.exists(sf_dir):
            print(f"  Warning: No structure_factor_results directory")
            continue
        
        qfi_data = parse_QFI_data_from_spectral(sf_dir, param_value=param_value)
        
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


def _plot_parameter_sweep_summary(data, data_dir, param_pattern):
    """Create summary plots for parameter sweep with comprehensive error checking.
    
    Parameters:
    data: Either merged_data dict (from parse_QFI_across_parameter) or species_data dict (from load_processed_qfi_data)
    data_dir: Root data directory
    param_pattern: Parameter name
    """
    
    print(f"\n{'='*70}")
    print(f"Starting heatmap plotting for {param_pattern} sweep")
    print(f"Data directory: {data_dir}")
    print(f"{'='*70}\n")
    
    plot_outdir = os.path.join(data_dir, f'plots_{param_pattern}')
    os.makedirs(plot_outdir, exist_ok=True)
    print(f"Output directory: {plot_outdir}")
    
    # Check if data is already organized by species or needs to be organized
    if data and isinstance(next(iter(data.values())), dict):
        # merged_data format: {param_value: {species: [(beta, qfi), ...]}}
        species_data = defaultdict(list)
        for param_value, qfi_dict in data.items():
            for species, qfi_list in qfi_dict.items():
                for beta, qfi in qfi_list:
                    species_data[species].append((param_value, beta, qfi))
    else:
        # Already organized by species: {species: [(param_value, beta, qfi), ...]}
        species_data = data
    
    print(f"Organized data for {len(species_data)} species")
    
    # Save raw data points
    _save_raw_data_points_param(species_data, plot_outdir, param_pattern)
    
    # Plot heatmaps for each species
    print(f"\n{'='*70}")
    print("Processing QFI heatmaps...")
    print(f"{'='*70}")
    
    # Separate base species from per-sample species
    base_species = {k: v for k, v in species_data.items() if '_sample_' not in k}
    sample_species = {k: v for k, v in species_data.items() if '_sample_' in k}
    
    # Plot base species (averaged) heatmaps
    for species, data_points in base_species.items():
        if data_points:
            try:
                _plot_parameter_beta_heatmap(species, data_points, plot_outdir, param_pattern, 
                                             data_dir=data_dir)
            except Exception as e:
                print(f"ERROR processing heatmap for {species}: {e}")
                import traceback
                traceback.print_exc()
    
    # Plot per-sample heatmaps
    if sample_species:
        print(f"\n{'='*70}")
        print("Processing per-sample QFI heatmaps...")
        print(f"{'='*70}")
        sample_outdir = os.path.join(plot_outdir, 'per_sample')
        os.makedirs(sample_outdir, exist_ok=True)
        
        # Group samples by base species
        samples_by_species = defaultdict(dict)
        for species_key, data_points in sample_species.items():
            # Extract base species and sample index
            # Format: {base_species}_sample_{N}
            match = re.match(r'^(.+?)_sample_(\d+)$', species_key)
            if match:
                base_species_name = match.group(1)
                sample_idx = int(match.group(2))
                samples_by_species[base_species_name][sample_idx] = data_points
        
        # Create individual heatmaps for each sample
        for base_species_name, samples_dict in samples_by_species.items():
            print(f"\nProcessing {len(samples_dict)} samples for {base_species_name}")
            for sample_idx, data_points in sorted(samples_dict.items()):
                sample_species_key = f"{base_species_name}_sample_{sample_idx}"
                try:
                    _plot_parameter_beta_heatmap(sample_species_key, data_points, 
                                                 sample_outdir, param_pattern, data_dir=data_dir)
                except Exception as e:
                    print(f"ERROR processing heatmap for {sample_species_key}: {e}")
                    import traceback
                    traceback.print_exc()
        
        # Also create subplot grids for easy comparison
        grid_outdir = os.path.join(plot_outdir, 'per_sample_grid')
        os.makedirs(grid_outdir, exist_ok=True)
        for base_species_name, samples_dict in samples_by_species.items():
            try:
                _plot_all_samples_grid(base_species_name, samples_dict, grid_outdir, param_pattern)
            except Exception as e:
                print(f"ERROR processing per-sample grid for {base_species_name}: {e}")
                import traceback
                traceback.print_exc()
    
    print(f"\n{'='*70}")
    print("Heatmap plotting complete!")
    print(f"{'='*70}\n")


def _save_raw_data_points_param(species_data, plot_outdir, param_pattern):
    """Save raw assembled points for parameter sweep."""
    for species, data_points in species_data.items():
        if data_points:
            arr = np.array(data_points, dtype=float)
            np.savetxt(
                os.path.join(plot_outdir, f'qfi_points_{species}.dat'), 
                arr, header=f'{param_pattern} beta qfi'
            )


def _create_closeup_heatmap_param(species, target_beta, Z_pos, param_pos, 
                                  filtered_data, plot_outdir, param_pattern, 
                                  param_range=(0.04, 0.10)):
    """Create a zoomed-in heatmap for a specific parameter range."""
    
    if Z_pos is None or param_pos.size == 0:
        print(f"  Warning: No positive parameter data for closeup heatmap")
        return
    
    # Filter parameter values to the specified range (excluding upper boundary)
    param_mask = (param_pos >= param_range[0]) & (param_pos < param_range[1])
    if not np.any(param_mask):
        print(f"  Warning: No data in parameter range {param_range} for closeup heatmap")
        return
    
    param_zoom = param_pos[param_mask]
    Z_zoom = Z_pos[:, param_mask]
    
    # Use filtered beta if available, otherwise use full beta grid
    if 'beta_pos_f' in filtered_data and 'mask_pos' in filtered_data:
        beta_zoom = filtered_data['beta_pos_f']
        Z_zoom_filtered = filtered_data['Z_pos_f'][:, param_mask]
    else:
        beta_zoom = target_beta
        Z_zoom_filtered = Z_zoom
    
    # Get color limits from the zoomed data
    vmin = np.nanmin(Z_zoom_filtered)
    vmax = np.nanmax(Z_zoom_filtered)
    
    # Create meshgrid
    P, B = np.meshgrid(param_zoom, beta_zoom)
    
    plt.figure(figsize=(10, 8))
    plt.pcolormesh(P, B, Z_zoom_filtered, shading='auto', cmap='RdYlBu_r', vmin=vmin, vmax=vmax)
    plt.yscale('log')
    plt.gca().invert_yaxis()  # large beta at bottom
    plt.xlabel(param_pattern)
    plt.ylabel('Beta (β)')
    plt.title(f'QFI Heatmap Closeup ({param_range[0]}-{param_range[1]}) for {species}')
    plt.colorbar(label='QFI')
    plt.xlim(param_range)
    
    fname = os.path.join(plot_outdir, f'qfi_heatmap_{species}_closeup_{param_range[0]}_{param_range[1]}.png')
    plt.savefig(fname, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save zoomed grid data
    zoom_filename = os.path.join(plot_outdir, f'qfi_grid_{species}_closeup_{param_range[0]}_{param_range[1]}.dat')
    header_cols = ['beta'] + [f'{param_pattern}={v:g}' for v in param_zoom]
    header = ' '.join(header_cols)
    out = np.column_stack((beta_zoom, Z_zoom_filtered))
    np.savetxt(zoom_filename, out, header=header)
    
    print(f"  Created closeup heatmap for {param_pattern} range {param_range}")


def _plot_all_samples_grid(base_species, samples_dict, plot_outdir, param_pattern):
    """Create subplot grid showing all samples together for direct comparison.
    
    Parameters:
    base_species: Base species name (without _sample_N suffix)
    samples_dict: Dictionary mapping sample_idx -> data_points
    plot_outdir: Output directory for plots
    param_pattern: Parameter name
    """
    print(f"\nCreating per-sample comparison grid for: {base_species}")
    print(f"Number of samples: {len(samples_dict)}")
    
    # Sort samples by index
    sample_indices = sorted(samples_dict.keys())
    n_samples = len(sample_indices)
    
    if n_samples == 0:
        return
    
    # Calculate subplot grid layout (prefer square-ish)
    n_cols = int(np.ceil(np.sqrt(n_samples)))
    n_rows = int(np.ceil(n_samples / n_cols))
    
    # Prepare data for all samples
    sample_grids = {}
    vmin_global = np.inf
    vmax_global = -np.inf
    
    for sample_idx in sample_indices:
        data_points = samples_dict[sample_idx]
        
        # Convert to array
        arr = np.array(data_points, dtype=float)
        param_vals, beta_vals, qfi_vals = arr[:, 0], arr[:, 1], arr[:, 2]
        
        # Get beta grid
        ref_target = np.median(param_vals)
        target_beta = _get_beta_grid_param(param_vals, beta_vals, ref_target)
        
        if target_beta.size < 2:
            print(f"  Sample {sample_idx}: Insufficient beta grid, skipping")
            continue
        
        # Split parameters
        param_neg, param_pos = _split_param_values(param_vals)
        
        # Interpolate to grid (focus on positive params for now)
        Z_neg, Z_pos = _interpolate_to_grid_param(
            param_vals, beta_vals, qfi_vals,
            param_neg, param_pos, target_beta, param_pattern
        )
        
        
        # Fill NaN values at highest beta (lowest temperature) by interpolating across Jpm
        if ENABLE_NAN_INTERPOLATION:
            Z_neg = _fill_nan_at_highest_beta(Z_neg, param_neg, param_pattern, target_beta)
            Z_pos = _fill_nan_at_highest_beta(Z_pos, param_pos, param_pattern, target_beta)
        # Filter NaN rows
        filtered_data = _filter_nan_rows_param(target_beta, Z_neg, Z_pos, True)
        
        # Store grid data
        sample_grids[sample_idx] = {
            'param_pos': param_pos,
            'beta': filtered_data.get('beta_pos_f', target_beta),
            'Z_pos': filtered_data.get('Z_pos_f', Z_pos),
            'param_neg': param_neg,
            'Z_neg': filtered_data.get('Z_neg_f', Z_neg)
        }
        
        # Update global color limits
        if Z_pos is not None:
            vmin_global = min(vmin_global, np.nanmin(Z_pos))
            vmax_global = max(vmax_global, np.nanmax(Z_pos))
    
    if not sample_grids:
        print(f"  No valid grids for {base_species}")
        return
    
    # Create subplot figure (positive parameters only)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows),
                             squeeze=False)
    
    for idx, sample_idx in enumerate(sample_indices):
        if sample_idx not in sample_grids:
            continue
            
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]
        
        grid = sample_grids[sample_idx]
        param_pos = grid['param_pos']
        beta = grid['beta']
        Z_pos = grid['Z_pos']
        
        if Z_pos is None or param_pos.size == 0:
            ax.text(0.5, 0.5, f'Sample {sample_idx}\n(No data)',
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_xticks([])
            ax.set_yticks([])
            continue
        
        # Create meshgrid and plot
        P, B = np.meshgrid(param_pos, beta)
        pcm = ax.pcolormesh(P, B, Z_pos, shading='auto', cmap='RdYlBu_r',
                           vmin=vmin_global, vmax=vmax_global)
        
        ax.set_yscale('log')
        ax.invert_yaxis()
        ax.set_title(f'Sample {sample_idx}', fontsize=10)
        
        # Only show axis labels on edge subplots
        if row == n_rows - 1:
            ax.set_xlabel(param_pattern, fontsize=9)
        if col == 0:
            ax.set_ylabel('Beta (β)', fontsize=9)
    
    # Hide unused subplots
    for idx in range(len(sample_indices), n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].axis('off')
    
    # Add shared colorbar
    fig.subplots_adjust(right=0.92, hspace=0.3, wspace=0.3)
    cbar_ax = fig.add_axes([0.94, 0.15, 0.02, 0.7])
    fig.colorbar(pcm, cax=cbar_ax, label='QFI')
    
    fig.suptitle(f'QFI Per-Sample Comparison: {base_species}', fontsize=14, y=0.98)
    
    # Save figure
    fname = os.path.join(plot_outdir, f'qfi_per_sample_grid_{base_species}.png')
    plt.savefig(fname, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Created per-sample grid with {len(sample_grids)} samples")


def _plot_parameter_beta_heatmap(species, data_points, plot_outdir, param_pattern, data_dir=None):
    """Create heatmap of QFI vs parameter and beta with comprehensive error checking.
    
    Parameters:
    species: Species name
    data_points: List of (param_value, beta, qfi) tuples
    plot_outdir: Output directory for plots
    param_pattern: Parameter name pattern (e.g., 'Jpm')
    data_dir: Root data directory containing parameter subdirectories (for peak data loading)
    """
    
    print(f"\n{'='*60}")
    print(f"Processing heatmap for species: {species}")
    print(f"Number of data points: {len(data_points)}")
    
    # Load peak data if data_dir is provided
    peak_data = None
    if data_dir is not None:
        peak_data = _load_peak_data_for_param_sweep(data_dir, species, param_pattern)
        if peak_data:
            print(f"  Loaded {len(peak_data)} lowest peak positions for overlay")
            # Save peak data to file
            _save_peak_data(peak_data, plot_outdir, species, param_pattern)
        else:
            print(f"  No peak data found for species {species}")
    
    # Apply per-parameter QFI floor to correct for numerical instabilities at high beta
    if ENABLE_HIGH_BETA_QFI_FLOOR:
        data_points = _apply_qfi_floor_per_param(data_points, HIGH_BETA_QFI_FLOOR_THRESHOLD)
    
    # Remove last row (highest beta) if beta extension is enabled
    if ENABLE_BETA_EXTENSION:
        # Group by param and find max beta for each
        from collections import defaultdict
        param_max_beta = defaultdict(float)
        for p, b, q in data_points:
            if not np.isinf(b):
                param_max_beta[p] = max(param_max_beta[p], b)
        
        # Remove points at the max beta for each param
        filtered_points = []
        removed_count = 0
        for p, b, q in data_points:
            if not np.isinf(b) and abs(b - param_max_beta[p]) < 1e-6:
                removed_count += 1
                continue
            filtered_points.append((p, b, q))
        data_points = filtered_points
        print(f"  Removed {removed_count} points at highest beta for extension")
    
    # Convert to array
    arr = np.array(data_points, dtype=float)
    param_vals, beta_vals, qfi_vals = arr[:, 0], arr[:, 1], arr[:, 2]
    
    print(f"{param_pattern} range: [{param_vals.min():.3f}, {param_vals.max():.3f}]")
    print(f"Beta range: [{beta_vals.min():.3f}, {beta_vals.max():.3f}]")
    print(f"QFI range: [{qfi_vals.min():.3f}, {qfi_vals.max():.3f}]")
    
    # Get beta grid - use reference value near middle of param range
    ref_target = np.median(param_vals)
    target_beta = _get_beta_grid_param(param_vals, beta_vals, ref_target)
    
    # Extend beta grid to EXTENDED_BETA_MAX if enabled
    if ENABLE_BETA_EXTENSION:
        max_existing = target_beta.max() if len(target_beta) > 0 else 0
        if max_existing < EXTENDED_BETA_MAX:
            # Add extended beta points (log-spaced from max to EXTENDED_BETA_MAX)
            n_extend = 5
            extended_betas = np.logspace(np.log10(max_existing * 1.2), np.log10(EXTENDED_BETA_MAX), n_extend)
            target_beta = np.concatenate([target_beta, extended_betas])
            target_beta = np.unique(target_beta)
            target_beta.sort()
            print(f"  Extended beta grid to {EXTENDED_BETA_MAX}, new size: {len(target_beta)}")
    if target_beta.size < 2:
        print(f"WARNING: Insufficient beta grid size ({target_beta.size}). Skipping species.")
        return
    
    print(f"Target beta grid size: {target_beta.size}")
    
    # Split into positive and negative parameter values
    param_neg, param_pos = _split_param_values(param_vals)
    
    print(f"Negative {param_pattern} values: {param_neg.size}, Positive {param_pattern} values: {param_pos.size}")
    
    # Interpolate data onto regular grid
    Z_neg, Z_pos = _interpolate_to_grid_param(
        param_vals, beta_vals, qfi_vals, 
        param_neg, param_pos, target_beta, param_pattern
    )
    
    
    # Fill NaN values at highest beta (lowest temperature) by interpolating across Jpm
    if ENABLE_NAN_INTERPOLATION:
        Z_neg = _fill_nan_at_highest_beta(Z_neg, param_neg, param_pattern, target_beta)
        Z_pos = _fill_nan_at_highest_beta(Z_pos, param_pos, param_pattern, target_beta)
    if Z_neg is not None:
        print(f"Z_neg shape: {Z_neg.shape}, NaN count: {np.isnan(Z_neg).sum()}")
    if Z_pos is not None:
        print(f"Z_pos shape: {Z_pos.shape}, NaN count: {np.isnan(Z_pos).sum()}")
    
    # Save grids
    _save_grids_param(species, target_beta, param_neg, param_pos, 
                      Z_neg, Z_pos, plot_outdir, param_pattern)
    
    # Filter rows with no NaN
    filtered_data = _filter_nan_rows_param(target_beta, Z_neg, Z_pos, True)
    
    print(f"Filtered data keys: {filtered_data.keys()}")
    
    # Save filtered grids
    _save_filtered_grids_param(species, filtered_data, param_neg, param_pos, 
                               plot_outdir, param_pattern)
    
    # Save gridded data as points format (binned data)
    _save_gridded_points_param(species, target_beta, param_neg, param_pos,
                               Z_neg, Z_pos, plot_outdir, param_pattern)
    
    # Create plots
    try:
        _create_heatmap_plots_param(species, filtered_data, param_neg, param_pos, 
                                    plot_outdir, param_pattern, peak_data=peak_data)
        print(f"Successfully created heatmap plots for {species}")
    except Exception as e:
        print(f"ERROR creating heatmap plots for {species}: {e}")
        import traceback
        traceback.print_exc()
    
    # Create line plots at fixed beta
    try:
        _create_fixed_beta_plots_param(species, target_beta, Z_neg, Z_pos, 
                                       param_neg, param_pos, filtered_data, 
                                       plot_outdir, param_pattern)
        print(f"Successfully created fixed beta plots for {species}")
    except Exception as e:
        print(f"ERROR creating fixed beta plots for {species}: {e}")
        import traceback
        traceback.print_exc()
    
    # Create closeup heatmap for specific parameter range
    try:
        _create_closeup_heatmap_param(species, target_beta, Z_pos, param_pos,
                                     filtered_data, plot_outdir, param_pattern,
                                     param_range=(0.04, 0.10))
        print(f"Successfully created closeup heatmap for {species}")
    except Exception as e:
        print(f"ERROR creating closeup heatmap for {species}: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"{'='*60}\n")


def _get_beta_grid_param(param_vals, beta_vals, ref_target):
    """Create beta grid based on reference parameter value."""
    unique_param = np.unique(param_vals)
    if unique_param.size == 0:
        return np.array([])
    
    # Find parameter closest to reference target
    ref_param = unique_param[np.argmin(np.abs(unique_param - ref_target))]
    ref_mask = np.isclose(param_vals, ref_param, rtol=1e-8, atol=1e-12)
    
    beta_ref = beta_vals[ref_mask]
    beta_ref = beta_ref[beta_ref > 0]
    target_beta = np.unique(beta_ref)
    target_beta.sort()
    
    # Fallback to all positive betas if reference grid insufficient
    if target_beta.size < 2:
        target_beta = np.unique(beta_vals[beta_vals > 0])
        target_beta.sort()
    
    
    return target_beta


def _split_param_values(param_vals):
    """Split parameter values into negative and positive arrays."""
    param_neg = np.unique(param_vals[param_vals < 0])
    param_neg.sort()
    param_pos = np.unique(param_vals[param_vals > 0])
    param_pos.sort()
    return param_neg, param_pos


def _interpolate_to_grid_param(param_vals, beta_vals, values, param_neg, param_pos, target_beta, param_pattern):
    """Interpolate data onto regular beta grid for each parameter value."""
    
    print(f"[DEBUG] interpolate_to_grid: target_beta size={len(target_beta)}, param_neg size={len(param_neg)}, param_pos size={len(param_pos)}")
    
    def interp_at_param(p):
        mask = np.isclose(param_vals, p, rtol=1e-8, atol=1e-12)
        b = beta_vals[mask]
        v = values[mask]
        
        if b.size == 0:
            print(f"[DEBUG] {param_pattern}={p:.3f}: No data points found")
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
            print(f"[DEBUG] {param_pattern}={p:.3f}: Insufficient unique beta values ({bu.size}), returning NaN")
            return np.full_like(target_beta, np.nan, dtype=float)
        
        # Use extrapolation: fill values beyond data range with the edge values
        # This extends QFI to higher beta using the last known value
        last_value = v_mean[-1]  # Value at highest beta in data
        f = interp1d(bu, v_mean, kind='linear', bounds_error=False, fill_value=(v_mean[0], last_value))
        result = f(target_beta)
        
        # For extended beta values, use the last known QFI (monotonicity)
        if ENABLE_BETA_EXTENSION:
            max_data_beta = bu.max()
            for i, tb in enumerate(target_beta):
                if tb > max_data_beta:
                    result[i] = last_value
        
        nan_count = np.isnan(result).sum()
        if nan_count > 0:
            print(f"[DEBUG] {param_pattern}={p:.3f}: Interpolation produced {nan_count} NaN values out of {len(result)}")
        return result
    
    Z_neg = np.column_stack([interp_at_param(p) for p in param_neg]) if param_neg.size > 0 else None
    Z_pos = np.column_stack([interp_at_param(p) for p in param_pos]) if param_pos.size > 0 else None
    
    if Z_neg is not None:
        print(f"[DEBUG] Z_neg created: shape={Z_neg.shape}, NaN count={np.isnan(Z_neg).sum()}")
    if Z_pos is not None:
        print(f"[DEBUG] Z_pos created: shape={Z_pos.shape}, NaN count={np.isnan(Z_pos).sum()}")
    
    return Z_neg, Z_pos









def _fill_nan_at_highest_beta(Z, params, param_pattern, target_beta=None):
    """
    Fill all NaN values by linear interpolation along the temperature (beta) axis
    at fixed parameter values.
    
    Parameters:
    Z: 2D array of shape (n_beta, n_params) - QFI values
    params: 1D array of parameter values (e.g., Jpm values)
    param_pattern: Parameter name for logging
    target_beta: 1D array of beta values (temperature axis)
    
    Returns:
    Z with all NaN values filled by interpolation along temperature axis
    """
    if Z is None or Z.size == 0:
        return Z
    
    if target_beta is None:
        target_beta = np.arange(Z.shape[0])
    
    from scipy.interpolate import interp1d
    
    Z_filled = Z.copy()
    total_filled = 0
    
    # Process each column (each Jpm value)
    for col_idx in range(Z.shape[1]):
        col = Z_filled[:, col_idx].copy()
        
        nan_mask = np.isnan(col)
        if not np.any(nan_mask):
            continue
        
        valid_mask = ~nan_mask
        n_valid = np.sum(valid_mask)
        
        if n_valid < 2:
            continue
        
        valid_betas = target_beta[valid_mask]
        valid_values = col[valid_mask]
        
        # Linear interpolation with extrapolation
        f = interp1d(valid_betas, valid_values, kind="linear",
                     bounds_error=False, fill_value="extrapolate")
        
        nan_betas = target_beta[nan_mask]
        filled_values = f(nan_betas)
        
        Z_filled[nan_mask, col_idx] = filled_values
        total_filled += np.sum(nan_mask)
    
    if total_filled > 0:
        print(f"  Filled {total_filled} NaN values by linear interpolation along beta")
    
    return Z_filled


def _filter_nan_rows_param(target_beta, Z_neg, Z_pos, naive=False):
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


def _save_grids_param(species, target_beta, param_neg, param_pos, Z_neg, Z_pos, plot_outdir, param_pattern):
    """Save unfiltered grid data."""
    
    if Z_neg is not None:
        _save_grid_data_param(plot_outdir, f'qfi_grid_neg_{species}', 
                             target_beta, param_neg, Z_neg, param_pattern)
    
    if Z_pos is not None:
        _save_grid_data_param(plot_outdir, f'qfi_grid_pos_{species}',
                             target_beta, param_pos, Z_pos, param_pattern)


def _save_filtered_grids_param(species, filtered_data, param_neg, param_pos, plot_outdir, param_pattern):
    """Save filtered grid data."""
    
    if 'Z_neg_f' in filtered_data:
        _save_grid_data_param(plot_outdir, f'qfi_grid_neg_filtered_{species}',
                             filtered_data['beta_neg_f'], param_neg, filtered_data['Z_neg_f'], param_pattern)
    
    if 'Z_pos_f' in filtered_data:
        _save_grid_data_param(plot_outdir, f'qfi_grid_pos_filtered_{species}',
                             filtered_data['beta_pos_f'], param_pos, filtered_data['Z_pos_f'], param_pattern)




def _save_gridded_points_param(species, target_beta, param_neg, param_pos,
                               Z_neg, Z_pos, plot_outdir, param_pattern):
    """
    Save gridded (binned) data in points format: param, beta, qfi.
    This uses the interpolated grid data rather than raw ungridded points.
    """
    points = []
    
    # Add negative param data
    if Z_neg is not None and param_neg.size > 0:
        for j, p in enumerate(param_neg):
            for i, b in enumerate(target_beta):
                qfi = Z_neg[i, j]
                if np.isfinite(qfi):
                    points.append((p, b, qfi))
    
    # Add positive param data
    if Z_pos is not None and param_pos.size > 0:
        for j, p in enumerate(param_pos):
            for i, b in enumerate(target_beta):
                qfi = Z_pos[i, j]
                if np.isfinite(qfi):
                    points.append((p, b, qfi))
    
    if points:
        arr = np.array(points, dtype=float)
        # Sort by param, then by beta
        arr = arr[np.lexsort((arr[:, 1], arr[:, 0]))]
        np.savetxt(
            os.path.join(plot_outdir, f'qfi_points_gridded_{species}.dat'),
            arr, header=f'{param_pattern} beta qfi'
        )
        print(f"  Saved {len(points)} gridded points for {species}")

def _save_grid_data_param(plot_outdir, base_name, beta, param, Z, param_pattern):
    """Save grid data in both npz and text formats."""
    np.savez(os.path.join(plot_outdir, f'{base_name}.npz'),
             beta=beta, param=param, Z=Z)
    
    # Save as text
    header_cols = ['beta'] + [f'{param_pattern}={v:g}' for v in param]
    header = ' '.join(header_cols)
    out = np.column_stack((beta, Z))
    np.savetxt(os.path.join(plot_outdir, f'{base_name}.dat'), 
               out, header=header)


def _create_heatmap_plots_param(species, filtered_data, param_neg, param_pos, plot_outdir, param_pattern,
                                peak_data=None):
    """Create heatmap plots for the species with optional peak position overlay.
    
    Parameters:
    peak_data: Species-specific peak data for gap curve overlay
    
    The gap curve label depends on species type:
    - SzSz species: labeled as "Vison gap"
    - SmSp species: labeled as "Spinon gap"
    """
    
    # Get color scale limits
    vmin, vmax = _get_color_limits_param(filtered_data)
    if vmin is None:
        return
    
    # Determine gap label and color based on species type
    if species.startswith('SzSz'):
        gap_label = 'Vison gap (β=1/|ω|)'
        gap_color = 'cyan'
    elif species.startswith('SmSp'):
        gap_label = 'Spinon gap (β=1/|ω|)'
        gap_color = 'magenta'
    else:
        gap_label = 'Gap (β=1/|ω|)'
        gap_color = 'lime'
    
    # Plot negative parameter heatmap
    if 'Z_neg_f' in filtered_data:
        _plot_single_heatmap_param(
            param_neg, filtered_data['beta_neg_f'], filtered_data['Z_neg_f'],
            vmin, vmax, 
            f'QFI Heatmap ({param_pattern}<0) for {species}',
            os.path.join(plot_outdir, f'qfi_heatmap_neg_{species}.png'),
            param_pattern,
            peak_data=peak_data,
            gap_label=gap_label,
            gap_color=gap_color
        )
    
    # Plot positive parameter heatmap
    if 'Z_pos_f' in filtered_data:
        _plot_single_heatmap_param(
            param_pos, filtered_data['beta_pos_f'], filtered_data['Z_pos_f'],
            vmin, vmax,
            f'QFI Heatmap ({param_pattern}>0) for {species}',
            os.path.join(plot_outdir, f'qfi_heatmap_pos_{species}.png'),
            param_pattern,
            peak_data=peak_data,
            gap_label=gap_label,
            gap_color=gap_color
        )
    
    # Plot side-by-side view
    if 'Z_neg_f' in filtered_data and 'Z_pos_f' in filtered_data:
        _plot_side_by_side_heatmap_param(
            species, filtered_data, param_neg, param_pos,
            vmin, vmax, plot_outdir, param_pattern,
            peak_data=peak_data,
            gap_label=gap_label,
            gap_color=gap_color
        )


def _get_color_limits_param(filtered_data):
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


def _load_peak_data_for_param_sweep(data_dir, species, param_pattern='Jpm'):
    """
    Load lowest peak positions from processed_data directories across parameter sweep.
    
    Returns:
    peak_data: List of (param_value, lowest_peak_frequency) tuples
    """
    peak_data = []
    
    # Find all subdirectories matching the pattern
    subdirs = sorted(glob.glob(os.path.join(data_dir, f'{param_pattern}=*')))
    param_regex = re.compile(rf'{param_pattern}=([-]?[\d\.]+)')
    
    for subdir in subdirs:
        param_match = param_regex.search(os.path.basename(subdir))
        if not param_match:
            continue
        
        param_value = float(param_match.group(1))
        
        # Look for processed data in processed_data/<species> directory
        processed_dir = os.path.join(subdir, 'structure_factor_results', 'processed_data', species)
        if not os.path.exists(processed_dir):
            continue
        
        # Find all peak files for this species
        peak_files = glob.glob(os.path.join(processed_dir, 'peaks_beta_*.dat'))
        
        # Collect lowest peak frequency across all beta values
        lowest_peaks_by_beta = []
        for peak_file in peak_files:
            try:
                # Extract beta from filename
                beta_match = re.search(r'peaks_beta_([0-9.]+|inf)\.dat', os.path.basename(peak_file))
                if not beta_match:
                    continue
                beta_str = beta_match.group(1)
                if beta_str == 'inf':
                    beta_val = np.inf
                else:
                    beta_val = float(beta_str)
                
                # Load peak data: frequency, height, prominence
                data = np.loadtxt(peak_file)
                if data.size == 0:
                    continue
                if data.ndim == 1:
                    data = data.reshape(1, -1)
                
                # Get the lowest frequency peak (first column is frequency)
                # Use absolute value to handle negative frequencies
                # Include peaks at exactly omega=0 (these will be plotted at lowest temperature)
                frequencies = np.abs(data[:, 0])
                valid_freqs = frequencies[frequencies >= 0]  # Include all peaks including omega=0
                if len(valid_freqs) == 0:
                    continue
                lowest_peak = np.min(valid_freqs)
                lowest_peaks_by_beta.append((beta_val, lowest_peak))
                
            except Exception as e:
                continue
        
        # Use the lowest peak from the lowest temperature (highest beta, excluding inf)
        if lowest_peaks_by_beta:
            # Filter out inf beta
            finite_peaks = [(b, p) for b, p in lowest_peaks_by_beta if not np.isinf(b)]
            if finite_peaks:
                # Sort by beta (descending) and take the lowest peak from highest beta
                finite_peaks.sort(key=lambda x: -x[0])
                _, lowest_peak = finite_peaks[0]
                peak_data.append((param_value, lowest_peak))
    
    return peak_data


def _load_global_gap_data(data_dir, operator_type, param_pattern='Jpm'):
    """
    Load global gap data by finding the minimum peak position across all momentum points
    for a given operator type (SzSz or SmSp).
    
    For SzSz: finds the minimum across all SzSz_q_Qx*_Qy*_Qz* species (vison gap)
    For SmSp: finds the minimum across all SmSp_q_Qx*_Qy*_Qz* species (spinon gap)
    
    Parameters:
    data_dir: Root data directory containing parameter subdirectories
    operator_type: 'SzSz' or 'SmSp'
    param_pattern: Parameter name pattern (e.g., 'Jpm')
    
    Returns:
    global_gap_data: List of (param_value, min_peak_frequency) tuples
    """
    # First, find all species matching the operator type pattern
    # We need to scan the processed_data directories to find all momentum point variants
    subdirs = sorted(glob.glob(os.path.join(data_dir, f'{param_pattern}=*')))
    
    if not subdirs:
        return []
    
    # Use the first subdir to discover all species of this operator type
    first_subdir = subdirs[0]
    processed_base = os.path.join(first_subdir, 'structure_factor_results', 'processed_data')
    
    if not os.path.exists(processed_base):
        return []
    
    # Find all species directories matching the operator type
    # Pattern: {operator_type}_q_Qx*_Qy*_Qz*
    matching_species = []
    for species_dir in os.listdir(processed_base):
        if species_dir.startswith(f'{operator_type}_q_'):
            matching_species.append(species_dir)
    
    if not matching_species:
        print(f"  No species found matching operator type: {operator_type}")
        return []
    
    print(f"  Found {len(matching_species)} momentum points for {operator_type}: {matching_species[:3]}...")
    
    # Load peak data for each matching species
    all_species_peaks = {}
    for species in matching_species:
        peak_data = _load_peak_data_for_param_sweep(data_dir, species, param_pattern)
        if peak_data:
            all_species_peaks[species] = dict(peak_data)  # Convert to dict for easy lookup
    
    if not all_species_peaks:
        return []
    
    # Get all parameter values
    all_params = set()
    for species_peaks in all_species_peaks.values():
        all_params.update(species_peaks.keys())
    
    # For each parameter value, find the minimum peak across all species
    global_gap_data = []
    for param_value in sorted(all_params):
        min_peak = float('inf')
        for species, species_peaks in all_species_peaks.items():
            if param_value in species_peaks:
                peak = species_peaks[param_value]
                if peak < min_peak:
                    min_peak = peak
        
        if min_peak < float('inf'):
            global_gap_data.append((param_value, min_peak))
    
    return global_gap_data


def _save_peak_data(peak_data, plot_outdir, species, param_pattern):
    """
    Save lowest peak position data to a file.
    
    Parameters:
    peak_data: List of (param_value, peak_frequency) tuples
    plot_outdir: Output directory for the file
    species: Species name
    param_pattern: Parameter name pattern (e.g., 'Jpm')
    """
    if not peak_data:
        return
    
    # Sort by parameter value
    sorted_data = sorted(peak_data, key=lambda x: x[0])
    
    # Create output array with param, frequency, and beta (=1/|frequency|)
    # Use absolute value of frequency for beta calculation to handle negative frequencies
    output = []
    for param_val, freq in sorted_data:
        abs_freq = np.abs(freq)
        beta = 1.0 / abs_freq if abs_freq > 0 else np.inf
        output.append([param_val, freq, beta])
    
    output_arr = np.array(output)
    
    # Save to file
    filename = os.path.join(plot_outdir, f'lowest_peak_positions_{species}.dat')
    header = f'{param_pattern} frequency beta(=1/|frequency|)'
    np.savetxt(filename, output_arr, header=header, fmt='%.8f')
    print(f"  Saved peak positions to {filename}")


def _overlay_peak_line_on_heatmap(ax, peak_data, param_range, color='lime', linewidth=2, label='Lowest peak (β=1/|ω|)', max_beta=1000.0):
    """
    Overlay the lowest peak position line on a heatmap.
    
    The line is plotted at beta = 1/|omega_peak| for each parameter value.
    Uses absolute value of frequency to handle negative frequency peaks.
    Peaks at omega=0 are plotted at the lowest temperature (highest beta).
    
    Parameters:
    ax: Matplotlib axis object
    peak_data: List of (param_value, peak_frequency) tuples
    param_range: Tuple of (param_min, param_max) for filtering
    color: Line color
    linewidth: Line width
    label: Legend label
    max_beta: Maximum beta value to use for omega=0 peaks (default: 1000.0)
    """
    if not peak_data:
        return
    
    # Filter peak data to parameter range
    # Accept both positive and negative frequencies (use abs for beta calculation)
    filtered_peaks = [(p, freq) for p, freq in peak_data 
                      if param_range[0] <= p <= param_range[1]]
    
    if not filtered_peaks:
        return
    
    # Sort by parameter value
    filtered_peaks.sort(key=lambda x: x[0])
    
    # Convert to arrays
    params = np.array([p for p, _ in filtered_peaks])
    frequencies = np.array([freq for _, freq in filtered_peaks])
    
    # Calculate beta = 1/|omega| (use absolute value for negative frequencies)
    # For omega=0, use max_beta (lowest temperature on plot)
    abs_freqs = np.abs(frequencies)
    betas = np.where(abs_freqs > 0, 1.0 / abs_freqs, max_beta)
    
    # Plot the line
    ax.plot(params, betas, color=color, linewidth=linewidth, linestyle='-', 
            marker='o', markersize=4, label=label, zorder=10)
    
    print(f"  Overlaid peak line: {len(params)} points, param range [{params.min():.3f}, {params.max():.3f}], "
          f"frequency range [{frequencies.min():.3f}, {frequencies.max():.3f}], "
          f"beta range [{betas.min():.3f}, {betas.max():.3f}]")


def _plot_single_heatmap_param(param, beta, Z, vmin, vmax, title, filename, param_pattern, 
                               peak_data=None, gap_label='Gap (β=1/|ω|)', gap_color='lime'):
    """Create a single heatmap plot with optional peak position overlay.
    
    Parameters:
    peak_data: Species-specific peak data
    gap_label: Label for the gap curve
    gap_color: Color for the gap curve
    """
    print(f"[DEBUG] plot_single_heatmap: param shape={param.shape}, beta shape={beta.shape}, Z shape={Z.shape}")
    print(f"[DEBUG] vmin={vmin:.6f}, vmax={vmax:.6f}")
    
    P, B = np.meshgrid(param, beta)
    print(f"[DEBUG] Meshgrid P shape={P.shape}, B shape={B.shape}")
    
    fig, ax = plt.subplots(figsize=(12, 8))
    try:
        mesh = ax.pcolormesh(P, B, Z, shading='auto', cmap='RdYlBu_r', vmin=vmin, vmax=vmax)
        ax.set_yscale('log')
        ax.invert_yaxis()  # large beta at bottom
        ax.set_xlabel(param_pattern)
        ax.set_ylabel('Beta (β)')
        ax.set_title(title)
        fig.colorbar(mesh, ax=ax, label='QFI')
        
        param_range = (param.min(), param.max())
        
        # Overlay species-specific gap curve if provided
        if peak_data:
            _overlay_peak_line_on_heatmap(ax, peak_data, param_range,
                                          color=gap_color, linewidth=2.5, label=gap_label)
            ax.legend(loc='upper right')
        
        fig.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"[DEBUG] Successfully saved heatmap to {filename}")
    except Exception as e:
        print(f"[ERROR] Failed to create heatmap: {e}")
        import traceback
        traceback.print_exc()
    finally:
        plt.close(fig)


def _plot_side_by_side_heatmap_param(species, filtered_data, param_neg, param_pos, 
                                     vmin, vmax, plot_outdir, param_pattern, peak_data=None,
                                     gap_label='Gap (β=1/|ω|)', gap_color='lime'):
    """Create side-by-side heatmap plot with optional peak position overlay.
    
    Parameters:
    peak_data: Species-specific peak data
    gap_label: Label for the gap curve
    gap_color: Color for the gap curve
    """
    
    beta_neg_f = filtered_data['beta_neg_f']
    beta_pos_f = filtered_data['beta_pos_f']
    Z_neg_f = filtered_data['Z_neg_f']
    Z_pos_f = filtered_data['Z_pos_f']
    
    # Create meshgrids
    PN, BN = np.meshgrid(param_neg, beta_neg_f)
    PP, BP = np.meshgrid(param_pos, beta_pos_f)
    
    # Calculate y-axis limits
    y_min = float(min(beta_neg_f.min(), beta_pos_f.min()))
    y_max = float(max(beta_neg_f.max(), beta_pos_f.max()))
    
    # Calculate width ratio
    wr = _calculate_width_ratio_param(param_neg, param_pos)
    
    # Create figure
    fig, (axL, axR) = plt.subplots(
        1, 2, figsize=(14, 8), sharey=True,
        gridspec_kw={'wspace': 0.0, 'hspace': 0.0, 'width_ratios': [wr, 1.0]}
    )
    
    # Plot heatmaps
    axL.pcolormesh(PN, BN, Z_neg_f, shading='auto', cmap='RdYlBu_r', vmin=vmin, vmax=vmax)
    axR.pcolormesh(PP, BP, Z_pos_f, shading='auto', cmap='RdYlBu_r', vmin=vmin, vmax=vmax)
    
    # Define parameter ranges
    param_range_neg = (param_neg.min(), param_neg.max())
    param_range_pos = (param_pos.min(), param_pos.max())
    
    # Overlay species-specific gap curve if provided
    if peak_data:
        _overlay_peak_line_on_heatmap(axL, peak_data, param_range_neg,
                                      color=gap_color, linewidth=2.5, label=gap_label)
        _overlay_peak_line_on_heatmap(axR, peak_data, param_range_pos,
                                      color=gap_color, linewidth=2.5, label=gap_label)
    
    # Format axes
    for ax in (axL, axR):
        ax.set_yscale('log')
        ax.set_ylim(y_max, y_min)  # large beta at bottom
        ax.set_xlabel(param_pattern)
    
    axL.set_ylabel('Beta (β)')
    axR.tick_params(labelleft=False)
    
    # Add colorbar
    fig.subplots_adjust(wspace=0.0)
    cbar = fig.colorbar(axL.collections[0], ax=[axL, axR], location='right', pad=0.02)
    cbar.set_label('QFI')
    
    # Add legend for gap lines (use left axis handles, avoid duplicates)
    if peak_data:
        handles, labels = axL.get_legend_handles_labels()
        if handles:
            # Remove duplicates while preserving order
            seen = set()
            unique_handles = []
            unique_labels = []
            for h, l in zip(handles, labels):
                if l not in seen:
                    seen.add(l)
                    unique_handles.append(h)
                    unique_labels.append(l)
            fig.legend(unique_handles, unique_labels, loc='upper center', 
                      bbox_to_anchor=(0.5, 0.02), ncol=len(unique_handles))
    
    fig.suptitle(f'QFI Heatmap ({param_pattern}<0 | {param_pattern}>0) for {species}')
    fig.savefig(os.path.join(plot_outdir, f'qfi_heatmap_side_by_side_{species}.png'),
                dpi=300, bbox_inches='tight')
    plt.close()


def _calculate_width_ratio_param(param_neg, param_pos):
    """Calculate width ratio for side-by-side plots."""
    neg_span = float(np.ptp(param_neg)) if param_neg.size > 0 else 0.0
    pos_span = float(np.ptp(param_pos)) if param_pos.size > 0 else 0.0
    
    if pos_span > 0:
        wr = neg_span / pos_span
    else:
        wr = 1.5
    
    return float(np.clip(wr, 1.2, 3.0))


def _create_fixed_beta_plots_param(species, target_beta, Z_neg, Z_pos, param_neg, param_pos, 
                                   filtered_data, plot_outdir, param_pattern):
    """Create line plots at fixed beta values."""
    
    # Find largest valid beta indices
    idx_neg = _find_largest_valid_beta_index_param(Z_neg, filtered_data.get('mask_neg'))
    idx_pos = _find_largest_valid_beta_index_param(Z_pos, filtered_data.get('mask_pos'))
    
    if idx_neg is None and idx_pos is None:
        return
    
    # Setup plot
    plt.figure(figsize=(10, 6))
    color = 'C0'
    
    # Plot negative parameter segment
    if idx_neg is not None:
        _plot_fixed_beta_segment_param(
            param_neg, Z_neg[idx_neg, :], target_beta[idx_neg],
            color, 'neg', plot_outdir, species, param_pattern
        )
    
    # Plot positive parameter segment
    if idx_pos is not None:
        _plot_fixed_beta_segment_param(
            param_pos, Z_pos[idx_pos, :], target_beta[idx_pos],
            color, 'pos', plot_outdir, species, param_pattern
        )
    
    plt.xlabel(param_pattern)
    plt.ylabel('QFI')
    plt.title(f'QFI vs {param_pattern} at largest β rows (no NaN) for {species}')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=9)
    
    fname = os.path.join(plot_outdir, f'qfi_vs_{param_pattern}_fixed_beta_{species}.png')
    plt.savefig(fname, dpi=300, bbox_inches='tight')
    plt.close()


def _find_largest_valid_beta_index_param(Z, mask):
    """Find the largest beta index with no NaN values, skipping the first (smallest) beta."""
    if Z is None or Z.size == 0 or mask is None or not np.any(mask):
        return None
    
    valid_idxs = np.where(mask)[0]
    # Skip the first (smallest) beta entry
    if valid_idxs.size > 1:
        valid_idxs = valid_idxs[1:]
    elif valid_idxs.size == 1:
        return None  # Only one valid beta, skip it
    
    return valid_idxs[-1] if valid_idxs.size > 0 else None


def _plot_fixed_beta_segment_param(param, values, beta, color, segment, plot_outdir, species, param_pattern):
    """Plot a segment of fixed beta data."""
    plt.plot(param, values, '-', lw=1.8, color=color, label=f'β={beta:.3g} ({segment})')
    
    # Save data
    filename = os.path.join(plot_outdir, f'qfi_vs_{param_pattern}_fixed_beta_{segment}_{species}.dat')
    header = f'{param_pattern} beta QFI'
    np.savetxt(filename, np.column_stack((param, np.full_like(param, beta), values)), header=header)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Calculate QFI from pre-computed spectral function files')
    parser.add_argument('directory', type=str,
                       help='Directory containing structure_factor_results or parameter sweep folders')
    parser.add_argument('--beta-tol', type=float, default=1e-2,
                       help='Tolerance for grouping beta values (default: 1e-2)')
    parser.add_argument('--param-sweep', type=str, default=None,
                       help='Parameter name for sweep analysis (e.g., Jpm, h, J)')
    parser.add_argument('--skip-processing', action='store_true',
                       help='Skip spectral processing and load existing processed QFI data')
    parser.add_argument('--clean', action='store_true',
                       help='Clean mode: disable all spectral corrections (rescaling, intensity scaling, NaN interpolation)')
    parser.add_argument('--no-rescale', action='store_true',
                       help='Disable spectral width rescaling (default: rescaling enabled)')
    parser.add_argument('--no-intensity-scale', action='store_true',
                       help='Disable intensity scaling to match static structure factor (default: enabled)')
    parser.add_argument('--aggressive-factor', type=float, default=1.5,
                       help='Aggressive rescaling factor (>1 for stronger high-T corrections, default: 1.5)')
    
    args = parser.parse_args()
    
    # Clean mode: disable all corrections
    if args.clean:
        ENABLE_SPECTRAL_RESCALING = False
        ENABLE_INTENSITY_SCALING = False
        ENABLE_NAN_INTERPOLATION = False
        print("=" * 60)
        print("CLEAN MODE: All spectral corrections DISABLED")
        print("  - No spectral width rescaling")
        print("  - No intensity scaling") 
        print("  - No NaN interpolation")
        print("=" * 60)
    else:
        # Set global rescaling flag based on command-line argument
        if args.no_rescale:
            ENABLE_SPECTRAL_RESCALING = False
            print("Spectral width rescaling is DISABLED")
        else:
            ENABLE_SPECTRAL_RESCALING = True
            SPECTRAL_RESCALE_AGGRESSIVE_FACTOR = args.aggressive_factor
            print(f"Spectral width rescaling is ENABLED (aggressive_factor={args.aggressive_factor})")
        
        # Set global intensity scaling flag based on command-line argument
        if args.no_intensity_scale:
            ENABLE_INTENSITY_SCALING = False
            print("Intensity scaling is DISABLED")
        else:
            ENABLE_INTENSITY_SCALING = True
            print("Intensity scaling is ENABLED (use --no-intensity-scale to disable)")
    
    if args.skip_processing:
        # Load already processed data mode
        if not args.param_sweep:
            print("ERROR: --skip-processing requires --param-sweep to be specified")
            print("Example: --skip-processing --param-sweep Jpm")
            sys.exit(1)
        
        print(f"Loading processed QFI data for {args.param_sweep} sweep (skipping spectral processing)")
        results = load_processed_qfi_data(args.directory, args.param_sweep)
        
    elif args.param_sweep:
        # Parameter sweep mode with full processing
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
