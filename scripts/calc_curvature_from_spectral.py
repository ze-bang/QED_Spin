"""
Calculate quantum geometric curvature from cross-spectral function data.

For a **pure ground state**, this computes the **Berry curvature** F_μν.
For a **thermal mixed state**, this computes the **mean Uhlmann curvature** (MUC) U_μν.

These are the antisymmetric/imaginary part of the quantum geometric tensor, and
serve as the curvature partner to the QFI (metric/Fubini-Study) part computed by
calc_QFI_from_spectral.py.

Physics
-------
Ground state (T=0, β → ∞):
    F_μν = -2 Im Σ_{n≠0} ⟨0|O_μ|n⟩⟨n|O_ν|0⟩ / (E_n - E_0)²
         = -(1/π) ∫₀^∞ dω Im[S_μν(ω)] / ω²

Thermal state (finite T):
    U_μν = (i/(2π)) ∫ dω/ω² tanh²(βω/2) [S_μν(ω) - S_νμ(-ω)]

    Using detailed balance S_νμ(-ω) = e^{-βω} S_μν(ω)*:
    U_μν = -(1/(2π)) ∫ dω tanh²(βω/2)(1+e^{-βω})/ω² · Im[S_μν(ω)]

    where S_μν(ω) is the (complex) cross-spectral function between operators
    O_μ and O_ν.

The curvature requires **cross-spectral** data between two different operators
(e.g., S^z_q and S^x_q). Both Re[S_μν(ω)] and Im[S_μν(ω)] must be available.
Im[S_μν(ω)] vanishes for auto-correlations (μ=ν), so the curvature is only
nonzero for genuinely distinct operator pairs.

Comparison with QFI:
    QFI uses the symmetric channel:    F_Q = 4 ∫₀^∞ Re[S(ω)] K_QFI(ω) dω
    MUC uses the antisymmetric channel: U  = -(1/π) ∫₀^∞ K_MUC(ω) Im[S_μν(ω)] dω

    K_QFI(ω,β) = tanh(βω/2)(1 - e^{-βω})
    K_MUC(ω,β) = tanh²(βω/2)(1 - e^{-βω}) / ω²  =  K_QFI · tanh(βω/2) / ω²

    Near ω=0:  K_MUC ≈ β³ω/4 → 0  (integrable, no divergence)
    For β→∞:   K_MUC → 1/ω²       (ground-state Berry curvature kernel)

Data sources (same as calc_QFI_from_spectral.py):
    1. HDF5 files: ed_results.h5 or dssf_results.h5 with both real and imag channels
    2. Text files: {species}_spectral_sample_{N}_beta_{beta}.txt
       Format: omega  Re[S(ω)]  Im[S(ω)]  Re[error]  Im[error]

Cross-spectral pair identification:
    Operator names like SxSz_q_Qx0_Qy0_Qz0 encode the pair (O_μ=Sx, O_ν=Sz).
    The script scans for all such cross-species and computes curvature for each.
"""

import os
import sys
import re
import glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict
from scipy.interpolate import interp1d

# Try to import h5py, but make it optional
try:
    import h5py
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False
    print("Warning: h5py not available, HDF5 reading disabled")

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


# ==============================================================================
# Configuration
# ==============================================================================

# Enable NaN interpolation in spectral data
ENABLE_NAN_INTERPOLATION = True

# Scale factor for curvature values
CURVATURE_SCALE_FACTOR = 1

# Minimum |ω| for integration (avoid exact ω=0 in discrete data)
OMEGA_MIN_CUTOFF = 1e-10

# ==============================================================================
# Spin operator labels used in cross-spectral pair identification
# ==============================================================================
XYZ_LABELS = {'Sx', 'Sy', 'Sz'}
LADDER_LABELS = {'Sp', 'Sm', 'Sz'}
ALL_SPIN_LABELS = XYZ_LABELS | LADDER_LABELS


# ==============================================================================
# Curvature Kernel Functions
# ==============================================================================

def curvature_kernel(omega, beta):
    """
    Compute the MUC kernel for positive frequencies:

        K(ω, β) = tanh²(βω/2) (1 - e^{-βω}) / ω²

    This is derived from the full-range integral by folding in the
    negative-frequency contribution using S_νμ(ω) = [S_μν(ω)]* (Hermitian
    operators) and the detailed balance relation S_μν(-ω) = e^{-βω}[S_μν(ω)]*.

    Relation to QFI kernel: K_MUC(ω) = K_QFI(ω) · tanh(βω/2) / ω²

    Limiting behaviour (ω > 0):
        ω → 0:   K → β³ω/4 → 0  (integrable)
        β → ∞:   K → 1/ω²
    """
    result = np.zeros_like(omega, dtype=float)
    pos = omega > OMEGA_MIN_CUTOFF

    if np.isinf(beta):
        result[pos] = 1.0 / omega[pos]**2
        return result

    omega_pos = omega[pos]
    bw2 = beta * omega_pos / 2.0

    # Vectorised, numerically stable evaluation
    # For very large βω/2, tanh→1, e^{-βω}→0
    safe = bw2 < 300
    result_pos = np.zeros_like(omega_pos)

    # Large βω regime
    result_pos[~safe] = 1.0 / omega_pos[~safe]**2

    # Normal regime
    if np.any(safe):
        th = np.tanh(bw2[safe])
        exp_neg = np.exp(-beta * omega_pos[safe])
        result_pos[safe] = th**2 * (1.0 - exp_neg) / omega_pos[safe]**2

    result[pos] = result_pos
    return result


def qfi_kernel(omega, beta):
    """
    QFI kernel for comparison: tanh(βω/2)(1 - e^{-βω}), only for ω > 0.
    This is the kernel used in calc_QFI_from_spectral.py.
    """
    result = np.zeros_like(omega, dtype=float)
    pos = omega > OMEGA_MIN_CUTOFF

    if np.isinf(beta):
        result[pos] = 1.0
    else:
        result[pos] = (np.tanh(beta * omega[pos] / 2.0)
                       * (1.0 - np.exp(-beta * omega[pos])))
    return result


# ==============================================================================
# Core Curvature Calculations
# ==============================================================================

def calculate_muc_from_cross_spectral(omega, spectral_imag, beta):
    """
    Calculate the mean Uhlmann curvature (MUC) from the imaginary part of
    the cross-spectral function, using positive frequencies only.

        U_μν = -(1/π) ∫₀^∞ dω K(ω,β) Im[S_μν(ω)]

    where K(ω,β) = tanh²(βω/2)(1-e^{-βω})/ω².

    This is derived from the full-range MUC integral
        U_μν = (i/(2π)) ∫ dω/ω² tanh²(βω/2) [S_μν(ω) − S_νμ(−ω)]
    by using the Hermitian-operator identity S_νμ(ω) = [S_μν(ω)]* and
    the detailed balance relation S_μν(−ω) = e^{−βω}[S_μν(ω)]*, which
    folds the negative-frequency piece into a factor of 2 on the
    positive-frequency integral.

    At T=0 this reduces to the Berry curvature:
        F_μν = U_μν(β→∞) = -(1/π) ∫₀^∞ dω Im[S_μν(ω)] / ω²

    Parameters
    ----------
    omega : ndarray
        Frequency array (positive frequencies used; negatives ignored).
    spectral_imag : ndarray
        Im[S_μν(ω)] — imaginary part of the cross-spectral function.
    beta : float
        Inverse temperature (use np.inf for ground state).

    Returns
    -------
    float
        The MUC value U_μν.
    """
    pos = omega > OMEGA_MIN_CUTOFF
    omega_pos = omega[pos]
    sim_pos = spectral_imag[pos]

    if len(omega_pos) == 0:
        return 0.0

    K = curvature_kernel(omega_pos, beta)
    integrand = K * sim_pos
    muc = -(1.0 / np.pi) * np_trapz(integrand, omega_pos)
    return muc


def calculate_berry_curvature_from_cross_spectral(omega, spectral_imag):
    """
    Calculate the Berry curvature F_μν from ground-state cross-spectral data.

        F_μν = -(1/π) ∫₀^∞ dω Im[S_μν(ω)] / ω²

    This is identical to U_μν(β→∞), the T=0 limit of the MUC.
    Provided separately for clarity when working with ground-state data.

    Parameters
    ----------
    omega : ndarray
        Frequency array.
    spectral_imag : ndarray
        Im[S_μν(ω)].

    Returns
    -------
    float
        The Berry curvature F_μν.
    """
    pos = omega > OMEGA_MIN_CUTOFF
    omega_pos = omega[pos]
    spec_pos = spectral_imag[pos]

    if len(omega_pos) == 0:
        return 0.0

    integrand = spec_pos / omega_pos**2
    F = -(1.0 / np.pi) * np_trapz(integrand, omega_pos)
    return F


def calculate_curvature_consistency_check(omega, spectral_real, beta):
    """
    Consistency check: compute the off-diagonal QFIM metric integral
    using the same kernel as MUC but with Re[S_μν].

    For hermitian operators, the positive-frequency integral with Re[S]
    gives the off-diagonal metric element, while Im[S] gives the curvature.
    The "imaginary part" of the full curvature tensor is proportional to
    this integral and should equal the QFIM off-diagonal up to
    normalisation — useful for cross-checking data quality.

    Returns (1/π) ∫₀^∞ K_MUC(ω,β) Re[S_μν(ω)] dω
    """
    pos = omega > OMEGA_MIN_CUTOFF
    omega_pos = omega[pos]
    sre_pos = spectral_real[pos]

    if len(omega_pos) == 0:
        return 0.0

    K = curvature_kernel(omega_pos, beta)
    integrand = K * sre_pos
    return (1.0 / np.pi) * np_trapz(integrand, omega_pos)


def calculate_qfi_from_cross_spectral(omega, spectral_real, beta):
    """
    Calculate the off-diagonal QFIM element from Re[S_μν(ω)].

    This is the metric (symmetric) part of the quantum geometric tensor,
    analogous to the diagonal QFI but for two different operators.

    F_Q^{μν} = 4 ∫₀^∞ Re[S_μν(ω)] tanh(βω/2)(1-e^{-βω}) dω

    Parameters
    ----------
    omega : ndarray
        Frequency array.
    spectral_real : ndarray
        Re[S_μν(ω)].
    beta : float
        Inverse temperature.

    Returns
    -------
    float
        Off-diagonal QFIM element.
    """
    pos = omega > OMEGA_MIN_CUTOFF
    omega_pos = omega[pos]
    spec_pos = spectral_real[pos]

    if len(omega_pos) == 0:
        return 0.0

    K = qfi_kernel(omega_pos, beta)
    integrand = spec_pos * K
    return 4.0 * np_trapz(integrand, omega_pos)


# ==============================================================================
# Data Loading — with both Real and Imaginary parts
# ==============================================================================

def load_spectral_complex_from_text(filepath):
    """
    Load spectral data from a text file, returning both Re and Im parts.

    Expected format:
        # Header lines
        omega  Re[S(ω)]  Im[S(ω)]  [Re[error]  Im[error]]

    Returns
    -------
    omega, spectral_real, spectral_imag : ndarray or (None, None, None)
    """
    try:
        data = np.loadtxt(filepath, comments='#')
        if data.ndim == 1:
            return None, None, None

        omega = data[:, 0]
        spectral_real = data[:, 1]
        # Im part may or may not be present
        if data.shape[1] >= 3:
            spectral_imag = data[:, 2]
        else:
            spectral_imag = np.zeros_like(spectral_real)

        return omega, spectral_real, spectral_imag

    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None, None, None


def load_spectral_complex_from_hdf5(h5_path, dataset_name):
    """
    Load spectral data from ed_results.h5, returning both Re and Im.

    Supports:
    1. time_correlations groups (FFT to frequency domain)
    2. Legacy spectral data with frequencies/spectral_real/spectral_imag

    Returns
    -------
    omega, spectral_real, spectral_imag : ndarray or (None, None, None)
    """
    if not HAS_H5PY:
        return None, None, None

    try:
        with h5py.File(h5_path, 'r') as f:
            if 'dynamical' not in f:
                return None, None, None
            dyn = f['dynamical']

            # New time_correlations format
            if 'time_correlations' in dyn:
                tc = dyn['time_correlations']
                if dataset_name in tc:
                    grp = tc[dataset_name]
                    times = grp['times'][:]
                    cr = grp['correlation_real'][:]
                    ci = grp['correlation_imag'][:]
                    # FFT to spectral domain (returns complex spectral)
                    omega, spec = _time_to_spectral_fft_complex(times, cr, ci)
                    if omega is not None:
                        return omega, spec.real, spec.imag
                    return None, None, None

            # Legacy format
            if dataset_name in dyn:
                ds = dyn[dataset_name]
                if 'frequencies' in ds and 'spectral_real' in ds:
                    omega = ds['frequencies'][:]
                    spec_re = ds['spectral_real'][:]
                    if 'spectral_imag' in ds:
                        spec_im = ds['spectral_imag'][:]
                    else:
                        spec_im = np.zeros_like(spec_re)
                    return omega, spec_re, spec_im

            return None, None, None
    except Exception as e:
        print(f"Error loading HDF5 {h5_path}/{dataset_name}: {e}")
        return None, None, None


def load_spectral_complex_from_dssf_hdf5(h5_path, operator_name,
                                          temperature_or_beta, sample_idx=None):
    """
    Load spectral data from dssf_results.h5, returning both Re and Im.

    Structure: /spectral/<operator_name>/<temp_or_beta>/sample_<N>/real,imag

    Returns
    -------
    omega, spectral_real, spectral_imag, beta : tuple or (None,None,None,None)
    """
    if not HAS_H5PY:
        return None, None, None, None

    try:
        with h5py.File(h5_path, 'r') as f:
            if '/spectral/frequencies' not in f:
                return None, None, None, None

            frequencies = f['/spectral/frequencies'][:]
            path = f'/spectral/{operator_name}/{temperature_or_beta}'
            if path not in f:
                return None, None, None, None

            group = f[path]

            if sample_idx is not None:
                sk = f'sample_{sample_idx}'
                if sk not in group:
                    return None, None, None, None
                sg = group[sk]
                spec_re = sg['real'][:]
                spec_im = sg['imag'][:] if 'imag' in sg else np.zeros_like(spec_re)
            else:
                all_re, all_im = [], []
                for k in group.keys():
                    if not k.startswith('sample_'):
                        continue
                    sg = group[k]
                    all_re.append(sg['real'][:])
                    if 'imag' in sg:
                        all_im.append(sg['imag'][:])
                    else:
                        all_im.append(np.zeros_like(all_re[-1]))
                if not all_re:
                    return None, None, None, None
                spec_re = np.nanmean(all_re, axis=0) if ENABLE_NAN_INTERPOLATION else np.mean(all_re, axis=0)
                spec_im = np.nanmean(all_im, axis=0) if ENABLE_NAN_INTERPOLATION else np.mean(all_im, axis=0)

            if len(frequencies) != len(spec_re):
                frequencies = np.linspace(frequencies[0], frequencies[-1], len(spec_re))

            beta = None
            if temperature_or_beta.startswith('T_'):
                T = float(temperature_or_beta.split('_')[1])
                beta = 1.0 / T if T > 0 else np.inf
            elif temperature_or_beta.startswith('beta_'):
                bs = temperature_or_beta.split('_')[1]
                beta = np.inf if bs.lower() in ('inf', 'infty') else float(bs)

            return frequencies, spec_re, spec_im, beta
    except Exception as e:
        print(f"Error loading dssf_results.h5: {e}")
        return None, None, None, None


def load_spectral_complex(filepath):
    """
    Unified loader: text, HDF5, or DSSF_HDF5 reference → (omega, Re, Im).
    """
    if filepath.startswith('DSSF_HDF5:'):
        parts = filepath[10:].split(':', 3)
        if len(parts) == 4:
            h5, op, tb, si = parts
            o, r, i, _ = load_spectral_complex_from_dssf_hdf5(h5, op, tb, int(si))
        elif len(parts) == 3:
            h5, op, tb = parts
            o, r, i, _ = load_spectral_complex_from_dssf_hdf5(h5, op, tb)
        else:
            return None, None, None
        return o, r, i

    if filepath.startswith('HDF5:'):
        parts = filepath[5:].split(':', 1)
        if len(parts) == 2:
            return load_spectral_complex_from_hdf5(parts[0], parts[1])
        return None, None, None

    return load_spectral_complex_from_text(filepath)


def _time_to_spectral_fft_complex(times, corr_real, corr_imag,
                                   omega_max=10.0, n_omega=1000):
    """FFT time-domain correlation → complex spectral function."""
    correlation = corr_real + 1j * corr_imag
    dt = times[1] - times[0] if len(times) > 1 else 1.0
    n = len(times)

    fft_vals = np.fft.fft(correlation) * dt
    freqs = np.fft.fftfreq(n, dt) * 2 * np.pi

    sort_idx = np.argsort(freqs)
    freqs = freqs[sort_idx]
    fft_vals = fft_vals[sort_idx]

    omega = np.linspace(-omega_max, omega_max, n_omega)
    mask = (freqs >= -omega_max) & (freqs <= omega_max)
    if np.sum(mask) > 2:
        f_re = interp1d(freqs[mask], fft_vals[mask].real,
                        kind='linear', bounds_error=False, fill_value=0.0)
        f_im = interp1d(freqs[mask], fft_vals[mask].imag,
                        kind='linear', bounds_error=False, fill_value=0.0)
        spectral = f_re(omega) + 1j * f_im(omega)
    else:
        spectral = np.zeros(n_omega, dtype=complex)

    return omega, spectral


# ==============================================================================
# Filename / Dataset Parsing
# ==============================================================================

def parse_spectral_filename(filename):
    """
    Extract species, beta, sample index from spectral filename.

    Returns (species, beta, sample_idx) or (None, None, None).
    """
    basename = os.path.basename(filename)
    m = re.match(
        r'^(.+?)_spectral_sample_(\d+)_beta_([0-9.+-eE]+|inf|infty)\.txt$',
        basename, re.IGNORECASE)
    if not m:
        return None, None, None

    species = m.group(1)
    sample_idx = int(m.group(2))
    bt = m.group(3)
    beta = np.inf if bt.lower() in ('inf', 'infty') else float(bt)
    return species, beta, sample_idx


def parse_time_correlation_name(name):
    """Parse HDF5 time_correlations group name → (species, beta, sample_idx)."""
    m = re.match(r'^(.+?)_sample(\d+)_beta([0-9.+-eE]+|inf|infty)_tpq$',
                 name, re.IGNORECASE)
    if not m:
        return None, None, None
    species = m.group(1)
    sample_idx = int(m.group(2))
    bt = m.group(3)
    beta = np.inf if bt.lower() in ('inf', 'infty') else float(bt)
    return species, beta, sample_idx


def parse_spectral_dataset_name(name):
    """Parse legacy HDF5 dataset name → (species, beta, sample_idx)."""
    m = re.match(
        r'^(.+?)_spectral_sample_(\d+)_beta_([0-9.+-eE]+|inf|infty)$',
        name, re.IGNORECASE)
    if not m:
        return None, None, None
    species = m.group(1)
    sample_idx = int(m.group(2))
    bt = m.group(3)
    beta = np.inf if bt.lower() in ('inf', 'infty') else float(bt)
    return species, beta, sample_idx


# ==============================================================================
# Cross-Spectral Pair Identification
# ==============================================================================

_SPIN_PAIR_RE = re.compile(
    r'^(Sx|Sy|Sz|Sp|Sm)(Sx|Sy|Sz|Sp|Sm)(_q_.+)$')


def decompose_operator_name(species):
    """
    Decompose operator name into (op1, op2, q_label).

    E.g. 'SxSz_q_Qx0_Qy0_Qz0' → ('Sx', 'Sz', '_q_Qx0_Qy0_Qz0')

    Returns (op1, op2, q_label) or (None, None, None) for non-matching names.
    """
    m = _SPIN_PAIR_RE.match(species)
    if m:
        return m.group(1), m.group(2), m.group(3)
    return None, None, None


def is_cross_spectral(species):
    """Return True if the species name encodes a cross-correlation (op1 ≠ op2)."""
    op1, op2, _ = decompose_operator_name(species)
    if op1 is None:
        return False
    return op1 != op2


def conjugate_species(species):
    """
    Return the conjugate pair name (swap op1 ↔ op2).

    E.g. 'SxSz_q_Qx0_Qy0_Qz0' → 'SzSx_q_Qx0_Qy0_Qz0'
    """
    op1, op2, ql = decompose_operator_name(species)
    if op1 is None:
        return None
    return f"{op2}{op1}{ql}"


def identify_cross_pairs(species_set):
    """
    From a set of species names, identify cross-spectral pairs.

    Returns a list of (species_μν, species_νμ_or_None) tuples.
    Only returns each unordered pair once.
    """
    cross = [s for s in species_set if is_cross_spectral(s)]
    seen = set()
    pairs = []

    for s in sorted(cross):
        if s in seen:
            continue
        conj = conjugate_species(s)
        seen.add(s)
        if conj in species_set:
            seen.add(conj)
            pairs.append((s, conj))
        else:
            pairs.append((s, None))

    return pairs


# ==============================================================================
# NaN Interpolation
# ==============================================================================

def interpolate_nan_values(omega, data):
    """Linearly interpolate NaN values in data."""
    if not np.any(np.isnan(data)):
        return data.copy()
    valid = ~np.isnan(data)
    if not np.any(valid):
        return np.zeros_like(data)
    f = interp1d(omega[valid], data[valid], kind='linear',
                 bounds_error=False, fill_value=0.0)
    out = data.copy()
    out[~valid] = f(omega[~valid])
    return out


# ==============================================================================
# Directory Scanning and Data Collection
# ==============================================================================

def _extract_beta_from_dirname(beta_dir):
    """Extract beta value from directory name like 'beta_3.456'."""
    basename = os.path.basename(beta_dir)
    m = re.match(r'^beta_([0-9.eE+-]+|inf|infty)$', basename, re.IGNORECASE)
    if not m:
        return None
    bt = m.group(1)
    return np.inf if bt.lower() in ('inf', 'infty') else float(bt)


def _assign_beta_bin(beta_val, bins, tol):
    """Assign beta to an existing bin or create a new one."""
    for idx, existing in enumerate(bins):
        if np.isinf(beta_val) and np.isinf(existing):
            return idx
        if not np.isinf(beta_val) and not np.isinf(existing):
            if abs(beta_val - existing) / max(abs(existing), 1e-12) < tol:
                return idx
    new_idx = len(bins)
    bins.append(beta_val)
    return new_idx


def list_spectral_datasets_hdf5(h5_path):
    """List all spectral datasets in ed_results.h5.

    Supports three naming conventions:
    1. time_correlations: {op}_sample{N}_beta{val}_tpq
    2. Legacy spectral: {op}_spectral_sample_{N}_beta_{val}
    3. Dynamical response: {op} or {op}_T{temperature}
       (from --dynamical-response mode, one group per operator/temperature)
    """
    if not HAS_H5PY:
        return []
    datasets = []
    try:
        with h5py.File(h5_path, 'r') as f:
            if 'dynamical' not in f:
                return datasets
            dyn = f['dynamical']

            if 'time_correlations' in dyn:
                for name in dyn['time_correlations'].keys():
                    species, beta, sidx = parse_time_correlation_name(name)
                    if species is not None:
                        datasets.append((name, species, beta, sidx))

            for name in dyn.keys():
                if name == 'time_correlations':
                    continue
                # Try legacy format first
                species, beta, sidx = parse_spectral_dataset_name(name)
                if species is not None:
                    datasets.append((name, species, beta, sidx))
                    continue
                # Try dynamical-response format: {op}_T{temperature}
                species, beta = _parse_dynamical_response_name(name, dyn)
                if species is not None:
                    datasets.append((name, species, beta, 0))
    except Exception as e:
        print(f"Error listing HDF5 datasets: {e}")
    return datasets


# Regex for dynamical-response operator names with temperature suffix
_DYN_RESP_T_RE = re.compile(r'^(.+?)_T([0-9.eE+-]+)$')


def _parse_dynamical_response_name(name, dyn_group=None):
    """Parse dynamical-response group name → (species, beta).

    Format: {operator}_T{temperature}  or  {operator} (with temperature attribute)
    The temperature is converted to beta = 1/T.
    """
    # Try {operator}_T{temperature} suffix first
    m = _DYN_RESP_T_RE.match(name)
    if m:
        species = m.group(1)
        T = float(m.group(2))
        beta = 1.0 / T if T > 0 else np.inf
        return species, beta

    # No _T suffix — check if group has 'temperature' attribute and spectral data
    if dyn_group is not None and name in dyn_group:
        grp = dyn_group[name]
        has_spectral = ('frequencies' in grp and 'spectral_real' in grp)
        if not has_spectral:
            return None, None
        T = 0.0
        if 'temperature' in grp.attrs:
            T = float(grp.attrs['temperature'])
        beta = 1.0 / T if T > 0 else np.inf
        return name, beta

    return None, None


def list_spectral_datasets_dssf_hdf5(h5_path):
    """List all spectral datasets in dssf_results.h5."""
    if not HAS_H5PY:
        return []
    datasets = []
    try:
        with h5py.File(h5_path, 'r') as f:
            if 'spectral' not in f:
                return datasets
            spec_grp = f['spectral']
            for op_name in spec_grp.keys():
                if op_name == 'frequencies':
                    continue
                op_grp = spec_grp[op_name]
                for tb_name in op_grp.keys():
                    tb_grp = op_grp[tb_name]
                    beta = None
                    if tb_name.startswith('T_'):
                        T = float(tb_name.split('_')[1])
                        beta = 1.0 / T if T > 0 else np.inf
                    elif tb_name.startswith('beta_'):
                        bs = tb_name.split('_')[1]
                        beta = np.inf if bs.lower() in ('inf', 'infty') else float(bs)
                    for sk in tb_grp.keys():
                        if sk.startswith('sample_'):
                            sidx = int(sk.split('_')[1])
                            datasets.append((op_name, tb_name, beta, sidx))
    except Exception as e:
        print(f"Error listing dssf HDF5 datasets: {e}")
    return datasets


def collect_spectral_files(structure_factor_dir, beta_tol=1e-2):
    """
    Scan directory tree for all spectral data files (HDF5 + text).

    Returns
    -------
    species_data : dict[species] → dict[beta_bin_idx] → list of file references
    beta_bins : list of representative beta values (one per bin)
    beta_bin_values : dict[beta_bin_idx] → list of all beta values in that bin
    species_names : set of all species found
    """
    species_data = defaultdict(lambda: defaultdict(list))
    species_names = set()
    beta_bins = []
    beta_bin_values = defaultdict(list)

    beta_dirs = glob.glob(os.path.join(structure_factor_dir, 'beta_*'))
    print(f"Found {len(beta_dirs)} beta directories")

    for beta_dir in beta_dirs:
        beta_val = _extract_beta_from_dirname(beta_dir)
        if beta_val is None:
            continue

        operator_subdirs = [d for d in glob.glob(os.path.join(beta_dir, '*'))
                           if os.path.isdir(d)]
        if not operator_subdirs:
            continue

        for op_subdir in operator_subdirs:
            h5_species_samples = set()

            # HDF5 first (preferred)
            h5_path = os.path.join(op_subdir, 'ed_results.h5')
            if HAS_H5PY and os.path.exists(h5_path):
                for ds_name, species, file_beta, sidx in list_spectral_datasets_hdf5(h5_path):
                    ref = f"HDF5:{h5_path}:{ds_name}"
                    bidx = _assign_beta_bin(file_beta, beta_bins, beta_tol)
                    beta_bin_values[bidx].append(file_beta)
                    species_data[species][bidx].append(ref)
                    species_names.add(species)
                    h5_species_samples.add((species, sidx, file_beta))

            # Text files for data not in HDF5
            for fpath in glob.glob(os.path.join(op_subdir,
                                                '*_spectral_sample*_beta_*.txt')):
                species, file_beta, sidx = parse_spectral_filename(fpath)
                if species is None:
                    continue
                if (species, sidx, file_beta) in h5_species_samples:
                    continue
                bidx = _assign_beta_bin(file_beta, beta_bins, beta_tol)
                beta_bin_values[bidx].append(file_beta)
                species_data[species][bidx].append(fpath)
                species_names.add(species)

    # Also scan dssf_results.h5
    dssf_h5 = os.path.join(structure_factor_dir, 'dssf_results.h5')
    if HAS_H5PY and os.path.exists(dssf_h5):
        print("Found dssf_results.h5, scanning...")
        for op, tb, beta_val, sidx in list_spectral_datasets_dssf_hdf5(dssf_h5):
            if beta_val is None:
                continue
            ref = f"DSSF_HDF5:{dssf_h5}:{op}:{tb}:{sidx}"
            bidx = _assign_beta_bin(beta_val, beta_bins, beta_tol)
            beta_bin_values[bidx].append(beta_val)
            species_data[op][bidx].append(ref)
            species_names.add(op)

    # Also scan for ed_results.h5 at the top level (--dynamical-response mode output)
    ed_h5 = os.path.join(structure_factor_dir, 'ed_results.h5')
    if HAS_H5PY and os.path.exists(ed_h5):
        ed_datasets = list_spectral_datasets_hdf5(ed_h5)
        if ed_datasets:
            print(f"Found ed_results.h5 with {len(ed_datasets)} dynamical datasets")
            for ds_name, species, file_beta, sidx in ed_datasets:
                ref = f"HDF5:{ed_h5}:{ds_name}"
                bidx = _assign_beta_bin(file_beta, beta_bins, beta_tol)
                beta_bin_values[bidx].append(file_beta)
                species_data[species][bidx].append(ref)
                species_names.add(species)

    return species_data, beta_bins, beta_bin_values, species_names


# ==============================================================================
# Processing Pipeline
# ==============================================================================

def _get_bin_beta(beta_vals):
    """Representative beta for a bin: median of collected values."""
    arr = np.array(beta_vals)
    if np.any(np.isinf(arr)):
        return np.inf
    return float(np.median(arr))


def _load_and_average_spectral_complex(file_list):
    """
    Load all spectral files in list and return averaged Re, Im and per-sample data.

    Returns
    -------
    mean_omega, mean_re, mean_im : ndarray
    individual_data : list of (omega, re, im, filepath)
    """
    all_data = []
    for fp in file_list:
        omega, sre, sim = load_spectral_complex(fp)
        if omega is not None:
            all_data.append((omega, sre, sim, fp))

    if not all_data:
        return None, None, None, []

    # Use the omega grid from the first file as reference
    ref_omega = all_data[0][0]

    re_stack, im_stack = [], []
    for omega, sre, sim, fp in all_data:
        if len(omega) == len(ref_omega) and np.allclose(omega, ref_omega):
            re_stack.append(sre)
            im_stack.append(sim)
        else:
            f_re = interp1d(omega, sre, kind='linear',
                            bounds_error=False, fill_value=0.0)
            f_im = interp1d(omega, sim, kind='linear',
                            bounds_error=False, fill_value=0.0)
            re_stack.append(f_re(ref_omega))
            im_stack.append(f_im(ref_omega))

    re_arr = np.array(re_stack)
    im_arr = np.array(im_stack)

    if ENABLE_NAN_INTERPOLATION:
        mean_re = np.nanmean(re_arr, axis=0)
        mean_im = np.nanmean(im_arr, axis=0)
    else:
        mean_re = np.mean(re_arr, axis=0)
        mean_im = np.mean(im_arr, axis=0)

    return ref_omega, mean_re, mean_im, all_data


def process_curvature_single_directory(structure_factor_dir, beta_tol=1e-2):
    """
    Process all cross-spectral data in a single structure_factor_results directory.

    Computes MUC and Berry curvature for every cross-spectral pair found.

    Returns
    -------
    all_curvature_data : dict[pair_label] → list of (beta, muc, berry, qfim_offdiag,
                                                      consistency, muc_std, n_samples)
    """
    print(f"Scanning: {structure_factor_dir}")
    species_data, beta_bins, beta_bin_values, species_names = \
        collect_spectral_files(structure_factor_dir, beta_tol)

    print(f"Found species: {sorted(species_names)}")

    # Identify cross-spectral pairs
    cross_pairs = identify_cross_pairs(species_names)
    # Also include all species that are cross-spectral (even without conjugate)
    cross_species = [s for s in species_names if is_cross_spectral(s)]

    if not cross_species:
        print("\nNo cross-spectral data found!")
        print("Curvature requires cross-correlations between different operators")
        print("(e.g., SxSz, SzSx). Configure spin_combinations in the ED config")
        print("to include off-diagonal pairs like '0,2' (SxSz in xyz basis).")
        return {}

    print(f"\nCross-spectral species: {sorted(cross_species)}")
    print(f"Cross pairs: {cross_pairs}")

    all_curvature_data = defaultdict(list)

    for species in sorted(cross_species):
        op1, op2, ql = decompose_operator_name(species)
        pair_label = f"{op1}{op2}{ql}"
        print(f"\n{'='*60}")
        print(f"Processing curvature for: {species}  (O_μ={op1}, O_ν={op2})")
        print(f"{'='*60}")

        if species not in species_data:
            print(f"  No data found for {species}")
            continue

        beta_groups = species_data[species]

        for bidx, file_list in sorted(beta_groups.items()):
            beta = _get_bin_beta(beta_bin_values[bidx])
            blabel = 'inf' if np.isinf(beta) else f'{beta:.6g}'
            print(f"\n  β ≈ {blabel}: {len(file_list)} files")

            omega, mean_re, mean_im, individual = \
                _load_and_average_spectral_complex(file_list)

            if omega is None:
                print(f"    Failed to load data")
                continue

            # Interpolate NaN values
            if ENABLE_NAN_INTERPOLATION:
                mean_re = interpolate_nan_values(omega, mean_re)
                mean_im = interpolate_nan_values(omega, mean_im)

            # Check if Im[S] is non-trivial
            im_norm = np.sqrt(np_trapz(mean_im**2, omega))
            re_norm = np.sqrt(np_trapz(mean_re**2, omega))
            if im_norm < 1e-15:
                print(f"    Warning: Im[S_μν] is essentially zero (norm={im_norm:.2e})")
                print(f"    This is expected for auto-correlations; curvature = 0.")

            # --- Compute MUC ---
            muc = calculate_muc_from_cross_spectral(omega, mean_im, beta) \
                  * CURVATURE_SCALE_FACTOR

            # --- Berry curvature (ground state formula, meaningful for β→∞) ---
            berry = calculate_berry_curvature_from_cross_spectral(omega, mean_im) \
                    * CURVATURE_SCALE_FACTOR

            # --- Off-diagonal QFIM for comparison ---
            qfim_od = calculate_qfi_from_cross_spectral(omega, mean_re, beta) \
                      * CURVATURE_SCALE_FACTOR

            # --- Consistency check ---
            cons = calculate_curvature_consistency_check(omega, mean_re, beta)

            # --- Per-sample MUC for error estimation ---
            per_sample_muc = []
            for s_omega, s_re, s_im, s_fp in individual:
                if ENABLE_NAN_INTERPOLATION:
                    s_im = interpolate_nan_values(s_omega, s_im)
                sm = calculate_muc_from_cross_spectral(s_omega, s_im, beta) \
                     * CURVATURE_SCALE_FACTOR
                per_sample_muc.append(sm)

            n_samples = len(per_sample_muc)
            muc_std = np.std(per_sample_muc) if n_samples > 1 else 0.0
            muc_sem = muc_std / np.sqrt(n_samples) if n_samples > 1 else 0.0

            print(f"    MUC U_μν       = {muc:+.6e} ± {muc_sem:.2e} ({n_samples} samples)")
            print(f"    Berry F_μν     = {berry:+.6e}  (T=0 formula)")
            print(f"    QFIM g_μν      = {qfim_od:+.6e}  (off-diagonal metric)")
            print(f"    Consistency    = {cons:+.6e}  (should ≈ 0)")
            print(f"    |Im[S]|/|Re[S]|= {im_norm/(re_norm+1e-30):.4f}")

            all_curvature_data[pair_label].append(
                (beta, muc, berry, qfim_od, cons, muc_sem, n_samples))

            # Save spectral data for this species/beta
            _save_curvature_results(species, beta, omega, mean_re, mean_im,
                                    muc, berry, qfim_od, cons, muc_sem,
                                    n_samples, structure_factor_dir)

    # Generate summary plots
    if all_curvature_data:
        _create_summary_plots(all_curvature_data, structure_factor_dir)

    print("\nCurvature processing complete!")
    return all_curvature_data


# ==============================================================================
# Output: Save Results
# ==============================================================================

def _save_curvature_results(species, beta, omega, spec_re, spec_im,
                            muc, berry, qfim_od, consistency, muc_sem,
                            n_samples, base_dir):
    """Save curvature results to text files."""
    blabel = 'inf' if np.isinf(beta) else f'{beta:.6g}'
    outdir = os.path.join(base_dir, 'curvature_results')
    os.makedirs(outdir, exist_ok=True)

    # Save spectral data with both Re and Im
    spec_file = os.path.join(outdir, f'{species}_cross_spectral_beta_{blabel}.txt')
    header = (f"# Cross-spectral function for {species} at beta={blabel}\n"
              f"# MUC = {muc:.8e}, Berry = {berry:.8e}, QFIM_offdiag = {qfim_od:.8e}\n"
              f"# Consistency check = {consistency:.8e}, MUC SEM = {muc_sem:.8e}\n"
              f"# n_samples = {n_samples}\n"
              f"# omega  Re[S_mn(omega)]  Im[S_mn(omega)]")
    np.savetxt(spec_file, np.column_stack([omega, spec_re, spec_im]),
               header=header, fmt='%.8e')

    # Save summary line to a cumulative file
    summary_file = os.path.join(outdir, f'{species}_curvature_summary.txt')
    write_header = not os.path.exists(summary_file)
    with open(summary_file, 'a') as f:
        if write_header:
            f.write("# beta  MUC  Berry_curvature  QFIM_offdiag  consistency  MUC_SEM  n_samples\n")
        f.write(f"{blabel}  {muc:.8e}  {berry:.8e}  {qfim_od:.8e}  "
                f"{consistency:.8e}  {muc_sem:.8e}  {n_samples}\n")


# ==============================================================================
# Plotting
# ==============================================================================

def _create_summary_plots(all_curvature_data, base_dir):
    """Generate summary plots for all cross-spectral pairs."""
    outdir = os.path.join(base_dir, 'curvature_results')
    os.makedirs(outdir, exist_ok=True)

    for pair_label, data_points in all_curvature_data.items():
        _plot_curvature_vs_temperature(pair_label, data_points, outdir)
        _plot_metric_vs_curvature(pair_label, data_points, outdir)
        _plot_kernel_comparison(pair_label, outdir)


def _plot_curvature_vs_temperature(pair_label, data_points, outdir):
    """Plot MUC and Berry curvature vs temperature (1/β)."""
    data_points = sorted(data_points, key=lambda x: x[0])

    betas = np.array([d[0] for d in data_points])
    mucs = np.array([d[1] for d in data_points])
    berrys = np.array([d[2] for d in data_points])
    muc_sems = np.array([d[5] for d in data_points])

    # Separate finite and infinite beta
    finite = np.isfinite(betas)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: MUC vs β
    ax = axes[0]
    if np.any(finite):
        ax.errorbar(betas[finite], mucs[finite], yerr=muc_sems[finite],
                     fmt='o-', color='C0', capsize=3, label='MUC $U_{\\mu\\nu}$')
    if np.any(~finite):
        ax.axhline(y=mucs[~finite][0], color='C1', linestyle='--',
                   label=f'$\\beta=\\infty$: {mucs[~finite][0]:.4e}')
    ax.set_xlabel('$\\beta$ (inverse temperature)')
    ax.set_ylabel('Mean Uhlmann curvature $U_{\\mu\\nu}$')
    ax.set_title(f'MUC vs $\\beta$: {pair_label}')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Right: MUC vs T = 1/β
    ax = axes[1]
    if np.any(finite):
        temps = 1.0 / betas[finite]
        ax.errorbar(temps, mucs[finite], yerr=muc_sems[finite],
                     fmt='s-', color='C2', capsize=3, label='MUC $U_{\\mu\\nu}$')
    ax.set_xlabel('Temperature $T = 1/\\beta$')
    ax.set_ylabel('Mean Uhlmann curvature $U_{\\mu\\nu}$')
    ax.set_title(f'MUC vs $T$: {pair_label}')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(os.path.join(outdir, f'{pair_label}_curvature_vs_T.png'), dpi=150)
    plt.close(fig)
    print(f"  Saved curvature vs T plot: {pair_label}_curvature_vs_T.png")


def _plot_metric_vs_curvature(pair_label, data_points, outdir):
    """
    Plot the metric (QFIM off-diagonal) and curvature (MUC) together
    to visualize the decomposition of the quantum geometric tensor.
    """
    data_points = sorted(data_points, key=lambda x: x[0])

    betas = np.array([d[0] for d in data_points])
    mucs = np.array([d[1] for d in data_points])
    qfim_ods = np.array([d[3] for d in data_points])
    muc_sems = np.array([d[5] for d in data_points])

    finite = np.isfinite(betas)
    if not np.any(finite):
        return

    temps = 1.0 / betas[finite]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(temps, np.abs(mucs[finite]), yerr=muc_sems[finite],
                fmt='o-', color='C0', capsize=3, label='|Curvature| $|U_{\\mu\\nu}|$')
    ax.plot(temps, np.abs(qfim_ods[finite]),
            's-', color='C3', label='|Metric| $|g_{\\mu\\nu}|$ (QFIM off-diag)')
    ax.set_xlabel('Temperature $T$')
    ax.set_ylabel('Magnitude')
    ax.set_title(f'Metric vs Curvature: {pair_label}')
    ax.legend()
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(os.path.join(outdir, f'{pair_label}_metric_vs_curvature.png'), dpi=150)
    plt.close(fig)
    print(f"  Saved metric vs curvature plot: {pair_label}_metric_vs_curvature.png")


def _plot_kernel_comparison(pair_label, outdir):
    """Plot the QFI and MUC kernels side-by-side for a few representative β values."""
    omega = np.linspace(0.01, 6.0, 500)
    betas = [1.0, 5.0, 20.0, 100.0]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # MUC kernel
    ax = axes[0]
    for b in betas:
        K = curvature_kernel(omega, b)
        ax.plot(omega, K, label=f'$\\beta={b}$')
    ax.set_xlabel('$\\omega$')
    ax.set_ylabel('$K_{\\mathrm{MUC}}(\\omega, \\beta)$')
    ax.set_title('MUC kernel: $\\tanh^2(\\beta\\omega/2)(1+e^{-\\beta\\omega})/\\omega^2$')
    ax.legend()
    ax.set_yscale('log')
    ax.set_ylim(bottom=1e-3)
    ax.grid(True, alpha=0.3)

    # QFI kernel
    ax = axes[1]
    for b in betas:
        K = qfi_kernel(omega, b)
        ax.plot(omega, K, label=f'$\\beta={b}$')
    ax.set_xlabel('$\\omega$')
    ax.set_ylabel('$K_{\\mathrm{QFI}}(\\omega, \\beta)$')
    ax.set_title('QFI kernel: $\\tanh(\\beta\\omega/2)(1-e^{-\\beta\\omega})$')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(os.path.join(outdir, f'{pair_label}_kernel_comparison.png'), dpi=150)
    plt.close(fig)


# ==============================================================================
# Parameter Sweep Support
# ==============================================================================

def parse_curvature_across_parameter(data_dir, param_pattern='Jpm'):
    """
    Run curvature analysis across a parameter sweep.

    Expects directory structure:
        data_dir/
            {param_pattern}_{value}/
                structure_factor_results/
                    beta_*/
                        ...

    Returns
    -------
    all_data : dict[pair_label] → list of (param_value, beta, muc, berry, qfim_od, ...)
    """
    # Find parameter directories
    param_re = re.compile(
        rf'^{re.escape(param_pattern)}_([0-9.eE+-]+)$')

    param_dirs = []
    for entry in sorted(os.listdir(data_dir)):
        m = param_re.match(entry)
        if m:
            pval = float(m.group(1))
            pdir = os.path.join(data_dir, entry)
            if os.path.isdir(pdir):
                param_dirs.append((pval, pdir))

    if not param_dirs:
        print(f"No parameter directories matching {param_pattern}_* found in {data_dir}")
        return {}

    print(f"Found {len(param_dirs)} parameter values for {param_pattern}")

    all_sweep_data = defaultdict(list)

    for pval, pdir in sorted(param_dirs):
        sf_dir = _resolve_data_dir(pdir)
        if not os.path.exists(sf_dir):
            print(f"  Skipping {param_pattern}={pval}: no data directory found")
            continue

        print(f"\n{'#'*70}")
        print(f"# {param_pattern} = {pval}")
        print(f"{'#'*70}")

        curv_data = process_curvature_single_directory(sf_dir)

        for pair_label, points in curv_data.items():
            for (beta, muc, berry, qfim_od, cons, muc_sem, nsamp) in points:
                all_sweep_data[pair_label].append(
                    (pval, beta, muc, berry, qfim_od, cons, muc_sem, nsamp))

    # Save parameter sweep summary
    if all_sweep_data:
        outdir = os.path.join(data_dir, 'curvature_sweep_results')
        os.makedirs(outdir, exist_ok=True)

        for pair_label, points in all_sweep_data.items():
            fname = os.path.join(outdir,
                                 f'{pair_label}_{param_pattern}_sweep.txt')
            with open(fname, 'w') as f:
                f.write(f"# {param_pattern} sweep curvature data for {pair_label}\n")
                f.write(f"# {param_pattern}  beta  MUC  Berry  QFIM_offdiag  "
                        f"consistency  MUC_SEM  n_samples\n")
                for row in sorted(points):
                    pv, b, m, br, q, c, se, ns = row
                    bl = 'inf' if np.isinf(b) else f'{b:.6g}'
                    f.write(f"{pv:.6g}  {bl}  {m:.8e}  {br:.8e}  {q:.8e}  "
                            f"{c:.8e}  {se:.8e}  {ns}\n")
            print(f"Saved sweep data: {fname}")

        _plot_parameter_sweep_summary(all_sweep_data, data_dir, param_pattern)

    return all_sweep_data


def _plot_parameter_sweep_summary(all_data, data_dir, param_pattern):
    """Generate heatmap-style plots for parameter sweep curvature data."""
    outdir = os.path.join(data_dir, 'curvature_sweep_results')
    os.makedirs(outdir, exist_ok=True)

    for pair_label, points in all_data.items():
        # Organize by param and beta
        param_vals = sorted(set(p[0] for p in points))
        beta_vals = sorted(set(p[1] for p in points if np.isfinite(p[1])))

        if not param_vals or not beta_vals:
            continue

        # Create grid
        n_p, n_b = len(param_vals), len(beta_vals)
        Z_muc = np.full((n_b, n_p), np.nan)
        Z_metric = np.full((n_b, n_p), np.nan)

        p_idx = {p: i for i, p in enumerate(param_vals)}
        b_idx = {b: i for i, b in enumerate(beta_vals)}

        for pv, b, muc, berry, qfim, cons, sem, ns in points:
            if not np.isfinite(b):
                continue
            pi = p_idx.get(pv)
            bi = b_idx.get(b)
            if pi is not None and bi is not None:
                Z_muc[bi, pi] = muc
                Z_metric[bi, pi] = qfim

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # MUC heatmap
        ax = axes[0]
        im = ax.pcolormesh(param_vals, beta_vals, Z_muc,
                           shading='nearest', cmap='RdBu_r')
        plt.colorbar(im, ax=ax, label='MUC $U_{\\mu\\nu}$')
        ax.set_xlabel(param_pattern)
        ax.set_ylabel('$\\beta$')
        ax.set_title(f'Mean Uhlmann Curvature: {pair_label}')

        # Metric heatmap
        ax = axes[1]
        im = ax.pcolormesh(param_vals, beta_vals, Z_metric,
                           shading='nearest', cmap='viridis')
        plt.colorbar(im, ax=ax, label='QFIM off-diag $g_{\\mu\\nu}$')
        ax.set_xlabel(param_pattern)
        ax.set_ylabel('$\\beta$')
        ax.set_title(f'Off-diagonal Metric: {pair_label}')

        plt.tight_layout()
        fig.savefig(os.path.join(outdir,
                                 f'{pair_label}_{param_pattern}_heatmap.png'),
                    dpi=150)
        plt.close(fig)
        print(f"Saved heatmap: {pair_label}_{param_pattern}_heatmap.png")


def _resolve_data_dir(directory):
    """Resolve the data directory, trying multiple conventions.

    Priority:
    1. directory itself if it ends with structure_factor_results
    2. directory/structure_factor_results/ if it exists
    3. directory/output/ if it contains ed_results.h5 or dssf_results.h5
    4. directory itself (may contain ed_results.h5 directly)
    """
    if directory.endswith('structure_factor_results'):
        return directory

    sf = os.path.join(directory, 'structure_factor_results')
    if os.path.isdir(sf):
        return sf

    out = os.path.join(directory, 'output')
    if os.path.isdir(out):
        for h5name in ('ed_results.h5', 'dssf_results.h5'):
            if os.path.exists(os.path.join(out, h5name)):
                return out

    # Check directory itself for HDF5 files
    for h5name in ('ed_results.h5', 'dssf_results.h5'):
        if os.path.exists(os.path.join(directory, h5name)):
            return directory

    # Fallback: still try structure_factor_results (will error later)
    return sf


# ==============================================================================
# CLI Entry Point
# ==============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description='Calculate quantum geometric curvature (Berry / MUC) '
                    'from cross-spectral function data')
    parser.add_argument('directory', type=str,
                        help='Directory containing structure_factor_results, '
                             'output/, or ed_results.h5')
    parser.add_argument('--beta-tol', type=float, default=1e-2,
                        help='Tolerance for grouping beta values (default: 1e-2)')
    parser.add_argument('--param-sweep', type=str, default=None,
                        help='Parameter name for sweep analysis (e.g., Jpm, h, J)')
    parser.add_argument('--list-species', action='store_true',
                        help='Only list available species and cross-spectral pairs, '
                             'then exit')

    args = parser.parse_args()

    if args.param_sweep:
        print(f"Running curvature analysis across {args.param_sweep} sweep")
        results = parse_curvature_across_parameter(
            args.directory, args.param_sweep)

    elif args.list_species:
        # Just scan and report available data
        sf_dir = _resolve_data_dir(args.directory)

        if not os.path.exists(sf_dir):
            print(f"Error: Directory not found: {sf_dir}")
            sys.exit(1)

        species_data, _, _, species_names = \
            collect_spectral_files(sf_dir, args.beta_tol)

        print(f"\nAll species found ({len(species_names)}):")
        for s in sorted(species_names):
            op1, op2, ql = decompose_operator_name(s)
            tag = " [CROSS]" if is_cross_spectral(s) else " [AUTO]"
            if op1:
                print(f"  {s}  →  O_μ={op1}, O_ν={op2}{tag}")
            else:
                print(f"  {s}")

        pairs = identify_cross_pairs(species_names)
        if pairs:
            print(f"\nCross-spectral pairs for curvature:")
            for s1, s2 in pairs:
                print(f"  {s1}  ↔  {s2 or '(conjugate not found)'}")
        else:
            print("\nNo cross-spectral pairs found.")
            print("To compute curvature, configure spin_combinations in ED config")
            print("to include off-diagonal pairs, e.g., '0,2;2,0' for SxSz/SzSx.")

    else:
        # Single directory mode
        sf_dir = _resolve_data_dir(args.directory)

        if not os.path.exists(sf_dir):
            print(f"Error: Directory not found: {sf_dir}")
            sys.exit(1)

        print(f"Processing cross-spectral data from: {sf_dir}")
        results = process_curvature_single_directory(sf_dir, args.beta_tol)

    print("\n" + "=" * 70)
    print("Curvature analysis complete!")
    print("=" * 70)
