#!/usr/bin/env python3
"""
Verify that the integral of TPQ_DSSF continued_fraction spectrum
equals the SSSF value at the same temperature.

For a TPQ state |ψ(β)⟩ at inverse temperature β:
- SSSF: S_static(q) = ⟨ψ|O†(q)O(q)|ψ⟩
- Continued fraction spectrum: S(q,ω) where ∫ S(q,ω) dω = S_static(q)

This script verifies this relationship holds.
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

def integrate_spectrum(frequencies, spectrum):
    """
    Compute integral of spectrum using trapezoidal rule.
    
    Args:
        frequencies: Array of frequency points
        spectrum: Array of spectral function values
    
    Returns:
        Integral value
    """
    return np.trapz(spectrum, frequencies)

def load_continued_fraction_data(h5_file, operator_name, sample_idx=0):
    """
    Load continued fraction spectral data from HDF5.
    
    Returns:
        Dictionary with keys: 'frequencies', 'beta', 'temperature', 'spectrum'
    """
    # Handle both string and Path objects
    if isinstance(h5_file, str):
        h5_file = Path(h5_file)
    
    # Check multiple possible file locations if file doesn't exist
    if not h5_file.exists():
        possible_files = [
            h5_file.parent / 'structure_factor_results' / 'dssf_results.h5',
            h5_file.parent / 'DSSF_results.h5'
        ]
        for f in possible_files:
            if f.exists():
                h5_file = f
                break
    
    results = {}
    
    with h5py.File(h5_file, 'r') as f:
        # Load frequencies
        if '/spectral/frequencies' in f:
            results['frequencies'] = f['/spectral/frequencies'][:]
        else:
            raise ValueError("No frequency data found")
        
        # Find operator group
        op_path = f'/spectral/{operator_name}'
        if op_path not in f:
            raise ValueError(f"Operator {operator_name} not found in spectral data")
        
        # Get temperature/beta groups
        op_group = f[op_path]
        temp_groups = list(op_group.keys())
        
        results['spectra'] = {}
        
        for temp_group in temp_groups:
            sample_path = f'{op_path}/{temp_group}/sample_{sample_idx}'
            if sample_path not in f:
                continue
                
            sample_grp = f[sample_path]
            
            # Get beta and temperature
            beta = sample_grp.attrs['beta']
            temperature = 1.0 / beta if beta > 0 else np.inf
            
            # Load spectrum (real part)
            if 'real' in sample_grp:
                spectrum = sample_grp['real'][:]
            else:
                raise ValueError(f"No real spectrum data in {sample_path}")
            
            results['spectra'][temperature] = {
                'beta': beta,
                'temperature': temperature,
                'spectrum': spectrum
            }
    
    return results

def load_sssf_data(h5_file, operator_name, sample_idx=0):
    """
    Load static structure factor (SSSF) data from HDF5.
    
    Returns:
        Dictionary with keys: 'temperatures', 'expectation'
    """
    # Handle both string and Path objects
    if isinstance(h5_file, str):
        h5_file = Path(h5_file)
    
    # Check multiple possible file locations if file doesn't exist
    if not h5_file.exists():
        possible_files = [
            h5_file.parent / 'structure_factor_results' / 'dssf_results.h5',
            h5_file.parent / 'DSSF_results.h5'
        ]
        for f in possible_files:
            if f.exists():
                h5_file = f
                break
    
    results = {}
    
    with h5py.File(h5_file, 'r') as f:
        # Find operator in static group
        op_path = f'/static/{operator_name}'
        if op_path not in f:
            raise ValueError(f"Operator {operator_name} not found in static data")
        
        sample_path = f'{op_path}/sample_{sample_idx}'
        if sample_path not in f:
            raise ValueError(f"Sample {sample_idx} not found for {operator_name}")
        
        sample_grp = f[sample_path]
        
        # Load temperature and expectation value data
        results['temperatures'] = sample_grp['temperatures'][:]
        results['expectation'] = sample_grp['expectation'][:]
    
    return results

def verify_integral_relation(data_dir, operator_name='SmSp_q_Qx0_Qy0_Qz0', sample_idx=0, 
                             tolerance=0.05, verbose=True):
    """
    Verify that ∫ S(q,ω) dω = S_static(q) for TPQ states.
    
    Args:
        data_dir: Directory containing DSSF results HDF5 file
        operator_name: Name of the operator to check
        sample_idx: Sample index (default 0)
        tolerance: Relative tolerance for verification (default 5%)
        verbose: Print detailed output
    
    Returns:
        Dictionary with verification results
    """
    # Check multiple possible HDF5 file locations
    possible_files = [
        Path(data_dir) / 'DSSF_results.h5',
        Path(data_dir) / 'structure_factor_results' / 'dssf_results.h5'
    ]
    
    h5_file = None
    for f in possible_files:
        if f.exists():
            h5_file = f
            break
    
    if h5_file is None:
        raise FileNotFoundError(f"HDF5 file not found in {data_dir} or {data_dir}/structure_factor_results/")
    
    # Load data
    if verbose:
        print(f"Loading data from {h5_file}")
        print(f"Operator: {operator_name}, Sample: {sample_idx}\n")
    
    cf_data = load_continued_fraction_data(h5_file, operator_name, sample_idx)
    sssf_data = load_sssf_data(h5_file, operator_name, sample_idx)
    
    frequencies = cf_data['frequencies']
    
    # Verify for each temperature
    results = {
        'passed': True,
        'details': []
    }
    
    if verbose:
        print("=" * 80)
        print("VERIFICATION: ∫ S(q,ω) dω = S_static(q)")
        print("=" * 80)
        print(f"{'Temperature':>12} {'S_static':>15} {'∫ S(ω) dω':>15} {'Ratio':>12} {'Status':>10}")
        print("-" * 80)
    
    for T_sssf, S_static in zip(sssf_data['temperatures'], sssf_data['expectation']):
        # Find matching temperature in continued fraction data
        # Allow small tolerance in temperature matching
        found = False
        for T_cf, cf_info in cf_data['spectra'].items():
            if np.abs(T_cf - T_sssf) / T_sssf < 1e-6:  # 0.0001% tolerance
                found = True
                spectrum = cf_info['spectrum']
                
                # Compute integral
                integral = integrate_spectrum(frequencies, spectrum)
                
                # Compute ratio
                ratio = integral / S_static if S_static != 0 else np.inf
                
                # Check if within tolerance
                passed = np.abs(ratio - 1.0) < tolerance
                
                result = {
                    'temperature': T_sssf,
                    'S_static': S_static,
                    'integral': integral,
                    'ratio': ratio,
                    'passed': passed
                }
                results['details'].append(result)
                
                if not passed:
                    results['passed'] = False
                
                if verbose:
                    status = "✓ PASS" if passed else "✗ FAIL"
                    print(f"{T_sssf:12.6f} {S_static:15.8f} {integral:15.8f} {ratio:12.6f} {status:>10}")
                
                break
        
        if not found and verbose:
            print(f"{T_sssf:12.6f} {'N/A':>15} {'N/A':>15} {'N/A':>12} {'MISSING':>10}")
    
    if verbose:
        print("-" * 80)
        print(f"\nOverall: {'PASS' if results['passed'] else 'FAIL'}")
        print(f"Tolerance: ±{tolerance*100:.1f}%\n")
    
    return results

def plot_comparison(data_dir, operator_name='SmSp_q_Qx0_Qy0_Qz0', sample_idx=0, 
                   temperature=None, output_file=None):
    """
    Plot spectrum and show integral vs SSSF comparison.
    
    Args:
        data_dir: Directory containing DSSF results HDF5 file
        operator_name: Name of the operator to plot
        sample_idx: Sample index
        temperature: Specific temperature to plot (or None for all)
        output_file: Save plot to file (optional)
    """
    # Check multiple possible HDF5 file locations
    possible_files = [
        Path(data_dir) / 'DSSF_results.h5',
        Path(data_dir) / 'structure_factor_results' / 'dssf_results.h5'
    ]
    
    h5_file = None
    for f in possible_files:
        if f.exists():
            h5_file = f
            break
    
    if h5_file is None:
        raise FileNotFoundError(f"HDF5 file not found in {data_dir} or {data_dir}/structure_factor_results/")
    
    cf_data = load_continued_fraction_data(h5_file, operator_name, sample_idx)
    sssf_data = load_sssf_data(h5_file, operator_name, sample_idx)
    
    frequencies = cf_data['frequencies']
    
    # Determine which temperatures to plot
    if temperature is not None:
        temps_to_plot = [T for T in cf_data['spectra'].keys() 
                        if np.abs(T - temperature) / temperature < 1e-3]
    else:
        temps_to_plot = sorted(cf_data['spectra'].keys())[:5]  # Plot first 5
    
    n_plots = len(temps_to_plot)
    if n_plots == 0:
        print("No data to plot")
        return
    
    fig, axes = plt.subplots(n_plots, 1, figsize=(10, 3*n_plots))
    if n_plots == 1:
        axes = [axes]
    
    for ax, T in zip(axes, temps_to_plot):
        cf_info = cf_data['spectra'][T]
        spectrum = cf_info['spectrum']
        
        # Find corresponding SSSF value
        idx = np.argmin(np.abs(sssf_data['temperatures'] - T))
        S_static = sssf_data['expectation'][idx]
        
        # Compute integral
        integral = integrate_spectrum(frequencies, spectrum)
        
        # Plot spectrum
        ax.plot(frequencies, spectrum, 'b-', linewidth=1.5, label='S(ω)')
        ax.fill_between(frequencies, 0, spectrum, alpha=0.3)
        ax.axhline(0, color='k', linestyle='--', linewidth=0.5)
        ax.set_xlabel('ω')
        ax.set_ylabel('S(ω)')
        ax.set_title(f'T = {T:.4f}: ∫ S(ω) dω = {integral:.6f}, S_static = {S_static:.6f}, '
                    f'ratio = {integral/S_static:.4f}')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output_file}")
    else:
        plt.show()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Verify TPQ_DSSF continued fraction integral equals SSSF"
    )
    parser.add_argument('data_dir', help='Directory containing DSSF_results.h5')
    parser.add_argument('--operator', default='SmSp_q_Qx0_Qy0_Qz0',
                       help='Operator name (default: SmSp_q_Qx0_Qy0_Qz0)')
    parser.add_argument('--sample', type=int, default=0,
                       help='Sample index (default: 0)')
    parser.add_argument('--tolerance', type=float, default=0.05,
                       help='Relative tolerance for pass/fail (default: 0.05)')
    parser.add_argument('--plot', action='store_true',
                       help='Generate comparison plots')
    parser.add_argument('--temperature', type=float, default=None,
                       help='Specific temperature for plotting (optional)')
    parser.add_argument('--output', help='Output file for plot (optional)')
    
    args = parser.parse_args()
    
    # Run verification
    results = verify_integral_relation(
        args.data_dir,
        operator_name=args.operator,
        sample_idx=args.sample,
        tolerance=args.tolerance,
        verbose=True
    )
    
    # Generate plots if requested
    if args.plot:
        plot_comparison(
            args.data_dir,
            operator_name=args.operator,
            sample_idx=args.sample,
            temperature=args.temperature,
            output_file=args.output
        )
    
    # Exit with appropriate code
    sys.exit(0 if results['passed'] else 1)
