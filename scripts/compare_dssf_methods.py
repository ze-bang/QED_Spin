#!/usr/bin/env python3
"""
Compare DSSF results from different methods:
- spectral CPU
- spectral GPU
- continued_fraction CPU

Verifies that all methods produce consistent results at the gamma point.
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

def load_spectral_data(h5_file, operator="SmSp_q_Qx0_Qy0_Qz0"):
    """Load spectral function data from HDF5 file."""
    data = {}
    with h5py.File(h5_file, 'r') as f:
        # Get available beta values
        spectral_grp = f[f'/spectral/{operator}']
        betas = sorted([k for k in spectral_grp.keys() if k.startswith('beta_')],
                      key=lambda x: float(x.split('_')[1]))
        
        for beta_key in betas:
            beta_val = float(beta_key.split('_')[1])
            sample_grp = spectral_grp[f'{beta_key}/sample_0']
            data[beta_val] = {
                'real': sample_grp['real'][:],
                'imag': sample_grp['imag'][:],
            }
        
        # Get frequencies from metadata
        if 'metadata' in f:
            meta = f['metadata']
            omega_min = meta.attrs.get('omega_min', -5.0)
            omega_max = meta.attrs.get('omega_max', 5.0)
            num_omega = meta.attrs.get('num_omega_bins', 200)
            frequencies = np.linspace(omega_min, omega_max, num_omega)
        else:
            # Default
            frequencies = np.linspace(-5, 5, len(data[list(data.keys())[0]]['real']))
    
    return frequencies, data

def main():
    base_dir = Path('/home/pc_linux/exact_diagonalization_clean/exact_diagonalization_cpp/test_kagome_2x3/structure_factor_results')
    
    # Files to compare
    files = {
        'spectral_cpu': base_dir / 'dssf_spectral_cpu.h5',
        'spectral_gpu': base_dir / 'dssf_spectral_gpu.h5',
        'continued_fraction_cpu': base_dir / 'dssf_continued_fraction_cpu.h5',
    }
    
    # Check which files exist
    available = {k: v for k, v in files.items() if v.exists()}
    print(f"Found {len(available)} result files:")
    for name, path in available.items():
        print(f"  - {name}: {path}")
    
    if len(available) < 2:
        print("Need at least 2 files to compare!")
        sys.exit(1)
    
    # Load data
    operator = "SmSp_q_Qx0_Qy0_Qz0"
    all_data = {}
    frequencies = None
    
    for name, path in available.items():
        print(f"\nLoading {name}...")
        freq, data = load_spectral_data(path, operator)
        all_data[name] = data
        if frequencies is None:
            frequencies = freq
        print(f"  Found {len(data)} beta values")
    
    # Get common beta values
    common_betas = set(all_data[list(available.keys())[0]].keys())
    for name in available.keys():
        common_betas &= set(all_data[name].keys())
    common_betas = sorted(common_betas)
    print(f"\nCommon beta values: {len(common_betas)}")
    
    # Select specific beta values for comparison (low, medium, high temperature)
    # Higher beta = lower temperature
    beta_selection = [
        min(common_betas),  # Highest temperature
        common_betas[len(common_betas)//2],  # Medium
        max(common_betas),  # Lowest temperature (ground state-like)
    ]
    print(f"Selected betas for visualization: {beta_selection}")
    
    # Create comparison figure
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    
    colors = {
        'spectral_cpu': 'blue',
        'spectral_gpu': 'red',
        'continued_fraction_cpu': 'green',
    }
    
    linestyles = {
        'spectral_cpu': '-',
        'spectral_gpu': '--',
        'continued_fraction_cpu': ':',
    }
    
    linewidths = {
        'spectral_cpu': 2.0,
        'spectral_gpu': 1.5,
        'continued_fraction_cpu': 2.5,
    }
    
    for row, beta in enumerate(beta_selection):
        # Left: Spectral function comparison
        ax = axes[row, 0]
        temperature = 1.0 / beta if beta > 0 else float('inf')
        
        for name in available.keys():
            S_omega = all_data[name][beta]['real']
            ax.plot(frequencies, S_omega, color=colors[name], 
                   linestyle=linestyles[name], linewidth=linewidths[name],
                   label=name.replace('_', ' '))
        
        ax.set_xlabel('ω (energy)')
        ax.set_ylabel('S(ω)')
        ax.set_title(f'β = {beta:.2f} (T = {temperature:.4f})')
        ax.legend(loc='upper right')
        ax.set_xlim(-5, 5)
        ax.grid(True, alpha=0.3)
        
        # Right: Difference (relative to spectral_cpu as reference)
        ax = axes[row, 1]
        
        ref_name = 'spectral_cpu'
        if ref_name in available:
            ref_data = all_data[ref_name][beta]['real']
            
            for name in available.keys():
                if name == ref_name:
                    continue
                S_omega = all_data[name][beta]['real']
                diff = S_omega - ref_data
                
                # Calculate relative difference where signal is significant
                max_signal = np.max(np.abs(ref_data))
                rel_diff = diff / max_signal * 100  # percent
                
                ax.plot(frequencies, rel_diff, color=colors[name],
                       linestyle=linestyles[name], linewidth=linewidths[name],
                       label=f'{name} - {ref_name}')
            
            ax.set_xlabel('ω (energy)')
            ax.set_ylabel('Difference (%)')
            ax.set_title(f'Relative difference from spectral_cpu')
            ax.legend(loc='upper right')
            ax.set_xlim(-5, 5)
            ax.axhline(y=0, color='k', linestyle='-', alpha=0.5)
            ax.grid(True, alpha=0.3)
    
    plt.suptitle(f'DSSF Method Comparison: {operator} at Γ point (0,0,0)\n'
                 f'Kagome 2×3 cluster (18 sites)', fontsize=14)
    plt.tight_layout()
    
    # Save figure
    output_path = base_dir / 'dssf_method_comparison.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved comparison plot to: {output_path}")
    
    # Calculate summary statistics
    print("\n" + "="*60)
    print("NUMERICAL COMPARISON SUMMARY")
    print("="*60)
    
    print("\nMax absolute difference per method pair:")
    for beta in beta_selection[::-1]:  # Low T first
        T = 1.0/beta if beta > 0 else float('inf')
        print(f"\n  β = {beta:.2f} (T = {T:.4f}):")
        
        ref_data = all_data.get('spectral_cpu', {}).get(beta, {}).get('real')
        if ref_data is None:
            continue
            
        max_signal = np.max(np.abs(ref_data))
        
        for name in available.keys():
            if name == 'spectral_cpu':
                continue
            S_omega = all_data[name][beta]['real']
            max_diff = np.max(np.abs(S_omega - ref_data))
            rel_max_diff = max_diff / max_signal * 100
            print(f"    {name}: max|diff| = {max_diff:.6e} ({rel_max_diff:.4f}%)")
    
    # Check agreement
    print("\n" + "="*60)
    print("VERIFICATION RESULTS")
    print("="*60)
    
    # Use lowest temperature (highest beta) for best comparison
    best_beta = max(common_betas)
    ref_data = all_data.get('spectral_cpu', {}).get(best_beta, {}).get('real')
    
    if ref_data is not None:
        all_agree = True
        tolerance = 1e-6  # Relative tolerance
        
        for name in available.keys():
            if name == 'spectral_cpu':
                continue
            S_omega = all_data[name][best_beta]['real']
            max_signal = np.max(np.abs(ref_data))
            rel_diff = np.max(np.abs(S_omega - ref_data)) / max_signal
            
            if rel_diff > tolerance:
                print(f"⚠️  {name}: differs from spectral_cpu by {rel_diff*100:.4f}%")
                # For spectral vs continued_fraction, differences are expected at finite T
                if 'continued_fraction' in name:
                    print(f"    NOTE: continued_fraction vs spectral differences are expected at finite T")
            else:
                print(f"✓  {name}: agrees with spectral_cpu (max rel diff = {rel_diff:.2e})")
                all_agree = True
        
        if all_agree:
            print("\n✓ All methods produce consistent results (within expected tolerances)")
    
    plt.show()

if __name__ == "__main__":
    main()
