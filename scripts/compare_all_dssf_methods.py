#!/usr/bin/env python3
"""
Compare all 4 DSSF computation methods:
1. Spectral CPU (eigendecomposition)
2. Spectral GPU (eigendecomposition)
3. Continued Fraction CPU
4. Continued Fraction GPU

This script loads results from HDF5 files and compares the spectral functions.
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

def load_dssf_data(h5_path, sample_idx=0, beta_idx=0):
    """Load DSSF data from HDF5 file for a specific sample at highest beta (lowest T)."""
    data = {}
    with h5py.File(h5_path, 'r') as f:
        # Get frequencies from spectral/frequencies
        if 'spectral' in f and 'frequencies' in f['spectral']:
            data['omega'] = f['spectral']['frequencies'][:]
        
        # Find the SmSp operator group
        if 'spectral' in f:
            for key in f['spectral'].keys():
                if key.startswith('SmSp') or key.startswith('Sz'):
                    grp = f['spectral'][key]
                    # Find the highest beta (lowest temperature)
                    beta_keys = [k for k in grp.keys() if k.startswith('beta_')]
                    if beta_keys:
                        # Sort by beta value (descending) to get lowest temperature
                        beta_keys_sorted = sorted(beta_keys, 
                                                  key=lambda x: float(x.split('_')[1]),
                                                  reverse=True)
                        beta_key = beta_keys_sorted[beta_idx] if beta_idx < len(beta_keys_sorted) else beta_keys_sorted[0]
                        
                        sample_key = f'sample_{sample_idx}'
                        if sample_key in grp[beta_key]:
                            sample_grp = grp[beta_key][sample_key]
                            data['S_real'] = sample_grp['real'][:]
                            data['S_imag'] = sample_grp['imag'][:]
                            data['operator'] = key
                            data['beta'] = float(beta_key.split('_')[1])
                            break
    return data

def compare_methods(base_dir):
    """Compare all 4 methods."""
    methods = {
        'Spectral CPU': 'spectral_cpu.h5',
        'Spectral GPU': 'spectral_gpu.h5',
        'CF CPU': 'cf_cpu.h5',
        'CF GPU': 'cf_gpu.h5'
    }
    
    # Timing data (from the runs)
    timings = {
        'Spectral CPU': 63.05,
        'Spectral GPU': 3.81,
        'CF CPU': 54.85,
        'CF GPU': 3.07
    }
    
    results = {}
    for name, filename in methods.items():
        path = os.path.join(base_dir, filename)
        if os.path.exists(path):
            results[name] = load_dssf_data(path)
            print(f"Loaded {name}: {len(results[name].get('omega', []))} frequency points")
        else:
            print(f"Warning: {path} not found")
    
    if len(results) < 2:
        print("Need at least 2 methods to compare")
        return
    
    # Create figure with 3 subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: All spectra overlaid
    ax1 = axes[0, 0]
    colors = {'Spectral CPU': 'blue', 'Spectral GPU': 'red', 
              'CF CPU': 'green', 'CF GPU': 'orange'}
    linestyles = {'Spectral CPU': '-', 'Spectral GPU': '--',
                  'CF CPU': '-.', 'CF GPU': ':'}
    
    for name, data in results.items():
        if 'omega' in data:
            ax1.plot(data['omega'], data['S_real'], 
                    color=colors.get(name, 'black'),
                    linestyle=linestyles.get(name, '-'),
                    linewidth=2 if 'CPU' in name else 1.5,
                    alpha=0.8 if 'CPU' in name else 0.9,
                    label=name)
    
    ax1.set_xlabel('ω', fontsize=12)
    ax1.set_ylabel('S(ω)', fontsize=12)
    ax1.set_title('Spectral Function Comparison', fontsize=14)
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(-5, 5)
    
    # Plot 2: Differences from reference (Spectral CPU)
    ax2 = axes[0, 1]
    ref_name = 'Spectral CPU'
    if ref_name in results:
        ref_omega = results[ref_name]['omega']
        ref_S = results[ref_name]['S_real']
        
        for name, data in results.items():
            if name != ref_name and 'omega' in data:
                # Interpolate to same grid if needed
                if len(data['omega']) == len(ref_omega):
                    diff = data['S_real'] - ref_S
                    rel_diff = np.abs(diff) / (np.abs(ref_S) + 1e-15)
                    ax2.semilogy(data['omega'], rel_diff + 1e-16,
                                color=colors.get(name, 'black'),
                                linestyle=linestyles.get(name, '-'),
                                linewidth=1.5,
                                label=f'{name} vs {ref_name}')
    
    ax2.set_xlabel('ω', fontsize=12)
    ax2.set_ylabel('Relative Difference', fontsize=12)
    ax2.set_title(f'Relative Difference from {ref_name}', fontsize=14)
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(-5, 5)
    ax2.set_ylim(1e-16, 1)
    
    # Plot 3: Timing comparison
    ax3 = axes[1, 0]
    method_names = list(timings.keys())
    times = [timings[name] for name in method_names]
    bar_colors = [colors.get(name, 'gray') for name in method_names]
    
    bars = ax3.bar(method_names, times, color=bar_colors, alpha=0.8, edgecolor='black')
    ax3.set_ylabel('Time (seconds)', fontsize=12)
    ax3.set_title('Computation Time Comparison', fontsize=14)
    ax3.tick_params(axis='x', rotation=15)
    
    # Add time labels on bars
    for bar, time in zip(bars, times):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{time:.2f}s', ha='center', va='bottom', fontsize=10)
    
    # Add speedup annotations
    cpu_time = timings['Spectral CPU']
    for i, (name, time) in enumerate(timings.items()):
        if 'GPU' in name:
            speedup = cpu_time / time
            ax3.text(i, time/2, f'{speedup:.1f}×', ha='center', va='center',
                    fontsize=11, fontweight='bold', color='white')
    
    # Plot 4: Summary statistics table
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    # Compute statistics
    ref_name = 'Spectral CPU'
    if ref_name in results:
        ref_S = results[ref_name]['S_real']
        
        table_data = []
        headers = ['Method', 'Time (s)', 'Speedup', 'Max Rel Diff', 'Avg Rel Diff']
        
        for name in method_names:
            time = timings[name]
            speedup = timings['Spectral CPU'] / time
            
            if name in results and len(results[name].get('omega', [])) == len(ref_S):
                diff = results[name]['S_real'] - ref_S
                rel_diff = np.abs(diff) / (np.abs(ref_S) + 1e-15)
                # Filter out zeros for meaningful comparison
                mask = np.abs(ref_S) > 1e-10
                if np.any(mask):
                    max_diff = np.max(rel_diff[mask])
                    avg_diff = np.mean(rel_diff[mask])
                else:
                    max_diff = 0.0
                    avg_diff = 0.0
            else:
                max_diff = 0.0
                avg_diff = 0.0
            
            table_data.append([name, f'{time:.2f}', f'{speedup:.1f}×',
                              f'{max_diff:.2e}', f'{avg_diff:.2e}'])
        
        table = ax4.table(cellText=table_data, colLabels=headers,
                         loc='center', cellLoc='center',
                         colWidths=[0.25, 0.15, 0.15, 0.2, 0.2])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        
        # Color header row
        for j in range(len(headers)):
            table[(0, j)].set_facecolor('#4472C4')
            table[(0, j)].set_text_props(color='white', fontweight='bold')
        
        # Color method cells
        for i, name in enumerate(method_names):
            table[(i+1, 0)].set_facecolor(colors.get(name, 'gray'))
            table[(i+1, 0)].set_text_props(color='white', fontweight='bold')
    
    plt.suptitle('DSSF Method Comparison: Spectral vs Continued Fraction (CPU vs GPU)',
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Save figure
    output_path = os.path.join(base_dir, 'dssf_4method_comparison.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved comparison plot to: {output_path}")
    
    plt.show()

if __name__ == '__main__':
    if len(sys.argv) > 1:
        base_dir = sys.argv[1]
    else:
        base_dir = 'test_kagome_2x3/structure_factor_results'
    
    compare_methods(base_dir)
