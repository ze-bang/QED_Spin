#!/usr/bin/env python3
"""
Plot NSF+SF sums for structure factor files across different K values.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import re
from pathlib import Path
import glob

def extract_K_value(dirname):
    """Extract K value from directory name like 'K=0.5'"""
    match = re.search(r'K=([0-9.]+)', dirname)
    if match:
        return float(match.group(1))
    return None

def load_spectral_data(filepath):
    """Load spectral data from file, skipping header lines"""
    data = np.loadtxt(filepath, comments='#')
    return data

def get_unique_file_bases(directory):
    """Get unique file base names (without NSF/SF suffix)"""
    files = os.listdir(directory)
    bases = set()
    for f in files:
        # Replace _NSF_ and _SF_ with a placeholder to get base name
        base = f.replace('_NSF_', '_PLACEHOLDER_').replace('_SF_', '_PLACEHOLDER_')
        if '_PLACEHOLDER_' in base:
            bases.add(base)
    return sorted(bases)

def plot_nsf_sf_sum_for_k_dirs(base_dir, output_dir='plots_nsf_sf'):
    """
    For each unique file pattern, sum NSF and SF files and plot across K values.
    """
    base_path = Path(base_dir)
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Get all K directories
    k_dirs = sorted([d for d in base_path.iterdir() if d.is_dir() and d.name.startswith('K=')])
    
    if not k_dirs:
        print(f"No K= directories found in {base_dir}")
        return
    
    # Extract K values
    k_values = []
    k_dir_dict = {}
    for k_dir in k_dirs:
        k_val = extract_K_value(k_dir.name)
        if k_val is not None:
            k_values.append(k_val)
            k_dir_dict[k_val] = k_dir
    
    k_values = sorted(k_values)
    print(f"Found K values: {k_values}")
    
    # Get file structure from first K directory
    first_k = k_values[0]
    sample_dir = k_dir_dict[first_k] / 'structure_factor_results' / 'beta_inf' / 'transverse'
    
    if not sample_dir.exists():
        print(f"Directory structure not found: {sample_dir}")
        return
    
    # Get unique file bases
    file_bases = get_unique_file_bases(sample_dir)
    print(f"Found {len(file_bases)} unique file patterns")
    
    # Process each file pattern
    for file_base in file_bases:
        # Reconstruct filenames
        nsf_name = file_base.replace('_PLACEHOLDER_', '_NSF_')
        sf_name = file_base.replace('_PLACEHOLDER_', '_SF_')
        
        print(f"\nProcessing: {nsf_name} + {sf_name}")
        
        # Storage for data across K values
        frequency = None
        nsf_sf_sums = {}
        
        for k_val in k_values:
            k_dir = k_dir_dict[k_val]
            data_dir = k_dir / 'structure_factor_results' / 'beta_inf' / 'transverse'
            
            nsf_path = data_dir / nsf_name
            sf_path = data_dir / sf_name
            
            if not nsf_path.exists() or not sf_path.exists():
                print(f"  Missing files for K={k_val}, skipping...")
                continue
            
            # Load data
            nsf_data = load_spectral_data(nsf_path)
            sf_data = load_spectral_data(sf_path)
            
            if frequency is None:
                frequency = nsf_data[:, 0]
            
            # Sum NSF + SF (Re[S(ω)] is column 1)
            nsf_sf_sum = nsf_data[:, 1] + sf_data[:, 1]
            nsf_sf_sums[k_val] = nsf_sf_sum
        
        if not nsf_sf_sums:
            print(f"  No valid data found for this pattern, skipping...")
            continue
        
        # Create plots
        # 1. Individual plots for each K showing NSF+SF sum
        fig1, ax1 = plt.subplots(figsize=(10, 6))
        
        # Use gradient colormap
        k_sorted = sorted(nsf_sf_sums.keys())
        n_colors = len(k_sorted)
        cmap = plt.cm.viridis
        colors = [cmap(i / (n_colors - 1)) for i in range(n_colors)]
        
        for idx, k_val in enumerate(k_sorted):
            ax1.plot(frequency, nsf_sf_sums[k_val], label=f'K={k_val:.1f}', 
                    alpha=0.8, linewidth=2, color=colors[idx])
        
        ax1.set_xlabel('Frequency (ω)', fontsize=12)
        ax1.set_ylabel('Re[S(ω)] (NSF+SF)', fontsize=12)
        ax1.set_title(f'NSF+SF Sum vs Frequency\n{nsf_name.split("_spectral")[0]}', fontsize=14)
        ax1.legend(loc='best', fontsize=9, ncol=2)
        ax1.grid(True, alpha=0.3)
        
        # Clean filename for saving
        clean_base = file_base.replace('_PLACEHOLDER_', '').replace('_spectral_sample_0_beta_inf.txt', '')
        output_file1 = output_path / f'{clean_base}_nsf_sf_sum_vs_frequency.png'
        plt.tight_layout()
        plt.savefig(output_file1, dpi=150)
        print(f"  Saved: {output_file1}")
        plt.close()
        
        # 2. Comparison plot: Peak intensity vs K
        peak_intensities = []
        for k_val in sorted(nsf_sf_sums.keys()):
            peak_intensities.append(np.max(nsf_sf_sums[k_val]))
        
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        k_vals = sorted(nsf_sf_sums.keys())
        
        # Create gradient scatter plot
        scatter = ax2.scatter(k_vals, peak_intensities, c=k_vals, cmap='viridis', 
                             s=100, edgecolors='black', linewidth=1.5, zorder=3)
        ax2.plot(k_vals, peak_intensities, '-', linewidth=2, alpha=0.5, color='gray', zorder=2)
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax2, label='K parameter')
        
        ax2.set_xlabel('K parameter', fontsize=12)
        ax2.set_ylabel('Peak Re[S(ω)] (NSF+SF)', fontsize=12)
        ax2.set_title(f'Peak Intensity vs K\n{nsf_name.split("_spectral")[0]}', fontsize=14)
        ax2.grid(True, alpha=0.3)
        
        output_file2 = output_path / f'{clean_base}_peak_vs_K.png'
        plt.tight_layout()
        plt.savefig(output_file2, dpi=150)
        print(f"  Saved: {output_file2}")
        plt.close()
        
        # 3. Integrated intensity vs K
        integrated_intensities = []
        for k_val in sorted(nsf_sf_sums.keys()):
            # Integrate using trapezoidal rule
            integrated = np.trapz(nsf_sf_sums[k_val], frequency)
            integrated_intensities.append(integrated)
        
        fig3, ax3 = plt.subplots(figsize=(10, 6))
        k_vals = sorted(nsf_sf_sums.keys())
        
        # Create gradient scatter plot with different colormap
        scatter = ax3.scatter(k_vals, integrated_intensities, c=k_vals, cmap='plasma', 
                             s=120, marker='s', edgecolors='black', linewidth=1.5, zorder=3)
        ax3.plot(k_vals, integrated_intensities, '-', linewidth=2, alpha=0.5, color='gray', zorder=2)
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax3, label='K parameter')
        
        ax3.set_xlabel('K parameter', fontsize=12)
        ax3.set_ylabel('Integrated Re[S(ω)] (NSF+SF)', fontsize=12)
        ax3.set_title(f'Integrated Intensity vs K\n{nsf_name.split("_spectral")[0]}', fontsize=14)
        ax3.grid(True, alpha=0.3)
        
        output_file3 = output_path / f'{clean_base}_integrated_vs_K.png'
        plt.tight_layout()
        plt.savefig(output_file3, dpi=150)
        print(f"  Saved: {output_file3}")
        plt.close()
    
    print(f"\n✓ All plots saved to {output_path}/")

if __name__ == '__main__':
    base_dir = '/home/pc_linux/exact_diagonalization_clean/exact_diagonalization_cpp/non_kramer_three_spins'
    output_dir = '/home/pc_linux/exact_diagonalization_clean/exact_diagonalization_cpp/non_kramer_three_spins/comparison_plots'
    
    plot_nsf_sf_sum_for_k_dirs(base_dir, output_dir)
