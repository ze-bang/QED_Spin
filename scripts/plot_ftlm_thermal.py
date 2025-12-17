#!/usr/bin/env python3
"""
Plot FTLM thermal spectral function results from HDF5 file.
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import argparse
import re
from pathlib import Path

def find_h5_file(path):
    """Find HDF5 file from path (handles directory input)."""
    path = Path(path)
    
    if path.is_file() and path.suffix in ['.h5', '.hdf5']:
        return path
    
    if path.is_dir():
        # Search for .h5 files in the directory and subdirectories
        h5_files = list(path.glob('**/*.h5')) + list(path.glob('**/*.hdf5'))
        if len(h5_files) == 1:
            print(f"Found HDF5 file: {h5_files[0]}")
            return h5_files[0]
        elif len(h5_files) > 1:
            print(f"Multiple HDF5 files found in {path}:")
            for i, f in enumerate(h5_files):
                print(f"  [{i}] {f}")
            # Try to find one with 'ftlm' or 'thermal' in the name
            for f in h5_files:
                if 'ftlm' in f.name.lower() or 'thermal' in f.name.lower():
                    print(f"Auto-selecting: {f}")
                    return f
            # Otherwise use the first one
            print(f"Auto-selecting first file: {h5_files[0]}")
            return h5_files[0]
        else:
            raise FileNotFoundError(f"No HDF5 files found in {path}")
    
    raise FileNotFoundError(f"Path does not exist or is not a valid HDF5 file: {path}")


def load_ftlm_thermal_data(h5_path):
    """Load all FTLM thermal data from HDF5 file."""
    h5_path = find_h5_file(h5_path)
    data = {}
    
    with h5py.File(h5_path, 'r') as f:
        # Load frequencies
        if 'spectral/frequencies' in f:
            data['frequencies'] = f['spectral/frequencies'][:]
        else:
            print("Warning: No frequencies found in HDF5 file")
            return None
        
        # Load momentum points
        if 'momentum_points/q_vectors' in f:
            data['q_vectors'] = f['momentum_points/q_vectors'][:]
        
        # Load metadata
        if 'metadata' in f:
            meta = f['metadata']
            data['metadata'] = {
                'num_sites': meta.attrs.get('num_sites', 'N/A'),
                'method': meta.attrs.get('method', 'N/A'),
                'broadening': meta.attrs.get('broadening', 'N/A'),
            }
        
        # Load spectral data for each operator
        data['spectral'] = {}
        if 'spectral' in f:
            for op_name in f['spectral'].keys():
                if op_name == 'frequencies':
                    continue
                    
                op_group = f['spectral'][op_name]
                data['spectral'][op_name] = {
                    'temperatures': [],
                    'spectra_real': [],
                    'spectra_imag': [],
                    'errors_real': [],
                    'errors_imag': [],
                }
                
                # Collect all temperature groups
                temp_groups = []
                for key in op_group.keys():
                    if key.startswith('T_'):
                        T = float(key[2:])
                        temp_groups.append((T, key))
                
                # Sort by temperature
                temp_groups.sort(key=lambda x: x[0])
                
                for T, key in temp_groups:
                    grp = op_group[key]
                    # Average over samples if multiple exist
                    real_spectra = []
                    imag_spectra = []
                    
                    for sample_key in grp.keys():
                        if sample_key.startswith('sample_'):
                            sample_grp = grp[sample_key]
                            if 'real' in sample_grp:
                                real_spectra.append(sample_grp['real'][:])
                            if 'imag' in sample_grp:
                                imag_spectra.append(sample_grp['imag'][:])
                    
                    if real_spectra:
                        data['spectral'][op_name]['temperatures'].append(T)
                        data['spectral'][op_name]['spectra_real'].append(np.mean(real_spectra, axis=0))
                        data['spectral'][op_name]['spectra_imag'].append(np.mean(imag_spectra, axis=0))
    
    return data


def plot_spectral_vs_omega(data, output_dir=None, show=True):
    """Plot spectral function vs frequency for different temperatures."""
    if data is None:
        print("No data to plot")
        return
    
    frequencies = data['frequencies']
    
    for op_name, op_data in data['spectral'].items():
        temperatures = np.array(op_data['temperatures'])
        spectra_real = np.array(op_data['spectra_real'])
        
        if len(temperatures) == 0:
            continue
        
        # Create figure with multiple temperature curves
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Select a subset of temperatures for clarity
        n_temps = len(temperatures)
        indices = np.linspace(0, n_temps-1, min(8, n_temps), dtype=int)
        
        colors = plt.cm.coolwarm(np.linspace(0, 1, len(indices)))
        
        for i, idx in enumerate(indices):
            T = temperatures[idx]
            spectrum = spectra_real[idx]
            ax1.plot(frequencies, spectrum, color=colors[i], 
                    label=f'T={T:.3f}', linewidth=1.5)
        
        ax1.set_xlabel(r'$\omega$', fontsize=12)
        ax1.set_ylabel(r'$S(\omega, T)$', fontsize=12)
        ax1.set_title(f'FTLM Spectral Function: {op_name}')
        ax1.legend(loc='upper right', fontsize=8)
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim([frequencies.min(), frequencies.max()])
        
        # Heatmap plot (ω vs T)
        T_grid, omega_grid = np.meshgrid(temperatures, frequencies)
        
        # Transpose to get correct orientation
        spectral_2d = spectra_real.T  # Shape: (n_omega, n_T)
        
        # Use positive values for log scale
        spectral_pos = np.maximum(spectral_2d, 1e-10)
        
        im = ax2.pcolormesh(T_grid, omega_grid, spectral_pos, 
                           shading='auto', cmap='hot',
                           norm=LogNorm(vmin=spectral_pos[spectral_pos > 0].min(), 
                                       vmax=spectral_pos.max()))
        ax2.set_xlabel('Temperature', fontsize=12)
        ax2.set_ylabel(r'$\omega$', fontsize=12)
        ax2.set_xscale('log')
        ax2.set_title(f'S(ω,T) Heatmap: {op_name}')
        plt.colorbar(im, ax=ax2, label=r'$S(\omega, T)$')
        
        plt.tight_layout()
        
        if output_dir:
            fname = op_name.replace('/', '_').replace(' ', '_')
            plt.savefig(f'{output_dir}/ftlm_spectral_{fname}.png', dpi=150, bbox_inches='tight')
            print(f"Saved: {output_dir}/ftlm_spectral_{fname}.png")
        
        if show:
            plt.show()
        else:
            plt.close()


def plot_sum_rule_check(data, output_dir=None, show=True):
    """Check sum rule: integral of S(ω) should be related to static susceptibility."""
    if data is None:
        print("No data to plot")
        return
    
    frequencies = data['frequencies']
    domega = frequencies[1] - frequencies[0]
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    for op_name, op_data in data['spectral'].items():
        temperatures = np.array(op_data['temperatures'])
        spectra_real = np.array(op_data['spectra_real'])
        
        if len(temperatures) == 0:
            continue
        
        # Compute integral of S(ω) for each temperature
        integrals = np.trapezoid(spectra_real, frequencies, axis=1)
        
        ax.plot(temperatures, integrals, 'o-', label=op_name, markersize=4)
    
    ax.set_xlabel('Temperature', fontsize=12)
    ax.set_ylabel(r'$\int S(\omega, T) d\omega$', fontsize=12)
    ax.set_title('Sum Rule Check: Integrated Spectral Weight')
    ax.set_xscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_dir:
        plt.savefig(f'{output_dir}/ftlm_sum_rule.png', dpi=150, bbox_inches='tight')
        print(f"Saved: {output_dir}/ftlm_sum_rule.png")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_temperature_evolution(data, omega_indices=None, output_dir=None, show=True):
    """Plot how spectral weight at specific frequencies evolves with temperature."""
    if data is None:
        print("No data to plot")
        return
    
    frequencies = data['frequencies']
    
    # Select some representative frequencies
    if omega_indices is None:
        n_omega = len(frequencies)
        omega_indices = [n_omega//4, n_omega//2, 3*n_omega//4]
    
    fig, axes = plt.subplots(1, len(data['spectral']), figsize=(6*len(data['spectral']), 5))
    if len(data['spectral']) == 1:
        axes = [axes]
    
    for ax, (op_name, op_data) in zip(axes, data['spectral'].items()):
        temperatures = np.array(op_data['temperatures'])
        spectra_real = np.array(op_data['spectra_real'])
        
        if len(temperatures) == 0:
            continue
        
        for idx in omega_indices:
            omega_val = frequencies[idx]
            ax.plot(temperatures, spectra_real[:, idx], 'o-', 
                   label=f'ω={omega_val:.2f}', markersize=4)
        
        ax.set_xlabel('Temperature', fontsize=12)
        ax.set_ylabel(r'$S(\omega, T)$', fontsize=12)
        ax.set_title(f'Temperature Evolution: {op_name}')
        ax.set_xscale('log')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_dir:
        plt.savefig(f'{output_dir}/ftlm_T_evolution.png', dpi=150, bbox_inches='tight')
        print(f"Saved: {output_dir}/ftlm_T_evolution.png")
    
    if show:
        plt.show()
    else:
        plt.close()


def main():
    parser = argparse.ArgumentParser(description='Plot FTLM thermal spectral function results')
    parser.add_argument('h5_file', help='Path to HDF5 file')
    parser.add_argument('--output-dir', '-o', help='Output directory for plots')
    parser.add_argument('--no-show', action='store_true', help='Do not display plots')
    args = parser.parse_args()
    
    # Find the actual HDF5 file path
    h5_path = find_h5_file(args.h5_file)
    
    # Load data
    print(f"Loading data from {h5_path}...")
    data = load_ftlm_thermal_data(args.h5_file)
    
    if data is None:
        print("Failed to load data")
        return
    
    print(f"Loaded {len(data['spectral'])} operators")
    for op_name in data['spectral']:
        n_temps = len(data['spectral'][op_name]['temperatures'])
        print(f"  - {op_name}: {n_temps} temperatures")
    
    # Set output directory - default to same directory as input file
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = str(h5_path.parent)
        print(f"Saving plots to: {output_dir}")
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    show = not args.no_show
    
    # Generate plots
    print("\nGenerating spectral function plots...")
    plot_spectral_vs_omega(data, output_dir, show)
    
    print("Generating sum rule check plot...")
    plot_sum_rule_check(data, output_dir, show)
    
    print("Generating temperature evolution plot...")
    plot_temperature_evolution(data, output_dir=output_dir, show=show)
    
    print("\nDone!")


if __name__ == '__main__':
    main()
