#!/usr/bin/env python3
"""
Plot FTLM (Finite Temperature Lanczos Method) Results

This script automatically generates publication-quality plots from FTLM output files.
Produces separate plots for energy, specific heat, entropy, and free energy vs temperature.

Usage:
    python plot_ftlm.py <ftlm_output_file> [--output <dir>] [--format <png|pdf|svg>]
    python plot_ftlm.py output/thermo/ftlm_thermo.txt
    python plot_ftlm.py output/thermo/ftlm_thermo.txt --output plots/ --format pdf
"""

import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import sys
from pathlib import Path

# Set publication-quality plot defaults
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 11
plt.rcParams['ytick.labelsize'] = 11
plt.rcParams['legend.fontsize'] = 11
plt.rcParams['figure.figsize'] = (8, 6)
plt.rcParams['lines.linewidth'] = 2
plt.rcParams['lines.markersize'] = 6


def load_ftlm_data(filename):
    """Load FTLM results from file."""
    if not os.path.exists(filename):
        raise FileNotFoundError(f"FTLM output file not found: {filename}")
    
    # Read the file, skipping comment lines
    data = np.loadtxt(filename)
    
    if data.shape[1] != 9:
        raise ValueError(f"Expected 9 columns in FTLM output, got {data.shape[1]}")
    
    results = {
        'temperature': data[:, 0],
        'energy': data[:, 1],
        'energy_error': data[:, 2],
        'specific_heat': data[:, 3],
        'specific_heat_error': data[:, 4],
        'entropy': data[:, 5],
        'entropy_error': data[:, 6],
        'free_energy': data[:, 7],
        'free_energy_error': data[:, 8],
    }
    
    # Extract number of samples from header
    with open(filename, 'r') as f:
        first_line = f.readline()
        if 'averaged over' in first_line:
            import re
            match = re.search(r'averaged over (\d+) samples', first_line)
            if match:
                results['num_samples'] = int(match.group(1))
            else:
                results['num_samples'] = None
        else:
            results['num_samples'] = None
    
    return results


def plot_energy(data, ax=None, show_errors=True):
    """Plot energy vs temperature."""
    if ax is None:
        fig, ax = plt.subplots()
    
    T = data['temperature']
    E = data['energy']
    E_err = data['energy_error']
    
    if show_errors and np.any(E_err > 0):
        ax.errorbar(T, E, yerr=E_err, fmt='o-', capsize=3, 
                   label='FTLM', color='C0', markersize=4)
    else:
        ax.plot(T, E, 'o-', label='FTLM', color='C0', markersize=4)
    
    ax.set_xlabel('Temperature (T)')
    ax.set_ylabel('Energy ⟨E⟩')
    ax.set_title('Energy vs Temperature')
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3, which='both')
    ax.legend()
    
    return ax


def plot_specific_heat(data, ax=None, show_errors=True):
    """Plot specific heat vs temperature."""
    if ax is None:
        fig, ax = plt.subplots()
    
    T = data['temperature']
    C = data['specific_heat']
    C_err = data['specific_heat_error']
    
    if show_errors and np.any(C_err > 0):
        ax.errorbar(T, C, yerr=C_err, fmt='o-', capsize=3,
                   label='FTLM', color='C1', markersize=4)
    else:
        ax.plot(T, C, 'o-', label='FTLM', color='C1', markersize=4)
    
    ax.set_xlabel('Temperature (T)')
    ax.set_ylabel('Specific Heat (C)')
    ax.set_title('Specific Heat vs Temperature')
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3, which='both')
    ax.legend()
    
    # Specific heat often has a peak
    ax.set_ylim(bottom=0)
    
    return ax


def plot_entropy(data, ax=None, show_errors=True):
    """Plot entropy vs temperature."""
    if ax is None:
        fig, ax = plt.subplots()
    
    T = data['temperature']
    S = data['entropy']
    S_err = data['entropy_error']
    
    if show_errors and np.any(S_err > 0):
        ax.errorbar(T, S, yerr=S_err, fmt='o-', capsize=3,
                   label='FTLM', color='C2', markersize=4)
    else:
        ax.plot(T, S, 'o-', label='FTLM', color='C2', markersize=4)
    
    ax.set_xlabel('Temperature (T)')
    ax.set_ylabel('Entropy (S)')
    ax.set_title('Entropy vs Temperature')
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3, which='both')
    ax.legend()
    
    return ax


def plot_free_energy(data, ax=None, show_errors=True):
    """Plot free energy vs temperature."""
    if ax is None:
        fig, ax = plt.subplots()
    
    T = data['temperature']
    F = data['free_energy']
    F_err = data['free_energy_error']
    
    if show_errors and np.any(F_err > 0):
        ax.errorbar(T, F, yerr=F_err, fmt='o-', capsize=3,
                   label='FTLM', color='C3', markersize=4)
    else:
        ax.plot(T, F, 'o-', label='FTLM', color='C3', markersize=4)
    
    ax.set_xlabel('Temperature (T)')
    ax.set_ylabel('Free Energy (F)')
    ax.set_title('Free Energy vs Temperature')
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3, which='both')
    ax.legend()
    
    return ax


def create_summary_plot(data, show_errors=True):
    """Create a 2x2 grid of all thermodynamic quantities."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('FTLM Thermodynamic Properties', fontsize=16, fontweight='bold')
    
    plot_energy(data, ax=axes[0, 0], show_errors=show_errors)
    plot_specific_heat(data, ax=axes[0, 1], show_errors=show_errors)
    plot_entropy(data, ax=axes[1, 0], show_errors=show_errors)
    plot_free_energy(data, ax=axes[1, 1], show_errors=show_errors)
    
    # Add info text
    info_text = f"Data from: FTLM"
    if data.get('num_samples'):
        info_text += f" ({data['num_samples']} samples)"
    fig.text(0.99, 0.01, info_text, ha='right', va='bottom', 
             fontsize=9, style='italic', alpha=0.7)
    
    plt.tight_layout()
    return fig


def main():
    parser = argparse.ArgumentParser(
        description='Plot FTLM thermodynamic results',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage - saves to same directory as input file
  python plot_ftlm.py output/thermo/ftlm_thermo.txt
  
  # Specify output directory and format
  python plot_ftlm.py output/thermo/ftlm_thermo.txt --output plots/ --format pdf
  
  # Generate high-DPI PNG for presentations
  python plot_ftlm.py output/thermo/ftlm_thermo.txt --dpi 300
  
  # Disable error bars
  python plot_ftlm.py output/thermo/ftlm_thermo.txt --no-errors
  
  # Individual plots instead of summary
  python plot_ftlm.py output/thermo/ftlm_thermo.txt --individual
        """
    )
    
    parser.add_argument('input', help='FTLM output file (ftlm_thermo.txt)')
    parser.add_argument('--output', '-o', default=None,
                       help='Output directory for plots (default: same as input file)')
    parser.add_argument('--format', '-f', default='png', choices=['png', 'pdf', 'svg', 'jpg'],
                       help='Output format (default: png)')
    parser.add_argument('--dpi', type=int, default=150,
                       help='DPI for raster formats (default: 150)')
    parser.add_argument('--no-errors', action='store_true',
                       help='Disable error bars')
    parser.add_argument('--individual', action='store_true',
                       help='Create individual plots instead of summary')
    parser.add_argument('--prefix', default='ftlm',
                       help='Prefix for output filenames (default: ftlm)')
    
    args = parser.parse_args()
    
    # Load data
    print(f"Loading FTLM data from: {args.input}")
    try:
        data = load_ftlm_data(args.input)
    except Exception as e:
        print(f"Error loading data: {e}", file=sys.stderr)
        return 1
    
    print(f"  Temperature range: [{data['temperature'].min():.3g}, {data['temperature'].max():.3g}]")
    print(f"  Number of points: {len(data['temperature'])}")
    if data.get('num_samples'):
        print(f"  Number of samples: {data['num_samples']}")
    
    # Determine output directory
    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = Path(args.input).parent
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    show_errors = not args.no_errors
    
    # Create plots
    if args.individual:
        # Individual plots
        plots = [
            ('energy', plot_energy),
            ('specific_heat', plot_specific_heat),
            ('entropy', plot_entropy),
            ('free_energy', plot_free_energy)
        ]
        
        for name, plot_func in plots:
            fig, ax = plt.subplots(figsize=(8, 6))
            plot_func(data, ax=ax, show_errors=show_errors)
            
            filename = output_dir / f"{args.prefix}_{name}.{args.format}"
            fig.savefig(filename, dpi=args.dpi, bbox_inches='tight')
            print(f"  Saved: {filename}")
            plt.close(fig)
    
    else:
        # Summary plot
        fig = create_summary_plot(data, show_errors=show_errors)
        filename = output_dir / f"{args.prefix}_summary.{args.format}"
        fig.savefig(filename, dpi=args.dpi, bbox_inches='tight')
        print(f"  Saved: {filename}")
        plt.close(fig)
    
    print("\nPlotting complete!")
    return 0


if __name__ == '__main__':
    sys.exit(main())
