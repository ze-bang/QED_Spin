#!/usr/bin/env python3
"""
Plot BFG order parameter scan results from compute_bfg_order_parameters --scan-dir

Usage:
    python plot_bfg_scan_results.py <scan_results.h5> [output_dir]
    python plot_bfg_scan_results.py --scan-dir <dir>  # auto-find scan_results.h5

Reads the scan_results.h5 file and produces plots of all order parameters vs Jpm:
  - Translation order (m_translation)
  - Nematic order (XY, S+S-, SzSz, Heisenberg)
  - VBS order (XY dimer, Heisenberg dimer)
"""

import numpy as np
import matplotlib.pyplot as plt
import h5py
import argparse
import os
from pathlib import Path


def load_scan_results(h5_file):
    """Load all datasets from scan_results.h5"""
    data = {}
    with h5py.File(h5_file, 'r') as f:
        for key in f.keys():
            data[key] = f[key][:]
    return data


def plot_all_order_parameters(data, output_dir, title_prefix=""):
    """Create comprehensive plots of all order parameters"""
    
    jpm = data.get('jpm_values', [])
    if len(jpm) == 0:
        print("No data found!")
        return
    
    # Sort by Jpm
    sort_idx = np.argsort(jpm)
    jpm = jpm[sort_idx]
    
    # Apply sort to all arrays
    sorted_data = {}
    for key, val in data.items():
        if len(val) == len(jpm):
            sorted_data[key] = val[sort_idx]
        else:
            sorted_data[key] = val
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Color scheme
    colors = {
        'translation': '#1f77b4',
        'xy': '#2ca02c',
        'spsm': '#9467bd',
        'szsz': '#d62728',
        'heisenberg': '#ff7f0e',
        'vbs_xy': '#17becf',
        'vbs_heis': '#e377c2',
    }
    
    # =========================================================================
    # Figure 1: Overview - All order parameters
    # =========================================================================
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'{title_prefix}BFG Order Parameters vs $J_{{\\pm}}$', fontsize=14)
    
    # Panel 1: Translation order
    ax = axes[0, 0]
    if 'm_translation' in sorted_data:
        ax.plot(jpm, sorted_data['m_translation'], 'o-', color=colors['translation'], 
                label='$m_{trans}$', markersize=4)
    ax.set_xlabel('$J_{\\pm}$')
    ax.set_ylabel('$m_{translation}$')
    ax.set_title('Translation Order (Spin Structure Factor)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Panel 2: All nematic orders
    ax = axes[0, 1]
    if 'm_nematic' in sorted_data:
        ax.plot(jpm, sorted_data['m_nematic'], 'o-', color=colors['xy'], 
                label='XY', markersize=4)
    if 'm_nematic_spsm' in sorted_data:
        ax.plot(jpm, sorted_data['m_nematic_spsm'], 's-', color=colors['spsm'], 
                label='$S^+S^-$', markersize=4)
    if 'm_nematic_szsz' in sorted_data:
        ax.plot(jpm, sorted_data['m_nematic_szsz'], '^-', color=colors['szsz'], 
                label='$S^zS^z$', markersize=4)
    if 'm_nematic_heisenberg' in sorted_data:
        ax.plot(jpm, sorted_data['m_nematic_heisenberg'], 'd-', color=colors['heisenberg'], 
                label='Heisenberg', markersize=4)
    ax.set_xlabel('$J_{\\pm}$')
    ax.set_ylabel('$m_{nematic}$')
    ax.set_title('Nematic Order (C3 Breaking)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Panel 3: VBS orders
    ax = axes[1, 0]
    if 'm_vbs_xy' in sorted_data:
        ax.plot(jpm, sorted_data['m_vbs_xy'], 'o-', color=colors['vbs_xy'], 
                label='XY dimer', markersize=4)
    if 'm_vbs_heis' in sorted_data:
        ax.plot(jpm, sorted_data['m_vbs_heis'], 's-', color=colors['vbs_heis'], 
                label='Heisenberg dimer', markersize=4)
    elif 'm_vbs' in sorted_data:
        ax.plot(jpm, sorted_data['m_vbs'], 'o-', color=colors['vbs_xy'], 
                label='VBS', markersize=4)
    ax.set_xlabel('$J_{\\pm}$')
    ax.set_ylabel('$m_{VBS}$')
    ax.set_title('VBS Order (4-site Dimer Correlations)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Panel 4: Dimer mean values
    ax = axes[1, 1]
    if 'D_mean_xy' in sorted_data:
        ax.plot(jpm, sorted_data['D_mean_xy'], 'o-', color=colors['vbs_xy'], 
                label='$\\langle D_{XY} \\rangle$', markersize=4)
    if 'D_mean_heis' in sorted_data:
        ax.plot(jpm, sorted_data['D_mean_heis'], 's-', color=colors['vbs_heis'], 
                label='$\\langle D_{Heis} \\rangle$', markersize=4)
    elif 'D_mean' in sorted_data:
        ax.plot(jpm, sorted_data['D_mean'], 'o-', color=colors['vbs_xy'], 
                label='$\\langle D \\rangle$', markersize=4)
    ax.set_xlabel('$J_{\\pm}$')
    ax.set_ylabel('Mean Dimer Value')
    ax.set_title('Average Bond Expectations')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/order_parameters_overview.png', dpi=150, bbox_inches='tight')
    plt.savefig(f'{output_dir}/order_parameters_overview.pdf', bbox_inches='tight')
    print(f"Saved: {output_dir}/order_parameters_overview.png")
    plt.close()
    
    # =========================================================================
    # Figure 2: Nematic order comparison
    # =========================================================================
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f'{title_prefix}Nematic Order Comparison', fontsize=14)
    
    # Left: absolute values
    ax = axes[0]
    labels_nem = []
    for key, label, color, marker in [
        ('m_nematic', 'XY', colors['xy'], 'o'),
        ('m_nematic_spsm', '$S^+S^-$', colors['spsm'], 's'),
        ('m_nematic_szsz', '$S^zS^z$', colors['szsz'], '^'),
        ('m_nematic_heisenberg', 'Heisenberg', colors['heisenberg'], 'd'),
    ]:
        if key in sorted_data:
            ax.plot(jpm, sorted_data[key], f'{marker}-', color=color, 
                    label=label, markersize=5)
            labels_nem.append(label)
    ax.set_xlabel('$J_{\\pm}$')
    ax.set_ylabel('$m_{nematic}$')
    ax.set_title('Absolute Nematic Order')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Right: normalized to max
    ax = axes[1]
    for key, label, color, marker in [
        ('m_nematic', 'XY', colors['xy'], 'o'),
        ('m_nematic_spsm', '$S^+S^-$', colors['spsm'], 's'),
        ('m_nematic_szsz', '$S^zS^z$', colors['szsz'], '^'),
        ('m_nematic_heisenberg', 'Heisenberg', colors['heisenberg'], 'd'),
    ]:
        if key in sorted_data:
            vals = sorted_data[key]
            max_val = np.max(np.abs(vals))
            if max_val > 1e-10:
                ax.plot(jpm, vals / max_val, f'{marker}-', color=color, 
                        label=label, markersize=5)
    ax.set_xlabel('$J_{\\pm}$')
    ax.set_ylabel('Normalized $m_{nematic}$')
    ax.set_title('Normalized Nematic Order')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/nematic_comparison.png', dpi=150, bbox_inches='tight')
    print(f"Saved: {output_dir}/nematic_comparison.png")
    plt.close()
    
    # =========================================================================
    # Figure 3: VBS order comparison
    # =========================================================================
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f'{title_prefix}VBS Order (4-site Dimer Correlations)', fontsize=14)
    
    # Left: VBS order parameter
    ax = axes[0]
    if 'm_vbs_xy' in sorted_data:
        ax.plot(jpm, sorted_data['m_vbs_xy'], 'o-', color=colors['vbs_xy'], 
                label='XY dimer $\\langle D_{XY} D_{XY} \\rangle$', markersize=5)
    if 'm_vbs_heis' in sorted_data:
        ax.plot(jpm, sorted_data['m_vbs_heis'], 's-', color=colors['vbs_heis'], 
                label='Heisenberg dimer $\\langle (S \\cdot S)(S \\cdot S) \\rangle$', markersize=5)
    ax.set_xlabel('$J_{\\pm}$')
    ax.set_ylabel('$m_{VBS}$')
    ax.set_title('VBS Order Parameter')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Right: VBS vs Translation
    ax = axes[1]
    if 'm_translation' in sorted_data:
        ax.plot(jpm, sorted_data['m_translation'], 'o-', color=colors['translation'], 
                label='$m_{trans}$', markersize=5)
    if 'm_vbs_xy' in sorted_data:
        ax.plot(jpm, sorted_data['m_vbs_xy'], 's-', color=colors['vbs_xy'], 
                label='$m_{VBS}$ (XY)', markersize=5)
    if 'm_vbs_heis' in sorted_data:
        ax.plot(jpm, sorted_data['m_vbs_heis'], '^-', color=colors['vbs_heis'], 
                label='$m_{VBS}$ (Heis)', markersize=5)
    ax.set_xlabel('$J_{\\pm}$')
    ax.set_ylabel('Order Parameter')
    ax.set_title('VBS vs Translation Order')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/vbs_comparison.png', dpi=150, bbox_inches='tight')
    print(f"Saved: {output_dir}/vbs_comparison.png")
    plt.close()
    
    # =========================================================================
    # Figure 4: Combined phase diagram style
    # =========================================================================
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Normalize each order parameter to [0, 1] for comparison
    def normalize(arr):
        if np.max(np.abs(arr)) > 1e-10:
            return arr / np.max(np.abs(arr))
        return arr
    
    if 'm_translation' in sorted_data:
        ax.fill_between(jpm, 0, normalize(sorted_data['m_translation']), 
                        alpha=0.3, color=colors['translation'], label='Translation')
        ax.plot(jpm, normalize(sorted_data['m_translation']), '-', 
                color=colors['translation'], linewidth=2)
    
    if 'm_nematic' in sorted_data:
        ax.plot(jpm, normalize(sorted_data['m_nematic']), '--', 
                color=colors['xy'], linewidth=2, label='Nematic (XY)')
    
    if 'm_vbs_xy' in sorted_data:
        ax.plot(jpm, normalize(sorted_data['m_vbs_xy']), '-.', 
                color=colors['vbs_xy'], linewidth=2, label='VBS (XY)')
    
    if 'm_vbs_heis' in sorted_data:
        ax.plot(jpm, normalize(sorted_data['m_vbs_heis']), ':', 
                color=colors['vbs_heis'], linewidth=2, label='VBS (Heis)')
    
    ax.set_xlabel('$J_{\\pm}$', fontsize=12)
    ax.set_ylabel('Normalized Order Parameter', fontsize=12)
    ax.set_title(f'{title_prefix}Phase Diagram Overview', fontsize=14)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.1)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/phase_diagram.png', dpi=150, bbox_inches='tight')
    plt.savefig(f'{output_dir}/phase_diagram.pdf', bbox_inches='tight')
    print(f"Saved: {output_dir}/phase_diagram.png")
    plt.close()
    
    # =========================================================================
    # Figure 5: Individual panels (publication quality)
    # =========================================================================
    for key, ylabel, title, color in [
        ('m_translation', '$m_{translation}$', 'Translation Order', colors['translation']),
        ('m_nematic', '$m_{nematic}$ (XY)', 'Nematic Order (XY)', colors['xy']),
        ('m_nematic_heisenberg', '$m_{nematic}$ (Heis)', 'Nematic Order (Heisenberg)', colors['heisenberg']),
        ('m_vbs_xy', '$m_{VBS}$ (XY)', 'VBS Order (XY Dimer)', colors['vbs_xy']),
        ('m_vbs_heis', '$m_{VBS}$ (Heis)', 'VBS Order (Heisenberg Dimer)', colors['vbs_heis']),
    ]:
        if key in sorted_data:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot(jpm, sorted_data[key], 'o-', color=color, markersize=5, linewidth=1.5)
            ax.set_xlabel('$J_{\\pm}$', fontsize=12)
            ax.set_ylabel(ylabel, fontsize=12)
            ax.set_title(f'{title_prefix}{title}', fontsize=12)
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            fname = key.replace('m_', '')
            plt.savefig(f'{output_dir}/{fname}.png', dpi=150, bbox_inches='tight')
            plt.close()
    
    print(f"\nAll plots saved to: {output_dir}/")


def plot_temperature_dependence(data, output_dir, title_prefix=""):
    """Plot order parameters vs temperature (for TPQ mode)"""
    
    if 'temperature' not in data or np.all(data['temperature'] == 0):
        return
    
    temp = data['temperature']
    if np.max(temp) < 1e-10:
        return
    
    jpm = data.get('jpm_values', [])
    
    # Check if we have multiple temperatures per Jpm (unlikely in scan mode)
    # For scan mode, each Jpm has one temperature
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f'{title_prefix}Order Parameters (TPQ)', fontsize=14)
    
    # Use color by Jpm
    unique_jpm = np.unique(jpm)
    cmap = plt.cm.viridis
    norm = plt.Normalize(vmin=np.min(unique_jpm), vmax=np.max(unique_jpm))
    
    ax = axes[0, 0]
    sc = ax.scatter(temp, data.get('m_translation', np.zeros_like(temp)), 
                    c=jpm, cmap=cmap, norm=norm, s=30)
    ax.set_xlabel('Temperature')
    ax.set_ylabel('$m_{translation}$')
    ax.set_title('Translation Order')
    plt.colorbar(sc, ax=ax, label='$J_{\\pm}$')
    
    ax = axes[0, 1]
    sc = ax.scatter(temp, data.get('m_nematic', np.zeros_like(temp)), 
                    c=jpm, cmap=cmap, norm=norm, s=30)
    ax.set_xlabel('Temperature')
    ax.set_ylabel('$m_{nematic}$')
    ax.set_title('Nematic Order')
    plt.colorbar(sc, ax=ax, label='$J_{\\pm}$')
    
    ax = axes[1, 0]
    if 'm_vbs_xy' in data:
        sc = ax.scatter(temp, data['m_vbs_xy'], c=jpm, cmap=cmap, norm=norm, 
                        s=30, marker='o', label='XY')
    if 'm_vbs_heis' in data:
        sc = ax.scatter(temp, data['m_vbs_heis'], c=jpm, cmap=cmap, norm=norm, 
                        s=30, marker='s', label='Heis')
    ax.set_xlabel('Temperature')
    ax.set_ylabel('$m_{VBS}$')
    ax.set_title('VBS Order')
    ax.legend()
    plt.colorbar(sc, ax=ax, label='$J_{\\pm}$')
    
    ax = axes[1, 1]
    ax.scatter(jpm, temp, c=data.get('m_translation', np.zeros_like(temp)), 
               cmap='coolwarm', s=50)
    ax.set_xlabel('$J_{\\pm}$')
    ax.set_ylabel('Temperature')
    ax.set_title('$J_{\\pm}$ - T Phase Space')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/temperature_dependence.png', dpi=150, bbox_inches='tight')
    print(f"Saved: {output_dir}/temperature_dependence.png")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Plot BFG order parameter scan results',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('input', nargs='?', help='Path to scan_results.h5')
    parser.add_argument('output_dir', nargs='?', default=None,
                        help='Output directory for plots (default: same as input)')
    parser.add_argument('--scan-dir', '-d', type=str, default=None,
                        help='Scan directory (will look for order_parameter_results/scan_results.h5)')
    parser.add_argument('--title', '-t', type=str, default='',
                        help='Title prefix for plots')
    
    args = parser.parse_args()
    
    # Find input file
    if args.scan_dir:
        h5_file = Path(args.scan_dir) / 'order_parameter_results' / 'scan_results.h5'
        if not h5_file.exists():
            h5_file = Path(args.scan_dir) / 'scan_results.h5'
    elif args.input:
        h5_file = Path(args.input)
    else:
        # Try current directory
        h5_file = Path('scan_results.h5')
        if not h5_file.exists():
            h5_file = Path('order_parameter_results/scan_results.h5')
    
    if not h5_file.exists():
        print(f"Error: Cannot find scan_results.h5")
        print(f"  Tried: {h5_file}")
        print("\nUsage:")
        print("  python plot_bfg_scan_results.py <scan_results.h5>")
        print("  python plot_bfg_scan_results.py --scan-dir <dir>")
        return 1
    
    # Determine output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = h5_file.parent / 'plots'
    
    print(f"Loading: {h5_file}")
    data = load_scan_results(h5_file)
    
    print(f"Found {len(data.get('jpm_values', []))} Jpm points")
    print(f"Datasets: {list(data.keys())}")
    
    title_prefix = args.title + ' ' if args.title else ''
    
    plot_all_order_parameters(data, output_dir, title_prefix)
    plot_temperature_dependence(data, output_dir, title_prefix)
    
    return 0


if __name__ == '__main__':
    exit(main())
