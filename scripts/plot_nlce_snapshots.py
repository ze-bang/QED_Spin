#!/usr/bin/env python3
"""Plot NLCE fit snapshots with experimental data."""

import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import re
import argparse

def parse_snapshot(filepath):
    """Parse a snapshot file and extract data."""
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    # Extract header info
    iteration = None
    chi_squared = None
    params = {}
    
    calc_temps = []
    calc_C = []
    exp_temps = []
    exp_C = []
    interp_C = []
    
    section = None
    for line in lines:
        line = line.strip()
        if line.startswith('# Iteration'):
            iteration = int(line.split()[-1])
        elif line.startswith('# Chi-squared:'):
            chi_squared = float(line.split()[-1])
        elif 'Jzz=' in line and 'Jpm=' in line:
            # Parse parameters
            parts = line.replace('#', '').strip().split(',')
            for part in parts:
                key, val = part.strip().split('=')
                params[key.strip()] = float(val.strip())
        elif line.startswith('# Calculated specific heat'):
            section = 'calc'
        elif line.startswith('# Experimental vs Interpolated'):
            section = 'exp'
        elif line.startswith('# T(K)') or line.startswith('# T_exp'):
            continue
        elif line.startswith('#'):
            continue
        elif line and section == 'calc':
            parts = line.split()
            if len(parts) >= 2:
                calc_temps.append(float(parts[0]))
                calc_C.append(float(parts[1]))
        elif line and section == 'exp':
            parts = line.split()
            if len(parts) >= 3:
                exp_temps.append(float(parts[0]))
                exp_C.append(float(parts[1]))
                interp_C.append(float(parts[2]))
    
    return {
        'iteration': iteration,
        'chi_squared': chi_squared,
        'params': params,
        'calc_temps': np.array(calc_temps),
        'calc_C': np.array(calc_C),
        'exp_temps': np.array(exp_temps),
        'exp_C': np.array(exp_C),
        'interp_C': np.array(interp_C)
    }

def main():
    parser = argparse.ArgumentParser(description='Plot NLCE fit snapshots')
    parser.add_argument('--snapshot_dir', type=str, required=True,
                       help='Directory containing snapshot files')
    parser.add_argument('--exp_data', type=str, default=None,
                       help='Experimental data file (optional, for full range)')
    parser.add_argument('--output', type=str, default='nlce_fit_snapshots.png',
                       help='Output plot file')
    parser.add_argument('--iterations', type=str, default='last',
                       help='Which iterations to plot: "all", "last", or comma-separated list')
    args = parser.parse_args()
    
    # Find snapshot files
    snapshot_files = sorted(glob.glob(os.path.join(args.snapshot_dir, 'snapshot_iter_*.txt')))
    if not snapshot_files:
        print(f"No snapshot files found in {args.snapshot_dir}")
        return
    
    print(f"Found {len(snapshot_files)} snapshot files")
    
    # Select which iterations to plot
    if args.iterations == 'last':
        snapshot_files = [snapshot_files[-1]]
    elif args.iterations == 'all':
        pass
    else:
        indices = [int(x) for x in args.iterations.split(',')]
        snapshot_files = [f for f in snapshot_files if any(f'iter_{i:04d}' in f for i in indices)]
    
    # Load experimental data if provided
    exp_full = None
    if args.exp_data and os.path.exists(args.exp_data):
        exp_full = np.loadtxt(args.exp_data)
        print(f"Loaded experimental data: {len(exp_full)} points")
    
    # Parse snapshots
    snapshots = [parse_snapshot(f) for f in snapshot_files]
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Color map for multiple iterations
    colors = plt.cm.viridis(np.linspace(0, 1, len(snapshots)))
    
    # Plot 1: Full range (linear scale)
    ax1 = axes[0]
    
    # Plot experimental data
    if exp_full is not None:
        ax1.scatter(exp_full[:, 0], exp_full[:, 1], c='black', s=20, 
                   label='Experiment', zorder=10, alpha=0.7)
    
    # Plot calculated curves for each snapshot
    for i, snap in enumerate(snapshots):
        label = f"Iter {snap['iteration']}: χ²={snap['chi_squared']:.2f}"
        if len(snapshots) == 1:
            params = snap['params']
            label = (f"NLCE (χ²={snap['chi_squared']:.2f})\n"
                    f"Jzz={params.get('Jzz', 0):.4f}, Jpm={params.get('Jpm', 0):.4f}\n"
                    f"Jpmpm={params.get('Jpmpm', 0):.4f}, Jzpm={params.get('Jzpm', 0):.4f}")
        
        # Filter out negative C values for plotting
        mask = snap['calc_C'] > 0
        ax1.plot(snap['calc_temps'][mask], snap['calc_C'][mask], 
                color=colors[i], linewidth=2, label=label)
    
    ax1.set_xlabel('Temperature (K)', fontsize=12)
    ax1.set_ylabel('Specific Heat (J/mol/K)', fontsize=12)
    ax1.set_title('NLCE Fit vs Experiment (Linear Scale)', fontsize=14)
    ax1.legend(loc='upper right', fontsize=9)
    ax1.set_xlim(0, None)
    ax1.set_ylim(0, None)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Log-log scale to see full range
    ax2 = axes[1]
    
    if exp_full is not None:
        ax2.scatter(exp_full[:, 0], exp_full[:, 1], c='black', s=20, 
                   label='Experiment', zorder=10, alpha=0.7)
    
    for i, snap in enumerate(snapshots):
        # Filter out non-positive values
        mask = snap['calc_C'] > 0
        if np.any(mask):
            ax2.plot(snap['calc_temps'][mask], snap['calc_C'][mask], 
                    color=colors[i], linewidth=2)
    
    ax2.set_xlabel('Temperature (K)', fontsize=12)
    ax2.set_ylabel('Specific Heat (J/mol/K)', fontsize=12)
    ax2.set_title('NLCE Fit vs Experiment (Log-Log Scale)', fontsize=14)
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3, which='both')
    ax2.legend(loc='upper right', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(args.output, dpi=150, bbox_inches='tight')
    print(f"Saved plot to {args.output}")
    plt.show()

if __name__ == '__main__':
    main()
