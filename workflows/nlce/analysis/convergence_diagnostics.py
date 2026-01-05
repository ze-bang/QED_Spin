#!/usr/bin/env python3
"""
NLCE Convergence Diagnostics

This script analyzes the order-by-order convergence of NLCE-FTLM results
to understand where and why the series converges or diverges.

Key insights from the analysis:

1. NLCE has a finite convergence radius - at low T (below the gap scale), 
   the series may diverge because the correlation length exceeds cluster sizes.

2. The increments δ_n = S_n - S_{n-1} should decrease with n for convergence.
   If |δ_n| > |δ_{n-1}|, the series is diverging at those temperatures.

3. Resummation methods (Euler, Wynn) can help accelerate convergence for 
   alternating or slowly convergent series, but cannot fix a fundamentally
   divergent series.

4. For NLCE to converge, we need T > T_min where T_min depends on:
   - Gap to first excited state
   - Cluster size (larger clusters → lower T_min)
   - Lattice geometry and coordination
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'run'))


def load_partial_sums(base_dir, max_order):
    """Load partial sum data files."""
    partial_sums = {}
    for order in range(1, max_order + 1):
        filepath = os.path.join(base_dir, f'nlc_partial_sum_order_{order}.txt')
        if os.path.exists(filepath):
            data = np.loadtxt(filepath, comments='#')
            partial_sums[order] = {
                'T': data[:, 0],
                'E': data[:, 1],
                'Cv': data[:, 2],
                'S': data[:, 3],
                'F': data[:, 4]
            }
    return partial_sums


def analyze_convergence_by_temperature(partial_sums):
    """
    Analyze where the series converges vs diverges as a function of temperature.
    
    Returns a dict with convergence metrics at each temperature.
    """
    orders = sorted(partial_sums.keys())
    if len(orders) < 2:
        return None
    
    T = partial_sums[orders[0]]['T']
    n_temps = len(T)
    
    analysis = {
        'T': T,
        'convergence_ratio': {},  # |δ_n|/|δ_{n-1}| for each quantity
        'increment_magnitude': {},  # |δ_n| at each T
        'converges_at_order': {},  # At which order does it start converging?
    }
    
    for quantity in ['E', 'Cv', 'S', 'F']:
        # Compute increments between successive orders
        increments = []
        for i, order in enumerate(orders):
            if i == 0:
                increments.append(partial_sums[order][quantity])
            else:
                increments.append(partial_sums[order][quantity] - partial_sums[orders[i-1]][quantity])
        
        increments = np.array(increments)  # Shape: (n_orders, n_temps)
        
        # Compute ratio of successive increment magnitudes
        if len(orders) >= 2:
            ratios = np.abs(increments[1:]) / (np.abs(increments[:-1]) + 1e-15)
            analysis['convergence_ratio'][quantity] = ratios  # Shape: (n_orders-1, n_temps)
        
        analysis['increment_magnitude'][quantity] = np.abs(increments)
        
        # Find temperature above which the series converges (ratio < 1 for last increment)
        if len(orders) >= 2:
            last_ratio = analysis['convergence_ratio'][quantity][-1]  # Ratio of last two increments
            converging_mask = last_ratio < 1.0
            # Find lowest T where it converges
            if np.any(converging_mask):
                first_converging_idx = np.where(converging_mask)[0][0]
                analysis['converges_at_order'][quantity] = {
                    'T_min_converges': T[first_converging_idx],
                    'T_min_idx': first_converging_idx
                }
            else:
                analysis['converges_at_order'][quantity] = {
                    'T_min_converges': float('inf'),
                    'T_min_idx': None
                }
    
    return analysis


def plot_convergence_diagnostics(partial_sums, output_dir):
    """Create detailed convergence diagnostic plots."""
    orders = sorted(partial_sums.keys())
    max_order = max(orders)
    T = partial_sums[orders[0]]['T']
    
    analysis = analyze_convergence_by_temperature(partial_sums)
    if analysis is None:
        print("Not enough orders for convergence analysis")
        return
    
    # Create figure with 4 panels
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    quantities = ['E', 'Cv', 'S', 'F']
    quantity_labels = {
        'E': 'Energy',
        'Cv': 'Specific Heat',
        'S': 'Entropy',
        'F': 'Free Energy'
    }
    
    for i, q in enumerate(quantities):
        ax = axes.flatten()[i]
        
        # Plot the convergence ratio |δ_n|/|δ_{n-1}| at each temperature
        ratios = analysis['convergence_ratio'][q]
        
        for j in range(len(ratios)):
            order = orders[j+1]
            ax.semilogy(T, ratios[j], '-', label=f'|δ_{order}|/|δ_{order-1}|', linewidth=1.5)
        
        # Add horizontal line at ratio=1 (divergence threshold)
        ax.axhline(y=1.0, color='red', linestyle='--', linewidth=2, label='Convergence threshold')
        ax.axhline(y=0.5, color='orange', linestyle=':', linewidth=1.5, label='Good convergence')
        
        ax.set_xlabel('Temperature', fontsize=12)
        ax.set_ylabel('Increment ratio |δ_n|/|δ_{n-1}|', fontsize=12)
        ax.set_title(f'{quantity_labels[q]} Convergence Ratio', fontsize=14, fontweight='bold')
        ax.set_xscale('log')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=9)
        
        # Shade the "diverging" region
        ax.fill_between(T, 1.0, ax.get_ylim()[1], alpha=0.1, color='red', label='Diverging')
    
    plt.suptitle(f'NLCE Convergence Diagnostics (max order {max_order})\n'
                 f'Ratio > 1: Diverging | Ratio < 1: Converging | Ratio < 0.5: Well-converged',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'convergence_diagnostics.png'), dpi=300)
    plt.close()
    
    print(f"Diagnostic plot saved to {output_dir}/convergence_diagnostics.png")
    
    # Print summary
    print("\n" + "="*80)
    print("CONVERGENCE ANALYSIS SUMMARY")
    print("="*80)
    
    for q in quantities:
        info = analysis['converges_at_order'][q]
        if info['T_min_idx'] is not None:
            print(f"\n{quantity_labels[q]}:")
            print(f"  Converges for T > {info['T_min_converges']:.4f}")
            
            # Check the ratio at different temperature ranges
            ratios = analysis['convergence_ratio'][q][-1]  # Last order ratio
            low_T_idx = len(T) // 10
            mid_T_idx = len(T) // 2
            high_T_idx = 9 * len(T) // 10
            
            print(f"  Ratio at T={T[low_T_idx]:.3f}: {ratios[low_T_idx]:.4f} {'(DIVERGING)' if ratios[low_T_idx] > 1 else '(converging)'}")
            print(f"  Ratio at T={T[mid_T_idx]:.3f}: {ratios[mid_T_idx]:.4f} {'(DIVERGING)' if ratios[mid_T_idx] > 1 else '(converging)'}")
            print(f"  Ratio at T={T[high_T_idx]:.3f}: {ratios[high_T_idx]:.4f} {'(DIVERGING)' if ratios[high_T_idx] > 1 else '(converging)'}")
        else:
            print(f"\n{quantity_labels[q]}: NOT CONVERGING at any temperature!")
    
    print("\n" + "="*80)
    
    return analysis


def main():
    import argparse
    parser = argparse.ArgumentParser(description='NLCE Convergence Diagnostics')
    parser.add_argument('--base_dir', type=str, required=True, help='Directory with NLC results')
    parser.add_argument('--max_order', type=int, required=True, help='Maximum order')
    args = parser.parse_args()
    
    # Find the nlc_results directory
    nlc_dir = os.path.join(args.base_dir, f'nlc_results_order_{args.max_order}')
    if not os.path.exists(nlc_dir):
        print(f"Error: NLC results directory not found: {nlc_dir}")
        sys.exit(1)
    
    # Load partial sums
    partial_sums = load_partial_sums(nlc_dir, args.max_order)
    if not partial_sums:
        print("Error: No partial sum data found")
        sys.exit(1)
    
    print(f"Loaded partial sums for orders: {list(partial_sums.keys())}")
    
    # Run analysis
    output_dir = os.path.join(args.base_dir, 'convergence_diagnostics')
    os.makedirs(output_dir, exist_ok=True)
    
    analysis = plot_convergence_diagnostics(partial_sums, output_dir)
    
    print(f"\nResults saved to: {output_dir}")


if __name__ == '__main__':
    main()
