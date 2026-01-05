#!/usr/bin/env python3
"""
Test script to compare different resummation methods for NLCE convergence.

This script:
1. Runs NLCE-FTLM for a given max order (once)
2. Re-analyzes the data with all resummation methods
3. Creates comparison plots to investigate convergence behavior

Usage:
    python test_resummation_methods.py --max_order 4 --base_dir test_resum
"""

import os
import sys
import argparse
import subprocess
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'run'))
from NLC_sum_ftlm import NLCExpansionFTLM


def run_initial_calculation(args):
    """Run NLCE-FTLM once to generate cluster data."""
    print("="*80)
    print("STEP 1: Running initial NLCE-FTLM calculation")
    print("="*80)
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    nlce_ftlm_script = os.path.join(script_dir, '..', 'run', 'nlce_ftlm.py')
    
    cmd = [
        sys.executable,
        nlce_ftlm_script,
        f'--max_order={args.max_order}',
        f'--base_dir={args.base_dir}',
        f'--ed_executable={args.ed_executable}',
        f'--Jxx={args.Jxx}',
        f'--Jyy={args.Jyy}',
        f'--Jzz={args.Jzz}',
        f'--h={args.h}',
        f'--ftlm_samples={args.ftlm_samples}',
        f'--krylov_dim={args.krylov_dim}',
        f'--temp_min={args.temp_min}',
        f'--temp_max={args.temp_max}',
        f'--temp_bins={args.temp_bins}',
        '--resummation=direct',  # Use direct for initial run
        '-v',  # Verbose
    ]
    
    if args.parallel:
        cmd.append('--parallel')
        if args.num_cores:
            cmd.append(f'--num_cores={args.num_cores}')
    
    if args.symmetrized:
        cmd.append('--symmetrized')
    
    if args.use_gpu:
        cmd.append('--use_gpu')
    
    print(f"Running: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error running NLCE-FTLM: {e}")
        return False


def analyze_with_resummation(cluster_dir, ftlm_dir, temp_min, temp_max, temp_bins, 
                              method, max_order, output_dir):
    """Re-analyze existing data with a specific resummation method."""
    print(f"\n{'='*60}")
    print(f"Analyzing with resummation method: {method}")
    print(f"{'='*60}")
    
    # Create NLC expansion object
    nlce = NLCExpansionFTLM(
        cluster_dir=cluster_dir,
        ftlm_dir=ftlm_dir,
        temp_min=temp_min,
        temp_max=temp_max,
        num_temps=temp_bins,
        SI_units=False
    )
    
    # Read cluster and FTLM data
    nlce.read_clusters()
    nlce.read_ftlm_data()
    
    # Calculate weights
    nlce.calculate_weights(verbose=False)
    
    # Perform NLC summation with the specified resummation method
    results = nlce.sum_nlc(resummation_method=method, verbose=True)
    
    # Save results
    method_dir = os.path.join(output_dir, f'resummation_{method}')
    os.makedirs(method_dir, exist_ok=True)
    
    # Save thermodynamic data
    for prop in ['energy', 'specific_heat', 'entropy', 'free_energy']:
        data = np.column_stack([
            results['temperatures'],
            results[prop],
            results[f'{prop}_error']
        ])
        np.savetxt(
            os.path.join(method_dir, f'nlc_{prop}.txt'),
            data,
            header='Temperature\tValue\tError',
            fmt='%.10e'
        )
    
    return results


def plot_comparison(all_results, temp_values, output_dir, max_order):
    """Create comparison plots for all resummation methods."""
    print("\n" + "="*80)
    print("Creating comparison plots")
    print("="*80)
    
    methods = list(all_results.keys())
    colors = plt.cm.tab10(np.linspace(0, 1, len(methods)))
    
    properties = ['energy', 'specific_heat', 'entropy', 'free_energy']
    property_labels = {
        'energy': 'Energy per site',
        'specific_heat': 'Specific Heat per site',
        'entropy': 'Entropy per site',
        'free_energy': 'Free Energy per site'
    }
    
    # Create 2x2 plot for all properties
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()
    
    for i, prop in enumerate(properties):
        ax = axes[i]
        ax.set_title(f'{property_labels[prop]} (max order {max_order})', fontsize=12, fontweight='bold')
        ax.set_xlabel('Temperature')
        ax.set_ylabel(property_labels[prop])
        ax.set_xscale('log')
        ax.grid(True, alpha=0.3)
        
        for j, method in enumerate(methods):
            results = all_results[method]
            ax.plot(temp_values, results[prop], '-', color=colors[j], 
                   label=method, linewidth=2, alpha=0.8)
            
            # Add error band if available
            if results[f'{prop}_error'] is not None and np.any(results[f'{prop}_error'] > 0):
                ax.fill_between(
                    temp_values,
                    results[prop] - results[f'{prop}_error'],
                    results[prop] + results[f'{prop}_error'],
                    alpha=0.2, color=colors[j]
                )
        
        ax.legend(loc='best', fontsize=9)
    
    plt.suptitle(f'NLCE Resummation Method Comparison\n(Order {max_order}, Isotropic Heisenberg J=1)', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'resummation_comparison_all.png'), dpi=300)
    plt.close()
    
    # Create individual plots for specific heat (most sensitive)
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_title(f'Specific Heat Comparison - Different Resummation Methods\n(max order {max_order})', 
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Temperature', fontsize=12)
    ax.set_ylabel('Specific Heat per site', fontsize=12)
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)
    
    for j, method in enumerate(methods):
        results = all_results[method]
        ax.plot(temp_values, results['specific_heat'], '-', color=colors[j], 
               label=method, linewidth=2.5, alpha=0.9)
        
        if results['specific_heat_error'] is not None and np.any(results['specific_heat_error'] > 0):
            ax.fill_between(
                temp_values,
                results['specific_heat'] - results['specific_heat_error'],
                results['specific_heat'] + results['specific_heat_error'],
                alpha=0.15, color=colors[j]
            )
    
    ax.set_ylim(bottom=0)  # Cv should be positive
    ax.legend(loc='best', fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'resummation_comparison_Cv.png'), dpi=300)
    plt.close()
    
    print(f"Plots saved to {output_dir}")


def plot_partial_sums_analysis(nlce, output_dir, max_order):
    """
    Create detailed plots showing partial sums at each order to diagnose convergence.
    """
    print("\n" + "="*80)
    print("Creating partial sum analysis plots")
    print("="*80)
    
    # Recompute partial sums for analysis
    order_contributions = {}
    for cluster_id in nlce.clusters:
        if not nlce.clusters[cluster_id].get('has_data', False):
            continue
        
        order = nlce.clusters[cluster_id]['order']
        multiplicity = nlce.clusters[cluster_id]['multiplicity']
        
        if order not in order_contributions:
            order_contributions[order] = {
                'energy': np.zeros_like(nlce.temp_values),
                'specific_heat': np.zeros_like(nlce.temp_values),
                'entropy': np.zeros_like(nlce.temp_values),
                'free_energy': np.zeros_like(nlce.temp_values)
            }
        
        for quantity in ['energy', 'specific_heat', 'entropy', 'free_energy']:
            weight = nlce.weights[quantity][cluster_id]
            order_contributions[order][quantity] += weight * multiplicity
    
    # Compute partial sums up to each order
    orders = sorted(order_contributions.keys())
    temp_values = nlce.temp_values
    
    properties = ['energy', 'specific_heat', 'entropy']
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # Top row: Partial sums S_n
    # Bottom row: Increments δ_n = S_n - S_{n-1}
    
    cmap = plt.cm.viridis
    colors = [cmap(i / len(orders)) for i in range(len(orders))]
    
    for col, prop in enumerate(properties):
        ax_sum = axes[0, col]
        ax_inc = axes[1, col]
        
        partial_sums = []
        for n, order in enumerate(orders):
            partial_sum = np.zeros_like(temp_values)
            for o in range(1, order + 1):
                if o in order_contributions:
                    partial_sum += order_contributions[o][prop]
            partial_sums.append(partial_sum)
            
            ax_sum.plot(temp_values, partial_sum, '-', color=colors[n], 
                       label=f'S_{order}', linewidth=1.5, alpha=0.8)
            
            # Increment
            if n > 0:
                increment = partial_sum - partial_sums[n-1]
            else:
                increment = partial_sum
            
            ax_inc.plot(temp_values, increment, '-', color=colors[n],
                       label=f'δ_{order}', linewidth=1.5, alpha=0.8)
        
        ax_sum.set_title(f'Partial Sums S_n for {prop}', fontsize=11, fontweight='bold')
        ax_sum.set_xlabel('Temperature')
        ax_sum.set_xscale('log')
        ax_sum.grid(True, alpha=0.3)
        ax_sum.legend(loc='best', fontsize=8)
        
        ax_inc.set_title(f'Increments δ_n for {prop}', fontsize=11, fontweight='bold')
        ax_inc.set_xlabel('Temperature')
        ax_inc.set_xscale('log')
        ax_inc.grid(True, alpha=0.3)
        ax_inc.legend(loc='best', fontsize=8)
        
        # Add horizontal line at y=0 for increments
        ax_inc.axhline(y=0, color='gray', linestyle='--', linewidth=0.5)
    
    plt.suptitle(f'NLCE Convergence Analysis: Partial Sums and Increments (max order {max_order})', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'partial_sums_analysis.png'), dpi=300)
    plt.close()
    
    # Also create a plot showing increment magnitudes |δ_n| to check convergence
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Select a few representative temperatures
    n_temps = len(temp_values)
    temp_indices = {
        f'T={temp_values[n_temps//10]:.3f}': n_temps // 10,
        f'T={temp_values[n_temps//4]:.3f}': n_temps // 4,
        f'T={temp_values[n_temps//2]:.3f}': n_temps // 2,
        f'T={temp_values[3*n_temps//4]:.3f}': 3 * n_temps // 4,
    }
    
    markers = ['o', 's', '^', 'D', 'v', '<', '>']
    
    for i, (label, idx) in enumerate(temp_indices.items()):
        increments_at_T = []
        for n, order in enumerate(orders):
            partial_sum = np.zeros_like(temp_values)
            for o in range(1, order + 1):
                if o in order_contributions:
                    partial_sum += order_contributions[o]['specific_heat']
            
            if n > 0:
                prev_partial = np.zeros_like(temp_values)
                for o in range(1, order):
                    if o in order_contributions:
                        prev_partial += order_contributions[o]['specific_heat']
                increment = partial_sum[idx] - prev_partial[idx]
            else:
                increment = partial_sum[idx]
            
            increments_at_T.append(np.abs(increment))
        
        ax.semilogy(orders, increments_at_T, f'-{markers[i]}', label=label, 
                   linewidth=2, markersize=8, alpha=0.8)
    
    ax.set_xlabel('Order n', fontsize=12)
    ax.set_ylabel('|δ_n| (Specific Heat increment magnitude)', fontsize=12)
    ax.set_title('Convergence Check: Should |δ_n| decrease with order?', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'convergence_check_increments.png'), dpi=300)
    plt.close()
    
    print(f"Partial sum analysis saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description='Test different resummation methods for NLCE convergence',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This script runs NLCE-FTLM once and then compares all resummation methods.
        """
    )
    
    parser.add_argument('--max_order', type=int, required=True,
                       help='Maximum NLCE order')
    parser.add_argument('--base_dir', type=str, default='./test_resummation',
                       help='Base directory for results')
    parser.add_argument('--ed_executable', type=str, default='./build/ED',
                       help='Path to ED executable')
    
    # Model parameters
    parser.add_argument('--Jxx', type=float, default=1.0)
    parser.add_argument('--Jyy', type=float, default=1.0)
    parser.add_argument('--Jzz', type=float, default=1.0)
    parser.add_argument('--h', type=float, default=0.0)
    
    # FTLM parameters
    parser.add_argument('--ftlm_samples', type=int, default=50,
                       help='Number of FTLM samples')
    parser.add_argument('--krylov_dim', type=int, default=200,
                       help='Krylov dimension')
    parser.add_argument('--temp_min', type=float, default=0.01)
    parser.add_argument('--temp_max', type=float, default=20.0)
    parser.add_argument('--temp_bins', type=int, default=100)
    
    # Options
    parser.add_argument('--parallel', action='store_true')
    parser.add_argument('--num_cores', type=int, default=None)
    parser.add_argument('--symmetrized', action='store_true')
    parser.add_argument('--use_gpu', action='store_true')
    parser.add_argument('--skip_calculation', action='store_true',
                       help='Skip initial FTLM calculation (reuse existing data)')
    
    args = parser.parse_args()
    
    # Create base directory
    os.makedirs(args.base_dir, exist_ok=True)
    
    # Step 1: Run initial NLCE-FTLM calculation (if not skipping)
    if not args.skip_calculation:
        success = run_initial_calculation(args)
        if not success:
            print("Initial calculation failed!")
            sys.exit(1)
    else:
        print("Skipping initial calculation, using existing data...")
    
    # Find cluster and FTLM directories (structure: base_dir/clusters_order_N/cluster_info_order_N)
    cluster_dir = os.path.join(args.base_dir, f'clusters_order_{args.max_order}', f'cluster_info_order_{args.max_order}')
    ftlm_dir = os.path.join(args.base_dir, f'ftlm_results_order_{args.max_order}')
    output_dir = os.path.join(args.base_dir, 'resummation_analysis')
    os.makedirs(output_dir, exist_ok=True)
    
    if not os.path.exists(cluster_dir):
        print(f"Error: Cluster directory not found: {cluster_dir}")
        print("Available directories:")
        for d in os.listdir(args.base_dir):
            print(f"  {d}")
        print("Run without --skip_calculation first")
        sys.exit(1)
    
    # Step 2: Analyze with different resummation methods
    methods = ['direct', 'euler', 'wynn', 'robust']
    all_results = {}
    
    nlce = None  # Keep for later analysis
    
    for method in methods:
        try:
            nlce_obj = NLCExpansionFTLM(
                cluster_dir=cluster_dir,
                ftlm_dir=ftlm_dir,
                temp_min=args.temp_min,
                temp_max=args.temp_max,
                num_temps=args.temp_bins,
                SI_units=False
            )
            nlce_obj.read_clusters()
            nlce_obj.read_ftlm_data()
            nlce_obj.calculate_weights(verbose=False)
            
            results = nlce_obj.sum_nlc(resummation_method=method, verbose=True)
            all_results[method] = results
            
            # Save for partial sum analysis
            if method == 'direct':
                nlce = nlce_obj
            
        except Exception as e:
            print(f"Error with method {method}: {e}")
            import traceback
            traceback.print_exc()
    
    # Step 3: Create comparison plots
    if len(all_results) > 0:
        temp_values = list(all_results.values())[0]['temperatures']
        plot_comparison(all_results, temp_values, output_dir, args.max_order)
        
        # Plot partial sums analysis for convergence diagnosis
        if nlce is not None:
            plot_partial_sums_analysis(nlce, output_dir, args.max_order)
    
    print("\n" + "="*80)
    print("Analysis complete!")
    print(f"Results saved to: {output_dir}")
    print("="*80)


if __name__ == '__main__':
    main()
