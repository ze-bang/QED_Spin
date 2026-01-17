#!/usr/bin/env python3
"""
Triangle-based NLCE convergence analysis for the triangular lattice.

This script runs triangle-based NLCE calculations at multiple orders and
plots the convergence of thermodynamic quantities.

Triangle-based NLCE:
- Order 0: single site
- Order n (n>=1): n up-pointing triangles sharing vertices
- More efficient than site-based expansion (fewer clusters)
- Each bond belongs to exactly one triangle (proper NLCE)
"""

import numpy as np
import os
import sys
import argparse
import subprocess
import json

# Add paths
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(script_dir, '..', 'run'))
sys.path.insert(0, os.path.join(script_dir, '..', '..', '..', 'python'))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def run_nlce_order(order, base_dir, model_params, workspace_root, visualize=False):
    """Run NLCE for a specific order."""
    order_dir = os.path.join(base_dir, f'order_{order}')
    
    # Build command - use nlce_triangular.py (triangle-based is now the default)
    nlce_script = os.path.join(script_dir, '..', 'run', 'nlce_triangular.py')
    
    cmd = [
        sys.executable,
        nlce_script,
        '--max_order', str(order),
        '--base_dir', order_dir,
        '--model', model_params['model'],
        '--thermo',  # Enable thermodynamic calculation
        '--temp_min', str(model_params['temp_min']),
        '--temp_max', str(model_params['temp_max']),
        '--temp_bins', str(model_params['temp_bins']),
        # triangle-based is now the default, no flag needed
    ]
    
    if visualize:
        cmd.append('--visualize')
    
    # Add model-specific parameters
    if model_params['model'] == 'heisenberg':
        cmd.extend(['--J1', str(model_params.get('J1', 1.0))])
    elif model_params['model'] == 'xxz':
        cmd.extend(['--Jxy', str(model_params.get('Jxy', 1.0))])
        cmd.extend(['--Jz', str(model_params.get('Jz', 1.0))])
    elif model_params['model'] == 'anisotropic':
        cmd.extend(['--Jzz', str(model_params.get('Jzz', 1.0))])
        cmd.extend(['--Jpm', str(model_params.get('Jpm', 0.0))])
        cmd.extend(['--Jpmpm', str(model_params.get('Jpmpm', 0.0))])
        cmd.extend(['--Jzpm', str(model_params.get('Jzpm', 0.0))])
    
    print(f"Running order {order}...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"  Error: {result.stderr[:500] if result.stderr else 'Unknown error'}")
        return None
    
    return order_dir


def load_nlce_results(order_dir, order):
    """Load NLCE results from a completed calculation."""
    # Results are stored in nlc_results_order_{order} subdirectory
    nlc_dir = os.path.join(order_dir, f'nlc_results_order_{order}')
    
    # Try different possible filenames
    possible_files = [
        os.path.join(nlc_dir, 'nlc_energy.txt'),
        os.path.join(order_dir, 'nlce_results', 'nlce_thermodynamics.txt'),
    ]
    
    results_file = None
    for f in possible_files:
        if os.path.exists(f):
            results_file = f
            break
    
    if results_file is None:
        return None
    
    try:
        data = np.loadtxt(results_file, comments='#')
        
        # Try to load other thermodynamic files
        result = {
            'temperature': data[:, 0],
            'energy': data[:, 1] if data.shape[1] > 1 else None,
        }
        
        # Load specific heat
        cv_file = os.path.join(nlc_dir, 'nlc_specific_heat.txt')
        if os.path.exists(cv_file):
            cv_data = np.loadtxt(cv_file, comments='#')
            result['specific_heat'] = cv_data[:, 1]
        else:
            result['specific_heat'] = data[:, 2] if data.shape[1] > 2 else None
        
        # Load entropy
        entropy_file = os.path.join(nlc_dir, 'nlc_entropy.txt')
        if os.path.exists(entropy_file):
            entropy_data = np.loadtxt(entropy_file, comments='#')
            result['entropy'] = entropy_data[:, 1]
        else:
            result['entropy'] = data[:, 3] if data.shape[1] > 3 else None
        
        return result
    except Exception as e:
        print(f"Error loading results from {results_file}: {e}")
        return None


def plot_convergence(results_by_order, output_dir, model_name):
    """Plot convergence of thermodynamic quantities across orders."""
    os.makedirs(output_dir, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    orders = sorted(results_by_order.keys())
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(orders)))
    
    for idx, order in enumerate(orders):
        data = results_by_order[order]
        if data is None:
            continue
        
        T = data['temperature']
        
        # Energy
        axes[0, 0].plot(T, data['energy'], '-', color=colors[idx], 
                       label=f'Order {order}', linewidth=1.5)
        
        # Specific heat
        axes[0, 1].plot(T, data['specific_heat'], '-', color=colors[idx],
                       label=f'Order {order}', linewidth=1.5)
        
        # Entropy
        if data['entropy'] is not None:
            axes[1, 0].plot(T, data['entropy'], '-', color=colors[idx],
                           label=f'Order {order}', linewidth=1.5)
    
    # Format energy plot
    axes[0, 0].set_xlabel('Temperature (J)')
    axes[0, 0].set_ylabel('Energy per site (J)')
    axes[0, 0].set_title('Energy Convergence')
    axes[0, 0].legend()
    axes[0, 0].set_xscale('log')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Format specific heat plot
    axes[0, 1].set_xlabel('Temperature (J)')
    axes[0, 1].set_ylabel('Specific heat per site')
    axes[0, 1].set_title('Specific Heat Convergence')
    axes[0, 1].legend()
    axes[0, 1].set_xscale('log')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Format entropy plot
    axes[1, 0].set_xlabel('Temperature (J)')
    axes[1, 0].set_ylabel('Entropy per site')
    axes[1, 0].set_title('Entropy Convergence')
    axes[1, 0].legend()
    axes[1, 0].set_xscale('log')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Convergence difference plot
    if len(orders) >= 2:
        for idx in range(1, len(orders)):
            prev_order = orders[idx - 1]
            curr_order = orders[idx]
            
            if results_by_order[prev_order] is None or results_by_order[curr_order] is None:
                continue
            
            diff = np.abs(results_by_order[curr_order]['energy'] - 
                         results_by_order[prev_order]['energy'])
            T = results_by_order[curr_order]['temperature']
            
            axes[1, 1].semilogy(T, diff + 1e-10, '-', 
                               label=f'Order {curr_order} - Order {prev_order}',
                               linewidth=1.5)
    
    axes[1, 1].set_xlabel('Temperature (J)')
    axes[1, 1].set_ylabel('|ΔE/N|')
    axes[1, 1].set_title('Order-by-Order Convergence')
    axes[1, 1].legend()
    axes[1, 1].set_xscale('log')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.suptitle(f'Triangle-Based NLCE Convergence: {model_name}', fontsize=14)
    plt.tight_layout()
    
    plot_path = os.path.join(output_dir, 'convergence_triangle_nlce.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved convergence plot: {plot_path}")


def print_cluster_summary(base_dir, max_order):
    """Print summary of clusters at each order."""
    print("\nCluster Summary:")
    print("-" * 50)
    
    for order in range(1, max_order + 1):
        order_dir = os.path.join(base_dir, f'order_{order}')
        summary_file = os.path.join(order_dir, 'cluster_summary.json')
        
        if os.path.exists(summary_file):
            with open(summary_file, 'r') as f:
                summary = json.load(f)
            
            total_clusters = 0
            nonzero_weight = 0
            
            for o_str, o_info in summary['orders'].items():
                total_clusters += o_info['n_clusters']
                for c in o_info['clusters']:
                    if c['weight'] != 0:
                        nonzero_weight += 1
            
            print(f"  Order {order}: {total_clusters} total clusters, "
                  f"{nonzero_weight} with non-zero weight")


def main():
    parser = argparse.ArgumentParser(
        description='Triangle-based NLCE convergence analysis')
    
    parser.add_argument('--max_order', type=int, default=4,
                       help='Maximum order to compute')
    parser.add_argument('--output_dir', type=str, default='./triangle_nlce_convergence',
                       help='Output directory')
    
    # Model parameters
    parser.add_argument('--model', type=str, default='heisenberg',
                       choices=['heisenberg', 'xxz', 'anisotropic'],
                       help='Model type')
    parser.add_argument('--J1', type=float, default=1.0, help='J1 coupling (Heisenberg)')
    parser.add_argument('--Jxy', type=float, default=1.0, help='Jxy (XXZ)')
    parser.add_argument('--Jz', type=float, default=1.0, help='Jz (XXZ)')
    parser.add_argument('--Jzz', type=float, default=1.0, help='Jzz (anisotropic)')
    parser.add_argument('--Jpm', type=float, default=0.0, help='J± (anisotropic)')
    parser.add_argument('--Jpmpm', type=float, default=0.0, help='J±± (anisotropic)')
    parser.add_argument('--Jzpm', type=float, default=0.0, help='Jz± (anisotropic)')
    
    # Temperature range
    parser.add_argument('--temp_min', type=float, default=0.1,
                       help='Minimum temperature')
    parser.add_argument('--temp_max', type=float, default=10.0,
                       help='Maximum temperature')
    parser.add_argument('--temp_bins', type=int, default=100,
                       help='Number of temperature bins')
    
    # Control
    parser.add_argument('--skip_calculations', action='store_true',
                       help='Skip calculations, only plot existing results')
    parser.add_argument('--visualize', action='store_true',
                       help='Generate cluster visualizations')
    
    args = parser.parse_args()
    
    # Find workspace root
    workspace_root = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.dirname(script_dir))))
    
    print("=" * 70)
    print("Triangle-Based NLCE Convergence Analysis")
    print("=" * 70)
    print(f"Maximum order: {args.max_order}")
    print(f"Model: {args.model}")
    print(f"Temperature range: {args.temp_min} - {args.temp_max}")
    print(f"Output directory: {args.output_dir}")
    print()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Build model parameters
    model_params = {
        'model': args.model,
        'J1': args.J1,
        'Jxy': args.Jxy,
        'Jz': args.Jz,
        'Jzz': args.Jzz,
        'Jpm': args.Jpm,
        'Jpmpm': args.Jpmpm,
        'Jzpm': args.Jzpm,
        'temp_min': args.temp_min,
        'temp_max': args.temp_max,
        'temp_bins': args.temp_bins
    }
    
    results_by_order = {}
    
    # Run NLCE for each order
    if not args.skip_calculations:
        for order in range(1, args.max_order + 1):
            order_dir = run_nlce_order(order, args.output_dir, model_params, 
                                        workspace_root, visualize=args.visualize)
            if order_dir:
                results_by_order[order] = load_nlce_results(order_dir, order)
    else:
        # Load existing results
        for order in range(1, args.max_order + 1):
            order_dir = os.path.join(args.output_dir, f'order_{order}')
            results_by_order[order] = load_nlce_results(order_dir, order)
    
    # Plot convergence
    model_name = {
        'heisenberg': f'Heisenberg (J={args.J1})',
        'xxz': f'XXZ (Jxy={args.Jxy}, Jz={args.Jz})',
        'anisotropic': f'Anisotropic (Jzz={args.Jzz}, J±={args.Jpm})'
    }[args.model]
    
    plot_convergence(results_by_order, args.output_dir, model_name)
    
    # Print cluster summary
    print_cluster_summary(args.output_dir, args.max_order)
    
    # Print sample results
    print("\nSample Results (highest order):")
    print("-" * 50)
    
    highest_order = max(o for o, r in results_by_order.items() if r is not None)
    data = results_by_order[highest_order]
    
    if data is not None:
        for idx in [0, len(data['temperature'])//2, -1]:
            T = data['temperature'][idx]
            E = data['energy'][idx]
            C = data['specific_heat'][idx]
            print(f"  T={T:.3f}: E/N={E:.6f}, C/N={C:.6f}")
    
    print("\nDone!")


if __name__ == '__main__':
    main()
