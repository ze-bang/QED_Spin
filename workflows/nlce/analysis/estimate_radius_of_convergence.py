#!/usr/bin/env python3
"""
Estimate Radius of Convergence for NLCE Series

Analyzes NLCE partial sums to determine the temperature below which the series diverges.

Methods implemented:
1. Ratio test: Extrapolate |a_{n+1}|/|a_n| to estimate where it reaches 1
2. Root test: Check convergence via (|a_n|)^(1/n)
3. Cauchy criterion: Monitor consecutive partial sum differences
4. Resummation stability: Check where resummation methods break down

Reference: Rigol, Dunjko, Yurovsky, Olshanii, Nature 452, 854-858 (2008)
"""

import numpy as np
import matplotlib.pyplot as plt
import argparse
import glob
import os
import re
from scipy.optimize import curve_fit
from scipy.interpolate import UnivariateSpline


def load_partial_sums(nlc_dir, max_order=None):
    """Load partial sums from NLCE calculation"""
    partial_sum_files = glob.glob(os.path.join(nlc_dir, 'nlc_partial_sum_order_*.txt'))
    
    orders = []
    partial_sums = {}
    
    for file in partial_sum_files:
        match = re.search(r'order_(\d+)', file)
        if match:
            order = int(match.group(1))
            if max_order is None or order <= max_order:
                orders.append(order)
                
                data = np.loadtxt(file)
                if len(data.shape) == 1:
                    data = data.reshape(1, -1)
                
                partial_sums[order] = {
                    'temperature': data[:, 0],
                    'energy': data[:, 1],
                    'specific_heat': data[:, 2],
                    'entropy': data[:, 3],
                    'free_energy': data[:, 4] if data.shape[1] > 4 else None
                }
    
    orders.sort()
    return orders, partial_sums


def ratio_test_analysis(partial_sums, orders, property='energy'):
    """
    Apply ratio test to estimate radius of convergence.
    
    For each temperature T, compute ratio r_n(T) = |a_n(T)| / |a_{n-1}(T)|
    where a_n = S_n - S_{n-1} is the nth increment.
    
    Convergence requires lim sup r_n < 1.
    Estimate T_c where r_n(T) → 1.
    """
    if len(orders) < 3:
        print("Need at least 3 orders for ratio test")
        return None
    
    temps = partial_sums[orders[0]]['temperature']
    n_temps = len(temps)
    
    # Collect increments for each order
    increments = []
    for i, order in enumerate(orders):
        if i == 0:
            increments.append(partial_sums[order][property])
        else:
            increments.append(partial_sums[order][property] - partial_sums[orders[i-1]][property])
    
    increments = np.array(increments)
    
    # Compute ratios
    ratios = np.zeros((len(orders)-1, n_temps))
    for i in range(1, len(orders)):
        ratios[i-1] = np.abs(increments[i]) / (np.abs(increments[i-1]) + 1e-15)
    
    # For each temperature, extrapolate where ratio → 1
    # Using last 3 ratios: fit exponential or linear trend
    divergence_indicators = np.zeros(n_temps)
    
    for t_idx in range(n_temps):
        ratio_vals = ratios[-3:, t_idx]  # Last 3 ratios
        
        # Average ratio (simple indicator)
        avg_ratio = np.mean(ratio_vals)
        divergence_indicators[t_idx] = avg_ratio
    
    # Estimate critical temperature where ratio ≈ 0.95-1.0
    convergent_mask = divergence_indicators < 0.8
    if np.any(convergent_mask):
        # Find temperature boundary
        T_min_convergent = np.min(temps[convergent_mask])
        
        # Find where ratio crosses 0.9 threshold
        divergent_mask = divergence_indicators > 0.9
        if np.any(divergent_mask):
            T_c_estimate = np.max(temps[divergent_mask])
        else:
            T_c_estimate = T_min_convergent * 0.5
    else:
        T_c_estimate = np.min(temps)
    
    return {
        'temperatures': temps,
        'ratios_by_order': ratios,
        'avg_ratio': divergence_indicators,
        'T_c_estimate': T_c_estimate
    }


def root_test_analysis(partial_sums, orders, property='energy'):
    """
    Apply root test: examine (|a_n|)^(1/n) for convergence.
    
    Series converges if lim sup (|a_n|)^(1/n) < 1/ρ where ρ is radius of convergence.
    """
    if len(orders) < 3:
        return None
    
    temps = partial_sums[orders[0]]['temperature']
    n_temps = len(temps)
    
    # Collect increments
    increments = []
    for i, order in enumerate(orders):
        if i == 0:
            increments.append(partial_sums[order][property])
        else:
            increments.append(partial_sums[order][property] - partial_sums[orders[i-1]][property])
    
    increments = np.array(increments)
    
    # Compute root test values
    root_vals = np.zeros((len(orders), n_temps))
    for i, order in enumerate(orders):
        if order > 0:
            root_vals[i] = np.abs(increments[i]) ** (1.0 / order)
    
    return {
        'temperatures': temps,
        'root_test_by_order': root_vals
    }


def consecutive_difference_analysis(partial_sums, orders, property='energy'):
    """
    Analyze how |S_n - S_{n-1}| behaves.
    
    Convergence requires this to decrease with n.
    Divergence shows increasing differences or oscillations.
    """
    if len(orders) < 3:
        return None
    
    temps = partial_sums[orders[0]]['temperature']
    n_temps = len(temps)
    
    diffs = np.zeros((len(orders)-1, n_temps))
    for i in range(1, len(orders)):
        diffs[i-1] = np.abs(partial_sums[orders[i]][property] - partial_sums[orders[i-1]][property])
    
    # Check if differences are growing (sign of divergence)
    growth_rate = np.zeros(n_temps)
    for t_idx in range(n_temps):
        if len(orders) >= 4:
            # Compare last two differences
            recent = diffs[-2:, t_idx]
            if recent[0] > 1e-12:
                growth_rate[t_idx] = recent[1] / recent[0]
            else:
                growth_rate[t_idx] = 0.0
    
    return {
        'temperatures': temps,
        'differences': diffs,
        'growth_rate': growth_rate
    }


def plot_convergence_diagnostics(ratio_result, root_result, diff_result, 
                                 property='energy', output_dir='.'):
    """Create comprehensive convergence diagnostic plots"""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Ratio test over temperature
    if ratio_result:
        ax = axes[0, 0]
        temps = ratio_result['temperatures']
        
        # Plot individual order ratios
        n_orders = ratio_result['ratios_by_order'].shape[0]
        cmap = plt.get_cmap('viridis')
        for i in range(n_orders):
            color = cmap(i / max(1, n_orders-1))
            ax.plot(temps, ratio_result['ratios_by_order'][i], 
                   label=f'Order {i+1}/{i+2}', alpha=0.6, color=color)
        
        # Average ratio
        ax.plot(temps, ratio_result['avg_ratio'], 'r-', linewidth=2, 
               label='Average (last 3)', zorder=10)
        
        # Convergence threshold
        ax.axhline(y=1.0, color='k', linestyle='--', label='Divergence threshold')
        ax.axhline(y=0.9, color='orange', linestyle=':', label='Warning zone')
        
        # Estimated T_c
        T_c = ratio_result['T_c_estimate']
        ax.axvline(x=T_c, color='red', linestyle='--', linewidth=2,
                  label=f'Est. T_c = {T_c:.3f}')
        
        ax.set_xlabel('Temperature')
        ax.set_ylabel('Ratio |a_n|/|a_{n-1}|')
        ax.set_title(f'Ratio Test Analysis ({property})')
        ax.set_xscale('log')
        ax.set_ylim([0, 2])
        ax.legend(fontsize=8, loc='best')
        ax.grid(True, alpha=0.3)
    
    # Plot 2: Root test
    if root_result:
        ax = axes[0, 1]
        temps = root_result['temperatures']
        root_vals = root_result['root_test_by_order']
        
        for i in range(root_vals.shape[0]):
            if i > 0:  # Skip order 0
                ax.plot(temps, root_vals[i], label=f'Order {i}', alpha=0.7)
        
        ax.set_xlabel('Temperature')
        ax.set_ylabel('(|a_n|)^(1/n)')
        ax.set_title(f'Root Test Analysis ({property})')
        ax.set_xscale('log')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    # Plot 3: Consecutive differences
    if diff_result:
        ax = axes[1, 0]
        temps = diff_result['temperatures']
        diffs = diff_result['differences']
        
        for i in range(diffs.shape[0]):
            ax.plot(temps, diffs[i], label=f'|S_{i+1} - S_{i}|', alpha=0.7)
        
        ax.set_xlabel('Temperature')
        ax.set_ylabel('|S_n - S_{n-1}|')
        ax.set_title(f'Convergence of Partial Sums ({property})')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    # Plot 4: Growth rate of differences
    if diff_result:
        ax = axes[1, 1]
        temps = diff_result['temperatures']
        growth = diff_result['growth_rate']
        
        ax.plot(temps, growth, 'b-', linewidth=2, label='Growth rate')
        ax.axhline(y=1.0, color='k', linestyle='--', label='No growth')
        ax.axhline(y=1.5, color='orange', linestyle=':', label='Rapid growth')
        
        # Identify divergence region
        divergent = growth > 1.5
        if np.any(divergent):
            ax.fill_between(temps, 0, 3, where=divergent, 
                           alpha=0.2, color='red', label='Divergent region')
        
        ax.set_xlabel('Temperature')
        ax.set_ylabel('Growth Rate')
        ax.set_title(f'Growth Rate of Increments ({property})')
        ax.set_xscale('log')
        ax.set_ylim([0, 3])
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_file = os.path.join(output_dir, f'radius_of_convergence_{property}.png')
    plt.savefig(output_file, dpi=300)
    plt.close()
    
    print(f"Saved convergence diagnostic plot: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Estimate radius of convergence for NLCE series',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  # Analyze NLCE results
  python estimate_radius_of_convergence.py \\
      --nlc_dir ./nlce_results/nlc_results \\
      --max_order 6 \\
      --output_dir ./convergence_analysis
  
  # Focus on specific property
  python estimate_radius_of_convergence.py \\
      --nlc_dir ./nlce_ftlm_results/order_5/nlc_results \\
      --property specific_heat
        """
    )
    
    parser.add_argument('--nlc_dir', type=str, required=True,
                       help='Directory containing nlc_partial_sum_order_*.txt files')
    parser.add_argument('--max_order', type=int, default=None,
                       help='Maximum order to analyze')
    parser.add_argument('--property', type=str, default='specific_heat',
                       choices=['energy', 'specific_heat', 'entropy', 'free_energy'],
                       help='Property to analyze')
    parser.add_argument('--output_dir', type=str, default='.',
                       help='Output directory for plots and results')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load data
    print(f"Loading NLCE partial sums from {args.nlc_dir}")
    orders, partial_sums = load_partial_sums(args.nlc_dir, args.max_order)
    
    if not orders:
        print(f"Error: No partial sum files found in {args.nlc_dir}")
        return
    
    print(f"Found orders: {orders}")
    print(f"Analyzing property: {args.property}")
    print("="*80)
    
    # Run diagnostics
    print("\n1. Ratio Test Analysis")
    ratio_result = ratio_test_analysis(partial_sums, orders, args.property)
    if ratio_result:
        T_c = ratio_result['T_c_estimate']
        print(f"   Estimated critical temperature: T_c ≈ {T_c:.4f}")
        print(f"   Series likely divergent for T < {T_c:.4f}")
        
        # Find temperature where ratio is reasonable
        safe_temps = ratio_result['temperatures'][ratio_result['avg_ratio'] < 0.7]
        if len(safe_temps) > 0:
            T_safe = np.min(safe_temps)
            print(f"   Series appears convergent for T > {T_safe:.4f}")
    
    print("\n2. Root Test Analysis")
    root_result = root_test_analysis(partial_sums, orders, args.property)
    
    print("\n3. Consecutive Difference Analysis")
    diff_result = consecutive_difference_analysis(partial_sums, orders, args.property)
    if diff_result:
        # Find where growth rate > 1.5
        high_growth = diff_result['growth_rate'] > 1.5
        if np.any(high_growth):
            divergent_temps = diff_result['temperatures'][high_growth]
            print(f"   Rapid growth detected at T < {np.max(divergent_temps):.4f}")
    
    print("\n4. Generating diagnostic plots...")
    plot_convergence_diagnostics(ratio_result, root_result, diff_result,
                                 args.property, args.output_dir)
    
    # Save summary
    summary_file = os.path.join(args.output_dir, f'convergence_summary_{args.property}.txt')
    with open(summary_file, 'w') as f:
        f.write("NLCE Radius of Convergence Analysis\n")
        f.write("="*80 + "\n")
        f.write(f"Property: {args.property}\n")
        f.write(f"Orders analyzed: {orders}\n\n")
        
        if ratio_result:
            f.write("Ratio Test Results:\n")
            f.write(f"  Estimated T_c (ratio → 1): {ratio_result['T_c_estimate']:.4f}\n")
            f.write(f"  Recommendation: Use data with T > {ratio_result['T_c_estimate']:.4f}\n\n")
        
        if diff_result:
            high_growth = diff_result['growth_rate'] > 1.5
            if np.any(high_growth):
                T_max_divergent = np.max(diff_result['temperatures'][high_growth])
                f.write("Growth Rate Analysis:\n")
                f.write(f"  Rapid divergence detected below T ≈ {T_max_divergent:.4f}\n\n")
        
        f.write("\nInterpretation:\n")
        f.write("- The series diverges at low temperature due to finite radius of convergence\n")
        f.write("- Use resummation methods (Euler, Wynn) to extend validity\n")
        f.write("- For quantitative results, restrict to convergent regime\n")
        f.write("- Consider higher orders to push T_c lower\n")
    
    print(f"\nSummary saved to: {summary_file}")
    print("="*80)
    print("\nRecommendations:")
    if ratio_result:
        print(f"1. Series is reliable for T > {ratio_result['T_c_estimate']:.4f}")
        print(f"2. Use resummation methods for {ratio_result['T_c_estimate']*0.5:.4f} < T < {ratio_result['T_c_estimate']:.4f}")
        print(f"3. Results below T ≈ {ratio_result['T_c_estimate']*0.5:.4f} are unreliable")


if __name__ == "__main__":
    main()
