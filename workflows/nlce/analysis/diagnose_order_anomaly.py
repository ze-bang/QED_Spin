#!/usr/bin/env python3
"""
Diagnose Order Anomalies in NLCE Results

When a specific order (e.g., order 6) shows sudden large changes in all properties,
this script helps identify the root cause by analyzing:
1. Number of clusters at each order
2. FTLM quality metrics for each cluster
3. Weight contributions and anomalies
4. Comparison with expected scaling

Usage:
    python diagnose_order_anomaly.py --nlc_dir ./nlce_results --problem_order 6
"""

import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import glob
import re
import h5py
from collections import defaultdict


def analyze_cluster_distribution(cluster_dir, max_order):
    """Count clusters and analyze topology distribution by order"""
    cluster_files = glob.glob(os.path.join(cluster_dir, "cluster_*_order_*.dat"))
    
    order_counts = defaultdict(int)
    order_sites = defaultdict(list)
    order_multiplicity = defaultdict(list)
    
    for file_path in cluster_files:
        basename = os.path.basename(file_path)
        match = re.search(r'cluster_(\d+)_order_(\d+)', basename)
        if match:
            cluster_id = int(match.group(1))
            order = int(match.group(2))
            
            if order <= max_order:
                order_counts[order] += 1
                
                # Read file to get number of sites and multiplicity
                with open(file_path, 'r') as f:
                    n_sites = 0
                    multiplicity = 1.0
                    for line in f:
                        if line.startswith("# Multiplicity") and ":" in line:
                            multiplicity = float(line.split(":")[-1].strip())
                        elif not line.startswith('#') and line.strip():
                            n_sites += 1
                    
                    order_sites[order].append(n_sites)
                    order_multiplicity[order].append(multiplicity)
    
    return order_counts, order_sites, order_multiplicity


def analyze_ftlm_quality(ftlm_dir, max_order):
    """Analyze FTLM quality metrics for each cluster"""
    cluster_ftlm_dirs = glob.glob(os.path.join(ftlm_dir, "cluster_*_order_*"))
    
    order_quality = defaultdict(list)
    order_errors = defaultdict(lambda: {'E': [], 'C': [], 'S': [], 'F': []})
    
    for cluster_dir in cluster_ftlm_dirs:
        basename = os.path.basename(cluster_dir)
        match = re.search(r'cluster_(\d+)_order_(\d+)', basename)
        if not match:
            continue
        
        cluster_id = int(match.group(1))
        order = int(match.group(2))
        
        if order > max_order:
            continue
        
        # Check for HDF5 output
        h5_file = os.path.join(cluster_dir, 'output', 'thermo', 'ed_results.h5')
        if not os.path.exists(h5_file):
            h5_file = os.path.join(cluster_dir, 'output', 'ed_results.h5')
        
        if os.path.exists(h5_file):
            try:
                with h5py.File(h5_file, 'r') as f:
                    if '/ftlm/averaged' in f:
                        ftlm_grp = f['/ftlm/averaged']
                        
                        # Get average errors
                        if 'energy_error' in ftlm_grp:
                            E_err = np.mean(ftlm_grp['energy_error'][:])
                            C_err = np.mean(ftlm_grp['specific_heat_error'][:]) if 'specific_heat_error' in ftlm_grp else 0
                            S_err = np.mean(ftlm_grp['entropy_error'][:]) if 'entropy_error' in ftlm_grp else 0
                            F_err = np.mean(ftlm_grp['free_energy_error'][:]) if 'free_energy_error' in ftlm_grp else 0
                            
                            order_errors[order]['E'].append(E_err)
                            order_errors[order]['C'].append(C_err)
                            order_errors[order]['S'].append(S_err)
                            order_errors[order]['F'].append(F_err)
                            
                            # Quality metric: relative error
                            E_val = np.mean(np.abs(ftlm_grp['energy'][:]))
                            quality = E_err / (E_val + 1e-10)
                            order_quality[order].append(quality)
            except Exception as e:
                print(f"Warning: Could not read {h5_file}: {e}")
    
    return order_quality, order_errors


def load_partial_sums(nlc_dir, max_order):
    """Load partial sums from NLCE results"""
    partial_sums = {}
    
    for order in range(1, max_order + 1):
        file_path = os.path.join(nlc_dir, f'nlc_partial_sum_order_{order}.txt')
        if os.path.exists(file_path):
            data = np.loadtxt(file_path)
            if len(data.shape) == 1:
                data = data.reshape(1, -1)
            
            partial_sums[order] = {
                'temperature': data[:, 0],
                'energy': data[:, 1],
                'specific_heat': data[:, 2],
                'entropy': data[:, 3],
                'free_energy': data[:, 4] if data.shape[1] > 4 else None
            }
    
    return partial_sums


def analyze_increments(partial_sums, problem_order):
    """Analyze increments between orders"""
    if problem_order not in partial_sums or problem_order - 1 not in partial_sums:
        return None
    
    temps = partial_sums[problem_order]['temperature']
    
    increments = {}
    for prop in ['energy', 'specific_heat', 'entropy']:
        increments[prop] = []
        
        for order in sorted(partial_sums.keys()):
            if order == 1:
                increments[prop].append(partial_sums[order][prop])
            else:
                prev_order = order - 1
                if prev_order in partial_sums:
                    inc = partial_sums[order][prop] - partial_sums[prev_order][prop]
                    increments[prop].append(inc)
    
    return temps, increments


def plot_diagnostics(order_counts, order_sites, order_quality, order_errors, 
                    temps, increments, problem_order, output_dir):
    """Create comprehensive diagnostic plots"""
    
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    orders = sorted(order_counts.keys())
    
    # 1. Cluster count by order
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.bar(orders, [order_counts[o] for o in orders])
    ax1.axvline(x=problem_order, color='red', linestyle='--', linewidth=2, label=f'Order {problem_order}')
    ax1.set_xlabel('Order')
    ax1.set_ylabel('Number of Clusters')
    ax1.set_title('Cluster Count by Order')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Average cluster size by order
    ax2 = fig.add_subplot(gs[0, 1])
    avg_sites = [np.mean(order_sites[o]) if order_sites[o] else 0 for o in orders]
    max_sites = [np.max(order_sites[o]) if order_sites[o] else 0 for o in orders]
    ax2.plot(orders, avg_sites, 'bo-', label='Average sites', linewidth=2)
    ax2.plot(orders, max_sites, 'r^--', label='Max sites', linewidth=2)
    ax2.axvline(x=problem_order, color='red', linestyle='--', linewidth=2, alpha=0.5)
    ax2.set_xlabel('Order')
    ax2.set_ylabel('Number of Sites')
    ax2.set_title('Cluster Size vs Order')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. FTLM quality by order
    ax3 = fig.add_subplot(gs[0, 2])
    if order_quality:
        avg_quality = [np.mean(order_quality[o]) if order_quality[o] else 0 for o in orders]
        max_quality = [np.max(order_quality[o]) if order_quality[o] else 0 for o in orders]
        ax3.semilogy(orders, avg_quality, 'go-', label='Avg rel. error', linewidth=2)
        ax3.semilogy(orders, max_quality, 'r^--', label='Max rel. error', linewidth=2)
        ax3.axvline(x=problem_order, color='red', linestyle='--', linewidth=2, alpha=0.5)
        ax3.axhline(y=0.01, color='orange', linestyle=':', label='1% threshold')
        ax3.set_xlabel('Order')
        ax3.set_ylabel('Relative Error')
        ax3.set_title('FTLM Quality by Order')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    
    # 4. Energy increments
    if increments:
        ax4 = fig.add_subplot(gs[1, :])
        ref_T_idx = len(temps) // 2
        ref_T = temps[ref_T_idx]
        
        energy_incs = [inc[ref_T_idx] for inc in increments['energy']]
        orders_inc = list(range(1, len(energy_incs) + 1))
        
        ax4.plot(orders_inc, energy_incs, 'bo-', linewidth=2, markersize=8)
        ax4.axvline(x=problem_order, color='red', linestyle='--', linewidth=2, 
                   label=f'Order {problem_order}')
        ax4.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
        
        # Highlight problem order
        if problem_order <= len(energy_incs):
            ax4.plot(problem_order, energy_incs[problem_order-1], 'r*', 
                    markersize=20, label=f'Problem: {energy_incs[problem_order-1]:.6e}')
        
        ax4.set_xlabel('Order')
        ax4.set_ylabel('Energy Increment')
        ax4.set_title(f'Order-by-Order Energy Increments (at T={ref_T:.3f})')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 5. Specific heat increments
        ax5 = fig.add_subplot(gs[2, 0])
        C_incs = [inc[ref_T_idx] for inc in increments['specific_heat']]
        ax5.plot(orders_inc, C_incs, 'go-', linewidth=2, markersize=8)
        ax5.axvline(x=problem_order, color='red', linestyle='--', linewidth=2)
        ax5.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
        if problem_order <= len(C_incs):
            ax5.plot(problem_order, C_incs[problem_order-1], 'r*', markersize=20)
        ax5.set_xlabel('Order')
        ax5.set_ylabel('Specific Heat Increment')
        ax5.set_title(f'C_v Increments (T={ref_T:.3f})')
        ax5.grid(True, alpha=0.3)
        
        # 6. Entropy increments
        ax6 = fig.add_subplot(gs[2, 1])
        S_incs = [inc[ref_T_idx] for inc in increments['entropy']]
        ax6.plot(orders_inc, S_incs, 'mo-', linewidth=2, markersize=8)
        ax6.axvline(x=problem_order, color='red', linestyle='--', linewidth=2)
        ax6.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
        if problem_order <= len(S_incs):
            ax6.plot(problem_order, S_incs[problem_order-1], 'r*', markersize=20)
        ax6.set_xlabel('Order')
        ax6.set_ylabel('Entropy Increment')
        ax6.set_title(f'Entropy Increments (T={ref_T:.3f})')
        ax6.grid(True, alpha=0.3)
        
        # 7. Temperature dependence of problem order
        ax7 = fig.add_subplot(gs[2, 2])
        if problem_order <= len(increments['specific_heat']):
            problem_C = increments['specific_heat'][problem_order - 1]
            ax7.plot(temps, problem_C, 'r-', linewidth=2)
            ax7.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
            ax7.set_xlabel('Temperature')
            ax7.set_ylabel(f'Order {problem_order} C_v Increment')
            ax7.set_title(f'Temperature Dependence')
            ax7.set_xscale('log')
            ax7.grid(True, alpha=0.3)
    
    plt.suptitle(f'NLCE Order {problem_order} Anomaly Diagnostics', fontsize=16, fontweight='bold')
    
    output_file = os.path.join(output_dir, f'order_{problem_order}_diagnostics.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nDiagnostic plot saved: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Diagnose anomalies in specific NLCE orders',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  # Diagnose why order 6 shows large changes
  python diagnose_order_anomaly.py \\
      --nlc_dir ./nlce_ftlm_results/order_6/nlc_results \\
      --cluster_dir ./nlce_ftlm_results/order_6/clusters_order_6/cluster_info_order_6 \\
      --ftlm_dir ./nlce_ftlm_results/order_6/ftlm_results \\
      --problem_order 6 \\
      --output_dir ./diagnostics
        """
    )
    
    parser.add_argument('--nlc_dir', type=str, required=True,
                       help='Directory with NLC results (partial_sum files)')
    parser.add_argument('--cluster_dir', type=str, required=True,
                       help='Directory with cluster info files')
    parser.add_argument('--ftlm_dir', type=str, required=True,
                       help='Directory with FTLM results')
    parser.add_argument('--problem_order', type=int, required=True,
                       help='Order that shows anomalous behavior')
    parser.add_argument('--output_dir', type=str, default='./diagnostics',
                       help='Output directory')
    
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("="*80)
    print(f"DIAGNOSING ORDER {args.problem_order} ANOMALY")
    print("="*80)
    
    # Analyze cluster distribution
    print("\n1. Analyzing cluster distribution...")
    order_counts, order_sites, order_multiplicity = analyze_cluster_distribution(
        args.cluster_dir, args.problem_order
    )
    
    print(f"\nCluster counts by order:")
    for order in sorted(order_counts.keys()):
        avg_size = np.mean(order_sites[order]) if order_sites[order] else 0
        max_size = np.max(order_sites[order]) if order_sites[order] else 0
        print(f"  Order {order}: {order_counts[order]} clusters, "
              f"avg size={avg_size:.1f}, max size={max_size}")
        
        if order == args.problem_order:
            print(f"    ▶ PROBLEM ORDER: {order_counts[order]} clusters")
            if order > 1 and order - 1 in order_counts:
                ratio = order_counts[order] / order_counts[order - 1]
                print(f"    ▶ Growth from order {order-1}: {ratio:.2f}x")
    
    # Analyze FTLM quality
    print("\n2. Analyzing FTLM quality...")
    order_quality, order_errors = analyze_ftlm_quality(args.ftlm_dir, args.problem_order)
    
    if order_quality:
        print(f"\nFTLM quality metrics (relative error):")
        for order in sorted(order_quality.keys()):
            if order_quality[order]:
                avg_qual = np.mean(order_quality[order])
                max_qual = np.max(order_quality[order])
                print(f"  Order {order}: avg={avg_qual:.4f}, max={max_qual:.4f}")
                
                if order == args.problem_order:
                    print(f"    ▶ PROBLEM ORDER quality: avg={avg_qual:.4f}")
                    if max_qual > 0.05:
                        print(f"    ⚠ WARNING: Some clusters have >5% error!")
    
    # Load and analyze partial sums
    print("\n3. Analyzing partial sum increments...")
    partial_sums = load_partial_sums(args.nlc_dir, args.problem_order)
    
    if partial_sums:
        result = analyze_increments(partial_sums, args.problem_order)
        if result:
            temps, increments = result
            ref_idx = len(temps) // 2
            
            print(f"\nIncrement analysis at T={temps[ref_idx]:.3f}:")
            for order in sorted(partial_sums.keys()):
                if order <= len(increments['energy']):
                    E_inc = increments['energy'][order - 1][ref_idx]
                    C_inc = increments['specific_heat'][order - 1][ref_idx]
                    S_inc = increments['entropy'][order - 1][ref_idx]
                    print(f"  Order {order}: ΔE={E_inc:.6e}, ΔC={C_inc:.6e}, ΔS={S_inc:.6e}")
                    
                    if order == args.problem_order:
                        print(f"    ▶ PROBLEM ORDER increment")
                        
                        # Compare with previous order
                        if order > 1:
                            prev_E = increments['energy'][order - 2][ref_idx]
                            prev_C = increments['specific_heat'][order - 2][ref_idx]
                            ratio_E = abs(E_inc / (prev_E + 1e-15))
                            ratio_C = abs(C_inc / (prev_C + 1e-15))
                            print(f"    ▶ Ratio vs Order {order-1}: E={ratio_E:.2f}x, C={ratio_C:.2f}x")
    
    # Generate diagnostic plots
    print("\n4. Generating diagnostic plots...")
    if partial_sums:
        temps, increments = analyze_increments(partial_sums, args.problem_order)
        plot_diagnostics(order_counts, order_sites, order_quality, order_errors,
                        temps, increments, args.problem_order, args.output_dir)
    
    # Summary and recommendations
    print("\n" + "="*80)
    print("SUMMARY AND RECOMMENDATIONS")
    print("="*80)
    
    if args.problem_order in order_counts:
        n_clusters = order_counts[args.problem_order]
        print(f"\nOrder {args.problem_order} has {n_clusters} clusters")
        
        if args.problem_order > 1 and args.problem_order - 1 in order_counts:
            growth = n_clusters / order_counts[args.problem_order - 1]
            if growth > 3:
                print(f"  ⚠ Large growth ({growth:.1f}x) in cluster count - expected for NLCE")
        
        if order_sites[args.problem_order]:
            max_size = np.max(order_sites[args.problem_order])
            print(f"  Largest cluster: {max_size} sites")
            if max_size > 20:
                print(f"    ⚠ Large clusters may need more FTLM samples/Krylov dim")
        
        if order_quality and args.problem_order in order_quality:
            max_qual = np.max(order_quality[args.problem_order]) if order_quality[args.problem_order] else 0
            if max_qual > 0.05:
                print(f"  ⚠ FTLM errors up to {max_qual*100:.1f}% - consider:")
                print(f"    - Increase --ftlm_samples (current: check your command)")
                print(f"    - Increase --krylov_dim (current: check your command)")
    
    print("\nPossible causes of anomaly:")
    print("  1. New cluster topology with different energy scale")
    print("  2. Numerical instability in larger clusters")
    print("  3. Insufficient sampling (FTLM samples, Krylov dim)")
    print("  4. Bug in cluster generation or weight calculation")
    print("  5. Series entering divergent regime")
    
    print("\nNext steps:")
    print("  1. Inspect individual cluster FTLM outputs in:")
    print(f"     {args.ftlm_dir}/cluster_*_order_{args.problem_order}/")
    print("  2. Rerun with increased --ftlm_samples and --krylov_dim")
    print("  3. Check cluster generation files for topology errors")
    print("  4. Run convergence radius analysis")
    print("="*80)


if __name__ == "__main__":
    main()
