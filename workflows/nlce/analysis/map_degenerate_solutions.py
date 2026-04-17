#!/usr/bin/env python3
"""
Map degenerate solutions from NLCE fit log files.

Extracts all evaluated parameter sets from the fitting log,
identifies solutions below a chi-squared threshold, and clusters
them to find distinct minima in parameter space.

Usage:
    python map_degenerate_solutions.py --log_file fit.log --threshold 2.0
    python map_degenerate_solutions.py --log_file fit.log --threshold_factor 3.0
"""

import argparse
import numpy as np
import re
import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from itertools import combinations


def parse_log_file(log_file, model='anisotropic'):
    """Parse NLCE fit log file to extract all evaluated parameter sets."""
    
    results = []
    
    if model == 'anisotropic':
        pattern = re.compile(
            r'Jzz=([-\d.]+),\s*Jpm=([-\d.]+),\s*Jpmpm=([-\d.]+),\s*'
            r'Jzpm=([-\d.]+),\s*Chi²=([-\d.]+(?:e[+-]?\d+)?)'
        )
        param_names = ['Jzz', 'Jpm', 'Jpmpm', 'Jzpm']
    else:
        pattern = re.compile(
            r'J1=([-\d.]+),\s*J2=([-\d.]+),\s*Chi²=([-\d.]+(?:e[+-]?\d+)?)'
        )
        param_names = ['J1', 'J2']
    
    with open(log_file, 'r') as f:
        for line in f:
            m = pattern.search(line)
            if m:
                groups = m.groups()
                chi2 = float(groups[-1])
                params = [float(g) for g in groups[:-1]]
                results.append(params + [chi2])
    
    if not results:
        print(f"No parameter evaluations found in {log_file}")
        return None, param_names
    
    data = np.array(results)
    print(f"Parsed {len(data)} evaluations from log file")
    print(f"Chi² range: {data[:,-1].min():.4f} to {data[:,-1].max():.2f}")
    
    return data, param_names


def find_good_solutions(data, param_names, threshold=None, threshold_factor=None,
                        best_n=None):
    """Find solutions below chi-squared threshold."""
    chi2 = data[:, -1]
    chi2_min = chi2.min()
    
    if threshold is not None:
        mask = chi2 <= threshold
        print(f"\nUsing absolute threshold: χ² ≤ {threshold}")
    elif threshold_factor is not None:
        threshold = chi2_min * threshold_factor
        mask = chi2 <= threshold
        print(f"\nUsing relative threshold: χ² ≤ {threshold_factor} × {chi2_min:.4f} = {threshold:.4f}")
    elif best_n is not None:
        sorted_idx = np.argsort(chi2)
        mask = np.zeros(len(chi2), dtype=bool)
        mask[sorted_idx[:best_n]] = True
        threshold = chi2[sorted_idx[min(best_n-1, len(chi2)-1)]]
        print(f"\nUsing best {best_n} solutions (χ² ≤ {threshold:.4f})")
    else:
        # Default: within 2x of best
        threshold = chi2_min * 2.0
        mask = chi2 <= threshold
        print(f"\nUsing default threshold: χ² ≤ 2 × {chi2_min:.4f} = {threshold:.4f}")
    
    good = data[mask]
    print(f"Found {len(good)} solutions below threshold")
    
    if len(good) == 0:
        return None, threshold
    
    # Sort by chi2
    good = good[np.argsort(good[:, -1])]
    
    return good, threshold


def cluster_solutions(good_solutions, param_names, distance_threshold=0.05):
    """
    Cluster good solutions to find distinct minima.
    Uses hierarchical clustering with normalized parameters.
    """
    from scipy.cluster.hierarchy import linkage, fcluster
    
    n_params = len(param_names)
    params = good_solutions[:, :n_params]
    chi2 = good_solutions[:, -1]
    
    if len(params) < 2:
        print("\nOnly 1 good solution found — no clustering needed")
        return [{'params': params[0], 'chi2': chi2[0], 'count': 1, 'spread': np.zeros(n_params)}]
    
    # Normalize parameters for clustering
    param_ranges = params.max(axis=0) - params.min(axis=0)
    param_ranges[param_ranges < 1e-10] = 1.0  # Avoid division by zero
    params_norm = (params - params.min(axis=0)) / param_ranges
    
    # Hierarchical clustering
    Z = linkage(params_norm, method='ward')
    clusters = fcluster(Z, t=distance_threshold, criterion='distance')
    n_clusters = len(set(clusters))
    
    print(f"\nIdentified {n_clusters} distinct solution cluster(s)")
    print("=" * 80)
    
    cluster_info = []
    for cl in sorted(set(clusters)):
        mask_cl = clusters == cl
        p_cl = params[mask_cl]
        chi2_cl = chi2[mask_cl]
        
        best_idx = np.argmin(chi2_cl)
        
        info = {
            'cluster_id': cl,
            'count': int(mask_cl.sum()),
            'params_mean': p_cl.mean(axis=0),
            'params_std': p_cl.std(axis=0),
            'params_min': p_cl.min(axis=0),
            'params_max': p_cl.max(axis=0),
            'best_params': p_cl[best_idx],
            'best_chi2': float(chi2_cl[best_idx]),
            'chi2_range': (float(chi2_cl.min()), float(chi2_cl.max())),
        }
        cluster_info.append(info)
        
        print(f"\n  Cluster {cl} ({info['count']} solutions):")
        print(f"    Best χ² = {info['best_chi2']:.6f}")
        for j, name in enumerate(param_names):
            print(f"    {name:>8s} = {info['params_mean'][j]:10.6f} ± {info['params_std'][j]:.6f}"
                  f"  (range: [{info['params_min'][j]:.6f}, {info['params_max'][j]:.6f}])")
        print(f"    Best:  " + ", ".join(f"{name}={info['best_params'][j]:.6f}" 
                                          for j, name in enumerate(param_names)))
    
    # Sort by best chi2
    cluster_info.sort(key=lambda x: x['best_chi2'])
    
    return cluster_info


def plot_solution_landscape(data, good_solutions, param_names, cluster_info, 
                           output_dir, threshold):
    """Create comprehensive plots of the solution landscape."""
    
    n_params = len(param_names)
    chi2 = data[:, -1]
    chi2_good = good_solutions[:, -1] if good_solutions is not None else np.array([])
    
    # 1. Pairwise scatter plots of good solutions
    n_pairs = n_params * (n_params - 1) // 2
    fig_rows = int(np.ceil(n_pairs / 3))
    fig, axes = plt.subplots(fig_rows, 3, figsize=(18, 5 * fig_rows))
    if fig_rows == 1:
        axes = axes.reshape(1, -1)
    axes_flat = axes.flatten()
    
    pair_idx = 0
    for i, j in combinations(range(n_params), 2):
        ax = axes_flat[pair_idx]
        
        # Background: all evaluations (faded)
        bg_mask = chi2 < np.percentile(chi2, 20)  # Show top 20% as background
        ax.scatter(data[bg_mask, i], data[bg_mask, j], c=chi2[bg_mask], 
                  cmap='Greys', alpha=0.15, s=5, vmin=chi2.min(), 
                  vmax=min(chi2.min() * 20, np.percentile(chi2, 20)))
        
        # Good solutions colored by chi2
        if len(chi2_good) > 0:
            sc = ax.scatter(good_solutions[:, i], good_solutions[:, j], 
                          c=chi2_good, cmap='viridis', s=30, alpha=0.7,
                          edgecolors='black', linewidths=0.3,
                          vmin=chi2_good.min(), 
                          vmax=min(chi2_good.max(), threshold))
            plt.colorbar(sc, ax=ax, label='χ²', shrink=0.8)
        
        # Mark cluster centers
        colors_cluster = plt.cm.Set1(np.linspace(0, 0.8, len(cluster_info)))
        for k, ci in enumerate(cluster_info):
            ax.scatter(ci['best_params'][i], ci['best_params'][j], 
                      marker='*', s=200, c=[colors_cluster[k]], 
                      edgecolors='black', linewidths=1.5, zorder=10,
                      label=f"Cl.{ci['cluster_id']} (χ²={ci['best_chi2']:.3f})")
        
        ax.set_xlabel(param_names[i])
        ax.set_ylabel(param_names[j])
        if pair_idx == 0:
            ax.legend(fontsize=7, loc='best')
        ax.grid(True, alpha=0.3)
        pair_idx += 1
    
    # Hide unused axes
    for idx in range(pair_idx, len(axes_flat)):
        axes_flat[idx].set_visible(False)
    
    plt.suptitle(f'Solution Landscape (χ² < {threshold:.4f}, {len(chi2_good)} solutions, '
                 f'{len(cluster_info)} clusters)', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'solution_landscape.png'), dpi=200, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {os.path.join(output_dir, 'solution_landscape.png')}")
    
    # 2. Chi² convergence plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Running best
    running_best = np.minimum.accumulate(chi2)
    ax1.semilogy(range(len(chi2)), running_best, 'b-', lw=1.5, label='Running best')
    ax1.semilogy(range(len(chi2)), chi2, 'r.', ms=1, alpha=0.3, label='All evaluations')
    ax1.axhline(threshold, color='green', ls='--', label=f'Threshold = {threshold:.4f}')
    ax1.set_xlabel('Evaluation #')
    ax1.set_ylabel('χ²')
    ax1.set_title('Convergence')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Chi² histogram of good solutions
    if len(chi2_good) > 0:
        ax2.hist(chi2_good, bins=min(50, len(chi2_good)), edgecolor='black', alpha=0.7)
        ax2.set_xlabel('χ²')
        ax2.set_ylabel('Count')
        ax2.set_title(f'Distribution of good solutions (n={len(chi2_good)})')
        ax2.axvline(chi2_good.min(), color='red', ls='--', label=f'Best = {chi2_good.min():.4f}')
        ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'convergence_and_distribution.png'), dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {os.path.join(output_dir, 'convergence_and_distribution.png')}")
    
    # 3. 1D marginal plots for each parameter
    if len(chi2_good) > 0:
        fig, axes = plt.subplots(1, n_params, figsize=(5 * n_params, 4))
        if n_params == 1:
            axes = [axes]
        
        for i, (ax, name) in enumerate(zip(axes, param_names)):
            ax.scatter(good_solutions[:, i], chi2_good, s=15, alpha=0.5, c='blue')
            for ci in cluster_info:
                ax.axvline(ci['best_params'][i], color='red', ls='--', alpha=0.7, 
                          label=f"Cl.{ci['cluster_id']}: {ci['best_params'][i]:.4f}")
            ax.set_xlabel(f'{name} (K)')
            ax.set_ylabel('χ²')
            ax.set_title(f'χ² vs {name}')
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'parameter_marginals.png'), dpi=200, bbox_inches='tight')
        plt.close()
        print(f"Saved: {os.path.join(output_dir, 'parameter_marginals.png')}")


def save_results(cluster_info, param_names, output_dir, threshold):
    """Save cluster information to JSON."""
    results = {
        'threshold': threshold,
        'n_clusters': len(cluster_info),
        'param_names': param_names,
        'clusters': []
    }
    
    for ci in cluster_info:
        cluster_dict = {
            'cluster_id': int(ci['cluster_id']),
            'n_solutions': ci['count'],
            'best_chi2': ci['best_chi2'],
            'chi2_range': list(ci['chi2_range']),
            'best_params': {name: float(ci['best_params'][j]) 
                          for j, name in enumerate(param_names)},
            'mean_params': {name: float(ci['params_mean'][j]) 
                          for j, name in enumerate(param_names)},
            'std_params': {name: float(ci['params_std'][j]) 
                         for j, name in enumerate(param_names)},
        }
        results['clusters'].append(cluster_dict)
    
    out_file = os.path.join(output_dir, 'degenerate_solutions.json')
    with open(out_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_file}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Map degenerate solutions from NLCE fit')
    parser.add_argument('--log_file', type=str, required=True,
                       help='Path to NLCE fit log file')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory (default: same as log file)')
    parser.add_argument('--model', type=str, default='anisotropic',
                       choices=['heisenberg', 'anisotropic'])
    parser.add_argument('--threshold', type=float, default=None,
                       help='Absolute chi-squared threshold')
    parser.add_argument('--threshold_factor', type=float, default=None,
                       help='Threshold as multiple of best chi² (e.g., 3.0 = within 3x of best)')
    parser.add_argument('--best_n', type=int, default=None,
                       help='Keep only the best N solutions')
    parser.add_argument('--cluster_distance', type=float, default=0.1,
                       help='Ward distance threshold for clustering (default: 0.1)')
    
    args = parser.parse_args()
    
    if args.output_dir is None:
        args.output_dir = os.path.dirname(args.log_file)
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Parse log file
    data, param_names = parse_log_file(args.log_file, args.model)
    if data is None:
        sys.exit(1)
    
    # Find good solutions
    good, threshold = find_good_solutions(
        data, param_names, 
        threshold=args.threshold,
        threshold_factor=args.threshold_factor,
        best_n=args.best_n
    )
    
    if good is None or len(good) == 0:
        print("No solutions found below threshold!")
        sys.exit(1)
    
    # Cluster solutions
    cluster_info = cluster_solutions(good, param_names, 
                                     distance_threshold=args.cluster_distance)
    
    # Save results
    save_results(cluster_info, param_names, args.output_dir, threshold)
    
    # Plot
    plot_solution_landscape(data, good, param_names, cluster_info, 
                           args.output_dir, threshold)
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY OF DISTINCT SOLUTIONS")
    print("=" * 80)
    for i, ci in enumerate(cluster_info):
        print(f"\n  Solution {i+1} (χ² = {ci['best_chi2']:.6f}, {ci['count']} nearby evaluations):")
        for j, name in enumerate(param_names):
            val = ci['best_params'][j]
            unit = " K" if not name.startswith('J_') else ""
            print(f"    {name:>8s} = {val:10.6f}{unit}")
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
