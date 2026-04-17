#!/usr/bin/env python3
"""
Plot cost function landscape from saved CostLandscapeLogger data.

Loads cost_landscape.npz (or .csv) and generates:
  1. Pairwise 2D scatter plots (parameter vs parameter, colored by log χ²)
  2. 1D marginal slices (each parameter vs χ²)
  3. Convergence curve (χ² vs evaluation number, with DE generation markers)
  4. Parameter trajectory (best-so-far for each parameter over time)

Usage:
    python plot_cost_landscape.py cost_landscape.npz [--top_frac 0.1] [--output_dir plots]
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from pathlib import Path
import os


def load_landscape(path):
    """Load landscape data from .npz or .csv"""
    path = Path(path)
    if path.suffix == '.npz':
        d = np.load(path, allow_pickle=True)
        evals = d['evaluations']
        timestamps = d['timestamps'] if 'timestamps' in d else None
        gen_best = d['gen_best'] if 'gen_best' in d and d['gen_best'].size > 0 else None
        param_names = list(d['param_names']) if 'param_names' in d else None
    elif path.suffix == '.csv':
        import csv
        with open(path) as f:
            header = f.readline().strip().split(',')
        data = np.loadtxt(path, delimiter=',', skiprows=1)
        # Last two columns are chi_squared and wall_time_s
        evals = data[:, :-1]  # params + chi_squared
        timestamps = data[:, -1]
        gen_best = None
        param_names = header[:-2]  # exclude chi_squared, wall_time_s
    else:
        raise ValueError(f"Unsupported file type: {path.suffix}")

    n_params = evals.shape[1] - 1
    params = evals[:, :n_params]
    chi2 = evals[:, -1]

    if param_names is None:
        param_names = [f'p{i}' for i in range(n_params)]

    return params, chi2, timestamps, gen_best, param_names


def plot_convergence(chi2, timestamps, gen_best, param_names, output_dir):
    """Plot χ² vs evaluation number and wall time."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Running best
    running_best = np.minimum.accumulate(chi2)

    # Left: vs eval number
    ax = axes[0]
    ax.scatter(np.arange(len(chi2)), chi2, s=1, alpha=0.3, c='steelblue', label='all evals')
    ax.plot(np.arange(len(chi2)), running_best, 'r-', lw=1.5, label='best so far')
    if gen_best is not None and gen_best.ndim == 2:
        # gen_best columns: gen, chi2_best, convergence, *xk
        # Mark generation boundaries — estimate eval number from gen count
        n_per_gen = len(chi2) // max(gen_best.shape[0], 1)
        gen_evals = (gen_best[:, 0] * n_per_gen).astype(int)
        gen_evals = np.clip(gen_evals, 0, len(chi2)-1)
        ax.scatter(gen_evals, gen_best[:, 1], marker='v', c='darkred', s=30,
                   zorder=5, label='DE gen best')
    ax.set_xlabel('Evaluation #')
    ax.set_ylabel('χ²')
    ax.set_yscale('log')
    ax.set_title('Convergence (eval #)')
    ax.legend(fontsize=8)

    # Right: vs wall time
    ax = axes[1]
    if timestamps is not None and len(timestamps) == len(chi2):
        hours = timestamps / 3600.0
        ax.scatter(hours, chi2, s=1, alpha=0.3, c='steelblue')
        ax.plot(hours, running_best, 'r-', lw=1.5)
        ax.set_xlabel('Wall time (hours)')
    else:
        ax.scatter(np.arange(len(chi2)), chi2, s=1, alpha=0.3, c='steelblue')
        ax.set_xlabel('Evaluation #')
    ax.set_ylabel('χ²')
    ax.set_yscale('log')
    ax.set_title('Convergence (wall time)')

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'convergence.png'), dpi=200)
    plt.close(fig)
    print(f"  Saved convergence.png")


def plot_pairwise(params, chi2, param_names, output_dir, top_frac=1.0):
    """Pairwise scatter of parameters colored by log(χ²)."""
    n = params.shape[1]
    if n < 2:
        print("  Skipping pairwise plot (< 2 parameters)")
        return

    # Optionally filter to top fraction (lowest χ²)
    if top_frac < 1.0:
        threshold = np.percentile(chi2, top_frac * 100)
        mask = chi2 <= threshold
        params = params[mask]
        chi2 = chi2[mask]
        suffix = f" (top {top_frac*100:.0f}%)"
    else:
        suffix = ""

    fig, axes = plt.subplots(n, n, figsize=(3*n + 1, 3*n + 1))
    if n == 1:
        axes = np.array([[axes]])

    vmin, vmax = np.nanpercentile(chi2[chi2 > 0], [1, 99])
    norm = LogNorm(vmin=max(vmin, 1e-10), vmax=vmax)

    for i in range(n):
        for j in range(n):
            ax = axes[i, j]
            if i == j:
                # Diagonal: 1D histogram
                ax.hist(params[:, i], bins=50, color='steelblue', alpha=0.7)
                ax.set_xlabel(param_names[i])
                ax.set_ylabel('count')
            elif i > j:
                # Lower triangle: scatter
                sc = ax.scatter(params[:, j], params[:, i], c=chi2,
                              s=2, alpha=0.4, cmap='viridis_r', norm=norm)
                ax.set_xlabel(param_names[j])
                ax.set_ylabel(param_names[i])
            else:
                ax.axis('off')

    fig.suptitle(f'Parameter space{suffix}', fontsize=14)
    fig.tight_layout(rect=[0, 0, 0.92, 0.96])

    # Add colorbar
    cbar_ax = fig.add_axes([0.93, 0.15, 0.02, 0.7])
    sm = plt.cm.ScalarMappable(cmap='viridis_r', norm=norm)
    fig.colorbar(sm, cax=cbar_ax, label='χ²')

    fig.savefig(os.path.join(output_dir, 'pairwise_landscape.png'), dpi=200)
    plt.close(fig)
    print(f"  Saved pairwise_landscape.png")


def plot_marginals(params, chi2, param_names, output_dir):
    """1D slices: each parameter vs χ²."""
    n = params.shape[1]
    fig, axes = plt.subplots(1, n, figsize=(4*n, 4))
    if n == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        ax.scatter(params[:, i], chi2, s=2, alpha=0.3, c='steelblue')
        ax.set_xlabel(param_names[i])
        ax.set_ylabel('χ²')
        ax.set_yscale('log')
        ax.set_title(f'{param_names[i]} marginal')

        # Mark best
        best_idx = np.argmin(chi2)
        ax.axvline(params[best_idx, i], color='red', ls='--', lw=1,
                   label=f'best={params[best_idx, i]:.4f}')
        ax.legend(fontsize=7)

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'marginals.png'), dpi=200)
    plt.close(fig)
    print(f"  Saved marginals.png")


def plot_parameter_trajectories(params, chi2, param_names, output_dir):
    """Best-so-far parameter values over the course of optimization."""
    n = params.shape[1]
    running_best_idx = np.zeros(len(chi2), dtype=int)
    current_best = np.inf
    for k in range(len(chi2)):
        if chi2[k] < current_best:
            current_best = chi2[k]
            running_best_idx[k] = k
        else:
            running_best_idx[k] = running_best_idx[k-1] if k > 0 else 0

    fig, axes = plt.subplots(n, 1, figsize=(10, 3*n), sharex=True)
    if n == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        best_params = params[running_best_idx, i]
        ax.plot(best_params, 'r-', lw=1)
        ax.set_ylabel(param_names[i])
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel('Evaluation #')
    fig.suptitle('Best-so-far parameter trajectories', fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(os.path.join(output_dir, 'param_trajectories.png'), dpi=200)
    plt.close(fig)
    print(f"  Saved param_trajectories.png")


def main():
    parser = argparse.ArgumentParser(description='Plot cost function landscape')
    parser.add_argument('landscape_file', help='Path to cost_landscape.npz or .csv')
    parser.add_argument('--output_dir', default=None,
                       help='Output directory for plots (default: same as input file)')
    parser.add_argument('--top_frac', type=float, default=0.1,
                       help='Fraction of lowest-χ² points to show in pairwise plot (default: 0.1)')
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = str(Path(args.landscape_file).parent)
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading {args.landscape_file}...")
    params, chi2, timestamps, gen_best, param_names = load_landscape(args.landscape_file)
    print(f"  {len(chi2)} evaluations, {len(param_names)} parameters: {param_names}")
    print(f"  χ² range: [{chi2.min():.6f}, {chi2.max():.2f}]")
    print(f"  Best: {' '.join(f'{p}={v:.6f}' for p,v in zip(param_names, params[np.argmin(chi2)]))} "
          f"→ χ²={chi2.min():.6f}")

    plot_convergence(chi2, timestamps, gen_best, param_names, args.output_dir)
    plot_pairwise(params, chi2, param_names, args.output_dir, top_frac=args.top_frac)
    plot_pairwise(params, chi2, param_names, args.output_dir, top_frac=1.0)
    # Rename the full one
    full_path = os.path.join(args.output_dir, 'pairwise_landscape.png')
    if os.path.exists(full_path):
        os.rename(full_path, os.path.join(args.output_dir, 'pairwise_landscape_all.png'))
        print(f"  Saved pairwise_landscape_all.png")
    plot_marginals(params, chi2, param_names, args.output_dir)
    plot_parameter_trajectories(params, chi2, param_names, args.output_dir)

    print("\nDone! All plots saved to:", args.output_dir)


if __name__ == '__main__':
    main()
